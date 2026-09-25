"""Transparent correction scoring from direct experiment metrics.

The scorer is deliberately a fixed six-feature calculator, not a trained
mini-ML model or hidden decision model. It exposes the exact feature vector,
direct inputs, calibration comparison, and disagreement so an experiment can
replace or reject it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(_number(value), low, high))


def _error_improvement(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    if "holdout_error" in before and "holdout_error" in after:
        return _number(before["holdout_error"]) - _number(after["holdout_error"])
    if "holdout_accuracy" in before and "holdout_accuracy" in after:
        return _number(after["holdout_accuracy"]) - _number(before["holdout_accuracy"])
    if "dynamic_functional_error" in before and "dynamic_functional_error" in after:
        return _number(before["dynamic_functional_error"]) - _number(after["dynamic_functional_error"])
    return 0.0


def _quality_score(value: Any, *, error: bool = False) -> float:
    number = _number(value)
    if error:
        return float(1.0 / (1.0 + max(0.0, number)))
    return _clip(number)


@dataclass(frozen=True)
class CorrectionAssessment:
    """Auditable diagnostic for one before/after correction comparison."""

    feature_names: tuple[str, ...]
    feature_vector: np.ndarray
    direct_metrics: Mapping[str, float | None]
    prediction: float
    confidence: float
    calibration: float | None
    disagreement: float
    novelty: float
    status: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.feature_vector, dtype=np.float64)
        if values.ndim != 1 or len(values) != len(self.feature_names) or not np.all(np.isfinite(values)):
            raise ValueError("assessor feature vector must be finite and match feature names")
        for value in (self.prediction, self.confidence, self.disagreement, self.novelty):
            if not np.isfinite(value):
                raise ValueError("assessor output must be finite")
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.novelty <= 1.0:
            raise ValueError("assessor confidence and novelty must be in [0, 1]")
        if self.calibration is not None and not 0.0 <= self.calibration <= 1.0:
            raise ValueError("assessor calibration must be in [0, 1]")
        object.__setattr__(self, "feature_vector", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessor": "surprise_novelty_scorer",
            "scorer_type": "fixed_direct_metric_scorer",
            "trained_model": False,
            "model_checksum": None,
            "calibration_type": "sign_calibration_only",
            "feature_names": list(self.feature_names),
            "feature_vector": [float(value) for value in self.feature_vector],
            "direct_metrics": {str(key): None if value is None else float(value) for key, value in self.direct_metrics.items()},
            "prediction": float(self.prediction),
            "confidence": float(self.confidence),
            "calibration": None if self.calibration is None else float(self.calibration),
            "disagreement": float(self.disagreement),
            "novelty": float(self.novelty),
            "status": self.status,
            "limitations": list(self.limitations),
        }


def assess_correction(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    uncertainty: float = 0.0,
    validation: Mapping[str, Any] | None = None,
) -> CorrectionAssessment:
    """Compute an interpretable fixed scorer from direct metrics.

    ``prediction`` is a fixed mean of exposed normalized metrics; it is not a
    trained prediction model and is not used as a substitute for the held-out
    guard. ``validation`` may contain a held-out improvement used only to
    report sign calibration.
    """

    improvement = _error_improvement(before, after)
    smoothness = _quality_score(after.get("smoothness", after.get("smoothness_error", 0.0)), error="smoothness_error" in after and "smoothness" not in after)
    stability = _quality_score(after.get("stability", after.get("stability_error", 0.0)), error="stability_error" in after and "stability" not in after)
    magnitude = _clip(after.get("observable_effect", after.get("correction_magnitude", 0.0)))
    teacher_consistency = _clip(after.get("teacher_consistency", 1.0 / (1.0 + max(0.0, _number(after.get("holdout_error", 0.0))))))
    uncertainty_score = _clip(1.0 - _clip(uncertainty))
    improvement_score = _clip(0.5 + 0.5 * np.tanh(10.0 * improvement))
    names = ("holdout_improvement_signal", "smoothness", "stability", "observable_effect", "teacher_consistency", "certainty")
    vector = np.asarray([improvement_score, smoothness, stability, magnitude, teacher_consistency, uncertainty_score], dtype=np.float64)
    prediction = float(np.mean(vector))
    disagreement = float(np.mean(np.abs(vector - prediction)))
    confidence = _clip(uncertainty_score * (1.0 - disagreement))
    novelty = _clip(0.5 * magnitude + 0.3 * abs(improvement_score - 0.5) + 0.2 * (1.0 - uncertainty_score))
    calibration = None
    if validation is not None:
        validation_improvement = _number(validation.get("holdout_improvement", validation.get("improvement", 0.0)))
        calibration = 1.0 if (improvement >= 0.0) == (validation_improvement >= 0.0) else 0.0
    if uncertainty_score < 0.5 or disagreement > 0.30:
        status = "uncertain"
    elif improvement > 0.0:
        status = "positive_direct_metrics"
    elif improvement < 0.0:
        status = "negative_direct_metrics"
    else:
        status = "unchanged_direct_metrics"
    return CorrectionAssessment(
        names,
        vector,
        {
            "holdout_improvement": improvement,
            "smoothness": smoothness,
            "stability": stability,
            "observable_effect": magnitude,
            "teacher_consistency": teacher_consistency,
            "uncertainty": _clip(uncertainty),
        },
        prediction,
        confidence,
        calibration,
        disagreement,
        novelty,
        status,
        (
            "fixed diagnostic score from direct metrics; it does not approve a correction",
            "novelty/surprise is an operational scorer label, not a semantic claim",
            "no learned model is fitted; calibration is sign agreement only",
            "causal rerun and student-capacity checks remain independent evidence",
        ),
    )
