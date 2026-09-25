"""Compatibility wrapper around the finite transfer scorecard.

Historically this module returned NaN placeholders when a control was not
applicable.  The public ablation API now preserves compact legacy fields but
represents missing measurements as ``None`` and carries status and reason.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .scorecard import ScorecardReport, run_transfer_scorecard
from .types import AlignmentResult, TrajectoryTrace, TransitionObservation


@dataclass(frozen=True)
class AblationCase:
    name: str
    geometric_error: float | None
    train_flow_error: float | None
    holdout_error: float | None
    mean_confidence: float | None
    correction_norm: float | None
    notes: tuple[str, ...] = ()
    status: str = "insufficient_data"
    reason: str | None = None
    validation_error: float | None = None
    baseline_holdout_error: float | None = None
    improvement: float | None = None
    relative_improvement: float | None = None
    validation_improvement: float | None = None
    holdout_improvement: float | None = None
    accepted_nodes: int = 0
    rollback_nodes: int = 0
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class AblationReport:
    cases: tuple[AblationCase, ...]
    teacher_probe_count: int
    student_probe_count: int
    holdout_used: bool
    metadata: dict[str, object]
    scorecard: ScorecardReport | None = None

    def to_dict(self) -> dict[str, object]:
        if self.scorecard is not None:
            return self.scorecard.to_dict()
        return {
            "schema_version": "faytuna-scorecard-v1",
            "status": "ok",
            "validation_used": False,
            "holdout_used": self.holdout_used,
            "entries": [
                {
                    "name": case.name,
                    "status": case.status,
                    "geometric_error": case.geometric_error,
                    "train_flow_error": case.train_flow_error,
                    "holdout_error": case.holdout_error,
                    "mean_confidence": case.mean_confidence,
                    "correction_norm": case.correction_norm,
                    "reason": case.reason,
                    "notes": list(case.notes),
                }
                for case in self.cases
            ],
            "metadata": self.metadata,
        }


def _perturb_traces(
    traces: Sequence[TrajectoryTrace],
    *,
    noise: float = 0.0,
    quantization_step: float | None = None,
    seed: int = 0,
) -> list[TrajectoryTrace]:
    """Perturb observed states while preserving trace schema and metadata."""

    rng = np.random.default_rng(seed)
    result: list[TrajectoryTrace] = []
    for trace in traces:
        states = trace.hidden_states.copy()
        if noise:
            scale = max(1.0, float(np.std(states)))
            states += noise * scale * rng.normal(size=states.shape)
        if quantization_step is not None:
            if quantization_step <= 0 or not np.isfinite(quantization_step):
                raise ValueError("quantization_step must be positive and finite")
            states = np.round(states / quantization_step) * quantization_step
        transitions = []
        for index, old in enumerate(trace.transitions):
            delta = states[index + 1] - states[index]
            ds = old.target_depth - old.source_depth
            transitions.append(
                TransitionObservation(
                    old.source_layer,
                    old.target_layer,
                    old.source_depth,
                    old.target_depth,
                    states[index],
                    states[index + 1],
                    delta,
                    delta / ds,
                    old.jacobian,
                    old.hessian_sketch,
                    old.curvature,
                    old.singular_values,
                    old.normalization_geometry,
                    old.attention_geometry,
                    old.uncertainty,
                )
            )
        result.append(replace(trace, hidden_states=states, transitions=tuple(transitions)))
    return result


def _case_from_entry(entry) -> AblationCase:
    metadata = dict(entry.metadata or {})
    metadata.setdefault("accepted_nodes", entry.accepted_nodes)
    return AblationCase(
        entry.name,
        entry.geometric_error,
        entry.train_error,
        entry.holdout_error,
        entry.mean_confidence,
        entry.correction_norm,
        entry.notes,
        entry.status,
        entry.reason,
        entry.validation_error,
        entry.baseline_holdout_error,
        entry.improvement,
        entry.relative_improvement,
        entry.validation_improvement,
        entry.holdout_improvement,
        entry.accepted_nodes,
        entry.rollback_nodes,
        metadata,
    )


def run_transfer_ablations(
    student_traces: Sequence[TrajectoryTrace],
    teacher_traces: Sequence[TrajectoryTrace],
    teacher_to_student: AlignmentResult,
    *,
    validation_student: Sequence[TrajectoryTrace] | None = None,
    validation_teacher: Sequence[TrajectoryTrace] | None = None,
    holdout_student: Sequence[TrajectoryTrace] | None = None,
    holdout_teacher: Sequence[TrajectoryTrace] | None = None,
    seed: int = 0,
    include_stress: bool = True,
) -> AblationReport:
    """Run accuracy ablations and return finite, structured control results."""

    scorecard = run_transfer_scorecard(
        student_traces,
        teacher_traces,
        teacher_to_student,
        validation_student=validation_student,
        validation_teacher=validation_teacher,
        holdout_student=holdout_student,
        holdout_teacher=holdout_teacher,
        seed=seed,
        include_stress=include_stress,
    )
    cases = tuple(_case_from_entry(entry) for entry in scorecard.entries)
    metadata = {
        **dict(scorecard.metadata),
        "scorecard_schema": scorecard.schema_version,
        "holdout_required_for_improvement": True,
        "operational_success_proxy": "disjoint transported-flow error; causal rerun is separate",
    }
    return AblationReport(cases, len(teacher_traces), len(student_traces), scorecard.holdout_used, metadata, scorecard)
