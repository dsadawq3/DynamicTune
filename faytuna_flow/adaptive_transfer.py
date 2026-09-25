"""Adaptive transfer policy for a fitted flow correction.

This is the single implementation for opt-in continuation.  The ordinary
scorecard and ``fit-flow`` path remain the clean baseline.  A policy applies
corrections by coherent depth blocks through an alpha sweep; a failed block is
rolled back as a whole, and no step is accepted without disjoint validation
  and holdout evidence.  ``safe`` and ``experimental`` are modes of this
  policy, not separate algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .scorecard import DYNAMIC_METRIC_SCOPE, FLOW_METRIC_SCOPE, _dynamic_one_step_error, _operator_error
from .assessor import assess_correction
from .math_profile import build_math_profile
from .solver import ConstrainedCorrection, TrustRegionConfig, solve_flow_correction
from .types import AlignmentResult, FlowFitResult, FlowOperator, TrajectoryTrace, stable_l2


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite(value)
    return value


@dataclass(frozen=True)
class AdaptiveTransferPolicy:
    mode: str = "safe"
    gain_grid: tuple[float, ...] | None = None
    block_size: int = 2
    min_observable_effect: float = 1e-3
    acceptance_tolerance: float = 1e-12
    max_step_norm: float = 0.25
    hard_max_step_norm: float = 1.0
    max_relative_step: float = 0.20
    max_spectral_norm: float = 2.0
    max_lipschitz: float = 2.5
    min_confidence: float = 0.05
    collapse_variance_ratio: float = 0.05
    max_stability_uncertainty: float = 0.50
    line_search: str = "auto"
    backend: str = "auto"
    depth_strategy: str = "auto"
    tensor_families: tuple[str, ...] | None = None
    block_order: str = "auto"
    calibrate_lm_head: bool = False
    memory_imprint_mlp: bool = False
    flow_distill_steps: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"safe", "experimental", "experimental_untrusted"}:
            raise ValueError("adaptive transfer mode must be 'safe', 'experimental', or 'experimental_untrusted'")
        if self.gain_grid is None:
            gains = (0.0, 0.25, 0.50, 0.75, 1.0) if self.mode == "safe" else (0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5)
        else:
            gains = tuple(float(value) for value in self.gain_grid)
        object.__setattr__(self, "gain_grid", gains)
        if not gains or gains[0] != 0.0 or any(not np.isfinite(value) or value < 0 for value in gains) or any(right < left for left, right in zip(gains, gains[1:])):
            raise ValueError("gain_grid must be finite, non-negative, sorted, and start at alpha=0")
        if self.block_size < 1 or self.min_observable_effect < 0 or self.acceptance_tolerance < 0:
            raise ValueError("adaptive transfer block and effect parameters are invalid")
        for value in (self.max_step_norm, self.hard_max_step_norm, self.max_relative_step, self.max_spectral_norm, self.max_lipschitz, self.max_stability_uncertainty):
            if value <= 0 or not np.isfinite(value):
                raise ValueError("adaptive transfer trust and stability limits must be positive and finite")
        if self.hard_max_step_norm < self.max_step_norm or not 0.0 <= self.min_confidence <= 1.0 or not 0.0 <= self.collapse_variance_ratio <= 1.0:
            raise ValueError("adaptive transfer bounds are inconsistent")
        if self.line_search not in {"auto", "backtracking", "grid"}:
            raise ValueError("line_search must be auto, backtracking, or grid")
        if self.backend not in {"auto", "dense", "scalable"}:
            raise ValueError("backend must be auto, dense, or scalable")
        if self.depth_strategy not in {"auto", "existing_monotone", "gap_aware_monotone"}:
            raise ValueError("depth_strategy is unsupported")
        if self.block_order not in {"auto", "depth", "confidence"}:
            raise ValueError("block_order must be auto, depth, or confidence")
        if self.tensor_families is not None and any(not str(item) for item in self.tensor_families):
            raise ValueError("tensor_families cannot contain empty names")
        if self.flow_distill_steps < 0:
            raise ValueError("flow_distill_steps must be non-negative")

    def tune(
        self,
        fit: FlowFitResult,
        target: FlowOperator,
        *,
        alignment: AlignmentResult,
        validation_student: Sequence[TrajectoryTrace] | None,
        validation_teacher: Sequence[TrajectoryTrace] | None,
        holdout_student: Sequence[TrajectoryTrace] | None,
        holdout_teacher: Sequence[TrajectoryTrace] | None,
        blocks: Sequence[Sequence[int]] | None = None,
    ) -> "AdaptiveTransferReport":
        """Select a bounded strategy on train/validation and judge on holdout.

        The low-level schedule below is intentionally deterministic. This
        method is the policy boundary: it selects strategy metadata and block
        order from the available evidence, then delegates all application and
        rollback decisions to the same schedule for both modes.
        """

        validation_used = validation_student is not None and validation_teacher is not None
        holdout_used = holdout_student is not None and holdout_teacher is not None
        if not validation_used:
            return AdaptiveTransferReport(
                "insufficient_validation", self.mode, False, holdout_used, (), fit.student,
                {"reason": "adaptive transfer requires a disjoint validation split for policy selection", "policy_mode": self.mode},
            )
        selection = _select_policy_strategy(self, fit, validation_student, blocks=blocks)
        selected_blocks = selection.pop("blocks")
        selected_config = replace(
            self,
            gain_grid=tuple(selection["gain_grid"]),
            block_size=int(selection["block_size"]),
            line_search=str(selection["line_search"]),
        )
        report = run_adaptive_transfer(
            fit,
            target,
            alignment=alignment,
            validation_student=validation_student,
            validation_teacher=validation_teacher,
            holdout_student=holdout_student,
            holdout_teacher=holdout_teacher,
            config=selected_config,
            blocks=selected_blocks,
        )
        metadata = dict(report.metadata)
        metadata["policy_selection"] = _jsonable(selection)
        metadata["holdout_is_final_verdict"] = True
        metadata["teacher_at_inference"] = False
        metadata["runtime_adapter"] = False
        return replace(report, metadata=metadata)


def _select_policy_strategy(
    policy: AdaptiveTransferPolicy,
    fit: FlowFitResult,
    validation_student: Sequence[TrajectoryTrace],
    *,
    blocks: Sequence[Sequence[int]] | None,
) -> dict[str, Any]:
    """Choose strategy knobs from measured fit/validation evidence.

    Selection is deliberately conservative: it never invents layers or
    tensors. The returned fields are also serialized into the final report so
    a real-model run can be audited rather than inferred from a label.
    """

    node_count = int(len(fit.correction_matrices))
    dense_size = int(np.prod(fit.correction_matrices.shape))
    measured_backend = str(fit.metadata.get("scalable_backend", "dense_local_field"))
    if policy.backend != "auto":
        backend = policy.backend
        backend_basis = "explicit policy setting"
    elif measured_backend not in {"", "dense_local_field", "dense"} or dense_size > 2_000_000:
        backend = "scalable"
        backend_basis = "fit metadata or dense allocation guard"
    else:
        backend = "dense"
        backend_basis = "bounded correction tensor"

    gap_count = int(getattr(fit.depth_report, "gap_count", 0) or 0)
    if policy.depth_strategy != "auto":
        depth_strategy = policy.depth_strategy
        depth_basis = "explicit policy setting"
    elif gap_count:
        depth_strategy = "gap_aware_monotone"
        depth_basis = "observed depth gaps"
    else:
        depth_strategy = "existing_monotone"
        depth_basis = "all fitted nodes have monotone support"

    if policy.tensor_families is not None:
        families = tuple(str(item) for item in policy.tensor_families)
        family_basis = "explicit policy setting"
    else:
        families = ["residual_flow", "trajectory_direction", "trajectory_magnitude"]
        if fit.correction_quadratic is not None:
            families.append("curvature")
        families = tuple(families)
        family_basis = "available fitted correction channels"

    base_blocks = _blocks(node_count, policy.block_size, blocks)
    order = policy.block_order
    if order == "auto":
        order = "confidence" if policy.mode in {"experimental", "experimental_untrusted"} else "depth"
    if order == "confidence":
        confidence = np.asarray(fit.confidence, dtype=np.float64)
        base_blocks = tuple(sorted(base_blocks, key=lambda block: (-float(np.mean(confidence[list(block)])), block)))
    elif order != "depth":
        raise ValueError("selected block order is unsupported")

    line_search = policy.line_search
    if line_search == "auto":
        line_search = "backtracking" if policy.mode == "safe" else "grid"
    gains = tuple(float(value) for value in (policy.gain_grid or (0.0,)))
    if policy.mode == "safe":
        gains = tuple(value for value in gains if value <= 1.0)
    if not gains or gains[0] != 0.0:
        gains = (0.0,)

    validation_support = int(sum(len(trace.transitions) for trace in validation_student))
    return {
        "backend": backend,
        "backend_basis": backend_basis,
        "depth_strategy": depth_strategy,
        "depth_basis": depth_basis,
        "tensor_families": families,
        "tensor_family_basis": family_basis,
        "block_order": order,
        "line_search": line_search,
        "gain_grid": gains,
        "block_size": max(1, max((len(block) for block in base_blocks), default=1)),
        "validation_transition_support": validation_support,
        "blocks": base_blocks,
    }


@dataclass(frozen=True)
class AdaptiveStep:
    alpha: float
    status: str
    validation_error: float | None
    holdout_error: float | None
    baseline_holdout_error: float | None
    dynamic_functional_error: float | None
    baseline_dynamic_functional_error: float | None
    holdout_improvement: float | None
    dynamic_functional_improvement: float | None
    observable_effect: float | None
    variance_ratio: float | None
    mean_confidence: float | None
    max_spectral_norm: float | None
    max_lipschitz: float | None
    accepted_blocks: tuple[tuple[int, ...], ...]
    rollback_blocks: tuple[tuple[int, ...], ...]
    ineffective_blocks: tuple[tuple[int, ...], ...]
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "alpha": self.alpha,
            "status": self.status,
            "validation_error": self.validation_error,
            "holdout_error": self.holdout_error,
            "baseline_holdout_error": self.baseline_holdout_error,
            "dynamic_functional_error": self.dynamic_functional_error,
            "baseline_dynamic_functional_error": self.baseline_dynamic_functional_error,
            "holdout_improvement": self.holdout_improvement,
            "dynamic_functional_improvement": self.dynamic_functional_improvement,
            "observable_effect": self.observable_effect,
            "variance_ratio": self.variance_ratio,
            "mean_confidence": self.mean_confidence,
            "max_spectral_norm": self.max_spectral_norm,
            "max_lipschitz": self.max_lipschitz,
            "accepted_blocks": self.accepted_blocks,
            "rollback_blocks": self.rollback_blocks,
            "ineffective_blocks": self.ineffective_blocks,
            "notes": self.notes,
            "metadata": self.metadata or {},
        })


@dataclass(frozen=True)
class AdaptiveTransferReport:
    status: str
    mode: str
    validation_used: bool
    holdout_used: bool
    steps: tuple[AdaptiveStep, ...]
    final_operator: FlowOperator | None
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "schema_version": "faytuna-adaptive-transfer-v1",
            "status": self.status,
            "mode": self.mode,
            "validation_used": self.validation_used,
            "holdout_used": self.holdout_used,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
        })


def _blocks(node_count: int, block_size: int, supplied: Sequence[Sequence[int]] | None) -> tuple[tuple[int, ...], ...]:
    if supplied is None:
        return tuple(tuple(range(start, min(node_count, start + block_size))) for start in range(0, node_count, block_size))
    result = tuple(tuple(int(index) for index in block) for block in supplied)
    flattened = [index for block in result for index in block]
    if sorted(flattened) != list(range(node_count)) or any(not block for block in result):
        raise ValueError("coherent blocks must partition every flow node exactly once")
    return result


def _operator_with_arrays(fit: FlowFitResult, matrices: np.ndarray, biases: np.ndarray, quadratic: np.ndarray | None, *, alpha: float) -> FlowOperator:
    spectral = np.asarray([np.linalg.svd(matrix, compute_uv=False)[0] for matrix in matrices])
    return FlowOperator(
        fit.student.coordinates,
        matrices,
        biases,
        fit.student.sample_counts,
        fit.student.residual_scales,
        spectral,
        {**fit.student.metadata, "adaptive_transfer": True, "alpha": alpha, "mode": "adaptive_transfer"},
        quadratic,
        fit.student.chart_projection,
    )


def _field_values(operator: FlowOperator, traces: Sequence[TrajectoryTrace]) -> np.ndarray:
    values = []
    for trace in traces:
        for transition in trace.transitions:
            values.append(operator.predict(transition.source_state, transition.source_depth))
    if not values:
        raise ValueError("adaptive transfer effect needs non-empty traces")
    return np.asarray(values)


def _observable_effect(candidate: FlowOperator, baseline: FlowOperator, traces: Sequence[TrajectoryTrace]) -> tuple[float, float]:
    candidate_values = _field_values(candidate, traces)
    baseline_values = _field_values(baseline, traces)
    effect = float(np.mean(np.asarray(stable_l2(candidate_values - baseline_values, axis=1, name="adaptive transfer effect")) / np.maximum(1.0, np.asarray(stable_l2(baseline_values, axis=1, name="baseline field")))))
    candidate_centered = candidate_values - candidate_values.mean(axis=0, keepdims=True)
    baseline_centered = baseline_values - baseline_values.mean(axis=0, keepdims=True)
    candidate_variance = float(np.mean(np.square(np.asarray(stable_l2(candidate_centered, axis=1, name="candidate field variance")))))
    baseline_variance = float(np.mean(np.square(np.asarray(stable_l2(baseline_centered, axis=1, name="baseline field variance")))))
    return effect, candidate_variance / max(baseline_variance, 1e-12)


def run_adaptive_transfer(
    fit: FlowFitResult,
    target: FlowOperator,
    *,
    alignment: AlignmentResult,
    validation_student: Sequence[TrajectoryTrace] | None,
    validation_teacher: Sequence[TrajectoryTrace] | None,
    holdout_student: Sequence[TrajectoryTrace] | None,
    holdout_teacher: Sequence[TrajectoryTrace] | None,
    config: AdaptiveTransferPolicy | None = None,
    blocks: Sequence[Sequence[int]] | None = None,
) -> AdaptiveTransferReport:
    """Run the explicit policy schedule with block-coherent rollback."""

    cfg = config or AdaptiveTransferPolicy()
    validation_used = validation_student is not None and validation_teacher is not None
    holdout_used = holdout_student is not None and holdout_teacher is not None
    if (validation_student is None) != (validation_teacher is None) or (holdout_student is None) != (holdout_teacher is None):
        raise ValueError("adaptive transfer teacher/student evaluation traces must be supplied in pairs")
    if not holdout_used:
        report = AdaptiveTransferReport("insufficient_holdout", cfg.mode, validation_used, False, (), fit.student, {"reason": "adaptive transfer requires a disjoint holdout", "policy_mode": cfg.mode, "target_fit_split": "train", "metric_scope": FLOW_METRIC_SCOPE, "dynamic_metric_scope": DYNAMIC_METRIC_SCOPE})
        return report
    node_blocks = _blocks(len(fit.correction_matrices), cfg.block_size, blocks)
    baseline = fit.student
    baseline_holdout = _operator_error(baseline, target, holdout_student)
    baseline_validation = None if not validation_used else _operator_error(baseline, target, validation_student)
    baseline_dynamic_holdout = _dynamic_one_step_error(baseline, holdout_student, holdout_teacher, alignment)
    baseline_dynamic_validation = None if not validation_used else _dynamic_one_step_error(baseline, validation_student, validation_teacher, alignment)
    current_matrices = np.zeros_like(fit.correction_matrices)
    current_biases = np.zeros_like(fit.correction_biases)
    current_quadratic = None if fit.correction_quadratic is None else np.zeros_like(fit.correction_quadratic)
    current_operator = baseline
    current_validation = baseline_validation
    current_holdout = baseline_holdout
    current_dynamic_validation = baseline_dynamic_validation
    current_dynamic_holdout = baseline_dynamic_holdout
    current_alpha = 0.0
    baseline_assessment = assess_correction(
        {"holdout_error": baseline_holdout, "dynamic_functional_error": baseline_dynamic_holdout},
        {"holdout_error": baseline_holdout, "dynamic_functional_error": baseline_dynamic_holdout, "stability": 1.0, "observable_effect": 0.0},
        uncertainty=1.0 - float(np.mean(fit.confidence)),
    )
    steps = [
        AdaptiveStep(
            0.0,
            "baseline",
            baseline_validation,
            baseline_holdout,
            baseline_holdout,
            baseline_dynamic_validation,
            baseline_dynamic_holdout,
            0.0,
            0.0,
            0.0,
            1.0,
            _finite(np.mean(fit.confidence)),
            _finite(np.max(fit.student.spectral_norms)),
            _finite(np.max(fit.student.spectral_norms)),
            (),
            (),
            (),
            ("clean student baseline; no correction applied",),
            {"target_fit_split": "train", "assessor": baseline_assessment.to_dict()},
        )
    ]
    adaptive_step = cfg.max_step_norm
    for alpha in cfg.gain_grid[1:]:
        proposed = replace(
            fit,
            correction_matrices=fit.correction_matrices * alpha,
            correction_biases=fit.correction_biases * alpha,
            correction_quadratic=None if fit.correction_quadratic is None else fit.correction_quadratic * alpha,
        )
        trust = TrustRegionConfig(max_step_norm=min(cfg.hard_max_step_norm, adaptive_step * max(1.0, alpha)), max_relative_step=min(1.0, cfg.max_relative_step * max(1.0, alpha)), max_spectral_norm=cfg.max_spectral_norm, max_lipschitz=cfg.max_lipschitz, min_confidence=cfg.min_confidence, collapse_variance_ratio=cfg.collapse_variance_ratio)
        constrained = solve_flow_correction(proposed, config=trust)
        accepted_blocks: list[tuple[int, ...]] = []
        rollback_blocks: list[tuple[int, ...]] = []
        ineffective_blocks: list[tuple[int, ...]] = []
        block_effects: dict[str, float] = {}
        for block in node_blocks:
            trial_matrices = current_matrices.copy()
            trial_biases = current_biases.copy()
            trial_quadratic = None if current_quadratic is None else current_quadratic.copy()
            trial_matrices[list(block)] = constrained.matrices[list(block)]
            trial_biases[list(block)] = constrained.biases[list(block)]
            if trial_quadratic is not None:
                trial_quadratic[list(block)] = constrained.quadratic_terms[list(block)]
            trial_operator = _operator_with_arrays(fit, fit.student.matrices + trial_matrices, fit.student.biases + trial_biases, None if fit.student.quadratic_terms is None else fit.student.quadratic_terms + trial_quadratic, alpha=alpha)
            effect, variance_ratio = _observable_effect(trial_operator, current_operator, holdout_student)
            block_effects["-".join(map(str, block))] = effect
            trial_validation = None if not validation_used else _operator_error(trial_operator, target, validation_student)
            trial_holdout = _operator_error(trial_operator, target, holdout_student)
            trial_dynamic_validation = None if not validation_used else _dynamic_one_step_error(trial_operator, validation_student, validation_teacher, alignment)
            trial_dynamic_holdout = _dynamic_one_step_error(trial_operator, holdout_student, holdout_teacher, alignment)
            solver_ok = bool(np.all(constrained.accepted[list(block)]))
            improves_validation = not validation_used or (trial_validation is not None and current_validation is not None and trial_validation < current_validation - cfg.acceptance_tolerance)
            preserves_holdout = trial_holdout <= baseline_holdout + cfg.acceptance_tolerance and trial_dynamic_holdout <= baseline_dynamic_holdout + cfg.acceptance_tolerance
            preserves_variance = variance_ratio >= cfg.collapse_variance_ratio
            stability = float(np.max(fit.metadata.get("ensemble_spread", [0.0]))) + float(np.max(fit.metadata.get("leave_one_probe_out_spread", [0.0])))
            stability_ok = stability <= cfg.max_stability_uncertainty
            if effect < cfg.min_observable_effect:
                ineffective_blocks.append(block)
            if solver_ok and improves_validation and preserves_holdout and preserves_variance and stability_ok and effect >= cfg.min_observable_effect:
                accepted_blocks.append(block)
                current_matrices = trial_matrices
                current_biases = trial_biases
                current_quadratic = trial_quadratic
                current_operator = trial_operator
                current_validation = trial_validation
                current_holdout = trial_holdout
                current_dynamic_validation = trial_dynamic_validation
                current_dynamic_holdout = trial_dynamic_holdout
            else:
                rollback_blocks.append(block)
        if accepted_blocks:
            current_alpha = alpha
            adaptive_step = min(cfg.hard_max_step_norm, adaptive_step * 1.25)
            status = "accepted" if len(rollback_blocks) == 0 else "partially_accepted"
        else:
            adaptive_step = max(cfg.max_step_norm * 0.25, adaptive_step * 0.5)
            status = "ineffective" if len(ineffective_blocks) == len(node_blocks) else "rejected"
        effect, variance_ratio = _observable_effect(current_operator, baseline, holdout_student)
        holdout_improvement = baseline_holdout - current_holdout
        dynamic_improvement = baseline_dynamic_holdout - current_dynamic_holdout
        diagnostics = constrained.diagnostics
        assessment = assess_correction(
            {"holdout_error": baseline_holdout, "dynamic_functional_error": baseline_dynamic_holdout},
            {"holdout_error": current_holdout, "dynamic_functional_error": current_dynamic_holdout, "stability": variance_ratio, "observable_effect": effect},
            uncertainty=1.0 - float(np.mean(constrained.confidence)),
            validation=None if current_validation is None or baseline_validation is None else {"improvement": baseline_validation - current_validation},
        )
        steps.append(
            AdaptiveStep(
                alpha,
                status,
                current_validation,
                current_holdout,
                baseline_holdout,
                current_dynamic_validation,
                current_dynamic_holdout,
                holdout_improvement,
                dynamic_improvement,
                effect,
                variance_ratio,
                _finite(np.mean(constrained.confidence)),
                _finite(np.max(diagnostics["spectral_norm"])),
                _finite(np.max(diagnostics["lipschitz"])),
                tuple(accepted_blocks),
                tuple(rollback_blocks),
                tuple(ineffective_blocks),
                ("coherent block rollback preserved the last accepted block state",),
                {
                    "adaptive_max_step_norm": adaptive_step,
                    "current_alpha": current_alpha,
                    "block_effects": block_effects,
                    "stability_uncertainty": stability,
                    "target_fit_split": "train",
                    "assessor": assessment.to_dict(),
                },
            )
        )
        if not accepted_blocks and alpha > current_alpha:
            break
    report_metadata = {"policy_mode": cfg.mode, "target_fit_split": "train", "metric_scope": FLOW_METRIC_SCOPE, "dynamic_metric_scope": DYNAMIC_METRIC_SCOPE, "block_partition": node_blocks, "gain_grid": cfg.gain_grid, "final_alpha": current_alpha, "baseline_guard": "validation improvement plus non-degrading holdout/operator and one-step metrics", "semantic_claim": "not measured"}
    report_metadata["math_profile"] = build_math_profile(fit.metadata, alignment_kind=str(fit.metadata.get("alignment_kind", "unknown")), signature_mode=str(fit.metadata.get("signature_mode", "full")), correction_matrices=fit.correction_matrices, correction_quadratic=fit.correction_quadratic, policy={"mode": "adaptive_transfer", "final_alpha": current_alpha})
    return AdaptiveTransferReport("accepted" if current_alpha > 0 else "unchanged", cfg.mode, validation_used, holdout_used, tuple(steps), current_operator, report_metadata)


def _strict_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _gpt2_gain_label(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def _load_gpt2_mapping(path: str | Path) -> list[Any]:
    """Load the explicit family mapping at the orchestration boundary."""

    from .gpt2 import GPT2TensorLiftMapping

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("GPT-2 mapping must be a JSON list")
    return [
        GPT2TensorLiftMapping(
            int(item["transition_index"]),
            str(item["tensor_name"]),
            side=str(item.get("side", "input")),
            block_index=None if item.get("block_index") is None else int(item["block_index"]),
        )
        for item in payload
    ]


def _text_ab_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    comparison = dict(report.get("comparison") or {})
    return {
        "status": report.get("status"),
        "candidate_minus_baseline_mean_target_nll": comparison.get("candidate_minus_baseline_mean_target_nll"),
        "candidate_minus_baseline_mean_target_perplexity": comparison.get("candidate_minus_baseline_mean_target_perplexity"),
        "candidate_minus_baseline_greedy_token_accuracy": comparison.get("candidate_minus_baseline_greedy_token_accuracy"),
        "mean_next_token_logit_relative_l2": comparison.get("mean_next_token_logit_relative_l2"),
        "tokenizer_equal": report.get("models", {}).get("baseline", {}).get("tokenizer_fingerprint") == report.get("models", {}).get("candidate", {}).get("tokenizer_fingerprint"),
    }


def run_gpt2_weight_tune(
    checkpoint_dir: str | Path,
    student_trace: str | Path,
    teacher_trace: str | Path,
    alignment: str | Path,
    mapping: str | Path | Sequence[Any],
    output_root: str | Path,
    *,
    sequence_length: int,
    gains: Sequence[float] = (0.01, 0.05, 0.1),
    mode: str = "safe",
    device: str = "auto",
    seed: int = 17,
    max_dense_features: int = 5_000_000,
    scalable_rank: int = 64,
    scalable_seed: int = 17,
    max_chart_bytes: int = 512 * 1024 * 1024,
    max_cross_token_ratio: float = 0.5,
    activation_inputs: Mapping[str, np.ndarray] | None = None,
    activation_target_deltas: Mapping[str, np.ndarray] | None = None,
    activation_ridge: float = 1e-6,
    activation_lift: str = "teacher_flow",
    teacher_flow_target_ratio: float = 0.05,
    site_policy: str = "dual_residual",
    dual_residual_ratio: float | str = 0.35,
    depth_schedule: str | Sequence[float] | None = "boost_deep",
    use_2jet: bool = True,
    teacher_checkpoint: str | Path | None = None,
    pilot_pulse: bool = False,
    piecewise_knots: bool = False,
    piecewise_num_pieces: int = 4,
    piecewise_tolerance: float = 0.35,
    piecewise_svd_rank_ratio: float = 0.50,
    piecewise_cosine_threshold: float = 0.50,
    piecewise_eta: float = 0.08,
    calibrate_lm_head: bool = False,
    teacher_lm_head: np.ndarray | None = None,
    lm_head_gain: float = 0.03,
    memory_imprint_mlp: bool = False,
    memory_imprint_gain: float = 1.0,
    flow_distill_steps: int = 0,
    flow_distill_lr: float = 1e-5,
) -> dict[str, Any]:
    """Run the central GPT-2 candidate state machine.

    The state machine is deliberately explicit: load local student → fit from
    train traces → build a shape-aware candidate for each gain → export an
    ordinary baseline/candidate pair → run the fixed text A/B evaluator →
    select by validation NLL and report holdout. ``experimental`` may
    materialize raw rejected flow nodes only through the explicit GPT-2
    mapping; ``safe`` uses the solver acceptance mask and can return no
    candidate. ``diagnostic`` builds no checkpoint and only records fit state.
    No teacher checkpoint or new observation is loaded here.
    """

    from .artifacts import load_alignment, load_traces
    from .connectors import CapabilityError
    from .flow import fit_flow_transfer
    from .gpt2 import build_gpt2_surgery_plan, build_gpt2_teacher_flow_activation_targets, export_gpt2_checkpoint_pair
    from .model_families import GPT2_XL_TO_SMALL, inspect_gpt2_variant
    from .solver import TrustRegionConfig, solve_flow_correction

    if mode not in {"diagnostic", "safe", "experimental", "experimental_untrusted"}:
        raise ValueError("GPT-2 tune mode must be diagnostic, safe, experimental, or experimental_untrusted")
    if (activation_inputs is None) != (activation_target_deltas is None):
        raise CapabilityError("exact GPT-2 tune activation lift requires both activation_inputs and activation_target_deltas")
    if not np.isfinite(activation_ridge) or activation_ridge < 0.0:
        raise ValueError("activation_ridge must be finite and non-negative")
    if activation_lift not in {"heuristic", "teacher_flow"}:
        raise ValueError("activation_lift must be 'heuristic' or 'teacher_flow'")
    if not np.isfinite(teacher_flow_target_ratio) or teacher_flow_target_ratio <= 0.0:
        raise ValueError("teacher_flow_target_ratio must be finite and positive")
    if activation_lift == "teacher_flow" and activation_inputs is not None:
        raise ValueError("activation_lift='teacher_flow' cannot be combined with caller-supplied activation maps")
    gain_values = tuple(float(value) for value in gains)
    if not gain_values or any(not np.isfinite(value) or value == 0.0 for value in gain_values):
        raise ValueError("GPT-2 tune gains must be finite and non-zero; use the exported baseline for alpha=0")
    checkpoint = Path(checkpoint_dir)
    if not checkpoint.is_dir():
        raise CapabilityError(f"student checkpoint directory does not exist locally: {checkpoint}")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    student = load_traces(student_trace)
    teacher = load_traces(teacher_trace)
    fitted_alignment = load_alignment(alignment)
    fit = fit_flow_transfer(
        student,
        teacher,
        fitted_alignment,
        signature_mode="full",
        max_dense_features=max_dense_features,
        scalable_rank=scalable_rank,
        scalable_seed=scalable_seed,
    )
    guarded = solve_flow_correction(fit, config=TrustRegionConfig())
    from dataclasses import replace as dataclass_replace

    try:
        import transformers
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional real-model dependency
        raise CapabilityError("GPT-2 tune requires optional torch and transformers; no checkpoint was loaded") from error
    try:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    except Exception as error:  # pragma: no cover - depends on local checkpoint
        raise CapabilityError(f"local GPT-2 checkpoint/tokenizer load failed with local_files_only=True: {error}") from error
    model.eval()

    from .model_families import inspect_transformer_architecture, generate_universal_mapping
    student_arch = inspect_transformer_architecture(getattr(model, "config", None))
    hidden_size = int(student_arch.hidden_size)
    student_blocks = int(student_arch.layers)
    if hidden_size < 1 or student_blocks < 1:
        raise CapabilityError("student model config does not expose valid hidden_size or layer count")

    if mode == "safe" and student_arch.model_type == "gpt2" and student_blocks == 12:
        architecture = inspect_gpt2_variant(getattr(model, "config", None), GPT2_XL_TO_SMALL.student)
        if not architecture["supported"]:
            raise CapabilityError("GPT-2 tune preflight rejected checkpoint: " + "; ".join(architecture["reasons"]))

    teacher_arch = None
    t_cand = Path(teacher_checkpoint) if teacher_checkpoint is not None else checkpoint.parent / "gpt2-xl"
    if t_cand.exists():
        teacher_arch = inspect_transformer_architecture(t_cand)

    if mapping is None or mapping == "auto" or (isinstance(mapping, str) and str(mapping).lower() == "auto"):
        if teacher_arch is None:
            from .model_families import TransformerArchitecture
            teacher_arch = TransformerArchitecture(
                model_type=student_arch.model_type,
                layers=student_arch.layers * 4,
                hidden_size=student_arch.hidden_size * 2,
                intermediate_size=student_arch.intermediate_size * 2,
                attention_heads=student_arch.attention_heads * 2,
                context_length=student_arch.context_length,
                vocab_size=student_arch.vocab_size,
                layer_prefix=student_arch.layer_prefix,
                attn_proj_suffix=student_arch.attn_proj_suffix,
                mlp_proj_suffix=student_arch.mlp_proj_suffix,
                is_conv1d=student_arch.is_conv1d,
            )
        from .gpt2 import GPT2TensorLiftMapping
        raw_map, _ = generate_universal_mapping(student_arch, teacher_arch)
        mapping_entries = [
            GPT2TensorLiftMapping(
                int(item["transition_index"]),
                str(item["tensor_name"]),
                side=str(item.get("side", "output")),
                block_index=int(item["block_index"]),
            )
            for item in raw_map
        ]
    elif isinstance(mapping, (str, Path)):
        mapping_entries = _load_gpt2_mapping(mapping)
    else:
        mapping_entries = list(mapping)

    activation_target_metadata: dict[str, Any] | None = None
    dissim_info: dict[str, Any] | None = None
    resolved_depth_schedule = depth_schedule
    if depth_schedule == "dissimilarity":
        if t_cand.exists():
            from .gpt2 import compute_universal_static_weight_dissimilarity
            if teacher_arch is None:
                teacher_arch = inspect_transformer_architecture(t_cand)
            dissim_info = compute_universal_static_weight_dissimilarity(
                model.state_dict(),
                t_cand,
                fitted_alignment,
                student_arch=student_arch,
                teacher_arch=teacher_arch,
                sequence_length=int(sequence_length),
            )
            resolved_depth_schedule = dissim_info["schedule"]

    if activation_lift == "teacher_flow":
        activation_inputs, activation_target_deltas, activation_target_metadata = build_gpt2_teacher_flow_activation_targets(
            model,
            student,
            fit.student,
            fit.transported_teacher,
            mapping_entries,
            sequence_length=int(sequence_length),
            hidden_size=hidden_size,
            target_effect_ratio=float(teacher_flow_target_ratio),
            site_policy=site_policy,
            dual_residual_ratio=dual_residual_ratio,
            depth_schedule=resolved_depth_schedule,
            use_2jet=use_2jet,
        )

    pilot_bound_info: dict[str, Any] | None = None
    if pilot_pulse and activation_inputs is not None and activation_target_deltas is not None:
        from .gpt2 import estimate_pilot_pulse_gain_bound
        pilot_bound_info = estimate_pilot_pulse_gain_bound(
            activation_inputs,
            activation_target_deltas,
        )

    raw = dataclass_replace(
        guarded,
        matrices=fit.correction_matrices.copy(),
        biases=fit.correction_biases.copy(),
        quadratic_terms=None if fit.correction_quadratic is None else fit.correction_quadratic.copy(),
    )
    plan_mode = {"diagnostic": "diagnostic", "safe": "apply", "experimental": "experimental", "experimental_untrusted": "experimental"}[mode]
    split_cases: dict[str, tuple[dict[str, str], ...]] = {}
    from scripts.text_ab_eval import DEFAULT_TUNE_CASES, run_ab

    if len(DEFAULT_TUNE_CASES) < 3 or len(DEFAULT_TUNE_CASES) % 3:
        raise RuntimeError("adaptive GPT-2 tune evaluation set must split evenly into train/validation/holdout")
    split_size = len(DEFAULT_TUNE_CASES) // 3
    split_cases.update({
        "train": tuple(DEFAULT_TUNE_CASES[:split_size]),
        "validation": tuple(DEFAULT_TUNE_CASES[split_size : 2 * split_size]),
        "holdout": tuple(DEFAULT_TUNE_CASES[2 * split_size :]),
    })
    records: list[dict[str, Any]] = []
    activation_ls_cache: dict[tuple[str, float], tuple[np.ndarray, dict[str, Any]]] = {}
    for gain in gain_values:
        label = _gpt2_gain_label(gain)
        pair_dir = output / f"gain-{label}"
        correction = raw if mode in {"experimental", "experimental_untrusted"} else guarded
        try:
            plan = build_gpt2_surgery_plan(
                model.state_dict(),
                correction,
                chart_projection=fit.student.chart_projection,
                sequence_length=int(sequence_length),
                hidden_size=hidden_size,
                mapping=mapping_entries,
                gain=gain,
                mode=plan_mode,
                max_chart_bytes=max_chart_bytes,
                max_cross_token_ratio=max_cross_token_ratio,
                activation_inputs=activation_inputs,
                activation_target_deltas=activation_target_deltas,
                activation_ridge=activation_ridge,
                activation_target_metadata=activation_target_metadata,
                activation_least_squares_cache=activation_ls_cache,
                piecewise_knots=piecewise_knots,
                piecewise_num_pieces=piecewise_num_pieces,
                piecewise_tolerance=piecewise_tolerance,
                piecewise_svd_rank_ratio=piecewise_svd_rank_ratio,
                piecewise_cosine_threshold=piecewise_cosine_threshold,
                piecewise_eta=piecewise_eta,
                calibrate_lm_head=calibrate_lm_head,
                teacher_lm_head=teacher_lm_head,
                lm_head_gain=lm_head_gain,
                memory_imprint_mlp=memory_imprint_mlp,
                memory_imprint_gain=memory_imprint_gain,
            )
        except CapabilityError as error:
            records.append({
                "gain": gain,
                "pair_directory": None,
                "export_seconds": 0.0,
                "status": "rejected",
                "reason": f"{type(error).__name__}: {error}",
                "plan": None,
                "text_ab": {},
            })
            continue
        split_reports: dict[str, Any] = {}
        if mode != "diagnostic":
            started = time.perf_counter()
            export_variant = "gpt2-small" if (student_arch.model_type == "gpt2" and student_blocks == 12) else "universal"
            export_gpt2_checkpoint_pair(
                model,
                pair_dir,
                variant=export_variant,
                experimental_plan=plan,
                tokenizer=tokenizer,
                metadata={
                    "adaptive_policy_mode": mode,
                    "gain": gain,
                    "source_traces": str(student_trace),
                    "teacher_traces": str(teacher_trace),
                    "alignment": str(alignment),
                    "observation_rerun": False,
                },
            )
            for split, cases in split_cases.items():
                report_path = pair_dir / f"text-ab-{split}.json"
                jsonl_path = pair_dir / f"text-ab-{split}.jsonl"
                report = run_ab(pair_dir / "baseline", pair_dir / "candidate", output=report_path, jsonl=jsonl_path, device=device, seed=seed, cases=cases)
                split_reports[split] = {"fixed_case_ids": [str(case["id"]) for case in cases], "summary": _text_ab_summary(report), "report": str(report_path), "jsonl": str(jsonl_path)}
            export_seconds = float(time.perf_counter() - started)
        else:
            export_seconds = 0.0
        records.append({
            "gain": gain,
            "pair_directory": str(pair_dir) if mode != "diagnostic" else None,
            "export_seconds": export_seconds,
            "status": "ok",
            "plan": {
                "mode": plan.metadata.get("mode"),
                "trust_status": plan.metadata.get("trust_status"),
                "acceptance_gate_bypassed": plan.metadata.get("acceptance_gate_bypassed"),
                "applied_tensors": list(plan.applied_tensors),
                "changed_tensor_details": list(plan.metadata.get("changed_tensor_details", ())),
                "rollback_layers": list(plan.rollback_layers),
                "rollback_reasons": dict(plan.metadata.get("rollback_reasons", {})),
                "skipped_tensors": dict(plan.skipped_tensors),
                "lift_reports": dict(plan.metadata.get("lift_reports", {})),
                "lift_method": plan.metadata.get("lift_method"),
                "exact_activation_ls_tensors": list(plan.metadata.get("exact_activation_ls_tensors", ())),
                "heuristic_chart_tensors": list(plan.metadata.get("heuristic_chart_tensors", ())),
            },
            "text_ab": split_reports,
        })
    scored = []
    for record in records:
        validation = record["text_ab"].get("validation", {}).get("summary", {})
        holdout = record["text_ab"].get("holdout", {}).get("summary", {})
        validation_delta = validation.get("candidate_minus_baseline_mean_target_nll")
        holdout_delta = holdout.get("candidate_minus_baseline_mean_target_nll")
        if validation.get("status") == "ok" and holdout.get("status") == "ok" and validation_delta is not None and holdout_delta is not None and float(holdout_delta) <= 0.0:
            scored.append(record)
    selected = None if not scored else min(scored, key=lambda record: (float(record["text_ab"]["validation"]["summary"]["candidate_minus_baseline_mean_target_nll"]), float(record["gain"])))
    status = "diagnostic" if mode == "diagnostic" else ("ok" if selected is not None else "no_valid_candidate")
    payload = {
        "schema_version": "faytuna-adaptive-gpt2-tune-v1",
        "status": status,
        "state_machine": ["local_preflight", "train_trace_fit", "shape_aware_lift", "atomic_pair_export", "validation_text_ab", "holdout_report", "selection"],
        "policy_mode": mode,
        "variant": export_variant,
        "checkpoint_dir": str(checkpoint),
        "architecture": {
            "student": student_arch.to_dict(),
            "teacher": teacher_arch.to_dict() if teacher_arch is not None else None,
            "student_blocks": student_blocks,
            "teacher_blocks": teacher_arch.layers if teacher_arch is not None else None,
        },
        "observation_rerun": False,
        "selection_rule": "minimum validation candidate_minus_baseline_mean_target_nll among candidates that do not degrade holdout",
        "target_fit_split": "train",
        "metric_scope": "transported_teacher_operator_approximation",
        "text_probe_set": {"name": "DEFAULT_TUNE_CASES", "case_count": len(DEFAULT_TUNE_CASES), "split_counts": {name: len(cases) for name, cases in split_cases.items()}, "disjoint_ids": True},
        "text_probe_limitations": "eighteen deterministic text probes split evenly into train/validation/holdout; text A/B is a behavioral proxy and is not semantic proof",
        "safe_solver_summary": {"accepted_nodes": int(np.count_nonzero(guarded.accepted)), "rollback_nodes": int(np.count_nonzero(~guarded.accepted)), "rollback_reasons": {str(index): str(reason) for index, reason in sorted(guarded.reasons.items())}},
        "selected_gain": None if selected is None else selected["gain"],
        "candidates": records,
        "runtime_validation": "stock llama.cpp remains a separate required stage",
        "semantic_claim": "not established",
        "activation_lift": {
            "requested": activation_lift,
            "ridge": float(activation_ridge),
            "teacher_flow_target_ratio": float(teacher_flow_target_ratio),
            "contract": "exact only with per-tensor hook inputs and separately defined target activation deltas; trace-only fitting cannot infer them",
            "target_metadata": activation_target_metadata,
        },
        "weight_dissimilarity": dissim_info,
        "pilot_pulse_bound": pilot_bound_info,
        "piecewise_knots": {
            "enabled": bool(piecewise_knots),
            "num_pieces": int(piecewise_num_pieces),
            "tolerance": float(piecewise_tolerance),
            "svd_rank_ratio": float(piecewise_svd_rank_ratio),
            "cosine_threshold": float(piecewise_cosine_threshold),
            "eta": float(piecewise_eta),
        },
    }
    _strict_json(output / "tune.json", payload)
    summary_lines = ["Adaptive GPT-2 tune", f"mode: {mode}", "selection: validation NLL subject to non-degrading holdout", "gain | validation delta NLL | holdout delta NLL | status", "--- | ---: | ---: | ---"]
    for record in records:
        validation = record["text_ab"].get("validation", {}).get("summary", {})
        holdout = record["text_ab"].get("holdout", {}).get("summary", {})
        summary_lines.append(f"{record['gain']} | {validation.get('candidate_minus_baseline_mean_target_nll')} | {holdout.get('candidate_minus_baseline_mean_target_nll')} | {validation.get('status')}")
    summary_lines.extend(["", f"selected_gain: {None if selected is None else selected['gain']}", f"status: {status}; semantic claim not established."])
    (output / "tune.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return payload


run_universal_flow_tune = run_gpt2_weight_tune
run_transformer_weight_tune = run_gpt2_weight_tune
