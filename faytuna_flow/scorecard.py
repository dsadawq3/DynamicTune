"""Finite, holdout-first accuracy scorecards for emergent-flow transfer.

The scorecard is deliberately separate from fitting.  Fitting produces a
candidate in the student chart; this module decides whether that candidate is
useful on disjoint probes.  A geometric match is therefore never reported as
an operational improvement by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .depth import MonotoneCorrespondence, monotone_correspondence
from .flow import _spectral_norm_diagnostic, calibrate_flow_confidence, fit_flow_transfer
from .geometry import fit_low_rank_map, relational_error
from .scalable import DEFAULT_MAX_DENSE_FEATURES, data_aware_chart_projection
from .math_profile import MATH_COMPONENTS, compare_math_profiles
from .solver import ConstrainedCorrection, TrustRegionConfig, solve_flow_correction
from .types import AlignmentResult, FlowFitResult, FlowOperator, TrajectoryTrace, TransitionObservation, stable_l2


SCORECARD_SCHEMA = "faytuna-scorecard-v1"
TARGET_FIT_SPLIT = "train"
FLOW_METRIC_SCOPE = "transported_teacher_operator_approximation"
DYNAMIC_METRIC_SCOPE = "heldout_one_step_transition_prediction"


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _jsonable(value: Any) -> Any:
    """Convert metadata to strict JSON values; non-finite numbers become null."""

    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return _finite(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _paired_ids(student: Sequence[TrajectoryTrace], teacher: Sequence[TrajectoryTrace], label: str) -> set[str]:
    if len(student) != len(teacher) or not student:
        raise ValueError(f"{label} teacher/student traces must be paired and non-empty")
    student_ids = [str(trace.probe_id) for trace in student]
    teacher_ids = [str(trace.probe_id) for trace in teacher]
    if student_ids != teacher_ids:
        raise ValueError(f"{label} teacher/student probe IDs are not paired")
    if len(set(student_ids)) != len(student_ids):
        raise ValueError(f"{label} contains duplicate probe IDs")
    return set(student_ids)


def _trace_content_fingerprint(student: TrajectoryTrace, teacher: TrajectoryTrace) -> str:
    """Fingerprint observed pair content independently of probe IDs.

    Split labels and IDs alone are insufficient: a faulty collector can write
    the same deterministic trace under three different IDs.  This exact
    content check does not claim semantic identity; it only rejects a direct
    artifact duplication that would make held-out scores optimistic.
    """

    digest = hashlib.sha256()
    for trace in (student, teacher):
        for value in (trace.hidden_states, trace.depth_coordinates):
            array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        for value in (trace.token_ids, trace.position_ids):
            if value is None:
                digest.update(b"<none>")
            else:
                array = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes())
    return digest.hexdigest()


def _paired_content_fingerprints(student: Sequence[TrajectoryTrace], teacher: Sequence[TrajectoryTrace]) -> set[str]:
    return {_trace_content_fingerprint(s_item, t_item) for s_item, t_item in zip(student, teacher)}


def _check_splits(
    train_student: Sequence[TrajectoryTrace],
    train_teacher: Sequence[TrajectoryTrace],
    validation_student: Sequence[TrajectoryTrace] | None,
    validation_teacher: Sequence[TrajectoryTrace] | None,
    holdout_student: Sequence[TrajectoryTrace] | None,
    holdout_teacher: Sequence[TrajectoryTrace] | None,
) -> tuple[set[str], set[str], set[str]]:
    train_ids = _paired_ids(train_student, train_teacher, "train")
    validation_ids: set[str] = set()
    holdout_ids: set[str] = set()
    if (validation_student is None) != (validation_teacher is None):
        raise ValueError("validation teacher/student traces must be supplied together")
    if (holdout_student is None) != (holdout_teacher is None):
        raise ValueError("holdout teacher/student traces must be supplied together")
    if validation_student is not None and validation_teacher is not None:
        validation_ids = _paired_ids(validation_student, validation_teacher, "validation")
        if train_ids & validation_ids:
            raise ValueError("validation probes overlap the training probes")
        duplicated = _paired_content_fingerprints(train_student, train_teacher) & _paired_content_fingerprints(validation_student, validation_teacher)
        if duplicated:
            raise ValueError("validation contains exact observed teacher/student trace content duplicated from train")
    if holdout_student is not None and holdout_teacher is not None:
        holdout_ids = _paired_ids(holdout_student, holdout_teacher, "holdout")
        if train_ids & holdout_ids or validation_ids & holdout_ids:
            raise ValueError("holdout probes overlap train or validation probes")
        duplicated = _paired_content_fingerprints(train_student, train_teacher) & _paired_content_fingerprints(holdout_student, holdout_teacher)
        if duplicated:
            raise ValueError("holdout contains exact observed teacher/student trace content duplicated from train")
        if validation_student is not None and validation_teacher is not None:
            duplicated = _paired_content_fingerprints(validation_student, validation_teacher) & _paired_content_fingerprints(holdout_student, holdout_teacher)
            if duplicated:
                raise ValueError("holdout contains exact observed teacher/student trace content duplicated from validation")
    return train_ids, validation_ids, holdout_ids


def _resampled_teacher_states(student: TrajectoryTrace, teacher: TrajectoryTrace, alignment: AlignmentResult) -> np.ndarray:
    mapped = alignment.apply(teacher.hidden_states, depth=teacher.depth_coordinates)
    return np.column_stack([
        np.interp(student.depth_coordinates, teacher.depth_coordinates, mapped[:, column])
        for column in range(mapped.shape[1])
    ])


def _perturb_traces(
    traces: Sequence[TrajectoryTrace],
    *,
    noise: float = 0.0,
    quantization_step: float | None = None,
    seed: int = 0,
) -> list[TrajectoryTrace]:
    rng = np.random.default_rng(seed)
    result: list[TrajectoryTrace] = []
    for trace in traces:
        states = trace.hidden_states.copy()
        if noise:
            states += noise * max(1.0, float(np.std(states))) * rng.normal(size=states.shape)
        if quantization_step is not None:
            if quantization_step <= 0 or not np.isfinite(quantization_step):
                raise ValueError("quantization_step must be positive and finite")
            states = np.round(states / quantization_step) * quantization_step
        transitions = []
        for index, old in enumerate(trace.transitions):
            delta = states[index + 1] - states[index]
            ds = old.target_depth - old.source_depth
            transitions.append(TransitionObservation(old.source_layer, old.target_layer, old.source_depth, old.target_depth, states[index], states[index + 1], delta, delta / ds, old.jacobian, old.hessian_sketch, old.curvature, old.singular_values, old.normalization_geometry, old.attention_geometry, old.uncertainty))
        result.append(replace(trace, hidden_states=states, transitions=tuple(transitions)))
    return result


def _state_geometry_error(student: Sequence[TrajectoryTrace], teacher: Sequence[TrajectoryTrace], alignment: AlignmentResult) -> float:
    errors: list[float] = []
    for student_trace, teacher_trace in zip(student, teacher):
        target = _resampled_teacher_states(student_trace, teacher_trace, alignment)
        numerator = float(stable_l2(student_trace.hidden_states - target, name="state geometry residual"))
        denominator = max(1.0, float(stable_l2(student_trace.hidden_states, name="student state geometry")), float(stable_l2(target, name="teacher state geometry")))
        errors.append(numerator / denominator)
    return float(np.mean(errors)) if errors else 0.0


def _operator_error(operator: FlowOperator, target: FlowOperator, student: Sequence[TrajectoryTrace]) -> float:
    errors: list[float] = []
    for trace in student:
        for transition in trace.transitions:
            prediction = operator.predict(transition.source_state, transition.source_depth)
            target_value = target.predict(transition.source_state, transition.source_depth)
            errors.append(float(stable_l2(prediction - target_value, name="operator residual") / max(1.0, float(stable_l2(target_value, name="operator target")))))
    if not errors:
        raise ValueError("cannot score an empty transition set")
    result = float(np.mean(errors))
    if not np.isfinite(result):
        raise FloatingPointError("operator score is non-finite")
    return result


def _operator_fingerprint(operator: FlowOperator) -> str:
    """Stable identity for a fitted target, without serializing its arrays."""

    digest = hashlib.sha256()
    for value in (operator.coordinates, operator.matrices, operator.biases, operator.sample_counts, operator.residual_scales):
        array = np.asarray(value, dtype=np.float64)
        digest.update(array.shape.__repr__().encode("ascii"))
        digest.update(array.tobytes())
    if operator.quadratic_terms is not None:
        digest.update(operator.quadratic_terms.shape.__repr__().encode("ascii"))
        digest.update(np.asarray(operator.quadratic_terms, dtype=np.float64).tobytes())
    if operator.chart_projection is not None:
        digest.update(operator.chart_projection.shape.__repr__().encode("ascii"))
        digest.update(np.asarray(operator.chart_projection, dtype=np.float64).tobytes())
    return digest.hexdigest()


def fit_target_operator(
    train_student: Sequence[TrajectoryTrace],
    train_teacher: Sequence[TrajectoryTrace],
    alignment: AlignmentResult,
    mode: str,
) -> FlowOperator:
    """Fit the fixed teacher target exclusively from the training split.

    This function is intentionally separate from evaluation.  A validation or
    holdout trace can be used to score this operator, but can never refit or
    replace it.
    """

    target = fit_flow_transfer(train_student, train_teacher, alignment, signature_mode=mode).transported_teacher
    metadata = dict(target.metadata)
    metadata.update({
        "target_fit_split": TARGET_FIT_SPLIT,
        "metric_scope": FLOW_METRIC_SCOPE,
        "target_operator_fingerprint": _operator_fingerprint(target),
    })
    return replace(target, metadata=metadata)


def _target_flow(
    train_student: Sequence[TrajectoryTrace],
    train_teacher: Sequence[TrajectoryTrace],
    alignment: AlignmentResult,
    mode: str,
) -> FlowOperator:
    """Compatibility alias for the train-only target fitting function."""

    return fit_target_operator(train_student, train_teacher, alignment, mode)


def _normalized_depth_map(
    student: TrajectoryTrace,
    teacher: TrajectoryTrace,
    correspondence: MonotoneCorrespondence | None = None,
    alignment: AlignmentResult | None = None,
) -> np.ndarray:
    """Map teacher coordinates to the student coordinate domain using monotone correspondence."""

    if len(student.depth_coordinates) == len(teacher.depth_coordinates) and np.allclose(student.depth_coordinates, teacher.depth_coordinates):
        return student.depth_coordinates.copy()
    if correspondence is None and alignment is not None and len(student.depth_coordinates) != len(teacher.depth_coordinates):
        try:
            mapped_teacher = alignment.apply(teacher.hidden_states, depth=teacher.depth_coordinates)
            correspondence = monotone_correspondence(
                student.hidden_states,
                mapped_teacher,
                student_coordinates=student.depth_coordinates,
                teacher_coordinates=teacher.depth_coordinates,
            )
        except Exception:
            correspondence = None
    if correspondence is not None and correspondence.pairs:
        pairs = sorted(correspondence.pairs, key=lambda pair: pair[1])
        teacher_index = np.asarray([pair[1] for pair in pairs], dtype=int)
        student_index = np.asarray([pair[0] for pair in pairs], dtype=int)
        _, unique_indices = np.unique(teacher_index, return_index=True)
        unique_indices = np.sort(unique_indices)
        t_coords = teacher.depth_coordinates[teacher_index[unique_indices]]
        s_coords = student.depth_coordinates[student_index[unique_indices]]
        if len(t_coords) > 1 and np.all(np.diff(t_coords) > 0):
            return np.interp(teacher.depth_coordinates, t_coords, s_coords)
    teacher_start, teacher_end = float(teacher.depth_coordinates[0]), float(teacher.depth_coordinates[-1])
    student_start, student_end = float(student.depth_coordinates[0]), float(student.depth_coordinates[-1])
    normalized = (teacher.depth_coordinates - teacher_start) / max(teacher_end - teacher_start, 1e-12)
    return student_start + normalized * (student_end - student_start)


def _dynamic_one_step_error(
    operator: FlowOperator,
    student: Sequence[TrajectoryTrace],
    teacher: Sequence[TrajectoryTrace],
    alignment: AlignmentResult,
    correspondences: Sequence[MonotoneCorrespondence | None] | None = None,
) -> float:
    """Score one-step prediction against observed teacher transitions.

    The comparison is operational and chart-dependent: teacher endpoints are
    transported into the student chart, and the candidate is evaluated at the
    transported teacher source.  It is deliberately separate from state
    alignment and from a causal rerun through model weights.
    """

    if len(student) != len(teacher) or not student:
        raise ValueError("paired traces are required for one-step scoring")
    errors: list[float] = []
    for idx_pair, (student_trace, teacher_trace) in enumerate(zip(student, teacher)):
        corr = correspondences[idx_pair] if correspondences is not None and idx_pair < len(correspondences) else None
        mapped_depth = _normalized_depth_map(student_trace, teacher_trace, correspondence=corr, alignment=alignment)
        for index, transition in enumerate(teacher_trace.transitions):
            ds = float(mapped_depth[index + 1] - mapped_depth[index])
            if ds <= 1e-12:
                continue
            source = alignment.apply(transition.source_state, depth=transition.source_depth)
            expected_velocity = alignment.linear_apply(transition.target_state - transition.source_state, depth=transition.source_depth) / ds
            prediction = operator.predict(source, float(mapped_depth[index]))
            field_scale = max(1.0, float(stable_l2(expected_velocity, name="one-step teacher velocity")))
            field_error = float(stable_l2(prediction - expected_velocity, name="one-step field residual") / field_scale)
            predicted_endpoint = source + ds * prediction
            expected_endpoint = alignment.apply(transition.target_state, depth=transition.target_depth)
            endpoint_scale = max(1.0, float(stable_l2(expected_endpoint - source, name="one-step endpoint displacement")))
            endpoint_error = float(stable_l2(predicted_endpoint - expected_endpoint, name="one-step endpoint residual") / endpoint_scale)
            errors.append(0.5 * (field_error + endpoint_error))
    if not errors:
        raise ValueError("cannot score an empty one-step transition set")
    result = float(np.mean(errors))
    if not np.isfinite(result):
        raise FloatingPointError("one-step score is non-finite")
    return result


def _operator_with_correction(fit: FlowFitResult, constrained: ConstrainedCorrection) -> FlowOperator:
    matrices = fit.student.matrices + constrained.matrices
    biases = fit.student.biases + constrained.biases
    quadratic = None
    if fit.student.quadratic_terms is not None:
        delta = constrained.quadratic_terms if constrained.quadratic_terms is not None else np.zeros_like(fit.student.quadratic_terms)
        quadratic = fit.student.quadratic_terms + delta
    # ``solve_flow_correction`` already computed the candidate spectral
    # diagnostics. Reuse them instead of decomposing every node again for
    # each validation/holdout trial. Unaccepted nodes remain the unchanged
    # student baseline and therefore reuse its stored norms.
    spectral = np.asarray(fit.student.spectral_norms, dtype=np.float64).copy()
    reported = constrained.diagnostics.get("spectral_norm")
    if reported is not None and np.asarray(reported).shape == spectral.shape:
        spectral[np.asarray(constrained.accepted, dtype=bool)] = np.asarray(reported, dtype=np.float64)[np.asarray(constrained.accepted, dtype=bool)]
    return FlowOperator(
        fit.student.coordinates,
        matrices,
        biases,
        fit.student.sample_counts,
        fit.student.residual_scales,
        spectral,
        {**fit.student.metadata, "guarded_correction": True},
        quadratic,
        fit.student.chart_projection,
    )


def _zero_constrained(fit: FlowFitResult, reasons: Mapping[int, str] | None = None) -> ConstrainedCorrection:
    count = len(fit.correction_matrices)
    quadratic = None if fit.correction_quadratic is None else np.zeros_like(fit.correction_quadratic)
    return ConstrainedCorrection(
        np.zeros_like(fit.correction_matrices),
        np.zeros_like(fit.correction_biases),
        np.zeros(count, dtype=bool),
        dict(reasons or {index: "rolled back by holdout baseline guard" for index in range(count)}),
        fit.confidence.copy(),
        {"spectral_norm": np.zeros(count), "lipschitz": np.zeros(count), "quadratic_jacobian_bound": np.zeros(count), "collapse_ratio": np.zeros(count), "quadratic_norm": np.zeros(count)},
        quadratic,
    )


def _guard_candidate(
    fit: FlowFitResult,
    validation: tuple[Sequence[TrajectoryTrace], Sequence[TrajectoryTrace], FlowOperator, str] | None,
    holdout: tuple[Sequence[TrajectoryTrace], Sequence[TrajectoryTrace], FlowOperator, str] | None,
    *,
    config: TrustRegionConfig,
    validation_functional: tuple[Sequence[TrajectoryTrace], Sequence[TrajectoryTrace], AlignmentResult] | None = None,
    holdout_functional: tuple[Sequence[TrajectoryTrace], Sequence[TrajectoryTrace], AlignmentResult] | None = None,
) -> tuple[FlowOperator, ConstrainedCorrection, float | None, float | None, float | None, float | None, int, tuple[str, ...]]:
    """Apply solver bounds, then keep only corrections supported by split scores.

    The greedy fallback is intentionally conservative.  It provides local
    rollback evidence when a complete correction is only train-fitting.
    """

    baseline = fit.student
    baseline_validation = None if validation is None else _operator_error(baseline, validation[2], validation[0])
    baseline_holdout = None if holdout is None else _operator_error(baseline, holdout[2], holdout[0])
    baseline_validation_functional = None if validation_functional is None else _dynamic_one_step_error(baseline, validation_functional[0], validation_functional[1], validation_functional[2])
    baseline_holdout_functional = None if holdout_functional is None else _dynamic_one_step_error(baseline, holdout_functional[0], holdout_functional[1], holdout_functional[2])
    constrained = solve_flow_correction(fit, config=config)
    if validation is None and holdout is None:
        return _operator_with_correction(fit, constrained), constrained, baseline_validation, baseline_holdout, None, None, int(np.count_nonzero(~constrained.accepted)), ("no disjoint evaluation split; correction is unguarded",)

    candidate = _operator_with_correction(fit, constrained)
    candidate_validation = None if validation is None else _operator_error(candidate, validation[2], validation[0])
    candidate_holdout = None if holdout is None else _operator_error(candidate, holdout[2], holdout[0])
    candidate_validation_functional = None if validation_functional is None else _dynamic_one_step_error(candidate, validation_functional[0], validation_functional[1], validation_functional[2])
    candidate_holdout_functional = None if holdout_functional is None else _dynamic_one_step_error(candidate, holdout_functional[0], holdout_functional[1], holdout_functional[2])

    def acceptable(value: float | None, reference: float | None) -> bool:
        return value is not None and reference is not None and value <= reference + 1e-12

    def strictly_improves(value: float | None, reference: float | None) -> bool:
        return value is not None and reference is not None and value < reference - 1e-12

    complete_ok = strictly_improves(candidate_validation, baseline_validation) and acceptable(candidate_holdout, baseline_holdout)
    if validation_functional is not None:
        complete_ok = complete_ok and acceptable(candidate_validation_functional, baseline_validation_functional)
    if holdout_functional is not None:
        complete_ok = complete_ok and acceptable(candidate_holdout_functional, baseline_holdout_functional)
    if validation is None:
        complete_ok = strictly_improves(candidate_holdout, baseline_holdout)
        if holdout_functional is not None:
            complete_ok = complete_ok and acceptable(candidate_holdout_functional, baseline_holdout_functional)
    if holdout is None:
        complete_ok = strictly_improves(candidate_validation, baseline_validation)
        if validation_functional is not None:
            complete_ok = complete_ok and acceptable(candidate_validation_functional, baseline_validation_functional)
    if complete_ok:
        return candidate, constrained, baseline_validation, baseline_holdout, candidate_validation, candidate_holdout, int(np.count_nonzero(~constrained.accepted)), ("complete correction passed the available split baseline guard",)

    # If the full candidate fails, test each solver-approved node in isolation
    # and keep only monotone improvements.  Every rejected node is explicit.
    current_matrices = np.zeros_like(fit.correction_matrices)
    current_biases = np.zeros_like(fit.correction_biases)
    current_quadratic = None if constrained.quadratic_terms is None else np.zeros_like(constrained.quadratic_terms)
    accepted = np.zeros_like(constrained.accepted)
    reasons = dict(constrained.reasons)
    current_operator = fit.student
    current_validation = baseline_validation
    current_holdout = baseline_holdout
    current_validation_functional = baseline_validation_functional
    current_holdout_functional = baseline_holdout_functional
    for index in range(len(accepted)):
        if not constrained.accepted[index]:
            reasons.setdefault(index, "solver rejected before baseline guard")
            continue
        trial_matrices = current_matrices.copy()
        trial_biases = current_biases.copy()
        trial_matrices[index] = constrained.matrices[index]
        trial_biases[index] = constrained.biases[index]
        trial_quadratic = None
        if current_quadratic is not None:
            trial_quadratic = current_quadratic.copy()
            trial_quadratic[index] = constrained.quadratic_terms[index]
        trial = _operator_with_correction(fit, ConstrainedCorrection(trial_matrices, trial_biases, accepted, reasons, constrained.confidence, constrained.diagnostics, trial_quadratic))
        trial_validation = None if validation is None else _operator_error(trial, validation[2], validation[0])
        trial_holdout = None if holdout is None else _operator_error(trial, holdout[2], holdout[0])
        trial_validation_functional = None if validation_functional is None else _dynamic_one_step_error(trial, validation_functional[0], validation_functional[1], validation_functional[2])
        trial_holdout_functional = None if holdout_functional is None else _dynamic_one_step_error(trial, holdout_functional[0], holdout_functional[1], holdout_functional[2])
        improves_primary = (validation is not None and trial_validation is not None and current_validation is not None and trial_validation < current_validation - 1e-12) or (validation is None and trial_holdout is not None and current_holdout is not None and trial_holdout < current_holdout - 1e-12)
        preserves_secondary = holdout is None or (trial_holdout is not None and baseline_holdout is not None and trial_holdout <= baseline_holdout + 1e-12)
        if validation is not None and holdout is not None:
            preserves_secondary = trial_holdout is not None and baseline_holdout is not None and trial_holdout <= baseline_holdout + 1e-12
        if validation_functional is not None:
            preserves_secondary = preserves_secondary and trial_validation_functional is not None and current_validation_functional is not None and trial_validation_functional <= baseline_validation_functional + 1e-12
        if holdout_functional is not None:
            preserves_secondary = preserves_secondary and trial_holdout_functional is not None and baseline_holdout_functional is not None and trial_holdout_functional <= baseline_holdout_functional + 1e-12
        if improves_primary and preserves_secondary:
            accepted[index] = True
            current_matrices[index] = constrained.matrices[index]
            current_biases[index] = constrained.biases[index]
            if current_quadratic is not None:
                current_quadratic[index] = constrained.quadratic_terms[index]
            current_operator = trial
            current_validation = trial_validation
            current_holdout = trial_holdout
            current_validation_functional = trial_validation_functional
            current_holdout_functional = trial_holdout_functional
        else:
            reasons[index] = "rolled back by disjoint baseline guard"
    guarded = ConstrainedCorrection(current_matrices, current_biases, accepted, reasons, constrained.confidence, constrained.diagnostics, current_quadratic)
    rollback = int(np.count_nonzero(~accepted))
    notes = ("complete candidate failed a split baseline guard; node-level rollback was attempted",)
    return current_operator, guarded, baseline_validation, baseline_holdout, current_validation, current_holdout, rollback, notes


@dataclass(frozen=True)
class ScorecardEntry:
    name: str
    status: str
    metric: str
    geometric_error: float | None = None
    train_error: float | None = None
    validation_error: float | None = None
    holdout_error: float | None = None
    baseline_train_error: float | None = None
    baseline_validation_error: float | None = None
    baseline_holdout_error: float | None = None
    improvement: float | None = None
    relative_improvement: float | None = None
    validation_improvement: float | None = None
    holdout_improvement: float | None = None
    dynamic_functional_error: float | None = None
    baseline_dynamic_functional_error: float | None = None
    dynamic_functional_improvement: float | None = None
    mean_confidence: float | None = None
    correction_norm: float | None = None
    confidence_reliability: float | None = None
    confidence_calibration_gap: float | None = None
    capacity_gate: float | None = None
    capacity_bottleneck: bool | None = None
    accepted_nodes: int = 0
    rollback_nodes: int = 0
    reason: str | None = None
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "name": self.name,
            "status": self.status,
            "metric": self.metric,
            "geometric_error": self.geometric_error,
            "train_error": self.train_error,
            "validation_error": self.validation_error,
            "holdout_error": self.holdout_error,
            "baseline_train_error": self.baseline_train_error,
            "baseline_validation_error": self.baseline_validation_error,
            "baseline_holdout_error": self.baseline_holdout_error,
            "improvement": self.improvement,
            "relative_improvement": self.relative_improvement,
            "validation_improvement": self.validation_improvement,
            "holdout_improvement": self.holdout_improvement,
            "dynamic_functional_error": self.dynamic_functional_error,
            "baseline_dynamic_functional_error": self.baseline_dynamic_functional_error,
            "dynamic_functional_improvement": self.dynamic_functional_improvement,
            "mean_confidence": self.mean_confidence,
            "correction_norm": self.correction_norm,
            "confidence_reliability": self.confidence_reliability,
            "confidence_calibration_gap": self.confidence_calibration_gap,
            "capacity_gate": self.capacity_gate,
            "capacity_bottleneck": self.capacity_bottleneck,
            "accepted_nodes": self.accepted_nodes,
            "rollback_nodes": self.rollback_nodes,
            "reason": self.reason,
            "notes": self.notes,
            "metadata": self.metadata or {},
        })


@dataclass(frozen=True)
class ScorecardReport:
    entries: tuple[ScorecardEntry, ...]
    validation_used: bool
    holdout_used: bool
    metadata: Mapping[str, Any]
    schema_version: str = SCORECARD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        complete = bool(self.holdout_used)
        return _jsonable({
            "schema_version": self.schema_version,
            "status": "ok" if complete else "incomplete",
            "decision_status": "holdout_decision_available" if complete else "insufficient_disjoint_holdout",
            "validation_used": self.validation_used,
            "holdout_used": self.holdout_used,
            "entries": [entry.to_dict() for entry in self.entries],
            "metadata": self.metadata,
        })


def _entry_from_fit(
    name: str,
    fit: FlowFitResult,
    train_student: Sequence[TrajectoryTrace],
    train_teacher: Sequence[TrajectoryTrace],
    fit_alignment: AlignmentResult,
    mode: str,
    validation_student: Sequence[TrajectoryTrace] | None,
    validation_teacher: Sequence[TrajectoryTrace] | None,
    holdout_student: Sequence[TrajectoryTrace] | None,
    holdout_teacher: Sequence[TrajectoryTrace] | None,
    evaluation_alignment: AlignmentResult,
    *,
    target_flow: FlowOperator | None = None,
    guard: bool = True,
    notes: tuple[str, ...] = (),
) -> ScorecardEntry:
    geometric = _state_geometry_error(train_student, train_teacher, fit_alignment)
    train_target = target_flow or _target_flow(train_student, train_teacher, evaluation_alignment, mode)
    target_fingerprint = _operator_fingerprint(train_target)
    baseline_train = _operator_error(fit.student, train_target, train_student)
    validation_tuple = None
    holdout_tuple = None
    if holdout_student is not None and holdout_teacher is not None:
        holdout_tuple = (holdout_student, holdout_teacher, train_target, mode)
    if validation_student is not None and validation_teacher is not None:
        validation_tuple = (validation_student, validation_teacher, train_target, mode)
    validation_functional = None if validation_student is None or validation_teacher is None else (validation_student, validation_teacher, evaluation_alignment)
    holdout_functional = None if holdout_student is None or holdout_teacher is None else (holdout_student, holdout_teacher, evaluation_alignment)
    if guard:
        candidate, constrained, base_validation, base_holdout, candidate_validation, candidate_holdout, rollback, guard_notes = _guard_candidate(
            fit,
            validation_tuple,
            holdout_tuple,
            config=TrustRegionConfig(min_confidence=0.01),
            validation_functional=validation_functional,
            holdout_functional=holdout_functional,
        )
    else:
        candidate = fit.student
        constrained = _zero_constrained(fit, {index: "reference baseline" for index in range(len(fit.correction_matrices))})
        base_validation = None if validation_tuple is None else _operator_error(fit.student, validation_tuple[2], validation_tuple[0])
        base_holdout = None if holdout_tuple is None else _operator_error(fit.student, holdout_tuple[2], holdout_tuple[0])
        candidate_validation, candidate_holdout = base_validation, base_holdout
        rollback = len(fit.correction_matrices)
        guard_notes = ()
    train_error = _operator_error(candidate, train_target, train_student)
    if validation_tuple is not None and candidate_validation is None:
        candidate_validation = _operator_error(candidate, validation_tuple[2], validation_tuple[0])
    if holdout_tuple is not None and candidate_holdout is None:
        candidate_holdout = _operator_error(candidate, holdout_tuple[2], holdout_tuple[0])
    baseline_dynamic_validation = None if validation_functional is None else _dynamic_one_step_error(fit.student, validation_functional[0], validation_functional[1], validation_functional[2])
    baseline_dynamic_holdout = None if holdout_functional is None else _dynamic_one_step_error(fit.student, holdout_functional[0], holdout_functional[1], holdout_functional[2])
    dynamic_validation = None if validation_functional is None else _dynamic_one_step_error(candidate, validation_functional[0], validation_functional[1], validation_functional[2])
    dynamic_holdout = None if holdout_functional is None else _dynamic_one_step_error(candidate, holdout_functional[0], holdout_functional[1], holdout_functional[2])
    baseline_dynamic = baseline_dynamic_holdout if baseline_dynamic_holdout is not None else baseline_dynamic_validation
    dynamic_error = dynamic_holdout if dynamic_holdout is not None else dynamic_validation
    dynamic_improvement = None if dynamic_error is None or baseline_dynamic is None else baseline_dynamic - dynamic_error
    validation_improvement = None if candidate_validation is None or base_validation is None else base_validation - candidate_validation
    holdout_improvement = None if candidate_holdout is None or base_holdout is None else base_holdout - candidate_holdout
    if holdout_improvement is None:
        improvement = None
        relative = None
        status = "insufficient_holdout" if validation_improvement is not None else "insufficient_data"
        reason = "a disjoint holdout split is required for an improvement or degradation decision"
    else:
        improvement = holdout_improvement
        relative = improvement / max(abs(base_holdout), 1e-12)
        if improvement > 1e-12:
            status = "improved"
            reason = None
        elif improvement < -1e-12:
            status = "degraded"
            reason = "candidate error is higher than its student-flow baseline"
        else:
            status = "unchanged"
            reason = "candidate did not reduce the selected holdout error"
    confidence = _finite(np.mean(fit.confidence))
    reliability = None if candidate_validation is None else _finite(np.exp(-np.clip(candidate_validation, 0.0, 700.0)))
    calibration_gap = None if confidence is None or reliability is None else abs(confidence - reliability)
    capacity_gate = _finite(fit.metadata.get("capacity_gate"))
    capacity_bottleneck = None if fit.capacity_diagnostics is None else bool(fit.capacity_diagnostics.bottleneck)
    metadata = {
        "fit_signature_mode": mode,
        "target_fit_split": TARGET_FIT_SPLIT,
        "metric_scope": FLOW_METRIC_SCOPE,
        "dynamic_metric_scope": DYNAMIC_METRIC_SCOPE,
        "target_operator_fingerprint": _operator_fingerprint(train_target),
        "evaluation_target_operator_fingerprint": target_fingerprint,
        "target_operator_source": "single_train_fitted_reference_target",
        "confidence_calibrated": bool(fit.metadata.get("confidence_calibrated", False)),
        "fit_alignment_direction": fit_alignment.metadata.get("direction"),
        "evaluation_alignment_direction": evaluation_alignment.metadata.get("direction"),
        "guarded": guard,
        "guard_notes": guard_notes,
        "depth_report": None if fit.depth_report is None else {
            "gap_count": fit.depth_report.gap_count,
            "matched_fraction": fit.depth_report.matched_fraction,
            "mean_gap_confidence": fit.depth_report.mean_gap_confidence,
        },
        "capacity": None if fit.capacity_diagnostics is None else {
            "state_subspace_coverage": fit.capacity_diagnostics.state_subspace_coverage,
            "velocity_subspace_coverage": fit.capacity_diagnostics.velocity_subspace_coverage,
            "residual_transport_error": fit.capacity_diagnostics.residual_transport_error,
            "irreducible_mismatch": fit.capacity_diagnostics.irreducible_mismatch,
            "map_condition_number": fit.capacity_diagnostics.map_condition_number,
        },
        "solver_reasons": constrained.reasons,
        "math_profile": fit.metadata.get("math_profile"),
    }
    return ScorecardEntry(
        name=name,
        status=status,
        metric="normalized_transported_flow_rmse",
        geometric_error=geometric,
        train_error=train_error,
        validation_error=candidate_validation,
        holdout_error=candidate_holdout,
        baseline_train_error=baseline_train,
        baseline_validation_error=base_validation,
        baseline_holdout_error=base_holdout,
        improvement=improvement,
        relative_improvement=relative,
        validation_improvement=validation_improvement,
        holdout_improvement=holdout_improvement,
        dynamic_functional_error=dynamic_error,
        baseline_dynamic_functional_error=baseline_dynamic,
        dynamic_functional_improvement=dynamic_improvement,
        mean_confidence=confidence,
        correction_norm=_finite(float(stable_l2(candidate.matrices - fit.student.matrices, name="scorecard correction")) + (0.0 if candidate.quadratic_terms is None else float(stable_l2(candidate.quadratic_terms, name="scorecard quadratic correction")))),
        confidence_reliability=reliability,
        confidence_calibration_gap=calibration_gap,
        capacity_gate=capacity_gate,
        capacity_bottleneck=capacity_bottleneck,
        accepted_nodes=int(np.count_nonzero(constrained.accepted)),
        rollback_nodes=rollback,
        reason=reason,
        notes=tuple(notes) + tuple(guard_notes),
        metadata=metadata,
    )


def _relabel_teacher(teacher: Sequence[TrajectoryTrace], student: Sequence[TrajectoryTrace], order: np.ndarray) -> list[TrajectoryTrace]:
    return [replace(teacher[int(source_index)], probe_id=student[target_index].probe_id) for target_index, source_index in enumerate(order)]


def _random_alignment(teacher: Sequence[TrajectoryTrace], student: Sequence[TrajectoryTrace], rng: np.random.Generator) -> AlignmentResult:
    source = np.asarray([trace.hidden_states[0] for trace in teacher])
    target = np.asarray([trace.hidden_states[0] for trace in student])
    if source.shape[1] * target.shape[1] > DEFAULT_MAX_DENSE_FEATURES:
        rank = min(64, int(np.sqrt(DEFAULT_MAX_DENSE_FEATURES)), source.shape[0] - 1, source.shape[1], target.shape[1])
        source_projection = data_aware_chart_projection(source, rank, int(rng.integers(0, 2**31 - 1)))
        target_projection = data_aware_chart_projection(target, rank, int(rng.integers(0, 2**31 - 1)))
        matrix = rng.normal(size=(rank, rank))
        matrix /= max(float(np.linalg.svd(matrix, compute_uv=False)[0]), 1e-12)
        source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
        mapped = (source - source_mean) @ source_projection @ matrix @ target_projection.T + target_mean
        singular = np.linalg.svd(matrix, compute_uv=False)
        condition = float(np.inf if singular[-1] <= 1e-12 else singular[0] / singular[-1])
        source_chart = (source - source_mean) @ source_projection
        target_chart = (target - target_mean) @ target_projection
        reverse, reverse_bias = fit_low_rank_map(target_chart, source_chart, rank=rank)
        cycle = float(np.mean(stable_l2(source_chart @ matrix @ reverse + reverse_bias - source_chart, axis=1, name="random-map compact cycle residual")))
        return AlignmentResult(
            "random_control", source_mean, target_mean, matrix, np.zeros(rank),
            int(np.linalg.matrix_rank(source - source_mean)), int(np.linalg.matrix_rank(target - target_mean)),
            float(np.mean(stable_l2(mapped - target, axis=1, name="random-map paired residual"))), relational_error(mapped, target), cycle, condition, 0.0,
            {"control": True, "direction": "teacher_to_student", "representation": "randomized_latent_compression", "dense_map_feature_count": int(source.shape[1] * target.shape[1]), "dense_map_skipped_reason": "random negative control also uses compact chart"},
            source_projection, target_projection,
        )
    matrix = rng.normal(size=(source.shape[1], target.shape[1]))
    random_norm, random_iterations, random_converged = _spectral_norm_diagnostic(matrix)
    matrix /= max(random_norm, 1e-12)
    bias = target.mean(axis=0) - source.mean(axis=0) @ matrix
    mapped = source @ matrix + bias
    # The random rectangular control is deliberately not assigned a costly
    # full spectrum. Its condition is conservatively unknown/infinite, while
    # the normalized spectral estimate and its convergence are logged.
    condition = float("inf")
    reverse, reverse_bias = fit_low_rank_map(target, source, rank=min(source.shape[1], target.shape[1]))
    cycle_residual = mapped @ reverse + reverse_bias - source
    cycle = float(np.mean(stable_l2(cycle_residual, axis=1, name="random-map cycle residual")))
    paired_residual = mapped - target
    paired_error = float(np.mean(stable_l2(paired_residual, axis=1, name="random-map paired residual")))
    return AlignmentResult(
        "random_control", source.mean(axis=0), target.mean(axis=0), matrix, bias,
        int(np.linalg.matrix_rank(source - source.mean(axis=0))),
        int(np.linalg.matrix_rank(target - target.mean(axis=0))),
        paired_error,
        relational_error(mapped, target), cycle, condition, 0.0,
        {"control": True, "direction": "teacher_to_student", "spectral_norm_method": "deterministic_power_iteration", "spectral_norm_estimate": random_norm, "spectral_norm_iterations": random_iterations, "spectral_norm_converged": random_converged, "condition_number_status": "conservative_infinite_unknown_spectrum"},
    )


def _rejected_entry(name: str, student: Sequence[TrajectoryTrace], teacher: Sequence[TrajectoryTrace], alignment: AlignmentResult, error: Exception, notes: tuple[str, ...]) -> ScorecardEntry:
    try:
        geometric = _finite(_state_geometry_error(student, teacher, alignment))
    except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError):
        geometric = None
    return ScorecardEntry(
        name=name,
        status="rejected",
        metric="normalized_transported_flow_rmse",
        geometric_error=geometric,
        reason=f"{type(error).__name__}: {error}",
        notes=notes,
        metadata={"structured_rejection": True},
    )


def _negative_control_entry(entry: ScorecardEntry, baseline: ScorecardEntry | None) -> ScorecardEntry:
    """Reject a control that accidentally looks better than the student baseline."""

    if baseline is None or entry.holdout_error is None or baseline.holdout_error is None:
        return entry
    if entry.holdout_error > baseline.holdout_error + 1e-12:
        return entry
    metadata = dict(entry.metadata)
    metadata["negative_control_valid"] = False
    metadata["negative_control_failure"] = "control did not produce a worse held-out fit; no transfer improvement is credited"
    return replace(entry, status="rejected", reason="negative control did not degrade relative to the student baseline", notes=tuple(entry.notes) + ("structured rejection: control was not a valid negative control on this synthetic split",), metadata=metadata)


def _calibrate_if_available(
    fit: FlowFitResult,
    validation_student: Sequence[TrajectoryTrace] | None,
    validation_teacher: Sequence[TrajectoryTrace] | None,
    alignment: AlignmentResult,
    mode: str,
    *,
    math_components: Mapping[str, bool] | None = None,
    stability_seeds: Sequence[int] | None = None,
) -> FlowFitResult:
    if validation_student is None or validation_teacher is None:
        return fit
    try:
        return calibrate_flow_confidence(fit, validation_student, validation_teacher, alignment, signature_mode=mode, math_components=None if math_components is None else dict(math_components), stability_seeds=stability_seeds)
    except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError):
        # A connector may expose a different visible depth grid on validation.
        # The fit remains scoreable, but the artifact states that confidence
        # calibration was unavailable instead of silently changing the grid.
        metadata = dict(fit.metadata)
        metadata.update({"confidence_calibrated": False, "confidence_calibration_unavailable": True})
        return replace(fit, metadata=metadata)


def _component_counterfactual_table(
    train_student: Sequence[TrajectoryTrace],
    train_teacher: Sequence[TrajectoryTrace],
    teacher_to_student: AlignmentResult,
    *,
    validation_student: Sequence[TrajectoryTrace] | None,
    validation_teacher: Sequence[TrajectoryTrace] | None,
    holdout_student: Sequence[TrajectoryTrace] | None,
    holdout_teacher: Sequence[TrajectoryTrace] | None,
    reference_target: FlowOperator | None,
    full_entry: ScorecardEntry | None,
    baseline_entry: ScorecardEntry | None,
    seed: int,
    stability_seeds: Sequence[int],
) -> dict[str, Any]:
    """Fit one same-split counterfactual per removable math component.

    The resulting error difference is directional: ``without_component``
    minus ``full``. Positive values mean removal worsened the held-out metric.
    Protocol components and gain scheduling are explicitly unavailable rather
    than being assigned a misleading zero contribution.
    """

    unavailable = {
        "alignment": "alignment is a required supplied protocol map; disabling it would be a no-op",
        "continuous_depth": "continuous depth is a required protocol coordinate; disabling it would be a no-op",
        "ot": "OT is fixed in the supplied alignment; a flow-only disable cannot remove it without refitting alignment",
        "gain_schedule": "gain schedule belongs to the adaptive transfer policy, not the standard full-flow scorecard",
    }
    table: dict[str, Any] = {}
    common = {
        "schema_version": "faytuna-component-counterfactual-v1",
        "seed": int(seed),
        "fit_stability_seeds": [int(value) for value in stability_seeds],
        "same_train_split": True,
        "same_validation_split": validation_student is not None and validation_teacher is not None,
        "same_holdout_split": holdout_student is not None and holdout_teacher is not None,
        "metric": "normalized_transported_flow_rmse",
        "metric_scope": "heldout_one_step_transport_operator_approximation",
        "target_fit_split": TARGET_FIT_SPLIT,
    }
    for component in MATH_COMPONENTS:
        if component in unavailable:
            table[component] = {**common, "component": component, "status": "unavailable", "disabled": False, "metric_delta": None, "skipped_reason": unavailable[component]}
            continue
        if holdout_student is None or holdout_teacher is None:
            table[component] = {**common, "component": component, "status": "unavailable", "disabled": False, "metric_delta": None, "skipped_reason": "disjoint holdout is required for component attribution"}
            continue
        if reference_target is None or full_entry is None or full_entry.holdout_error is None:
            table[component] = {**common, "component": component, "status": "unavailable", "disabled": False, "metric_delta": None, "skipped_reason": "full-flow reference target or holdout score is unavailable"}
            continue
        full_profile = dict(full_entry.metadata.get("math_profile", {})).get("components", {})
        full_component_audit = dict(full_profile.get(component, {}))
        if not bool(full_component_audit.get("applied", False)):
            table[component] = {
                **common,
                "component": component,
                "status": "unavailable",
                "disabled": False,
                "metric_delta": None,
                "full_component_applied": False,
                "skipped_reason": "full-flow reference did not apply this component: " + str(full_component_audit.get("skipped_reason") or "component unavailable"),
            }
            continue
        # OT is part of the supplied alignment and therefore unavailable for a
        # flow-only counterfactual unless the alignment itself is OT.  Marking
        # it false in the inherited switch set prevents the internal refit
        # from mistaking a resolved default for an explicit OT request.
        disabled = {"ot": teacher_to_student.kind == "ot_barycentric", component: False}
        try:
            fit = fit_flow_transfer(train_student, train_teacher, teacher_to_student, signature_mode="full", math_components=disabled, stability_seeds=stability_seeds)
            fit = _calibrate_if_available(fit, validation_student, validation_teacher, teacher_to_student, "full", math_components=disabled, stability_seeds=stability_seeds)
            counterfactual = _entry_from_fit(
                f"full_without_{component}", fit, train_student, train_teacher, teacher_to_student, "full",
                validation_student, validation_teacher, holdout_student, holdout_teacher, teacher_to_student,
                target_flow=reference_target,
                guard=True,
                notes=(f"leave-one-component-out counterfactual: {component} disabled",),
            )
            without_error = counterfactual.holdout_error
            full_error = full_entry.holdout_error
            delta = None if without_error is None or full_error is None else float(without_error - full_error)
            without_profile = dict(counterfactual.metadata.get("math_profile", {})).get("components", {})
            table[component] = {
                **common,
                "component": component,
                "status": "scored" if delta is not None and np.isfinite(delta) else "unavailable",
                "disabled": True,
                "full_status": full_entry.status,
                "without_component_status": counterfactual.status,
                "full_holdout_error": full_error,
                "without_component_holdout_error": without_error,
                "metric_delta": delta,
                "metric_delta_direction": "without_component_minus_full; positive means removal worsened holdout",
                "component_helped_holdout": None if delta is None else bool(delta > 1e-12),
                "full_improvement_vs_baseline": full_entry.holdout_improvement,
                "without_component_improvement_vs_baseline": counterfactual.holdout_improvement,
                "full_component_applied": dict(full_profile.get(component, {})).get("applied"),
                "without_component_applied": dict(without_profile.get(component, {})).get("applied"),
                "accepted_nodes": counterfactual.accepted_nodes,
                "rollback_nodes": counterfactual.rollback_nodes,
                "skipped_reason": None if delta is not None else "counterfactual did not yield a finite holdout score",
            }
        except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as error:
            table[component] = {**common, "component": component, "status": "rejected", "disabled": True, "metric_delta": None, "skipped_reason": f"{type(error).__name__}: {error}"}
    return table


def run_transfer_scorecard(
    train_student: Sequence[TrajectoryTrace],
    train_teacher: Sequence[TrajectoryTrace],
    teacher_to_student: AlignmentResult,
    *,
    validation_student: Sequence[TrajectoryTrace] | None = None,
    validation_teacher: Sequence[TrajectoryTrace] | None = None,
    holdout_student: Sequence[TrajectoryTrace] | None = None,
    holdout_teacher: Sequence[TrajectoryTrace] | None = None,
    seed: int = 0,
    include_stress: bool = True,
) -> ScorecardReport:
    """Compare static, delta, differential and full transfer on disjoint splits."""

    _check_splits(train_student, train_teacher, validation_student, validation_teacher, holdout_student, holdout_teacher)
    rng = np.random.default_rng(seed)
    # Every main method and every counterfactual uses the same deterministic
    # stability ensemble.  This makes the leave-one-component-out comparison
    # a controlled experiment rather than a comparison of different solver
    # noise levels.
    fit_stability_seeds = (0, 17, 31)
    entries: list[ScorecardEntry] = []
    modes = (("baseline_student_flow", "none", False), ("delta_flow", "none", True), ("curvature_jacobian", "differential", True), ("full_flow", "full", True))
    fitted: dict[str, tuple[FlowFitResult, FlowOperator]] = {}
    fit_errors: dict[str, Exception] = {}
    for name, mode, guard in modes:
        try:
            fit = fit_flow_transfer(train_student, train_teacher, teacher_to_student, signature_mode=mode, stability_seeds=fit_stability_seeds)
            fit = _calibrate_if_available(fit, validation_student, validation_teacher, teacher_to_student, mode, stability_seeds=fit_stability_seeds)
            target = fit.transported_teacher
            target_metadata = dict(target.metadata)
            target_metadata.update({"target_fit_split": TARGET_FIT_SPLIT, "metric_scope": FLOW_METRIC_SCOPE, "target_operator_fingerprint": _operator_fingerprint(target)})
            target = replace(target, metadata=target_metadata)
            fit = replace(fit, transported_teacher=target)
            fitted[name] = (fit, target)
        except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as error:
            fit_errors[name] = error

    # Every method is scored against one train-fitted teacher target.  Using a
    # different target estimator for each mode would make the metric compare
    # targets rather than transfer methods and could hide the nonlinear gain.
    reference_target = None
    for candidate_name in ("full_flow", "curvature_jacobian", "delta_flow", "baseline_student_flow"):
        if candidate_name in fitted:
            reference_target = fitted[candidate_name][1]
            break

    for name, mode, guard in modes:
        if name in fit_errors:
            entries.append(_rejected_entry(name, train_student, train_teacher, teacher_to_student, fit_errors[name], ("fit rejected before scoring",)))
            continue
        fit, target = fitted[name]
        entries.append(_entry_from_fit(name, fit, train_student, train_teacher, teacher_to_student, mode, validation_student, validation_teacher, holdout_student, holdout_teacher, teacher_to_student, target_flow=reference_target or target, guard=guard, notes=("baseline is the same fitted student operator for this signature mode",) if name != "baseline_student_flow" else ("reference: no teacher correction applied",)))

    static_train = _state_geometry_error(train_student, train_teacher, teacher_to_student)
    static_validation = None if validation_student is None or validation_teacher is None else _state_geometry_error(validation_student, validation_teacher, teacher_to_student)
    static_holdout = None if holdout_student is None or holdout_teacher is None else _state_geometry_error(holdout_student, holdout_teacher, teacher_to_student)
    entries.insert(0, ScorecardEntry("static_hidden", "reference", "normalized_state_trajectory_rmse", static_train, static_train, static_validation, static_holdout, notes=("state geometry is reported separately; it is not causal or flow success",), metadata={"teacher_and_student_required": True}))

    baseline_entry = next((entry for entry in entries if entry.name == "baseline_student_flow"), None)
    shuffled = _relabel_teacher(train_teacher, train_student, rng.permutation(len(train_teacher)))
    try:
        fit = fit_flow_transfer(train_student, shuffled, teacher_to_student, signature_mode="full", stability_seeds=fit_stability_seeds)
        fit = _calibrate_if_available(fit, validation_student, validation_teacher, teacher_to_student, "full", stability_seeds=fit_stability_seeds)
        if reference_target is None:
            raise RuntimeError("no train-fitted reference target is available for negative control")
        fixed_target = reference_target
        entries.append(_negative_control_entry(_entry_from_fit("shuffled_probes", fit, train_student, shuffled, teacher_to_student, "full", validation_student, validation_teacher, holdout_student, holdout_teacher, teacher_to_student, target_flow=fixed_target, notes=("negative pairing control: teacher traces were permuted",)), baseline_entry))
    except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as error:
        entries.append(_rejected_entry("shuffled_probes", train_student, shuffled, teacher_to_student, error, ("negative control was rejected with a finite structured status",)))

    random_map = _random_alignment(train_teacher, train_student, rng)
    try:
        fit = fit_flow_transfer(train_student, train_teacher, random_map, signature_mode="full", stability_seeds=fit_stability_seeds)
        fit = _calibrate_if_available(fit, validation_student, validation_teacher, teacher_to_student, "full", stability_seeds=fit_stability_seeds)
        if reference_target is None:
            raise RuntimeError("no train-fitted reference target is available for negative control")
        fixed_target = reference_target
        entries.append(_negative_control_entry(_entry_from_fit("random_map", fit, train_student, train_teacher, random_map, "full", validation_student, validation_teacher, holdout_student, holdout_teacher, teacher_to_student, target_flow=fixed_target, notes=("negative map control: fit uses a seeded random teacher-to-student chart",)), baseline_entry))
    except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as error:
        entries.append(_rejected_entry("random_map", train_student, train_teacher, random_map, error, ("negative control was rejected with a finite structured status",)))

    if include_stress:
        stress_cases = (
            ("noise_stress", _perturb_traces(train_student, noise=0.02, seed=seed + 1), _perturb_traces(train_teacher, noise=0.02, seed=seed + 2)),
            ("quantization_stress", _perturb_traces(train_student, quantization_step=0.02, seed=seed + 3), _perturb_traces(train_teacher, quantization_step=0.02, seed=seed + 4)),
        )
        for name, stress_student, stress_teacher in stress_cases:
            try:
                fit = fit_flow_transfer(stress_student, stress_teacher, teacher_to_student, signature_mode="full", stability_seeds=fit_stability_seeds)
                fit = _calibrate_if_available(fit, validation_student, validation_teacher, teacher_to_student, "full", stability_seeds=fit_stability_seeds)
                if reference_target is None:
                    raise RuntimeError("no train-fitted reference target is available for stress control")
                fixed_target = reference_target
                entries.append(_entry_from_fit(name, fit, stress_student, stress_teacher, teacher_to_student, "full", validation_student, validation_teacher, holdout_student, holdout_teacher, teacher_to_student, target_flow=fixed_target, notes=("stress control; training traces were perturbed while evaluation probes stayed disjoint",)))
            except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as error:
                entries.append(_rejected_entry(name, stress_student, stress_teacher, teacher_to_student, error, ("stress case rejected with a finite structured status",)))

    profile_by_name = {name: fit.metadata.get("math_profile") for name, (fit, _) in fitted.items() if fit.metadata.get("math_profile") is not None}
    entry_by_name = {entry.name: entry for entry in entries}
    baseline_profile = profile_by_name.get("baseline_student_flow", {})
    contribution_table = {
        name: compare_math_profiles(
            baseline_profile,
            profile,
            reference_metric=entry_by_name.get("baseline_student_flow").holdout_error if entry_by_name.get("baseline_student_flow") is not None else None,
            candidate_metric=entry_by_name.get(name).holdout_error if entry_by_name.get(name) is not None else None,
        )
        for name, profile in profile_by_name.items()
    }
    component_counterfactuals = _component_counterfactual_table(
        train_student,
        train_teacher,
        teacher_to_student,
        validation_student=validation_student,
        validation_teacher=validation_teacher,
        holdout_student=holdout_student,
        holdout_teacher=holdout_teacher,
        reference_target=reference_target,
        full_entry=entry_by_name.get("full_flow"),
        baseline_entry=baseline_entry,
        seed=seed,
        stability_seeds=fit_stability_seeds,
    )
    return ScorecardReport(
        tuple(entries),
        validation_student is not None and validation_teacher is not None,
        holdout_student is not None and holdout_teacher is not None,
        {
            "train_probe_count": len(train_student),
            "validation_probe_count": 0 if validation_student is None else len(validation_student),
            "holdout_probe_count": 0 if holdout_student is None else len(holdout_student),
            "seed": seed,
            "fit_stability_seeds": list(fit_stability_seeds),
            "teacher_and_student_required": True,
            "target_fit_split": TARGET_FIT_SPLIT,
            "metric_scope": FLOW_METRIC_SCOPE,
            "dynamic_metric_scope": DYNAMIC_METRIC_SCOPE,
            "target_operator_fingerprints": {name: _operator_fingerprint(target) for name, (_, target) in fitted.items()},
            "reference_target_operator_fingerprint": None if reference_target is None else _operator_fingerprint(reference_target),
            "reference_target_source": "full_flow_train_fit" if "full_flow" in fitted else "first_available_train_fit",
            "decision_rule": "positive improvement means baseline_error minus candidate_error is strictly positive",
            "semantic_claim": "not measured; causal rerun is a separate validation stage",
            "causal_claim": "not measured by scorecard",
            "math_profiles": profile_by_name,
            "math_contribution_table": contribution_table,
            "math_component_counterfactual_table": component_counterfactuals,
            "component_counterfactuals_available": bool(holdout_student is not None and holdout_teacher is not None),
        },
    )
