"""Memory bounded flow fitting for sequence-flattened hidden states.

The ordinary field backend stores dense compact matrices.  GPT-2 sequence
charts can make even their first-order ``d x d`` field too large, before a
quadratic term is considered.  This module uses a deterministic orthonormal
random chart ``P in R^(d x r)`` and fits the transported field in ``r``
coordinates.  The returned :class:`FlowOperator` keeps ``P`` as an explicit
chart projection and lifts predictions with ``P.T``; no dense ``d x d`` or
``d x d x d`` array is allocated.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from .types import AlignmentResult, FlowFitResult, FlowOperator, TrajectoryTrace, TransitionObservation, finite_array


DEFAULT_MAX_DENSE_FEATURES = 5_000_000
DEFAULT_SCALABLE_RANK = 64


def dense_memory_estimate(state_dim: int, *, nodes: int = 1, quadratic: bool = False, dtype_bytes: int = 8) -> dict[str, int | bool]:
    """Estimate dense field storage without allocating the requested tensor."""

    d = int(state_dim)
    n = int(nodes)
    if d < 1 or n < 1 or int(dtype_bytes) < 1:
        raise ValueError("state_dim, nodes, and dtype_bytes must be positive")
    linear_features = d
    quadratic_features = d * d if quadratic else 0
    linear_operator_elements = d * d
    quadratic_operator_elements = d * d * d if quadratic else 0
    return {
        "state_dim": d,
        "nodes": n,
        "linear_feature_count": linear_features,
        "quadratic_feature_count": quadratic_features,
        "linear_operator_elements": linear_operator_elements,
        "quadratic_operator_elements": quadratic_operator_elements,
        "linear_feature_bytes_per_sample": linear_features * int(dtype_bytes),
        "quadratic_feature_bytes_per_sample": quadratic_features * int(dtype_bytes),
        "linear_bytes": linear_operator_elements * n * int(dtype_bytes),
        "quadratic_bytes": quadratic_operator_elements * n * int(dtype_bytes),
        "total_bytes": (linear_operator_elements + quadratic_operator_elements) * n * int(dtype_bytes),
        "finite": all(value >= 0 for value in (linear_features, quadratic_features, linear_operator_elements, quadratic_operator_elements)),
    }


def requires_scalable_backend(state_dim: int, *, max_dense_features: int = DEFAULT_MAX_DENSE_FEATURES, quadratic: bool = False) -> bool:
    estimate = dense_memory_estimate(state_dim, quadratic=quadratic)
    return (
        int(estimate["linear_operator_elements"]) > int(max_dense_features)
        or (quadratic and (int(estimate["quadratic_feature_count"]) > int(max_dense_features) or int(estimate["quadratic_operator_elements"]) > int(max_dense_features)))
    )


def deterministic_chart_projection(input_dim: int, rank: int, seed: int = 0) -> np.ndarray:
    if input_dim < 1 or rank < 1 or rank > input_dim:
        raise ValueError("scalable chart dimensions are invalid")
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(size=(int(input_dim), int(rank)))
    q, _ = np.linalg.qr(raw, mode="reduced")
    return finite_array(q, ndim=2, name="scalable chart projection")


def data_aware_chart_projection(values: np.ndarray, rank: int, seed: int = 0) -> np.ndarray:
    """Build a data-aware low-rank chart with O(samples * rank * dim) work.

    A randomized range finder avoids the d-by-d covariance used by ordinary
    whitening.  The result is an explicit orthonormal basis in the original
    chart, so the approximation and the discarded subspace can be reported.
    """

    samples = finite_array(values, ndim=2, name="chart samples")
    if samples.shape[0] < 2 or rank < 1 or rank > samples.shape[1]:
        raise ValueError("data-aware chart dimensions require at least two samples")
    effective_rank = min(int(rank), samples.shape[0] - 1, samples.shape[1])
    centered = samples - samples.mean(axis=0)
    scale = max(1.0, float(np.max(np.abs(centered))))
    scaled = centered / scale
    if effective_rank < 1:
        return deterministic_chart_projection(samples.shape[1], 1, seed)
    oversample = min(samples.shape[1], effective_rank + 8)
    rng = np.random.default_rng(int(seed))
    omega = rng.normal(size=(samples.shape[1], oversample))
    sketch = scaled @ omega
    basis, _ = np.linalg.qr(sketch, mode="reduced")
    small = basis.T @ scaled
    _, _, right_transpose = np.linalg.svd(small, full_matrices=False)
    projection = right_transpose[:effective_rank].T
    if effective_rank < rank:
        # A probe set can have lower rank than the requested chart.  Complete
        # it with a deterministic orthogonal complement instead of silently
        # changing the advertised chart rank.
        extra = deterministic_chart_projection(samples.shape[1], rank, seed + 7919)
        projection = np.column_stack([projection, extra])
        projection, _ = np.linalg.qr(projection, mode="reduced")
        projection = projection[:, :rank]
    return finite_array(projection, ndim=2, name="data-aware chart projection")


def _compress_trace(trace: TrajectoryTrace, projection: np.ndarray, *, role: str) -> TrajectoryTrace:
    states = trace.hidden_states @ projection
    residuals = None if trace.residual_states is None else trace.residual_states @ projection
    transitions: list[TransitionObservation] = []
    for transition in trace.transitions:
        source = transition.source_state @ projection
        target = transition.target_state @ projection
        delta = transition.delta @ projection
        vector = transition.vector_field @ projection
        transitions.append(TransitionObservation(
            transition.source_layer,
            transition.target_layer,
            transition.source_depth,
            transition.target_depth,
            source,
            target,
            delta,
            vector,
            jacobian=None,
            hessian_sketch=None,
            curvature=transition.curvature,
            singular_values=None,
            normalization_geometry=transition.normalization_geometry,
            attention_geometry=transition.attention_geometry,
            uncertainty={**transition.uncertainty, "scalable_chart": 1.0},
        ))
    metadata = {
        **dict(trace.metadata),
        "scalable_backend": "randomized_latent_compression",
        "scalable_chart_role": role,
        "scalable_original_state_dim": trace.state_dim,
        "scalable_rank": projection.shape[1],
        "scalable_differential_note": "Jacobian/Hessian arrays are not copied; flow is fitted in the compact chart",
    }
    return replace(trace, hidden_states=states, residual_states=residuals, transitions=tuple(transitions), metadata=metadata, feature_space=f"{trace.feature_space}:compact_rank_{projection.shape[1]}")


def _compact_identity_alignment(rank: int, parent: AlignmentResult) -> AlignmentResult:
    zeros = np.zeros(rank, dtype=np.float64)
    return AlignmentResult(
        "scalable_compact_identity",
        zeros,
        zeros,
        np.eye(rank),
        zeros,
        rank,
        rank,
        0.0,
        0.0,
        0.0,
        1.0,
        metadata={
            "direction": "teacher_to_student",
            "source_role": "teacher",
            "target_role": "student",
            "parent_alignment_kind": parent.kind,
            "parent_alignment_fit_scope": parent.metadata.get("alignment_fit_scope", "unspecified"),
            "parent_alignment_paired_error": parent.paired_error,
            "parent_alignment_relational_error": parent.relational_error,
            "alignment_fit_scope": parent.metadata.get("alignment_fit_scope", "unspecified"),
            "scalable_compact_chart": True,
        },
    )


def _wrap_operator(operator: FlowOperator, projection: np.ndarray, *, original_dim: int, estimate: Mapping[str, Any], role: str) -> FlowOperator:
    metadata = {
        **dict(operator.metadata),
        "representation": "randomized_latent_compression",
        "scalable_backend": "randomized_latent_compression",
        "scalable_chart_role": role,
        "scalable_original_state_dim": original_dim,
        "scalable_rank": projection.shape[1],
        "dense_memory_guard": dict(estimate),
    }
    return FlowOperator(operator.coordinates, operator.matrices, operator.biases, operator.sample_counts, operator.residual_scales, operator.spectral_norms, metadata, operator.quadratic_terms, projection)


def fit_scalable_flow_transfer(
    student_traces: Sequence[TrajectoryTrace],
    teacher_traces: Sequence[TrajectoryTrace],
    teacher_to_student: AlignmentResult,
    *,
    signature_mode: str = "full",
    rank: int = DEFAULT_SCALABLE_RANK,
    seed: int = 0,
    max_dense_features: int = DEFAULT_MAX_DENSE_FEATURES,
    math_components: Mapping[str, bool] | None = None,
    projection: np.ndarray | None = None,
) -> FlowFitResult:
    """Fit a chart-compressed field and explicitly downgrade full to differential."""

    if not student_traces or len(student_traces) != len(teacher_traces):
        raise ValueError("scalable flow transfer needs paired non-empty traces")
    if signature_mode not in {"none", "differential", "full"}:
        raise ValueError("signature_mode must be 'none', 'differential', or 'full'")
    student_dim = student_traces[0].state_dim
    teacher_dim = teacher_traces[0].state_dim
    if any(trace.state_dim != student_dim for trace in student_traces) or any(trace.state_dim != teacher_dim for trace in teacher_traces):
        raise ValueError("scalable flow traces have inconsistent model dimensions")
    if any(student.probe_id != teacher.probe_id for student, teacher in zip(student_traces, teacher_traces)):
        raise ValueError("scalable flow traces must retain paired probe IDs")
    if teacher_to_student.source_dim != teacher_dim or teacher_to_student.target_dim != student_dim:
        raise ValueError(f"teacher_to_student alignment must have shape ({teacher_dim}, {student_dim})")
    if teacher_to_student.metadata.get("direction") not in {None, "teacher_to_student"}:
        raise ValueError("scalable flow requires a teacher_to_student alignment")
    if rank < 1 or rank > student_dim:
        raise ValueError(f"scalable rank must be in [1, {student_dim}]")
    if max_dense_features < 1:
        raise ValueError("max_dense_features must be positive")
    rank = min(int(rank), int(np.sqrt(max_dense_features)))
    if rank < 1:
        raise ValueError("scalable rank is zero under max_dense_features")
    estimate = dense_memory_estimate(student_dim, nodes=student_traces[0].layer_count - 1, quadratic=signature_mode == "full")
    student_states = np.concatenate([trace.hidden_states for trace in student_traces], axis=0)
    if projection is None:
        projection = data_aware_chart_projection(student_states, int(rank), seed)
        projection_source = "fit_student_states"
    else:
        projection = finite_array(projection, ndim=2, name="reused scalable chart projection")
        if projection.shape != (student_dim, int(rank)):
            raise ValueError(f"reused scalable chart projection must have shape ({student_dim}, {int(rank)}); got {projection.shape}")
        if not np.allclose(projection.T @ projection, np.eye(int(rank)), atol=2e-5, rtol=2e-5):
            raise ValueError("reused scalable chart projection must be approximately orthonormal")
        projection_source = "reused_train_chart"
    compact_student = tuple(_compress_trace(trace, projection, role="student") for trace in student_traces)
    compact_teacher = []
    for trace in teacher_traces:
        mapped_states = teacher_to_student.apply(trace.hidden_states, depth=trace.depth_coordinates) @ projection
        mapped_transitions = []
        for transition in trace.transitions:
            source = teacher_to_student.apply(transition.source_state, depth=transition.source_depth) @ projection
            target = teacher_to_student.apply(transition.target_state, depth=transition.target_depth) @ projection
            delta = target - source
            ds = transition.target_depth - transition.source_depth
            mapped_transitions.append(TransitionObservation(transition.source_layer, transition.target_layer, transition.source_depth, transition.target_depth, source, target, delta, delta / ds, curvature=transition.curvature, uncertainty={**transition.uncertainty, "scalable_chart": 1.0}))
        compact_teacher.append(replace(trace, hidden_states=mapped_states, residual_states=None, transitions=tuple(mapped_transitions), metadata={**dict(trace.metadata), "scalable_backend": "randomized_latent_compression", "scalable_chart_role": "teacher_transported_to_student", "scalable_original_state_dim": teacher_dim, "scalable_rank": rank}, feature_space=f"{trace.feature_space}:transported_compact_rank_{rank}"))
    compact_alignment = _compact_identity_alignment(int(rank), teacher_to_student)
    compact_components = dict(math_components or {})
    compact_components.update({"quadratic_flow": False, "hessian_2jet": False})
    from .flow import fit_flow_transfer
    compact_mode = "none" if signature_mode == "none" else "differential"
    compact_fit = fit_flow_transfer(compact_student, tuple(compact_teacher), compact_alignment, signature_mode=compact_mode, math_components=compact_components, max_dense_features=max_dense_features, scalable_rank=rank, scalable_seed=seed)
    requested_mode = signature_mode
    metadata = dict(compact_fit.metadata)
    metadata.update({
        # The compact identity map is an internal chart bridge. Preserve the
        # parent teacher->student alignment evidence in the public artifact;
        # otherwise a forensic reader could mistake the zero-error identity
        # bridge for the real cross-model alignment.
        "alignment_direction": teacher_to_student.metadata.get("direction", "teacher_to_student"),
        "alignment_fit_scope": teacher_to_student.metadata.get("alignment_fit_scope", "unspecified"),
        "alignment_paired_error": teacher_to_student.paired_error,
        "alignment_relational_error": teacher_to_student.relational_error,
        "parent_alignment_kind": teacher_to_student.kind,
        "requested_signature_mode": requested_mode,
        "effective_signature_mode": compact_mode,
        "scalable_backend": "randomized_latent_compression",
        "scalable_chart_rank": int(rank),
        "scalable_seed": int(seed),
        "max_dense_features": int(max_dense_features),
        "scalable_chart_projection_shape": list(projection.shape),
        "scalable_chart_projection_source": projection_source,
        "scalable_original_student_dim": student_dim,
        "scalable_original_teacher_dim": teacher_dim,
        "dense_memory_guard": dict(estimate),
        "quadratic_skipped_reason": "dense quadratic and dense first-order field exceed max_dense_features; safe differential compact chart selected" if signature_mode == "full" else "quadratic channel was not requested",
        "local_jet_skipped_reason": "local jets are evaluated in compact chart; full-dimensional dense jet is prohibited",
        "capacity_projection": "explicit randomized chart projector with residual reported by capacity diagnostics",
        "capacity_report_scope": "compact_chart; original full-dimensional residual was not formed",
    })
    student = _wrap_operator(compact_fit.student, projection, original_dim=student_dim, estimate=estimate, role="student")
    transported = _wrap_operator(compact_fit.transported_teacher, projection, original_dim=student_dim, estimate=estimate, role="teacher_transported_to_student")
    operator_metadata = {
        "requested_signature_mode": requested_mode,
        "effective_signature_mode": compact_mode,
        "quadratic_skipped_reason": metadata["quadratic_skipped_reason"],
        "max_dense_features": int(max_dense_features),
        "target_fit_split": "train",
        "metric_scope": "transported_teacher_operator_approximation",
    }
    student = replace(student, metadata={**dict(student.metadata), **operator_metadata})
    transported = replace(transported, metadata={**dict(transported.metadata), **operator_metadata})
    from .math_profile import build_math_profile
    metadata["math_profile"] = build_math_profile(
        metadata,
        alignment_kind=teacher_to_student.kind,
        signature_mode=requested_mode,
        correction_matrices=compact_fit.correction_matrices,
        correction_quadratic=None,
        requested=math_components,
    )
    metadata["math_profile"].update({
        "scalable_backend": "randomized_latent_compression",
        "requested_signature_mode": requested_mode,
        "effective_signature_mode": compact_mode,
        "dense_quadratic_blocked": True,
        "hessian_2jet_applied": False,
        "quadratic_flow_applied": False,
        "quadratic_skipped_reason": metadata["quadratic_skipped_reason"],
    })
    return FlowFitResult(student, transported, compact_fit.correction_matrices, compact_fit.correction_biases, compact_fit.confidence, compact_fit.validation_error, metadata, compact_fit.depth_report, compact_fit.capacity_diagnostics, None)
