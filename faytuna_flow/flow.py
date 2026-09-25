"""Multi-scale estimation and transport of depth-indexed emergent flow.

The fit is local in continuous depth rather than a nearest-layer lookup. It
uses weighted ridge surrogates only as a numerically stable estimator of local
vector-field samples; all samples retain their observed depth and gap weight.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .depth import MonotoneCorrespondence, monotone_correspondence
from .capacity import diagnose_capacity, project_correction_to_student_tangent
from .jets import LocalJetTransport, fit_rank_aware_local_jet_transport
from .math_profile import build_math_profile, resolve_math_components
from .signatures import TangentTransport, differential_compatibility, fit_tangent_transport
from .types import AlignmentResult, CapacityDiagnostics, DepthTransferReport, FlowFitResult, FlowOperator, TrajectoryTrace, finite_array, stable_l2
from .scalable import DEFAULT_MAX_DENSE_FEATURES, DEFAULT_SCALABLE_RANK, dense_memory_estimate, requires_scalable_backend


@dataclass(frozen=True)
class FlowSample:
    state: np.ndarray
    velocity: np.ndarray
    depth: float
    weight: float = 1.0
    source_transition: int | None = None
    source_probe_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", finite_array(self.state, ndim=1, name="flow sample state"))
        object.__setattr__(self, "velocity", finite_array(self.velocity, ndim=1, name="flow sample velocity"))
        if self.state.size != self.velocity.size or not np.isfinite(self.depth) or not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("flow sample dimensions, depth, or weight are invalid")


def _spectral_norm_diagnostic(matrix: np.ndarray, *, max_iterations: int = 24, tolerance: float = 1e-6) -> tuple[float, int, bool]:
    """Estimate the largest singular value without a full dense SVD.

    This is used for per-node diagnostics only.  The solver keeps its
    constraint checks exact.  Power iteration uses a deterministic start and
    returns the iteration count/convergence flag so the approximation is
    visible in artifacts rather than being presented as an exact spectrum.
    """

    values = finite_array(matrix, ndim=2, name="spectral diagnostic operator")
    if values.size == 0:
        return 0.0, 0, True
    if max_iterations < 1 or tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("spectral diagnostic parameters are invalid")
    vector = np.ones(values.shape[1], dtype=np.float64)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    previous = 0.0
    estimate = 0.0
    converged = False
    for iteration in range(1, int(max_iterations) + 1):
        image = values @ vector
        estimate = float(np.linalg.norm(image))
        if estimate <= 1e-15:
            return 0.0, iteration, True
        vector = values.T @ image
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm <= 1e-15:
            return estimate, iteration, True
        vector /= vector_norm
        if abs(estimate - previous) <= tolerance * max(1.0, estimate):
            converged = True
            break
        previous = estimate
    return float(estimate), iteration, converged


def _weighted_ridge_operator(samples: Sequence[FlowSample], ridge: float, *, quadratic: bool = False, quadratic_ridge: float | None = None, robust_loss: str = "huber", robust_iterations: int = 4, robust_tuning: float = 1.345) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    if not samples:
        raise ValueError("at least one flow sample is required")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("flow ridge must be finite and non-negative")
    if robust_loss not in {"none", "huber", "tukey"} or not 1 <= robust_iterations <= 20 or robust_tuning <= 0 or not np.isfinite(robust_tuning):
        raise ValueError("robust flow fitting parameters are invalid")
    states = np.asarray([sample.state for sample in samples], dtype=np.float64)
    velocities = np.asarray([sample.velocity for sample in samples], dtype=np.float64)
    weights = np.asarray([sample.weight for sample in samples], dtype=np.float64)
    if states.shape != velocities.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("flow samples are inconsistent or have zero weight")
    weight_scale = float(np.max(weights))
    if weight_scale <= 0:
        raise ValueError("flow samples are inconsistent or have zero weight")
    weights = weights / weight_scale
    # Scale before forming the Gram matrix. This prevents overflow for large
    # but finite observations and leaves the returned operator in raw units.
    state_scale = max(1.0, float(np.max(np.abs(states))))
    velocity_scale = max(1.0, float(np.max(np.abs(velocities))))
    x = states / state_scale
    v = velocities / velocity_scale
    quadratic_scale = state_scale * state_scale
    if quadratic and not np.isfinite(quadratic_scale):
        # The normalized design remains representable even when the raw
        # square of the chart scale is not.  The returned coefficient is then
        # the largest finite conservative raw scale; evaluation still fails
        # loudly if the physical quadratic value itself is unrepresentable.
        quadratic_scale = np.finfo(np.float64).max
    pair_features = np.einsum("ni,nj->nij", x, x).reshape(len(x), -1) if quadratic else np.empty((len(x), 0))
    design = np.column_stack([x, np.ones(len(x)), pair_features])
    sqrt_w = np.sqrt(np.clip(weights, 0.0, np.inf))
    weighted_design = design * sqrt_w[:, None]
    weighted_velocity = v * sqrt_w[:, None]
    penalties = np.full(design.shape[1], float(ridge), dtype=np.float64)
    if quadratic:
        penalties[states.shape[1] + 1 :] = max(float(ridge), float(quadratic_ridge if quadratic_ridge is not None else 0.05))
    coefficient = np.zeros((design.shape[1], velocities.shape[1]), dtype=np.float64)
    robust_weights = weights.copy()
    for _ in range(robust_iterations):
        sqrt_w = np.sqrt(np.clip(robust_weights, 0.0, np.inf))
        weighted_design = design * sqrt_w[:, None]
        weighted_velocity = v * sqrt_w[:, None]
        # With few probes, the primal Gram matrix is (d+1)x(d+1) even
        # though the design has only n rows.  For the uniform linear ridge,
        # the dual identity is exact:
        # X.T (X X.T + lambda I)^-1 Y ==
        # (X.T X + lambda I)^-1 X.T Y.
        if not quadratic and weighted_design.shape[1] > weighted_design.shape[0] and ridge > 0.0:
            dual_gram = weighted_design @ weighted_design.T + ridge * np.eye(weighted_design.shape[0])
            dual_gram = (dual_gram + dual_gram.T) / 2.0
            try:
                dual_solution = np.linalg.solve(dual_gram, weighted_velocity)
            except np.linalg.LinAlgError:
                dual_solution = np.linalg.lstsq(dual_gram, weighted_velocity, rcond=1e-10)[0]
            coefficient = weighted_design.T @ dual_solution
        else:
            gram = weighted_design.T @ weighted_design + np.diag(penalties)
            gram = (gram + gram.T) / 2.0
            try:
                coefficient = np.linalg.solve(gram, weighted_design.T @ weighted_velocity)
            except np.linalg.LinAlgError:
                coefficient = np.linalg.lstsq(weighted_design, weighted_velocity, rcond=1e-10)[0]
        residual_rows = np.asarray(stable_l2(design @ coefficient - v, axis=1, name="robust flow residual"), dtype=np.float64)
        if robust_loss == "none":
            break
        scale = max(float(np.median(residual_rows)), 1e-8)
        normalized_residual = residual_rows / (robust_tuning * scale)
        if robust_loss == "huber":
            robust_factor = np.minimum(1.0, 1.0 / np.maximum(normalized_residual, 1.0))
        else:
            robust_factor = np.where(normalized_residual < 1.0, (1.0 - normalized_residual ** 2) ** 2, 0.0)
        robust_weights = weights * robust_factor
        if not np.any(robust_weights > 0):
            robust_weights = weights.copy()
    prediction = design @ coefficient
    residual = float(np.sqrt(np.sum(weights[:, None] * (prediction - v) ** 2) / max(np.sum(weights), 1e-12)) * velocity_scale)
    matrix = (velocity_scale / state_scale) * coefficient[: states.shape[1]]
    bias = velocity_scale * coefficient[states.shape[1]]
    quadratic_terms = None
    if quadratic:
        quadratic_terms = (velocity_scale / quadratic_scale) * coefficient[states.shape[1] + 1 :].T.reshape(states.shape[1], states.shape[1], states.shape[1])
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(bias)) and (quadratic_terms is None or np.all(np.isfinite(quadratic_terms))) and np.isfinite(residual)):
        raise FloatingPointError("flow operator fit produced non-finite values")
    return matrix, bias, residual, quadratic_terms


def _validate_coordinates(coordinates: np.ndarray) -> np.ndarray:
    values = finite_array(coordinates, ndim=1, name="flow coordinates")
    if len(values) == 0 or np.any(np.diff(values) <= 0):
        raise ValueError("flow coordinates must be non-empty and strictly increasing")
    return values


def _continuous_local_fit(samples: Sequence[FlowSample], coordinates: np.ndarray, *, ridge: float, min_samples_per_node: int, bandwidth: float | None, quadratic: bool = False, quadratic_ridge: float | None = None, robust_loss: str = "huber", robust_iterations: int = 4) -> FlowOperator:
    node_coordinates = _validate_coordinates(coordinates)
    if not samples:
        raise ValueError("at least one flow sample is required")
    if any(sample.state.size != samples[0].state.size for sample in samples):
        raise ValueError("flow samples have inconsistent state dimensions")
    scale = np.median(np.diff(node_coordinates)) if len(node_coordinates) > 1 else 1.0
    # A narrow default kernel preserves layer-local dynamics when observations
    # land on a node while still permitting interpolation for genuine gaps.
    # It is a continuous kernel, never an index/nearest-node assignment.
    if min_samples_per_node < 1 or ridge < 0 or not np.isfinite(ridge):
        raise ValueError("flow fit support and ridge parameters are invalid")
    if bandwidth is not None and (bandwidth <= 0 or not np.isfinite(bandwidth)):
        raise ValueError("flow bandwidth must be positive and finite")
    band = max(float(bandwidth or (0.30 * scale)), 1e-4)
    global_matrix, global_bias, global_residual, global_quadratic = _weighted_ridge_operator(samples, ridge, quadratic=quadratic, quadratic_ridge=quadratic_ridge, robust_loss=robust_loss, robust_iterations=robust_iterations)
    matrices, biases, counts, residuals, norms, quadratic_terms = [], [], [], [], [], []
    confidence_masses = []
    spectral_iterations, spectral_converged = [], []
    for coordinate in node_coordinates:
        local = []
        effective_count = 0.0
        confidence_mass = 0.0
        exact = [sample for sample in samples if abs(sample.depth - coordinate) <= 1e-10 and sample.weight > 0]
        if len(exact) >= min_samples_per_node:
            # Exact observations anchor a node without borrowing dynamics from
            # adjacent depth. This remains a continuous estimator because the
            # kernel path is used whenever a node lies inside a genuine gap.
            local = exact
            # min_samples_per_node measures observation support.  Confidence
            # remains a regression weight, but fractional gap confidence must
            # not turn every valid interpolated node into zero support.
            effective_count = float(len(exact))
            confidence_mass = float(sum(sample.weight for sample in exact))
        else:
            for sample in samples:
                kernel = float(np.exp(-0.5 * ((sample.depth - coordinate) / band) ** 2))
                weight = sample.weight * kernel
                if weight > 1e-9:
                    local.append(FlowSample(sample.state, sample.velocity, sample.depth, weight, sample.source_transition))
                    effective_count += kernel
                    confidence_mass += weight
        if effective_count < min_samples_per_node:
            matrix, bias, residual, quadratic_term = global_matrix, global_bias, global_residual, global_quadratic
            count = 0.0
            confidence_mass = 0.0
        else:
            matrix, bias, residual, quadratic_term = _weighted_ridge_operator(local, ridge, quadratic=quadratic, quadratic_ridge=quadratic_ridge, robust_loss=robust_loss, robust_iterations=robust_iterations)
            count = effective_count
        matrices.append(matrix)
        biases.append(bias)
        counts.append(count)
        confidence_masses.append(confidence_mass)
        residuals.append(residual)
        norm, iterations, converged = _spectral_norm_diagnostic(matrix)
        norms.append(norm)
        spectral_iterations.append(iterations)
        spectral_converged.append(converged)
        if quadratic:
            if quadratic_term is None:
                raise RuntimeError("quadratic flow fit returned no quadratic term")
            quadratic_terms.append(quadratic_term)
    input_radius = max(1.0, float(np.max(np.abs(np.asarray([sample.state for sample in samples])))))
    metadata = {"ridge": ridge, "quadratic_ridge": max(float(ridge), float(quadratic_ridge if quadratic_ridge is not None else 0.05)) if quadratic else None, "min_samples_per_node": min_samples_per_node, "fitting_method": "continuous_depth_weighted_local_robust_ridge", "robust_loss": robust_loss, "robust_iterations": robust_iterations, "bandwidth": band, "nearest_node_grouping": False, "support_count_definition": "effective_kernel_observation_count_before_confidence_weights", "confidence_mass": confidence_masses, "state_feature_degree": 2 if quadratic else 1, "input_radius": input_radius, "spectral_norm_method": "deterministic_power_iteration", "spectral_norm_iterations": spectral_iterations, "spectral_norm_converged": spectral_converged, "spectral_norm_tolerance": 1e-6}
    return FlowOperator(node_coordinates, np.asarray(matrices), np.asarray(biases), np.asarray(counts), np.asarray(residuals), np.asarray(norms), metadata, np.asarray(quadratic_terms) if quadratic else None)


def _trace_samples(traces: Sequence[TrajectoryTrace]) -> list[FlowSample]:
    return [FlowSample(transition.source_state, transition.vector_field, transition.source_depth, 1.0, index, trace.probe_id) for trace in traces for index, transition in enumerate(trace.transitions)]


def fit_flow_operator(traces: Sequence[TrajectoryTrace], *, coordinates: np.ndarray | None = None, ridge: float = 1e-4, min_samples_per_node: int = 2, observations: Sequence[FlowSample] | None = None, bandwidth: float | None = None, quadratic: bool = False, quadratic_ridge: float | None = None, robust_loss: str = "huber", robust_iterations: int = 4, max_dense_features: int = DEFAULT_MAX_DENSE_FEATURES) -> FlowOperator:
    if not traces and not observations:
        raise ValueError("at least one trace or flow observation is required")
    samples = list(observations) if observations is not None else _trace_samples(traces)
    if samples and requires_scalable_backend(samples[0].state.size, max_dense_features=max_dense_features, quadratic=quadratic):
        raise ValueError("dense flow operator fit is prohibited by max_dense_features; use fit_flow_transfer for the scalable chart backend")
    if coordinates is None:
        if not traces:
            raise ValueError("coordinates are required when fitting from standalone observations")
        coordinates = traces[0].depth_coordinates[:-1]
    return _continuous_local_fit(samples, coordinates, ridge=ridge, min_samples_per_node=min_samples_per_node, bandwidth=bandwidth, quadratic=quadratic, quadratic_ridge=quadratic_ridge, robust_loss=robust_loss, robust_iterations=robust_iterations)


def _operator_spread(reference: FlowOperator, members: Sequence[FlowOperator]) -> np.ndarray:
    if not members:
        return np.zeros(len(reference.coordinates), dtype=np.float64)
    matrix_delta = np.asarray([stable_l2(member.matrices - reference.matrices, axis=(1, 2), name="operator stability delta") for member in members])
    matrix_scale = np.maximum(1.0, np.asarray(stable_l2(reference.matrices, axis=(1, 2), name="operator stability reference")))
    spread = np.mean(matrix_delta / matrix_scale[None, :], axis=0)
    if reference.quadratic_terms is not None and all(member.quadratic_terms is not None for member in members):
        q_delta = np.asarray([stable_l2(member.quadratic_terms - reference.quadratic_terms, axis=(1, 2, 3), name="quadratic stability delta") for member in members])
        q_scale = np.maximum(1.0, np.asarray(stable_l2(reference.quadratic_terms, axis=(1, 2, 3), name="quadratic stability reference")))
        spread = np.clip(spread + 0.5 * np.mean(q_delta / q_scale[None, :], axis=0), 0.0, 1.0)
    return np.clip(spread, 0.0, 1.0)


def _fit_stability_diagnostics(
    reference: FlowOperator,
    observations: Sequence[FlowSample],
    *,
    coordinates: np.ndarray,
    ridge: float,
    min_samples_per_node: int,
    bandwidth: float | None,
    quadratic: bool,
    quadratic_ridge: float | None,
    robust_loss: str,
    robust_iterations: int,
    seeds: Sequence[int],
    leave_one_probe_out: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate deterministic ensemble and leave-one-probe-out instability."""

    if not observations:
        return np.zeros(len(coordinates)), np.zeros(len(coordinates))
    members: list[FlowOperator] = []
    for seed in tuple(seeds):
        rng = np.random.default_rng(int(seed))
        perturbed = [replace(sample, weight=float(sample.weight * np.exp(rng.normal(0.0, 0.02)))) for sample in observations]
        try:
            members.append(fit_flow_operator([], coordinates=coordinates, observations=perturbed, ridge=ridge, min_samples_per_node=min_samples_per_node, bandwidth=bandwidth, quadratic=quadratic, quadratic_ridge=quadratic_ridge, robust_loss=robust_loss, robust_iterations=robust_iterations))
        except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError):
            continue
    ensemble_spread = _operator_spread(reference, members)
    loo_members: list[FlowOperator] = []
    probe_ids = sorted({sample.source_probe_id for sample in observations if sample.source_probe_id is not None})
    if leave_one_probe_out and len(probe_ids) > 2:
        selected = probe_ids[: min(len(probe_ids), 4)]
        for probe_id in selected:
            reduced = [sample for sample in observations if sample.source_probe_id != probe_id]
            if len(reduced) < 2:
                continue
            try:
                loo_members.append(fit_flow_operator([], coordinates=coordinates, observations=reduced, ridge=ridge, min_samples_per_node=min_samples_per_node, bandwidth=bandwidth, quadratic=quadratic, quadratic_ridge=quadratic_ridge, robust_loss=robust_loss, robust_iterations=robust_iterations))
            except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError):
                continue
    return ensemble_spread, _operator_spread(reference, loo_members)


def _depth_map(teacher: TrajectoryTrace, student: TrajectoryTrace, correspondence: MonotoneCorrespondence) -> tuple[np.ndarray, np.ndarray, set[int], set[int]]:
    pairs = list(correspondence.pairs)
    if len(pairs) < 2:
        raise ValueError("depth correspondence has fewer than two matched observations")
    pairs.sort(key=lambda item: item[1])
    teacher_indices = np.asarray([item[1] for item in pairs], dtype=int)
    student_indices = np.asarray([item[0] for item in pairs], dtype=int)
    teacher_depth = teacher.depth_coordinates[teacher_indices]
    student_depth = student.depth_coordinates[student_indices]
    order = np.argsort(teacher_depth)
    teacher_depth = teacher_depth[order]
    student_depth = student_depth[order]
    # np.interp is used only for the continuous coordinate map. It does not
    # create an observation or layer; unmatched nodes remain gap-weighted.
    mapped = np.interp(teacher.depth_coordinates, teacher_depth, student_depth)
    matched_teacher = set(int(x) for x in teacher_indices)
    matched_student = set(int(x) for x in student_indices)
    return mapped, student_depth, matched_teacher, matched_student


def _transport_teacher_samples(student: TrajectoryTrace, teacher: TrajectoryTrace, alignment: AlignmentResult, correspondence: MonotoneCorrespondence, gap_weight_floor: float, compatibility: np.ndarray | None = None, tangent_confidence: np.ndarray | None = None, tangent_transport: TangentTransport | None = None, jet_transport: LocalJetTransport | None = None, path_score: float = 1.0) -> tuple[list[FlowSample], np.ndarray, int, float]:
    mapped_depth, _, matched_teacher, matched_student = _depth_map(teacher, student, correspondence)
    samples: list[FlowSample] = []
    dropped = 0
    weights = []
    interpolation_weights = []
    for index, transition in enumerate(teacher.transitions):
        source = alignment.apply(transition.source_state, depth=transition.source_depth)
        target = alignment.apply(transition.target_state, depth=transition.target_depth)
        ds = float(mapped_depth[index + 1] - mapped_depth[index])
        if ds <= 1e-8:
            dropped += 1
            continue
        # Extra teacher layers inside two matched anchors are observed points
        # on a continuous interval, not missing dynamics.  Penalize their
        # interpolation uncertainty according to the bracket width.  Only a
        # transition outside any matched bracket receives the hard gap floor.
        if index in matched_teacher and index + 1 in matched_teacher:
            interpolation_confidence = 1.0
        else:
            left = max((anchor for anchor in matched_teacher if anchor <= index), default=None)
            right = min((anchor for anchor in matched_teacher if anchor >= index + 1), default=None)
            if left is not None and right is not None and right > left:
                bracket_fraction = float(right - left - 1) / max(1, teacher.layer_count - 1)
                interpolation_confidence = max(gap_weight_floor, float(np.exp(-0.5 * bracket_fraction)))
            else:
                interpolation_confidence = gap_weight_floor
        confidence = interpolation_confidence
        interpolation_weights.append(interpolation_confidence)
        if compatibility is not None and len(compatibility):
            confidence *= float(np.interp(mapped_depth[index], student.depth_coordinates[:-1], compatibility, left=compatibility[0], right=compatibility[-1]))
        confidence *= float(np.clip(path_score, 0.0, 1.0))
        if tangent_confidence is not None and len(tangent_confidence):
            midpoint = 0.5 * (mapped_depth[index] + mapped_depth[index + 1])
            confidence *= float(np.interp(midpoint, student.depth_coordinates, tangent_confidence, left=tangent_confidence[0], right=tangent_confidence[-1]))
        velocity = (target - source) / ds
        if jet_transport is not None:
            midpoint = 0.5 * (mapped_depth[index] + mapped_depth[index + 1])
            velocity = jet_transport.project_velocity(velocity, midpoint)
            confidence *= jet_transport.confidence_at(midpoint)
        elif tangent_transport is not None:
            midpoint = 0.5 * (mapped_depth[index] + mapped_depth[index + 1])
            tt_coords = tangent_transport.coordinates
            tt_mats = tangent_transport.transport_matrices
            if len(tt_coords) <= 1 or midpoint <= tt_coords[0]:
                local_transport = tt_mats[0]
            elif midpoint >= tt_coords[-1]:
                local_transport = tt_mats[-1]
            else:
                pos = int(np.searchsorted(tt_coords, midpoint))
                c0, c1 = float(tt_coords[pos - 1]), float(tt_coords[pos])
                alpha = (midpoint - c0) / max(c1 - c0, 1e-12)
                local_transport = (1.0 - alpha) * tt_mats[pos - 1] + alpha * tt_mats[pos]
            velocity = velocity @ local_transport.T
        samples.append(FlowSample(source, velocity, float(mapped_depth[index]), confidence, index, teacher.probe_id))
        weights.append(confidence)
    node_confidence = np.asarray([1.0 if index in matched_student else 0.25 for index in range(student.layer_count)], dtype=np.float64)
    return samples, node_confidence, dropped, float(np.mean(interpolation_weights)) if interpolation_weights else 0.0


def _aggregate_depth_reports(reports: list[DepthTransferReport]) -> DepthTransferReport:
    if not reports:
        raise ValueError("cannot aggregate empty depth reports")
    same_shape = len({report.node_confidence.shape for report in reports}) == 1
    node_confidence = np.mean([report.node_confidence for report in reports], axis=0) if same_shape else reports[0].node_confidence
    notes = ("correspondences/gaps are retained from the first paired trace; matched and gap counts are sums across all paired traces",) if len(reports) > 1 else ()
    if not same_shape:
        notes += ("student traces have different visible depths; node confidence is reported for the first trace",)
    return DepthTransferReport(reports[0].student_nodes, reports[0].teacher_nodes, int(np.sum([report.matched_nodes for report in reports])), int(np.sum([report.gap_count for report in reports])), int(np.sum([report.dropped_transitions for report in reports])), float(np.mean([report.matched_fraction for report in reports])), float(np.mean([report.mean_gap_confidence for report in reports])), node_confidence, reports[0].correspondences, reports[0].gaps, notes)


def fit_flow_transfer(
    student_traces: Sequence[TrajectoryTrace],
    teacher_traces: Sequence[TrajectoryTrace],
    teacher_to_student: AlignmentResult,
    *,
    ridge: float = 1e-4,
    min_samples_per_node: int = 2,
    bandwidth: float | None = None,
    gap_weight_floor: float = 0.25,
    signature_mode: str = "full",
    robust_loss: str = "huber",
    robust_iterations: int = 4,
    stability_seeds: Sequence[int] = (0, 17, 31),
    leave_one_probe_out: bool = True,
    math_components: dict[str, bool] | None = None,
    max_dense_features: int = DEFAULT_MAX_DENSE_FEATURES,
    scalable_rank: int = DEFAULT_SCALABLE_RANK,
    scalable_seed: int = 0,
    scalable_projection: np.ndarray | None = None,
) -> FlowFitResult:
    """Fit teacher flow in student coordinates using monotone continuous depth.

    ``teacher_to_student`` is directional and must map exactly from the
    teacher hidden dimension to the student hidden dimension. A reverse or
    dimensionally incompatible map fails before any fitting begins.
    """

    if len(student_traces) != len(teacher_traces) or not student_traces:
        raise ValueError("student and teacher trace sets must be paired and non-empty")
    if len({trace.probe_id for trace in student_traces}) != len(student_traces) or len({trace.probe_id for trace in teacher_traces}) != len(teacher_traces):
        raise ValueError("paired trace sets must not contain duplicate probe IDs")
    student_dim = student_traces[0].state_dim
    teacher_dim = teacher_traces[0].state_dim
    if teacher_to_student.source_dim != teacher_dim or teacher_to_student.target_dim != student_dim:
        raise ValueError(f"teacher_to_student alignment must have shape ({teacher_dim}, {student_dim}); got ({teacher_to_student.source_dim}, {teacher_to_student.target_dim})")
    declared_direction = teacher_to_student.metadata.get("direction")
    if declared_direction is not None and declared_direction != "teacher_to_student":
        raise ValueError(f"alignment direction is {declared_direction!r}; fit_flow_transfer requires teacher_to_student")
    if not (0.0 <= gap_weight_floor < 1.0):
        raise ValueError("gap_weight_floor must be in [0, 1)")
    if signature_mode not in {"none", "differential", "full"}:
        raise ValueError("signature_mode must be 'none', 'differential', or 'full'")
    if int(max_dense_features) < 1:
        raise ValueError("max_dense_features must be positive")
    if int(scalable_rank) < 1:
        raise ValueError("scalable_rank must be positive")
    for student, teacher in zip(student_traces, teacher_traces):
        if student.state_dim != student_dim or teacher.state_dim != teacher_dim:
            raise ValueError("all traces in each model set must have consistent dimensions")
        if student.probe_id != teacher.probe_id:
            raise ValueError(f"trace pairing mismatch: student probe {student.probe_id!r}, teacher probe {teacher.probe_id!r}")
    components = resolve_math_components(math_components)
    if math_components is not None and bool(math_components.get("ot", False)) and teacher_to_student.kind != "ot_barycentric":
        raise ValueError("math component 'ot' was explicitly enabled, but alignment kind is not ot_barycentric")
    dense_guard = dense_memory_estimate(
        student_dim,
        nodes=max(1, student_traces[0].layer_count - 1),
        quadratic=signature_mode == "full",
    )
    if requires_scalable_backend(student_dim, max_dense_features=max_dense_features, quadratic=signature_mode == "full"):
        # Keep the guard before every dense design/tensor construction.  The
        # imported function only allocates O(d*r), then fits the field in the
        # compact chart; it never materializes d*d or d*d*d features.
        from .scalable import fit_scalable_flow_transfer

        safe_rank = min(int(scalable_rank), student_dim, max(1, int(np.sqrt(max_dense_features))))
        if safe_rank < 1:
            raise ValueError("scalable chart rank is zero under max_dense_features")
        return fit_scalable_flow_transfer(
            student_traces,
            teacher_traces,
            teacher_to_student,
            signature_mode=signature_mode,
            rank=safe_rank,
            seed=int(scalable_seed),
            max_dense_features=int(max_dense_features),
            math_components=math_components,
            projection=scalable_projection,
        )
    fit_robust_loss = robust_loss if components["robust_loss"] else "none"
    use_quadratic = signature_mode == "full" and components["quadratic_flow"] and components["hessian_2jet"]
    # The student operator is the measured baseline and stays deliberately
    # low variance.  A full transfer estimates the richer quadratic channel
    # from the teacher target; it must not spend the bottlenecked student
    # sample budget fitting an independent quadratic model before correction.
    student_flow = fit_flow_operator(student_traces, ridge=ridge, min_samples_per_node=min_samples_per_node, bandwidth=bandwidth, quadratic=False, robust_loss=fit_robust_loss, robust_iterations=robust_iterations)
    tangent_transport = None
    tangent_note = ""
    if components["tangent_projector_transport"]:
        try:
            tangent_transport = fit_tangent_transport(student_traces, teacher_traces, teacher_to_student)
        except ValueError as error:
            # A mixed connector may expose different per-probe node grids. The
            # core remains usable, but it reports why tangent evidence was absent.
            tangent_note = f"tangent transport unavailable: {error}"
    else:
        tangent_note = "tangent/projector transport explicitly disabled"
    correspondences: list[MonotoneCorrespondence] = []
    for student, teacher in zip(student_traces, teacher_traces):
        mapped_teacher_states = teacher_to_student.apply(teacher.hidden_states, depth=teacher.depth_coordinates)
        mapped_teacher_velocity = teacher_to_student.linear_apply(
            np.asarray([transition.vector_field for transition in teacher.transitions]),
            depth=teacher.depth_coordinates[:-1],
        )
        correspondence = monotone_correspondence(
            student.hidden_states,
            mapped_teacher_states,
            student_coordinates=student.depth_coordinates,
            teacher_coordinates=teacher.depth_coordinates,
            gap_penalty=1.0 - gap_weight_floor,
            student_vector_fields=np.asarray([transition.vector_field for transition in student.transitions]),
            teacher_vector_fields=mapped_teacher_velocity,
            student_curvature=np.asarray([transition.curvature for transition in student.transitions]) if components["curvature"] else np.zeros(len(student.transitions)),
            teacher_curvature=np.asarray([transition.curvature for transition in teacher.transitions]) if components["curvature"] else np.zeros(len(teacher.transitions)),
            student_uncertainty=np.zeros(student.layer_count, dtype=np.float64) if student.uncertainty is None else student.uncertainty,
            teacher_uncertainty=np.zeros(teacher.layer_count, dtype=np.float64) if teacher.uncertainty is None else teacher.uncertainty,
            curvature_weight=0.20 if components["curvature"] else 0.0,
        )
        correspondences.append(correspondence)
    jet_transport: LocalJetTransport | None = None
    jet_note = ""
    if components["local_1jet"] and signature_mode in {"differential", "full"}:
        try:
            jet_transport = fit_rank_aware_local_jet_transport(
                student_traces,
                teacher_traces,
                teacher_to_student,
                correspondences,
                rank=min(student_dim, teacher_dim),
                quadratic=signature_mode == "full" and components["hessian_2jet"],
                robust_loss=fit_robust_loss,
                robust_iterations=4,
            )
        except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as error:
            jet_note = f"rank-aware local jet unavailable: {type(error).__name__}: {error}"
    teacher_samples: list[FlowSample] = []
    depth_reports: list[DepthTransferReport] = []
    signature_scores = []
    for pair_index, (student, teacher) in enumerate(zip(student_traces, teacher_traces)):
        correspondence = correspondences[pair_index]
        mapped_depth, _, _, _ = _depth_map(teacher, student, correspondence)
        compatibility = None
        path_score = 1.0
        if components["path_signature"] and signature_mode in {"differential", "full"}:
            compatibility, path_score = differential_compatibility(student, teacher, mapped_depth)
        tangent_confidence = None if tangent_transport is None else tangent_transport.confidence
        if signature_mode != "full" or not components["tangent_projector_transport"]:
            tangent_confidence = None
            path_score = 1.0
        samples, node_confidence, dropped, mean_weight = _transport_teacher_samples(student, teacher, teacher_to_student, correspondence, gap_weight_floor, compatibility, tangent_confidence, tangent_transport if signature_mode == "full" and jet_transport is None and components["tangent_projector_transport"] else None, jet_transport if signature_mode in {"differential", "full"} and components["local_1jet"] else None, path_score)
        teacher_samples.extend(samples)
        pair_count = len(correspondence.pairs)
        endpoint_anchor_added = any("endpoint anchor retained" in gap.reason for gap in correspondence.gaps)
        # ``pairs`` includes (0, 0).  When the DP reaches the final corner
        # through gaps, the protocol adds a synthetic endpoint *anchor* to
        # make interpolation well-defined.  That anchor is not a DP diagonal
        # move, so its arithmetic has a separate, explicit branch.
        expected_gap_count = (
            student.layer_count + teacher.layer_count - 2 * pair_count
            if not endpoint_anchor_added
            else student.layer_count + teacher.layer_count + 3 - 2 * pair_count
        )
        # The dynamic-programming path consumes one student or teacher node on
        # every gap move.  The explicit endpoint anchor can add one reported
        # interval after backtrace.  Keep this arithmetic in the artifact so
        # an aggregate gap count cannot be mistaken for a different pairing.
        gap_count_consistent = len(correspondence.gaps) in {expected_gap_count, expected_gap_count + 1}
        signature_scores.append({
            "probe_id": student.probe_id,
            "path_score": float(path_score),
            "mean_differential_weight": 1.0 if compatibility is None else float(np.mean(compatibility)),
            "correspondence_cost": float(correspondence.cost),
            "correspondence_confidence": float(correspondence.confidence),
            "correspondence_metadata": correspondence.metadata,
            "student_nodes": int(student.layer_count),
            "teacher_nodes": int(teacher.layer_count),
            "matched_nodes": int(pair_count),
            "student_coverage": float(pair_count / max(student.layer_count, 1)),
            "teacher_coverage": float(pair_count / max(teacher.layer_count, 1)),
            "matched_fraction": float(pair_count / max(student.layer_count, teacher.layer_count)),
            "gap_count": int(len(correspondence.gaps)),
            "expected_gap_count": int(expected_gap_count),
            "endpoint_anchor_added": bool(endpoint_anchor_added),
            "gap_count_consistent": bool(gap_count_consistent),
            "pairs": [[int(a), int(b)] for a, b in correspondence.pairs],
        })
        depth_reports.append(DepthTransferReport(student.layer_count, teacher.layer_count, len(correspondence.pairs), len(correspondence.gaps), dropped, len(correspondence.pairs) / max(student.layer_count, teacher.layer_count), mean_weight, node_confidence, correspondence.pairs, correspondence.gaps, ("path-level transition/curvature/uncertainty costs are included",)))
    if not teacher_samples:
        raise ValueError("all teacher transitions were dropped by depth mapping")
    # Fit the first-order field independently from the optional quadratic
    # channel.  Sharing an uncentred [x, 1, x⊗x] design lets a harmless
    # quadratic ridge bias the linear Jacobian even when the planted system is
    # exactly affine.  Keeping the channels separate improves fidelity while
    # retaining the directional 2-jet as an explicit additional term.
    teacher_linear = fit_flow_operator(student_traces[:1], coordinates=student_flow.coordinates, observations=teacher_samples, ridge=ridge, min_samples_per_node=min_samples_per_node, bandwidth=bandwidth, quadratic=False, robust_loss=fit_robust_loss, robust_iterations=robust_iterations)
    teacher_quadratic = None if not use_quadratic else fit_flow_operator(student_traces[:1], coordinates=student_flow.coordinates, observations=teacher_samples, ridge=ridge, min_samples_per_node=min_samples_per_node, bandwidth=bandwidth, quadratic=True, quadratic_ridge=max(ridge, 0.05), robust_loss=fit_robust_loss, robust_iterations=robust_iterations)
    teacher_flow = teacher_linear if teacher_quadratic is None else FlowOperator(
        teacher_linear.coordinates,
        teacher_linear.matrices,
        teacher_linear.biases,
        teacher_linear.sample_counts,
        teacher_linear.residual_scales,
        teacher_linear.spectral_norms,
        {**teacher_linear.metadata, "quadratic_channel_fitted_separately": True},
        teacher_quadratic.quadratic_terms,
    )
    ensemble_spread, loo_spread = _fit_stability_diagnostics(
        teacher_flow,
        teacher_samples,
        coordinates=student_flow.coordinates,
        ridge=ridge,
        min_samples_per_node=min_samples_per_node,
        bandwidth=bandwidth,
        quadratic=use_quadratic,
        quadratic_ridge=max(ridge, 0.05),
        robust_loss=fit_robust_loss,
        robust_iterations=robust_iterations,
        seeds=stability_seeds,
        leave_one_probe_out=leave_one_probe_out and components["stability_barriers"],
    ) if components["stability_barriers"] else (np.zeros(len(student_flow.coordinates)), np.zeros(len(student_flow.coordinates)))
    # A jet can refine a genuinely nonlinear or cross-dimensional target.  On
    # an exactly affine same-chart system the independently fitted first-order
    # field is the lower-variance estimate, so do not blend a noisier local
    # basis into it.
    jet_matrix_blend = jet_transport is not None and (teacher_dim != student_dim or float(np.max(teacher_linear.residual_scales)) > 1e-5)
    if jet_matrix_blend:
        jet_confidence = np.clip(jet_transport.confidence, 0.0, 1.0)
        blended_matrices = (1.0 - jet_confidence[:, None, None]) * teacher_flow.matrices + jet_confidence[:, None, None] * jet_transport.transported_jacobians
        blended_norms = np.asarray([_spectral_norm_diagnostic(matrix)[0] for matrix in blended_matrices])
        teacher_flow = FlowOperator(
            teacher_flow.coordinates,
            blended_matrices,
            teacher_flow.biases,
            teacher_flow.sample_counts,
            teacher_flow.residual_scales,
            blended_norms,
            {**teacher_flow.metadata, "rank_aware_jet_blend": True, "jet_blend_confidence": jet_confidence.tolist()},
            teacher_flow.quadratic_terms,
        )
    if use_quadratic:
        student_flow = FlowOperator(
            student_flow.coordinates,
            student_flow.matrices,
            student_flow.biases,
            student_flow.sample_counts,
            student_flow.residual_scales,
            student_flow.spectral_norms,
            {**student_flow.metadata, "student_baseline_quadratic": "zero_teacher_channel_only"},
            np.zeros_like(teacher_flow.quadratic_terms),
        )
    # A fit-flow artifact is a train-fitted transported teacher target.  Make
    # that contract intrinsic to the operator so a later scorecard can verify
    # identity instead of relying on a CLI wrapper to add the labels.
    teacher_flow = replace(teacher_flow, metadata={
        **dict(teacher_flow.metadata),
        "target_fit_split": "train",
        "metric_scope": "transported_teacher_operator_approximation",
    })
    report = _aggregate_depth_reports(depth_reports)
    all_student_states = np.concatenate([trace.hidden_states for trace in student_traces], axis=0)
    all_teacher_states_original = np.concatenate([trace.hidden_states for trace in teacher_traces], axis=0)
    all_teacher_states_student = np.concatenate([
        teacher_to_student.apply(trace.hidden_states, depth=trace.depth_coordinates)
        for trace in teacher_traces
    ], axis=0)
    all_teacher_velocities_student = np.asarray([sample.velocity for sample in teacher_samples])
    capacity = diagnose_capacity(all_student_states, all_teacher_states_original, all_teacher_states_student, all_teacher_velocities_student, teacher_to_student)
    delta_matrices = teacher_flow.matrices - student_flow.matrices
    delta_biases = teacher_flow.biases - student_flow.biases
    delta_quadratic = None if not use_quadratic else teacher_flow.quadratic_terms - student_flow.quadratic_terms
    if components["capacity_projection"]:
        projected_matrices = []
        projected_biases = []
        for matrix, bias in zip(delta_matrices, delta_biases):
            projected_matrix, projected_bias = project_correction_to_student_tangent(matrix, bias, all_student_states)
            projected_matrices.append(projected_matrix)
            projected_biases.append(projected_bias)
        delta_matrices = np.asarray(projected_matrices)
        delta_biases = np.asarray(projected_biases)
    # Remove the component the measured student chart cannot express before
    # the trust-region solver sees it. This is a local capability gate, not a
    # claim that the compressed target has been recovered.
    capacity_gate = float(np.clip(min(capacity.state_subspace_coverage, capacity.velocity_subspace_coverage, 1.0 - capacity.irreducible_mismatch), 0.0, 1.0)) if components["capacity_projection"] else 1.0
    delta_matrices *= capacity_gate
    delta_biases *= capacity_gate
    if delta_quadratic is not None:
        delta_quadratic *= capacity_gate
    local_tangent_gate = np.ones(len(delta_matrices), dtype=np.float64)
    if tangent_transport is not None and signature_mode == "full" and components["tangent_projector_transport"]:
        local_tangent_gate = np.interp(student_flow.coordinates, tangent_transport.coordinates, tangent_transport.confidence, left=tangent_transport.confidence[0], right=tangent_transport.confidence[-1])
        delta_matrices *= local_tangent_gate[:, None, None]
        delta_biases *= local_tangent_gate[:, None]
        if delta_quadratic is not None:
            delta_quadratic *= local_tangent_gate[:, None, None, None]
    support = np.minimum(teacher_flow.sample_counts, student_flow.sample_counts)
    support_factor = support / (support + 2.0)
    gap_factor = np.clip(report.node_confidence[: len(support)], 0.0, 1.0)
    teacher_velocity_scale = max(1.0, float(np.median(np.asarray(stable_l2(all_teacher_velocities_student, axis=1, name="teacher velocity scale")))))
    student_velocity_values = np.asarray([transition.vector_field for trace in student_traces for transition in trace.transitions], dtype=np.float64)
    student_velocity_scale = max(1.0, float(np.median(np.asarray(stable_l2(student_velocity_values, axis=1, name="student velocity scale")))))
    residual_factor = np.exp(-np.clip(teacher_flow.residual_scales / teacher_velocity_scale + student_flow.residual_scales / student_velocity_scale, 0.0, 40.0))
    # The report's aggregate gap confidence measures the actual continuous
    # interpolation quality.  It is high for dense teacher observations inside
    # matched brackets and low for extrapolated unmatched regions.
    depth_factor = float(np.clip(report.mean_gap_confidence, 0.0, 1.0))
    confidence = support_factor * gap_factor * depth_factor * residual_factor
    if teacher_to_student.condition_number > 1e4:
        confidence *= 1e4 / teacher_to_student.condition_number
    confidence *= capacity.confidence_penalty
    stability_factor = np.exp(-np.clip(ensemble_spread + loo_spread, 0.0, 40.0)) if components["stability_barriers"] else np.ones(len(confidence))
    confidence *= stability_factor
    confidence = np.clip(confidence, 0.0, 1.0)
    validation_errors = []
    for trace in student_traces:
        for transition in trace.transitions:
            predicted = student_flow.predict(transition.source_state, transition.source_depth)
            validation_errors.append(np.linalg.norm(predicted - transition.vector_field))
    validation_error = float(np.mean(validation_errors)) if validation_errors else float("inf")
    split_labels = {trace.metadata.get("probe_split") for trace in student_traces if trace.metadata.get("probe_split") is not None}
    split_label = next(iter(split_labels)) if len(split_labels) == 1 else "unspecified"
    metadata = {"alignment_kind": teacher_to_student.kind, "alignment_direction": "teacher_to_student", "alignment_fit_scope": teacher_to_student.metadata.get("alignment_fit_scope", "unspecified"), "alignment_paired_error": teacher_to_student.paired_error, "alignment_relational_error": teacher_to_student.relational_error, "ot_mass_error": teacher_to_student.ot_mass_error, "continuous_depth": True, "depth_map_direction": "teacher_depth_to_student_depth", "depth_map_domain": {"teacher": [float(teacher_traces[0].depth_coordinates[0]), float(teacher_traces[0].depth_coordinates[-1])], "student": [float(student_traces[0].depth_coordinates[0]), float(student_traces[0].depth_coordinates[-1])]}, "nearest_node_grouping": False, "path_level_correspondence": True, "support_factor": support_factor, "gap_confidence": gap_factor, "depth_interpolation_confidence": depth_factor, "support_count_definition": teacher_flow.metadata.get("support_count_definition"), "teacher_confidence_mass": teacher_flow.metadata.get("confidence_mass"), "teacher_velocity_scale": teacher_velocity_scale, "student_velocity_scale": student_velocity_scale, "residual_confidence_factor": residual_factor.tolist(), "student_manifold_projection": components["capacity_projection"], "capacity_gate": capacity_gate, "local_tangent_gate": local_tangent_gate.tolist(), "signature_mode": signature_mode, "multi_scale_path_signatures": components["path_signature"] and signature_mode in {"differential", "full"}, "differential_signature_weighting": components["path_signature"] and signature_mode in {"differential", "full"}, "curvature_weight": 0.20 if components["curvature"] else 0.0, "tangent_transport": tangent_transport is not None and signature_mode == "full" and components["tangent_projector_transport"], "tangent_velocity_transport": tangent_transport is not None and signature_mode == "full" and components["tangent_projector_transport"], "rank_aware_local_jet_transport": jet_transport is not None and components["local_1jet"], "jet_hessian_kind": None if jet_transport is None else "directional_sketch", "jet_confidence": None if jet_transport is None else jet_transport.confidence.tolist(), "jet_confidence_interval": None if jet_transport is None else {"low": jet_transport.confidence_low.tolist(), "high": jet_transport.confidence_high.tolist()}, "jet_residual_budget": None if jet_transport is None else jet_transport.residual_budget.tolist(), "robust_loss": fit_robust_loss, "robust_iterations": robust_iterations, "deterministic_stability_seeds": [int(seed) for seed in stability_seeds] if components["stability_barriers"] else [], "leave_one_probe_out": bool(leave_one_probe_out and components["stability_barriers"]), "ensemble_spread": ensemble_spread.tolist(), "leave_one_probe_out_spread": loo_spread.tolist(), "stability_factor": stability_factor.tolist(), "stability_confidence_interval": {"low": np.clip(confidence - 1.96 * (ensemble_spread + loo_spread), 0.0, 1.0).tolist(), "high": np.clip(confidence + 1.96 * (ensemble_spread + loo_spread), 0.0, 1.0).tolist()}, "state_feature_degree": 2 if use_quadratic else 1, "signature_scores": signature_scores, "depth_correspondence_summary": {"mean_matched_nodes": float(np.mean([item["matched_nodes"] for item in signature_scores])), "mean_student_coverage": float(np.mean([item["student_coverage"] for item in signature_scores])), "mean_teacher_coverage": float(np.mean([item["teacher_coverage"] for item in signature_scores])), "mean_matched_fraction": float(np.mean([item["matched_fraction"] for item in signature_scores])), "mean_gap_count": float(np.mean([item["gap_count"] for item in signature_scores])), "all_gap_counts_consistent": bool(all(item["gap_count_consistent"] for item in signature_scores))}, "probe_ids": tuple(trace.probe_id for trace in student_traces), "training_split": split_label, "target_fit_split": "train", "metric_scope": "transported_teacher_operator_approximation", "dense_memory_guard": dense_guard, "scalable_backend": "dense_local_field", "max_dense_features": int(max_dense_features), "requested_signature_mode": signature_mode, "effective_signature_mode": signature_mode}
    if tangent_note:
        metadata["tangent_transport_note"] = tangent_note
    if jet_note:
        metadata["jet_transport_note"] = jet_note
    metadata["jet_matrix_blend_applied"] = bool(jet_matrix_blend)
    metadata["training_student_flow_error"] = validation_error
    metadata["math_components"] = components
    metadata["math_profile"] = build_math_profile(metadata, alignment_kind=teacher_to_student.kind, signature_mode=signature_mode, correction_matrices=delta_matrices, correction_quadratic=delta_quadratic, requested=math_components)
    return FlowFitResult(student_flow, teacher_flow, delta_matrices, delta_biases, confidence, validation_error, metadata, report, capacity, delta_quadratic)


def calibrate_flow_confidence(flow_fit: FlowFitResult, validation_student: Sequence[TrajectoryTrace], validation_teacher: Sequence[TrajectoryTrace], teacher_to_student: AlignmentResult, *, signature_mode: str | None = None, math_components: dict[str, bool] | None = None, stability_seeds: Sequence[int] | None = None) -> FlowFitResult:
    """Calibrate node confidence on a disjoint validation probe split.

    The validation split is refit only to estimate out-of-sample stability;
    the returned correction remains the training fit. This avoids treating a
    low training residual as evidence of transferable dynamics.
    """

    mode = signature_mode or str(flow_fit.metadata.get("requested_signature_mode", flow_fit.metadata.get("signature_mode", "full")))
    train_ids = set(str(item) for item in flow_fit.metadata.get("probe_ids", ()))
    if not train_ids:
        train_ids = set()
    validation_ids = {trace.probe_id for trace in validation_student}
    if train_ids.intersection(validation_ids):
        raise ValueError("validation confidence calibration reuses training probe IDs")
    if len(validation_student) != len(validation_teacher) or any(a.probe_id != b.probe_id for a, b in zip(validation_student, validation_teacher)):
        raise ValueError("validation teacher/student traces must remain paired")
    inherited_components = math_components if math_components is not None else flow_fit.metadata.get("math_components")
    if math_components is None and inherited_components is not None and teacher_to_student.kind != "ot_barycentric":
        # ``fit_flow_transfer`` treats an explicitly enabled OT flag as a
        # contract request.  The stored resolved defaults contain ot=True even
        # for a non-OT alignment, so normalize inherited metadata here rather
        # than turning confidence calibration into a false capability error.
        inherited_components = dict(inherited_components)
        inherited_components["ot"] = False
    validation_kwargs: dict[str, object] = {
        "signature_mode": mode,
        "math_components": inherited_components,
        "stability_seeds": tuple(stability_seeds) if stability_seeds is not None else tuple(flow_fit.metadata.get("deterministic_stability_seeds", (0, 17, 31))),
    }
    if flow_fit.metadata.get("scalable_backend") == "randomized_latent_compression":
        validation_kwargs.update({
            "max_dense_features": int(flow_fit.metadata.get("max_dense_features", DEFAULT_MAX_DENSE_FEATURES)),
            "scalable_rank": int(flow_fit.metadata.get("scalable_chart_rank", DEFAULT_SCALABLE_RANK)),
            "scalable_seed": int(flow_fit.metadata.get("scalable_seed", 0)),
            # Coefficient stability is meaningful only in one chart.  Reuse
            # the train projection instead of fitting a fresh validation chart
            # whose basis rotation would be mistaken for transfer uncertainty.
            "scalable_projection": flow_fit.student.chart_projection,
        })
    validation_fit = fit_flow_transfer(validation_student, validation_teacher, teacher_to_student, **validation_kwargs)
    if validation_fit.student.matrices.shape != flow_fit.student.matrices.shape:
        raise ValueError("validation flow has a different student node grid; confidence calibration needs comparable depth nodes")
    train_target = flow_fit.student.matrices + flow_fit.correction_matrices
    target_delta = train_target - validation_fit.transported_teacher.matrices
    target_scale = np.maximum(1.0, np.linalg.norm(validation_fit.transported_teacher.matrices, axis=(1, 2)))
    target_error = np.linalg.norm(target_delta, axis=(1, 2)) / target_scale
    if flow_fit.correction_quadratic is not None and validation_fit.transported_teacher.quadratic_terms is not None:
        quadratic_difference = flow_fit.student.quadratic_terms + flow_fit.correction_quadratic - validation_fit.transported_teacher.quadratic_terms
        target_error += np.linalg.norm(quadratic_difference.reshape(len(quadratic_difference), -1), axis=1) / np.maximum(1.0, np.linalg.norm(validation_fit.transported_teacher.quadratic_terms.reshape(len(validation_fit.transported_teacher.quadratic_terms), -1), axis=1))
    train_student_error = np.linalg.norm(flow_fit.student.matrices - validation_fit.student.matrices, axis=(1, 2)) / np.maximum(1.0, np.linalg.norm(validation_fit.student.matrices, axis=(1, 2)))
    factor = np.exp(-np.clip(target_error + 0.5 * train_student_error, 0.0, 40.0))
    confidence = np.clip(flow_fit.confidence * factor, 0.0, 1.0)
    metadata = dict(flow_fit.metadata)
    metadata.update({"confidence_calibrated": True, "calibration_probe_count": len(validation_student), "calibration_split": "validation", "calibration_node_error": target_error.tolist(), "calibration_factor": factor.tolist()})
    return replace(flow_fit, confidence=confidence, metadata=metadata)
