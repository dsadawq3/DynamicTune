"""Core immutable-ish data contracts for the emergent-flow pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def finite_array(value: Any, *, ndim: int | None = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def stable_l2(value: Any, *, axis: int | tuple[int, ...] | None = None, name: str = "array") -> np.ndarray | float:
    """Finite L2 norm with max-abs scaling before reduction.

    ``numpy.linalg.norm`` can overflow for a finite float64 vector because it
    squares before summing.  Scaling preserves the representable result and
    lets callers distinguish an impossible true distance from an arithmetic
    overflow.
    """

    arr = finite_array(value, name=name)
    if arr.size == 0:
        return 0.0 if axis is None else np.zeros(np.asarray(arr).sum(axis=axis).shape)
    if axis is None:
        scale = max(1.0, float(np.max(np.abs(arr))))
        result = float(np.linalg.norm(arr / scale) * scale)
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        scale = np.maximum(1.0, np.max(np.abs(arr), axis=axes, keepdims=True))
        # ``numpy.linalg.norm`` only supports one or two reduction axes.
        # Explicit scaled sum of squares keeps the same overflow protection
        # for arbitrary tensor reductions used by Hessian/jet diagnostics.
        scaled_squared = np.sum(np.square(arr / scale), axis=axes)
        result = np.sqrt(scaled_squared) * np.squeeze(scale, axis=axes)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"{name} norm overflowed despite finite inputs")
    return result


@dataclass(frozen=True)
class CapabilityMatrix:
    """What a connector can observe, with an explicit reason for omissions."""

    hidden_states: bool = True
    residual_states: bool = False
    layer_transitions: bool = True
    vector_fields: bool = True
    jacobian_sketch: bool = False
    hessian_sketch: bool = False
    attention_geometry: bool = False
    normalization_geometry: bool = False
    weight_surgery: bool = False
    token_position_correspondence: bool = False
    reasons: Mapping[str, str] = field(default_factory=dict)
    automatic: Mapping[str, bool] = field(default_factory=dict)
    required_callbacks: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            name: bool(getattr(self, name))
            for name in (
                "hidden_states",
                "residual_states",
                "layer_transitions",
                "vector_fields",
                "jacobian_sketch",
                "hessian_sketch",
                "attention_geometry",
                "normalization_geometry",
                "weight_surgery",
                "token_position_correspondence",
            )
        }
        result["reasons"] = dict(self.reasons)
        result["automatic"] = dict(self.automatic)
        result["required_callbacks"] = dict(self.required_callbacks)
        return result


@dataclass(frozen=True)
class Probe:
    """A structured probe. Text is optional; the generator's structure is primary."""

    probe_id: str
    family: str
    payload: Mapping[str, Any]
    initial_state: np.ndarray
    pair_id: str | None = None
    perturbation: np.ndarray | None = None
    split: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_state", finite_array(self.initial_state, ndim=1, name="initial_state"))
        if self.perturbation is not None:
            object.__setattr__(self, "perturbation", finite_array(self.perturbation, ndim=1, name="perturbation"))


@dataclass(frozen=True)
class ProbeSplit:
    train: tuple[Probe, ...]
    validation: tuple[Probe, ...]
    holdout: tuple[Probe, ...]

    def all(self) -> tuple[Probe, ...]:
        return self.train + self.validation + self.holdout


@dataclass(frozen=True)
class TransitionObservation:
    source_layer: int
    target_layer: int
    source_depth: float
    target_depth: float
    source_state: np.ndarray
    target_state: np.ndarray
    delta: np.ndarray
    vector_field: np.ndarray
    jacobian: np.ndarray | None = None
    hessian_sketch: np.ndarray | None = None
    curvature: float = 0.0
    singular_values: np.ndarray | None = None
    normalization_geometry: Mapping[str, Any] | None = None
    attention_geometry: Mapping[str, Any] | None = None
    uncertainty: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source_state", "target_state", "delta", "vector_field"):
            object.__setattr__(self, name, finite_array(getattr(self, name), ndim=1, name=name))
        dimension = self.source_state.size
        if self.target_state.size != dimension or self.delta.size != dimension or self.vector_field.size != dimension:
            raise ValueError("transition state and vector dimensions are inconsistent")
        if self.jacobian is not None:
            jacobian = finite_array(self.jacobian, ndim=2, name="jacobian")
            # Exact coordinate Jacobians are square. A scalable observation
            # may instead store the explicit directional product J U with
            # shape [output_dim, k]; k is recorded in trace metadata.
            if jacobian.shape[0] != dimension or jacobian.shape[1] < 1:
                raise ValueError("jacobian must have observed output dimension and at least one coordinate or sketch direction")
            object.__setattr__(self, "jacobian", jacobian)
        if self.hessian_sketch is not None:
            hessian = finite_array(self.hessian_sketch, ndim=2, name="hessian_sketch")
            if hessian.shape[1] != dimension:
                raise ValueError("hessian sketch direction dimension does not match the state chart")
            object.__setattr__(self, "hessian_sketch", hessian)
        if self.singular_values is not None:
            object.__setattr__(self, "singular_values", finite_array(self.singular_values, ndim=1, name="singular_values"))
        if not np.isfinite(self.curvature) or self.target_depth <= self.source_depth:
            raise ValueError("transition depth and curvature must be finite and strictly increasing")
        if any(not np.isfinite(float(value)) or float(value) < 0.0 for value in self.uncertainty.values()):
            raise ValueError("transition uncertainty must be finite and non-negative")


@dataclass(frozen=True)
class TrajectoryTrace:
    model_id: str
    probe_id: str
    layer_ids: tuple[int, ...]
    depth_coordinates: np.ndarray
    hidden_states: np.ndarray
    residual_states: np.ndarray | None
    transitions: tuple[TransitionObservation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    feature_space: str = "hidden"
    token_ids: np.ndarray | None = None
    position_ids: np.ndarray | None = None
    token_position_map: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: np.ndarray | None = None

    def __post_init__(self) -> None:
        coordinates = finite_array(self.depth_coordinates, ndim=1, name="depth_coordinates")
        states = finite_array(self.hidden_states, ndim=2, name="hidden_states")
        if len(self.layer_ids) != len(coordinates) or states.shape[0] != len(coordinates):
            raise ValueError("layer_ids, depth_coordinates, and hidden_states must have equal length")
        if len(coordinates) < 2 or np.any(np.diff(coordinates) <= 0):
            raise ValueError("depth_coordinates must contain at least two strictly increasing values")
        if self.residual_states is not None:
            residual = finite_array(self.residual_states, ndim=2, name="residual_states")
            if residual.shape != states.shape:
                raise ValueError("residual_states must have the same shape as hidden_states")
            object.__setattr__(self, "residual_states", residual)
        if len(self.transitions) != len(coordinates) - 1:
            raise ValueError("one transition is required for each adjacent pair of states")
        if not self.feature_space or not isinstance(self.feature_space, str):
            raise ValueError("feature_space must be a non-empty string")
        for name in ("token_ids", "position_ids"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value)
                if array.ndim != 1:
                    raise ValueError(f"{name} must be one-dimensional")
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} contains non-finite values")
                object.__setattr__(self, name, array.astype(np.int64, copy=False))
        if self.token_ids is not None and self.position_ids is not None and len(self.token_ids) != len(self.position_ids):
            raise ValueError("token_ids and position_ids must have equal lengths")
        if self.uncertainty is not None:
            uncertainty = finite_array(self.uncertainty, ndim=1, name="trajectory uncertainty")
            if len(uncertainty) != len(coordinates):
                raise ValueError("trajectory uncertainty must have one value per depth node")
            object.__setattr__(self, "uncertainty", np.maximum(uncertainty, 0.0))
        object.__setattr__(self, "depth_coordinates", coordinates)
        object.__setattr__(self, "hidden_states", states)
        object.__setattr__(self, "layer_ids", tuple(int(x) for x in self.layer_ids))

    @property
    def state_dim(self) -> int:
        return int(self.hidden_states.shape[1])

    @property
    def layer_count(self) -> int:
        return len(self.layer_ids)


@dataclass(frozen=True)
class GapInterval:
    source_index: int
    target_index: int
    source_depth: float
    target_depth: float
    uncertainty: float
    reason: str


@dataclass(frozen=True)
class BifurcationPoint:
    depth: float
    level: str
    score: float
    confidence: float
    evidence: Mapping[str, float]


@dataclass(frozen=True)
class AlignmentResult:
    kind: str
    source_mean: np.ndarray
    target_mean: np.ndarray
    matrix: np.ndarray
    bias: np.ndarray
    source_rank: int
    target_rank: int
    paired_error: float
    relational_error: float
    cycle_error: float
    condition_number: float
    ot_mass_error: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_projection: np.ndarray | None = None
    target_projection: np.ndarray | None = None
    depth_coordinates: np.ndarray | None = None
    depth_matrices: np.ndarray | None = None
    depth_biases: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_mean", finite_array(self.source_mean, ndim=1, name="source_mean"))
        object.__setattr__(self, "target_mean", finite_array(self.target_mean, ndim=1, name="target_mean"))
        object.__setattr__(self, "matrix", finite_array(self.matrix, ndim=2, name="matrix"))
        object.__setattr__(self, "bias", finite_array(self.bias, ndim=1, name="bias"))
        if (self.source_projection is None) != (self.target_projection is None):
            raise ValueError("source_projection and target_projection must be supplied together")
        if self.source_projection is None:
            if self.matrix.shape != (self.source_mean.size, self.target_mean.size) or self.bias.size != self.target_mean.size:
                raise ValueError("alignment dimensions are inconsistent")
        else:
            source_projection = finite_array(self.source_projection, ndim=2, name="source chart projection")
            target_projection = finite_array(self.target_projection, ndim=2, name="target chart projection")
            if source_projection.shape[0] != self.source_mean.size or target_projection.shape[0] != self.target_mean.size:
                raise ValueError("alignment chart projections do not match original chart dimensions")
            if self.matrix.shape != (source_projection.shape[1], target_projection.shape[1]) or self.bias.size != target_projection.shape[1]:
                raise ValueError("compact alignment dimensions are inconsistent")
            object.__setattr__(self, "source_projection", source_projection)
            object.__setattr__(self, "target_projection", target_projection)
        depth_values = self.depth_coordinates
        depth_matrix_values = self.depth_matrices
        depth_bias_values = self.depth_biases
        if (depth_values is None) != (depth_matrix_values is None) or (depth_values is None) != (depth_bias_values is None):
            raise ValueError("depth-conditioned alignment requires coordinates, matrices, and biases together")
        if depth_values is not None and depth_matrix_values is not None and depth_bias_values is not None:
            coordinates = finite_array(depth_values, ndim=1, name="depth-conditioned alignment coordinates")
            matrices = finite_array(depth_matrix_values, ndim=3, name="depth-conditioned alignment matrices")
            biases = finite_array(depth_bias_values, ndim=2, name="depth-conditioned alignment biases")
            if len(coordinates) < 1 or np.any(np.diff(coordinates) <= 0):
                raise ValueError("depth-conditioned alignment coordinates must be strictly increasing")
            if matrices.shape[0] != len(coordinates) or matrices.shape[1:] != self.matrix.shape or biases.shape != (len(coordinates), self.bias.size):
                raise ValueError("depth-conditioned alignment arrays have inconsistent shapes")
            object.__setattr__(self, "depth_coordinates", coordinates)
            object.__setattr__(self, "depth_matrices", matrices)
            object.__setattr__(self, "depth_biases", biases)

    @property
    def source_dim(self) -> int:
        return int(self.source_mean.size)

    @property
    def target_dim(self) -> int:
        return int(self.target_mean.size)

    def _depth_operator(self, depth: float) -> tuple[np.ndarray, np.ndarray]:
        if self.depth_coordinates is None or self.depth_matrices is None or self.depth_biases is None:
            return self.matrix, self.bias
        if not np.isfinite(depth):
            raise ValueError("alignment depth must be finite")
        coordinates = self.depth_coordinates
        if depth <= coordinates[0]:
            return self.depth_matrices[0], self.depth_biases[0]
        if depth >= coordinates[-1]:
            return self.depth_matrices[-1], self.depth_biases[-1]
        index = int(np.searchsorted(coordinates, depth, side="right") - 1)
        fraction = (float(depth) - coordinates[index]) / (coordinates[index + 1] - coordinates[index])
        return (
            (1.0 - fraction) * self.depth_matrices[index] + fraction * self.depth_matrices[index + 1],
            (1.0 - fraction) * self.depth_biases[index] + fraction * self.depth_biases[index + 1],
        )

    def _apply_depth_conditioned(self, values: np.ndarray, depth: Any, *, linear: bool) -> np.ndarray:
        if self.depth_coordinates is None:
            return self._apply_global(values, linear=linear)
        flattened = values.reshape(-1, self.source_dim)
        raw_depth = np.asarray(depth, dtype=np.float64)
        if raw_depth.ndim == 0:
            depths = np.full(flattened.shape[0], float(raw_depth), dtype=np.float64)
        else:
            depths = raw_depth.reshape(-1)
            if len(depths) != flattened.shape[0]:
                raise ValueError("alignment depth count must match the leading state rows")
        result = np.empty((flattened.shape[0], self.target_dim), dtype=np.float64)
        unique_depths, inverse_indices = np.unique(depths, return_inverse=True)
        for u_idx, u_depth in enumerate(unique_depths):
            mask = (inverse_indices == u_idx)
            rows = flattened[mask]
            matrix, bias = self._depth_operator(float(u_depth))
            if self.source_projection is not None and self.target_projection is not None:
                chart = rows @ self.source_projection if linear else (rows - self.source_mean) @ self.source_projection
                output = chart @ matrix
                if not linear:
                    output = output + bias
                block_res = output @ self.target_projection.T
                if not linear:
                    block_res = block_res + self.target_mean
                result[mask] = block_res
            else:
                block_res = rows @ matrix
                if not linear:
                    block_res = block_res + bias
                result[mask] = block_res
        return result.reshape(values.shape[:-1] + (self.target_dim,))

    def _apply_global(self, values: np.ndarray, *, linear: bool) -> np.ndarray:
        if self.source_projection is not None and self.target_projection is not None:
            chart = values @ self.source_projection if linear else (values - self.source_mean) @ self.source_projection
            output = chart @ self.matrix
            if not linear:
                output = output + self.bias
            result = output @ self.target_projection.T
            if not linear:
                result = result + self.target_mean
            return result
        result = values @ self.matrix
        return result if linear else result + self.bias

    def apply(self, x: np.ndarray, *, depth: Any | None = None) -> np.ndarray:
        values = finite_array(x, name="alignment input")
        if values.shape[-1] != self.source_dim:
            raise ValueError("alignment input has the wrong last dimension")
        return self._apply_depth_conditioned(values, depth, linear=False) if depth is not None and self.depth_coordinates is not None else self._apply_global(values, linear=False)

    def linear_apply(self, x: np.ndarray, *, depth: Any | None = None) -> np.ndarray:
        """Apply only the tangent map, excluding affine centering/bias."""

        values = finite_array(x, name="alignment tangent input")
        if values.shape[-1] != self.source_dim:
            raise ValueError("alignment tangent input has the wrong last dimension")
        return self._apply_depth_conditioned(values, depth, linear=True) if depth is not None and self.depth_coordinates is not None else self._apply_global(values, linear=True)


@dataclass(frozen=True)
class FlowOperator:
    """Piecewise local affine/quadratic approximation on continuous depth."""

    coordinates: np.ndarray
    matrices: np.ndarray
    biases: np.ndarray
    sample_counts: np.ndarray
    residual_scales: np.ndarray
    spectral_norms: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    quadratic_terms: np.ndarray | None = None
    chart_projection: np.ndarray | None = None

    def __post_init__(self) -> None:
        coordinates = finite_array(self.coordinates, ndim=1, name="flow coordinates")
        matrices = finite_array(self.matrices, ndim=3, name="flow matrices")
        biases = finite_array(self.biases, ndim=2, name="flow biases")
        sample_counts = finite_array(self.sample_counts, ndim=1, name="sample_counts")
        residual_scales = finite_array(self.residual_scales, ndim=1, name="residual_scales")
        spectral_norms = finite_array(self.spectral_norms, ndim=1, name="spectral_norms")
        n = len(coordinates)
        if n == 0 or np.any(np.diff(coordinates) <= 0):
            raise ValueError("flow coordinates must be non-empty and strictly increasing")
        if matrices.shape[0] != n or matrices.shape[1] != matrices.shape[2]:
            raise ValueError("flow matrices must be [nodes, dim, dim]")
        if biases.shape != (n, matrices.shape[1]) or any(len(x) != n for x in (sample_counts, residual_scales, spectral_norms)):
            raise ValueError("flow arrays have inconsistent node counts")
        if any(np.any(values < 0) for values in (sample_counts, residual_scales, spectral_norms)):
            raise ValueError("flow support, residual, and spectral norms must be non-negative")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "biases", biases)
        object.__setattr__(self, "sample_counts", sample_counts)
        object.__setattr__(self, "residual_scales", residual_scales)
        object.__setattr__(self, "spectral_norms", spectral_norms)
        if self.quadratic_terms is not None:
            quadratic = finite_array(self.quadratic_terms, ndim=4, name="flow quadratic terms")
            if quadratic.shape != (n, matrices.shape[1], matrices.shape[1], matrices.shape[1]):
                raise ValueError("flow quadratic terms must be [nodes, output_dim, input_dim, input_dim]")
            object.__setattr__(self, "quadratic_terms", quadratic)
        if self.chart_projection is not None:
            projection = finite_array(self.chart_projection, ndim=2, name="flow chart projection")
            if projection.shape[1] != matrices.shape[1]:
                raise ValueError("flow chart projection columns must match the compact operator dimension")
            object.__setattr__(self, "chart_projection", projection)

    @property
    def state_dim(self) -> int:
        return int(self.chart_projection.shape[0] if self.chart_projection is not None else self.matrices.shape[1])

    def at(self, depth: float) -> tuple[np.ndarray, np.ndarray]:
        depth = float(depth)
        if not np.isfinite(depth):
            raise ValueError("depth must be finite")
        idx = int(np.searchsorted(self.coordinates, depth, side="right") - 1)
        idx = max(0, min(idx, len(self.coordinates) - 1))
        if idx == len(self.coordinates) - 1 or depth <= self.coordinates[0]:
            return self.matrices[idx].copy(), self.biases[idx].copy()
        t = (depth - self.coordinates[idx]) / (self.coordinates[idx + 1] - self.coordinates[idx])
        return (
            (1 - t) * self.matrices[idx] + t * self.matrices[idx + 1],
            (1 - t) * self.biases[idx] + t * self.biases[idx + 1],
        )

    def predict(self, state: np.ndarray, depth: float) -> np.ndarray:
        x = finite_array(state, name="flow state")
        if x.shape[-1] != self.state_dim:
            raise ValueError("state has the wrong dimension for the flow")
        compact_x = x if self.chart_projection is None else x @ self.chart_projection
        matrix, bias = self.at(depth)
        result = compact_x @ matrix + bias
        if self.quadratic_terms is not None:
            idx = int(np.searchsorted(self.coordinates, depth, side="right") - 1)
            idx = max(0, min(idx, len(self.coordinates) - 1))
            quadratic = self.quadratic_terms[idx]
            if idx < len(self.coordinates) - 1 and depth > self.coordinates[0]:
                t = (depth - self.coordinates[idx]) / (self.coordinates[idx + 1] - self.coordinates[idx])
                quadratic = (1.0 - t) * quadratic + t * self.quadratic_terms[idx + 1]
            result = result + np.einsum("...i,jik,...k->...j", compact_x, quadratic, compact_x)
        if self.chart_projection is not None:
            result = result @ self.chart_projection.T
        return finite_array(result, name="flow prediction")


@dataclass(frozen=True)
class DepthTransferReport:
    """Evidence attached to continuous teacher-to-student depth transport.

    For paired trace collections, ``matched_nodes`` and ``gap_count`` are
    totals across traces while ``matched_fraction`` and
    ``mean_gap_confidence`` are means. The retained correspondence fields are
    from the first paired trace and are labelled as such in ``notes``.
    """

    student_nodes: int
    teacher_nodes: int
    matched_nodes: int
    gap_count: int
    dropped_transitions: int
    matched_fraction: float
    mean_gap_confidence: float
    node_confidence: np.ndarray
    correspondences: tuple[tuple[int, int], ...] = ()
    gaps: tuple[GapInterval, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        confidence = finite_array(self.node_confidence, ndim=1, name="depth node confidence")
        if not (0.0 <= self.matched_fraction <= 1.0 and 0.0 <= self.mean_gap_confidence <= 1.0):
            raise ValueError("depth report fractions must be in [0, 1]")
        if self.student_nodes < 2 or self.teacher_nodes < 2 or self.matched_nodes < 0 or self.gap_count < 0 or self.dropped_transitions < 0:
            raise ValueError("depth report counts are invalid")
        object.__setattr__(self, "node_confidence", np.clip(confidence, 0.0, 1.0))


@dataclass(frozen=True)
class CapacityDiagnostics:
    """Measured student bottleneck evidence for a particular paired dataset."""

    student_state_rank: int
    teacher_state_rank: int
    student_tangent_rank: int
    teacher_velocity_rank: int
    state_subspace_coverage: float
    velocity_subspace_coverage: float
    residual_transport_error: float
    irreducible_mismatch: float
    map_condition_number: float
    bottleneck: bool
    confidence_penalty: float
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = (self.state_subspace_coverage, self.velocity_subspace_coverage, self.residual_transport_error, self.irreducible_mismatch, self.map_condition_number, self.confidence_penalty)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("capacity diagnostics contain non-finite values")
        if not all(0.0 <= value <= 1.0 for value in (self.state_subspace_coverage, self.velocity_subspace_coverage, self.confidence_penalty)):
            raise ValueError("capacity fractions must be in [0, 1]")


@dataclass(frozen=True)
class FlowFitResult:
    student: FlowOperator
    transported_teacher: FlowOperator
    correction_matrices: np.ndarray
    correction_biases: np.ndarray
    confidence: np.ndarray
    validation_error: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    depth_report: DepthTransferReport | None = None
    capacity_diagnostics: CapacityDiagnostics | None = None
    correction_quadratic: np.ndarray | None = None

    def __post_init__(self) -> None:
        mats = finite_array(self.correction_matrices, ndim=3, name="correction_matrices")
        bias = finite_array(self.correction_biases, ndim=2, name="correction_biases")
        conf = finite_array(self.confidence, ndim=1, name="confidence")
        if mats.shape != self.student.matrices.shape or bias.shape != self.student.biases.shape or len(conf) != mats.shape[0]:
            raise ValueError("flow correction arrays do not match student flow")
        if np.any(conf < 0) or np.any(conf > 1):
            raise ValueError("flow confidence must be in [0, 1]")
        if not np.isfinite(self.validation_error) or self.validation_error < 0:
            raise ValueError("flow validation error must be finite and non-negative")
        object.__setattr__(self, "correction_matrices", mats)
        object.__setattr__(self, "correction_biases", bias)
        object.__setattr__(self, "confidence", conf)
        if self.correction_quadratic is not None:
            quadratic = finite_array(self.correction_quadratic, ndim=4, name="correction quadratic terms")
            expected = (mats.shape[0], mats.shape[1], mats.shape[1], mats.shape[1])
            if quadratic.shape != expected:
                raise ValueError("correction quadratic terms do not match student flow")
            object.__setattr__(self, "correction_quadratic", quadratic)

    def corrected_operator(self) -> FlowOperator:
        """Return the student-chart candidate before trust-region application."""

        quadratic = None
        if self.student.quadratic_terms is not None:
            correction = self.correction_quadratic if self.correction_quadratic is not None else np.zeros_like(self.student.quadratic_terms)
            quadratic = self.student.quadratic_terms + correction
        return FlowOperator(self.student.coordinates, self.student.matrices + self.correction_matrices, self.student.biases + self.correction_biases, self.student.sample_counts, self.student.residual_scales, self.student.spectral_norms, {**self.student.metadata, "corrected_from": "student_flow_fit"}, quadratic, self.student.chart_projection)


@dataclass(frozen=True)
class TensorSchemaEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class TensorTransitionMapping:
    """Explicit relation between a flow transition and an existing tensor."""

    transition_index: int
    tensor_name: str
    orientation: str = "row_right"
    role: str = "dense_square"

    def __post_init__(self) -> None:
        if self.transition_index < 0 or not self.tensor_name:
            raise ValueError("tensor transition mapping has invalid index or name")
        if self.orientation not in {"row_right", "column_left"}:
            raise ValueError("orientation must be 'row_right' or 'column_left'")
        if self.role != "dense_square":
            raise ValueError("only dense_square mappings are currently supported; fused/gated/sharded tensors need a family connector")


@dataclass(frozen=True)
class SurgeryPlan:
    updates: Mapping[str, np.ndarray]
    rollback_layers: tuple[int, ...]
    rollback_reasons: Mapping[int, str]
    confidence: np.ndarray
    schema: tuple[TensorSchemaEntry, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    skipped_tensors: Mapping[str, str] = field(default_factory=dict)
    applied_tensors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    geometric_alignment: float
    baseline_functional_error: float
    intervened_functional_error: float
    causal_effect: float
    stability_score: float
    collapse_score: float
    holdout_count: int
    passed: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
