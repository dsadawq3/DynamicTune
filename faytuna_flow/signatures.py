"""Fidelity features for comparing teacher and student dynamics.

The objects in this module are operational signatures.  They summarize paths
and local differential observations; they do not identify meaning or prove a
semantic correspondence.  Scalar invariants are used where charts differ,
while tangent bases are compared only after the teacher has been transported
into the student chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .types import AlignmentResult, TrajectoryTrace, finite_array


def _stable_differences(values: np.ndarray) -> np.ndarray:
    scale = max(1.0, float(np.max(np.abs(values))))
    with np.errstate(over="raise", invalid="raise"):
        differences = (values[1:] / scale - values[:-1] / scale) * scale
    return finite_array(differences, ndim=2, name="path differences")


def _stable_norm(values: np.ndarray, axis: int = -1) -> np.ndarray:
    scale = max(1.0, float(np.max(np.abs(values))))
    result = np.linalg.norm(values / scale, axis=axis) * scale
    return finite_array(result, name="stable norm")


@dataclass(frozen=True)
class PathSignature:
    scales: tuple[int, ...]
    values: np.ndarray
    endpoint_displacement: float
    total_arc_length: float
    turning_total: float
    area_backend: str = "exact_dense"
    area_sketch_rank: int | None = None

    def __post_init__(self) -> None:
        values = finite_array(self.values, ndim=2, name="path signature")
        if values.shape[0] != len(self.scales) or values.shape[1] != 8:
            raise ValueError("path signature must have one eight-feature row per scale")
        if any(int(scale) < 1 for scale in self.scales):
            raise ValueError("path signature scales must be positive")
        if self.area_backend not in {"exact_dense", "randomized_frobenius_sketch"}:
            raise ValueError("path signature area_backend is unsupported")
        if self.area_backend == "randomized_frobenius_sketch" and (self.area_sketch_rank is None or int(self.area_sketch_rank) < 1):
            raise ValueError("randomized path signature requires a positive sketch rank")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class DifferentialSignature:
    """Per-transition invariant and differential geometry features."""

    values: np.ndarray
    available: np.ndarray
    feature_names: tuple[str, ...] = (
        "field_log_norm",
        "jacobian_log_frobenius",
        "jacobian_log_singular_mean",
        "jacobian_log_singular_spread",
        "jacobian_log_condition",
        "hessian_log_norm",
        "curvature_log",
        "uncertainty_log",
    )

    def __post_init__(self) -> None:
        values = finite_array(self.values, ndim=2, name="differential signature")
        available = np.asarray(self.available, dtype=bool)
        if values.shape[0] != len(available) or values.shape[1] != len(self.feature_names):
            raise ValueError("differential signature shapes are inconsistent")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "available", available)


@dataclass(frozen=True)
class TangentTransport:
    """Local teacher-tangent to student-tangent transport evidence."""

    coordinates: np.ndarray
    overlap: np.ndarray
    student_rank: np.ndarray
    teacher_rank: np.ndarray
    transport_matrices: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        coords = finite_array(self.coordinates, ndim=1, name="tangent coordinates")
        overlap = finite_array(self.overlap, ndim=2, name="tangent overlap")
        student_rank = finite_array(self.student_rank, ndim=1, name="student tangent rank")
        teacher_rank = finite_array(self.teacher_rank, ndim=1, name="teacher tangent rank")
        matrices = finite_array(self.transport_matrices, ndim=3, name="tangent transport")
        confidence = finite_array(self.confidence, ndim=1, name="tangent confidence")
        if matrices.shape[0] != len(coords) or overlap.shape[0] != len(coords) or len(student_rank) != len(coords) or len(teacher_rank) != len(coords) or len(confidence) != len(coords):
            raise ValueError("tangent transport node counts are inconsistent")
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "overlap", overlap)
        object.__setattr__(self, "student_rank", student_rank)
        object.__setattr__(self, "teacher_rank", teacher_rank)
        object.__setattr__(self, "transport_matrices", matrices)
        object.__setattr__(self, "confidence", np.clip(confidence, 0.0, 1.0))


def _ordered_area_frobenius_sketch(normalized_deltas: np.ndarray, start: int, stop: int, projection: np.ndarray, *, projected_deltas: np.ndarray | None = None) -> float:
    """Estimate ordered area Frobenius norm without materializing a d×d matrix."""

    if stop - start <= 1:
        return 0.0
    sub_proj = (normalized_deltas[start:stop] @ projection) if projected_deltas is None else projected_deltas[start:stop]
    sub_deltas = normalized_deltas[start:stop]
    suffix_sums = np.cumsum(sub_proj[::-1], axis=0)[::-1][1:]
    area_projection = sub_deltas[:-1].T @ suffix_sums
    # Rademacher columns satisfy E ||A r||² = ||A||_F². The result is a
    # compact deterministic estimator for the same ordered-area aggregate.
    return float(np.sqrt(np.mean(np.sum(np.square(area_projection), axis=0))))


def multi_scale_path_signature(states: np.ndarray, coordinates: np.ndarray, *, scales: Sequence[int] = (1, 2, 4), eps: float = 1e-8, max_dense_area_dim: int = 512, area_sketch_rank: int = 32, area_seed: int = 0) -> PathSignature:
    """Compute path-integral surrogates at several transition scales.

    The eighth-order-free feature vector deliberately contains both additive
    path quantities and second-order ordered area magnitude.  It is compact,
    chart-aware, and useful for weighting transport without pretending that a
    raw hidden vector has the same coordinates in two models.
    """

    values = finite_array(states, ndim=2, name="signature states")
    coords = finite_array(coordinates, ndim=1, name="signature coordinates")
    if len(values) != len(coords) or len(values) < 2 or np.any(np.diff(coords) <= 0):
        raise ValueError("signature states and coordinates must be aligned and increasing")
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("signature eps must be positive and finite")
    if max_dense_area_dim < 1 or area_sketch_rank < 1:
        raise ValueError("path signature area limits must be positive")
    unique_scales = tuple(dict.fromkeys(int(scale) for scale in scales))
    if not unique_scales or any(scale < 1 for scale in unique_scales):
        raise ValueError("signature scales must contain positive integers")
    deltas = _stable_differences(values)
    ds = np.diff(coords)
    velocities = deltas / ds[:, None]
    lengths = _stable_norm(deltas)
    speed = _stable_norm(velocities)
    turns = np.zeros(len(deltas), dtype=np.float64)
    for index in range(1, len(deltas)):
        denominator = max(float(lengths[index - 1] * lengths[index]), eps)
        turns[index] = np.arccos(np.clip(float(np.dot(deltas[index - 1], deltas[index]) / denominator), -1.0, 1.0))
    rows = []
    delta_scale = max(1.0, float(np.max(np.abs(deltas))))
    normalized_deltas = deltas / delta_scale
    use_dense_area = values.shape[1] <= int(max_dense_area_dim)
    area_projection = None
    all_projected = None
    if not use_dense_area:
        area_rng = np.random.default_rng(int(area_seed))
        area_projection = area_rng.choice(np.asarray([-1.0, 1.0]), size=(values.shape[1], int(area_sketch_rank)))
        all_projected = normalized_deltas @ area_projection
    for window in unique_scales:
        block_features = []
        for start in range(0, len(deltas), window):
            stop = min(len(deltas), start + window)
            block_delta = np.sum(deltas[start:stop], axis=0)
            block_length = float(np.sum(lengths[start:stop]))
            block_speed = float(np.mean(speed[start:stop]))
            block_turn = float(np.mean(turns[start:stop]))
            block_acceleration = float(np.mean(_stable_norm(np.diff(velocities[start:stop], axis=0))) if stop - start > 1 else 0.0)
            if use_dense_area:
                block_len = stop - start
                if block_len > 1:
                    sub_deltas = normalized_deltas[start:stop]
                    suffix_sums = np.cumsum(sub_deltas[::-1], axis=0)[::-1][1:]
                    ordered_area = sub_deltas[:-1].T @ suffix_sums
                    normalized_area_norm = float(np.linalg.norm(ordered_area))
                else:
                    normalized_area_norm = 0.0
            else:
                assert area_projection is not None
                normalized_area_norm = _ordered_area_frobenius_sketch(normalized_deltas, start, stop, area_projection, projected_deltas=all_projected)
            area_log = float(np.logaddexp(0.0, -np.inf if normalized_area_norm == 0.0 else np.log(normalized_area_norm) + 2.0 * np.log(delta_scale)))
            block_roughness = float(np.sum(_stable_norm(np.diff(velocities[start:stop], axis=0))) if stop - start > 1 else 0.0)
            block_features.append((float(_stable_norm(block_delta[None, :])[0]), block_length, block_speed, block_acceleration, block_turn, area_log, block_roughness))
        block_array = np.asarray(block_features, dtype=np.float64)
        aggregate = np.mean(block_array, axis=0)
        rows.append(np.asarray([
            np.log1p(max(0.0, aggregate[0])),
            np.log1p(max(0.0, aggregate[1])),
            np.log1p(max(0.0, aggregate[2])),
            np.log1p(max(0.0, aggregate[3])),
            aggregate[4] / np.pi,
            max(0.0, aggregate[5]),
            np.log1p(max(0.0, aggregate[6])),
            float(len(block_features)),
        ]))
    endpoint = float(_stable_norm(_stable_differences(values[[0, -1]]))[0])
    return PathSignature(unique_scales, np.asarray(rows), endpoint, float(np.sum(lengths)), float(np.sum(turns)), "exact_dense" if use_dense_area else "randomized_frobenius_sketch", None if use_dense_area else int(area_sketch_rank))


def differential_signature(trace: TrajectoryTrace, *, eps: float = 1e-8) -> DifferentialSignature:
    """Extract log-scaled Jacobian/Hessian/curvature invariants from a trace."""

    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("differential signature eps must be positive and finite")
    rows = []
    availability = []
    for transition in trace.transitions:
        field = max(float(np.linalg.norm(transition.vector_field)), eps)
        jacobian = transition.jacobian
        singular_values = transition.singular_values
        if singular_values is None and jacobian is not None:
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
        if singular_values is None:
            jac_log_fro = jac_mean = jac_spread = jac_condition = 0.0
            available = False
        else:
            singular = np.maximum(finite_array(singular_values, ndim=1, name="signature singular values"), eps)
            logs = np.log(singular)
            jac_log_fro = float(np.log(max(float(np.linalg.norm(singular)), eps)))
            jac_mean = float(np.mean(logs))
            jac_spread = float(np.std(logs))
            jac_condition = float(np.max(logs) - np.min(logs))
            available = True
        hessian_norm = 0.0 if transition.hessian_sketch is None else float(np.linalg.norm(transition.hessian_sketch))
        uncertainty = float(np.mean(list(transition.uncertainty.values()))) if transition.uncertainty else 0.0
        rows.append([
            np.log(field), jac_log_fro, jac_mean, jac_spread, jac_condition,
            np.log1p(max(0.0, hessian_norm)), np.log1p(abs(float(transition.curvature))), np.log1p(max(0.0, uncertainty)),
        ])
        availability.append(available)
    return DifferentialSignature(np.asarray(rows, dtype=np.float64), np.asarray(availability, dtype=bool))


def _resample_rows(values: np.ndarray, source_coordinates: np.ndarray, query_coordinates: np.ndarray) -> np.ndarray:
    rows = finite_array(values, ndim=2, name="signature rows")
    source = finite_array(source_coordinates, ndim=1, name="signature source coordinates")
    query = finite_array(query_coordinates, ndim=1, name="signature query coordinates")
    if len(rows) != len(source) or len(source) == 0 or np.any(np.diff(source) <= 0):
        raise ValueError("signature resampling coordinates are invalid")
    return np.column_stack([np.interp(query, source, rows[:, column]) for column in range(rows.shape[1])])


def differential_compatibility(student: TrajectoryTrace, teacher: TrajectoryTrace, mapped_teacher_depth: np.ndarray, *, uncertainty_scale: float = 1.0) -> tuple[np.ndarray, float]:
    """Return per-student-transition compatibility weights and a path score."""

    if len(mapped_teacher_depth) != teacher.layer_count or np.any(np.diff(mapped_teacher_depth) < 0):
        raise ValueError("mapped teacher depth must be monotone and cover all teacher nodes")
    student_signature = differential_signature(student)
    teacher_signature = differential_signature(teacher)
    student_depth = student.depth_coordinates[:-1]
    teacher_depth = np.asarray(mapped_teacher_depth[:-1], dtype=np.float64)
    # Collapse repeated teacher coordinates caused by an explicitly reported
    # gap before interpolation; repeated coordinates are not invented layers.
    unique_depth, unique_index = np.unique(teacher_depth, return_index=True)
    teacher_values = teacher_signature.values[unique_index]
    sampled_teacher = _resample_rows(teacher_values, unique_depth, student_depth)
    joint_scale = np.maximum(1.0, np.median(np.abs(np.vstack([student_signature.values, sampled_teacher])), axis=0))
    discrepancy = np.sqrt(np.mean(((student_signature.values - sampled_teacher) / joint_scale) ** 2, axis=1))
    # Differential disagreement is evidence about transfer confidence, not a
    # reason to erase the teacher target.  Keep a bounded half-weight floor;
    # otherwise a chart-dependent scale mismatch suppresses exactly the
    # dynamics the transfer is meant to test.
    weights = 0.5 + 0.5 * np.exp(-np.clip(discrepancy * max(0.0, uncertainty_scale), 0.0, 40.0))
    weights *= np.where(student_signature.available, 1.0, 0.85)
    path_student = multi_scale_path_signature(student.hidden_states, student.depth_coordinates)
    path_teacher = multi_scale_path_signature(teacher.hidden_states, teacher.depth_coordinates)
    scale = np.maximum(1.0, np.median(np.abs(np.vstack([path_student.values, path_teacher.values])), axis=0))
    path_discrepancy = float(np.sqrt(np.mean(((path_student.values - path_teacher.values) / scale) ** 2)))
    path_score = 0.5 + 0.5 * float(np.exp(-min(path_discrepancy, 40.0)))
    return np.clip(weights, 0.0, 1.0), path_score


def _local_basis(samples: np.ndarray, *, tolerance: float = 1e-8) -> np.ndarray:
    centered = samples - samples.mean(axis=0, keepdims=True)
    if centered.size == 0:
        return np.zeros((samples.shape[1], 0))
    _, singular, right_transpose = np.linalg.svd(centered, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0:
        return np.zeros((samples.shape[1], 0))
    keep = singular > tolerance * max(1.0, singular[0])
    return right_transpose[keep].T


def fit_tangent_transport(student_traces: Sequence[TrajectoryTrace], teacher_traces: Sequence[TrajectoryTrace], alignment: AlignmentResult, *, rank: int | None = None) -> TangentTransport:
    """Estimate local tangent overlap after teacher→student chart transport."""

    if len(student_traces) != len(teacher_traces) or not student_traces:
        raise ValueError("paired traces are required for tangent transport")
    student_dim = student_traces[0].state_dim
    teacher_dim = teacher_traces[0].state_dim
    if alignment.source_dim != teacher_dim or alignment.target_dim != student_dim:
        raise ValueError("teacher-to-student alignment dimensions do not match tangent transport")
    node_count = student_traces[0].layer_count
    if any(trace.layer_count != node_count for trace in student_traces):
        raise ValueError("tangent transport currently requires a common student node grid")
    coordinates = student_traces[0].depth_coordinates
    overlaps = []
    student_ranks = []
    teacher_ranks = []
    matrices = []
    confidences = []
    teacher_samples = [
        alignment.apply(trace.hidden_states, depth=trace.depth_coordinates)
        for trace in teacher_traces
    ]
    resampled_teacher_grids = [
        _resample_rows(values, trace.depth_coordinates, coordinates)
        for values, trace in zip(teacher_samples, teacher_traces)
    ]
    for node in range(node_count):
        student_samples = np.asarray([trace.hidden_states[node] for trace in student_traces])
        teacher_grid = np.asarray([grid[node] for grid in resampled_teacher_grids])
        student_basis = _local_basis(student_samples)
        teacher_basis = _local_basis(teacher_grid)
        if rank is not None:
            student_basis = student_basis[:, : min(rank, student_basis.shape[1])]
            teacher_basis = teacher_basis[:, : min(rank, teacher_basis.shape[1])]
        if student_basis.shape[1] and teacher_basis.shape[1]:
            left, singular, right_transpose = np.linalg.svd(teacher_basis.T @ student_basis, full_matrices=False)
            overlap_row = np.zeros(max(student_dim, teacher_dim), dtype=np.float64)
            overlap_row[: len(singular)] = singular
            # Procrustes tangent transport is defined in the shared student
            # chart and is rectangular-safe through the two orthonormal bases.
            transport = student_basis @ right_transpose.T @ left.T @ teacher_basis.T
            confidence = float(np.mean(singular))
        else:
            overlap_row = np.zeros(max(student_dim, teacher_dim), dtype=np.float64)
            transport = np.zeros((student_dim, student_dim), dtype=np.float64)
            confidence = 0.0
        overlaps.append(overlap_row)
        student_ranks.append(student_basis.shape[1])
        teacher_ranks.append(teacher_basis.shape[1])
        matrices.append(transport)
        confidences.append(confidence)
    return TangentTransport(coordinates.copy(), np.asarray(overlaps), np.asarray(student_ranks), np.asarray(teacher_ranks), np.asarray(matrices), np.asarray(confidences))
