"""Cross-model geometry: whitening, transport maps, OT, and relational tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .types import AlignmentResult, finite_array, stable_l2
from .scalable import DEFAULT_MAX_DENSE_FEATURES, DEFAULT_SCALABLE_RANK, data_aware_chart_projection


DEFAULT_MAX_PAIRWISE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_DIRECT_JACOBIAN_ROW_ELEMENTS = 200_000


@dataclass(frozen=True)
class Whitening:
    mean: np.ndarray
    components: np.ndarray
    inverse_components: np.ndarray
    rank: int
    eigenvalues: np.ndarray

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (finite_array(x, name="whitening input") - self.mean) @ self.components

    def decode(self, z: np.ndarray) -> np.ndarray:
        return finite_array(z, name="unwhitening input") @ self.inverse_components + self.mean


def fit_whitening(x: np.ndarray, *, rank: int | None = None, eps: float = 1e-8) -> Whitening:
    values = finite_array(x, ndim=2, name="whitening samples")
    if values.shape[0] < 2:
        raise ValueError("at least two samples are needed for whitening")
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("whitening eps must be positive and finite")
    mean = values.mean(axis=0)
    centered = values - mean
    # Compute the covariance in a scaled chart. The scale is cancelled in
    # ``components``/``inverse_components`` below, so encode/decode retain
    # their raw-coordinate contract without overflowing on large finite data.
    scale = max(1.0, float(np.max(np.abs(centered))))
    scaled = centered / scale
    covariance = scaled.T @ scaled / max(1, values.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    if rank is None:
        rank = int(np.sum(eigenvalues > eps * max(1.0, eigenvalues[0])))
        rank = max(1, rank)
    rank = min(int(rank), values.shape[1])
    vals = np.maximum(eigenvalues[:rank], eps)
    basis = eigenvectors[:, :rank]
    components = basis / (scale * np.sqrt(vals))[None, :]
    inverse_components = np.diag(scale * np.sqrt(vals)) @ basis.T
    return Whitening(mean, components, inverse_components, rank, vals)


def _distance_matrix_input(value: np.ndarray, *, name: str) -> np.ndarray:
    """Validate a 2-D real chart while preserving float32 memory savings."""

    values = np.asarray(value)
    if values.ndim != 2:
        raise ValueError(f"{name} must have ndim=2, got {values.ndim}")
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    return values


def _pairwise_squared_gram(
    source: np.ndarray,
    target: np.ndarray,
    *,
    scale: float,
    max_pairwise_bytes: int,
) -> np.ndarray:
    """Compute scaled squared distances without constructing ``N×M×D``."""

    if source.shape[1] != target.shape[1]:
        raise ValueError("distance charts must have equal feature dimensions")
    if source.shape[0] < 1 or target.shape[0] < 1:
        raise ValueError("distance charts must be non-empty")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("distance scale must be positive and finite")
    if int(max_pairwise_bytes) < 1:
        raise ValueError("max_pairwise_bytes must be positive")
    # Dividing before matmul keeps finite large inputs in range. The only
    # feature-sized temporaries are these two 2-D arrays; there is no
    # broadcasted difference tensor.
    source_scaled = source / scale
    target_scaled = source_scaled if target is source else target / scale
    source_norm = np.einsum("ij,ij->i", source_scaled, source_scaled)
    target_norm = np.einsum("ij,ij->i", target_scaled, target_scaled)
    result = np.empty((source.shape[0], target.shape[0]), dtype=np.float64)
    rows_per_block = max(1, min(source.shape[0], int(max_pairwise_bytes) // max(8 * target.shape[0], 8)))
    negative_tolerance = 1e-10 * max(1.0, float(np.max(source_norm)), float(np.max(target_norm)))
    for start in range(0, source.shape[0], rows_per_block):
        stop = min(source.shape[0], start + rows_per_block)
        # ||x-y||² = ||x||² + ||y||² - 2<x,y>. The output is N×M, which is
        # intentional and bounded; the dangerous feature broadcast is absent.
        gram = source_scaled[start:stop] @ target_scaled.T
        squared = source_norm[start:stop, None] + target_norm[None, :] - 2.0 * gram
        if not np.all(np.isfinite(squared)):
            raise FloatingPointError("Gram distance computation produced non-finite values")
        if np.any(squared < -negative_tolerance):
            raise FloatingPointError("Gram distance computation lost numerical accuracy; rescale the supplied chart")
        squared[squared < 0.0] = 0.0
        result[start:stop] = squared
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("pairwise squared distance is non-finite")
    return result


def pairwise_distances(x: np.ndarray, *, max_pairwise_bytes: int = DEFAULT_MAX_PAIRWISE_BYTES) -> np.ndarray:
    values = _distance_matrix_input(x, name="distance samples")
    scale = max(1.0, float(np.max(np.abs(values))))
    squared = _pairwise_squared_gram(values, values, scale=scale, max_pairwise_bytes=max_pairwise_bytes)
    distances = np.sqrt(squared) * scale
    if not np.all(np.isfinite(distances)):
        raise FloatingPointError("pairwise distance overflowed; rescale the supplied chart")
    return distances


def relational_error(source: np.ndarray, target: np.ndarray, *, max_pairwise_bytes: int = DEFAULT_MAX_PAIRWISE_BYTES) -> float:
    a = pairwise_distances(source, max_pairwise_bytes=max_pairwise_bytes)
    b = pairwise_distances(target, max_pairwise_bytes=max_pairwise_bytes)
    scale = max(float(np.linalg.norm(b)), 1e-12)
    return float(np.linalg.norm(a - b) / scale)


def orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = finite_array(source, ndim=2, name="procrustes source")
    y = finite_array(target, ndim=2, name="procrustes target")
    if x.shape != y.shape:
        raise ValueError("orthogonal Procrustes requires equal reduced dimensions")
    cross = x.T @ y
    left, _, right_transpose = np.linalg.svd(cross, full_matrices=False)
    return left @ right_transpose


def sinkhorn_plan(
    source: np.ndarray,
    target: np.ndarray,
    *,
    regularization: float = 0.05,
    max_iter: int = 500,
    tolerance: float = 1e-8,
    max_pairwise_bytes: int = 256 * 1024 * 1024,
) -> tuple[np.ndarray, float]:
    """Uniform entropic OT plan and marginal error for diagnostic transport."""

    x = _distance_matrix_input(source, name="OT source")
    y = _distance_matrix_input(target, name="OT target")
    if x.shape[0] < 1 or y.shape[0] < 1:
        raise ValueError("OT source and target must be non-empty")
    if regularization <= 0 or not np.isfinite(regularization) or max_iter < 1 or tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("OT regularization, iteration count, and tolerance must be positive and finite")
    coordinate_scale = max(1.0, float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    cost = _pairwise_squared_gram(x, y, scale=coordinate_scale, max_pairwise_bytes=max_pairwise_bytes)
    positive_cost = cost[cost > 0.0]
    cost = cost / max(float(np.median(positive_cost)) if positive_cost.size else 1.0, 1e-12)
    log_kernel = -np.clip(cost / float(regularization), 0.0, 700.0)
    if not np.all(np.isfinite(log_kernel)):
        raise FloatingPointError("Sinkhorn log-kernel is non-finite")
    a = np.full(x.shape[0], 1.0 / x.shape[0])
    b = np.full(y.shape[0], 1.0 / y.shape[0])
    log_a = np.log(a)
    log_b = np.log(b)
    log_u = np.zeros_like(a)
    log_v = np.zeros_like(b)

    def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
        maximum = np.max(values, axis=axis, keepdims=True)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            shifted = np.exp(values - maximum)
            summed = np.sum(shifted, axis=axis, keepdims=True)
            result = (maximum + np.log(summed)).squeeze(axis)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Sinkhorn log-sum-exp is non-finite")
        return result

    for _ in range(max_iter):
        old_log_u = log_u.copy()
        old_log_v = log_v.copy()
        log_u = log_a - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_kernel.T + log_u[None, :], axis=1)
        if not np.all(np.isfinite(log_u)) or not np.all(np.isfinite(log_v)):
            raise FloatingPointError("Sinkhorn dual potentials became non-finite")
        if max(float(np.max(np.abs(log_u - old_log_u))), float(np.max(np.abs(log_v - old_log_v)))) < tolerance:
            break
    log_plan = np.clip(log_u[:, None] + log_kernel + log_v[None, :], -745.0, 0.0)
    plan = np.exp(log_plan)
    if not np.all(np.isfinite(plan)):
        raise FloatingPointError("Sinkhorn produced non-finite transport mass")
    marginal_error = float(max(np.linalg.norm(plan.sum(axis=1) - a, 1), np.linalg.norm(plan.sum(axis=0) - b, 1)))
    return plan, marginal_error


def fit_low_rank_map(source: np.ndarray, target: np.ndarray, *, rank: int, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    x = finite_array(source, ndim=2, name="low-rank source")
    y = finite_array(target, ndim=2, name="low-rank target")
    if x.shape[0] != y.shape[0] or rank < 1 or ridge < 0 or not np.isfinite(ridge):
        raise ValueError("low-rank samples or rank are invalid")
    x_scale = max(1.0, float(np.max(np.abs(x))))
    y_scale = max(1.0, float(np.max(np.abs(y))))
    xs = x / x_scale
    ys = y / y_scale
    x_aug = np.column_stack([xs, np.ones(x.shape[0])])
    dual_solve = x_aug.shape[0] < x_aug.shape[1] and ridge > 0.0
    if dual_solve:
        # Exact ridge dual identity.  With real GPT-2 traces the number of
        # probes is tiny compared with the chart dimension, so solving the
        # d-by-d primal system needlessly dominates alignment.
        dual_gram = x_aug @ x_aug.T + ridge * np.eye(x_aug.shape[0])
        dual_gram = (dual_gram + dual_gram.T) / 2.0
        try:
            dual_solution = np.linalg.solve(dual_gram, ys)
        except np.linalg.LinAlgError:
            dual_solution = np.linalg.lstsq(dual_gram, ys, rcond=1e-10)[0]
        full = x_aug.T @ dual_solution
    else:
        gram = x_aug.T @ x_aug + ridge * np.eye(x_aug.shape[1])
        gram = (gram + gram.T) / 2.0
        try:
            full = np.linalg.solve(gram, x_aug.T @ ys)
        except np.linalg.LinAlgError:
            full = np.linalg.lstsq(x_aug, ys, rcond=1e-10)[0]
    matrix = (y_scale / x_scale) * full[:-1]
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("low-rank map overflowed; rescale the supplied charts")
    if dual_solve:
        # ``matrix`` is a product of two tall/thin factors.  Its non-zero
        # singular spectrum is the spectrum of the small core after QR, so
        # this computes the same truncated SVD without decomposing a
        # 1536-by-3200 dense matrix.
        left_factor = xs.T
        right_factor = (y_scale / x_scale) * dual_solution
        left_basis, left_triangular = np.linalg.qr(left_factor, mode="reduced")
        right_basis, right_triangular = np.linalg.qr(right_factor.T, mode="reduced")
        core = left_triangular @ right_triangular.T
        core_left, singular, core_right_transpose = np.linalg.svd(core, full_matrices=False)
        k = min(rank, len(singular))
        low_rank = (left_basis @ core_left[:, :k] * singular[:k]) @ (right_basis @ core_right_transpose[:k].T).T
    else:
        left, singular, right_transpose = np.linalg.svd(matrix, full_matrices=False)
        k = min(rank, len(singular))
        low_rank = (left[:, :k] * singular[:k]) @ right_transpose[:k]
    bias = y.mean(axis=0) - x.mean(axis=0) @ low_rank
    if not np.all(np.isfinite(bias)):
        raise FloatingPointError("low-rank map bias overflowed; rescale the supplied charts")
    return low_rank, bias


def _fit_compressed_alignment(x: np.ndarray, y: np.ndarray, *, kind: str, rank: int | None, ot_regularization: float, source_role: str | None, target_role: str | None, max_dense_features: int, scalable_rank: int, scalable_seed: int, max_pairwise_bytes: int) -> AlignmentResult:
    """Fit a map between explicit data charts without allocating source_dim*target_dim."""

    if kind not in {"whitened_orthogonal", "riemannian", "affine", "low_rank", "ot_barycentric"}:
        raise ValueError(f"unknown alignment kind: {kind}")
    chart_rank = min(int(rank or scalable_rank), int(scalable_rank), int(np.sqrt(max_dense_features)), x.shape[0] - 1, x.shape[1], y.shape[1])
    if chart_rank < 1:
        raise ValueError("compressed alignment has no rank after finite memory guard")
    source_projection = data_aware_chart_projection(x, chart_rank, scalable_seed)
    target_projection = data_aware_chart_projection(y, chart_rank, scalable_seed + 1)
    x_chart = (x - x.mean(axis=0)) @ source_projection
    y_chart = (y - y.mean(axis=0)) @ target_projection
    if kind in {"whitened_orthogonal", "riemannian"}:
        wx = fit_whitening(x_chart, rank=chart_rank)
        wy = fit_whitening(y_chart, rank=chart_rank)
        rotation = orthogonal_procrustes(wx.encode(x_chart), wy.encode(y_chart))
        matrix = wx.components @ rotation @ wy.inverse_components
        bias = wy.mean - wx.mean @ matrix
        label = kind
        ot_mass_error = 0.0
    elif kind == "affine":
        full = np.linalg.lstsq(np.column_stack([x_chart, np.ones(x.shape[0])]), y_chart, rcond=1e-10)[0]
        matrix, bias, label, ot_mass_error = full[:-1], full[-1], kind, 0.0
    elif kind == "low_rank":
        matrix, bias = fit_low_rank_map(x_chart, y_chart, rank=chart_rank)
        label, ot_mass_error = kind, 0.0
    else:
        plan, ot_mass_error = sinkhorn_plan(x_chart, y_chart, regularization=ot_regularization)
        pseudo_target = (plan / np.maximum(plan.sum(axis=1, keepdims=True), 1e-300)) @ y_chart
        matrix, bias = fit_low_rank_map(x_chart, pseudo_target, rank=chart_rank)
        label = kind
    mapped = (x - x.mean(axis=0)) @ source_projection @ matrix @ target_projection.T + bias @ target_projection.T + y.mean(axis=0)
    rel = relational_error(mapped, y, max_pairwise_bytes=max_pairwise_bytes)
    reverse_matrix, reverse_bias = fit_low_rank_map(y_chart, x_chart, rank=chart_rank)
    reverse_chart = (y - y.mean(axis=0)) @ target_projection
    cycle_chart = (reverse_chart @ reverse_matrix + reverse_bias) - x_chart
    # A map fitted from fewer samples than both chart dimensions has an
    # exactly non-trivial null space.  Report its condition as infinite
    # directly instead of decomposing a large dense matrix just to discover
    # the zero singular values.  This is common for real probe pilots.
    if x.shape[0] < min(x.shape[1], y.shape[1]):
        condition = float("inf")
    else:
        singular = np.linalg.svd(matrix, compute_uv=False)
        condition = float(np.inf if singular.size == 0 or singular[-1] < 1e-12 else singular[0] / singular[-1])
    metadata: dict[str, Any] = {
        "representation": "randomized_latent_compression",
        "direction": None if source_role is None else f"{source_role}_to_{target_role}",
        "source_role": source_role,
        "target_role": target_role,
        "source_projection_shape": list(source_projection.shape),
        "target_projection_shape": list(target_projection.shape),
        "compressed_rank": chart_rank,
        "max_dense_features": int(max_dense_features),
        "dense_map_feature_count": int(x.shape[1] * y.shape[1]),
        "dense_map_estimated_bytes": int(x.shape[1] * y.shape[1] * 8),
        "dense_map_skipped_reason": "source_dim*target_dim exceeds max_dense_features",
        "discarded_source_rank": max(0, x.shape[1] - chart_rank),
        "discarded_target_rank": max(0, y.shape[1] - chart_rank),
        "pairwise_distance_backend": "scaled_gram_identity_blockwise",
        "pairwise_distance_feature_broadcast_allocated": False,
        "pairwise_distance_max_pairwise_bytes": int(max_pairwise_bytes),
        "pairwise_distance_output_shape": [int(mapped.shape[0]), int(y.shape[0])],
    }
    return AlignmentResult(label, x.mean(axis=0), y.mean(axis=0), matrix, bias, int(np.linalg.matrix_rank(x - x.mean(axis=0))), int(np.linalg.matrix_rank(y - y.mean(axis=0))), float(np.sqrt(np.mean(np.sum((mapped - y) ** 2, axis=1)))), rel, float(np.sqrt(np.mean(np.sum(cycle_chart ** 2, axis=1)))), condition, float(ot_mass_error), metadata, source_projection, target_projection)


def fit_alignment(source: np.ndarray, target: np.ndarray, *, kind: str = "whitened_orthogonal", rank: int | None = None, ot_regularization: float = 0.05, source_role: str | None = None, target_role: str | None = None, max_dense_features: int = DEFAULT_MAX_DENSE_FEATURES, scalable_rank: int = DEFAULT_SCALABLE_RANK, scalable_seed: int = 0, max_pairwise_bytes: int = DEFAULT_MAX_PAIRWISE_BYTES) -> AlignmentResult:
    """Fit a map x_s -> x_t while reporting geometry and cycle diagnostics."""

    x = finite_array(source, ndim=2, name="alignment source")
    y = finite_array(target, ndim=2, name="alignment target")
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("paired alignment requires equal sample counts >= 2")
    if max_dense_features < 1 or scalable_rank < 1 or max_pairwise_bytes < 1:
        raise ValueError("alignment memory guards and rank must be positive")
    if x.shape[1] * y.shape[1] > int(max_dense_features):
        return _fit_compressed_alignment(x, y, kind=kind, rank=rank, ot_regularization=ot_regularization, source_role=source_role, target_role=target_role, max_dense_features=int(max_dense_features), scalable_rank=int(scalable_rank), scalable_seed=int(scalable_seed), max_pairwise_bytes=int(max_pairwise_bytes))
    if kind in {"whitened_orthogonal", "riemannian"}:
        k = rank or min(x.shape[1], y.shape[1])
        if k < 1 or k > min(x.shape[1], y.shape[1]):
            raise ValueError("alignment rank must be between 1 and the smaller chart dimension")
        wx = fit_whitening(x, rank=k)
        wy = fit_whitening(y, rank=k)
        xr = wx.encode(x)
        yr = wy.encode(y)
        rotation = orthogonal_procrustes(xr, yr)
        matrix = wx.components @ rotation @ wy.inverse_components
        bias = wy.mean - wx.mean @ matrix
        label = kind
        metadata: dict[str, Any] = {"reduced_rank": k, "orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(k)))}
        ot_mass_error = 0.0
    elif kind == "affine":
        augmented = np.column_stack([x, np.ones(x.shape[0])])
        full = np.linalg.lstsq(augmented, y, rcond=1e-10)[0]
        matrix = full[:-1]
        bias = full[-1]
        label = kind
        metadata = {}
        ot_mass_error = 0.0
    elif kind == "low_rank":
        k = rank or min(x.shape[1], y.shape[1])
        matrix, bias = fit_low_rank_map(x, y, rank=k)
        label = kind
        metadata = {"requested_rank": k}
        ot_mass_error = 0.0
    elif kind == "ot_barycentric":
        if x.shape[1] != y.shape[1]:
            raise ValueError("OT barycentric map requires equal dimensions")
        plan, ot_mass_error = sinkhorn_plan(x, y, regularization=ot_regularization)
        barycentric = plan / np.maximum(plan.sum(axis=1, keepdims=True), 1e-300)
        pseudo_target = barycentric @ y
        matrix, bias = fit_low_rank_map(x, pseudo_target, rank=rank or x.shape[1])
        label = kind
        metadata = {"ot_regularization": ot_regularization}
    else:
        raise ValueError(f"unknown alignment kind: {kind}")
    if (source_role is None) != (target_role is None):
        raise ValueError("source_role and target_role must be supplied together")
    if source_role is not None:
        if source_role not in {"teacher", "student", "source"} or target_role not in {"teacher", "student", "target"}:
            raise ValueError("alignment roles must be teacher/student or source/target")
        metadata = {**metadata, "source_role": source_role, "target_role": target_role, "direction": f"{source_role}_to_{target_role}"}
    mapped = x @ matrix + bias
    paired = float(np.sqrt(np.mean(np.sum((mapped - y) ** 2, axis=1))))
    rel = relational_error(mapped, y, max_pairwise_bytes=max_pairwise_bytes)
    reverse_matrix, reverse_bias = fit_low_rank_map(y, x, rank=min(x.shape[1], y.shape[1]))
    cycle = float(np.sqrt(np.mean(np.sum((mapped @ reverse_matrix + reverse_bias - x) ** 2, axis=1))))
    # A map fitted from fewer samples than both chart dimensions has an
    # exactly non-trivial null space.  Avoid a large dense SVD whose only
    # useful conclusion would be that the condition number is infinite.
    if x.shape[0] < min(x.shape[1], y.shape[1]):
        condition = float("inf")
    else:
        singular = np.linalg.svd(matrix, compute_uv=False)
        condition = float(np.inf if singular.size == 0 or singular[-1] < 1e-12 else singular[0] / singular[-1])
    representation = "dense_low_rank_matrix" if label == "low_rank" else "dense_affine_map"
    metadata = {**metadata, "representation": representation, "dense_map_feature_count": int(x.shape[1] * y.shape[1]), "dense_map_estimated_bytes": int(x.shape[1] * y.shape[1] * 8), "pairwise_distance_backend": "scaled_gram_identity_blockwise", "pairwise_distance_feature_broadcast_allocated": False, "pairwise_distance_max_pairwise_bytes": int(max_pairwise_bytes), "pairwise_distance_output_shape": [int(mapped.shape[0]), int(y.shape[0])]}
    return AlignmentResult(label, x.mean(axis=0), y.mean(axis=0), matrix, bias, int(np.linalg.matrix_rank(x - x.mean(axis=0))), int(np.linalg.matrix_rank(y - y.mean(axis=0))), paired, rel, cycle, condition, float(ot_mass_error), metadata)


def trajectory_alignment_samples(source_traces: Sequence[Any], target_traces: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build paired all-depth samples for a teacher-to-student chart fit.

    The source depth grid is retained.  Target states are linearly
    interpolated only as training observations at those existing coordinates;
    this does not create a model layer or alter either trace artifact.  The
    helper is explicit because an initial-state-only map is often
    underdetermined for a many-layer cross-model transfer.
    """

    if len(source_traces) != len(target_traces) or not source_traces:
        raise ValueError("trajectory alignment needs paired non-empty trace sets")
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for source_trace, target_trace in zip(source_traces, target_traces):
        if source_trace.probe_id != target_trace.probe_id:
            raise ValueError("trajectory alignment traces must have paired probe IDs")
        if source_trace.state_dim < 1 or target_trace.state_dim < 1:
            raise ValueError("trajectory alignment traces must have non-empty state charts")
        source_depth = np.asarray(source_trace.depth_coordinates, dtype=np.float64)
        target_depth = np.asarray(target_trace.depth_coordinates, dtype=np.float64)
        if np.any(np.diff(source_depth) <= 0) or np.any(np.diff(target_depth) <= 0):
            raise ValueError("trajectory alignment depth coordinates must be strictly increasing")
        # Coordinates are normalized depth coordinates in the observation
        # contract, but connectors may expose different finite domains. Map
        # the source domain into the target domain before interpolation so a
        # 48-layer teacher and 12-layer student are compared at the same
        # relative depth rather than at the target endpoint after depth 12.
        source_query = _map_depth_domain(source_depth, target_depth)
        target_at_source = np.asarray([
            _interpolate_state_row(target_trace.hidden_states, target_depth, query)
            for query in source_query
        ])
        source_rows.append(np.asarray(source_trace.hidden_states, dtype=np.float64))
        target_rows.append(target_at_source)
    source = finite_array(np.concatenate(source_rows, axis=0), ndim=2, name="trajectory alignment source")
    target = finite_array(np.concatenate(target_rows, axis=0), ndim=2, name="trajectory alignment target")
    return source, target, {
        "alignment_fit_scope": "all_observed_source_depths_with_target_interpolation",
        "sample_count": int(source.shape[0]),
        "source_trace_count": int(len(source_traces)),
        "source_nodes_per_trace": [int(trace.layer_count) for trace in source_traces],
        "target_nodes_per_trace": [int(trace.layer_count) for trace in target_traces],
        "interpolation_creates_no_layer": True,
    }


def _map_depth_domain(source_coordinates: np.ndarray, target_coordinates: np.ndarray) -> np.ndarray:
    """Map one monotone depth domain to another by relative position."""

    source = finite_array(source_coordinates, ndim=1, name="source depth coordinates")
    target = finite_array(target_coordinates, ndim=1, name="target depth coordinates")
    if len(source) < 1 or len(target) < 1 or np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0):
        raise ValueError("depth domains must be non-empty and strictly increasing")
    fraction = (source - source[0]) / max(float(source[-1] - source[0]), 1e-12)
    return target[0] + fraction * float(target[-1] - target[0])


def _interpolate_state_row(states: np.ndarray, coordinates: np.ndarray, query: float) -> np.ndarray:
    """Interpolate one full state row without one ``np.interp`` per feature."""

    values = np.asarray(states)
    depth = np.asarray(coordinates)
    if query <= depth[0]:
        return values[0].copy()
    if query >= depth[-1]:
        return values[-1].copy()
    index = int(np.searchsorted(depth, query, side="right") - 1)
    fraction = (float(query) - float(depth[index])) / float(depth[index + 1] - depth[index])
    return (1.0 - fraction) * values[index] + fraction * values[index + 1]


def _fit_weighted_compact_map(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    rank: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a compact affine map with separately weighted state/jet rows."""

    x = finite_array(source, ndim=2, name="weighted compact source")
    y = finite_array(target, ndim=2, name="weighted compact target")
    w = finite_array(weights, ndim=1, name="weighted compact rows")
    if x.shape[0] != y.shape[0] or len(w) != len(x) or np.any(w <= 0.0):
        raise ValueError("weighted compact rows must have equal non-empty positive weights")
    if rank < 1 or rank > min(x.shape[1], y.shape[1]) or ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("weighted compact rank or ridge is invalid")
    augmented = np.column_stack([x, np.ones(len(x))])
    sqrt_weights = np.sqrt(w)[:, None]
    weighted_augmented = augmented * sqrt_weights
    weighted_target = y * sqrt_weights
    regularizer = np.eye(augmented.shape[1], dtype=np.float64) * float(ridge)
    # The intercept is not a chart direction and must not be shrunk.
    regularizer[-1, -1] = 0.0
    gram = weighted_augmented.T @ weighted_augmented + regularizer
    gram = (gram + gram.T) / 2.0
    try:
        solved = np.linalg.solve(gram, weighted_augmented.T @ weighted_target)
    except np.linalg.LinAlgError:
        solved = np.linalg.lstsq(weighted_augmented, weighted_target, rcond=1e-10)[0]
    matrix = solved[:-1]
    if rank < min(matrix.shape):
        left, singular, right_transpose = np.linalg.svd(matrix, full_matrices=False)
        matrix = (left[:, :rank] * singular[:rank]) @ right_transpose[:rank]
    weighted_mean_source = np.average(x, axis=0, weights=w)
    weighted_mean_target = np.average(y, axis=0, weights=w)
    bias = weighted_mean_target - weighted_mean_source @ matrix
    return finite_array(matrix, ndim=2, name="weighted compact matrix"), finite_array(bias, ndim=1, name="weighted compact bias")


def _smooth_depth_coefficients(values: np.ndarray, weight: float) -> np.ndarray:
    """Apply a deterministic quadratic smoothness prior along observed depth."""

    array = finite_array(values, ndim=3 if values.ndim == 3 else 2, name="depth coefficients")
    if weight <= 0.0:
        return array.copy()
    if not np.isfinite(weight):
        raise ValueError("depth smoothness weight must be finite")
    node_count = array.shape[0]
    if node_count < 2:
        return array.copy()
    diagonal = np.full(node_count, 1.0 + 2.0 * float(weight), dtype=np.float64)
    diagonal[[0, -1]] = 1.0 + float(weight)
    system = np.diag(diagonal)
    system += np.diag(np.full(node_count - 1, -float(weight)), k=1)
    system += np.diag(np.full(node_count - 1, -float(weight)), k=-1)
    flattened = array.reshape(node_count, -1)
    smoothed = np.linalg.solve(system, flattened)
    return finite_array(smoothed.reshape(array.shape), name="smoothed depth coefficients")


def _coefficient_discontinuity(matrices: np.ndarray, biases: np.ndarray) -> float:
    if len(matrices) < 2:
        return 0.0
    matrix_delta = np.diff(matrices, axis=0)
    bias_delta = np.diff(biases, axis=0)
    numerator = float(stable_l2(matrix_delta, name="depth matrix discontinuity") ** 2 + stable_l2(bias_delta, name="depth bias discontinuity") ** 2) ** 0.5
    denominator = max(1.0, float(stable_l2(matrices, name="depth matrix scale")), float(stable_l2(biases, name="depth bias scale")))
    return float(numerator / denominator)


def _jacobian_spectrum_gate(source_transition: Any, target_transition: Any) -> float | None:
    """Compare stored Jacobian singular descriptors without chart-sized matmul."""

    source_values = getattr(source_transition, "singular_values", None)
    target_values = getattr(target_transition, "singular_values", None)
    if source_values is None or target_values is None:
        return None
    source = finite_array(source_values, ndim=1, name="source Jacobian singular spectrum")
    target = finite_array(target_values, ndim=1, name="target Jacobian singular spectrum")
    count = min(len(source), len(target))
    if count < 1:
        return None
    source = np.log1p(np.abs(source[:count]))
    target = np.log1p(np.abs(target[:count]))
    scale = max(1.0, float(stable_l2(source, name="source Jacobian spectrum")), float(stable_l2(target, name="target Jacobian spectrum")))
    discrepancy = float(stable_l2(source - target, name="Jacobian spectrum discrepancy") / scale)
    return float(np.exp(-np.clip(discrepancy, 0.0, 40.0)))


def fit_depth_conditioned_alignment(
    source_traces: Sequence[Any],
    target_traces: Sequence[Any],
    *,
    kind: str = "low_rank",
    rank: int | None = None,
    ridge: float = 1e-6,
    source_role: str | None = None,
    target_role: str | None = None,
    max_dense_features: int = DEFAULT_MAX_DENSE_FEATURES,
    scalable_rank: int = DEFAULT_SCALABLE_RANK,
    scalable_seed: int = 0,
    max_pairwise_bytes: int = DEFAULT_MAX_PAIRWISE_BYTES,
    velocity_weight: float = 0.0,
    jacobian_weight: float = 0.0,
    smoothness_weight: float = 0.0,
) -> AlignmentResult:
    """Fit one shared chart basis with affine coefficients per source depth.

    The maps are fitted only from the supplied traces, normally the training
    split.  Each target trajectory is interpolated at the existing source
    coordinates; no target layer is created.  Validation/holdout application
    uses the stored depth maps and never refits their coefficients.
    """

    if kind not in {"low_rank", "affine"}:
        raise ValueError("depth-conditioned alignment currently supports low_rank or affine maps")
    if not source_traces or len(source_traces) != len(target_traces):
        raise ValueError("depth-conditioned alignment needs paired non-empty traces")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("depth-conditioned alignment ridge must be finite and non-negative")
    if max_dense_features < 1 or scalable_rank < 1 or max_pairwise_bytes < 1:
        raise ValueError("depth-conditioned alignment guards and rank must be positive")
    for name, value in (("velocity_weight", velocity_weight), ("jacobian_weight", jacobian_weight), ("smoothness_weight", smoothness_weight)):
        if value < 0.0 or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
    dynamic_aware = bool(velocity_weight > 0.0 or jacobian_weight > 0.0 or smoothness_weight > 0.0)
    source_dim = int(source_traces[0].state_dim)
    target_dim = int(target_traces[0].state_dim)
    source_coordinates = np.asarray(source_traces[0].depth_coordinates, dtype=np.float64)
    if len(source_coordinates) < 1 or np.any(np.diff(source_coordinates) <= 0):
        raise ValueError("source depth coordinates must be strictly increasing")
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    target_queries: list[np.ndarray] = []
    for source_trace, target_trace in zip(source_traces, target_traces):
        if source_trace.probe_id != target_trace.probe_id:
            raise ValueError("depth-conditioned alignment traces must retain paired probe IDs")
        if source_trace.state_dim != source_dim or target_trace.state_dim != target_dim:
            raise ValueError("depth-conditioned alignment traces have inconsistent dimensions")
        if not np.array_equal(np.asarray(source_trace.depth_coordinates, dtype=np.float64), source_coordinates):
            raise ValueError("depth-conditioned alignment requires a common source depth grid")
        target_coordinates = finite_array(target_trace.depth_coordinates, ndim=1, name="target depth coordinates")
        if len(target_coordinates) < 1 or np.any(np.diff(target_coordinates) <= 0):
            raise ValueError("target depth coordinates must be strictly increasing")
        target_query = _map_depth_domain(source_coordinates, target_coordinates)
        target_queries.append(target_query)
        target_at_source = np.asarray([
            _interpolate_state_row(target_trace.hidden_states, target_coordinates, query)
            for query in target_query
        ])
        source_rows.append(finite_array(source_trace.hidden_states, ndim=2, name="depth-conditioned source states"))
        target_rows.append(finite_array(target_at_source, ndim=2, name="depth-conditioned target states"))
    source_all = finite_array(np.concatenate(source_rows, axis=0), ndim=2, name="depth-conditioned source samples")
    target_all = finite_array(np.concatenate(target_rows, axis=0), ndim=2, name="depth-conditioned target samples")
    sample_count = int(source_all.shape[0])
    chart_rank = min(int(rank or scalable_rank), int(scalable_rank), int(np.sqrt(max_dense_features)), sample_count - 1, source_dim, target_dim)
    if chart_rank < 1:
        raise ValueError("depth-conditioned alignment has no retained chart rank")
    source_projection = data_aware_chart_projection(source_all, chart_rank, scalable_seed)
    target_projection = data_aware_chart_projection(target_all, chart_rank, scalable_seed + 1)
    source_mean = source_all.mean(axis=0)
    target_mean = target_all.mean(axis=0)
    source_chart_all = (source_all - source_mean) @ source_projection
    target_chart_all = (target_all - target_mean) @ target_projection
    local_matrices: list[np.ndarray] = []
    local_biases: list[np.ndarray] = []
    reverse_matrices: list[np.ndarray] = []
    reverse_biases: list[np.ndarray] = []
    local_mapped_rows = np.empty_like(target_all, dtype=np.float64)
    depth_metrics: list[dict[str, Any]] = []
    local_conditions: list[float] = []
    velocity_row_count = 0
    jacobian_row_count = 0
    jacobian_descriptor_count = 0
    jacobian_direct_skipped = False
    for node, depth in enumerate(source_coordinates):
        source_node = np.asarray([row[node] for row in source_rows], dtype=np.float64)
        target_node = np.asarray([row[node] for row in target_rows], dtype=np.float64)
        source_node_chart = (source_node - source_mean) @ source_projection
        target_node_chart = (target_node - target_mean) @ target_projection
        if dynamic_aware:
            fit_sources = [source_node_chart]
            fit_targets = [target_node_chart]
            state_weights = np.ones(len(source_node), dtype=np.float64)
            fit_weights = [state_weights]
            if velocity_weight > 0.0 and node < len(source_coordinates) - 1:
                velocity_sources = []
                velocity_targets = []
                for source_trace, target_trace, target_query in zip(source_traces, target_traces, target_queries):
                    source_delta = source_trace.transitions[node].delta
                    target_delta = _interpolate_state_row(target_trace.hidden_states, target_trace.depth_coordinates, target_query[node + 1]) - _interpolate_state_row(target_trace.hidden_states, target_trace.depth_coordinates, target_query[node])
                    velocity_sources.append(source_delta @ source_projection)
                    velocity_targets.append(target_delta @ target_projection)
                if velocity_sources:
                    fit_sources.append(np.asarray(velocity_sources, dtype=np.float64))
                    fit_targets.append(np.asarray(velocity_targets, dtype=np.float64))
                    fit_weights.append(np.full(len(velocity_sources), float(velocity_weight), dtype=np.float64))
                    velocity_row_count += len(velocity_sources)
            if jacobian_weight > 0.0 and node < len(source_coordinates) - 1:
                jacobian_sources = []
                jacobian_targets = []
                for trace_index, (source_trace, target_trace, target_query) in enumerate(zip(source_traces, target_traces, target_queries)):
                    teacher_jacobian = source_trace.transitions[node].jacobian
                    if teacher_jacobian is None:
                        continue
                    target_transition_index = int(np.argmin(np.abs(target_trace.depth_coordinates[:-1] - target_query[node])))
                    student_jacobian = target_trace.transitions[target_transition_index].jacobian
                    if student_jacobian is None:
                        continue
                    direction_count = min(teacher_jacobian.shape[1], student_jacobian.shape[1])
                    if direction_count < 1:
                        continue
                    direct_elements = (source_dim + target_dim) * direction_count
                    if direct_elements <= DEFAULT_MAX_DIRECT_JACOBIAN_ROW_ELEMENTS:
                        jacobian_sources.append(teacher_jacobian[:, :direction_count].T @ source_projection)
                        jacobian_targets.append(student_jacobian[:, :direction_count].T @ target_projection)
                    else:
                        jacobian_direct_skipped = True
                        jacobian_descriptor_count += 1
                        gate = _jacobian_spectrum_gate(source_trace.transitions[node], target_trace.transitions[target_transition_index])
                        if gate is not None:
                            # The descriptor is a bounded confidence weight on
                            # the same state/velocity equations. It changes
                            # the solve without allocating d×k chart rows.
                            state_weights[trace_index] *= 0.5 + 0.5 * float(gate)
                if jacobian_sources:
                    fit_sources.append(np.concatenate(jacobian_sources, axis=0))
                    fit_targets.append(np.concatenate(jacobian_targets, axis=0))
                    fit_weights.append(np.full(sum(len(value) for value in jacobian_sources), float(jacobian_weight), dtype=np.float64))
                    jacobian_row_count += sum(len(value) for value in jacobian_sources)
            matrix, bias = _fit_weighted_compact_map(
                np.concatenate(fit_sources, axis=0),
                np.concatenate(fit_targets, axis=0),
                np.concatenate(fit_weights, axis=0),
                rank=chart_rank,
                ridge=ridge,
            )
            reverse_matrix, reverse_bias = _fit_weighted_compact_map(
                target_node_chart,
                source_node_chart,
                np.ones(len(target_node_chart), dtype=np.float64),
                rank=chart_rank,
                ridge=ridge,
            )
        elif kind == "affine":
            augmented = np.column_stack([source_node_chart, np.ones(len(source_node_chart))])
            solved = np.linalg.lstsq(augmented, target_node_chart, rcond=1e-10)[0]
            matrix = solved[:-1]
            bias = solved[-1]
            reverse_augmented = np.column_stack([target_node_chart, np.ones(len(target_node_chart))])
            reverse_solved = np.linalg.lstsq(reverse_augmented, source_node_chart, rcond=1e-10)[0]
            reverse_matrix = reverse_solved[:-1]
            reverse_bias = reverse_solved[-1]
        else:
            matrix, bias = fit_low_rank_map(source_node_chart, target_node_chart, rank=chart_rank, ridge=ridge)
            reverse_matrix, reverse_bias = fit_low_rank_map(target_node_chart, source_node_chart, rank=chart_rank, ridge=ridge)
        if not dynamic_aware and kind == "affine":
            reverse_augmented = np.column_stack([target_node_chart, np.ones(len(target_node_chart))])
            reverse_solved = np.linalg.lstsq(reverse_augmented, source_node_chart, rcond=1e-10)[0]
            reverse_matrix = reverse_solved[:-1]
            reverse_bias = reverse_solved[-1]
        local_matrices.append(finite_array(matrix, ndim=2, name="depth-conditioned local map"))
        local_biases.append(finite_array(bias, ndim=1, name="depth-conditioned local bias"))
        reverse_matrices.append(finite_array(reverse_matrix, ndim=2, name="depth-conditioned reverse map"))
        reverse_biases.append(finite_array(reverse_bias, ndim=1, name="depth-conditioned reverse bias"))
    local_matrices_array = np.asarray(local_matrices, dtype=np.float64)
    local_biases_array = np.asarray(local_biases, dtype=np.float64)
    reverse_matrices_array = np.asarray(reverse_matrices, dtype=np.float64)
    reverse_biases_array = np.asarray(reverse_biases, dtype=np.float64)
    raw_discontinuity = _coefficient_discontinuity(local_matrices_array, local_biases_array)
    local_matrices_array = _smooth_depth_coefficients(local_matrices_array, smoothness_weight)
    local_biases_array = _smooth_depth_coefficients(local_biases_array, smoothness_weight)
    reverse_matrices_array = _smooth_depth_coefficients(reverse_matrices_array, smoothness_weight)
    reverse_biases_array = _smooth_depth_coefficients(reverse_biases_array, smoothness_weight)
    smoothed_discontinuity = _coefficient_discontinuity(local_matrices_array, local_biases_array)
    for node, depth in enumerate(source_coordinates):
        source_node = np.asarray([row[node] for row in source_rows], dtype=np.float64)
        target_node = np.asarray([row[node] for row in target_rows], dtype=np.float64)
        source_node_chart = (source_node - source_mean) @ source_projection
        mapped_chart = source_node_chart @ local_matrices_array[node] + local_biases_array[node]
        mapped_node = mapped_chart @ target_projection.T + target_mean
        # ``source_all`` is trace-major: [trace0 node0..nodeN, trace1 ...].
        row_indices = np.arange(len(source_rows), dtype=np.int64) * len(source_coordinates) + node
        local_mapped_rows[row_indices] = mapped_node
        residual = mapped_node - target_node
        normalized_error = float(stable_l2(residual, name="depth-conditioned local residual") / max(1.0, float(stable_l2(target_node, name="depth-conditioned target node"))))
        singular = np.linalg.svd(local_matrices_array[node], compute_uv=False)
        condition = float(np.inf if singular.size == 0 or singular[-1] <= 1e-12 else singular[0] / singular[-1])
        local_conditions.append(condition)
        depth_metrics.append({
            "depth": float(depth),
            "sample_count": int(len(source_node)),
            "paired_error": float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
            "normalized_error": normalized_error,
            "source_norm": float(stable_l2(source_node, name="depth-conditioned source norm")),
            "target_norm": float(stable_l2(target_node, name="depth-conditioned target norm")),
            "map_condition_number": condition,
            "map_rank": int(np.linalg.matrix_rank(local_matrices_array[node])),
        })
    # Fit a genuine single global reference in the same shared compact chart.
    # Averaging local matrices is useful as a diagnostic, but it is not a
    # fair global baseline because it is not itself least-squares fitted.
    if kind == "affine":
        global_augmented = np.column_stack([source_chart_all, np.ones(sample_count)])
        global_solved = np.linalg.lstsq(global_augmented, target_chart_all, rcond=1e-10)[0]
        global_matrix = global_solved[:-1]
        global_bias = global_solved[-1]
    else:
        global_matrix, global_bias = fit_low_rank_map(source_chart_all, target_chart_all, rank=chart_rank, ridge=ridge)
    global_mapped = source_chart_all @ global_matrix @ target_projection.T + global_bias @ target_projection.T + target_mean
    global_error = float(np.sqrt(np.mean(np.sum((global_mapped - target_all) ** 2, axis=1))))
    mean_local_matrix = np.mean(local_matrices_array, axis=0)
    mean_local_bias = np.mean(local_biases_array, axis=0)
    mean_local_mapped = source_chart_all @ mean_local_matrix @ target_projection.T + mean_local_bias @ target_projection.T + target_mean
    mean_local_error = float(np.sqrt(np.mean(np.sum((mean_local_mapped - target_all) ** 2, axis=1))))
    local_error = float(np.sqrt(np.mean(np.sum((local_mapped_rows - target_all) ** 2, axis=1))))
    reverse_chart_rows: list[np.ndarray] = []
    for node in range(len(source_coordinates)):
        # Rows are stored trace-major, so take the node slice explicitly.
        mapped_node = local_mapped_rows[np.arange(len(source_rows)) * len(source_coordinates) + node]
        mapped_chart = (mapped_node - target_mean) @ target_projection
        source_node_chart = source_chart_all[np.arange(len(source_rows)) * len(source_coordinates) + node]
        reverse_chart_rows.append(mapped_chart @ reverse_matrices_array[node] + reverse_biases_array[node] - source_node_chart)
    reverse_residual = np.concatenate(reverse_chart_rows, axis=0)
    cycle_error = float(np.sqrt(np.mean(np.sum(reverse_residual * reverse_residual, axis=1))))
    finite_conditions = [value for value in local_conditions if np.isfinite(value)]
    condition = float(np.median(finite_conditions)) if finite_conditions else float("inf")
    metadata: dict[str, Any] = {
        "representation": "shared_low_rank_basis_depth_conditioned_affine",
        "alignment_strategy": "depth_conditioned_dynamic_aware" if dynamic_aware else "depth_conditioned_local_maps",
        "depth_conditioned": True,
        "dynamic_aware": dynamic_aware,
        "dynamic_objective": "weighted_state_transition_jacobian_smoothness" if dynamic_aware else "state_only",
        "dynamic_weights": {
            "velocity": float(velocity_weight),
            "jacobian": float(jacobian_weight),
            "smoothness": float(smoothness_weight),
        },
        "velocity_row_count": int(velocity_row_count),
        "jacobian_row_count": int(jacobian_row_count),
        "jacobian_descriptor_count": int(jacobian_descriptor_count),
        "jacobian_direct_row_transport": bool(jacobian_row_count > 0),
        "jacobian_direct_skipped": bool(jacobian_direct_skipped),
        "jacobian_skipped_reason": (
            "direct chart Jacobian rows exceed memory guard; stored singular-spectrum descriptor applied"
            if jacobian_direct_skipped else None
        ),
        "local_map_discontinuity_before": float(raw_discontinuity),
        "local_map_discontinuity_after": float(smoothed_discontinuity),
        "smoothness_applied": bool(smoothness_weight > 0.0),
        "depth_fit_domain": "source_depth_coordinates",
        "direction": None if source_role is None else f"{source_role}_to_{target_role}",
        "source_role": source_role,
        "target_role": target_role,
        "source_projection_shape": list(source_projection.shape),
        "target_projection_shape": list(target_projection.shape),
        "compressed_rank": int(chart_rank),
        "depth_node_count": int(len(source_coordinates)),
        "depth_metrics": depth_metrics,
        "global_shared_map_paired_error": global_error,
        "mean_local_operator_paired_error": mean_local_error,
        "depth_local_paired_error": local_error,
        "depth_local_improvement_over_global": float(global_error - local_error),
        "depth_local_map_condition_median": condition,
        "max_dense_features": int(max_dense_features),
        "dense_map_feature_count": int(source_dim * target_dim),
        "dense_map_estimated_bytes": int(source_dim * target_dim * 8),
        "dense_map_skipped_reason": "source_dim*target_dim exceeds max_dense_features" if source_dim * target_dim > max_dense_features else "shared compact basis selected for depth conditioning",
        "discarded_source_rank": max(0, source_dim - chart_rank),
        "discarded_target_rank": max(0, target_dim - chart_rank),
        "alignment_fit_scope": "train_depth_conditioned_source_nodes_with_target_interpolation",
        "target_fit_split": "train",
        "pairwise_distance_backend": "scaled_gram_identity_blockwise",
        "pairwise_distance_feature_broadcast_allocated": False,
        "pairwise_distance_max_pairwise_bytes": int(max_pairwise_bytes),
        "pairwise_distance_output_shape": [sample_count, sample_count],
        "validation_refit": False,
    }
    return AlignmentResult(
        f"depth_conditioned_{kind}",
        source_mean,
        target_mean,
        global_matrix,
        global_bias,
        int(np.linalg.matrix_rank(source_chart_all)),
        int(np.linalg.matrix_rank(target_chart_all)),
        local_error,
        relational_error(local_mapped_rows, target_all, max_pairwise_bytes=max_pairwise_bytes),
        cycle_error,
        condition,
        0.0,
        metadata,
        source_projection,
        target_projection,
        source_coordinates,
        local_matrices_array,
        local_biases_array,
    )


def fit_dynamic_depth_conditioned_alignment(
    source_traces: Sequence[Any],
    target_traces: Sequence[Any],
    *,
    velocity_weight: float = 1.0,
    jacobian_weight: float = 0.25,
    smoothness_weight: float = 0.25,
    **kwargs: Any,
) -> AlignmentResult:
    """Fit depth-local maps with explicit transition and tangent objectives."""

    if velocity_weight <= 0.0:
        raise ValueError("dynamic-aware velocity_weight must be positive")
    return fit_depth_conditioned_alignment(
        source_traces,
        target_traces,
        velocity_weight=velocity_weight,
        jacobian_weight=jacobian_weight,
        smoothness_weight=smoothness_weight,
        **kwargs,
    )


def evaluate_dynamic_alignment(
    source_traces: Sequence[Any],
    target_traces: Sequence[Any],
    alignment: AlignmentResult,
    *,
    velocity_weight: float = 1.0,
    jacobian_weight: float = 0.25,
    smoothness_weight: float = 0.25,
) -> dict[str, Any]:
    """Score state, transition, tangent, and coefficient continuity separately."""

    if len(source_traces) != len(target_traces) or not source_traces:
        raise ValueError("dynamic alignment evaluation needs paired non-empty traces")
    if any(source.probe_id != target.probe_id for source, target in zip(source_traces, target_traces)):
        raise ValueError("dynamic alignment evaluation requires paired probe IDs")
    if any(value < 0.0 or not np.isfinite(value) for value in (velocity_weight, jacobian_weight, smoothness_weight)):
        raise ValueError("dynamic alignment evaluation weights must be finite and non-negative")
    state_errors: list[float] = []
    velocity_errors: list[float] = []
    jacobian_errors: list[float] = []
    for source_trace, target_trace in zip(source_traces, target_traces):
        source_depth = finite_array(source_trace.depth_coordinates, ndim=1, name="dynamic source depth")
        target_depth = finite_array(target_trace.depth_coordinates, ndim=1, name="dynamic target depth")
        query_depth = _map_depth_domain(source_depth, target_depth)
        mapped_states = alignment.apply(source_trace.hidden_states, depth=source_depth)
        target_at_query = np.asarray([
            _interpolate_state_row(target_trace.hidden_states, target_depth, query)
            for query in query_depth
        ])
        state_errors.append(float(stable_l2(mapped_states - target_at_query, name="dynamic state residual") / max(1.0, float(stable_l2(target_at_query, name="dynamic target states")))))
        for node, transition in enumerate(source_trace.transitions):
            expected_delta = _interpolate_state_row(target_trace.hidden_states, target_depth, query_depth[node + 1]) - _interpolate_state_row(target_trace.hidden_states, target_depth, query_depth[node])
            predicted_delta = alignment.linear_apply(transition.delta, depth=transition.source_depth)
            velocity_errors.append(float(stable_l2(predicted_delta - expected_delta, name="dynamic transition residual") / max(1.0, float(stable_l2(expected_delta, name="dynamic target transition")))))
            teacher_jacobian = transition.jacobian
            if teacher_jacobian is None:
                continue
            target_index = int(np.argmin(np.abs(target_depth[:-1] - query_depth[node])))
            student_jacobian = target_trace.transitions[target_index].jacobian
            if student_jacobian is None:
                continue
            direction_count = min(teacher_jacobian.shape[1], student_jacobian.shape[1])
            if direction_count < 1:
                continue
            predicted_jacobian = alignment.linear_apply(teacher_jacobian[:, :direction_count].T, depth=transition.source_depth)
            expected_jacobian = student_jacobian[:, :direction_count].T
            if alignment.target_projection is not None:
                expected_jacobian = expected_jacobian @ alignment.target_projection @ alignment.target_projection.T
            jacobian_errors.append(float(stable_l2(predicted_jacobian - expected_jacobian, name="dynamic Jacobian residual") / max(1.0, float(stable_l2(expected_jacobian, name="dynamic target Jacobian")))))
    discontinuity = _coefficient_discontinuity(alignment.depth_matrices, alignment.depth_biases) if alignment.depth_matrices is not None and alignment.depth_biases is not None else 0.0
    metrics = {
        "state_error": float(np.mean(state_errors)),
        "velocity_error": float(np.mean(velocity_errors)) if velocity_errors else None,
        "jacobian_error": float(np.mean(jacobian_errors)) if jacobian_errors else None,
        "smoothness_error": float(discontinuity),
        "state_sample_count": int(len(state_errors)),
        "velocity_sample_count": int(len(velocity_errors)),
        "jacobian_sample_count": int(len(jacobian_errors)),
        "velocity_weight": float(velocity_weight),
        "jacobian_weight": float(jacobian_weight),
        "smoothness_weight": float(smoothness_weight),
    }
    objective = metrics["state_error"]
    if metrics["velocity_error"] is not None:
        objective += velocity_weight * metrics["velocity_error"]
    if metrics["jacobian_error"] is not None:
        objective += jacobian_weight * metrics["jacobian_error"]
    objective += smoothness_weight * metrics["smoothness_error"]
    metrics["objective"] = float(objective)
    if not np.isfinite(metrics["objective"]):
        raise FloatingPointError("dynamic alignment objective is non-finite")
    return metrics


def select_dynamic_depth_conditioned_alignment(
    train_source_traces: Sequence[Any],
    train_target_traces: Sequence[Any],
    validation_source_traces: Sequence[Any],
    validation_target_traces: Sequence[Any],
    *,
    candidates: Sequence[Mapping[str, float]] = (
        {"velocity_weight": 0.5, "jacobian_weight": 0.0, "smoothness_weight": 0.10},
        {"velocity_weight": 1.0, "jacobian_weight": 0.25, "smoothness_weight": 0.25},
        {"velocity_weight": 2.0, "jacobian_weight": 0.50, "smoothness_weight": 0.50},
    ),
    **fit_kwargs: Any,
) -> tuple[AlignmentResult, dict[str, Any]]:
    """Select dynamic-aware weights on validation while fitting only on train."""

    if not validation_source_traces or len(validation_source_traces) != len(validation_target_traces):
        raise ValueError("dynamic-aware selection requires paired validation traces")
    records: list[dict[str, Any]] = []
    fitted: list[tuple[AlignmentResult, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        values = {
            "velocity_weight": float(candidate.get("velocity_weight", 0.0)),
            "jacobian_weight": float(candidate.get("jacobian_weight", 0.0)),
            "smoothness_weight": float(candidate.get("smoothness_weight", 0.0)),
        }
        record: dict[str, Any] = {"candidate_index": int(index), "candidate": values, "selection_split": "validation", "fit_split": "train"}
        try:
            fit = fit_dynamic_depth_conditioned_alignment(train_source_traces, train_target_traces, **fit_kwargs, **values)
            # Candidate fit weights select the estimator, while candidates
            # are compared with one fixed validation policy. Otherwise a
            # candidate could win merely by assigning smaller coefficients to
            # the objective terms rather than by improving the measurements.
            metrics = evaluate_dynamic_alignment(
                validation_source_traces,
                validation_target_traces,
                fit,
                velocity_weight=1.0,
                jacobian_weight=1.0,
                smoothness_weight=1.0,
            )
            record.update({
                "status": "scored",
                "selection_objective_weights": {"velocity": 1.0, "jacobian": 1.0, "smoothness": 1.0},
                "fit_weights": values,
                **metrics,
                "train_fit_paired_error": float(fit.paired_error),
                "train_fit_relational_error": float(fit.relational_error),
                "train_fit_local_map_discontinuity": float(fit.metadata.get("local_map_discontinuity_after", 0.0)),
                "train_fit_jacobian_direct_skipped": bool(fit.metadata.get("jacobian_direct_skipped", False)),
            })
            fitted.append((fit, record))
        except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as error:
            record.update({"status": "rejected", "objective": None, "skipped_reason": f"{type(error).__name__}: {error}"})
        records.append(record)
    if not fitted:
        raise ValueError("all dynamic-aware alignment candidates were rejected")

    # Report a multi-objective frontier in addition to the fixed validation
    # objective.  This prevents a single scalar from hiding a tradeoff between
    # state geometry, transition fidelity, tangent fidelity, and depth
    # smoothness.  Missing metrics are treated as unavailable (infinite for
    # frontier purposes), never as zero.
    metric_names = ("state_error", "velocity_error", "jacobian_error", "smoothness_error")
    scored_indices = [index for index, record in enumerate(records) if record.get("status") == "scored"]
    pareto_indices: list[int] = []
    for candidate_index in scored_indices:
        candidate_metrics = tuple(float("inf") if records[candidate_index].get(name) is None else float(records[candidate_index][name]) for name in metric_names)
        dominated = False
        for other_index in scored_indices:
            if other_index == candidate_index:
                continue
            other_metrics = tuple(float("inf") if records[other_index].get(name) is None else float(records[other_index][name]) for name in metric_names)
            if all(other <= current for other, current in zip(other_metrics, candidate_metrics)) and any(other < current for other, current in zip(other_metrics, candidate_metrics)):
                dominated = True
                break
        records[candidate_index]["pareto_optimal"] = not dominated
        if not dominated:
            pareto_indices.append(candidate_index)
    selected, selected_record = min(fitted, key=lambda pair: (float(pair[1]["objective"]), int(pair[1]["candidate_index"])))
    metadata = dict(selected.metadata)
    metadata.update({
        "alignment_strategy": "depth_conditioned_dynamic_aware",
        "dynamic_selection": "validation_minimum_objective",
        "selection_split": "validation",
        "fit_split": "train",
        "selected_candidate_index": int(selected_record["candidate_index"]),
        "selected_candidate": dict(selected_record["candidate"]),
        "validation_selection_table": records,
        "pareto_front_candidate_indices": pareto_indices,
        "pareto_metrics": list(metric_names),
        "validation_refit": False,
    })
    return replace(selected, metadata=metadata), {
        "schema_version": "faytuna-dynamic-selection-v1",
        "status": "selected",
        "fit_split": "train",
        "selection_split": "validation",
        "selected_candidate_index": int(selected_record["candidate_index"]),
        "selected_candidate": dict(selected_record["candidate"]),
        "pareto_front_candidate_indices": pareto_indices,
        "pareto_metrics": list(metric_names),
        "candidates": records,
    }


def project_teacher_weight(
    teacher_weight: np.ndarray,
    projection: np.ndarray,
    *,
    site: str = "hidden_to_hidden",
) -> np.ndarray:
    """Project teacher weight matrix into student coordinate dimension.

    For GPT-2 Conv1D convention (x @ W):
    - ``hidden_to_hidden`` (e.g. attn.c_proj): W_T has shape (d_T, d_T).
      With orthonormalized projection U (d_T, d_S), the projected weight is
      U.T @ W_T @ U with shape (d_S, d_S).
    - ``mlp_to_hidden`` (e.g. mlp.c_proj): W_T has shape (4*d_T, d_T).
      Output dimension contracts via @ U. Input dimension contracts via
      (I_4 (x) U.T), reshaping W_T to (4, d_T, d_T) and multiplying each
      block, producing shape (4*d_S, d_S).
    """
    wt = finite_array(teacher_weight, ndim=2, name="teacher_weight")
    proj = finite_array(projection, ndim=2, name="alignment projection")
    dt, ds = proj.shape
    if dt < ds:
        raise ValueError(f"projection must map teacher ({dt}) to student ({ds}), expected dt >= ds")

    u, _, _ = np.linalg.svd(proj, full_matrices=False)
    u = u[:, :ds]

    if site == "hidden_to_hidden":
        if wt.shape != (dt, dt):
            raise ValueError(f"hidden_to_hidden weight shape {wt.shape} does not match teacher dim {dt}")
        return u.T @ wt @ u
    elif site == "mlp_to_hidden":
        if wt.shape[0] % dt == 0:
            mult = wt.shape[0] // dt
            w_out = wt @ u
            blocks = w_out.reshape(mult, dt, ds)
            proj_blocks = np.stack([u.T @ b for b in blocks], axis=0)
            return proj_blocks.reshape(mult * ds, ds)
        elif wt.shape[1] % dt == 0 and wt.shape[0] == dt:
            mult = wt.shape[1] // dt
            w_in = wt.T
            w_out = w_in @ u
            blocks = w_out.reshape(mult, dt, ds)
            proj_blocks = np.stack([u.T @ b for b in blocks], axis=0)
            return proj_blocks.reshape(mult * ds, ds).T
        else:
            raise ValueError(f"mlp_to_hidden weight shape {wt.shape} does not match (k*{dt}, {dt}) or ({dt}, k*{dt})")
    else:
        raise ValueError(f"unsupported projection site: {site!r}; expected 'hidden_to_hidden' or 'mlp_to_hidden'")


def compute_frobenius_dissimilarity(
    w_teacher_proj: np.ndarray,
    w_student: np.ndarray,
) -> float:
    """Compute normalized Frobenius distance between projected teacher and student weights."""
    wt_p = finite_array(w_teacher_proj, ndim=2, name="w_teacher_proj")
    ws = finite_array(w_student, ndim=2, name="w_student")
    if wt_p.shape != ws.shape:
        raise ValueError(f"shape mismatch between projected teacher {wt_p.shape} and student {ws.shape}")
    diff_norm = float(np.linalg.norm(wt_p - ws, ord="fro"))
    base_norm = float(np.linalg.norm(ws, ord="fro"))
    return diff_norm / max(base_norm, 1e-12)


def dissimilarity_gain_schedule(
    distances: Sequence[float],
    *,
    min_gain: float = 0.25,
    max_gain: float = 2.0,
) -> tuple[float, ...]:
    """Derive layer-wise gain weights from normalized Frobenius distances.

    Layers with large distance (student severely deficient relative to teacher)
    receive higher gain; layers with small distance (weights already close)
    receive suppressed gain to protect foundational representations.
    """
    arr = np.asarray(distances, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("distances must be a non-empty 1D sequence")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("distances must be finite and non-negative")
    mean_dist = float(np.mean(arr))
    if mean_dist < 1e-12:
        return tuple(1.0 for _ in range(arr.size))
    raw_ratios = arr / mean_dist
    clipped = np.clip(raw_ratios, min_gain, max_gain)
    return tuple(float(g) for g in clipped)

