"""Piecewise MLP Knot Detection, Multi-Trial Retry & Attention Detour Routing.

In transformer architectures, MLP layers function as key-value associative
memories. In a student model undergoing flow-transfer surgery, certain
subspaces exhibit polysemantic chaotic superposition ('knots') where
monolithic gradient or least-squares projection degrades existing model
capabilities.

This module provides:
1. Spectral Subspace Entropy & Knot Detection (compute_subspace_entropy, is_knot_subspace)
2. Piecewise Subspace Fitter with Multi-Trial Retry and Safe Knot Skip (fit_piecewise_mlp_with_retries)
3. Attention Detour Projector for Routing Bypass (build_attention_detour_projector)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .types import finite_array


def compute_subspace_entropy(X: np.ndarray) -> float:
    """Compute normalized spectral entropy H in [0, 1] of a representation matrix.

    For singular values sigma_i of X, the normalized distribution is:
        p_i = sigma_i / sum_j(sigma_j)
    and the normalized spectral entropy is:
        H = - sum_i (p_i * ln(p_i)) / ln(min(N, D))

    Low entropy (H -> 0) corresponds to clear low-rank semantic structure.
    High entropy (H -> 1, e.g. H > 0.85) corresponds to isotropic chaos or
    superposition ('knot').

    Handles 1D arrays, empty inputs, zero matrices, and non-finite values safely.
    """
    if X is None:
        return 0.0
    arr = np.asarray(X, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return 0.0
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    min_dim = min(arr.shape[0], arr.shape[1])
    if min_dim <= 1:
        return 0.0

    singular = np.linalg.svd(arr, compute_uv=False)
    positive = singular[np.isfinite(singular) & (singular > 1e-12)]
    if positive.size <= 1:
        return 0.0

    total = float(np.sum(positive))
    if total <= 1e-15:
        return 0.0

    p = positive / total
    p = p[p > 1e-15]
    if p.size <= 1:
        return 0.0

    shannon = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(min_dim))
    if max_entropy <= 1e-15:
        return 0.0

    normalized = shannon / max_entropy
    return float(np.clip(normalized, 0.0, 1.0))


def is_knot_subspace(X: np.ndarray, threshold: float = 0.85) -> bool:
    """Return True if spectral subspace entropy exceeds the knot threshold."""
    return compute_subspace_entropy(X) > float(threshold)


@dataclass(frozen=True)
class PieceReport:
    """Detailed telemetry for a single MLP subspace piece."""

    piece_index: int
    columns: tuple[int, ...]
    status: str
    attempt_converged: int | None
    relative_residual: float
    cosine_similarity: float
    subspace_entropy: float
    is_knot: bool
    svd_rank: int


def fit_piecewise_mlp_with_retries(
    input_activations: np.ndarray,
    target_activation_delta: np.ndarray,
    *,
    num_pieces: int = 4,
    tolerance: float = 0.35,
    svd_rank_ratio: float = 0.50,
    cosine_threshold: float = 0.50,
    base_ridge: float = 1e-6,
    retry_ridge: float = 1e-2,
    entropy_threshold: float = 0.85,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit an MLP Conv1D delta using piecewise subspace partitioning and multi-trial retry.

    Splits the target delta space into K column chunks. For each chunk c:
    - Attempt 1 (Direct Least Squares): solves min ||X D_c - Y_c||^2.
      Accepted if RelRes = ||X D_c - Y_c|| / ||Y_c|| <= tolerance.
      Status: 'applied_exact'.
    - Attempt 2 (Truncated SVD Denoising): projects Y_c onto top-k singular
      vectors to filter high-frequency noise.
      Accepted if RelRes <= tolerance.
      Status: 'applied_svd'.
    - Attempt 3 (Directional Cosine Alignment & Ridge Shrinkage): solves with
      increased regularization retry_ridge and checks cosine(X D_c, Y_c) >= cosine_threshold.
      Accepted if cosine similarity >= cosine_threshold.
      Status: 'applied_cosine'.
    - Final (SKIP): if all attempts fail, declared 'skipped_knot'. Delta is
      zeroed (Delta W_c = 0) and singular basis vectors of Y_c are preserved.

    Returns:
        tuple (delta, report_dict)
    """
    inputs = finite_array(input_activations, ndim=2, name="piecewise input activations")
    targets = finite_array(target_activation_delta, ndim=2, name="piecewise target activation delta")

    if inputs.shape[0] != targets.shape[0] or inputs.shape[0] < 1:
        raise ValueError("piecewise inputs and targets must have matching non-empty leading dimensions")
    if int(num_pieces) < 1:
        raise ValueError("num_pieces must be >= 1")
    if not (0.0 <= float(tolerance) <= 10.0):
        raise ValueError("tolerance must be positive and reasonable")
    if not (0.0 < float(svd_rank_ratio) <= 1.0):
        raise ValueError("svd_rank_ratio must be in (0, 1]")
    if not (-1.0 <= float(cosine_threshold) <= 1.0):
        raise ValueError("cosine_threshold must be in [-1, 1]")
    if not np.isfinite(base_ridge) or base_ridge < 0.0:
        raise ValueError("base_ridge must be finite and non-negative")
    if not np.isfinite(retry_ridge) or retry_ridge < 0.0:
        raise ValueError("retry_ridge must be finite and non-negative")

    n_samples, d_in = inputs.shape
    _, d_out = targets.shape

    k_pieces = min(int(num_pieces), d_out)
    col_chunks = [tuple(int(idx) for idx in chunk) for chunk in np.array_split(np.arange(d_out), k_pieces) if len(chunk) > 0]
    total_pieces = len(col_chunks)

    input_scale = max(1.0, float(np.max(np.abs(inputs))))
    x_scaled = inputs / input_scale
    gram = x_scaled.T @ x_scaled
    gram_base = gram + float(base_ridge) * np.eye(d_in, dtype=np.float64)
    gram_retry = gram + float(retry_ridge) * np.eye(d_in, dtype=np.float64)

    def _solve_ridge(gram_mat: np.ndarray, rhs_mat: np.ndarray) -> np.ndarray:
        sym = (gram_mat + gram_mat.T) / 2.0
        try:
            return np.linalg.solve(sym, rhs_mat)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(sym, rhs_mat, rcond=1e-10)[0]

    full_delta = np.zeros((d_in, d_out), dtype=np.float64)
    piece_reports: list[dict[str, Any]] = []
    knot_bases: list[np.ndarray] = []

    applied_exact_count = 0
    applied_svd_count = 0
    applied_cosine_count = 0
    skipped_knot_count = 0

    for c_idx, cols in enumerate(col_chunks):
        y_c = targets[:, cols]
        y_c_norm = float(np.linalg.norm(y_c))
        entropy_c = compute_subspace_entropy(y_c)
        is_knot_flag = entropy_c > float(entropy_threshold)

        if y_c_norm <= 1e-12:
            piece_reports.append({
                "piece_index": c_idx,
                "columns": cols,
                "status": "applied_exact",
                "attempt_converged": 1,
                "relative_residual": 0.0,
                "cosine_similarity": 1.0,
                "subspace_entropy": entropy_c,
                "is_knot": is_knot_flag,
                "svd_rank": 0,
            })
            applied_exact_count += 1
            continue

        target_c_scale = max(1.0, float(np.max(np.abs(y_c))))
        y_c_scaled = y_c / target_c_scale

        # ---------------------------------------------------------------------
        # Attempt 1: Direct Least Squares
        # ---------------------------------------------------------------------
        rhs_1 = x_scaled.T @ y_c_scaled
        scaled_d1 = _solve_ridge(gram_base, rhs_1)
        d1 = scaled_d1 * (target_c_scale / input_scale)
        pred_1 = inputs @ d1
        res_1 = float(np.linalg.norm(pred_1 - y_c))
        rel_res_1 = res_1 / max(y_c_norm, 1e-12)
        pred_1_norm = float(np.linalg.norm(pred_1))
        cos_1 = float(np.sum(pred_1 * y_c) / max(pred_1_norm * y_c_norm, 1e-12))

        if rel_res_1 <= float(tolerance):
            full_delta[:, cols] = d1
            applied_exact_count += 1
            piece_reports.append({
                "piece_index": c_idx,
                "columns": cols,
                "status": "applied_exact",
                "attempt_converged": 1,
                "relative_residual": rel_res_1,
                "cosine_similarity": cos_1,
                "subspace_entropy": entropy_c,
                "is_knot": is_knot_flag,
                "svd_rank": int(np.linalg.matrix_rank(y_c)),
            })
            continue

        # ---------------------------------------------------------------------
        # Attempt 2: Truncated SVD Denoising
        # ---------------------------------------------------------------------
        u_y, s_y, vt_y = np.linalg.svd(y_c, full_matrices=False)
        rank_y = len(s_y)
        k_svd = max(1, int(round(float(svd_rank_ratio) * rank_y)))
        if rank_y > 1 and k_svd >= rank_y:
            k_svd = rank_y - 1

        y_denoised = u_y[:, :k_svd] @ np.diag(s_y[:k_svd]) @ vt_y[:k_svd, :]
        y_denoised_norm = float(np.linalg.norm(y_denoised))
        target_denoised_scale = max(1.0, float(np.max(np.abs(y_denoised))))
        y_denoised_scaled = y_denoised / target_denoised_scale

        rhs_2 = x_scaled.T @ y_denoised_scaled
        scaled_d2 = _solve_ridge(gram_base, rhs_2)
        d2 = scaled_d2 * (target_denoised_scale / input_scale)
        pred_2 = inputs @ d2
        res_2_denoised = float(np.linalg.norm(pred_2 - y_denoised))
        rel_res_2 = res_2_denoised / max(y_denoised_norm, 1e-12)
        pred_2_norm = float(np.linalg.norm(pred_2))
        cos_2 = float(np.sum(pred_2 * y_c) / max(pred_2_norm * y_c_norm, 1e-12))

        if rel_res_2 <= float(tolerance) or (float(np.linalg.norm(pred_2 - y_c)) / max(y_c_norm, 1e-12) <= float(tolerance)):
            full_delta[:, cols] = d2
            applied_svd_count += 1
            piece_reports.append({
                "piece_index": c_idx,
                "columns": cols,
                "status": "applied_svd",
                "attempt_converged": 2,
                "relative_residual": rel_res_2,
                "cosine_similarity": cos_2,
                "subspace_entropy": entropy_c,
                "is_knot": is_knot_flag,
                "svd_rank": k_svd,
            })
            continue

        # ---------------------------------------------------------------------
        # Attempt 3: Directional Cosine Alignment & Ridge Shrinkage
        # ---------------------------------------------------------------------
        rhs_3 = x_scaled.T @ y_c_scaled
        scaled_d3 = _solve_ridge(gram_retry, rhs_3)
        d3 = scaled_d3 * (target_c_scale / input_scale)
        pred_3 = inputs @ d3
        pred_3_norm = float(np.linalg.norm(pred_3))
        res_3 = float(np.linalg.norm(pred_3 - y_c))
        rel_res_3 = res_3 / max(y_c_norm, 1e-12)
        cos_3 = float(np.sum(pred_3 * y_c) / max(pred_3_norm * y_c_norm, 1e-12))

        if cos_3 >= float(cosine_threshold):
            full_delta[:, cols] = d3
            applied_cosine_count += 1
            piece_reports.append({
                "piece_index": c_idx,
                "columns": cols,
                "status": "applied_cosine",
                "attempt_converged": 3,
                "relative_residual": rel_res_3,
                "cosine_similarity": cos_3,
                "subspace_entropy": entropy_c,
                "is_knot": is_knot_flag,
                "svd_rank": rank_y,
            })
            continue

        # ---------------------------------------------------------------------
        # Final: SKIP (Unresolvable Knot)
        # ---------------------------------------------------------------------
        full_delta[:, cols] = 0.0
        skipped_knot_count += 1

        # Preserve singular basis of the unresolvable knot
        if len(vt_y) > 0:
            top_v = vt_y[0, :]
            v_norm = float(np.linalg.norm(top_v))
            if v_norm > 1e-12:
                top_v = top_v / v_norm
            v_full = np.zeros(d_out, dtype=np.float64)
            v_full[list(cols)] = top_v
            knot_bases.append(v_full)

        piece_reports.append({
            "piece_index": c_idx,
            "columns": cols,
            "status": "skipped_knot",
            "attempt_converged": None,
            "relative_residual": rel_res_1,
            "cosine_similarity": cos_3,
            "subspace_entropy": entropy_c,
            "is_knot": True,
            "svd_rank": rank_y,
        })

    # Overall fit diagnostics
    pred_full = inputs @ full_delta
    target_norm_full = float(np.linalg.norm(targets))
    pred_norm_full = float(np.linalg.norm(pred_full))
    res_norm_full = float(np.linalg.norm(pred_full - targets))
    rel_res_full = res_norm_full / max(target_norm_full, 1e-12)
    cos_full = float(np.sum(pred_full * targets) / max(pred_norm_full * target_norm_full, 1e-12))

    summary: dict[str, Any] = {
        "backend": "piecewise_mlp_subspace_retry",
        "status": "applied_all" if skipped_knot_count == 0 else ("partial_applied" if skipped_knot_count < total_pieces else "all_skipped_knots"),
        "total_pieces": total_pieces,
        "applied_exact_count": applied_exact_count,
        "applied_svd_count": applied_svd_count,
        "applied_cosine_count": applied_cosine_count,
        "skipped_knot_count": skipped_knot_count,
        "knot_fraction": float(skipped_knot_count / total_pieces),
        "accepted_fraction": float((total_pieces - skipped_knot_count) / total_pieces),
        "subspace_entropies": [float(p["subspace_entropy"]) for p in piece_reports],
        "mean_subspace_entropy": float(np.mean([p["subspace_entropy"] for p in piece_reports])),
        "piece_reports": piece_reports,
        "knot_bases": knot_bases,
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "delta_shape": list(full_delta.shape),
        "delta_frobenius_norm": float(np.linalg.norm(full_delta)),
        "fit_residual_frobenius_norm": res_norm_full,
        "fit_relative_residual": rel_res_full,
        "cosine_after": cos_full,
        "target_frobenius_norm": target_norm_full,
        "predicted_target_frobenius_norm": pred_norm_full,
        "tolerance": float(tolerance),
        "svd_rank_ratio": float(svd_rank_ratio),
        "cosine_threshold": float(cosine_threshold),
        "base_ridge": float(base_ridge),
        "retry_ridge": float(retry_ridge),
        "entropy_threshold": float(entropy_threshold),
        "finite": bool(np.all(np.isfinite(full_delta))),
    }

    return full_delta, summary


def build_attention_detour_projector(
    knot_bases: Sequence[np.ndarray],
    hidden_size: int,
    eta: float = 0.08,
) -> np.ndarray:
    """Construct a soft detour projection operator bypassing tangled memory cells.

    For unit knot basis vectors v_j in R^H, the soft detour projector is:
        P_bypass = I - eta * sum_j (v_j * v_j^T)

    The singular values are strictly bounded in [1 - eta, 1.0].
    When applied to the preceding attention projection matrix W_attn:
        W_attn_new = W_attn @ P_bypass
    the attention layer smoothly detours signal away from chaotic MLP memory
    cells through the residual connection.

    If knot_bases is empty, returns the identity matrix I_H with singular values 1.0.
    """
    h_dim = int(hidden_size)
    if h_dim < 1:
        raise ValueError("hidden_size must be >= 1")
    eta_val = float(eta)
    if not (0.0 <= eta_val < 1.0):
        raise ValueError("eta must be in [0, 1)")

    if not knot_bases:
        return np.eye(h_dim, dtype=np.float64)

    # Collect and validate basis vectors
    valid_vectors: list[np.ndarray] = []
    for vec in knot_bases:
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if v.size != h_dim or not np.all(np.isfinite(v)):
            continue
        v_norm = float(np.linalg.norm(v))
        if v_norm > 1e-12:
            valid_vectors.append(v / v_norm)

    if not valid_vectors:
        return np.eye(h_dim, dtype=np.float64)

    # Orthonormalize the knot subspace using SVD to form an exact orthogonal projector
    v_matrix = np.column_stack(valid_vectors)
    u_k, s_k, _ = np.linalg.svd(v_matrix, full_matrices=False)
    knot_rank_mask = s_k > 1e-6
    if not np.any(knot_rank_mask):
        return np.eye(h_dim, dtype=np.float64)

    q_basis = u_k[:, knot_rank_mask]
    knot_projector = q_basis @ q_basis.T

    # Soft detour operator: P = I - eta * P_knot
    p_detour = np.eye(h_dim, dtype=np.float64) - eta_val * knot_projector

    # Strictly enforce singular value bounds in [1 - eta, 1.0]
    u_p, s_p, vt_p = np.linalg.svd(p_detour)
    min_sv = max(0.0, 1.0 - eta_val)
    max_sv = 1.0
    s_bounded = np.clip(s_p, min_sv, max_sv)
    p_bounded = u_p @ np.diag(s_bounded) @ vt_p
    p_symmetric = (p_bounded + p_bounded.T) / 2.0

    return p_symmetric
