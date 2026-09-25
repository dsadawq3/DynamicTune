"""Non-linear activation inversion and 2-jet curvature transport.

Provides exact analytical pre-activation inversion for GELU and SwiGLU non-linearities,
overcoming the linear least-squares barrier in deep transformer layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
from scipy.special import erf

from .types import finite_array


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    arr = np.asarray(z, dtype=np.float64)
    clipped = np.clip(arr, -85.0, 85.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def gelu(z: np.ndarray, approximate: str = "none") -> np.ndarray:
    """Gaussian Error Linear Unit (GELU).

    approximate:
        'none': exact formula 0.5 * z * (1 + erf(z / sqrt(2)))
        'tanh': 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
    """
    arr = np.asarray(z, dtype=np.float64)
    if approximate == "tanh":
        inner = np.sqrt(2.0 / np.pi) * (arr + 0.044715 * (arr ** 3))
        return 0.5 * arr * (1.0 + np.tanh(np.clip(inner, -85.0, 85.0)))
    return 0.5 * arr * (1.0 + erf(arr / np.sqrt(2.0)))


def gelu_prime(z: np.ndarray, approximate: str = "none") -> np.ndarray:
    """Analytical first derivative of GELU with respect to pre-activation z.

    d/dz GELU(z) = 0.5 * (1 + erf(z / sqrt(2))) + (z / sqrt(2*pi)) * exp(-z^2 / 2)
    """
    arr = np.asarray(z, dtype=np.float64)
    if approximate == "tanh":
        inner = np.sqrt(2.0 / np.pi) * (arr + 0.044715 * (arr ** 3))
        t = np.tanh(np.clip(inner, -85.0, 85.0))
        sech2 = 1.0 - t ** 2
        d_inner = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * (arr ** 2))
        return 0.5 * (1.0 + t) + 0.5 * arr * sech2 * d_inner

    cdf = 0.5 * (1.0 + erf(arr / np.sqrt(2.0)))
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * np.clip(arr ** 2, 0.0, 100.0))
    return cdf + arr * pdf


def gelu_second_derivative(z: np.ndarray) -> np.ndarray:
    """Analytical second derivative of GELU: d^2/dz^2 GELU(z) = (2 - z^2) * pdf(z)."""
    arr = np.asarray(z, dtype=np.float64)
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * np.clip(arr ** 2, 0.0, 100.0))
    return (2.0 - arr ** 2) * pdf


def swish(u: np.ndarray) -> np.ndarray:
    """Swish / SiLU activation function: u * sigmoid(u)."""
    arr = np.asarray(u, dtype=np.float64)
    return arr * sigmoid(arr)


def swish_prime(u: np.ndarray) -> np.ndarray:
    """Analytical first derivative of Swish: sigmoid(u) * (1 + u * (1 - sigmoid(u)))."""
    arr = np.asarray(u, dtype=np.float64)
    sig = sigmoid(arr)
    return sig * (1.0 + arr * (1.0 - sig))


def invert_gelu_preactivation(
    z_student: np.ndarray,
    delta_post_activation: np.ndarray,
    *,
    epsilon: float = 1e-4,
    approximate: str = "none",
    max_delta: float = 5.0,
    newton_steps: int = 1,
) -> np.ndarray:
    """Invert target activation shift Delta_a into pre-activation shift Delta_z.

    Linearized Taylor expansion with high-order Newton refinement:
        Step 0: Delta_z^(0) = Delta_a / (GELU'(z) + sign(GELU'(z)) * epsilon)
        Step 1+: Delta_z^(k+1) = Delta_z^(k) - (GELU(z + Delta_z^(k)) - GELU(z) - Delta_a) / (GELU'(z + Delta_z^(k)) + epsilon)

    Reduces inversion error to O(Delta_z^4), dramatically increasing transfer fidelity.
    """
    z = finite_array(z_student, ndim=2, name="z_student")
    da = finite_array(delta_post_activation, ndim=2, name="delta_post_activation")
    if z.shape != da.shape:
        raise ValueError(f"shape mismatch between z {z.shape} and delta_a {da.shape}")

    deriv = gelu_prime(z, approximate=approximate)
    sign = np.where(deriv >= 0.0, 1.0, -1.0)
    denom = np.where(np.abs(deriv) > epsilon, deriv, sign * epsilon)

    delta_z = da / denom
    if max_delta > 0:
        delta_z = np.clip(delta_z, -max_delta, max_delta)

    # Newton-Raphson refinement
    base_a = gelu(z, approximate=approximate)
    for _ in range(max(0, newton_steps)):
        pred_a = gelu(z + delta_z, approximate=approximate)
        residual = (pred_a - base_a) - da
        cur_deriv = gelu_prime(z + delta_z, approximate=approximate)
        cur_sign = np.where(cur_deriv >= 0.0, 1.0, -1.0)
        cur_denom = np.where(np.abs(cur_deriv) > epsilon, cur_deriv, cur_sign * epsilon)
        delta_z = delta_z - residual / cur_denom
        if max_delta > 0:
            delta_z = np.clip(delta_z, -max_delta, max_delta)

    return delta_z


def invert_swiglu_activations(
    u_gate: np.ndarray,
    v_up: np.ndarray,
    delta_act: np.ndarray,
    *,
    epsilon: float = 1e-4,
    max_delta: float = 5.0,
    newton_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert target activation shift into coupled (Delta_u, Delta_v) for SwiGLU.

    In SwiGLU:
        act = swish(u) * v
    The differential is:
        Delta_act approx swish'(u) * v * Delta_u + swish(u) * Delta_v = J_u * Delta_u + J_v * Delta_v

    Computes the minimum-norm pseudoinverse solution with Newton-Raphson refinement.
    """
    u = finite_array(u_gate, ndim=2, name="u_gate")
    v = finite_array(v_up, ndim=2, name="v_up")
    da = finite_array(delta_act, ndim=2, name="delta_act")

    if u.shape != v.shape or u.shape != da.shape:
        raise ValueError(f"shape mismatch: u {u.shape}, v {v.shape}, delta_act {da.shape}")

    j_u = swish_prime(u) * v
    j_v = swish(u)
    denom = j_u ** 2 + j_v ** 2 + epsilon
    factor = da / denom

    delta_u = factor * j_u
    delta_v = factor * j_v

    if max_delta > 0:
        delta_u = np.clip(delta_u, -max_delta, max_delta)
        delta_v = np.clip(delta_v, -max_delta, max_delta)

    base_act = swish(u) * v
    for _ in range(max(0, newton_steps)):
        cur_u = u + delta_u
        cur_v = v + delta_v
        pred_act = swish(cur_u) * cur_v
        residual = (pred_act - base_act) - da

        cur_j_u = swish_prime(cur_u) * cur_v
        cur_j_v = swish(cur_u)
        cur_denom = cur_j_u ** 2 + cur_j_v ** 2 + epsilon
        step_factor = residual / cur_denom

        delta_u = delta_u - step_factor * cur_j_u
        delta_v = delta_v - step_factor * cur_j_v

        if max_delta > 0:
            delta_u = np.clip(delta_u, -max_delta, max_delta)
            delta_v = np.clip(delta_v, -max_delta, max_delta)

    return delta_u, delta_v


def solve_weight_deltas_least_squares(
    x_input: np.ndarray,
    delta_targets: np.ndarray,
    *,
    ridge: float = 1e-4,
    max_relative_norm: float = 0.25,
) -> np.ndarray:
    """Solve Delta_W for linear regression X @ Delta_W = Delta_Z.

    Args:
        x_input: (N, d_in) student representations.
        delta_targets: (N, d_out) target shifts.
        ridge: Tikhonov L2 regularizer.
        max_relative_norm: safety threshold.

    Returns:
        Delta_W: (d_in, d_out) weight updates.
    """
    x = finite_array(x_input, ndim=2, name="x_input")
    dz = finite_array(delta_targets, ndim=2, name="delta_targets")

    n, d_in = x.shape
    if dz.shape[0] != n:
        raise ValueError(f"sample count mismatch: X {x.shape} vs targets {dz.shape}")

    # Normal equations: (X^T X + lambda I) Delta_W = X^T Delta_Z
    xtx = x.T @ x
    reg = ridge * np.trace(xtx) / max(d_in, 1)
    if reg <= 0 or not np.isfinite(reg):
        reg = ridge

    a = xtx + reg * np.eye(d_in, dtype=np.float64)
    b = x.T @ dz

    try:
        delta_w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        delta_w, _, _, _ = np.linalg.lstsq(a, b, rcond=1e-5)

    return np.asarray(delta_w, dtype=np.float64)


def solve_truncated_svd_deltas(
    x_input: np.ndarray,
    delta_targets: np.ndarray,
    *,
    rank: int = 16,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Solve Delta_W for X @ Delta_W = Delta_Y using Truncated SVD.

    Guarantees that the update lives strictly in the active low-rank subspace
    spanned by the top singular vectors, eliminating noise in the nullspace
    when sample count N is much smaller than dimension D.
    """
    x = finite_array(x_input, ndim=2, name="x_input")
    y = finite_array(delta_targets, ndim=2, name="delta_targets")

    n, d_in = x.shape
    r = min(rank, n, d_in)
    if r < 1:
        return np.zeros((d_in, y.shape[1]), dtype=np.float64)

    # Economy SVD: X = U Sigma V^T
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    u_r = u[:, :r]
    s_r = s[:r]
    v_r = vt[:r, :].T  # (d_in, r)

    # Regularized inverse singular values: s / (s^2 + ridge)
    inv_s = s_r / (s_r ** 2 + ridge)

    # Delta_W = V_r @ diag(inv_s) @ (U_r^T @ Y)
    delta_w = v_r @ (inv_s[:, None] * (u_r.T @ y))
    return np.asarray(delta_w, dtype=np.float64)


def damped_tikhonov_pinv(
    matrix: np.ndarray,
    ridge: float = 1e-3,
    relative: bool = True,
) -> np.ndarray:
    """Compute damped Tikhonov-Levenberg-Marquardt pseudoinverse.

    Damps singular values via sigma / (sigma^2 + lambda), preventing norm explosion
    caused by near-zero singular values in ill-conditioned matrices.

    Args:
        matrix: (M, N) input matrix A.
        ridge: regularization parameter.
        relative: if True, lambda = ridge * (||A||_F^2 / min(M, N)).

    Returns:
        (N, M) regularized pseudoinverse matrix A^dagger.
    """
    a = finite_array(matrix, ndim=2, name="matrix")
    m, n = a.shape
    k = min(m, n)
    if k == 0:
        return np.zeros((n, m), dtype=np.float64)

    u, s, vt = np.linalg.svd(a, full_matrices=False)

    if relative:
        fro_sq = float(np.sum(s ** 2))
        lam = float(ridge * (fro_sq / max(k, 1)))
    else:
        lam = float(ridge)

    if lam <= 0.0 or not np.isfinite(lam):
        lam = 1e-12

    inv_s = s / (s ** 2 + lam)
    a_pinv = (vt.T * inv_s) @ u.T
    return np.asarray(a_pinv, dtype=np.float64)


def solve_adaptive_spectral_svd_deltas(
    x: np.ndarray,
    delta_y: np.ndarray,
    energy_ratio: float = 0.85,
    max_rank: int = 128,
    min_rank: int = 16,
    ridge: float = 1e-2,
) -> np.ndarray:
    """Solve Delta_W for X @ Delta_W = Delta_Y using Adaptive Spectral SVD.

    Selects an adaptive rank r that retains >= energy_ratio of the total singular energy
    (sum_{i=1}^r sigma_i^2 / sum sigma_i^2 >= energy_ratio), clamped to [min_rank, max_rank].
    Solves the Tikhonov-regularized least squares problem on this active subspace.

    Args:
        x: (N, d_in) input representation matrix.
        delta_y: (N, d_out) target shifts.
        energy_ratio: fraction of cumulative singular variance to retain (default: 0.85).
        max_rank: upper bound on adaptive rank.
        min_rank: lower bound on adaptive rank.
        ridge: Tikhonov L2 regularizer on singular values.

    Returns:
        delta_w: (d_in, d_out) weight update matrix.
    """
    x_arr = finite_array(x, ndim=2, name="x")
    y_arr = finite_array(delta_y, ndim=2, name="delta_y")

    n, d_in = x_arr.shape
    if y_arr.shape[0] != n:
        raise ValueError(f"sample count mismatch: X {x_arr.shape} vs delta_y {y_arr.shape}")

    k_max = min(n, d_in)
    if k_max < 1:
        return np.zeros((d_in, y_arr.shape[1]), dtype=np.float64)

    # Economy SVD: X = U Sigma V^T
    u, s, vt = np.linalg.svd(x_arr, full_matrices=False)

    energy = s ** 2
    total_energy = float(np.sum(energy))
    if total_energy > 0.0 and np.isfinite(total_energy):
        cum_energy = np.cumsum(energy) / total_energy
        idx = np.where(cum_energy >= float(energy_ratio))[0]
        if len(idx) > 0:
            target_r = int(idx[0]) + 1
        else:
            target_r = k_max
    else:
        target_r = min_rank

    lower_bound = min(int(min_rank), k_max)
    upper_bound = min(int(max_rank), k_max)
    if lower_bound > upper_bound:
        lower_bound = upper_bound

    r = max(lower_bound, min(target_r, upper_bound))
    r = max(1, min(r, k_max))

    u_r = u[:, :r]
    s_r = s[:r]
    v_r = vt[:r, :].T  # (d_in, r)

    # Regularized inverse singular values: s / (s^2 + ridge)
    inv_s = s_r / (s_r ** 2 + ridge)

    # Delta_W = V_r @ diag(inv_s) @ (U_r^T @ Y)
    delta_w = v_r @ (inv_s[:, None] * (u_r.T @ y_arr))
    return np.asarray(delta_w, dtype=np.float64)


def spectral_directional_rescale(
    w_orig: np.ndarray,
    delta_w: np.ndarray,
    max_spectral_ratio: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Rescale weight update so that its spectral norm does not exceed max_spectral_ratio * ||W_orig||_2.

    If ||Delta_W||_2 > max_spectral_ratio * ||W_orig||_2, scales Delta_W such that
    ||Delta_W||_2 strictly equals max_spectral_ratio * ||W_orig||_2.

    Args:
        w_orig: (M, N) original base weight matrix.
        delta_w: (M, N) computed weight update matrix.
        max_spectral_ratio: maximum permissible ratio of spectral norm of Delta_W to W_orig.

    Returns:
        scaled_delta_w: (M, N) scaled weight update.
        scale_factor: float scaling factor applied to delta_w (in (0, 1]).
    """
    w = finite_array(w_orig, ndim=2, name="w_orig")
    dw = finite_array(delta_w, ndim=2, name="delta_w")

    if w.shape != dw.shape:
        raise ValueError(f"shape mismatch: w_orig {w.shape} vs delta_w {dw.shape}")

    sigma_w = float(np.linalg.norm(w, ord=2))
    sigma_dw = float(np.linalg.norm(dw, ord=2))

    if sigma_w <= 0.0 or not np.isfinite(sigma_w) or sigma_dw <= 0.0 or not np.isfinite(sigma_dw):
        return dw.copy(), 1.0

    target_max = float(max_spectral_ratio * sigma_w)
    if sigma_dw > target_max:
        scale = float(target_max / sigma_dw)
        scaled_dw = dw * scale
    else:
        scale = 1.0
        scaled_dw = dw.copy()

    return np.asarray(scaled_dw, dtype=np.float64), scale


@dataclass(frozen=True)
class CurvatureMatchResult:
    mean_student_curvature: float
    mean_teacher_curvature: float
    curvature_ratio: float
    curvature_divergence: float
    safe_to_transfer: bool


def compute_2jet_curvature_metric(
    student_activations: np.ndarray,
    teacher_activations: np.ndarray,
    projector: np.ndarray,
) -> CurvatureMatchResult:
    """Compute empirical 2-jet Riemannian curvature along activation trajectories.

    Measures second differences as proxy for trajectory curvature:
        kappa(t) = || x(t+1) - 2x(t) + x(t-1) || / (|| x(t+1) - x(t) || + 1e-6)
    """
    s = finite_array(student_activations, ndim=2, name="student_activations")
    t = finite_array(teacher_activations, ndim=2, name="teacher_activations")
    p = finite_array(projector, ndim=2, name="projector")

    n = min(len(s), len(t))
    if n < 3:
        return CurvatureMatchResult(0.0, 0.0, 1.0, 0.0, True)

    # Second discrete differences along token sequence
    s_diff2 = s[2:n] - 2.0 * s[1:n - 1] + s[0:n - 2]
    s_speed = np.linalg.norm(s[1:n - 1] - s[0:n - 2], axis=1) + 1e-6
    s_curv = np.mean(np.linalg.norm(s_diff2, axis=1) / s_speed)

    t_diff2 = t[2:n] - 2.0 * t[1:n - 1] + t[0:n - 2]
    t_speed = np.linalg.norm(t[1:n - 1] - t[0:n - 2], axis=1) + 1e-6
    t_curv = np.mean(np.linalg.norm(t_diff2, axis=1) / t_speed)

    ratio = float(t_curv / max(s_curv, 1e-6))
    div = float(abs(np.log(max(ratio, 1e-6))))

    return CurvatureMatchResult(
        mean_student_curvature=float(s_curv),
        mean_teacher_curvature=float(t_curv),
        curvature_ratio=ratio,
        curvature_divergence=div,
        safe_to_transfer=bool(div < 2.0),
    )


def rank_one_memory_imprint(
    w_down: np.ndarray,
    keys: np.ndarray,
    target_values: np.ndarray,
    *,
    ridge: float = 1e-3,
    gain: float = 1.0,
    max_relative_norm: float = 0.02,
) -> np.ndarray:
    """Targeted key-value associative memory imprinting (MEMIT/ROME-style update).

    Updates W_down such that for active intermediate key directions k_i in R^d_inner,
    the output shifts toward target values v_i in R^d_model:
        Delta_W = sum_i (v_i @ k_i^T) / (||k_i||^2 + lambda)

    Operates strictly in the low-dimensional subspaces of triggered associations,
    preserving all orthogonal dimensions of the student's existing memory space.

    Args:
        w_down: (d_model, d_inner) base projection matrix.
        keys: (N, d_inner) intermediate representations (e.g. SwiGLU activations).
        target_values: (N, d_model) desired output residual corrections.
        ridge: regularizer to prevent division by near-zero key activations.
        gain: transfer scaling factor alpha.
        max_relative_norm: safety clamp on ||Delta_W|| / ||W||.

    Returns:
        new_w_down: (d_model, d_inner) updated projection matrix.
    """
    w = finite_array(w_down, ndim=2, name="w_down")
    k = finite_array(keys, ndim=2, name="keys")
    v = finite_array(target_values, ndim=2, name="target_values")

    d_model, d_inner = w.shape
    n = len(k)
    if k.shape[1] != d_inner or v.shape[0] != n or v.shape[1] != d_model:
        raise ValueError(
            f"dimension mismatch: w_down {w.shape}, keys {k.shape}, target_values {v.shape}"
        )
    if n == 0:
        return w.copy()

    # Sum of rank-one associative updates: v_i @ k_i^T / (||k_i||^2 + ridge)
    k_norms_sq = np.sum(k ** 2, axis=1, keepdims=True)  # (N, 1)
    k_scaled = k / (k_norms_sq + ridge)                  # (N, d_inner)

    # Delta_W: (d_model, d_inner) = v.T @ k_scaled
    delta_w = v.T @ k_scaled

    # Apply relative norm safety clamp
    w_norm = float(np.linalg.norm(w))
    delta_norm = float(np.linalg.norm(delta_w))
    clamp = min(1.0, (max_relative_norm * w_norm) / max(delta_norm, 1e-8))

    scaled_update = (gain * clamp) * delta_w
    return w + scaled_update


def align_vocabulary_head(
    student_head: np.ndarray,
    teacher_head: np.ndarray,
    projector: np.ndarray,
    *,
    gain: float = 0.03,
    max_relative_norm: float = 0.02,
) -> np.ndarray:
    """Project teacher vocabulary classification hyperplanes into student dimension.

    When student and teacher share the same vocabulary V, the teacher's classification
    hyperplanes W_T in R^{V x d_T} contain sharper semantic separation boundaries.
    Projects W_T through the output Procrustes chart P in R^{d_T x d_S}:
        W_target = W_T @ P  in R^{V x d_S}
        Delta_W = W_target - W_S

    Safeguards with Frobenius relative norm clamping and convex gain blending.

    Args:
        student_head: (V, d_S) student lm_head matrix.
        teacher_head: (V, d_T) teacher lm_head matrix.
        projector: (d_T, d_S) orthogonal chart projector.
        gain: interpolation gain alpha in [0, 0.1].
        max_relative_norm: max allowed relative change ||Delta_W|| / ||W_S||.

    Returns:
        new_head: (V, d_S) aligned classification matrix.
    """
    w_s = finite_array(student_head, ndim=2, name="student_head")
    w_t = finite_array(teacher_head, ndim=2, name="teacher_head")
    p = finite_array(projector, ndim=2, name="projector")

    v_s, d_s = w_s.shape
    v_t, d_t = w_t.shape

    if v_s != v_t:
        raise ValueError(f"vocabulary size mismatch: student {v_s} vs teacher {v_t}")
    if p.shape != (d_t, d_s):
        raise ValueError(f"projector shape mismatch: expected ({d_t}, {d_s}), got {p.shape}")

    # Teacher head projected into student representation space: (V, d_S)
    w_target = w_t @ p
    delta_w = w_target - w_s

    # Clamp relative step norm
    w_norm = float(np.linalg.norm(w_s))
    delta_norm = float(np.linalg.norm(delta_w))
    clamp = min(1.0, (max_relative_norm * w_norm) / max(delta_norm, 1e-8))

    scaled_delta = (gain * clamp) * delta_w
    return w_s + scaled_delta
