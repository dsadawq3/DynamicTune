"""Unit tests for non-linear pre-activation inversion, 2-jet curvature, and multi-chart manifold transport."""

from __future__ import annotations

import numpy as np
import pytest

from faytuna_flow.nonlinear_transfer import (
    compute_2jet_curvature_metric,
    damped_tikhonov_pinv,
    gelu,
    gelu_prime,
    invert_gelu_preactivation,
    invert_swiglu_activations,
    solve_adaptive_spectral_svd_deltas,
    solve_weight_deltas_least_squares,
    spectral_directional_rescale,
    swish,
    swish_prime,
)
from faytuna_flow.manifold_charts import (
    build_multi_chart_atlas,
    compute_grassmannian_geodesic,
    spherical_kmeans,
)


def test_gelu_derivatives_vs_finite_differences():
    """Verify analytical GELU derivatives match numerical central differences."""
    z = np.linspace(-3.0, 3.0, 50).reshape(10, 5)
    h = 1e-6

    for mode in ("none", "tanh"):
        analytical = gelu_prime(z, approximate=mode)
        numerical = (gelu(z + h, approximate=mode) - gelu(z - h, approximate=mode)) / (2.0 * h)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-5)


def test_invert_gelu_preactivation():
    """Verify pre-activation inversion accurately recovers small input perturbations."""
    rng = np.random.RandomState(42)
    z0 = rng.randn(20, 16)
    delta_z_true = 0.05 * rng.randn(20, 16)

    a0 = gelu(z0)
    a1 = gelu(z0 + delta_z_true)
    delta_a = a1 - a0

    delta_z_est = invert_gelu_preactivation(z0, delta_a, epsilon=1e-4)

    # For non-zero gradient regimes, reconstructed delta_z should match delta_z_true closely
    active = np.abs(gelu_prime(z0)) > 0.05
    assert np.mean(active) > 0.5
    np.testing.assert_allclose(delta_z_est[active], delta_z_true[active], rtol=0.10, atol=0.01)


def test_swish_derivative_vs_finite_differences():
    """Verify Swish derivative matches central differences."""
    u = np.linspace(-4.0, 4.0, 40).reshape(8, 5)
    h = 1e-6
    analytical = swish_prime(u)
    numerical = (swish(u + h) - swish(u - h)) / (2.0 * h)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-5)


def test_invert_swiglu_activations():
    """Verify coupled minimum-norm SwiGLU inversion produces valid forward delta."""
    rng = np.random.RandomState(42)
    u0 = rng.randn(25, 32)
    v0 = rng.randn(25, 32)

    delta_u_true = 0.02 * rng.randn(25, 32)
    delta_v_true = 0.02 * rng.randn(25, 32)

    act0 = swish(u0) * v0
    act1 = swish(u0 + delta_u_true) * (v0 + delta_v_true)
    delta_act = act1 - act0

    du_est, dv_est = invert_swiglu_activations(u0, v0, delta_act, epsilon=1e-4)

    # Reconstructed forward activation shift should match delta_act
    j_u = swish_prime(u0) * v0
    j_v = swish(u0)
    delta_act_reconstructed = j_u * du_est + j_v * dv_est

    # Relative error should be small where Jacobian norm is non-degenerate
    j_norm_sq = j_u ** 2 + j_v ** 2
    valid = j_norm_sq > 0.01
    np.testing.assert_allclose(
        delta_act_reconstructed[valid],
        delta_act[valid],
        rtol=0.08,
        atol=0.01,
    )


def test_solve_weight_deltas_least_squares():
    """Verify least-squares solver recovers true weight deltas on linear systems."""
    rng = np.random.RandomState(42)
    x = rng.randn(100, 32)
    delta_w_true = rng.randn(32, 16) * 0.1
    delta_z = x @ delta_w_true + 1e-4 * rng.randn(100, 16)

    recovered = solve_weight_deltas_least_squares(x, delta_z, ridge=1e-6)
    np.testing.assert_allclose(recovered, delta_w_true, rtol=0.05, atol=0.01)


def test_spherical_kmeans():
    """Verify spherical k-means produces unit-length centroids and valid labels."""
    rng = np.random.RandomState(42)
    data = rng.randn(120, 24)
    centroids, labels = spherical_kmeans(data, n_clusters=4, random_state=42)

    assert len(centroids) == 4
    assert len(labels) == 120
    assert set(np.unique(labels)).issubset({0, 1, 2, 3})

    # Centroids must be on unit sphere
    norms = np.linalg.norm(centroids, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_multi_chart_atlas():
    """Verify multi-chart atlas partition-of-unity and smooth projection."""
    rng = np.random.RandomState(42)
    x = rng.randn(150, 32)
    # Teacher is higher-dimensional
    p_true = rng.randn(64, 32)
    y = x @ p_true.T + 0.05 * rng.randn(150, 64)

    atlas = build_multi_chart_atlas(x, y, n_charts=3, temperature=3.0, random_state=42)
    assert atlas.n_charts == 3
    assert atlas.student_dim == 32
    assert atlas.teacher_dim == 64
    assert len(atlas.projectors) == 3

    # Check partition of unity
    weights = atlas.compute_weights(x)
    assert weights.shape == (150, 3)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, atol=1e-6)
    assert np.all(weights >= 0.0)

    # Lift student to teacher
    lifted = atlas.lift_student_to_teacher(x)
    assert lifted.shape == (150, 64)
    assert np.all(np.isfinite(lifted))

    # Project back to student
    projected = atlas.project_teacher_to_student(lifted, reference_student_x=x)
    assert projected.shape == (150, 32)
    assert np.all(np.isfinite(projected))


def test_grassmannian_geodesic():
    """Verify Grassmannian geodesic maintains orthonormality and endpoints."""
    rng = np.random.RandomState(42)
    d_amb = 24
    d_sub = 6

    u1, _ = np.linalg.qr(rng.randn(d_amb, d_sub))
    u2, _ = np.linalg.qr(rng.randn(d_amb, d_sub))

    geodesic = compute_grassmannian_geodesic(u1, u2)
    assert len(geodesic.principal_angles) == d_sub
    assert np.all(geodesic.principal_angles >= 0.0)
    assert np.all(geodesic.principal_angles <= np.pi / 2.0 + 1e-6)

    # Test t = 0 matches span(u1)
    g0 = geodesic.interpolate(0.0)
    proj_diff_0 = np.linalg.norm(g0 @ g0.T - u1 @ u1.T)
    assert proj_diff_0 < 1e-4

    # Test t = 1 matches span(u2)
    g1 = geodesic.interpolate(1.0)
    proj_diff_1 = np.linalg.norm(g1 @ g1.T - u2 @ u2.T)
    assert proj_diff_1 < 1e-4

    # Test intermediate point maintains strict orthonormality
    g_half = geodesic.interpolate(0.5)
    np.testing.assert_allclose(g_half.T @ g_half, np.eye(d_sub), atol=1e-5)


def test_2jet_curvature_metric():
    """Verify 2-jet curvature metric handles sample trajectories correctly."""
    rng = np.random.RandomState(42)
    s = rng.randn(30, 16)
    t = rng.randn(30, 32)
    p = rng.randn(32, 16)

    curv = compute_2jet_curvature_metric(s, t, p)
    assert curv.mean_student_curvature > 0.0
    assert curv.mean_teacher_curvature > 0.0
    assert curv.curvature_ratio > 0.0
    assert isinstance(curv.safe_to_transfer, bool)


def test_damped_tikhonov_pinv_stability():
    """Verify damped Tikhonov pseudoinverse prevents norm explosion on ill-conditioned matrices."""
    rng = np.random.RandomState(42)
    m, n = 30, 20
    u, _ = np.linalg.qr(rng.randn(m, n))
    v, _ = np.linalg.qr(rng.randn(n, n))

    # Ill-conditioned spectrum: dominant singular values + near-zero singular values (1e-6)
    s = np.concatenate([np.linspace(5.0, 1.0, 15), np.full(5, 1e-6)])
    a = u @ np.diag(s) @ v.T

    # Standard pseudoinverse inverts 1e-6 to 1e6, exploding the norm
    pinv_std = np.linalg.pinv(a)
    norm_std = float(np.linalg.norm(pinv_std))
    assert norm_std > 1e5

    # Damped Tikhonov pseudoinverse dampens sigma / (sigma^2 + lambda)
    pinv_damped = damped_tikhonov_pinv(a, ridge=1e-3, relative=True)
    norm_damped = float(np.linalg.norm(pinv_damped))

    # Norm must remain well-conditioned (< 10.0), orders of magnitude smaller than unregularized pinv
    assert norm_damped < 10.0
    assert norm_damped < norm_std * 1e-4

    # On dominant singular directions, damped pinv accurately approximates pseudo-inverse
    dominant_proj = v[:, :15] @ v[:, :15].T
    reconstructed_dom = a @ (pinv_damped @ a)
    a_dom = a @ dominant_proj
    np.testing.assert_allclose(reconstructed_dom @ dominant_proj, a_dom, rtol=0.05, atol=1e-3)


def test_solve_adaptive_spectral_svd_energy_coverage():
    """Verify adaptive SVD solver retains >= 85% energy and captures active subspace."""
    rng = np.random.RandomState(42)
    n, d_in, d_out = 120, 60, 50

    # Generate synthetic representations X with decaying spectrum
    u, _ = np.linalg.qr(rng.randn(n, d_in))
    v, _ = np.linalg.qr(rng.randn(d_in, d_in))

    i_idx = np.arange(d_in, dtype=np.float64)
    s = 10.0 / ((i_idx + 1.0) ** 0.6)
    cum_energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    expected_r_85 = int(np.where(cum_energy >= 0.85)[0][0]) + 1

    x = u @ np.diag(s) @ v.T

    # Generate target shift aligned with representation + noise
    w_true = rng.randn(d_in, d_out) * 0.1
    delta_y = x @ w_true + 1e-4 * rng.randn(n, d_out)

    # Solve with adaptive spectral SVD
    delta_w = solve_adaptive_spectral_svd_deltas(
        x,
        delta_y,
        energy_ratio=0.85,
        min_rank=1,
        max_rank=50,
        ridge=1e-4,
    )

    # Effective rank of delta_w must match the rank required for 85% energy
    sv_w = np.linalg.svd(delta_w, compute_uv=False)
    effective_rank = int(np.sum(sv_w > 1e-6))
    assert effective_rank == expected_r_85

    # Check forward reconstruction quality on active subspace
    y_pred = x @ delta_w
    energy_pred = np.sum(y_pred ** 2)
    energy_target = np.sum((x @ w_true) ** 2)
    assert energy_pred / energy_target >= 0.80

    # Higher energy ratio (0.95) must adaptively select higher rank
    expected_r_95 = int(np.where(cum_energy >= 0.95)[0][0]) + 1
    assert expected_r_95 > expected_r_85
    delta_w_95 = solve_adaptive_spectral_svd_deltas(
        x,
        delta_y,
        energy_ratio=0.95,
        min_rank=1,
        max_rank=50,
        ridge=1e-4,
    )
    sv_w_95 = np.linalg.svd(delta_w_95, compute_uv=False)
    assert int(np.sum(sv_w_95 > 1e-6)) == expected_r_95


def test_spectral_directional_rescale_bounds():
    """Verify spectral norm of delta is strictly bounded by max_spectral_ratio * ||W_orig||_2."""
    rng = np.random.RandomState(42)
    m, n = 40, 30

    # Case 1: Delta exceeds spectral threshold -> must be strictly scaled down
    w_orig = rng.randn(m, n)
    delta_w_large = rng.randn(m, n) * 10.0

    sigma_orig = float(np.linalg.norm(w_orig, ord=2))
    sigma_delta_pre = float(np.linalg.norm(delta_w_large, ord=2))
    max_ratio = 0.05
    target_sigma = max_ratio * sigma_orig

    assert sigma_delta_pre > target_sigma

    scaled_dw, scale = spectral_directional_rescale(w_orig, delta_w_large, max_spectral_ratio=max_ratio)
    sigma_delta_post = float(np.linalg.norm(scaled_dw, ord=2))

    # Must strictly equal max_spectral_ratio * sigma_orig
    np.testing.assert_allclose(sigma_delta_post, target_sigma, rtol=1e-5, atol=1e-6)
    assert 0.0 < scale < 1.0
    np.testing.assert_allclose(scale, target_sigma / sigma_delta_pre, rtol=1e-5)

    # Case 2: Delta is within spectral threshold -> must NOT be altered
    delta_w_small = rng.randn(m, n) * (0.01 * sigma_orig / np.linalg.norm(rng.randn(m, n), ord=2))
    scaled_dw_small, scale_small = spectral_directional_rescale(w_orig, delta_w_small, max_spectral_ratio=max_ratio)

    assert scale_small == 1.0
    np.testing.assert_allclose(scaled_dw_small, delta_w_small, rtol=1e-6, atol=1e-8)
    assert np.linalg.norm(scaled_dw_small, ord=2) <= target_sigma

