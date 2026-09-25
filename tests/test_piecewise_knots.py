from __future__ import annotations

import argparse
import numpy as np
import pytest

from faytuna_flow.knots import (
    PieceReport,
    build_attention_detour_projector,
    compute_subspace_entropy,
    fit_piecewise_mlp_with_retries,
    is_knot_subspace,
)
from faytuna_flow.gpt2 import GPT2TensorLiftMapping, build_gpt2_surgery_plan
from faytuna_flow.solver import ConstrainedCorrection
from scripts.run_auto_tune import _parser


def test_compute_subspace_entropy_rank1():
    """Rank-1 matrix has singular value distribution concentrated on 1 value (entropy ~ 0)."""
    rng = np.random.default_rng(42)
    u = rng.normal(size=(64, 1))
    v = rng.normal(size=(1, 32))
    x_rank1 = u @ v
    entropy = compute_subspace_entropy(x_rank1)
    assert 0.0 <= entropy <= 0.05
    assert not is_knot_subspace(x_rank1, threshold=0.85)


def test_compute_subspace_entropy_isotropic_chaos():
    """Isotropic orthogonal/random matrix has high entropy (> 0.85, representing a knot)."""
    rng = np.random.default_rng(123)
    # High-dimensional random Gaussian matrix has nearly flat singular value spectrum
    x_chaos = rng.normal(size=(64, 64))
    entropy = compute_subspace_entropy(x_chaos)
    assert entropy > 0.85
    assert is_knot_subspace(x_chaos, threshold=0.85)


def test_compute_subspace_entropy_edge_cases():
    """Handles 1D, empty, zero, and non-finite inputs safely."""
    assert compute_subspace_entropy(None) == 0.0
    assert compute_subspace_entropy(np.array([])) == 0.0
    assert compute_subspace_entropy(np.zeros((10, 10))) == 0.0
    assert compute_subspace_entropy(np.ones(10)) == 0.0
    assert compute_subspace_entropy(np.array([[np.nan, 1.0], [0.0, 2.0]])) == 0.0
    assert compute_subspace_entropy(np.array([[np.inf, 1.0], [0.0, 2.0]])) == 0.0


def test_piecewise_mlp_attempt_1_exact():
    """Clean linear mapping converges on Attempt 1 (Direct Least Squares)."""
    rng = np.random.default_rng(77)
    n_samples, d_in, d_out = 128, 64, 32
    x = rng.normal(size=(n_samples, d_in))
    planted_delta = rng.normal(scale=0.1, size=(d_in, d_out))
    y = x @ planted_delta

    delta, report = fit_piecewise_mlp_with_retries(
        x,
        y,
        num_pieces=4,
        tolerance=0.35,
        base_ridge=1e-6,
    )

    assert report["backend"] == "piecewise_mlp_subspace_retry"
    assert report["status"] == "applied_all"
    assert report["applied_exact_count"] == 4
    assert report["applied_svd_count"] == 0
    assert report["applied_cosine_count"] == 0
    assert report["skipped_knot_count"] == 0
    assert np.allclose(delta, planted_delta, atol=1e-4)
    assert report["fit_relative_residual"] < 1e-4
    assert report["cosine_after"] > 0.999


def test_piecewise_mlp_attempt_2_svd_denoising():
    """Noisy target where Attempt 1 fails tolerance but Attempt 2 (Truncated SVD) converges."""
    rng = np.random.default_rng(99)
    n_samples, d_in, d_out = 100, 10, 8
    u_true = rng.normal(size=(n_samples, 2))
    v_true = rng.normal(size=(2, d_out))
    clean_target = u_true @ v_true

    x = u_true @ rng.normal(size=(2, d_in)) + 0.001 * rng.normal(size=(n_samples, d_in))
    noise = rng.normal(scale=0.8, size=(n_samples, d_out))
    noisy_target = clean_target + noise

    delta, report = fit_piecewise_mlp_with_retries(
        x,
        noisy_target,
        num_pieces=1,
        tolerance=0.35,
        svd_rank_ratio=0.25,
        cosine_threshold=0.99,  # make attempt 3 fail so attempt 2 is isolated
        base_ridge=1e-6,
    )

    assert report["applied_svd_count"] == 1
    assert report["applied_exact_count"] == 0
    assert report["skipped_knot_count"] == 0
    assert report["delta_shape"] == [d_in, d_out]


def test_piecewise_mlp_attempt_3_cosine_alignment():
    """Target where Attempt 1 & 2 fail residual tolerance, but Attempt 3 (Cosine Alignment) converges."""
    rng = np.random.default_rng(101)
    n_samples, d_in, d_out = 100, 30, 10
    x = rng.normal(size=(n_samples, d_in))

    # Target that correlates positively (cosine ~ 0.65) but has large variance causing residual > 0.35
    w_plant = rng.normal(size=(d_in, d_out))
    base = x @ w_plant
    # Substantial orthogonal noise to fail relative residual tolerance 0.20
    noise = rng.normal(scale=1.5 * np.std(base), size=(n_samples, d_out))
    target = base + noise

    delta, report = fit_piecewise_mlp_with_retries(
        x,
        target,
        num_pieces=2,
        tolerance=0.15,  # strict tolerance so attempt 1 and 2 fail
        cosine_threshold=0.45,  # attainable for attempt 3
        retry_ridge=1e-1,
    )

    assert report["applied_cosine_count"] > 0
    assert report["skipped_knot_count"] == 0
    assert report["status"] == "applied_all"


def test_piecewise_mlp_final_skip_unresolvable_knot():
    """Independent isotropic noise cannot be solved by any attempt; skipped as knot."""
    rng = np.random.default_rng(2024)
    n_samples, d_in, d_out = 100, 20, 8
    # X and Y are completely independent random variables
    x = rng.normal(size=(n_samples, d_in))
    y_knot = rng.normal(size=(n_samples, d_out))

    # With very strict tolerance and high cosine threshold, all attempts must fail
    delta, report = fit_piecewise_mlp_with_retries(
        x,
        y_knot,
        num_pieces=2,
        tolerance=0.01,
        cosine_threshold=0.99,
        base_ridge=1e-4,
    )

    assert report["status"] == "all_skipped_knots"
    assert report["skipped_knot_count"] == 2
    assert report["applied_exact_count"] == 0
    assert report["applied_svd_count"] == 0
    assert report["applied_cosine_count"] == 0
    # Crucial rule: delta is zeroed out to prevent knowledge degradation
    assert np.all(delta == 0.0)
    # Knot bases must be populated with unit vectors
    assert len(report["knot_bases"]) == 2
    for basis in report["knot_bases"]:
        assert basis.shape == (d_out,)
        assert np.isclose(np.linalg.norm(basis), 1.0, atol=1e-5)


def test_build_attention_detour_projector():
    """Constructs detour projector with singular values strictly bounded in [1 - eta, 1.0]."""
    hidden_size = 32
    eta = 0.08

    # Empty knot basis -> returns exact identity
    p_identity = build_attention_detour_projector([], hidden_size, eta=eta)
    assert np.allclose(p_identity, np.eye(hidden_size))
    svs_id = np.linalg.svd(p_identity, compute_uv=False)
    assert np.allclose(svs_id, 1.0)

    # Populate with 2 knot basis vectors
    rng = np.random.default_rng(55)
    v1 = rng.normal(size=hidden_size)
    v1 /= np.linalg.norm(v1)
    v2 = rng.normal(size=hidden_size)
    v2 /= np.linalg.norm(v2)

    p_detour = build_attention_detour_projector([v1, v2], hidden_size, eta=eta)
    assert p_detour.shape == (hidden_size, hidden_size)
    # Must be symmetric
    assert np.allclose(p_detour, p_detour.T)

    # Singular values must be strictly in [1 - eta, 1.0] -> [0.92, 1.0]
    svs = np.linalg.svd(p_detour, compute_uv=False)
    assert np.all(svs >= (1.0 - eta - 1e-9))
    assert np.all(svs <= (1.0 + 1e-9))
    assert np.isclose(np.min(svs), 1.0 - eta, atol=1e-4)
    assert np.isclose(np.max(svs), 1.0, atol=1e-4)

    # Signal along knot basis direction v1 is attenuated by (1 - eta)
    v1_detoured = v1 @ p_detour
    assert np.isclose(np.linalg.norm(v1_detoured), 1.0 - eta, atol=1e-3)


def test_gpt2_surgery_plan_with_piecewise_knots_and_detour():
    """End-to-end integration: build_gpt2_surgery_plan with piecewise_knots=True."""
    hidden = 8
    weights = {
        "transformer.h.0.attn.c_proj.weight": np.eye(hidden, dtype=np.float32),
        "transformer.h.0.mlp.c_proj.weight": np.ones((4 * hidden, hidden), dtype=np.float32),
    }
    mapping = [
        GPT2TensorLiftMapping(0, "transformer.h.0.mlp.c_proj.weight", side="output", block_index=0),
    ]

    correction = ConstrainedCorrection(
        matrices=np.eye(hidden)[None, :, :] * 0.01,
        biases=np.zeros((1, hidden)),
        accepted=np.array([True]),
        reasons={},
        confidence=np.array([0.9]),
        diagnostics={},
    )

    rng = np.random.default_rng(888)
    n_samples = 40
    mlp_in = rng.normal(size=(n_samples, 4 * hidden))

    # Unresolvable knot target for MLP
    mlp_target = rng.normal(size=(n_samples, hidden))

    plan = build_gpt2_surgery_plan(
        weights,
        correction,
        chart_projection=None,
        sequence_length=4,
        hidden_size=hidden,
        mapping=mapping,
        mode="apply",
        activation_inputs={"transformer.h.0.mlp.c_proj.weight": mlp_in},
        activation_target_deltas={"transformer.h.0.mlp.c_proj.weight": mlp_target},
        activation_ridge=1e-5,
        piecewise_knots=True,
        piecewise_num_pieces=2,
        piecewise_tolerance=0.001,  # force knot skip
        piecewise_cosine_threshold=0.99,  # force knot skip
        piecewise_eta=0.08,
    )

    assert plan.metadata["piecewise_knots"] is True
    summary = plan.metadata["piecewise_knots_summary"]
    assert summary is not None
    assert summary["enabled"] is True
    assert summary["total_knots_skipped"] == 2
    assert "transformer.h.0.attn.c_proj.weight" in plan.applied_tensors

    # Attention matrix was updated by detour projector!
    attn_orig = weights["transformer.h.0.attn.c_proj.weight"]
    attn_updated = plan.updates["transformer.h.0.attn.c_proj.weight"]
    assert not np.array_equal(attn_orig, attn_updated)

    # Singular values of the updated attention matrix reflect the detour [0.92, 1.0]
    svs_attn = np.linalg.svd(attn_updated, compute_uv=False)
    assert np.isclose(np.min(svs_attn), 0.92, atol=1e-3)
    assert np.isclose(np.max(svs_attn), 1.0, atol=1e-3)


def test_run_auto_tune_cli_flag():
    """Verify --piecewise-knots CLI flag is present and parsed correctly."""
    parser = _parser()
    args = parser.parse_args([
        "--checkpoint-dir", "dummy_dir",
        "--student-trace", "dummy_s.npz",
        "--teacher-trace", "dummy_t.npz",
        "--alignment", "dummy_align.json",
        "--piecewise-knots",
    ])
    assert args.piecewise_knots is True

    # Default should be False
    args_default = parser.parse_args([
        "--checkpoint-dir", "dummy_dir",
        "--student-trace", "dummy_s.npz",
        "--teacher-trace", "dummy_t.npz",
        "--alignment", "dummy_align.json",
    ])
    assert args_default.piecewise_knots is False


def test_piecewise_mlp_all_four_outcomes_in_single_fit():
    """Simultaneously exercise exact, svd, cosine, and skipped knot across 4 chunks."""
    rng = np.random.default_rng(42)
    n_samples, d_in = 120, 20
    x = rng.normal(size=(n_samples, d_in))

    # Chunk 0: clean linear mapping -> applied_exact
    d0 = rng.normal(size=(d_in, 4))
    y0 = x @ d0

    # Chunk 1: low-rank true signal + orthogonal noise -> applied_svd
    u = rng.normal(size=(d_in, 1))
    v = rng.normal(size=(1, 4))
    clean1 = x @ (u @ v)
    noise1 = rng.normal(size=(n_samples, 4))
    noise1_ortho, _ = np.linalg.qr(noise1 - clean1 @ np.linalg.lstsq(clean1, noise1, rcond=None)[0])
    y1 = clean1 + 0.50 * np.linalg.norm(clean1) / np.linalg.norm(noise1_ortho) * noise1_ortho

    # Chunk 2: correlated signal with high variance -> applied_cosine
    d2 = rng.normal(size=(d_in, 4))
    y2 = x @ d2 + rng.normal(scale=1.5 * np.std(x @ d2), size=(n_samples, 4))

    # Chunk 3: independent isotropic noise -> skipped_knot
    y3 = rng.normal(size=(n_samples, 4))

    y_full = np.concatenate([y0, y1, y2, y3], axis=1)

    delta, report = fit_piecewise_mlp_with_retries(
        x,
        y_full,
        num_pieces=4,
        tolerance=0.35,
        svd_rank_ratio=0.25,
        cosine_threshold=0.45,
        base_ridge=1e-6,
        retry_ridge=0.1,
    )

    assert report["total_pieces"] == 4
    assert report["applied_exact_count"] == 1
    assert report["applied_svd_count"] == 1
    assert report["applied_cosine_count"] == 1
    assert report["skipped_knot_count"] == 1

    statuses = [p["status"] for p in report["piece_reports"]]
    assert statuses == ["applied_exact", "applied_svd", "applied_cosine", "skipped_knot"]

    # Chunk 3 delta is zeroed
    assert np.all(delta[:, 12:16] == 0.0)
    # Chunks 0, 1, 2 deltas are non-zero
    assert np.linalg.norm(delta[:, 0:4]) > 0.0
    assert np.linalg.norm(delta[:, 4:8]) > 0.0
    assert np.linalg.norm(delta[:, 8:12]) > 0.0

    # Knot basis for chunk 3 is recorded
    assert len(report["knot_bases"]) == 1
    kbasis = report["knot_bases"][0]
    assert kbasis.shape == (16,)
    assert np.isclose(np.linalg.norm(kbasis), 1.0)
    assert np.all(kbasis[:12] == 0.0)
    assert np.linalg.norm(kbasis[12:16]) > 0.999


def test_build_attention_detour_projector_varying_eta():
    """Validates detour projector singular value bounds across a range of eta values."""
    hidden_size = 16
    rng = np.random.default_rng(33)
    bases = [rng.normal(size=hidden_size) for _ in range(3)]

    for eta in [0.01, 0.05, 0.08, 0.15, 0.25, 0.50]:
        p = build_attention_detour_projector(bases, hidden_size, eta=eta)
        svs = np.linalg.svd(p, compute_uv=False)
        assert np.all(svs >= 1.0 - eta - 1e-9)
        assert np.all(svs <= 1.0 + 1e-9)
        assert np.isclose(np.min(svs), 1.0 - eta, atol=1e-4)
        assert np.isclose(np.max(svs), 1.0, atol=1e-4)

