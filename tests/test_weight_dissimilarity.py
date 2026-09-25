"""Unit tests for static weight dissimilarity and pilot flow pulse bound."""

from __future__ import annotations

import numpy as np
import pytest

from faytuna_flow.geometry import (
    compute_frobenius_dissimilarity,
    dissimilarity_gain_schedule,
    project_teacher_weight,
)
from faytuna_flow.gpt2 import (
    compute_gpt2_static_weight_dissimilarity,
    estimate_pilot_pulse_gain_bound,
    gpt2_depth_gain_weight,
)


def test_project_teacher_weight_hidden_to_hidden():
    dt, ds = 16, 8
    np.random.seed(42)
    wt = np.random.randn(dt, dt)
    proj = np.random.randn(dt, ds)

    w_proj = project_teacher_weight(wt, proj, site="hidden_to_hidden")
    assert w_proj.shape == (ds, ds)
    assert np.all(np.isfinite(w_proj))

    # Orthogonal projection property: (U.T @ W @ U)
    u, _, _ = np.linalg.svd(proj, full_matrices=False)
    u = u[:, :ds]
    expected = u.T @ wt @ u
    np.testing.assert_allclose(w_proj, expected, atol=1e-10)


def test_project_teacher_weight_mlp_to_hidden():
    dt, ds = 16, 8
    np.random.seed(42)
    wt = np.random.randn(4 * dt, dt)
    proj = np.random.randn(dt, ds)

    w_proj = project_teacher_weight(wt, proj, site="mlp_to_hidden")
    assert w_proj.shape == (4 * ds, ds)
    assert np.all(np.isfinite(w_proj))


def test_project_teacher_weight_invalid():
    with pytest.raises(ValueError, match="dt >= ds"):
        project_teacher_weight(np.eye(4), np.ones((4, 8)))

    with pytest.raises(ValueError, match="unsupported projection site"):
        project_teacher_weight(np.eye(8), np.ones((8, 4)), site="invalid_site")


def test_compute_frobenius_dissimilarity():
    np.random.seed(42)
    w = np.random.randn(8, 8)
    assert compute_frobenius_dissimilarity(w, w) == pytest.approx(0.0, abs=1e-10)

    w_scaled = 2.0 * w
    assert compute_frobenius_dissimilarity(w_scaled, w) == pytest.approx(1.0, abs=1e-10)

    # Shape mismatch error
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_frobenius_dissimilarity(w, np.ones((8, 4)))


def test_dissimilarity_gain_schedule():
    distances = [1.0, 1.2, 1.5, 2.0, 0.8]
    schedule = dissimilarity_gain_schedule(distances, min_gain=0.3, max_gain=1.8)
    assert len(schedule) == 5
    assert all(0.3 <= g <= 1.8 for g in schedule)
    # Highest distance must yield highest gain
    assert schedule[3] == max(schedule)
    # Lowest distance must yield lowest gain
    assert schedule[4] == min(schedule)

    # Zero distance edge case returns 1.0
    zero_sched = dissimilarity_gain_schedule([0.0, 0.0, 0.0])
    assert zero_sched == (1.0, 1.0, 1.0)


def test_gpt2_depth_gain_weight_flexible():
    # Sequence/vector support
    vec = (0.5, 0.8, 1.2, 1.5)
    assert gpt2_depth_gain_weight(0, 4, schedule=vec) == 0.5
    assert gpt2_depth_gain_weight(2, 4, schedule=vec) == 1.2
    assert gpt2_depth_gain_weight(10, 4, schedule=vec) == 1.0  # out-of-bounds fallback

    # Mapping/dict support
    mapping = {0: 0.4, "1": 0.9}
    assert gpt2_depth_gain_weight(0, schedule=mapping) == 0.4
    assert gpt2_depth_gain_weight(1, schedule=mapping) == 0.9

    # Standard named schedules
    assert 0.0 < gpt2_depth_gain_weight(6, 12, schedule="boost_deep") <= 2.0
    assert 0.0 < gpt2_depth_gain_weight(6, 12, schedule="sine") <= 2.0


def test_estimate_pilot_pulse_gain_bound():
    inputs = {
        "layer_0": np.ones((16, 768), dtype=np.float64),
        "layer_1": np.ones((16, 768), dtype=np.float64),
    }
    targets = {
        "layer_0": 0.1 * np.ones((16, 768), dtype=np.float64),
        "layer_1": 0.1 * np.ones((16, 768), dtype=np.float64),
    }

    result = estimate_pilot_pulse_gain_bound(inputs, targets, target_logit_shift=0.01)
    assert "safe_gain_bound" in result
    assert "recommended_gains" in result
    assert result["safe_gain_bound"] > 0
    assert result["relative_shift"] == pytest.approx(0.1, rel=1e-5)
    assert result["safe_gain_bound"] == pytest.approx(0.1, rel=1e-5)
    assert len(result["recommended_gains"]) == 4


def test_compute_gpt2_static_weight_dissimilarity_mock():
    np.random.seed(17)
    dt, ds = 32, 16
    student_dict = {}
    teacher_dict = {}
    for i in range(4):
        student_dict[f"h.{i}.attn.c_proj.weight"] = np.random.randn(ds, ds)
    for i in range(16):
        teacher_dict[f"h.{i}.attn.c_proj.weight"] = np.random.randn(dt, dt)

    proj = np.random.randn(dt, ds)
    res = compute_gpt2_static_weight_dissimilarity(
        student_dict,
        teacher_dict,
        proj,
        student_blocks=4,
        teacher_blocks=16,
        teacher_dim=dt,
        student_dim=ds,
    )

    assert len(res["distances"]) == 4
    assert len(res["schedule"]) == 4
    assert all(0.25 <= g <= 2.0 for g in res["schedule"])
    assert res["mean_distance"] > 0

