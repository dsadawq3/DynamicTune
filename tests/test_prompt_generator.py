"""Unit tests for the advanced calibration prompt generator and rank diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from faytuna_flow.prompt_generator import (
    CalibrationPromptEngine,
    compute_spectral_effective_rank,
    compute_spectral_entropy,
)


def test_baseline_prompts_integrity() -> None:
    engine = CalibrationPromptEngine(seed=42)
    baseline = engine.get_curated_baseline_prompts()
    assert len(baseline) >= 15
    for p in baseline:
        assert isinstance(p, str)
        assert len(p.strip()) > 30


def test_dual_superposition_prompts_tension() -> None:
    engine = CalibrationPromptEngine(seed=42)
    duals = engine.get_dual_superposition_prompts()
    assert len(duals) >= 8
    for d in duals:
        assert isinstance(d, str)
        # Verify dual structure has multi-component tension
        assert "\n" in d or "Compare" in d or "Contrast" in d or "Сопоставьте" in d


def test_generate_calibration_suite_scaling() -> None:
    engine = CalibrationPromptEngine(seed=42)
    for target_count in [10, 47, 64, 100]:
        suite = engine.generate_calibration_suite(num_prompts=target_count)
        assert len(suite) == target_count
        ids = [s["id"] for s in suite]
        assert len(set(ids)) == target_count, "IDs must be unique across the suite"
        for item in suite:
            assert "category" in item
            assert "text" in item
            assert len(item["text"].strip()) > 20
            assert "strategy" in item


def test_save_and_load_calibration_dataset(tmp_path: Path) -> None:
    engine = CalibrationPromptEngine(seed=42)
    suite = engine.generate_calibration_suite(num_prompts=32)
    json_file = tmp_path / "test_calibration.json"

    saved_path = engine.save_calibration_dataset(json_file, suite)
    assert saved_path.exists()

    loaded_prompts = CalibrationPromptEngine.load_calibration_dataset(json_file)
    assert len(loaded_prompts) == 32
    assert loaded_prompts[0] == suite[0]["text"]
    assert loaded_prompts[-1] == suite[-1]["text"]


def test_spectral_effective_rank_properties() -> None:
    # 1. Rank-1 matrix must have effective rank very close to 1.0
    u = np.ones((100, 1))
    v = np.ones((1, 50))
    rank_1 = u @ v
    eff_rank_1 = compute_spectral_effective_rank(rank_1)
    assert 0.99 <= eff_rank_1 <= 1.05

    # 2. Identity matrix of size D must have maximal effective rank D
    d = 32
    identity = np.eye(d)
    eff_rank_id = compute_spectral_effective_rank(identity)
    assert abs(eff_rank_id - d) < 1e-4

    # 3. Spectral entropy on rank-1 must be 0, on identity must be 1.0
    h_rank1 = compute_spectral_entropy(rank_1)
    assert abs(h_rank1) < 1e-4

    h_id = compute_spectral_entropy(identity)
    assert abs(h_id - 1.0) < 1e-4

    # 4. Empty / zero matrix guards
    zero_mat = np.zeros((10, 10))
    assert compute_spectral_effective_rank(zero_mat) == 1.0
    assert compute_spectral_entropy(zero_mat) == 0.0
