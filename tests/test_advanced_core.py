"""Tests for advanced transfer vectors: memory imprinting, vocabulary alignment & flow distillation."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
import torch
import torch.nn as nn

from faytuna_flow.adaptive_transfer import AdaptiveTransferPolicy
from faytuna_flow.nonlinear_transfer import align_vocabulary_head, rank_one_memory_imprint
from faytuna_flow.solver import ConstrainedCorrection
from faytuna_flow.transformer_core import (
    GPT2TensorLiftMapping,
    build_transformer_surgery_plan,
    run_flow_guided_distillation,
)


def test_rank_one_memory_imprint_recall_and_nullspace() -> None:
    """Verify Vector 5: exact associative memory update along keys and zero disturbance on nullspace."""
    rng = np.random.default_rng(42)
    d_model, d_inner = 32, 128
    w_down = rng.standard_normal((d_model, d_inner)) * 0.05

    # 3 target keys and target residual adjustments
    keys = rng.standard_normal((3, d_inner))
    target_values = rng.standard_normal((3, d_model)) * 0.1

    new_w = rank_one_memory_imprint(
        w_down, keys, target_values, ridge=1e-4, gain=1.0, max_relative_norm=0.1
    )

    assert new_w.shape == w_down.shape
    assert np.all(np.isfinite(new_w))

    # Test 1: Nullspace preservation
    # Any vector orthogonal to all keys should experience ZERO shift (w_down @ z == new_w @ z)
    q, _ = np.linalg.qr(keys.T)
    # The nullspace basis starts from index 3 to 127
    nullspace_basis = q[:, 3:]  # (128, 125)
    random_null = nullspace_basis @ rng.standard_normal((nullspace_basis.shape[1], 5))  # (128, 5)

    base_null_resp = w_down @ random_null
    new_null_resp = new_w @ random_null
    np.testing.assert_allclose(new_null_resp, base_null_resp, atol=1e-12)

    # Test 2: Key direction response shifts toward target values
    # For a key k_0, (new_w - w_down) @ k_0 should have strong positive projection onto target_values[0]
    delta_w = new_w - w_down
    assert np.linalg.norm(delta_w) > 0.0
    k0 = keys[0]
    shift_response = delta_w @ k0
    cosine = np.dot(shift_response, target_values[0]) / (
        np.linalg.norm(shift_response) * np.linalg.norm(target_values[0]) + 1e-12
    )
    assert cosine > 0.50, f"Expected associative alignment > 0.5, got {cosine}"


def test_align_vocabulary_head_geometry() -> None:
    """Verify Vector 4: vocabulary logit alignment projection and Frobenius norm bounding."""
    rng = np.random.default_rng(42)
    vocab_size = 500
    d_student = 32
    d_teacher = 64

    w_s = rng.standard_normal((vocab_size, d_student)) * 0.02
    w_t = rng.standard_normal((vocab_size, d_teacher)) * 0.02
    # Orthogonal chart projector
    q, _ = np.linalg.qr(rng.standard_normal((d_teacher, d_student)))
    projector = q[:, :d_student]  # (64, 32)

    aligned = align_vocabulary_head(w_s, w_t, projector, gain=0.05, max_relative_norm=0.02)

    assert aligned.shape == (vocab_size, d_student)
    assert np.all(np.isfinite(aligned))

    # Norm change must respect max_relative_norm (0.02)
    rel_change = np.linalg.norm(aligned - w_s) / np.linalg.norm(w_s)
    assert rel_change <= 0.02 + 1e-6
    assert rel_change > 0.0

    # Test mismatch error handling
    with pytest.raises(ValueError, match="vocabulary size mismatch"):
        align_vocabulary_head(w_s, rng.standard_normal((600, d_teacher)), projector)


class TinyStudentLM(nn.Module):
    """Minimal PyTorch transformer-like module for testing autograd flow distillation."""

    def __init__(self, vocab: int = 100, hidden: int = 32) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab, hidden)
        self.layer = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = True) -> Any:
        h0 = self.wte(input_ids)
        h1 = torch.relu(self.layer(h0))
        logits = self.lm_head(h1)
        if output_hidden_states:
            return type("Out", (), {"logits": logits, "hidden_states": (h0, h1)})()
        return logits


class TinyTeacherLM(nn.Module):
    """Minimal PyTorch teacher module."""

    def __init__(self, vocab: int = 100, hidden: int = 64) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab, hidden)
        self.layer = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = True) -> Any:
        h0 = self.wte(input_ids)
        h1 = torch.relu(self.layer(h0))
        logits = self.lm_head(h1)
        if output_hidden_states:
            return type("Out", (), {"logits": logits, "hidden_states": (h0, h1)})()
        return logits


def test_run_flow_guided_distillation_convergence() -> None:
    """Verify Vector 2: flow-guided micro-distillation decreases loss without exploding drift."""
    torch.manual_seed(42)
    student = TinyStudentLM(vocab=50, hidden=16)
    teacher = TinyTeacherLM(vocab=50, hidden=32)

    # Synthetic chart projector: (d_T=32, d_S=16)
    chart = np.random.randn(32, 16).astype(np.float32)

    # Single fixed batch of token IDs for deterministic gradient descent verification
    batches = [{"input_ids": torch.randint(0, 50, (2, 8))}]

    report = run_flow_guided_distillation(
        student,
        teacher,
        batches,
        alignment_chart=chart,
        steps=15,
        lr=1e-2,
        lambda_flow=0.5,
        lambda_ce=1.0,
        device="cpu",
        freeze_embeddings=True,
    )

    assert report["status"] == "completed"
    assert report["steps"] == 15
    assert np.isfinite(report["initial_loss"])
    assert np.isfinite(report["final_loss"])
    assert report["final_loss"] < report["initial_loss"]
    assert report["relative_parameter_drift"] < 0.20
    assert len(report["loss_history"]) == 15


def test_adaptive_policy_advanced_configuration() -> None:
    """Verify AdaptiveTransferPolicy exposes and validates advanced transfer fields."""
    policy = AdaptiveTransferPolicy(
        mode="safe",
        calibrate_lm_head=True,
        memory_imprint_mlp=True,
        flow_distill_steps=25,
    )
    assert policy.calibrate_lm_head is True
    assert policy.memory_imprint_mlp is True
    assert policy.flow_distill_steps == 25

    # Invalid negative steps must be rejected
    with pytest.raises(ValueError, match="flow_distill_steps must be non-negative"):
        AdaptiveTransferPolicy(flow_distill_steps=-1)


def test_transformer_surgery_plan_advanced_vectors() -> None:
    """Verify build_transformer_surgery_plan applies memory imprinting and vocabulary alignment."""
    rng = np.random.default_rng(42)
    v_size, d_s, d_t = 200, 16, 32

    # Mock Conv1D / Linear weights for student
    weights = {
        "h.0.mlp.c_proj.weight": rng.standard_normal((64, 16)).astype(np.float32),
        "lm_head.weight": rng.standard_normal((v_size, d_s)).astype(np.float32),
    }
    teacher_head = rng.standard_normal((v_size, d_t)).astype(np.float32)
    chart_proj = rng.standard_normal((d_t, d_s)).astype(np.float32)

    correction = ConstrainedCorrection(
        matrices=np.zeros((1, 16, 16)),
        biases=np.zeros((1, 16)),
        accepted=np.array([True]),
        reasons={},
        confidence=np.array([1.0]),
        diagnostics={},
        quadratic_terms=None,
    )

    mapping = [
        GPT2TensorLiftMapping(0, "h.0.mlp.c_proj.weight", side="output", block_index=0),
    ]

    # Mock activation inputs and target deltas for memory imprinting
    activation_inputs = {
        "h.0.mlp.c_proj.weight": rng.standard_normal((10, 64)).astype(np.float32),
    }
    activation_target_deltas = {
        "h.0.mlp.c_proj.weight": rng.standard_normal((10, 16)).astype(np.float32) * 0.05,
    }

    plan = build_transformer_surgery_plan(
        weights,
        correction,
        chart_projection=chart_proj,
        sequence_length=8,
        hidden_size=d_s,
        mapping=mapping,
        gain=0.5,
        mode="apply",
        activation_inputs=activation_inputs,
        activation_target_deltas=activation_target_deltas,
        calibrate_lm_head=True,
        teacher_lm_head=teacher_head,
        lm_head_gain=0.04,
        memory_imprint_mlp=True,
    )

    assert "h.0.mlp.c_proj.weight" in plan.applied_tensors
    assert "lm_head.weight" in plan.applied_tensors
    assert "vocabulary_head_alignment:lm_head.weight" in plan.metadata["lift_reports"]
    assert "activation_ls:h.0.mlp.c_proj.weight" in plan.metadata["lift_reports"]
    rep = plan.metadata["lift_reports"]["activation_ls:h.0.mlp.c_proj.weight"]
    assert rep.get("method") == "rank_one_memory_imprint"
