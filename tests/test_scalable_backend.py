from pathlib import Path

import numpy as np
import pytest

from faytuna_flow.artifacts import load_alignment, load_flow, save_alignment, save_flow
from faytuna_flow.geometry import fit_alignment
from faytuna_flow.flow import FlowSample, fit_flow_operator, fit_flow_transfer
import faytuna_flow.flow as flow_module
from faytuna_flow.jets import _fit_local_jet
from faytuna_flow.scalable import dense_memory_estimate, requires_scalable_backend
from faytuna_flow.synthetic import LinearResidualSystem
from faytuna_flow.types import AlignmentResult, Probe, TrajectoryTrace, TransitionObservation


def _traces(dim: int, *, layers: int = 2, count: int = 8, seed: int = 41, model_id: str) -> list[TrajectoryTrace]:
    rng = np.random.default_rng(seed)
    coordinates = np.linspace(0.0, 1.0, layers + 1)
    matrices = np.stack([(-0.08 + 0.002 * i) * np.eye(dim) for i in range(layers)])
    traces = []
    for index in range(count):
        initial = rng.normal(0.0, 0.2, size=dim)
        states = [initial]
        transitions = []
        for layer in range(layers):
            ds = coordinates[layer + 1] - coordinates[layer]
            delta = ds * (states[-1] @ matrices[layer])
            target = states[-1] + delta
            transitions.append(TransitionObservation(layer, layer + 1, coordinates[layer], coordinates[layer + 1], states[-1], target, delta, delta / ds))
            states.append(target)
        traces.append(TrajectoryTrace(model_id, f"p{index}", tuple([-1] + list(range(layers))), coordinates, np.asarray(states), None, tuple(transitions), {"probe_split": "train"}))
    return traces


def _elementwise_traces(dim: int, *, layers: int = 2, count: int = 4, seed: int = 61, model_id: str) -> list[TrajectoryTrace]:
    rng = np.random.default_rng(seed)
    coordinates = np.linspace(0.0, 1.0, layers + 1)
    traces = []
    for index in range(count):
        initial = rng.normal(0.0, 0.03, size=dim)
        states = [initial]
        transitions = []
        for layer in range(layers):
            ds = coordinates[layer + 1] - coordinates[layer]
            field = (-0.06 + 0.003 * layer) * states[-1] + 0.001 * np.sin(states[-1])
            delta = ds * field
            target = states[-1] + delta
            transitions.append(TransitionObservation(layer, layer + 1, coordinates[layer], coordinates[layer + 1], states[-1], target, delta, field))
            states.append(target)
        traces.append(TrajectoryTrace(model_id, f"p{index}", tuple([-1] + list(range(layers))), coordinates, np.asarray(states), None, tuple(transitions), {"probe_split": "train"}))
    return traces


def test_gpt2_like_memory_estimate_is_integer_only_and_forces_scalable_backend():
    estimate = dense_memory_estimate(16 * 1600, nodes=48, quadratic=True)
    assert estimate["state_dim"] == 25600
    assert estimate["linear_feature_count"] == 25600
    assert estimate["quadratic_feature_count"] == 25600**2
    assert estimate["quadratic_operator_elements"] == 25600**3
    assert estimate["total_bytes"] > 10**12
    assert requires_scalable_backend(25600, max_dense_features=5_000_000, quadratic=True)
    sample = FlowSample(np.zeros(25600), np.zeros(25600), 0.0)
    with pytest.raises(ValueError, match="dense flow operator fit is prohibited"):
        fit_flow_operator([], observations=[sample], coordinates=np.asarray([0.0]), quadratic=True)


def test_non_quadratic_local_jet_does_not_allocate_dense_cubic_tensor_for_gpt2_chart():
    dimension = 16 * 96
    rng = np.random.default_rng(191)
    states = rng.normal(0.0, 0.01, size=(3, dimension))
    velocities = rng.normal(0.0, 0.01, size=(3, dimension))
    center, jacobian, quadratic_tensor, hessian, residual, uncertainty, basis = _fit_local_jet(
        states,
        velocities,
        np.ones(3),
        rank=4,
        quadratic=False,
        ridge=1e-4,
        robust_loss="huber",
        robust_iterations=1,
    )
    assert center.shape == (dimension,)
    assert jacobian.shape == (dimension, dimension)
    assert quadratic_tensor is None
    assert hessian.shape == (4, dimension)
    assert np.all(np.isfinite(hessian))
    assert np.allclose(hessian, 0.0)
    assert np.isfinite(residual) and np.isfinite(uncertainty)
    assert basis.shape[0] == dimension


def test_non_quadratic_flow_operator_keeps_quadratic_terms_absent_at_gpt2_dimension(monkeypatch):
    dimension = 1536
    samples = [FlowSample(np.zeros(dimension), np.zeros(dimension), 0.0) for _ in range(2)]

    def fake_weighted_ridge(local_samples, ridge, **kwargs):
        assert not kwargs["quadratic"]
        return np.eye(dimension), np.zeros(dimension), 0.0, None

    monkeypatch.setattr(flow_module, "_weighted_ridge_operator", fake_weighted_ridge)
    operator = fit_flow_operator(
        [],
        observations=samples,
        coordinates=np.asarray([0.0]),
        quadratic=False,
        min_samples_per_node=2,
    )
    assert operator.matrices.shape == (1, dimension, dimension)
    assert operator.quadratic_terms is None


def test_large_sequence_alignment_is_compressed_before_dense_teacher_student_map():
    rng = np.random.default_rng(19)
    source = rng.normal(size=(6, 16 * 1600))
    target = rng.normal(size=(6, 16 * 768))
    alignment = fit_alignment(source, target, source_role="teacher", target_role="student", scalable_rank=4, max_dense_features=5_000_000)
    assert alignment.metadata["representation"] == "randomized_latent_compression"
    assert alignment.matrix.shape == (4, 4)
    assert alignment.source_projection.shape == (25600, 4)
    assert alignment.target_projection.shape == (12288, 4)
    assert alignment.source_dim == 25600
    assert alignment.target_dim == 12288
    mapped = alignment.apply(source[:1])
    assert mapped.shape == (1, 12288)
    assert np.all(np.isfinite(mapped))
    tangent = rng.normal(size=(1, 25600))
    assert np.allclose(alignment.apply(source[:1] + tangent) - alignment.apply(source[:1]), alignment.linear_apply(tangent))


def test_gpt2_sequence_dimensions_complete_full_fit_through_scalable_backend():
    teacher = _elementwise_traces(16 * 1600, count=4, model_id="gpt2-xl")
    student = _elementwise_traces(16 * 768, count=4, seed=62, model_id="gpt2")
    alignment = fit_alignment(
        np.asarray([trace.hidden_states[0] for trace in teacher]),
        np.asarray([trace.hidden_states[0] for trace in student]),
        kind="low_rank",
        rank=4,
        source_role="teacher",
        target_role="student",
        max_dense_features=5_000_000,
        scalable_rank=4,
    )
    fit = fit_flow_transfer(teacher_traces=teacher, student_traces=student, teacher_to_student=alignment, signature_mode="full", scalable_rank=4)
    assert fit.metadata["scalable_backend"] == "randomized_latent_compression"
    assert fit.student.state_dim == 16 * 768
    assert fit.student.matrices.shape[1:] == (4, 4)
    assert fit.metadata["dense_memory_guard"]["quadratic_feature_count"] == (16 * 768) ** 2
    assert np.all(np.isfinite(fit.corrected_operator().predict(student[0].hidden_states[0], 0.25)))


def test_compressed_alignment_round_trip_preserves_original_chart_contract(tmp_path: Path):
    rng = np.random.default_rng(20)
    source = rng.normal(size=(7, 128))
    target = rng.normal(size=(7, 96))
    alignment = fit_alignment(source, target, source_role="teacher", target_role="student", scalable_rank=4, max_dense_features=100)
    path = save_alignment(alignment, tmp_path / "alignment.json")
    restored = load_alignment(path)
    assert restored.matrix.shape == (4, 4)
    assert restored.source_projection.shape == (128, 4)
    assert restored.target_projection.shape == (96, 4)
    assert np.allclose(restored.apply(source), alignment.apply(source))


def test_full_mode_uses_compact_chart_without_dense_quadratic_allocation():
    student = _traces(512, model_id="student")
    teacher = _traces(512, seed=42, model_id="teacher")
    alignment = AlignmentResult(
        "whitened_orthogonal",
        np.zeros(512),
        np.zeros(512),
        np.eye(512),
        np.zeros(512),
        512,
        512,
        0.0,
        0.0,
        0.0,
        1.0,
        metadata={"direction": "teacher_to_student"},
    )
    fit = fit_flow_transfer(
        student,
        teacher,
        alignment,
        signature_mode="full",
        max_dense_features=100,
        scalable_rank=8,
        scalable_seed=13,
    )
    assert fit.metadata["scalable_backend"] == "randomized_latent_compression"
    assert fit.metadata["requested_signature_mode"] == "full"
    assert fit.metadata["effective_signature_mode"] == "differential"
    assert fit.metadata["dense_memory_guard"]["quadratic_feature_count"] == 512**2
    assert fit.metadata["dense_memory_guard"]["quadratic_operator_elements"] == 512**3
    assert fit.correction_quadratic is None
    assert fit.student.matrices.shape[1:] == (8, 8)
    assert fit.student.chart_projection.shape == (512, 8)
    prediction = fit.corrected_operator().predict(np.ones(512), 0.25)
    assert prediction.shape == (512,)
    assert np.all(np.isfinite(prediction))
    profile = fit.metadata["math_profile"]
    assert profile["components"]["quadratic_flow"]["enabled"] is True
    assert profile["components"]["quadratic_flow"]["applied"] is False
    assert profile["components"]["quadratic_flow"]["skipped_reason"]
    assert profile["hessian_2jet_applied"] is False
    assert profile["quadratic_flow_applied"] is False
    assert profile["effective_signature_mode"] == "differential"


def test_scalable_fit_preserves_parent_alignment_scope_for_forensics():
    student = _traces(32, model_id="student")
    teacher = _traces(32, seed=42, model_id="teacher")
    alignment = AlignmentResult(
        "affine",
        np.zeros(32),
        np.zeros(32),
        np.eye(32),
        np.zeros(32),
        32,
        32,
        0.0,
        0.0,
        0.0,
        1.0,
        metadata={"direction": "teacher_to_student", "alignment_fit_scope": "initial_state_only"},
    )
    fit = fit_flow_transfer(
        student,
        teacher,
        alignment,
        signature_mode="full",
        max_dense_features=100,
        scalable_rank=6,
        scalable_seed=7,
    )
    assert fit.metadata["alignment_fit_scope"] == "initial_state_only"
    assert fit.metadata["alignment_paired_error"] == 0.0
    assert fit.transported_teacher.metadata["target_fit_split"] == "train"
    assert fit.transported_teacher.metadata["metric_scope"] == "transported_teacher_operator_approximation"


def test_scalable_flow_round_trip_preserves_chart_and_external_prediction(tmp_path: Path):
    student = _traces(32, model_id="student")
    teacher = _traces(32, seed=42, model_id="teacher")
    alignment = AlignmentResult("affine", np.zeros(32), np.zeros(32), np.eye(32), np.zeros(32), 32, 32, 0.0, 0.0, 0.0, 1.0, metadata={"direction": "teacher_to_student"})
    fit = fit_flow_transfer(student, teacher, alignment, signature_mode="full", max_dense_features=100, scalable_rank=6, scalable_seed=7)
    path = save_flow(fit.transported_teacher, tmp_path / "transported.npz")
    restored = load_flow(path)
    x = np.linspace(-0.2, 0.2, 32)
    assert restored.chart_projection.shape == (32, 6)
    assert np.allclose(restored.predict(x, 0.5), fit.transported_teacher.predict(x, 0.5))
