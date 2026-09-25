from pathlib import Path

import numpy as np

from faytuna_flow.flow import fit_flow_operator, fit_flow_transfer
from faytuna_flow.geometry import fit_alignment
from faytuna_flow.observation import ObservationProtocol
from faytuna_flow.solver import TrustRegionConfig, solve_flow_correction
from faytuna_flow.surgery import apply_surgery, build_surgery_plan, export_weights, import_weights
from faytuna_flow.synthetic import LinearResidualSystem
from faytuna_flow.types import FlowFitResult, FlowOperator, Probe, TensorTransitionMapping, TrajectoryTrace, TransitionObservation


def _linear_traces(matrices, biases, count=50, model_id="linear", initial_states=None, coordinates=None):
    rng = np.random.default_rng(23)
    probes = [Probe(f"p{i}", "synthetic", {}, rng.normal(size=matrices.shape[1])) for i in range(count)]
    traces = []
    coordinates = np.linspace(0.0, 1.0, len(matrices) + 1) if coordinates is None else np.asarray(coordinates)
    for probe in probes:
        initial = probe.initial_state if initial_states is None else np.asarray(initial_states[len(traces)])
        probe = Probe(probe.probe_id, probe.family, probe.payload, initial)
        states = [initial.copy()]
        transitions = []
        for i, (matrix, bias) in enumerate(zip(matrices, biases)):
            ds = coordinates[i + 1] - coordinates[i]
            delta = ds * (states[-1] @ matrix + bias)
            target = states[-1] + delta
            transitions.append(TransitionObservation(i, i + 1, coordinates[i], coordinates[i + 1], states[-1], target, delta, delta / ds))
            states.append(target)
        traces.append(TrajectoryTrace(model_id, probe.probe_id, tuple(range(-1, len(matrices))), coordinates, np.asarray(states), None, tuple(transitions)))
    return traces


def test_flow_fit_recovers_transportable_linear_operator():
    layers, dim = 4, 3
    student_a = np.asarray([np.diag([-0.2, -0.1, -0.15]) + 0.01 * i for i in range(layers)])
    teacher_a = student_a + np.asarray([np.eye(dim) * (0.03 + 0.005 * i) for i in range(layers)])
    zero = np.zeros((layers, dim))
    student = _linear_traces(student_a, zero, model_id="student")
    teacher = _linear_traces(teacher_a, zero, model_id="teacher")
    x = np.asarray([t.hidden_states[0] for t in student])
    alignment = fit_alignment(x, x.copy(), kind="whitened_orthogonal")
    fit = fit_flow_transfer(student, teacher, alignment)
    assert fit.correction_matrices.shape == (layers, dim, dim)
    assert np.all(fit.confidence > 0.5)
    assert np.allclose(fit.correction_matrices, teacher_a - student_a, atol=1e-3)
    assert fit.depth_report is not None
    assert fit.depth_report.gap_count == 0
    assert fit.depth_report.matched_fraction == 1.0


def test_cross_dimensional_teacher_to_student_flow_uses_student_coordinates_and_reports_bottleneck():
    rng = np.random.default_rng(24)
    teacher_dim, student_dim = 5, 3
    count = 64
    teacher_initial = rng.normal(size=(count, teacher_dim))
    projection = rng.normal(size=(teacher_dim, student_dim))
    student_initial = teacher_initial @ projection
    teacher_a = np.stack([np.diag([-0.10, -0.08, -0.06, -0.04, -0.02]) for _ in range(5)])
    student_a = np.stack([np.diag([-0.12, -0.09, -0.07]) for _ in range(3)])
    teacher = _linear_traces(teacher_a, np.zeros((5, teacher_dim)), count=count, model_id="large-teacher", initial_states=teacher_initial)
    student = _linear_traces(student_a, np.zeros((3, student_dim)), count=count, model_id="small-student", initial_states=student_initial, coordinates=np.linspace(0.0, 1.0, 4))
    alignment = fit_alignment(teacher_initial, student_initial, kind="low_rank", rank=student_dim)
    fit = fit_flow_transfer(student, teacher, alignment)
    assert fit.transported_teacher.state_dim == student_dim
    assert fit.student.state_dim == student_dim
    assert fit.correction_matrices.shape == (3, student_dim, student_dim)
    assert fit.metadata["alignment_direction"] == "teacher_to_student"
    assert fit.depth_report.teacher_nodes == 6
    assert fit.depth_report.student_nodes == 4
    assert fit.capacity_diagnostics is not None
    assert fit.capacity_diagnostics.teacher_state_rank >= fit.capacity_diagnostics.student_state_rank
    try:
        fit_flow_transfer(student, teacher, fit_alignment(student_initial, teacher_initial, kind="low_rank", rank=student_dim))
    except ValueError as error:
        assert "teacher_to_student alignment" in str(error)
    else:
        raise AssertionError("reverse alignment was accepted")


def test_solver_enforces_trust_region_and_rolls_back_low_confidence():
    dim = 3
    baseline = FlowOperator(np.array([0.0, 1.0]), np.stack([np.eye(dim), np.eye(dim)]), np.zeros((2, dim)), np.full(2, 10.0), np.ones(2) * 0.01, np.ones(2))
    target = FlowOperator(baseline.coordinates, baseline.matrices + np.stack([np.ones((dim, dim)) * 10, np.eye(dim) * 0.1]), baseline.biases, baseline.sample_counts, baseline.residual_scales, baseline.spectral_norms)
    fit = FlowFitResult(baseline, target, target.matrices - baseline.matrices, np.zeros((2, dim)), np.array([0.9, 0.1]), 0.0)
    result = solve_flow_correction(fit, config=TrustRegionConfig(max_step_norm=0.2, max_relative_step=0.1, max_lipschitz=1.5))
    assert result.accepted[0]
    assert not result.accepted[1]
    assert result.reasons[1] == "confidence below threshold"
    assert np.all(result.diagnostics["spectral_norm"] <= 1.5 + 1e-8)
    assert np.all(np.isfinite(result.matrices))


def test_surgery_preserves_tensor_schema_and_rolls_back_incompatible_nodes():
    dim = 3
    baseline = FlowOperator(np.array([0.0, 1.0]), np.stack([np.eye(dim), np.eye(dim)]), np.zeros((2, dim)), np.ones(2), np.ones(2) * 0.01, np.ones(2))
    fit = FlowFitResult(baseline, baseline, np.stack([np.eye(dim) * 0.05, np.eye(dim) * 0.05]), np.zeros((2, dim)), np.array([1.0, 0.05]), 0.0)
    from faytuna_flow.solver import solve_flow_correction

    constrained = solve_flow_correction(fit)
    weights = {"layers.0.weight": np.eye(dim), "layers.1.weight": np.eye(dim), "head.weight": np.ones((2, dim), dtype=np.float32)}
    plan = build_surgery_plan(weights, constrained, mapping=[TensorTransitionMapping(0, "layers.0.weight"), TensorTransitionMapping(1, "layers.1.weight")])
    updated = apply_surgery(weights, plan)
    assert updated.keys() == weights.keys()
    assert updated["layers.0.weight"].shape == weights["layers.0.weight"].shape
    assert updated["head.weight"].dtype == np.float32
    assert 1 in plan.rollback_layers


def test_surgery_requires_explicit_mapping_and_fails_fast_in_apply_mode():
    dim = 3
    baseline = FlowOperator(np.array([0.0]), np.eye(dim)[None, :, :], np.zeros((1, dim)), np.ones(1), np.ones(1) * 0.01, np.ones(1))
    fit = FlowFitResult(baseline, baseline, np.eye(dim)[None, :, :] * 0.05, np.zeros((1, dim)), np.ones(1), 0.0)
    from faytuna_flow.solver import solve_flow_correction

    constrained = solve_flow_correction(fit)
    weights = {"layers.0.weight": np.eye(dim)}
    diagnostic = build_surgery_plan(weights, constrained)
    assert not diagnostic.applied_tensors
    assert diagnostic.skipped_tensors["layers.0.weight"].startswith("not selected")
    try:
        build_surgery_plan(weights, constrained, mode="apply")
    except Exception as error:
        assert "no applicable tensor updates" in str(error)
    else:
        raise AssertionError("apply mode must fail without explicit mapping")


def test_surgery_apply_mode_rejects_effective_noop_even_with_mapping():
    dim = 2
    baseline = FlowOperator(np.array([0.0]), np.eye(dim)[None, :, :], np.zeros((1, dim)), np.ones(1), np.ones(1) * 0.01, np.ones(1))
    fit = FlowFitResult(baseline, baseline, np.eye(dim)[None, :, :] * 0.05, np.zeros((1, dim)), np.ones(1), 0.0)
    constrained = solve_flow_correction(fit)
    try:
        build_surgery_plan({"layers.0.weight": np.eye(dim)}, constrained, mapping=[TensorTransitionMapping(0, "layers.0.weight")], gain=0.0, mode="apply")
    except Exception as error:
        assert "no applicable tensor updates" in str(error)
    else:
        raise AssertionError("apply mode must reject a mapped but unchanged tensor")


def test_direction_metadata_rejects_reverse_alignment_even_when_dimensions_match():
    dim = 3
    student = _linear_traces(np.stack([np.eye(dim) * -0.1 for _ in range(2)]), np.zeros((2, dim)), model_id="student")
    teacher = _linear_traces(np.stack([np.eye(dim) * -0.08 for _ in range(2)]), np.zeros((2, dim)), model_id="teacher")
    source = np.asarray([trace.hidden_states[0] for trace in student])
    target = np.asarray([trace.hidden_states[0] for trace in teacher])
    reverse = fit_alignment(source, target, kind="affine", source_role="student", target_role="teacher")
    try:
        fit_flow_transfer(student, teacher, reverse)
    except ValueError as error:
        assert "alignment direction" in str(error)
    else:
        raise AssertionError("declared reverse alignment must be rejected")


def test_round_trip_export_of_ordinary_weights(tmp_path: Path):
    weights = {"layers.0.weight": np.eye(3), "layers.0.bias": np.array([0.1, 0.2, 0.3]), "head/projection": np.arange(6, dtype=np.float32).reshape(2, 3)}
    path = export_weights(weights, tmp_path / "weights.npz")
    restored = import_weights(path)
    assert restored.keys() == weights.keys()
    for key in weights:
        assert restored[key].dtype == weights[key].dtype
        assert np.array_equal(restored[key], weights[key])
    no_suffix = export_weights(weights, tmp_path / "weights_without_suffix")
    assert no_suffix.suffix == ".npz"
    assert import_weights(tmp_path / "weights_without_suffix").keys() == weights.keys()


def test_surgery_reports_nonfloating_tensors_instead_of_casting_them():
    dim = 2
    baseline = FlowOperator(np.array([0.0]), np.eye(dim)[None, :, :], np.zeros((1, dim)), np.ones(1), np.ones(1) * 0.01, np.ones(1))
    fit = FlowFitResult(baseline, baseline, np.eye(dim)[None, :, :] * 0.01, np.zeros((1, dim)), np.ones(1), 0.0)
    constrained = solve_flow_correction(fit)
    plan = build_surgery_plan({"layers.0.weight": np.ones((dim, dim), dtype=np.int32)}, constrained, mapping=[TensorTransitionMapping(0, "layers.0.weight")])
    assert "not a floating point" in plan.skipped_tensors["layers.0.weight"]
    assert not plan.applied_tensors


def test_quadratic_surgery_requires_and_honors_explicit_callback_contract():
    dim = 2
    zero_quadratic = np.zeros((1, dim, dim, dim))
    baseline = FlowOperator(np.array([0.0]), np.eye(dim)[None, :, :], np.zeros((1, dim)), np.ones(1), np.ones(1) * 0.01, np.ones(1), {}, zero_quadratic)
    fit = FlowFitResult(baseline, baseline, np.eye(dim)[None, :, :] * 0.02, np.zeros((1, dim)), np.ones(1), 0.0, correction_quadratic=np.ones_like(zero_quadratic) * 0.01)
    constrained = solve_flow_correction(fit, config=TrustRegionConfig(min_confidence=0.0))
    weights = {"layers.0.weight": np.eye(dim), "layers.0.quadratic": np.zeros((dim, dim, dim))}

    def apply_quadratic(values, quadratic, gain):
        assert quadratic.shape == (1, dim, dim, dim)
        return {"layers.0.quadratic": values["layers.0.quadratic"] + gain * quadratic[0]}

    plan = build_surgery_plan(
        weights,
        constrained,
        mapping=[TensorTransitionMapping(0, "layers.0.weight")],
        quadratic_callback=apply_quadratic,
        mode="apply",
    )
    updated = apply_surgery(weights, plan)
    assert plan.metadata["quadratic_terms_applied"] is True
    assert "layers.0.quadratic" in plan.applied_tensors
    assert not np.array_equal(updated["layers.0.quadratic"], weights["layers.0.quadratic"])


def test_solver_binds_state_dependent_quadratic_jacobian():
    dim = 2
    baseline = FlowOperator(np.array([0.0]), np.eye(dim)[None, :, :], np.zeros((1, dim)), np.ones(1), np.ones(1) * 0.01, np.ones(1), {"input_radius": 3.0}, np.zeros((1, dim, dim, dim)))
    fit = FlowFitResult(baseline, baseline, np.zeros((1, dim, dim)), np.zeros((1, dim)), np.ones(1), 0.0, correction_quadratic=np.ones((1, dim, dim, dim)))
    constrained = solve_flow_correction(fit, config=TrustRegionConfig(min_confidence=0.0, max_lipschitz=1.5, max_quadratic_norm=10.0))
    assert constrained.diagnostics["quadratic_jacobian_bound"][0] <= 0.5 + 1e-8
    assert constrained.diagnostics["lipschitz"][0] <= 1.5 + 1e-8
    assert np.all(np.isfinite(constrained.quadratic_terms))
