from pathlib import Path

import numpy as np

from faytuna_flow.ablations import run_transfer_ablations
from faytuna_flow.geometry import fit_alignment, fit_whitening, sinkhorn_plan
from faytuna_flow.pipeline import TraceBundle, collect_staged_synthetic_splits, collect_synthetic_pair, fit_bundle, load_bundle, save_bundle
from faytuna_flow.signatures import differential_signature, fit_tangent_transport, multi_scale_path_signature
from faytuna_flow.flow import fit_flow_operator
from faytuna_flow.jets import fit_rank_aware_local_jet_transport
from faytuna_flow.depth import MonotoneCorrespondence
from faytuna_flow.types import Probe, TrajectoryTrace, TransitionObservation


def test_multiscale_signature_is_rigid_chart_invariant_and_differential_signature_is_finite():
    rng = np.random.default_rng(401)
    states = np.cumsum(rng.normal(size=(11, 4)), axis=0)
    coordinates = np.linspace(0.0, 1.0, len(states))
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    transformed = states @ rotation + np.array([10.0, -2.0, 0.5, 3.0])
    first = multi_scale_path_signature(states, coordinates)
    second = multi_scale_path_signature(transformed, coordinates)
    assert first.values.shape == (3, 8)
    assert first.area_backend == second.area_backend == "exact_dense"
    assert np.allclose(first.values, second.values, atol=1e-8)

    from faytuna_flow.connectors import SyntheticConnector
    from faytuna_flow.observation import ObservationProtocol
    from faytuna_flow.probes import ProbeGenerator, ProbeGeneratorConfig
    from faytuna_flow.synthetic import random_stable_system

    connector = SyntheticConnector(random_stable_system(layers=5, state_dim=4, seed=402, nonlinear=True))
    probe = ProbeGenerator(ProbeGeneratorConfig(state_dim=4, per_family=3, seed=403)).generate()[0]
    trace = ObservationProtocol(hessian_rank=3).collect(connector, [probe], seed=404)[0]
    signature = differential_signature(trace)
    assert signature.values.shape == (trace.layer_count - 1, 8)
    assert np.all(np.isfinite(signature.values))


def test_high_dimensional_path_signature_forbids_dense_ordered_area(monkeypatch):
    import faytuna_flow.signatures as signatures

    original_zeros = signatures.np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2 and shape[0] > 512 and shape[1] > 512:
            raise AssertionError("dense high-dimensional ordered area was allocated")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(signatures.np, "zeros", guarded_zeros)
    rng = np.random.default_rng(405)
    states = np.cumsum(rng.normal(size=(7, 1024)), axis=0)
    signature = signatures.multi_scale_path_signature(states, np.linspace(0.0, 1.0, len(states)), area_seed=406)
    assert signature.area_backend == "randomized_frobenius_sketch"
    assert signature.area_sketch_rank == 32
    assert np.all(np.isfinite(signature.values))


def test_cross_dimensional_tangent_transport_and_holdout_calibration(tmp_path: Path):
    train = collect_synthetic_pair(teacher_dim=6, student_dim=3, teacher_layers=7, student_layers=4, per_family=3, seed=410, split="train")
    validation = collect_synthetic_pair(teacher_dim=6, student_dim=3, teacher_layers=7, student_layers=4, per_family=3, seed=410, split="validation")
    alignment, fit = fit_bundle(train, validation=validation, kind="low_rank", signature_mode="full")
    assert alignment.matrix.shape == (6, 3)
    assert fit.metadata["confidence_calibrated"] is True
    assert fit.metadata["multi_scale_path_signatures"] is True
    assert np.all(np.isfinite(fit.confidence))
    assert fit.capacity_diagnostics is not None

    tangent = fit_tangent_transport(train.student, train.teacher, alignment)
    assert tangent.transport_matrices.shape == (train.student[0].layer_count, 3, 3)
    assert np.all(np.isfinite(tangent.confidence))

    restored_dir = save_bundle(train, tmp_path / "paired")
    restored = load_bundle(restored_dir)
    assert restored.stage == train.stage
    assert restored.teacher[0].feature_space == train.teacher[0].feature_space
    assert restored.student[0].probe_id == train.student[0].probe_id


def test_local_jet_teacher_variation_uses_continuous_depth_correspondence():
    coordinates_student = np.linspace(0.0, 1.0, 4)
    coordinates_teacher = np.linspace(0.0, 1.0, 7)

    def make_trace(model: str, probe_id: str, coordinates: np.ndarray, offset: float) -> TrajectoryTrace:
        states = np.asarray([np.array([offset + 0.01 * index, 0.02 * index, -0.01 * index]) for index in range(len(coordinates))])
        transitions = tuple(
            TransitionObservation(
                index,
                index + 1,
                coordinates[index],
                coordinates[index + 1],
                states[index],
                states[index + 1],
                states[index + 1] - states[index],
                (states[index + 1] - states[index]) / (coordinates[index + 1] - coordinates[index]),
            )
            for index in range(len(coordinates) - 1)
        )
        return TrajectoryTrace(model, probe_id, tuple(range(len(coordinates))), coordinates, states, None, transitions, {"probe_pair_id": "bundle-0"})

    student = [make_trace("student", "bundle-0-base", coordinates_student, 0.0), make_trace("student", "bundle-0-perturbed", coordinates_student, 0.2)]
    teacher = [make_trace("teacher", "bundle-0-base", coordinates_teacher, 0.0), make_trace("teacher", "bundle-0-perturbed", coordinates_teacher, 0.2)]
    identity = fit_alignment(np.asarray([trace.hidden_states[0] for trace in teacher]), np.asarray([trace.hidden_states[0] for trace in student]), kind="affine")
    correspondence = MonotoneCorrespondence(((0, 0), (1, 2), (2, 4), (3, 6)), (), 0.0, 1.0)
    transport = fit_rank_aware_local_jet_transport(student, teacher, identity, [correspondence, correspondence], rank=1, quadratic=False)
    mapped = transport.metadata["teacher_variation_node_indices"]
    assert [row[0] for row in mapped] == [0, 2, 4]
    assert [row[0] for row in mapped] != [0, 1, 2]


def test_ablation_suite_keeps_teacher_and_student_and_records_controls():
    train = collect_synthetic_pair(teacher_dim=5, student_dim=3, teacher_layers=6, student_layers=4, per_family=3, seed=420, split="train")
    initial_teacher = np.asarray([trace.hidden_states[0] for trace in train.teacher])
    initial_student = np.asarray([trace.hidden_states[0] for trace in train.student])
    alignment = fit_alignment(initial_teacher, initial_student, kind="low_rank", rank=3)
    report = run_transfer_ablations(train.student, train.teacher, alignment, seed=421, include_stress=True)
    names = {case.name for case in report.cases}
    assert {"static_hidden", "delta_flow", "curvature_jacobian", "full_flow", "shuffled_probes", "random_map", "noise_stress", "quantization_stress"} <= names
    assert report.metadata["teacher_and_student_required"] is True
    assert all(np.isfinite(case.geometric_error) for case in report.cases)
    assert report.to_dict()["holdout_used"] is False
    assert all(case["status"] in {"reference", "insufficient_data", "rejected", "improved", "degraded", "unchanged"} for case in report.to_dict()["entries"])


def test_rank_deficient_and_large_coordinate_geometry_remains_finite():
    values = np.asarray([[1e150, 2e140, 0.0], [1e150, 2e140, 0.0], [1e150 + 1e140, 2e140 - 1e139, 0.0], [1e150 - 1e140, 2e140 + 1e139, 0.0]])
    whitening = fit_whitening(values, rank=3)
    assert np.all(np.isfinite(whitening.encode(values)))
    assert np.all(np.isfinite(whitening.decode(whitening.encode(values))))
    plan, error = sinkhorn_plan(values, values[::-1], regularization=0.15)
    assert np.all(np.isfinite(plan))
    assert error < 1e-6


def test_staged_split_artifacts_are_labelled_and_quadratic_flow_round_trips(tmp_path: Path):
    stages = collect_staged_synthetic_splits(per_family=3, seed=430)
    assert set(stages) == {"synthetic_ground_truth", "tiny_small", "larger_teacher_smaller_student"}
    for stage, splits in stages.items():
        assert set(splits) == {"train", "validation", "holdout"}
        assert all(bundle.teacher[0].metadata["probe_split"] == name for name, bundle in splits.items())
    train = stages["tiny_small"]["train"]
    alignment, fit = fit_bundle(train, kind="low_rank")
    from faytuna_flow.artifacts import load_flow, save_flow
    restored = load_flow(save_flow(fit.transported_teacher, tmp_path / "quadratic-flow.npz"))
    assert restored.quadratic_terms is not None
    assert np.allclose(restored.quadratic_terms, fit.transported_teacher.quadratic_terms)


def test_quadratic_local_flow_captures_nonlinear_ground_truth_beyond_affine_fit():
    rng = np.random.default_rng(440)
    dimension = 3
    linear = np.asarray([[-0.18, 0.04, 0.01], [0.02, -0.12, 0.03], [0.01, 0.02, -0.15]])
    quadratic = np.zeros((dimension, dimension, dimension))
    quadratic[0, 0, 0] = 0.22
    quadratic[1, 1, 2] = -0.17
    quadratic[2, 0, 1] = 0.13
    coordinates = np.linspace(0.0, 1.0, 5)

    def make_traces(initials):
        traces = []
        for index, initial in enumerate(initials):
            states = [initial.copy()]
            transitions = []
            for node in range(len(coordinates) - 1):
                state = states[-1]
                velocity = state @ linear + np.einsum("jik,i,k->j", quadratic, state, state)
                delta = (coordinates[node + 1] - coordinates[node]) * velocity
                target = state + delta
                transitions.append(TransitionObservation(node, node + 1, coordinates[node], coordinates[node + 1], state, target, delta, velocity))
                states.append(target)
            traces.append(TrajectoryTrace("quadratic-ground-truth", f"q-{index}", tuple(range(-1, len(coordinates) - 1)), coordinates, np.asarray(states), None, tuple(transitions)))
        return traces

    train = make_traces(rng.normal(size=(80, dimension)))
    holdout = make_traces(rng.normal(size=(40, dimension)))
    affine = fit_flow_operator(train, min_samples_per_node=2, quadratic=False)
    full = fit_flow_operator(train, min_samples_per_node=2, quadratic=True, quadratic_ridge=1e-8)
    affine_error = np.mean([np.linalg.norm(affine.predict(t.transitions[0].source_state, 0.0) - t.transitions[0].vector_field) for t in holdout])
    full_error = np.mean([np.linalg.norm(full.predict(t.transitions[0].source_state, 0.0) - t.transitions[0].vector_field) for t in holdout])
    assert full.quadratic_terms is not None
    assert full_error < affine_error * 0.75
