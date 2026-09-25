from dataclasses import replace

import numpy as np

import faytuna_flow.geometry as geometry
from faytuna_flow.artifacts import load_alignment, save_alignment
from faytuna_flow.geometry import _map_depth_domain, evaluate_dynamic_alignment, fit_alignment, fit_depth_conditioned_alignment, fit_dynamic_depth_conditioned_alignment, fit_whitening, pairwise_distances, relational_error, select_dynamic_depth_conditioned_alignment, sinkhorn_plan, trajectory_alignment_samples
from faytuna_flow.connectors import SyntheticConnector
from faytuna_flow.observation import ObservationProtocol
from faytuna_flow.probes import ProbeGenerator, ProbeGeneratorConfig
from faytuna_flow.synthetic import random_stable_system
from faytuna_flow.types import TrajectoryTrace, TransitionObservation


def test_whitening_has_identity_retained_covariance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 4)) @ np.diag([3.0, 2.0, 1.0, 0.4])
    whitening = fit_whitening(x, rank=4)
    z = whitening.encode(x)
    covariance = z.T @ z / (len(z) - 1)
    assert np.allclose(covariance, np.eye(4), atol=1e-7)
    assert np.allclose(whitening.decode(z), x, atol=1e-7)


def test_whitened_alignment_preserves_a_known_chart_transform():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 4)) @ np.diag([2.5, 1.7, 0.9, 0.4])
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    y = x @ q + np.array([1.0, -0.5, 0.25, 0.75])
    result = fit_alignment(x, y, kind="whitened_orthogonal")
    assert result.paired_error < 1e-6
    assert result.relational_error < 1e-6
    assert result.cycle_error < 1e-5
    assert np.isfinite(result.condition_number)


def test_alignment_supports_different_dimensions_and_low_rank():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 5))
    transform = rng.normal(size=(5, 3))
    y = x @ transform + 0.1
    result = fit_alignment(x, y, kind="low_rank", rank=3)
    assert result.matrix.shape == (5, 3)
    assert result.bias.shape == (3,)
    assert np.all(np.isfinite(result.apply(x)))


def test_sinkhorn_plan_has_uniform_marginals():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(15, 3)), rng.normal(size=(17, 3))
    plan, error = sinkhorn_plan(x, y, regularization=0.2)
    assert plan.shape == (15, 17)
    assert np.all(plan >= 0)
    assert error < 1e-6
    assert np.isclose(plan.sum(), 1.0, atol=1e-6)


def test_relational_error_detects_non_isometric_scaling():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 3))
    assert relational_error(x, x.copy()) < 1e-12
    assert relational_error(x, 2.0 * x) > 0.1


def test_ot_barycentric_alignment_stays_finite():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(24, 3))
    y = x + np.array([0.4, -0.2, 0.1])
    result = fit_alignment(x, y, kind="ot_barycentric", ot_regularization=0.2)
    assert np.all(np.isfinite(result.matrix))
    assert np.all(np.isfinite(result.bias))
    assert result.ot_mass_error < 1e-6


def test_rank_deficient_whitening_and_log_sinkhorn_are_deterministic_under_scale_noise():
    rng = np.random.default_rng(91)
    base = rng.normal(size=(32, 2))
    rank_deficient = np.column_stack([base[:, 0], 2.0 * base[:, 0], base[:, 1], np.zeros(len(base))])
    first = fit_whitening(rank_deficient, rank=4)
    second = fit_whitening(rank_deficient, rank=4)
    assert np.all(np.isfinite(first.encode(rank_deficient)))
    assert np.array_equal(first.encode(rank_deficient), second.encode(rank_deficient))
    extreme = rank_deficient * 1e150 + rng.normal(scale=1e-2, size=rank_deficient.shape)
    plan, marginal_error = sinkhorn_plan(extreme, extreme[::-1], regularization=0.25)
    assert np.all(np.isfinite(plan))
    assert np.isfinite(marginal_error)
    assert marginal_error < 1e-6


def test_trajectory_alignment_uses_all_existing_depths_without_fabricating_layers():
    probes = ProbeGenerator(ProbeGeneratorConfig(state_dim=4, per_family=3, seed=101)).generate()[:4]
    source = ObservationProtocol(jacobian_rank=1, hessian_rank=1).collect(
        SyntheticConnector(random_stable_system(layers=5, state_dim=4, seed=102), model_id="source"), probes
    )
    target_probes = [replace(probe, initial_state=probe.initial_state[:3]) for probe in probes]
    target = ObservationProtocol(jacobian_rank=1, hessian_rank=1).collect(
        SyntheticConnector(random_stable_system(layers=2, state_dim=3, seed=103), model_id="target"), target_probes
    )
    x, y, metadata = trajectory_alignment_samples(source, target)
    assert x.shape == (len(source) * source[0].layer_count, 4)
    assert y.shape == (len(source) * source[0].layer_count, 3)
    assert metadata["sample_count"] == x.shape[0]
    assert metadata["interpolation_creates_no_layer"] is True


def test_depth_domain_mapping_uses_relative_coordinate_for_different_units():
    mapped = _map_depth_domain(np.asarray([10.0, 20.0, 30.0]), np.asarray([100.0, 160.0, 220.0, 280.0]))
    assert np.allclose(mapped, np.asarray([100.0, 190.0, 280.0]))


def test_gpt2_sized_pairwise_gram_path_has_no_feature_broadcast():
    # This is the flattened GPT-2 XL chart for sequence_length=16. The array
    # itself is intentionally real-sized, while the output remains only N×N.
    samples = 1764
    features = 16 * 768
    values = np.zeros((samples, features), dtype=np.float32)
    values[:, 0] = np.linspace(-1.0, 1.0, samples, dtype=np.float32)
    distances = pairwise_distances(values, max_pairwise_bytes=4 * 1024 * 1024)
    assert distances.shape == (samples, samples)
    assert distances.dtype == np.float64
    assert np.all(np.isfinite(distances))
    assert np.allclose(np.diag(distances), 0.0, atol=1e-7)


def test_gpt2_sized_sinkhorn_uses_bounded_gram_blocks():
    samples = 1764
    features = 16 * 768
    values = np.zeros((samples, features), dtype=np.float32)
    values[:, 0] = np.linspace(-1.0, 1.0, samples, dtype=np.float32)
    plan, error = sinkhorn_plan(values, values, regularization=0.25, max_iter=80, tolerance=1e-6, max_pairwise_bytes=4 * 1024 * 1024)
    assert plan.shape == (samples, samples)
    assert np.all(np.isfinite(plan))
    assert np.isfinite(error)
    assert np.isclose(plan.sum(), 1.0, atol=1e-5)


def test_compressed_alignment_records_gram_distance_backend():
    rng = np.random.default_rng(122)
    source = rng.normal(size=(24, 5))
    target = rng.normal(size=(24, 3))
    result = fit_alignment(
        source,
        target,
        kind="low_rank",
        rank=2,
        max_dense_features=1,
        scalable_rank=2,
        max_pairwise_bytes=1024,
    )
    assert result.metadata["pairwise_distance_backend"] == "scaled_gram_identity_blockwise"
    assert result.metadata["pairwise_distance_feature_broadcast_allocated"] is False
    assert result.metadata["pairwise_distance_output_shape"] == [24, 24]


def _depth_varying_pair(*, count: int = 48, seed: int = 177) -> tuple[list[TrajectoryTrace], list[TrajectoryTrace]]:
    rng = np.random.default_rng(seed)
    coordinates = np.linspace(0.0, 1.0, 5)
    teacher = []
    student = []
    maps = [
        np.asarray([[1.20, 0.10, 0.00], [0.00, 0.75, 0.15], [0.05, 0.00, 1.05]]),
        np.asarray([[0.85, -0.25, 0.10], [0.20, 1.15, 0.00], [0.00, 0.10, 0.70]]),
        np.asarray([[1.05, 0.30, -0.10], [-0.10, 0.80, 0.25], [0.15, 0.00, 1.25]]),
        np.asarray([[0.70, 0.05, 0.20], [0.00, 1.30, -0.15], [-0.20, 0.10, 0.90]]),
        np.asarray([[1.25, -0.10, 0.05], [0.15, 0.65, 0.20], [0.00, -0.05, 1.10]]),
    ]
    biases = [np.asarray([0.2 * node, -0.1 * node, 0.05 * node]) for node in range(len(coordinates))]
    for index in range(count):
        base = rng.normal(size=3)
        source_states = np.asarray([base @ np.linalg.matrix_power(np.asarray([[0.92, 0.04, 0.00], [0.00, 0.88, 0.06], [0.03, 0.00, 0.90]]), node) for node in range(len(coordinates))])
        target_states = np.asarray([source_states[node] @ maps[node] + biases[node] for node in range(len(coordinates))])
        source_transitions = []
        target_transitions = []
        for node in range(len(coordinates) - 1):
            ds = coordinates[node + 1] - coordinates[node]
            source_delta = source_states[node + 1] - source_states[node]
            target_delta = target_states[node + 1] - target_states[node]
            source_transitions.append(TransitionObservation(node, node + 1, coordinates[node], coordinates[node + 1], source_states[node], source_states[node + 1], source_delta, source_delta / ds, jacobian=np.eye(3)))
            target_transitions.append(TransitionObservation(node, node + 1, coordinates[node], coordinates[node + 1], target_states[node], target_states[node + 1], target_delta, target_delta / ds, jacobian=np.eye(3)))
        teacher.append(TrajectoryTrace("teacher-depth", f"p{index}", tuple(range(5)), coordinates, source_states, None, tuple(source_transitions)))
        student.append(TrajectoryTrace("student-depth", f"p{index}", tuple(range(5)), coordinates, target_states, None, tuple(target_transitions)))
    return teacher, student


def test_depth_conditioned_alignment_beats_global_map_and_round_trips(tmp_path):
    teacher, student = _depth_varying_pair()
    x, y, _ = trajectory_alignment_samples(teacher, student)
    global_map = fit_alignment(x, y, kind="affine", source_role="teacher", target_role="student")
    local_map = fit_depth_conditioned_alignment(
        teacher,
        student,
        kind="affine",
        source_role="teacher",
        target_role="student",
        scalable_rank=3,
        max_dense_features=1_000,
    )
    assert local_map.metadata["alignment_strategy"] == "depth_conditioned_local_maps"
    assert len(local_map.metadata["depth_metrics"]) == 5
    assert local_map.metadata["depth_local_paired_error"] < global_map.paired_error * 0.1
    assert local_map.metadata["depth_local_improvement_over_global"] > 0.0
    assert np.isclose(local_map.metadata["global_shared_map_paired_error"], global_map.paired_error, rtol=1e-6, atol=1e-8)
    mapped = local_map.apply(teacher[0].hidden_states, depth=teacher[0].depth_coordinates)
    assert np.allclose(mapped, student[0].hidden_states, atol=1e-8)
    tangent = np.asarray([[0.3, -0.2, 0.1]])
    assert np.allclose(
        local_map.apply(teacher[2].hidden_states[2] + tangent[0], depth=teacher[2].depth_coordinates[2])
        - local_map.apply(teacher[2].hidden_states[2], depth=teacher[2].depth_coordinates[2]),
        local_map.linear_apply(tangent, depth=teacher[2].depth_coordinates[2]),
        atol=1e-8,
    )
    path = save_alignment(local_map, tmp_path / "depth-local.json")
    restored = load_alignment(path)
    assert restored.depth_matrices is not None
    assert restored.depth_biases is not None
    assert np.allclose(restored.apply(teacher[3].hidden_states, depth=teacher[3].depth_coordinates), local_map.apply(teacher[3].hidden_states, depth=teacher[3].depth_coordinates))


def test_dynamic_aware_alignment_uses_transition_jacobian_and_smoothness_terms():
    train_teacher, train_student = _depth_varying_pair(count=36, seed=177)
    validation_teacher, validation_student = _depth_varying_pair(count=12, seed=911)
    candidate = fit_dynamic_depth_conditioned_alignment(
        train_teacher,
        train_student,
        kind="affine",
        source_role="teacher",
        target_role="student",
        velocity_weight=1.0,
        jacobian_weight=0.25,
        smoothness_weight=0.25,
        scalable_rank=3,
        max_dense_features=1_000,
    )
    metadata = candidate.metadata
    assert metadata["alignment_strategy"] == "depth_conditioned_dynamic_aware"
    assert metadata["velocity_row_count"] == 36 * 4
    assert metadata["jacobian_row_count"] == 36 * 4 * 3
    assert metadata["smoothness_applied"] is True
    assert metadata["local_map_discontinuity_after"] <= metadata["local_map_discontinuity_before"]
    validation_metrics = evaluate_dynamic_alignment(
        validation_teacher,
        validation_student,
        candidate,
        velocity_weight=1.0,
        jacobian_weight=0.25,
        smoothness_weight=0.25,
    )
    assert np.isfinite(validation_metrics["objective"])
    selected, report = select_dynamic_depth_conditioned_alignment(
        train_teacher,
        train_student,
        validation_teacher,
        validation_student,
        candidates=(
            {"velocity_weight": 0.25, "jacobian_weight": 0.0, "smoothness_weight": 0.05},
            {"velocity_weight": 1.0, "jacobian_weight": 0.25, "smoothness_weight": 0.25},
        ),
        kind="affine",
        source_role="teacher",
        target_role="student",
        scalable_rank=3,
        max_dense_features=1_000,
    )
    assert report["status"] == "selected"
    assert report["selection_split"] == "validation"
    assert len(report["candidates"]) == 2
    assert report["pareto_metrics"] == ["state_error", "velocity_error", "jacobian_error", "smoothness_error"]
    assert report["pareto_front_candidate_indices"]
    assert all("train_fit_paired_error" in item for item in report["candidates"] if item["status"] == "scored")
    assert selected.metadata["fit_split"] == "train"
    assert selected.metadata["selection_split"] == "validation"


def test_dynamic_aware_large_chart_skips_dense_jacobian_rows(monkeypatch):
    teacher, student = _depth_varying_pair(count=8, seed=191)
    # Force the same branch that GPT-2-sized charts take without allocating a
    # chart-sized Jacobian or quadratic feature tensor in the test.
    monkeypatch.setattr(geometry, "DEFAULT_MAX_DIRECT_JACOBIAN_ROW_ELEMENTS", 1)
    result = fit_dynamic_depth_conditioned_alignment(
        teacher,
        student,
        kind="affine",
        velocity_weight=0.5,
        jacobian_weight=0.5,
        smoothness_weight=0.1,
        scalable_rank=3,
        max_dense_features=1_000,
    )
    metadata = result.metadata
    assert metadata["jacobian_direct_skipped"] is True
    assert metadata["jacobian_row_count"] == 0
    assert metadata["jacobian_descriptor_count"] == 8 * 4
    assert metadata["jacobian_skipped_reason"]
    assert np.isfinite(result.paired_error)
