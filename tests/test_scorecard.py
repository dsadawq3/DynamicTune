import json

import numpy as np

from faytuna_flow.geometry import fit_alignment
from faytuna_flow.pipeline import collect_synthetic_pair_splits
from faytuna_flow.scorecard import run_transfer_scorecard
from faytuna_flow.types import TrajectoryTrace, TransitionObservation


def test_scorecard_is_holdout_first_and_negative_controls_are_finite():
    splits = collect_synthetic_pair_splits(teacher_dim=5, student_dim=3, teacher_layers=6, student_layers=4, per_family=3, seed=601)
    train, validation, holdout = splits["train"], splits["validation"], splits["holdout"]
    alignment = fit_alignment(
        np.asarray([trace.hidden_states[0] for trace in train.teacher]),
        np.asarray([trace.hidden_states[0] for trace in train.student]),
        kind="low_rank",
        rank=3,
        source_role="teacher",
        target_role="student",
    )
    report = run_transfer_scorecard(
        train.student,
        train.teacher,
        alignment,
        validation_student=validation.student,
        validation_teacher=validation.teacher,
        holdout_student=holdout.student,
        holdout_teacher=holdout.teacher,
        seed=601,
    )
    payload = report.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["schema_version"] == "faytuna-scorecard-v1"
    assert payload["validation_used"] is True
    assert payload["holdout_used"] is True
    entries = {entry.name: entry for entry in report.entries}
    assert {"static_hidden", "baseline_student_flow", "delta_flow", "curvature_jacobian", "full_flow", "shuffled_probes", "random_map"} <= set(entries)
    assert entries["shuffled_probes"].status in {"rejected", "improved", "degraded", "unchanged", "insufficient_data"}
    assert entries["random_map"].status in {"rejected", "improved", "degraded", "unchanged", "insufficient_data"}
    for entry in report.entries:
        rendered = json.dumps(entry.to_dict(), allow_nan=False)
        assert "NaN" not in rendered and "Infinity" not in rendered
    for name in ("delta_flow", "curvature_jacobian", "full_flow"):
        entry = entries[name]
        if entry.holdout_error is not None and entry.baseline_holdout_error is not None:
            assert np.isclose(entry.improvement, entry.baseline_holdout_error - entry.holdout_error)
            assert (entry.status == "improved") == bool(entry.improvement > 1e-12)


def _quadratic_traces(initial_states: np.ndarray, split: str, model: str, quadratic: np.ndarray) -> list[TrajectoryTrace]:
    dimension = 3
    coordinates = np.linspace(0.0, 1.0, 7)
    linear = np.asarray([[-0.18, 0.04, 0.01], [0.02, -0.12, 0.03], [0.01, 0.02, -0.15]])
    traces = []
    for index, initial in enumerate(initial_states):
        states = [initial.copy()]
        transitions = []
        for node in range(len(coordinates) - 1):
            state = states[-1]
            q = quadratic * (1.0 + 0.10 * node) if model == "teacher" else np.zeros_like(quadratic)
            velocity = state @ linear + np.einsum("jik,i,k->j", q, state, state)
            delta = (coordinates[node + 1] - coordinates[node]) * velocity
            target = state + delta
            transitions.append(TransitionObservation(node, node + 1, coordinates[node], coordinates[node + 1], state, target, delta, velocity))
            states.append(target)
        traces.append(TrajectoryTrace(model, f"{split}-{index}", tuple(range(-1, len(coordinates) - 1)), coordinates, np.asarray(states), None, tuple(transitions), {"probe_split": split}))
    return traces


def test_full_flow_wins_on_disjoint_nonlinear_ground_truth():
    rng = np.random.default_rng(701)
    quadratic = np.zeros((3, 3, 3))
    quadratic[0, 0, 0] = 0.12
    quadratic[1, 1, 2] = -0.09
    quadratic[2, 0, 1] = 0.07
    datasets = {}
    for split, count in (("train", 70), ("validation", 50), ("holdout", 50)):
        initial = rng.uniform(-0.7, 0.7, size=(count, 3))
        datasets[split] = (
            _quadratic_traces(initial, split, "student", quadratic),
            _quadratic_traces(initial, split, "teacher", quadratic),
        )
    train_student, train_teacher = datasets["train"]
    alignment = fit_alignment(
        np.asarray([trace.hidden_states[0] for trace in train_teacher]),
        np.asarray([trace.hidden_states[0] for trace in train_student]),
        kind="affine",
        source_role="teacher",
        target_role="student",
    )
    validation_student, validation_teacher = datasets["validation"]
    holdout_student, holdout_teacher = datasets["holdout"]
    report = run_transfer_scorecard(
        train_student,
        train_teacher,
        alignment,
        validation_student=validation_student,
        validation_teacher=validation_teacher,
        holdout_student=holdout_student,
        holdout_teacher=holdout_teacher,
        include_stress=False,
    )
    entries = {entry.name: entry for entry in report.entries}
    assert entries["full_flow"].holdout_error is not None
    assert entries["full_flow"].holdout_error < entries["delta_flow"].holdout_error
    assert entries["full_flow"].holdout_error < entries["baseline_student_flow"].holdout_error
    assert entries["curvature_jacobian"].holdout_error < entries["baseline_student_flow"].holdout_error
    assert entries["full_flow"].status == "improved"
    assert entries["full_flow"].metadata["fit_signature_mode"] == "full"
