import json
from dataclasses import replace

import numpy as np
import pytest

from faytuna_flow.artifacts import save_alignment, save_traces
from faytuna_flow.cli import main
from faytuna_flow.flow import fit_flow_transfer
from faytuna_flow.geometry import fit_alignment
from faytuna_flow.model_families import GPT2_XL_TO_SMALL, preflight_gpt2_pair
from faytuna_flow.pipeline import collect_synthetic_pair
from faytuna_flow.scorecard import _perturb_traces, fit_target_operator, run_transfer_scorecard


def _train_and_alignment():
    train = collect_synthetic_pair(teacher_dim=4, student_dim=3, teacher_layers=5, student_layers=3, per_family=3, seed=1201, split="train")
    alignment = fit_alignment(
        np.asarray([trace.hidden_states[0] for trace in train.teacher]),
        np.asarray([trace.hidden_states[0] for trace in train.student]),
        kind="low_rank",
        rank=3,
        source_role="teacher",
        target_role="student",
    )
    return train, alignment


def test_target_operator_is_train_fitted_and_evaluation_changes_cannot_leak():
    train, alignment = _train_and_alignment()
    validation = collect_synthetic_pair(teacher_dim=4, student_dim=3, teacher_layers=5, student_layers=3, per_family=3, seed=1201, split="validation")
    holdout = collect_synthetic_pair(teacher_dim=4, student_dim=3, teacher_layers=5, student_layers=3, per_family=3, seed=1201, split="holdout")
    first = run_transfer_scorecard(train.student, train.teacher, alignment, validation_student=validation.student, validation_teacher=validation.teacher, holdout_student=holdout.student, holdout_teacher=holdout.teacher, seed=4, include_stress=False)
    changed_validation = _perturb_traces(validation.student, noise=2.0, seed=7)
    changed_validation_teacher = _perturb_traces(validation.teacher, noise=2.0, seed=8)
    changed_holdout = _perturb_traces(holdout.student, noise=2.0, seed=9)
    changed_holdout_teacher = _perturb_traces(holdout.teacher, noise=2.0, seed=10)
    second = run_transfer_scorecard(train.student, train.teacher, alignment, validation_student=changed_validation, validation_teacher=changed_validation_teacher, holdout_student=changed_holdout, holdout_teacher=changed_holdout_teacher, seed=4, include_stress=False)
    assert first.metadata["target_fit_split"] == second.metadata["target_fit_split"] == "train"
    assert first.metadata["metric_scope"] == "transported_teacher_operator_approximation"
    assert first.metadata["target_operator_fingerprints"] == second.metadata["target_operator_fingerprints"]
    counterfactuals = first.metadata["math_component_counterfactual_table"]
    assert set(counterfactuals) == {
        "alignment", "ot", "continuous_depth", "path_signature", "local_1jet",
        "hessian_2jet", "curvature", "robust_loss", "tangent_projector_transport",
        "capacity_projection", "quadratic_flow", "stability_barriers", "gain_schedule",
    }
    assert counterfactuals["alignment"]["status"] == "unavailable"
    assert counterfactuals["alignment"]["metric_delta"] is None
    assert counterfactuals["path_signature"]["same_holdout_split"] is True
    assert counterfactuals["path_signature"]["metric_delta"] is not None
    assert first.metadata["math_contribution_table"]["full_flow"]["component_deltas_available"] is False
    assert all(item["metric_delta"] is None for item in first.metadata["math_contribution_table"]["full_flow"]["components"].values())
    direct = fit_target_operator(train.student, train.teacher, alignment, "full")
    assert direct.metadata["target_fit_split"] == "train"
    assert direct.metadata["target_operator_fingerprint"] == first.metadata["target_operator_fingerprints"]["full_flow"]
    entries = {entry.name: entry for entry in first.entries}
    reference = first.metadata["reference_target_operator_fingerprint"]
    assert reference == entries["full_flow"].metadata["evaluation_target_operator_fingerprint"]
    assert all(
        entry.metadata.get("evaluation_target_operator_fingerprint") in {None, reference}
        for entry in first.entries
    )


def test_scorecard_without_disjoint_holdout_is_strict_and_incomplete():
    train, alignment = _train_and_alignment()
    report = run_transfer_scorecard(train.student, train.teacher, alignment, include_stress=False)
    payload = report.to_dict()
    rendered = json.dumps(payload, allow_nan=False)
    assert "NaN" not in rendered and "Infinity" not in rendered
    assert payload["status"] == "incomplete"
    assert payload["decision_status"] == "insufficient_disjoint_holdout"
    assert payload["holdout_used"] is False
    assert payload["metadata"]["component_counterfactuals_available"] is False
    assert all(item["status"] == "unavailable" for item in payload["metadata"]["math_component_counterfactual_table"].values())


def test_scorecard_rejects_exact_trace_duplication_hidden_behind_new_probe_ids():
    train, alignment = _train_and_alignment()
    duplicated_student = [replace(train.student[0], probe_id="validation-renamed")]
    duplicated_teacher = [replace(train.teacher[0], probe_id="validation-renamed")]
    with pytest.raises(ValueError, match="exact observed teacher/student trace content duplicated"):
        run_transfer_scorecard(
            train.student,
            train.teacher,
            alignment,
            validation_student=duplicated_student,
            validation_teacher=duplicated_teacher,
            include_stress=False,
        )


def test_scorecard_cli_reports_duplicate_split_as_strict_structured_json(tmp_path, capsys):
    train, alignment = _train_and_alignment()
    student_path = save_traces(train.student, tmp_path / "student")
    teacher_path = save_traces(train.teacher, tmp_path / "teacher")
    validation_student = [replace(train.student[0], probe_id="validation-renamed")]
    validation_teacher = [replace(train.teacher[0], probe_id="validation-renamed")]
    validation_student_path = save_traces(validation_student, tmp_path / "validation-student")
    validation_teacher_path = save_traces(validation_teacher, tmp_path / "validation-teacher")
    alignment_path = save_alignment(alignment, tmp_path / "alignment.json")
    code = main([
        "ablate",
        "--student", str(student_path),
        "--teacher", str(teacher_path),
        "--alignment", str(alignment_path),
        "--validation-student", str(validation_student_path),
        "--validation-teacher", str(validation_teacher_path),
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "rejected"
    assert payload["decision_status"] == "invalid_split_or_input"
    json.dumps(payload, allow_nan=False)


def test_scorecard_cli_without_holdout_has_structured_exit_and_strict_json(tmp_path, capsys):
    train, alignment = _train_and_alignment()
    student_path = save_traces(train.student, tmp_path / "student.npz")
    teacher_path = save_traces(train.teacher, tmp_path / "teacher.npz")
    alignment_path = save_alignment(alignment, tmp_path / "alignment.json")
    code = main(["ablate", "--student", str(student_path), "--teacher", str(teacher_path), "--alignment", str(alignment_path)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "incomplete"
    json.dumps(payload, allow_nan=False)


def test_exact_gpt2_family_manifest_ids_and_architecture_preflight_are_truthful():
    assert GPT2_XL_TO_SMALL.teacher.repository_id == "openai-community/gpt2-xl"
    assert GPT2_XL_TO_SMALL.student.repository_id == "openai-community/gpt2"
    payload = preflight_gpt2_pair(
        {"model_type": "gpt2", "n_layer": 48, "n_embd": 1600, "n_head": 25, "n_positions": 1024, "vocab_size": 50257},
        {"model_type": "gpt2", "n_layer": 12, "n_embd": 768, "n_head": 12, "n_positions": 1024, "vocab_size": 50257},
    )
    assert payload["status"] == "pass"
    bad = preflight_gpt2_pair({"model_type": "llama", "n_layer": 48}, {"model_type": "gpt2"})
    assert bad["status"] == "rejected"
    json.dumps(bad, allow_nan=False)
