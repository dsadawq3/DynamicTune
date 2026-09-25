import json
from pathlib import Path

import numpy as np

from faytuna_flow.assessor import assess_correction
from faytuna_flow.adaptive_transfer import AdaptiveTransferPolicy, run_adaptive_transfer
from faytuna_flow.flow import fit_flow_transfer
from faytuna_flow.geometry import fit_alignment
from faytuna_flow.journal import ExperimentJournal, JOURNAL_SCHEMA, journal_math_profile, journal_scorecard, journal_traces
from faytuna_flow.math_profile import MATH_COMPONENTS
from faytuna_flow.pipeline import collect_synthetic_pair


def _paired_fit(*, mode: str = "full", components=None):
    bundle = collect_synthetic_pair(teacher_dim=5, student_dim=3, teacher_layers=6, student_layers=4, per_family=3, seed=902, split="train")
    alignment = fit_alignment(
        np.asarray([trace.hidden_states[0] for trace in bundle.teacher]),
        np.asarray([trace.hidden_states[0] for trace in bundle.student]),
        kind="low_rank",
        rank=3,
        source_role="teacher",
        target_role="student",
    )
    return bundle, fit_flow_transfer(bundle.student, bundle.teacher, alignment, signature_mode=mode, math_components=components)


def test_activation_profile_has_every_component_and_full_is_not_a_noop():
    _, none = _paired_fit(mode="none")
    _, differential = _paired_fit(mode="differential")
    _, full = _paired_fit(mode="full")
    profile = full.metadata["math_profile"]
    assert tuple(profile["components"]) == MATH_COMPONENTS
    assert profile["no_op_flags_rejected"] is True
    assert profile["components"]["path_signature"]["applied"] is True
    assert profile["components"]["hessian_2jet"]["applied"] is True
    assert none.metadata["math_profile"]["components"]["path_signature"]["applied"] is False
    assert differential.metadata["math_profile"]["components"]["hessian_2jet"]["applied"] is False
    assert full.correction_quadratic is not None
    assert not np.allclose(full.correction_quadratic, 0.0)
    assert full.metadata["state_feature_degree"] != none.metadata["state_feature_degree"]


def test_explicit_component_disables_change_calculation_and_are_audited():
    disabled = {name: False for name in MATH_COMPONENTS if name not in {"alignment", "continuous_depth"}}
    _, default = _paired_fit(mode="full")
    _, candidate = _paired_fit(mode="full", components=disabled)
    assert candidate.correction_quadratic is None
    assert candidate.metadata["robust_loss"] == "none"
    assert candidate.metadata["deterministic_stability_seeds"] == []
    assert candidate.metadata["capacity_gate"] == 1.0
    for name, audit in candidate.metadata["math_profile"]["components"].items():
        if name in {"alignment", "continuous_depth"}:
            assert audit["applied"] is True
        else:
            assert audit["applied"] is False
            assert audit["skipped_reason"]
    assert not np.allclose(default.correction_matrices, candidate.correction_matrices)


def test_no_op_protocol_flags_fail_fast():
    bundle, _ = _paired_fit(mode="none")
    alignment = fit_alignment(
        np.asarray([trace.hidden_states[0] for trace in bundle.teacher]),
        np.asarray([trace.hidden_states[0] for trace in bundle.student]),
        kind="low_rank",
        rank=3,
        source_role="teacher",
        target_role="student",
    )
    for component in ("alignment", "continuous_depth"):
        try:
            fit_flow_transfer(bundle.student, bundle.teacher, alignment, math_components={component: False})
        except ValueError as error:
            assert "no-op" in str(error)
        else:
            raise AssertionError(f"{component} disable unexpectedly became a silent no-op")


def test_journal_is_deterministic_strict_and_contains_math_events(tmp_path: Path):
    paths = []
    bundle, fit = _paired_fit(mode="full")
    for index in range(2):
        path = tmp_path / f"run-{index}.jsonl"
        summary = tmp_path / f"run-{index}.summary.txt"
        journal = ExperimentJournal(path, summary, run_id="deterministic", seed=19)
        journal_traces(journal, bundle.student[:1], model_role="student")
        journal_math_profile(journal, fit.metadata["math_profile"], stage="fit-flow")
        journal_scorecard(journal, {"metadata": {"math_profiles": {}, "math_component_counterfactual_table": {"quadratic_flow": {"status": "scored", "metric_delta": 0.25}}}})
        journal.record("fit-flow", "test", payload={"accepted_count": 2, "rollback_count": 1, "nonfinite_probe": float("nan")})
        journal.close()
        paths.append(path)
    assert paths[0].read_text(encoding="utf-8") == paths[1].read_text(encoding="utf-8")
    lines = paths[0].read_text(encoding="utf-8").splitlines()
    assert lines
    assert all(json.loads(line)["schema_version"] == JOURNAL_SCHEMA for line in lines)
    assert all("NaN" not in line and "Infinity" not in line for line in lines)
    assert any(json.loads(line)["event_type"] == "math-component" for line in lines)
    assert any(json.loads(line)["event_type"] == "math-component-counterfactual" for line in lines)
    assert "accepted_count: 2" in (tmp_path / "run-0.summary.txt").read_text(encoding="utf-8")
    assert "rollback_count: 1" in (tmp_path / "run-0.summary.txt").read_text(encoding="utf-8")
    assert "component_counterfactual: quadratic_flow; status=scored; metric_delta=0.25" in (tmp_path / "run-0.summary.txt").read_text(encoding="utf-8")


def test_assessor_exposes_direct_features_prediction_calibration_and_disagreement():
    assessment = assess_correction(
        {"holdout_error": 0.60, "dynamic_functional_error": 0.50},
        {"holdout_error": 0.45, "dynamic_functional_error": 0.43, "smoothness": 0.90, "stability": 0.85, "observable_effect": 0.30, "teacher_consistency": 0.80},
        uncertainty=0.15,
        validation={"improvement": 0.10},
    )
    payload = assessment.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["assessor"] == "surprise_novelty_scorer"
    assert payload["scorer_type"] == "fixed_direct_metric_scorer"
    assert payload["trained_model"] is False
    assert payload["model_checksum"] is None
    assert len(payload["feature_vector"]) == len(payload["feature_names"]) == 6
    assert payload["calibration"] == 1.0
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["limitations"]


def test_adaptive_transfer_is_explicit_and_accounts_for_whole_block_rollback():
    train, fit = _paired_fit(mode="full")
    validation = collect_synthetic_pair(teacher_dim=5, student_dim=3, teacher_layers=6, student_layers=4, per_family=3, seed=903, split="validation")
    holdout = collect_synthetic_pair(teacher_dim=5, student_dim=3, teacher_layers=6, student_layers=4, per_family=3, seed=904, split="holdout")
    alignment = fit_alignment(
        np.asarray([trace.hidden_states[0] for trace in train.teacher]),
        np.asarray([trace.hidden_states[0] for trace in train.student]),
        kind="low_rank",
        rank=3,
        source_role="teacher",
        target_role="student",
    )
    report = run_adaptive_transfer(
        fit,
        fit.transported_teacher,
        alignment=alignment,
        validation_student=validation.student,
        validation_teacher=validation.teacher,
        holdout_student=holdout.student,
        holdout_teacher=holdout.teacher,
        config=AdaptiveTransferPolicy(mode="safe", gain_grid=(0.0, 0.5), block_size=2, min_observable_effect=1e9),
    )
    payload = report.to_dict()
    json.dumps(payload, allow_nan=False)
    assert report.mode == "safe"
    assert report.steps[0].status == "baseline"
    assert report.status in {"unchanged", "accepted"}
    assert report.metadata["math_profile"]["components"]["gain_schedule"]["applied"] is True
    step = report.steps[-1]
    assert step.rollback_blocks or step.ineffective_blocks
    assert step.metadata["assessor"]["assessor"] == "surprise_novelty_scorer"


def test_adaptive_transfer_without_holdout_never_applies():
    train, fit = _paired_fit(mode="full")
    alignment = fit_alignment(
        np.asarray([trace.hidden_states[0] for trace in train.teacher]),
        np.asarray([trace.hidden_states[0] for trace in train.student]),
        kind="low_rank",
        rank=3,
        source_role="teacher",
        target_role="student",
    )
    report = run_adaptive_transfer(
        fit,
        fit.transported_teacher,
        alignment=alignment,
        validation_student=None,
        validation_teacher=None,
        holdout_student=None,
        holdout_teacher=None,
        config=AdaptiveTransferPolicy(mode="safe", gain_grid=(0.0, 1.0)),
    )
    assert report.status == "insufficient_holdout"
    assert report.holdout_used is False
    json.dumps(report.to_dict(), allow_nan=False)


def test_adaptive_policy_selects_strategy_and_keeps_holdout_as_final_verdict():
    train, fit = _paired_fit(mode="full")
    validation = collect_synthetic_pair(teacher_dim=5, student_dim=3, teacher_layers=6, student_layers=4, per_family=3, seed=903, split="validation")
    holdout = collect_synthetic_pair(teacher_dim=5, student_dim=3, teacher_layers=6, student_layers=4, per_family=3, seed=904, split="holdout")
    alignment = fit_alignment(
        np.asarray([trace.hidden_states[0] for trace in train.teacher]),
        np.asarray([trace.hidden_states[0] for trace in train.student]),
        kind="low_rank",
        rank=3,
        source_role="teacher",
        target_role="student",
    )
    report = AdaptiveTransferPolicy(mode="safe", block_size=2).tune(
        fit,
        fit.transported_teacher,
        alignment=alignment,
        validation_student=validation.student,
        validation_teacher=validation.teacher,
        holdout_student=holdout.student,
        holdout_teacher=holdout.teacher,
    )
    selection = report.metadata["policy_selection"]
    assert report.mode == "safe"
    assert selection["backend"] in {"dense", "scalable"}
    assert selection["depth_strategy"] in {"existing_monotone", "gap_aware_monotone"}
    assert selection["line_search"] == "backtracking"
    assert selection["tensor_families"]
    assert report.metadata["holdout_is_final_verdict"] is True
    assert report.metadata["teacher_at_inference"] is False
    json.dumps(report.to_dict(), allow_nan=False)


def test_adaptive_journal_event_has_new_policy_contract(tmp_path: Path):
    journal = ExperimentJournal(tmp_path / "adaptive.jsonl", run_id="adaptive", seed=7)
    from faytuna_flow.journal import journal_adaptive
    journal_adaptive(journal, {"status": "insufficient_validation", "mode": "safe"})
    journal.close()
    events = [json.loads(line) for line in (tmp_path / "adaptive.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["event_type"] == "adaptive-report"
