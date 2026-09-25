import json

import numpy as np

from faytuna_flow.forensics import FORENSIC_SCHEMA, audit_flow_fit
from faytuna_flow.geometry import fit_alignment
from faytuna_flow.pipeline import collect_synthetic_pair
from faytuna_flow.flow import fit_flow_transfer
from faytuna_flow.types import AlignmentResult


def _fit():
    bundle = collect_synthetic_pair(
        teacher_dim=5,
        student_dim=3,
        teacher_layers=6,
        student_layers=4,
        per_family=3,
        seed=901,
        split="train",
    )
    alignment = fit_alignment(
        np.asarray([trace.hidden_states[0] for trace in bundle.teacher]),
        np.asarray([trace.hidden_states[0] for trace in bundle.student]),
        kind="low_rank",
        rank=3,
        source_role="teacher",
        target_role="student",
    )
    alignment = AlignmentResult(
        alignment.kind,
        alignment.source_mean,
        alignment.target_mean,
        alignment.matrix,
        alignment.bias,
        alignment.source_rank,
        alignment.target_rank,
        alignment.paired_error,
        alignment.relational_error,
        alignment.cycle_error,
        alignment.condition_number,
        alignment.ot_mass_error,
        {**dict(alignment.metadata), "alignment_fit_scope": "initial_state_only"},
        alignment.source_projection,
        alignment.target_projection,
    )
    return bundle, alignment, fit_flow_transfer(bundle.student, bundle.teacher, alignment, signature_mode="differential")


def test_forensic_audit_separates_initial_chart_match_from_trajectory_and_is_strict_json():
    bundle, alignment, fit = _fit()
    report = audit_flow_fit(fit, alignment, train_student=bundle.student, train_teacher=bundle.teacher)
    json.dumps(report, allow_nan=False)
    assert report["schema_version"] == FORENSIC_SCHEMA
    assert report["alignment"]["initial_state_only_warning"] is True
    assert report["depth_correspondence"]["all_gap_counts_consistent"] is True
    assert report["depth_correspondence"]["mean_student_coverage"] is not None
    assert report["depth_correspondence"]["mean_teacher_coverage"] is not None
    assert set(report["confidence"]["factors"]) >= {"support", "residual", "stability", "capacity"}
    assert report["claims"]["semantic_transfer"] == "not measured"


def test_forensic_audit_marks_missing_scorecard_identity_as_unavailable():
    _, alignment, fit = _fit()
    report = audit_flow_fit(fit, alignment, scorecard=None)
    assert report["scorecard_consistency"]["status"] == "unavailable"
    json.dumps(report, allow_nan=False)
