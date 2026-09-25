"""Focused post-fit consistency diagnostics.

This module does not collect observations and does not relax any guard.  It
explains whether a fit is internally consistent, and separates an
initial-state chart match from whole-trajectory transport evidence.  The
output is an operational diagnostic; it is not a semantic or causal claim.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .types import AlignmentResult, FlowFitResult, FlowOperator, TrajectoryTrace, stable_l2


FORENSIC_SCHEMA = "faytuna-forensic-audit-v1"


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if is_dataclass(value):
        return _safe(asdict(value))
    return value


def _stats(values: Any) -> dict[str, Any]:
    if values is None:
        return {"count": 0, "mean": None, "min": None, "max": None, "finite": False}
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "finite": False}
    finite = bool(np.all(np.isfinite(array)))
    if not finite:
        return {"count": int(array.size), "mean": None, "min": None, "max": None, "finite": False}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "finite": True,
    }


def operator_fingerprint(operator: FlowOperator) -> str:
    """Return the stable identity used to compare fit and scorecard targets."""

    digest = hashlib.sha256()
    for value in (operator.coordinates, operator.matrices, operator.biases, operator.sample_counts, operator.residual_scales):
        array = np.asarray(value, dtype=np.float64)
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    if operator.quadratic_terms is not None:
        digest.update(repr(operator.quadratic_terms.shape).encode("ascii"))
        digest.update(np.asarray(operator.quadratic_terms, dtype=np.float64).tobytes())
    if operator.chart_projection is not None:
        digest.update(repr(operator.chart_projection.shape).encode("ascii"))
        digest.update(np.asarray(operator.chart_projection, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _trajectory_geometry_error(student: Sequence[TrajectoryTrace], teacher: Sequence[TrajectoryTrace], alignment: AlignmentResult) -> float:
    errors: list[float] = []
    if len(student) != len(teacher):
        return float("nan")
    for student_trace, teacher_trace in zip(student, teacher):
        if student_trace.probe_id != teacher_trace.probe_id:
            return float("nan")
        mapped = alignment.apply(teacher_trace.hidden_states, depth=teacher_trace.depth_coordinates)
        target = np.column_stack([
            np.interp(student_trace.depth_coordinates, teacher_trace.depth_coordinates, mapped[:, column])
            for column in range(mapped.shape[1])
        ])
        numerator = float(stable_l2(student_trace.hidden_states - target, name="forensic trajectory residual"))
        denominator = max(1.0, float(stable_l2(student_trace.hidden_states, name="forensic student trajectory")), float(stable_l2(target, name="forensic mapped teacher trajectory")))
        errors.append(numerator / denominator)
    return float(np.mean(errors)) if errors else float("nan")


def _scorecard_dict(scorecard: Any) -> Mapping[str, Any] | None:
    if scorecard is None:
        return None
    if isinstance(scorecard, Mapping):
        return scorecard
    to_dict = getattr(scorecard, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return value if isinstance(value, Mapping) else None
    return None


def audit_flow_fit(
    fit: FlowFitResult,
    alignment: AlignmentResult,
    *,
    train_student: Sequence[TrajectoryTrace] | None = None,
    train_teacher: Sequence[TrajectoryTrace] | None = None,
    scorecard: Any | None = None,
    method: str = "full_flow",
) -> dict[str, Any]:
    """Build a strict forensic report for an existing fit.

    ``train_student`` and ``train_teacher`` are optional and are only used to
    recompute a normalized whole-trajectory geometry diagnostic.  No model
    forward pass is performed.  When a scorecard is supplied, target identity
    is checked against its single train-fitted reference target.
    """

    metadata = dict(fit.metadata)
    depth_items = [item for item in metadata.get("signature_scores", ()) if isinstance(item, Mapping)]
    depth_summary = dict(metadata.get("depth_correspondence_summary", {}))
    if depth_items:
        for key in ("matched_nodes", "student_coverage", "teacher_coverage", "matched_fraction", "gap_count", "correspondence_cost", "correspondence_confidence"):
            if key not in depth_summary:
                depth_summary[f"mean_{key}"] = float(np.mean([float(item[key]) for item in depth_items if key in item])) if any(key in item for item in depth_items) else None
        consistency = [bool(item.get("gap_count_consistent")) for item in depth_items if "gap_count_consistent" in item]
        depth_summary.setdefault("all_gap_counts_consistent", bool(consistency) and all(consistency))
        depth_summary["probe_count"] = len(depth_items)
        if fit.depth_report is not None:
            reported_gaps = int(fit.depth_report.gap_count)
            observed_gaps = int(sum(int(item.get("gap_count", 0)) for item in depth_items))
            depth_summary["aggregate_gap_count_matches"] = reported_gaps == observed_gaps
            reported_matches = float(fit.depth_report.matched_nodes)
            observed_matches = float(sum(float(item.get("matched_nodes", 0)) for item in depth_items))
            depth_summary["aggregate_matched_count_matches"] = bool(np.isclose(reported_matches, observed_matches, rtol=0.0, atol=1e-12))
    report: dict[str, Any] = {
        "schema_version": FORENSIC_SCHEMA,
        "status": "consistent",
        "claims": {
            "semantic_transfer": "not measured",
            "causal_effect": "not measured",
            "operational_proxy": "whole-trajectory chart and transported-flow consistency only",
        },
        "alignment": {
            "direction": metadata.get("alignment_direction", alignment.metadata.get("direction")),
            "source_dim": alignment.source_dim,
            "target_dim": alignment.target_dim,
            "fit_scope": metadata.get("alignment_fit_scope", alignment.metadata.get("alignment_fit_scope", "unspecified")),
            "paired_error": alignment.paired_error,
            "relational_error": alignment.relational_error,
            "condition_number": alignment.condition_number,
            "initial_state_only_warning": metadata.get("alignment_fit_scope", alignment.metadata.get("alignment_fit_scope")) == "initial_state_only",
        },
        "flow_output": {
            "role": "train_fitted_transported_teacher_target",
            "target_fit_split": fit.transported_teacher.metadata.get("target_fit_split", metadata.get("target_fit_split", "unspecified")),
            "metric_scope": fit.transported_teacher.metadata.get("metric_scope", metadata.get("metric_scope", "unspecified")),
            "operator_fingerprint": operator_fingerprint(fit.transported_teacher),
            "student_operator_shape": list(fit.student.matrices.shape),
            "correction_shape": list(fit.correction_matrices.shape),
            "requested_signature_mode": metadata.get("requested_signature_mode", metadata.get("signature_mode")),
            "effective_signature_mode": metadata.get("effective_signature_mode", metadata.get("signature_mode")),
            "training_student_flow_error": fit.validation_error,
        },
        "depth_correspondence": {
            **depth_summary,
            "report_matched_fraction": None if fit.depth_report is None else fit.depth_report.matched_fraction,
            "report_gap_count": None if fit.depth_report is None else fit.depth_report.gap_count,
            "report_student_nodes": None if fit.depth_report is None else fit.depth_report.student_nodes,
            "report_teacher_nodes": None if fit.depth_report is None else fit.depth_report.teacher_nodes,
        },
        "confidence": {
            "final": _stats(fit.confidence),
            "factors": {
                "support": _stats(metadata.get("support_factor")),
                "student_node_gap": _stats(metadata.get("gap_confidence")),
                "depth_interpolation": _stats([metadata.get("depth_interpolation_confidence")]) if metadata.get("depth_interpolation_confidence") is not None else _stats(None),
                "residual": _stats(metadata.get("residual_confidence_factor")),
                "capacity": _stats([metadata.get("capacity_gate")]) if metadata.get("capacity_gate") is not None else _stats(None),
                "tangent": _stats(metadata.get("local_tangent_gate")),
                "stability": _stats(metadata.get("stability_factor")),
                "calibration": _stats(metadata.get("calibration_factor")),
            },
            "calibrated": bool(metadata.get("confidence_calibrated", False)),
            "calibration_chart_reused": bool(metadata.get("scalable_backend") == "randomized_latent_compression" and metadata.get("scalable_chart_projection_source") == "reused_train_chart"),
        },
        "capacity": None if fit.capacity_diagnostics is None else {
            "student_state_rank": fit.capacity_diagnostics.student_state_rank,
            "teacher_state_rank": fit.capacity_diagnostics.teacher_state_rank,
            "state_subspace_coverage": fit.capacity_diagnostics.state_subspace_coverage,
            "velocity_subspace_coverage": fit.capacity_diagnostics.velocity_subspace_coverage,
            "irreducible_mismatch": fit.capacity_diagnostics.irreducible_mismatch,
            "bottleneck": fit.capacity_diagnostics.bottleneck,
            "confidence_penalty": fit.capacity_diagnostics.confidence_penalty,
        },
    }

    findings: list[dict[str, Any]] = []
    if report["alignment"]["initial_state_only_warning"]:
        findings.append({"level": "explanation", "code": "alignment_scope_initial_only", "message": "low paired/relational alignment error describes the fitted initial-state chart; it does not constrain hidden trajectories at later depths"})
    if depth_summary.get("all_gap_counts_consistent") is False:
        findings.append({"level": "error", "code": "gap_count_arithmetic_inconsistent", "message": "reported gap counts do not agree with the dynamic-programming path arithmetic"})
    if depth_summary.get("aggregate_gap_count_matches") is False or depth_summary.get("aggregate_matched_count_matches") is False:
        findings.append({"level": "error", "code": "aggregate_depth_report_mismatch", "message": "the aggregate depth report does not match its per-probe correspondence records"})
    teacher_coverage = depth_summary.get("mean_teacher_coverage")
    if teacher_coverage is not None and teacher_coverage < 0.25:
        findings.append({"level": "limitation", "code": "low_teacher_depth_coverage", "value": teacher_coverage, "message": "only a small fraction of teacher depth nodes are exact correspondence anchors; continuous interpolation is therefore weakly supported"})
    confidence_mean = report["confidence"]["final"].get("mean")
    if confidence_mean is not None and confidence_mean < 0.10:
        factors = report["confidence"]["factors"]
        means = [(name, value.get("mean")) for name, value in factors.items() if value.get("mean") is not None]
        means.sort(key=lambda pair: pair[1])
        findings.append({"level": "explanation", "code": "low_confidence_is_factorized", "value": confidence_mean, "lowest_factors": [{"name": name, "mean": value} for name, value in means[:3]], "message": "confidence is the product of support, correspondence, capacity, residual, stability, and optional calibration evidence; it is not inferred from paired alignment alone"})
    if report["capacity"] is not None and report["capacity"]["bottleneck"]:
        findings.append({"level": "limitation", "code": "student_capacity_bottleneck", "message": "the measured student chart cannot express all transported teacher variation; the guard is expected to reduce or reject correction"})

    scorecard_value = _scorecard_dict(scorecard)
    scorecard_check: dict[str, Any] = {"status": "unavailable", "method": method}
    if scorecard_value is not None:
        scorecard_metadata = dict(scorecard_value.get("metadata", {}))
        reference_fingerprint = scorecard_metadata.get("reference_target_operator_fingerprint")
        entries = [item for item in scorecard_value.get("entries", ()) if isinstance(item, Mapping)]
        method_entry = next((item for item in entries if item.get("name") == method), None)
        entry_fingerprint = None if method_entry is None else dict(method_entry.get("metadata", {})).get("evaluation_target_operator_fingerprint")
        comparable = [dict(item.get("metadata", {})).get("evaluation_target_operator_fingerprint") for item in entries if dict(item.get("metadata", {})).get("evaluation_target_operator_fingerprint") is not None]
        all_same = bool(comparable) and len(set(comparable)) == 1
        flow_fingerprint = report["flow_output"]["operator_fingerprint"]
        scorecard_check = {
            "status": "consistent" if reference_fingerprint == flow_fingerprint and (entry_fingerprint in {None, reference_fingerprint}) and (not comparable or all_same) else "mismatch",
            "method": method,
            "flow_target_fingerprint": flow_fingerprint,
            "scorecard_reference_target_fingerprint": reference_fingerprint,
            "scorecard_method_target_fingerprint": entry_fingerprint,
            "scorecard_entries_share_target": all_same,
            "scorecard_status": scorecard_value.get("status"),
            "scorecard_decision_status": scorecard_value.get("decision_status"),
        }
        if scorecard_check["status"] == "mismatch":
            findings.append({"level": "error", "code": "scorecard_target_mismatch", "message": "the supplied scorecard does not identify the same train-fitted target operator as this fit"})
        if method_entry is not None and method_entry.get("status") == "unchanged" and int(method_entry.get("accepted_nodes", 0) or 0) == 0:
            findings.append({"level": "decision", "code": "guard_rollback_expected", "message": "the scorecard kept the student baseline because no correction node passed the disjoint baseline guard"})
    report["scorecard_consistency"] = scorecard_check
    report["findings"] = findings
    report["status"] = "inconsistent" if any(item.get("level") == "error" for item in findings) else "consistent"
    if report["status"] == "consistent" and any(item.get("level") == "limitation" for item in findings):
        report["status"] = "consistent_with_bottleneck"
    return _safe(report)
