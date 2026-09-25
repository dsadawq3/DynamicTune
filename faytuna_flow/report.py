"""Human-readable diagnostics assembled from measured artifact fields."""

from __future__ import annotations

from typing import Any, Sequence

import json
import numpy as np

from .depth import detect_bifurcations
from .types import AlignmentResult, FlowFitResult, TrajectoryTrace, stable_l2


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def make_report(traces: Sequence[TrajectoryTrace], *, alignment: AlignmentResult | None = None, flow_fit: FlowFitResult | None = None) -> dict[str, Any]:
    bifurcations = detect_bifurcations(traces)
    flow = None
    if flow_fit is not None:
        flow = {
            "training_student_flow_error": flow_fit.validation_error,
            "mean_confidence": float(np.mean(flow_fit.confidence)),
            "max_correction_norm": float(np.max(stable_l2(flow_fit.correction_matrices, axis=(1, 2), name="report correction"))),
            "quadratic_correction_norm": 0.0 if flow_fit.correction_quadratic is None else float(stable_l2(flow_fit.correction_quadratic, name="report quadratic correction")),
            "metadata": {key: value for key, value in flow_fit.metadata.items() if isinstance(value, (str, int, float, bool))},
            "depth": None if flow_fit.depth_report is None else {
                "student_nodes": flow_fit.depth_report.student_nodes,
                "teacher_nodes": flow_fit.depth_report.teacher_nodes,
                "matched_nodes": flow_fit.depth_report.matched_nodes,
                "gap_count": flow_fit.depth_report.gap_count,
                "dropped_transitions": flow_fit.depth_report.dropped_transitions,
                "matched_fraction": flow_fit.depth_report.matched_fraction,
                "mean_gap_confidence": flow_fit.depth_report.mean_gap_confidence,
                "notes": list(flow_fit.depth_report.notes),
            },
            "capacity": None if flow_fit.capacity_diagnostics is None else {
                "student_state_rank": flow_fit.capacity_diagnostics.student_state_rank,
                "teacher_state_rank": flow_fit.capacity_diagnostics.teacher_state_rank,
                "student_tangent_rank": flow_fit.capacity_diagnostics.student_tangent_rank,
                "teacher_velocity_rank": flow_fit.capacity_diagnostics.teacher_velocity_rank,
                "state_subspace_coverage": flow_fit.capacity_diagnostics.state_subspace_coverage,
                "velocity_subspace_coverage": flow_fit.capacity_diagnostics.velocity_subspace_coverage,
                "residual_transport_error": flow_fit.capacity_diagnostics.residual_transport_error,
                "irreducible_mismatch": flow_fit.capacity_diagnostics.irreducible_mismatch,
                "map_condition_number": flow_fit.capacity_diagnostics.map_condition_number,
                "bottleneck": flow_fit.capacity_diagnostics.bottleneck,
                "confidence_penalty": flow_fit.capacity_diagnostics.confidence_penalty,
                "notes": list(flow_fit.capacity_diagnostics.notes),
            },
        }
    return {
        "trace_count": len(traces),
        "model_ids": sorted({t.model_id for t in traces}),
        "state_dimension": traces[0].state_dim if traces else 0,
        "layer_counts": sorted({t.layer_count for t in traces}),
        "mean_path_length": float(np.mean([stable_l2(np.diff(t.hidden_states, axis=0), axis=1, name="report path deltas").sum() for t in traces])) if traces else 0.0,
        "bifurcations": [{"depth": p.depth, "level": p.level, "score": p.score, "confidence": p.confidence, "candidate": True, "status": "hypothesis_requires_causal_intervention", "evidence": dict(p.evidence)} for p in bifurcations],
        "alignment": None if alignment is None else {"kind": alignment.kind, "paired_error": alignment.paired_error, "relational_error": alignment.relational_error, "cycle_error": alignment.cycle_error, "condition_number": alignment.condition_number},
        "flow": flow,
    }


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False)
