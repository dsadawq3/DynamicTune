"""Deterministic JSONL experiment journal and human-readable summary.

The journal is an append-only, artifact-first observation layer.  It records
what each stage measured and decided without changing the numerical pipeline.
Wall-clock timestamps are opt-in so identical runs can be compared byte for
byte in tests and in reproducibility reports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .types import FlowFitResult, FlowOperator, TrajectoryTrace, stable_l2


JOURNAL_SCHEMA = "faytuna-experiment-journal-v1"


def _safe(value: Any, path: str, nonfinite: list[str]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item, f"{path}.{key}", nonfinite) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item, f"{path}[{index}]", nonfinite) for index, item in enumerate(value)]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist(), path, nonfinite)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isfinite(number):
            return number
        nonfinite.append(path)
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if is_dataclass(value):
        return _safe(asdict(value), path, nonfinite)
    try:
        return _safe(value.item(), path, nonfinite)
    except (AttributeError, ValueError, TypeError):
        return str(value)


def strict_json_value(value: Any) -> Any:
    """Return a strict JSON-safe value, mapping non-finite floats to null."""

    nonfinite: list[str] = []
    result = _safe(value, "$", nonfinite)
    json.dumps(result, allow_nan=False)
    return result


def _array_checksum(value: Any) -> str | None:
    try:
        array = np.asarray(value)
        if array.dtype == object:
            return None
        return hashlib.sha256(array.tobytes()).hexdigest()
    except (TypeError, ValueError):
        return None


def describe_model(model: Any, *, model_id: str | None = None, seed: int | None = None, checksum: bool = False) -> dict[str, Any]:
    """Describe an already constructed model without loading or downloading it."""

    result: dict[str, Any] = {"model_id": model_id or getattr(model, "model_id", None) or getattr(model, "name_or_path", None), "seed": seed}
    config = getattr(model, "config", None)
    if config is not None:
        try:
            result["config"] = strict_json_value(config.to_dict() if callable(getattr(config, "to_dict", None)) else config)
        except Exception:
            result["config"] = str(config)
    tensors: dict[str, Any] = {}
    state = None
    if callable(getattr(model, "state_dict", None)):
        state = model.state_dict()
    elif isinstance(model, Mapping):
        state = model
    if state is not None:
        for name, value in state.items():
            try:
                array = np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
                entry: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
                if checksum:
                    entry["sha256"] = _array_checksum(array)
                tensors[str(name)] = entry
            except (TypeError, ValueError):
                tensors[str(name)] = {"shape": None, "dtype": type(value).__name__}
    result["tensor_count"] = len(tensors)
    result["tensors"] = tensors
    result["checksum_requested"] = bool(checksum)
    return strict_json_value(result)


class ExperimentJournal:
    """Write deterministic structured events and a compact text summary."""

    def __init__(self, jsonl_path: str | Path, summary_path: str | Path | None = None, *, run_id: str = "run-0", seed: int = 0, common_metadata: Mapping[str, Any] | None = None, include_wallclock: bool = False):
        self.jsonl_path = Path(jsonl_path)
        self.summary_path = Path(summary_path) if summary_path is not None else self.jsonl_path.with_suffix(".summary.txt")
        self.run_id = str(run_id)
        self.seed = int(seed)
        self.common_metadata = dict(common_metadata or {})
        self.include_wallclock = bool(include_wallclock)
        self._events: list[dict[str, Any]] = []
        self._closed = False
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.write_text("", encoding="utf-8")

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)

    def record(self, stage: str, event_type: str, *, scope: str = "run", payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("experiment journal is already closed")
        nonfinite: list[str] = []
        event: dict[str, Any] = {
            "schema_version": JOURNAL_SCHEMA,
            "event_index": len(self._events),
            "run_id": self.run_id,
            "seed": self.seed,
            "stage": str(stage),
            "event_type": str(event_type),
            "scope": str(scope),
            **self.common_metadata,
            **dict(payload or {}),
        }
        if self.include_wallclock:
            from datetime import datetime, timezone
            event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        safe = _safe(event, "$", nonfinite)
        safe["finite_json"] = not bool(nonfinite)
        safe["nonfinite_fields"] = sorted(set(nonfinite))
        rendered = json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.jsonl_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered + "\n")
        self._events.append(safe)
        return safe

    def close(self) -> Path:
        if not self._closed:
            counts = Counter(str(event.get("stage")) for event in self._events)
            statuses = Counter(str(event.get("status")) for event in self._events if event.get("status") is not None)
            accepted = sum(int(event.get("accepted_count", 0) or 0) for event in self._events)
            rejected = sum(int(event.get("rejected_count", 0) or 0) for event in self._events)
            rollback = sum(int(event.get("rollback_count", 0) or 0) for event in self._events)
            nonfinite = sum(not bool(event.get("finite_json", False)) for event in self._events)
            counterfactual_lines = []
            for event in self._events:
                if event.get("event_type") != "math-component-counterfactual":
                    continue
                counterfactual_lines.append(
                    "component_counterfactual: "
                    f"{event.get('component')}; status={event.get('status')}; "
                    f"metric_delta={event.get('metric_delta')}"
                )
            lines = [
                "Faytuna Emergent Flow experiment journal",
                f"schema_version: {JOURNAL_SCHEMA}",
                f"run_id: {self.run_id}",
                f"seed: {self.seed}",
                f"events: {len(self._events)}",
                f"stages: {dict(sorted(counts.items()))}",
                f"statuses: {dict(sorted(statuses.items()))}",
                f"accepted_count: {accepted}",
                f"rejected_count: {rejected}",
                f"rollback_count: {rollback}",
                f"nonfinite_events: {nonfinite}",
            ]
            lines.extend(sorted(counterfactual_lines))
            self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._closed = True
        return self.summary_path

    def __enter__(self) -> "ExperimentJournal":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _trace_common(trace: TrajectoryTrace, model_role: str, include_arrays: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_role": model_role,
        "model_id": trace.model_id,
        "probe_id": trace.probe_id,
        "probe_split": trace.metadata.get("probe_split"),
        "probe_family": trace.metadata.get("probe_family"),
        "probe_pair_id": trace.metadata.get("probe_pair_id"),
        "perturbation_norm": trace.metadata.get("perturbation_norm"),
        "perturbation_metadata": {key: value for key, value in trace.metadata.items() if "perturb" in str(key).lower()},
        "feature_space": trace.feature_space,
        "layer_count": trace.layer_count,
        "state_shape": list(trace.hidden_states.shape),
        "state_dtype": str(trace.hidden_states.dtype),
        "depth_coordinates": trace.depth_coordinates,
        "token_ids": trace.token_ids,
        "position_ids": trace.position_ids,
        "token_position_map": trace.token_position_map,
        "trace_uncertainty": trace.uncertainty,
        "metadata": trace.metadata,
        "finite_hidden_states": bool(np.all(np.isfinite(trace.hidden_states))),
    }
    if include_arrays:
        payload["hidden_states"] = trace.hidden_states
        if trace.residual_states is not None:
            payload["residual_states"] = trace.residual_states
    return payload


def journal_traces(journal: ExperimentJournal, traces: Sequence[TrajectoryTrace], *, stage: str = "collect-trace", model_role: str = "model", model_metadata: Mapping[str, Any] | None = None, include_arrays: bool = False) -> None:
    for trace in traces:
        journal.record(stage, "probe", scope="probe", payload={**_trace_common(trace, model_role, include_arrays), "model_metadata": model_metadata or {}})
        for node, (layer_id, depth, state) in enumerate(zip(trace.layer_ids, trace.depth_coordinates, trace.hidden_states)):
            transition = trace.transitions[node] if node < len(trace.transitions) else None
            differential = trace.metadata.get("differential_observation", {})
            payload: dict[str, Any] = {
                "model_role": model_role,
                "model_id": trace.model_id,
                "probe_id": trace.probe_id,
                "probe_split": trace.metadata.get("probe_split"),
                "perturbation_norm": trace.metadata.get("perturbation_norm"),
                "node": node,
                "layer_id": layer_id,
                "depth": depth,
                "state_shape": list(state.shape),
                "state_norm": stable_l2(state, name="journal state norm"),
                "state_dtype": str(state.dtype),
                "uncertainty": None if trace.uncertainty is None else trace.uncertainty[node],
                "observation_backend": differential.get("backend"),
                "jacobian_kind": differential.get("jacobian_kind"),
                "jacobian_rank": differential.get("jacobian_rank"),
                "jacobian_directions": differential.get("jacobian_directions"),
                "hessian_kind": differential.get("hessian_kind"),
                "hessian_rank": differential.get("hessian_rank"),
                "estimated_forward_calls": differential.get("estimated_forward_calls"),
                "estimated_peak_memory_bytes": differential.get("estimated_peak_memory_bytes"),
                "observation_auto_downgraded": differential.get("auto_downgraded"),
                "path_signature_area_backend": trace.metadata.get("path_signature", {}).get("area_backend"),
                "path_signature_area_sketch_rank": trace.metadata.get("path_signature", {}).get("area_sketch_rank"),
            }
            if transition is not None:
                payload.update({
                    "transition_to_layer": transition.target_layer,
                    "transition_depth": transition.target_depth,
                    "transition_delta_norm": stable_l2(transition.delta, name="journal transition delta"),
                    "observed_velocity_norm": stable_l2(transition.vector_field, name="journal observed velocity"),
                    "jacobian_shape": None if transition.jacobian is None else list(transition.jacobian.shape),
                    "jacobian_singular_values": None if transition.jacobian is None else np.linalg.svd(transition.jacobian, compute_uv=False),
                    "hessian_shape": None if transition.hessian_sketch is None else list(transition.hessian_sketch.shape),
                    "hessian_kind": trace.metadata.get("hessian_kind", "directional_sketch" if transition.hessian_sketch is not None else None),
                    "curvature": transition.curvature,
                    "transition_singular_values": transition.singular_values,
                    "normalization_geometry": transition.normalization_geometry,
                    "attention_geometry": transition.attention_geometry,
                    "transition_uncertainty": transition.uncertainty,
                })
                if include_arrays:
                    payload.update({"state": state, "delta": transition.delta, "vector_field": transition.vector_field})
            journal.record(stage, "depth", scope="depth", payload=payload)


def journal_flow_fit(journal: ExperimentJournal, fit: FlowFitResult, *, stage: str = "fit-flow") -> None:
    journal.record(stage, "fit", scope="run", payload={
        "status": "fitted",
        "student_shape": list(fit.student.matrices.shape),
        "transported_teacher_shape": list(fit.transported_teacher.matrices.shape),
        "student_state_dim": fit.student.state_dim,
        "transported_teacher_state_dim": fit.transported_teacher.state_dim,
        "representation": fit.metadata.get("scalable_backend", fit.student.metadata.get("representation", "dense_local_field")),
        "chart_projection_shape": None if fit.student.chart_projection is None else list(fit.student.chart_projection.shape),
        "dense_memory_guard": fit.metadata.get("dense_memory_guard", fit.student.metadata.get("dense_memory_guard")),
        "feature_count": None if fit.metadata.get("dense_memory_guard", fit.student.metadata.get("dense_memory_guard")) is None else fit.metadata.get("dense_memory_guard", fit.student.metadata.get("dense_memory_guard")).get("quadratic_feature_count"),
        "estimated_dense_bytes": None if fit.metadata.get("dense_memory_guard", fit.student.metadata.get("dense_memory_guard")) is None else fit.metadata.get("dense_memory_guard", fit.student.metadata.get("dense_memory_guard")).get("total_bytes"),
        "requested_signature_mode": fit.metadata.get("requested_signature_mode", fit.metadata.get("signature_mode")),
        "effective_signature_mode": fit.metadata.get("effective_signature_mode", fit.metadata.get("signature_mode")),
        "hessian_2jet_applied": bool(fit.metadata.get("math_profile", {}).get("components", {}).get("hessian_2jet", {}).get("applied", False)),
        "quadratic_flow_applied": bool(fit.metadata.get("math_profile", {}).get("components", {}).get("quadratic_flow", {}).get("applied", False)),
        "quadratic_skipped_reason": fit.metadata.get("quadratic_skipped_reason"),
        "training_error": fit.validation_error,
        "mean_confidence": np.mean(fit.confidence),
        "target_fit_split": fit.transported_teacher.metadata.get("target_fit_split", "train"),
        "metric_scope": fit.transported_teacher.metadata.get("metric_scope", "transported_teacher_operator_approximation"),
        "alignment_fit_scope": fit.metadata.get("alignment_fit_scope"),
        "alignment_paired_error": fit.metadata.get("alignment_paired_error"),
        "alignment_relational_error": fit.metadata.get("alignment_relational_error"),
        "depth_correspondence_summary": fit.metadata.get("depth_correspondence_summary"),
        "depth_report": fit.depth_report,
        "capacity_diagnostics": fit.capacity_diagnostics,
        "metadata": fit.metadata,
        "finite_correction": bool(np.all(np.isfinite(fit.correction_matrices)) and np.all(np.isfinite(fit.correction_biases))),
        })
    journal_math_profile(journal, fit.metadata.get("math_profile"), stage=stage)
    for node in range(len(fit.correction_matrices)):
        journal.record(stage, "flow-depth", scope="depth", payload={
            "node": node,
            "student_matrix_shape": list(fit.student.matrices[node].shape),
            "correction_matrix_shape": list(fit.correction_matrices[node].shape),
            "student_matrix_norm": stable_l2(fit.student.matrices[node], name="journal student matrix"),
            "correction_norm": stable_l2(fit.correction_matrices[node], name="journal correction matrix"),
            "spectral_norm": fit.student.spectral_norms[node],
            "confidence": fit.confidence[node],
            "residual_scale": fit.student.residual_scales[node],
            "quadratic_present": fit.correction_quadratic is not None,
            "backend": fit.metadata.get("scalable_backend", fit.student.metadata.get("scalable_backend", "dense_local_field")),
            "feature_count": None if fit.metadata.get("dense_memory_guard") is None else fit.metadata["dense_memory_guard"].get("quadratic_feature_count"),
            "estimated_dense_bytes": None if fit.metadata.get("dense_memory_guard") is None else fit.metadata["dense_memory_guard"].get("total_bytes"),
        })


def journal_scorecard(journal: ExperimentJournal, report: Any, *, stage: str = "scorecard") -> None:
    payload = report.to_dict() if hasattr(report, "to_dict") else report
    journal.record(stage, "scorecard", scope="run", payload=payload)
    journal_math_profile(journal, payload.get("metadata", {}).get("math_profiles"), stage=stage)
    # Keep aggregate profile deltas separate from direct same-split
    # leave-one-component-out measurements.
    counterfactuals = payload.get("metadata", {}).get("math_component_counterfactual_table", {})
    if isinstance(counterfactuals, Mapping):
        for component, data in counterfactuals.items():
            journal.record(stage, "math-component-counterfactual", scope="math", payload={"component": component, **(dict(data) if isinstance(data, Mapping) else {"value": data})})
    for entry in getattr(report, "entries", ()):
        data = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
        journal.record(stage, "scorecard-entry", scope="probe-set", payload={**data, "accepted_count": data.get("accepted_nodes", 0), "rollback_count": data.get("rollback_nodes", 0), "rejected_count": data.get("rollback_nodes", 0)})


def journal_adaptive(journal: ExperimentJournal, report: Any, *, stage: str = "adaptive-transfer") -> None:
    journal.record(stage, "adaptive-report", scope="run", payload=report.to_dict() if hasattr(report, "to_dict") else report)
    payload = report.to_dict() if hasattr(report, "to_dict") else report
    journal_math_profile(journal, payload.get("metadata", {}).get("math_profile"), stage=stage)
    for step in getattr(report, "steps", ()):
        data = step.to_dict() if hasattr(step, "to_dict") else dict(step)
        accepted = len(data.get("accepted_blocks", ()))
        rollback = len(data.get("rollback_blocks", ()))
        journal.record(stage, "gain-step", scope="gain", payload={**data, "accepted_count": accepted, "rollback_count": rollback, "rejected_count": rollback})


def journal_preflight(journal: ExperimentJournal, payload: Mapping[str, Any], *, stage: str = "preflight") -> None:
    journal.record(stage, "preflight", scope="runtime", payload=dict(payload))


def journal_surgery(journal: ExperimentJournal, plan: Any, *, stage: str = "apply-surgery") -> None:
    payload = {
        "status": "applied" if getattr(plan, "applied_tensors", ()) else "rejected",
        "mode": getattr(plan, "metadata", {}).get("mode"),
        "applied_tensors": list(getattr(plan, "applied_tensors", ())),
        "skipped_tensors": dict(getattr(plan, "skipped_tensors", {})),
        "rollback_layers": list(getattr(plan, "rollback_layers", ())),
        "quadratic_terms_applied": bool(getattr(plan, "metadata", {}).get("quadratic_terms_applied", False)),
        "actual_weight_changes": bool(getattr(plan, "applied_tensors", ())),
        "accepted_count": len(getattr(plan, "applied_tensors", ())),
        "rollback_count": len(getattr(plan, "rollback_layers", ())),
        "rejected_count": len(getattr(plan, "skipped_tensors", {})),
    }
    journal.record(stage, "weight-surgery", scope="weights", payload=payload)
    for name, reason in dict(getattr(plan, "skipped_tensors", {})).items():
        journal.record(stage, "tensor-skip", scope="tensor", payload={"tensor_name": name, "status": "skipped", "reason": reason})


def journal_runtime(journal: ExperimentJournal, payload: Mapping[str, Any], *, stage: str = "runtime-validation") -> None:
    journal.record(stage, "runtime-preflight-or-result", scope="runtime", payload=dict(payload))


def journal_math_profile(journal: ExperimentJournal, profile: Any, *, stage: str) -> None:
    """Emit one machine-readable activation event for every math component."""

    if not isinstance(profile, Mapping):
        return
    components = profile.get("components", {})
    if isinstance(components, Mapping):
        for component, audit in components.items():
            journal.record(stage, "math-component", scope="math", payload={"component": component, **(dict(audit) if isinstance(audit, Mapping) else {"value": audit})})
