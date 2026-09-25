"""Artifact-first serialization for paired teacher/student experiments.

Trace artifacts keep observed feature charts, continuous coordinates, token
position metadata, differential signatures, uncertainty, and capability
metadata. A large teacher is therefore replaceable at the connector boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import json
import os
import tempfile
import numpy as np

from .types import AlignmentResult, FlowOperator, TrajectoryTrace, TransitionObservation
from .signatures import differential_signature, multi_scale_path_signature


def _npz_path(path: str | Path) -> Path:
    destination = Path(path)
    return destination if destination.suffix.lower() == ".npz" else destination.with_suffix(destination.suffix + ".npz")


def _json_safe(value):
    """Encode manifests without non-standard JSON numbers."""

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


def _stack_optional(values: Sequence[np.ndarray | None], shape: tuple[int, ...], *, dtype: type = float) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray([value is not None for value in values], dtype=bool)
    result = np.zeros((len(values),) + shape, dtype=dtype)
    for index, value in enumerate(values):
        if value is not None:
            array = np.asarray(value)
            if array.shape != shape:
                raise ValueError("optional trace fields have inconsistent shapes")
            result[index] = array
    return result, mask


def _atomic_npz(destination: Path, arrays: dict[str, np.ndarray], *, compressed: bool, durable: bool) -> None:
    """Commit an NPZ by rename so an interrupted probe cannot look complete."""

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if compressed:
            np.savez_compressed(temporary, **arrays)
        else:
            np.savez(temporary, **arrays)
        if durable:
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(destination: Path, content: str, *, durable: bool) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_traces(traces: Sequence[TrajectoryTrace], path: str | Path, *, compressed: bool = True, durable: bool = False) -> Path:
    if not traces:
        raise ValueError("cannot save an empty trace collection")
    destination = _npz_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len({trace.hidden_states.shape for trace in traces}) != 1 or len({trace.depth_coordinates.shape for trace in traces}) != 1:
        raise ValueError("artifact format requires equal trace state/coordinate shapes")
    if len({trace.layer_ids for trace in traces}) != 1 or len({trace.feature_space for trace in traces}) != 1:
        raise ValueError("artifact format requires a common layer schema and feature space")
    transition_count = traces[0].layer_count - 1
    if any(len(trace.transitions) != transition_count for trace in traces):
        raise ValueError("artifact format requires equal transition counts")
    jacobian_shape = next((value.jacobian.shape for trace in traces for value in trace.transitions if value.jacobian is not None), None)
    hessian_shape = next((value.hessian_sketch.shape for trace in traces for value in trace.transitions if value.hessian_sketch is not None), None)
    singular_shape = next((value.singular_values.shape for trace in traces for value in trace.transitions if value.singular_values is not None), None)
    jacobians, jacobian_mask = _stack_optional([value.jacobian for trace in traces for value in trace.transitions], jacobian_shape or (0,)) if jacobian_shape else (None, None)
    hessians, hessian_mask = _stack_optional([value.hessian_sketch for trace in traces for value in trace.transitions], hessian_shape or (0,)) if hessian_shape else (None, None)
    singular_values, singular_mask = _stack_optional([value.singular_values for trace in traces for value in trace.transitions], singular_shape or (0,)) if singular_shape else (None, None)
    residual_values, residual_mask = _stack_optional([trace.residual_states for trace in traces], traces[0].hidden_states.shape) 
    uncertainty_values, uncertainty_mask = _stack_optional([trace.uncertainty for trace in traces], (traces[0].layer_count,))
    arrays = {
        "states": np.asarray([trace.hidden_states for trace in traces]),
        "coordinates": np.asarray([trace.depth_coordinates for trace in traces]),
        "layer_ids": np.asarray(traces[0].layer_ids, dtype=np.int64),
        "vector_fields": np.asarray([[transition.vector_field for transition in trace.transitions] for trace in traces]),
        "curvatures": np.asarray([[transition.curvature for transition in trace.transitions] for trace in traces]),
        "residual_states": residual_values,
        "residual_mask": residual_mask,
        "uncertainty": uncertainty_values,
        "uncertainty_mask": uncertainty_mask,
    }
    if jacobians is not None:
        arrays.update({"jacobians": jacobians.reshape(len(traces), transition_count, *jacobian_shape), "jacobian_mask": jacobian_mask.reshape(len(traces), transition_count)})
    if hessians is not None:
        arrays.update({"hessian_sketches": hessians.reshape(len(traces), transition_count, *hessian_shape), "hessian_mask": hessian_mask.reshape(len(traces), transition_count)})
    if singular_values is not None:
        arrays.update({"singular_values": singular_values.reshape(len(traces), transition_count, *singular_shape), "singular_mask": singular_mask.reshape(len(traces), transition_count)})
    # Resume checkpoints are written once and read at most once. Avoiding
    # deflate there reduces CPU time while the atomic rename prevents a
    # partially written NPZ from being accepted by a later resume.
    _atomic_npz(destination, arrays, compressed=compressed, durable=durable)
    trace_metadata = []
    for trace in traces:
        trace_info = dict(trace.metadata)
        if "path_signature" not in trace_info or "differential_signature" not in trace_info:
            path_signature = multi_scale_path_signature(trace.hidden_states, trace.depth_coordinates, area_seed=int(trace.metadata.get("probe_seed", 0)))
            differential = differential_signature(trace)
            trace_info.update({
                "path_signature": {"scales": list(path_signature.scales), "values": path_signature.values.tolist(), "endpoint_displacement": path_signature.endpoint_displacement, "total_arc_length": path_signature.total_arc_length, "turning_total": path_signature.turning_total, "area_backend": path_signature.area_backend, "area_sketch_rank": path_signature.area_sketch_rank},
                "differential_signature": {"values": differential.values.tolist(), "available": differential.available.tolist(), "feature_names": list(differential.feature_names)},
            })
        trace_metadata.append({
            "model_id": trace.model_id,
            "probe_id": trace.probe_id,
            "feature_space": trace.feature_space,
            "metadata": trace_info,
            "token_ids": None if trace.token_ids is None else trace.token_ids.tolist(),
            "position_ids": None if trace.position_ids is None else trace.position_ids.tolist(),
            "token_position_map": dict(trace.token_position_map),
            "transition_uncertainty": [dict(value.uncertainty) for value in trace.transitions],
            "normalization_geometry": [None if value.normalization_geometry is None else dict(value.normalization_geometry) for value in trace.transitions],
            "attention_geometry": [None if value.attention_geometry is None else dict(value.attention_geometry) for value in trace.transitions],
        })
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    _atomic_text(metadata_path, json.dumps(_json_safe({"format": "faytuna-traces-v2", "trace_metadata": trace_metadata}), allow_nan=False, indent=2), durable=durable)
    return destination


def load_traces(path: str | Path) -> list[TrajectoryTrace]:
    source = _npz_path(path)
    metadata = json.loads(source.with_suffix(source.suffix + ".json").read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as archive:
        states = archive["states"].copy()
        coordinates = archive["coordinates"].copy()
        layer_ids = tuple(int(x) for x in archive["layer_ids"])
        vectors = archive["vector_fields"].copy() if "vector_fields" in archive else None
        curvatures = archive["curvatures"].copy() if "curvatures" in archive else None
        residuals = archive["residual_states"].copy() if "residual_states" in archive else None
        residual_mask = archive["residual_mask"].copy() if "residual_mask" in archive else np.zeros(len(states), dtype=bool)
        uncertainty_values = archive["uncertainty"].copy() if "uncertainty" in archive else None
        uncertainty_mask = archive["uncertainty_mask"].copy() if "uncertainty_mask" in archive else np.zeros(len(states), dtype=bool)
        jacobians = archive["jacobians"].copy() if "jacobians" in archive else None
        jacobian_mask = archive["jacobian_mask"].copy() if "jacobian_mask" in archive else None
        hessians = archive["hessian_sketches"].copy() if "hessian_sketches" in archive else None
        hessian_mask = archive["hessian_mask"].copy() if "hessian_mask" in archive else None
        singular_values = archive["singular_values"].copy() if "singular_values" in archive else None
        singular_mask = archive["singular_mask"].copy() if "singular_mask" in archive else None
    if metadata.get("format") not in {"faytuna-traces-v1", "faytuna-traces-v2"}:
        raise ValueError("unsupported trace artifact")
    trace_metadata = metadata.get("trace_metadata")
    if trace_metadata is None:
        trace_metadata = [{"model_id": item, "probe_id": probe, "metadata": {}} for item, probe in zip(metadata["model_ids"], metadata["probe_ids"])]
    traces = []
    for index in range(states.shape[0]):
        info = trace_metadata[index]
        transitions = []
        for node in range(len(layer_ids) - 1):
            delta = states[index, node + 1] - states[index, node]
            ds = float(coordinates[index, node + 1] - coordinates[index, node])
            transition_index = index * (len(layer_ids) - 1) + node
            transition_uncertainty = info.get("transition_uncertainty", [{} for _ in range(len(layer_ids) - 1)])[node]
            jacobian = None if jacobians is None or not jacobian_mask[index, node] else jacobians[index, node]
            hessian = None if hessians is None or not hessian_mask[index, node] else hessians[index, node]
            singular = None if singular_values is None or not singular_mask[index, node] else singular_values[index, node]
            transitions.append(TransitionObservation(layer_ids[node], layer_ids[node + 1], float(coordinates[index, node]), float(coordinates[index, node + 1]), states[index, node], states[index, node + 1], delta, vectors[index, node] if vectors is not None else delta / ds, jacobian, hessian, float(curvatures[index, node]) if curvatures is not None else 0.0, singular, info.get("normalization_geometry", [None] * (len(layer_ids) - 1))[node], info.get("attention_geometry", [None] * (len(layer_ids) - 1))[node], transition_uncertainty))
        traces.append(TrajectoryTrace(info["model_id"], info["probe_id"], layer_ids, coordinates[index], states[index], residuals[index] if residuals is not None and residual_mask[index] else None, tuple(transitions), info.get("metadata", {}), info.get("feature_space", "hidden"), None if info.get("token_ids") is None else np.asarray(info["token_ids"]), None if info.get("position_ids") is None else np.asarray(info["position_ids"]), info.get("token_position_map", {}), uncertainty_values[index] if uncertainty_values is not None and uncertainty_mask[index] else None))
    return traces


def save_alignment(alignment: AlignmentResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_safe({"kind": alignment.kind, "source_mean": alignment.source_mean.tolist(), "target_mean": alignment.target_mean.tolist(), "matrix": alignment.matrix.tolist(), "bias": alignment.bias.tolist(), "source_rank": alignment.source_rank, "target_rank": alignment.target_rank, "paired_error": alignment.paired_error, "relational_error": alignment.relational_error, "cycle_error": alignment.cycle_error, "condition_number": alignment.condition_number, "ot_mass_error": alignment.ot_mass_error, "metadata": dict(alignment.metadata), "source_projection": None if alignment.source_projection is None else alignment.source_projection.tolist(), "target_projection": None if alignment.target_projection is None else alignment.target_projection.tolist(), "depth_coordinates": None if alignment.depth_coordinates is None else alignment.depth_coordinates.tolist(), "depth_matrices": None if alignment.depth_matrices is None else alignment.depth_matrices.tolist(), "depth_biases": None if alignment.depth_biases is None else alignment.depth_biases.tolist()}), allow_nan=False, indent=2), encoding="utf-8")
    return destination


def load_alignment(path: str | Path) -> AlignmentResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AlignmentResult(data["kind"], np.asarray(data["source_mean"]), np.asarray(data["target_mean"]), np.asarray(data["matrix"]), np.asarray(data["bias"]), int(data["source_rank"]), int(data["target_rank"]), float(data["paired_error"]), float(data["relational_error"]), float(data["cycle_error"]), float("inf") if data.get("condition_number") is None else float(data["condition_number"]), float(data.get("ot_mass_error", 0.0) or 0.0), data.get("metadata", {}), None if data.get("source_projection") is None else np.asarray(data["source_projection"]), None if data.get("target_projection") is None else np.asarray(data["target_projection"]), None if data.get("depth_coordinates") is None else np.asarray(data["depth_coordinates"]), None if data.get("depth_matrices") is None else np.asarray(data["depth_matrices"]), None if data.get("depth_biases") is None else np.asarray(data["depth_biases"]))


def save_flow(flow: FlowOperator, path: str | Path) -> Path:
    destination = _npz_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"coordinates": flow.coordinates, "matrices": flow.matrices, "biases": flow.biases, "sample_counts": flow.sample_counts, "residual_scales": flow.residual_scales, "spectral_norms": flow.spectral_norms}
    if flow.quadratic_terms is not None:
        arrays["quadratic_terms"] = flow.quadratic_terms
    if flow.chart_projection is not None:
        arrays["chart_projection"] = flow.chart_projection
    np.savez_compressed(destination, **arrays)
    destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(_json_safe({"format": "faytuna-flow-v1", "metadata": dict(flow.metadata)}), allow_nan=False, indent=2), encoding="utf-8")
    return destination


def load_flow(path: str | Path) -> FlowOperator:
    source = _npz_path(path)
    data = json.loads(source.with_suffix(source.suffix + ".json").read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as archive:
        return FlowOperator(archive["coordinates"], archive["matrices"], archive["biases"], archive["sample_counts"], archive["residual_scales"], archive["spectral_norms"], data.get("metadata", {}), archive["quadratic_terms"] if "quadratic_terms" in archive else None, archive["chart_projection"] if "chart_projection" in archive else None)
