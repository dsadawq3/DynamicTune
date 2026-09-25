"""Strict ordinary-weight surgery with explicit tensor/flow contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import json
import numpy as np

from .connectors import CapabilityError
from .solver import ConstrainedCorrection, TrustRegionConfig, solve_flow_correction
from .types import FlowFitResult, SurgeryPlan, TensorSchemaEntry, TensorTransitionMapping, finite_array


def tensor_schema(weights: Mapping[str, np.ndarray]) -> tuple[TensorSchemaEntry, ...]:
    entries = []
    for name, value in weights.items():
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"weight {name} is non-finite")
        entries.append(TensorSchemaEntry(str(name), tuple(int(s) for s in array.shape), str(array.dtype)))
    return tuple(entries)


def _resolve_mapping(weights: Mapping[str, np.ndarray], correction: ConstrainedCorrection, mapping: Sequence[TensorTransitionMapping] | None, mapping_callback: Callable[[Mapping[str, np.ndarray], ConstrainedCorrection], Sequence[TensorTransitionMapping]] | None) -> tuple[TensorTransitionMapping, ...]:
    if mapping is not None and mapping_callback is not None:
        raise ValueError("provide an explicit tensor mapping or a mapping callback, not both")
    resolved = tuple(mapping_callback(weights, correction) if mapping_callback is not None else (mapping or ()))
    if any(not isinstance(item, TensorTransitionMapping) for item in resolved):
        raise TypeError("tensor mapping callback must return TensorTransitionMapping entries")
    if len({item.tensor_name for item in resolved}) != len(resolved):
        raise ValueError("tensor mapping contains duplicate tensor names")
    if len({item.transition_index for item in resolved}) != len(resolved):
        raise ValueError("tensor mapping contains duplicate transition indices")
    return resolved


QuadraticSurgeryCallback = Callable[[Mapping[str, np.ndarray], np.ndarray, float], Mapping[str, np.ndarray]]


def _validate_callback_updates(originals: Mapping[str, np.ndarray], callback_updates: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], tuple[str, ...], dict[str, str]]:
    updates: dict[str, np.ndarray] = {}
    applied: list[str] = []
    skipped: dict[str, str] = {}
    for raw_name, raw_value in callback_updates.items():
        name = str(raw_name)
        if name not in originals:
            skipped[name] = "quadratic callback returned a tensor outside the current schema"
            continue
        candidate = np.asarray(raw_value)
        original = originals[name]
        if candidate.shape != original.shape or candidate.dtype != original.dtype:
            skipped[name] = f"quadratic callback changed tensor schema: {candidate.shape}/{candidate.dtype} vs {original.shape}/{original.dtype}"
            continue
        if not np.all(np.isfinite(candidate)):
            skipped[name] = "quadratic callback returned non-finite values"
            continue
        updates[name] = candidate.copy()
        if not np.array_equal(candidate, original):
            applied.append(name)
        else:
            skipped[name] = "quadratic callback returned an unchanged tensor"
    return updates, tuple(applied), skipped


def build_surgery_plan(weights: Mapping[str, np.ndarray], correction: ConstrainedCorrection, *, mapping: Sequence[TensorTransitionMapping] | None = None, mapping_callback: Callable[[Mapping[str, np.ndarray], ConstrainedCorrection], Sequence[TensorTransitionMapping]] | None = None, quadratic_callback: QuadraticSurgeryCallback | None = None, gain: float = 1.0, step_sizes: Sequence[float] | None = None, mode: str = "diagnostic") -> SurgeryPlan:
    """Build a plan using only explicitly named compatible tensors.

    ``row_right`` means the layer computes ``x @ W`` and receives the flow
    delta directly. ``column_left`` means it computes ``W @ x`` and receives
    the transpose. No tensor is selected by a name regex or shape coincidence.
    """

    if not np.isfinite(gain) or gain < 0:
        raise ValueError("gain must be finite and non-negative")
    if mode not in {"diagnostic", "apply"}:
        raise ValueError("mode must be 'diagnostic' or 'apply'")
    originals = {str(k): np.asarray(v).copy() for k, v in weights.items()}
    schema = tensor_schema(originals)
    resolved = _resolve_mapping(originals, correction, mapping, mapping_callback)
    updates = {name: value.copy() for name, value in originals.items()}
    rollback_layers = {int(i) for i, ok in enumerate(correction.accepted) if not ok}
    reasons = {int(i): str(reason) for i, reason in correction.reasons.items()}
    skipped: dict[str, str] = {}
    applied: list[str] = []
    callback_names: set[str] = set()
    quadratic_present = bool(getattr(correction, "quadratic_terms", None) is not None and np.any(np.abs(correction.quadratic_terms) > 0))
    quadratic_applied = False
    if quadratic_present and quadratic_callback is None:
        skipped["<flow.quadratic_terms>"] = "quadratic flow residual requires quadratic_callback(weights, quadratic_terms, gain); dense-square tensor mapping applies only linear transitions"
    if quadratic_present and quadratic_callback is not None:
        callback_result = quadratic_callback(originals, correction.quadratic_terms.copy(), gain)
        if not isinstance(callback_result, Mapping):
            raise CapabilityError("quadratic surgery callback must return a mapping of named tensors")
        callback_names = {str(name) for name in callback_result}
        callback_updates, callback_applied, callback_skipped = _validate_callback_updates(originals, callback_result)
        updates.update(callback_updates)
        applied.extend(callback_applied)
        skipped.update({name: f"quadratic callback: {reason}" for name, reason in callback_skipped.items()})
        quadratic_applied = bool(callback_applied)
        if not callback_result:
            skipped["<flow.quadratic_terms>"] = "quadratic callback returned no tensor updates"
    mapped_names = {item.tensor_name for item in resolved}
    overlap = mapped_names.intersection(applied)
    if overlap:
        raise ValueError(f"quadratic callback and linear mapping both update tensor(s): {sorted(overlap)}")
    applied = list(dict.fromkeys(applied))
    for name in originals:
        if name not in mapped_names and name not in applied and name not in callback_names:
            skipped[name] = "not selected by an explicit tensor-transition mapping"
    for item in resolved:
        name = item.tensor_name
        if name not in originals:
            skipped[name] = "mapped tensor does not exist in current schema"
            continue
        if item.transition_index >= len(correction.matrices):
            skipped[name] = "mapping references a missing flow transition"
            continue
        if not correction.accepted[item.transition_index]:
            skipped[name] = f"flow transition rolled back: {reasons.get(item.transition_index, 'unsafe correction')}"
            continue
        value = originals[name]
        if not np.issubdtype(value.dtype, np.floating):
            skipped[name] = f"tensor dtype {value.dtype} is not a floating point weight"
            continue
        delta = finite_array(correction.matrices[item.transition_index], ndim=2, name=f"correction for {name}")
        tensor_delta = delta if item.orientation == "row_right" else delta.T
        if value.ndim != 2 or value.shape != tensor_delta.shape:
            skipped[name] = f"shape/orientation mismatch: tensor {value.shape}, correction {tensor_delta.shape}"
            continue
        ds = 1.0
        if step_sizes is not None and item.transition_index < len(step_sizes):
            ds = float(step_sizes[item.transition_index])
        elif getattr(correction, "step_sizes", None) is not None and item.transition_index < len(correction.step_sizes):
            ds = float(correction.step_sizes[item.transition_index])
        candidate = value.astype(np.float64) + (gain * ds) * tensor_delta
        if not np.all(np.isfinite(candidate)):
            skipped[name] = "candidate tensor is non-finite"
            rollback_layers.add(item.transition_index)
            reasons[item.transition_index] = "candidate tensor is non-finite"
            continue
        try:
            cast_candidate = candidate.astype(value.dtype, copy=False)
        except (OverflowError, ValueError) as error:
            skipped[name] = f"tensor dtype conversion failed: {error}"
            continue
        if not np.all(np.isfinite(cast_candidate)):
            skipped[name] = f"candidate overflows tensor dtype {value.dtype}"
            rollback_layers.add(item.transition_index)
            reasons[item.transition_index] = "candidate overflows tensor dtype"
            continue
        if np.array_equal(cast_candidate, value):
            skipped[name] = "candidate tensor is unchanged; no effective update"
            continue
        updates[name] = cast_candidate
        applied.append(name)
    if mode == "apply" and not applied:
        detail = "; ".join(f"{name}: {reason}" for name, reason in skipped.items()) or "no mapping supplied"
        raise CapabilityError(f"apply mode produced no applicable tensor updates; {detail}")
    return SurgeryPlan(updates, tuple(sorted(rollback_layers)), reasons, correction.confidence.copy(), schema, {"gain": gain, "mode": mode, "mapping_count": len(resolved), "adapter": None, "lora": False, "quadratic_terms_applied": quadratic_applied, "quadratic_callback": quadratic_callback is not None, "applied_tensors": tuple(applied)}, skipped, tuple(applied))


def apply_surgery(weights: Mapping[str, np.ndarray], plan: SurgeryPlan, *, require_update: bool = True) -> dict[str, np.ndarray]:
    current_schema = tensor_schema(weights)
    if current_schema != plan.schema:
        raise ValueError("weight schema changed since the surgery plan was built")
    if require_update and not plan.applied_tensors:
        raise CapabilityError("surgery plan has no applied tensors; use diagnostic mode to inspect skipped mappings")
    result = {str(k): np.asarray(v).copy() for k, v in plan.updates.items()}
    if set(result) != set(weights):
        raise ValueError("surgery plan does not preserve tensor keys")
    for name, value in result.items():
        if value.shape != np.asarray(weights[name]).shape or value.dtype != np.asarray(weights[name]).dtype:
            raise ValueError(f"schema changed for tensor {name}")
    return result


def apply_plan_to_connector(connector: Any, plan: SurgeryPlan) -> None:
    """Commit ordinary tensors through an explicit connector setter."""

    if not getattr(connector.capabilities, "weight_surgery", False):
        raise CapabilityError("connector capability matrix does not authorize weight surgery")
    current = connector.weights()
    connector.set_weights(apply_surgery(current, plan, require_update=True))


def plan_from_flow(weights: Mapping[str, np.ndarray], flow_fit: FlowFitResult, *, mapping: Sequence[TensorTransitionMapping] | None = None, mapping_callback: Callable[[Mapping[str, np.ndarray], ConstrainedCorrection], Sequence[TensorTransitionMapping]] | None = None, quadratic_callback: QuadraticSurgeryCallback | None = None, config: TrustRegionConfig | None = None, gain: float = 1.0, mode: str = "diagnostic") -> SurgeryPlan:
    constrained = solve_flow_correction(flow_fit, config=config, baseline_operators=flow_fit.student.matrices)
    return build_surgery_plan(weights, constrained, mapping=mapping, mapping_callback=mapping_callback, quadratic_callback=quadratic_callback, gain=gain, step_sizes=constrained.step_sizes, mode=mode)


def export_weights(weights: Mapping[str, np.ndarray], path: str | Path) -> Path:
    """Export ordinary named tensors as NPZ with a JSON schema manifest."""

    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        destination = destination.with_suffix(destination.suffix + ".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean = {str(k): np.asarray(v).copy() for k, v in weights.items()}
    schema = tensor_schema(clean)
    encoded: dict[str, np.ndarray] = {}
    encoded_names: dict[str, str] = {}
    for index, (name, value) in enumerate(clean.items()):
        archive_name = f"tensor_{index:06d}"
        encoded[archive_name] = value
        encoded_names[name] = archive_name
    np.savez_compressed(destination, **encoded)
    manifest = destination.with_suffix(destination.suffix + ".json")
    manifest.write_text(json.dumps({"format": "faytuna-ordinary-weights-v1", "schema": [entry.__dict__ for entry in schema], "archive_names": encoded_names}, indent=2), encoding="utf-8")
    return destination


def import_weights(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if source.suffix.lower() != ".npz":
        source = source.with_suffix(source.suffix + ".npz")
    manifest_path = source.with_suffix(source.suffix + ".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "faytuna-ordinary-weights-v1":
        raise ValueError("unsupported weights artifact")
    archive_names = manifest.get("archive_names")
    if archive_names is None:
        archive_names = {key: key.replace("/", "__") for key in manifest["keys"]}
    with np.load(source, allow_pickle=False) as archive:
        result = {key: archive[archive_name].copy() for key, archive_name in archive_names.items()}
    expected = tuple(TensorSchemaEntry(item["name"], tuple(item["shape"]), item["dtype"]) for item in manifest["schema"])
    if tensor_schema(result) != expected:
        raise ValueError("weights manifest schema mismatch")
    return result
