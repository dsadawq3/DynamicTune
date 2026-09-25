"""Model-independent observation protocol and local differential sketches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import hashlib
import json
import time
import numpy as np

from .types import CapabilityMatrix, Probe, TrajectoryTrace, TransitionObservation, finite_array, stable_l2
from .signatures import differential_signature, multi_scale_path_signature


DEFAULT_MAX_EXACT_JACOBIAN_DIM = 512


def probe_fingerprint(probe: Probe) -> str:
    """Hash the probe identity and numerical payload used by a trace."""

    def canonical(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        if isinstance(value, np.ndarray):
            return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    digest = hashlib.sha256()
    header = {"probe_id": canonical(probe.probe_id), "family": canonical(probe.family), "payload": canonical(probe.payload), "split": canonical(probe.split), "pair_id": canonical(probe.pair_id)}
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    digest.update(np.asarray(probe.initial_state, dtype=np.float64).tobytes())
    if probe.perturbation is not None:
        digest.update(np.asarray(probe.perturbation, dtype=np.float64).tobytes())
    return digest.hexdigest()


def finite_jacobian(function: Any, x: np.ndarray, step: float = 1e-4, *, base_output: np.ndarray | None = None, max_dimension: int | None = DEFAULT_MAX_EXACT_JACOBIAN_DIM) -> np.ndarray:
    """Central finite-difference Jacobian J[i,j]=d f_i / d x_j."""

    point = finite_array(x, ndim=1, name="jacobian point")
    if step <= 0 or not np.isfinite(step):
        raise ValueError("finite-difference step must be positive and finite")
    if max_dimension is not None and point.size > int(max_dimension):
        raise ValueError(f"coordinate finite Jacobian prohibited above max_exact_jacobian_dim={int(max_dimension)}; use directional_jacobian_sketch")
    base = finite_array(function(point) if base_output is None else base_output, ndim=1, name="function output")
    jacobian = np.empty((base.size, point.size), dtype=np.float64)
    for j in range(point.size):
        direction = np.zeros_like(point)
        local_step = float(step) * max(1.0, abs(float(point[j])))
        direction[j] = local_step
        with np.errstate(over="raise", invalid="raise"):
            plus = finite_array(function(point + direction), ndim=1, name="plus output")
            minus = finite_array(function(point - direction), ndim=1, name="minus output")
        if plus.shape != base.shape or minus.shape != base.shape:
            raise ValueError("function output shape changed under finite difference")
        jacobian[:, j] = (plus - minus) / (2.0 * local_step)
    return finite_array(jacobian, ndim=2, name="finite-difference Jacobian")


def directional_jacobian_sketch(function: Any, x: np.ndarray, directions: np.ndarray, step: float = 1e-4, *, base_output: np.ndarray | None = None) -> np.ndarray:
    """Estimate ``J U`` with central finite differences along ``k`` directions.

    ``directions`` has shape ``[k, input_dim]`` and the result has shape
    ``[output_dim, k]``.  This is a directional sketch, not a recoverable full
    coordinate Jacobian; it needs two function calls per direction and never
    allocates an ``input_dim x input_dim`` matrix.
    """

    point = finite_array(x, ndim=1, name="directional Jacobian point")
    dirs = finite_array(directions, ndim=2, name="directional Jacobian directions")
    if dirs.shape[1] != point.size or dirs.shape[0] < 1:
        raise ValueError("directional Jacobian directions have the wrong shape")
    if step <= 0 or not np.isfinite(step):
        raise ValueError("finite-difference step must be positive and finite")
    base = finite_array(function(point) if base_output is None else base_output, ndim=1, name="directional Jacobian output")
    local_step = float(step) * max(1.0, float(np.linalg.norm(point)))
    columns: list[np.ndarray] = []
    for direction in dirs:
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError("directional Jacobian directions must be finite and nonzero")
        unit = direction / norm
        with np.errstate(over="raise", invalid="raise"):
            plus = finite_array(function(point + local_step * unit), ndim=1, name="directional Jacobian plus output")
            minus = finite_array(function(point - local_step * unit), ndim=1, name="directional Jacobian minus output")
        if plus.shape != base.shape or minus.shape != base.shape:
            raise ValueError("function output shape changed under directional Jacobian difference")
        columns.append((plus - minus) / (2.0 * local_step))
    return finite_array(np.column_stack(columns), ndim=2, name="directional Jacobian sketch")


def hessian_directional_sketch(function: Any, x: np.ndarray, directions: np.ndarray, step: float = 2e-3, *, base_output: np.ndarray | None = None) -> np.ndarray:
    """Return H[u,u] for several directions u, without materializing a tensor Hessian."""

    point = finite_array(x, ndim=1, name="hessian point")
    dirs = finite_array(directions, ndim=2, name="hessian directions")
    if dirs.shape[1] != point.size or step <= 0:
        raise ValueError("hessian directions or step has the wrong shape")
    if step <= 0 or not np.isfinite(step):
        raise ValueError("hessian step must be positive and finite")
    center = finite_array(function(point) if base_output is None else base_output, ndim=1, name="hessian center output")
    local_step = float(step) * max(1.0, float(np.linalg.norm(point)))
    step_sq = local_step * local_step
    two_center = 2.0 * center
    rows = []
    for direction in dirs:
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("hessian directions must be nonzero")
        unit = direction / norm
        with np.errstate(over="raise", invalid="raise"):
            plus = finite_array(function(point + local_step * unit), ndim=1, name="hessian plus output")
            minus = finite_array(function(point - local_step * unit), ndim=1, name="hessian minus output")
        rows.append((plus - two_center + minus) / step_sq)
    return finite_array(np.asarray(rows, dtype=np.float64), ndim=2, name="hessian sketch")


def trajectory_curvature(previous_delta: np.ndarray | None, delta: np.ndarray, ds: float) -> float:
    if previous_delta is None:
        return 0.0
    if ds <= 0:
        raise ValueError("depth increment must be positive")
    previous = finite_array(previous_delta, ndim=1, name="previous delta")
    current = finite_array(delta, ndim=1, name="delta")
    scale = max(1.0, float(np.max(np.abs(np.concatenate([previous, current])))))
    a = ((current / scale) - (previous / scale)) * scale / ds
    velocity = current / ds
    velocity_norm = float(stable_l2(velocity, name="curvature velocity"))
    speed_sq = velocity_norm * velocity_norm
    if speed_sq <= 1e-18:
        return float(stable_l2(a, name="curvature acceleration"))
    acceleration_norm = float(stable_l2(a, name="curvature acceleration"))
    cosine = float(np.dot(a / max(acceleration_norm, 1e-12), velocity / max(velocity_norm, 1e-12)))
    tangential = velocity * (cosine * acceleration_norm / max(velocity_norm, 1e-12))
    normal = a - tangential
    normal_norm = float(stable_l2(normal, name="curvature normal"))
    result = normal_norm / (speed_sq + 1e-12)
    if not np.isfinite(result):
        raise FloatingPointError("curvature calculation overflowed")
    return float(result)


@dataclass
class ObservationProtocol:
    """Collect comparable observations without requiring a model framework."""

    jacobian_step: float = 1e-4
    hessian_step: float = 2e-3
    hessian_rank: int = 4
    jacobian_rank: int = 8
    max_exact_jacobian_dim: int = DEFAULT_MAX_EXACT_JACOBIAN_DIM
    jacobian_mode: str = "auto"
    depth_mode: str = "adaptive"

    def collect(
        self,
        connector: Any,
        probes: Sequence[Probe],
        *,
        seed: int = 0,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        trace_callback: Callable[[TrajectoryTrace, int, int], None] | None = None,
    ) -> list[TrajectoryTrace]:
        if not probes:
            return []
        if self.jacobian_mode not in {"auto", "exact", "directional"}:
            raise ValueError("jacobian_mode must be 'auto', 'exact', or 'directional'")
        if self.max_exact_jacobian_dim < 1 or self.jacobian_rank < 1 or self.hessian_rank < 1:
            raise ValueError("jacobian/hessian ranks and max_exact_jacobian_dim must be positive")
        capabilities: CapabilityMatrix = connector.capabilities
        traces: list[TrajectoryTrace] = []
        probe_total = len(probes)
        for probe_index, probe in enumerate(probes):
            # Make randomized directions independent of collection order. A
            # resumed single-probe collection therefore uses the same
            # directional estimator as its position in a complete run.
            digest = hashlib.blake2b(f"{int(seed)}:{probe.probe_id}".encode("utf-8"), digest_size=8).digest()
            probe_seed = int.from_bytes(digest, "little", signed=False)
            rng = np.random.default_rng(probe_seed)
            probe_started = time.perf_counter()
            if progress_callback is not None:
                progress_callback({"event": "probe_start", "probe_index": probe_index, "probe_total": probe_total, "probe_id": probe.probe_id, "layer_index": -1, "layer_total": len(connector.layer_ids)})
            states = [finite_array(connector.initial_state(probe), ndim=1, name="initial state")]
            for layer_index in range(len(connector.layer_ids)):
                transition_started = time.perf_counter()
                states.append(finite_array(connector.transition(states[-1], layer_index, probe), ndim=1, name="hidden state"))
                if progress_callback is not None:
                    progress_callback({"event": "layer_transition_complete", "probe_index": probe_index, "probe_total": probe_total, "probe_id": probe.probe_id, "layer_index": layer_index, "layer_total": len(connector.layer_ids), "finite": True, "transition_elapsed_seconds": float(time.perf_counter() - transition_started)})
            state_matrix = np.asarray(states)
            if self.depth_mode == "uniform":
                coordinates = np.linspace(0.0, 1.0, len(states))
            elif self.depth_mode == "adaptive":
                coordinates = self._adaptive_coordinates(state_matrix)
            else:
                raise ValueError("depth_mode must be 'uniform' or 'adaptive'")
            layer_ids = tuple([-1] + [int(x) for x in connector.layer_ids])
            transition_records: list[TransitionObservation] = []
            differential_records: list[dict[str, Any]] = []
            residuals: list[np.ndarray] | None = [] if capabilities.residual_states else None
            for node_index, state in enumerate(state_matrix):
                if residuals is not None:
                    residual = connector.residual_state(state, max(0, node_index - 1), probe)
                    residuals.append(state.copy() if residual is None else residual)
            for layer_index in range(len(connector.layer_ids)):
                differential_started = time.perf_counter()
                source = state_matrix[layer_index]
                target = state_matrix[layer_index + 1]
                ds = float(coordinates[layer_index + 1] - coordinates[layer_index])
                delta = target - source
                fn = lambda z, li=layer_index: connector.transition(z, li, probe)
                jacobian = None
                hessian = None
                singular = None
                jacobian_sensitivity = 0.0
                hessian_sensitivity = 0.0
                state_dim = int(source.size)
                exact_allowed = state_dim <= int(self.max_exact_jacobian_dim)
                use_exact_jacobian = (self.jacobian_mode in {"auto", "exact"}) and exact_allowed
                jacobian_kind = "unavailable"
                jacobian_rank = 0
                jacobian_directions = 0
                jacobian_calls = 0
                hessian_kind = "unavailable"
                hessian_rank = 0
                hessian_directions = 0
                hessian_calls = 0
                jacobian_skip_reason: str | None = None
                hessian_skip_reason: str | None = None
                if capabilities.jacobian_sketch:
                    if use_exact_jacobian:
                        jacobian_kind = "coordinate_finite_difference"
                        jacobian_rank = state_dim
                        jacobian_directions = state_dim
                    else:
                        jacobian_kind = "directional_sketch"
                        jacobian_rank = min(int(self.jacobian_rank), state_dim)
                        jacobian_directions = jacobian_rank
                        if not exact_allowed:
                            jacobian_skip_reason = f"coordinate finite Jacobian auto-downgraded above max_exact_jacobian_dim={int(self.max_exact_jacobian_dim)}"
                        elif self.jacobian_mode == "directional":
                            jacobian_skip_reason = "directional Jacobian sketch explicitly selected"
                    jacobian_directions_array = None if use_exact_jacobian else rng.normal(size=(jacobian_rank, state_dim))
                    callback = getattr(connector, "jacobian_sketch", None)
                    if use_exact_jacobian:
                        jacobian = finite_jacobian(fn, source, self.jacobian_step, base_output=target, max_dimension=self.max_exact_jacobian_dim)
                        jacobian_alt = finite_jacobian(fn, source, self.jacobian_step * 2.0, base_output=target, max_dimension=self.max_exact_jacobian_dim)
                        jacobian_calls = 4 * state_dim
                    elif callable(callback):
                        callback_value = callback(source, layer_index, probe, jacobian_directions_array, self.jacobian_step)
                        if callback_value is None:
                            jacobian = directional_jacobian_sketch(fn, source, jacobian_directions_array, self.jacobian_step, base_output=target)
                            jacobian_alt = directional_jacobian_sketch(fn, source, jacobian_directions_array, self.jacobian_step * 2.0, base_output=target)
                            jacobian_calls = 4 * jacobian_rank
                            jacobian_skip_reason = "connector sketch callback unavailable; directional finite sketch used"
                        else:
                            callback_alt = callback(source, layer_index, probe, jacobian_directions_array, self.jacobian_step * 2.0)
                            jacobian = finite_array(callback_value, ndim=2, name="connector Jacobian sketch")
                            jacobian_alt = finite_array(callback_alt, ndim=2, name="connector Jacobian alternate sketch")
                            jacobian_calls = int(getattr(connector, "jacobian_two_step_calls", 2))
                            jacobian_kind = "directional_sketch"
                    else:
                        jacobian = directional_jacobian_sketch(fn, source, jacobian_directions_array, self.jacobian_step, base_output=target)
                        jacobian_alt = directional_jacobian_sketch(fn, source, jacobian_directions_array, self.jacobian_step * 2.0, base_output=target)
                        jacobian_calls = 4 * jacobian_rank
                    if jacobian.shape != jacobian_alt.shape or jacobian.shape[0] != target.size:
                        raise ValueError("Jacobian sketch output shape is incompatible with the observed state chart")
                    jacobian_sensitivity = float(stable_l2(jacobian_alt - jacobian, name="jacobian step sensitivity") / max(1.0, float(stable_l2(jacobian, name="jacobian magnitude"))))
                    if jacobian.shape[0] > jacobian.shape[1] and jacobian.shape[1] <= 64:
                        gram = jacobian.T @ jacobian
                        eig = np.linalg.eigvalsh(gram)
                        singular = np.sqrt(np.maximum(eig[::-1], 0.0))
                    else:
                        singular = np.linalg.svd(jacobian, compute_uv=False)
                else:
                    jacobian_skip_reason = "connector capability jacobian_sketch=false"
                if capabilities.hessian_sketch:
                    rank = min(self.hessian_rank, source.size)
                    directions = rng.normal(size=(rank, source.size))
                    hessian_rank = rank
                    hessian_directions = rank
                    hessian_kind = "directional_sketch"
                    callback = getattr(connector, "hessian_sketch", None)
                    if callable(callback):
                        hessian_value = callback(source, layer_index, probe, directions, self.hessian_step)
                        if hessian_value is None:
                            hessian = hessian_directional_sketch(fn, source, directions, self.hessian_step, base_output=target)
                            hessian_alt = hessian_directional_sketch(fn, source, directions, self.hessian_step * 2.0, base_output=target)
                            hessian_calls = 4 * rank
                            hessian_skip_reason = "connector sketch callback unavailable; directional finite sketch used"
                        else:
                            hessian_alt_value = callback(source, layer_index, probe, directions, self.hessian_step * 2.0)
                            hessian = finite_array(hessian_value, ndim=2, name="connector Hessian sketch")
                            hessian_alt = finite_array(hessian_alt_value, ndim=2, name="connector Hessian alternate sketch")
                            hessian_kind = "directional_sketch"
                            hessian_calls = int(getattr(connector, "hessian_two_step_calls", 2))
                    else:
                        hessian = hessian_directional_sketch(fn, source, directions, self.hessian_step, base_output=target)
                        hessian_alt = hessian_directional_sketch(fn, source, directions, self.hessian_step * 2.0, base_output=target)
                        hessian_calls = 4 * rank
                    if hessian.shape != hessian_alt.shape or hessian.shape[0] != rank:
                        raise ValueError("Hessian sketch output shape is incompatible with the requested directions")
                    hessian_sensitivity = float(stable_l2(hessian_alt - hessian, name="hessian step sensitivity") / max(1.0, float(stable_l2(hessian, name="hessian magnitude"))))
                else:
                    hessian_skip_reason = "connector capability hessian_sketch=false"
                backend = "exact_coordinate_finite_difference" if jacobian_kind == "coordinate_finite_difference" else "directional_randomized_sketch" if jacobian_kind in {"directional_sketch", "directional_sketch_callback"} else "unavailable"
                differential_calls = jacobian_calls + hessian_calls
                differential_memory = ((0 if jacobian is None else int(np.prod(jacobian.shape))) + (0 if hessian is None else int(np.prod(hessian.shape))) + (0 if jacobian is None else int(np.prod(jacobian.shape))) + (0 if hessian is None else int(np.prod(hessian.shape)))) * np.dtype(np.float64).itemsize
                callback_memory = 0
                memory_estimator = getattr(connector, "estimate_differential_memory_bytes", None)
                if callable(memory_estimator):
                    callback_memory = int(memory_estimator(jacobian_rank, hessian_rank))
                differential_memory = max(differential_memory, callback_memory)
                transition_differential_metadata = {"backend": backend, "jacobian_kind": jacobian_kind, "jacobian_rank": jacobian_rank, "jacobian_directions": jacobian_directions, "jacobian_direction_count": jacobian_directions, "jacobian_estimated_calls": jacobian_calls, "hessian_kind": hessian_kind, "hessian_rank": hessian_rank, "hessian_directions": hessian_directions, "hessian_direction_count": hessian_directions, "hessian_estimated_calls": hessian_calls, "estimated_differential_calls": differential_calls, "estimated_forward_calls": 1 + differential_calls, "estimated_stencil_samples": 1 + (4 * jacobian_rank) + (4 * hessian_rank), "estimated_calls": 1 + differential_calls, "estimated_peak_memory_bytes": differential_memory, "estimated_memory_bytes": differential_memory, "estimated_callback_batch_memory_bytes": callback_memory, "callback_batched": bool(callback_memory), "callback_two_step_cache": bool(callback_memory and (jacobian_calls == 1 or hessian_calls == 1)), "callback_nominal_reuse": bool(getattr(connector, "hessian_reuses_nominal_output", False)), "max_exact_jacobian_dim": int(self.max_exact_jacobian_dim), "auto_downgraded": bool(jacobian_kind == "directional_sketch" and not exact_allowed), "jacobian_skipped_reason": jacobian_skip_reason, "hessian_skipped_reason": hessian_skip_reason}
                differential_records.append(transition_differential_metadata)
                if progress_callback is not None:
                    progress_callback({"event": "finite_difference_dispatch", "probe_index": probe_index, "probe_total": probe_total, "probe_id": probe.probe_id, "layer_index": layer_index, "layer_total": len(connector.layer_ids), "backend": backend, "finite": True, "estimated_forward_calls": 1 + differential_calls, "estimated_stencil_samples": 1 + (4 * jacobian_rank) + (4 * hessian_rank), "estimated_differential_calls": differential_calls, "jacobian_kind": jacobian_kind, "jacobian_rank": jacobian_rank, "hessian_kind": hessian_kind, "hessian_rank": hessian_rank, "callback_nominal_reuse": bool(getattr(connector, "hessian_reuses_nominal_output", False)), "estimated_peak_memory_bytes": differential_memory, "differential_elapsed_seconds": float(time.perf_counter() - differential_started)})
                normalization = connector.normalization_geometry(source, layer_index, probe) if capabilities.normalization_geometry else None
                attention = connector.attention_geometry(source, layer_index, probe) if capabilities.attention_geometry else None
                previous_delta = transition_records[-1].delta if transition_records else None
                transition_records.append(
                    TransitionObservation(
                        source_layer=layer_ids[layer_index],
                        target_layer=layer_ids[layer_index + 1],
                        source_depth=float(coordinates[layer_index]),
                        target_depth=float(coordinates[layer_index + 1]),
                        source_state=source,
                        target_state=target,
                        delta=delta,
                        vector_field=delta / ds,
                        jacobian=jacobian,
                        hessian_sketch=hessian,
                        curvature=trajectory_curvature(previous_delta, delta, ds),
                        singular_values=singular,
                        normalization_geometry=normalization,
                        attention_geometry=attention,
                        # TransitionObservation.uncertainty is a numeric
                        # uncertainty map. String backend/capability fields
                        # live in trace.metadata and the journal payload.
                        uncertainty={"depth_coordinate": 1e-3, "finite_difference": max(self.jacobian_step, self.hessian_step), "jacobian_step_sensitivity": jacobian_sensitivity, "hessian_step_sensitivity": hessian_sensitivity, "jacobian_rank": jacobian_rank, "hessian_rank": hessian_rank, "estimated_differential_calls": differential_calls, "estimated_forward_calls": 1 + differential_calls, "estimated_peak_memory_bytes": differential_memory},
                    )
                )
            token_ids, position_ids, token_position_map = self._token_metadata(connector, probe)
            path_signature = multi_scale_path_signature(state_matrix, coordinates, area_seed=probe_seed)
            differential = differential_signature(TrajectoryTrace(
                model_id=str(connector.model_id),
                probe_id=probe.probe_id,
                layer_ids=layer_ids,
                depth_coordinates=coordinates,
                hidden_states=state_matrix,
                residual_states=None if residuals is None else np.asarray(residuals),
                transitions=tuple(transition_records),
            ))
            first_differential = differential_records[0] if differential_records else {}
            backends = {str(item.get("backend")) for item in differential_records}
            jacobian_kinds = {str(item.get("jacobian_kind")) for item in differential_records}
            hessian_kinds = {str(item.get("hessian_kind")) for item in differential_records}
            trace_differential = {"backend": next(iter(backends)) if len(backends) == 1 else "mixed", "jacobian_kind": next(iter(jacobian_kinds)) if len(jacobian_kinds) == 1 else "mixed", "jacobian_rank": first_differential.get("jacobian_rank", 0), "jacobian_directions": first_differential.get("jacobian_directions", 0), "hessian_kind": next(iter(hessian_kinds)) if len(hessian_kinds) == 1 else "mixed", "hessian_rank": first_differential.get("hessian_rank", 0), "estimated_differential_calls": int(sum(item.get("estimated_differential_calls", 0) for item in differential_records)), "estimated_forward_calls": int(sum(item.get("estimated_forward_calls", 0) for item in differential_records)), "estimated_stencil_samples": int(sum(item.get("estimated_stencil_samples", 0) for item in differential_records)), "estimated_calls": int(sum(item.get("estimated_calls", 0) for item in differential_records)), "estimated_peak_memory_bytes": int(max((item.get("estimated_peak_memory_bytes", 0) for item in differential_records), default=0)), "estimated_callback_batch_memory_bytes": int(max((item.get("estimated_callback_batch_memory_bytes", 0) for item in differential_records), default=0)), "callback_batched": any(bool(item.get("callback_batched", False)) for item in differential_records), "callback_two_step_cache": any(bool(item.get("callback_two_step_cache", False)) for item in differential_records), "callback_nominal_reuse": any(bool(item.get("callback_nominal_reuse", False)) for item in differential_records), "max_exact_jacobian_dim": int(self.max_exact_jacobian_dim), "auto_downgraded": any(bool(item.get("auto_downgraded", False)) for item in differential_records), "skipped_reason": next((item.get("jacobian_skipped_reason") or item.get("hessian_skipped_reason") for item in differential_records if item.get("jacobian_skipped_reason") or item.get("hessian_skipped_reason")), None)}
            traces.append(
                TrajectoryTrace(
                    model_id=str(connector.model_id),
                    probe_id=probe.probe_id,
                    layer_ids=layer_ids,
                    depth_coordinates=coordinates,
                    hidden_states=state_matrix,
                    residual_states=None if residuals is None else np.asarray(residuals),
                    transitions=tuple(transition_records),
                    metadata={"capabilities": capabilities.to_dict(), "depth_mode": self.depth_mode, "probe_split": probe.split, "probe_family": probe.family, "probe_pair_id": probe.pair_id, "probe_seed": probe_seed, "probe_fingerprint": probe_fingerprint(probe), "perturbation_norm": None if probe.perturbation is None else float(stable_l2(probe.perturbation, name="probe perturbation")), "state_layout": getattr(connector, "state_shape", None), "differential_observation": trace_differential, "path_signature": {"scales": list(path_signature.scales), "values": path_signature.values.tolist(), "endpoint_displacement": path_signature.endpoint_displacement, "total_arc_length": path_signature.total_arc_length, "turning_total": path_signature.turning_total, "area_backend": path_signature.area_backend, "area_sketch_rank": path_signature.area_sketch_rank}, "differential_signature": {"values": differential.values.tolist(), "available": differential.available.tolist(), "feature_names": list(differential.feature_names)}},
                    feature_space=str(getattr(connector, "feature_space", "hidden_state")),
                    token_ids=token_ids,
                    position_ids=position_ids,
                    token_position_map=token_position_map,
                    uncertainty=np.full(len(coordinates), 1e-3, dtype=np.float64),
                )
            )
            if trace_callback is not None:
                trace_callback(traces[-1], probe_index, probe_total)
            if progress_callback is not None:
                    progress_callback({"event": "probe_complete", "probe_index": probe_index, "probe_total": probe_total, "probe_id": probe.probe_id, "layer_index": len(connector.layer_ids) - 1, "layer_total": len(connector.layer_ids), "finite": True, "probe_elapsed_seconds": float(time.perf_counter() - probe_started), "estimated_forward_calls": trace_differential.get("estimated_forward_calls", 0), "estimated_stencil_samples": trace_differential.get("estimated_stencil_samples", 0), "callback_nominal_reuse": trace_differential.get("callback_nominal_reuse", False), "estimated_peak_memory_bytes": trace_differential.get("estimated_peak_memory_bytes", 0)})
        return traces

    @staticmethod
    def _token_metadata(connector: Any, probe: Probe) -> tuple[np.ndarray | None, np.ndarray | None, Mapping[str, Any]]:
        callback = getattr(connector, "token_position_metadata", None)
        supplied = dict(callback(probe) if callback is not None else {})
        payload = dict(probe.payload)
        token_ids = supplied.get("token_ids", payload.get("token_ids"))
        position_ids = supplied.get("position_ids", payload.get("position_ids"))
        mapping = supplied.get("token_position_map", payload.get("token_position_map", {}))
        return (None if token_ids is None else np.asarray(token_ids), None if position_ids is None else np.asarray(position_ids), dict(mapping))

    @staticmethod
    def _adaptive_coordinates(states: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(np.diff(states, axis=0), axis=1)
        distances = np.maximum(distances, 1e-8)
        cumulative = np.concatenate([[0.0], np.cumsum(distances)])
        return cumulative / cumulative[-1]
