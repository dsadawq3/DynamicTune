"""Reproducible observation performance benchmark without model checkpoints.

The benchmark deliberately uses an elementwise nonlinear system at the real
GPT-2 XL sequence chart dimension.  It measures dispatch overhead and batch
construction separately from model-family behavior while preserving the full
directional finite-difference stencil.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

from .connectors import SyntheticConnector
from .model_families import strict_json_payload
from .observation import ObservationProtocol
from .types import Probe, TrajectoryTrace, finite_array


def memory_snapshot() -> dict[str, int | None]:
    """Read current process memory with no mandatory psutil dependency."""

    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            process = kernel32.GetCurrentProcess()
            query = psapi.GetProcessMemoryInfo
            query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
            query.restype = ctypes.c_int
            ok = query(process, ctypes.byref(counters), counters.cb)
            if ok:
                return {"rss_bytes": int(counters.WorkingSetSize), "private_bytes": int(counters.PrivateUsage), "commit_bytes": int(counters.PagefileUsage)}
        except (AttributeError, OSError, TypeError):
            pass
    else:
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                key, _, raw = line.partition(":")
                if key in {"VmRSS", "VmSize"}:
                    values[key] = int(raw.strip().split()[0]) * 1024
            return {"rss_bytes": values.get("VmRSS"), "private_bytes": None, "commit_bytes": values.get("VmSize")}
        except (FileNotFoundError, OSError, ValueError):
            pass
    return {"rss_bytes": None, "private_bytes": None, "commit_bytes": None}


class _MemorySampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak = memory_snapshot()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.update()

    def update(self) -> None:
        current = memory_snapshot()
        for key in self.peak:
            value = current.get(key)
            if value is not None and (self.peak[key] is None or value > self.peak[key]):
                self.peak[key] = value

    def __enter__(self) -> "_MemorySampler":
        self.update()
        self._thread = threading.Thread(target=self._sample, name="faytuna-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, 4.0 * self.interval_seconds))
        self.update()


class _ElementwiseNonlinearSystem:
    """Actual-size nonlinear chart with a vectorizable, exact batch path."""

    def __init__(self, *, layers: int, state_dim: int, seed: int) -> None:
        if layers < 1 or state_dim < 1:
            raise ValueError("benchmark layers and state_dim must be positive")
        rng = np.random.default_rng(seed)
        self.layer_count = int(layers)
        self.state_dim = int(state_dim)
        self.gains = np.asarray(0.70 + 0.04 * rng.normal(size=(layers, state_dim)), dtype=np.float64)
        self.biases = np.asarray(0.01 * rng.normal(size=(layers, state_dim)), dtype=np.float64)

    def initial_state(self, probe: Probe) -> np.ndarray:
        return finite_array(probe.initial_state, ndim=1, name="benchmark initial state").copy()

    def _evaluate(self, values: np.ndarray, layer_index: int) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        z = x * self.gains[int(layer_index)] + self.biases[int(layer_index)]
        return x + 0.01 * np.tanh(z) + 0.0005 * np.sin(x)

    def step(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray:
        return finite_array(self._evaluate(finite_array(state, ndim=1, name="benchmark state"), layer_index), ndim=1, name="benchmark next state")

    def step_batch(self, states: np.ndarray, layer_index: int) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.state_dim:
            raise ValueError("benchmark batch has the wrong shape")
        result = self._evaluate(values, layer_index)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("benchmark batch produced non-finite values")
        return result


class _CountingSingleConnector(SyntheticConnector):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.dispatches = 0
        self.samples = 0

    def transition(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray:
        self.dispatches += 1
        self.samples += 1
        return super().transition(state, layer_index, probe)


class _CountingBatchConnector(SyntheticConnector):
    """Synthetic connector with GPT-2-style two-step batched callbacks."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dispatches = 0
        self.samples = 0
        self.jacobian_two_step_calls = 1
        self.hessian_two_step_calls = 1
        self.hessian_reuses_nominal_output = True
        self._jacobian_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._hessian_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._nominal_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    def _dispatch_batch(self, states: np.ndarray, layer_index: int) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        self.dispatches += 1
        self.samples += int(values.shape[0])
        return self.model.step_batch(values, int(layer_index))  # type: ignore[attr-defined]

    @staticmethod
    def _key(kind: str, layer_index: int, probe: Probe, directions: np.ndarray, step: float) -> tuple[Any, ...]:
        packed = np.ascontiguousarray(directions, dtype=np.float64)
        return (kind, int(layer_index), str(probe.probe_id), float(step), packed.shape, packed.tobytes())

    @staticmethod
    def _units(state: np.ndarray, directions: np.ndarray, step: float) -> tuple[np.ndarray, float]:
        values = np.asarray(directions, dtype=np.float64)
        norms = np.linalg.norm(values, axis=1)
        if values.ndim != 2 or values.shape[0] < 1 or not np.all(np.isfinite(values)) or np.any(norms <= 0.0):
            raise ValueError("benchmark directions must be finite and nonzero")
        local_step = float(step) * max(1.0, float(np.linalg.norm(state)))
        return values / norms[:, None], local_step

    def transition(self, state: np.ndarray, layer_index: int, probe: Probe) -> np.ndarray:
        self.dispatches += 1
        self.samples += 1
        values = super().transition(state, layer_index, probe)
        self._nominal_cache[(int(layer_index), str(probe.probe_id))] = (np.asarray(state, dtype=np.float64).copy(), values.copy())
        return values

    def jacobian_sketch(self, state: np.ndarray, layer_index: int, probe: Probe, directions: np.ndarray, step: float) -> np.ndarray:
        key = self._key("jacobian", layer_index, probe, directions, step)
        cached = self._jacobian_cache.pop(key, None)
        if cached is not None:
            return cached
        units, local_step = self._units(state, directions, step)
        offsets = local_step * units
        points = np.concatenate((state[None, :] + offsets, state[None, :] - offsets, state[None, :] + 2.0 * offsets, state[None, :] - 2.0 * offsets), axis=0)
        outputs = self._dispatch_batch(points, layer_index)
        count = len(units)
        current = ((outputs[:count] - outputs[count : 2 * count]) / (2.0 * local_step)).T
        alternate = ((outputs[2 * count : 3 * count] - outputs[3 * count :]) / (4.0 * local_step)).T
        self._jacobian_cache[self._key("jacobian", layer_index, probe, directions, step * 2.0)] = alternate
        return current

    def hessian_sketch(self, state: np.ndarray, layer_index: int, probe: Probe, directions: np.ndarray, step: float) -> np.ndarray:
        key = self._key("hessian", layer_index, probe, directions, step)
        cached = self._hessian_cache.pop(key, None)
        if cached is not None:
            return cached
        units, local_step = self._units(state, directions, step)
        offsets = local_step * units
        nominal = self._nominal_cache.get((int(layer_index), str(probe.probe_id)))
        reuse_nominal = nominal is not None and np.array_equal(nominal[0], state)
        if reuse_nominal:
            center = nominal[1]
            points = np.concatenate((state[None, :] + offsets, state[None, :] - offsets, state[None, :] + 2.0 * offsets, state[None, :] - 2.0 * offsets), axis=0)
            outputs = self._dispatch_batch(points, layer_index)
        else:
            points = np.concatenate((state[None, :], state[None, :] + offsets, state[None, :] - offsets, state[None, :] + 2.0 * offsets, state[None, :] - 2.0 * offsets), axis=0)
            outputs = self._dispatch_batch(points, layer_index)
            center = outputs[0]
            outputs = outputs[1:]
        count = len(units)
        current = (outputs[:count] - 2.0 * center[None, :] + outputs[count : 2 * count]) / (local_step * local_step)
        alternate = (outputs[2 * count : 3 * count] - 2.0 * center[None, :] + outputs[3 * count :]) / (4.0 * local_step * local_step)
        self._hessian_cache[self._key("hessian", layer_index, probe, directions, step * 2.0)] = alternate
        self._nominal_cache.pop((int(layer_index), str(probe.probe_id)), None)
        return current

    def estimate_differential_memory_bytes(self, jacobian_rank: int, hessian_rank: int) -> int:
        batch = max(4 * int(jacobian_rank), 4 * int(hessian_rank))
        return int(2 * batch * self.state_dim * np.dtype(np.float64).itemsize)


@dataclass(frozen=True)
class BenchmarkConfig:
    state_dim: int = 16 * 1600
    layers: int = 48
    probes: int = 1
    jacobian_rank: int = 8
    hessian_rank: int = 4
    seed: int = 17
    memory_interval_seconds: float = 0.02

    def __post_init__(self) -> None:
        if min(self.state_dim, self.layers, self.probes, self.jacobian_rank, self.hessian_rank) < 1:
            raise ValueError("benchmark dimensions, ranks, and probe count must be positive")
        if self.memory_interval_seconds <= 0.0 or not np.isfinite(self.memory_interval_seconds):
            raise ValueError("benchmark memory interval must be positive and finite")


class _ProgressWriter:
    def __init__(self, jsonl_path: str | Path | None, text_path: str | Path | None, *, run_id: str, total_samples: int) -> None:
        self.jsonl = None if jsonl_path is None else Path(jsonl_path)
        self.text = None if text_path is None else Path(text_path)
        self.run_id = str(run_id)
        self.total_samples = int(total_samples)
        self.started = time.perf_counter()
        self.completed_samples = 0
        if self.jsonl is not None:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl.write_text("", encoding="utf-8")
        if self.text is not None:
            self.text.parent.mkdir(parents=True, exist_ok=True)
            self.text.write_text("", encoding="utf-8")

    def emit(self, variant: str, event: Mapping[str, Any]) -> None:
        now = time.perf_counter()
        elapsed = float(now - self.started)
        payload = dict(event)
        delta = int(payload.get("estimated_stencil_samples", 0)) if payload.get("event") == "finite_difference_dispatch" else 0
        self.completed_samples += max(0, delta)
        throughput = None if elapsed <= 0.0 else self.completed_samples / elapsed
        remaining = max(0, self.total_samples - self.completed_samples)
        eta = None if throughput is None or throughput <= 0.0 else remaining / throughput
        record = {
            "schema_version": "faytuna-benchmark-progress-v1",
            "run_id": self.run_id,
            "stage": "benchmark-observation",
            "variant": str(variant),
            "elapsed_seconds": elapsed,
            "work_unit_name": "stencil_samples",
            "work_units_delta": delta,
            "work_units_completed": self.completed_samples,
            "work_units_total": self.total_samples,
            "throughput_stencil_samples_per_second": throughput,
            "eta_seconds": eta,
            "memory": memory_snapshot(),
            **payload,
        }
        rendered = json.dumps(strict_json_payload(record), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if self.jsonl is not None:
            with self.jsonl.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered + "\n")
        if self.text is not None:
            memory = record["memory"]
            rss = "?" if memory["rss_bytes"] is None else f"{memory['rss_bytes'] / 2**20:.0f}MiB"
            line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} variant={variant} event={payload.get('event')} elapsed={elapsed:.2f}s samples/s={throughput or 0.0:.1f} eta={eta if eta is not None else '?'} rss={rss}"
            with self.text.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")


def _trace_difference(first: TrajectoryTrace, second: TrajectoryTrace) -> dict[str, float | bool]:
    differences: list[np.ndarray] = [np.asarray(first.hidden_states) - np.asarray(second.hidden_states), np.asarray(first.depth_coordinates) - np.asarray(second.depth_coordinates)]
    differences.extend([np.asarray(first.transitions[index].delta) - np.asarray(second.transitions[index].delta) for index in range(len(first.transitions))])
    differences.extend([np.asarray(first.transitions[index].vector_field) - np.asarray(second.transitions[index].vector_field) for index in range(len(first.transitions))])
    differences.extend([np.asarray(first.transitions[index].curvature) - np.asarray(second.transitions[index].curvature) for index in range(len(first.transitions))])
    differences.extend([np.asarray(first.transitions[index].jacobian) - np.asarray(second.transitions[index].jacobian) for index in range(len(first.transitions)) if first.transitions[index].jacobian is not None and second.transitions[index].jacobian is not None])
    differences.extend([np.asarray(first.transitions[index].hessian_sketch) - np.asarray(second.transitions[index].hessian_sketch) for index in range(len(first.transitions)) if first.transitions[index].hessian_sketch is not None and second.transitions[index].hessian_sketch is not None])
    differences.extend([np.asarray(first.transitions[index].singular_values) - np.asarray(second.transitions[index].singular_values) for index in range(len(first.transitions)) if first.transitions[index].singular_values is not None and second.transitions[index].singular_values is not None])
    absolute = np.concatenate([np.abs(value).reshape(-1) for value in differences])
    return {"max_abs_error": float(np.max(absolute)), "mean_abs_error": float(np.mean(absolute)), "all_finite": bool(np.all(np.isfinite(absolute)))}


def _run_variant(label: str, connector: Any, probes: tuple[Probe, ...], config: BenchmarkConfig, writer: _ProgressWriter | None) -> tuple[dict[str, Any], list[TrajectoryTrace]]:
    protocol = ObservationProtocol(jacobian_rank=config.jacobian_rank, hessian_rank=config.hessian_rank, jacobian_mode="directional", depth_mode="uniform")
    callback: Callable[[Mapping[str, Any]], None] | None = None if writer is None else lambda event: writer.emit(label, event)
    started = time.perf_counter()
    with _MemorySampler(config.memory_interval_seconds) as sampler:
        traces = protocol.collect(connector, probes, seed=config.seed, progress_callback=callback)
    elapsed = float(time.perf_counter() - started)
    samples = int(getattr(connector, "samples", 0))
    dispatches = int(getattr(connector, "dispatches", 0))
    memory = sampler.peak
    result: dict[str, Any] = {
        "variant": label,
        "elapsed_seconds": elapsed,
        "dispatches": dispatches,
        "stencil_samples": samples,
        "estimated_stencil_samples": int(sum(trace.metadata["differential_observation"]["estimated_stencil_samples"] for trace in traces)),
        "throughput_stencil_samples_per_second": None if elapsed <= 0.0 else samples / elapsed,
        "throughput_dispatches_per_second": None if elapsed <= 0.0 else dispatches / elapsed,
        "peak_memory": memory,
        "finite": bool(all(np.all(np.isfinite(trace.hidden_states)) for trace in traces)),
        "trace_count": len(traces),
        "path_signature_area_backend": traces[0].metadata.get("path_signature", {}).get("area_backend") if traces else None,
        "path_signature_area_sketch_rank": traces[0].metadata.get("path_signature", {}).get("area_sketch_rank") if traces else None,
    }
    return result, traces


def run_benchmark(config: BenchmarkConfig = BenchmarkConfig(), *, output: str | Path | None = None, progress_jsonl: str | Path | None = None, progress_log: str | Path | None = None) -> dict[str, Any]:
    """Run same-seed single and batch differential traces and compare them."""

    rng = np.random.default_rng(config.seed + 101)
    probes = tuple(Probe(f"benchmark-{index}", "actual_dimension", {"seed": config.seed, "index": index}, rng.normal(0.0, 0.12, config.state_dim)) for index in range(config.probes))
    system = _ElementwiseNonlinearSystem(layers=config.layers, state_dim=config.state_dim, seed=config.seed + 313)
    total_samples = config.probes * config.layers * (1 + 4 * config.jacobian_rank + 4 * config.hessian_rank)
    writer = None if progress_jsonl is None and progress_log is None else _ProgressWriter(progress_jsonl, progress_log, run_id=f"benchmark-{config.seed}", total_samples=2 * total_samples)
    single_result, single_traces = _run_variant("single_dispatch", _CountingSingleConnector(system, model_id="benchmark-single"), probes, config, writer)
    batch_result, batch_traces = _run_variant("batch_dispatch", _CountingBatchConnector(system, model_id="benchmark-batch"), probes, config, writer)
    errors = [_trace_difference(first, second) for first, second in zip(single_traces, batch_traces)]
    max_error = max((float(item["max_abs_error"]) for item in errors), default=0.0)
    mean_error = float(np.mean([float(item["mean_abs_error"]) for item in errors])) if errors else 0.0
    result: dict[str, Any] = {
        "schema_version": "faytuna-benchmark-v1",
        "status": "pass" if single_result["finite"] and batch_result["finite"] and max_error <= 1e-12 and single_result["stencil_samples"] == batch_result["stencil_samples"] == total_samples else "rejected",
        "claim_scope": "dispatch_and_observation_overhead_on_actual_size_synthetic_chart",
        "model_execution": "synthetic_only; no checkpoint or tokenizer",
        "config": {"state_dim": config.state_dim, "layers": config.layers, "probes": config.probes, "jacobian_rank": config.jacobian_rank, "hessian_rank": config.hessian_rank, "seed": config.seed},
        "expected_stencil_samples": total_samples,
        "single_dispatch": single_result,
        "batch_dispatch": batch_result,
        "trace_comparison": {"same_seed": True, "per_probe": errors, "max_abs_error": max_error, "mean_abs_error": mean_error, "all_finite": bool(all(bool(item["all_finite"]) for item in errors))},
        "dispatch_reduction_factor": None if batch_result["dispatches"] == 0 else single_result["dispatches"] / batch_result["dispatches"],
        "memory_note": "peak memory is sampled process memory; model-internal allocator peaks can be missed between samples",
    }
    safe = strict_json_payload(result)
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return safe
