"""Run a bounded real GPT-2 observation experiment from local checkpoints."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faytuna_flow.artifacts import load_traces, save_traces
from faytuna_flow.gpt2 import GPT2Connector, GPT2ProbePolicy, make_gpt2_probe_splits, make_gpt2_python_probe_splits
from faytuna_flow.model_families import strict_json_payload
from faytuna_flow.observation import ObservationProtocol, probe_fingerprint


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models" / "gpt2-clean"


def _event(path: Path, event: str, **payload: object) -> None:
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **payload}
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _memory_snapshot() -> dict[str, int | None]:
    """Read process memory without adding a required psutil dependency."""

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


def _strict(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return strict_json_payload(value)


class ProgressMonitor:
    """Line-buffered progress journal with probe/layer ETA and memory state."""

    def __init__(self, output: Path, *, run_id: str) -> None:
        self.output = output
        self.run_id = run_id
        self.started = time.perf_counter()
        self.completed_artifact_count = 0
        self.work_units_completed = 0.0
        output.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output / "progress.jsonl"
        self.text_path = output / "progress.log"
        self.jsonl = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
        self.text = self.text_path.open("a", encoding="utf-8", buffering=1)

    def emit(self, stage: str, event: str, *, progress_units: float | None = None, progress_total: int | None = None, completed_artifact: str | Path | None = None, work_units: float | None = None, work_total: float | None = None, work_unit_name: str = "work_units", **payload: object) -> None:
        elapsed = float(time.perf_counter() - self.started)
        total = None if progress_total is None else int(progress_total)
        current = None if progress_units is None else float(progress_units)
        rate = None if current is None or current <= 0.0 or elapsed <= 0.0 else current / elapsed
        eta = None if rate is None or total is None else max(0.0, (float(total) - current) / rate)
        work_delta = 0.0 if work_units is None else float(work_units)
        if not np.isfinite(work_delta) or work_delta < 0.0:
            raise ValueError("work_units must be finite and non-negative")
        self.work_units_completed += work_delta
        work_rate = None if self.work_units_completed <= 0.0 or elapsed <= 0.0 else self.work_units_completed / elapsed
        work_eta = None if work_rate is None or work_total is None else max(0.0, (float(work_total) - self.work_units_completed) / work_rate)
        if completed_artifact is not None:
            self.completed_artifact_count += 1
        record = {
            "schema_version": "faytuna-progress-v1",
            "run_id": self.run_id,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stage": stage,
            "event": event,
            "elapsed_seconds": elapsed,
            "progress_units": current,
            "progress_total": total,
            "rate_units_per_second": rate,
            "eta_seconds": eta,
            "work_unit_name": str(work_unit_name),
            "work_units_delta": work_delta,
            "work_units_completed": self.work_units_completed,
            "work_units_total": None if work_total is None else float(work_total),
            "throughput_work_units_per_second": work_rate,
            "work_eta_seconds": work_eta,
            "memory": _memory_snapshot(),
            "backend": payload.pop("backend", "HF/PyTorch"),
            "device": payload.pop("device", "cpu"),
            "completed_artifact": None if completed_artifact is None else str(completed_artifact),
            "completed_artifact_count": self.completed_artifact_count,
            **payload,
        }
        if "layer_index" in record and "block_index" not in record:
            record["block_index"] = record["layer_index"]
        encoded = json.dumps(_strict(record), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self.jsonl.write(encoded + "\n")
        self.jsonl.flush()
        probe = ""
        if "probe_index" in record and "probe_total" in record:
            probe = f" probe={int(record['probe_index']) + 1}/{record['probe_total']}"
        layer = ""
        if "layer_index" in record and "layer_total" in record and int(record["layer_total"]) > 0:
            layer = f" layer={int(record['layer_index']) + 1}/{record['layer_total']}"
        memory = record["memory"]
        rss = "?" if memory["rss_bytes"] is None else f"{memory['rss_bytes'] / 2**20:.0f}MiB"
        eta_text = "?" if eta is None else f"{eta:.1f}s"
        throughput_text = "?" if work_rate is None else f" {work_rate:.1f} {work_unit_name}/s"
        artifact = "" if completed_artifact is None else f" artifact={completed_artifact}"
        line = f"{record['time']} stage={stage} event={event}{probe}{layer} elapsed={elapsed:.1f}s eta={eta_text}{throughput_text} rss={rss}{artifact}"
        self.text.write(line + "\n")
        self.text.flush()

    def close(self) -> None:
        self.jsonl.close()
        self.text.close()


def _model_parameter_checksum(model: object) -> str:
    """Hash parameter tensors once; no copy of the full model is created."""

    digest = hashlib.sha256()
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        return "unavailable"
    for name, tensor in state_dict().items():
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        values = tensor.detach().to("cpu").contiguous().numpy()
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    rendered = json.dumps(_strict(payload), ensure_ascii=False, indent=2, allow_nan=False)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _trace_matches_probe(candidate: object, probe: object, connector: object, layer_schema: tuple[int, ...], jacobian_rank: int, hessian_rank: int) -> bool:
    """Validate cache identity before reusing a trace or split artifact."""

    if candidate is None:
        return False
    metadata = getattr(candidate, "metadata", {})
    differential = dict(metadata.get("differential_observation", {})) if isinstance(metadata, dict) else {}
    return bool(
        getattr(candidate, "probe_id", None) == getattr(probe, "probe_id", None)
        and getattr(candidate, "model_id", None) == getattr(connector, "model_id", None)
        and getattr(candidate, "layer_ids", None) == layer_schema
        and getattr(candidate, "state_dim", None) == getattr(connector, "state_dim", None)
        and tuple(metadata.get("state_layout", ())) == tuple(getattr(connector, "state_shape", ()))
        and metadata.get("probe_fingerprint") == probe_fingerprint(probe)
        and int(differential.get("jacobian_rank", -1)) == min(jacobian_rank, int(getattr(connector, "state_dim")))
        and int(differential.get("hessian_rank", -1)) == min(hessian_rank, int(getattr(connector, "state_dim")))
    )


def _policy(tokenizer: object, sequence_length: int) -> GPT2ProbePolicy:
    eos = int(getattr(tokenizer, "eos_token_id", 50256) or 50256)

    def encode(probe: object) -> dict[str, list[int]]:
        cached_payload = getattr(probe, "payload", {})
        # ``make_gpt2_probe_splits`` materializes tokens once.  Reusing those
        # exact ids avoids re-running the tokenizer for every layer/stencil
        # while keeping the observation chart and probe bytes unchanged.
        if isinstance(cached_payload, dict) and "input_ids" in cached_payload:
            return {
                "input_ids": [int(value) for value in cached_payload["input_ids"]],
                "position_ids": [int(value) for value in cached_payload.get("position_ids", range(sequence_length))],
                "attention_mask": [int(value) for value in cached_payload.get("attention_mask", [1] * sequence_length)],
            }
        family = str(getattr(probe, "family"))
        probe_id = str(getattr(probe, "probe_id"))
        code_text = cached_payload.get("code") or cached_payload.get("text")
        if code_text:
            text = str(code_text)
        else:
            payload = json.dumps(getattr(probe, "payload"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            text = f"{family} {probe_id} {payload}"
        values = tokenizer(text, add_special_tokens=False, truncation=True, max_length=sequence_length)["input_ids"]  # type: ignore[operator]
        values = [int(value) for value in values[:sequence_length]]
        values += [eos] * (sequence_length - len(values))
        return {
            "input_ids": values,
            "position_ids": list(range(sequence_length)),
            "attention_mask": [1] * sequence_length,
        }

    return GPT2ProbePolicy(encode=encode, sequence_length=sequence_length)


def _load_model(path: Path) -> tuple[object, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=False)
    except TypeError:
        # Older transformers releases use the deprecated spelling. The
        # fallback keeps the runner compatible without changing dtype.
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=False)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return tokenizer, model


def collect_variant(
    variant: str,
    checkpoint: Path,
    policy: GPT2ProbePolicy,
    split: object,
    output: Path,
    layers: tuple[int, ...] | None,
    jacobian_rank: int,
    hessian_rank: int,
    monitor: ProgressMonitor | None = None,
    resume: bool = True,
) -> dict[str, object]:
    import torch

    log = output / "run.jsonl"
    started = time.perf_counter()
    _event(log, "model_load_start", variant=variant, checkpoint=str(checkpoint))
    if monitor is not None:
        monitor.emit("model", "model_load_start", variant=variant, checkpoint=str(checkpoint), backend="HF/PyTorch", device="cpu")
    tokenizer, model = _load_model(checkpoint)
    del tokenizer
    connector = GPT2Connector(model, policy, variant=variant, model_id=f"real-{variant}", device="cpu")
    if layers is not None:
        if not layers or min(layers) < 0 or max(layers) >= len(connector.layers):
            raise ValueError(f"invalid layer subset {layers} for {len(connector.layers)} layers")
        connector.layer_ids = layers
    parameter = next(model.parameters())
    parameter_bytes = int(sum(int(value.numel()) * int(value.element_size()) for value in model.parameters()))
    checksum = _model_parameter_checksum(model)
    _event(log, "model_ready", variant=variant, total_layers=len(connector.layers), observed_layers=list(connector.layer_ids), state_dim=connector.state_dim, state_shape=list(connector.state_shape), dtype=str(parameter.dtype), device=str(parameter.device), parameter_bytes=parameter_bytes, model_checksum=checksum, model_checksum_kind="parameter_tensor_sha256", ram_policy="full_float32_model_in_host_ram; no quantization; no checkpoint reload between probes", capabilities=connector.capabilities.to_dict())
    if monitor is not None:
        monitor.emit("model", "model_ready", variant=variant, backend="HF/PyTorch", device=str(parameter.device), total_layers=len(connector.layers), observed_layers=list(connector.layer_ids), state_dim=connector.state_dim, state_shape=list(connector.state_shape), dtype=str(parameter.dtype), parameter_bytes=parameter_bytes, model_checksum=checksum, model_checksum_kind="parameter_tensor_sha256")
    protocol = ObservationProtocol(jacobian_rank=jacobian_rank, hessian_rank=hessian_rank, jacobian_mode="directional", depth_mode="uniform")
    all_probes = tuple(getattr(split, name) for name in ("train", "validation", "holdout"))
    estimated_samples_per_probe = len(connector.layer_ids) * (1 + 4 * min(jacobian_rank, connector.state_dim) + 4 * min(hessian_rank, connector.state_dim))
    estimated_work_total = float(sum(len(probes) for probes in all_probes) * estimated_samples_per_probe)
    counts: dict[str, int] = {}
    variant_dir = output / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    layer_schema = tuple([-1] + [int(value) for value in connector.layer_ids])
    manifest_path = variant_dir / "resume_manifest.json"
    signature = {
        "variant": variant,
        "checkpoint": str(checkpoint.resolve()),
        "model_checksum": checksum,
        "sequence_length": policy.sequence_length,
        "state_dim": connector.state_dim,
        "state_shape": list(connector.state_shape),
        "layer_ids": list(connector.layer_ids),
        "jacobian_rank": jacobian_rank,
        "hessian_rank": hessian_rank,
        "jacobian_step": protocol.jacobian_step,
        "hessian_step": protocol.hessian_step,
        "jacobian_mode": protocol.jacobian_mode,
        "depth_mode": protocol.depth_mode,
        "seed": 17,
    }
    resume_allowed = bool(resume)
    previous_manifest: dict[str, object] | None = None
    if resume_allowed and manifest_path.exists():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_manifest = loaded_manifest if isinstance(loaded_manifest, dict) else None
            resume_allowed = previous_manifest is not None and previous_manifest.get("signature") == signature
        except (OSError, ValueError, TypeError):
            resume_allowed = False
            previous_manifest = None
        if not resume_allowed and monitor is not None:
            monitor.emit("resume", "resume_manifest_mismatch", variant=variant, backend="HF/PyTorch", device="cpu", skipped_reason="checkpoint, model checksum, chart, layer schema, rank, step, or seed changed")
    preserved_splits = previous_manifest.get("splits", {}) if resume_allowed and previous_manifest is not None else {}
    manifest: dict[str, object] = {"schema_version": "faytuna-gpt2-resume-v1", "status": "running", "signature": signature, "splits": dict(preserved_splits) if isinstance(preserved_splits, dict) else {}}
    _write_json(manifest_path, manifest)
    if monitor is not None:
        monitor.emit("resume", "resume_ready", variant=variant, backend="HF/PyTorch", device="cpu", resume_requested=bool(resume), resume_allowed=resume_allowed, work_total=estimated_work_total, work_unit_name="stencil_samples", completed_artifact=manifest_path if resume_allowed else None)
    for name, probes in zip(("train", "validation", "holdout"), all_probes):
        split_started = time.perf_counter()
        _event(log, "observation_start", variant=variant, split=name, probes=len(probes), jacobian_rank=jacobian_rank, hessian_rank=hessian_rank)
        if monitor is not None:
            monitor.emit("observation", "split_start", variant=variant, split=name, probe_total=len(probes), progress_units=0.0, progress_total=len(probes), backend="HF/PyTorch", device="cpu")
        current: list[object] = []
        probe_dir = variant_dir / "probes" / name
        probe_dir.mkdir(parents=True, exist_ok=True)
        previous_splits = {} if previous_manifest is None else previous_manifest.get("splits", {})
        previous_split = previous_splits.get(name) if isinstance(previous_splits, dict) else None
        split_artifact = variant_dir / f"{name}.npz"
        if resume_allowed and isinstance(previous_split, dict) and previous_split.get("status") == "complete" and split_artifact.exists():
            try:
                split_loaded = load_traces(split_artifact)
                if len(split_loaded) == len(probes) and all(_trace_matches_probe(candidate, probe, connector, layer_schema, jacobian_rank, hessian_rank) for candidate, probe in zip(split_loaded, probes)):
                    current.extend(split_loaded)
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                current.clear()
        if len(current) == len(probes):
            counts[name] = len(current)
            for probe_index, probe in enumerate(probes):
                if monitor is not None:
                    monitor.emit("observation", "probe_resumed", variant=variant, split=name, probe_index=probe_index, probe_total=len(probes), layer_index=len(connector.layer_ids) - 1, layer_total=len(connector.layer_ids), progress_units=float(probe_index + 1), progress_total=len(probes), finite=True, backend="HF/PyTorch", device="cpu", completed_artifact=probe_dir / f"probe_{probe_index:05d}.npz", cache_source=split_artifact, resume_validation="split_artifact_and_probe_fingerprint")
            if monitor is not None:
                monitor.emit("observation", "split_resumed", variant=variant, split=name, probe_count=len(current), progress_units=float(len(current)), progress_total=len(probes), finite=True, backend="HF/PyTorch", device="cpu", completed_artifact=split_artifact, cache_source=split_artifact, resume_validation="split_artifact_and_probe_fingerprint")
            del current
            continue
        for probe_index, probe in enumerate(probes):
            probe_path = probe_dir / f"probe_{probe_index:05d}.npz"
            cached = None
            if resume_allowed and probe_path.exists() and probe_path.with_suffix(probe_path.suffix + ".json").exists():
                try:
                    loaded = load_traces(probe_path)
                    candidate = loaded[0] if len(loaded) == 1 else None
                    if _trace_matches_probe(candidate, probe, connector, layer_schema, jacobian_rank, hessian_rank):
                        cached = candidate
                except (OSError, ValueError, KeyError, IndexError, TypeError):
                    cached = None
            if cached is not None:
                current.append(cached)
                if monitor is not None:
                    monitor.emit("observation", "probe_resumed", variant=variant, split=name, probe_index=probe_index, probe_total=len(probes), layer_index=len(connector.layer_ids) - 1, layer_total=len(connector.layer_ids), progress_units=float(probe_index + 1), progress_total=len(probes), finite=True, backend="HF/PyTorch", device="cpu", completed_artifact=probe_path)
                continue

            def report(event: object, *, _probe_index: int = probe_index, _probe_total: int = len(probes), _name: str = name) -> None:
                if monitor is None or not isinstance(event, dict):
                    return
                payload = dict(event)
                event_name = str(payload.pop("event", "observation"))
                layer_index = int(payload.get("layer_index", -1))
                layer_total = int(payload.get("layer_total", len(connector.layer_ids)))
                fraction = 0.0 if layer_index < 0 else min(1.0, float(layer_index + 1) / max(1, layer_total))
                completed = float(_probe_index) + (1.0 if event_name == "probe_complete" else fraction)
                payload.pop("probe_index", None)
                payload.pop("probe_total", None)
                payload.pop("layer_index", None)
                payload.pop("layer_total", None)
                payload.pop("backend", None)
                payload.pop("device", None)
                work_units = payload.get("estimated_stencil_samples") if event_name == "finite_difference_dispatch" else None
                monitor.emit("observation", event_name, variant=variant, split=_name, probe_index=_probe_index, probe_total=_probe_total, progress_units=completed, progress_total=_probe_total, work_units=None if work_units is None else float(work_units), work_total=estimated_work_total, work_unit_name="stencil_samples", backend="HF/PyTorch", device="cpu", **payload)

            # Observation is inference-only. inference_mode removes autograd
            # bookkeeping without changing weights, dtype, points, or outputs.
            with torch.inference_mode():
                result = protocol.collect(connector, [probe], seed=17, progress_callback=report)
            if len(result) != 1:
                raise RuntimeError(f"observation returned {len(result)} traces for one probe")
            trace = result[0]
            serialization_started = time.perf_counter()
            save_traces([trace], probe_path, compressed=False, durable=True)
            serialization_elapsed = float(time.perf_counter() - serialization_started)
            current.append(trace)
            if monitor is not None:
                monitor.emit("observation", "probe_artifact_saved", variant=variant, split=name, probe_index=probe_index, probe_total=len(probes), layer_index=len(connector.layer_ids) - 1, layer_total=len(connector.layer_ids), progress_units=float(probe_index + 1), progress_total=len(probes), finite=True, backend="HF/PyTorch", device="cpu", completed_artifact=probe_path, serialization_elapsed_seconds=serialization_elapsed, artifact_durable=True)
        counts[name] = len(current)
        metadata = current[0].metadata if current else {}
        save_traces(current, variant_dir / f"{name}.npz")
        manifest["splits"] = {**dict(manifest.get("splits", {})), name: {"status": "complete", "probe_count": len(current), "artifact": str(variant_dir / f"{name}.npz"), "probe_artifacts": [str(variant_dir / "probes" / name / f"probe_{index:05d}.npz") for index in range(len(current))]}}
        _write_json(manifest_path, manifest)
        _event(log, "observation_complete", variant=variant, split=name, traces=len(current), elapsed_seconds=round(time.perf_counter() - split_started, 3), sample_metadata=metadata)
        if monitor is not None:
            monitor.emit("observation", "split_complete", variant=variant, split=name, probe_count=len(current), progress_units=float(len(current)), progress_total=len(probes), finite=True, backend="HF/PyTorch", device="cpu", completed_artifact=variant_dir / f"{name}.npz")
        del current
    result = {"variant": variant, "counts": counts, "observed_layers": list(connector.layer_ids), "state_dim": connector.state_dim, "model_checksum": checksum, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    _write_json(variant_dir / "summary.json", result)
    manifest["status"] = "complete"
    manifest["summary"] = result
    _write_json(manifest_path, manifest)
    _event(log, "variant_complete", **result)
    if monitor is not None:
        monitor.emit("run", "variant_complete", variant=variant, backend="HF/PyTorch", device="cpu", progress_units=1.0, progress_total=1, completed_artifact=variant_dir / "summary.json", counts=counts, observed_layers=list(connector.layer_ids), state_dim=connector.state_dim, model_checksum=checksum)
    del connector, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _proportional_student_layers(teacher_layers: tuple[int, ...] | None, teacher_total: int = 48, student_total: int = 12) -> tuple[int, ...] | None:
    if teacher_layers is None:
        return None
    if all(0 <= x < student_total for x in teacher_layers) and len(teacher_layers) >= 4:
        return tuple(sorted(set(teacher_layers)))
    mapped = sorted({int(round(l * (student_total - 1) / max(1, teacher_total - 1))) for l in teacher_layers if 0 <= l < teacher_total})
    return tuple(mapped) if mapped else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "real_runs" / "gpt2")
    parser.add_argument("--sequence-length", type=int, default=2)
    parser.add_argument("--per-family", type=int, default=3)
    parser.add_argument("--jacobian-rank", type=int, default=1)
    parser.add_argument("--hessian-rank", type=int, default=1)
    parser.add_argument("--layers", default="0,11,23,35,47", help="comma-separated observed teacher layer ids; empty means every layer")
    parser.add_argument("--student-layers", default=None, help="comma-separated observed student layer ids; defaults to proportional depth mapping of --layers")
    parser.add_argument("--variant", choices=("gpt2-small", "gpt2-xl", "both"), default="both")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="reuse validated per-probe artifacts; use --no-resume to recompute")
    parser.add_argument("--torch-threads", type=int, default=8, help="intra-op CPU threads; default 8 preserves the baseline setting")
    parser.add_argument("--domain", choices=("generic", "python"), default="generic", help="probe domain (generic synthetic or authentic python code)")
    args = parser.parse_args()
    if args.sequence_length < 1 or args.per_family < 3:
        raise SystemExit("sequence length must be positive and per-family must be >= 3")
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "run.jsonl"
    layers = None if not args.layers.strip() else tuple(int(value) for value in args.layers.split(","))
    student_layers = (
        tuple(int(value) for value in args.student_layers.split(","))
        if args.student_layers is not None and args.student_layers.strip()
        else _proportional_student_layers(layers, teacher_total=48, student_total=12)
    )
    import torch

    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    _event(log, "experiment_start", output=str(args.output), sequence_length=args.sequence_length, per_family=args.per_family, jacobian_rank=args.jacobian_rank, hessian_rank=args.hessian_rank, layers=None if layers is None else list(layers), student_layers=None if student_layers is None else list(student_layers), domain=args.domain, torch_threads=torch.get_num_threads())
    monitor = ProgressMonitor(args.output, run_id=f"gpt2-{int(time.time())}")
    monitor.emit("run", "experiment_start", output=str(args.output), variant=args.variant, sequence_length=args.sequence_length, per_family=args.per_family, jacobian_rank=args.jacobian_rank, hessian_rank=args.hessian_rank, layers=None if layers is None else list(layers), student_layers=None if student_layers is None else list(student_layers), domain=args.domain, torch_threads=torch.get_num_threads(), resume_requested=bool(args.resume), backend="HF/PyTorch", device="cpu")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / "gpt2-small", local_files_only=True, use_fast=True)
        policy = _policy(tokenizer, args.sequence_length)
        if args.domain == "python":
            split = make_gpt2_python_probe_splits(policy, per_family=args.per_family, seed=17, state_dim=8)
            paired_families = ["python_counterfactual", "python_adversarial"]
        else:
            split = make_gpt2_probe_splits(policy, per_family=args.per_family, seed=17, state_dim=8)
            paired_families = ["counterfactual", "adversarial_stability"]
        probe_records = len(split.train) + len(split.validation) + len(split.holdout)
        probe_quality = "pilot_low_sample" if args.per_family < 6 else "baseline_operator_fit" if args.per_family < 9 else "operator_fit_with_more_stable_family_statistics"
        _event(log, "probe_split_ready", train=len(split.train), validation=len(split.validation), holdout=len(split.holdout), total_records=probe_records, family_count=7, units_per_family=args.per_family, paired_families=paired_families, domain=args.domain, probe_quality=probe_quality, probe_quality_warning=None if args.per_family >= 6 else "per-family unit count below 6; rank/confidence estimates are pilot measurements")
        monitor.emit("probes", "probe_split_ready", train=len(split.train), validation=len(split.validation), holdout=len(split.holdout), total_records=probe_records, family_count=7, units_per_family=args.per_family, domain=args.domain, probe_quality=probe_quality, probe_quality_warning=None if args.per_family >= 6 else "per-family unit count below 6; rank/confidence estimates are pilot measurements", backend="HF/PyTorch", device="cpu")
        if args.variant in {"gpt2-xl", "both"}:
            collect_variant("gpt2-xl", MODEL_ROOT / "gpt2-xl", policy, split, args.output, layers, args.jacobian_rank, args.hessian_rank, monitor=monitor, resume=args.resume)
        if args.variant in {"gpt2-small", "both"}:
            collect_variant("gpt2-small", MODEL_ROOT / "gpt2-small", policy, split, args.output, student_layers, args.jacobian_rank, args.hessian_rank, monitor=monitor, resume=args.resume)
        _event(log, "experiment_complete")
        monitor.emit("run", "experiment_complete", variant=args.variant, backend="HF/PyTorch", device="cpu", progress_units=1.0, progress_total=1)
        return 0
    except Exception as error:
        monitor.emit("run", "experiment_failed", variant=args.variant, backend="HF/PyTorch", device="cpu", finite=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        monitor.close()


if __name__ == "__main__":
    sys.exit(main())
