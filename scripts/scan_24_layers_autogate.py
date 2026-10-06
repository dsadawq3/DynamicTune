"""Cross-Layer Representation Manifold & Spectral Entropy Auto-Gate Scanner.

Profiles the depth-wise dynamical flow between student and teacher transformers:
1. Extracts hidden state trajectories for student and teacher models on calibration prompts.
2. Maps student layer s_idx to proportional teacher layer t_idx across arbitrary model depths.
3. Fits local multi-chart Procrustes atlases between student and teacher representations.
4. Computes normalized Shannon spectral entropy H in [0, 1] of the trajectory velocity residual.
5. Diagnoses each layer as either a coherent anchor candidate (low entropy) or a chaotic
   polysemantic superposition knot (high entropy) that must be skipped to avoid collapse.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from faytuna_flow.knots import compute_subspace_entropy
from faytuna_flow.manifold_charts import build_multi_chart_atlas
from scripts.run_qwen35_transfer import (
    TRAIN_PROMPTS,
    collect_hidden_states,
    resolve_device,
)


def default_model_path(local_preferred: str, fallback_hf: str) -> str:
    """Return local path if it exists on disk, otherwise default to Hugging Face ID."""
    p = Path(local_preferred)
    if p.exists():
        return str(p)
    return fallback_hf


def get_prompt_hash(prompts: Sequence[str]) -> str:
    """Compute deterministic short hash for prompt list."""
    return hashlib.sha256("".join(prompts).encode("utf-8")).hexdigest()[:10]


def collect_or_load_traces(
    model_identifier: str,
    prompts: Sequence[str],
    *,
    role: str = "model",
    device_obj: Any = "cpu",
    batch_size: int = 4,
    cache_dir: Path | None = None,
    force_recompute: bool = False,
) -> tuple[list[dict[int, np.ndarray]], int]:
    """Collect hidden states or load from cache if available."""
    p_hash = get_prompt_hash(prompts)
    safe_name = "".join(c if c.isalnum() else "_" for c in model_identifier).strip("_")
    
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{role}_{safe_name}_{p_hash}.pkl"

    if cache_file is not None and cache_file.exists() and not force_recompute:
        print(f"[*] Found cached {role} traces at {cache_file}. Loading...")
        t0 = time.time()
        with open(cache_file, "rb") as f:
            traces = pickle.load(f)
        num_layers = len(traces[0]) - 1
        print(f"[+] Loaded {role} traces ({len(traces)} prompts, {num_layers} layers) in {time.time() - t0:.2f}s")
        return traces, num_layers

    print(f"\n[*] Loading {role} tokenizer & model from '{model_identifier}'...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_identifier,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"[+] {role} loaded in {time.time() - t0:.2f}s ({sum(p.numel() for p in model.parameters()):,} params)")

    print(f"[*] Collecting {role} hidden state traces on device '{device_obj}' (batch_size={batch_size})...")
    t0 = time.time()
    traces = collect_hidden_states(model, tokenizer, prompts, device=device_obj, batch_size=batch_size)
    num_layers = len(traces[0]) - 1
    print(f"[+] Collected traces for {len(traces)} prompts ({num_layers} layers) in {time.time() - t0:.2f}s")

    # Save to cache if cache directory provided
    if cache_file is not None:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(traces, f)
            print(f"[+] Saved {role} traces to cache: {cache_file}")
        except Exception as e:
            print(f"[-] Warning: Failed to save cache ({e})")

    # Free model memory immediately
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return traces, num_layers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DynamicTune: Cross-Layer Representation Manifold & Spectral Entropy Auto-Gate Scanner"
    )
    parser.add_argument(
        "--student-path",
        "--student-dir",
        dest="student_path",
        type=str,
        default=default_model_path("C:/models/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-0.8B-Base"),
        help="Path or Hugging Face ID of the student model (e.g. Qwen/Qwen3.5-0.8B-Base or Qwen/Qwen3.5-4B-Base)",
    )
    parser.add_argument(
        "--teacher-path",
        "--teacher-dir",
        dest="teacher_path",
        type=str,
        default=default_model_path("C:/models/Qwen3.5-4B-Base", "Qwen/Qwen3.5-4B-Base"),
        help="Path or Hugging Face ID of the teacher model (e.g. Qwen/Qwen3.5-4B-Base or Qwen/Qwen3.5-27B-Base)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "dml"],
        help="Execution device for trace extraction (auto, cuda, cpu, dml)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for hidden state trace collection",
    )
    parser.add_argument(
        "--n-charts",
        type=int,
        default=4,
        help="Number of manifold charts for local Procrustes atlas",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.88,
        help="Subspace spectral entropy threshold to classify as anchor candidate vs knot (default: 0.88)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="runs/cache",
        help="Directory to cache extracted hidden state traces",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of hidden state traces, ignoring cache files",
    )
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="default",
        choices=["default", "v2", "file"],
        help="Calibration prompt source (default: 8 standard baseline prompts)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Custom prompt JSON file if --prompt-source file",
    )
    args = parser.parse_args()

    device_obj, device_type = resolve_device(args.device)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve calibration prompts
    if args.prompt_source == "v2":
        from faytuna_flow.prompt_generator import CalibrationPromptEngine
        engine = CalibrationPromptEngine(seed=42)
        suite = engine.generate_calibration_suite(num_prompts=16)
        prompts = [s["text"] for s in suite]
    elif args.prompt_source == "file" and args.prompt_file:
        from faytuna_flow.prompt_generator import CalibrationPromptEngine
        prompts = CalibrationPromptEngine.load_calibration_dataset(args.prompt_file)
    else:
        prompts = list(TRAIN_PROMPTS)

    print("=" * 80)
    print("  DYNAMICTUNE: CROSS-LAYER SPECTRAL ENTROPY & KNOT AUTO-GATE SCANNER")
    print(f"  Student Model   : {args.student_path}")
    print(f"  Teacher Model   : {args.teacher_path}")
    print(f"  Device          : {device_type} ({device_obj})")
    print(f"  Prompts Count   : {len(prompts)}")
    print(f"  Manifold Charts : {args.n_charts}")
    print(f"  Knot Threshold  : H < {args.entropy_threshold:.2f} (Candidates) | H >= {args.entropy_threshold:.2f} (Knots)")
    print("=" * 80)

    # 2. Extract Teacher Traces first, then unload
    t_traces, n_t = collect_or_load_traces(
        args.teacher_path,
        prompts,
        role="teacher",
        device_obj=device_obj,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        force_recompute=args.force_recompute,
    )

    # 3. Extract Student Traces, then unload
    s_traces, n_s = collect_or_load_traces(
        args.student_path,
        prompts,
        role="student",
        device_obj=device_obj,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        force_recompute=args.force_recompute,
    )

    print("\n" + "=" * 80)
    print(f"  SCANNING {n_s} STUDENT LAYERS MAPPED TO {n_t} TEACHER LAYERS")
    print("=" * 80)
    print(f"{'Student Layer':<14} | {'Mapped Teacher':<14} | {'Spectral Entropy H':<20} | {'Auto-Gate Verdict'}")
    print("-" * 80)

    candidates: list[tuple[int, int, float]] = []
    layer_results: list[dict[str, Any]] = []

    for s_idx in range(n_s):
        t_idx = int(round(s_idx * (n_t - 1) / max(1, n_s - 1)))

        s_in = np.concatenate([tr[s_idx] for tr in s_traces], axis=0)
        s_out = np.concatenate([tr[s_idx + 1] for tr in s_traces], axis=0)
        t_in = np.concatenate([tr[t_idx] for tr in t_traces], axis=0)
        t_out = np.concatenate([tr[t_idx + 1] for tr in t_traces], axis=0)

        atlas_in = build_multi_chart_atlas(s_in, t_in, n_charts=args.n_charts, random_state=42 + s_idx)
        atlas_out = build_multi_chart_atlas(s_out, t_out, n_charts=args.n_charts, random_state=100 + s_idx)

        t_in_proj = atlas_in.project_teacher_to_student(t_in, s_in)
        t_out_proj = atlas_out.project_teacher_to_student(t_out, s_out)

        flow_res = (t_out_proj - t_in_proj) - (s_out - s_in)
        ent = float(compute_subspace_entropy(flow_res))

        is_cand = ent < args.entropy_threshold
        if is_cand:
            candidates.append((s_idx, t_idx, ent))
            verdict = "*** ANCHOR CANDIDATE (Low Entropy) ***"
        else:
            verdict = "Knot / Polysemantic Superposition (Skip)"

        print(f"Layer {s_idx:02d}       | Layer {t_idx:02d}       | {ent:18.4f}   | {verdict}")
        layer_results.append({
            "student_layer": s_idx,
            "teacher_layer": t_idx,
            "spectral_entropy": ent,
            "is_candidate": is_cand,
            "verdict": verdict,
        })

    print("-" * 80)
    print(f"\n[Summary] Total Transferable Anchor Candidates Found: {len(candidates)} / {n_s}")
    for c in candidates:
        print(f"  * Student Layer {c[0]:02d} -> Teacher Layer {c[1]:02d} (Spectral Entropy: {c[2]:.4f})")

    if not candidates:
        print("  [Warning] No layers fell below threshold. Recommend checking Layer 0 and final output layer.")

    report_file = cache_dir / "scan_entropy_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "student_model": args.student_path,
            "teacher_model": args.teacher_path,
            "student_layers": n_s,
            "teacher_layers": n_t,
            "entropy_threshold": args.entropy_threshold,
            "candidates": candidates,
            "layers": layer_results,
        }, f, indent=2)
    print(f"\n[+] Scan report written to {report_file}")


if __name__ == "__main__":
    main()
