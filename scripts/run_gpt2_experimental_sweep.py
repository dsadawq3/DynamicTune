"""Compatibility CLI wrapper for the central adaptive GPT-2 tune policy.

The orchestration implementation lives in
``faytuna_flow.adaptive_transfer.run_gpt2_weight_tune``. This historical
script name remains usable for saved commands and writes ``sweep.json`` and
``sweep.txt`` aliases beside the canonical ``tune.json`` and ``tune.txt``.
It never downloads a checkpoint or runs a new observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faytuna_flow.adaptive_transfer import run_gpt2_weight_tune
from faytuna_flow.connectors import CapabilityError


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    gains = tuple(float(value.strip()) for value in args.gains.split(",") if value.strip())
    if not gains or any(not np.isfinite(value) or value == 0.0 for value in gains):
        raise ValueError("--gains must be a non-empty comma-separated list of finite non-zero values")
    depth_sched = None if getattr(args, "depth_schedule", "boost_deep") in {None, "none", "flat"} else args.depth_schedule
    dual_ratio = args.dual_residual_ratio
    if dual_ratio != "auto":
        try:
            dual_ratio = float(dual_ratio)
        except ValueError:
            pass
    payload = run_gpt2_weight_tune(
        args.checkpoint_dir,
        args.student_trace,
        args.teacher_trace,
        args.alignment,
        args.mapping,
        args.output_root,
        sequence_length=args.sequence_length,
        gains=gains,
        mode="experimental_untrusted",
        device=args.device,
        seed=args.seed,
        max_dense_features=args.max_dense_features,
        scalable_rank=args.scalable_rank,
        scalable_seed=args.scalable_seed,
        max_chart_bytes=args.max_chart_bytes,
        max_cross_token_ratio=args.max_cross_token_ratio,
        activation_lift=getattr(args, "activation_lift", "teacher_flow"),
        teacher_flow_target_ratio=getattr(args, "teacher_flow_target_ratio", 0.05),
        activation_ridge=getattr(args, "activation_ridge", 1e-6),
        site_policy=getattr(args, "site_policy", "dual_residual"),
        dual_residual_ratio=dual_ratio,
        depth_schedule=depth_sched,
        use_2jet=getattr(args, "use_2jet", True),
        teacher_checkpoint=getattr(args, "teacher_checkpoint", None),
        pilot_pulse=getattr(args, "pilot_pulse", False),
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root.joinpath("sweep.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tune_summary = output_root / "tune.txt"
    if tune_summary.exists():
        output_root.joinpath("sweep.txt").write_text(tune_summary.read_text(encoding="utf-8"), encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for the local-only adaptive GPT-2 tune policy")
    parser.add_argument("--checkpoint-dir", required=True, help="local GPT-2 small HF checkpoint; local_files_only is enforced")
    parser.add_argument("--student-trace", required=True)
    parser.add_argument("--teacher-trace", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--mapping", default="auto", help="explicit GPT2TensorLiftMapping JSON or 'auto' to infer automatically")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--gains", default="0.01,0.05,0.1")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-dense-features", type=int, default=5_000_000)
    parser.add_argument("--scalable-rank", type=int, default=64)
    parser.add_argument("--scalable-seed", type=int, default=17)
    parser.add_argument("--max-chart-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-cross-token-ratio", type=float, default=0.5)
    parser.add_argument("--activation-lift", choices=["heuristic", "teacher_flow"], default="teacher_flow")
    parser.add_argument("--teacher-flow-target-ratio", type=float, default=0.05)
    parser.add_argument("--activation-ridge", type=float, default=1e-6)
    parser.add_argument("--site-policy", choices=["mlp_residual_only", "dual_residual"], default="dual_residual")
    parser.add_argument("--dual-residual-ratio", default="0.35", help="float or 'auto' for dynamic variance balance")
    parser.add_argument("--depth-schedule", choices=["none", "flat", "sine", "boost_deep", "dissimilarity"], default="boost_deep")
    parser.add_argument("--teacher-checkpoint", default=None, help="path to teacher checkpoint for static weight dissimilarity")
    parser.add_argument("--pilot-pulse", action="store_true", default=False, help="calculate pilot pulse sensitivity bound")
    parser.add_argument("--use-2jet", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run_sweep(args)
    except (CapabilityError, RuntimeError, ValueError, OSError, KeyError) as error:
        payload = {
            "schema_version": "faytuna-adaptive-gpt2-tune-v1",
            "status": "error",
            "reason": f"{type(error).__name__}: {error}",
            "observation_rerun": False,
            "semantic_claim": "not established",
        }
        if getattr(args, "output_root", None):
            output = Path(args.output_root)
            output.mkdir(parents=True, exist_ok=True)
            output.joinpath("sweep.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("status") in {"ok", "diagnostic"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
