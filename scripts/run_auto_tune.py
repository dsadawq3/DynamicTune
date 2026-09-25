"""Universal autonomous flow transfer and weight surgery runner.

Model-agnostic: automatically inspects student and teacher architectures,
detects layer counts, hidden dimensions, and tensor layouts, automatically generates
block correspondence without hardcoded mappings, computes static weight dissimilarity
and dynamic attention/MLP ratios, evaluates NLL/PPL before and after surgery, and
prints a detailed empirical report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faytuna_flow.adaptive_transfer import run_universal_flow_tune
from faytuna_flow.model_families import inspect_transformer_architecture


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)
    row_lines = [" | ".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths)) for row in rows]
    return f"{header_line}\n{sep_line}\n" + "\n".join(row_lines)


def run_autonomous_tune(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_dir = Path(args.checkpoint_dir)
    teacher_cand = Path(args.teacher_checkpoint) if args.teacher_checkpoint else checkpoint_dir.parent / "gpt2-xl"
    
    print("\n" + "=" * 80)
    print(" [*] FAYTUNA AUTONOMOUS MODEL-AGNOSTIC FLOW SURGERY ENGINE")
    print("=" * 80)
    
    # 1. Inspect Student Architecture
    print(f"\n[1/4] Inspecting Student Architecture: {checkpoint_dir}")
    student_arch = inspect_transformer_architecture(checkpoint_dir)
    print(f"      • Model Family:        {student_arch.model_type.upper()}")
    print(f"      • Number of Layers:    {student_arch.layers}")
    print(f"      • Hidden Dimension:    {student_arch.hidden_size}")
    print(f"      • Intermediate (MLP):  {student_arch.intermediate_size}")
    print(f"      • Attention Heads:     {student_arch.attention_heads}")
    print(f"      • Tensor Conventions:  Prefix='{student_arch.layer_prefix}', Attn='{student_arch.attn_proj_suffix}', Conv1D={student_arch.is_conv1d}")
    
    # 2. Inspect Teacher Architecture
    teacher_arch = None
    if teacher_cand.exists():
        print(f"\n[2/4] Inspecting Teacher Architecture: {teacher_cand}")
        teacher_arch = inspect_transformer_architecture(teacher_cand)
        print(f"      • Model Family:        {teacher_arch.model_type.upper()}")
        print(f"      • Number of Layers:    {teacher_arch.layers}")
        print(f"      • Hidden Dimension:    {teacher_arch.hidden_size}")
        print(f"      • Intermediate (MLP):  {teacher_arch.intermediate_size}")
        print(f"      • Attention Heads:     {teacher_arch.attention_heads}")
        ratio_layers = teacher_arch.layers / max(1, student_arch.layers)
        ratio_dim = teacher_arch.hidden_size / max(1, student_arch.hidden_size)
        print(f"      • Teacher/Student Scale: Depth {ratio_layers:.1f}x ({teacher_arch.layers} vs {student_arch.layers}), Width {ratio_dim:.1f}x ({teacher_arch.hidden_size} vs {student_arch.hidden_size})")
    else:
        print(f"\n[2/4] Teacher checkpoint not found locally at {teacher_cand}; using dynamic synthetic teacher scaling.")

    # 3. Parse parameters
    gains = tuple(float(v.strip()) for v in args.gains.split(",") if v.strip())
    dual_ratio = args.dual_residual_ratio
    if dual_ratio != "auto":
        try:
            dual_ratio = float(dual_ratio)
        except ValueError:
            pass

    depth_sched = None if args.depth_schedule in {None, "none", "flat"} else args.depth_schedule

    print("\n[3/4] Launching End-to-End Surgical State Machine:")
    print(f"      • Gains:                 {gains}")
    print(f"      • Mapping Mode:          {args.mapping}")
    print(f"      • Depth Schedule:        {depth_sched}")
    print(f"      • Dual Residual Ratio:   {dual_ratio}")
    print(f"      • Pilot Pulse Bound:     {args.pilot_pulse}")
    print(f"      • 2-Jet Flow Curvature:  {args.use_2jet}")
    print(f"      • Piecewise Knots:       {getattr(args, 'piecewise_knots', False)}")
    
    t0 = time.perf_counter()
    payload = run_universal_flow_tune(
        checkpoint_dir,
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
        activation_lift=args.activation_lift,
        teacher_flow_target_ratio=args.teacher_flow_target_ratio,
        activation_ridge=args.activation_ridge,
        site_policy=args.site_policy,
        dual_residual_ratio=dual_ratio,
        depth_schedule=depth_sched,
        use_2jet=args.use_2jet,
        teacher_checkpoint=teacher_cand if teacher_cand.exists() else None,
        pilot_pulse=args.pilot_pulse,
        piecewise_knots=getattr(args, "piecewise_knots", False),
    )
    elapsed = time.perf_counter() - t0
    
    # 4. Print Empirical Summary Tables
    print(f"\n[4/4] Execution Completed in {elapsed:.2f}s! Status: {payload.get('status')}")
    print("=" * 80)
    
    # Dissimilarity Table
    dissim = payload.get("weight_dissimilarity")
    if dissim and "distances" in dissim:
        print("\n[+] 1. STATIC WEIGHT DISSIMILARITY & DYNAMIC LAYER GAIN SCHEDULE:")
        headers = ["Block (Student)", "Matched Layer (Teacher)", "Frobenius Distance", "Auto-Gain Weight", "Status"]
        rows = []
        dists = dissim["distances"]
        sched = dissim.get("schedule", [1.0] * len(dists))
        lmap = dissim.get("teacher_layer_map", {})
        mean_d = dissim.get("mean_distance", float(np.mean(dists)))
        for b, (d, g) in enumerate(zip(dists, sched)):
            t_idx = lmap.get(b, lmap.get(str(b), "-"))
            status = "[BOOST]" if g > 1.05 else ("[TAPER]" if g < 0.95 else "[NOMINAL]")
            rows.append([f"Block {b}", f"Layer {t_idx}", f"{d:.4f}", f"{g:.4f}", status])
        print(_format_table(headers, rows))
        print(f"Mean Divergence Distance: {mean_d:.4f} (contrast={dissim.get('contrast')})\n")

    # Pilot Pulse Bound Table
    pulse = payload.get("pilot_pulse_bound")
    if pulse:
        print("[+] 2. PILOT FLOW SENSITIVITY & SAFE LOGIT BOUNDS:")
        print(f"      • Max Relative Perturbation:  {pulse.get('relative_perturbation', 0.0):.6f}")
        print(f"      • Safe Gain Upper Bound:      {pulse.get('safe_gain_bound', 0.0):.4f}")
        print(f"      • Target Logit Shift eps:     {pulse.get('target_logit_shift', 0.01)}")
        print(f"      • Recommended Sweep Gains:    {pulse.get('recommended_sweep_gains', [])}\n")

    # Candidate Performance Table
    candidates = payload.get("candidates", [])
    if candidates:
        print("[+] 3. EMPIRICAL NLL / PPL PERFORMANCE SWEEP:")
        headers = ["Gain alpha", "Val dNLL", "Val PPL Gain", "Holdout dNLL", "Holdout PPL Gain", "Status"]
        rows = []
        for cand in candidates:
            g = cand.get("gain")
            val_sum = cand.get("text_ab", {}).get("validation", {}).get("summary", {})
            hold_sum = cand.get("text_ab", {}).get("holdout", {}).get("summary", {})
            v_dnll = val_sum.get("candidate_minus_baseline_mean_target_nll")
            h_dnll = hold_sum.get("candidate_minus_baseline_mean_target_nll")
            
            v_str = f"{v_dnll:+.4f}" if v_dnll is not None else "N/A"
            h_str = f"{h_dnll:+.4f}" if h_dnll is not None else "N/A"
            
            v_ppl = f"{(math.exp(-v_dnll) - 1.0) * 100:+.1f}%" if v_dnll is not None else "N/A"
            h_ppl = f"{(math.exp(-h_dnll) - 1.0) * 100:+.1f}%" if h_dnll is not None else "N/A"
            
            stat = "[ACCEPTED]" if (h_dnll is not None and h_dnll <= 0) else "[DEGRADED]"
            rows.append([str(g), v_str, v_ppl, h_str, h_ppl, stat])
        print(_format_table(headers, rows))

    # 4. Piecewise Knots & Detour Routing Summary
    pk_info = payload.get("piecewise_knots")
    if pk_info and pk_info.get("enabled"):
        print("\n[+] 4. PIECEWISE MLP KNOT DETECTION & ATTENTION DETOUR ROUTING:")
        print(f"      • Subspace Pieces (K):        {pk_info.get('num_pieces')}")
        print(f"      • RelRes Tolerance:           {pk_info.get('tolerance')}")
        print(f"      • SVD Rank Ratio:             {pk_info.get('svd_rank_ratio')}")
        print(f"      • Cosine Threshold:           {pk_info.get('cosine_threshold')}")
        print(f"      • Detour Bypass Factor (eta): {pk_info.get('eta')}")
        for cand in candidates:
            plan = cand.get("plan")
            if isinstance(plan, dict) and "metadata" in plan:
                pks = plan["metadata"].get("piecewise_knots_summary")
                if pks:
                    print(f"      • Gain {cand.get('gain')}: {pks.get('total_pieces', 0)} pieces, "
                          f"{pks.get('total_knots_skipped', 0)} knots skipped, "
                          f"detours: {len(pks.get('detours_applied', []))} applied")

    selected = payload.get("selected_gain")
    print("\n" + "=" * 80)
    if selected is not None:
        print(f" [SELECTED] OPTIMAL CANDIDATE SELECTED: Gain alpha = {selected}")
    else:
        print(" [WARN] No candidate passed strict zero-degradation criteria on holdout.")
    print("=" * 80 + "\n")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root.joinpath("autonomous_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal autonomous model-agnostic flow surgery runner")
    parser.add_argument("--checkpoint-dir", required=True, help="path to local student HF model checkpoint")
    parser.add_argument("--teacher-checkpoint", default=None, help="path to teacher checkpoint (auto-located if omitted)")
    parser.add_argument("--student-trace", required=True, help="student training observation trace (.npz)")
    parser.add_argument("--teacher-trace", required=True, help="teacher training observation trace (.npz)")
    parser.add_argument("--alignment", required=True, help="alignment file (.json)")
    parser.add_argument("--mapping", default="auto", help="tensor lift mapping JSON or 'auto' for dynamic layer matching")
    parser.add_argument("--output-root", default="runs/auto_tune_run", help="output directory for candidate checkpoints and reports")
    parser.add_argument("--sequence-length", type=int, default=16)
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
    parser.add_argument("--dual-residual-ratio", default="auto", help="energy split: float or 'auto' for dynamic variance balance")
    parser.add_argument("--depth-schedule", choices=["none", "flat", "sine", "boost_deep", "dissimilarity"], default="dissimilarity")
    parser.add_argument("--pilot-pulse", action="store_true", default=True, help="calculate pilot pulse sensitivity bound")
    parser.add_argument("--use-2jet", action="store_true", default=True, help="use 2-jet midpoint flow curvature correction")
    parser.add_argument("--piecewise-knots", action="store_true", default=False, help="enable piecewise MLP knot detection, multi-trial retry, and attention detour routing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_autonomous_tune(args)
    return 0 if payload.get("status") in {"ok", "diagnostic"} else 1


if __name__ == "__main__":
    sys.exit(main())
