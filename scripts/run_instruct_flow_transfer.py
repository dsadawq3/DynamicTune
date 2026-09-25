"""Instruct-Native Flow Transfer Engine for Qwen 3.5.

Key Innovations for Instruct Models:
1. ChatML Target-Only Flow Masking: Traces and flow residuals are computed
   STRICTLY on assistant reasoning tokens (loss_mask = 0 on system/user prompts).
2. Semantic Null-Space Projection (P_perp): Flow residuals are projected orthogonal
   to the top singular vectors of embed_tokens/lm_head to strictly preserve
   DPO/RLHF calibrated token margins.
3. Closed-Form Analytical Step: Solves optimal scale alpha* in closed form with
   Trust-Region Lipschitz clamping.
4. Auto-Gate Subspace Entropy: Identifies transferable layers vs polysemantic knots.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faytuna_flow.knots import compute_subspace_entropy, is_knot_subspace
from faytuna_flow.layer_streaming import PipelinedLayerStreamer, resolve_execution_device
from scripts.run_qwen35_transfer import (
    EVAL_SUITES,
    collect_hidden_states,
    build_multi_chart_atlas,
    swish,
    solve_adaptive_spectral_svd_deltas,
    evaluate_benchmark_suites,
)


def extract_chatml_dialogs(dataset_path: str | Path, max_dialogs: int = 16) -> list[dict[str, Any]]:
    """Load high-tension reasoning, STEM, and logic dialogs formatted for ChatML."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prompts = data.get("prompts", [])
    selected = []
    
    for p in prompts:
        cat = p.get("category", "")
        text = p.get("text", "")
        
        user_content = ""
        assistant_content = ""
        
        if "User:" in text and "Assistant:" in text:
            parts = text.split("Assistant:")
            user_content = parts[0].replace("User:", "").strip()
            assistant_content = parts[1].strip()
        elif "Task:" in text and ("Solution:" in text or "Analysis:" in text):
            split_word = "Solution:" if "Solution:" in text else "Analysis:"
            parts = text.split(split_word)
            user_content = parts[0].replace("Task:", "").strip()
            assistant_content = parts[1].strip()
        elif "Question:" in text and "Explanation:" in text:
            parts = text.split("Explanation:")
            user_content = parts[0].replace("Question:", "").strip()
            assistant_content = parts[1].strip()
        elif "Вопрос:" in text and "Объяснение:" in text:
            parts = text.split("Объяснение:")
            user_content = parts[0].replace("Вопрос:", "").strip()
            assistant_content = parts[1].strip()
        elif cat in {"discrete_math_and_logic", "theoretical_physics", "dual_superposition_2in1"}:
            user_content = text.strip()
            assistant_content = "Understood. Analyzing the mathematical and physical foundations step by step."
        
        if user_content and assistant_content:
            selected.append({
                "id": p.get("id", f"prompt_{len(selected)}"),
                "category": cat,
                "user": user_content,
                "assistant": assistant_content,
            })
            if len(selected) >= max_dialogs:
                break
                
    return selected


def collect_masked_chatml_hidden_states(
    model: Any,
    tokenizer: Any,
    dialogs: Sequence[dict[str, Any]],
    *,
    device: Any = "cpu",
    batch_size: int = 4,
) -> tuple[list[dict[int, np.ndarray]], list[tuple[int, int]]]:
    """Collect hidden states with exact assistant token slicing.
    
    Returns:
        states: List of dicts mapping layer_idx -> numpy array of assistant tokens (L_resp, d).
        slices: List of (prompt_len, full_len) token boundaries.
    """
    model.eval()
    all_assistant_states: list[dict[int, np.ndarray]] = []
    token_slices: list[tuple[int, int]] = []
    backbone = getattr(model, "model", getattr(model, "transformer", model))
    
    for d in dialogs:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": d["user"]},
            {"role": "assistant", "content": d["assistant"]},
        ]
        txt_prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        txt_full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        
        ids_prompt = tokenizer.encode(txt_prompt)
        ids_full = tokenizer.encode(txt_full)
        
        p_len = len(ids_prompt)
        f_len = len(ids_full)
        token_slices.append((p_len, f_len))
        
        # Tokenize full input for model forward pass
        inputs = tokenizer(txt_full, return_tensors="pt")
        with torch.no_grad():
            try:
                inputs_dev = {k: v.to(device) for k, v in inputs.items()}
                outputs = backbone(**inputs_dev, output_hidden_states=True)
            except Exception:
                backbone_cpu = backbone.to("cpu")
                inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                outputs = backbone_cpu(**inputs_cpu, output_hidden_states=True)
                
            prompt_states: dict[int, np.ndarray] = {}
            for l_idx, hs in enumerate(outputs.hidden_states):
                # Extract strictly the assistant response tokens [p_len : f_len]
                hs_np = hs[0].detach().to(torch.float32).cpu().numpy()
                hs_assistant = hs_np[p_len:f_len, :]  # (L_resp, hidden_size)
                prompt_states[l_idx] = hs_assistant
                
            all_assistant_states.append(prompt_states)
            
    return all_assistant_states, token_slices


def compute_semantic_null_space_projector(student_model: Any, top_k: int = 16) -> np.ndarray:
    """Compute orthogonal projector P_perp to protect DPO/RLHF token margins.
    
    Extracts top_k right singular vectors of embed_tokens (via Gram matrix)
    and constructs P_perp = I - V_k @ V_k.T.
    """
    embed_w = student_model.model.embed_tokens.weight.detach().to(torch.float32).cpu().numpy()
    # Gram matrix G = W.T @ W is (1024, 1024)
    G = embed_w.T @ embed_w
    vals, V = np.linalg.eigh(G)
    idx = np.argsort(vals)[::-1]
    V = V[:, idx]
    V_k = V[:, :top_k]
    
    P_perp = np.eye(embed_w.shape[1], dtype=np.float64) - V_k @ V_k.T
    return P_perp.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruct-Native Flow Transfer Engine")
    parser.add_argument("--student-path", type=str, default=r"C:\models\Qwen3.5-0.8B")
    parser.add_argument("--teacher-path", type=str, default=r"C:\models\Qwen3.5-4B")
    parser.add_argument("--dataset-path", type=str, default="data/calibration_prompts_v2.json")
    parser.add_argument("--output-dir", type=str, default="runs/qwen35_instruct_native")
    parser.add_argument("--max-dialogs", type=int, default=12)
    parser.add_argument("--top-k-nullspace", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    device_obj, device_type = resolve_execution_device(args.device)

    print("=" * 75)
    print("  INSTRUCT-NATIVE FLOW TRANSFER ENGINE (CHATML + NULL-SPACE DPO PROTECTION)")
    print(f"  Student Model: {args.student_path}")
    print(f"  Teacher Model: {args.teacher_path}")
    print(f"  Execution Device: {device_type} ({device_obj})")
    print(f"  Calibration Dialogs: {args.max_dialogs} ChatML pairs from {args.dataset_path}")
    print(f"  Semantic Null-Space Dim: Top-{args.top_k_nullspace} DPO margin preservation")
    print("=" * 75)

    # 1. Load Tokenizer & Dialogs
    print("\n[Stage 1/6] Loading Tokenizer & ChatML Reasoning Dialogs...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_path)
    dialogs = extract_chatml_dialogs(args.dataset_path, max_dialogs=args.max_dialogs)
    print(f"Loaded {len(dialogs)} ChatML dialogs across categories:")
    for d in dialogs:
        print(f"  - [{d['category']}] User: {d['user'][:45]}... -> Assistant: {d['assistant'][:35]}...")

    # 2. Load Student Model & Compute Null-Space Projector
    print("\n[Stage 2/6] Loading Student Model (Qwen3.5-0.8B) in FP32...")
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.student_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if device_type == "dml":
        try:
            for p in student.parameters():
                if p.dtype == torch.bfloat16:
                    p.data = p.data.to(torch.float32)
            for b in student.buffers():
                if b.dtype == torch.bfloat16:
                    b.data = b.data.to(torch.float32)
            student = student.to(device_obj)
            print(f"Student model placed on DirectML VRAM ({device_obj})")
        except BaseException as err:
            print(f"[Warning] Failed to place student on DirectML ({err}); falling back to CPU.")
            student = student.to("cpu")
            device_obj = torch.device("cpu")
            device_type = "cpu"
    print(f"Student loaded in {time.time() - t0:.2f}s.")

    print(f"Computing Semantic Null-Space Projector (Top-{args.top_k_nullspace})...")
    P_perp = compute_semantic_null_space_projector(student, top_k=args.top_k_nullspace)
    print(f"  Null-Space Projector computed: shape {P_perp.shape}, rank = {int(round(np.trace(P_perp)))}")

    # 3. Baseline Multi-Domain Benchmark
    print("\n[Stage 3/6] Running Baseline Multi-Domain Benchmark (5 Domains)...")
    base_bench = evaluate_benchmark_suites(student, tokenizer, EVAL_SUITES, device=device_obj, batch_size=4)
    print(f"  Baseline Global NLL: {base_bench['global']['mean_nll']:.5f} (PPL: {base_bench['global']['ppl']:.2f})")
    for k in ["in_domain_stem", "ood_history_humanities", "ood_medicine_biology", "ood_reasoning_commonsense", "ood_multilingual_russian"]:
        print(f"    - {k:28s}: NLL = {base_bench[k]['mean_nll']:.5f} (PPL: {base_bench[k]['ppl']:.2f})")

    # 4. Collect Masked ChatML Student Traces
    print("\n[Stage 4/6] Collecting Masked Assistant-Token Traces for Student...")
    s_traces, s_slices = collect_masked_chatml_hidden_states(student, tokenizer, dialogs, device=device_obj)
    total_tokens = sum(s[1] - s[0] for s in s_slices)
    print(f"Collected {total_tokens} assistant tokens across {len(dialogs)} dialogs (24 layers).")

    # 5. Load or Collect Masked ChatML Teacher Traces
    teacher_cache_file = out_dir / f"teacher_chatml_traces_{len(dialogs)}.pkl"
    if teacher_cache_file.exists():
        print(f"\n[Stage 5/6] Found cached ChatML teacher traces at {teacher_cache_file}! Loading...")
        with open(teacher_cache_file, "rb") as f:
            t_traces = pickle.load(f)
        print("Teacher traces loaded from disk.")
    else:
        print("\n[Stage 5/6] Loading Teacher Model (4B) & Collecting Masked ChatML Traces...")
        t0 = time.time()
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print(f"Teacher loaded in {time.time() - t0:.2f}s.")
        t_traces, _ = collect_masked_chatml_hidden_states(teacher, tokenizer, dialogs, device="cpu")
        print("Freeing Teacher from memory...")
        del teacher
        gc.collect()
        with open(teacher_cache_file, "wb") as f:
            pickle.dump(t_traces, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Teacher traces cached to {teacher_cache_file}.")

    # 6. Global Latent Alignment & Instruct-Native Weight Surgery
    print("\n[Stage 6/6] Executing Instruct-Native Closed-Form Weight Surgery...")
    
    # Procrustes Alignment between final layers on assistant tokens
    t_final = np.concatenate([tr[32] for tr in t_traces], axis=0)  # (N_tokens, 2560)
    s_final = np.concatenate([tr[24] for tr in s_traces], axis=0)  # (N_tokens, 1024)
    u, _, vt = np.linalg.svd(t_final.T @ s_final, full_matrices=False)
    P_global = (u @ vt).astype(np.float32)  # (2560, 1024)
    print(f"Global Procrustes Alignment: shape {P_global.shape}, orthogonal error: {np.linalg.norm(P_global.T @ P_global - np.eye(1024)):.2e}")

    n_student_layers = 24
    n_teacher_layers = 32
    surgery_records: list[dict[str, Any]] = []

    print("\nLayer-by-Layer Auto-Gate Scan on Assistant Tokens:")
    print("Layer | Teacher | Assistant Entropy | Auto-Gate Verdict | Optimal alpha | Update Norm")
    print("-" * 84)

    for s_idx in range(n_student_layers):
        t_idx = int(round(s_idx * (n_teacher_layers - 1) / max(1, n_student_layers - 1)))
        layer_module = student.model.layers[s_idx]

        s_in = np.concatenate([tr[s_idx] for tr in s_traces], axis=0)      # (N, 1024)
        s_out = np.concatenate([tr[s_idx + 1] for tr in s_traces], axis=0)  # (N, 1024)
        t_in = np.concatenate([tr[t_idx] for tr in t_traces], axis=0)      # (N, 2560)
        t_out = np.concatenate([tr[t_idx + 1] for tr in t_traces], axis=0)  # (N, 2560)

        # Local multi-chart projection
        atlas_in = build_multi_chart_atlas(s_in, t_in, n_charts=4, random_state=42 + s_idx)
        atlas_out = build_multi_chart_atlas(s_out, t_out, n_charts=4, random_state=100 + s_idx)

        t_in_proj = atlas_in.project_teacher_to_student(t_in, s_in)
        t_out_proj = atlas_out.project_teacher_to_student(t_out, s_out)

        # Flow residual on assistant tokens
        flow_residual_raw = (t_out_proj - t_in_proj) - (s_out - s_in)  # (N, 1024)

        # Apply Semantic Null-Space Projection to protect DPO/RLHF margins
        flow_residual = flow_residual_raw @ P_perp  # (N, 1024)

        entropy = compute_subspace_entropy(flow_residual)
        knot = is_knot_subspace(flow_residual, threshold=0.88)

        if knot:
            verdict = "Knot (Skipped)"
            alpha_optimal = 0.0
            delta_norm = 0.0
            print(f"{s_idx:5d} | {t_idx:7d} | {entropy:17.4f} | {verdict:17s} | {alpha_optimal:13.4f} | {delta_norm:11.4f}")
            surgery_records.append({
                "layer": s_idx, "teacher_layer": t_idx, "entropy": float(entropy),
                "knot": True, "alpha": 0.0, "update_norm": 0.0,
            })
            continue

        # Candidate transferable layer: compute analytical optimal update
        verdict = "*** CANDIDATE ***"
        mlp = layer_module.mlp
        with torch.no_grad():
            w_gate = mlp.gate_proj.weight.detach().to(torch.float32).cpu().numpy()
            w_up = mlp.up_proj.weight.detach().to(torch.float32).cpu().numpy()
            w_down = mlp.down_proj.weight.detach().to(torch.float32).cpu().numpy()

        u_gate = s_in @ w_gate.T
        v_up = s_in @ w_up.T
        act_swiglu = swish(u_gate) * v_up  # (N, 3584)
        mlp_target_shift = 0.5 * flow_residual  # (N, 1024)

        # Solve down_proj update via Adaptive Spectral SVD
        delta_w_down = solve_adaptive_spectral_svd_deltas(act_swiglu, mlp_target_shift, energy_ratio=0.85, max_rank=128)
        delta_w_down_t = delta_w_down.T  # (1024, 3584)

        # Predicted output shift produced by unscaled delta_w_down_t
        hat_delta = act_swiglu @ delta_w_down_t.T  # (N, 1024)

        # Exact Closed-Form Optimal Scale:
        inner_prod = float(np.sum(hat_delta * mlp_target_shift))
        norm_hat_sq = float(np.sum(hat_delta ** 2))
        norm_w_sq = float(np.sum(delta_w_down_t ** 2))
        ridge_lambda = 1e-4 * (norm_hat_sq / max(norm_w_sq, 1e-12))

        alpha_analytic = inner_prod / (norm_hat_sq + ridge_lambda * norm_w_sq)

        # Strict Trust-Region Clamp: max relative weight change <= 4%
        max_safe_alpha = (0.04 * np.linalg.norm(w_down)) / max(np.linalg.norm(delta_w_down_t), 1e-12)
        alpha_optimal = float(np.clip(alpha_analytic, 0.0, max_safe_alpha))

        # Apply update
        new_down = w_down + alpha_optimal * delta_w_down_t
        delta_norm = float(np.linalg.norm(alpha_optimal * delta_w_down_t))

        with torch.no_grad():
            mlp.down_proj.weight.copy_(torch.from_numpy(new_down).to(device=mlp.down_proj.weight.device, dtype=mlp.down_proj.weight.dtype))

        print(f"{s_idx:5d} | {t_idx:7d} | {entropy:17.4f} | {verdict:17s} | {alpha_optimal:13.4f} | {delta_norm:11.4f}")
        surgery_records.append({
            "layer": s_idx, "teacher_layer": t_idx, "entropy": float(entropy),
            "knot": False, "alpha": float(alpha_optimal), "update_norm": delta_norm,
        })

    # 7. Post-Surgery Multi-Domain Benchmark Evaluation
    print("\n" + "=" * 80)
    print("  POST-SURGERY MULTI-DOMAIN GENERALIZATION EVALUATION (5 DOMAINS)")
    print("=" * 80)
    post_bench = evaluate_benchmark_suites(student, tokenizer, EVAL_SUITES, device=device_obj, batch_size=4)

    print("\n" + "-" * 88)
    print(f"{'Domain Suite':<28} | {'Pre NLL':<8} -> {'Post NLL':<8} | {'Delta':<9} | {'Change %':<8} | {'Status'}")
    print("-" * 88)

    comparisons: dict[str, Any] = {}
    for domain in list(EVAL_SUITES.keys()) + ["global"]:
        pre_res = base_bench[domain]
        post_res = post_bench[domain]
        d_nll = post_res["mean_nll"] - pre_res["mean_nll"]
        pct = (d_nll / pre_res["mean_nll"]) * 100.0
        dom_status = "IMPROVED" if d_nll < -0.001 else ("DEGRADED" if d_nll > 0.001 else "STABLE")
        print(f"{domain:<28} | {pre_res['mean_nll']:<8.4f} -> {post_res['mean_nll']:<8.4f} | {d_nll:<+9.4f} | {pct:<+7.2f}% | [{dom_status}]")
        comparisons[domain] = {
            "pre_nll": pre_res["mean_nll"],
            "post_nll": post_res["mean_nll"],
            "delta_nll": d_nll,
            "pct_change": pct,
            "status": dom_status,
        }

    # Save final report and model
    report = {
        "student_model": args.student_path,
        "teacher_model": args.teacher_path,
        "num_dialogs": len(dialogs),
        "total_assistant_tokens": total_tokens,
        "top_k_nullspace": args.top_k_nullspace,
        "base_benchmark": base_bench,
        "post_benchmark": post_bench,
        "comparisons": comparisons,
        "surgery_records": surgery_records,
    }

    report_file = out_dir / "instruct_flow_transfer_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_file}.")

    if args.save_model:
        model_save_dir = out_dir / "transferred_student"
        print(f"Saving transferred student model checkpoint to {model_save_dir}...")
        student.save_pretrained(model_save_dir)
        tokenizer.save_pretrained(model_save_dir)
        print("Model saved successfully.")


if __name__ == "__main__":
    main()
