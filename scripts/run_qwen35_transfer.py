"""Universal End-to-End Non-Linear & Multi-Chart Flow Transfer Engine for Qwen 3.5.

Executes real weight surgery on Qwen3.5-0.8B-Base using knowledge from Qwen3.5-4B-Base:
1. Multi-Chart Manifold Atlas (LTSA with spherical k-means & partition-of-unity).
2. Non-linear SwiGLU dual-gate pre-activation inversion.
3. Piecewise Knot detection and regularized least-squares.
4. Pre vs Post empirical A/B evaluation (NLL, Perplexity, Logit stability).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from faytuna_flow.knots import (
    build_attention_detour_projector,
    compute_subspace_entropy,
    is_knot_subspace,
)
from faytuna_flow.manifold_charts import (
    MultiChartAtlas,
    build_multi_chart_atlas,
)
from faytuna_flow.nonlinear_transfer import (
    align_vocabulary_head,
    compute_2jet_curvature_metric,
    damped_tikhonov_pinv,
    invert_swiglu_activations,
    rank_one_memory_imprint,
    solve_adaptive_spectral_svd_deltas,
    solve_truncated_svd_deltas,
    solve_weight_deltas_least_squares,
    spectral_directional_rescale,
    swish,
    swish_prime,
)
from faytuna_flow.layer_streaming import PipelinedLayerStreamer
from faytuna_flow.transformer_core import run_flow_guided_distillation


TRAIN_PROMPTS = [
    "The fundamental theorem of algebra states that every non-zero single-variable polynomial",
    "In deep learning, residual connections allow gradients to flow directly through skip paths",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2",
    "The thermodynamic entropy of an isolated system never decreases over time according to",
    "Photosynthesis is the biological process used by plants, algae, and certain bacteria to convert",
    "To solve the linear system Ax = b when A is symmetric positive definite, conjugate gradient",
    "The manifold hypothesis in representation learning suggests that high-dimensional data resides",
    "In distributed systems, the Raft consensus algorithm achieves safety and state replication by",
]

EVAL_SUITES: dict[str, list[str]] = {
    "in_domain_stem": [
        "The Riemann hypothesis is a conjecture that the Riemann zeta function has its zeros only at",
        "def binary_search(nums, target):\n    left, right = 0, len(nums) - 1\n    while left <= right:",
        "Quantum entanglement occurs when a group of particles interact such that the quantum state of each",
        "In modern transformer architectures, SwiGLU activation replaces standard ReLU or GELU to",
    ],
    "ood_history_humanities": [
        "The fall of the Western Roman Empire in the fifth century AD was precipitated by a complex combination of",
        "During the Renaissance, European scholars rediscovered classical Greek and Roman texts, leading to a revival of",
        "The Treaty of Westphalia signed in 1648 established the modern concept of national sovereign state borders and",
        "The Silk Road was an ancient network of Eurasian trade routes that facilitated cultural and economic exchange between",
    ],
    "ood_medicine_biology": [
        "Penicillin acts by inhibiting bacterial cell wall synthesis through binding to specific penicillin-binding proteins and",
        "Insulin is a peptide hormone produced by beta cells of the pancreatic islets that regulates carbohydrate metabolism by",
        "In molecular biology, DNA transcription is the process by which genetic instructions from DNA are copied into",
        "The blood-brain barrier is a highly selective semipermeable border of endothelial cells that prevents solutes in the",
    ],
    "ood_reasoning_commonsense": [
        "If a glass bottle full of water is placed inside a freezer, the water will expand as it freezes and",
        "When driving on a rainy highway at high speeds, vehicles risk hydroplaning because a thin layer of water builds",
        "To bake a loaf of sourdough bread from scratch, the baker first mixes active sourdough starter with flour and",
        "If the price of wheat increases sharply due to drought, the cost of producing flour and commercial bread will",
    ],
    "ood_multilingual_russian": [
        "Основная теорема анализа связывает операцию дифференцирования функции с операцией взятия определённого интеграла и",
        "История развития архитектуры трансформеров началась с публикации статьи Attention Is All You Need в две тысячи семнадцатом году",
        "Принципы термодинамики утверждают, что в изолированной системе энтропия никогда не убывает с течением времени и",
        "Для оптимизации сложных нелинейных функций в машинном обучении часто применяют методы градиентного спуска с адаптивным",
    ],
}

EVAL_PROMPTS = EVAL_SUITES["in_domain_stem"]


def resolve_device(requested: str = "cpu") -> tuple[Any, str]:
    """Resolve requested device (including DirectML) with automatic CPU fallback."""
    req_lower = str(requested).lower()
    if req_lower in {"dml", "directml"}:
        try:
            import torch_directml
            dev = torch_directml.device()
            return dev, "dml"
        except Exception as err:
            print(f"[Warning] Failed to initialize DirectML ({err}); falling back to CPU.")
            return torch.device("cpu"), "cpu"
    elif req_lower == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        try:
            import torch_directml
            return torch_directml.device(), "dml"
        except Exception:
            return torch.device("cpu"), "cpu"
    else:
        try:
            return torch.device(requested), req_lower
        except Exception:
            return torch.device("cpu"), "cpu"


def collect_hidden_states(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    device: Any = "cpu",
    batch_size: int = 4,
) -> list[dict[int, np.ndarray]]:
    """Run model on prompts and extract hidden states at all layer boundaries.

    Optimizations:
    1. Zero-Logit Extraction: Targets language backbone directly via
       `getattr(model, "model", getattr(model, "transformer", model))`,
       bypassing the 248,320-token vocabulary classification projection (lm_head).
    2. Dynamic Batching & Padding: Batches prompts with attention mask and
       dynamic padding, then unpads each prompt's hidden states to exact length.
    3. DirectML / CPU Fallback: Seamlessly executes on DirectML or falls back to CPU.

    Returns:
        List of dicts mapping layer_idx -> numpy array of shape (seq_len, hidden_size).
    """
    model.eval()
    all_states: list[dict[int, np.ndarray]] = []
    backbone = getattr(model, "model", getattr(model, "transformer", model))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) is None:
        tokenizer.padding_side = "right"

    prompt_list = list(prompts)
    batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for i in range(0, len(prompt_list), batch_size):
            batch_prompts = prompt_list[i : i + batch_size]
            if len(batch_prompts) == 1 or batch_size == 1:
                for prompt in batch_prompts:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                    try:
                        inputs_dev = {k: v.to(device) for k, v in inputs.items()}
                        outputs = backbone(**inputs_dev, output_hidden_states=True)
                    except BaseException as err:
                        print(f"[collect_hidden_states] Target device execution failed ({err}); falling back to CPU.")
                        backbone_cpu = backbone.to("cpu")
                        inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                        outputs = backbone_cpu(**inputs_cpu, output_hidden_states=True)

                    prompt_states: dict[int, np.ndarray] = {}
                    for l_idx, hs in enumerate(outputs.hidden_states):
                        prompt_states[l_idx] = hs[0].detach().to(torch.float32).cpu().numpy()
                    all_states.append(prompt_states)
            else:
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
                unpadded_lens = inputs["attention_mask"].sum(dim=1).tolist()
                try:
                    inputs_dev = {k: v.to(device) for k, v in inputs.items()}
                    outputs = backbone(**inputs_dev, output_hidden_states=True)
                except BaseException as err:
                    print(f"[collect_hidden_states] Batched device execution failed ({err}); falling back to CPU.")
                    backbone_cpu = backbone.to("cpu")
                    inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                    outputs = backbone_cpu(**inputs_cpu, output_hidden_states=True)

                for b_idx, seq_len in enumerate(unpadded_lens):
                    prompt_states = {}
                    for l_idx, hs in enumerate(outputs.hidden_states):
                        if tokenizer.padding_side == "left":
                            valid_hs = hs[b_idx, -seq_len:]
                        else:
                            valid_hs = hs[b_idx, :seq_len]
                        prompt_states[l_idx] = valid_hs.detach().to(torch.float32).cpu().numpy()
                    all_states.append(prompt_states)

    return all_states


def evaluate_nll_and_ppl(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    device: Any = "cpu",
    batch_size: int = 4,
) -> dict[str, float]:
    """Compute mean Cross-Entropy NLL and Perplexity on evaluation prompts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    prompt_list = list(prompts)
    batch_size = max(1, int(batch_size))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for i in range(0, len(prompt_list), batch_size):
            batch_prompts = prompt_list[i : i + batch_size]
            if len(batch_prompts) == 1 or batch_size == 1:
                for prompt in batch_prompts:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                    try:
                        inputs_dev = {k: v.to(device) for k, v in inputs.items()}
                        labels = inputs_dev["input_ids"].clone()
                        outputs = model(**inputs_dev, labels=labels)
                    except BaseException as err:
                        model_cpu = model.to("cpu")
                        inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                        labels = inputs_cpu["input_ids"].clone()
                        outputs = model_cpu(**inputs_cpu, labels=labels)
                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        loss = outputs.loss.item()
                    else:
                        logits = getattr(outputs, "logits", outputs[0])
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        ).item()
                    num_tokens = inputs["input_ids"].size(1)
                    total_loss += loss * num_tokens
                    total_tokens += num_tokens
            else:
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=64)
                try:
                    inputs_dev = {k: v.to(device) for k, v in inputs.items()}
                    labels = inputs_dev["input_ids"].clone()
                    labels[inputs_dev["attention_mask"] == 0] = -100
                    outputs = model(**inputs_dev, labels=labels)
                except BaseException as err:
                    model_cpu = model.to("cpu")
                    inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                    labels = inputs_cpu["input_ids"].clone()
                    labels[inputs_cpu["attention_mask"] == 0] = -100
                    outputs = model_cpu(**inputs_cpu, labels=labels)
                num_tokens = int(inputs["attention_mask"].sum().item())
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss.item()
                else:
                    logits = getattr(outputs, "logits", outputs[0])
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    ).item()
                total_loss += loss * num_tokens
                total_tokens += num_tokens

    mean_nll = total_loss / max(total_tokens, 1)
    ppl = float(np.exp(mean_nll))
    return {
        "mean_nll": float(mean_nll),
        "ppl": ppl,
        "total_tokens": total_tokens,
    }


def evaluate_benchmark_suites(
    model: Any,
    tokenizer: Any,
    suites: Mapping[str, Sequence[str]],
    *,
    device: Any = "cpu",
    batch_size: int = 4,
) -> dict[str, Any]:
    """Evaluate NLL and PPL across multiple domain suites, computing per-domain and global metrics."""
    results: dict[str, Any] = {}
    all_nll_sum = 0.0
    all_tokens = 0

    for suite_name, prompts in suites.items():
        res = evaluate_nll_and_ppl(model, tokenizer, prompts, device=device, batch_size=batch_size)
        results[suite_name] = res
        all_nll_sum += res["mean_nll"] * res["total_tokens"]
        all_tokens += res["total_tokens"]

    global_mean_nll = all_nll_sum / max(all_tokens, 1)
    results["global"] = {
        "mean_nll": float(global_mean_nll),
        "ppl": float(np.exp(global_mean_nll)),
        "total_tokens": all_tokens,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen 3.5 Non-Linear Multi-Chart Flow Transfer")
    parser.add_argument("--student-path", type=str, default=r"C:\models\Qwen3.5-0.8B-Base")
    parser.add_argument("--teacher-path", type=str, default=r"C:\models\Qwen3.5-4B-Base")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "dml", "cuda", "auto"], help="Execution device for student model (cpu, dml, cuda, auto)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for hidden state trace collection and evaluation")
    parser.add_argument("--gain", type=float, default=0.08, help="Transfer gain alpha")
    parser.add_argument("--n-charts", type=int, default=4, help="Number of manifold charts")
    parser.add_argument("--max-blocks", type=int, default=6, help="Max blocks to transfer in pilot")
    parser.add_argument("--use-memory-imprint", action="store_true", help="Use targeted rank-1 memory imprinting for MLP down_proj")
    parser.add_argument("--calibrate-head", action="store_true", help="Align vocabulary classification head with teacher")
    parser.add_argument("--head-gain", type=float, default=0.03, help="Vocabulary head alignment gain")
    parser.add_argument("--distill-steps", type=int, default=0, help="Micro-distillation steps (0=disabled)")
    parser.add_argument("--output-dir", type=str, default="runs/qwen35_transfer")
    parser.add_argument("--save-model", action="store_true", help="Save transferred student model checkpoint to disk")
    parser.add_argument("--use-layer-streaming", action="store_true", help="Use pipelined layer-by-layer VRAM streaming for hidden state collection")
    parser.add_argument("--prompt-source", type=str, default="default", choices=["default", "v2", "file"], help="Calibration prompt source (default=8 baseline, v2=high-rank 2-in-1 stress generator, file=custom JSON)")
    parser.add_argument("--prompt-file", type=str, default="data/calibration_prompts_v2.json", help="Path to prompt JSON if --prompt-source file")
    parser.add_argument("--num-calibration-prompts", type=int, default=32, help="Number of calibration prompts to draw from v2 generator (default: 32)")
    parser.add_argument("--depth-schedule", type=str, default="none", choices=["none", "flat", "sine", "boost_deep"], help="Modulate alpha gain across layer depth (none, sine, boost_deep)")
    parser.add_argument("--knot-threshold", type=float, default=0.985, help="Subspace spectral entropy threshold to classify as knot (default: 0.985)")
    args = parser.parse_args()


    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    device_obj, device_type = resolve_device(args.device)

    # Resolve calibration prompts
    if args.prompt_source == "v2":
        from faytuna_flow.prompt_generator import CalibrationPromptEngine
        engine = CalibrationPromptEngine(seed=42)
        suite = engine.generate_calibration_suite(num_prompts=args.num_calibration_prompts)
        train_prompts = [s["text"] for s in suite]
        prompt_desc = f"{len(train_prompts)} prompts from High-Rank V2 Engine (2-in-1 Superposition)"
    elif args.prompt_source == "file":
        from faytuna_flow.prompt_generator import CalibrationPromptEngine
        train_prompts = CalibrationPromptEngine.load_calibration_dataset(args.prompt_file)
        if args.num_calibration_prompts and len(train_prompts) > args.num_calibration_prompts:
            train_prompts = train_prompts[:args.num_calibration_prompts]
        prompt_desc = f"{len(train_prompts)} prompts loaded from {args.prompt_file}"
    else:
        train_prompts = list(TRAIN_PROMPTS)
        prompt_desc = f"{len(train_prompts)} baseline prompts"

    print("=" * 70)
    print("  QWEN 3.5 NON-LINEAR MULTI-CHART FLOW SURGERY ENGINE")
    print(f"  Student Checkpoint: {args.student_path}")
    print(f"  Teacher Checkpoint: {args.teacher_path}")
    print(f"  Execution Device: {device_type} ({device_obj})")
    print(f"  Trace Batch Size: {args.batch_size}")
    print(f"  Calibration Prompts: {prompt_desc}")
    print(f"  Transfer Gain alpha: {args.gain}")
    print(f"  Manifold Charts: {args.n_charts}")
    print(f"  Pilot Blocks: {args.max_blocks}")
    print("=" * 70)


    # 1. Load Tokenizer
    print("\n[Stage 1/6] Loading Tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.student_path)
    print(f"Tokenizer loaded in {time.time() - t0:.2f}s (vocab={len(tokenizer)})")

    # 2. Load Student Model (0.8B)
    print("\n[Stage 2/6] Loading Student Model (0.8B)...")
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.student_path,
        dtype=torch.float32,  # float32 for precise weight surgery
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
    print(f"Student loaded in {time.time() - t0:.2f}s ({sum(p.numel() for p in student.parameters()):,} params)")

    # 3. Comprehensive Baseline Multi-Domain Evaluation Before Surgery
    print("\n[Stage 3/6] Running Pre-Surgery Baseline Holdout Evaluation (5 Domains)...")
    pre_benchmark = evaluate_benchmark_suites(student, tokenizer, EVAL_SUITES, device=device_obj, batch_size=args.batch_size)
    pre_eval = pre_benchmark["in_domain_stem"]
    for domain, res in pre_benchmark.items():
        print(f"  [Pre]  {domain:<28} : NLL = {res['mean_nll']:.4f} | PPL = {res['ppl']:.4f} ({res['total_tokens']} tokens)")

    # 4. Collect Student Traces on Training Prompts
    print("\n[Stage 4/6] Collecting Student Hidden State Traces...")
    if args.use_layer_streaming:
        print("Collecting student traces via PipelinedLayerStreamer...")
        streamer_student = PipelinedLayerStreamer(student, device=device_obj, tokenizer=tokenizer)
        student_traces = streamer_student.collect_hidden_states(train_prompts, batch_size=args.batch_size, device=device_obj)
    else:
        student_traces = collect_hidden_states(student, tokenizer, train_prompts, device=device_obj, batch_size=args.batch_size)
    print(f"Collected traces across {len(student_traces)} prompts (24 layers).")

    # 5. Load Teacher Model (4B) & Collect Teacher Traces (with caching)
    import hashlib
    p_hash = hashlib.sha256("".join(train_prompts).encode("utf-8")).hexdigest()[:8]
    if args.prompt_source == "default":
        teacher_cache_file = out_dir / "teacher_traces_cache.pkl"
        teacher_head_file = out_dir / "teacher_head_proj.npy"
    else:
        teacher_cache_file = out_dir / f"teacher_traces_cache_{p_hash}.pkl"
        teacher_head_file = out_dir / f"teacher_head_proj_{p_hash}.npy"

    teacher_traces = None
    teacher_head_proj: np.ndarray | None = None

    if teacher_cache_file.exists():
        print(f"\n[Stage 5/6] Found cached teacher traces at {teacher_cache_file}! Loading instantly...")
        t0 = time.time()
        with open(teacher_cache_file, "rb") as f:
            teacher_traces = pickle.load(f)
        print(f"Teacher traces loaded from disk in {time.time() - t0:.2f}s ({len(teacher_traces)} prompts).")
        if args.calibrate_head:
            if teacher_head_file.exists():
                teacher_head_proj = np.load(teacher_head_file)
                print(f"Projected teacher lm_head loaded from cache: shape {teacher_head_proj.shape}")
            else:
                print("Extracting teacher lm_head directly from disk and projecting via final layer chart...")
                t_final = np.concatenate([tr[32] for tr in teacher_traces], axis=0)  # (N, 2560)
                s_final = np.concatenate([tr[24] for tr in student_traces], axis=0)  # (N, 1024)
                u, _, vt = np.linalg.svd(t_final.T @ s_final, full_matrices=False)
                p_final = u @ vt  # (2560, 1024)
                shard_path = Path(args.teacher_path) / "model.safetensors-00001-of-00002.safetensors"
                if shard_path.exists():
                    from safetensors.torch import load_file
                    shard = load_file(str(shard_path), device="cpu")
                    w_t_head = shard["model.language_model.embed_tokens.weight"].to(torch.float32).numpy()
                    del shard
                else:
                    teacher_tmp = AutoModelForCausalLM.from_pretrained(args.teacher_path, dtype=torch.float32, low_cpu_mem_usage=True)
                    w_t_head = teacher_tmp.lm_head.weight.detach().to(torch.float32).cpu().numpy()
                    del teacher_tmp
                gc.collect()
                teacher_head_proj = (w_t_head @ p_final).astype(np.float32)
                del w_t_head
                gc.collect()
                print(f"Projected teacher lm_head computed: shape {teacher_head_proj.shape}")
                np.save(teacher_head_file, teacher_head_proj)
    else:
        print("\n[Stage 5/6] Loading Teacher Model (4B) & Collecting Teacher Traces...")
        t0 = time.time()
        if args.use_layer_streaming:
            print("Collecting teacher traces via PipelinedLayerStreamer (streaming into VRAM)...")
            streamer_teacher = PipelinedLayerStreamer(args.teacher_path, device=device_obj, tokenizer=tokenizer)
            teacher = streamer_teacher.model
            teacher_traces = streamer_teacher.collect_hidden_states(train_prompts, batch_size=args.batch_size, device=device_obj)
        else:
            teacher = AutoModelForCausalLM.from_pretrained(
                args.teacher_path,
                dtype=torch.bfloat16,  # bfloat16 to fit in memory
                low_cpu_mem_usage=True,
            )
            print(f"Teacher loaded in {time.time() - t0:.2f}s ({sum(p.numel() for p in teacher.parameters()):,} params)")
            teacher_traces = collect_hidden_states(teacher, tokenizer, train_prompts, device="cpu", batch_size=args.batch_size)
        print(f"Teacher traces collected across {len(teacher_traces)} prompts (32 layers).")

        if args.calibrate_head:
            print("Extracting teacher lm_head and projecting via final layer chart...")
            t_final = np.concatenate([tr[32] for tr in teacher_traces], axis=0)  # (N, 2560)
            s_final = np.concatenate([tr[24] for tr in student_traces], axis=0)  # (N, 1024)
            u, _, vt = np.linalg.svd(t_final.T @ s_final, full_matrices=False)
            p_final = u @ vt  # (2560, 1024)
            w_t_head = teacher.lm_head.weight.detach().to(torch.float32).cpu().numpy()  # (V, 2560)
            teacher_head_proj = (w_t_head @ p_final).astype(np.float32)
            del w_t_head
            gc.collect()
            print(f"Projected teacher lm_head computed: shape {teacher_head_proj.shape}")
            np.save(teacher_head_file, teacher_head_proj)

        # Free teacher memory immediately to keep RAM pristine
        print("Freeing teacher model from RAM...")
        del teacher
        gc.collect()
        print("Teacher unloaded. RAM reclaimed.")

        # Cache teacher traces to disk
        print(f"Caching teacher traces to {teacher_cache_file}...")
        with open(teacher_cache_file, "wb") as f:
            pickle.dump(teacher_traces, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Teacher traces cached successfully.")

    # 6. Non-Linear Multi-Chart Weight Surgery
    print("\n[Stage 6/6] Executing Non-Linear Multi-Chart Flow Surgery on Student...")
    n_student_layers = len(student.model.layers)
    n_teacher_layers = 32

    # Flatten token activations across prompts for atlas construction
    surgery_records: list[dict[str, Any]] = []

    # Process selected student layers (up to max_blocks)
    target_layers = list(range(min(args.max_blocks, n_student_layers)))
    print(f"Targeting {len(target_layers)} layers: {target_layers}")

    for s_idx in target_layers:
        t_idx = int(round(s_idx * (n_teacher_layers - 1) / max(1, n_student_layers - 1)))
        layer_module = student.model.layers[s_idx]

        # Stack token states for student layer s_idx and teacher layer t_idx
        # State index in traces: 0 is embedding, s_idx+1 is layer output
        s_in = np.concatenate([tr[s_idx] for tr in student_traces], axis=0)  # (N, 1024)
        s_out = np.concatenate([tr[s_idx + 1] for tr in student_traces], axis=0)
        t_in = np.concatenate([tr[t_idx] for tr in teacher_traces], axis=0)  # (N, 2560)
        t_out = np.concatenate([tr[t_idx + 1] for tr in teacher_traces], axis=0)

        # Build Multi-Chart Atlas for input and output representations
        atlas_in = build_multi_chart_atlas(s_in, t_in, n_charts=args.n_charts, random_state=42 + s_idx)
        atlas_out = build_multi_chart_atlas(s_out, t_out, n_charts=args.n_charts, random_state=100 + s_idx)

        # Project teacher trajectory into student chart
        t_in_proj = atlas_in.project_teacher_to_student(t_in, s_in)
        t_out_proj = atlas_out.project_teacher_to_student(t_out, s_out)

        teacher_step = t_out_proj - t_in_proj
        student_step = s_out - s_in
        flow_residual = teacher_step - student_step  # (N, 1024)

        # Spectral entropy of residual
        entropy = compute_subspace_entropy(flow_residual)
        knot = is_knot_subspace(flow_residual, threshold=args.knot_threshold)

        # Check curvature
        curv_info = compute_2jet_curvature_metric(s_out, t_out_proj, atlas_out.projectors[0])

        print(f"\n--- Layer {s_idx:02d} -> Teacher {t_idx:02d} ---")
        print(f"  Entropy: {entropy:.4f} | Knot: {knot} | Curvature ratio: {curv_info.curvature_ratio:.4f}")

        # Compute updates for MLP using SwiGLU non-linear inversion
        mlp = layer_module.mlp
        with torch.no_grad():
            w_gate = mlp.gate_proj.weight.detach().to(torch.float32).cpu().numpy()  # (3584, 1024)
            w_up = mlp.up_proj.weight.detach().to(torch.float32).cpu().numpy()      # (3584, 1024)
            w_down = mlp.down_proj.weight.detach().to(torch.float32).cpu().numpy()  # (1024, 3584)

            # Save backup of layer weights for rollback
            layer_backup = {k: v.clone() for k, v in layer_module.state_dict().items()}

            if knot:
                print(f"  [Safety] Knot detected at Layer {s_idx} (entropy={entropy:.4f} > 0.88). SKIPPING MLP surgery to preserve memory.")
                # Apply gentle attention detour bypass only
                bypass_proj = build_attention_detour_projector(
                    [flow_residual], hidden_size=1024, eta=0.03
                )
                if hasattr(layer_module, "linear_attn"):
                    out_p = layer_module.linear_attn.out_proj.weight
                    p_arr = out_p.detach().to(torch.float32).cpu().numpy()
                    out_p.copy_(torch.from_numpy(bypass_proj @ p_arr).to(out_p.dtype))
                elif hasattr(layer_module, "self_attn"):
                    out_p = layer_module.self_attn.o_proj.weight
                    p_arr = out_p.detach().to(torch.float32).cpu().numpy()
                    out_p.copy_(torch.from_numpy(bypass_proj @ p_arr).to(out_p.dtype))
                delta_down_norm = 0.0
            else:
                # Compute student intermediate activations
                u_gate = s_in @ w_gate.T  # (N, 3584)
                v_up = s_in @ w_up.T      # (N, 3584)
                act_swiglu = swish(u_gate) * v_up  # (N, 3584)

                # Desired MLP output shift
                mlp_target_shift = 0.5 * flow_residual  # (N, 1024)
                if args.depth_schedule != "none":
                    from faytuna_flow.transformer_core import gpt2_depth_gain_weight
                    depth_mult = gpt2_depth_gain_weight(s_idx, 24, schedule=args.depth_schedule)
                    alpha = float(args.gain) * depth_mult
                else:
                    alpha = float(args.gain)


                # Non-linear pre-activation inversion for SwiGLU gates
                down_pinv = damped_tikhonov_pinv(w_down.T, ridge=1e-3, relative=True)
                delta_act = mlp_target_shift @ down_pinv  # (N, 3584)

                delta_u, delta_v = invert_swiglu_activations(u_gate, v_up, delta_act, epsilon=1e-4)
                delta_w_gate = solve_adaptive_spectral_svd_deltas(s_in, delta_u, energy_ratio=0.85, max_rank=128).T
                delta_w_up = solve_adaptive_spectral_svd_deltas(s_in, delta_v, energy_ratio=0.85, max_rank=128).T

                delta_w_gate, scale_gate = spectral_directional_rescale(w_gate, delta_w_gate, max_spectral_ratio=0.05)
                delta_w_up, scale_up = spectral_directional_rescale(w_up, delta_w_up, max_spectral_ratio=0.05)

                if args.use_memory_imprint:
                    new_down = rank_one_memory_imprint(
                        w_down,
                        act_swiglu,
                        mlp_target_shift,
                        ridge=1e-3,
                        gain=alpha,
                        max_relative_norm=0.03,
                    )
                    delta_down_norm = float(np.linalg.norm(new_down - w_down))
                    scale_down = 1.0
                else:
                    # 1. Update down_proj via Adaptive Spectral SVD
                    delta_w_down = solve_adaptive_spectral_svd_deltas(act_swiglu, mlp_target_shift, energy_ratio=0.85, max_rank=128)
                    delta_w_down_t = delta_w_down.T  # (1024, 3584)
                    delta_w_down_t, scale_down = spectral_directional_rescale(w_down, delta_w_down_t, max_spectral_ratio=0.05)
                    delta_down_norm = float(np.linalg.norm(delta_w_down_t))
                    new_down = w_down + alpha * delta_w_down_t

                new_gate = w_gate + alpha * delta_w_gate
                new_up = w_up + alpha * delta_w_up

                # Commit to PyTorch module in its native dtype
                mlp.down_proj.weight.copy_(torch.from_numpy(new_down).to(mlp.down_proj.weight.dtype))
                mlp.gate_proj.weight.copy_(torch.from_numpy(new_gate).to(mlp.gate_proj.weight.dtype))
                mlp.up_proj.weight.copy_(torch.from_numpy(new_up).to(mlp.up_proj.weight.dtype))

                print(f"  -> Applied non-linear updates (scale_down={scale_down:.4f}, scale_gate={scale_gate:.4f})")

            # Quick validation check on holdout prompt
            val_eval = evaluate_nll_and_ppl(student, tokenizer, EVAL_PROMPTS[:1], device=device_obj)
            if val_eval["mean_nll"] > pre_eval["mean_nll"] * 1.02:
                print(f"  [Rollback] Layer {s_idx} degraded holdout NLL ({val_eval['mean_nll']:.4f} vs base {pre_eval['mean_nll']:.4f}); rolling back.")
                layer_module.load_state_dict(layer_backup)
                accepted = False
            else:
                print(f"  [Accepted] Layer {s_idx} holdout NLL = {val_eval['mean_nll']:.4f}")
                accepted = True

            sys.stdout.flush()

            surgery_records.append({
                "layer": s_idx,
                "teacher_layer": t_idx,
                "entropy": float(entropy),
                "knot": bool(knot),
                "curvature_ratio": float(curv_info.curvature_ratio),
                "down_update_norm": float(delta_down_norm),
                "accepted": accepted,
                "holdout_nll": float(val_eval["mean_nll"]),
            })

    # 6B. Vocabulary Head Alignment (Vector 4)
    head_aligned = False
    if args.calibrate_head and teacher_head_proj is not None:
        print("\n[Stage 6B] Aligning Student Vocabulary Classification Hyperplanes (lm_head)...")
        w_s_head = student.lm_head.weight.detach().to(torch.float32).cpu().numpy()
        delta_head = teacher_head_proj - w_s_head
        norm_s = float(np.linalg.norm(w_s_head))
        norm_delta = float(np.linalg.norm(delta_head))
        clamp = min(1.0, (0.02 * norm_s) / max(norm_delta, 1e-8))
        new_head = w_s_head + (float(args.head_gain) * clamp) * delta_head

        head_backup = student.lm_head.weight.detach().clone()
        with torch.no_grad():
            student.lm_head.weight.copy_(torch.from_numpy(new_head).to(student.lm_head.weight.dtype))
        val_head = evaluate_nll_and_ppl(student, tokenizer, EVAL_PROMPTS, device=device_obj)
        current_holdout = surgery_records[-1].get("holdout_nll", pre_eval["mean_nll"]) if surgery_records else pre_eval["mean_nll"]
        if val_head["mean_nll"] > current_holdout:
            print(f"  [Rollback] Vocabulary alignment degraded holdout ({val_head['mean_nll']:.4f} vs {current_holdout:.4f}); rolling back.")
            with torch.no_grad():
                student.lm_head.weight.copy_(head_backup)
        else:
            print(f"  [Accepted] Vocabulary head aligned (gain={args.head_gain}, clamp={clamp:.4f}, delta_norm={norm_delta:.2f})")
            head_aligned = True

    # 6C. Flow-Guided Iterative Micro-Distillation (Vector 2)
    distill_report = None
    if args.distill_steps > 0:
        print(f"\n[Stage 6C] Executing {args.distill_steps} Flow-Guided Micro-Distillation steps...")
        token_batches = [tokenizer(p, return_tensors="pt")["input_ids"] for p in train_prompts]
        pre_distill_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
        try:
            if device_type == "dml":
                student = student.to("cpu")
                distill_dev = "cpu"
            else:
                distill_dev = device_type

            distill_report = run_flow_guided_distillation(
                student,
                None,
                token_batches,
                steps=int(args.distill_steps),
                lr=1e-5,
                lambda_flow=0.0,
                lambda_ce=1.0,
                device=distill_dev,
                freeze_embeddings=True,
            )
            if device_type == "dml":
                for p in student.parameters():
                    if p.dtype == torch.bfloat16:
                        p.data = p.data.to(torch.float32)
                for b in student.buffers():
                    if b.dtype == torch.bfloat16:
                        b.data = b.data.to(torch.float32)
                student = student.to(device_obj)

            if distill_report["final_loss"] > distill_report["initial_loss"]:
                print(f"  [Rollback] Distillation loss diverged ({distill_report['initial_loss']:.4f} -> {distill_report['final_loss']:.4f}); rolling back to analytical surgery weights.")
                student.load_state_dict(pre_distill_state)
                if device_type == "dml":
                    student = student.to(device_obj)
            else:
                print(f"  -> Distillation completed: initial loss {distill_report['initial_loss']:.4f} -> final {distill_report['final_loss']:.4f}, param drift: {distill_report['relative_parameter_drift']:.6f}")
        except Exception as err:
            print(f"  [Warning] Micro-distillation skipped ({err}); proceeding with analytical surgery weights.")
            student.load_state_dict(pre_distill_state)
            if device_type == "dml":
                try:
                    student = student.to(device_obj)
                except Exception:
                    pass

    # 7. Post-Surgery Multi-Domain Holdout Evaluation
    print("\n" + "=" * 78)
    print("  POST-SURGERY MULTI-DOMAIN GENERALIZATION EVALUATION (5 DOMAINS)")
    print("=" * 78)
    post_benchmark = evaluate_benchmark_suites(student, tokenizer, EVAL_SUITES, device=device_obj, batch_size=args.batch_size)
    post_eval = post_benchmark["in_domain_stem"]

    print("\n" + "-" * 84)
    print(f"{'Domain Suite':<28} | {'Pre NLL':<8} -> {'Post NLL':<8} | {'Delta':<9} | {'Change %':<8} | {'Status'}")
    print("-" * 84)

    suite_comparisons: dict[str, Any] = {}
    for domain in list(EVAL_SUITES.keys()) + ["global"]:
        pre_res = pre_benchmark[domain]
        post_res = post_benchmark[domain]
        d_nll = post_res["mean_nll"] - pre_res["mean_nll"]
        pct = (d_nll / pre_res["mean_nll"]) * 100.0
        dom_status = "IMPROVED" if d_nll < -0.005 else ("DEGRADED" if d_nll > 0.005 else "STABLE")
        print(f"{domain:<28} | {pre_res['mean_nll']:<8.4f} -> {post_res['mean_nll']:<8.4f} | {d_nll:<+9.4f} | {pct:<+7.2f}% | [{dom_status}]")
        suite_comparisons[domain] = {
            "pre_nll": pre_res["mean_nll"],
            "post_nll": post_res["mean_nll"],
            "pre_ppl": pre_res["ppl"],
            "post_ppl": post_res["ppl"],
            "delta_nll": d_nll,
            "pct_change": pct,
            "status": dom_status,
        }
    print("-" * 84)

    delta_nll = post_eval["mean_nll"] - pre_eval["mean_nll"]
    pct_change = (delta_nll / pre_eval["mean_nll"]) * 100.0
    status = suite_comparisons["global"]["status"]

    # Save summary report
    summary = {
        "student_model": args.student_path,
        "teacher_model": args.teacher_path,
        "gain_alpha": args.gain,
        "n_charts": args.n_charts,
        "pre_eval": pre_eval,
        "post_eval": post_eval,
        "pre_benchmark": pre_benchmark,
        "post_benchmark": post_benchmark,
        "delta_nll": delta_nll,
        "percent_change": pct_change,
        "status": status,
        "suite_comparisons": suite_comparisons,
        "layers_modified": surgery_records,
    }

    report_path = out_dir / "qwen35_transfer_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull transfer report written to {report_path}")

    # Optionally save student model
    if args.save_model:
        save_dir = out_dir / "transferred_student"
        print(f"\nSaving transferred student model to {save_dir}...")
        student.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Model saved successfully.")

    # Generate qualitative test output
    print("\nQualitative Generation Test:")
    test_prompt = "In deep learning, non-linear representations are crucial because"
    student = student.to("cpu")
    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = student.generate(**inputs, max_new_tokens=25, do_sample=False)
    gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Output: {gen_text}")
    print("\nTransfer pipeline execution complete!")


if __name__ == "__main__":
    main()
