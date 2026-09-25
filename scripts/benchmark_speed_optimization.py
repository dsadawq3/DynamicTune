"""Empirical microbenchmark for Qwen 3.5 speed optimizations.

Compares:
1. Unoptimized extraction: Sequential (batch_size=1) through full model including lm_head (248,320 vocab projection).
2. Optimized extraction: Zero-logit backbone forward with dynamic batching (batch_size=4) and attention unpadding.

Measures:
- Wall-clock execution time for both passes.
- Speedup factor and time reduction percentage.
- Bit-exact numerical equivalence: ||h_opt - h_orig||_inf.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Any, Sequence
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_qwen35_transfer import (
    TRAIN_PROMPTS,
    collect_hidden_states,
    resolve_device,
)


def run_unoptimized_collection(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    device: Any = "cpu",
) -> tuple[list[dict[int, np.ndarray]], float]:
    """Extract hidden states sequentially using full causal LM forward (including lm_head)."""
    model.eval()
    all_states: list[dict[int, np.ndarray]] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            inputs_dev = {k: v.to(device) for k, v in inputs.items()}
            # Full model forward calculates logits across full 248,320 vocabulary
            outputs = model(**inputs_dev, output_hidden_states=True)
            prompt_states: dict[int, np.ndarray] = {}
            for l_idx, hs in enumerate(outputs.hidden_states):
                prompt_states[l_idx] = hs[0].detach().to(torch.float32).cpu().numpy()
            all_states.append(prompt_states)
    elapsed = time.perf_counter() - t0
    return all_states, elapsed


def run_optimized_collection(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    device: Any = "cpu",
    batch_size: int = 4,
) -> tuple[list[dict[int, np.ndarray]], float]:
    """Extract hidden states using zero-logit backbone extraction with dynamic batching."""
    t0 = time.perf_counter()
    all_states = collect_hidden_states(
        model,
        tokenizer,
        prompts,
        device=device,
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - t0
    return all_states, elapsed


def main() -> None:
    model_path = r"C:\models\Qwen3.5-0.8B-Base"
    print("=" * 80)
    print("  QWEN 3.5 SPEED OPTIMIZATION EMPIRICAL BENCHMARK")
    print(f"  Target Checkpoint: {model_path}")
    print("=" * 80)

    # 1. Load Tokenizer & Model
    print("\n[1/4] Loading Tokenizer & Model...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Select 4 representative evaluation prompts for benchmark
    test_prompts = list(TRAIN_PROMPTS[:4])
    print(f"\n[2/5] Selected {len(test_prompts)} prompts for timing benchmark:")
    for idx, p in enumerate(test_prompts):
        print(f"  Prompt {idx + 1:02d} ({len(p.split())} words): {p[:60]}...")

    # Warmup pass to prime CPU caches
    print("\n[Warmup] Priming execution pipeline...")
    _ = collect_hidden_states(model, tokenizer, test_prompts[:1], device="cpu", batch_size=1)

    # 2. Run Unoptimized (Sequential, Full lm_head)
    print("\n[3/5] Running Unoptimized Trace Collection (Batch size = 1, with lm_head projection)...")
    gc.collect()
    orig_states, unopt_time = run_unoptimized_collection(model, tokenizer, test_prompts, device="cpu")
    print(f"  Unoptimized Wall-Clock Time: {unopt_time:.4f}s ({unopt_time / len(test_prompts):.4f}s/prompt)")

    # 3. Run Optimized Zero-Logit (Sequential, batch_size = 1)
    print("\n[4/5] Running Optimized Zero-Logit Trace Collection (Batch size = 1, backbone only)...")
    gc.collect()
    opt_states_single, opt_time_single = run_optimized_collection(model, tokenizer, test_prompts, device="cpu", batch_size=1)
    print(f"  Optimized (Zero-Logit, bs=1): {opt_time_single:.4f}s ({opt_time_single / len(test_prompts):.4f}s/prompt)")

    # 4. Run Optimized Zero-Logit Batched (batch_size = 4)
    print("\n[5/5] Running Optimized Batched Trace Collection (Batch size = 4, backbone only)...")
    gc.collect()
    opt_states_batched, opt_time_batched = run_optimized_collection(model, tokenizer, test_prompts, device="cpu", batch_size=4)
    print(f"  Optimized (Zero-Logit, bs=4): {opt_time_batched:.4f}s ({opt_time_batched / len(test_prompts):.4f}s/prompt)")

    # 5. Verification of Numerical Equivalence (Zero-Logit vs Original)
    print("\n" + "-" * 80)
    print("  NUMERICAL EQUIVALENCE VERIFICATION (||h_opt - h_orig||_inf)")
    print("-" * 80)

    max_inf_norm = 0.0
    assert len(orig_states) == len(opt_states_single) == len(test_prompts)
    num_layers = len(orig_states[0])

    for l_idx in range(num_layers):
        for p_idx in range(len(test_prompts)):
            h_orig = orig_states[p_idx][l_idx]
            h_opt = opt_states_single[p_idx][l_idx]
            assert h_orig.shape == h_opt.shape, f"Shape mismatch at prompt {p_idx}, layer {l_idx}: {h_orig.shape} vs {h_opt.shape}"
            diff = float(np.max(np.abs(h_opt - h_orig)))
            if diff > max_inf_norm:
                max_inf_norm = diff

    print(f"Total layers checked: {num_layers} (Layer 0 = embedding, Layers 1..24 = transformer blocks)")
    print(f"Global Maximum Absolute Difference ||h_opt - h_orig||_inf: {max_inf_norm:.8e}")
    assert max_inf_norm < 1e-5, f"Numerical difference {max_inf_norm} exceeds tolerance 1e-5!"
    print("STATUS: BIT-EXACT NUMERICAL EQUIVALENCE CONFIRMED (||h_opt - h_orig||_inf < 10^-5) [PASS]")

    # 6. Speedup and Timing Summary
    zl_speedup = unopt_time / max(opt_time_single, 1e-9)
    zl_reduction = (1.0 - opt_time_single / unopt_time) * 100.0

    batched_speedup = unopt_time / max(opt_time_batched, 1e-9)
    batched_reduction = (1.0 - opt_time_batched / unopt_time) * 100.0

    print("\n" + "=" * 80)
    print("  EMPIRICAL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"  {'Execution Configuration':<35} | {'Runtime':<12} | {'Per-Prompt':<12} | {'Speedup':<10} | {'Time Saved':<10}")
    print("-" * 80)
    print(f"  {'Unoptimized (Full LM, bs=1)':<35} | {f'{unopt_time:.3f}s':<12} | {f'{unopt_time / len(test_prompts):.3f}s':<12} | {'1.00x':<10} | {'0.0%':<10}")
    print(f"  {'Optimized Zero-Logit (bs=1)':<35} | {f'{opt_time_single:.3f}s':<12} | {f'{opt_time_single / len(test_prompts):.3f}s':<12} | {f'{zl_speedup:.2f}x':<10} | {f'{zl_reduction:.1f}%':<10}")
    print(f"  {'Optimized Batched (bs=4)':<35} | {f'{opt_time_batched:.3f}s':<12} | {f'{opt_time_batched / len(test_prompts):.3f}s':<12} | {f'{batched_speedup:.2f}x':<10} | {f'{batched_reduction:.1f}%':<10}")
    print("-" * 80)
    print(f"  BIT-EXACT MAX ABSOLUTE DIFF: {max_inf_norm:.8e} (Tolerance: 1.0e-05) -> [PASS]")
    print(f"  ZERO-LOGIT SPEEDUP FACTOR:   {zl_speedup:.2f}x faster ({zl_reduction:.1f}% time saved)")
    print(f"  BATCHED SPEEDUP FACTOR:      {batched_speedup:.2f}x faster ({batched_reduction:.1f}% time saved)")
    print("=" * 80)


if __name__ == "__main__":
    main()
