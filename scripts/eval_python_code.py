"""Evaluate Python code completion and generation for baseline vs candidate GPT-2 models."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.text_ab_eval import PYTHON_CODE_EVAL_CASES, _encoded, _logits, _target_token_positions


FREEFORM_CODE_PROMPTS: tuple[dict[str, str], ...] = (
    {"id": "gen_is_prime", "prompt": "def is_prime(n: int) -> bool:\n    \"\"\"Check if n is prime.\"\"\"\n    if n < 2:\n        return "},
    {"id": "gen_factorial", "prompt": "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return "},
    {"id": "gen_binary_search", "prompt": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return "},
    {"id": "gen_filter_positive", "prompt": "def filter_positive(numbers):\n    return [x for x in numbers if "},
    {"id": "gen_class_stack", "prompt": "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items."},
)


def evaluate_target_nll(model: Any, tokenizer: Any, cases: Sequence[dict[str, str]], device: Any) -> dict[str, Any]:
    model.eval()
    results = []
    for case in cases:
        prompt = case["prompt"]
        continuation = case["continuation"]
        full_ids, full_mask = _encoded(tokenizer, prompt + continuation, device)
        target_positions, boundary_method = _target_token_positions(tokenizer, prompt, continuation, full_ids)
        target_positions = [p for p in target_positions if p > 0]
        with torch.no_grad():
            output = model(input_ids=full_ids, attention_mask=full_mask)
            logits = _logits(output)
        labels = torch.full_like(full_ids, -100)
        labels[:, target_positions] = full_ids[:, target_positions]
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        valid = shifted_labels != -100
        loss_sum = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        mean_nll = float((loss_sum / valid.sum()).detach().cpu().item())
        ppl = math.exp(min(mean_nll, 700.0))

        # Greedy completion match
        input_ids, attention_mask = _encoded(tokenizer, prompt, device)
        target_count = len(target_positions)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=target_count,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_tokens = gen[:, input_ids.shape[1] :]
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=False)
        target_text = continuation

        results.append({
            "id": case["id"],
            "prompt": prompt,
            "target": target_text,
            "nll": mean_nll,
            "ppl": ppl,
            "greedy_gen": gen_text,
            "exact_match": (gen_text == target_text),
        })

    avg_nll = sum(r["nll"] for r in results) / len(results)
    avg_ppl = sum(r["ppl"] for r in results) / len(results)
    exact_acc = sum(1 for r in results if r["exact_match"]) / len(results)
    return {
        "mean_nll": avg_nll,
        "mean_ppl": avg_ppl,
        "exact_accuracy": exact_acc,
        "cases": results,
    }


def evaluate_freeform(model: Any, tokenizer: Any, prompts: Sequence[dict[str, str]], device: Any, max_tokens: int = 32) -> list[dict[str, Any]]:
    model.eval()
    generations = []
    for item in prompts:
        prompt = item["prompt"]
        input_ids, attention_mask = _encoded(tokenizer, prompt, device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = tokenizer.decode(gen[0, input_ids.shape[1] :], skip_special_tokens=True)
        full_code = prompt + continuation

        # Check Python syntax validity of snippet
        syntax_valid = False
        try:
            ast.parse(full_code)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False

        generations.append({
            "id": item["id"],
            "prompt": prompt,
            "continuation": continuation,
            "full_code": full_code,
            "syntax_valid": syntax_valid,
        })
    return generations


def main():
    parser = argparse.ArgumentParser(description="Evaluate Python code capabilities of GPT-2 small baseline vs candidate")
    parser.add_argument("--baseline", required=True, help="path to baseline model checkpoint")
    parser.add_argument("--candidate", required=False, help="path to candidate model checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=False)
    args = parser.parse_args()

    def resolve_model_path(path_str: str) -> Path:
        p = Path(path_str)
        if (p / "candidate").exists() and (p / "candidate" / "config.json").exists():
            return (p / "candidate").resolve()
        if p.exists():
            return p.resolve()
        parent_candidate = PROJECT_ROOT.parent / path_str
        if (parent_candidate / "candidate").exists() and (parent_candidate / "candidate" / "config.json").exists():
            return (parent_candidate / "candidate").resolve()
        if parent_candidate.exists():
            return parent_candidate.resolve()
        return p.resolve()

    device = torch.device(args.device)
    baseline_path = resolve_model_path(args.baseline)
    print(f"Loading baseline from: {baseline_path}")
    tok = AutoTokenizer.from_pretrained(str(baseline_path), local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(str(baseline_path), local_files_only=True).to(device)

    base_eval = evaluate_target_nll(base_model, tok, PYTHON_CODE_EVAL_CASES, device)
    base_freeform = evaluate_freeform(base_model, tok, FREEFORM_CODE_PROMPTS, device)

    print("\n=== BASELINE PYTHON CODE EVALUATION ===")
    print(f"Baseline Mean NLL:    {base_eval['mean_nll']:.4f} (PPL: {base_eval['mean_ppl']:.2f})")
    print(f"Exact Match Accuracy: {base_eval['exact_accuracy']:.1%}")
    for c in base_eval["cases"]:
        status = "OK" if c["exact_match"] else "MISMATCH"
        print(f"  [{c['id']}] NLL={c['nll']:.3f} | gen: {repr(c['greedy_gen'])} vs target: {repr(c['target'])} -> {status}")

    report = {
        "baseline": {
            "path": str(args.baseline),
            "target_eval": base_eval,
            "freeform": base_freeform,
        }
    }

    if args.candidate:
        cand_path = resolve_model_path(args.candidate)
        if cand_path.exists():
            print(f"Loading candidate from: {cand_path}")
            cand_model = AutoModelForCausalLM.from_pretrained(str(cand_path), local_files_only=True).to(device)
            cand_eval = evaluate_target_nll(cand_model, tok, PYTHON_CODE_EVAL_CASES, device)
            cand_freeform = evaluate_freeform(cand_model, tok, FREEFORM_CODE_PROMPTS, device)

        delta_nll = cand_eval["mean_nll"] - base_eval["mean_nll"]
        delta_ppl = cand_eval["mean_ppl"] - base_eval["mean_ppl"]
        acc_gain = cand_eval["exact_accuracy"] - base_eval["exact_accuracy"]

        report["candidate"] = {
            "path": str(args.candidate),
            "target_eval": cand_eval,
            "freeform": cand_freeform,
        }
        report["comparison"] = {
            "baseline_nll": base_eval["mean_nll"],
            "candidate_nll": cand_eval["mean_nll"],
            "delta_nll": delta_nll,
            "baseline_ppl": base_eval["mean_ppl"],
            "candidate_ppl": cand_eval["mean_ppl"],
            "delta_ppl": delta_ppl,
            "baseline_exact_acc": base_eval["exact_accuracy"],
            "candidate_exact_acc": cand_eval["exact_accuracy"],
            "accuracy_gain": acc_gain,
        }

        print("\n=== PYTHON CODE EVALUATION RESULTS ===")
        print(f"Baseline Mean NLL:    {base_eval['mean_nll']:.4f} (PPL: {base_eval['mean_ppl']:.2f})")
        print(f"Candidate Mean NLL:   {cand_eval['mean_nll']:.4f} (PPL: {cand_eval['mean_ppl']:.2f})")
        print(f"Delta NLL:            {delta_nll:+.4f} ({'IMPROVEMENT' if delta_nll < 0 else 'DEGRADATION'})")
        print(f"Exact Match Accuracy: {base_eval['exact_accuracy']:.1%} -> {cand_eval['exact_accuracy']:.1%}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
