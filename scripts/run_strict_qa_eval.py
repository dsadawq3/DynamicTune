"""Strict Target-Only Conditional QA & Knowledge Benchmark for Qwen 3.5.

Evaluates Cross-Entropy NLL and Perplexity STRICTLY on ground-truth target answers
by masking question/prompt tokens with -100 (loss ignored).

Head-to-head comparison:
- Baseline Student: C:\\models\\Qwen3.5-0.8B-Base
- Transferred Student: runs/qwen35_transfer/transferred_student
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 25 Strictly Held-Out Virgin Question -> Ground-Truth Answer Benchmarks
STRICT_QA_DATASET: dict[str, list[dict[str, str]]] = {
    "history_and_facts": [
        {
            "question": "In what year was the Treaty of Westphalia signed, ending the Thirty Years' War in Europe?",
            "answer": "1648.",
        },
        {
            "question": "What was the capital of the Byzantine Empire before its fall to the Ottoman Empire in 1453?",
            "answer": "Constantinople.",
        },
        {
            "question": "Who was the primary author of the United States Declaration of Independence in 1776?",
            "answer": "Thomas Jefferson.",
        },
        {
            "question": "Which ancient civilization built the fortress city of Machu Picchu in Peru?",
            "answer": "The Inca Empire.",
        },
        {
            "question": "The ancient Silk Road primarily connected Han Dynasty China with which major western empire?",
            "answer": "The Roman Empire.",
        },
    ],
    "science_and_medicine": [
        {
            "question": "Which pancreatic islet cells are responsible for the synthesis and secretion of insulin?",
            "answer": "Beta cells.",
        },
        {
            "question": "What antibiotic substance was discovered in mold by Alexander Fleming in 1928?",
            "answer": "Penicillin.",
        },
        {
            "question": "In molecular biology, which primary enzyme transcribes genetic information from DNA into messenger RNA?",
            "answer": "RNA polymerase.",
        },
        {
            "question": "Which fundamental quantum mechanical principle states that one cannot simultaneously determine both the exact position and momentum of a subatomic particle?",
            "answer": "Heisenberg uncertainty principle.",
        },
        {
            "question": "Which cellular organelle generates most of the chemical energy needed by eukaryotic cells via oxidative phosphorylation?",
            "answer": "Mitochondria.",
        },
    ],
    "logic_and_reasoning": [
        {
            "question": "If a rigid glass bottle filled completely to the top with liquid water is tightly sealed and frozen, why does the bottle shatter?",
            "answer": "Because water expands as it crystallizes into ice, increasing pressure beyond the tensile strength of glass.",
        },
        {
            "question": "Why does an automobile risk hydroplaning when traveling at high speeds on a water-covered highway?",
            "answer": "Because water pressure builds in front of the tire until a thin film separates the tire from the road surface, eliminating traction.",
        },
        {
            "question": "If severe drought cuts the global wheat harvest by half while consumer demand remains unchanged, what happens to the market price of bread?",
            "answer": "The market price of bread increases significantly due to the reduced supply of wheat.",
        },
        {
            "question": "When a solid copper rod is heated uniformly, what happens to its physical length?",
            "answer": "Its length increases due to thermal expansion.",
        },
        {
            "question": "Why does acoustic sound propagate significantly faster through solid steel than through air?",
            "answer": "Because steel has much higher elastic stiffness and atomic density than compressible gas.",
        },
    ],
    "russian_qa": [
        {
            "question": "Какая фундаментальная математическая формула связывает операцию взятия первообразной с определённым интегралом?",
            "answer": "Формула Ньютона — Лейбница.",
        },
        {
            "question": "В каком году и в какой научной статье исследователи из Google впервые представили архитектуру трансформеров?",
            "answer": "В 2017 году в статье Attention Is All You Need.",
        },
        {
            "question": "Как в физике называется состояние двух или более частиц, при котором их квантовые состояния неразрывно связаны независимо от расстояния?",
            "answer": "Квантовая запутанность.",
        },
        {
            "question": "Какой химический элемент составляет наибольшую объёмную долю в атмосфере Земли?",
            "answer": "Азот.",
        },
        {
            "question": "Как формулируется второе начало классической термодинамики для изолированной физической системы?",
            "answer": "Энтропия изолированной системы никогда не убывает со временем.",
        },
    ],
    "code_and_algorithms": [
        {
            "question": "What is the average time complexity of the quicksort algorithm when sorting an array of n elements?",
            "answer": "O(n log n).",
        },
        {
            "question": "Which algorithmic paradigm does Dijkstra's shortest path algorithm utilize to find the shortest paths from a single source node?",
            "answer": "A greedy algorithm paradigm with a priority queue.",
        },
        {
            "question": "Which linear data structure strictly enforces a Last-In, First-Out (LIFO) operational discipline?",
            "answer": "A stack.",
        },
        {
            "question": "In deep residual neural networks (ResNet), what architectural feature allows gradients to propagate directly across layers without vanishing?",
            "answer": "Skip connections (or residual shortcut connections).",
        },
        {
            "question": "Which consensus algorithm is specifically structured as an understandable state-machine replication alternative to Paxos?",
            "answer": "The Raft consensus algorithm.",
        },
    ],
}


def evaluate_target_only_nll(
    model: Any,
    tokenizer: Any,
    question: str,
    target_answer: str,
    is_russian: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compute Cross-Entropy NLL STRICTLY on target answer tokens.
    
    Question and prompt tokens are assigned label -100 so that they do not
    contribute to the loss. Loss is calculated exclusively on P(Target | Prompt).
    """
    if is_russian:
        prompt_str = f"Вопрос: {question}\nОтвет: "
    else:
        prompt_str = f"Question: {question}\nAnswer: "

    prompt_ids = tokenizer(prompt_str, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target_answer, add_special_tokens=False)["input_ids"]

    # Full sequence is prompt + target
    full_ids = prompt_ids + target_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    # Mask: prompt tokens = -100, target tokens = actual IDs
    labels = torch.full_like(input_ids, -100)
    labels[0, len(prompt_ids):] = torch.tensor(target_ids, dtype=torch.long, device=device)

    model.eval()
    try:
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            nll = float(outputs.loss.item())
    except BaseException:
        model_cpu = model.to("cpu")
        input_ids_cpu = input_ids.to("cpu")
        labels_cpu = labels.to("cpu")
        with torch.no_grad():
            outputs = model_cpu(input_ids_cpu, labels=labels_cpu)
            nll = float(outputs.loss.item())

    return {
        "nll": nll,
        "target_tokens": len(target_ids),
        "prompt_tokens": len(prompt_ids),
    }


def resolve_device(requested: str = "cpu") -> tuple[Any, str]:
    req_lower = str(requested).lower()
    if req_lower in {"dml", "directml"}:
        try:
            import torch_directml
            return torch_directml.device(), "dml"
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


def load_eval_model(model_path: str, device_obj: Any, device_type: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if device_type == "dml":
        try:
            for p in model.parameters():
                if p.dtype == torch.bfloat16:
                    p.data = p.data.to(torch.float32)
            for b in model.buffers():
                if b.dtype == torch.bfloat16:
                    b.data = b.data.to(torch.float32)
            model = model.to(device_obj)
        except Exception as err:
            print(f"[Warning] Failed to place model on DirectML ({err}); falling back to CPU.")
            model = model.to("cpu")
    elif device_type != "cpu":
        model = model.to(device_obj)
    return model


def evaluate_model_on_strict_qa(
    model: Any,
    tokenizer: Any,
    dataset: dict[str, list[dict[str, str]]],
    device: Any = "cpu",
) -> dict[str, Any]:
    """Evaluate target-only NLL across all domain suites."""
    results: dict[str, Any] = {}
    global_weighted_nll = 0.0
    global_tokens = 0

    for domain_name, pairs in dataset.items():
        is_ru = domain_name == "russian_qa"
        domain_weighted_nll = 0.0
        domain_tokens = 0
        pair_records = []

        for pair in pairs:
            q = pair["question"]
            a = pair["answer"]
            res = evaluate_target_only_nll(model, tokenizer, q, a, is_russian=is_ru, device=device)
            domain_weighted_nll += res["nll"] * res["target_tokens"]
            domain_tokens += res["target_tokens"]
            pair_records.append({
                "question": q,
                "answer": a,
                "nll": res["nll"],
                "target_tokens": res["target_tokens"],
            })

        mean_nll = domain_weighted_nll / max(domain_tokens, 1)
        results[domain_name] = {
            "mean_nll": mean_nll,
            "ppl": float(np.exp(mean_nll)),
            "total_target_tokens": domain_tokens,
            "pairs": pair_records,
        }

        global_weighted_nll += domain_weighted_nll
        global_tokens += domain_tokens

    overall_nll = global_weighted_nll / max(global_tokens, 1)
    results["global"] = {
        "mean_nll": overall_nll,
        "ppl": float(np.exp(overall_nll)),
        "total_target_tokens": global_tokens,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict Target-Only QA Evaluation Benchmark")
    parser.add_argument("--base-model", type=str, default=r"C:\models\Qwen3.5-0.8B-Base")
    parser.add_argument("--transferred-model", type=str, default="runs/qwen35_transfer/transferred_student")
    parser.add_argument("--output-report", type=str, default="runs/qwen35_transfer/strict_qa_audit_report.json")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "dml", "cuda", "auto"], help="Evaluation device")
    args = parser.parse_args()

    out_file = Path(args.output_report)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    device_obj, device_type = resolve_device(args.device)

    print("=" * 80)
    print("  STRICT TARGET-ONLY CONDITIONAL QA BENCHMARK (QWEN 3.5)")
    print("  Evaluating P(Target Answer | Question) with question tokens masked to -100")
    print(f"  Baseline Model   : {args.base_model}")
    print(f"  Transferred Model: {args.transferred_model}")
    print(f"  Device           : {device_obj} ({device_type})")
    print(f"  Total QA Pairs   : {sum(len(v) for v in STRICT_QA_DATASET.values())} across {len(STRICT_QA_DATASET)} domains")
    print("=" * 80)

    # 1. Load Tokenizer
    print("\n[Stage 1/4] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print(f"Tokenizer loaded (vocab={len(tokenizer)}).")

    # 2. Evaluate Baseline Model
    print(f"\n[Stage 2/4] Evaluating BASELINE Student on Strict Target-Only QA ({device_type})...")
    t0 = time.time()
    base_model = load_eval_model(args.base_model, device_obj, device_type)
    print(f"Baseline loaded in {time.time() - t0:.2f}s.")
    base_results = evaluate_model_on_strict_qa(base_model, tokenizer, STRICT_QA_DATASET, device=device_obj)
    print("Baseline evaluation complete.")

    for domain, res in base_results.items():
        if domain != "global":
            print(f"  [Base]  {domain:<24} : Target NLL = {res['mean_nll']:.4f} | Target PPL = {res['ppl']:.4f}")
    print(f"  [Base]  {'GLOBAL (ALL ANSWERS)':<24} : Target NLL = {base_results['global']['mean_nll']:.4f} | Target PPL = {base_results['global']['ppl']:.4f}")

    # Free base model from RAM/VRAM
    print("Unloading baseline model...")
    del base_model
    import gc
    gc.collect()

    # 3. Evaluate Transferred Model
    print(f"\n[Stage 3/4] Evaluating TRANSFERRED Student on Strict Target-Only QA ({device_type})...")
    t0 = time.time()
    trans_model = load_eval_model(args.transferred_model, device_obj, device_type)
    print(f"Transferred model loaded in {time.time() - t0:.2f}s.")
    trans_results = evaluate_model_on_strict_qa(trans_model, tokenizer, STRICT_QA_DATASET, device=device_obj)
    print("Transferred model evaluation complete.")

    # 4. Comparative Audit Table
    print("\n" + "=" * 86)
    print("  STRICT TARGET-ONLY CONDITIONAL QA COMPARISON TABLE")
    print("=" * 86)
    print(f"{'Domain Suite':<24} | {'Base NLL':<9} -> {'Trans NLL':<9} | {'Delta':<9} | {'Change %':<8} | {'Status'}")
    print("-" * 86)

    comparisons: dict[str, Any] = {}
    for domain in list(STRICT_QA_DATASET.keys()) + ["global"]:
        b_nll = base_results[domain]["mean_nll"]
        t_nll = trans_results[domain]["mean_nll"]
        b_ppl = base_results[domain]["ppl"]
        t_ppl = trans_results[domain]["ppl"]
        delta = t_nll - b_nll
        pct = (delta / b_nll) * 100.0
        status = "IMPROVED" if delta < -0.005 else ("DEGRADED" if delta > 0.005 else "STABLE")

        print(f"{domain:<24} | {b_nll:<9.4f} -> {t_nll:<9.4f} | {delta:<+9.4f} | {pct:<+7.2f}% | [{status}]")
        comparisons[domain] = {
            "base_nll": b_nll,
            "trans_nll": t_nll,
            "base_ppl": b_ppl,
            "trans_ppl": t_ppl,
            "delta_nll": delta,
            "pct_change": pct,
            "status": status,
        }
    print("-" * 86)

    # 5. Qualitative Side-by-Side Generation Comparison
    print("\nQualitative Generation Check on 3 Sample Questions:")
    sample_questions = [
        ("In what year was the Treaty of Westphalia signed?", False),
        ("Какая фундаментальная формула связывает первообразную с определённым интегралом?", True),
        ("Why does acoustic sound propagate faster through solid steel than through air?", False),
    ]

    trans_model = trans_model.to("cpu")
    generation_samples = []
    for q, is_ru in sample_questions:
        prompt_text = f"Вопрос: {q}\nОтвет: " if is_ru else f"Question: {q}\nAnswer: "
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = trans_model.generate(**inputs, max_new_tokens=30, do_sample=False)
        gen_text = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\nQ: {q}")
        print(f"Transferred Model Generated: {gen_text}")
        generation_samples.append({"question": q, "generated": gen_text})

    # Save full report
    audit_report = {
        "base_model": args.base_model,
        "transferred_model": args.transferred_model,
        "comparisons": comparisons,
        "base_results": base_results,
        "trans_results": trans_results,
        "generation_samples": generation_samples,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)
    print(f"\nStrict QA audit report successfully written to {out_file}")


if __name__ == "__main__":
    main()
