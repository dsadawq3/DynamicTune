"""Vulkan-Accelerated llama.cpp Benchmark Middleware for Qwen 3.5.

Evaluates Base GGUF vs Transferred 4-Block Anchor GGUF on 30 hardcore expert tasks
across 5 domains: Mathematics & Logic, Python Algorithms, Russian Nuance,
Biomedicine, and Deep Learning Architecture.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

HARDCORE_BENCHMARK_SUITE: dict[str, list[dict[str, Any]]] = {
    "mathematics_and_logic": [
        {
            "id": "math_modular_arithmetic",
            "question": "What is the remainder when 7^222 is divided by 11? Provide the exact integer remainder.",
            "target": "9",
            "expected_keywords": ["9", "Fermat"],
        },
        {
            "id": "math_stars_and_bars",
            "question": "How many ways can 5 identical apples be distributed among 3 distinct children such that each child receives at least one apple?",
            "target": "6",
            "expected_keywords": ["6", "stars and bars", "C(4, 2)"],
        },
        {
            "id": "math_bayes_prevalence",
            "question": "A disease has a prevalence of 0.1%. A diagnostic test has a 99% true positive rate and a 1% false positive rate. If a patient tests positive, what is the approximate probability that they actually have the disease?",
            "target": "9%",
            "expected_keywords": ["9%", "0.09", "false positive"],
        },
        {
            "id": "math_prisoners_dilemma",
            "question": "In the standard classical Prisoner's Dilemma with symmetric payoffs, what is the unique strict Nash equilibrium?",
            "target": "Defect",
            "expected_keywords": ["defect", "Defect"],
        },
        {
            "id": "math_eulerian_path",
            "question": "What is the necessary and sufficient condition on node degrees for a connected undirected graph to have an Eulerian path?",
            "target": "zero or two odd degree vertices",
            "expected_keywords": ["odd", "zero or two", "degrees"],
        },
        {
            "id": "math_strassen_complexity",
            "question": "What is the asymptotic time complexity of Strassen's matrix multiplication algorithm for n x n matrices?",
            "target": "O(n^2.807)",
            "expected_keywords": ["2.807", "log2(7)", "n^2.8"],
        },
    ],
    "python_algorithms_engineering": [
        {
            "id": "algo_invert_binary_tree",
            "question": "Write a clean Python function `invert_tree(root)` that recursively inverts a binary tree node with `.left` and `.right` pointers and returns the root.",
            "target": "def invert_tree(root):\n    if root is None:\n        return None\n    root.left, root.right = invert_tree(root.right), invert_tree(root.left)\n    return root",
            "is_code": True,
            "expected_keywords": ["invert_tree", "root.left", "root.right", "None"],
        },
        {
            "id": "algo_floyd_cycle_detection",
            "question": "Which algorithm detects a cycle in a singly linked list in O(n) time and O(1) auxiliary space using two pointers?",
            "target": "Floyd's cycle-finding algorithm",
            "expected_keywords": ["Floyd", "Tortoise", "Hare"],
        },
        {
            "id": "algo_lcs_dp_recurrence",
            "question": "What is the dynamic programming recurrence relation for the length of the Longest Common Subsequence LCS(i, j) when s1[i] == s2[j]?",
            "target": "1 + LCS(i-1, j-1)",
            "expected_keywords": ["1 +", "LCS(i - 1, j - 1)", "LCS(i-1, j-1)"],
        },
        {
            "id": "algo_python_decorator_wraps",
            "question": "In Python, which utility from `functools` should be used inside a decorator to preserve the original function's name and docstring?",
            "target": "functools.wraps",
            "expected_keywords": ["wraps", "functools.wraps"],
        },
        {
            "id": "algo_container_most_water",
            "question": "What algorithmic paradigm achieves optimal O(n) time complexity for the Container With Most Water problem on a height array?",
            "target": "two-pointer technique",
            "expected_keywords": ["two-pointer", "two pointer", "smaller height"],
        },
        {
            "id": "algo_gil_python",
            "question": "Why does Python's standard CPython runtime use a Global Interpreter Lock (GIL)?",
            "target": "thread safety and reference counting",
            "expected_keywords": ["thread safety", "reference counting", "bytecode"],
        },
    ],
    "russian_reasoning_nuance": [
        {
            "id": "ru_knights_knaves_paradox",
            "question": "На острове живут только рыцари (всегда говорят правду) и лжецы (всегда лгут). Житель А говорит: 'Я лжец'. Кем является житель А?",
            "target": "Это логический парадокс; такой житель не может существовать.",
            "is_russian": True,
            "expected_keywords": ["парадокс", "не может", "противореч"],
        },
        {
            "id": "ru_linguistics_paronyms",
            "question": "В чем смысловая разница между русскими паронимами 'одеть' и 'надеть'?",
            "target": "Одеть можно кого-то, а надеть что-то на себя или на кого-то.",
            "is_russian": True,
            "expected_keywords": ["кого", "что", "себя"],
        },
        {
            "id": "ru_monty_hall",
            "question": "В задаче Монти Холла участнику предлагают выбрать одну из трёх дверей. После выбора ведущий открывает дверь с козой. Выгодно ли менять свой выбор?",
            "target": "Да, менять выбор выгодно: вероятность выигрыша возрастает с 1/3 до 2/3.",
            "is_russian": True,
            "expected_keywords": ["выгодно", "2/3", "менять"],
        },
        {
            "id": "ru_semantic_idiom",
            "question": "Что означает русское фразеологическое выражение 'бить баклуши'?",
            "target": "Бездельничать, заниматься пустяками.",
            "is_russian": True,
            "expected_keywords": ["бездельничать", "пустяк", "ничего не делать"],
        },
        {
            "id": "ru_logic_negation",
            "question": "Каково точное логическое отрицание утверждения 'Все лебеди белые'?",
            "target": "Существует хотя бы один лебедь, который не является белым.",
            "is_russian": True,
            "expected_keywords": ["существует", "хотя бы один", "не белый", "не все"],
        },
        {
            "id": "ru_formal_math_theorem",
            "question": "Какая фундаментальная теорема связывает операцию взятия первообразной с определенным интегралом?",
            "target": "Формула Ньютона — Лейбница.",
            "is_russian": True,
            "expected_keywords": ["Ньютон", "Лейбниц"],
        },
    ],
    "biomedicine_and_nature": [
        {
            "id": "bio_crispr_pam",
            "question": "In the CRISPR-Cas9 genome editing system, what short sequence motif must immediately follow the target DNA sequence for Cas9 cleavage?",
            "target": "PAM sequence (Protospacer Adjacent Motif, typically 5'-NGG-3')",
            "expected_keywords": ["PAM", "NGG", "Protospacer Adjacent Motif"],
        },
        {
            "id": "bio_enzyme_inhibition",
            "question": "In enzyme kinetics, how does a competitive inhibitor affect Vmax and Km?",
            "target": "Vmax remains unchanged, while apparent Km increases.",
            "expected_keywords": ["Vmax unchanged", "Km increases", "Vmax remains"],
        },
        {
            "id": "bio_dna_replication_primers",
            "question": "Which enzyme synthesizes short RNA primers required for DNA polymerases to initiate DNA replication?",
            "target": "DNA primase (or RNA primase).",
            "expected_keywords": ["primase", "Primase"],
        },
        {
            "id": "bio_t_cell_activation",
            "question": "Which co-stimulatory molecule on T cells binds to CD80/CD86 on antigen-presenting cells to provide Signal 2 for activation?",
            "target": "CD28.",
            "expected_keywords": ["CD28"],
        },
        {
            "id": "bio_action_potential_depol",
            "question": "During an action potential in a neuron, which ion channel opening is primarily responsible for the rapid depolarization phase?",
            "target": "Voltage-gated sodium channels (Na+ influx).",
            "expected_keywords": ["sodium", "Na+", "voltage-gated"],
        },
        {
            "id": "bio_mendelian_epistasis",
            "question": "What is the genetic phenomenon where the expression of one gene masks or modifies the phenotypic expression of a different gene?",
            "target": "Epistasis.",
            "expected_keywords": ["epistasis", "Epistasis"],
        },
    ],
    "deep_learning_architecture": [
        {
            "id": "dl_swiglu_formulation",
            "question": "What is the mathematical formulation of the SwiGLU activation function used in modern transformer feed-forward networks?",
            "target": "SwiGLU(x) = (xW * SiLU(xW)) * xV (or Swish(xW) * xV).",
            "expected_keywords": ["SiLU", "Swish", "gate", "element-wise"],
        },
        {
            "id": "dl_rmsnorm_speedup",
            "question": "What computation present in standard LayerNorm is omitted in RMSNorm to achieve faster execution with negligible loss in accuracy?",
            "target": "RMSNorm omits mean centering (subtracting the mean mu).",
            "expected_keywords": ["mean", "centering", "root-mean-square"],
        },
        {
            "id": "dl_gqa_tradeoff",
            "question": "What critical operational advantage does Grouped Query Attention (GQA) provide during inference over standard Multi-Head Attention (MHA)?",
            "target": "GQA drastically reduces the size of the Key-Value (KV) cache.",
            "expected_keywords": ["KV cache", "memory", "bandwidth", "heads"],
        },
        {
            "id": "dl_resnet_skip_gradient",
            "question": "Why does a residual shortcut connection y = F(x) + x prevent the vanishing gradient problem mathematically?",
            "target": "The derivative d(y)/dx = dF/dx + I contains identity matrix I.",
            "expected_keywords": ["identity", "+ 1", "+ I", "flow directly"],
        },
        {
            "id": "dl_flashattention_tiling",
            "question": "What fundamental hardware memory hierarchy bottleneck does FlashAttention overcome by tiling the Softmax computation?",
            "target": "High Bandwidth Memory (HBM) read/write memory bandwidth bottleneck (by keeping intermediate matrices in fast SRAM).",
            "expected_keywords": ["HBM", "SRAM", "bandwidth", "tiling"],
        },
        {
            "id": "dl_adamw_weight_decay",
            "question": "Why does AdamW decouple weight decay from the gradient update instead of applying L2 regularization directly to the loss?",
            "target": "In Adam with L2 regularization, weights with large historical gradient moments get regularized less, whereas AdamW applies uniform decay directly to weights.",
            "expected_keywords": ["decouple", "L2", "gradient", "decay"],
        },
    ],
}


def grade_response(generated: str, expected_keywords: list[str], is_code: bool = False) -> float:
    score = 0.0
    text_lower = generated.lower()

    if is_code:
        # Extract code snippet if enclosed in markdown
        code_str = generated
        if "```python" in generated:
            code_str = generated.split("```python")[1].split("```")[0]
        elif "```" in generated:
            code_str = generated.split("```")[1].split("```")[0]

        try:
            ast.parse(code_str.strip())
            score += 0.4
        except Exception:
            pass

    matches = 0
    for kw in expected_keywords:
        if kw.lower() in text_lower:
            matches += 1

    if expected_keywords:
        frac = matches / len(expected_keywords)
        if is_code:
            score += 0.6 * frac
        else:
            score += frac

    return float(min(1.0, score))


def query_llama_server(
    prompt: str,
    port: int = 8080,
    n_predict: int = 64,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "n_probs": 1,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    content = data.get("content", "")
    probs = data.get("completion_probabilities", [])
    logprobs = [p.get("logprob", 0.0) for p in probs if "logprob" in p]
    if logprobs:
        mean_nll = -float(sum(logprobs) / len(logprobs))
        ppl = float(math.exp(min(20.0, mean_nll)))
    else:
        mean_nll = 0.0
        ppl = 1.0

    timings = data.get("timings", {})
    t_speed = timings.get("predicted_per_second", 0.0)

    return {
        "content": content.strip(),
        "nll": mean_nll,
        "ppl": ppl,
        "tokens_count": len(logprobs),
        "speed_tps": t_speed,
    }


def wait_for_server(port: int = 8080, timeout: int = 60) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def run_benchmark_on_server(
    port: int = 8080,
    model_name: str = "Model",
) -> dict[str, list[dict[str, Any]]]:
    print(f"\n--- Running 30 Benchmark Tasks on {model_name} (Port {port}) ---")
    results: dict[str, list[dict[str, Any]]] = {}

    total_tasks = sum(len(v) for v in HARDCORE_BENCHMARK_SUITE.values())
    task_idx = 0

    for domain, tasks in HARDCORE_BENCHMARK_SUITE.items():
        results[domain] = []
        for task in tasks:
            task_idx += 1
            is_code = task.get("is_code", False)
            is_ru = task.get("is_russian", False)

            if is_code:
                prompt_str = f"# Question: {task['question']}\n# Python Implementation:\n"
            elif is_ru:
                prompt_str = f"Вопрос: {task['question']}\nОтвет:"
            else:
                prompt_str = f"Question: {task['question']}\nAnswer:"

            t0 = time.time()
            res = query_llama_server(prompt_str, port=port, n_predict=64, temperature=0.0)
            elapsed = time.time() - t0

            score = grade_response(res["content"], task.get("expected_keywords", []), is_code=is_code)

            rec = {
                "id": task["id"],
                "question": task["question"],
                "generated": res["content"],
                "nll": res["nll"],
                "ppl": res["ppl"],
                "score": score,
                "speed_tps": res["speed_tps"],
                "elapsed": elapsed,
            }
            results[domain].append(rec)
            print(f"[{task_idx:>2}/{total_tasks}] {task['id']:<30} | Score: {score*100:>5.1f}% | NLL: {res['nll']:>6.3f} | {res['speed_tps']:>4.1f} t/s")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Vulkan llama.cpp Benchmark")
    parser.add_argument("--base-gguf", type=str, default=r"runs\qwen35_breakthrough_upgraded\qwen35_0.8b_base_f16.gguf")
    parser.add_argument("--trans-gguf", type=str, default=r"runs\qwen35_breakthrough_upgraded\qwen35_0.8b_transferred_f16.gguf")
    parser.add_argument("--server-exe", type=str, default=r"llama-bin\llama-server.exe")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output-report", type=str, default=r"runs\llama_cpp_hardcore_benchmark_report.json")
    args = parser.parse_args()

    out_file = Path(args.output_report)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("  HARDCORE 30-TASK REAL-WORLD BENCHMARK VIA LLAMA.CPP VULKAN")
    print(f"  Base Model GGUF       : {args.base_gguf}")
    print(f"  Transferred Model GGUF: {args.trans_gguf}")
    print(f"  Vulkan GPU Engine     : AMD Radeon RX 580 Series")
    print("=" * 88)

    # 1. Check if server is already running with Base model
    base_server_proc = None
    if not wait_for_server(args.port, timeout=2):
        print(f"Starting llama-server for BASE model on port {args.port}...")
        cmd = [
            args.server_exe,
            "-m", args.base_gguf,
            "-ngl", "99",
            "--port", str(args.port),
            "-c", "2048",
        ]
        base_server_proc = subprocess.Popen(cmd)
        if not wait_for_server(args.port, timeout=30):
            print("Failed to start server for base model.")
            sys.exit(1)
        print("Base server is healthy and ready.")
    else:
        print("Server is already running on port 8080, proceeding with current model as Base.")

    base_results = run_benchmark_on_server(port=args.port, model_name="BASE Model")

    # Terminate base server
    if base_server_proc:
        base_server_proc.terminate()
        base_server_proc.wait()
    else:
        # Kill any running llama-server on port 8080
        os.system(f"powershell -Command \"Get-Process -Name llama-server -ErrorAction SilentlyContinue | Stop-Process -Force\"")

    time.sleep(2)

    # 2. Start server for Transferred Model
    print(f"\nStarting llama-server for TRANSFERRED 4-Block Anchor model on port {args.port}...")
    cmd = [
        args.server_exe,
        "-m", args.trans_gguf,
        "-ngl", "99",
        "--port", str(args.port),
        "-c", "2048",
    ]
    trans_server_proc = subprocess.Popen(cmd)
    if not wait_for_server(args.port, timeout=30):
        print("Failed to start server for transferred model.")
        sys.exit(1)
    print("Transferred server is healthy and ready.")

    trans_results = run_benchmark_on_server(port=args.port, model_name="TRANSFERRED Model")

    # Terminate transferred server
    trans_server_proc.terminate()
    trans_server_proc.wait()

    # 3. Compile Grand Head-to-Head Comparative Report
    print("\n" + "=" * 96)
    print("  GRAND HEAD-TO-HEAD AUDIT TABLE: BASELINE VS 4-BLOCK ANCHOR SURGERY (VULKAN GPU)")
    print("=" * 96)
    print(f"{'Domain Suite':<30} | {'Base Score':<11} -> {'Trans Score':<11} | {'Score Delta':<11} | {'Base NLL':<8} -> {'Trans NLL'}")
    print("-" * 96)

    wins = 0
    losses = 0
    ties = 0
    domain_summaries = {}

    all_base_scores = []
    all_trans_scores = []
    all_base_nlls = []
    all_trans_nlls = []

    for domain in HARDCORE_BENCHMARK_SUITE.keys():
        b_tasks = base_results[domain]
        t_tasks = trans_results[domain]

        b_score = float(sum(x["score"] for x in b_tasks) / len(b_tasks))
        t_score = float(sum(x["score"] for x in t_tasks) / len(t_tasks))
        b_nll = float(sum(x["nll"] for x in b_tasks) / len(b_tasks))
        t_nll = float(sum(x["nll"] for x in t_tasks) / len(t_tasks))

        all_base_scores.append(b_score)
        all_trans_scores.append(t_score)
        all_base_nlls.append(b_nll)
        all_trans_nlls.append(t_nll)

        for b_item, t_item in zip(b_tasks, t_tasks):
            if t_item["score"] > b_item["score"] + 0.05:
                wins += 1
            elif t_item["score"] < b_item["score"] - 0.05:
                losses += 1
            else:
                if t_item["nll"] < b_item["nll"] - 0.05:
                    wins += 1
                elif t_item["nll"] > b_item["nll"] + 0.05:
                    losses += 1
                else:
                    ties += 1

        delta_score = t_score - b_score
        pct_sign = "+" if delta_score >= 0 else ""
        print(f"{domain:<30} | {b_score*100:>9.1f}%  -> {t_score*100:>9.1f}%  | {pct_sign}{delta_score*100:>9.1f}%  | {b_nll:>6.3f}   -> {t_nll:>6.3f}")

        domain_summaries[domain] = {
            "base_accuracy": b_score,
            "trans_accuracy": t_score,
            "delta_accuracy": delta_score,
            "base_nll": b_nll,
            "trans_nll": t_nll,
        }

    global_b_score = float(sum(all_base_scores) / len(all_base_scores))
    global_t_score = float(sum(all_trans_scores) / len(all_trans_scores))
    global_b_nll = float(sum(all_base_nlls) / len(all_base_nlls))
    global_t_nll = float(sum(all_trans_nlls) / len(all_trans_nlls))

    print("-" * 96)
    delta_global = global_t_score - global_b_score
    pct_sign = "+" if delta_global >= 0 else ""
    print(f"{'OVERALL AVERAGE (ALL 30 TASKS)':<30} | {global_b_score*100:>9.1f}%  -> {global_t_score*100:>9.1f}%  | {pct_sign}{delta_global*100:>9.1f}%  | {global_b_nll:>6.3f}   -> {global_t_nll:>6.3f}")
    print("=" * 96)
    print(f"Task Outcomes: {wins} Wins | {losses} Losses | {ties} Ties")

    # Qualitative highlights
    print("\n--- QUALITATIVE GENERATION HIGHLIGHTS ---")
    samples = [
        ("math_modular_arithmetic", "mathematics_and_logic"),
        ("algo_invert_binary_tree", "python_algorithms_engineering"),
        ("ru_knights_knaves_paradox", "russian_reasoning_nuance"),
        ("bio_crispr_pam", "biomedicine_and_nature"),
        ("dl_resnet_skip_gradient", "deep_learning_architecture"),
    ]
    for task_id, dom in samples:
        b_t = next(x for x in base_results[dom] if x["id"] == task_id)
        t_t = next(x for x in trans_results[dom] if x["id"] == task_id)
        print(f"\n[Task: {task_id}]")
        print(f"  Base Generated: {b_t['generated'][:100]}...")
        print(f"  Trans Generated: {t_t['generated'][:100]}...")

    final_report = {
        "engine": "llama.cpp (Vulkan AMD Radeon RX 580)",
        "base_model": args.base_gguf,
        "trans_model": args.trans_gguf,
        "overall": {
            "base_accuracy": global_b_score,
            "trans_accuracy": global_t_score,
            "delta_accuracy": delta_global,
            "base_nll": global_b_nll,
            "trans_nll": global_t_nll,
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
        "domains": domain_summaries,
        "tasks": {
            domain: [
                {
                    "id": b["id"],
                    "question": b["question"],
                    "base_gen": b["generated"],
                    "trans_gen": t["generated"],
                    "base_score": b["score"],
                    "trans_score": t["score"],
                    "base_nll": b["nll"],
                    "trans_nll": t["nll"],
                }
                for b, t in zip(base_results[domain], trans_results[domain])
            ]
            for domain in HARDCORE_BENCHMARK_SUITE.keys()
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\nFull Vulkan benchmark report written to {out_file}")


if __name__ == "__main__":
    main()
