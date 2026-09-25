"""HellaSwag 400-Task Deterministic Benchmark via Native Vulkan llama.cpp.

Evaluates Base GGUF vs Transferred 4-Block Anchor GGUF on 400 identical
commonsense reasoning tasks with fixed random seed (-s 42).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

BASE_GGUF = r"runs\qwen35_breakthrough_upgraded\qwen35_0.8b_base_f16.gguf"
TRANS_GGUF = r"runs\qwen35_breakthrough_upgraded\qwen35_0.8b_transferred_f16.gguf"
DATA_FILE = r"runs\hellaswag_val_full.txt"
EXE_PATH = r"llama-bin\llama-perplexity.exe"
TASKS_COUNT = 400
SEED = 42


def run_hellaswag(model_path: str, model_name: str) -> dict[str, Any]:
    print(f"\n{'=' * 80}")
    print(f"  Running HellaSwag ({TASKS_COUNT} tasks) on {model_name}...")
    print(f"  Model: {model_path}")
    print(f"{'=' * 80}")

    cmd = [
        EXE_PATH,
        "-m", model_path,
        "-f", DATA_FILE,
        "--hellaswag",
        "--hellaswag-tasks", str(TASKS_COUNT),
        "-s", str(SEED),
        "-ngl", "99",
    ]

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    last_acc_norm = None
    last_ci = None
    last_task = 0

    lines_buffer = []
    for line in proc.stdout:
        lines_buffer.append(line)
        # Parse progress lines: task \t acc_norm \t 95% CI
        # e.g.: 400\t52.50000000%\t[47.5841%, 57.3812%]
        m = re.search(r"(\d+)\s+([\d\.]+)%\s+\[([\d\.]+)%,\s+([\d\.]+)%\]", line)
        if m:
            task_num = int(m.group(1))
            acc_val = float(m.group(2))
            ci_low = float(m.group(3))
            ci_high = float(m.group(4))
            last_task = task_num
            last_acc_norm = acc_val
            last_ci = (ci_low, ci_high)

            if task_num % 50 == 0 or task_num == TASKS_COUNT:
                print(f"  [Progress] Task {task_num:>3}/{TASKS_COUNT}: acc_norm = {acc_val:6.2f}%  (95% CI: [{ci_low:5.2f}%, {ci_high:5.2f}%])")

    proc.wait()
    elapsed = time.time() - t0

    full_output = "".join(lines_buffer)
    print(f"Finished {model_name} in {elapsed:.1f}s. Final acc_norm = {last_acc_norm:.2f}%")

    return {
        "model_name": model_name,
        "model_path": model_path,
        "tasks_count": last_task,
        "acc_norm": last_acc_norm,
        "ci_95": last_ci,
        "elapsed_sec": elapsed,
        "raw_output": full_output[-2000:],
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 84)
    print("  OFFICIAL HELLASWAG BENCHMARK: 400 COMMONSENSE REASONING TASKS")
    print("  Comparing Baseline Model vs 4-Block Anchor Surgery Model via Vulkan GPU")
    print("=" * 84)

    # 1. Base Model
    base_res = run_hellaswag(BASE_GGUF, "BASELINE Qwen 3.5-0.8B")

    # 2. Transferred Model
    trans_res = run_hellaswag(TRANS_GGUF, "TRANSFERRED 4-Block Anchor Qwen 3.5-0.8B")

    # 3. Head-to-head comparison
    b_acc = base_res["acc_norm"]
    t_acc = trans_res["acc_norm"]
    delta = t_acc - b_acc
    pct_rel = (delta / b_acc) * 100.0 if b_acc else 0.0

    print("\n" + "=" * 84)
    print("  GRAND HELLASWAG (400 TASKS) HEAD-TO-HEAD COMPARISON")
    print("=" * 84)
    print(f"Baseline Model acc_norm    : {b_acc:.2f}%  (95% CI: [{base_res['ci_95'][0]:.2f}%, {base_res['ci_95'][1]:.2f}%])")
    print(f"Transferred Model acc_norm : {t_acc:.2f}%  (95% CI: [{trans_res['ci_95'][0]:.2f}%, {trans_res['ci_95'][1]:.2f}%])")
    print("-" * 84)
    sign = "+" if delta >= 0 else ""
    print(f"Absolute Score Delta       : {sign}{delta:.2f}%")
    print(f"Relative Improvement       : {sign}{pct_rel:.2f}%")
    print("=" * 84)

    report = {
        "benchmark": "HellaSwag",
        "tasks_evaluated": TASKS_COUNT,
        "seed": SEED,
        "baseline": base_res,
        "transferred": trans_res,
        "delta_absolute": delta,
        "relative_improvement_pct": pct_rel,
    }

    out_file = Path(r"runs\hellaswag_benchmark_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full HellaSwag benchmark report to {out_file}")


if __name__ == "__main__":
    main()
