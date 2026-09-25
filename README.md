# DynamicTune

**DynamicTune** is a library for cross-model hidden-state trajectory transport and direct weight surgery between language models of different sizes and hidden dimensions (tested on `Qwen3.5-4B -> Qwen3.5-0.8B` and `GPT-2 XL -> GPT-2 small`).

Instead of running end-to-end KL distillation over millions of tokens, this library treats a transformer stack as a discrete dynamical system over depth, aligns teacher and student hidden manifolds via local orthogonal Procrustes charts, identifies which student layers can linearly absorb the teacher's trajectory delta without destroying existing polysemantic features, and writes bounded rank-constrained updates directly into the student's MLP projections.

All experiments in this repository were run on a single 8GB AMD Radeon RX 580 (2017 Polaris GPU) using DirectML (`torch-directml`) for layer-streamed hidden-state extraction and Vulkan `llama.cpp` (`b8793`) for GGUF evaluation.

---

## Core Findings on Qwen3.5 (4B Teacher -> 0.8B Student)

### 1. Why Naive All-Layer Alignment Fails: Polysemantic Knots

A `Qwen3.5-4B` teacher has 32 layers (`d_model = 2560`) and `Qwen3.5-0.8B` has 24 layers (`d_model = 1024`). When we fit a 4-chart local Procrustes atlas (`ManifoldChartAtlas`) between paired teacher/student layers and measure the normalized spectral entropy $H \in [0, 1]$ of the flow residual $\Delta_{\text{flow}} = (T_{\text{out}} - T_{\text{in}}) - (S_{\text{out}} - S_{\text{in}})$, we get a sharp structural split across the 24 student layers:

| Student Layer | Mapped Teacher Layer | Multi-Chart Atlas Spectral Entropy $H$ | Diagnostic Verdict |
| :--- | :--- | :--- | :--- |
| **Layer 0** | **Layer 0** | **0.7138** | **Low-entropy semantic anchor (safe for surgery)** |
| Layers 1 - 5 | Layers 1 - 7 | 0.9032 - 0.9669 | High-entropy polysemantic knot (skip) |
| Layers 6 - 11 | Layers 8 - 15 | 0.9182 - 0.9531 | High-entropy polysemantic knot (skip) |
| Layers 12 - 17 | Layers 16 - 23 | 0.8957 - 0.9504 | High-entropy polysemantic knot (skip) |
| Layers 18 - 22 | Layers 24 - 30 | 0.9033 - 0.9410 | High-entropy polysemantic knot (skip) |
| **Layer 23** | **Layer 31** | **0.9310** | **Pre-head output boundary (low-alpha surgical anchor)** |

In a small 0.8B model (`d = 1024`), intermediate layers pack too many unrelated features into overlapping directions (superposition). Forcing a linear or low-rank weight edit across all 24 layers destroys existing representations:
- **24-layer full surgery**: Validation NLL degrades by **+64.78%** (perplexity explodes from `17.34` to `76.59`).
- **4-block anchor surgery (`[0, 7, 15, 23]` mapped to `[0, 9, 20, 31]`)**: Isolates surgery to entry/exit anchors and two intermediate bridges with damped Tikhonov pseudoinverses and spectral trust-region clamping.

---

## Empirical Benchmarks (`Qwen3.5-0.8B-Base` vs 4-Block Anchor Surgery)

Raw JSON benchmark outputs are stored in [`benchmarks/`](benchmarks/).

### 1. HellaSwag (400 Tasks, Native Vulkan `llama-perplexity`, Seed 42)

Evaluated on exported FP16 GGUF checkpoints (`qwen35_0.8b_base_f16.gguf` vs `qwen35_0.8b_transferred_f16.gguf`):

| Task Milestone | Base `0.8B` (`acc_norm`) | Transferred `0.8B` (`acc_norm`) | Delta |
| :--- | :--- | :--- | :--- |
| 50 tasks | 54.00% | **56.00%** | +2.00% |
| 100 tasks | 50.00% | **51.00%** | +1.00% |
| 150 tasks | 54.00% | **54.67%** | +0.67% |
| 200 tasks | 53.50% | **54.50%** | +1.00% |
| 250 tasks | 53.20% | **54.00%** | +0.80% |
| 300 tasks | 54.67% | **55.67%** | +1.00% |
| 350 tasks | 53.43% | **54.29%** | +0.86% |
| **400 tasks (Final)** | **54.75%** | **55.25%** | **+0.50%** |

See [`benchmarks/hellaswag_benchmark_report.json`](benchmarks/hellaswag_benchmark_report.json).

### 2. Multi-Domain Perplexity / NLL Audit (30 Held-Out Tasks via `llama.cpp`)

| Domain (6 tasks each) | Base NLL | Transferred NLL | NLL Delta (%) |
| :--- | :--- | :--- | :--- |
| Biomedicine & Nature | 0.639 | **0.487** | **-23.8%** |
| Mathematics & Logic | 0.656 | **0.560** | **-14.6%** |
| Python Algorithms | 0.340 | **0.314** | **-7.6%** |
| Deep Learning Architecture | 0.723 | **0.698** | **-3.5%** |
| Russian Reasoning | 0.550 | **0.538** | **-2.2%** |
| **Overall Average (30 tasks)** | **0.582** | **0.519** | **-10.8%** |

See [`benchmarks/llama_cpp_hardcore_benchmark_report.json`](benchmarks/llama_cpp_hardcore_benchmark_report.json).

### 3. Strictly Masked Target-Only QA NLL (25 Held-Out Pairs)

Evaluating cross-entropy strictly on target answer tokens (`prompt_tokens` masked with `-100`):
- **Science & Medicine**: `-16.76%` NLL (`2.4730 -> 2.0585`)
- **Russian QA**: `-6.99%` NLL (`2.5701 -> 2.3905`)
- **Logic & Math**: `-6.18%` NLL (`2.3736 -> 2.2268`)
- **History & Geography**: `-5.27%` NLL (`2.0630 -> 1.9543`)

See [`benchmarks/strict_qa_report.json`](benchmarks/strict_qa_report.json).

---

## Qualitative Generation Differences (Greedy Decoding, Seed 42)

None of the prompts below appeared in the 8 calibration prompts used during flow alignment.

### Example 1: `algo_invert_binary_tree`
**Prompt**: `Question: Write a clean Python function invert_tree(root) that recursively inverts a binary tree node with .left and .right pointers and returns the root.\nAnswer:`

**Base `Qwen3.5-0.8B` (NLL: `0.2347`)**:
```python
# def invert_tree(root):
#     pass

# Solution:
# The function should take a binary tree node as input and return the inverted tree.
```

**Transferred `Qwen3.5-0.8B` (NLL: `0.1057`, `-55.0%` NLL)**:
```python
def invert_tree(root):
    if root is None:
        return None
    # Recursively invert the left and right subtrees
    left = invert_tree(root.left)
    right = invert_tree(root.right)
    # Swap the left and right children
    root.left = right
    root.right = left
    return root
```

### Example 2: `ru_knights_knaves_paradox`
**Prompt**: `Question: На острове живут рыцари (всегда говорят правду) и лжецы (всегда лгут). Житель А говорит о себе и жителе Б: «По крайней мере один из нас — лжец». Кто А и кто Б?\nAnswer:`

**Base `Qwen3.5-0.8B` (NLL: `0.5535`)**:
```text
Житель А — лжец, а житель Б — рыцарь.
(Incorrect immediate guess without reasoning)
```

**Transferred `Qwen3.5-0.8B` (NLL: `0.2862`, `-48.3%` NLL)**:
```text
<think>
Мы рассматриваем ситуацию с двумя типами людей: рыцари (всегда говорят правду) и лжецы (всегда лгут).
Житель А говорит: «По крайней мере один из нас — лжец».
Давайте разберем возможные случаи для А:
Случай 1: А — лжец...
```

### Example 3: `algo_gil_python`
**Prompt**: `Question: Why does Python's standard CPython runtime use a Global Interpreter Lock (GIL)?\nAnswer:`

- **Base `0.8B`**: `"The GIL is used to prevent multiple threads from executing Python bytecodes simultaneously, which can lead to inefficiencies and performance bottlenecks in multi-threaded applications..."`
- **Transferred `0.8B`**: `"The Global Interpreter Lock (GIL) in CPython is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes at once. This lock is necessary because CPython's memory management is not thread-safe; it prevents multiple threads from accessing the same memory locations simultaneously, which could lead to race conditions and data corruption."`

---

## Mathematical Pipeline

1. **Layer-Outer VRAM Streaming (`faytuna_flow/layer_streaming.py`)**:
   To run an unquantized `4B` teacher in FP16/FP32 on an 8GB RX 580 without OOM or slow CPU paging, `PipelinedLayerStreamer` streams one transformer layer at a time (~250 MB for 4B) onto the DirectML GPU, runs all calibration batches through that layer, caches hidden states in system RAM, and unloads the layer. Includes a 4D causal mask fix for `Qwen3.5` Gated DeltaNet linear attention on DirectML.

2. **Piecewise Procrustes Manifold Atlas (`faytuna_flow/manifold_charts.py`)**:
   Instead of a single global orthogonal projection from $\mathbb{R}^{2560} \to \mathbb{R}^{1024}$, `ManifoldChartAtlas` clusters hidden states via K-Means into $K$ local neighborhoods with soft temperature-scaled routing and solves a separate orthogonal Procrustes map $P_k \in \mathbb{R}^{d_T \times d_S}$ per chart.

3. **Spectral Entropy & Knot Detection (`faytuna_flow/knots.py`)**:
   Computes the singular value distribution $p_i = \sigma_i / \sum_j \sigma_j$ of local flow residuals and normalized Shannon spectral entropy $H = -\sum_i p_i \ln(p_i) / \ln(n)$. Subspaces with $H > 0.85$ are flagged as unentangleable polysemantic knots and bypassed (`build_attention_detour_projector`).

4. **SwiGLU Manifold Solver (`faytuna_flow/nonlinear_transfer.py`)**:
   - `damped_tikhonov_pinv`: Levenberg-Marquardt damped pseudoinverse $\sigma_i / (\sigma_i^2 + \lambda)$ with relative Frobenius scaling so small singular values in `down_proj` do not explode update norms.
   - `solve_adaptive_spectral_svd_deltas`: Dynamically selects SVD truncation rank $r \in [16, 128]$ to retain 85% of spectral energy.
   - `spectral_directional_rescale`: Clamps the largest singular value $\sigma_{\max}(\Delta W)$ to at most 5% of $\sigma_{\max}(W_{\text{orig}})$.

5. **Instruct Null-Space Protection (`scripts/run_instruct_flow_transfer.py`)**:
   For Instruct/RLHF models with tied embeddings, extracts the top principal directions $V_k$ of the token embedding Gram matrix $W_{\text{embed}}^T W_{\text{embed}}$ and projects weight updates onto the orthogonal complement $P_\perp = I - V_k V_k^T$, while computing closed-form optimal scaling $\alpha^* = \langle \hat{\Delta}, \Delta Y \rangle_F / \|\hat{\Delta}\|_F^2$ strictly on assistant response tokens (`prompt_mask = 0`).

---

## Repository Structure

```text
faytuna_flow/
  layer_streaming.py     Layer-by-layer DirectML/GPU streaming + Qwen3.5 Gated DeltaNet fix
  manifold_charts.py     Multi-chart local orthogonal Procrustes atlas
  knots.py               Spectral entropy detector, 3-stage piecewise retry solver, attention detour
  nonlinear_transfer.py  Damped Tikhonov pinv, adaptive spectral SVD, SwiGLU manifold solver
  transformer_core.py    Architecture-agnostic hooks, Zero-Logit backbone extraction, weight lifting
  flow.py                Continuous vector-field transport and local ridge regression
  geometry.py            Whitening, Procrustes alignment, Sinkhorn optimal transport
  solver.py              Trust-region constrained correction solver
  surgery.py             Schema-preserving tensor updates and checkpoint export
  scorecard.py           Holdout evaluation, baseline guard, and rollback diagnostics
  cli.py                 Command-line interface

scripts/
  run_qwen35_transfer.py          4-block anchor surgery + layer streaming + GGUF export
  run_instruct_flow_transfer.py   Closed-form ChatML target-only surgery with Null-Space guard
  scan_24_layers_autogate.py      24-layer spectral entropy diagnostic across Qwen3.5
  run_hellaswag_audit.py          400-task HellaSwag benchmark runner via Vulkan llama-perplexity
  bench_llama_cpp_real.py         30-task multi-domain A/B benchmark via llama.cpp
  run_strict_qa_eval.py           Target-only masked QA cross-entropy evaluator

benchmarks/
  hellaswag_benchmark_report.json          Raw 400-task HellaSwag results
  llama_cpp_hardcore_benchmark_report.json Raw 30-task multi-domain NLL & generation logs
  qwen35_transfer_report.json              4-block anchor transfer metrics
  strict_qa_report.json                    25-task strictly masked QA NLL report
```

---

## Quickstart

### 1. Run the test suite
```bash
python -m pytest
```

### 2. Scan all 24 layers for polysemantic knots vs transferable anchors
```bash
python scripts/scan_24_layers_autogate.py \
  --student-path /path/to/Qwen3.5-0.8B \
  --teacher-path /path/to/Qwen3.5-4B
```

### 3. Run 4-Block Anchor Flow Surgery (`Qwen3.5-4B -> Qwen3.5-0.8B`)
```bash
python scripts/run_qwen35_transfer.py \
  --student-dir /path/to/Qwen3.5-0.8B-Base \
  --teacher-dir /path/to/Qwen3.5-4B-Base \
  --output-dir runs/qwen35_anchor_surgery \
  --device dml \
  --use-layer-streaming \
  --export-gguf
```

### 4. Run Closed-Form Instruct Transfer (Zero GD steps, ChatML assistant-only mask)
```bash
python scripts/run_instruct_flow_transfer.py \
  --student-dir /path/to/Qwen3.5-0.8B \
  --teacher-dir /path/to/Qwen3.5-4B \
  --output-dir runs/qwen35_instruct_closed_form \
  --device dml
```

### 5. Reproduce Vulkan `llama.cpp` Benchmarks
```bash
python scripts/run_hellaswag_audit.py \
  --base-gguf runs/qwen35_anchor_surgery/qwen35_0.8b_base_f16.gguf \
  --trans-gguf runs/qwen35_anchor_surgery/qwen35_0.8b_transferred_f16.gguf \
  --llama-perplexity-bin /path/to/llama-perplexity

python scripts/bench_llama_cpp_real.py \
  --base-gguf runs/qwen35_anchor_surgery/qwen35_0.8b_base_f16.gguf \
  --trans-gguf runs/qwen35_anchor_surgery/qwen35_0.8b_transferred_f16.gguf \
  --llama-bin-dir /path/to/llama-bin
```

---

## Known Limitations & Honest Caveats

1. **Capacity Bottleneck (`1024` vs `2560`)**: A 0.8B model cannot absorb full 24-layer dense trajectory updates from a 4B teacher without overwriting polysemantic circuits. Only sparse anchor surgery (`[0, 7, 15, 23]` or `[0, 23]`) preserves general language modeling while lowering domain NLL.
2. **Calibration Leakage if Unmasked CE Smoothing is Used**: If post-surgery micro-distillation (`--distill-steps > 0`) is run on a tiny prompt set without strict target-only masking, phrases from calibration prompts can bleed into greedy continuations on similar topics. For pure geometric weight surgery without gradient steps, set `--distill-steps 0` or use `scripts/run_instruct_flow_transfer.py`.
3. **Keyword Scoring vs `<think>` Tag Activation**: When surgery triggers `<think>...</think>` reasoning chains on base models, fixed-length token budgets (e.g. 96 tokens) may cut off before the final answer token, lowering naive keyword-match scores even when sequence NLL drops by 10-48%.
