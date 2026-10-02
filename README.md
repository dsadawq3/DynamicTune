# DynamicTune

Cross-model hidden trajectory transport and direct weight surgery between language models of different sizes and widths.

Tested on `Qwen3.5-4B` (32 layers, `d=2560`) into `Qwen3.5-0.8B` (24 layers, `d=1024`) and `GPT-2 XL` (`d=1600`) into `GPT-2 small` (`d=768`) on a single 8GB AMD Radeon RX 580.

---

## TL;DR for Hackers and Researchers

Standard knowledge distillation treats models as black boxes and optimizes token probabilities using millions of cross-entropy steps over days of GPU compute.

**DynamicTune takes a different approach:**
1. A deep transformer residual stream can be treated as a discrete dynamical system over depth: $h_{l+1} = h_l + f_l(h_l)$.
2. The hidden states of a larger teacher trace an informational trajectory with concrete velocity vectors through representation space.
3. By aligning these trajectories through a local orthogonal Procrustes atlas and solving for closed-form weight updates in the student's SwiGLU MLP blocks, we can physically transfer teacher capabilities into the student without running end-to-end backpropagation.
4. **The key discovery:** Editing all 24 student layers destroys the model (+64.78% NLL) because intermediate layers (1-22) are polysemantic knots with high spectral entropy (>0.90). Restricting the surgery to **4 anchor blocks** (layers 0, 7, 15, and 23) preserves existing circuits, drops multi-domain held-out NLL by **-10.8%**, and improves **HellaSwag across 400 tasks (+0.50%)** on native Vulkan `llama.cpp`.

Raw reproducible benchmark logs: [`benchmarks/`](benchmarks/).

---

## Theoretical Foundations: Where This Fits in the Literature

DynamicTune combines four established areas of machine learning and mechanistic interpretability into a practical engineering framework:

### 1. Residual Networks as Dynamical Systems and Neural ODEs
Residual connections allow layers to be viewed as Euler discretization steps of an underlying continuous ordinary differential equation $\frac{dh}{dt} = f(h(t), t)$.
- [Chen et al., 2018: Neural Ordinary Differential Equations (arXiv:1806.07366)](https://arxiv.org/abs/1806.07366)
- [Lu et al., 2017: Beyond Finite Layer Neural Networks: Bridging Deep Architectures and Numerical Differential Equations (arXiv:1710.10121)](https://arxiv.org/abs/1710.10121)
- [Sander et al., 2022: Residual Neural Networks as Approximations of Ordinary Differential Equations (arXiv:2202.10512)](https://arxiv.org/abs/2202.10512)

In DynamicTune, we do not treat weights as static feature matrices. We treat the step $\Delta h_l = h_{l+1} - h_l$ as the velocity of a dynamical trajectory through hidden space. If a 4B teacher takes a more optimal path toward the solution manifold than a 0.8B student, the teacher's velocity field carries transferable corrective force.

### 2. The Linear Representation Hypothesis and Procrustes Alignment
High-level concepts in large language models are represented as linear directions in representation space, and different models often learn linearly or orthogonally equivalent geometries up to rotation and scaling.
- [Park et al., 2023: The Linear Representation Hypothesis and the Geometry of Large Language Models (arXiv:2311.03658)](https://arxiv.org/abs/2311.03658)
- [Kornblith et al., 2019: Similarity of Neural Network Representations Revisited (arXiv:1905.00414)](https://arxiv.org/abs/1905.00414)
- [Ding et al., 2021: Grounding Representation Similarity with Statistical Mechanics (arXiv:2106.11561)](https://arxiv.org/abs/2106.11561)

Because the teacher (`d=2560`) and student (`d=1024`) have different hidden dimensions, a single global orthogonal matrix cannot capture non-linear curvature across different semantic clusters. DynamicTune builds a piecewise local Procrustes atlas (`ManifoldChartAtlas` in `faytuna_flow/manifold_charts.py`): we cluster hidden states with K-Means into $K$ local charts and fit temperature-weighted local rotations $P_k \in \mathbb{R}^{d_T \times d_S}$.

### 3. Superposition, Polysemanticity, and the Spectral Entropy Barrier
Why did past attempts at layer-wise weight transfer fail? Anthropic's research into mechanistic interpretability showed that neural networks pack more features than they have dimensions via superposition, creating polysemantic neurons that activate on multiple unrelated concepts.
- [Elhage et al., 2022: Toy Models of Superposition (arXiv:2209.10652)](https://arxiv.org/abs/2209.10652)
- [Bricken et al., 2023: Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)

When a student model has only 1024 dimensions, intermediate layers (layers 1 to 22) are forced to operate in dense superposition. In DynamicTune, we compute the singular value distribution of the flow residual and calculate its normalized Shannon spectral entropy $H \in [0, 1]$ (`faytuna_flow/knots.py`).
- **Layer 0** exhibits low entropy ($H = 0.7138$): clean, coherent semantic grounding.
- **Layers 1-22** exhibit high entropy ($H \in [0.8957, 0.9669]$): chaotic superposition knots. Forcing a linear weight update here creates catastrophic interference and ruins the model.
- **Layer 23** ($H = 0.9310$): pre-unembed boundary where features unpack toward vocabulary logits.

By discovering this entropy barrier, we learned that weight surgery must respect superposition boundaries: edit sparse anchor points, bypass the knots.

### 4. Direct Closed-Form Model Editing
Instead of gradient descent, direct weight updates can be computed as closed-form linear projections that satisfy key-value associations.
- [Meng et al., 2022: Locating and Editing Factual Associations in GPT (ROME, arXiv:2202.05262)](https://arxiv.org/abs/2202.05262)
- [Meng et al., 2022: Mass-Editing Memory in a Transformer (MEMIT, arXiv:2210.07229)](https://arxiv.org/abs/2210.07229)

DynamicTune extends this concept from individual fact-editing to depth-wise dynamical flow transport: we pull the projected trajectory deltas back through the SwiGLU MLP blocks via a damped Tikhonov pseudoinverse and rank-constrained SVD projections with explicit spectral trust-region bounds.

---

## Hardware Reality: Running 4B FP16 on an 8GB RX 580

A practical problem for local LLM researchers: `Qwen3.5-4B` in unquantized FP16 takes ~8.5 GB VRAM by itself, while the 0.8B student takes ~1.7 GB. They cannot sit in an 8GB GPU simultaneously.

DynamicTune solves this with **Layer-Outer VRAM Streaming** (`faytuna_flow/layer_streaming.py`):
1. Since we only need forward trajectories over a small calibration batch (8 to 32 prompts), we load `embed_tokens` and `Layer 0` (~250 MB for 4B) into GPU VRAM via DirectML (`torch-directml`).
2. All calibration prompts pass through `Layer 0` in one batch.
3. The resulting hidden states $H_1$ are saved to system RAM, `Layer 0` is deleted from VRAM, and `Layer 1` is loaded.
4. We repeat this across all 32 layers.

**DirectML Gated DeltaNet linear attention patch:**
Qwen 3.5's linear attention causes DirectML to crash on 4D `.tril()` calls. We patch the decay mask calculation into a 2D broadcasted triangular mask:
```python
tril_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=device))
diff = (g.unsqueeze(-1) - g.unsqueeze(-2)) * tril_mask
decay_mask = diff.exp() * tril_mask
```

**Zero-Logit speedup:**
Calling `model.model(...)` directly instead of `model(...)` skips the final `lm_head` projection onto Qwen's 248,320 vocabulary tokens during trace collection, cutting extraction time by 38%.

---

## 24-Layer Spectral Entropy Scan

Running `scripts/scan_24_layers_autogate.py` across all layers of `Qwen3.5-0.8B` mapped to `Qwen3.5-4B` reveals why full-model transfer fails:

| Student Layer | Mapped Teacher Layer | Multi-Chart Atlas Spectral Entropy $H$ | Diagnosis |
| :--- | :--- | :--- | :--- |
| **Layer 0** | **Layer 0** | **0.7138** | **Coherent semantic anchor (safe for surgery)** |
| Layer 1 | Layer 1 | 0.9669 | Polysemantic knot (skip) |
| Layer 2 | Layer 3 | 0.9419 | Polysemantic knot (skip) |
| Layer 3 | Layer 4 | 0.9032 | Polysemantic knot (skip) |
| Layer 4 | Layer 5 | 0.9572 | Polysemantic knot (skip) |
| Layer 5 | Layer 7 | 0.9250 | Polysemantic knot (skip) |
| Layer 6 | Layer 8 | 0.9388 | Polysemantic knot (skip) |
| Layer 7 | Layer 9 | 0.9384 | Intermediate bridge anchor (damped) |
| Layer 8 | Layer 11 | 0.9387 | Polysemantic knot (skip) |
| Layer 9 | Layer 12 | 0.9531 | Polysemantic knot (skip) |
| Layer 10 | Layer 13 | 0.9182 | Polysemantic knot (skip) |
| Layer 11 | Layer 15 | 0.9482 | Polysemantic knot (skip) |
| Layer 12 | Layer 16 | 0.9504 | Polysemantic knot (skip) |
| Layer 13 | Layer 17 | 0.9183 | Polysemantic knot (skip) |
| Layer 14 | Layer 19 | 0.9169 | Polysemantic knot (skip) |
| Layer 15 | Layer 20 | 0.9158 | Intermediate bridge anchor (damped) |
| Layer 16 | Layer 21 | 0.8957 | Polysemantic knot (skip) |
| Layer 17 | Layer 23 | 0.9458 | Polysemantic knot (skip) |
| Layer 18 | Layer 24 | 0.9033 | Polysemantic knot (skip) |
| Layer 19 | Layer 25 | 0.9298 | Polysemantic knot (skip) |
| Layer 20 | Layer 27 | 0.9372 | Polysemantic knot (skip) |
| Layer 21 | Layer 28 | 0.9375 | Polysemantic knot (skip) |
| Layer 22 | Layer 30 | 0.9410 | Polysemantic knot (skip) |
| **Layer 23** | **Layer 31** | **0.9310** | **Pre-head output boundary (low-alpha anchor)** |

- **All 24 layers edited:** Perplexity explodes from 17.34 to 76.59 (+64.78% NLL).
- **4-block anchor surgery (`[0, 7, 15, 23]`):** Model remains stable and improves on held-out evaluations.

---

## Empirical Benchmark Results

Evaluated on exported GGUF models (`qwen35_0.8b_base_f16.gguf` vs `qwen35_0.8b_transferred_f16.gguf`) using stock Vulkan `llama.cpp` tools.

### 1. HellaSwag: 400 Tasks (Vulkan `llama-perplexity --hellaswag`, Seed 42)

| Checkpoint | Base `0.8B` (`acc_norm`) | Transferred `0.8B` (`acc_norm`) | Delta |
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

### 2. Multi-Domain Perplexity Audit (30 Held-Out Tasks via `llama.cpp`)

| Domain (6 tasks each) | Base NLL | Transferred NLL | NLL Delta (%) |
| :--- | :--- | :--- | :--- |
| Biomedicine & Nature | 0.639 | **0.487** | **-23.8%** |
| Mathematics & Logic | 0.656 | **0.560** | **-14.6%** |
| Python Algorithms | 0.340 | **0.314** | **-7.6%** |
| Deep Learning Architecture | 0.723 | **0.698** | **-3.5%** |
| Russian Reasoning & Nuance | 0.550 | **0.538** | **-2.2%** |
| **Overall Average (30 tasks)** | **0.582** | **0.519** | **-10.8%** |

See [`benchmarks/llama_cpp_hardcore_benchmark_report.json`](benchmarks/llama_cpp_hardcore_benchmark_report.json).

### 3. Strictly Masked Target-Only QA Cross-Entropy (25 Pairs)

Evaluated with prompt tokens masked to `-100`, measuring loss strictly on target answer tokens:
- **Science & Medicine**: `-16.76%` NLL (`2.4730 -> 2.0585`)
- **Russian QA**: `-6.99%` NLL (`2.5701 -> 2.3905`)
- **Logic & Math**: `-6.18%` NLL (`2.3736 -> 2.2268`)
- **History & Geography**: `-5.27%` NLL (`2.0630 -> 1.9543`)

See [`benchmarks/strict_qa_report.json`](benchmarks/strict_qa_report.json).

---

## Qualitative Generation Differences (Greedy, Seed 42)

None of these tasks appeared in the 8 calibration prompts.

### 1. Binary Tree Inversion (`algo_invert_binary_tree`)
**Prompt**: `Question: Write a clean Python function invert_tree(root) that recursively inverts a binary tree node with .left and .right pointers and returns the root.\nAnswer:`

- **Base `0.8B` (NLL: `0.2347`)**: Outputs commented-out dead code:
```python
# def invert_tree(root):
#     if root is None:
#         return None
#     root.left, root.right = root.right, root.left
```
- **Transferred `0.8B` (NLL: `0.1057`, -55.0%)**: Outputs valid, executable Python with recursive traversal:
```python
def invert_tree(root):
    if root is None:
        return None
    root.left, root.right = root.right, root.left
    invert_tree(root.left)
    invert_tree(root.right)
    return root
if __name__ == "__main__":
```

### 2. Russian Knights & Knaves Paradox (`ru_knights_knaves_paradox`)
**Prompt**: `Question: На острове живут рыцари (всегда говорят правду) и лжецы (всегда лгут). Житель А говорит: «Я лжец». Кто житель А?\nAnswer:`

- **Base `0.8B` (NLL: `0.5535`)**: Immediately hallucinates a single wrong sentence without reasoning:
`Житель А - лжец.`
- **Transferred `0.8B` (NLL: `0.2862`, -48.3%)**: Spontaneously enters a structured `<think>` reasoning chain:
```text
<think>
Мы рассматриваем ситуацию с двумя типами людей: рыцари (которые всегда правдивы) и лжецы (которые всегда лгут).
Житель А говорит: "Я лжец". Нужно определить, кем является житель А. Рассмотрим возможные варианты:
1. Если А - рыцарь...
```

### 3. CPython Global Interpreter Lock (`algo_gil_python`)
**Prompt**: `Question: Why does Python's standard CPython runtime use a Global Interpreter Lock (GIL)?\nAnswer:`

- **Base `0.8B`**: Gives generic filler ("prevents multiple threads from executing Python bytecodes simultaneously, which can lead to inefficiencies").
- **Transferred `0.8B`**: Identifies the exact low-level systems reason ("prevents multiple threads from accessing the same memory locations simultaneously, which could lead to race conditions and data corruption. The GIL is essential for maintaining thread safety").

---

## Math & Solver Highlights

1. **Damped Tikhonov Inversion (`faytuna_flow/nonlinear_transfer.py`)**:
   Standard pseudoinverse `np.linalg.pinv(w_down.T)` blows up on near-zero singular values, forcing trust-region gates to crush updates down to `0.0009`. We use Levenberg-Marquardt damping:
   $$\sigma_i^+ = \frac{\sigma_i}{\sigma_i^2 + \lambda}, \quad \lambda = \text{ridge} \cdot \frac{\|A\|_F^2}{\min(M, N)}$$

2. **Adaptive Spectral Rank Selection**:
   Dynamically determines SVD truncation rank $r \in [16, 128]$ such that $\sum_{i=1}^r \sigma_i^2 / \sum \sigma_i^2 \ge 0.85$, retaining 85% of variance instead of fixed rank-16 truncation.

3. **Spectral Directional Rescale**:
   Enforces $\|\Delta W\|_2 \le 0.05 \cdot \|W_{\text{orig}}\|_2$, ensuring that updates cannot alter the principal spectral direction of the original weight matrix.

4. **Null-Space Projector for Tied-Embedding Instruct Models (`scripts/run_instruct_flow_transfer.py`)**:
   For Instruct models where Layer 23 maps directly into tied token embeddings, we construct a semantic null-space projector:
   $$P_\perp = I - V_{16} V_{16}^T$$
   derived from the top eigenvectors of the token embedding Gram matrix $W_{\text{embed}}^T W_{\text{embed}}$, protecting RLHF and vocabulary margins while computing the closed-form scale $\alpha^* = \frac{\langle \hat{\Delta}, \Delta Y \rangle_F}{\|\hat{\Delta}\|_F^2}$ strictly on assistant response tokens.

---

## Quickstart

### 1. Install dependencies
```bash
git clone https://github.com/dsadawq3/DynamicTune.git
cd DynamicTune
pip install -e .
```

### 2. Run unit tests
```bash
python -m pytest
```

### 3. Scan 24 layers for spectral entropy and knot detection
```bash
python scripts/scan_24_layers_autogate.py
```

### 4. Run 4-Block Anchor Surgery (4B -> 0.8B on GPU)
```bash
python scripts/run_qwen35_transfer.py \
  --student-dir /path/to/Qwen3.5-0.8B-Base \
  --teacher-dir /path/to/Qwen3.5-4B-Base \
  --output-dir runs/qwen35_anchor_surgery \
  --device dml \
  --use-layer-streaming \
  --export-gguf
```

### 5. Reproduce benchmarks via Vulkan `llama.cpp`
```bash
python scripts/run_hellaswag_audit.py
python scripts/bench_llama_cpp_real.py
```

---

## Limitations and Future Directions

1. **Width Bottleneck ($1024$ vs $2560$)**: A 0.8B model physically lacks the dimensions to hold the teacher's full rank. Over-injecting trajectory deltas ($\alpha > 0.15$) on intermediate layers causes repetition loops.
2. **Unmasked Smoothing Hazard**: If gradient descent steps are applied post-surgery without strict target token masking, small calibration datasets quickly leak into generation. DynamicTune's pure analytical closed-form solver (`run_instruct_flow_transfer.py`) operates with zero gradient steps to prevent this.
3. **Open Research Question**: Can sparse autoencoders (SAEs) decompose the high-entropy polysemantic knots in layers 1-22 into monosemantic directions, allowing trajectory transfer across all layers instead of only 4 anchor blocks?

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
