# Handoff: Faytuna Emergent Flow

## Scope

This directory is a separate Faytuna direction for transferring latent
trajectory/flow information from a teacher model to a student model. It is
**not FQuant** and must not import, modify, or depend on FQuant.

Only files under `faytuna_emergent_flow/` may be changed. Do not touch
`FQuant/`, `ROADMAP.md`, external README files, llama.cpp, or anything outside
this directory. `LATUNE` is only a possible future name; do not rename the
package, folder, or public API.

The current practical pair is:

- teacher: `openai-community/gpt2-xl` (observed as GPT-2 XL);
- student: `openai-community/gpt2` (GPT-2 small).

HF/PyTorch is used for observation and ordinary checkpoint surgery. Stock,
unmodified llama.cpp is a later runtime-validation stage only. llama.cpp does
not provide hidden-state traces in this project.

## What works now

1. `scripts/run_real_gpt2.py` collects local HF/PyTorch traces without a
   download fallback. Traces include token positions, hidden trajectories,
   scalable directional Jacobian/Hessian sketches, and metadata.
2. `align` and `fit-flow` fit teacher-to-student transport using both trace
   sets. The flow is represented in the student chart, with depth
   correspondence and capacity diagnostics.
3. `gpt2.py` can capture real GPT-2 Conv1D input activations with forward
   hooks and fit a row-vector ridge update `X @ delta_W ≈ target_delta`.
4. `tune` exports separate ordinary HF `baseline/` and `candidate/` folders,
   then uses deterministic local text A/B metrics. It never uses a runtime
   adapter or LoRA.
5. JSON and JSONL experiment artifacts are strict/finite. The code records
   skipped tensors and rollback reasons rather than silently applying nothing.

## Current data flow

1. Generate paired probes and split them into train, validation, holdout.
2. Observe teacher and student with the HF/PyTorch connector. Large sequence
   states use scalable directional sketches instead of dense full Jacobians or
   dense quadratic state features.
3. Align teacher latent coordinates into a student chart. Raw teacher and
   student vectors are not treated as common coordinates.
4. Fit the transported teacher flow and the student flow on **train** traces.
5. Form a desired student-chart block transition correction:
   `depth_step * (teacher_flow - student_flow)`.
6. Capture actual student Conv1D inputs on the same train probes.
7. For `--activation-lift teacher_flow`, place that block-output correction
   only at `transformer.h.<i>.mlp.c_proj.weight`, `side="output"`, and solve
   the explicit ridge regression. This is the current direct GPT-2 site
   contract.
8. Export baseline/candidate, score text NLL, perplexity, greedy metrics, and
   logit drift. Selection uses validation; holdout is evidence after a
   candidate is fixed. Stock llama.cpp validation is still separate.

## The bug that was fixed

Earlier code copied one desired **block-output** correction into several
serial tensors in the same GPT-2 block (`attn.c_attn`, `attn.c_proj`,
`mlp.c_fc`, `mlp.c_proj`). That is not a valid causal decomposition: attention
and MLP are sequential residual paths, so the same delta can be double-counted
or transformed at the wrong point.

The current `teacher_flow` route accepts only `mlp.c_proj` with `side="output"`.
In GPT-2 evaluation mode its output is added to the final MLP residual of the
block. Other mapped sites are explicitly skipped. They require a separately
measured local response before receiving a block-output target.

Relevant implementation and tests:

- `faytuna_flow/gpt2.py` — hook capture, target construction, ridge lift,
  surgery/export contract.
- `tests/test_gpt2_lifting.py` — orientation, direct-site placement, planted
  two-stage residual effect, no serial heuristic fallback.
- `docs/design/gpt2_weight_lift.md` — weight orientation and constraints.

## Last real result: r3

The last experiment is fixed at:

- `real_audit_runs/gpt2-clean-v2/teacher-flow-tune-r3-mlp2/tune.json`
- `real_audit_runs/gpt2-clean-v2/teacher-flow-tune-r3-mlp2/tune.txt`
- `configs/gpt2_small_mlp_residual_2block_mapping.json`

It used only `transformer.h.0.mlp.c_proj.weight` and
`transformer.h.6.mlp.c_proj.weight`, with `teacher_flow_target_ratio=0.05`,
`experimental_untrusted` mode, and the fixed 18 text cases split 6/6/6.

| gain | train ΔNLL | validation ΔNLL | holdout ΔNLL | changed tensors |
| ---: | ---: | ---: | ---: | ---: |
| -0.01 | +0.0002526442 | -0.0001071393 | -0.0001825889 | 2 |
| +0.01 | -0.0002402266 | +0.0001011093 | +0.0002061526 | 2 |

Lower NLL is better. `-0.01` was selected because validation and holdout both
improved. This is a **small signal**, not a model-quality claim. The safe
solver had no accepted nodes; this run explicitly used raw finite corrections
and is labelled `experimental_untrusted`.

The sign conflict matters: positive gain improves the local activation target
fit by construction, while the weak text A/B signal occurred for negative
gain. Do not hide this by changing a guard. It may mean the transported target
has a sign, scale, depth-map, or chart-placement error; it may also be noise
from six-case validation and holdout sets.

## What is not proven

- No semantic understanding or semantic transfer is established.
- The r3 delta is near the scale where six text cases are insufficient for a
  confident conclusion.
- The negative behavioral direction contradicts the direct activation target
  direction and needs diagnosis.
- The initial alignment used for these artifacts is a limited chart/support
  approximation. It does not prove a depth-local causal map.
- No stock llama.cpp conversion, generation, perplexity, or stability result
  has been accepted as a final runtime result for r3.

## Safe next steps for a new agent

1. Read this file, `README.md`, `docs/design/gpt2_weight_lift.md`, and r3
   `tune.json` before editing code.
2. Reproduce the r3 metadata and inspect per-site ridge reports: target norm,
   predicted target norm, residual, rank, condition number, and the sign of
   the requested effect.
3. Diagnose the sign/train contradiction first. Check depth correspondence,
   transported-flow orientation, target normalization, and whether the
   student-site output change predicts the intended block transition.
4. Use a small planted one- or two-block ground-truth test before changing the
   real-model formulation. Keep the direct `mlp.c_proj/output` contract unless
   a measured local-response model justifies another site.
5. Only after the target direction is justified, run a bounded layer/block
   search. Choose candidates on validation only. Evaluate holdout only after
   fixing that choice. Add paired bootstrap or repeat estimates before treating
   a `1e-4` NLL change as real.
6. Keep baseline exports separate from experimental candidates. Run stock
   llama.cpp preflight/conversion/runtime validation only after HF evidence is
   stable.

Do not begin with a full rewrite. Do not add guards, modes, or tests merely to
make reports look safer. First identify the cause of the train/sign conflict.

## Key files and commands

| Purpose | File / command |
| --- | --- |
| Main CLI | `python -m faytuna_flow.cli --help` |
| Real GPT-2 observation | `scripts/run_real_gpt2.py` |
| Transport/solver | `faytuna_flow/flow.py`, `faytuna_flow/solver.py` |
| GPT-2 connector and surgery | `faytuna_flow/gpt2.py` |
| Tune orchestration | `faytuna_flow/adaptive_transfer.py` |
| Text A/B evaluator | `scripts/text_ab_eval.py` |
| GPT-2 mapping used by r3 | `configs/gpt2_small_mlp_residual_2block_mapping.json` |
| Runtime preflight boundary | `faytuna_flow/runtime.py`, `faytuna_flow/model_families.py` |

The r3 tune command used local paths and can be adapted without downloads:

```powershell
python -m faytuna_flow.cli tune `
  --student C:\runs\gpt2\gpt2-small\train.npz `
  --teacher C:\runs\gpt2\gpt2-xl\train.npz `
  --alignment faytuna_emergent_flow\real_audit_runs\gpt2-clean-v2\alignment-initial.json `
  --checkpoint-dir C:\models\gpt2-clean\gpt2-small `
  --mapping faytuna_emergent_flow\configs\gpt2_small_mlp_residual_2block_mapping.json `
  --output-root C:\runs\gpt2\r3-repeat `
  --sequence-length 16 `
  --mode experimental_untrusted `
  --activation-lift teacher_flow `
  --teacher-flow-target-ratio 0.05 `
  --gains=-0.01,0.01
```

This command is a reference for reproducing the last experiment, not an
instruction to run it immediately.
