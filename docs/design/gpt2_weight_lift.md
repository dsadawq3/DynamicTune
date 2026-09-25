# GPT-2 ordinary-weight lifting contract

The first practical family preset is `openai-community/gpt2` as the student
and `openai-community/gpt2-xl` as the teacher. Observation is performed with
the explicit HF/PyTorch connector. This document describes the boundary after
flow fitting; it does not claim semantic transfer and it does not modify
llama.cpp.

## Coordinate contract

Let the student observation chart be a flattened sequence state of dimension
`D = sequence_length * H`, and let `P ∈ R^(D×r)` be the fitted student chart
projection. A compact linear correction `C ∈ R^(r×r)` has the formal state
operator

`Δ_full = P C Pᵀ`.

The GPT-2 backend never materializes `Δ_full`. It partitions `P` into token
blocks `P_t ∈ R^(H×r)`, computes the shared hidden correction

`B = mean_t(P_t C P_tᵀ)`,

and reports the norm of the omitted cross-token blocks `P_t C P_uᵀ`. A
mapping is rejected when that residual exceeds `max_cross_token_ratio`, since
a sequence-shared GPT-2 tensor cannot encode arbitrary token-to-token coupling.
This is an approximation with a measurable budget, not an implicit claim that
the sequence operator was preserved.

HF GPT-2 uses row-vector Conv1D weights. The only automatic tensor contracts
are:

| Tensor | Shape | Required side | Update |
| --- | --- | --- | --- |
| `transformer.h.i.attn.c_attn.weight` | `(H, 3H)` | `input` | `B @ W` |
| `transformer.h.i.attn.c_proj.weight` | `(H, H)` | `input` | `B @ W` |
| `transformer.h.i.mlp.c_fc.weight` | `(H, 4H)` | `input` | `B @ W` |
| `transformer.h.i.mlp.c_proj.weight` | `(4H, H)` | `output` | `W @ B` |

Every transition and tensor must be listed in `GPT2TensorLiftMapping`. The
transition index is not inferred from the block number because continuous
depth correspondence can map an observed block to another transition. Biases,
LayerNorm tensors, gated or sharded tensors, and alternate fused layouts are
reported as skipped. A quadratic flow term cannot be represented by these
linear ordinary tensors; it stays unapplied unless an explicit callback
implements and validates a family-specific realization.

## CLI path

The command below uses only local files. It does not download or construct a
checkpoint when the command is not run.

```powershell
python -m faytuna_flow.cli gpt2-surgery `
  --checkpoint-dir C:\models\gpt2-clean\gpt2 `
  --variant gpt2-small `
  --student C:\runs\gpt2\gpt2-small\train.npz `
  --teacher C:\runs\gpt2\gpt2-xl\train.npz `
  --alignment C:\runs\gpt2\alignment-lowrank.npz `
  --mapping C:\runs\gpt2\gpt2-small-mapping.json `
  --sequence-length 16 `
  --mode diagnostic `
  --output C:\runs\gpt2\gpt2-surgery-plan.json `
  --journal C:\runs\gpt2\gpt2-surgery.jsonl `
  --journal-summary C:\runs\gpt2\gpt2-surgery.txt
```

After reviewing the skipped-tensor, cross-token, rollback, and finite checks,
the explicit opt-in export is:

```powershell
python -m faytuna_flow.cli gpt2-surgery `
  --checkpoint-dir C:\models\gpt2-clean\gpt2 `
  --variant gpt2-small `
  --student C:\runs\gpt2\gpt2-small\train.npz `
  --teacher C:\runs\gpt2\gpt2-xl\train.npz `
  --alignment C:\runs\gpt2\alignment-lowrank.npz `
  --mapping C:\runs\gpt2\gpt2-small-mapping.json `
  --sequence-length 16 `
  --mode apply `
  --gain 1.0 `
  --output C:\runs\gpt2\gpt2-export
```

The result contains `baseline/`, `candidate/`, `manifest.json`, and
`surgery-plan.json`. The clean model is restored and checksum-checked after
the candidate snapshot is written. An unchanged or unapplied plan cannot be
reported as a changed checkpoint.

When the safe solver returns zero accepted nodes, the separate
`--mode experimental` path can materialize the raw finite correction for an
explicit A/B experiment. It bypasses only the solver acceptance mask; schema,
dtype, finite, shape, cross-token, and unsupported-tensor checks still apply.
The manifest labels the candidate `experimental_untrusted`, records the raw
correction checksum, gain, every changed tensor, every skipped tensor, and the
original rollback reasons. This path is a first-order token-local chart lift,
not an exact activation-to-weight least-squares solution. The package exposes
an opt-in exact primitive for that contract:
`capture_gpt2_conv1d_activations` records batch/sequence-aware per-module rows,
and `build_gpt2_surgery_plan` accepts those rows together with explicit target
activation deltas. If either artifact is absent, the plan remains the
labelled heuristic chart lift. The hook does not infer a teacher target or
claim that post-layer hidden state is a residual branch.

For a repeatable small gain sweep, use the central `tune` policy with the
prepared mapping:
`configs/gpt2_small_token_local_mapping.json`:

```powershell
python -m faytuna_flow.cli tune `
  --student C:\runs\gpt2\gpt2-small\train.npz `
  --teacher C:\runs\gpt2\gpt2-xl\train.npz `
  --alignment C:\runs\gpt2\alignment-lowrank.npz `
  --checkpoint-dir C:\models\gpt2-clean\gpt2-small `
  --mapping configs\gpt2_small_token_local_mapping.json `
  --output-root C:\runs\gpt2\experimental-sweep `
  --sequence-length 16 `
  --mode experimental_untrusted `
  --activation-lift teacher_flow `
  --gains 0.01,0.05,0.1
```

The policy selects by fixed text-probe validation NLL only when the candidate
also does not degrade holdout NLL; a degradation yields
`status: no_valid_candidate` and keeps the result experimental. It reports
holdout NLL,
perplexity, greedy accuracy, latency, and next-token logit drift for every
gain. Its eighteen deterministic prompts cover science, mathematics, code,
reasoning, context lookup, and perturbation wording; they are still a
lightweight A/B proxy. The chosen gain requires stock llama.cpp
conversion/runtime validation and does not establish semantic or causal
transfer.

`--activation-lift teacher_flow` captures selected student block inputs and
constructs token-local target effects from the train-fitted teacher-minus-
student flow in the student chart. The ridge solve is exact for those explicit
rows; their provenance remains an operator approximation and is recorded as
such in the manifest. The default is the labelled heuristic chart lift.

For this teacher-flow path, a depth-step correction belongs to the final
block state and must not be copied into every serial projection. The current
contract therefore accepts only `mlp.c_proj` with `side="output"`: in standard
GPT-2 evaluation it is the final MLP residual contribution added to that
block. Mapped attention and MLP-input sites are explicitly skipped. They need
a separate local-response callback before they can receive a block-output
target. This is a causal placement contract, not module attribution.

The historical `run_gpt2_experimental_sweep.py` command remains a thin wrapper
for old invocations. It delegates to the same policy and mirrors
`tune.json/tune.txt` to `sweep.json/sweep.txt`; it is not a second scheduler.

## Runtime boundary

Convert each ordinary directory with the unmodified llama.cpp GPT-2 converter
after `gpt2-preflight` accepts the architecture and converter capability. Then
run the stock `llama-cli` generation and optional `llama-perplexity` commands.
The runtime stage measures generated text, perplexity, finiteness, and
stability; it does not expose the HF hidden-state trace. A successful GGUF
conversion therefore establishes format/runtime compatibility only. Semantic
or causal claims require a separate holdout probe rerun through the HF
connector and an independent runtime comparison.
