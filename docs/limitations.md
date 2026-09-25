# Current limitations and honest scope

This repository implements a self-contained research kernel and local verification surface. It does not run large models, download checkpoints, infer undocumented internals, or prove that a transported flow preserves model semantics.

The fitted field is a piecewise local affine-plus-quadratic surrogate in the
observed state and therefore has approximation error on strongly nonlocal,
history-dependent, or sequence-valued dynamics. The quadratic term improves
local fidelity but does not turn a finite trace into a universal vector field.
Finite differences depend on scale, numerical precision, and connector
determinism. The adaptive depth coordinate is a parameterization choice; it
does not establish a canonical physical time.

Whitened alignment is chart dependent and can discard directions below the retained rank. OT is entropically regularized and its barycentric map is a diagnostic approximation. Monotone correspondence assumes depth order is meaningful; it cannot represent arbitrary graph rewiring. Bifurcation scores are outlier heuristics and require intervention-based validation.

Generic weight surgery edits only explicitly mapped compatible tensors. The GPT-2
family path adds a separate shape-aware Conv1D contract and a token-local chart
contraction, but it still rejects LayerNorm/bias, gated, sharded, and alternate
fused layouts unless a family callback is supplied. Its omitted cross-token
energy is measured rather than hidden. The GPT-2 export path writes ordinary HF
baseline/candidate directories; the generic export format remains ordinary
named NPZ tensors plus a manifest and is not a framework checkpoint format.

The success criteria are deliberately local: positive causal effect on explicitly
labelled holdout structured probes, validation-calibrated confidence, finite and
stable trajectories, and no collapse under the configured intervention.
Passing them is evidence for that experiment, not a general claim about
transferability. A larger teacher can expose a richer target flow, but the
student bottleneck diagnostics can reject directions outside the measured
student tangent/rank capacity; no extra student capacity is created.

The scorecard's transported-flow target is fitted once from the training
traces and frozen for validation and holdout. It is an estimator for a
transported teacher operator approximation, not a semantic label, and it does
not replace a rerun through model weights. A quadratic flow term has no
ordinary tensor realization in the core contract; without an explicit family
callback it is recorded as unapplied. Causal validation marks connector
reruns separately from geometric alignment and emits
`semantic_claim: not_established`.

The activation profile records execution, but a nonzero component
contribution is not proof that the component improved fidelity. Counterfactual
metric deltas require a separate mode comparison on the same held-out probes.
The novelty scorer is a transparent fixed diagnostic average with no learned
mini-ML model; its prediction and confidence are subordinate to direct
metrics, capacity gates, and causal reruns. Per-component metric attribution
is available only through the same-split leave-one-component-out table; the
activation profile's aggregate delta is not a causal contribution.

Sequence-flattened GPT-2 charts are subject to a finite memory guard. Dense
teacher-to-student maps and dense linear/quadratic flow features are not
constructed above the configured limit. The scalable backend uses an explicit
data-aware randomized chart and fits a compact differential field. Its
capacity report is scoped to that compact chart, and a requested full mode is
marked as downgraded with a machine-readable reason. Projection rank and
discarded directions remain experimental choices requiring real-model
validation; they do not establish semantic transfer.

Relational geometry and Sinkhorn compute squared distances with the scaled
Gram identity and optional row blocks. They retain the requested `N×M`
cost/output matrix, but never construct a feature broadcast of shape
`N×M×D`; tiny negative round-off in squared distances is clamped and larger
numerical loss is rejected.

Trajectory-local alignment can reduce a depth-mixture error by allowing a
train-fitted compact chart map to vary over the existing source depth grid.
The local maps are interpolated between observed nodes and do not add model
layers. This is still a chart approximation: a lower local paired error does
not establish a better student operator, causal effect, or semantic transfer.
The scorecard must evaluate the resulting flow on disjoint validation and
holdout probes; in the GPT-2 v2 audit the local map improved geometric and
flow proxy errors while its dynamic one-step proxy degraded relative to the
global map, so all corrections remained rolled back.

The dynamic-aware depth mode adds transition and tangent objectives to the
train fit and uses validation to select among fixed candidate weights. This
does not guarantee that the selected map improves the downstream field: in
the GPT-2 clean-v2 run, the selected candidate had train-fit paired error
`192.3632907` versus `148.3318071` for the shared global reference, and the
downstream holdout scorecard left the student unchanged at flow error
`3.5102450` with zero accepted correction nodes. This is a recorded failure
case, not evidence for dynamic-aware improvement. High-dimensional GPT-2
charts use singular-spectrum descriptors when direct Jacobian rows exceed the
guard; no full Jacobian or Hessian is implied.

Calling the standalone dense `fit_flow_operator` directly on a chart above
the guard fails with a directed error because it has no paired chart context
from which to choose a compression. `fit_flow_transfer` is the entry point
that can select and record the compact backend.

The performance benchmark is a checkpoint-free synthetic dispatch benchmark.
It verifies full-stencil numerical parity and measures sampled process memory;
it cannot predict latency of GPT-2 attention, normalization, allocator, or
thread scheduling. Batch dispatch can use more peak activation memory and can
be slower on a cheap elementwise reference even while reducing Python/model
call count. A real HF/PyTorch run is required before choosing thread counts or
claiming wall-clock improvement.
