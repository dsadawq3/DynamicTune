# Architecture

The system is a staged pipeline with explicit artifact boundaries:

```text
structured probes
      |
      v
connector -> observation protocol -> trace artifact
                                      |
                 +--------------------+--------------------+
                 v                                         v
          depth correspondence                      chart alignment
          gaps / bifurcations                       OT / whitening
                 |                                         |
                 +--------------------+--------------------+
                                      v
                           transported teacher flow
                         /        |          \
                 signatures   capacity     depth evidence
                                      |
                                      v
                           constrained correction solver
                          /                         \
                         v                           v
                 diagnostic report              ordinary weights
                                                     |
                                                     v
                                                causal validation
```

`types.py` contains the boundary contracts. Each boundary checks dimensions and finiteness so that a numerical failure is localized to the stage that introduced it. `observation.py` only asks a connector for transitions and optional hooks. `depth.py` operates on observations and keeps unmatched layers as uncertainty-bearing gap records. `geometry.py` builds a directional chart map and reports condition, relational, paired, cycle, and OT marginal errors. For trajectory alignment, `fit_depth_conditioned_alignment` first fits one shared data-aware compact chart and then fits an explicit affine/low-rank map at each already observed source depth; target states are interpolated on relative depth only, so layer-count mismatch produces uncertainty rather than fabricated layers. `AlignmentResult.apply(..., depth=...)` is the only path that activates these local maps, and artifacts retain the local metrics plus a same-chart global reference. `signatures.py` measures path and local differential invariants and estimates tangent transport after chart alignment. `capacity.py` measures rank coverage and irreducible student mismatch. `flow.py` transports teacher state/velocity samples at continuous student depth, uses compatibility weights, and fits affine plus optional local quadratic terms. `solver.py` applies trust-region, spectral, Lipschitz, minimum-singular-value, finite, quadratic-norm, and collapse checks per depth node. `surgery.py` is the generic ordinary-tensor contract and records unsupported quadratic terms as skipped. `gpt2.py` adds a separate explicit GPT-2 Conv1D lifting contract: it contracts compact corrections token-locally, reports omitted cross-token energy, checks exact standard GPT-2 shapes/orientations, and exports clean/candidate HF directories with restoration checks. It does not infer mappings from shape coincidence and does not handle fused, gated, or sharded layouts without a callback.

The four operational modes are represented by command boundaries rather than a hidden runtime mechanism:

* `inspect` and `report` are diagnostic-only;
* `collect-trace`, `collect-paired`, `align`, and `fit-flow` are trace-fitting operations;
* `ablate` compares dynamic methods and deliberately corrupted controls;
* `apply-surgery` creates a schema-checked ordinary-weight artifact;
* `gpt2-surgery` creates a GPT-2-specific diagnostic plan or ordinary HF baseline/candidate export;
* `validate` measures causal or record-level outcomes on held-out traces.

`pipeline.py` makes the artifact-first stages explicit: paired synthetic ground
truth, tiny/small teacher/student, and larger-teacher/smaller-student. The
optional `TorchConnector` accepts an ordinary `torch.nn.Module` only with an
explicit state encoder, decoder, and observer. It can discover common layer
containers, but family-specific forward arguments, token alignment, and weight
mutation remain callback capabilities. Missing layers and unavailable
capabilities remain explicit in artifacts.

`scorecard.py` is the decision boundary after fitting. It compares each
candidate with the fitted student operator on validation and holdout target
flows, recalibrates confidence from validation, and performs node-level
rollback when a correction fails the split guard. The report records geometric
alignment separately from dynamic fidelity; it does not infer semantic
understanding.

`scalable.py` is selected before dense field fitting when the chart or
teacher-to-student map exceeds `max_dense_features`. It uses a deterministic,
data-aware randomized basis for each original chart, stores compact maps and
operators, and lifts `FlowOperator.predict` back to the original student
state dimension. A compact flow carries its projection and capacity scope in
the artifact; it is not silently treated as a full GPT-2 weight update.

The explicit `align --depth-conditioned --dynamic-aware` mode is a train-only
estimator-selection path. It fits depth-local compact maps using state rows,
teacher/student transition rows, and, when affordable, directional Jacobian
rows. On high-dimensional charts the direct Jacobian rows are refused by the
memory guard; a bounded singular-spectrum descriptor can change row weights,
but it is not a Jacobian matrix. Local coefficients receive a depth smoothness
prior, and target states are interpolated on the relative depth domain. The
candidate weights are selected by a fixed-policy validation objective and the
holdout is used only by the downstream scorecard. The artifact records the
candidate table, train-fit errors, Pareto frontier, gap policy, and whether
direct Jacobian transport was skipped. This mode is an operational dynamical
proxy and must not be read as semantic or causal evidence.
