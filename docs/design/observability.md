# Experiment journal and activation audit

`faytuna_flow.journal.ExperimentJournal` writes a deterministic JSONL stream
and a short human-readable summary.  The default stream has no wall-clock
field, so two runs with the same artifacts and seed can be compared byte for
byte.  Non-finite numeric values are converted to JSON `null` and the event
lists their original field in `nonfinite_fields`; `allow_nan` is never used.

The journal has run, stage, probe, depth, math, gain, tensor, and runtime
scopes.  Trace events retain model IDs, feature space, state shapes and dtype,
coordinates, token/position metadata, pair/split identifiers,
perturbation norms, differential availability, singular spectra, curvature,
uncertainty, and optional raw arrays.  Fit and scorecard events add target
fit split, metric scope, depth gaps, capacity diagnostics, predicted versus
observed flow errors, correction norms, confidence, stability spreads, and
per-node rollback.  Weight surgery emits the applied tensor list plus every
skipped tensor and reason.  Runtime events describe conversion and stock
llama.cpp preflight; they do not claim hidden-state access from llama-cli.

Every fitted flow carries a `faytuna-math-profile-v1` activation audit.  The
profile contains one record for alignment, OT, continuous depth, path
signatures, local 1-jet, directional 2-jet/Hessian sketch, curvature, robust
loss, tangent/projector transport, capacity projection, quadratic flow,
stability barriers, and gain schedule.  Each record has:

* `enabled`: requested by the mode or explicit switch;
* `applied`: the calculation actually changed the executed path;
* `contribution`: a local magnitude or confidence diagnostic;
* `metric_delta`: populated only by a counterfactual comparison, otherwise
  `null` with an explicit scope;
* `skipped_reason`: why an enabled component was unavailable or disabled.

Alignment and continuous depth are protocol requirements.  Disabling either
raises an error instead of silently returning the same calculation.  The
other switches are explicit and alter a calculation when supported; for
example, `signature_mode=none` suppresses path/differential weighting,
`hessian_2jet=false` suppresses the quadratic channel, and
`stability_barriers=false` suppresses the deterministic stability ensemble.

`assess_correction` is the `surprise_novelty_scorer`, a fixed direct-metric
calculator. It is not a trained mini-ML model: `trained_model` is `false` and
`model_checksum` is `null`. It exposes its six-feature vector, direct metrics,
prediction, confidence, sign calibration, disagreement, novelty value, and
limitations. Its prediction is diagnostic and never approves a correction.
Held-out accuracy, one-step functional fidelity, causal connector rerun, and
capacity gates remain the decision evidence.

The scorecard also emits `math_component_counterfactual_table`. For each
removable component it refits with exactly that component disabled, using the
same train/validation/holdout IDs, seed, and stability ensemble, then reports
the direct held-out delta
`without_component_holdout_error - full_holdout_error`. Positive values mean
that removal worsened the held-out metric. Required protocol components and
components that cannot be disabled without refitting alignment are marked
`unavailable` with a reason. The older `math_contribution_table` contains
execution diagnostics and an aggregate candidate-minus-reference delta only;
its per-component `metric_delta` remains `null`.

The real GPT-2 runner additionally writes `progress.jsonl` and `progress.log`.
Progress events cover model load, split, probe, block/layer transition, and
finite-difference dispatch. They include probe/layer counters, observed rate
and ETA, RSS/private/commit memory when the host exposes it, backend/device,
model tensor checksum, and the most recently completed artifact. Each probe
is persisted as an independent uncompressed NPZ after its trace is finite and complete;
`resume_manifest.json` records the chart, layer schema, ranks, steps, seed, and
model checksum. A restart reuses a probe only when all those fields and its
probe/model/schema identity match. An incomplete or stale artifact is
recomputed rather than treated as an observation.

The runner also supports an explicit `benchmark-observation` command. Its
actual-size synthetic chart uses `state_dim=25600`, compares scalar and batch
dispatches with the same differential stencil, and reports max/mean absolute
trace error, dispatches, samples, wall-clock, throughput, and sampled
RSS/private/commit peaks. The result is an overhead benchmark, not a claim
about GPT-2 model latency. High-dimensional path ordered-area aggregates use a
deterministic randomized Frobenius sketch; exact dense area remains restricted
to small charts.

The runner also supports an explicit `benchmark-observation` command. Its
actual-size synthetic chart uses `state_dim=25600`, compares scalar and batch
dispatches with the same differential stencil, and reports max/mean absolute
trace error, dispatches, samples, wall-clock, throughput, and sampled
RSS/private/commit peaks. The result is an overhead benchmark, not a claim
about GPT-2 model latency. High-dimensional path ordered-area aggregates use a
deterministic randomized Frobenius sketch; exact dense area remains restricted
to small charts.
