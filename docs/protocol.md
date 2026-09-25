# Observation protocol and capability matrix

The connector contract has five required operations: expose visible layer IDs, produce an initial state for a structured probe, apply one layer transition, and optionally expose normalization/attention geometry and mutable named weights. A trace always records hidden states, layer IDs, continuous coordinates, transition deltas, vector fields, and finite checks.

The protocol has three levels of evidence. Raw observations are retained,
derived signatures are serialized with the trace, and cross split scorecards
are evaluated only after collection.

| Observable | Required hook | Meaning when available | Meaning when absent |
|---|---|---|---|
| hidden trajectory | `initial_state`, `transition` | state chart sampled at layer boundaries | collection cannot run |
| residual stream | `residual_state` or explicit model hook | residual chart, kept separate from hidden state | reported unavailable |
| vector field | state deltas and depth gaps | finite-depth velocity samples | never inferred from logits |
| Jacobian sketch | transition callable | local first-order sensitivity | no local sensitivity claim |
| Hessian sketch | transition callable | directional second-order probes | no curvature-from-Hessian claim |
| normalization geometry | explicit connector hook | norms/scales/statistics supplied by model | omitted |
| attention geometry | explicit connector hook | attention-specific geometry supplied by model | omitted |
| weight surgery | `weights` and `set_weights` | named ordinary tensors can be edited | diagnostic-only for that connector |
| path/differential signatures | derived from the observed trace | multi-scale path and local Jacobian/Hessian invariants | absent only if the trace itself cannot be made finite |

The default connectors are:

* `SyntheticConnector`, used by the local test systems. It exposes exact transitions and finite-difference sketches and marks residuals only when the synthetic system declares that it has them.
* `HFLikeConnector`, which is a framework-free callback adapter for fake/HF-like objects. It recognizes common `layers`, `transformer.h`, `encoder.layer`, and `model.layers` containers. It does not import Transformers and does not load checkpoints. A torch module is routed to `TorchConnector`; it never receives numpy or a spare `probe` argument through a guessed standard layer call.
* `TorchConnector`, an optional PyTorch connector. `TorchHooks.state_encoder` and `state_decoder` define the chart, while `state_observer` reduces a `[batch, sequence, hidden]` tensor to the one-dimensional observed vector. The default forward call is `layer(hidden)` and preserves batch/sequence shape. A family-specific `layer_call` is required for keyword-heavy or cached forwards. The capability matrix distinguishes automatic layer discovery and finite-difference access from callback-dependent residual, token-position, attention, normalization, and weight surgery capabilities.

Capability reasons are serialized with each trace. A downstream report must preserve those reasons. In particular, missing attention or residual hooks are missing measurements, not zeros.

## Collection rules

Probe IDs, model IDs, layer IDs, and coordinates are retained. Every numerical array is checked for finiteness at the boundary. Central differences use an explicit step. Hessian sketches use normalized random directions and a seed recorded by the caller. Adaptive depth uses observed arc length; it does not rename or manufacture layers. Each collected artifact also stores `path_signature` and `differential_signature` in metadata, alongside their source feature space and uncertainty. These are operational summaries of both observed models, not semantic labels.

For a model family with fused blocks, recurrent state, sparse routing, or sequence-valued hidden states, a custom connector should first define the state chart and flattening/aggregation convention. That convention is part of the experiment metadata and must be held fixed across train, validation, and holdout probes. Capability `weight_surgery` is false unless both a getter and setter are supplied; fused, gated, convolutional, and sharded tensors are not inferred from shape.

High-dimensional sequence charts also record `dense_memory_guard`, feature
counts, backend, chart projection shape, and effective signature mode. The
scalable path is entered before a dense field or cross-model map is allocated;
it stores an explicit compact chart projection and preserves the original
state dimension at the `predict` API. Compact operator matrices are not
interpreted as full-dimensional GPT-2 weights. A full request that exceeds
the guard is a finite differential approximation with a skipped quadratic
component, never a silent dense fallback.
