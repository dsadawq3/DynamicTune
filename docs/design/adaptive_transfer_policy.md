# Adaptive Transfer Policy

`AdaptiveTransferPolicy` is the single policy boundary for continuation after
a flow fit. It selects a bounded backend, depth strategy, tensor-family view,
block order, gain grid, and line-search policy from train metadata and
validation traces. `safe` and `experimental` are modes of this same policy.

The clean student remains the baseline. The policy applies a candidate only by
coherent depth blocks. A failed block is rolled back as a whole, and the
validation split controls selection. A disjoint holdout is the final verdict;
without it no result is called accepted.

The policy never invents layers. If the depth fit reports gaps, the selected
`gap_aware_monotone` strategy keeps interpolation on existing nodes and marks
the gap in serialized metadata. Tensor-family selection is descriptive until
an explicit connector supplies a shape-aware ordinary-tensor mapping.

The schedule evaluates

\[
F_{b,\alpha}=F_{student,b}+\alpha\Delta F_b
\]

under trust-region, confidence, spectral, Lipschitz, finite-output, and
anti-collapse guards. `safe` bounds gains at one and uses conservative
backtracking. `experimental` may test the wider configured grid and orders
blocks by measured confidence, but it remains subject to the same holdout
report and is never a semantic claim.

The policy is intentionally separate from checkpoint export. GPT-2 ordinary
weight export is exposed through the same implementation source and requires
an explicit tensor mapping. Unsupported fused or cross-token mappings fail
with a capability error instead of silently changing tensor semantics.
