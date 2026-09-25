# Formal model and success criteria

This document separates definitions from conjectures. The package implements estimators for the defined quantities; an estimator output is evidence about a model, not a theorem about the model.

## State and flow

For a probe (p), a connector exposes a finite trajectory

\[
 h_0(p),h_1(p),\ldots,h_L(p), \qquad h_{\ell+1}=F_\ell(h_\ell,p).
\]

The layer index is an observation label. It is not assumed to be a physical time variable. A strictly monotone coordinate (s_\ell\in[0,1]) is reconstructed from cumulative path length and a curvature penalty. The observed average velocity is

\[
 v_\ell(p) = \frac{h_{\ell+1}(p)-h_\ell(p)}{s_{\ell+1}-s_\ell}.
\]

The fitted local field is a local low-order surrogate,

\[
\hat v(x,s)=A(s)x+b(s)+Q(s)[x\otimes x],
\]

with ridge regularization; the quadratic term is enabled by `full` mode and is
scaled before solving. `none` and `differential` modes provide lower-complexity
ablations. This is still a low-order surrogate of a nonlinear field, not an
assumption that the network is globally linear. Residual scale, sample count,
spectral norm, and feature degree are carried with every depth node.

The fidelity layer also records a multi-scale path signature (displacement,
arc length, speed, acceleration, turning, ordered second-level area, and
roughness) and log-scaled differential invariants from Jacobian singular
values, Hessian sketches, curvature, and uncertainty. After teacher chart
transport, local tangent bases are compared by principal-angle overlap. These
quantities weight a transfer candidate; they are operational stability
features, not semantic labels.

## Differential observations

For a transition function (F_\ell), the central finite-difference estimate is

\[
 \hat J_{ij} = \frac{F_i(h+\epsilon e_j)-F_i(h-\epsilon e_j)}{2\epsilon}.
\]

The Hessian is not materialized. For normalized directions (u_k), the protocol records

\[
 \hat H[u_k,u_k] = \frac{F(h+\eta u_k)-2F(h)+F(h-\eta u_k)}{\eta^2}.
\]

Curvature is estimated from the normal component of a finite acceleration. Attention and normalization geometry are optional capabilities and are never synthesized when a connector cannot expose them.

## Cross-model transport

Teacher and student coordinates are treated as different charts. Given paired samples (X_s,X_t), the default map centers and whitens each chart, fits an orthogonal map in the shared retained rank, and decodes into the target chart:

\[
 T(x)= (x-\mu_s)W_s R W_t^{-1}+\mu_t.
\]

The package reports paired reconstruction error, relational distance error, condition number, and a fitted reverse-map cycle error. Optimal transport is available as an entropic barycentric diagnostic. Low-rank and affine maps are explicit alternatives. A small relational error is necessary for a credible chart match but is not sufficient for functional transfer.

When a teacher field is transported to student coordinates, the velocity samples are mapped and the student-side field is refit. The correction is therefore a correction to the observed operator `(A(s), b(s), Q(s))`, rather than a copy of teacher logits, token distributions, or text labels. The quadratic part is currently a diagnostic flow term: ordinary surgery applies only an explicitly mapped dense-square linear tensor and records the quadratic term as unapplied.

## Depth correspondence and gaps

Monotone dynamic programming matches observed trajectory samples. Unmatched nodes become `GapInterval` records with a penalty and uncertainty. Interpolation is only performed on the continuous path between observed endpoints. The implementation does not create a missing layer or claim that an interpolated state was observed.

Bifurcation candidates are reported at micro, meso, and macro scales: curvature outliers, local singular-spectrum changes, and growth in cross-probe separation. These are diagnostic hypotheses. They require downstream causal intervention before being called a functional bifurcation.

## Constrained objective

The unconstrained per-node target is

\[
 \Delta A(s)=A_T^{\rightarrow S}(s)-A_S(s),\qquad
 \Delta b(s)=b_T^{\rightarrow S}(s)-b_S(s).
\]

The solver seeks a clipped update under trust-region and stability constraints:

\[
 \min_{\Delta A,\Delta b} \|\Delta A-\Delta A^*\|_F^2+\lambda\|\Delta A\|_F^2
\]

subject to step norm, relative step, spectral/Lipschitz bounds, finite values, minimum singular value, and a variance-collapse ratio. For a quadratic correction it also bounds the local Jacobian with the conservative estimate (||J_Q(x)||le 2r||Q||) on the observed student radius (r). The current implementation realizes this objective with sequential projection and rollback. It is a conservative numerical solver, not a proof of global optimality.

Confidence decreases with field residual, poor conditioning, low sample support,
gap uncertainty, differential mismatch, tangent misalignment, and measured
student rank/velocity bottlenecks. The training confidence can be recalibrated
on a disjoint validation probe set. A node below the configured threshold is
rolled back independently. A tensor is edited only when its existing name
resolves through an explicit mapping or connector callback to a compatible
square layer matrix; no adapter, LoRA tensor, or invented layer is emitted.

## Causal success criteria

For a holdout set, a transfer passes only if all of the following hold:

1. geometric metrics are recorded and finite;
2. the intervened student has lower trajectory error to the teacher than the baseline student;
3. the measured causal effect is positive under the chosen intervention;
4. stability remains above the configured threshold;
5. trajectory variance does not collapse below the collapse threshold.

The package intentionally does not claim that these criteria establish semantic
equivalence. They establish only that the selected intervention produced the
specified measured effect on the selected holdout probes. The ablation suite
compares static hidden matching, delta-flow, curvature/Jacobian weighting, and
full-flow, plus shuffled-pair, random-map, noise, and quantization controls.

## Accuracy scorecard

The scorecard fits one target operator from the paired `train` traces and
freezes it before any validation or holdout score is computed. It reports
normalized transported-flow error for the student baseline, delta flow,
differential weighting, and full flow; static hidden geometry is a separate
reference metric. For candidate
$c$ on split $D$,

\[
 I_D(c)=E_D(S)-E_D(c),
\]

where $S$ is the fitted student operator for that signature mode. A
candidate is labelled `improved` only when the selected holdout value of
\(I_D\) is strictly positive. The baseline guard checks the complete
correction and then tries accepted depth nodes individually; nodes that fail
the disjoint score are rolled back. With no holdout, the report is
`incomplete`, entries use `insufficient_holdout`, and no improvement claim is
supported. Target metadata contains `target_fit_split: train` and
`metric_scope: transported_teacher_operator_approximation`.

The separate dynamic metric `heldout_one_step_transition_prediction` evaluates
the candidate at transported teacher source states and compares its one-step
field and endpoint prediction with observed teacher transitions. It is an
operational functional proxy, distinct from geometric state alignment and
from a causal rerun through edited model weights. Neither metric is a semantic
claim.

Confidence is recalibrated from the validation refit and accompanied by a
calibration gap to the bounded validation reliability proxy
\(\exp(-E_{validation})\). This is a diagnostic reliability score, not a
probability theorem. Shuffled probes, random maps, noise, and quantization are
controls. A control may be rejected by a valid contract error, but its JSON
still contains a finite structured status and uses `null` for absent values.
