# Post-fit forensic audit

`audit_flow_fit` is a read-only diagnostic over an existing trace fit. It
does not run a model, collect a new observation, change weights, or relax a
guard. It answers four contract questions:

1. Was the alignment fitted on initial states or on a trajectory sample?
2. How much of each teacher and student depth path has an exact monotone
   anchor, and are gap counts arithmetically consistent with the dynamic
   programming path?
3. Which multiplicative evidence factors reduce the reported confidence?
4. Does a supplied scorecard identify the same train-fitted transported
   teacher operator as the fit artifact?

The initial-state alignment numbers and the path-level numbers use different
observations. A small `paired_error` or `relational_error` from an
`initial_state_only` alignment is evidence about the input chart only. It
does not predict later hidden states, transition fields, or functional model
behavior. The audit therefore reports trajectory geometry, teacher coverage,
student coverage, correspondence cost, and gap support separately.

The confidence decomposition is diagnostic. It reports support, exact-node
gap evidence, interpolation, residual, capacity, tangent, stability, and
validation calibration factors. It does not assign causal credit to a factor.
The scorecard remains the authority for accepting a correction: a candidate
must improve its declared holdout metric and preserve the independent guards.

## Existing artifact command

The command below fits only from already saved trace artifacts and writes a
strict JSON audit. It does not load a checkpoint or call a model connector.

```powershell
python -m faytuna_flow.cli audit-fit `
  --student .\student\train.npz `
  --teacher .\teacher\train.npz `
  --alignment .\alignment-lowrank.npz `
  --scorecard .\scorecard.json `
  --signature-mode full `
  --scalable-rank 64 `
  --output .\forensic-audit.json
```

For the real GPT-2 pair, a typical result with good initial alignment and
low teacher depth coverage should be classified as
`consistent_with_bottleneck`. That classification means the guards have
evidence to reject or limit the correction. It is not a semantic-transfer
claim. `inconsistent` is reserved for contract failures such as a scorecard
target fingerprint mismatch or impossible gap arithmetic.
