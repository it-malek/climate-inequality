# Numerical corrections used in 2.1.0

Version 2.1.0 incorporates the corrections described here. The original
correction-set identifiers and separate output namespace are retained for
reproducibility; the frozen historical Model V2 specification remains unchanged.

## Evidence availability

Both vulnerability criteria require finite responsibility and area-warming rank
correlations and permutation p-values. Missing required evidence produces
`status=insufficient_evidence`, `holds=false`, and explicit reasons. Counts refer
to finite pairs. Meeting the descriptive criterion does not establish equal
warming or a causal relationship.

Global and local spatial diagnostics report unavailable statistics explicitly
when residuals, samples or neighbor weights cannot define them. Reasons propagate
through local labels, FDR summaries and influence diagnostics. Available cases
retain the existing formulas, random seeds, permutation counts and tail
conventions. Moran's permutation tail uses the absolute statistic, not distance
from its null expectation. An unavailable ancillary z-score is distinguished from
an unavailable statistic or p-value.

## Conditional forecast intervals

For the fitted model `y_t = x_t beta + e_t`, with
`e_t = rho e_(t-1) + u_t`, conditioning on the final training observation gives

```text
y_(T+h) = rho^h y_T + z_h beta + sum_(j=0..h-1) rho^j u_(T+h-j)
z_h = x_h - rho^h x_T
c_h = sum_(j=0..h-1) rho^(2j)
```

The predictive Student-t distribution has `2 alpha_n` degrees of freedom and
scale squared `(beta_n / alpha_n) [c_h + z_h V_n z_h']`. This includes the
coefficient uncertainty in the final training residual. The conditional mean,
in-sample bands, and estimation of rho and lambda are unchanged.

The scale is not the predictive variance. The latter uses
`beta_n / (alpha_n - 1)` when `alpha_n > 1`. Rho and lambda remain fixed at their
fitted values; their estimation uncertainty and future-driver uncertainty are
not included. Correcting this formula alone does not establish better coverage.

## Interpretation

The historical interpolation comparison conditions on a variogram fitted to the
full field, including held-out outcomes. Coordinate-group exclusion does not
remove that dependence. Its recorded numbers are preserved. Version 2.1.0 adds
a separate training-only interpolation evaluation, documented in
[current validation](current_validation.md).

ND-GAIN vulnerability combines exposure, sensitivity and adaptive capacity
across six sectors. It is not a standalone adaptive-capacity score. See the
[ND-GAIN methodology](https://gain.nd.edu/our-work/country-index/methodology/).

## Execution and provenance

The correction set is `prospective-corrections-2026-10-22`, the evidence schema
is `finite-evidence-v1`, and the forecast method is `fixed-rho-conditional-t-v1`.
These implementation identifiers distinguish corrected execution from the
frozen historical result generation.

Writers require an explicit prospective scope. A `run_id` confines its output
under `outputs/prospective/prospective-corrections-2026-10-22/<run_id>/`.
The orchestrator creates the directory exclusively and verifies checkpoint hashes
before continuing. Input generations must be supplied explicitly. Selecting an
output scope does not redirect historical input defaults.

`research.model_v2.prospective_diagnostics` verifies saved result artifacts and
recomputes diagnostics from unaffected predictions in their original country
order. Its `spatial-diagnostic-replay-v1` supplement records input reuse and
availability without fitting or selecting models. Reconstruction of an unsaved
residual requires a separate recorded step with the original fixed inputs and
parameters. Rounded saved inputs must be distinguished from original in-memory
values when comparing historical diagnostics.

The historical evaluator entrypoints remain guarded. Frozen protocols and
independent checkers remain unchanged. Test scopes use temporary synthetic inputs.
Compact input identities and released result records are committed under
`reports/current_validation/`. Bulk execution logs and raw inputs are not
distributed with the repository.
