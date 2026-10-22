# Historical numerical corrections, 22 October 2026

These notes describe the isolated correction design at this point in development.
The published bundle and frozen Model V2 specification were separate. Later
validation results are not implied by this engineering description.

## Evidence states

Both vulnerability criteria require finite responsibility and area rho and
permutation p. Missing or nonfinite required evidence yields `holds=false`,
`status=insufficient_evidence`, and reasons. Gradients count finite pairs and
identify insufficient samples or constant inputs. Both the income and ND-GAIN
summary/view paths carry the state and coverage. The prospective summary schema
is `finite-evidence-v1`. Legacy saved summaries remain readable.
A supported result means the prespecified descriptive criterion is met;
nonsignificance does not establish equal or no-lower warming.

The main-pipeline Moran result retains two-value unpacking and adds availability
and a reason. The matrix-based V2 global Moran result adds a reason; unavailable
statistics/p-values are null. Local diagnostics use missing numeric values,
explicit unavailable labels and reasons (including a frame-level reason for an
empty input). Invalid samples, constant residuals, nonfinite values and unusable
neighbors do not reach a permutation comparison that can fabricate a small p.
The companion Geary statistic in the same report uses the same input guard so
an unavailable Moran report is not followed by a fabricated Geary p-value.

Diagnostic summaries, local cluster counts, FDR labels, influence summaries and
CLI reporting retain unavailable states. Valid cases preserve the existing
formula, absolute-statistic tail convention, RNG order and numbers. No spatial
weights, inference tail convention or model-selection rule was changed.
Independent checker implementations and every protocol remain untouched.

## Conditional forecast uncertainty

The implemented model is y_t = x_t beta + e_t with e_t = rho e_(t-1) + u_t.
At the final fixed-rho/lambda refit, beta | sigma²,D has mean m_n and covariance
sigma² V_n, and sigma² | D is inverse-gamma(alpha_n,beta_n).

Since e_T = y_T - x_T beta, recursive substitution gives

    y_(T+h) = rho^h y_T + z_h beta + sum_(j=0..h-1) rho^j u_(T+h-j),
    z_h = x_h - rho^h x_T,
    c_h = sum_(j=0..h-1) rho^(2j) = (1-rho^(2h))/(1-rho²).

Integration gives Student-t degrees of freedom 2 alpha_n, unchanged conditional
mean rho^h y_T + z_h m_n, and scale squared

    (beta_n / alpha_n) [c_h + z_h V_n z_h'].

The independent covariance expansion is x_h V_n x_h' + rho^(2h) x_T V_n x_T'
- 2 rho^h x_h V_n x_T'. Scale squared is not variance: variance replaces
beta_n/alpha_n with beta_n/(alpha_n-1) when alpha_n > 1. Quantile intervals
remain defined for positive alpha_n even when variance does not exist.
Future candidate summaries identify `fixed-rho-conditional-t-v1`.
In-sample bands, the conditional mean, and estimation of rho/lambda are unchanged.
No estimated-rho/lambda uncertainty, driver uncertainty or new model selection
is introduced.

Tests use synthetic frames and arrays, actual SciPy quantiles, expanded
covariance, finite innovation sums, rho zero/negative/near-unit controls,
coefficient-design cancellation, a long-horizon limit and hierarchical posterior
simulation. Independent review also checks Gaussian Schur conditioning against
the actual implemented posterior. These checks do not measure climate coverage.

## Historical interpolation

The historical comparison is conditional on a variogram fitted to the full
field, including held-out outcomes. Whole-coordinate-group exclusion removes
neighbor duplication, not that fitted-parameter dependence. Its recorded
rankings and numbers are preserved; no replacement study was launched.

## Integrated execution and output boundary

This revision isolated prospective-development integration from the published analysis.
The identifiers have separate roles:

- Correction set: `prospective-corrections-2026-10-22`.
- Evidence schema: `finite-evidence-v1`.
- Conditional forecast method: `fixed-rho-conditional-t-v1`.

These are implementation identifiers, not revisions of the frozen Model V2
scientific identity or a new validated release. The package version, release
manifest and retained-model record are unchanged. New summaries carry
`computation_status=prospective-development`; historical/unversioned summaries
remain readable and are identified as saved records not recomputed by this code.
An available Moran statistic and permutation p-value can coexist with an
unavailable ancillary z-score; `z_score_reason` records that separate condition.

The production output root is derived from `src.data_io.OUTPUTS_DIR`:

    outputs/prospective/prospective-corrections-2026-10-22/

Within it, standalone stage summaries/tables go under `processed/`, preview
assets under `app_bundle/`, surface figures under `figures/`, and diagnostics
under `diagnostics/` (product comparisons under `diagnostics/product_stability/`).
No writer is redirected into the published `app/data` or the frozen research
output tree. Merely importing modules or selecting an execution scope creates
no output directory.

Adapted command-line entrypoints require `--prospective`. Programmatic writers
require an explicit `with src.prospective.prospective_execution():` scope before
any input read/fit; omitted historical output defaults are replaced inside this
scope. Explicit destinations must resolve within its root. Escaping paths,
symlink aliases and existing hardlinked output files are refused; all declared
output channels are checked before the first write. Builders that acquire inputs
implicitly now require prepared inputs read-only. Acquisition is a separate
authorized operation.

The synthetic test harness opts in explicitly with
`prospective_execution(scratch=tmp_path_factory.getbasetemp())`. Scratch scopes
are restricted to unaliased OS-temporary subdirectories and confine writes there.
They assert synthetic provenance; the API cannot infer whether arbitrary caller
supplied arrays came from real data. This is an execution/API boundary, not an
operating-system sandbox for arbitrary Python code.

Read destinations are not silently rewritten. Defaults continue to read the
named prepared historical input files. For a future chain using fresh prospective
intermediates, select those inputs explicitly: for example, pass scoped
`processed/explain_summary.json` and `processed/explain_features.parquet` through
`build_app_assets`'s existing `explain_summary_path` / `explain_bundle_path`
parameters. The preview labels a copied historical explanation by its own origin,
not the origin of newly built core assets. Border and product diagnostics likewise
retain named historical input defaults; an execution plan must specify whether
those historical inputs or new intermediate tables are intended. Output isolation
does not validate a mixed-generation scientific comparison.

## Specific evaluator work still required before any real-data run

Historical scoring writers are closed to normal prospective execution. The
explicit temporary synthetic harness may exercise destination-parameterized
`evaluate` / conditional functions; legacy writers with hardcoded destinations
are fully refused. Read-only result comparison remains available; writing a
comparison record requires the output scope. Existing scientific provenance,
reference, integrity and independent-checker gates remain in place.

The remaining adaptations are limited to these existing contracts:

1. `cv.scorecard` and `m1a_evaluate.score` currently project scalar Moran fields.
   A prospective scorecard must retain availability/reasons with those fields.
2. M1b/M2 `score_model` and land-centroid summaries must preserve that evidence
   state; M2 verdict/qualification deltas must handle unavailable inputs explicitly
   before subtraction or threshold comparisons. Missing diagnostics cannot qualify
   a model or become an ordinary nonsignificant result.
3. M3 `protocol_card` / `run_family` and qualification must retain availability and
   avoid numeric delta calculations when either leg is unavailable. M4 and the
   conditional runners need a declared policy for those dependent assessments.
4. The legacy M0/CV/territory/rank/M1a writers and M1b–M4/conditional entrypoints
   need approved prospective input identities, output manifests, version admission
   and namespace-aware invocation before their real-data execution gates can open.
   Current source-closure and checker requirements must not be silently relaxed.

These items are not implemented as a new evaluation framework. Pure estimators
remain testable; frozen protocols and independent checkers remain unchanged. Any
future checker adaptation is a separate scientific decision, not an automatic
consequence of a passing engineering suite.

## Targeted ND-GAIN definition clarification

ND-GAIN vulnerability combines exposure, sensitivity and adaptive capacity across
food, water, health, ecosystem services, human habitat and infrastructure.
It is not a standalone adaptive-capacity measure. The prospective view, source
comments and corresponding explanation use that distinction, without changing
any score, ranking or recorded number. Source checked 2026-10-22:
[ND-GAIN Country Index methodology](https://gain.nd.edu/our-work/country-index/methodology/).
