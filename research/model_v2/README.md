# Model V2: residual structure of the cross-country warming decomposition

> Current primary M0: `outputs/m0_scorecard_territory_corrected.json`.
> Session 1 artifacts retain legacy 1° centre-approximation results.
> See `TERRITORIAL_CV_CORRECTION.md`; intended protocol and score definitions unchanged.

Research line branched from the frozen V1 baseline. Nothing here changes the
public model, the dashboard bundle or the tagged results.

| | |
|---|---|
| Branch | `research/model-v2-residual-structure` |
| Base | `v1.3.0` = commit `a6733cb78292b9b27466e7ebfe7c4fdc6b59592f` (annotated tag `16b94d3`), verified as the branch's merge base; `main` was at the same commit when the branch was cut |
| Companion spec | `climate-inequality-instructions@v1.3.0-aligned` (design context only; this repository and its `docs/` are authoritative) |
| Status | 2026-09-16: M1a not promoted (registered measurement sensitivity); **M1b closed as a completed negative experiment, not promoted** (`M1B_CLOSURE.md`); M0\* remains the development baseline. **M2 design and evaluation contract frozen; no M2 fit or score** (`M2_EVALUATION_CONTRACT.md`) |

## Question

What physically and statistically accounts for the cross-country variation in
area-weighted land-warming rates that the V1 model leaves unexplained, and does
geography remain the dominant explanatory axis under a richer, spatially
appropriate model? The aim is not a larger R²; it is to say how much of the V1
residual is recoverable physical structure, spatial dependence, functional-form
error, observational-product uncertainty, internal variability, or irreducible
at country resolution.

## M0, the immutable comparison row

Read from the `v1.3.0` bundle (`app/data/decomposition_summary.json`,
`stability_summary.json`) and reproduced from the local artifacts to 1e-11
(`outputs/m0_baseline.json`). M0 is never recalculated under later definitions.

| Quantity | M0 |
|---|---|
| Outcome | area-weighted national land-warming rate, 1950-01 to 2013-09, Berkeley Earth 1° grid, °C/decade |
| n | 151 |
| In-sample R² | 0.636 |
| Residual share | 0.364 |
| Geography share | 0.513 |
| Emissions / responsibility share | 0.021 |
| Socioeconomic share | 0.069 |
| Population share | 0.033 |
| Residual Moran's I | 0.269 (p = 0.005; station-centroid kNN, k = 8, 199 permutations) |
| Spatial-CV R² / RMSE (primary protocol, added here) | 0.153 / 0.0429 °C/decade |

Full anatomy in [`M0_BASELINE.md`](M0_BASELINE.md).

## Evaluation contract (frozen before any predictor work)

* **Primary spatial validation**: leave-one-country-out with a 500 km
  exclusion buffer on the minimum inter-territory distance, so every bordering
  and near-bordering country is out of training
  ([`SPATIAL_CV_PROTOCOL.md`](SPATIAL_CV_PROTOCOL.md)).
* **Secondary stress test**: leave-one-UN-M49-sub-region-out (20 folds).
* **Reference, never a criterion**: random 10-fold (seed 0), to show the
  spatial optimism gap.
* **Scorecard**: the fixed metric set every stage reports
  ([`SCORECARD.md`](SCORECARD.md)); M0's corrected row is in `outputs/m0_scorecard_territory_corrected.json`.
* **Stages and gates**: M0 → M1 (physical covariates, preceded by an
  area-consistent re-measurement of the existing geography features) → M2
  (pre-specified nonlinear terms) → M3 (explicit spatial structure) → M4
  (final out-of-sample assessment), with the decision rules in
  [`EXPERIMENTAL_PLAN.md`](EXPERIMENTAL_PLAN.md).
* **Hypotheses** are derived from the residual, not from a variable list
  ([`HYPOTHESIS_REGISTER.md`](HYPOTHESIS_REGISTER.md)); the residual itself is
  characterised in [`M0_RESIDUAL_DIAGNOSTICS.md`](M0_RESIDUAL_DIAGNOSTICS.md).
* **Decisions that need the project owner** are listed in
  [`OPEN_DECISIONS.md`](OPEN_DECISIONS.md). M1 does not start before the
  protocol and plan are approved.

## Guardrails inherited from V1

* The emissions result is: the historical national emissions/responsibility
  feature group contributes little independent explanatory variance to
  cross-country differences in area-weighted land-warming rates. It is not
  "emissions do not explain warming".
* The station-to-area shift is consistent with station-network geography and
  observational sampling; it is not proof that station siting caused the
  earlier relationship.
* Shapley/LMG shares are a descriptive variance decomposition, never causal
  attribution.
* The residual is variance in cross-country warming differences not captured by
  the model, not "climate change of unknown cause".
* A spatial term that lowers residual Moran's I accounts for covariance; it
  does not explain a mechanism.

## Layout

```
research/model_v2/
  m0.py            refit of the tagged model (same code path), influence diagnostics, bundle verification
  geometry.py      country geometry from the GPW national grid, ETOPO and Köppen (descriptive only)
  cells.py         within-country structure of the 1° trend field (what the collapse discards)
  regions.py       UN M49 sub-regions for the 151 countries
  spatial.py       spatial weights, Moran's I, Geary's C, local Moran's I
  borders.py       LEGACY 1° approximation and historical correlogram
  territory.py     certified WGS84 clearance for all GPW cell footprints
  run_territory_correction.py  corrected scores, membership audit and provenance
  diagnostics.py   the residual diagnostics -> outputs/m0_countries.csv, m0_residual_diagnostics.json
  figures.py       research figures -> outputs/figures/ (needs matplotlib: uv run --with matplotlib ...)
  cv.py            fold constructions, the V1 estimator as fit/predict, the scorecard
  run_cv_design.py evaluation of the candidate validation designs on M0
  run_m0_scorecard.py  the frozen M0 scorecard row
  tests/           synthetic tests plus a data-gated bundle verification
  outputs/         committed tables, JSON and figures
```

Historical Session 1 reproduction (these commands overwrite legacy artifacts;
do not run for current CV). From the repository root, with the V1 `data/processed` artifacts and
the raw grids in place):

```bash
uv run python -m research.model_v2.geometry
uv run python -m research.model_v2.cells            # needs data/processed/model_v2/berkeley_cell_trends_1950_2013.npz (see cells.py)
uv run python -m research.model_v2.diagnostics
uv run python -m research.model_v2.borders
uv run --with matplotlib python -m research.model_v2.figures
uv run python -m research.model_v2.run_cv_design
uv run python -m research.model_v2.run_m0_scorecard
uv run pytest -q research/model_v2/tests
```

The per-cell trend field is computed once with the V1 operator
(`src.area_weighting.cell_trends`, about three minutes) and cached under the
gitignored `data/processed/model_v2/`; `cells.py` checks that its country means
reproduce the V1 `trend_c_per_decade_area_weighted` column exactly.

Current primary CV: `uv run python -m research.model_v2.run_territory_correction`.

## M0.5 (research only)

- `RANK_DEFICIENCY_AUDIT.md`: recovered full-rank sensitivities on corrected folds.
- `V2_PREDICTOR_CONTRACT.md`: primary/sensitivity representation proposals.
- `PRODUCT_STABILITY_AUDIT.md`: matched-country external product audit, with
  absolute-temperature legacy and preprocessing-aligned ERA5 arms distinguished.
- `M1A_MEASUREMENT_SPEC.md`: specification only; five remeasurements and one
  unchanged geography feature.
- `OPEN_DECISIONS.md`: both M0.5 decisions approved on 2026-09-15.
- `M1A_EVALUATION_CONTRACT.md`: M1a comparison and interpretation rules, frozen
  before M1a measurement or scoring.
- `M1A_FEASIBILITY_AUDIT.md`: original-gate measurement build (147/151 pass), kept as
  the evidence for the pre-result `M1A_MEASUREMENT_SPEC.md` Amendment 1.
- `M1A_MEASUREMENT_RECORD.md`: amended measurements frozen before scoring.
- `M1A_REPORT.md`: M1a scorecards and verdict (neither improved generalization nor
  spatial specification; conclusion stable). Not promoted; registered as the
  area-consistent geography measurement sensitivity.
- `M1B_DESIGN.md`, `M1B_HYPOTHESIS_REGISTER.md`: approved M1b design, baseline
  hydroclimatic dryness only (original-rule C2 built for the feasibility audit; nothing fitted).
- `M1B_EVALUATION_CONTRACT.md`: C2 measurement and M1b evaluation rules, frozen before
  any CRU acquisition. Amendment 1 (structural CRU land-mask harmonization) was adopted
  before any amended C2 value, redundancy diagnostic or score.
- `M1B_AMENDMENT1_BUILD_STOP.md`: the stopped Amendment 1 build (one non-structural Peru cell), resolved by
  contract Amendment 3 (non-structural support counts against coverage; package-integrity gates).
- `M1B_PREDICTOR_PACKAGE.md`: the frozen Amendment 3 C2 package (build `58e64bf`), its integrity digests,
  determinism and invariance evidence; frozen before any redundancy diagnostic or score.
- `M1B_REDUNDANCY_DIAGNOSTIC.md`: pre-outcome redundancy of C2 against M0\* (pass; R² 0.650, VIF 2.86),
  computed from verified package bytes and outcome-free predictor inputs; M1b not scored.
- `M1B_FEASIBILITY_AUDIT.md`: original-rule C2 build (148/151 pass the coverage gate),
  kept as the evidence for Amendment 1.
- `M1B_REPORT.md`: the M1b result (`not supported`): C2 worsens out-of-fold error and carries the
  opposite of the pre-stated sign, with the unconditional per-capita arm, the reproducibility and
  independent-verification evidence, and the contract-based non-execution of the conditional arms.
- `M1B_CLOSURE.md`: closure of M1b from saved evidence only (2026-09-16): verification record, evidence
  limits, and the dated post-result corrections to `M1B_REPORT.md`. M1b is a completed negative
  experiment and is not promoted.
- `M2_PROVENANCE_AUDIT.md`: what `af97bd2` actually proposed for M2, classified A/B/C from Git. It
  records the disclosures: H1/H5 are residual-derived, not outcome-naive; the retained-covariate slot is
  void; nonlinear dryness is barred.
- `M2_DESIGN.md`: the single joint candidate, M0\* + two natural cubic `abs_latitude` columns +
  `I(SH)·abs_latitude` (+3 coefficients, all geography). It specifies the train-only knots and the exact
  basis (`m2_latitude_basis.py`), and γ as the only shape diagnostic.
- `M2_FEASIBILITY_AUDIT.md`: predictor-only structural feasibility (`m2_feasibility.py`,
  `outputs/m2_feasibility/`, run from pushed `fbaa3ff`). It passes in every required fit and in the
  conditional M1a arm. No outcome was read and nothing was estimated.
- `M2_EVALUATION_CONTRACT.md`: frozen M2 rules. Predictive support is ΔRMSE < 0 with the interval
  below 0. Promotion adds ≤ −0.002, the M49 and worst-region vetoes, and structural integrity. There is
  no sign criterion, no penalty and no rescue. It **authorizes no execution**.
- `M1B_CLI_MAINTENANCE.md`: post-M1b auxiliary maintenance (2026-09-16). It fixes the `--compare`
  exit status and changes no scoring definition, constant, pin or artifact. The M1b result stays
  pinned to `618c8b8`.
- `M1B_EVALUATOR_SPEC.md`: the operational evaluator specification, frozen and pushed before the
  first outcome-facing M1b run. It fixes the execution boundary, the strict scoring-input gate, the
  two reported representations, the saved evidence, the prespecified equivalence tolerances and the
  conditional runners' authorization; it changes no contract rule.

All public V1 values and artifacts are unchanged. The two-product comparison
does not tune or select predictors and must be repeated after V2 is frozen.
