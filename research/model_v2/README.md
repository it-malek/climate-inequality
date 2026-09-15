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
| Status | session 1 (2026-09-14): diagnostics and experimental design only; **no predictor has been added, removed, transformed or tested** |

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
