# Residual structure of the cross-country warming decomposition

The public decomposition (`docs/findings.md`) leaves about 36% of the
cross-country variance in area-weighted land-warming rates outside the static
model, and that residual is spatially clustered. This directory holds the
follow-up investigation of that residual: what it looks like, whether
re-measuring or extending the static model recovers any of it under
geographically separated validation, how much of it explicit spatial dependence
represents, and how much of each answer survives a change of temperature
product. The public bundle, dashboard and pipeline in `src/` are unchanged by
anything here; the dashboard reads a small derived summary of these records
(`app/data/residual_structure_summary.json`, built by `src.residual_assets`).

The investigation is closed. Its conclusions, with every number, are in
[`RESULTS.md`](RESULTS.md); the validation design is in
[`VALIDATION_PROTOCOL.md`](VALIDATION_PROTOCOL.md); the characterisation of the
residual that motivated the work is in
[`RESIDUAL_DIAGNOSTICS.md`](RESIDUAL_DIAGNOSTICS.md).

## Summary

* The static model transfers weakly to geographically separated countries:
  leave-one-country-out with a 500 km territorial exclusion gives CV R² 0.153
  and RMSE 0.0429 °C/decade against an in-sample R² of 0.636, and the
  out-of-fold residual has Moran's I 0.32.
* Re-measuring the geography features over national land area instead of
  station locations, adding a pre-1950 baseline dryness index, and a richer
  latitude functional form each failed to demonstrate transferable improvement.
  The dryness index made prediction worse; the latitude form improved the point
  estimate but its paired interval narrowly included zero.
* Spatial error and spatial lag models on the static specification both improve
  geographically separated prediction under Berkeley Earth and represent about
  39–40% of the static in-sample residual as neighbour-conditional dependence.
  Neither is designated the single final model.
* The static decomposition conclusion (geography the largest group, historical
  responsibility small) holds under every check it was registered for. The
  spatial models' predictive advantage does not reproduce with ERA5 outcomes.

## Stage labels

File names use the labels under which the work was carried out: `m0` (the
baseline and its validation), `rank_audit` and `product_stability` (the
representation and product audits), `m1a` (geography re-measurement), `m1b`
(baseline dryness), `m2` (latitude functional form), `m3` (spatial models) and
`m4` (final assessment). `M0*` denotes the full-rank representation of the
public model described in `VALIDATION_PROTOCOL.md`. The dashboard and
`docs/` describe the same analyses by their content rather than by these
labels.

## Layout

```
research/model_v2/
  m0.py                    refit of the public model from local artifacts; bundle verification
  geometry.py, cells.py    country geometry and within-country structure of the 1-degree trend field
  regions.py               UN M49 sub-regions for the 151 countries
  spatial.py               spatial weights, Moran's I, Geary's C, local Moran's I
  borders.py               legacy 1-degree territorial distance (superseded, kept for the record)
  territory.py             certified WGS84 clearance between GPW cell footprints
  run_territory_correction.py   the corrected primary spatial-CV scores
  diagnostics.py, figures.py    residual diagnostics and figures (outputs/figures/)
  cv.py, run_cv_design.py, run_m0_scorecard.py   folds, estimator wrapper, scorecard
  rank_audit.py            full-rank representations of the design matrix
  product_stability.py     Berkeley versus ERA5 outcome and residual comparison
  m1a_geography.py, m1a_evaluate.py            area-consistent geography measurement and its score
  m1b_hydroclimate.py, m1b_redundancy.py, m1b_evaluate.py, m1b_sensitivity.py,
  m1b_independent_check.py                      baseline dryness: build, redundancy, score, checks
  m2_latitude_basis.py, m2_feasibility.py, m2_evaluate.py, m2_conditional.py,
  m2_independent_check.py                       latitude functional form
  m3_spatial.py, m3_independent_math.py, m3_feasibility.py, m3_evaluate.py,
  m3_independent_check.py                       spatial error and spatial lag models
  m4_evaluate.py, m4_independent_check.py       final assessment, bootstraps, product check
  v2_provenance.py, v2_final_record.py          execution guards, digests, the final record
  feasibility/             coverage and package-integrity tooling for the dryness predictor
  tests/                   synthetic tests plus data-gated checks of the committed records
  outputs/                 committed tables, JSON records, verification records and figures
```

## Records

Every stage writes a result manifest with SHA-256 digests of its deterministic
artifacts, a provenance file describing the code closure and inputs, and a
run-metadata file (the one non-deterministic artifact). Each outcome-facing
stage was also reconstructed by an independently written checker that imports
neither the estimator nor the evaluator (`outputs/*_verification/*_independent_check.json`)
and rerun once with unchanged code (`outputs/*_verification/*_reproducibility.json`,
byte-identical on every deterministic artifact). The closing record is
`outputs/v2_final/v2_final_record.json`.

The provenance files and the final record carry Git commit identifiers from the
development history in which the analyses were run. The public repository is a
fresh snapshot, so those identifiers do not resolve here; they are kept as
immutable labels of the recorded runs. The records also cite, by path and in
some cases by digest, the pre-specification documents under which each stage
was scored, and they record the digests of the module bytes each run imported.
Five necessary contracts and final-record citations are retained byte-for-byte in
`evidence/`; `HISTORICAL_RECORDS.md` describes their path and digest mapping.
The current source is not byte-identical to the original scoring closure.
Changes include editorial edits, relocated evidence, fail-closed validation of
invalid distance inputs, and a repair to the M4 comparison command. Frozen
outputs have not been regenerated by these changes. Source hashes identify the
original execution; they do not certify execution of the current checkout.
`tests/test_v2_final_record.py` verifies every final citation, including mapped
evidence. `CLIMATE_RESEARCH_ARCHIVE` is an optional fallback for a separate archive.
Missing evidence or a digest mismatch fails the check.

## Reproduction

The outcome-facing evaluators refuse to run unless `PYTHONHASHSEED=0` is set,
the code they import is committed and matches the remote head, their input
manifests verify, and the output directory is fresh. They need the local
`data/processed/` artifacts of the main pipeline (see `docs/reproducibility.md`)
plus the raw grids listed below. The commands below describe the original
research checkout, including its pre-specified documents and history. Exact
scoring reruns require that preserved checkout; this publication snapshot
cannot currently satisfy the original live-upstream guard because the historical
remote branch is absent. It does not substitute missing documents with null hashes or relax the execution
guards. The application and its derived presentation bundle run independently
of that archive.

```bash
uv run python -m research.model_v2.geometry
uv run python -m research.model_v2.cells            # per-cell trend field, about three minutes
uv run python -m research.model_v2.diagnostics
uv run --locked --extra figures python -m research.model_v2.figures
uv run python -m research.model_v2.run_territory_correction     # primary spatial-CV scores
uv run python -m research.model_v2.rank_audit
OPENBLAS_NUM_THREADS=1 uv run python -m research.model_v2.product_stability
OPENBLAS_NUM_THREADS=1 uv run python -m research.model_v2.product_stability --build-aligned
uv run python -m research.model_v2.m1a_geography                # about ten minutes
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 uv run python -m research.model_v2.m1a_evaluate
PYTHONHASHSEED=0 uv run python -m research.model_v2.m1b_evaluate --out <fresh dir>
PYTHONHASHSEED=0 uv run python -m research.model_v2.m2_evaluate --out <fresh dir>
PYTHONHASHSEED=0 uv run python -m research.model_v2.m3_evaluate --out <fresh dir>
PYTHONHASHSEED=0 uv run python -m research.model_v2.m4_evaluate --out <fresh dir>
uv run pytest -q research/model_v2/tests
```

Additional inputs beyond the main pipeline: the ERA5 monthly 2 m temperature
grid (`scripts/fetch_era5.py`), GSHHG 2.3.7 shorelines and ETOPO 2022 for the
area-consistent geography measurement, and CRU TS v4.10 precipitation and
potential evapotranspiration for the dryness index. The file digests each stage
expects are recorded in its manifest.
