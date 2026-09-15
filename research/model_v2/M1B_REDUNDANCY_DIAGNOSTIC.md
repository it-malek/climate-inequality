# M1b pre-outcome redundancy diagnostic — C2 against the M0\* predictors

**Recorded 2026-09-15. Status: pass. No warming outcome was loaded, and M1b has not been scored.**

This diagnostic describes how much of C2's cross-country variation the frozen M0\* predictor design already spans. It is computed from predictors only (contract §3, Amendments 2 and 3). It says nothing about whether C2 predicts warming, and it is not a verdict condition beyond its hard stops.

## Inputs and provenance

* **C2 package:** frozen at `222e2c1`, built at `58e64bf` under gate `Amendment 3`.
  * manifest SHA-256 `1974686199b9c7960ead2b105e6225f34b9b6fb33627858d2bc15239d7de4c9c`;
  * C2 features SHA-256 `cf33ba7cfb87c2184a94df5e5686f0951adf692819af8de27329ea698d72e1f4`;
  * `verify_package` passed before C2 was read, and C2 was parsed from the verified bytes.
* **Predictor inputs:** asserted against the frozen records before any read.
  * `data/processed/country_inequality.parquet` `74bbad76…`, `data/processed/city_features.parquet` `7d4a95fa…` and `data/raw/worldbank/world-bank-income-groups.csv` `79569499…`, as recorded at `9fb56e1`;
  * `outputs/m1a_geography_qa.csv` `bdc4f1ca…`, as at `a7dea34`, which supplies the frozen identifiers and row order.
* **Columns read.** Only allow-listed predictor columns were read from those parquet containers; the containers also hold outcome columns, but no target, fitted, residual or score column was requested.
  * Country table: `Country`, `owid_country`, `continent`, `cumulative_co2_mt`, `population`.
  * City features: `Country`, `abs_latitude`, `hemisphere`, `coast_km`, `elevation_m`, `koppen`, `station_density`.
* **Sample.** The 151 frozen countries, selected by identifier, with no outcome-dependent filtering. The outcome-free frame reproduces the frozen M0\* predictor columns and design exactly; this is tested separately in `test_m1b.py`, and that test never involves C2.
* **Design identity:**
  * ordered predictor frame SHA-256 `b741f2a0…a738`;
  * design matrix SHA-256 `5d3703e7…24c4`, 20 columns;
  * C2 vector SHA-256 `dd951521…2d47`;
  * redundancy code SHA-256 `a7155ec5…ec4`.
* **Instrumentation** (`m1b_redundancy_instrumentation.json`), recorded during the actual run:
  * parquet reads were exactly the two allow-lists, and no forbidden column was requested;
  * CSV reads were exactly the verified C2 bytes, `m1a_geography_qa.csv[iso3, Country]` and the income table;
  * all file opens fell within the package, the pinned inputs and the output record;
  * `m0.load_inputs`, `m0.m0_complete_design` and `build_country_design` were never called, and there is no `m0_countries.csv` dependency.
* **Instrumentation scope.** The audit hook sees Python-level opens under the repository. The parquet column spy is the primary guard. The instrumentation script (`feasibility/m1b_redundancy_instrumented.py`) is pinned by SHA-256 in its record.
* **Reproducibility.** Records are in `m1b_process_evidence.json`.
  * A second run in a separate process (`PYTHONHASHSEED` 91) reproduced `m1b_redundancy_diagnostic.json` byte for byte (`9acb680c…`).
  * The instrumented rerun wrote the same bytes.

## Diagnostics

| Quantity | Value |
|---|---|
| Design columns / rank without C2 | 20 / 20 |
| Rank with C2 | 21 |
| R²(C2 ~ M0\*) | 0.650191 |
| Adjusted R² | 0.599456 |
| VIF of C2 | 2.8587 |
| Residual SD of C2 | 0.3429 (C2 mean −0.2227, SD 0.5418) |
| Condition number, raw: without → with C2 | 17,827.5 → 17,844.3 |
| Condition number, unit-column-scaled: without → with C2 | 100.83 → 102.08 |

**R² of C2 on each group alone:**

| Group | R² |
|---|---|
| Geography (six features) | 0.6215 |
| Responsibility (log total CO₂) | 1.33e−5 |
| Socioeconomic (income group) | 0.0353 |
| Population (log population, station density) | 0.0661 |

**Correlations with C2:**

| Predictor | Pearson | Spearman |
|---|---:|---:|
| abs_latitude | 0.020 | −0.051 |
| elevation | −0.153 | −0.271 |
| continentality | −0.166 | −0.257 |
| log10 total CO₂ | 0.004 | −0.007 |
| log10 population | −0.072 | −0.118 |
| station_density | 0.194 | 0.161 |

**Numerical caveat.** Rank correlations can depend on floating-point ties in the aggregated predictors.
* An independent recomputation matched every other recorded statistic to ≤ 4e−15 relative.
* It changed the Spearman correlation with `abs_latitude` by about 4e−4 (−0.0512 vs −0.0516) when country means were summed with NumPy instead of the frozen pandas aggregation, because float rounding decides which tied latitudes tie.
* The recorded value uses the frozen M0\* aggregation.

**Category means of C2 (n, mean, SD).**
* **Köppen:**
  * A: 54, 0.037, 0.297;
  * B: 29, −1.006, 0.526;
  * C: 37, −0.202, 0.446;
  * D: 30, 0.021, 0.165;
  * E: 1, 0.418, SD undefined.
* **Hemisphere:** N 122, −0.243; S 29, −0.138.
* **Continent:** Africa 47, −0.427; Asia 38, −0.401; Europe 38, 0.040; North America 14, 0.031; Oceania 3, −0.012; South America 11, −0.023.
* **Income group:** high 47, −0.105; upper-middle 41, −0.208; lower-middle 40, −0.269; low 23, −0.410.

The complete values are in `outputs/m1b_redundancy_diagnostic.json`.

## Stops and flags

* **Hard stops:** none.
  * C2 is finite, with non-zero variance.
  * Adding C2 raises the design rank.
  * R² is well below 1 − 1e−10.
* **Near-redundancy flag (R² > 0.9):** not set.
* **Condition numbers:** diagnostic only.

## What this does and does not show

* **What it shows.**
  * About 65% of the cross-country variance in C2 lies in the span of the M0\* design, mostly the geography block, which alone reaches R² 0.62.
  * C2 is not mathematically redundant with M0\*, and adding it changes the design's conditioning only slightly.
* **What it does not show.** It does not show that C2 carries information about warming, that M1b will generalize, or any mechanism.
  * C2 is CRU-reconstructed 1920–1949 baseline hydroclimatic dryness over resolved support, not a predictor built only from information available before 1950.
  * Nothing here was tuned: C2 was frozen before this diagnostic ran.

**STOPPED BEFORE M1b SCORING.** The package and this record would satisfy `scoring_gate`, so scoring is blocked in code as well as by procedure: `m1b_evaluate.main()` refuses to run while `SCORING_ENABLED` is `False`. Before scoring, two things are required:
* owner review of this pre-outcome package;
* implementation and review of the per-capita representation that A3.6 requires with any M1b score.
