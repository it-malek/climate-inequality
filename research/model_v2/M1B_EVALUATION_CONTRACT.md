# M1b measurement and evaluation contract — frozen before any CRU acquisition

**Recorded 2026-09-15, amended the same day with owner decisions, and committed and pushed before any CRU TS data are downloaded, before C2 is constructed, and before any M1b model is fitted.** The design was approved by the owner and pushed as `da4455d`.

Nothing in this file may be revised after C2 values, the redundancy diagnostic or any M1b score have been seen. Any change must be a new, dated, separately justified pre-score record, with the original rule still reported.

**Amended 2026-09-15 before any amended C2 value, redundancy diagnostic or M1b score existed.** Amendment 1 (end of this file) changes the following.

For structural CRU land-mask cells only:
* the §2.2 prohibition on neighbour and fill values (the prohibition on floors stays);
* the §2.3 country formula and the §2.4 coverage formula, which now sum over native-valid and harmonized cells;
* §2.4's prohibition on fill and nearest-cell substitution, and §6's prohibition on alternative coverage rules. Amendment 1's rule is the only permitted coverage rule.

Globally:
* the §2.6 QA column list and output set, where byte-identity covers the deterministic outputs and not the run-metadata file;
* two added fail-closed stops (A1.7);
* the manifest's provenance fields (A1.10).

The original text is kept below for the audit trail. The evidence is in `M1B_FEASIBILITY_AUDIT.md`.

**Hypothesis:** M1b = M0\* + baseline hydroclimatic dryness (C2).
**Schematic model:** `y = G + H_hydroclimate + R + S + P + error`.

## 1. Sources and acquisition (not yet performed)

### 1.1 Files proposed for acquisition

The source is CRU TS v4.10 (Climatic Research Unit, University of East Anglia; released 2026-06-25; covers 1901–2025; no DOI at review). Base URL: `https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.10/cruts.2604091129.v4.10/`

| Role | Path | Bytes (HTTP Content-Length at review) | Last-Modified at review |
|---|---|---:|---|
| Precipitation (includes the `stn` station-count variable) | `pre/cru_ts4.10.1901.2025.pre.dat.nc.gz` | 698,006,393 | Sun, 07 Jun 2026 00:06:19 GMT |
| Potential evapotranspiration | `pet/cru_ts4.10.1901.2025.pet.dat.nc.gz` | 72,951,284 | Sun, 07 Jun 2026 09:36:47 GMT |

Documentation already read:
* `Release_Notes_CRU_TS_4.10.txt` (2,086 bytes, SHA-256 `fb3c8cb596c6fe00ca4ee26ec4d5889b7037250013d5139f2bbab641e0180c42`);
* Harris, Osborn, Jones & Lister (2020), *Scientific Data* 7:109, doi:10.1038/s41597-020-0453-3 (open accepted manuscript).

**Why full-length files, not decadal.** The v4.10 release notes state that decadal files "may not be archived when superseded". The 1920-01 to 1949-12 subset is selected in code from the full-length files, which are the reproducible archival unit.

### 1.2 Acquisition rules

1. **Storage.** Files are stored under gitignored `data/raw/cru_ts_4.10/`.
2. **Hash pin.** No published checksum was found in the CRU directory. SHA-256 of each `.gz` and of each decompressed `.nc` is computed at acquisition and becomes the pin. It is recorded in the measurement manifest and committed before any C2 value is examined.
3. **Match check.** Acquisition must reproduce the byte sizes above. A size mismatch, or a Last-Modified change, stops the work for review. There is no silent substitution of another release, mirror (CEDA) or variable.
4. **Inside-file checks.** These must hold or the work stops:
   * variables `pre` (with `stn`) and `pet`;
   * units `mm/month` (pre) and `mm/day` (pet);
   * latitude centres −89.75…89.75 and longitude centres −179.75…179.75 at 0.5° steps;
   * monthly time axis;
   * version metadata consistent with 4.10.
5. **Licence and citation.** The terms stated at acquisition are recorded.

### 1.3 Provenance assessment (how PRE and PET are built)

Taken from Harris et al. (2020) and the v4.10 release notes (the interpolation algorithm is unchanged since v4.03 and the process since v4.09).

**PRE (primary variable, mm/month).**
* **Anomalies.** Each station series is converted to monthly percentage anomalies relative to that station's 1961–1990 means. A station needs at least 75% of 1961–1990 observations per calendar month to be anomalised.
* **Interpolation.** Anomalies are interpolated onto the 0.5° land grid by angular-distance weighting, using up to eight stations within a 450 km correlation decay distance.
* **Cells without stations.** Cells with no station in range receive a zero anomaly, i.e. the climatology.
* **Absolute values.** The gridded anomalies are converted to absolute values with the published CRU CL v1.0 1961–1990 climatology. The dataset has no missing land values by construction.
* **Station counts.** The NetCDF `stn` variable gives the number of contributing stations (0–8) per cell-month; 0 means climatology inserted.

**PET (derived variable, mm/day).**
* **Formula.** FAO Penman–Monteith, computed from gridded absolute mean temperature, minimum and maximum temperature, vapour pressure and cloud cover, plus a static (temporally invariant except the annual cycle) 1961–1990 wind climatology.
* **Synthetic inputs.** Vapour pressure and cloud cover include synthetic station estimates derived from temperature and diurnal-range observations.
* **No station counts.** PET has none; the paper states no meaningful count can be produced.

**Implications for C2.**
* **A pre-outcome temporal window, not a pre-1950-information predictor.** 1920–1949 is a pre-outcome *temporal window*. C2 is **not** a predictor constructed exclusively from information available before 1950.
* **How the 1920–1949 values are built.** They combine 1920–1949 station anomalies, where stations exist and also have 1961–1990 normals, with the dataset's 1961–1990 climatological construction. That climatology is used exactly where stations are absent.
* **PET inherits this construction** for each of its inputs, and adds a static 1961–1990 wind field.
* **Anchoring.** C2's spatial pattern is therefore partly anchored to climatological normals that lie inside the outcome window.
* **Not target leakage:** C2 uses no Berkeley data and no 1950–2013 trend. It does not disqualify the dataset.
* **Limits the wording:** C2 must not be described as an independently observed pre-1950 state.
* **Handling:** the limitation is quantified by the PRE station-support QA (§2.5) and reported with any result. PET support is not quantifiable from the pinned files, and that is recorded as a limitation.

## 2. C2 construction (frozen formula)

### 2.1 Time selection
* **Months:** t ∈ 1920-01 … 1949-12, exactly T = 360 unique year-months (assert).
* **Days in month:** N_t uses the Gregorian calendar, including 29 days in February of 1920, 1924, 1928, 1932, 1936, 1940, 1944 and 1948.

### 2.2 Per-cell climatology
For each 0.5° CRU cell j:

```
P̄_j  = (1/30) · Σ_t PRE_{j,t}                 [mm yr⁻¹]   (PRE in mm/month)
Ē_j  = (1/30) · Σ_t PET_{j,t} · N_t           [mm yr⁻¹]   (PET in mm/day)
AI_j = P̄_j / Ē_j
x_j  = log10(AI_j)
```

**Valid cell:** all 360 PRE and all 360 PET values are finite and not the file's fill or missing value, **and** P̄_j > 0 **and** Ē_j > 0. Otherwise the cell is invalid, and no floor, fill or neighbour value is substituted.

*(Amendment 1: for structural CRU land-mask cells only, the neighbour and fill prohibition is replaced by the one-ring native-donor rule. No floor is ever substituted. Native-valid cells and their values are unchanged.)*

### 2.3 Country aggregation over terrestrial support

* **Terrestrial area.** A_{c,j} = Σ_{g ∈ c, g ⊂ j} L_g, where:
  * L_g is the frozen M1a terrestrial area (km²) of GPW 0.25° cell g, from `outputs/m1a_cell_land_area_km2.npz` at `a7dea34`;
  * g is assigned to country c by GPW v4 rev11 band 11 (the frozen territorial ontology);
  * each 0.5° CRU cell contains exactly 2 × 2 GPW cells (assert that edges align).
* **Consistency check.** Σ_j A_{c,j} must equal `terrestrial_area_km2` in `outputs/m1a_geography_qa.csv` to a relative 1e−9 (assert).
* **Country value.**

```
C2_c = Σ_{j valid} A_{c,j} · x_j  /  Σ_{j valid} A_{c,j}
```

*(Amendment 1 A1.6: the sums run over native-valid and harmonized cells.)*

* **Primary definition.** The mean of cell-level log aridity weighted by terrestrial area, i.e. the log of the area-weighted geometric mean of AI_j.
* **Why cell level.** The mechanism is local (land–atmosphere coupling), and the outcome is itself an area mean of cell-level trends. A national ratio of pooled precipitation and PET volumes would weight wet regions' water volumes, not the typical local regime.
* **The pooled national ratio is not computed at any point in M1b, not even as a sensitivity.** It represents a different, integrated water-budget concept.
* **Equivalence (owner-approved).** C2 is the area-weighted mean of local log aridity, which equals the log of the area-weighted geometric mean of local P/PET ratios.

### 2.4 Coverage gate (threshold unchanged from the M1a numeric gate)

```
coverage_c = Σ_{j valid} A_{c,j} / Σ_j A_{c,j}  ≥  0.98
```

**Any failure stops the work** before the redundancy diagnostic and before any fit:
* the country, achieved coverage, missing area and cause (CRU land-mask or small-island resolution, invalid P̄ or Ē) are reported for owner review;
* no threshold relaxation, fill, nearest-cell substitution, source change, country removal or reduced-sample fit.

*(Amendment 1 A1.6–A1.7: coverage counts native-valid and harmonized area. The 0.98 threshold and the stop are unchanged. Fill and nearest-cell substitution are permitted only through the one-ring structural-mask rule.)*

### 2.5 Pre-outcome support-quality checkpoint (saved and committed before the outcome is loaded)

For each country, over valid cells, weighted by A_{c,j} and averaged over the 360 months:
* **station-supported share:** the proportion of land-months with PRE `stn` ≥ 1;
* **mean PRE station count;**
* **pure-climatology share:** the proportion of terrestrial area whose `stn` = 0 in all 360 months (zero-anomaly climatology fallback throughout).

**Also recorded:**
* the country-level distribution of the three quantities (min, 10th, 25th, 50th, 75th and 90th percentiles, max);
* the ten most extreme countries on each;
* any `stn` value outside the documented 0–8 range, or missing.

**Handling rules.**
* **No exclusion threshold.** No station-count exclusion threshold is defined, and none is created afterwards. These quantities never exclude, reweight or adjust a country.
* **Qualitatively pathological case → stop before the outcome is touched.** A case is pathological if it makes the "1920–1949 baseline" description misleading. Declared now as definitional (not tuned) cases:
  * a country whose entire terrestrial area is pure climatology for all 360 months (pure-climatology share = 1), so its C2 is exactly a 1961–1990 climatological value;
  * `stn` data that are internally invalid (out of range or missing).
* **Other judgement cases.** Any other pathology identified in the audit is reported and stops the work for owner review. The work is not continued under a post-hoc rule.
* **PET limitation.** PET has no meaningful station-count field. This remains an explicit, unquantified limitation.

### 2.6 Determinism and outputs
* **Determinism.** C2 is built twice; the outputs must be byte-identical.
* **Outputs** (only `m1b_`-prefixed research files):
  * `outputs/m1b_hydroclimate_features.csv` (iso3, Country, `baseline_dryness`);
  * `outputs/m1b_hydroclimate_qa.csv` (P̄/Ē area means, coverage, invalid area and reasons, §2.5 QA);
  * `outputs/m1b_measurement_manifest.json` (source hashes, checks, formula, code hash).
* **Commit.** These are committed before §3.

*(Amendment 1 A1.9 replaces the QA column list. The outputs add `m1b_harmonization_cells.csv` and the run-metadata file `m1b_build_run.json`. Byte-identity applies to the five deterministic outputs (A1.10), not to the run-metadata file, which records wall-clock times.)*

## 3. Pre-outcome redundancy diagnostic (committed before the outcome is loaded or fitted)

**Inputs.** The frozen M0\* design with the `warming_trend` column **removed** (assert absent), and C2.

**Reported:**
1. **Full-design redundancy.** R², adjusted R² and VIF = 1/(1 − R²) of OLS `C2 ~ X_{M0*}`, where X_{M0*} is exactly the 20-column M0\* encoded design including the intercept; plus the residual SD of C2.
2. **Geography-block redundancy.** R² of `C2 ~ intercept + geography block` (the six features, encoded).
3. **Per-group redundancy.** R² of C2 on each other group alone: responsibility, socioeconomic, population.
4. **Correlations.** Pearson and Spearman correlations of C2 with `abs_latitude`, `elevation`, `continentality`, log10 total CO₂, log10 population and `station_density`.
5. **Category summaries.** Count, mean and SD of C2 by `climate_zone`, `hemisphere`, `spatial_block` and `income_group`.

**Decision rule (mechanical, fixed now):**
* **Mathematical redundancy** — rank([X_{M0*}, C2]) = rank(X_{M0*}), or R² ≥ 1 − 1e−10: stop for a pre-score redesign decision.
* **Data defect** — any non-finite C2, or zero variance: stop for review.
* **Otherwise:** proceed regardless of R², with **no tuning or redefinition of C2**, however inconvenient its correlation with Köppen, latitude or other predictors. R² > 0.9 is flagged "near-redundant" only to interpret a non-supported result as possible non-identifiability.

## 4. M1b evaluation

### 4.1 Models
* **Comparator M0\*.** The approved primary full-rank representation: drop `cum_co2_per_capita`; station geography; 151 countries in frozen row order. It must reproduce the rank-audit `drop_per_capita` card to 1e−10 (in-sample R² 0.6364460028, CV R² 0.1530799130, CV RMSE 0.0428725807; assert).
* **M1b.** M0\* + `baseline_dryness`:
  * a numeric linear column with no further transform (it is already log10);
  * a new declared group `hydroclimate`, never folded into geography;
  * implemented as a research-only schema extension that reproduces M0\* exactly when the column is absent.
* **Encoders.** Train-only encoders and the mean-training-level-effect rule are unchanged.
* **Reproducibility.** Scoring runs with `PYTHONHASHSEED=0` so outputs are byte-reproducible.

### 4.2 Protocols and reported scorecard
* **Protocols:**
  * primary: corrected 500 km footprint-certified LOCO, with lower bounds from `4263429`, never recomputed;
  * secondary: leave one M49 subregion out;
  * reference only: random 10-fold (seed 0).
* **Reported for both models:**
  * in-sample R²;
  * spatial-CV R², RMSE and MAE;
  * M49 R² and RMSE;
  * random-reference R² and RMSE;
  * residual Moran's I in-sample and out-of-fold (V1 station-centroid kNN8, 999 permutations, seed 0), with area-centroid I recorded;
  * group shares (geography, responsibility, socioeconomic, population, hydroclimate, residual);
  * fold RMSE median/min/max and worst fold country;
  * IQR of \|OOF error\|;
  * per-region RMSE and bias, and the worst region with n ≥ 3;
  * unseen-level rows by feature;
  * calibration slope and intercept;
  * matrix rank;
  * coefficient and sign stability.
* **Paired comparison:**
  * ΔRMSE with the frozen **paired resampling interval** (below);
  * ΔR² in-sample and CV;
  * ΔI;
  * share changes;
  * per-country change in absolute OOF error, with counts improved and worsened.

### 4.3 Uncertainty interval (the existing frozen method, retained for comparability)

The interval comes from `run_territory_correction.paired_rmse_interval`, unchanged:
* **Resampling.** The 151 countries are resampled with replacement, 2,000 times, seed 0.
* **Pairing.** The same resampled country indices are used for both models' fixed out-of-fold errors (the predictions are not refitted).
* **Statistic.** RMSE(M1b) − RMSE(M0\*) on each resample.
* **Interval.** The 2.5th and 97.5th percentiles.

This is a **paired resampling interval** for the difference in out-of-fold error. It treats countries as exchangeable and does **not** resolve spatial dependence among countries. The primary safeguards against geographic overoptimism remain:
* the corrected 500 km spatial CV;
* the M49 holdout;
* the worst-region veto.

No other uncertainty method is introduced.

### 4.4 Two-level verdict (frozen before C2 or any warming-model result is seen; one hypothesis, so no multiplicity adjustment)

**Level 1 — linear hydroclimate association supported.** Requires both:
* **S1:** paired spatial-CV ΔRMSE < 0 with the §4.3 interval entirely below zero;
* **S2:** the `baseline_dryness` coefficient is negative in the full fit and in ≥ 80% of the 151 primary training fits.

The M49 and worst-region results are essential robustness evidence. They are reported prominently alongside Level 1, but they are not Level-1 conditions.

**Level 2 — M1b promoted to the primary V2 baseline.** Requires all of A1–A5:

| # | Condition |
|---|---|
| A1 | ΔRMSE = RMSE(M1b) − RMSE(M0\*) < 0, and the upper bound of its 95% country-bootstrap interval is < 0 |
| A2 | Practical requirement: ΔRMSE ≤ −0.002 °C/decade |
| A3 | M49 secondary RMSE(M1b) − RMSE(M0\*) ≤ 0.001 °C/decade |
| A4 | Worst-region RMSE (n ≥ 3) of M1b − that of M0\* ≤ 0.001 °C/decade |
| A5 | Pre-stated sign: the `baseline_dryness` coefficient is < 0 in the full fit and in ≥ 80% of the 151 primary training fits |

* **A1 and A3–A5** are `EXPERIMENTAL_PLAN.md`'s genuine-improvement and covariate-retention definitions, unchanged.
* **A2** is the plan's only frozen practical magnitude: the 0.002 °C/decade threshold in the M4 stopping rule. Here it is applied as the practical RMSE-improvement requirement. It does not weaken any earlier condition.
* **Verdicts (exactly one is reported):**
  * **supported and promoted:** A1–A5 all hold;
  * **supported but sub-material / not promoted:** S1 and S2 hold, but A2, A3 or A4 fails;
  * **not supported:** S1 or S2 fails.
* **Improvement without the pre-stated sign.** If predictive performance improves but S2 fails, the result is **not** support for the dryness mechanism, and it is not promoted.
* **Separately reported:** "worsened generalization" if ΔRMSE > 0 with the lower interval bound > 0.

### 4.5 Diagnostics reported (not verdict conditions)
* **Overfitting signal** (plan definition): the in-sample R² gain exceeds the CV R² gain by > 0.05, or OOF RMSE worsens while in-sample RMSE improves.
* **Residual spatial structure:**
  * reduced if in-sample and OOF I both fall by ≥ 0.05;
  * increased if both rise by ≥ 0.05;
  * otherwise essentially unchanged.
* **Scientific stability:** a material change is geography no longer the largest named group, or the responsibility share > 0.10. The hydroclimate share is reported.
* **Three-dimension summary**, as in `M1A_EVALUATION_CONTRACT.md` §4: transfer, spatial specification, stability.
* **Context:** the redundancy diagnostic (§3) and the provenance QA (§2.5) are reported alongside.

### 4.6 Interpretation (fixed now)
* **Supported (either level).** Baseline hydroclimatic dryness carries transferable, directionally pre-specified out-of-fold information about cross-country differences in area-weighted warming beyond M0\*, consistent with moisture-limited amplification.
  * This is associational, not causal.
  * It is subject to the CRU provenance limitation (§1.3) and the registered sensitivities (§5).
  * "Sub-material / not promoted" means the information is reproducible in direction but falls below the plan's practical improvement threshold, or fails a geographic robustness veto. M0\* then remains the primary baseline.
* **Not supported.** The pre-registered **linear** C2 association is not demonstrated at n = 151 under the frozen criteria.
  * This is **not** evidence that hydroclimate is irrelevant.
  * It may reflect the geography block already carrying the signal (§3), climatological anchoring or noise in C2, limited power, or a non-linear relationship.
  * Non-linear dryness effects belong to M2 only if independently justified later, and may not be introduced in response to this score.
* **Responsibility statement unchanged.** The narrow responsibility statement stays as before: explained variance of cross-country differences in area-weighted warming, not greenhouse gases' physical role.

## 5. Registered sensitivities (only if the linear association is supported, at either level; after the M1b result is committed)

**Order.** First commit the M1b result. Then, if S1 and S2 hold, replicate M1b, with the identical C2, under:
1. the frozen M1a area-consistent geography measurements, in place of station geography, for both comparator and M1b;
2. the per-capita responsibility representation;
3. the preprocessing-aligned ERA5 outcome, with the comparator refitted on the same outcome.

**Use.** Results are reported side by side and cannot change either verdict level. A gain confined to Berkeley is flagged product-sensitive. If the association is not supported, no sensitivity runs are performed.

## 6. Not permitted at any point
* Alternative dryness windows, transforms (for example unlogged AI or PET-free precipitation), datasets (other CRU versions, GPCC, TerraClimate, CGIAR), thresholds, aggregation orders (including the pooled national P/PET ratio) or coverage rules.
  *(Amendment 1 replaces the §2.4 coverage rule with its structural-mask rule. No other coverage rule is permitted.)*
* Station-count exclusion thresholds, or any post-hoc support rule.
* Changing M0\*, M1a outputs, folds, the outcome or the scorecard.
* C1, C3, joint models, nonlinear or interaction terms, or spatial terms.
* Using ERA5 or residual clusters to motivate changes.

## 7. Commit sequence after this contract

The owner approved acquisition once this amended contract is on the remote.

1. **Acquire and pin.** Push this contract, then acquire the §1.1 files and pin the SHA-256 of the compressed and decompressed files.
2. **Build and check.** Construct C2 twice (determinism), then run the file, grid and unit checks and the coverage gate. Stop on any failure.
3. **Pre-outcome checkpoint.** Run the support audit (§2.5) and the redundancy diagnostic (§3) **without loading the warming outcome**. Commit the measurements and both pre-fit records. Stop on any hard stop or pathology.
4. **Score.** If all hard stops pass, score M1b under §4, then commit the result and report.
5. **Sensitivities.** Only if the linear association is supported: run §5 and commit.

## Amendment 1 — structural CRU land-mask harmonization of C2 support

**Adopted 2026-09-15, before any amended C2 value was computed, before the redundancy diagnostic, and before any M1b model was fitted or scored.**

Order of events:

1. The measurement and evaluation contract was frozen and pushed (`60111ae`).
2. The builder was implemented (`ef09196`), and its station-count validation was corrected to run over valid cells (`c6ef919`).
3. The first valid C2 build failed the §2.4 coverage gate in three countries. After that failure, the project owner approved in principle a pre-outcome, global harmonization rule for structural land-mask cells.
4. The failed build was preserved and diagnosed in `M1B_FEASIBILITY_AUDIT.md` (`c5ff9b5`). That audit's statement that the amendment "is recorded and committed separately" anticipated this record.
5. The builder's input checks were hardened without any rule change (`81c5dca`):
   * the source SHA-256 pins and the GPW/CRU grid nesting are now asserted;
   * country identifiers now come from the outcome-free M1a support record instead of `m0_countries.csv`, which also carries warming, fitted and residual columns. The original build had read only that file's `Country` and `iso3` columns.
6. This text, its implementation and tests were reviewed, committed and pushed before the amended rule was computed for any country. The rule is the owner-approved one.

**Outcome boundary, stated precisely.**
* **What was never used.** C2 construction, the feasibility audit and this amendment use no warming outcome, fitted value, residual or score. No C2 value has been joined with, compared with or correlated with any of them.
* **What the test suite loads.** This protocol runs the repository's unit-test suite. Some of its existing tests load the frozen V1 design, which contains the warming outcome, to verify M0/M0\* reproduction and the evaluator's schema. Those tests never involve C2 values.
* **What already exists.** Original-rule (native-only) C2 values for all 151 countries were committed with the feasibility audit (`c5ff9b5`). This rule was fixed from the mask diagnostics, not from those values.
* **What has not run.** The §3 redundancy diagnostic has not been run. Before it runs, its inputs are restricted to predictor source columns, and that restriction is recorded separately.

**Why it was needed.** Under §2.4, 148 of 151 countries passed. The failures were Bahamas (0.8865), Panama (0.9728) and the Philippines (0.9761). The audit established the cause from the raw files:
* CRU's 0.5° PRE and PET land masks are constant over all 1,500 months;
* every invalid terrestrial support cell is fill in every month of PRE, PET or both;
* those cells hold small land fragments that the finer M1a support resolves;
* no invalid support cell has partial temporal missingness or a non-positive climatological mean.

The failure is therefore a mismatch between the source's land mask and the frozen terrestrial support. It is not missing observations and not a defect.

**Constraints.**

* **Nothing here was chosen from results.** No rule was chosen from C2 values, the outcome, residuals, redundancy, any model score or any amended coverage value.
* **One global rule.** It applies identically to every cell and all 151 countries, with no country-specific exception.
* **Unchanged:**
  * the source files and pins, and the 1920-01…1949-12 window;
  * the per-cell formula and native validity (§2.2);
  * the M1a terrestrial support and area weights (§2.3);
  * the 0.98 coverage threshold;
  * the station-support audit and its stops (§2.5);
  * the redundancy decision rule (§3);
  * all of §4: models, protocols, interval, verdict levels and criteria.
* **Changed, for structural CRU land-mask cells only:**
  * §2.2: the prohibition on neighbour and fill values is replaced by the one-ring native-donor rule. No floor is ever substituted.
  * §2.3 and §2.4: the country formula and the coverage formula sum over native-valid and harmonized cells (A1.6).
  * §2.4 and §6: fill and nearest-cell substitution are permitted only through this rule, which becomes the only permitted coverage rule. Threshold relaxation, source change, country removal and reduced samples remain prohibited.
* **Changed globally:**
  * §2.6: the QA column list and output set are replaced by A1.9. Byte-identity covers the deterministic outputs only (A1.10).
  * the manifest's provenance fields (A1.10).
* **Added fail-closed stops (A1.7).** These can only halt work for owner review: any non-structural invalid support cell, and any country with no native-valid area.

### A1.1 Native cells (unchanged)

A cell is **native-valid** under §2.2:
* all 360 PRE and PET values are finite and not fill;
* P̄_j > 0 and Ē_j > 0.

Its value x_j = log10(P̄_j / Ē_j) is never altered, replaced or averaged.

### A1.2 Structural-mask target cells

Each variable (PRE, PET) in each cell has one state:
* **masked:** every raw value in every month of the pinned file (1901-01…2025-12, 1,500 months) equals the variable's declared `_FillValue` or `missing_value`. This is CRU's fixed land mask;
* **complete:** every decoded value in the 360 window months is finite, and the climatological mean is positive (P̄ > 0 for PRE, Ē > 0 for PET);
* **defective:** anything else, checked in this order:
  * `nonpositive_mean`: every window month finite, mean not positive;
  * `partial_fill`: fill in some window months, but not in every record month;
  * `malformed`: a non-finite window value that is not the fill value.

A cell is native-valid exactly when both variables are complete, and the builder asserts this equivalence.

A cell is a **structural-mask target** if and only if all four conditions hold:

1. the frozen M1a support assigns terrestrial area in it to at least one of the 151 countries;
2. it is not native-valid;
3. at least one variable is masked;
4. each variable is masked or complete.

Any other invalid support cell (a cell with a defective variable) is **not** a target. It stays unresolved, receives no value and is a hard stop for owner review (A1.7). For the pinned files, the audit found the masks constant over the full record and no such support cell.

### A1.3 Donor search: one ring, native donors only

* **Candidates:** exactly the eight CRU cells adjacent to the target, (i+Δi, k+Δk) for Δi, Δk ∈ {−1, 0, +1}, excluding (0, 0).
* **Longitude wraps** (index modulo 720), so the 179.75°E and 179.75°W columns are adjacent.
* **Latitude does not wrap.** Rows −1 and 360 do not exist, and no candidate crosses a pole, including the cell 180° away in the same polar row.
* **Eligibility.** A candidate is eligible only if it is native-valid (A1.1).
  * Eligibility is evaluated once, on the native grid, before any assignment.
  * A harmonized target is therefore never a candidate. No value moves more than one cell, and there is no recursive or iterative fill.
* **No political constraint.** The donor need not belong to the target's country, or to any of the 151 countries' support. The cell field is a continuous physical climate field, and national borders do not constrain it.
* **No wider search.** Ring 2, the nearest valid cell elsewhere, another source or another variable is never used.

### A1.4 Distance and ties

* **Distance.** The geodesic distance between cell centres on the WGS84 ellipsoid, using Karney's inverse geodesic (pyproj `Geod(ellps="WGS84")`).
  * This is the ellipsoid of the frozen territorial CV (`territory.py`).
  * It is tabulated once per latitude row from whole-cell offsets with a non-negative longitude step, so east/west mirror-image candidates share one stored value.
  * Rows beyond a pole have infinite distance and are never selected.
* **Why the ellipsoid.** The owner rule asks for the geodesically nearest donor, and the sphere misorders neighbours near the equator.
  * On a sphere the east/west neighbour is always nearer than the north/south one, by about 0.5 m at ±0.25° rising to about 330 m at ±6.25°.
  * On WGS84 the north/south neighbour is nearer for cell-centre latitudes within ±6.25°, by about 372 m at ±0.25° falling to about 38 m at ±6.25°. From ±6.75° east/west is nearer on both.
  * On a sphere, north and south neighbours tie exactly. On WGS84 the equatorward neighbour is nearer, by about 4 cm to 1 m within ±6.25° and by more at higher latitudes.
  * WGS84 was fixed here before any amended value, and it is not a sensitivity.
* **Selection.** The eligible candidate with the smallest distance.
* **Ties.** Exact equality occurs only between east/west mirror-image candidates (E/W, NE/NW, SE/SW), which share a stored distance.
  * A tie goes to the smaller latitude index, then the smaller longitude index.
  * The longitude index is the wrapped index 0…719, ascending from −179.75°. Away from the antimeridian a tie therefore goes west; in columns 0 and 719 the wrapped index sends it east.

### A1.5 Assigned value

* **What is transferred.** A resolved target's terrestrial area carries the donor's already-computed native x_donor, as one derived value.
* **What is never done.**
  * PRE and PET are never borrowed or interpolated separately, and never taken from different cells.
  * Donors are never averaged.
  * No trend or temporal information is transferred.
* **Shared cells.** A target cell shared by several countries has one donor and one value for all of them.
* **Unresolved.** A target with no eligible candidate receives no value.

### A1.6 Country value and coverage

For country c, let N_c be its native cells, H_c its resolved (harmonized) targets and U_c its unresolved cells:

```
C2_c       = [ Σ_{j∈N_c} A_cj · x_j  +  Σ_{j∈H_c} A_cj · x_donor(j) ]  /  Σ_{j∈N_c ∪ H_c} A_cj
coverage_c =   Σ_{j∈N_c ∪ H_c} A_cj  /  Σ_j A_cj   ≥  0.98   (threshold unchanged)
```

The area partitions exactly: native + harmonized + unresolved = terrestrial area. It is asserted to a relative 1e−10, which bounds float64 summation error over at most ~10⁵ cell addends.

### A1.7 Hard stops

Any one of these stops the work before §3. Every comparison is written so that a non-finite value triggers the stop.
* coverage_c < 0.98 for any country;
* any non-structural invalid support cell (A1.2) *(added)*;
* any country with no native-valid terrestrial area, since its §2.5 station audit is undefined *(added)*;
* the unchanged §2.5 stops: pure-climatology share = 1, or invalid `stn` in a native cell;
* any non-finite C2;
* source size or SHA-256 differing from the pins, M1a support files differing from `a7dea34`, failed GPW/CRU grid nesting, or country area not reproducing M1a to 1e−9.

If a country remains below 0.98 after this rule, it is reported for owner review. No farther search, threshold change, source change, country removal, reduced sample or further amendment is introduced to force a pass.

### A1.8 Station-support audit and wording

* **Unchanged audit.** The §2.5 quantities are computed over native-valid cells exactly as before.
* **Harmonized fragments.** They carry no station information of their own and are reported as area, not as station support.
* **Wording.** C2 is **CRU-reconstructed 1920–1949 baseline hydroclimatic dryness**. It is never described as purely observed pre-1950 dryness, or as built only from information available before 1950.

### A1.9 Reported QA

**Per country** (`m1b_hydroclimate_qa.csv`):
* total, native-valid, harmonized and unresolved terrestrial area and their fractions, with unresolved area split into structural and non-structural;
* cell counts by status;
* unweighted mean, area-weighted mean and maximum donor distance;
* cross-border harmonized cells and area;
* C2 over native support only (the original-rule value; NA with no native area), the harmonized C2, and their difference;
* P̄ and Ē native-area means;
* the §2.5 quantities.

**Per target cell and country** (`m1b_harmonization_cells.csv`):
* target and donor indices and centres;
* the PRE and PET states (A1.2);
* donor distance and donor C2;
* the donor cell's support countries (`none` when it supports none) and a cross-border flag;
* the country's terrestrial area in the cell;
* status and reason (`structural_cru_mask` or `non_structural_invalid`).

Unresolved rows carry NA donor fields. Cross-border rows are kept.

**Global** (manifest):
* the number of target, harmonized and unresolved cells;
* total harmonized and unresolved area;
* donor-distance minimum, median, mean, 95th percentile and maximum over distinct harmonized target cells (NumPy linear interpolation);
* cross-border counts;
* the PRE and PET state counts over the support.

**Use.** The coverage, area and non-structural columns feed the A1.7 stops. The native-only C2, the harmonization difference and the donor-distance statistics are reported only for transparency. They are not an alternative definition and are never scored.

### A1.10 Provenance and determinism

* **Builder checks.** It asserts the source pins and grid nesting (`81c5dca`), and the SHA-256 of the M1a land-support files at `a7dea34`.
* **Manifest contents:**
  * the git commit, the commit that last changed this contract, and whether the construction code path (builder, contract, and the M1a support and GPW modules it imports) matches the commit;
  * source sizes and SHA-256, and the CRU version and run ID;
  * the window, formula and amendment identifiers;
  * the land-support file hashes and commit, the GPW grid and national-identifier lookup hashes, and the ordered country-list hash;
  * the builder hash and software versions, including pyproj and PROJ.
* **Run metadata.** Wall-clock metadata goes to a separate file, so the deterministic outputs can be compared byte-for-byte.
* **Two builds.** After this amendment is pushed, C2 is built twice, in separate processes and into separate output roots. The deterministic outputs must be byte-identical.
