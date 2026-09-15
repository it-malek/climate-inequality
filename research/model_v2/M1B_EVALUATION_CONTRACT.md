# M1b measurement and evaluation contract — frozen before any CRU acquisition

**Recorded 2026-09-15, amended the same day with owner decisions, and committed and pushed before any CRU TS data are downloaded, before C2 is constructed, and before any M1b model is fitted.** The design was approved by the owner and pushed as `da4455d`.

Nothing in this file may be revised after C2 values, the redundancy diagnostic or any M1b score have been seen. Any change must be a new, dated, separately justified pre-score record, with the original rule still reported.

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
