# M1a — area-consistent geography measurement: result

**Stage verdict under the frozen contract (`M1A_EVALUATION_CONTRACT.md`, `e3e5901`):**

* **Summary: neither improved generalization nor improved spatial specification.**
* **Scientific conclusion: stable.**
* **Gate M1a → M1: M1a does not become the base.**

Every point estimate of predictive transfer is worse than the rank-only M0-equivalent comparator. The primary ΔRMSE interval covers zero, so the contract classifies transfer as "no detectable change", not as a statistically established deterioration.

The per-capita representation sensitivity gives the same scores, residuals and verdict, with different group shares.

M1a was a measurement-alignment experiment only. No physical concept, nonlinear term, interaction or spatial term was added. Outcome, sample, groups, folds and success criteria are unchanged.

## 1. Order of events (all local, branch `research/model-v2-residual-structure`)

| Commit | What | Before any M1a score? |
|---|---|---|
| `e3e5901` | Predictor contract approved; M1a evaluation contract frozen | yes |
| `ff67545` | Feasibility audit under original gates (147/151 pass; builder, evaluator, tests) | yes |
| `10e117b` | Amendment 1 (Köppen identifiability, coastal elevation support) + implementation | yes |
| `a7dea34` | Final measurements (two byte-identical builds, 151/151 valid) + pre-scoring record | yes |
| this commit | Evaluator run, unmodified since `ff67545`; scorecards and this report | the score |

## 2. Geography measurement changes implemented

Support for each country is its GPW v4 rev11 band-11 0.25° cells intersected with GSHHG 2.3.7 full-resolution land, (L1−L2) ∪ (L3−L4). Quadrature uses the ETOPO 2022 60″ pixel lattice with exact clipping and spherical area.

| Feature | V1 (station) | M1a (terrestrial area) | Median \|Δ\| vs V1 | V1–M1a correlation |
|---|---|---|---:|---:|
| `abs_latitude` | mean station \|lat\| | area mean \|lat\| (absolute before averaging) | 0.68° | 0.992 |
| `elevation` | mean station-sampled ETOPO | area mean ETOPO over centre-resolved land, coastal support completed from adjacent resolved land; bathymetry never used | 184 m | 0.684 |
| `continentality` | station mean, planar-nearest point | area mean exact great-circle distance to Natural Earth 110m shoreline, seams removed, 3′ quadrature (≤0.1 km error vs 5′/60″) | 45 km | 0.916 |
| `climate_zone` | station-modal A–E | area-dominant A–E, identified only if A1 > A2 + U | 29 class switches | — |
| `hemisphere` | station-modal N/S | majority terrestrial area | 2 switches (GAB N→S, KEN S→N) | — |
| `spatial_block` | OWID continent | unchanged | 0 | 1 |

## 3. Deviations from the original specification, and why

1. **Amendment 1, adopted before any score.**
   * **Köppen:** the ≥95% classified-coverage gate is replaced by strict identifiability, A1 > A2 + U.
   * **Elevation:** land in coastline-straddling pixels whose centres are water is completed from queen-adjacent centre-resolved pixels of the same unit. Unresolved area is never valued, and the 98% gate is unchanged.
   * **Evidence:** the original rules failed 5 gates in 4 countries (PHL, DNK, NOR, BHS) through source-resolution limitations, not defects (`M1A_FEASIBILITY_AUDIT.md`).
   * **Effect:** the only Köppen change is BHS (missing → A). The elevation change has median −0.49 m, and its extremes are listed in `M1A_MEASUREMENT_RECORD.md`.
2. **Distance quadrature.**
   * **What:** the 60″/30″ comparison in spec §6.4 was replaced by 3′ block centroids, checked against 5′ blocks for all countries and exact 60″ pixels for 10 representative countries.
   * **Why:** a proportionate method accepted by the owner before any score.
   * **Error:** max \|3′−5′\| = 0.095 km, max \|3′−60″\| = 0.034 km. The adaptive 60″ fallback was never triggered.
3. **Natural Earth seam edges.** The 5 edges on ±180° and the south-pole cap are excluded as cut lines.
4. **GSHHG repair.** One invalid GSHHG L1 polygon (id 2380) was repaired with `make_valid`.
5. **River-lakes.** GSHHG L2 river-lakes are treated as water.
6. **Exact ties.** A Köppen tie is not identified; the original alphabetical tie-break is withdrawn by Amendment 1.
7. **Evaluator reproducibility.** Group shares vary at ≤2.2e-16 between runs because V1's `group_lmg_shares` iterates `frozenset` keys. With `PYTHONHASHSEED=0` the evaluator is byte-identical; committed outputs use that setting. No code changed.

## 4. Primary contract: M0-equivalent (C0) vs M1a

Primary representation: total cumulative CO₂ plus population. 151 countries, corrected 500 km footprint-certified LOCO, rank 20 of 20 for both models.

| Metric | C0 | M1a | Δ |
|---|---:|---:|---:|
| In-sample R² | 0.636446 | 0.612171 | −0.024275 |
| Spatial-CV R² (500 km) | 0.153080 | 0.057988 | −0.095092 |
| Spatial-CV RMSE (°C/decade) | 0.042873 | 0.045215 | +0.002343 [95% CI −0.000350, +0.005264] |
| Spatial-CV MAE | 0.034465 | 0.035929 | +0.001465 |
| M49 R² / RMSE | 0.302970 / 0.038894 | 0.241136 / 0.040583 | −0.061834 / +0.001689 |
| Random 10-fold R² (reference only) | 0.428353 | 0.366148 | −0.062205 |
| Residual Moran's I, in-sample (V1 weights) | 0.268689 | 0.315303 | +0.046614 |
| Residual Moran's I, out-of-fold | 0.321547 | 0.340804 | +0.019257 |
| Area-centroid I, in-sample / OOF (recorded) | 0.289919 / 0.350153 | 0.340722 / 0.360198 | +0.050803 / +0.010045 |
| Geography share | 0.518837 | 0.487667 | −0.031170 |
| Responsibility share | 0.013134 | 0.013851 | +0.000717 |
| Socioeconomic share | 0.062821 | 0.062648 | −0.000173 |
| Population share | 0.041653 | 0.048005 | +0.006352 |
| Residual share | 0.363554 | 0.387829 | +0.024275 |
| Fold RMSE median / max | 0.030213 / 0.114735 (Brazil) | 0.030306 / 0.127233 (Tajikistan) | |
| IQR of \|OOF error\| | 0.043031 | 0.039751 | −0.003279 |
| Worst region (n≥3) | Central Asia 0.069364 | Central Asia 0.098009 | +0.028645 |
| Calibration slope | 0.609 | 0.540 | |
| Rows with unseen level (primary / M49) | 1 (Iceland, Köppen E) / 12 | 0 / 11 | |

C0 reproduces the rank-audit `drop_per_capita` card to 1e−10.

## 5. Registered per-capita representation sensitivity

C0 and M1a scores, residuals, Moran's I, folds and region errors are **identical** to the primary rows: equal column space, as established by the rank audit. Only the group shares differ:

| Share | C0 | M1a | Δ |
|---|---:|---:|---:|
| Geography | 0.513520 | 0.482446 | −0.031074 |
| Responsibility | 0.020435 | 0.021249 | +0.000814 |
| Socioeconomic | 0.069282 | 0.069505 | +0.000223 |
| Population | 0.033208 | 0.038972 | +0.005764 |
| Residual | 0.363554 | 0.387829 | +0.024275 |

The sensitivity classification is the same as the primary ("neither", stable), so it agrees with the primary conclusion.

## 6. Fold-level spatial-CV behaviour

* **Countries:** absolute OOF error improves for 72 countries and worsens for 79. Quantiles of the change are p10 −0.0111, median +0.0002, p90 +0.0143.
* **Largest improvements:** Angola −0.039, Algeria −0.038, Eswatini −0.034, Kenya −0.026, Gabon −0.022, Morocco −0.021, Egypt −0.020, Argentina −0.017.
* **Largest deteriorations:** Tajikistan +0.097, Bolivia +0.044, Jordan +0.037, Iceland +0.031, Malawi +0.030, Chad +0.028, Zimbabwe +0.024, Pakistan +0.024, Central African Republic +0.023, Uzbekistan +0.019.
* **Regions improved:** Northern Africa (−0.013), Australia/NZ (−0.011), Southern Africa (−0.008), Eastern Europe, Eastern Asia, Western Africa.
* **Regions worsened:** Central Asia (+0.029), South-eastern Asia (+0.009), Northern Europe, Western Europe, Northern America (+0.006–0.007) and the other nine subregions.
* **Dispersion:** the error IQR narrows (0.0430 → 0.0398) while the tail grows (max fold RMSE 0.115 → 0.127).
* **Large-country gate:**

| Country | C0 abs error | M1a abs error | Change |
|---|---:|---:|---|
| Canada | 0.086 | 0.097 | worse |
| Brazil | 0.1147 | 0.1148 | unchanged |
| Russia | 0.032 | 0.040 | worse |
| Algeria | 0.042 | 0.004 | much better |

  The large-country pattern is not removed.
* **Unseen levels:** the primary unseen-level count falls from 1 to 0, because Iceland's area-dominant class is no longer a singleton.

## 7. Residual spatial autocorrelation

With V1 weights, in-sample I rises by +0.047 and out-of-fold I by +0.019; both p = 0.001.

Both changes are below the frozen 0.05 materiality threshold, so the contract classification is **essentially unchanged**. The direction is an increase, not a reduction. The recorded area-centroid weights show the same direction (+0.051 in-sample, +0.010 OOF).

## 8. Group contributions

* **Geography** remains by far the largest named group, but falls by 3.1 percentage points (0.519 → 0.488 primary).
* **The residual share** rises by the same amount as the in-sample R² falls (+2.4 pp).
* **Responsibility** is essentially unchanged: 1.31% → 1.39% (primary), 2.04% → 2.12% (sensitivity), far below the 10% materiality line.
* **Population** rises slightly (+0.6 pp); **socioeconomic** is unchanged.
* **Signs:**
  * The responsibility coefficient keeps its full-fit sign in 99.3% of primary training fits.
  * Elevation sign stability rises from 0.72 to 0.90.
  * The Köppen B and lower-middle-income dummies become less stable (0.92 → 0.82, 0.97 → 0.73).

## 9. Countries whose geography representation changed most, and why

* **Latitude:** large, elongated or unevenly sampled countries.

  | Country | V1 (station) | M1a (area) | Change |
  |---|---:|---:|---:|
  | Canada | 46.8° | 58.7° | +12.0° |
  | Brazil | 19.3° | 10.7° | −8.6° |
  | Algeria | 35.2° | 28.0° | −7.2° |
  | Australia | 32.2° | 25.4° | −6.7° |
  | Russia | 53.8° | 60.3° | +6.6° |
  | Chad | 8.8° | 15.3° | +6.5° |
  | USA | 36.6° | 42.9° | +6.3° |

  Stations cluster in the populated south, the coast or the north, depending on the country.
* **Elevation:** V1's grid-snapped station samples included bathymetry. All 17 negative V1 country means become positive:
  * Jamaica −1528 → +319 m;
  * Lebanon −873 → +976 m;
  * Dominican Republic −1174 → +399 m;
  * Taiwan −342 → +768 m.

  Sampling support also changes highland countries: Bolivia 3353 → 1251 m (lowlands), Nepal 572 → 2122 m, China 453 → 1828 m, Tajikistan 1916 → 2929 m.
* **Continentality:** coastal station clustering understates interiors: Algeria +607 km, China +542, Brazil +500, Mauritania +451, Libya +412, Angola +329. Uzbekistan moves the other way (−462 km).
  * The algorithm correction alone (planar-nearest → exact arc, at the same station points) is small: median 0.04 km, but 116 km for Uzbekistan, which accounts for part of its move.
  * Almost all of the change is support.
* **Climate zone (29 switches):**
  * Stations sit in wetter or temperate zones while arid B dominates the area: Algeria, Chad, Mali, Ethiopia, Kenya, Morocco, South Africa, China, Mexico, Australia, Chile.
  * Temperate C station modes give way to D over area: Japan, Turkey, Denmark, Germany, USA.
  * Brazil, Bolivia and Guatemala move C → A, and Tajikistan D → E.
* **Hemisphere:** Gabon (34.8% of land north) and Kenya (58.0% north) are equator-crossing countries whose station modes sat on the minority side.

## 10. Interpretation under the three dimensions

1. **Predictive transfer.** There is no detectable improvement. Point estimates are worse on primary RMSE (+0.0023) and CV R² (−0.095). The M49 secondary and worst-region vetoes both fail (+0.0017, +0.029). The contract does not call this "worsened generalization" only because the country-bootstrap interval covers zero. The mechanical overfitting flag is set: the in-sample R² change (−0.024) exceeds the CV R² change (−0.095) by more than 0.05. That reflects a larger out-of-sample loss than in-sample loss, not an in-sample gain.
2. **Residual spatial structure.** Essentially unchanged by the frozen threshold, with a small increase in the point estimates. This is not improved spatial specification.
3. **Scientific stability.** Stable:
   * geography remains the largest group;
   * responsibility stays at about 1.4–2.1%;
   * the narrow statement is unchanged, namely that responsibility explains little of the cross-country differences in area-weighted warming in this descriptive decomposition, which is not a statement about greenhouse gases' physical role in warming.

**Summary classification: neither.** Better-aligned measurement of the existing geography concepts did not improve the model. This is informative. V1's station-based measurements were misaligned with the area-weighted outcome, but they were not a hidden source of transfer failure, and correcting them does not reduce residual spatial structure. One plausible reading, not tested here, is that station-sampled geography partly tracks the station-sampled construction behind parts of the gridded outcome. That is a hypothesis, not a finding.

**Gate M1a → M1 (plan):**
* M1a is not a genuine improvement.
* It is not worse by the ΔRMSE interval.
* It does not remove the large-country error pattern.

It therefore does not become the base. M0 measurements under the approved contract remain the comparison row. The plan's text does not name this exact case (neither condition met, but no genuine deterioration). This report applies the plan's first sentence literally and flags the ambiguity for the owner.

## 11. Reproducibility and validation

```sh
uv run python -m research.model_v2.m1a_geography              # ~7–11 min; deterministic
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 uv run python -m research.model_v2.m1a_evaluate
```

* **Measurement builds:** byte-identical twice, under both the original and the amended rules.
* **Evaluator:** byte-identical twice with a fixed hash seed; ≤2.2e-16 without one.
* **Comparators:** C0 reproduces the rank audit.
* Test, Ruff and diff-check results are recorded in the session report.

Artifacts:
* `outputs/m1a_scorecard.json` (complete cards, sign stability, unseen levels, classification);
* `outputs/m1a_scorecard_summary.csv`;
* `outputs/m1a_country_cv_errors.csv`.

## 12. Not done, by design

* No new physical predictor.
* No ERA5 scoring.
* No retuning of any measurement rule, source, gate, fold or threshold after the result.
* No change to V1, the bundle, the dashboard or public findings.

## 13. Owner decision (2026-09-15, after this report)

**M1a is complete and is not promoted.** It did not satisfy the preregistered criteria for replacing M0:
* spatial-CV R² 0.1531 → 0.0580;
* spatial-CV RMSE 0.042873 → 0.045215;
* in-sample residual Moran's I 0.269 → 0.315;
* M49 performance worsened;
* both frozen vetoes failed.

The RMSE interval includes zero, so this is not described as a statistically established deterioration.

**What follows from the decision:**
* **Primary V2 development** stays anchored to the approved full-rank M0-equivalent station-geography baseline, **M0\*** (row C0 above).
* **M1a is retained permanently** as the registered **area-consistent geography measurement sensitivity**. After an M1b specification is approved and frozen, its result is replicated with the frozen M1a measurements. The two measurement sets are never searched jointly.
* **Scientific result preserved:** remeasuring the existing geography variables consistently over national land area materially changes several country measurements, but does not improve spatial generalization or residual spatial structure. Geography nevertheless remains the dominant variance group, and the historical-responsibility contribution remains small.
* **Not a validity judgement:** this decision does not mean station-based geography is physically or conceptually more correct because it predicts better.
