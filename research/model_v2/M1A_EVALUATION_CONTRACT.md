# M1a evaluation contract — frozen before any M1a measurement or score

**Recorded 2026-09-15, in a commit made before the M1a measurement builder exists
and before any M1a model is fitted.** The M1a stage was approved by the project
owner on the same date, under [M1A_MEASUREMENT_SPEC.md](M1A_MEASUREMENT_SPEC.md)
and the approved [V2_PREDICTOR_CONTRACT.md](V2_PREDICTOR_CONTRACT.md). Nothing in
this file may be revised after M1a scores have been seen. Any later change must be
a new, dated, separately justified record, and the original rule must still be
reported.

## 1. What M1a is

M1a is strictly a **measurement-alignment experiment**. It remeasures only the
existing geography concepts, over the terrestrial support of each GPW/ISO
analytical unit:

| Feature | M0 (station-based) | M1a (area-based) |
|---|---|---|
| `abs_latitude` | mean station \|latitude\| | mean absolute terrestrial latitude |
| `elevation` | mean station-sampled ETOPO | terrestrial ETOPO with independent land/water masking, genuine below-sea-level land retained |
| `continentality` | mean station coast distance | area-mean great-circle distance to coast |
| `climate_zone` | station-modal Köppen A–E | area-dominant Köppen A–E |
| `hemisphere` | station-modal N/S | majority-area hemisphere |
| `spatial_block` | OWID continent | unchanged |

**Not part of M1a:**

* new physical concepts: aridity, precipitation, snow, albedo, soil moisture,
  vegetation, aerosols, circulation indices, additional forcings;
* model changes: nonlinear terms, interactions, spatial-error/lag terms;
* design changes: any change to the outcome, temporal window, temperature
  product, CV design, 500 km threshold, feature-group taxonomy, V1, the public
  bundle, dashboard or findings.

## 2. Comparison design

Four fits are compared, in two pairs:

| Row | Responsibility representation | Geography |
|---|---|---|
| `C0_primary` (rank-only M0 equivalent) | total CO₂ (per-capita dropped) | M0 station-based |
| `M1a_primary` | total CO₂ (per-capita dropped) | M1a area-based |
| `C0_percapita` (registered sensitivity comparator) | per-capita CO₂ (total dropped) | M0 station-based |
| `M1a_percapita` (registered sensitivity) | per-capita CO₂ (total dropped) | M1a area-based |

Within each pair:

* the same 151 countries, in the frozen M0 row order;
* the same area-weighted Berkeley outcome values;
* identical non-geography values and an identical `spatial_block`;
* the same four feature groups;
* the same frozen folds and training-exclusion sets. These are the corrected
  territorial lower bounds from `4263429`, read from committed outputs and never
  recomputed from the new land mask.

Only the five remeasured geography columns differ. The **primary conclusion
comes only from the primary pair**. The per-capita pair is reported as the
predeclared sensitivity, whichever pair looks better. `C0_primary` and
`C0_percapita` must reproduce the rank-audit `drop_per_capita` and `drop_total`
cards to 1e−10, or the run aborts. Legacy M0 (all three log columns) remains the
historical row.

## 3. Reported for every row

Computed by the existing `run_territory_correction.complete_score` /
`cv.scorecard` code.

**Scores**

* in-sample R²;
* corrected 500 km spatial-CV R², RMSE and MAE;
* M49 secondary R² and RMSE;
* random 10-fold reference R² and RMSE, as a reference only.

**Residual structure**

* residual Moran's I in-sample and out-of-fold, with frozen V1 station-centroid
  kNN8 weights, 999 permutations, seed 0.

**Group contributions**

* LMG/Shapley shares: geography, responsibility, socioeconomic and population,
  plus the residual share.

**Fold behaviour**

* fold RMSE median/min/max and worst fold country;
* IQR of per-country absolute OOF error;
* per-M49-region RMSE and bias, and the worst region with n≥3.

**Other diagnostics**

* rows with an unseen categorical level (primary and secondary), by feature;
* calibration slope/intercept;
* effective rank and encoded columns and levels present.

**Recorded, not criteria**

* area-centroid kNN8 residual Moran's I, as a weight sensitivity;
* coefficient sign stability across primary training fits.

**Paired rows (M1a vs its comparator)**

* ΔRMSE with the 2,000-resample country-bootstrap 95% interval (seed 0);
* ΔR² in-sample and CV;
* ΔI in-sample and OOF;
* share changes;
* per-country change in absolute OOF error, with counts improved and worsened;
* OOF errors of Canada, Brazil, Russia and Algeria (the plan's large-country
  gate).

## 4. Interpretation rule (three dimensions, fixed now)

Success is **not** "R² increased". Each dimension is judged separately, and only
for the primary pair.

### 4.1 Predictive transfer

* **Improved generalization** requires *all* of the following:
  * primary ΔRMSE < 0 with a 95% interval excluding zero;
  * M49 secondary RMSE no worse than the comparator by more than 0.001 °C/decade;
  * the worst-region RMSE (n≥3) no worse by more than 0.001.

  This is the plan's genuine-improvement definition with its vetoes.
* **Worsened generalization:** ΔRMSE > 0 with a 95% interval excluding zero.
* **Otherwise:** no detectable change in transfer. Report point estimates, and
  never describe a non-significant decrease as an improvement.
* **Overfitting signal** (plan definition):
  * the in-sample R² gain exceeds the CV R² gain by more than 0.05; or
  * OOF RMSE worsens while in-sample RMSE improves.
* **An in-sample improvement without improved spatial-CV performance is not
  evidence that M1a improved the model.**

### 4.2 Residual spatial structure

This dimension uses the frozen V1-weight residual Moran's I.

* **Reduced autocorrelation:** in-sample I and OOF I both fall by at least 0.05
  in absolute terms (about 19% of M0's in-sample 0.269).
* **Increased autocorrelation:** both rise by at least 0.05.
* **Otherwise:** essentially unchanged.

The 0.05 threshold is a descriptive magnitude fixed before results, not a
significance test. A reduction without improved transfer is described as
**improved spatial specification, not improved generalization**.

### 4.3 Scientific stability

**Material change to the V1/V2 conclusion** (plan definition): geography is no
longer the largest named group, or the responsibility share exceeds 0.10 of total
variance.

* **Otherwise:** the group-level conclusion is stable. Report every share change
  under the same representation.
* The narrow emissions statement is unchanged: responsibility explains little of
  the *cross-country differences in area-weighted warming* in this descriptive
  decomposition. It is not a statement about greenhouse gases' physical role in
  global warming.

### 4.4 Summary classification

Report exactly one of:

* improved generalization and spatial specification;
* improved generalization only;
* improved spatial specification only;
* neither.

Report any worsening explicitly. The sensitivity pair is classified the same way
and reported as agreeing or disagreeing; it cannot override the primary.

### 4.5 Gate M1a → M1 (plan, unchanged)

M1a becomes the base for later stages if either:

* it is a genuine improvement; or
* it is not worse (the ΔRMSE interval covers zero) and it removes the
  large-country OOF error pattern.

A genuine deterioration is reported as a finding about station-based versus
area-based measurement, and later stages then build on M0 measurements. **This
gate does not authorize any M1 covariate** (§6).

## 5. Measurement freeze and failure handling

* The measurement manifest is committed before any M1a score is computed. It
  records sources, hashes, mask hierarchy, the area convention, algorithms,
  numerical tolerances and coverage gates.
* Nothing may be tuned after seeing a score: measurement definitions, coverage
  gates, masks, weighting, quadrature and sources are all fixed.
* If any frozen country fails a coverage gate, the primary 151-country result is
  ineligible. The builder reports the affected IDs, areas and reasons, and
  scoring stops for an owner decision. No threshold is relaxed and no station
  fallback is used.
* If M1a worsens spatial-CV performance, that is reported directly.

## 6. Product-audit guardrail for later stages

The M0.5 product audit ([PRODUCT_STABILITY_AUDIT.md](PRODUCT_STABILITY_AUDIT.md))
materially limits what residual structure can motivate later M1 predictors:

* Berkeley and preprocessing-aligned ERA5 agree only moderately on outcomes
  (Pearson 0.540, Spearman 0.619).
* Residual agreement is substantially weaker (Pearson 0.334; signs agree in 89 of
  151 countries).
* No FDR-significant local residual cluster is shared between the products.
* Many regional patterns are product-sensitive (7 sensitive and 12 unresolved of
  20 regions).
* The previous aerosol-cluster premise (H4, Levant/eastern-Mediterranean LL) is
  **not product-robust**.

Rules that follow:

* Product-sensitive Berkeley residual patterns may not justify adding a later
  physical covariate.
* M1a is justified independently: it corrects an identified support/measurement
  mismatch in variables already in M0. It is not motivated by residual clusters.
* Any later genuinely new physical predictor requires product-robust residual
  evidence, a strong independently specified physical hypothesis, or both, fixed
  before fitting.
* Do not search predictors against Berkeley residuals and rationalize them
  afterwards.

**Interpreting the audit.** Neither temperature product is "right" or "wrong".
Aligning ERA5 preprocessing substantially reduced the mean Berkeley–ERA5 bias
without improving agreement. The disagreement is therefore not primarily the
anomaly-versus-absolute mismatch. The remaining disagreement is observational
and product uncertainty that bounds how far country-level inference can go.
ERA5 stays outside M1a scoring and model selection.
