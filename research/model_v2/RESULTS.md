# Results of the residual-structure investigation

Recorded at closure, September 2026. Every number is read from the committed
records under `outputs/`; the machine-readable summary is
`outputs/v2_final/v2_final_record.json`. Units are °C/decade unless stated.
The validation protocol, the decision rules and the full-rank representation
(M0\*) that every comparison uses are in
[`VALIDATION_PROTOCOL.md`](VALIDATION_PROTOCOL.md).

## 1. The static model under geographically separated validation

The public decomposition fits a linear model of the area-weighted national
land-warming rate, 1950–2013, on four feature groups over 151 countries.
Re-expressed in full rank (M0\*) it gives:

| Quantity | Value |
|---|---:|
| In-sample R² | 0.636 |
| Residual share of total variance | 0.364 |
| Primary spatial CV (500 km territorial exclusion): R² / RMSE / MAE | 0.153 / 0.0429 / 0.0345 |
| M49 sub-region hold-out: R² / RMSE | 0.303 / 0.0389 |
| Random 10-fold (reference): R² / RMSE | 0.428 / 0.0352 |
| Worst sub-region (n >= 3) | Central Asia, RMSE 0.0694 |
| Residual Moran's I, in-sample / out-of-fold | 0.269 / 0.322 |
| Calibration slope of out-of-fold predictions | 0.61 |

Group shares of total cross-country variance with their bootstrap intervals
(2,000 draws each; `outputs/m4_final/m4_bootstrap_summary.json`):

| Share | Point | Country bootstrap 95% | Continent-block bootstrap 95% |
|---|---:|---|---|
| Geography | 0.519 | 0.452–0.630 | 0.385–0.594 |
| Socioeconomic development | 0.063 | 0.036–0.126 | 0.038–0.153 |
| Population | 0.042 | 0.021–0.090 | 0.015–0.117 |
| Historical emissions responsibility | 0.013 | 0.006–0.035 | 0.008–0.116 |
| Residual | 0.364 | 0.233–0.407 | 0.189–0.504 |

Geography is the largest named group in every draw under both bootstraps; the
2.5th percentile of its margin over the next group is 0.35 (country) and 0.28
(block). Under the per-capita representation the point shares are geography
0.514 and responsibility 0.020, with the same conclusions.

## 2. Overview of the investigations

Each row compares one extension with M0\* on the same 151 countries, folds and
estimator (`outputs/m4_final/m4_consolidated_table.json`). Negative changes
are improvements.

| Extension | Verdict | Primary CV RMSE | M49 RMSE | Out-of-fold Moran's I | Paired change in RMSE vs M0\* [95% interval] |
|---|---|---:|---:|---:|---|
| M0\* (comparator) | | 0.04287 | 0.03889 | 0.322 | |
| Area-consistent geography measurement | not promoted; kept as measurement sensitivity | 0.04522 | 0.04058 | 0.341 | +0.00234 [−0.00035, +0.00526] |
| Baseline hydroclimatic dryness | not supported | 0.04460 | 0.03969 | 0.327 | +0.00172 [+0.00089, +0.00285] |
| Flexible latitude functional form | not supported | 0.03992 | 0.03969 | 0.345 | −0.00295 [−0.00592, +0.00008] |
| Spatial error model (SEM), station-centroid kNN8 | qualifies for prediction | 0.03977 | 0.03697 | 0.259 | −0.00310 [−0.00592, −0.00022] |
| Spatial lag model (SAR), station-centroid kNN8 | qualifies for prediction | 0.03811 | 0.03589 | 0.256 | −0.00476 [−0.00790, −0.00154] |
| SEM, land-centroid kNN8 (weight sensitivity) | descriptive; M49 veto fails | 0.03805 | 0.04684 | 0.273 | −0.00483 [−0.00809, −0.00102] |
| SAR, land-centroid kNN8 (weight sensitivity) | descriptive; conditions hold | 0.03782 | 0.03514 | 0.262 | −0.00505 [−0.00759, −0.00208] |

The final identities: the retained static specification is M0\*; the
qualifying spatial extensions are SEM and SAR on M0\* with station-centroid
kNN8 weights, listed in fixed order and not ranked; no single final primary
predictive model is designated, because two families qualify and no rule
selects between them. No material change to the public conclusion occurred in
any stage or representation.

## 3. Area-consistent geography measurement

The public geography features are measured at station locations (mean station
latitude, elevation sampled at grid-snapped station coordinates, distance from
stations to the coast, the modal Köppen class and hemisphere over stations)
while the outcome describes the whole land area. The features were re-measured
over each country's terrestrial area (GPW national cells intersected with
GSHHG full-resolution land, exact great-circle coast distance, area-dominant
Köppen class, majority-area hemisphere; `outputs/m1a_geography_features.csv`).
The measurements changed a great deal: Canada's absolute latitude rises from
46.8° to 58.7° and Brazil's falls from 19.3° to 10.7°; all 17 negative
station-sampled elevations become positive (Jamaica −1,528 m to +319 m);
continentality rises by 300–600 km for Algeria, China, Brazil, Mauritania and
Libya; 29 countries change Köppen class (stations in wetter or temperate zones
while arid B dominates the area: Algeria, Chad, Mali, Ethiopia, Kenya, Morocco,
South Africa, China, Mexico, Australia, Chile).

Fitting the same model on the re-measured features did not improve transfer:
in-sample R² fell from 0.636 to 0.612, primary CV R² from 0.153 to 0.058, the
paired RMSE change was +0.0023 with an interval covering zero, the M49 and
worst-region errors rose (Central Asia 0.069 to 0.098), and the residual
Moran's I rose slightly (in-sample 0.269 to 0.315). The group shares moved by
less than 0.04 (geography 0.519 to 0.488; responsibility 0.013 to 0.014). The
measurement was therefore kept as a registered sensitivity rather than adopted.
Better-aligned measurement of the existing geography concepts was informative
but did not recover residual structure; one untested reading is that
station-sampled geography partly tracks the station-based construction behind
parts of the gridded outcome.

## 4. Baseline hydroclimatic dryness

One pre-specified physical covariate was tested: a baseline dryness index, the
area-weighted mean over each country's land of log10 of the ratio of mean
annual precipitation to mean annual potential evapotranspiration over
1920–1949 (CRU TS v4.10, 0.5°; `outputs/m1b_hydroclimate_features.csv`). The
hypothesis, fixed in advance, was that drier baseline regimes warm faster
(negative coefficient on log P/PET), on the physical rationale that in
water-limited regimes extra surface energy goes into sensible rather than
latent heat. A pre-outcome redundancy check found that 65% of the index's
cross-country variance already lies in the span of the M0\* design (R² 0.650,
VIF 2.86), which is below the pre-set exclusion level.

The result is not supported: the paired RMSE change is +0.0017 with the whole
interval above zero (+0.0009 to +0.0028), the primary CV R² falls from 0.153
to 0.084 while the in-sample R² rises by 0.00001, and the coefficient is
positive in the full fit (+0.00047) and negative in only 68 of 151 training
fits. The worst-region error rises by +0.004. Per country, 107 of 151 out-of-fold
errors worsen, most in Egypt, Libya, Chad, Jordan and Kazakhstan. In sample the
new group takes a share of 0.046, drawn mostly from geography; that is an
allocation of shared explained variance, not a physical fraction, and it
coexists with worse transfer.

This does not show that hydroclimate is irrelevant to cross-country warming
differences. It may reflect information the geography block already carries,
the climatological anchoring of the CRU pre-1950 fields (CRU inserts a
1961–1990 climatology where stations are absent, and PET uses partly synthetic
inputs), noise in the index, limited power at n = 151, or a relationship that
is not linear. No alternative window, transform, dataset, threshold or
nonlinear dryness term was tried in response, and the registered conditional
replications (area-measured geography, per-capita responsibility, ERA5 outcome)
were not run because the primary association was not supported.

## 5. Flexible latitude functional form

The residual's hemispheric asymmetry (`RESIDUAL_DIAGNOSTICS.md` §4) motivated
one joint candidate: M0\* plus two natural cubic spline columns in absolute
latitude (knots at the training tertiles and range, derived inside each fold)
and one southern-hemisphere latitude-slope difference, three added
coefficients, all in the geography group (`m2_latitude_basis.py`). No partial
form (interaction only, spline only, northern-hemisphere-only knots) was
scored, before or after.

The candidate improves the point estimate but is not supported under the
pre-set rule: the paired RMSE change is −0.00295 [−0.00592, +0.00008], with
the interval reaching just above zero. Primary CV R² rises from 0.153 to
0.266, the M49 change (+0.0008) and the worst-region change (−0.0044) pass
their vetoes, and the in-sample R² rises to 0.697 (geography share 0.582). The
southern-minus-northern slope difference is negative in all 151 training fits
(full fit −0.0026 °C/decade per degree), which is descriptive and
directionally consistent with the residual pattern that generated the
hypothesis; it is not independent confirmation, and it identifies no
mechanism. The out-of-fold residual Moran's I is essentially unchanged (0.322
to 0.345). The static stopping indicator did not fire because the point gain
exceeded 0.002; the result is therefore that the tested static extensions did
not demonstrate transferable improvement, not that the residual is
unrecoverable in principle.

## 6. Explicit spatial dependence

A spatial error model (dependence in the error) and a spatial lag model
(dependence in the outcome) were fitted on M0\* by maximum likelihood, with a
fixed row-standardised 8-nearest-neighbour graph on station centroids and,
as a sensitivity, on land-area centroids (`m3_spatial.py`). Held-out
predictions under the primary protocol attach the held-out country to its
eight nearest training countries, all at least 500 km away (nearest at
644–3,616 km, eighth at 1,069–9,182 km), so the gain reported here depends on
the observed outcomes of distant neighbours being available.

| Station-centroid kNN8 | SEM | SAR |
|---|---:|---:|
| Paired RMSE change vs M0\* [95% interval] | −0.00310 [−0.00592, −0.00022] | −0.00476 [−0.00790, −0.00154] |
| Primary CV R² | 0.271 | 0.331 |
| M49 RMSE change | −0.0019 | −0.0030 |
| Out-of-fold residual Moran's I | 0.259 | 0.256 |
| Dependence parameter, full sample (Wald 95%) | 0.83 (0.74–0.92) | 0.74 (0.62–0.85) |
| Moran's I of in-sample innovations | 0.077 | 0.056 |
| Countries improved / worsened | 88 / 63 | 94 / 57 |

Both families meet all five qualification conditions. In-sample accounting
splits the outcome variance into the static part (0.636, the M0\* R²), a
dependence part (SEM 0.143, SAR 0.146) and an innovation part (0.220, 0.217);
the dependence part is 39.4% (SEM) and 40.3% (SAR) of the static residual. That
quantity is the reduction in one-step squared error from a neighbour-conditional
predictor. It is not a Shapley share, not a variance component, not a mechanism,
and not additive with the geography share. Within the spatially filtered
outcome, geography remains the largest named group (filtered shares 0.280 SEM,
0.313 SAR) and responsibility stays small (0.012, 0.015).

Under land-centroid weights SAR's conditions all hold (−0.00505 [−0.00759,
−0.00208]); SEM's primary gain reproduces (−0.00483) but its M49 error rises by
+0.0079, failing the veto, and its dependence estimate approaches the domain
bound (0.97). SEM's qualification is therefore weight-sensitive on the
secondary protocol.

## 7. Robustness

**Exclusion distance** (`outputs/m4_final/m4_sensitivities.json`). SAR's
advantage over M0\* holds at a 1000 km territorial exclusion (−0.00306
[−0.00580, −0.00023]) and a 1500 km land-centroid exclusion (−0.00437
[−0.00695, −0.00163]). SEM's holds at 1500 km (−0.00289 [−0.00523, −0.00015])
but its interval covers zero at 1000 km (−0.00141 [−0.00427, +0.00150]).

**Temperature product** (`outputs/m4_final/m4_products.json`). The static
model and both spatial extensions were refitted with the ERA5 area-weighted
outcome on the same 151 countries, under two constructions: anomalies against
ERA5's own 1951–1980 monthly climatology (preferred, matching Berkeley Earth's
convention) and absolute monthly temperature (the construction behind the
public ERA5 cross-check).

| | ERA5, anomaly-aligned | ERA5, absolute temperature |
|---|---|---|
| M0\* in-sample R² | 0.465 | 0.500 |
| M0\* primary CV RMSE / R² | 0.0672 / 0.053 | 0.0711 / 0.126 |
| M0\* geography / responsibility share | 0.406 / 0.008 | 0.444 / 0.015 |
| Residual agreement with Berkeley Earth: Pearson / Spearman / signs | 0.334 / 0.326 / 89 of 151 | 0.362 / 0.355 / 90 of 151 |
| SEM change vs M0\* [interval]; dependence estimate | +0.00146 [−0.00107, +0.00411]; 0.49 | +0.00156 [−0.00107, +0.00434]; 0.48 |
| SAR change vs M0\* [interval]; dependence estimate | +0.00008 [−0.00158, +0.00178]; 0.38 | −0.00029 [−0.00218, +0.00171]; 0.39 |

The static conclusion survives both constructions: geography remains the
largest named group and responsibility stays far below 0.10. The spatial
extensions' predictive advantage does not: neither family improves transfer
with ERA5 outcomes, although positive dependence estimates are recovered.
Product disagreement is not an identified amount or cause of observational
error, and the ERA5 results selected nothing.

| Conclusion | Aligned ERA5 | Absolute ERA5 | Per-capita representation | Area-measured geography | Block bootstrap | Land-centroid weights |
|---|---|---|---|---|---|---|
| Geography largest named group; responsibility small (static) | survives | survives | survives | survives | survives | not applicable |
| SAR predictive gain over M0\* | does not survive | does not survive | survives | not run | not applicable | survives |
| SEM predictive gain over M0\* | does not survive | does not survive | survives | not run | not applicable | M49 veto fails |

## 8. Hypotheses recorded from the residual, and their status

| Hypothesis | Status |
|---|---|
| The latitudinal gradient differs by hemisphere | tested jointly with curvature (§5): not supported; slope difference negative in all fits, descriptive only |
| Geography re-measured over land area changes the residual | not supported as an improvement (§3) |
| Baseline aridity and land–atmosphere coupling | not supported as the linear dryness index (§4) |
| Regional aerosol forcing histories | not advanced: its residual evidence was not product-robust, and an aerosol term would need its own declared group |
| Snow–albedo amplification at northern high latitudes as latitude curvature | tested only as common curvature within §5: not supported; a distinct northern onset was not representable |
| Synoptic-scale ocean influence on southern mid-latitude land | not pursued |
| Observational-product structure | run as the product check in §7: the spatial gains are product-sensitive |
| Internal decadal variability aliased into the linear trend; the unit of analysis; station density as observational structure | not pursued |

Not pursued is not rejected, and a rejected operational model does not
disprove its motivating mechanism.

## 9. What remains, and the limits of these results

The M0\* residual share is 0.364 (country bootstrap 0.233–0.407). Its
out-of-fold residual stays spatially organised (Moran's I 0.32 for M0\*, 0.26
for the spatial models), while the in-sample innovations of the spatial models
are nearly unstructured (0.06–0.08). This remaining variance is not an unknown
climate cause; it is variance in cross-country warming differences that the
tested country-level models do not capture with these observations. It may
contain omitted regional physical heterogeneity, internal variability,
functional-form limitations, temperature-product differences, spatial
dependence, predictor measurement error, country aggregation and finite-sample
error, in proportions this design cannot separate.

Limits that travel with every result above: 151 countries; hypotheses
generated and tested on the same outcome data; equal country weighting; the
Berkeley–ERA5 disagreement; spatial-model gains that rely on neighbouring
observed outcomes; paired country-resampling intervals that are not spatially
corrected; model-based dependence-parameter intervals; descriptive and
associational throughout, with no causal identification.

## 10. Artifacts

| Stage | Result manifest | Scorecard | Verification |
|---|---|---|---|
| Baseline and corrected spatial CV | | `outputs/m0_scorecard_territory_corrected.json` | `outputs/m0_baseline.json` (bundle refit to 1e-11) |
| Full-rank representations | | `outputs/rank_audit.json` | |
| Product audit | | `outputs/product_stability_summary.json`, `outputs/product_stability_aligned_summary.json` | |
| Geography re-measurement | `outputs/m1a_measurement_manifest.json` | `outputs/m1a_scorecard.json` | two byte-identical builds |
| Baseline dryness | `outputs/m1b_primary/m1b_result_manifest.json` | `outputs/m1b_primary/m1b_scorecard.json` | `outputs/m1b_primary_verification/` |
| Latitude functional form | `outputs/m2_primary/m2_result_manifest.json` | `outputs/m2_primary/m2_scorecard.json` | `outputs/m2_primary_verification/` |
| Spatial models, station weights | `outputs/m3_station/m3_result_manifest.json` | `outputs/m3_station/m3_scorecard.json` | `outputs/m3_station_verification/` |
| Spatial models, land-centroid weights | `outputs/m3_land_centroid/m3_result_manifest.json` | `outputs/m3_land_centroid/m3_scorecard.json` | `outputs/m3_land_centroid_verification/` |
| Final assessment | `outputs/m4_final/m4_result_manifest.json` | `outputs/m4_final/m4_consolidated_table.json`, `m4_bootstrap_summary.json`, `m4_sensitivities.json`, `m4_products.json` | `outputs/m4_final_verification/` |
| Closing record | `outputs/v2_final/v2_final_record.json` | | `tests/test_v2_final_record.py` |

Every independent reconstruction passes with its negative controls detected,
and every unchanged-code rerun is byte-identical on every deterministic
artifact. Two auxiliary defects are recorded rather than hidden: the dryness
evaluator's comparison mode originally raised after writing its record (fixed
in a later change to the exit mapping only), and the final-assessment
evaluator's comparison mode calls a helper that lists the wrong artifact names,
so its comparison record was produced by a direct digest comparison. The
comparison command was repaired during the later repository audit; the original
record remains unchanged. Neither defect affects a computed number.
