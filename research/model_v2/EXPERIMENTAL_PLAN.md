# Model V2 experimental plan

Stages, what each is allowed to change, and the decision gates between them.
Every stage is scored on the frozen scorecard (`SCORECARD.md`) under the frozen
protocol (`SPATIAL_CV_PROTOCOL.md`) and compared with the frozen M0 model.
Its corrected primary CV baseline is recorded separately in commit `4263429`;
verification refits do not change the frozen outcome or predictors. Hypotheses come from `HYPOTHESIS_REGISTER.md`; nothing enters a
stage that is not registered there first, with its expected sign.

## Definitions used by the gates

* **Genuine improvement** of stage B over comparison stage A: paired ΔRMSE
  under the primary protocol is negative and its 95% country-bootstrap interval
  excludes zero; the secondary-protocol RMSE of B is not worse than A's by more
  than 0.001 °C/decade; for a covariate, its pre-stated sign holds in the full
  fit and in at least 80% of the primary training fits.
* **Overfitting signal**: the in-sample R² gain exceeds the primary-protocol
  R² gain by more than 0.05; or out-of-fold RMSE worsens while in-sample RMSE
  improves; or a coefficient's sign is unstable across folds (< 80%); or the
  effective degrees of freedom rise by more than 5 for an RMSE gain below
  0.002 °C/decade.
* **Covariate rejected**: wrong sign; or no genuine improvement; or the gain
  disappears under the secondary protocol; or it changes the meaning of an
  existing group without a declared new group. Rejected covariates stay in the
  report with their scorecard row.
* **Material change to the V1 conclusion**: the geography share of the retained
  final model, or its 95% interval, no longer makes geography the largest named
  group; or the emissions/responsibility share exceeds 0.10 of total variance.
  Either is reported as such, not protected against.

## M0 — frozen baseline

M0 model unchanged. Corrected validation row is
`outputs/m0_scorecard_territory_corrected.json`; the Session 1 row
`outputs/m0_scorecard.json` is legacy 1° approximation. Baseline: in-sample R² 0.636;
corrected primary spatial-CV R² 0.153080 and RMSE 0.042873 °C/decade; residual Moran's I 0.269;
shares geography 0.513, emissions 0.021, socioeconomic 0.069, population 0.033,
residual 0.364.

## M1a — area-consistent measurement of the existing geography features (H2)

**Current specification, not implemented:** `M1A_MEASUREMENT_SPEC.md` supersedes
Session 1's preliminary measurement recipes. Remeasure station-derived
`abs_latitude`, `elevation`, `continentality`, `climate_zone` and `hemisphere`
over the specified terrestrial support; retain the sixth geography variable,
OWID `spatial_block`, unchanged. Hemisphere is the majority-area category,
not centroid sign. Climate remains one A–E category, not five numeric shares.
Independent shoreline masking must preserve genuine below-sea-level land.
The proposed mask is a measurement-support source, not a new predictor.

First obtain approval for `V2_PREDICTOR_CONTRACT.md`. Compare M1a to the matching
rank-only M0 equivalent with the same approved primary/sensitivity representation;
never credit changed Shapley allocation from deleting a redundant log column to
geographic measurement alignment. V1's original decomposition remains separate.

**Why first.** It isolates remeasurement of concepts already in M0 before adding
physical concepts. Station-derived predictors and the area-weighted outcome
currently describe different spatial supports. No M1a result exists yet.

**Gate M1a → M1.** Report the scorecard. If M1a is a genuine improvement over
M0, or is not worse (ΔRMSE interval covers zero) and removes the large-country
pattern (Canada, Brazil, Russia, Algeria out-of-fold errors), M1a becomes the
base on which M1 covariates are added. If M1a is a genuine deterioration, M1
covariates are added to M0's measurements and the deterioration is reported as
a finding about station-based versus area-based measurement. Either way M0
remains the comparison row.

## M1 — pre-registered physical covariates (H3, H6; H4 only with approval)

**Change.** Add, one at a time and then jointly, the registered baseline
quantities: baseline aridity index (H3), synoptic-scale ocean fraction (H6);
regional aerosol history (H4) only if the owner admits it as a declared
separate group. Each is a static or pre-window quantity, area-averaged with the
same operator as the outcome, with provenance recorded in the register. No
transformation search: one pre-stated transform per covariate (log for
positive skewed quantities, identity otherwise).

**Data rule.** Fetch only the registered quantities; metadata research first;
no dataset enters the design without a provenance note and a pinned vintage.

**Gates.** A covariate is retained under the improvement rule and rejected
under the rejection rule above. **M1 → M2** proceeds only if the retained M1
model's out-of-fold errors still show structure the register says is nonlinear:
the SH latitude gradient (Spearman \|ρ\| > 0.4 within the SH), or out-of-fold
Moran's I > 0.2, or at least ten countries in significant LISA clusters. If none
holds, M2 is skipped and the reason recorded.

## M2 — pre-specified nonlinear terms (H1, H5; at most three)

**Change.** (a) `hemisphere × abs_latitude` (H1); (b) a natural cubic spline in
`abs_latitude` with three degrees of freedom and knots at the sample tertiles
(H5), fixed a priori, no df search; (c) at most one interaction between a
retained M1 covariate and continentality or latitude, chosen before fitting
and written into the register. Nothing else: no GAM over all features, no
automated term selection.

**Gates.** Retention as in M1, with the overfitting signal checked explicitly
(effective degrees of freedom is recorded). **M2 → M3** always proceeds,
because M3 is an accounting step, not a model-selection step.

## M3 — explicit spatial structure

**Change.** Fit a spatial error model and a spatial lag model on the retained
M2 (or M1, or M1a) specification with the V1 weights (station-centroid kNN,
k = 8), and with land-centroid kNN k = 8 as a sensitivity; optionally a
geostatistical residual (exponential covariance, range and nugget estimated)
as a third form. Out-of-fold predictions use only training-fold neighbours;
under the primary protocol the buffer removes every neighbour within 500 km,
so a spatial term can improve the primary score only through whatever
dependence survives beyond the buffer.

**Interpretation rule (fixed now).** A fall in residual Moran's I or a rise in
in-sample fit from a spatial term accounts for spatial covariance; it does not
explain a mechanism and is never described as one. M3's reportable outputs are:
the estimated spatial range and dependence parameter; how much of the residual
share is absorbed; whether the group contributions of the non-spatial part
change materially when the spatial term is present; and whether the spatial
term generalises under the primary protocol at all. A spatial term is retained
for prediction only under the improvement rule; its accounting value is
reported regardless.

## M4 — final spatial out-of-sample assessment

**Change.** None to the models. One table: M0, M1a, M1, M2, M3 under the frozen
scorecard; bootstrap intervals for the shares of the retained stages; the
paired ΔRMSE intervals; the per-region tables; the sensitivity protocols. If the
owner approves the product-robustness arm (H7), every retained stage is also
refit on the ERA5 area-weighted outcome and on the two-product mean, and the
stage-by-product table is part of M4.

**What M4 concludes.** Three numbers per retained stage answer the research
question: the primary-protocol RMSE relative to M0 (how much of the residual
is recoverable and transferable), the out-of-fold Moran's I (how much of what
remains is still spatially organised), and the geography share with its
interval (whether geography remains the dominant axis). The residual share of
the final model is described as variance in cross-country warming differences
not captured at country resolution with these observations, with the
product-robustness arm, if run, bounding how much of it is observational.

**Stopping rule.** If the best of M1a–M2 improves the primary RMSE by less than
0.002 °C/decade (5% of M0's), the residual is declared not recoverable with
country-level static covariates; M3 then quantifies its spatial covariance and
M4 reports that result as the finding. A defensible "substantial variance
remains" is preferred to a fragile high-fitting model.

## What every stage must not do

Change the outcome, the window, the temperature product (outside the approved
robustness arm), the country set, the primary protocol or the scorecard; tune
anything against the primary protocol without nesting; search variables or
transformations; regroup features without a declared schema revision; use
causal language for any share; touch `v1.3.0` or `main`.
