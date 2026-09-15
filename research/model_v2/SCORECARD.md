# The frozen scorecard

Every model stage (M0, M1a, M1, M2, M3, and the M4 table) reports the same
metrics, computed by `research/model_v2/cv.py::scorecard` and
`region_errors` from the same 151-country design, under the protocols in
`SPATIAL_CV_PROTOCOL.md`. M0's row is `outputs/m0_scorecard.json`.

## Core metrics (decision-relevant)

| # | Metric | Definition |
|---|---|---|
| 1 | n, mean n_train | countries scored; mean training-set size under the primary protocol |
| 2 | In-sample R² | 1 − SS_res / SS_tot of the full-sample fit |
| 3 | Spatial-CV R² | pooled over the 151 out-of-fold predictions of the primary protocol: 1 − Σ(y − ŷ_oof)² / Σ(y − ȳ)² |
| 4 | Spatial-CV RMSE | √mean(y − ŷ_oof)² under the primary protocol, °C/decade |
| 5 | Spatial-CV MAE | mean\|y − ŷ_oof\| under the primary protocol |
| 6 | Paired ΔRMSE vs M0 | RMSE(stage) − RMSE(M0) on the same 151 out-of-fold errors, with a 95% percentile interval from 2,000 country-bootstrap resamples of the per-country squared-error pairs (seed 0). **The improvement criterion.** |
| 7 | Calibration slope, intercept | OLS of y on ŷ_oof (primary); 1 and 0 for a calibrated model |
| 8 | Residual Moran's I, in-sample | V1 weights (station-centroid kNN, k = 8, row-standardised), two-sided permutation p from 999 permutations (seed 0); the statistic equals V1's, the p-value is finer than V1's 199-permutation value |
| 9 | Residual Moran's I, out-of-fold | the same statistic on the primary-protocol errors y − ŷ_oof |
| 10 | Group contributions | in-sample LMG/Shapley shares over the stage's declared group partition: geography, historical responsibility (emissions), socioeconomic, population, residual share, plus one row per newly declared group (for example "regional forcing" or "spatial"), never folded into geography |
| 11 | Secondary-protocol R² and RMSE | pooled, leave-one-M49-sub-region-out; may not reverse an improvement seen under the primary |
| 12 | Worst-region RMSE | the largest per-sub-region RMSE of the primary-protocol errors among M49 sub-regions with at least 3 countries, with the region named; the full per-region table is recorded alongside |
| 13 | Effective degrees of freedom | rank of the design for linear stages; trace of the smoother matrix for penalised or spline stages; parameter count plus the spatial parameter for spatial models |

## Recorded, not criteria

* Reference random 10-fold R² and RMSE (seed 0), to show the spatial optimism gap.
* Sensitivity protocols: a 1000 km border buffer and a 1500 km land-centroid buffer.
* Coefficient sign stability: the share of primary-protocol training fits in
  which each coefficient keeps its full-sample sign (retention rule for M1
  covariates: at least 0.80, and the pre-stated sign).
* Group-contribution stability: the V1 country bootstrap (2,000 resamples,
  seed 0) and continent block bootstrap (seed 1) of the shares, computed at M4
  for the retained stages (they are the expensive part and change nothing
  about retention decisions).
* Per-region RMSE and mean error (bias) tables for the primary and secondary
  protocols.
* Fold-level dispersion: the interquartile range of the per-country absolute
  out-of-fold error under the primary protocol.

## Deliberately not recorded

* AIC/BIC or likelihood-based criteria: not comparable across the estimator
  families the stages use (OLS, splines, spatial error/lag).
* Per-fold R²: undefined for single-country folds; the per-region RMSE table
  carries the same information at an interpretable resolution.
* Any metric under a protocol designed after M1–M3 results were seen.

## Conventions

* The outcome, the country set (151), the V1 weights for Moran's I and the
  fold assignments are identical for every stage; a stage that needs a
  different country set (a covariate missing for some countries) is scored on
  the 151 with the missing values imputed by a documented rule or is not
  admitted.
* Predictions for a categorical level absent from a training fold use the mean
  of the training level effects (`unseen="mean_effect"`); under the primary
  protocol this affects only Iceland (Köppen E).
* RMSE and MAE are in °C/decade and reported to four decimals; shares to three.
* "Improvement" is judged on metric 6; metrics 11 and 12 can veto it; metrics
  2, 8, 9, 10 and 13 are for interpretation.
