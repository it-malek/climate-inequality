# Model V2 — final report

**Recorded 2026-09-16, at scientific closure.** Machine-readable record:
[`outputs/v2_final/v2_final_record.json`](outputs/v2_final/v2_final_record.json). It carries source digests,
evaluator and result commits, and verification records. Detailed numbers are in the stage reports and
artifacts:
* [`M2_REPORT.md`](M2_REPORT.md), [`M3_REPORT.md`](M3_REPORT.md), [`M4_REPORT.md`](M4_REPORT.md);
* [`outputs/m4_final/`](outputs/m4_final/).

The public V1 model, bundle and tag `v1.3.0` are unchanged.

## 1. Final identities

| Field | Value |
|---|---|
| **Retained static specification** | **M0\***: V1's covariates in the approved full-rank representation (total cumulative CO₂ primary; per-capita registered sensitivity) |
| **Qualifying spatial extensions** | **M3-SEM(M0\*)** and **M3-SAR(M0\*)**: station-centroid kNN8 spatial error and spatial lag models on M0\*; listed, not ranked |
| **Final primary predictive model** | **None designated.** Both families meet the frozen retention rule, and no historical rule selects one. The frozen disposition is to report both. |

## 2. What changed from V1

* **Validation.** Spatial validation is now the corrected 500 km footprint-certified leave-one-country-out
  protocol. The legacy 1° approximation (CV R² 0.212) is superseded; M0's corrected CV R² is 0.153 and RMSE
  0.0429 °C/decade.
* **Representation.** V1's design had an exact rank deficiency (21 columns, rank 20). V2 uses M0\*, which
  drops per-capita CO₂. Fitted values are identical, but group allocations differ, so V1's shares are never
  restated.
* **The descriptive conclusion is unchanged.**
  * Geography is the largest named group: M0\* share 0.519, country-bootstrap 95% [0.452, 0.630], continent
    block bootstrap [0.385, 0.594]. It is largest in every draw.
  * The historical-responsibility group is small: 0.013 [0.006, 0.035]; the block-bootstrap upper bound is
    0.116.
* **New: spatial covariance accounting and spatial prediction (M3).** On M0\*, SEM and SAR absorb about 39–40%
  of the static in-sample residual as neighbour-conditional dependence, and improve held-out prediction on
  Berkeley.

## 3. Findings by stage and their limits

* **M0.5.**
  * The territorial CV defect was corrected.
  * The rank-20 identity was resolved by the V2 predictor contract.
  * The product audit found that residual autocorrelation is present in both Berkeley and ERA5, while its
    magnitude and most regional or extreme patterns are product-sensitive. Aligned ERA5 residual Pearson
    correlation with Berkeley is 0.334.
* **M1a** (area-consistent geography). Not promoted: ΔRMSE +0.0023 [−0.0003, +0.0053], and the M49 and
  worst-region vetoes fail. It remains the measurement sensitivity; the group conclusion was stable.
* **M1b** (baseline hydroclimatic dryness C2). `not supported`: generalization worsened (ΔRMSE +0.0017
  [+0.0009, +0.0028]) and the coefficient carried the wrong sign.
* **M2** (joint pre-registered latitude form: two natural cubic columns + southern slope difference).
  `not supported`.
  * ΔRMSE −0.00295, but the paired interval [−0.00592, +0.00008] reaches above zero.
  * γ was negative in all 151 training fits. That is descriptive and directionally consistent with H1's
    expectation; it is not independent confirmation, since H1 was generated from the same outcome.
  * It is not evidence that latitude acts linearly.
  * The static stopping indicator did **not** fire, because the point gain exceeds 0.002.
* **M3** (station kNN8 on M0\*). Both families qualify for prediction.
  * SEM: ΔRMSE −0.00310 [−0.00592, −0.00022]. SAR: −0.00476 [−0.00790, −0.00154]. M49 and worst-region
    errors fall for both.
  * The non-spatial part of each model still ranks geography first.
  * Limits:
    * held-out predictions borrow observed outcomes of training countries at least 500 km away;
    * dependence parameters are model-based and are not read as dependence strength;
    * one SEM training fit was a constrained boundary estimate.
* **M4.**
  * Inventory complete (27 rows).
  * No material change to the V1 conclusion in either representation.
  * Buffer sensitivities: SAR's advantage holds at 1000 km and 1500 km. SEM's interval covers zero at 1000 km.

## 4. Robustness: what survives products and measurement choices

| Conclusion | Aligned ERA5 | Legacy ERA5 | Per-capita | Area geography (M1a) | Block bootstrap | Land-centroid weights |
|---|---|---|---|---|---|---|
| Geography largest named group; responsibility small (static M0\*) | survives | survives | survives | survives | survives | n/a |
| M2 not supported | not run (no support; aligned-only arm conditional) | not run | agrees | not run (conditional) | n/a | n/a |
| SAR predictive gain over M0\* | **does not survive** (ΔRMSE +0.0001) | **does not survive** (−0.0003, interval covers 0) | agrees | not registered | n/a | survives |
| SEM predictive gain over M0\* | **does not survive** (+0.0015) | **does not survive** (+0.0016) | agrees | not registered | n/a | M49 veto fails |

**Reading.** The static decomposition conclusion is robust across every check it was registered for. The
spatial models' transfer gain is a Berkeley-specific result: it is product-sensitive, and SEM is also
weight-sensitive. Product disagreement is not an identified amount or cause of observational error, and ERA5
selected nothing.

## 5. Hypotheses: status

| Entry | Status |
|---|---|
| H1 (hemispheric latitude slope) | Tested only jointly with H5 in M2: **not supported**. γ negative in all fits, which is directionally consistent and descriptive only. The H1-only form was deliberately not tested. |
| H2 (area remeasurement) | **not supported** as an improvement (M1a) |
| H3 (aridity / dryness, as C2) | **not supported** (M1b) |
| H4 (aerosol histories) | not advanced (cluster premise not product-robust) |
| H5 (NH high-latitude curvature) | Tested only as common curvature within M2: **not supported**. A distinct NH onset was not representable and was not tested. |
| H6 (synoptic ocean influence, C1) | not pursued (held) |
| H7 (product structure) | eventual-model product check run in M4, with no two-product mean; the spatial gains are product-sensitive |
| H8, H9, H10 | not pursued |

"Not pursued" is not a rejection. A rejected operational model does not disprove its motivating mechanism.

## 6. Remaining residual structure and principal limitations

* **What remains.** M0\*'s residual share is 0.364 [0.233, 0.407]. Its out-of-fold residual stays spatially
  organized: Moran's I 0.32 for M0\* and 0.26 for SEM and SAR. The in-sample innovations of the spatial
  models are nearly unstructured (I ≈ 0.06–0.08).
* **What it is not.** This remaining variance is not an unknown climate cause. It is variance in
  cross-country warming differences that the tested country-level models do not capture with these
  observations.
* **Principal limitations:**
  * n = 151;
  * same-dataset hypothesis generation and testing;
  * equal country weighting (the V1 estimand);
  * the Berkeley–ERA5 disagreement;
  * spatial-model gains that rely on neighbouring observations;
  * paired country-resampling intervals that are not spatially corrected;
  * model-based (not spatially robust) dependence-parameter intervals;
  * pre-freeze baseline-only anchor calculations, disclosed in each evaluator spec.

## 7. Why this research line ends here

Every mandatory branch of the frozen state machine has reached a terminal disposition, as specified before
any M2 fit:
* M2 reached its verdict;
* the conditional arms were not executed;
* M3 was run in both weight arms;
* M4's inventory is complete.

No registered static candidate remains:
* M1a, M1b and M2 were not supported or not promoted;
* C1, C3 and alternative latitude forms were not pursued;
* no branch authorizes further static-model search.

The M4 stopping rule did not declare the residual unrecoverable, because M2's point gain exceeded the
threshold. The stopping conclusion is therefore narrower: the **tested** country-level static specifications
did not demonstrate transferable improvement beyond M0\*. The remaining transferable signal the line found is
spatial dependence, which is product-sensitive, and it is accounted for, not explained. Further work (for
example the H8 decadal-variability diagnostic, a product-harmonized outcome or a different unit of analysis)
would ask different questions and belongs to a new line.
