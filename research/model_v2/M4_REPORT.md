# M4 — final spatial out-of-sample assessment: result

**Recorded 2026-09-16.** Run once from pushed evaluator-freeze commit `0366281` with `PYTHONHASHSEED=0` into
`outputs/m4_final/`. M4 changes no model. Scope, uncertainty applicability and completion criteria were frozen
before any M2 fit (`DOWNSTREAM_COMPLETION_SPEC.md` §3, `M4_EVIDENCE_INVENTORY.md`; `4dc2fad`); the operational
detail is in `M4_EVALUATOR_SPEC.md`.

## 1. Final model identities (`m4_final_specification.json`)

| Field | Value |
|---|---|
| `retained_static_specification` | **M0\*** |
| `qualifying_spatial_extensions` | **M3-SEM(M0\*)**, **M3-SAR(M0\*)**, listed in fixed order and not ranked |
| `final_primary_predictive_model` | **null**: two qualifying spatial extensions, and no historical unique-family selection rule |
| M2 label | `not supported` |
| Static stopping indicator | does **not** fire (best point ΔRMSE −0.002952 ≤ −0.002) |

## 2. The consolidated table (primary representation; details in `m4_consolidated_table.json`)

| Stage | Status | Primary CV RMSE | M49 RMSE | OOF Moran's I | Paired ΔRMSE vs comparator [interval] |
|---|---|---:|---:|---:|---|
| M0 (V1, corrected 500 km) | frozen baseline | 0.042873 | 0.038894 | 0.3215 | — |
| M0\* (total CO₂; per capita identical) | development baseline | 0.042873 | 0.038894 | 0.3215 | — |
| M1a (area geography) | not promoted; measurement sensitivity | 0.045215 | 0.040583 | 0.3408 | +0.002343 [−0.000350, +0.005264] |
| M1b (M0\* + C2 dryness) | not supported | 0.044596 | 0.039693 | 0.3269 | +0.001723 [+0.000891, +0.002849] |
| M2 (latitude functional form) | not supported | 0.039920 | 0.039694 | 0.3446 | −0.002952 [−0.005921, +0.000078] |
| M3-SEM(M0\*), station kNN8 | qualifying | 0.039768 | 0.036970 | 0.2591 | −0.003105 [−0.005918, −0.000224] |
| M3-SAR(M0\*), station kNN8 | qualifying | 0.038108 | 0.035892 | 0.2562 | −0.004764 [−0.007901, −0.001537] |
| M3-SEM(M0\*), land kNN8 (sensitivity) | descriptive; M49 veto fails | 0.038046 | 0.046838 | 0.2727 | −0.004826 [−0.008095, −0.001016] |
| M3-SAR(M0\*), land kNN8 (sensitivity) | descriptive; conditions hold | 0.037822 | 0.035143 | 0.2616 | −0.005051 [−0.007591, −0.002076] |

* The legacy 1° M0 row (CV RMSE 0.041355, R² 0.212) is a defective approximation and stays in a separate block.
* V1's rank-deficient allocations stay separate from V2's full-rank allocations.

**Three numbers per retained stage.**

| Retained stage | Primary RMSE − M0\* | OOF Moran's I | Geography share |
|---|---:|---:|---|
| M0\* | 0 | 0.3215 | 0.519, country-bootstrap 95% [0.452, 0.630] |
| M3-SEM(M0\*) | −0.003105 | 0.2591 | filtered 0.280; interval not applicable |
| M3-SAR(M0\*) | −0.004764 | 0.2562 | filtered 0.313; interval not applicable |

## 3. Static share uncertainty for M0\* (both representations)

2,000 draws each. Every draw was usable; 25 block draws were rank-deficient but usable.

| Share (total CO₂) | Point | Country bootstrap 95% | Continent block bootstrap 95% |
|---|---:|---|---|
| geography | 0.5188 | 0.452–0.630 | 0.385–0.594 |
| emissions (responsibility) | 0.0131 | 0.006–0.035 | 0.008–0.116 |
| socioeconomic | 0.0628 | 0.036–0.126 | 0.038–0.153 |
| population | 0.0417 | 0.021–0.090 | 0.015–0.117 |
| residual | 0.3636 | 0.233–0.407 | 0.189–0.504 |

* **Geography is the largest named group** in 100% of draws under both bootstraps. The 2.5th percentile of
  its margin over the next group is 0.348 (country) and 0.277 (block).
* **Material change to the V1 conclusion: none**, in either representation.
* **Block-bootstrap emissions interval.** Its upper end exceeds 0.10: 0.116 for total CO₂ and 0.195 for per
  capita. The frozen materiality rule is defined on the point share, so this is reported descriptively.
* **Not applicable:** V1's rank-deficient intervals are not reused, and the spatial accounting parts and
  filtered shares have no bootstrap (spec §3.3).

## 4. Buffer sensitivities (reporting only)

| Protocol | M0\* CV RMSE (anchor to committed M0) | SEM − M0\* [interval] | SAR − M0\* [interval] |
|---|---:|---|---|
| territorial 1000 km | 0.044950 (reproduced) | −0.001405 [−0.004266, +0.001499] | −0.003056 [−0.005799, −0.000228] |
| land-centroid 1500 km | 0.041005 (reproduced) | −0.002887 [−0.005235, −0.000151] | −0.004373 [−0.006948, −0.001628] |

Both representations are identical. SAR's ranking over M0\* survives both widths and metrics. SEM's paired
interval covers zero under the 1000 km buffer.

## 5. Product check of the eventual model (spec §3.5)

| | Aligned ERA5 (preferred) | Legacy ERA5 (preprocessing sensitivity) |
|---|---|---|
| M0\* in-sample R² (anchor to M0.5) | 0.4654 (reproduced) | 0.5003 (reproduced) |
| M0\* primary CV RMSE / R² | 0.067232 / 0.053 | 0.071120 / 0.126 |
| M0\* geography / emissions share | 0.406 / 0.008 | 0.444 / 0.015 |
| Point materiality (geography largest; emissions ≤ 0.10) | no material change | no material change |
| Residual agreement with Berkeley: Pearson / Spearman / sign | 0.334 / 0.326 / 89 of 151 | 0.362 / 0.355 / 90 of 151 |
| SEM − M0\* under the product [interval]; θ̂ | +0.001459 [−0.001072, +0.004108]; 0.485 | +0.001560 [−0.001071, +0.004344]; 0.484 |
| SAR − M0\* under the product [interval]; θ̂ | +0.000075 [−0.001581, +0.001778]; 0.377 | −0.000286 [−0.002176, +0.001708]; 0.391 |
| Extensions' Q1–Q5 under the product | neither holds | neither holds |

* **Product-sensitive: yes, for both extensions.** Their transfer gain over M0\* is not reproduced with ERA5
  outcomes, although positive dependence estimates are.
* **What survives both products and both preprocessing constructions:** the static decomposition conclusion
  (geography the largest named group; responsibility small).
* **What does not survive:** the spatial extensions' predictive advantage.
* **Limits.** Product disagreement is not an identified amount or cause of observational error, and ERA5
  selects nothing.

## 6. Measurement sensitivity

Reused from `m1a_scorecard.json`, an identical estimand (C0 = M0\* to 1e-10). Area-consistent geography did
not improve transfer (M1a row above), and the group conclusion was stable. No M3 geography remeasurement is
registered.

## 7. Inventory and verification

* **Inventory** (`m4_inventory_status.json`). All 27 rows end in verified evidence (22) or a permitted
  disposition (5):
  * spatial range not applicable;
  * two-product mean never authorized;
  * residual description in the closure;
  * H8–H10 not pursued;
  * geostatistical, C1, C3 and alternative latitude forms not pursued.
* **Independent reconstruction** (`outputs/m4_final_verification/m4_independent_check.json`; no import of
  the evaluators or `src.decomposition`). All sections pass:
  * all 8,000 bootstrap draws regenerated and refitted (8.3e-16);
  * summaries and material flags;
  * sensitivity scores, intervals and anchors (6.3e-14);
  * product scores, R² anchors and residual agreement;
  * final-record consistency with the M2 and M3 manifests.

  Four negative controls are detected.
* **Reproducibility.** An unchanged-code rerun is byte-identical on every deterministic artifact and the
  manifest (`m4_reproducibility.json`).
* **Disclosed auxiliary defect.** `m4_evaluate --compare` fails: it calls the M3 comparison helper, which
  lists M3 artifact names. The comparison record was therefore produced by a direct SHA-256 comparison. No
  computed number is affected, and the frozen evaluator was not patched after its result.
