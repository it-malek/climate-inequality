# M3 — explicit spatial structure on M0\*: station-centroid kNN8 result

**Recorded 2026-09-16. First and only primary M3 score (station arm)**, run from pushed evaluator-freeze
commit `a6df380` with `PYTHONHASHSEED=0` into `outputs/m3_station/`. The method was frozen before any M2 fit
(`DOWNSTREAM_COMPLETION_SPEC.md`, `4dc2fad`; Amendment A1 `ef04958`); the operational detail is in
`M3_EVALUATOR_SPEC.md`. **Branch A:** the retained static specification is M0\* (M2 `not supported`).
The land-centroid kNN8 weight sensitivity is a separate package (§8).

## 1. Qualification (retention for prediction) and final naming

| Condition (station arm, primary total CO₂) | SEM: `M3-SEM(M0*)` | SAR: `M3-SAR(M0*)` |
|---|---|---|
| Q1: ΔRMSE vs M0\* < 0 | −0.0031049 | −0.0047643 |
| Q2: paired interval upper endpoint < 0 | [−0.0059176, −0.0002240] | [−0.0079013, −0.0015371] |
| Q3: M49 ΔRMSE ≤ +0.001 | −0.0019237 | −0.0030022 |
| Q4: worst(f) − worst(M0\*) ≤ +0.001 | −0.0079545 | −0.0047253 |
| Q5: computable in every required fit | yes (182 fits) | yes (182 fits) |
| **Qualifies** | **yes** | **yes** |

**Final naming** (spec §2.14):
* `retained_static_specification` = **M0\***;
* `qualifying_spatial_extensions` = **[M3-SEM(M0\*), M3-SAR(M0\*)]**, listed in fixed order, not ranked;
* `final_primary_predictive_model` = **null**. Two spatial extensions qualify, and no historical rule selects
  one family. No winner is chosen.

## 2. Scorecard (primary representation, n = 151)

| Metric | M0\* | SEM | SAR |
|---|---:|---:|---:|
| Primary 500 km LOCO CV R² / RMSE / MAE | 0.15308 / 0.042873 / 0.034465 | 0.27131 / 0.039768 / 0.031922 | 0.33085 / 0.038108 / 0.030326 |
| M49 holdout RMSE / R² | 0.038894 / 0.30297 | 0.036970 / 0.37020 | 0.035892 / 0.40640 |
| Random 10-fold RMSE (reference) | 0.035223 | 0.029226 | 0.028466 |
| Worst eligible region | Central Asia 0.069364 | Central America 0.061410 | Central America 0.064639 |
| OOF residual Moran's I (station kNN8) | 0.3215 | 0.2591 | 0.2562 |
| In-sample: OLS R² (M0\*) / one-step R² / trend R² | 0.63645 | 0.77951 / 0.50664 | 0.78286 / 0.67527 |
| Moran's I of innovations / structural residual | 0.2687 (OLS residual) | 0.0767 / 0.4846 | 0.0557 / 0.3157 |
| Calibration slope / intercept | 0.609 / 0.069 | 0.722 / 0.049 | 0.764 / 0.041 |
| Effective degrees of freedom | 20 | 21 | 21 |
| Countries improved / worsened vs M0\* (primary) | — | 88 / 63 | 94 / 57 |

The one-step R² uses neighbours' observed outcomes. It is not an OLS R² and is shown only for the accounting
and the overfitting diagnostic. **Transfer evidence is the out-of-fold rows.** For both families the primary
and M49 RMSE fall and OOF Moran's I falls by about 0.06.

## 3. Dependence parameter (model-based, descriptive)

| | SEM λ̂ | SAR ρ̂ |
|---|---|---|
| Full sample (se; Wald 95%) | 0.832 (0.046; 0.742–0.923) | 0.735 (0.059; 0.620–0.850) |
| 151 primary training fits: median (min–max) | 0.820 (0.768–0.990) | 0.722 (0.632–0.797) |
| Fits on the domain bound | 1: the Bahamas held out, constrained at 0.99 | 0 |
| Strictly positive, all protocols | 181 of 181 | 181 of 181 |

* **The Wald interval** is asymptotic, conditional on W and the Gaussian model, and not spatially robust.
* **Direction of the design bias.** Under no dependence, the frozen designs pull θ̂ strongly negative
  (`M3_FEASIBILITY_AUDIT.md` §3). The positive estimates are therefore not a design artefact of that kind, but
  they are still not read as a dependence strength.
* **The boundary fit.** Under the pre-result convention (commit `99f2c3d`) the fit on the bound is a flagged
  constrained estimate. Under the earlier draft rule the SEM family would have been non-computable.
* **Multimodality.** Some SEM grids have two local maxima. The frozen global-grid rule decides, and the
  independent check confirms that each estimate lies in the global basin.
* **Spatial range: not applicable.** The fixed kNN8 graph has no estimated range.
  * The full-sample 8th-neighbour distance has median 1,442 km and maximum 9,182 km.
  * Under the primary protocol, held-out nodes attach to training neighbours at 644–3,616 km (nearest) and
    1,069–9,182 km (8th). Dependence beyond the 500 km buffer carries the transfer gain.

## 4. Accounting (full sample, spec §2.11)

| Part | SEM | SAR |
|---|---:|---:|
| A_static (= M0\* OLS R² = Σ M0\* LMG shares) | 0.63645 | 0.63645 |
| A_dependence | 0.14306 | 0.14641 |
| A_innovation | 0.22049 | 0.21714 |
| Identity sum | 1.00000 | 1.00000 |
| Absorbed fraction of the static residual | 0.3935 | 0.4027 |

**What A_dependence is.** It is the in-sample reduction in squared one-step error from the ML spatial fit
and its neighbour-conditional predictor.
* It is **not** an LMG share, a variance component or a mechanism.
* It is not added to geography or to any other group.

**Non-spatial-part group contributions** are fractions of the filtered outcome's variance, a different
denominator from M0\*'s shares.

| | M0\* static (of outcome variance) | SEM filtered (R²\* 0.334) | SAR filtered (R²\* 0.402) |
|---|---:|---:|---:|
| geography | 0.5188 | 0.2795 | 0.3131 |
| emissions | 0.0131 | 0.0119 | 0.0152 |
| socioeconomic | 0.0628 | 0.0110 | 0.0291 |
| population | 0.0417 | 0.0316 | 0.0442 |
| geography's composition (share ÷ R²) | 0.815 | 0.837 | 0.780 |

**Material change of the non-spatial part: no, for either family.** Geography remains strictly the largest
named filtered share, and emissions stays ≤ 0.10.

## 5. Diagnostics, representation identity

* **Overfitting clauses (i), (ii), (iv): no signal for either family.**
  * The one-step in-sample gain (0.143, 0.146) does not exceed the CV R² gain (0.118, 0.178) by more than 0.05.
  * The df rise is 1.
* **Representation identity (Amendment A1): passes.**
  * θ̂ agrees to 3.2e-8 (SEM) and 4.3e-8 (SAR) across the full and training fits.
  * Predictions agree to 2.2e-9.
  * Per-capita Booleans are identical and descriptive.
* **Countries.** The largest per-country error reductions are:
  * SEM: Brazil, Libya, Uzbekistan, Kazakhstan, UAE;
  * SAR: Brazil, UAE, Laos, Oman, Libya.

  The largest increases are:
  * SEM: Mongolia, Canada, the United States, Congo, Tunisia;
  * SAR: Mongolia, Canada, Tunisia, the United States, Niger.

## 6. Verification

* **Independent reconstruction** (`outputs/m3_station_verification/m3_independent_check.json`). It uses the
  separately written reference math and imports neither the estimator nor the evaluator. It covers all 1,456
  fits:
  * kNN8 lists re-derived from coordinates;
  * log-likelihoods through eigenvalues (to 3.4e-13);
  * local optimality and global basin;
  * β̂ by QR (1.2e-14) and predictions by loops (4.4e-16);
  * metrics and paired intervals (5e-15), Q1–Q5 and naming;
  * accounting and permutation-Shapley filtered shares (1.7e-14);
  * representation identity.

  All checks pass. Five negative controls are detected: θ̂, a neighbour, a prediction, the veto threshold and β̂.
  The M0\* static reproduction matches the committed M2 result exactly.
* **Reproducibility.** An unchanged-code rerun (`reproducibility_run_b/`) is byte-identical on every
  deterministic artifact and the manifest (`m3_reproducibility.json`).

## 7. Interpretation (fixed before the result)

* Adding kNN8 spatial dependence to M0\* improves transfer to held-out countries under the frozen 500 km
  protocol and the M49 holdout, beyond what M0\*'s covariates carry. This accounts for spatial covariance. It
  does not explain a mechanism.
* Qualification is retention for prediction only. The descriptive decomposition conclusion still rests on
  M0\*'s OLS LMG shares, and both families leave that conclusion's structure unchanged.
* Held-out predictions borrow observed outcomes of training countries at least 500 km away. The gain is
  therefore conditional on neighbouring observations being available, and this is not a better static model.
* **Next:** the land-centroid kNN8 sensitivity (checkpoint 7), then M4.

## 8. Addition (2026-09-16, after §1–§7 were committed at `e049eba`): land-centroid kNN8 weight sensitivity

Run once from the unchanged frozen code (closure identical to `a6df380`) into `outputs/m3_land_centroid/`,
after the station package was committed and pushed. Graphs and held-out attachments use land-area centroids;
folds, buffers, S, optimizer and rules are unchanged. **Descriptive only: it cannot change the qualifying set or
the final naming, and no geometry is selected.**

| Primary representation | SEM (land) | SAR (land) |
|---|---:|---:|
| ΔRMSE vs M0\* [interval] | −0.0048265 [−0.0080948, −0.0010164] | −0.0050508 [−0.0075912, −0.0020762] |
| M49 ΔRMSE | **+0.0079440** | −0.0037507 |
| worst(f) − worst(M0\*) | −0.0085039 | −0.0087754 |
| Q1–Q5 (descriptive `arm_conditions`) | Q3 fails → would not qualify | all hold |
| θ̂ full sample (se); primary fits median (range) | 0.971 (0.010); 0.969 (0.865–0.989) | 0.808 (0.047); 0.792 (0.729–0.839) |
| OOF Moran's I (station weights) | 0.2727 | 0.2616 |
| A_dependence / A_innovation | 0.1922 / 0.1713 | 0.1740 / 0.1896 |
| Material change of the non-spatial part | no | no |

**Reading.** SAR's qualification conditions hold under both weight geometries. SEM's primary-protocol gain
reproduces under land-centroid weights, but its M49 holdout error rises above the veto and its dependence
estimate approaches the domain bound. The SEM qualification is therefore weight-sensitive on the secondary
protocol. This is reported as found; the station arm remains the deciding arm. Verification: independent
reconstruction passes with all five negative controls detected
(`outputs/m3_land_centroid_verification/m3_independent_check.json`), and an unchanged-code rerun is
byte-identical (`m3_reproducibility.json`).
