# M2 — joint latitude functional form: result

**Recorded 2026-09-16. First and only primary M2 score**, run from pushed evaluator-freeze commit `b13bb04`
with `PYTHONHASHSEED=0` into `outputs/m2_primary/`. Rules: `M2_EVALUATION_CONTRACT.md` with Amendment 1
(`M2_PRE_SCORE_RESOLUTIONS.md`) and `M2_EVALUATOR_SPEC.md`. The downstream M3/M4 rules were frozen and pushed
before this score (`4dc2fad`, Amendment A1 `ef04958`).

## 1. Verdict: `not supported`

| Gate | Value | Holds? |
|---|---|---|
| P1: ΔRMSE = RMSE(M2) − RMSE(M0\*) < 0 | **−0.0029521** °C/decade | yes |
| P2: paired interval upper endpoint < 0 | interval **[−0.0059210, +0.0000779]** | **no** |
| R1: ΔRMSE ≤ −0.002 | −0.0029521 | yes |
| R2: M49 ΔRMSE ≤ +0.001 | +0.0008002 | yes |
| R3: worst(M2) − worst(M0\*) ≤ +0.001 | −0.0043891 (Central Asia both) | yes |
| R4: structural integrity in every required fit | 728 fits | yes |

P2 fails, so the label is `not supported`; `worsened_generalization` is false. The interval is a paired
country-resampling interval (2,000 draws, seed 0), not a spatially corrected confidence interval.
**Retained static specification: M0\*** (branch A of `DOWNSTREAM_COMPLETION_SPEC.md` §1).

**Static stopping indicator: does not fire.** The best point ΔRMSE over M1a (+0.002343), M1b (+0.001723) and
M2 (−0.002952) is −0.002952, which is not > −0.002. The rule concerns point estimates and is independent of the
label. So the M4 stopping conclusion ("not recoverable with country-level static covariates") is **not**
declared by the rule, while M2 is also not supported. No branch authorizes a further static-model search.

## 2. Scorecard (primary total-CO₂ representation, n = 151)

| Metric | M0\* | M2 |
|---|---:|---:|
| In-sample R² | 0.63645 | 0.69744 |
| Primary 500 km LOCO: CV R² / RMSE / MAE | 0.15308 / 0.042873 / 0.034465 | 0.26570 / 0.039920 / 0.031692 |
| M49 holdout: R² / RMSE | 0.30297 / 0.038894 | 0.27399 / 0.039694 |
| Random 10-fold (reference): R² / RMSE | 0.42835 / 0.035223 | 0.50318 / 0.032837 |
| Worst eligible region (Central Asia, n = 4) RMSE | 0.069364 | 0.064975 |
| Residual Moran's I in-sample / OOF (station kNN8) | 0.2687 / 0.3215 | 0.2657 / 0.3446 |
| Calibration slope / intercept | 0.609 / 0.069 | 0.684 / 0.055 |
| Effective degrees of freedom | 20 | 23 |
| Shares: geography / emissions / socioeconomic / population | 0.5188 / 0.0131 / 0.0628 / 0.0417 | 0.5821 / 0.0110 / 0.0621 / 0.0423 |
| Residual share | 0.3636 | 0.3026 |

M0\* reproduces its frozen rank-audit cards to 1e-10. Per country (primary): 81 improved, 70 worsened. The
largest improvements are Papua New Guinea, Argentina, South Africa, Botswana and Kenya; the largest
deteriorations are DR Congo, Norway, Peru, Iceland and Congo.

## 3. Diagnostics (never verdict conditions)

* **γ** (southern-minus-northern latitude-slope difference): full fit −0.002554 °C/decade per degree; strictly
  negative in **151 of 151** primary training fits (median −0.002540, range −0.003407 to −0.001944). No cutoff
  applies. A negative γ means a lower southern slope relative to the north; it does not establish positive
  northern or non-positive southern absolute slopes, and identifies no mechanism.
* **Overfitting clauses:** (i) in-sample R² gain 0.0610 minus CV R² gain 0.1126 is not > 0.05; (ii) CV RMSE
  did not worsen; (iv) df rise 3; no signal. Clause (iii) is reported without a cutoff (γ same-sign fraction 1.0).
* **Residual spatial structure:** essentially unchanged (in-sample I −0.003, OOF I +0.023).
* **Scientific stability:** stable; geography is the largest named group, and emissions is 0.011.
* **Three-dimension summary:** no detectable change in transfer; spatial autocorrelation essentially unchanged;
  stable.
* **Extrapolation disclosure** (counts equal the pre-registered feasibility record). Iceland, above its
  primary training maximum, has OOF error −0.086 (M0\*) versus −0.124 (M2). With Northern Europe held out
  (M49), eight countries lie above the training maximum and their errors change in both directions. Gabon
  lies below its training minimum.

Raw spline-basis, linear `abs_latitude` and `hemisphere=S` coefficients are recorded but not interpreted.

## 4. Representation identity (per-capita, reported unconditionally)

Predictions agree to 2.8e-14 in every protocol, and ΔRMSE and interval bounds to below 1e-15. The latitude
coefficients agree to 3.1e-16 in all 151 training fits. The same Booleans give `not supported` descriptively.
Shares differ by construction.

## 5. Verification

* **Independent reconstruction** (`m2_independent_check.py`, a separate path using only numpy, pandas and the
  stdlib): it rebuilds the designs from saved states and labels and refits all 728 fits by QR (coefficients
  to 1.3e-14). It recomputes every prediction (2.7e-14), the metrics, the paired interval, the Booleans, the
  label, the stopping indicator, γ, coalition R² and permutation-Shapley shares, and the representation
  identity. Every check passes, and five negative controls are detected: a perturbed prediction, a shifted
  threshold, an altered knot, swapped training membership and an altered coefficient.
  * Evidence: `outputs/m2_primary_verification/m2_independent_check.json`.
  * Scope limits recorded there: same inputs; bounds, M49 labels, Moran and calibration not re-derived;
    written in the same session before the result.
* **Reproducibility:** one unchanged-code rerun into a fresh root
  (`outputs/m2_primary_verification/reproducibility_run_b/`). All ten deterministic artifacts and the manifest
  are byte-identical (`m2_reproducibility.json`). Only run metadata differs.
* **Pre-freeze disclosure** (`M2_EVALUATOR_SPEC.md` §8a). Baseline-only M0\* anchors and predictor-only
  identity checks ran from uncommitted code during implementation. No M2 candidate was fitted to the real
  outcome before this score.

## 6. Interpretation (contract §11)

* The joint pre-registered latitude form does not demonstrate transferable improvement at n = 151 under the
  frozen criteria. The point improvement exceeds the practical threshold, but the paired interval reaches
  above zero.
* This is not evidence that the latitude relationship is linear, and no rescue follows: no H1-only,
  spline-only or alternative form.
* M2 is a geography functional-form result. It is not evidence about hydroclimate, snow/ice feedback or any
  mechanism.
* The responsibility statement is unchanged: shares are explained variance of cross-country differences in
  area-weighted warming, not greenhouse gases' physical role.

## 7. Next (frozen consequences)

* **Conditional arms: not executed.** The label records no predictive support, so the non-execution record is
  written before any arm is constructed (checkpoint 4).
* **M3** runs on **M0\*** with both families and both weights arms (`DOWNSTREAM_COMPLETION_SPEC.md` §2).
