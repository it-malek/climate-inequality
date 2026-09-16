# M1b result — baseline hydroclimatic dryness is not supported

> **Post-result documentation corrections, 2026-09-16.** Three passages below (the opening chronology,
> the §3 allocation sentence and one §6 sentence) were corrected after the result. They are marked
> *[corrected 2026-09-16]* and quoted in their original form in [`M1B_CLOSURE.md`](M1B_CLOSURE.md) §4.
> The original report is preserved at `1680d85`. No number, verdict or artifact changed.

**Scored 2026-09-15 from the frozen evaluator commit `618c8b8`, pushed before the run, with a clean
working tree and `PYTHONHASHSEED=0`.** *[corrected 2026-09-16]* The original measurement and
evaluation contract, [`M1B_EVALUATION_CONTRACT.md`](M1B_EVALUATION_CONTRACT.md), was frozen and pushed
before any CRU data were acquired (`60111ae`). Amendments 1–3 were adopted after acquisition and after
C2 values existed under the original coverage rule, but before the redundancy diagnostic and before any
outcome-facing fit involving C2. The operational detail in [`M1B_EVALUATOR_SPEC.md`](M1B_EVALUATOR_SPEC.md) was
frozen before this run. Nothing was changed between the freeze and the score.

**Verdict (contract §4.4, applied mechanically): `not supported`.**
Adding C2 made out-of-fold error **worse**, and the coefficient carries the **opposite** of the
pre-stated sign.

## 1. Headline comparison (primary total-CO₂ representation, 151 countries)

| Quantity | M0\* | M1b = M0\* + C2 | Change |
|---|---:|---:|---:|
| In-sample R² | 0.6364460028 | 0.6364565669 | +0.0000106 |
| Spatial-CV R² (500 km LOCO) | 0.1530799130 | 0.0836391291 | **−0.0694408** |
| Spatial-CV RMSE (°C/decade) | 0.0428725807 | 0.0445955657 | **+0.0017230** |
| Spatial-CV MAE | 0.0344645875 | 0.0356116337 | +0.0011470 |
| Fold RMSE median / max | 0.0302127 / 0.1147346 | 0.0313481 / 0.1203339 | +0.0011 / +0.0056 |
| \|OOF error\| IQR | 0.0430306 | 0.0448220 | +0.0017914 |
| Worst fold country | Brazil | Egypt | — |
| M49 secondary RMSE | 0.0388941369 | 0.0396928954 | +0.0007988 |
| M49 secondary R² | 0.3029700198 | 0.2740466069 | −0.0289234 |
| Random 10-fold RMSE (reference only) | 0.0352226782 | 0.0356079866 | +0.0003853 |
| Worst eligible region (n ≥ 3) | Central Asia 0.0693644 | Central Asia 0.0734948 | **+0.0041304** |
| Residual Moran's I, in-sample | 0.2686890161 | 0.2680749683 | −0.0006140 |
| Residual Moran's I, out-of-fold | 0.3215471677 | 0.3269143648 | +0.0053672 |
| Matrix columns / rank | 20 / 20 | 21 / 21 | — |
| Rows with an unseen level | 1 | 1 | — |

**Paired spatial-CV comparison** (contract §4.3, unchanged: 2,000 resamples, seed 0, the same
resampled countries for both models, fixed out-of-fold errors, no refitting):

* ΔRMSE = **+0.0017229851 °C/decade**;
* 95% paired resampling interval **[+0.0008914058, +0.0028494442]**, entirely **above** zero.

This is a paired country resampling interval for the difference in out-of-fold error. It is **not** a
spatially corrected confidence interval; the geographic safeguards remain the corrected 500 km
spatial CV, the M49 holdout and the worst-region veto.

**`baseline_dryness` coefficient:** full fit **+0.00047407**, i.e. positive, where the registered
hypothesis required negative. Over the 151 primary training fits it is negative in **68** (fraction
**0.4503**, against the required ≥ 0.80), with median +0.00026, range [−0.02009, +0.00928], none
exactly zero, and all 151 finite and identified.

**Per country:** 44 improved, **107 worsened**. The largest deteriorations are Egypt (+0.0278),
Libya (+0.0193), Chad (+0.0136), Jordan (+0.0124) and Kazakhstan (+0.0118); the largest improvements
are Iraq (−0.0064), Saudi Arabia (−0.0024) and Türkiye (−0.0020).

## 2. The verdict, condition by condition

| Condition | Value | Result |
|---|---|---|
| S1: ΔRMSE < 0 and interval upper bound < 0 | +0.0017230, upper +0.0028494 | **fails** |
| S2: coefficient < 0 in the full fit and in ≥ 80% of 151 training fits | +0.00047407; 68/151 = 0.4503 | **fails** |
| A1 (= S1) | — | fails |
| A2: ΔRMSE ≤ −0.002 | +0.0017230 | fails |
| A3: M49 RMSE change ≤ +0.001 | +0.0007988 | passes |
| A4: worst-eligible-region RMSE change ≤ +0.001 | +0.0041304 | fails |
| A5 (= S2) | — | fails |

**Reported verdict: `not supported`.** Also reported: `worsened_generalization = true` (ΔRMSE > 0
with the lower interval bound > 0); `improvement_without_prestated_sign = false`.

Because the linear association is **not** supported, the registered conditional replications
(§5 items 1 and 3, and the Amendment 2 five-country station-support arm) are **not run**. That is the
contract's rule, not a choice made after seeing these numbers, and no favourable sensitivity was
searched for. The runner records the non-execution rather than being skipped silently.

## 3. Diagnostics (contract §4.5; never verdict conditions)

* **Three-dimension summary:** transfer *worsened generalization*; spatial specification *residual
  spatial autocorrelation essentially unchanged*; stability *stable*.
* **Overfitting signal: true** by the plan's second clause — out-of-fold RMSE worsens while in-sample
  RMSE improves. The in-sample R² gain is +0.0000106 against a CV R² loss of −0.0694; C2 buys
  essentially nothing in sample and costs transfer.
* **Residual spatial structure:** in-sample I −0.00061, out-of-fold I +0.00537, both far inside the
  ±0.05 band. The recorded area-centroid values move likewise (0.28992 → 0.28911 in sample).
* **Scientific stability: stable.** Geography remains the largest named group and the responsibility
  share stays far below 0.10 (0.0131 → 0.0111).
* **Group shares (in-sample LMG/Shapley, primary representation).** M0\*: geography 0.5188,
  socioeconomic 0.0628, population 0.0417, responsibility 0.0131, residual 0.3636. M1b: geography
  0.4840, socioeconomic 0.0593, population 0.0362, responsibility 0.0111, **hydroclimate 0.0458**,
  residual 0.3635. *[corrected 2026-09-16]* Relative to M0\*, the named-group allocations shift by:
  geography −0.03484, population −0.00543, socioeconomic −0.00348, responsibility −0.00206 and
  hydroclimate +0.04583. The net change equals the in-sample R² gain (+0.0000106) by construction,
  because LMG shares sum to R². Geography accounts for about 76% of the displaced allocation. That
  split reflects C2's correlation with each group under LMG's averaging over coalitions, and it does
  not identify a physical source. It is consistent with the pre-outcome redundancy diagnostic: about 65% of C2's cross-country variance
  already lies in the span of the M0\* design, and geography alone reaches R² 0.62. **A nonzero
  hydroclimate share allocates shared explanatory variance in sample; it is not a fraction of warming
  physically caused by hydroclimate, and here it coexists with worse out-of-fold error.**

## 4. Per-capita representation (Amendment 3 A3.6, reported unconditionally)

The per-capita responsibility representation was scored on the same sample, outcome, encoders, folds
and C2. Exactly one contract verdict is reported — the primary one; the per-capita arm carries the
identical Booleans under a descriptive label and decides nothing.

As the representation identity `log10(total Mt) = log10(per-capita t) + log10(population) − 6`
requires (it held to 1.4e-13 on the frozen design), the two representations agree:

* predictions, in-sample and all three protocols: max absolute difference **4.4e-14** (tolerance 1e-9);
* non-allocation scores: max absolute difference **4.9e-14**;
* ΔRMSE differs by 1.2e-16 and the interval bounds by ≤ 3.9e-16;
* the `baseline_dryness` coefficient differs by 1.6e-17 in the full fit and by ≤ 2.6e-16 across all
  151 training fits.

Allocations differ, as expected and permitted: responsibility 0.0175 vs 0.0111 and geography 0.4801
vs 0.4840 in M1b, and the hydroclimate share is 0.0449 vs 0.0458 (difference 0.0009). No
representation was chosen by its shares, and the per-capita arm reaches the same conditions: S1 and
S2 both fail.

## 5. Verification

* **Reproducibility.** An unchanged-code rerun in a separate process and a separate output root
  reproduced **all ten deterministic artifacts and the result manifest byte for byte**; only
  `m1b_run_metadata.json` differs, by design (timestamps, host, wall time, upstream head).
  Record: `outputs/m1b_primary_verification/m1b_reproducibility.json`.
* **Independent recomputation** (`m1b_independent_check.py`, written without reading the evaluator,
  `cv.py`, `spatial.py`, `run_territory_correction.py` or `src/decomposition.py`, and sharing no code
  with them): from the saved predictions, coefficients and coalition R² values it re-derived the
  in-sample and out-of-fold R²/RMSE/MAE, the fold and region summaries, the paired interval from the
  contract text, the strictly-negative coefficient count, every verdict Boolean and the label, the
  representation-equivalence differences, the Shapley aggregation, and the residual Moran statistics
  on both station-centroid and area-centroid weights rebuilt from scratch. It also re-derived the
  saved fold evidence's internal consistency: the per-country fold ids against the predictions, the
  training sizes against the pipe-joined memberships, no country in its own training set, symmetry of
  the 21,332 directed memberships (as a distance threshold requires), the M49 partition, and
  `folds_random(151, 10, seed=0)` exactly. **All 392 comparisons pass**; the worst absolute difference
  is 5.6e-16. Record: `outputs/m1b_primary_verification/m1b_independent_check.json`.
  * Negative controls were run for every family (perturbing a reported value, k=7 instead of k=8,
    station centroids in place of area centroids, a rolled random fold, a swapped membership): each
    fires, so no check passes vacuously.
  * Its stated limits: predictions, coefficients and coalition R² values are inputs to that check, so
    it validates the **aggregation** of the Shapley decomposition, not the coalition regressions; it
    does not re-derive the encoders, the outcome's provenance, or the 500 km territorial bounds, so
    it shows the fold memberships are coherent and symmetric, not that the excluded set is the right
    one. Those are covered instead by the evaluator's own input gate, which verified the folds
    against the frozen `4263429` memberships and the outcome against the frozen ordered record
    before scoring, and by the Phase A check that the derived memberships reproduce that artifact.
  * One methodological note the check surfaced: pandas' default CSV float parser is not round-trip
    exact (~1e-14 relative on this platform), so anyone re-reading these artifacts should parse with
    `float_precision="round_trip"`, as the module does.
* **Frozen-card anchors, checked before either candidate was fitted:** both comparators reproduced
  their `rank_audit.json` cards (`drop_per_capita`, `drop_total`) on 24 metrics to ≤ 2.8e-17, against
  a 1e-10 tolerance.
* **Share accounting:** named shares sum to the in-sample R² and, with the residual, to 1, to
  ≤ 5.6e-16 for every model and representation.

## 6. Interpretation (fixed in the contract before C2 was seen)

**The pre-registered linear C2 association is not demonstrated at n = 151 under the frozen criteria.**

* This is **not** evidence that hydroclimate is irrelevant to cross-country warming differences.
* It may reflect the geography block already carrying this information (the pre-outcome redundancy
  diagnostic: R² 0.650 of C2 on the M0\* design, geography alone 0.622, VIF 2.86), the climatological
  anchoring of the CRU 1920–1949 fields, noise in C2, limited power at n = 151, or a relationship
  that is not linear.
* The sign is also wrong, not merely weak: the full-fit coefficient is positive and fewer than half
  the training fits are negative. Under the contract, improvement without the pre-stated sign would
  not be support either — and here there is no improvement to begin with.
* **Non-linear dryness terms, interactions and alternative C2 definitions are not introduced in
  response to this result.** *[corrected 2026-09-16]* No alternative window, transform, dataset,
  threshold or aggregation order was tried, before or after. The coverage rule changed only through
  the recorded pre-outcome contract Amendments 1 and 3, and no coverage rule was tried in response to
  this result.
* The responsibility statement is unchanged: these shares are explained variance in cross-country
  differences in area-weighted warming, not greenhouse gases' physical role.

**Limitations that travel with this result.** C2 is **CRU-reconstructed** 1920–1949 baseline
hydroclimatic dryness over *resolved* support — not an independently observed pre-1950 state: CRU
converts station series to anomalies against 1961–1990 normals and inserts that climatology where
stations are absent, and PET adds partly synthetic inputs and a static wind climatology. PET support
is not quantifiable from the pinned files. Resolved coverage is ≥ 0.98 everywhere (minimum Bahamas
0.9888), which bounds but does not eliminate bias from missing support; 6,599 km² stays unresolved,
including Peru's zero-precipitation cell. The paired interval treats countries as exchangeable and
does not resolve spatial dependence.

## 7. Artifacts

All under `research/model_v2/outputs/`, with SHA-256 digests in `m1b_primary/m1b_result_manifest.json`:

| File | Contents |
|---|---|
| `m1b_primary/m1b_scorecard.json` | both representations' cards, paired comparison, verdict, diagnostics, equivalence, redundancy and support context |
| `m1b_primary/m1b_country_predictions.csv` | 2,416 rows: representation × model × protocol × country, with errors, fold ids, training sizes and unseen-level counts |
| `m1b_primary/m1b_primary_country_comparison.csv` | per-country primary OOF comparison and the change in absolute error |
| `m1b_primary/m1b_cv_folds.csv` | per-country primary fold, training size and full training membership, M49 and random folds |
| `m1b_primary/m1b_coefficients.csv` | every full-sample and primary-training-fit coefficient, with each fit's column count, rank and identification flag |
| `m1b_primary/m1b_coefficient_summary.json` | per-column full value, fold distribution, same-sign fraction, strictly-negative count |
| `m1b_primary/m1b_shapley_coalitions.csv` | every group-coalition R² behind the shares |
| `m1b_primary/m1b_design_matrices.csv` | the encoded full-sample designs by country, column and group |
| `m1b_primary/m1b_provenance.json` | evaluator commit, code-path digests, every input digest and the frame, fold and weight identities |
| `m1b_primary/m1b_run_metadata.json` | the one non-deterministic artifact |
| `m1b_primary_verification/m1b_reproducibility.json` | the two-run byte comparison |
| `m1b_primary_verification/m1b_independent_check.json` | the independent recomputation |

**Reproduce:**

```
PYTHONHASHSEED=0 uv run python -m research.model_v2.m1b_evaluate --out <fresh dir>          # at 618c8b8
PYTHONHASHSEED=0 uv run python -m research.model_v2.m1b_independent_check                   # independent check
```

## 8. Known defect in an auxiliary code path, disclosed, not patched

After scoring, the evaluator's `--compare` mode (the two-run byte comparison) was found to raise
`KeyError: 'integrity_passed'` in its `__main__` exit line, because that line assumes a scoring
result. **It has no numerical effect of any kind:** the comparison itself ran, found the runs
byte-identical and wrote its record before the failing line, and the scoring path returns a result
that carries `integrity_passed` and exited 0. It is recorded here rather than patched, because
changing the evaluator now would break the code-path identity between this result and the commit it
records. Any fix belongs to a separate, dated change after this milestone.

**STOPPED AFTER M1b.** No conditional analysis was run, and no C1, C3, other M1 predictor, M2 or M3
work follows from this result.
