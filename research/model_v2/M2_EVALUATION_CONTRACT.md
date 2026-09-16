# M2 evaluation contract — frozen before any M2 fit

**Recorded 2026-09-16. Frozen and pushed before any M2 model is fitted, scored or cross-validated.
This contract does not authorize execution (§9).**
* The candidate and its mathematics are in [`M2_DESIGN.md`](M2_DESIGN.md).
* Their historical basis and limits are in [`M2_PROVENANCE_AUDIT.md`](M2_PROVENANCE_AUDIT.md).
* Structural feasibility is in [`M2_FEASIBILITY_AUDIT.md`](M2_FEASIBILITY_AUDIT.md). The canonical
  record `outputs/m2_feasibility/` was written from pushed commit `fbaa3ff`.

> **Amendment 1 (owner, 2026-09-16, before any M2 fit):** [`M2_PRE_SCORE_RESOLUTIONS.md`](M2_PRE_SCORE_RESOLUTIONS.md)
> authorizes execution and the downstream stages, confirms the §12 interpretations, corrects the H5 wording
> (its §4), and removes the sign-stability cutoff from the γ diagnostic, so §6.2 clause (iii) is reported as a
> fraction without a thresholded Boolean (its §5). The rest of this contract is unchanged. The downstream M3
> and M4 rules are frozen in [`DOWNSTREAM_COMPLETION_SPEC.md`](DOWNSTREAM_COMPLETION_SPEC.md).

**Where each rule comes from.**
* **Inherited:** unchanged from the `af97bd2` plan, the corrected protocol (`4263429`), the M1a
  contract (`e3e5901`) or the M1b contract and specification (`60111ae`…`618c8b8`).
* **Owner, 2026-09-16:** directed by the owner in this session, before any M2 fit and after the M1a
  and M1b outcomes.
* **Design, 2026-09-16:** proposed and frozen in this design and **not separately owner-approved**. It
  may be changed only by a dated amendment committed before the execution freeze (§9), never after any
  M2 number exists. All design choices are listed in §12.

## 1. Question and status

**Question.** Does M0\* plus the pre-registered latitude functional form (§2) transfer better to held-out
countries than M0\*, on the same outcome and sample, under the frozen spatial protocol?

**This is one pre-registered operational comparison.** Its hypotheses were generated historically from
the same outcome's M0 residuals. The eventual test is a **prospectively frozen follow-up within the same
outcome dataset**, not untouched-data or independent external confirmation. It is reported primarily as
a **predictive functional-form test** (`M2_DESIGN.md` §6). No multiplicity adjustment is made for a
single comparison.

## 2. Models (owner, 2026-09-16; design §1–§4)

* **Comparator M0\*** (inherited, unchanged).
  * **Specification.** Drop `cum_co2_per_capita`; station geography; 151 countries in frozen row order;
    20 columns, rank 20.
  * **Reproduction.** It must reproduce the `rank_audit.json` `drop_per_capita` card, and the per-capita
    arm the `drop_total` card, on the M1b reference metrics to 1e-10.
* **Candidate M2** = M0\* + `abs_latitude_ns1` + `abs_latitude_ns2` + `hemisphere=S:abs_latitude`,
  all in `geography`.
  * **Budget.** +3 coefficients; full-sample rank 23.
  * **Latitude state.** From each fit's training predictors only, under basis `m2-latitude-basis-v1`
    (`m2_latitude_basis.py` at `fbaa3ff`).
  * **Interaction coding.** `I(SH)·abs_latitude` with centre 0°.
* **No other model is fitted at any point**, including as an "exploratory" variant:
  * H1-only;
  * spline-only;
  * any other df, knots, quantile rule, boundary convention, basis, spline package or centering;
  * NH-only curvature;
  * any C2, C1 or C3 term.

  No such latitude-form variant is scored on this outcome dataset in this research line, before or
  after the M2 result, and the M2 result is not cited as motivation for one.
* **Complexity control** (owner). The fixed +3 budget and the out-of-fold gates below are the complexity
  control. **No AIC, BIC, cross-validated penalty or other numerical penalty** is introduced, before or
  after fitting.
* **Encoders** (inherited). Train-only categorical encoders; the mean-training-level-effect rule for
  unseen levels; the V1 estimator (`lstsq`, `rcond=None`); `PYTHONHASHSEED=0`.

## 3. Entry gate (owner, 2026-09-16; `M2_PROVENANCE_AUDIT.md` §5)

* **The gate's object.** The historical M1 → M2 gate applies to "the retained M1 model". No M1 addition
  was promoted, so the retained model is **M0\***.
* **The gate is met on a frozen value.** Condition (ii) holds on the frozen M0\* out-of-fold residual
  Moran's I **0.3215471676927579** > 0.2 (`rank_audit.json`, `9fb56e1`, before the M1a score).
* **Nothing new is computed for it.** No M1b residual is analysed, and conditions (i) and (iii) are not
  computed.
* **The gate is procedural, not evidence.** Its thresholds were set with M0 values already above them,
  and residual autocorrelation does not by itself establish nonlinear misspecification.

## 4. Protocols and paired uncertainty (inherited, unchanged)

* **Protocols.**
  * Primary: the corrected 500 km footprint-certified leave-one-country-out, lower bounds from
    `4263429`, memberships identical to the frozen `m1b_cv_folds.csv`.
  * Secondary: leave one M49 subregion out (20 folds).
  * Reference only: random 10-fold, seed 0.
* **Sample and outcome.** 151 countries; area-weighted Berkeley Earth trend 1950-01…2013-09 (°C/decade);
  identical to M0\*.
* **Paired interval.** `run_territory_correction.paired_rmse_interval`, unchanged. The 151 countries
  are resampled 2,000 times with replacement (seed 0); the same indices are used for both models' fixed
  out-of-fold errors; the statistic is RMSE(M2) − RMSE(M0\*); the interval is the 2.5th and 97.5th
  percentiles.
  * It is a paired country-resampling interval, **not** a spatially corrected confidence interval.
  * The geographic safeguards remain the 500 km CV, the M49 holdout and the worst-region veto.
* **Worst eligible region** (inherited definition, M1b A4 as pinned in `M1B_EVALUATOR_SPEC.md` §5). Each
  model's **own** worst M49 subregion with n ≥ 3 under the primary protocol:
  * `worst(M) = max over eligible subregions of RMSE_M(subregion)`;
  * the veto compares worst(M2) − worst(M0\*).

  This is not the candidate's RMSE in the comparator's worst region, and it is not a per-region
  deterioration rule.

## 5. Verdict (owner, 2026-09-16; at most one label, from the primary representation; none on an integrity failure)

**Level 1 — predictive support.** Both of:
* **P1:** primary paired ΔRMSE = RMSE(M2) − RMSE(M0\*) **< 0**; and
* **P2:** the upper endpoint of the §4 interval **< 0**.

**Level 2 — promotion to the next primary development baseline.** P1 and P2, plus all of:

| # | Condition | Origin |
|---|---|---|
| R1 | ΔRMSE ≤ **−0.002 °C/decade** | inherited (M4 practical threshold, applied as in M1b A2) |
| R2 | M49 RMSE(M2) − RMSE(M0\*) ≤ **+0.001 °C/decade** | inherited (plan's secondary veto; M1b A3) |
| R3 | worst(M2) − worst(M0\*) ≤ **+0.001 °C/decade** (§4) | inherited (M1a/M1b A4) |
| R4 | structural integrity (§7) holds in every required fit | owner (requirement); design (scope, §12) |

**Labels.**
* **`supported and promoted`:** P1, P2 and R1–R4 all hold.
* **`predictive support, not promoted`:** P1, P2 and R4 hold, but R1, R2 or R3 fails.
* **`not supported`:** R4 holds, but P1 or P2 fails.
* **Reported alongside every label:** **`worsened_generalization`**, which is true when ΔRMSE > 0 and
  the lower interval endpoint > 0.

**Integrity failures (§7).** No label is reported. What happens depends on when the failure is found:
* **Before any candidate is fitted:** it is a §9 refusal. Nothing is written, and it may be retried only
  from the identical frozen commit.
* **After any candidate fit:** it is recorded in the output root with no label, and it stops the line
  for owner review (§9).

**Reconciliation with the historical gates.**
* **Genuine improvement.** The plan defines it as a negative ΔRMSE with an interval excluding zero,
  secondary not worse by more than 0.001, and, for a covariate, a sign rule. Promotion contains the first
  two parts; predictive support omits the secondary veto and is reported as a weaker label.
* **The sign rule is an owner-directed departure.** Spline basis coefficients have no invariant sign,
  but γ does, and H1 pre-states it as negative. Under the plan's "retention as in M1", γ would face the
  covariate sign rule: full fit and ≥ 80% of primary training fits.
  * **The owner removed every coefficient-sign criterion for M2 on 2026-09-16**, after the M1b result
    (in which C2 was wrong-signed).
  * This is a dated, disclosed **loosening** relative to the `af97bd2` rule and to M1b S2/A5.
  * γ's sign and sign stability are reported only as diagnostics (§6.2) and cannot change a label.
* **The plan's M2 overfitting gate.** The plan says "retention as in M1, with the overfitting signal
  checked explicitly". The owner's requirement 4 defines promotion exhaustively, so the overfitting
  signal is reported in full (§6.2) but does not veto. As in M1b, this is a diagnostic, not a verdict
  condition.

## 6. Reported evidence

### 6.1 Scorecard (inherited, both models, both representations)

* **Scores.** In-sample R²; spatial-CV R², RMSE and MAE; M49 R² and RMSE; random-reference R² and RMSE.
* **Folds.** Fold RMSE median, minimum and maximum, and the worst fold country; IQR of |OOF error|.
* **Regions.** Per-region RMSE and bias, and each model's worst eligible region.
* **Calibration and design.** Calibration slope and intercept; unseen-level rows by feature; matrix
  columns and rank.
* **Residual Moran's I**, in-sample and out-of-fold. V1 station-centroid kNN8, 999 permutations,
  seed 0, with area-centroid I recorded.
* **Group shares** (§8).
* **Coefficients, with limits on interpretation.** Every full-fit coefficient is reported. For the
  ns1, ns2, linear `abs_latitude` and `hemisphere=S` columns:
  * they are reported **only as basis coefficients**;
  * no sign, same-sign fraction or cross-fold summary of them is interpreted or presented as stability
    evidence. Their meaning changes with each fit's knots (`M2_DESIGN.md` §6).

  Only γ is an interpretable shape quantity.
* **Paired comparison.** ΔRMSE and its interval; ΔR² in-sample and CV; ΔI; share changes; per-country
  change in |OOF error|, with counts improved and worsened.

### 6.2 Diagnostics (never verdict conditions)

* **Overfitting signal.** All four clauses of the plan's definition are reported. M1a and M1b reported
  only (i) and (ii); here all four are restored (design).
  * (i) in-sample R² gain − CV R² gain > 0.05;
  * (ii) OOF RMSE worsens while in-sample RMSE improves;
  * (iii) sign instability < 80%, evaluated **only for γ**. Basis coefficients are excluded as
    parameterization-dependent (design);
  * (iv) effective df rise > 5 with an RMSE gain < 0.002. It is **evaluated**: effective df rises
    20 → 23 (+3 ≤ 5), so clause (iv) is recorded as not triggered.
* **Residual spatial structure** (inherited). Reduced if in-sample and OOF I both fall by ≥ 0.05;
  increased if both rise by ≥ 0.05; otherwise essentially unchanged.
* **Scientific stability** (inherited). A material change is geography no longer being the largest
  named group, or the responsibility share exceeding 0.10.
* **Three-dimension summary** (inherited): transfer, spatial specification, stability.
* **Shape diagnostic: γ only** (design; `M2_DESIGN.md` §6). The full fit, the γ < 0 fraction over the 151
  primary training fits, and the median and range.
  * No curvature contrast is reported as a shape test: the pooled tertile spline cannot represent an
    onset near 50°N.
  * Nothing here is interpreted as confirming a mechanism.
* **Extrapolation disclosure** (design). Out-of-fold errors are reported separately for held-out rows
  outside their fit's knot range, as listed in advance in `M2_FEASIBILITY_AUDIT.md` §4:
  * primary: Gabon and Iceland;
  * M49: Middle Africa (Gabon) and Northern Europe (8 countries);
  * random reference: folds 5 and 6.

  This is descriptive only.
* **Explicitly exploratory.** Per-region, per-country and extrapolated-row breakdowns, and γ
  distributions. They describe the single candidate. **No variant model is fitted**, and nothing is
  selected from them.

## 7. Structural integrity and refusals (owner requirement; design scope)

**Required fits.** In both representations, for the station design:
* the full sample;
* every primary training fit (151);
* every M49 training fit (20);
* every random-reference fit (10).

Including the random-reference fits as integrity conditions is a design choice (§12). Their structural
integrity is required like any other required fit's; their scores are reported only and are never
verdict criteria.

**Integrity failures.** Each of the following stops the run, writes no verdict label and applies no
fallback: no df reduction, knot change, column drop, regularization or substituted basis.
1. A degenerate latitude state in any required fit.
2. M0\* rank ≠ M0\* columns in any required fit (design: per-fit full rank is required).
3. Candidate rank ≠ candidate columns, or candidate rank − M0\* rank ≠ **3**, in any required fit. At
   full sample the ranks must be 20/20 and 23/23.
4. `lb.fit_state(full-sample abs_latitude)` ≠ `LatitudeState.from_json` of the committed record's
   `full_sample/station/<representation>/latitude_state`. This is exact equality through the hex knots.
5. The full-sample M0\* encoded design ≠ the frozen `M0star` rows, value for value after aligning
   columns by name.
6. M0\* fails to reproduce its frozen `drop_per_capita` / `drop_total` card to 1e-10 (inherited).
7. The representation identity fails (§8).
8. Any verdict input is non-finite (inherited: a refusal, not a False).

When a failure is detected decides its handling (§5): before any candidate fit it is a refusal that
writes nothing; after, it is recorded with no label and stops the line.

Rank uses `numpy.linalg.matrix_rank` at its default tolerance (inherited).

## 8. Decomposition (owner, 2026-09-16; design §7)

* **Groups.** There are four named groups: `geography`, `emissions`/responsibility, `socioeconomic` and
  `population`. That gives 16 coalitions with a common intercept.
* **The geography block stays whole.** It contains linear `abs_latitude`, both spline columns, the
  hemisphere main effect, the interaction and every other geography predictor, together in every
  geography coalition, with no split allocation.
* **Latitude state.** All coalitions use the full-data state. Any later refit derives its own state from
  its own rows (`M2_DESIGN.md` §3.3).
* **Definition.** The existing R²-based LMG/Shapley, unchanged; M2 is OLS. Shares are fractions of total
  outcome variance, and named plus residual equals 1 within the inherited share-accounting tolerance of
  1e-9.
* **Per-capita representation** (inherited, A3.6). Reported **unconditionally**, with the same sample,
  outcome, folds, encoders, latitude states and group definitions. It decides nothing.
  * The representation identity must hold, or the run stops (§7 item 7). The prediction and
    non-allocation-score tolerances of 1e-9 are inherited from M1b.
  * Design extension: the γ, ns1, ns2 and linear `abs_latitude` coefficients must agree to **1e-8** in
    the full fit and in every primary training fit. M1b applied that tolerance to its added coefficient.
  * Shapley allocations and the population and intercept coefficients may differ, and are reported.
* **Interpretation.** A descriptive allocation among correlated groups, never causal attribution. A
  change in the geography share is an accounting consequence, not a physical source.

## 9. Execution boundary (owner, 2026-09-16)

**This contract authorizes no M2 execution.** A later, separately authorized step must first commit and
push:
* an M2 evaluator specification, an implementation and its tests, with no outcome-facing run;
* frozen identities for:
  * every scoring input, using the M1b scoring-input pins minus the C2 package;
  * the fold memberships;
  * the M0\* reference cards and encoded design;
  * this contract and `M2_DESIGN.md`;
  * `m2_latitude_basis.py`;
  * the feasibility record;
* the evaluator's own code path, including the latitude basis and every local dependency, with a
  provenance guard of the M1b kind: committed, unmodified and pushed.

**Then:**
* the first and only score runs from that **exact clean, pushed commit**, with `PYTHONHASHSEED=0`, into
  a fresh output root. Platform, Python and numpy versions are recorded; scores are platform-scoped in
  their last bits;
* **a refusal that happens before any candidate is fitted** writes nothing. It may be retried only from
  the identical frozen commit. If passing requires any code change, the change goes to owner review as a
  new freeze;
* the first score is preserved as produced;
* a numerical or implementation defect found after any M2 number exists **stops** the line for owner
  review. There are no scientific code fixes followed by silent reruns;
* reproducibility reruns use unchanged code only.

## 10. Robustness (inherited requirements and design decisions)

* **Product.**
  * Inherited (`PRODUCT_STABILITY_AUDIT.md` §7): ERA5 is never used to choose M2's form, knots,
    transforms, thresholds or CV design. Its role is external robustness after the specification is
    frozen.
  * Design:
    * an **aligned-ERA5** replication of the frozen M0\*-vs-M2 comparison is run only if predictive
      support holds, and only after the primary result is committed;
    * it has the same predictors and 151 countries, so it **reuses the primary per-fit latitude states
      unchanged**;
    * it is descriptive, reports both representations, and cannot change the verdict.
  * Departure from §7: running the aligned construction only, not "both constructions", is a design
    departure. It follows the M0.5 decision that prefers the aligned arm and the M1b §5 precedent
    (§12).
* **Measurement.**
  * Inherited (`M1A_REPORT.md` §13): the two measurement sets are never searched jointly.
  * Design: under the same condition and timing, the frozen comparison is replicated with the frozen M1a
    area measurements. The latitude state is recomputed per fit from M1a `abs_latitude`, and the
    interaction uses M1a `hemisphere`; Gabon and Kenya switch hemisphere. It is descriptive only.
  * Its structure was verified in advance: it passes every fit (`M2_FEASIBILITY_AUDIT.md` §2.2).
* **Conditional-arm failures.** A structural failure in either conditional arm is recorded as
  **non-computable**, with no repair, and cannot affect the verdict.
* **Not transplanted.** M1b's five-country station-support sensitivity was specific to the CRU support
  of C2. M2 contains no C2, so it does not apply.
* **No other sensitivity** is run, and none is searched for.

## 11. Interpretation (fixed now)

* **`supported and promoted`.** The pre-registered richer latitude functional form improves transferable
  prediction relative to M0\* and becomes the next station-geography development baseline.
  * The result is associational.
  * It does **not** separately establish hemispheric slope asymmetry, curvature, an onset near 50°N, or
    snow-albedo feedback.
* **`predictive support, not promoted`.** Reproducible out-of-fold improvement that falls below the
  practical threshold or fails a geographic veto. M0\* remains the baseline.
* **`not supported`.** The joint pre-registered latitude form does not demonstrate transferable
  improvement at n = 151 under the frozen criteria.
  * This is not evidence that the latitude relationship is linear.
  * **No rescue follows** (§2).
* **Under any label:**
  * M2 is a geography functional-form result. It is not evidence about hydroclimate, dryness or C2, and
    not a rescue of C2.
  * The responsibility statement is unchanged: shares are explained variance of cross-country
    differences in area-weighted warming, not greenhouse gases' physical role.
* **M4 stopping rule** (design operationalization of the inherited rule). The rule **fires if and only
  if** the smallest primary-protocol **point** ΔRMSE against the M0/M0\* comparator row, over M1a, M1b
  and M2, is **> −0.002 °C/decade**:
  * total-CO₂ representation, Berkeley area-weighted outcome;
  * whatever the verdict labels or intervals;
  * the per-capita, ERA5 and M1a-measurement arms never enter.

  M1a (+0.002343) and M1b (+0.001723) are already recorded, so the rule fires unless M2's point ΔRMSE is
  ≤ −0.002. It is reported as it falls. The incentive this creates is disclosed, not acted on.

## 12. Recorded interpretations and design choices (not separately owner-approved)

These do not block this contract. The owner may confirm or overrule any of them by a dated amendment
before the execution freeze (§9), never after an M2 number exists.

1. **The shortlist-entry rule.** M1b's rule ("a physical rationale specified independently of the
   Berkeley residuals, or product-robust residual evidence") is read as governing **new physical
   covariates**, not functional forms of existing M0\* features pre-registered at `af97bd2`. H1 was never
   re-based, and H5's residual evidence was withdrawn at `da4455d`.
2. **The product rule.** `PRODUCT_STABILITY_AUDIT.md` §7.2 ("product-sensitive and unresolved patterns …
   must not be the justification for an M1 predictor") is read likewise. M2's admission rests on the
   `af97bd2` roadmap, not on residual evidence (`M2_PROVENANCE_AUDIT.md` §4).
3. **Design choices:**
   * γ as the only shape diagnostic, and no curvature contrast;
   * the restored four-clause overfitting report, with clause (iii) evaluated for γ only;
   * the 1e-6° knot-gap tolerance, and the minimum of four training values;
   * IEEE product cubes and the ESL truncated-power parameterization;
   * random-reference fits and per-fit M0\* full rank as integrity conditions;
   * the 1e-8 per-capita coefficient identity for the latitude columns;
   * the extrapolation disclosure;
   * the conditional timing of the aligned-ERA5 and M1a-measurement replications, and aligned-only ERA5;
   * the state policy for later refits;
   * the operational M4 stopping-rule definition (§11).
