# M2 evaluator specification — frozen before the first outcome-facing M2 fit

**Recorded 2026-09-16, committed and pushed before any M2 candidate is fitted to the warming outcome.**
This file fixes the operational detail that [`M2_EVALUATION_CONTRACT.md`](M2_EVALUATION_CONTRACT.md)
(with Amendment 1, [`M2_PRE_SCORE_RESOLUTIONS.md`](M2_PRE_SCORE_RESOLUTIONS.md)) and
[`M2_DESIGN.md`](M2_DESIGN.md) leave to implementation. It changes no rule, threshold, protocol, verdict
condition or interpretation; where they speak, they govern. Implementation: `m2_evaluate.py` (primary),
`m2_conditional.py` (conditional arms or their non-execution), `m2_independent_check.py` (post-result
reconstruction). Nothing here may be revised after any M2 number exists.

## 1. Execution boundary

`m2_evaluate.main()` refuses, before reading any input or outcome, unless all hold:
* `PYTHONHASHSEED=0`;
* every file in the **code closure** is tracked, committed and identical to `HEAD`. The closure is the
  transitive set of local modules (`research.*`, `src.*`) imported by `m2_evaluate.py` and
  `m2_conditional.py`, computed at run time from the AST, plus: this file, the contract, the design, the
  provenance and feasibility audits, `M2_PRE_SCORE_RESOLUTIONS.md`, `DOWNSTREAM_COMPLETION_SPEC.md`,
  `pyproject.toml` and `uv.lock`. Its SHA-256 digests are recorded;
* `HEAD` is contained in the **live** remote head of its upstream branch: `git ls-remote` gives the remote
  head, which must equal `HEAD` or have `HEAD` as an ancestor (the object must exist locally);
* the output root does not exist.

A refusal before any candidate fit writes nothing and may be retried from identical frozen bytes. Exit
status: 0 success; 3 when the run wrote its evidence but a post-write integrity check failed (evidence is
never deleted); non-zero otherwise. `--compare RUN_A RUN_B --record PATH` byte-compares two runs (exit 0
when identical, 4 otherwise) and never scores.

## 2. Scoring-input gate (before any outcome is read)

**Pins** (SHA-256, each with the commit that froze it): the M1b `FROZEN_SCORING_INPUTS` (unchanged,
`m1b_evaluate.py`), plus
* `outputs/m1b_primary/m1b_design_matrices.csv`, `m1b_cv_folds.csv`, `m1b_result_manifest.json`,
  `m1b_provenance.json`, `m1b_country_predictions.csv`, `m1b_scorecard.json` (`1680d85`);
* `outputs/m2_feasibility/m2_feasibility_summary.json`, `m2_feasibility_fits.csv` and
  `m2_feasibility_manifest.json` (`23d31f4`).

The M1b artifacts must also equal their entries in the committed `m1b_result_manifest.json`, and the
feasibility summary and fits their entries in the feasibility manifest.

**Identities checked on the frames actually scored:**
1. **Country order.** `m0_countries.csv` (`Country`, `iso3`, `m49_subregion`, station coordinates and the
   four categorical labels only) gives the canonical 151; `regions.m49_table()` equals its M49 labels.
2. **Numeric predictors.** For each representation, the `M0star` rows of `m1b_design_matrices.csv`
   (`float_precision="round_trip"`) pivoted by `iso3` × `column`, aligned to canonical order. Columns:
   primary `cum_co2_total, abs_latitude, elevation, continentality, population, station_density`;
   per-capita the same with `cum_co2_per_capita`. They are already transformed and never logged again.
3. **Encoded design.** The full-sample M0\* encoding (§3) equals the frozen `M0star` design value for value
   after aligning columns by name; every dummy column rebuilt from the labels is bit-identical; column and
   group sets agree.
4. **Representation identity.** log10 total = log10 per-capita + log10 population − 6 on the frozen values
   to 1e-12.
5. **Outcome.** y is the `observed` column of `rank_audit_country_predictions.csv` (variant
   `drop_per_capita`, round-trip parse) in canonical order; it must equal, exactly, the outcome of
   `m0.m0_complete_design(*m0.load_inputs())` and the `observed` column of the committed M1b prediction
   record for `M0star` in-sample rows. Its float64 digest is recorded.
6. **Folds.** Primary LOCO training sets derived from `territory.load_bounds` with the 500 km buffer equal
   the `corrected_train` memberships of `territory_cv_membership_comparison.csv`, the
   `primary_training_iso3` of `m1b_cv_folds.csv` and the `n_train` of `m0_cv_errors_territory_corrected.csv`;
   M49 fold ids equal `cv.folds_from_labels(m49_subregion)`, `cv_fold_membership.csv`
   `m49_subregion_lo` and `m1b_cv_folds.csv` `m49_fold`; random ids equal `cv.folds_random(151, 10, 0)`,
   `random_10fold` and `random10_fold`.
7. **Weights.** Station and area-centroid kNN8 matrices equal the M1b identity digests.
8. **Latitude state.** `lb.fit_state(full-sample abs_latitude)` equals `LatitudeState.from_json` of the
   feasibility summary's `full_sample/station/<representation>/latitude_state` exactly (hex knots).

Only identifiers, M49 labels, coordinates and categorical labels are read from the country table.

## 3. Models, encoder and fits

* **Encoder** = `m2_feasibility.baseline_encoder` / `encode` (V1 feature order: numerics then drop-first
  dummies on the training levels), which is `cv.V1Design._matrix` on transformed numerics.
* **M0\*:** X = encode(rows, encoder(training rows)). **M2:** X = [M0\* X, `lb.added_block(state, a, h)`]
  with `state = lb.fit_state(training abs_latitude)`; columns `abs_latitude_ns1`, `abs_latitude_ns2`,
  `hemisphere=S:abs_latitude`, group `geography`.
* **Estimator:** `numpy.linalg.lstsq(X, y, rcond=None)`.
* **Prediction** for held-out rows: X built with the training encoder and (M2) the training state;
  prediction = X β̂ plus, for each categorical feature whose held-out level is absent from the training
  levels, the mean of the list [0, β̂ of each non-reference training level] (`cv.V1Design.predict`).
* **CV** through `cv.cross_validate` with the frozen fold ids and territorial bounds (buffer 500 km for the
  primary protocol, none for M49 and random).
* **Order before any M2 fit:** (1) every fit's latitude state (hex), M0\* and M2 column counts and ranks, and
  held-out extrapolation counts, for both representations (364 fits), must equal the committed predictor-only
  feasibility record; (2) both M0\* representations are scored and checked against `rank_audit.json` cards
  `drop_per_capita` / `drop_total` on the M1b `REFERENCE_METRICS` to 1e-10. A mismatch is a pre-candidate
  refusal (nothing written).
* **Structural integrity per required fit** (full, 151 primary, 20 M49, 10 random; both representations):
  M0\* rank = columns; M2 state non-degenerate; M2 rank = columns; M2 rank − M0\* rank = 3; full sample
  20/20 and 23/23. Rank is `numpy.linalg.matrix_rank` at its default tolerance. A failure after any M2 fit
  is written with no label and stops the line.

## 4. Scores and group shares

Per model and representation, the M1b card fields (`m1b_evaluate.protocol_card` and `score_model`
definitions): in-sample R² and RMSE; for primary, M49 and random: CV R², RMSE, MAE, fold RMSE
median/min/max, worst fold and its units, IQR of |OOF error|, calibration slope/intercept, residual Moran's
I in-sample and OOF (V1 station weights, 999 permutations, seed 0) with area-centroid values recorded,
per-M49-region RMSE/MAE/bias and each model's own worst eligible region (n ≥ 3), unseen-level rows by
categorical, mean n_train and nearest training distance; matrix columns and rank; effective degrees of
freedom (rank).

**LMG shares** are computed on the encoded blocks: groups `emissions`, `geography`, `socioeconomic`,
`population` (the added M2 columns in `geography`); 16 coalitions; coalition R² = 1 − RSS/TSS from lstsq
with an intercept; Shapley weights as `src.decomposition`. All coalitions use the full-sample latitude
state. Share accounting to 1e-9. For M0\* these must reproduce the rank-audit card shares to 1e-10.

## 5. Paired comparison, verdict and diagnostics

* **Paired:** `run_territory_correction.paired_rmse_interval(y, M2 primary OOF, M0* primary OOF)`, primary
  only; M49 and random differences reported as plain differences.
* **Verdict** (primary representation only), exactly `M2_PRE_SCORE_RESOLUTIONS.md` §7: P1 d < 0; P2 hi < 0;
  R1 d ≤ −0.002; R2 M49 ΔRMSE ≤ +0.001; R3 worst(M2) − worst(M0\*) ≤ +0.001; R4 structural integrity.
  Labels `supported and promoted` / `predictive support, not promoted` / `not supported`; none on an
  integrity failure; `worsened_generalization` = d > 0 and lo > 0. Every input must be finite (else an
  integrity refusal). The per-capita representation emits the same Booleans under
  `representation_conditions` with a descriptive label.
* **Static stopping indicator:** `fires = d > -0.002`, with M1a (+0.002343) and M1b (+0.001723) recorded
  from their committed scorecards.
* **γ summary:** full-fit γ; over the 151 primary training fits: strictly-negative count and fraction, zero
  count, median, minimum, maximum; no cutoff.
* **Overfitting diagnostics:** (i) in-sample R² gain − CV R² gain > 0.05; (ii) CV RMSE worsens while
  in-sample RMSE improves; (iii) γ same-sign fraction reported without a cutoff; (iv) effective df rise > 5
  with d > −0.002 (df rise is 3, so false). Summary Boolean from (i), (ii), (iv).
* **Residual spatial structure** (±0.05 on both in-sample and OOF I), **scientific stability** (geography
  largest named share; emissions ≤ 0.10), **three-dimension summary** as M1b.
* **Extrapolation disclosure:** for every fit, held-out rows below ξ₁ or above ξ₄, with their OOF errors
  under both models; must equal the feasibility record's counts (Gabon, Iceland; M49 Middle Africa 1,
  Northern Europe 8; random folds 5 and 6), otherwise an integrity failure.
* **Countries improved/worsened/tied** under the primary protocol.

## 6. Representation equivalence

Per model and protocol: predictions ≤ 1e-9; the M1b `EQUIVALENT_METRICS`, d and both interval bounds
≤ 1e-9; γ, `abs_latitude_ns1`, `abs_latitude_ns2` and `abs_latitude` coefficients ≤ 1e-8 in the full fit and
in every primary training fit. Shares and population/intercept coefficients may differ and are reported.

## 7. Evidence written (deterministic unless stated)

| File | Contents |
|---|---|
| `m2_scorecard.json` | both representations: M0\* and M2 cards, reference reproduction, paired comparison, verdict / representation conditions, diagnostics, γ summary, extrapolation disclosure, stopping indicator, equivalence, integrity |
| `m2_country_predictions.csv` | representation × model × protocol (`in_sample`, `primary_loco`, `m49_subregion_lo`, `random10`) × country: observed, prediction, error, fold id, n_train, nearest training km, unseen levels, M49 label, extrapolation flag |
| `m2_primary_country_comparison.csv` | per representation and country: both primary OOF predictions and absolute errors, change |
| `m2_cv_folds.csv` | as `m1b_cv_folds.csv` |
| `m2_fit_states.csv` | representation × model × protocol × fit: training ISO3 count, categorical levels per feature, column names, matrix columns and rank; for M2 the serialized latitude state (hex), extrapolation counts, SH count |
| `m2_coefficients.csv` | every coefficient of the full fit and of every primary, M49 and random training fit |
| `m2_coefficient_summary.json` | per column over primary training fits: full value, n, min/median/max (γ with its strictly-negative count); basis coefficients labelled "not interpreted" |
| `m2_shapley_coalitions.csv` | every coalition R² |
| `m2_design_matrices.csv` | full-sample encoded designs by representation, model, country and column, with groups |
| `m2_provenance.json` | code closure digests and commit, input pins and identities, software and platform |
| `m2_result_manifest.json` | SHA-256 of every deterministic artifact; evaluator commit; label; integrity |
| `m2_run_metadata.json` | not deterministic: times, host, interpreter, argv, output root, live remote head |

## 8. Conditional arms (`m2_conditional.py`)

Refuses unless: the primary result verifies against its manifest; those files are tracked, identical to
`HEAD` and contained in the live remote head; the code closure is unchanged since the recorded evaluator
commit; the scoring-input gate passes again; `PYTHONHASHSEED=0`; fresh output root.

* **No predictive support:** write `m2_conditional_not_run.json` (the primary digests, the verdict and the
  rule) and return **before** any arm frame is constructed or any fit is made.
* **Predictive support (promoted or not):** run exactly two arms, each refitting M0\* and M2 in both
  representations under all protocols:
  1. `m1a_area_geography`: `abs_latitude`, `elevation`, `continentality`, `climate_zone`, `hemisphere`
     from the pinned `m1a_geography_features.csv` (round-trip), `spatial_block` unchanged; states recomputed
     per fit from M1a `abs_latitude`, interaction from M1a hemisphere labels. M0\* must reproduce
     `m1a_scorecard.json` `primary.M1a` / `percapita_sensitivity.M1a` to 1e-10 on the metrics that card
     records.
  2. `aligned_era5_outcome`: y replaced by `trend_c_per_decade_era5_area` of the pinned aligned file
     (digest equal to the aligned summary's `aligned_outcome_sha256`; all 151 finite); every fit's
     latitude state identical to the primary run's (checked by hex). M0\* in-sample R² must reproduce
     `r2.era5` to 1e-10.
* Each arm writes its own cards, predictions, states, identities, equivalence and descriptive
  `arm_conditions` (P1, P2, R1–R4 Booleans); `product_sensitive` when primary P1∧P2 holds and the aligned
  arm's does not. A structural failure in an arm (degenerate latitude state, a rank failure in a required fit)
  is recorded `non_computable` with no repair. A reference-reproduction, share-accounting, state-identity or
  representation-equivalence failure is a software or input defect, not that disposition: it stops the run
  and writes nothing. No arm changes the verdict.

## 8a. Disclosure: calculations before this freeze

* During implementation the evaluator tests ran, from uncommitted code: identity checks that bind the real
  outcome without fitting (`numpy.linalg.lstsq` raises in that fixture); the predictor-only comparison of all
  364 fits with the feasibility record; and a **baseline-only anchor** that scores both M0\* representations
  on real inputs against their frozen rank-audit cards, in which any M2 fit raises. The committed M1b test
  suite already scores the same M0\* comparators on real inputs whenever it runs.
* **No M2 candidate was fitted to any real outcome** before this freeze. Git cannot prove the absence of
  unrecorded executions; this statement rests on the session record.

## 9. Independent reconstruction (`m2_independent_check.py`, after the result)

A separate calculation path that imports nothing from `m2_evaluate.py` or `m2_feasibility.py`: rebuilds
the natural cubic columns from the saved hex knots by its own formula; rebuilds each design from the saved
states and labels; refits every full and training fit by QR; recomputes every prediction (including the
unseen-level rule) and compares with `m2_country_predictions.csv`; recomputes RMSE, R², MAE, the paired
interval, worst regions, verdict Booleans, γ summary, stopping indicator, coalition R² and Shapley shares by
permutation averaging, and the representation identity. Negative controls (perturbed prediction, shifted
threshold, altered knot, swapped fold membership, altered coefficient) must each be detected and are saved.
Output: `outputs/m2_primary_verification/m2_independent_check.json`.

## 10. Reproducibility

One unchanged-code rerun in a separate process and fresh root; `--compare` must report every deterministic
artifact and the manifest byte-identical. The comparison record and the rerun's manifest and run metadata
are committed under `outputs/m2_primary_verification/`.
