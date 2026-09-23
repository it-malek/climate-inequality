# M2 pre-score resolutions and completion authorization (owner, 2026-09-16)

**Recorded 2026-09-16, before any M2 candidate fit, score or cross-validated prediction, and before any
M3 or M4 computation on the warming outcome.** This file records the owner's completion instruction of
2026-09-16 as dated decisions. It is Amendment 1 to [`M2_EVALUATION_CONTRACT.md`](M2_EVALUATION_CONTRACT.md)
(which permits dated amendments before its execution freeze) and the authority for
[`DOWNSTREAM_COMPLETION_SPEC.md`](DOWNSTREAM_COMPLETION_SPEC.md). Historical files keep their original
text; where a sentence is corrected, the correction is dated and the original is quoted.

**Session disclosure (this checkpoint).** Before this record was committed, the session read committed
documents, code and committed JSON summaries (for example `rank_audit.json` card keys and the M0.5
product summaries' `r2` values). It fitted no model to any outcome, computed no M2 or M3 quantity, and ran
no evaluator. Synthetic-outcome tests of the M3 estimator library ran on simulated data only.

## 1. Authority and precedence

1. This owner instruction resolves the earlier prompts and amendments.
2. Owner-approved contracts (`V2_PREDICTOR_CONTRACT.md`, the corrected protocol, the M1a, M1b and M2
   contracts) stay binding where not resolved here.
3. Older roadmap text (`af97bd2`) supplies historical scope and provenance.

**Superseded session-only restrictions.** The instruction authorizes M2 execution, its contract-authorized
conditional robustness, the required M3 accounting, the M4 final assessment, commits and normal pushes at
checkpoints, and integration into `main` after scientific completion and verification. It therefore
supersedes, for this completion only: "This contract authorizes no M2 execution" (`M2_EVALUATION_CONTRACT.md`
§9, the authorization part only; the freeze requirements stand); "no M3" (`M1B_CLOSURE.md` §1, which recorded
that no follow-up was automatic); "touch `v1.3.0` or `main`" (`EXPERIMENTAL_PLAN.md`, "What every stage must
not do", the `main` part only); and "never merge to `main` in this line" (`OPEN_DECISIONS.md` item 17).
**Preserved:** the `v1.3.0` tag and its scientific outputs, the public V1 bundle, all historical outputs and
chronology; no amend, squash, rebase, force-push or retag.

## 2. M2 admission and the latitude measurements

* **Admission.** M2 rests on the `af97bd2` functional-form registration and the frozen M0\* entry-gate value
  (out-of-fold Moran's I 0.3215471676927579 > 0.2), not on treating unresolved residual evidence as
  independently confirmed. The shortlist-entry rule and `PRODUCT_STABILITY_AUDIT.md` §7.2 govern **new
  physical measurements**; they do not exclude the historically registered transformations of existing
  latitude and hemisphere. This confirms `M2_EVALUATION_CONTRACT.md` §12 items 1–2.
* **Three measurements, not interchangeable:**

  | Role | Latitude | Hemisphere |
  |---|---|---|
  | **Primary M2** | M0\* station-based `abs_latitude` (exact frozen scoring values) | M0\* station hemisphere |
  | Historical H1/H5 generation (`af97bd2` diagnostics) | absolute land-area-centroid latitude (`abs_centroid_lat`) | station hemisphere |
  | M1a measurement sensitivity | mean absolute land latitude (`M1A_MEASUREMENT_SPEC.md`) | majority-area hemisphere |

  The station measurements are primary for operational continuity with the retained baseline. The M1a arm
  assesses measurement sensitivity; it does not redefine the primary hypothesis. Primary M2 is **not** a
  literal replication of the centroid-latitude residual finding.

## 3. The sole candidate (confirmed)

M2 = M0\* + θ₁N₁(a) + θ₂N₂(a) + γ I(SH) a, with a the exact frozen M0\* `abs_latitude`, the frozen
`m2_latitude_basis.py` (`fbaa3ff`) state rules and M0\*'s linear latitude, hemisphere main effect and all
other columns kept: exactly three added coefficients, all in `geography`; full-sample ranks 20/23.
Prospective extrapolation (Gabon, Iceland, the Northern-Europe M49 holdout, random folds 5 and 6) is not a
reason to redesign. **Never scored:** H1-only, spline-only, alternative latitude forms, NH-only curvature,
alternative knots or df, nonlinear C2, C1, C3 or any rescue variant.

## 4. H5 wording correction and the shape interpretation

**Correction (dated 2026-09-16, pre-score).** `M2_DESIGN.md` §6 says the pooled tertile spline "cannot
represent or test H5's "steepening above ~50°N"". That is too strong. Corrected statement:

> The spline can exhibit positive curvature and increasing slope above 50°. It cannot identify a distinct
> NH-specific onset near 50°: curvature is common across hemispheres, boundary behaviour is constrained,
> and γ supplies only a hemisphere-specific linear slope difference.

(For ξ₃ ≤ u < v ≤ ξ₄ the slope increases by f″(ξ₃)(v − u)(ξ₄ − (u + v)/2)/(ξ₄ − ξ₃), which is positive when
f″(ξ₃) > 0, so increasing slope above 50° is representable; what is not representable is curvature that
begins near 50° in the northern hemisphere only.) The same reading applies to "cannot test an onset of
curvature near 50°N" (`M2_DESIGN.md` §1) and "cannot represent an onset near 50°N" (contract §6.2,
`M2_PROVENANCE_AUDIT.md` §4 item 4): they are correct about a distinct onset, not about increasing slope.

* M2 is primarily a **predictive functional-form test**, not a direct snow/ice-feedback test.
* No separate curvature mechanism contrast is introduced.

## 5. γ policy

Always reported (primary and per-capita): the full-fit γ; the **number and fraction of the 151 primary
training-fit γ values strictly below zero**; their median, minimum and maximum.

* **No sign-stability cutoff.** No pre-M2 source establishes an 80% threshold for H1: the `af97bd2` H1
  entry states only a direction ("NH slope positive …, SH slope negative or zero"); 80% appears only in the
  plan's generic covariate-retention rule and generic overfitting clause and in M1b's C2 rule, which are
  insufficient. The contract §6.2 overfitting clause (iii) is therefore reported as the γ same-sign
  fraction **without a thresholded Boolean**; the overfitting summary Boolean uses clauses (i), (ii) and
  (iv) only. Sign stability never affects any M2 label.
* **Interpretation.** Negative γ means a lower SH latitude slope relative to NH. It does not establish a
  positive NH or non-positive SH absolute slope, and it identifies no mechanism. Raw spline-basis
  coefficients, the linear `abs_latitude` coefficient and the `hemisphere=S` coefficient are never
  interpreted, and their cross-fold summaries are never presented as physical stability evidence.

## 6. Exact inputs and shared protocol (restated as binding operational detail)

* 151 countries in canonical order; Berkeley area-weighted outcome 1950-01…2013-09, **bound to the frozen
  ordered reference** (`rank_audit_country_predictions.csv`, variant `drop_per_capita`, `observed`) by exact
  equality, not by an approximately matching summary.
* Corrected 500 km footprint-certified LOCO (`4263429` territorial lower bounds); fold memberships verified
  against **both** the saved memberships (`territory_cv_membership_comparison.csv` and the M1b
  `m1b_cv_folds.csv`) and the corrected construction; 20 M49 holdouts; random 10-fold seed 0.
* Train-only categorical encoding; mean-training-level-effect handling of unseen levels; V1 estimator
  (`numpy.linalg.lstsq`, `rcond=None`); V1 group definitions.
* **Numeric predictors** come from the pinned `M0star` rows of
  `outputs/m1b_primary/m1b_design_matrices.csv`, read with `float_precision="round_trip"`, aligned by
  country and column name, with group identities verified and the file bound to its committed
  `m1b_result_manifest.json`. They are already transformed and are not logged again. The country table's
  rounded numeric copies are not scoring inputs; its pinned categorical labels rebuild the dummy columns,
  which must equal the frozen dummy columns.
* Primary total-CO₂ representation decides; the per-capita representation is reported unconditionally.

## 7. Verdict (restated; unchanged from contract §5)

d = primary RMSE(M2) − primary RMSE(M0\*); [lo, hi] = `paired_rmse_interval` (2,000 seed-0 country
resamples of the fixed paired OOF errors, identical indices, percentile 2.5/97.5; not a spatially corrected
interval).

* **Predictive support:** d < 0 and hi < 0 (both strict).
* **Promotion:** predictive support and d ≤ −0.002; M49 ΔRMSE ≤ +0.001; worst(M2) − worst(M0\*) ≤ +0.001
  (each model's own maximum primary regional RMSE over M49 regions with n ≥ 3); structural integrity in
  every contract-required fit (all inclusive comparisons).
* **Labels:** `supported and promoted`; `predictive support, not promoted`; `not supported`. No label on an
  integrity failure. `worsened_generalization` is true exactly when d > 0 and lo > 0.
* Only the primary representation determines the label. γ, overfitting diagnostics, random-reference
  performance, product or measurement arms and raw basis coefficients add no condition.
* **Tolerances:** frozen baseline reference metrics 1e-10; representation predictions and non-allocation
  scores 1e-9; latitude-coefficient identity (γ, ns1, ns2, linear `abs_latitude`) 1e-8; share accounting
  1e-9; exact encoded-design and authoritative full-sample hex knot-state identities.

## 8. Decomposition and conditional robustness

* Four named groups, 16 intercept-containing coalitions; all latitude additions stay in `geography`; each
  fit's coalitions share that fit's latitude state; named shares sum to R², named plus residual to one.
* **Predictive support (promoted or not) authorizes exactly:** the M1a-geography replication, the
  aligned-ERA5 replication, each in both representations. They run only **after the verified primary result
  is committed and pushed**. ERA5 reuses the identical predictor and fold latitude states; M1a recomputes the
  states from its frozen training measurements and uses its own hemisphere labels. Neither arm changes the
  primary verdict. A contract-defined structural failure in an arm is recorded as non-computable without
  fallback.
* **Without support:** verified non-execution is recorded before any arm is constructed or fitted. No
  C2-specific five-country sensitivity applies. Aligned-only ERA5 for M2 remains a disclosed prospective
  departure from the older "both constructions" product wording.

## 9. Completion state machine

* Retained static specification: M2 if promoted, else M0\*. Static stopping indicator: fires iff d > −0.002.
* Every integrity-valid path: required M3 accounting → M4 final assessment → scientific closure → branch
  verification → integration → integrated-tree verification → normal `main` push.
* M3 evaluates both historical families (SEM, SAR) under station-centroid kNN8 (primary) and land-centroid
  kNN8 (sensitivity). The optional geostatistical family is **not pursued**.
* **Historical unique-family selection rule:** none exists. Final naming follows the owner's disposition
  table as frozen in `DOWNSTREAM_COMPLETION_SPEC.md` §2.14, with three separate machine-readable fields.
  A non-unique result is a valid conclusion.

## 10. Interpretation limits (binding for every report)

Descriptive and associational throughout: geography is not causal attribution; historical-responsibility
allocation is not greenhouse-gas forcing's physical contribution; residual variance is not "unknown climate
cause"; spatial covariance accounting is not mechanism identification; rejected operational models do not
disprove their motivating mechanisms; same-dataset hypothesis generation and follow-up are not independent
confirmation; product disagreement is not an identified decomposition of observational error.

## 11. Evidence limits carried forward (unchanged)

Before M1b's evaluator freeze, baseline-only real-outcome calculations (including aligned-ERA5 comparator
calculations) ran from uncommitted code; no C2-containing pre-freeze fit was evidenced (`M1B_CLOSURE.md`
§3 item 7). This does not invalidate M1b, and Git cannot prove that no unrecorded execution ever occurred.
The closure's limits on deleted scratch runs, reproducibility evidence and local push timing stand.
