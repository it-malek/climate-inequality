# M1b evaluator specification — frozen before the first outcome-facing score

**Recorded 2026-09-15, committed and pushed before any M1b model is fitted to the warming outcome.**

This file fixes the **operational detail** the evaluator needs and that
[`M1B_EVALUATION_CONTRACT.md`](M1B_EVALUATION_CONTRACT.md) does not spell out. It changes no rule,
threshold, gate, protocol, verdict condition or interpretation. Where the contract speaks, the
contract governs; where it is silent on implementation, this file is the frozen answer, so that no
operational choice can be made after a number is seen.

Nothing here may be revised after any M1b score exists. A later change must be a new, dated,
separately justified record that leaves the original rule reported.

## 1. Execution boundary

The former `SCORING_ENABLED` flag is replaced by a provenance guard, so the implementation can only
execute from a frozen, published state. `m1b_evaluate.main()` refuses, before reading any gate,
input or outcome, unless all of these hold:

* `PYTHONHASHSEED=0` in the environment (contract §4.1);
* every file on the recorded code path is tracked, committed and identical to `HEAD`
  (no unstaged, staged or untracked change);
* `HEAD` is contained in the upstream remote-tracking branch, i.e. the evaluator commit is pushed;
* the output root does not already exist (a run never overwrites a prior run).

The guard is the enabling mechanism: scoring runs only through committed, pushed code.

Exit status: `0` success; non-zero on any refusal; `3` when the run completed and wrote its
evidence but a post-write integrity check (representation equivalence, share accounting, reference
reproduction or result-manifest digest) failed. Written evidence is never deleted or rewritten to
make a failed check disappear.

## 2. Scoring-input gate (before any target column is loaded)

1. **Accepted measurement package, by exact digest** (not merely mutually consistent documents):
   * `m1b_hydroclimate_features.csv` `cf33ba7c…8f1e4`, full value in code;
   * `m1b_measurement_manifest.json` `19746861…de4c9c`;
   * `m1b_redundancy_diagnostic.json` `9acb680c…b0f5585`.
2. `m1b_hydroclimate.verify_package` and `m1b_redundancy.verify_record` must pass on that package
   (passing gate, empty `hard_stops`, Amendment 3 gate version, clean committed build, artifact
   digests, and a redundancy record whose recorded manifest and feature digests are that package's).
3. **Required manifest and record fields** are validated by name and type: gate version, build
   commit and `code_path_matches_commit`, contract freeze commit, `artifact_sha256` covering exactly
   the package artifacts, `country_list_sha256`, and the redundancy record's `status`,
   `all_hard_stops_pass`, `hard_stops`, `design_rank`, `rank_with_c2` and provenance digests.
   A missing, misspelled, wrongly typed, stale or unapproved-but-self-consistent package fails closed.
4. **Frozen scoring inputs, by SHA-256 pin** (each recorded with the commit that froze it):
   the three predictor sources and the M1a support record already pinned by
   `m1b_redundancy.PREDICTOR_INPUT_SHA256`; `m0_countries.csv`, `country_geometry.csv`,
   `cv_fold_membership.csv` (`af97bd2`); `territory_distance_lower_km.csv`,
   `territory_cv_membership_comparison.csv`, `m0_cv_errors_territory_corrected.csv`,
   `m0_scorecard_territory_corrected.json` (`4263429`); `rank_audit.json`,
   `rank_audit_country_predictions.csv`, `product_stability_aligned_era5_trends.csv`,
   `product_stability_aligned_summary.json` (`9fb56e1`); `m1a_geography_features.csv`,
   `m1a_geography_qa.csv`, `m1a_measurement_manifest.json` (`a7dea34`); `m1a_scorecard.json`
   (`7003ff4`). The conditional M1a and aligned-ERA5 artifacts are pinned and verified at primary
   scoring time as well, so the conditional arms cannot later run on changed inputs.
5. **Identities checked on the actual scored frames** (not merely on the files):
   * C2 is parsed from the verified bytes; its columns, 151-row order and country list match the
     manifest and the frozen M1a support record; its float64 vector digest matches the redundancy
     record's `c2_vector_sha256`;
   * the scored design's country order equals the C2 order and `m0_countries.csv`;
   * the outcome-free predictor frame taken from the scored design reproduces the redundancy
     record's `predictor_frame_sha256`, `design_columns` and `design_matrix_sha256`;
   * the ordered Berkeley outcome vector equals the frozen `observed` column of
     `rank_audit_country_predictions.csv` (variant `drop_per_capita`) exactly, and its digest is
     recorded; the outcome source column is `trend_c_per_decade_area_weighted` (area-weighted V1);
   * M49 labels equal `regions.m49_table()` and the frozen `cv_fold_membership.csv` labels; the M49
     fold ids equal `m49_subregion_lo` and the random-reference ids equal `random_10fold`
     (`folds_random(151, 10, seed=0)`);
   * the territorial bound matrix matches the frozen `4263429` artifact, and the derived primary
     LOCO training memberships equal the frozen `corrected_train` memberships and `n_train` counts;
   * station-centroid and area-centroid geometry come from the pinned frozen tables.
   A complete-case population that differs from the frozen 151 in identity or order is a refusal,
   never a silently recreated sample.

Hashing a container's bytes for provenance is not reading its target column. The country table's
outcome and residual columns are never read; identifiers, M49 labels and station centroids are.

## 3. Model arms (contract §4.1, `V2_PREDICTOR_CONTRACT.md`, Amendment 3 A3.6)

| Representation | Comparator | Candidate | Role |
|---|---|---|---|
| `primary_total_co2` | frozen design minus `cum_co2_per_capita` | comparator + `baseline_dryness` | the only acceptance/promotion arm |
| `per_capita` | frozen design minus `cum_co2_total` | comparator + the identical `baseline_dryness` | reported unconditionally, never decides |

Same sample, outcome, encoders, folds, weights and C2 in both. Both comparators are scored and
checked **before either candidate is fitted**:

* `primary_total_co2` against `rank_audit.json` card `drop_per_capita`;
* `per_capita` against card `drop_total`;
* tolerance 1e-10, absolute, on every compared metric, each of which must be finite:
  `n`, `in_sample_r2`, `in_sample_rmse`, `cv_r2`, `cv_rmse`, `cv_mae`, `fold_rmse_median`,
  `fold_rmse_min`, `fold_rmse_max`, `fold_error_iqr`, `calibration_slope`, `calibration_intercept`,
  `residual_morans_i_in_sample`, `residual_morans_i_cv`, `share_geography`, `share_emissions`,
  `share_socioeconomic`, `share_population`, `residual_share`, `rows_with_unseen_level`,
  `worst_region.rmse`, `secondary.cv_r2`, `secondary.cv_rmse`, `random_reference_only.cv_rmse`.

On the frozen full sample the expected encoded matrices are 20 columns / rank 20 (comparator) and
21 / 21 (candidate), asserted for both representations. These full-sample counts are **not** imposed
on training folds, where a categorical level may legitimately be absent; train-only encoding and the
mean-training-level-effect rule are unchanged, and each training fit records its own column count,
rank and whether `baseline_dryness` is identified.

## 4. Protocols, scores and saved evidence

Protocols are unchanged: primary corrected 500 km footprint-certified LOCO with the `4263429`
bounds; secondary leave-one-M49-subregion-out; random 10-fold (seed 0) as reference only. Metric
definitions, the Moran weights (V1 station-centroid kNN8, 999 permutations, seed 0), the recorded
area-centroid Moran values, the worst-eligible-region rule (n ≥ 3) and the LMG/Shapley group shares
are the frozen definitions, reused from the existing code path.

Alongside the metrics, each representation reports the contract's §4.5 diagnostics, including the
three-dimension summary (transfer, spatial specification, stability) in the form
`M1A_EVALUATION_CONTRACT.md` §4 uses, and the result carries the §3 redundancy context and the §2.5
support-quality QA from the verified checkpoint. All of these are context, never verdict conditions.

**Paired-comparison field.** The frozen `paired_rmse_interval` is applied to the **primary**
protocol only, between a candidate and its own comparator. Secondary and random cards no longer
carry a `paired_delta_vs_m0` field computed against themselves; the M49 and random comparisons are
reported as plain RMSE/R² differences against the matching comparator protocol. This removes
meaningless zeros; primary paired inference is unchanged.

**Written per run** (deterministic unless stated):

| File | Contents |
|---|---|
| `m1b_scorecard.json` | both representations: comparator and candidate cards, paired primary comparison, verdict, diagnostics, equivalence block, reference-reproduction block, share accounting |
| `m1b_country_predictions.csv` | representation × model × protocol (`in_sample`, `primary_loco`, `m49_subregion_lo`, `random10`) × country: observed, prediction, error, absolute error, fold id, training size, nearest training distance, unseen-level count, region label |
| `m1b_primary_country_comparison.csv` | per representation and country: observed, both primary OOF predictions, both absolute errors, the change, unseen-level counts |
| `m1b_cv_folds.csv` | per country: primary fold, training size, the pipe-joined training ISO3 membership, M49 label and fold, random fold |
| `m1b_coefficients.csv` | every full-sample and primary-training-fit coefficient by representation, model, held-out country and column, with the fit's column count, rank and `baseline_dryness` identification flag |
| `m1b_coefficient_summary.json` | per column: full value, fold count, min/median/max, same-sign fraction, and for `baseline_dryness` the strictly-negative count and fraction |
| `m1b_shapley_coalitions.csv` | every group-coalition R² behind the LMG/Shapley shares |
| `m1b_design_matrices.csv` | the encoded full-sample design of each representation and model, by country and column, with group labels |
| `m1b_provenance.json` | commit, code-path digests, contract and spec digests, input digests and identities, fold identities, software versions |
| `m1b_result_manifest.json` | SHA-256 of every deterministic artifact above |
| `m1b_run_metadata.json` | **not deterministic**: timestamps, wall time, host, interpreter, argv, output root |

`m1b_coefficient_summary.json`'s `baseline_dryness` count is over the primary training fits, which
must number exactly the sample size (151 primary; 146 in the support arm) and be finite; the
**strictly-negative** proportion is used (a coefficient of exactly zero is not negative).

## 5. Verdict (contract §4.4, restated mechanically)

With `d = RMSE(candidate) − RMSE(comparator)` on the primary protocol, `[lo, hi]` its frozen paired
resampling interval, `b` the full-fit `baseline_dryness` coefficient and `f` the strictly-negative
fraction of its primary training-fit coefficients:

* **S1** `d < 0` and `hi < 0`; **S2** `b < 0` and `f ≥ 0.80` (compared exactly, as a rational);
* **A1** = S1; **A2** `d ≤ −0.002`; **A3** `M49 RMSE(candidate) − M49 RMSE(comparator) ≤ 0.001`;
  **A4** worst-eligible-region RMSE(candidate) − worst-eligible-region RMSE(comparator) ≤ 0.001
  (the frozen difference-of-worst-region definition, each model's own worst eligible region, not a
  per-region deterioration rule); **A5** = S2;
* verdict labels, exactly one: `supported and promoted` (A1–A5); `supported but sub-material / not
  promoted` (S1 and S2 hold, A2/A3/A4 not all); `not supported` (S1 or S2 fails);
* separately reported: `improvement_without_prestated_sign`, `worsened_generalization`.

Every underlying Boolean and every input value is emitted. Any non-finite verdict input is a
refusal, not a `False`. The verdict is a pure function of the primary representation's own cards;
no sensitivity, no per-capita result and no diagnostic can enter it.

Exactly one §4.4 verdict is reported, the primary representation's. The per-capita representation
emits the identical Booleans under `representation_conditions`, whose label is
`label_descriptive_only`, exactly as a conditional robustness arm does. A pre-scoring refusal (a
comparator that misses its frozen card, or a full-sample design of unexpected rank) aborts before the
output root is created, so it writes nothing and leaves the path free for a retry; the exit-3
behaviour applies to the post-write integrity checks.

## 6. Representation equivalence (prespecified tolerances)

`log10(total Mt) = log10(per-capita t) + log10(population) − 6` (asserted to 1e-12 on the frozen
design), so the two representations span the same column space and must agree, per corresponding
model and protocol:

* predictions (in-sample and all three out-of-fold protocols): max absolute difference ≤ 1e-9 °C/decade;
* non-allocation scores (`in_sample_r2`, `in_sample_rmse`, `cv_r2`, `cv_rmse`, `cv_mae`, Moran
  statistics, worst-region RMSE, M49 and random RMSE, `delta_rmse` and both interval bounds): ≤ 1e-9;
* `baseline_dryness` coefficients, full fit and every training fit where identified in both: ≤ 1e-8.

Allocation quantities are expected to differ: LMG/Shapley group shares, including the hydroclimate
share, and the population/intercept coefficients, whose interpretation changes. Differences are
reported, never required to match, and a representation is **never** chosen by its shares.

Share accounting, checked per model to 1e-9: `Σ named group shares = in-sample R²` and
`Σ named group shares + residual share = 1`. All shares are fractions of explained outcome variance,
not causal contributions.

## 7. Conditional robustness runners (contract §5, A2.1; run only after the primary freeze)

`m1b_sensitivity.py` **refuses to execute** — it does not merely defer writing — unless:

* the primary result directory verifies against its own `m1b_result_manifest.json`;
* those artifacts are tracked, committed, identical to `HEAD`, and contained in the pushed upstream
  branch;
* the evaluator code path is unchanged between the primary run's recorded commit and `HEAD`;
* the scoring-input gate of §2 passes again;
* the committed primary verdict records **association support** (S1 and S2), independent of
  promotion. If association support fails, the runner writes a contract-based non-execution record
  and runs nothing; no favourable sensitivity is searched for.

Three authorized arms, each refitting **both** the comparator and the candidate under that arm's own
inputs and sample, retaining the exact frozen country C2 values (never rebuilt, never renormalized),
and each reporting both representations as A3.6 requires:

1. `m1a_area_geography`: the frozen M1a measurements replace `abs_latitude`, `elevation`,
   `continentality`, `climate_zone`, `hemisphere`; `spatial_block` is copied unchanged; everything
   else is held fixed. Its comparators must reproduce `m1a_scorecard.json` cards
   `primary.M1a` and `percapita_sensitivity.M1a` to 1e-10.
2. `aligned_era5_outcome`: the committed preprocessing-aligned ERA5 outcome
   (`product_stability_aligned_era5_trends.csv`, `trend_c_per_decade_era5_area`, digest matching the
   aligned summary's `aligned_outcome_sha256`) replaces the Berkeley outcome for both models; no
   reconstruction, no model selection, no new missing-country rule; all 151 countries have a finite
   value. Its comparators' in-sample R² must reproduce the aligned summary's `r2.era5` to 1e-10.
Each arm records the identity digests of **its own** frame — outcome vector, C2 vector, predictor
frame, ISO3 list, fold memberships and both weight matrices — so no arm inherits the primary run's
digests, and an independent verifier can pin exactly what that arm fitted.

3. `station_support_146`: exactly Saudi Arabia, Yemen, Haiti, Oman and Chad (SAU, YEM, HTI, OMN,
   TCD — the five lowest PRE station-supported shares in the verified support checkpoint) are
   removed by ISO3. Country rows, the bound matrix, the geometry and the M49 labels are subset; the
   same fold-construction algorithm is re-applied to the 146 rows (LOCO, M49 leave-one-out,
   `folds_random(146, 10, seed=0)`); the paired resampling procedure runs at n = 146 with 2,000
   resamples and seed 0. Its candidate is compared only with its own 146-country comparator.

No arm is combined with another. Each arm emits the same condition Booleans as §5 under the label
`arm_conditions`, explicitly **descriptive**: they cannot change either verdict level. A gain
confined to Berkeley is flagged `product_sensitive` when the primary association holds but the
aligned-ERA5 arm's S1 does not.

## 8. Determinism and reproducibility

`PYTHONHASHSEED=0`; the paired interval, fold constructions, permutations and `lstsq` solves are
deterministic. An unchanged-code rerun in a separate process and a separate output root must
reproduce every deterministic artifact, and the result manifest, byte for byte; only
`m1b_run_metadata.json` may differ, and it holds the upstream head as well as the timestamps, since
neither is a property of the scored inputs. The comparison is produced by the evaluator itself
(`--compare RUN_A RUN_B --record PATH`) and records both runs' digests per artifact.

## 9. What this specification does not do

It does not change C2, the measurement package, the redundancy record, the folds, the outcome, the
protocols, the metric definitions, the uncertainty method, the thresholds, the verdict, the
registered sensitivities or the interpretation limits. It adds no predictor, no transform, no
nonlinear or interaction term, no alternative dryness definition and no new uncertainty method. It
introduces no outcome-dependent tolerance, gate or interpretation.
