# Decisions register

## Resolved on 2026-09-15 (after the M0.5 review, commit `9fb56e1`)

1. **V2 representation pair: approved.** The primary uses total cumulative CO₂
   plus population, dropping per-capita. The registered sensitivity uses
   per-capita plus population, dropping total. See `V2_PREDICTOR_CONTRACT.md`.
   V1's decomposition remains V1.
2. **M1a: approved** under `M1A_MEASUREMENT_SPEC.md`, as a measurement-alignment
   experiment only. Its evaluation rule was frozen in `M1A_EVALUATION_CONTRACT.md`
   before any M1a measurement or score. That file also records the product-audit
   guardrail for later predictors.

## Resolved after M1a (owner, 2026-09-15)

3. **M1a not promoted.** Primary V2 development stays on M0\* (full-rank station
   geography). M1a is the registered area-consistent geography measurement sensitivity.
4. **Push.** The M1a history `e3e5901`…`7003ff4` was pushed without rewriting.

## M1b (baseline hydroclimatic dryness): status and open decisions

M1b is narrowed to C2 baseline hydroclimatic dryness (owner, 2026-09-15). The decisions
are recorded in `M1B_DESIGN.md`:
* land-area support;
* 1920-01 to 1949-12 window;
* CRU TS v4.10;
* baseline hydroclimate group;
* no Bonferroni and no joint model;
* sensitivities only for a surviving model (Amendment 3 A3.6: the per-capita representation is
  reported with any M1b score unconditionally).

The measurement and evaluation rules are frozen in `M1B_EVALUATION_CONTRACT.md`.

**Status 2026-09-15.**
* CRU TS v4.10 PRE/PET were acquired and hash-pinned.
* The first C2 build failed the 98% coverage gate in Bahamas, Panama and the Philippines,
  because of CRU's fixed land mask (`M1B_FEASIBILITY_AUDIT.md`).
* The owner approved a global pre-outcome harmonization rule, recorded as contract
  Amendment 1: one ring, native donors only, threshold unchanged.
* Next: an amended build must pass every hard stop. The redundancy diagnostic then runs on
  predictor columns only; contract Amendment 2 restricts its inputs to allow-listed predictor
  source columns.
* The amended build stopped (`M1B_AMENDMENT1_BUILD_STOP.md`).
  * All 151 countries reach ≥ 0.98 coverage (minimum Bahamas 0.9888).
  * Hard stop A1.7 fired on one non-structural Peru cell: 23 km², PET masked, PRE identically zero
    across the record.
* **Resolved by the owner (contract Amendment 3, 2026-09-15).**
  * This is a pre-outcome relaxation: non-structural invalid support stays unresolved and counts
    against the unchanged 0.98 coverage threshold, and its presence no longer stops the build.
  * The amendment also adds package-integrity gates, and it records that the per-capita
    representation is reported unconditionally when M1b is scored.
* **Pre-outcome package complete.**
  * C2 was frozen at `222e2c1` (build `58e64bf`; all gates pass; minimum coverage Bahamas 0.9888).
  * The redundancy diagnostic passed (`M1B_REDUNDANCY_DIAGNOSTIC.md`: R² 0.650, VIF 2.86, no hard
    stop, no near-redundancy flag).
  * Passing coverage and redundancy establish neither predictive value nor mechanism.
* **Evaluator frozen before scoring (owner-authorized, 2026-09-15).**
  * `M1B_EVALUATOR_SPEC.md` freezes the operational detail the contract leaves open. It changes no
    rule, threshold, protocol or verdict.
  * The A3.6 per-capita representation is implemented and reported unconditionally; only the primary
    total-CO₂ pair determines acceptance.
  * The conditional M1a, aligned-ERA5 and five-country runners are implemented and tested, and
    refuse to execute without a committed, digest-verified primary result recording association
    support.
  * `SCORING_ENABLED` is replaced by a provenance guard: scoring runs only from a committed,
    unmodified, pushed code path, with `PYTHONHASHSEED=0` and a fresh output root.
* **M1b scored 2026-09-15 from evaluator commit `618c8b8`: `not supported`** (`M1B_REPORT.md`).
  * Delta RMSE +0.00172 degC/decade with the paired interval [+0.00089, +0.00285] entirely above
    zero, so generalization worsened; the `baseline_dryness` coefficient is positive (+0.000474) and
    negative in only 68 of 151 training fits.
  * The per-capita representation agrees to 4.4e-14 and decides nothing (A3.6).
  * **Conditional analyses are not run**, exactly as contract §5 and A2.1 require when the linear
    association is not supported. No favourable sensitivity was searched for.
  * No nonlinear term, alternative C2 definition, window, transform, dataset or threshold follows
    from this result.
* Contract Amendment 2 also pre-registers one station-support sensitivity. If and only if the
  linear association is supported, the frozen comparison is repeated excluding Saudi Arabia,
  Yemen, Haiti, Oman and Chad. It cannot change either verdict level.
* The pre-outcome package is reviewed by the owner before the warming outcome is loaded
  for M1b scoring.

## M1b closed (2026-09-16, `M1B_CLOSURE.md`)

* **M1b is a completed negative experiment and is not promoted.** It was verified from saved evidence
  only: no rerun, no regenerated artifact.
* **M1a** remains not promoted and is the registered area-consistent geography measurement sensitivity.
* **M0\*** remains the station-geography development baseline.
* **No automatic follow-up:** no nonlinear or alternative C2, no other M1 predictor, no C1 or C3
  acquisition, no conditional replication, no M3. Any M2 work rests on the `af97bd2` roadmap and a
  separate owner authorization, not on this result.
* `M1B_REPORT.md` carries three dated post-result documentation corrections (chronology of the
  amendments against CRU acquisition, the allocation sentence, and one coverage-rule sentence). No
  number, verdict or artifact changed.
* Twelve evidence limits are recorded in the closure. Two of them:
  * Before the freeze, the M0\* comparators were scored on real inputs, from uncommitted evaluator
    code outside the provenance guard. This happened in ad hoc runs in the main session, in runs by
    review subagents, and in a committed test. No model containing C2 was fitted to any outcome before
    the primary score.
  * The `067bc98` commit message is inaccurate about the scoring-input gate.

No territorial-unit, 500 km threshold, M49, random-reference, unseen-level or
scorecard decision is reopened. No new physical covariate is authorized. Items in
the historical register below are not current authorizations.

## Session 1 methodological decisions (historical register)

**Status update:** items 1–4 are approved and frozen by the owner. The territorial
implementation is corrected in `TERRITORIAL_CV_CORRECTION.md`, using the approved
GPW/ISO analytical units. The options below retain their Session 1 wording as
history and are not requests to reopen those decisions.

The following is the original Session 1 register. Its 'open' and 'not acted on'
wording reflects that session, not current authorization. Items 1–4 are approved;
item 11 is resolved analytically by M0.5, with contract approval requested above.
The current M1a and product-audit specifications supersede preliminary recipes.

1. **Approve the primary protocol.** Leave-one-country-out with a 500 km
   exclusion buffer on the minimum inter-territory distance, unseen levels
   given the mean training effect (`SPATIAL_CV_PROTOCOL.md`). Recommendation:
   approve. Alternatives evaluated and why they were not chosen are in the
   protocol document.

2. **Buffer metric and width.** The buffer is defined on border-to-border
   (minimum 1° cell) distance because a centroid buffer leaves the neighbours of
   Russia, Canada, the United States and Mexico in training; its width (500 km)
   is the first correlogram ring on that metric with no positive residual
   dependence. The earlier candidate, 1500 km on centroid distance, is kept as a
   sensitivity. Recommendation: 500 km on the border metric, frozen for all
   stages even if a later residual has a different range.

3. **Secondary stress test.** Leave-one-M49-sub-region-out. South America is a
   sub-region and a continent at once, so its 11 countries are predicted with
   an unseen `spatial_block` level (mean-effect rule). Recommendation: accept
   as a stress test with that caveat stated; do not merge or split regions to
   avoid it.

4. **Unseen-level rule.** Mean of the training level effects (affects only
   Iceland under the primary; South America and Iceland under the secondary).
   The alternative, the reference level, is arbitrary (Africa, Köppen A, High
   income, Northern Hemisphere). Recommendation: mean effect.

5. **Improvement threshold.** Paired ΔRMSE with a country-bootstrap 95%
   interval excluding zero, vetoed by the secondary protocol. With n = 151 this
   detects RMSE changes of roughly 0.002–0.003 °C/decade. Recommendation:
   accept; report point estimates for smaller changes without calling them
   improvements.

6. **M1a as a stage.** Re-measuring the five geography features over the land
   area keeps their names and groups but changes their values. Treat it (a) as
   a re-measurement under `SCHEMA_V1` names, recorded as stage M1a, or (b) as a
   `SCHEMA_V2`. Recommendation: (a) for the research line, with the schema
   revision decided only if M1a is retained at M4.

7. **Whether M1a becomes the base for M1.** The plan makes M1a the base if it
   is not worse than M0. Alternative: add M1 covariates to M0's measurements
   regardless. Recommendation: as planned.

8. **Product-robustness arm (H7).** Refitting the stages on the ERA5 outcome
   (and the two-product mean) changes the temperature product, which is
   forbidden for the primary results and for this session. The Berkeley–ERA5
   country disagreement (sd 0.063 °C/decade) is larger than the residual, and
   the low–low clusters sit where the products disagree most. Recommendation:
   approve it as an M4 robustness arm only, never as a primary, with the ERA5
   outcome built by the same operator and window.

9. **Aerosol forcing histories (H4).** Admitting a regional-forcing covariate
   touches the responsibility-versus-warming guardrail because aerosol
   precursors are co-emitted with CO₂. Options: exclude; admit as a declared
   separate group "regional forcing" with fixed wording; admit only in a
   sensitivity. Recommendation: separate declared group, M1, with the wording
   agreed in advance; otherwise exclude.

10. **Group contributions for non-OLS stages.** LMG shares are defined by
    coalition R² of the stage's own estimator (refit each coalition); for spatial
    models the spatial term is its own group and the non-spatial shares are
    computed on the trend part. Recommendation: accept; record the definition
    in the scorecard before M2.

11. **The rank-20 identity in M0** (`log10 per-capita = log10 total − log10
    population + 6`). It changes no V1 number; it does mean one exact shared
    direction between the emissions and population groups. Options: leave as is
    (M0 is frozen); in a later schema drop `cum_co2_total` or `population`;
    document only. Recommendation: document now (done in `M0_BASELINE.md`);
    decide the schema question only when V2 results exist, and never restate
    V1 shares.

12. **Station density and Japan.** `station_density` (population group) gives
    Japan leverage 0.47 and partly encodes observational sampling. Moving it to
    an "observational" group is a schema revision. Recommendation: leave for
    this line; revisit with item 6 at M4.

13. **Unit of analysis (H9).** Equal-weight countries (V1 estimand) versus
    area-weighted regression or a cell-level hierarchical model. Recommendation:
    keep the V1 estimand for M1–M4; open a separate line if the owner wants the
    "how land warms" question.

14. **Decadal-variability diagnostic (H8).** A time-series arm that removes
    Pacific/Atlantic mode imprints from each country's series changes the
    outcome definition. Recommendation: not in M1–M4; a candidate for a later
    line, recorded now so a remaining residual is not over-interpreted.

15. **Permutations for Moran's I.** The harness uses 999 (V1: 199); the
    statistic is identical, the p-value finer. Recommendation: 999 in V2,
    with the V1 value quoted as recorded.

16. **Figure tooling.** Research figures use an ephemeral matplotlib
    (`uv run --with matplotlib`) so the project's dependencies are untouched.
    Recommendation: keep unless the paper track needs pinned figure builds,
    then add matplotlib to the `dev` extra.

17. **Publishing this branch.** Nothing has been pushed. Recommendation: push
    the branch after approval so the tag-derived base and the scaffolding are
    backed up; never merge to `main` in this line.
