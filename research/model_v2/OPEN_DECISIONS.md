# Open methodological decisions

Items that need the project owner's judgement before M1 starts. Each has a
recommendation; none has been acted on.

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
