# Model V2 M0.5 — completion report

## Scope and repository lineage

Research branch: `research/model-v2-residual-structure`.
Immutable V1 baseline: `v1.3.0` → `a6733cb78292b9b27466e7ebfe7c4fdc6b59592f`.
Session 1 scaffold: `af97bd21c30bc7b09708e0bd7c8d71f4d61a334c`, directly based on V1.
Separate territorial enforcement correction:
`4263429f08dccfb63f79bfb165df9d803748feef`.
The substantive M0.5 commit follows that correction without amend/squash.
No public code, docs, dashboard, bundle or baseline tag changes; no push.

## Recovery after the usage interruptions

M0.5 was interrupted twice by usage limits: after 23:55 on 14 September and
after 08:57 on 15 September. Neither interruption was a methodological stop.

**First resumption (15 September morning).**

* The territorial correction `4263429` was already complete.
* The rank runner had completed all four variants and saved scores,
  coefficients, country predictions and deterministic-repeat evidence. These were
  recovered rather than rerun under new variants. An independent test refits the
  saved representations on the frozen data and folds and checks their outputs.
* The legacy product audit's metrics and 9,999-permutation arrays were reused.
  Classification and overlap summaries were patched in.
* The absolute-temperature versus monthly-anomaly mismatch was identified, and
  the aligned ERA5 arm was built and run.

**Final resumption (this commit).** Branch, HEAD and tags were re-verified. The
working tree, every partial output and the `/tmp` run and test logs were
inspected, and all prior product/rank outputs and logs were backed up with
SHA-256 hashes outside the repository.

The inspection found two provenance defects:

* The aligned summary lacked the cross-arm block that the final source always
  writes. It also recorded the *final* source hash, because the file had been
  edited while that run was executing.
* The legacy summary was a hand-patched hybrid of a 23:55 run.

What was reused versus recomputed:

* **Rank audit:** reused. Its fresh real-data reproducibility test passes.
* **M1a specification:** reviewed and accepted as complete.
* **Product audit:** code extended with explicit systematic-bias metrics and the
  cross-arm pattern classification. Both arms were then regenerated in place from
  that single final source, including the aligned anomaly → cell-slope rebuild.
  * Legacy arm: every saved metric, permutation p-value, local label, country row
    and region row reproduced within 1e−12.
  * Aligned arm: likewise. Its country trends and preprocessing record are
    byte-identical and its cell-slope arrays equal.
  * The regeneration therefore recomputed nothing that changed. It repaired
    provenance and added the missing cross-arm outputs.

A final whitespace-only source fix was followed by rerunning both arms, reusing the verified aligned build. Data artifacts were byte-identical and only the summaries' code hash changed. No scratch directory was left in the repository. The aligned build's temporary
anomaly file deletes itself.

## Frozen validation confirmation

- Primary: leave one country out; every represented GPW/ISO cell footprint of
  each coded unit counts; retain a training unit only when its WGS84 certified
  clearance lower bound is **strictly greater than 500 km**. Equality and uncertain
  bounds exclude. No sovereign-state aggregation or new sensitivity.
- Secondary: leave one UN M49 sub-region out, unchanged.
- Random 10-fold: optimistic reference only, never selection criterion.
- Unseen categories: fit levels on training data only; use the mean of training
  category-level effects including reference zero; record affected observations.
- Original scorecard definitions and V1 outcome/predictor values unchanged.
  The correction supplements missing already-defined diagnostics without changing
  definitions. No CV design or threshold is revised in response to results.

The corrected M0 primary baseline is **R² 0.153079913, RMSE 0.042872581 °C/decade**.
The old **0.211980511 / 0.041354892** belongs only to the defective 1° approximation.
M49 remains **0.302970020 / 0.038894137**. The correction changed 100/151 folds,
added 218 directed exclusions, removed none; minimum retained certified lower
bound is 500.682949 km. Its geometry/source proof and qualifications remain in
[TERRITORIAL_CV_CORRECTION.md](TERRITORIAL_CV_CORRECTION.md). This certifies the
adopted full-GPW-footprint geometry, not unrepresented real-world land.

## Rank audit

See [RANK_DEFICIENCY_AUDIT.md](RANK_DEFICIENCY_AUDIT.md) and
[rank_audit_scorecard.csv](outputs/rank_audit_scorecard.csv) for the complete
side-by-side table, standalone R², all coalition values and coefficients.

Exact identity: `log10(per capita) − log10(total Mt) + log10(population) = 6`.
Maximum floating error 1.78e-15; rank 20/21. NumPy least squares returns a
minimum-norm coefficient representative, not uniquely identified log coefficients.
Removing each redundant representation in turn preserves the full column space.

| Representation | Rank / columns | Geography share | Responsibility share | Socioeconomic share | Population share | Residual |
|---|---:|---:|---:|---:|---:|---:|
| Original M0 | 20 / 21 | .513033 | .021393 | .069323 | .032697 | .363554 |
| Drop per-capita | 20 / 20 | .518837 | .013134 | .062821 | .041653 | .363554 |
| Drop total | 20 / 20 | .513520 | .020435 | .069282 | .033208 | .363554 |
| Drop population | 20 / 20 | .513276 | .022190 | .069339 | .031642 | .363554 |

Every row: n = 151; in-sample R² .636446; corrected CV R² .153080/RMSE .042873;
in-sample residual Moran's I .268689. CV predictions differ by at most 4.29e-14.
Partial group coalitions change, explaining the different Shapley allocations.
Responsibility remains small (1.31–2.22% of total country-outcome variance).
This is a descriptive/associational country comparison, not causal climate
attribution and not evidence against greenhouse gases causing global warming.

Proposed V2 primary: total cumulative CO₂ in responsibility plus population in
its existing group, dropping per-capita; proposed registered sensitivity:
per-capita plus population, dropping total. Station density stays in the population
group unchanged. The recommendation is conceptual, not score-based.
[V2_PREDICTOR_CONTRACT.md](V2_PREDICTOR_CONTRACT.md) awaits owner approval.

## M1a specification only

[M1A_MEASUREMENT_SPEC.md](M1A_MEASUREMENT_SPEC.md) defines source versions,
weights, mask hierarchy, island/dateline handling, coverage gates and leakage
limits for all six current geography features:

| Feature | V1 measurement | Future M1a specification |
|---|---|---|
| Absolute latitude | Mean absolute station latitude | Mean absolute terrestrial latitude, absolute before averaging |
| Elevation | Mean station-sampled ETOPO surface height | Area-weighted ETOPO surface, independent land/water mask; retain genuine negative dry land |
| Continentality | Mean station coast distance | Area-mean minimum great-circle distance to the declared coastline; source-resolution limitations recorded |
| Köppen class | Station-modal A–E | Area-dominant A–E with coverage and unclassified-area winner checks; one categorical feature |
| Hemisphere | Station-modal N/S | Hemisphere containing most terrestrial area; no mixed class |
| Spatial block | OWID continent | Unchanged country label |

FRA/GUF and other separately coded units remain separate; every represented
component participates. The proposed GSHHG independent terrestrial mask is
measurement support, not a new physical predictor, and must not shrink the
frozen CV exclusion geometry. Numeric coverage defaults are 98%, climate 95%
plus a robust winner, geometry 100%; failures remain explicit rather than
triggering score-driven threshold relaxation. Current data have 26 Köppen class
differences and 17 negative country-mean elevations; the latter is not a verified
complete bathymetry inventory. Canada's 14.1872° station/centroid displacement
is not the proposed mean-absolute-area-latitude correction. The ~29% within-country
cell variance is distinct from the 36.36% national-regression residual.

M1a must compare to the rank-only M0 equivalent under the **same approved
representation**, so representation-induced share changes are not credited to
remeasurement. No M1a features or model have been built.

## Product stability (Berkeley vs ERA5)

Full methodology, tables and caveats are in
[PRODUCT_STABILITY_AUDIT.md](PRODUCT_STABILITY_AUDIT.md), with pattern classes in
[product_stability_pattern_classification.csv](outputs/product_stability_pattern_classification.csv).

**Comparability.** The comparison covers:

* all 151 M0 countries;
* the exact overlap window Jan 1950–Sep 2013 (765 months);
* per-cell Theil–Sen slopes with a 689-month (90%) coverage gate;
* cos-latitude country aggregation over GPW band-11 native-cell assignment;
* the unchanged rank-20 M0 design, refitted separately to each outcome.

Berkeley is a monthly anomaly relative to 1951–1980. The **legacy ERA5 field is
absolute temperature**. The **preprocessing-aligned research arm** subtracts
ERA5's own 1951–1980 cell/calendar-month climatology and is otherwise unchanged,
and it is the preferred arm here. The native-grid half-degree offset and the
product-specific masks remain in both arms.

| Quantity (Berkeley − ERA5) | Legacy arm | Aligned arm |
|---|---:|---:|
| ERA5 mean / SD | .205882 / .076340 | .192175 / .069326 |
| Berkeley mean / SD | .176156 / .046741 | same |
| Mean / median difference | −.029726 / −.021974 | −.016019 / −.008181 |
| SD of differences / RMSE | .062898 / .069380 | .059077 / .061022 |
| Outcome Pearson / Spearman | .5684 / .6375 | .5402 / .6189 |
| ERA5 > Berkeley countries | 105 | 83 |
| Moran's I of difference, station / area kNN8 | .1703 / .1849 | .1206 / .1238 |
| Residual Moran's I, Berkeley (station / area) | .268689 / .289919 | same |
| Residual Moran's I, ERA5 (station / area) | .0792 / .1180 | .0813 / .1118 |
| Residual Pearson / Spearman | .3625 / .3549 | .3337 / .3260 |
| Residual sign agreement | 90/151 | 89/151 |
| FDR local residual clusters shared | none | none |

Largest regional bias shares (aligned arm, share of squared gaps): Western Asia
18.7%, Northern Europe 17.7%, Central America 10.6%, Northern Africa 10.0%.
Dominant countries are Iceland (12.0%, exactly fitted), Saudi Arabia, Guatemala,
Iraq, Yemen, Bolivia, Norway, Mexico, Tunisia and Australia. The top ten account
for 47.9%.

Alignment halves the mean bias, mostly by lowering ERA5 trends in strongly
seasonal continental countries (Russia −.047). It does not improve agreement.
Disagreement is therefore dominated by product, grid and coverage differences
that M0.5 does not separate.

**Product-robust** (both products, both ERA5 constructions):

* presence of positive residual spatial autocorrelation (both weights);
* the Southern Africa negative regional residual;
* single-country extremes: Canada, Estonia, Indonesia, Kenya (positive);
  Argentina, Central African Republic, South Korea (negative).

**Product-sensitive:**

* residual-autocorrelation magnitude (ERA5 I is 30–41% of Berkeley's);
* regions: Caribbean (reversal), Central America, Central Asia, Western Africa,
  Eastern Europe, Eastern Asia, Northern Africa;
* Berkeley FDR LL clusters in Cyprus, Jordan, Lebanon and Sri Lanka, hence the
  Levant LL structure;
* all three top-15 residual lists;
* 28 single-product extremes, including Egypt, Mexico, Libya, Iran and the
  ERA5-only Gulf/Norway positives.

**Unresolved:**

* 12 regions: n<3, FE-constrained South America, internally cancelling regions,
  and preprocessing-dependent Northern Europe;
* UAE HH and Lesotho LL;
* 22 extremes that are near-top in the other product or that depend on the ERA5
  construction.

None of these is a mechanism claim. ERA5 remains outside predictor choice, tuning
and model ranking, and a post-freeze ERA5 check of the eventual V2 model (both
constructions) remains required. The hypothesis register now states the product
status of the evidence each Session 1 entry cites.

## Validation and commit state

* Research tests: `uv run python -m pytest research/model_v2/tests -q`.
* Full suite: `uv run python -m pytest tests research -q`.
* Ruff: `uv run ruff check .`.
* `git diff --check`.

All were run before this commit; the results are in the session report.

Relative to `v1.3.0`, nothing outside `research/` changed. `research/__init__.py`
was added empty by `af97bd2`. Every output committed in `af97bd2` and `4263429` is
byte-identical in this commit. The corrected M0 baseline (R² 0.153079913, RMSE
0.042872581) is unchanged. The commit is local and unpushed.

## Decisions remaining before M1a

Only the two items in [OPEN_DECISIONS.md](OPEN_DECISIONS.md):

1. approve the V2 representation pair;
2. approve starting M1a under the completed measurement specification.
