# M0.5 Berkeley–ERA5 product-stability audit

## Decision

Two ERA5 constructions are compared with the frozen Berkeley M0 outcome on the
same 151 countries and the unchanged M0 design:

* **Legacy/existing ERA5 construction** (absolute monthly 2 m temperature). This
  is the field behind V1's ERA5 cross-check and Session 1's statements. It is kept
  unchanged for the audit trail.
* **Preprocessing-aligned ERA5 research arm** (ERA5 monthly anomalies against its
  own 1951–1980 calendar-month climatology, matching Berkeley's anomaly
  convention). This is the **preferred arm for interpreting product robustness**.
  It is research-only: it does not change M0, the frozen Berkeley outcome, V1, or
  any predictor, and it was not selected for agreement (§2.3).

The two constructions agree on the classification of 82 of the 92 classified
patterns. The exceptions are Northern Europe and nine near-threshold extreme
countries, all of which combine to unresolved (§5). In the aligned arm:

* Residual Pearson is **0.334**, Spearman **0.326**, and signs agree for
  **89/151** countries (legacy: 0.362, 0.355, 90/151).
* Positive global residual autocorrelation appears in both products under both
  weight schemes, so its **presence is product-robust**.
* Its **magnitude is product-sensitive**. ERA5's Moran's I is only 30–41% of
  Berkeley's.
* Southern Africa's negative regional residual is the only product-robust
  regional pattern.
* The individual extreme residuals of **Canada, Estonia, Indonesia and Kenya
  (positive)** and **Argentina, Central African Republic and South Korea
  (negative)** replicate in both products under both constructions.
* The Berkeley-FDR local clusters are not replicated:
  * Cyprus, Jordan, Lebanon and Sri Lanka LL are product-sensitive. Jordan's
    and Lebanon's ERA5 residuals reverse sign.
  * UAE HH is unresolved.
* Most regional and extreme-country patterns are product-sensitive or unresolved.

Aligning preprocessing roughly halves the mean Berkeley−ERA5 gap. It reduces the
spatial structure of the gap without changing residual agreement materially. The
disagreement is therefore not mainly an anomaly-convention artifact.

This is an **external outcome robustness diagnostic**, not an ERA5-trained model
selection exercise. Nothing was changed: predictors, transformations, feature
selection, CV folds, exclusion buffers, tuning and the primary outcome are all as
frozen. A post-freeze ERA5 check of the eventual model remains necessary. Nothing
here establishes a physical mechanism or a measurement-error variance.

## 1. Provenance and comparability

All inputs are local; no downloads were performed. The summary JSON of each arm
stores exact SHA-256 hashes and byte counts for:

* raw grids;
* the GPW mask and lookup;
* consumed processed and bundled inputs;
* M0 source, spatial code and audit code.

The files are
[`product_stability_summary.json`](outputs/product_stability_summary.json) and
[`product_stability_aligned_summary.json`](outputs/product_stability_aligned_summary.json).
Raw coordinate/time axes and country mask counts are inspected directly rather
than inferred from filenames. The legacy ERA5 cell slopes are not recomputed;
that arm reads the bundled `era5_area_trends.parquet`. The aligned arm recomputes
ERA5 cell slopes from the raw grid (§2.3). Hashes establish which bytes were
inspected, not independent validation of all historical upstream computations.

| Property | Berkeley | ERA5 |
|---|---|---|
| Raw file | `data/raw/berkeley_gridded/Complete_TAVG_LatLong1.nc` | `data/raw/era5/era5_t2m_monthly_1950_2013_1deg.nc` |
| Variable | `temperature`, °C anomaly vs Jan 1951–Dec 1980 climatology | `t2m`, absolute K |
| Raw temporal coverage | Jan 1750–Dec 2024, 3300 months | Jan 1950–Dec 2013, 768 months |
| Fitted window (exact overlap) | Jan 1950–Sep 2013 inclusive, 765 months | identical |
| Grid | 180 × 360; latitude −89.5…89.5, longitude −179.5…179.5 | 181 × 360; latitude 90…−90, longitude 0…359 |
| Grid spacing | 1° | 1°; half-degree offset relative to Berkeley |
| GPW-assigned cells, all countries | 16,659 | 16,720 |
| Distinct GPW-assigned ISO3 codes | 202 | 207 |
| Common-M0 country cell coverage | mean 0.9822; minimum 0.5 | all 1.0 |

**Trend estimator.** `src/area_weighting.py::cell_trends` fits
`scipy.stats.theilslopes` to each cell's finite monthly observations against
decimal decades, `(year + (month - 0.5)/12)/10`.

* Both products and both ERA5 constructions use this function.
* All use the same 90% temporal coverage requirement: at least
  ceil(0.9 × 765) = **689 observations**.
* Berkeley fractional years and ERA5 CF timestamps are decoded to first-of-month
  for window selection.

**Aggregation.** Country outcomes average the fitted cell slopes using normalized
cos(latitude) weights. They are not slopes of a country-average time series, and
for a nonlinear estimator the two operations need not commute.

**Country operator.** Both paths sample GPW v4 rev11 national identifiers at each
native temperature cell centre. The source is band 11 of the 15-arc-minute file,
with numeric codes mapped to ISO3.

* Ocean, missing and unmapped codes do not contribute.
* ERA5 longitudes are normalized to [−180, 180) for sampling, without reordering
  its temperature array.
* Country masks are neither polygon-intersection area fractions nor a harmonized
  shared raster.
* Native grid-centre offsets change coast, island and border assignment. Assigned
  cells, Berkeley versus ERA5: Egypt 92 vs 102; Guatemala 11 vs 9; Jamaica 1 vs 4;
  Australia 739 vs 758.
* Reported cell coverage is fitted cells over assigned cells. It does not
  guarantee national representativeness or observation quality.
* Berkeley's supplied fractional land-mask variable is not used as an additional
  weight.

**Preprocessing difference.** Berkeley's temperature is a monthly anomaly. Its
`climatology` metadata identifies **Jan 1951–Dec 1980**. The legacy ERA5 field is
absolute monthly temperature, and the shared estimator neither deseasonalizes nor
annualizes.

* A scalar Kelvin-to-Celsius offset leaves every pairwise slope unchanged.
* Subtracting a *different* climatology for each calendar month generally
  **does** change a Theil–Sen slope.
* The window ends in September, so the final partial year adds summer-heavy
  months at its end. This can interact with a large seasonal cycle.

The legacy construction therefore combines several kinds of difference: product,
native grid/mask, coverage, and anomaly/seasonal preprocessing. V1 code comments
saying that only the data source differs must be read with this qualification.
The aligned arm removes the anomaly-convention difference. It leaves the product,
the native grid and mask, coverage and the half-degree offset unchanged, so it is
*more* comparable, not *fully harmonized*.

Supporting code paths:

* `src/era5_weighting.py`, `src/area_weighting.py`, `src/grids.py`,
  `src/cleaning.py`: the shared trend and country code.
* `scripts/fetch_era5.py`: requests ERA5 monthly-averaged reanalysis at 1°. Its
  normalization squeezes and renames dimensions to `t2m(time, latitude, longitude)`.

Berkeley's file metadata records a January 2025 production history. ERA5's
normalized file has no global production-history attributes. The preserved GRIB
variable attributes identify the field and grid, not a complete acquisition and
version history.

## 2. Country population, paired fits and outcome agreement

### 2.1 Sample and fits

The inequality input contains 157 countries. Frozen M0 complete cases contain 151.
Bahrain, Gambia, Guyana, Hong Kong, Mauritius and Singapore are outside that
design.

* **Every M0 country has a finite ERA5 outcome in both constructions.** Common
  n = **151**, with no additional exclusions.
* ERA5 outcomes exist for 207 ISO3 codes.
* The older validation bundle's 153 countries common across warming lenses form a
  different analysis population. They must not replace the M0 sample.

The audit calls `m0.load_inputs`, `m0.m0_complete_design` and `m0.design_matrix`.
It builds one finite common-row mask before either fit, then solves
`np.linalg.lstsq(X_common, y_product, rcond=None)` separately for each outcome.

* The exact column names, group assignments and rank **20** are recorded in JSON.
* This keeps the original rank-deficient M0 column space and its continent fixed
  effects. By the rank audit, residuals are identical under every full-rank
  representation.
* Berkeley's reproduced R² is **0.6364460028** and its station-kNN8 residual
  Moran's I is **0.268689**, matching frozen M0.
* The fits are in-sample diagnostics, not predictive validation or new scorecard
  rows.

Conventions for the tables below:

* Country means and correlations weight each country equally, independently of
  the within-country area weighting.
* All temperature quantities are °C/decade; sample SDs use n−1.
* Differences are **Berkeley minus ERA5**.

### 2.2 Outcome and residual agreement

| Quantity | Berkeley | ERA5 legacy | ERA5 aligned |
|---|---:|---:|---:|
| Mean outcome | 0.176156 | 0.205882 | 0.192175 |
| Outcome sample SD | 0.046741 | 0.076340 | 0.069326 |
| Residual sample SD | 0.028183 | 0.053962 | 0.050690 |
| In-sample R² (same design) | 0.636446 | 0.500341 | 0.465366 |

| Paired quantity | Legacy arm | Aligned arm |
|---|---:|---:|
| Mean difference (systematic bias) | −0.029726 | −0.016019 |
| Median difference | −0.021974 | −0.008181 |
| Difference sample SD | 0.062898 | 0.059077 |
| Difference RMSE | 0.069380 | 0.061022 |
| ERA5 > Berkeley / Berkeley > ERA5 (countries) | 105 / 46 | 83 / 68 |
| OLS ERA5-on-Berkeley slope / intercept | 0.9284 / 0.0423 | 0.8012 / 0.0510 |
| ERA5/Berkeley outcome SD ratio | 1.633 | 1.483 |
| Outcome Pearson r | 0.568414 | 0.540172 |
| Outcome Spearman rho | 0.637470 | 0.618899 |
| Residual Pearson r | 0.362477 | 0.333736 |
| Residual Spearman rho | 0.354876 | 0.325988 |
| Residual sign agreement | 90/151 (59.6%) | 89/151 (58.9%) |
| Berkeley residual vs ERA5−Berkeley, Pearson / Spearman | −0.1371 / −0.1422 | −0.1907 / −0.2160 |

Signs treat absolute residuals below 1e−10 as zero, so floating-point noise in
exactly fitted countries is not read as a sign. Both residual vectors use the
same projection, so their difference equals the residualized outcome difference.
This is a refitted residual correlation, not a correlation of the Berkeley
residual with an unrefitted ERA5 outcome.

### 2.3 The preprocessing-aligned arm

`product_stability.build_aligned_outcome` does four things:

1. Checks that Berkeley's climatology metadata still reads Jan 1951–Dec 1980.
2. Subtracts, for every ERA5 cell and calendar month, that cell's mean over
   exactly the 30 baseline years for that month. The baseline is checked to
   contain 30 per month. No fitted trend or other product is used.
3. Writes the anomaly field only to a temporary file that is deleted afterwards.
4. Runs the **unchanged** `era5_cell_slopes` / `reduce_era5_slopes` operator.
   Window, coverage gate, GPW operator and cos-latitude aggregation are the same.

Outputs are the cell slopes (`product_stability_aligned_cell_slopes.npz`), the
country trends (`product_stability_aligned_era5_trends.csv`) and the
preprocessing record (`product_stability_aligned_preprocessing.json`). The
preprocessing record includes the raw-grid SHA-256.

The alignment rule is fixed by Berkeley's own documented convention, and it was
recorded before aligned results existed (`selected_before_aligned_results`). No
alternative baseline, annualization or grid harmonization was tried.

Effect on ERA5 country trends (aligned minus legacy): mean −0.01371, sample SD
0.01624, range −0.0466 to +0.0212.

* The largest reductions are in strongly seasonal continental or high-latitude
  countries: Russia −0.047, Canada −0.046, Mongolia −0.044, Kazakhstan −0.042,
  Djibouti −0.042, Moldova −0.040.
* This is consistent with a seasonal-cycle interaction in the absolute-temperature
  slopes. It is not an independent proof of that mechanism.
* Across constructions, ERA5 residual Pearson is 0.9878, and 144/151 residual
  signs are unchanged.
* The exact values are in the aligned summary's `across_preprocessing_arms` block.

**Interpretation.** Alignment removes about half of the systematic mean bias and
reduces the difference RMSE by 12%. Countries where ERA5 exceeds Berkeley drop
from 105 to 83. Correlations and residual agreement do not improve; they decrease
slightly. Most country-level disagreement therefore comes from the remaining
product, grid and mask, and coverage differences. Their relative contributions are
**unresolved**, and separating them would need grid harmonization, which is
outside M0.5.

## 3. Global and local spatial structure

Both products and their difference use identical spatial weights within the
audit:

* haversine kNN with k=8, directed and row-standardized, with deterministic
  index tie-breaking;
* the main row uses station centroids, matching V1; the sensitivity row uses the
  archived area centroids.

Neither point summary claims to represent the full territory used in the separate
corrected territorial CV audit.

Each test uses **9,999 permutations, seed 0**.

* **Global Moran p-values** follow the existing V1 two-sided absolute-statistic
  convention: `(|I_perm| >= |I_obs|)` with a plus-one correction, rather than
  distance from the theoretical null mean.
* **Local Moran** uses conditional permutations holding the focal value fixed,
  with the same absolute-statistic comparison and plus-one correction.
* p = 0.0001 is the simulation resolution, not zero.
* No asymptotic correlation p-values are interpreted as independent-country
  evidence.

| Weights | Arm | Difference Moran I (p) | Berkeley residual I (p) | ERA5 residual I (p) | ERA5/Berkeley I |
|---|---|---:|---:|---:|---:|
| Station centroid kNN8 | legacy | 0.170311 (0.0001) | 0.268689 (0.0001) | 0.079219 (0.0261) | 0.295 |
| Station centroid kNN8 | aligned | 0.120609 (0.0023) | 0.268689 (0.0001) | 0.081327 (0.0229) | 0.303 |
| Area centroid kNN8 | legacy | 0.184864 (0.0001) | 0.289919 (0.0001) | 0.117953 (0.0009) | 0.407 |
| Area centroid kNN8 | aligned | 0.123784 (0.0012) | 0.289919 (0.0001) | 0.111825 (0.0020) | 0.386 |

**Global patterns.**

* The **presence** of positive residual autocorrelation is product-robust:
  positive with p<0.05 in both outcomes, under both weights, in both arms.
* Its **strength** is product-sensitive: ERA5's statistic is under half of
  Berkeley's in every row.
* The Berkeley−ERA5 difference is itself spatially autocorrelated. Alignment
  weakens that but does not remove it, so the product disagreement has regional
  structure.

None of this identifies which mechanism explains either field.

### Local cluster and spatial-outlier overlap

Terms used here:

* HH and LL: above- or below-mean residuals with similarly signed spatial lags
  (clusters).
* HL and LH: spatial outliers.
* "Nominal": conditional p<0.05.

Complete country identities, statistics, p-values, quadrants and adjusted q-values
are in the country CSVs. All overlap sets are in JSON.

| Weights | Class | Berkeley nominal n | ERA5 nominal n (legacy / aligned) | Same-country overlap (both arms) |
|---|---|---:|---:|---|
| Station | HH | 14 | 6 / 4 | UAE only |
| Station | LL | 15 | 11 / 11 | Lesotho only |
| Station | HL | 0 | 1 / 2 | none |
| Station | LH | 0 | 2 / 1 | none |
| Area | HH | 16 | 8 / 8 | UAE only |
| Area | LL | 15 | 10 / 9 | Lesotho only |
| Area | HL | 1 | 1 / 1 | none |
| Area | LH | 1 | 3 / 1 | none |

**Nominal patterns by product.**

* Berkeley HH spans Central Asia and western Africa.
* ERA5 HH is in northern Europe (Estonia, Finland, Sweden, plus Norway under area
  weights) and in the Gulf/Horn under area weights.
* Berkeley LL covers the Levant/Egypt/Turkey and parts of southern and
  southeastern Asia.
* ERA5 LL is concentrated in southern Africa and the Balkans.
* Spatial outliers are not cross-product replicated.

**FDR-adjusted results.** Benjamini–Hochberg q-values are calculated separately
over the 151 countries for each map (field × weights × arm).

* At q≤0.05, Berkeley's station map retains UAE HH, and Cyprus, Jordan, Lebanon
  and Sri Lanka LL.
* **Neither ERA5 construction retains any residual local cluster** under either
  weight system.
* Neither Berkeley area-centroid map retains a country.
* The legacy difference field retains Qatar HL (both weights) and Eritrea, Israel
  and Jordan LL (area weights). The **aligned difference field retains none**.
* There is **no FDR-supported cross-product residual cluster overlap**.

Limits of the adjustment:

* It is within each map, not across every map or analytical question.
* BH's usual guarantee is not asserted under the arbitrary spatial dependence
  present here.
* Nominal overlaps are exploratory. Absence of adjusted significance does not
  prove absence of local structure.

## 4. Regional biases and dominating countries

### 4.1 Regional rule and table

The regional rule is descriptive. It was specified for this audit before the
aligned arm existed and is applied identically to both arms; it was not tuned to
obtain preferred regions.

A regional pattern needs **n≥3**. A regional mean counts as material when its
absolute value is at least 0.5 × that product's full-sample residual SD. The
thresholds are:

* Berkeley **0.014091**;
* ERA5 legacy **0.026981**;
* ERA5 aligned **0.025345**.

Per-arm classes:

* **Product-robust:** both regional means are material, have the same sign, and
  at least 75% of country residual signs agree.
* **Product-sensitive:** exactly one regional mean is material, or both are
  material with opposite signs. The one-material rule applies symmetrically.
* **Unresolved:** all other cases, including n<3 and weak aggregate means.

Caveats on reading the table:

* The thresholds are a reproducible description, not confidence intervals or
  selection criteria for M1. A near-threshold region can change class under
  another reasonable description.
* Local inference remains a separate, stricter evidence layer.
* Fixed effects force continent-average residuals to zero. South America's
  subregion coincides with such a block, so its near-zero means are a
  construction constraint, **not evidence that its countries lack structure**.
* Other subregions can cancel internally too, notably Western Asia.

B−E is the aligned-arm mean Berkeley−ERA5 regional outcome difference (the
regional bias). Residual means are aligned; legacy ERA5 means are in the regions
CSV.

| Subregion | n | B−E mean (aligned) | B−E mean (legacy) | B residual mean | E residual mean (aligned) | Signs agree (aligned) | Legacy class | Aligned class |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Australia and New Zealand | 2 | +0.0522 | +0.0704 | −0.0153 | −0.0278 | 0% | unresolved | unresolved |
| Caribbean | 5 | +0.0473 | +0.0404 | +0.0181 | −0.0180 | 60% | sensitive | sensitive |
| Central America | 7 | −0.0625 | −0.0662 | −0.0240 | +0.0032 | 43% | sensitive | sensitive |
| Central Asia | 4 | +0.0180 | −0.0181 | +0.0276 | +0.0028 | 50% | sensitive | sensitive |
| Eastern Africa | 14 | −0.0104 | −0.0189 | +0.0044 | −0.0007 | 64% | unresolved | unresolved |
| Eastern Asia | 5 | +0.0215 | −0.0087 | −0.0011 | −0.0409 | 60% | sensitive | sensitive |
| Eastern Europe | 10 | −0.0072 | −0.0400 | −0.0198 | −0.0144 | 80% | sensitive | sensitive |
| Melanesia | 1 | −0.0050 | −0.0096 | +0.0305 | +0.0555 | 100% | unresolved | unresolved |
| Middle Africa | 8 | −0.0019 | +0.0063 | −0.0131 | −0.0095 | 38% | unresolved | unresolved |
| Northern Africa | 6 | −0.0940 | −0.1142 | −0.0136 | +0.0367 | 67% | sensitive | sensitive |
| Northern America | 2 | −0.0171 | −0.0543 | +0.0388 | +0.0337 | 50% | unresolved | unresolved |
| Northern Europe | 10 | −0.0333 | −0.0572 | +0.0093 | +0.0260 | 80% | unresolved | sensitive |
| South America | 11 | −0.0210 | −0.0202 | −0.0000 | −0.0000 | 45% | unresolved | unresolved |
| South-eastern Asia | 8 | −0.0405 | −0.0411 | −0.0097 | +0.0156 | 50% | unresolved | unresolved |
| Southern Africa | 5 | +0.0203 | +0.0272 | −0.0265 | −0.0497 | 100% | **robust** | **robust** |
| Southern Asia | 7 | −0.0185 | −0.0415 | −0.0024 | −0.0220 | 57% | unresolved | unresolved |
| Southern Europe | 11 | +0.0099 | −0.0108 | +0.0054 | −0.0198 | 64% | unresolved | unresolved |
| Western Africa | 14 | −0.0131 | −0.0132 | +0.0183 | +0.0082 | 50% | sensitive | sensitive |
| Western Asia | 15 | −0.0414 | −0.0684 | −0.0019 | +0.0143 | 53% | unresolved | unresolved |
| Western Europe | 6 | −0.0092 | −0.0379 | +0.0105 | +0.0185 | 67% | unresolved | unresolved |

**Region notes.**

* Southern Africa is the only region meeting the robust rule in both arms; all
  five country signs agree.
* Central America, Central Asia, Western Africa and Eastern Europe have material
  Berkeley patterns that attenuate under ERA5.
* Caribbean residual means reverse sign.
* Eastern Asia and Northern Africa have material ERA5 patterns but weak Berkeley
  regional means. Northern Africa's Berkeley mean narrowly misses the threshold.
* **Northern Europe** is sensitive only in the aligned arm, where ERA5's mean
  crosses the threshold at 0.0260 vs 0.0253. It is combined as unresolved (§5).
* Australia/New Zealand and Northern America have only two countries and
  Melanesia only one, so aggregate claims are unresolved. The ANZ gap is driven
  mainly by Australia.

### 4.2 Countries and regions dominating disagreement

Country discrepancy is concentrated rather than uniformly large. In the aligned
arm, shares of the sum of squared outcome differences are:

* by region: Western Asia **18.7%**, Northern Europe **17.7%**, Central America
  **10.6%**, Northern Africa **10.0%**, Western Africa **7.3%**;
* by country: the largest five countries **32.6%**, the largest ten **47.9%**.

The legacy arm gives Western Asia 21.9%, Northern Europe 15.7%, Northern Africa
11.0%, Central America 8.8%, top five 30.7%, top ten 45.4%. These are
equal-country squared-gap shares, not shares of world land or uncertainty.

| Country (aligned rank) | Berkeley | ERA5 legacy | ERA5 aligned | B residual | E residual (aligned) | Squared-gap share (aligned) |
|---|---:|---:|---:|---:|---:|---:|
| Iceland | 0.1274 | 0.3897 | 0.3868 | ~0 | ~0 | 11.97% |
| Saudi Arabia | 0.1913 | 0.4225 | 0.3900 | −0.0072 | +0.1455 | 7.02% |
| Guatemala | 0.0508 | 0.2292 | 0.2266 | −0.0221 | +0.0710 | 5.50% |
| Iraq | 0.1763 | 0.3745 | 0.3360 | +0.0071 | +0.1429 | 4.53% |
| Yemen | 0.2015 | 0.3743 | 0.3441 | +0.0357 | +0.1113 | 3.62% |
| Bolivia | 0.1078 | 0.2475 | 0.2467 | −0.0008 | +0.0667 | 3.43% |
| Norway | 0.2019 | 0.3671 | 0.3384 | −0.0089 | +0.1271 | 3.31% |
| Mexico | 0.0407 | 0.1849 | 0.1761 | −0.0615 | −0.0068 | 3.26% |
| Tunisia | 0.2375 | 0.3788 | 0.3634 | +0.0031 | +0.0904 | 2.82% |
| Australia | 0.1630 | 0.0245 | 0.0457 | +0.0159 | −0.0643 | 2.45% |

**Country notes.**

* Iceland has the largest outcome discrepancy but essentially zero residual in
  both fits, because the design fits it exactly. Discrepancy magnitude alone does
  not identify a residual hotspot.
* The Gulf and Arabian countries and Norway carry large ERA5-only positive
  residuals.
* Mexico and Egypt carry Berkeley-only negative residuals. Egypt is −0.0721 in
  Berkeley vs −0.0257 in aligned ERA5.

## 5. Final pattern classification

[`outputs/product_stability_pattern_classification.csv`](outputs/product_stability_pattern_classification.csv)
is generated by `classify_patterns`. For every pattern it records the evidence and
the class in each arm, plus the combined class.

**Combination rule.** If the legacy and aligned arms agree, that class stands.
If they differ, the pattern is **unresolved**. A pattern is product-robust only
when both constructions say so.

**Rules per pattern type.** These were declared during M0.5, after the legacy arm
had been inspected, and are applied identically to both arms. They are
descriptive screens, not significance tests or M1 selection rules.

* **Global presence:** positive Moran's I with p<0.05 in both products.
* **Global magnitude:** robust if ERA5 I ≥ 0.5 × Berkeley I.
* **Regional:** the §4 rule.
* **Local clusters:**
  * "strong" means BH q≤0.05 in the given quadrant; "weak" means nominal only.
  * Robust = strong in both products. Sensitive = strong in one, absent in the
    other. Otherwise unresolved.
  * Candidates are all strong entries in either product plus nominal
    cross-product overlaps.
* **Extreme countries:** the same strong/weak logic, where strong is a
  same-direction top-15 residual and weak is rank 16–30.
* **Top-15 lists:** robust only if more than half of the entries overlap.

| Type | Product-robust | Product-sensitive | Unresolved |
|---|---:|---:|---:|
| Global (2 weights × presence/magnitude) | 2 | 2 | 0 |
| Regional (20 M49 subregions) | 1 | 7 | 12 |
| Local clusters | 0 | 4 | 4 |
| Top-15 lists | 0 | 3 | 0 |
| Extreme countries | 7 | 28 | 22 |

### Product-robust

Reasonable descriptive grounds for later hypothesis generation, not mechanisms:

1. **Presence of positive residual spatial autocorrelation**, under both station
   and area kNN8 weights.
2. **Southern Africa's negative regional residual** (Berkeley −0.0265; ERA5
   −0.0612 legacy and −0.0497 aligned; 5/5 signs agree).
3. **Single-country extreme residuals** in the same-direction top 15 of both
   products under both constructions:
   * positive: Canada, Estonia, Indonesia, Kenya;
   * negative: Argentina, Central African Republic, South Korea.

   These are individual countries, not spatial patterns. Canada is also the
   country most affected by station-versus-land displacement, which M1a addresses
   by remeasurement without any new predictor.

### Product-sensitive

Do not use these to justify M1 predictors without additional evidence.

* **Residual autocorrelation magnitude.** ERA5's I is 30–41% of Berkeley's.
* **Regions:**
  * Caribbean (sign reversal);
  * Central America, Central Asia, Western Africa, Eastern Europe (Berkeley
    pattern attenuated in ERA5);
  * Eastern Asia, Northern Africa (ERA5-only patterns).
* **Berkeley FDR local LL clusters** in Cyprus, Jordan, Lebanon and Sri Lanka
  (station weights). None is even nominal in ERA5, and the Jordan and Lebanon
  residuals are positive in ERA5. Berkeley's Levant/Egypt LL structure as a whole
  is not replicated.
* **All three top-15 extreme lists**: overlap 4, 4 and 1 of 15 in the aligned arm.
* **Berkeley-only extremes:**
  * positive: Angola, Brazil, Dominican Republic, Haiti, Iran, Senegal;
  * negative: Egypt, Israel, Libya, Mexico, Myanmar, New Zealand, Romania,
    Slovakia, Sudan.
* **ERA5-only extremes:**
  * positive: Iraq, Mali, Norway, Paraguay, Saudi Arabia, Slovenia, Tunisia;
  * negative: Georgia, Ireland, Liberia, Pakistan, Sierra Leone, Somalia.

### Unresolved

* **Regions:**
  * n<3: Australia/New Zealand, Melanesia, Northern America;
  * FE-constrained: South America;
  * internally cancelling or weak: Western Asia, Eastern Africa, Middle Africa,
    South-eastern Asia, Southern Asia, Southern Europe, Western Europe;
  * preprocessing-dependent: Northern Europe.
* **UAE HH** (Berkeley FDR under station weights; ERA5 nominal only) and
  **Lesotho LL** (nominal in both products, FDR in neither), plus both under area
  weights.
* **Extremes in the top 15 of one product and rank 16–30 of the other:**
  * Latvia, Morocco, Mauritania, Turkmenistan, Algeria, Yemen, Papua New Guinea
    (positive);
  * Botswana, Syria, Cameroon, Greece, Hungary, Eswatini (negative).
* **Extremes whose class depends on the ERA5 construction:**
  * Tanzania, Bolivia, Guatemala, Spain (positive);
  * India, Australia, South Africa, Colombia, Mongolia (negative).

## 6. Check against Session 1 statements

All checks below use the legacy arm, which is what Session 1 used.

* **Rounded summary statistics.** The 0.063 difference SD, +0.030
  ERA5-minus-Berkeley mean and 0.64 Spearman agreement reproduce on 151
  countries. This report uses Berkeley-minus-ERA5, so reverse the sign when
  comparing with the old prose. Under aligned preprocessing these become 0.059,
  +0.016 and 0.62.
* **Difference SD vs outcome SD.** A difference SD of 0.0629 exceeds Berkeley's
  outcome SD (0.0467) and residual SD (0.0282), but not ERA5's outcome SD
  (0.0763). "Larger than the outcome's own SD" is correct only with Berkeley named
  explicitly. It is a discrepancy comparison, not a calibrated uncertainty lower
  bound.
* **Regional directions and example trends** (Egypt, Mexico, Guatemala, Libya,
  Saudi Arabia, Iraq) agree with the saved products at the stated rounding.
* **Difference Moran's I.** The rounded I≈0.18 matches **area-centroid** weights
  (0.1849), not station-centroid weights (0.1703). The old p=0.001 was limited by
  its permutation count. With 9,999 permutations both weightings give p=0.0001.
  Under aligned preprocessing the statistic falls to about 0.12.
* **Weak Berkeley-residual relationship with ERA5−Berkeley.** Reproduced: Pearson
  −0.1371, Spearman −0.1422. The old "rho" notation was ambiguous; neither
  statistic is a refitted residual correlation.
* **"Low–low clusters sit exactly where the products disagree most" is too
  strong.**
  * Iceland is the largest gap and is exactly fitted.
  * The Gulf contains large ERA5 positive residuals.
  * Only Lesotho is nominally LL in both products, and no LL overlap survives FDR.
  * Overlap and refitting qualify the original visual observation. They do not
    support a station-sparsity or desert mechanism claim.

## 7. What M1 may safely use

1. The product-robust patterns in §5 may **motivate** independently justified
   physical hypotheses, within the descriptive and local-inference limits above.
   They are not evidence that any particular candidate variable works.
2. Product-sensitive and unresolved patterns may stay in the hypothesis register
   as observations. They must not be the justification for an M1 predictor.
   * Do not promote a mechanism because it fits a Berkeley-only pattern or
     reproduces an ERA5-only one.
   * Region-level cancellation must not erase country differences from the
     register.
3. Keep ERA5 out of predictor choice, feature transforms, model ranking, threshold
   tuning and CV design.
   * Freeze the eventual M1 specification on the primary protocol before its
     **final post-freeze ERA5 check**, using both constructions.
   * This M0 audit cannot substitute for that check.
   * ERA5 has now been inspected, so describe its role transparently as external
     robustness, not as an unseen validation sample.
4. Separating product disagreement from grid/mask and coverage differences would
   need a harmonized-grid construction. That is additional research, not
   permission to change this frozen comparison or to average the products into a
   new training outcome.

## Reproduction and verification

From the repository root, in this order (the aligned run reads the legacy
artifacts):

```sh
OPENBLAS_NUM_THREADS=1 uv run python -m research.model_v2.product_stability
OPENBLAS_NUM_THREADS=1 uv run python -m research.model_v2.product_stability --build-aligned
uv run python -m pytest research/model_v2/tests/test_product_stability.py -q
```

The first command took about 1 minute and the aligned build plus run about 90 minutes wall time on the audit machine.

Generated research artifacts (prefix `product_stability_` for the legacy arm,
`product_stability_aligned_` for the aligned arm):

* `*_countries.csv`: both outcomes and refits, signs, gap shares, and complete
  local diagnostics for both geometries.
* `*_regions.csv`: regional means, disagreement medians and SDs, counts, sign
  agreement, gap shares and classifications.
* `*_summary.json`: full-sample metrics, systematic bias, exact predictor columns,
  top-tail sets, local overlap sets, global tests and source hashes. The aligned
  summary also has `across_preprocessing_arms`, `alignment` and
  `pattern_classification_counts`.
* `product_stability_aligned_{cell_slopes.npz, era5_trends.csv, preprocessing.json}`:
  the aligned ERA5 construction.
* `product_stability_pattern_classification.csv`: the §5 classification.

Nine focused tests cover:

* common-row-before-fit behaviour and the shared residual projection identity;
* BH ordering and missing values;
* regional classification and its one-material symmetry;
* local overlap;
* monthly-anomaly normalization (a seasonal cycle changes a Theil–Sen slope; a
  constant offset does not);
* the strong/weak evidence rule and arm combination;
* tail ranks consistent with the tail lists;
* an end-to-end synthetic cross-arm classification.

**Recovery record.** Two usage interruptions affected this audit.

* The **legacy arm** was first computed at 23:55 on 14 September. Its summary and
  regions were then patched with classification and overlap summaries at 08:48 on
  15 September, so its code hash described an intermediate source.
* The **aligned arm** was built and run at 08:56–08:57 on 15 September under a
  source version that predated the cross-arm block. The source file was edited
  during that run, so the recorded code hash did not describe the executing code.

In the resuming session, every prior artifact and log was first backed up with
hashes outside the repository. Both arms were then regenerated from the final
source, deterministically and in place:

* The legacy arm reproduced every saved number, permutation p-value and local
  label within 1e−12.
* The aligned arm reproduced likewise (see `M0_5_REPORT.md`).
* Only the new `systematic_bias`, `across_preprocessing_arms` and pattern
  classification outputs are additions.

After a final whitespace-only source fix, both runs were repeated with `--aligned`, reusing the verified build. All data artifacts were byte-identical and only the recorded code hash changed. The committed summaries carry the hash of the code that produced them. No
production artifacts or older research outputs were overwritten.
