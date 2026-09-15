# M1a — area-consistent geography measurement specification

**Status: specification only; no predictor construction, download or model fit.**
This specifies a measurement-only contrast against an M0-equivalent comparator
using the same owner-approved full-rank V2 feature contract on both sides. It
replaces where existing geography is measured, retaining its concepts, groups
and functional form. It does not establish a better model or a causal explanation.

## 1. Frozen scientific and geographic contract

M0 is `v1.3.0`, commit `a6733cb78292b9b27466e7ebfe7c4fdc6b59592f`.
The active contract is the actual `build_country_design` / `feature_block` path
in [src/decomposition.py](../../src/decomposition.py), checked by
[research/model_v2/m0.py](m0.py), rather than every feature mentioned anywhere
in the repository. [SCHEMA_V1](../../src/feature_schema.py) and its
[YAML mirror](../../docs/feature_schema_v1.yaml) declare six active geography
features; **all six are actually materialized** (see §4). M0 has 151 complete
countries, 21 design columns including the intercept, rank 20. That legacy
identity is documented rather than silently carried into the future V2 design.

**Rank choice is separate from measurement.** The rank audit compares minimal
full-rank representations of M0's full-model column space. Its recommended V2
primary retains cumulative CO₂ total and population (drops cumulative CO₂ per
capita); the registered sensitivity retains cumulative CO₂ per capita and
population (drops cumulative CO₂ total). **This recommendation is pending owner
approval; this specification does not make it final.** Use the approved contract
for both the rank-only M0 comparator (legacy geography) and M1a (remeasured
geography). Compare those two to isolate measurement. Preserve legacy M0 as a
historical row and report the rank-only comparison separately: equal full-model
column space does not imply equal subgroup coalition spaces or LMG allocations.
Do not attribute a rank-contract change in group shares to geography measurement.
The [rank audit](rank_audit.py) and its [registered results](outputs/rank_audit.json)
provide the controlled representation comparison.

Keep unchanged:

- `warming_trend = trend_c_per_decade_area_weighted`, Berkeley Earth monthly
  TAVG, 1950-01–2013-09, per-cell Theil–Sen slopes, the existing 90% temporal
  coverage gate and cos-latitude country aggregation on the 1° outcome grid.
- The existing `Country`–OWID–ISO3 bridge, exactly 151 primary comparison IDs,
  outcome values and values of all retained non-geography features. Feature
  inclusion follows the approved rank contract above; income and station-density
  definitions are unchanged.
  Geography area weights operate **within** a country; regression and LMG
  remain unweighted across countries (one country, one outcome).
- Linear numeric features; existing categorical fixed-effect blocks; log10
  only for retained members of cumulative CO₂ per capita, cumulative CO₂ total
  and population;
  four named groups plus `1 − R²` residual. No interactions, splines or
  country-area weights in the regression.

The owner already approved the **existing GPW/ISO analytical units before
corrected CV scores**, documented in [TERRITORIAL_CV_CORRECTION.md](TERRITORIAL_CV_CORRECTION.md).
Use GPW v4 rev11 national-identifier band 11, the same lookup and country bridge.
A unit contains every assigned 0.25° footprint, including disconnected islands,
enclaves and exclaves. FRA excludes separately coded GUF; do not merge
territories by sovereignty or substitute a new political boundary dataset.
This is an analytical ontology, not a political-status assertion.

Area-consistent predictors describe terrestrial area inside these footprints,
not just the Berkeley 1° cells with estimable outcomes, stations, inhabited
cells or a country's largest polygon. This aligns unit and weighting intent;
it does **not** claim that the finer predictor support is identical to the
coarse outcome support. That remaining resolution mismatch must be reported.

## 2. Verified local evidence and its limits

The following were checked by reading existing outputs and code, without
refitting. [m0_countries.csv](outputs/m0_countries.csv) has 151 unique country
rows, including the frozen inputs and diagnostic descriptors;
[country_geometry.csv](outputs/country_geometry.csv) supplies the area summaries.
Values printed in CSV are rounded; the diagnostic JSON retains more precision.

| Claim | Verified definition and value | What it does not establish |
|---|---|---|
| 26 Köppen differences | `climate_zone != koppen_area_dominant`: 26/151. Station modal A–E vs the diagnostic area-dominant A–E on GPW cells with a valid class. | Not 26 independently verified station-siting errors. The comparison also changes spatial support, raster sampling and missing-class treatment. |
| 17 negative country-mean elevations | Exactly 17/151 have **negative country-mean V1 `elevation`**. Jamaica −1527.6362 m; Dominican Republic −1174.1370 m; Lebanon −872.99272 m; Australia −465.43547 m. | This count alone is not an independently masked count of countries with any ocean samples. Positive means can hide bathymetry; negative values can also be real terrestrial depressions. Large negative island means and grid-snapped sampling support the artifact diagnosis, not a sign-based land rule. |
| Canada approximately 14° | V1 mean station `abs_latitude` 46.7736°; diagnostic absolute spherical land centroid latitude 60.9608°; gap 14.1872°. | The centroid latitude is not mean absolute land latitude, the proposed feature. This is a diagnostic displacement, not M1a's already-computed correction. |
| Approximately 71/29 between/within | All fitted cells: 16,424 in 181 countries, between 70.712902%, within 29.287098%. M0 subset: 15,479 in 151, between 71.279341%, within 28.720659%. | These are area-weighted cell-trend variance partitions, not the regression's equal-country explained/residual shares. |

The 26 disagreements are Algeria, Angola, Burkina Faso, Chad, Ethiopia, Kenya,
Mali, Morocco, South Africa, Swaziland, Zimbabwe, Azerbaijan, China, Iraq,
Jordan, Tajikistan, Croatia, Germany, Guatemala, Mexico, United States,
Australia, Bolivia, Brazil, Chile and Peru. The 17 negative country means are
Gabon, Libya, Madagascar, Senegal, Lebanon, Philippines, Sri Lanka, Taiwan,
Albania, Cyprus, Greece, Cuba, Dominican Republic, Haiti, Jamaica, Australia
and New Zealand.

The variance partition is from
[m0_residual_diagnostics.json](outputs/m0_residual_diagnostics.json),
`collapse_information_loss`: for cell slopes `t_ic` and area weights `a_ic`,
`Var_a(t) = Var_country-area(mean_a(t|c)) + mean_country-area(Var_a(t|c))`.
Country weights here are summed fitted-cell areas. **The approximately 29%
within-country component is distinct from M0's 36.355400% regression residual**,
which is `1 − 0.6364460028` over equally weighted national outcomes. They cannot
be added or subtracted as nested shares. M1a does not recover discarded
within-country variation, seasonal behavior or temporal curvature.

[geometry.py](geometry.py) is diagnostic, not a ready M1a feature builder:
it uses GPW centres for Köppen, classified-area renormalization without a hard
coverage gate, a spherical vector centroid, and ETOPO every fifth sample.
Its elevation filter `z >= 0` removes both bathymetry and genuinely below-sea-level
land. **Do not promote those elevation summaries directly into M1a.**

## 3. Common spatial operator, sources and quality gates

### 3.1 Land mask independent of elevation sign

Use **GSHHG 2.3.7 full-resolution shoreline polygons**, pinned by release and
archive/member SHA-256 before construction, as a new *measurement support*
source. It is not a predictor. Its documented hierarchy distinguishes ocean
shorelines, lakes, islands in lakes and ponds. Terrestrial support is L1 minus
L2 plus L3 minus L4, intersected with each GPW unit. Thus ocean and inland-water
beds are excluded; below-sea-level dry land remains. Use the ice-front option
if Antarctic geometry is encountered; Antarctica is not added to the sample.
Record it explicitly in the manifest. The hierarchy and resolution options are
published by the [GSHHG maintainers](https://www.soest.hawaii.edu/pwessel/gshhg/).
Use the shapefile archive linked there and its full-resolution polygon members;
verify the release and member hashes before construction. The
[versioned GMT deposit](https://zenodo.org/records/7007502) independently records
version 2.3.7 and WGS84 but is not the proposed shapefile archive. Both primary
source pages were checked directly; no dataset was downloaded for this spec.

ETOPO surface data combine topography and bathymetry, as documented by
[NOAA NCEI](https://www.ncei.noaa.gov/products/etopo-global-relief-model).
“Independent” here means classification does not depend on the sign of ETOPO;
it does not assert statistical independence between source production chains.
Keep ETOPO ice-surface height (not bedrock). Neither ice presence nor snow is
materialized as a new feature.

The land mask only subdivides already assigned GPW footprints. It cannot assign
unrepresented islands, annex a separately coded territory or invent a national
code. Record GPW area, masked terrestrial area and removed ocean/inland-water
area separately. A large removed fraction is a coastline-resolution diagnostic,
not automatically missing data. A zero-area unit is invalid. Unassigned GPW
space is outside this analytical universe, not a nearest-country fill.

### 3.2 Area integration and topology

Define `D_c = union(GPW footprints with ISO3 c) ∩ terrestrial mask`.
Use the same spherical area convention as the existing outcome: latitude-longitude
integration with `dA = R² cos(phi) dphi dlambda`, `R=6371.0088 km`. On equal-angle
full cells this is proportional to the existing cos-centre-latitude weights.
Integrate clipped cell polygons using this measure; do not count pixels equally
or use degrees squared. WGS84 coordinates locate inputs; this spherical area
weighting is separate from the WGS84 distance bounds used for CV.

Use native raster cell footprints and explicit overlaps with `D_c`, retaining
fractional coast/border cells. For analytic latitude/hemisphere, integrate over
`D_c` directly. For distance quadrature use the existing 60 arc-second ETOPO
lattice as a computational grid only; clipped-cell representative points must
lie on terrestrial support (area centroid when inside, deterministic interior
point otherwise). Split longitude-crossing polygons at ±180°, wrap distances,
handle pole closures and preserve holes and every multipart component.
No mainland-only simplification; validate area conservation after seam splitting.

### 3.3 Coverage defaults fixed on methodological grounds

For feature `j`, store `coverage_cj = valid terrestrial area / area(D_c)`.
Use finite values and metadata nodata codes, not numeric sign. Coverage is
independent of outcome availability, predictor effect size and CV performance.
These thresholds are specification choices, not empirically validated guarantees:

- Numeric elevation and continentality: require **at least 98%** coverage.
  Means are sensitive to selective loss of extreme terrain/interiors, so cap
  unmeasured area at 2%. Renormalize only over valid area and report omitted
  area; this still cannot bound bias from arbitrarily extreme missing values.
- Latitude and hemisphere: require complete valid geometry (100% of `D_c`),
  since coordinates themselves have no raster nodata justification.
- Climate: require **at least 95%** classified terrestrial area **and** a robust
  winner. With `A1` and `A2` the leading classified areas and `U` unclassified
  area, require `A1 − A2 > U` for incomplete coverage. Missing area cannot then
  overturn the category. A 5% ceiling tolerates coarse coastal raster holes
  while the winner test protects the categorical decision. With full coverage,
  exact ties use alphabetic A–E order, reproducing the V1 mode convention.
- `spatial_block`: require one nonmissing existing OWID continent for every
  frozen country; no area coverage concept applies to this unchanged label.

Failed gates produce a missing feature plus an explicit reason, never zero,
station fallback, ocean-to-land nearest fill, label interpolation or relaxed
threshold. No threshold search after inspecting scores. Primary 151-country M1a
results are ineligible until all rows pass; a reduced-sample paired M0/M1a result,
if later authorized, must be separately named and recompute both on identical
IDs. Newly measurable excluded V1 countries are not silently appended.

## 4. Feature-by-feature executable contract

All row definitions inherit §3's land/ocean mask, coverage logging, all-island
and antimeridian treatment. “Measurement-only” below means the physical concept
and model feature identity persist; numerical values are deliberately changed.

### `abs_latitude` — mean absolute terrestrial latitude (degrees)

- **V1:** `add_latitude_features` takes `abs(Latitude)` at grid-snapped city
  locations; `aggregate_city_features_to_country` takes their unweighted mean.
  Source is the existing Berkeley-derived city-coordinate table, not an area
  raster. Missing station values are skipped by the mean.
- **M1a:** `integral_Dc |phi| dA / area(D_c)`, linear, using GPW footprints and
  independent terrestrial support. Absolute value is applied **before**
  averaging; neither `abs(mean(phi))` nor absolute vector-centroid latitude is
  equivalent for equator-crossing or broad countries.
- **Edges:** split integrals at the equator; retain both hemispheres and remote
  islands. Coordinates at ±180° do not alter latitude. Geometry gate is 100%.
- **Scope/leakage:** measurement-only; no latitude spread, polar fraction,
  interaction or inferred amplification feature. Geometry has no response
  information; do not choose support by warm/cold cells.

### `elevation` — mean terrestrial surface elevation (metres)

- **V1:** nearest static-grid ETOPO 2022 v1 **60 arc-second surface** `z`,
  sampled at approximately 1° grid-snapped station locations, then unweighted
  country mean. Input identity is `src.explain.ETOPO_PATH/ETOPO_URL`; negative
  bathymetric samples currently survive.
- **M1a:** retain that exact ETOPO version, resolution, surface variable and
  vertical reference. For every native source pixel, weight its piecewise
  constant value by `area(pixel ∩ D_c)`. A source pixel whose centre is classified
  as water by the independent mask is **not valid elevation support**, even if
  it overlaps land; report that land fraction as missing. This prevents an
  offshore height sample from being spread over a coastal fragment.
- **Edges:** retain negative `z` when the source centre is independently on land;
  no `z >= 0`, absolute value or clipping. Check terrestrial depressions and
  coastal negative extremes separately against mask geometry. A land-centred
  mixed pixel may retain source-resolution error; log this limitation and mixed
  fractions, rather than asserting the mask certifies every height. Islands
  with no land-centred ETOPO pixel fail coverage; no mainland substitution.
- **Scope/leakage:** measurement-only support repair, including increased
  integration fidelity over the legacy 5′ diagnostic sample. No roughness,
  elevation range, lapse-rate correction or height-dependent interaction.
  ETOPO is a later static map used descriptively, not a 1950 forecast input.
  Selection of suspect cells must not use residuals or fitted gains.

### `continentality` — mean distance to the existing coastline (km)

- **V1:** station mean `coast_km`; Natural Earth **110m means 1:110 million
  map scale, not 110-metre resolution** land geometry from
  `src.interpolate.LAND_ZIP_PATH`. `src.explain.coast_distance_km` selects a
  nearest boundary point in planar longitude/latitude and applies haversine
  distance to that point. This is not necessarily the globally closest
  geodesic boundary point.
- **M1a:** area mean of the minimum spherical great-circle distance to the same
  cached Natural Earth land-boundary geometry. Compute the global minimum over
  boundary segments with wrapped longitude, treating each segment as its shorter
  spherical arc; use radius 6371.0088 km. Average over §3.2 quadrature weights.
  This explicitly corrects the nearest-point metric as well as sampling support;
  retain a diagnostic comparison to the legacy planar-nearest algorithm to
  separate these two measurement effects. Boundary includes whatever holes are
  present in that frozen source; do not add a lake-distance predictor.
- **Mask/edges:** GSHHG determines which locations are land; Natural Earth
  remains the distance source, so differing shoreline resolution is disclosed.
  Some small islands are absent from Natural Earth and can have implausibly
  large distances. Report represented-island coverage and those cases without
  swapping sources country by country. Distances are to global coastline, not
  country borders; landlocked-country distance is valid. Preserve dateline
  segments and remote islands. Numeric gate is 98%.
- **Scope/leakage:** same maritime-distance concept; no coastal fraction,
  ocean-temperature or ocean-circulation feature. Resolve numerical nearest
  distance independently of warming or CV. GSHHG distance as a replacement would
  be a separately declared source sensitivity, not the primary definition.

### `climate_zone` — dominant terrestrial Köppen major group (A–E)

- **V1:** Beck et al. 2018 present-day 0.5° raster
  `Beck_KG_V1_present_0p5.tif`, converted to `koppen_present_0p5.nc` by
  `src.explain.prepare_koppen_grid`; nearest station sampling, codes 1–30 mapped
  by `KOPPEN_GROUPS` to A–E. Code 0/unmapped is missing; first sorted station
  mode wins ties. Country feature is one categorical variable.
- **M1a:** retain the exact cached source, version and code mapping. Sum
  `area(native climate pixel ∩ D_c)` by major group and choose the dominant
  group under §3.3. No bilinear interpolation of codes, nearest-land fill,
  reassignment of ocean code 0, or silent higher-resolution map substitution.
- **Edges:** equatorial countries, high-altitude E, coastal islands and multipart
  countries all use the same rule. Track valid area and winner/runner-up/missing
  areas; near-tied incomplete countries fail rather than receive a guessed class.
- **Scope/leakage:** same categorical climate-regime concept. Shares A–E are
  permitted in a **QA table only**, never as five numeric model columns. No
  diversity, entropy, subtype, new aridity variable or changing categories to
  optimize fit. The existing contemporary classification uses climatic
  information overlapping the warming window; retain that inherited descriptive
  limitation. It is not an independent historical exposure or causal instrument.

### `hemisphere` — hemisphere containing most terrestrial area (N/S)

- **V1:** station `N` for latitude ≥0, `S` otherwise, then first-sorted mode.
- **M1a:** integrate terrestrial area north/south of the equator in `D_c`;
  choose N if north area ≥ south area, else S. The exact equator has zero area;
  numeric ties use N, matching V1 sorted-label behavior. Geometry gate is 100%.
- **Edges:** no centroid-sign shortcut for countries spanning the equator or
  multipart territories. Retain a near-tie diagnostic but no third “mixed” class.
- **Scope/leakage:** measurement-only categorical reassignment. No north-area
  fraction, latitude×hemisphere term or extra feature. This deterministic
  coordinate operation uses no outcome or fold labels.

### `spatial_block` — existing OWID continent (unchanged)

- **V1:** `inequality.continent` copied into `spatial_block`, then categorical
  dummy coding. It was never inferred from station locations or a raster.
- **M1a:** copy the same value, source and one-to-one country mapping byte for
  byte. There is no scientifically equivalent area-mean continent category
  that must be recomputed. Land/ocean weighting is not applicable.
- **Edges:** transcontinental countries retain OWID assignment, and islands
  remain attached to their frozen analytical unit. No centroid reassignment,
  sovereign aggregation, M49 replacement, spatial cluster refit or new level.
  Missing labels fail the frozen-row contract.
- **Scope/leakage:** unchanged; M49 is a validation partition, not an active
  geography predictor. Do not derive blocks from residuals or outcomes.

## 5. Features and derived quantities that do not enter M1a

`station_density` stays in the population group unchanged: mean station count
within 100 km, excluding self, at most 50 neighbours examined. It is a sampling
and concentration proxy whose meaning would change if averaged over country
land. Calling all spatially located data “geography” would silently change the
model. Non-geography values remain exactly as in M0; the separately approved
rank contract determines which redundant magnitude column is omitted in both
the comparator and M1a.

Schema-declared but inactive `co2_intensity_gdp`, `gdp_per_capita` and
`urbanization_rate` stay absent. Diagnostic `land_area_km2`, centroid,
`lat_sd_area_weighted`, coastal fraction, elevation spread/quantiles and Köppen
proportions are also absent from the model. No aridity index, snow, albedo,
soil moisture, aerosol, vegetation or other new physical concept; no ERA5
predictor or outcome replacement. A diagnostic table is not authorization to
expand the design.

## 6. Future implementation and validation contract

The subsequent implementation should emit a separately named research artifact,
never overwrite production predictors or legacy M0 outputs:

1. A manifest recording this spec version, source URLs/release names/file and
   lookup hashes, CRS, grid registration/orientation, nodata metadata, spherical
   area convention, shoreline boundary rule, mask hierarchy, algorithms and
   software versions. Freeze these before viewing any M1a score. Require the
   existing geometry snapshot/source hashes to match the corrected CV manifest.
2. One row per frozen ISO3 with exactly six replacement/retained geography
   columns plus IDs; a separate QA table with old/new values, difference or
   category switch, source and mask area, valid coverage, excluded-water area,
   genuine negative-land area, mixed-pixel area, missing reasons and category
   ambiguity. No QA field may enter the design matrix.
3. Geometry checks: FRA/GUF separation; a represented distant island/exclave;
   landlocked and equator-crossing countries; seam-crossing multipart geometry;
   conservation of full/partial cell area and classification hierarchy. Synthetic
   fixtures must show a negative land elevation retained and a negative ocean
   sample excluded, and distinguish mean absolute latitude from centroid latitude.
4. Source checks: axes and longitude convention, correct pixel edges, no
   interpolation of categorical classes, positive area, weight sums and
   reproducibility. Require deterministic country ordering and repeated-output
   equality. Compare distance quadrature at 60″ and 30″ using the same coastline:
   per-country difference ≤max(0.1 km, 0.1% of the 30″ estimate); use 30″ if needed
   and continue deterministic refinement until met. This tolerance controls
   numerical integration, not shoreline-source accuracy, and is not score-tuned.
5. Record the owner's approved rank contract and compare like with like. Assert
   equality of all 151 IDs, outcome and retained non-geography values between
   legacy M0, its approved full-rank comparator and M1a. Assert the same feature
   inclusion in the comparator and M1a and no unexpected feature identities.
   Categorical support may change naturally, so report encoded columns, present
   levels and effective rank; do not hard-code the legacy 21-column count if a
   magnitude is omitted or a category disappears. Record `spatial_block` equality
   and verify the selected rank contract removes the original redundant log
   direction; flag any new dependence introduced by category support.
6. Only after measurement QA passes, evaluate the paired full-rank M0/M1a comparison using
   the **corrected** territorial scorer. Primary is leave-one-country-out with
   training units retained only when their certified all-GPW-footprint WGS84
   lower bound is **strictly >500 km**. Equality/uncertainty excludes. Reuse
   frozen memberships; the finer measurement land mask must not shrink CV
   exclusion geometry. M49 stays secondary, random CV reference-only. Do not
   optimize threshold, folds or categorical handling against M1a scores.
7. Fit preprocessing/encoders and OLS on training rows only. Test categories
   unseen in training receive the existing **mean of training level effects**
   (including reference effect zero; unweighted across levels), not a new
   global-mean rule. Deterministic response-independent map aggregation can be
   shared across folds. No label imputation or support selection from test
   outcomes. Report changed/unseen category counts, effective rank, the existing
   full scorecard and paired country-level prediction errors.

The detailed distance proof, historical/corrected artifact distinction and
score definitions are in [TERRITORIAL_CV_CORRECTION.md](TERRITORIAL_CV_CORRECTION.md)
and [cv.py](cv.py). Existing corrected scores are M0 scores, not an M1a result.
Measurement source/threshold choices here must not be represented as having
been frozen before those already-observed M0 scores; the **territorial ontology
and CV correction** were frozen before corrected scores, while this spec is
being fixed before any M1a result.

## 7. Interpretation and remaining decisions

The **full-rank V2 feature contract** is the substantive owner decision still
required before fitting (§1); that choice fixes the allocation of overlapping
emissions/population information, not just numerical implementation. No new
owner decision is needed for the routine measurement definitions in this
specification. The territorial ontology was already approved. Implementation
or fitting is a separate future task, not performed by this document. If the
chosen sources cannot satisfy a frozen country's coverage, the concrete failure
will require a source-resolution or explicitly labelled sample-sensitivity
choice; the implementation must show the affected IDs/areas before requesting
that choice, rather than create a speculative approval gate now.

A better paired score would support the practical value of area-consistent
measurement in this observational decomposition. It would not uniquely identify
station siting as the original cause: aggregation support, coarse coordinate
snapping, coastline detail, coverage, source vintage and product construction
also differ. Historical station-versus-area warming sensitivity likewise does
not isolate station placement. Group LMG shares are associational allocation of
cross-country variance, not causal attribution. The measurement correction
cannot by itself distinguish regional physics, product error or internal
variability, or turn a national regression into an explanation of within-country
inequality.
