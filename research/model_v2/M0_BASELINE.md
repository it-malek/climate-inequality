# M0: the frozen V1 model

M0 is the decomposition shipped at `v1.3.0` (commit `a6733cb`). This page records
its values, verifies them, and describes the model as it is actually implemented,
so that every later stage is compared with the same object. Nothing here is a
proposal; the proposals are in the hypothesis register and the plan.

## 1. Values (bundle-authoritative) and verification

| Quantity | Bundle (`v1.3.0`) | Refit from local artifacts | Difference |
|---|---|---|---|
| n | 151 | 151 | 0 |
| In-sample R² | 0.6364460028 | 0.6364460028 | 5e-11 |
| Residual share | 0.3635539972 | 0.3635539972 | 5e-11 |
| Geography share | 0.5130328657 | 0.5130328657 | 8e-13 |
| Emissions share | 0.02139283879 | 0.02139283879 | 5e-12 |
| Socioeconomic share | 0.06932302426 | 0.06932302426 | 4e-12 |
| Population share | 0.032697274 | 0.032697274 | 3e-12 |
| Residual Moran's I | 0.2686890161 | 0.2686890161 | 4e-11 |
| Moran's I permutation p | 0.005 | 0.005 | 0 |

Verification code: `research/model_v2/m0.py::verify_against_bundle`
(`outputs/m0_baseline.json`). The refit uses the V1 functions themselves
(`src.decomposition.build_country_design`, `feature_block`, the same complete-case
rule and least-squares solve), so the residuals inspected in this line are the
residuals of the tagged model. The bundle's `country_inequality.parquet` is
byte-identical to the local `data/processed` copy; the city-feature and income
inputs are the ones the bundle was built from (the committed bundle was built on
this machine, which is also why Moran's I reproduces beyond the ~1e-4
cross-platform tie-breaking drift documented in `docs/reproducibility.md`).

Companion values (for context, not part of the M0 row): country-bootstrap 95%
intervals geography 0.447–0.623, emissions 0.009–0.054, residual 0.235–0.407;
continent block bootstrap geography 0.384–0.577, emissions 0.011–0.197;
P(geography largest) = 1.00; station-weighted sensitivity on the same 151
countries: geography 0.458, emissions 0.085, R² 0.635, Moran's I 0.314.

## 2. Outcome

`warming_trend` := `trend_c_per_decade_area_weighted` (°C/decade). For every
1°×1° Berkeley Earth land cell (`Complete_TAVG_LatLong1.nc`, monthly TAVG
anomalies), a Theil–Sen slope over 1950-01 to 2013-09 (765 months; cells with at
least 90% finite months), on a decimal-decade axis. Cells are assigned to
countries by sampling band 11 of the GPW v4.11 national-identifier grid (15
arc-minutes) at the 1° cell centre and mapping the code to ISO3; each country's
cell slopes are averaged with cos(latitude) weights. 16,424 cells are fitted for
181 countries; the 151 M0 countries own 15,479 of them. The operator is the same
as the station pipeline's (`src.area_weighting`); only the weighting differs.

Sample statistics over the 151 countries: mean 0.176, sd 0.0467, minimum 0.041
(Mexico), maximum 0.277 (Uzbekistan) °C/decade.

## 3. Countries

151 = the 157 Berkeley→OWID name-matched countries minus three with no 1° land
cell resolving to them (Bahrain, Hong Kong, Singapore) minus three with a
missing Köppen class (Gambia, Guyana, Mauritius: every grid-snapped station
coordinate falls in an ocean cell of the 0.5° Köppen grid). Puerto Rico and
Réunion were dropped earlier at the OWID join. Continents (OWID): Africa 47,
Asia 38, Europe 38, North America 14, South America 11, Oceania 3.

## 4. Predictors, transforms and groups (`SCHEMA_V1`)

| Group | Feature | Construction in V1 | Enters as |
|---|---|---|---|
| emissions | `cum_co2_per_capita` | OWID production-based cumulative CO₂ through 2013 (from record start) ÷ 2013 population, t/person | log10 |
| emissions | `cum_co2_total` | the same cumulative, Mt | log10 |
| geography | `abs_latitude` | mean of the country's city-locations' \|latitude\| | linear |
| geography | `elevation` | mean ETOPO 2022 surface value sampled at the grid-snapped station coordinates, m | linear |
| geography | `continentality` | mean great-circle distance from the stations to the Natural Earth 110 m land boundary, km | linear |
| geography | `climate_zone` | modal Köppen–Geiger major group (A–E) over the stations, Beck et al. 0.5° | 4 dummies (A reference) |
| geography | `hemisphere` | modal N/S over the stations | 1 dummy (N reference) |
| geography | `spatial_block` | OWID continent | 5 dummies (Africa reference) |
| socioeconomic | `income_group` | World Bank classification (vendored table) | 3 dummies (High reference) |
| population | `population` | OWID 2013 population | log10 |
| population | `station_density` | mean over the stations of the number of other city-locations within 100 km (at most 50 examined) | linear |

Proposed-but-unwired schema features (`co2_intensity_gdp`, `gdp_per_capita`,
`urbanization_rate`) are absent. Design: 21 columns including the intercept.

**The design has rank 20, not 21.** Per-capita cumulative CO₂ is the cumulative
total (Mt × 10⁶) over the same 2013 population that is the `population`
feature, so after the logs `log10(per capita) − log10(total) + log10(population)
= 6` holds for every country (maximum deviation 1.8e-15;
`outputs/m0_baseline.json` and `m0.py::collinearity_report`). Consequences: the
least-norm solve V1 uses returns the projection R², so every V1 number is
unaffected; individual coefficients of those three columns are not identified;
and the emissions and population groups share one exact linear direction, so
the Shapley averaging splits that direction between them (the population
group's share partly credits information the emissions group carries and vice
versa). This is a documented property of M0, not a defect to be fixed in this
line; a later schema revision could replace `cum_co2_total` or `population`, but
that is a V2 decision, not a V1 correction.

## 5. Estimator

Ordinary least squares with an intercept (`numpy.linalg.lstsq`, least-norm),
unweighted: every country is one unit regardless of area or population. R² of
the full model, 0.636, equals the full-coalition R² of the decomposition
(checked to 1e-12).

## 6. Shapley/LMG contributions

Group-level LMG (Shapley–Owen): with G = 4 named groups the 2⁴ = 16 coalition
models are fitted and each group's share is its incremental R² averaged over
orderings with weight |S|!(G−|S|−1)!/G!. Shares plus 1 − R² sum to one.
Categorical blocks are attributed to their group as a unit. Standalone
(univariate) group R²: geography 0.597, socioeconomic 0.096, population 0.060,
emissions 0.003.

## 7. Stability layer (V1)

Country bootstrap: 2,000 resamples with replacement (seed 0), percentile 95%
intervals of every share; continent block bootstrap: 2,000 resamples of whole
`spatial_block` groups (seed 1); leave-one-country-out shifts of each share (top
ten per group); Moran's I of the full-model residual. A resample that makes the
design unusable is skipped and counted (none was).

## 8. Moran's I and the neighbour definition

`src.explain.morans_i`: each country's location is the **mean of its station
coordinates** (not its land-area centroid); neighbours are the k = 8 nearest
countries by great-circle distance (KD-tree, ties broken by the library, hence
the ~1e-4 cross-platform drift); weights row-standardised (1/8 each);
I = Σ z_i·mean(z_neighbours) / Σ z²; two-sided permutation p from 199
permutations (seed 0). The dense re-implementation in `spatial.py` breaks ties
by index and reproduces 0.2686890161 on this machine.

## 9. What collapsing the gridded field to one country trend discards

Quantified in `outputs/m0_residual_diagnostics.json` (`collapse_information_loss`)
and `outputs/country_cell_trend_stats.csv`:

* **Within-country spatial variance.** Of the cos-latitude-weighted variance of
  the 16,424 fitted 1° cell trends, 71% lies between countries and 29% within
  them (the 151 M0 countries: 71% / 29%). The median within-country sd of cell
  trends is 0.008 °C/decade, the 90th percentile 0.019, the maximum 0.063
  (Canada); seven countries (Libya, China, Saudi Arabia, Russia, Canada, Mexico,
  United States) have within-country sd larger than the M0 residual sd (0.028).
* **The time path.** One linear slope per cell over 64 years; acceleration,
  the mid-century NH cooling and decadal variability are folded into the slope.
* **The seasonal cycle.** Annual-mean anomalies only; no seasonal trends.
* **Trend uncertainty.** Sen intervals per cell are not propagated to the
  country value.
* **Predictor–outcome mismatch.** The outcome describes the land area; the
  geography predictors describe where the stations are. Median \|station-mean
  latitude − land-area centroid latitude\| is 0.7°, but 14.2° for Canada, 12.6°
  for Russia, 8.7° for the United States and 8.6° for Brazil. 26 of 151
  countries have a modal station Köppen class that differs from the
  area-dominant class (Algeria: station-modal C, 96% of land B; Mali A vs 88% B;
  Chad A vs 85% B; Jordan C vs 95% B; Iraq C vs 88% B; Brazil C vs 85% A;
  Mexico C vs 54% B). 17 countries have a negative "elevation" because their
  grid-snapped station coordinates sample ETOPO bathymetry (Jamaica −1,528 m,
  Dominican Republic −1,174 m, Lebanon −873 m, Gabon −641 m, Cuba −602 m,
  Australia −465 m).
* **Product structure.** The ERA5 area trend (bundle cross-check) disagrees
  with the Berkeley outcome by a country sd of 0.063 °C/decade, larger than the
  outcome's own sd (0.047); rank agreement ρ = 0.64.

## 10. Substantive scientific choices versus implementation conventions

Substantive (changing them changes what is measured):

* outcome = area-weighted land trend, Berkeley Earth product, 1950–2013 window,
  Theil–Sen linear slope, annual mean;
* four named groups plus residual, and the membership of each group
  (`station_density` inside population; continent inside geography);
* emissions as a responsibility axis measured by cumulative production CO₂
  through 2013 per 2013 resident;
* one country = one unit, unweighted, complete cases only;
* log10 on the three magnitude features;
* linear, additive, non-spatial functional form.

Conventions and implementation choices (a different choice would be the same
science, though some carry measurable artefacts):

* geography features aggregated from stations (means; modal classes) rather
  than measured over the land area — with the bathymetric elevations, the Köppen
  mismatches and the latitude displacement listed above;
* drop-first dummy coding and the least-norm solve (no effect on R² or shares);
* the station-mean location and k = 8 for Moran's I; 199 permutations; seeds;
* the station-density definition (100 km, at most 50 neighbours);
* the Köppen 30→5 collapse; the OWID continent table as the spatial block;
* JSON rounding to 10 significant figures.
