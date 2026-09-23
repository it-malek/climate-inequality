# Findings

Every number here is read from the committed `app/data/` bundle and is
reproduced by the pipeline described in [`reproducibility.md`](reproducibility.md).
All results are descriptive: they measure how observed warming aligns with
country attributes, never a causal effect of a country's emissions on its own
climate.

## 1. Warming at the station level

3,510 Berkeley Earth city-locations with at least 90% monthly coverage over
1950-01 to 2013-09, each fit with a Theil–Sen slope on anomalies relative to
its own 1951–1980 monthly climatology.

| Statistic | Value |
|---|---|
| Land mean trend | 0.146 °C/decade |
| Interquartile range | 0.11–0.18 °C/decade |
| 5th / 95th percentile | 0.06 / 0.25 °C/decade |
| Locations with the whole 95% CI above zero | 99.1% |
| Mean above 60°N (25 locations) | 0.228 °C/decade (1.56× the land mean) |

The ratio to the land mean is lower than the textbook Arctic amplification of
about 2× because that figure is measured against the land-plus-ocean global
mean, and because only 25 stations sit above 60°N. The fastest-warming cluster
is the Iranian plateau and Central Asia (Mashhad, Herat, Ashgabat, up to
0.34 °C/decade), alongside Norilsk in Siberia (0.33). The slowest warming is on
the southern coast of China (0.02–0.03 °C/decade).

Interpolation: inverse-distance weighting beats local ordinary kriging under
leave-location-out cross-validation (RMSE 0.0083 vs 0.0099 °C/decade). Plain
leave-one-row-out CV reverses the ranking (0.0057 vs 0.0160) because 2,821 of
the 3,510 locations share grid-snapped coordinates with another city, and a
held-out point's coordinate twin leaks its value into the fold. Holding out the
whole coordinate group removes the leak.

## 2. Country-level inequality and its decomposition

For the inequality metrics, country warming is the unweighted mean of its
city-location slopes (157 countries match the OWID emissions table; Puerto
Rico and Réunion do not).

| Metric | Value |
|---|---|
| Mean country warming | 0.161 °C/decade |
| 5th–95th percentile | 0.086–0.254 °C/decade |
| Gini | 0.175 |
| Theil-T | 0.050, of which 18% is between continents |
| Coefficient of variation | 0.31 |

A group-level Shapley/LMG decomposition attributes the cross-country variance
of warming to four fixed feature groups. The outcome is the **area-weighted**
land-warming rate: a Theil–Sen trend per Berkeley Earth 1° grid cell,
averaged over each country's land with cos-latitude weights (§3 describes
the construction). Its national aggregation is set by land area rather than
directly by where the stations are, which makes the country-level result
less sensitive to station-network geography; it remains an observational
product, since Berkeley Earth's field is interpolated from the same station
record. 151 countries have both an area-weighted value and complete features
(Bahrain, Hong Kong and Singapore have no 1° land cell).

| Group | Features | Standalone R² | Shapley share | 95% country bootstrap | Share of explained |
|---|---|:-:|:-:|:-:|:-:|
| Geography | abs. latitude, elevation, coast distance, Köppen class, hemisphere, continent | 0.597 | **0.519** | 0.452–0.630 | 82% |
| Socioeconomic | World Bank income group | 0.096 | 0.063 | 0.036–0.126 | 10% |
| Population | log population, station density | 0.060 | 0.042 | 0.021–0.090 | 7% |
| Historical responsibility | log cumulative CO₂ | 0.001 | 0.013 | 0.006–0.035 | 2% |
| Residual | (1 − R²) | | **0.364** | 0.233–0.407 | |

Total R² is 0.636. Cross-country differences in land-warming rates are
associated primarily with geography rather than with countries' historical
emissions responsibility, socioeconomic characteristics or population
characteristics: the geography group contributes 0.52 of total variance, or
roughly 82% of the variance the model explains, and is the largest group in
every one of 2,000 country-bootstrap resamples (continent-block bootstrap
interval 0.385–0.594), while the responsibility group contributes very little
independent explanatory variance (0.013 of total) and has almost no standalone
association with the outcome. About 36% of the variance lies outside the
model. This is a descriptive variance decomposition of cross-country
differences, not a physical attribution: greenhouse gases remain the driver of
global warming, and the question here is why warming rates differ spatially
among countries.

The shares above use the full-rank representation of the model, in which the
responsibility group holds log cumulative total CO₂ and the population group
holds log population and station density. Cumulative CO₂ per capita is the
total divided by that same population, so keeping all three columns (the
representation of `decomposition_summary.json`) leaves the fitted model,
its R² and its residual unchanged to numerical precision but makes the
allocation between those two groups representation-dependent: geography
0.513, responsibility 0.021, socioeconomic 0.069, population 0.033. The
representation that keeps per-capita CO₂ and drops the total gives geography
0.514 and responsibility 0.020. The conclusion is the same under all three
(`research/model_v2/VALIDATION_PROTOCOL.md` §6).

**Outcome sensitivity.** The same decomposition on the same 151 countries
with the station-weighted outcome (the unweighted mean of each country's
city-location slopes) isolates the effect of the outcome construction. This
comparison uses the original three-column representation on both sides, so
its area-weighted column is the 0.513 / 0.021 allocation rather than the
full-rank headline. The
last column is the station-weighted decomposition on all 154 countries that
have complete features, the sample the station outcome supports.

| Share of total variance | Area-weighted (primary), n = 151 | Station-weighted, same 151 | Station-weighted, all 154 |
|---|:-:|:-:|:-:|
| Geography | 0.513 | 0.458 | 0.455 |
| Emissions | 0.021 | 0.085 | 0.084 |
| Socioeconomic | 0.069 | 0.062 | 0.056 |
| Population | 0.033 | 0.029 | 0.036 |
| Residual | 0.364 | 0.365 | 0.369 |
| Total R² | 0.636 | 0.635 | 0.631 |
| Emissions standalone R² | 0.003 | 0.142 | 0.132 |

Geography is the dominant group and the explained variance is 63–64% under
both constructions; the socioeconomic and population shares stay small. What
depends on the construction is the emissions group: under the station-weighted
outcome it has a standalone R² of 0.14 and a share of 0.085 (13% of
explained), under the area-weighted outcome 0.003 and 0.021. The two outcomes
agree on country ranking only at ρ = 0.81; the countries that warm faster
under area weighting are those whose stations sit in slower-warming
populated fringes (the Sahel, Central Africa, Canada, the Philippines), and
those that warm more slowly are ones whose few stations sit in fast-warming
interiors (Central Asia, Central America, the Gulf). This pattern, together
with the coupling and ERA5 results in §3, is consistent with station-network
effects. The comparison also changes the observational representation (a
gridded field against station means), so it does not isolate station siting
as the sole cause of every difference.

The older single-coefficient framing shows the same fragility. Regressing
the station-weighted country warming on log₁₀ cumulative per-capita CO₂
gives +0.021 °C/decade per tenfold increase pooled, and +0.029 (95% CI +0.014
to +0.045) with continent fixed effects; adding each country's mean absolute
latitude within continents cuts it to +0.012 (CI −0.005 to +0.029). Adding
income group on top brings it back to +0.026 (p = 0.03). The coefficient
depends on which correlated controls enter, which is why the decomposition
reports shares rather than a coefficient.

Stability ([`stability.md`](stability.md)): over 2,000 country-bootstrap
resamples geography is the largest group every time (95% interval 0.45–0.62
in the original representation; 0.40–0.57 under the station-weighted
outcome). The responsibility share's interval is 0.01–0.05 area-weighted
against 0.05–0.13 station-weighted; the two touch only at the boundary.
Moran's I on the full-model residual is 0.27 (permutation p = 0.005, k = 8
nearest country centroids; 0.31 station-weighted): the unexplained 36% is
regionally clustered rather than noise. Section 7 follows that residual.

## 3. Responsibility versus warming exposure

The coupling comparator ranks countries by cumulative production-based CO₂ per
2013 resident (responsibility) and by warming rate (impact), and summarises
their relationship by Spearman ρ and by a Lorenz-style inequality coefficient
(twice the area between the cumulative-impact-vs-cumulative-responsibility
curve and the diagonal; 0 means aligned, 1 maximally divergent).

| Definition of country warming | n | Spearman ρ vs responsibility | p | Inequality coefficient | Rank ρ vs station |
|---|:-:|:-:|:-:|:-:|:-:|
| Station-weighted (mean over city-locations) | 157 | +0.359 | 4×10⁻⁶ | 0.563 | |
| People-weighted (GPW v4 population at each station) | 157 | +0.342 | | 0.566 | 0.988 |
| Area-weighted (per-cell trend on the Berkeley 1° grid, cos-latitude) | 154 | **+0.011** | 0.90 | 0.607 | 0.807 |
| Area-weighted, ERA5 reanalysis (common 153-country set) | 153 | +0.118 | 0.15 | 0.603 | 0.639 |

People-weighting barely changes anything because people live near stations.
Area-weighting changes a great deal: the significant station-based coupling
disappears under Berkeley Earth and is weak and non-significant under ERA5,
while the inequality coefficient rises. The world land mean of the
area-weighted Berkeley trend is 0.193 °C/decade, matching Berkeley Earth's
published global-land figure of about 0.19, against a station mean of 0.146:
cities cluster in slower-warming temperate mid-latitudes and under-sample the
fast-warming Sahel, Sahara and Central Asian interiors. Details of the ERA5
comparison, including the caveat that the two gridded products agree on
country ranking only at ρ = 0.62, are in
[`era5_validation_results.md`](era5_validation_results.md).

Consumption-based accounting (OWID `consumption_co2`, available from 1990 for
114 countries, with the production cumulative summed over the same window for
comparability) reorders responsibility only slightly (rank ρ 0.97 between the
two accountings).

## 4. Income and vulnerability

Stratifying the area-weighted warming and responsibility columns by World Bank
income group and by the ND-GAIN vulnerability score
([`vulnerability_results.md`](vulnerability_results.md)):

| Gradient | Spearman ρ | Permutation p |
|---|:-:|:-:|
| Responsibility vs income rank (157 countries) | +0.885 | 0.001 |
| Area-weighted warming vs income rank (154) | −0.145 | 0.06 |
| Responsibility vs ND-GAIN vulnerability (155) | −0.882 | 0.001 |
| Area-weighted warming vs ND-GAIN vulnerability (153) | +0.019 | 0.80 |

Mean cumulative CO₂ per capita rises from 7 t (low income) to 493 t (high
income), and falls from 537 t in the least vulnerable ND-GAIN quartile to 8 t
in the most vulnerable. Population-weighted area warming by income tier is
0.205 (low), 0.146 (lower-middle), 0.168 (upper-middle) and 0.182 (high)
°C/decade. Under the station-weighted definition, warming instead rises with
income (ρ = +0.17, p = 0.03), one more face of the station-siting effect.

## 5. Out-of-sample check of the trends

The stored 1950–2013 lines were scored against the Berkeley Earth 1° gridded
product through December 2024 (135 forecast months). 3,347 of the 3,510
locations pass the agreement gate (Pearson r ≥ 0.80 between the city series and
its grid cell over the fit window; the failures are mostly islands and coasts
where a 1° land average is not the city's series).

| Statistic | Value |
|---|---|
| Mean forecast residual, observed − predicted | +0.48 °C (+0.41 excluding 2023–24) |
| Mean stored slope (fit window) | 0.147 °C/decade |
| Grid slope over the same window | 0.159 °C/decade |
| Grid slope over the full record | 0.200 °C/decade |

Warming after 2013 ran ahead of the fitted lines; the trends are backcasts of a
period slower than the present.

## 6. Global temperature and radiative forcing

A separate regression of the annual global mean temperature anomaly (GISTEMP,
1951–2024) on lagged effective radiative forcings and the ENSO index, with an
AR(1) error, trained through 2013 ([`physical_model.md`](physical_model.md)):

| Statistic | Value |
|---|---|
| Train R² (1951–2013) | 0.91 |
| Test RMSE (2014–2024) | 0.10 °C |
| 95% band coverage on the 11 test years | 91% |
| AR(1) ρ | −0.03 |
| CO₂ sensitivity | 0.37 °C per W/m² (95% interval 0.06–0.68) |
| ONI sensitivity | 0.064 °C per index unit (0.03–0.10) |

The greenhouse-gas forcings are collinear, so the individual CO₂, CH₄ and N₂O
coefficients are only partly identified; the N₂O point estimate (3.2 °C per
W/m², interval −0.1 to 6.6) is an artefact of that collinearity and carries no
physical meaning on its own. The volcanic and ENSO terms are the tightly
estimated ones.
The model describes the global trajectory and is kept apart from the
cross-country analysis: a global signal has no cross-country variance to
explain.

## 7. The residual and its investigation

The 36% of cross-country variance outside the static model was investigated
under geographically separated validation: leave-one-country-out with every
country whose territory lies within 500 km of the held-out country removed
from training, so that no prediction borrows from a neighbour. Full detail is
in [`research/model_v2/RESULTS.md`](../research/model_v2/RESULTS.md); the
protocol and decision rules are in
[`research/model_v2/VALIDATION_PROTOCOL.md`](../research/model_v2/VALIDATION_PROTOCOL.md).

| Model | Out-of-fold RMSE (°C/decade) | Out-of-fold Moran's I | Paired change vs static [95% interval] | Verdict |
|---|:-:|:-:|:-:|---|
| Static model (in-sample R² 0.636; out-of-fold R² 0.153) | 0.0429 | 0.32 | | comparator |
| Geography re-measured over land area | 0.0452 | 0.34 | +0.0023 [−0.0003, +0.0053] | not adopted; kept as measurement sensitivity |
| Plus baseline hydroclimatic dryness (1920–1949 P/PET) | 0.0446 | 0.33 | +0.0017 [+0.0009, +0.0028] | not supported; coefficient wrong-signed |
| Plus flexible latitude form (natural spline, hemisphere slope) | 0.0399 | 0.34 | −0.0030 [−0.0059, +0.0001] | not supported; interval includes zero |
| Spatial error model on the static specification | 0.0398 | 0.26 | −0.0031 [−0.0059, −0.0002] | improves transfer |
| Spatial lag model on the static specification | 0.0381 | 0.26 | −0.0048 [−0.0079, −0.0015] | improves transfer |

The two spatial models both meet the same conditions and neither is declared
the single final model. In sample they represent 39% and 40% of the static
residual as neighbour-conditional dependence; that is represented spatial
covariance, not a fraction of warming, not a causal share, and not additive
with the geography share. Their predictive advantage holds under a
land-centroid neighbour graph and wider exclusion distances (1000 km
territorial, 1500 km centroid) for the spatial lag model, is weaker for the
spatial error model, and does not reproduce when the outcome is taken from
ERA5 under either construction (anomaly-aligned: spatial error +0.0015
[−0.0011, +0.0041], spatial lag +0.0001 [−0.0016, +0.0018]). The static
decomposition conclusion survives both ERA5 constructions (geography share
0.41 and 0.44; responsibility 0.008 and 0.015). The remaining variance is not
an unknown climate cause; it is variance in cross-country warming differences
that the tested country-level models do not capture with these observations.

## What these results do not show

- No causal effect of a country's emissions on its own warming.
- No policy counterfactual.
- Nothing about heat extremes, exposure-days or damages; the outcome throughout
  is the mean warming rate.
- Nothing within countries: every country is one unit.
- No recovery of the residual by the static extensions that were tested, and
  no mechanism behind the spatial dependence that the spatial models represent.
