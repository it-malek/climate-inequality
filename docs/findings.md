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

Country warming is the unweighted mean of its city-location slopes (157
countries match the OWID emissions table; Puerto Rico and Réunion do not).

| Metric | Value |
|---|---|
| Mean country warming | 0.161 °C/decade |
| 5th–95th percentile | 0.086–0.254 °C/decade |
| Gini | 0.175 |
| Theil-T | 0.050, of which 18% is between continents |
| Coefficient of variation | 0.31 |

A group-level Shapley/LMG decomposition attributes the variance of country
warming to four fixed feature groups over the 154 complete cases:

| Axis | Features | Standalone R² | Shapley share | Share of explained |
|---|---|:-:|:-:|:-:|
| Geography | abs. latitude, elevation, coast distance, Köppen class, hemisphere, continent | 0.580 | **0.455** | 72% |
| Emissions | log cumulative CO₂ per capita, log cumulative CO₂ total | 0.132 | 0.084 | 13% |
| Socioeconomic | World Bank income group | 0.080 | 0.056 | 9% |
| Population | log population, station density | 0.049 | 0.036 | 6% |
| Residual | (1 − R²) | | **0.369** | |

Total R² is 0.631. The gap between the emissions axis's standalone R² (0.13)
and its Shapley share (0.08) is the variance it shares with geography:
historically high-emitting countries sit at the mid-to-high northern latitudes
where warming is fastest.

The same fragility shows in the older single-coefficient framing. Regressing
country warming on log₁₀ cumulative per-capita CO₂ gives +0.021 °C/decade per
tenfold increase pooled, and +0.029 (95% CI +0.014 to +0.045) with continent
fixed effects; adding each country's mean absolute latitude within continents
cuts it to +0.012 (CI −0.005 to +0.029). Adding income group on top brings it
back to +0.026 (p = 0.03). The coefficient depends on which correlated controls
enter, which is why the decomposition reports shares rather than a coefficient.

Stability ([`stability.md`](stability.md)): over 2,000 country-bootstrap
resamples geography is the largest axis every time (95% interval 0.39–0.56);
the emissions share stays positive (0.05–0.13). Moran's I on the full-model
residual is 0.33 (permutation p = 0.005, k = 8 nearest country centroids): the
unexplained 37% is regionally clustered rather than noise.

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
coefficients are only partly identified (the N₂O interval spans −0.1 to 6.6).
The model describes the global trajectory and is kept apart from the
cross-country analysis: a global signal has no cross-country variance to
explain.

## What these results do not show

- No causal effect of a country's emissions on its own warming.
- No policy counterfactual.
- Nothing about heat extremes, exposure-days or damages; the outcome throughout
  is the mean warming rate.
- Nothing within countries: every country is one unit.
