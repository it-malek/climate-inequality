# Climate inequality

How is cross-country variation in observed land-warming rates structured among
physical geography, historical emissions responsibility, socioeconomic
development and population characteristics, and what can be learned from the
spatial structure that remains?

Interactive presentation: [climate-inequality.streamlit.app](https://climate-inequality.streamlit.app/)

This repository is the completed, fixed snapshot of an independent research
project. It fits a warming trend to every Berkeley Earth city record over
1950–2013, defines national warming three ways (station-, population- and
area-weighted), decomposes the cross-country variance of the area-weighted
rate among four groups of country attributes, and then investigates the
spatially clustered residual under geographically separated validation. A
separate module regresses the global mean temperature on radiative forcings;
it explains the global trajectory through time, not the cross-country
pattern, and is never combined with the decomposition.

Everything is descriptive. CO₂ is well mixed, so a country's own emissions do
not preferentially heat its own territory; every statistic here measures how
warming *aligns* with a country attribute, not what caused the warming.

## Principal findings

Numbers are read from the committed result records (`app/data/` and
`research/model_v2/outputs/`); the tables behind every claim are in
[`docs/findings.md`](docs/findings.md).

- **Warming is universal but uneven.** All 3,510 city locations warmed over
  1950–2013 (sample mean 0.146 °C/decade; 5th to 95th percentile 0.06 to
  0.25). Above 60°N the mean is 0.23 °C/decade; the fastest national rates are
  in Central Asia, the Iranian plateau and the Sahel rather than the Arctic.
- **Historical responsibility and experienced warming are different axes.**
  Ranked by station-weighted warming, countries with more cumulative CO₂ per
  person warmed faster (Spearman ρ +0.36, n = 157). Ranked by area-weighted
  warming the correlation is +0.01 under Berkeley Earth (n = 154) and −0.02
  under anomaly-aligned ERA5 (n = 151). The shift is consistent with station-network geography, but the gridded
  product also changes the observational representation; station siting is
  not isolated as the cause.
- **Geography structures most of what a static model explains.** Over the 151
  countries with complete features, four attribute groups explain 63.6% of the
  cross-country variance in area-weighted warming. Physical geography accounts
  for 0.519 of total variance (country-bootstrap 95% interval 0.452 to 0.630),
  socioeconomic development 0.063, population 0.042 and historical emissions
  responsibility 0.013 (0.006 to 0.035); the residual is 0.364. Geography is the
  largest group in every one of 2,000 resamples. These are descriptive
  allocations of explained cross-country variance, not causal contributions.
- **The residual is spatially structured, and hard to recover.** About 36% of
  the variance lies outside the static model, with residual Moran's I 0.27.
  Under leave-one-country-out validation with a 500 km territorial exclusion
  the static model transfers weakly (R² 0.153 against 0.636 in sample).
  Re-measuring geography over land area, a pre-1950 baseline dryness index and
  a richer latitude functional form did not demonstrate transferable
  improvement. Spatial error and spatial lag models both improve
  geographically separated prediction under Berkeley Earth and represent
  39–40% of the static residual as neighbour-conditional covariance; that gain
  does not reproduce with ERA5 outcomes, while the static decomposition
  conclusion survives both products.
- **The least responsible warm no less and can adapt least.** Responsibility
  rises steeply with income (ρ +0.89) and falls with ND-GAIN vulnerability
  (ρ −0.88), while area-weighted warming is flat across both (ρ −0.15 and
  +0.02, neither significant).
- **The 1950–2013 trends underpredict what came after.** Against gridded
  observations through 2024, the stored lines run 0.48 °C below observed
  anomalies on average.
- **A forcing regression reproduces the global series.** Trained through 2013,
  it explains 91% of the annual variance and its 95% band covers 10 of the 11
  held-out years (2014–2024).

## How it is built

```
Berkeley Earth city CSV -> DuckDB -> monthly anomalies -> Theil-Sen trend per city location
                                                              |
        +-----------------------------------------------------+------------------+
        v                                                     v                  v
  IDW / kriging surface                     country means (station / population / area weighted)
  (leave-location-out CV)                                     |
                                       OWID CO2 + population, continents, income groups, ETOPO,
                                       Koppen, GPW population + national-identifier grids
                                                              |
                              +-------------------------------+-------------------------+
                              v                               v                         v
                 Gini / Theil + Shapley decomposition   responsibility-warming         income and
                 + bootstrap / jackknife / Moran's I    comparison (3 weightings,      ND-GAIN
                                                        consumption lens, ERA5)       strata
                                                              |
GISTEMP + AR6 ERF + ONI -> forcings table -> AR(1) ridge regression -> hindcast
                                                              |
                                          app/data/ bundle (committed) -> Streamlit
                                                              |
            research/model_v2/: spatial CV, geography re-measurement, baseline dryness,
            latitude form, spatial error and lag models, ERA5 product check -> committed records
```

Each pipeline stage is a module under `src/` with a `python -m src.<module>`
entry point; every published number is produced by tested code, and the bundle
builder refuses to publish from a stale or inconsistent pipeline state. The
methods, parameters and integrity checks are documented in
[`docs/reproducibility.md`](docs/reproducibility.md); the scientific rationale
for the decomposition is in
[`docs/decomposition_design_memo.md`](docs/decomposition_design_memo.md); the
residual investigation is documented in
[`research/model_v2/README.md`](research/model_v2/README.md).

## Running it

The dashboard and the test suite need no data download: the app reads only the
committed bundle, and the tests run on synthetic fixtures.

```bash
uv sync --locked --extra dev
uv run --locked pytest -q
python -m src.research_audit
uv run --locked streamlit run app/streamlit_app.py
```

Rebuilding the bundle from the raw sources takes a few hours of downloads and
computation; the step-by-step sequence is in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Documentation

| Document | What it covers |
|----------|----------------|
| [`reports/canonical_results.json`](reports/canonical_results.json) | Generated FINAL / SUPERSEDED / HISTORICAL result index with exact source pointers |
| [`docs/findings.md`](docs/findings.md) | Results, with the numbers behind every claim above |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Data sources, pipeline stages, parameters, determinism and integrity checks |
| [`docs/decomposition_design_memo.md`](docs/decomposition_design_memo.md) | Why a variance decomposition, why these groups, what it can and cannot support |
| [`docs/stability.md`](docs/stability.md) | Bootstrap, leave-one-out and spatial diagnostics on the decomposition |
| [`docs/era5_validation_results.md`](docs/era5_validation_results.md) | The area-weighting result re-tested on ERA5 reanalysis |
| [`docs/vulnerability_results.md`](docs/vulnerability_results.md) | Income and ND-GAIN stratification |
| [`docs/physical_model.md`](docs/physical_model.md) | The forcing regression: model, fit, hindcast |
| [`docs/future_work.md`](docs/future_work.md) | Limitations and open scientific questions |
| [`research/model_v2/RESULTS.md`](research/model_v2/RESULTS.md) | The residual-structure investigation, stage by stage |
| [`research/model_v2/VALIDATION_PROTOCOL.md`](research/model_v2/VALIDATION_PROTOCOL.md) | The geographically separated validation and decision rules |

## Limitations

- **Station sampling.** Trends exist only where Berkeley Earth has city
  records: dense in the populated mid-latitudes, sparse over the Arctic,
  Sahara, Amazon and Siberia. The area-weighted outcome sets national
  aggregation by land area, but it is not independent of the station record:
  Berkeley Earth's gridded field is interpolated from the same stations.
- **Products disagree.** Berkeley Earth and ERA5 differ on national
  area-weighted trends by more than the decomposition residual (rank agreement
  about 0.6), and ERA5 carries its own reanalysis error structure. The static
  conclusion holds under both; the spatial models' predictive gain does not.
- **About a third of cross-country variance is outside the static model**,
  and the tested extensions did not recover it in a transferable way. The
  remaining variance may mix omitted regional physical heterogeneity, internal
  variability, functional-form limitations, product differences, spatial
  dependence, predictor measurement error, country aggregation and
  finite-sample error, in proportions this design cannot separate.
- **Land only, monthly means, ending September 2013.** Ocean warming, heat
  extremes and the post-2013 acceleration are outside the trend fits.
- **Country aggregation and a 151-country complete-case sample** with equal
  country weighting; nothing describes variation within countries.
- **Uncertainty is not fully propagated.** Trend uncertainty and predictor
  measurement error do not enter the shares; resampling intervals are not
  spatially corrected, and spatial dependence persists beyond the 500 km
  exclusion used for validation.
- **Same-data hypotheses.** Several follow-up hypotheses were motivated by the
  same outcome they were tested on.
- **Correlation, not attribution.** The forcing regression explains the global
  mean; the decomposition and comparisons describe cross-country alignment.
  Neither identifies a causal effect of a country's emissions on its own
  warming, and a small responsibility share does not mean emissions do not
  explain warming: greenhouse gases drive the global trend, while the
  decomposition asks why warming rates differ among countries.

## Data

- [Berkeley Earth: Climate Change - Earth Surface Temperature Data](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data) (Kaggle) and the [Berkeley Earth 1° gridded land product](https://berkeleyearth.org/data/)
- [Our World in Data: CO₂ and Greenhouse Gas Emissions](https://github.com/owid/co2-data), continents, and the World Bank income classification
- [SEDAC GPW v4.11](https://sedac.ciesin.columbia.edu/data/collection/gpw-v4) population count and national identifier grids
- [ETOPO 2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model) elevation, [Beck et al. 2018](https://www.gloh2o.org/koppen/) Köppen–Geiger classes, [Natural Earth](https://www.naturalearthdata.com/) land polygons, GSHHG shorelines
- [ND-GAIN Country Index](https://gain.nd.edu/our-work/country-index/), [NASA GISTEMP v4](https://data.giss.nasa.gov/gistemp/), [Forster et al. ERF time series](https://github.com/ClimateIndicator/forcing-timeseries), [NOAA ONI](https://www.cpc.ncep.noaa.gov/data/indices/), [ERA5](https://cds.climate.copernicus.eu/) (product check), [CRU TS v4.10](https://crudata.uea.ac.uk/cru/data/hrg/) (baseline dryness)

## Author

Malek Elaghel. The project began as a 2022 undergraduate research proposal to
map warming from the Berkeley Earth city data; the current pipeline and
analyses were built in 2026 as an independent personal project with no
external funding.
