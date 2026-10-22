# Climate inequality

How is cross-country variation in observed land-warming rates structured among
physical geography, historical emissions responsibility, socioeconomic
development and population characteristics, and what can be learned from the
spatial structure that remains?

Interactive presentation: [climate-inequality.streamlit.app](https://climate-inequality.streamlit.app/)

This independent research project examines how warming differs across countries.
The 2.1.0 analysis fits a warming trend to every
Berkeley Earth city-location record over 1950–2013, defines national warming three ways (station-, population- and
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

These findings describe the 2.1.0 analysis at this development revision.
Historical result records remain available and are distinguished from the
corrected results.

- **Warming in the sample is uneven.** All 3,510 city-location estimates warmed over
  1950–2013 (sample mean 0.146 °C/decade; 5th to 95th percentile 0.06 to
  0.25). Above 60°N the mean is 0.23 °C/decade; the fastest national rates are
  in Central Asia, the Iranian plateau and the Sahel rather than the Arctic.
- **Historical responsibility and experienced warming are different axes.**
  Ranked by station-weighted warming, countries with more cumulative CO₂ per
  person warmed faster (Spearman ρ +0.36, n = 157). Ranked by area-weighted
  warming the correlation is +0.01 under Berkeley Earth (n = 154) and −0.01
  under anomaly-aligned ERA5 (n = 154). Both sampling and observational
  representation change across these comparisons, so they do not isolate the
  cause of the shift.
- **Geography structures most of what a static model explains.** Over the 151
  countries with complete features, four attribute groups explain 63.6% of the
  cross-country variance in area-weighted warming. Physical geography accounts
  for 0.519 of total variance in the retained full-rank model (country-bootstrap
  95% interval 0.452 to 0.630),
  socioeconomic development 0.063, population 0.042 and historical emissions
  responsibility 0.013 (0.006 to 0.035); the residual is 0.364. Geography is the
  largest group in every one of 2,000 resamples. These are descriptive
  allocations of total cross-country variance, not causal contributions. The
  separate legacy decomposition uses the same 151 complete countries; its
  revised bootstrap population is explained in [docs/stability.md](docs/stability.md).
- **The residual is spatially structured, and hard to recover.** About 36% of
  the variance lies outside the static model, with residual Moran's I 0.27.
  Under leave-one-country-out validation with a 500 km territorial exclusion
  the static model transfers weakly (R² 0.153 against 0.636 in sample).
  Re-measuring geography over land area, a pre-1950 baseline dryness index and
  a richer latitude functional form did not demonstrate transferable
  improvement. Spatial error and spatial lag models both improve
  geographically separated prediction under Berkeley Earth and represent
  a 39–40% in-sample reduction in static residual squared error through
  neighbour-conditional prediction. This is covariance accounting, not a
  physical mechanism. The predictive gain does not reproduce with ERA5 outcomes, while the static decomposition
  conclusion survives both products.
- **Lower responsibility aligns with greater composite vulnerability.**
  Responsibility rises with income (ρ +0.89, n = 157) and falls with ND-GAIN
  vulnerability (ρ −0.84, n = 155). With the official 2026 v2 release and a
  fixed 2023 observation year, the area-warming gradient is +0.10 (n = 153;
  country-bootstrap 95% interval −0.06 to +0.26). These comparisons use
  different available-country samples. Neither nonsignificance nor an interval
  spanning zero establishes equal or no-lower warming. ND-GAIN combines
  exposure, sensitivity and adaptive capacity across six sectors; it is not a
  standalone measure of ability to adapt.
- **Kriging has lower error in the replacement interpolation evaluation.**
  With matched folds and variograms fitted only to training data,
  coordinate-group RMSE is 0.0104 °C/decade for kriging versus 0.0119 for IDW.
  Under ten spatial folds with 250 km exclusion it is 0.0438 versus 0.0515.
  This descriptive advantage concerns the tested designs on previously
  examined data, rather than an untouched confirmatory holdout.
- **The 1950–2013 trends underpredict what came after.** Against gridded
  observations through 2024, the stored lines run 0.48 °C below observed
  anomalies on average.
- **The forcing regression supports a conditional hindcast.** Its training fit
  through 2013 explains 91% of annual variance; corrected conditional 95% bands
  cover 10 of 11 subsequent years (2014–2024). Rolling evaluation refits at 24
  training origins and produces 264 predictions over horizons of 1–11 years.
  These calculations use observed future drivers. They do not demonstrate a
  forecast available at each origin or establish interval calibration from
  small, overlapping samples. Bands condition on the fitted AR(1) parameter
  and ridge penalty and omit future-driver uncertainty.

## How it is built

```
Berkeley Earth city CSV -> DuckDB -> monthly anomalies -> Theil-Sen trend per city location
                                                              |
        +-----------------------------------------------------+------------------+
        v                                                     v                  v
  IDW / kriging surface                     country means (station / population / area weighted)
  (grouped / spatial CV)                                       |
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
[`docs/current_validation.md`](docs/current_validation.md) for 2.1.0 and
[`docs/reproducibility.md`](docs/reproducibility.md) for the historical pipeline;
the scientific rationale
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

Rebuilding this analysis requires the verified raw sources and offline
computation. The step-by-step sequence is in
[`docs/current_validation.md`](docs/current_validation.md);
[`docs/reproducibility.md`](docs/reproducibility.md) documents the historical pipeline.

## Documentation

| Document | What it covers |
|----------|----------------|
| [`reports/canonical_results.json`](reports/canonical_results.json) | Generated FINAL / SUPERSEDED / HISTORICAL result index with exact source pointers |
| [`docs/findings.md`](docs/findings.md) | Results, with the numbers behind every claim above |
| [`docs/current_validation.md`](docs/current_validation.md) | 2.1.0 reconstruction, fixed settings and verification |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Historical pipeline, data sources, parameters, determinism and integrity checks |
| [`docs/decomposition_design_memo.md`](docs/decomposition_design_memo.md) | Why a variance decomposition, why these groups, what it can and cannot support |
| [`docs/stability.md`](docs/stability.md) | Bootstrap, leave-one-out and spatial diagnostics on the decomposition |
| [`docs/era5_validation_results.md`](docs/era5_validation_results.md) | The area-weighting result re-tested on ERA5 reanalysis |
| [`docs/vulnerability_results.md`](docs/vulnerability_results.md) | Income and ND-GAIN stratification |
| [`docs/physical_model.md`](docs/physical_model.md) | The forcing regression: model, fit, hindcast |
| [`docs/future_work.md`](docs/future_work.md) | Limitations and open scientific questions |
| [`research/model_v2/RESULTS.md`](research/model_v2/RESULTS.md) | The residual-structure investigation, stage by stage |
| [`research/model_v2/VALIDATION_PROTOCOL.md`](research/model_v2/VALIDATION_PROTOCOL.md) | The geographically separated validation and decision rules |

## Limitations

- **Source-location sampling.** Distinct city labels share Berkeley Earth
  source locations: 3,510 records have 1,366 distinct coordinate pairs. Shared
  rounded coordinates and identical monthly values support common source-field
  assignment, rather than independent weather-station observations. No duplicate
  identity-month records were found. Records are retained and coincident
  locations stay together in interpolation validation. Coverage is dense in
  populated mid-latitudes and sparse over the Arctic,
  Sahara, Amazon and Siberia. The area-weighted outcome sets national
  aggregation by land area. The city series and Berkeley Earth gridded field
  share an observational system; changing aggregation does not create an
  independent temperature record.
- **Products disagree.** Berkeley Earth and ERA5 differ on national
  area-weighted trends by more than the decomposition residual (rank agreement
  about 0.6), and ERA5 carries its own reanalysis error structure. The static
  conclusion holds under both; the spatial models' predictive gain does not.
- **About a third of cross-country variance is outside the static model.**
  The tested static extensions did not establish transferable improvement,
  while spatial predictive gains depend on the temperature product. The
  remaining variance may mix omitted regional physical heterogeneity, internal
  variability, functional-form limitations, product differences, spatial
  dependence, predictor measurement error, country aggregation and
  finite-sample error, in proportions this design cannot separate.
- **Land only, monthly means, ending September 2013.** Ocean warming, heat
  extremes and the post-2013 acceleration are outside the trend fits.
- **Country aggregation and a 151-country complete-case sample** with equal
  country weighting; nothing describes variation within countries.
- **Uncertainty is not fully propagated.** Trend uncertainty and predictor
  measurement error do not enter the shares. Resampling describes stability
  conditional on observed countries; six continent blocks do not establish
  calibrated spatial uncertainty, and dependence persists beyond the 500 km
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
- [ND-GAIN Country Index](https://gain.nd.edu/our-work/country-index/download-data/)
  2026 release, `release_data_v2`, downloaded 22 October 2026, using the exact
  2023 columns. This separately versioned source update preserves the historical
  extract; its original archive could not be recovered, so exact historical
  source reconstruction is not claimed.
- [NASA GISTEMP v4](https://data.giss.nasa.gov/gistemp/), [Forster et al. ERF time series](https://github.com/ClimateIndicator/forcing-timeseries), [NOAA ONI](https://www.cpc.ncep.noaa.gov/data/indices/), [ERA5](https://cds.climate.copernicus.eu/) (product check), [CRU TS v4.10](https://crudata.uea.ac.uk/cru/data/hrg/) (baseline dryness)

## Author

Malek Elaghel. The project began as a 2022 undergraduate research proposal to
map warming from the Berkeley Earth city data; the current pipeline and
analyses were built in 2026 as an independent personal project with no
external funding.
