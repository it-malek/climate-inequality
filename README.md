# Climate inequality

Where has land warmed fastest since 1950, and how does that pattern line up with
who is responsible for the emissions?

Live dashboard: [climate-inequality.streamlit.app](https://climate-inequality.streamlit.app/)

The project fits a warming trend to every Berkeley Earth city record, aggregates
those trends to countries, and then asks three questions of the result:

1. **How is warming distributed across countries, and what structures it?**
   A group-level Shapley (LMG) decomposition of the cross-country variance in
   warming rate over four fixed axes: emissions responsibility, physical
   geography, socioeconomic development, and population. With bootstrap and
   leave-one-out stability checks.
2. **Does warming track historical responsibility?** Rank correlation and a
   Lorenz-style inequality coefficient between cumulative per-capita CO₂ and
   warming rate, under station-, population- and area-weighted definitions of a
   country's warming, cross-checked against ERA5 reanalysis.
3. **Who is most vulnerable?** The same quantities stratified by World Bank
   income group and by ND-GAIN vulnerability.

A separate, deliberately isolated module fits the global mean temperature
series to effective radiative forcings (a closed-form Bayesian AR(1) regression
with an out-of-sample hindcast); it explains the global trajectory, not the
cross-country pattern, and is never combined with the decomposition.

Everything is descriptive. CO₂ is well mixed, so a country's own emissions do
not preferentially heat its own territory; every statistic here measures how
warming *aligns* with a country attribute, not what caused the warming.

## What the data say

Numbers are from the committed `app/data/` bundle; see
[`docs/findings.md`](docs/findings.md) for the full tables.

- **Warming is universal but uneven.** All 3,510 city-locations warmed over
  1950–2013 (land mean 0.146 °C/decade; 5th–95th percentile 0.06–0.25). Above
  60°N the mean is 0.23 °C/decade; the fastest cluster is the Iranian plateau
  and Central Asia, not the Arctic.
- **Cross-country inequality is mostly geography.** Across 154 countries the
  four axes explain 63% of the variance in country-mean warming; geography
  takes 46 points of that, emissions 8, socioeconomic 6, population 4, and 37%
  is unexplained. The residual is regionally clustered (Moran's I 0.33,
  p = 0.005). Geography is the largest share in every one of 2,000 bootstrap
  resamples.
- **The station-based emissions–warming link is a sampling artefact.** With
  each country's warming defined as the mean over its stations, warming rate
  and cumulative per-capita CO₂ correlate at Spearman ρ = +0.36. Weighting
  every square kilometre equally (a per-cell trend on the Berkeley Earth 1°
  grid, cos-latitude weighted) collapses this to ρ = +0.01; the same
  computation on ERA5 gives ρ = +0.12, also not significant. Stations sit
  where people and infrastructure are, which happens to be where high emitters
  warm fastest.
- **Vulnerability runs the other way.** Responsibility rises steeply with
  income (ρ = +0.89) and falls with ND-GAIN vulnerability (ρ = −0.88), while
  area-weighted warming is flat across both (ρ = −0.15 and +0.02, neither
  significant). Per resident, low-income countries warm slightly faster than
  high-income ones (0.205 vs 0.182 °C/decade).
- **The 1950–2013 trends underpredict what came after.** Against gridded
  Berkeley Earth observations through 2024, the stored lines run 0.48 °C below
  observed anomalies on average; refitting on the full record raises the mean
  slope from 0.147 to 0.200 °C/decade.
- **A forcing regression reproduces the global series.** Trained through 2013,
  the physical model explains 91% of the annual variance and its 95% band
  covers 10 of the 11 held-out years (2014–2024); the CO₂ sensitivity is
  0.37 °C per W/m² (95% interval 0.06–0.68).

## How it is built

```
Berkeley Earth city CSV ─► DuckDB ─► monthly anomalies ─► Theil–Sen trend per city-location
                                                              │
        ┌─────────────────────────────────────────────────────┼──────────────────┐
        ▼                                                     ▼                  ▼
  IDW / kriging surface                     country means (station / people / area weighted)
  (leave-location-out CV)                                     │
                                       OWID CO₂ + population, continents, income groups, ETOPO,
                                       Köppen, GPW population + national-identifier grids
                                                              │
                              ┌───────────────────────────────┼─────────────────────────┐
                              ▼                               ▼                         ▼
                 Gini / Theil + Shapley decomposition   responsibility–impact         income and
                 + bootstrap / jackknife / Moran's I    coupling (3 weightings,        ND-GAIN
                                                        consumption lens, ERA5)       strata
                                                              │
GISTEMP + AR6 ERF + ONI ─► forcings table ─► AR(1) ridge regression ─► hindcast
                                                              │
                                          app/data/ bundle (≈5 MB, committed) ─► Streamlit
```

Each stage is a module under `src/` with a `python -m src.<module>` entry
point; every published number is produced by tested code, and the bundle
builder refuses to publish from a stale or inconsistent pipeline state. The
methods, parameters and integrity checks are documented in
[`docs/reproducibility.md`](docs/reproducibility.md); the scientific rationale
for the decomposition is in
[`docs/decomposition_design_memo.md`](docs/decomposition_design_memo.md).

## Running it

The dashboard and the test suite need no data download: the app reads only the
committed bundle, and the tests run on synthetic fixtures.

```bash
uv sync --extra dev
uv run pytest -q
uv run streamlit run app/streamlit_app.py
```

Rebuilding the bundle from the raw sources takes a few hours of downloads and
computation; the step-by-step sequence is in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Documentation

| Document | What it covers |
|----------|----------------|
| [`docs/findings.md`](docs/findings.md) | Results, with the numbers behind every claim above |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Data sources, pipeline stages, parameters, determinism and integrity checks |
| [`docs/decomposition_design_memo.md`](docs/decomposition_design_memo.md) | Why a variance decomposition, why these axes, what it can and cannot support |
| [`docs/stability.md`](docs/stability.md) | Bootstrap, leave-one-out and spatial diagnostics on the decomposition |
| [`docs/era5_validation_results.md`](docs/era5_validation_results.md) | The area-weighting result re-tested on ERA5 reanalysis |
| [`docs/vulnerability_results.md`](docs/vulnerability_results.md) | Income and ND-GAIN stratification |
| [`docs/physical_model.md`](docs/physical_model.md) | The forcing regression: model, fit, hindcast |
| [`docs/future_work.md`](docs/future_work.md) | Open questions and known gaps |

## Limitations

- **Station sampling.** Trends exist only where Berkeley Earth has city
  records: dense in the populated mid-latitudes, sparse over the Arctic,
  Sahara, Amazon and Siberia. This is the single largest caveat on the
  station-based results and the reason the area-weighted lens exists.
- **Land only, monthly means, ending September 2013.** Ocean warming, heat
  extremes and the post-2013 acceleration are outside the trend fits (the
  validation stage measures the last of these).
- **Coordinates are grid-snapped to about 1°**, so 18 same-named city pairs
  share coordinates with other cities; the pipeline keys on the full
  (city, country, latitude, longitude) identity and the interpolation CV holds
  out whole coordinate groups.
- **Responsibility is production-based cumulative CO₂ per 2013 resident.** No
  land-use change, no non-CO₂ gases, and a population basis that flatters
  countries whose populations grew late. A window-matched consumption-based
  variant is provided for 114 countries.
- **Measurement uncertainty is not propagated** into the trend fits.
- **Correlation, not attribution.** The physical model explains the global
  mean; the decomposition and coupling describe cross-country alignment. Neither
  identifies a causal effect of a country's emissions on its own warming.

## Data

- [Berkeley Earth: Climate Change — Earth Surface Temperature Data](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data) (Kaggle) and the [Berkeley Earth 1°×1° gridded land product](https://berkeleyearth.org/data/)
- [Our World in Data: CO₂ and Greenhouse Gas Emissions](https://github.com/owid/co2-data), continents, and the World Bank income classification
- [SEDAC GPW v4.11](https://sedac.ciesin.columbia.edu/data/collection/gpw-v4) population count and national identifier grids
- [ETOPO 2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model) elevation, [Beck et al. 2018](https://www.gloh2o.org/koppen/) Köppen–Geiger classes, [Natural Earth](https://www.naturalearthdata.com/) land polygons
- [ND-GAIN Country Index](https://gain.nd.edu/our-work/country-index/), [NASA GISTEMP v4](https://data.giss.nasa.gov/gistemp/), [Forster et al. ERF time series](https://github.com/ClimateIndicator/forcing-timeseries), [NOAA ONI](https://www.cpc.ncep.noaa.gov/data/indices/), [ERA5](https://cds.climate.copernicus.eu/) (optional cross-check)

The project began as a 2022 undergraduate research proposal to map warming
from the Berkeley Earth city data; the current pipeline was built in 2026.
