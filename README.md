# Climate Inequality: Mapping Where Warming Hits Hardest

> Which regions are warming fastest — and is warming proportional to
> emissions responsibility?

**Status:** 🚧 Phases 1–5 complete (data layer; trend fitting; spatial
interpolation; emissions join; dashboard). Remaining: findings writeup and
the Community Cloud deploy click; stretch goal is validating the fitted
trends against post-2013 observations. See `README` for the full plan.

*Originally proposed as an undergraduate research fellowship project (2022);
rebuilt and substantially upgraded in 2026.*

## Question

Global warming is not geographically uniform. This project (1) quantifies
per-city warming rates from 1950–2013 using the Berkeley Earth surface
temperature dataset, (2) interpolates those rates into continuous spatial
surfaces, and (3) tests whether the countries warming fastest are the ones
most responsible for cumulative CO₂ emissions.

## Methods

- **Data layer.** The 8.6M-row Berkeley Earth by-city CSV is ingested into
  DuckDB with hemisphere-suffixed coordinates parsed to signed floats.
  Because 18 same-named (City, Country) pairs exist at multiple grid
  coordinates, every step keys on the full (City, Country, Latitude,
  Longitude) identity — 3,510 city-locations. Analysis window: 1950-01 to
  2013-09 (the dataset's end), where coverage is strong.
- **Trends.** Monthly anomalies relative to each location's 1951–1980
  monthly climatology (deseasonalization), gated on ≥90% non-null monthly
  coverage, then a per-location Theil–Sen slope in °C/decade with a 95%
  confidence interval (OLS fit alongside for comparison).
- **Interpolation.** Inverse-distance weighting (own implementation, on
  great-circle distances) vs ordinary kriging (pykrige, local k=30
  neighborhoods under a single globally-fit variogram — a global kriging
  system over 3,510 points is ill-conditioned). Methods are compared with
  leave-*location*-out cross-validation, which holds out entire
  same-coordinate groups: Berkeley Earth's grid-snapped coordinates would
  otherwise leak bit-identical duplicate values into their own folds. IDW
  wins (RMSE 0.0083 vs 0.0099 °C/decade) and renders the published surface,
  masked to Natural Earth land polygons.
- **Inequality analysis.** Per-country warming = unweighted mean of its
  city-location trends (no city-population data in the project datasets —
  see limitations). Emissions responsibility = cumulative production-based
  CO₂ summed through 2013, divided by 2013 population (OWID). Relationship
  quantified by Spearman rank correlation plus OLS of warming on log₁₀
  emissions — pooled and with continent fixed effects — using HC1 robust
  standard errors, so effects read as °C/decade per 10× emissions.

## Findings

<!-- TODO: headline result with effect size and uncertainty -->

## Dashboard

Three-page Streamlit app: an interpolated warming map with a city-station
layer toggle, a city explorer showing any location's anomaly series with
its fitted trend, and the country-level inequality scatter.

```bash
uv run streamlit run app/streamlit_app.py
```

The app reads only the committed `app/data/` bundle (~5 MB, built by
`uv run python -m src.app_assets`), so it runs identically on a fresh
clone and on Streamlit Community Cloud — no raw-data download needed.

**Deploying to Community Cloud:** push to GitHub, then at
[share.streamlit.io](https://share.streamlit.io) create an app from
`it-malek/climate-inequality`, branch `main`, main file path
`app/streamlit_app.py`, Python 3.11. The cloud environment installs
`app/requirements.txt` (the dependency file in the entrypoint's directory
takes precedence over root-level files).

<!-- TODO: Streamlit Community Cloud link -->

## Limitations

<!-- TODO: station coverage bias, dataset ends Sept 2013, land-only
interpolation, grid-snapped coordinates -->

## Reproducing

```bash
uv sync --extra dev
uv run pytest    # no data needed — tests run on synthetic fixtures
uv run ruff check src tests app

# Build the data layer (public Kaggle datasets, no credentials needed):
uv run python -c "from src.data_io import download_raw_data; download_raw_data()"
uv run python -c "from src.data_io import load_city_temperatures, city_csv_path; load_city_temperatures(city_csv_path())"

# Phase outputs (each prints its sanity checks):
uv run python -m src.trends       # city_trends.parquet
uv run python -m src.interpolate  # outputs/trend_surface.html
uv run python -m src.emissions    # country_inequality.parquet + scatter
uv run python -m src.app_assets   # app/data/ dashboard bundle (committed)

uv run streamlit run app/streamlit_app.py
```

The dashboard bundle is committed, so the app (and its tests) work without
any of the above data steps. On a fresh machine, start with
`docs/new_machine_setup.md`.

## Data

- [Berkeley Earth — Climate Change: Earth Surface Temperature Data](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
- [Our World in Data — CO₂ and Greenhouse Gas Emissions](https://github.com/owid/co2-data)
- [Natural Earth — 110m land polygons](https://www.naturalearthdata.com/downloads/110m-physical-vectors/) (land mask)
