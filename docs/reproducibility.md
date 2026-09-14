# Reproducibility

How every published number is produced, the choices made along the way, and
the checks that keep a rebuild honest.

Two guarantees:

1. **The dashboard and the tests run from a clean clone with no data and no
   network.** The app reads only the committed `app/data/` bundle; every test
   runs on synthetic fixtures (`tests/conftest.py` builds a small end-to-end
   bundle through the real builder). `tests/test_artifacts.py` additionally
   checks the committed bundle on disk for internal consistency.
2. **The pipeline regenerates the bundle deterministically.** Every stage is a
   `python -m src.<module>` entry point that reads upstream artifacts and writes
   its own; the bundle builder cross-checks refit values against stored ones
   and fails loudly on drift. Which artifacts are byte-identical across
   platforms is stated in [§7](#7-determinism).

## 1. Environment

Python ≥ 3.11 managed by [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev          # pipeline + tests + lint
uv run pytest -q
uv run ruff check src tests app scripts
```

The deployed app installs `app/requirements.txt` instead (Streamlit Community
Cloud picks the dependency file in the entrypoint's directory): streamlit,
plotly, pandas, numpy, pyarrow, statsmodels, pinned to the versions in
`uv.lock`. `uv sync --extra era5` adds the Copernicus client for the optional
ERA5 fetch. Continuous integration runs lint and the test suite on Python 3.11,
plus a smoke test of every dashboard page on 3.11 and 3.14 with only the
dashboard dependencies installed.

## 2. Data

`data/raw/` and `data/processed/` are gitignored except for two small vendored
classification tables. Sizes are approximate.

| Input | File | Size | How obtained |
|---|---|---|---|
| Berkeley Earth city temperatures (Kaggle) | `GlobalLandTemperaturesByCity.csv` | 500 MB | `src.data_io.download_raw_data()` (kagglehub, no credentials) |
| Berkeley Earth 1°×1° gridded land anomalies | `berkeley_gridded/Complete_TAVG_LatLong1.nc` | 200 MB | downloaded by `src.validation` |
| OWID CO₂ and population | `owid/owid-co2-data.csv` | 14 MB | downloaded by `src.emissions` (live file; revisions change the numbers, see §7) |
| OWID continents | `owid/continents.csv` | | **committed** (`data/raw/owid/NOTICE.md`) |
| World Bank income groups (OWID mirror) | `worldbank/world-bank-income-groups.csv` | 350 KB | **committed** (`data/raw/worldbank/NOTICE.md`) |
| ETOPO 2022 elevation, 60 arc-second | `etopo/ETOPO_2022_v1_60s_N90W180_surface.nc` | 480 MB | downloaded by `src.explain` |
| Köppen–Geiger (Beck et al. 2018), 0.5° | `koppen/Beck_KG_V1.zip` → `koppen_present_0p5.nc` | 70 MB | downloaded and converted by `src.explain` |
| Natural Earth 110m land | `natural_earth/ne_110m_land.zip` | 70 KB | downloaded by `src.interpolate` |
| SEDAC GPW v4.11 population count, 15 arc-minute, with the national-identifier grid and lookup | `gpw-v4-…_15_min_nc/…rev11_15_min.nc` and `…lookup.txt` | 84 MB | **manual**: requires a free NASA Earthdata login; place the unpacked directory under `data/raw/` |
| ND-GAIN Country Index | `ndgain/ndgain_latest.csv` | | **committed** slim extract (`scripts/fetch_ndgain.py`, `data/raw/ndgain/NOTICE.md`) |
| GISTEMP v4, Forster et al. ERF, NOAA ONI | `forcings/*` | | downloaded by `src.forcings` |
| ERA5 monthly 2 m temperature, 1950–2013, 1° | `era5/era5_t2m_monthly_1950_2013_1deg.nc` | 200 MB | **optional, manual**: `scripts/fetch_era5.py` with a Copernicus CDS account |

Stages whose inputs are absent skip with a warning and the dashboard shows a
pending state for the affected page, so a partial rebuild still produces a
working bundle.

## 3. Pipeline

Run in this order from the repository root.

```bash
uv run python -c "from src.data_io import download_raw_data; download_raw_data()"
uv run python -c "from src.data_io import load_city_temperatures, city_csv_path; load_city_temperatures(city_csv_path())"
uv run python -m src.trends          # city_trends.parquet
uv run python -m src.emissions       # country_inequality.parquet (station, people, area weightings)
uv run python -m src.interpolate     # outputs/trend_surface.html + cross-validation
uv run python -m src.app_assets      # first pass: the bundle the validation stage reads
uv run python -m src.validation      # validation_*.{json,parquet}
uv run python -m src.explain         # city_features.parquet + explain_*
uv run python -m src.forcings        # forcings.parquet
uv run python -m src.era5_validation # optional, needs the ERA5 grid
uv run python -m src.app_assets      # final bundle: app/data/
```

`src.app_assets` recomputes the trends, interpolation and inequality assets and
then builds every downstream summary (decomposition, stability, coupling
lenses, ERA5 cross-check, vulnerability strata, physical model) from
`data/processed/`. The individual modules (`src.inequality`,
`src.decomposition`, `src.stability`, `src.projections`, `src.coupling`,
`src.vulnerability`, `src.physical_model`) can also be run on their own; each
prints its sanity checks. `src.feature_schema` and `src.pcs` regenerate the
YAML mirrors of the feature contract and projection registry in `docs/`.

| Stage | Module | Output | Parameters | Check |
|---|---|---|---|---|
| Ingest | `data_io` | `climate.duckdb` | hemisphere-suffixed coordinates → signed floats; keyed on (City, Country, Lat, Lon) | row count preserved; 3,510 locations |
| Trends | `trends` | `city_trends.parquet` | baseline 1951–1980; window 1950-01 to 2013-09; ≥ 90% coverage; Theil–Sen with 95% CI, OLS alongside | land mean ≈ 0.146; > 60°N ≈ 0.228 °C/decade |
| Interpolation | `interpolate` | `trend_surface.html` | IDW power 2 vs local ordinary kriging (spherical variogram fit once globally); k = 30; 2° grid; leave-location-out CV | IDW RMSE 0.0083 vs 0.0099 |
| Country table | `emissions` | `country_inequality.parquet` | cumulative CO₂ through 2013 ÷ 2013 population; Spearman and OLS with HC1; people weights from GPW 2020 counts; area weights cos(lat) on the Berkeley grid | 157 countries; ρ ≈ +0.36 |
| Features | `explain` | `city_features.parquet`, `explain_*` | ETOPO elevation, coast distance, Köppen class, station density (k = 50 within 100 km); preflight gate | latitude baseline R² ≈ 0.32 |
| Validation | `validation` | `validation_*` | forecast from 2013-10; agreement gate r ≥ 0.80; El Niño sensitivity from 2023 | positive residuals (acceleration) |
| Inequality | `inequality` | `inequality_summary.json` | Gini, Theil-T with continent split, CV, variance; unweighted | Gini ≈ 0.175 |
| Decomposition | `decomposition` | `decomposition_summary.json` | group LMG/Shapley over `SCHEMA_V1`; log₁₀ for CO₂ and population; complete cases | n = 154; R² ≈ 0.63 |
| Stability | `stability` | `stability_summary.json` | B = 2000 country and block bootstraps (seeds 0, 1); leave-one-out; Moran's I k = 8, 199 permutations | geography largest in 100% |
| Projections and coupling | `pcs`, `projections`, `coupling` | `projections_*.parquet`, `coupling_*` | ranks (min), z (ddof 0), Lorenz coefficient, Spearman; three lenses | station ρ ≈ +0.36 → area +0.01 |
| ERA5 cross-check | `era5_weighting`, `era5_validation` | `era5_*` | same operator on ERA5 `t2m` | world land mean ≈ 0.22 |
| Strata | `vulnerability` | `vulnerability_*` | income rank and ND-GAIN score; permutation p with 999 draws (seed 0) | triple inequality holds |
| Forcings and physical model | `forcings`, `physical_model` | `forcings.parquet`, `physical_*` | 1-year lag on GHG/aerosol; ridge λ by evidence; AR(1) by iteration; train ≤ 2013 | train R² ≈ 0.91 |
| Bundle | `app_assets` | `app/data/` | rebuilds and cross-checks everything above | refit slopes match stored to 1e-8 |

## 4. Preprocessing choices

- **Location identity.** (City, Country) is not unique: 18 same-named pairs sit
  at 2–3 grid-snapped coordinates. Every stage keys on the full (City, Country,
  Latitude, Longitude) tuple, and the interpolation CV holds out whole
  coordinate groups so duplicates cannot leak across folds.
- **Anomalies** are taken against each location's own 1951–1980 monthly
  climatology. The baseline anchors levels only; slopes are insensitive to it.
- **Coverage gate.** A location needs non-null observations in at least 90% of
  the 765 window months. All 3,510 locations pass.
- **Trend estimator.** Theil–Sen (median of pairwise slopes) with the
  Sen–Kendall 95% interval; OLS is stored for comparison only.
- **Country warming** is the unweighted mean of the location slopes (the
  station-weighted definition). The people-weighted mean uses GPW 2020
  population counts sampled at each location as weights, with no latitude
  correction (a count already reflects cell area). The area-weighted mean fits a
  trend per Berkeley grid cell, assigns cells to countries through the GPW
  national-identifier band, and averages with cos(latitude) weights (a trend is
  intensive; cell area shrinks toward the poles).
- **Emissions responsibility** is cumulative production-based CO₂ summed from
  the start of each country's OWID record through 2013, divided by 2013
  population. The consumption variant sums both consumption- and
  production-based CO₂ over each country's consumption-available window
  (from 1990 for most) so the two are comparable.
- **Geography features** aggregate city → country as means (absolute latitude,
  elevation, coast distance, station density) and modal class (Köppen,
  hemisphere). 157 countries match the OWID join; 154 have complete cases over
  the decomposition features.
- **Decomposition transforms.** Cumulative CO₂ per capita, cumulative CO₂ total
  and population enter as log₁₀; categoricals (income group, Köppen class,
  hemisphere, continent) are dummy-coded with the first sorted level dropped
  and attributed to their group as a block. Features with `status="proposed"`
  in the schema (GDP per capita, urbanisation, carbon intensity) have no data
  source and are excluded.

## 5. Integrity checks built into the pipeline

- `app_assets.theil_sen_intercepts` refits Theil–Sen on the rebuilt anomalies
  and requires the slopes to match `city_trends.parquet` to 1e-8;
  `attach_city_ids` requires a one-to-one match between anomaly locations and
  trend rows with identical observation counts. A stale trends file raises.
- `feature_schema.validate_design_matrix` rejects any design column that is not
  a schema feature; `assert_groups_disjoint` guarantees the groups partition the
  matrix so Shapley shares cannot double-count. `DecompositionResult.check_sums_to_one`
  asserts the LMG identity.
- `coupling.validate_projection_frame` rejects any column outside the registered
  projections of a lens; `pcs` checks its registries at import, and each
  contract carries a definition hash mirrored in `docs/pcs_*.yaml`.
- `forcings.ForcingsResult.check` requires contiguous years, no gaps in the
  estimator columns, GISTEMP–Berkeley correlation ≥ 0.95 and a plausible recent
  anomaly level; `physical_model.PhysicalModelResult.check` requires a
  stationary ρ and a bounded band coverage.
- `explain.run_geo_preflight` verifies grid orientation, spot-checks three
  cities, checks sampling determinism and NaN rates, and checks the
  country-name joins before any feature is used.
- `app_assets._copy_findings_parquet` and `app.loaders._read_bundle_parquet`
  fail loudly on schema drift at build and at load.
- `tests/test_artifacts.py` validates the committed bundle on disk: coupling
  summaries recompute from their tables, projections equal their source
  columns, decomposition shares partition to one, stability point shares match
  the decomposition, the physical trajectory is contiguous and its band brackets
  the mean.

## 6. Interpretation limits

Every statistic is descriptive. The decomposition attributes variance, not
cause; the coupling and strata describe alignment; the physical model is a
predictive association validated by hindcast. None supports causal claims
about a country's emissions and its own warming, policy counterfactuals,
statements about heat extremes or damages, or within-country inequality. The
`interpretation` field in each summary JSON restates this so it travels with
the numbers. The full threats-to-validity list is in
[`decomposition_design_memo.md`](decomposition_design_memo.md) §7.

## 7. Determinism

- All randomness goes through `numpy.random.default_rng` with recorded seeds
  (stability: seed 0 for the country bootstrap, 1 for the block bootstrap, 0
  for the Moran's I permutations; vulnerability: seed 0). The physical model
  uses no sampler.
- Parquet outputs go through `data_io.write_typed_parquet` (explicit column
  types, fixed row order, zstd); the bundle's `city_trends`, `city_anomalies`
  and `trend_surface` files are written with pandas in a fixed sort order.
- Every committed JSON summary passes through `data_io.round_floats`, which
  rounds floats to 10 significant figures at serialization. Rebuilding the
  same inputs on a different platform reproduces these files byte for byte,
  with one exception: the residual Moran's I in `stability_summary.json`
  depends on nearest-neighbour tie-breaking in the KD-tree and moves by about
  1e-4 between macOS and Linux. Parquet files carrying unrounded floats
  (`coupling*.parquet`, `physical_trajectory.parquet`, the `intercept` column
  of `city_trends.parquet`) reproduce to within 1e-11 across platforms and
  byte for byte on the same platform. `stats.json` carries unrounded OLS
  statistics that differ at 1e-16 across platforms.
- The pipeline does not pin upstream data vintages except for the vendored
  tables. OWID revises its CO₂ series, and the Kaggle and Berkeley Earth files
  are snapshots; the committed bundle was built from files retrieved in
  June 2026, and `physical_summary.json` records the SHA-256 of the forcings
  table it was fit on.

To check a rebuild:

```bash
uv run python -m src.app_assets
git diff --stat app/data/       # expect no change on the same platform
uv run pytest -q
```
