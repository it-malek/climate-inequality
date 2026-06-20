# Reproducibility

How every published number is produced, which choices were made along the way,
and the integrity checks that keep a rebuild honest. This document is the
authoritative reproduction reference; the README is the summary, and
`docs/decomposition_design_memo.md` is the scientific rationale.

## Two guarantees

1. **The dashboard reproduces from a clean clone with no data and no network.**
   The app reads only the committed `app/data/` bundle (~5 MB). `git clone`,
   `uv sync --extra dev`, `uv run streamlit run app/streamlit_app.py` — done.
   `uv run pytest` is likewise data-free: every test runs on synthetic fixtures
   (`tests/conftest.py` builds a tiny end-to-end bundle through the *real*
   builder).

2. **The full pipeline regenerates the committed bundle deterministically.**
   From the public raw datasets, the pipeline rebuilds `app/data/` to the same
   bytes on any platform. Determinism is enforced, not hoped for (see
   [§7](#7-built-in-reproducibility--integrity-checks)).

---

## 1. Environment

- **Python ≥ 3.11**, managed by [uv](https://docs.astral.sh/uv/). `uv sync
  --extra dev` installs the locked dependency set (`uv.lock`) including the dev
  tools (pytest, ruff, jupyter, xarray/netcdf4 for the gridded inputs).
- The **deployed app** uses the minimal pinned stack in `app/requirements.txt`
  (streamlit, plotly, pandas, numpy, pyarrow, statsmodels) — Streamlit Community
  Cloud picks the entrypoint-directory requirements over `uv.lock`. Keep the two
  in step.
- New-machine bootstrap, including the WSL/macOS specifics, is in
  `docs/new_machine_setup.md`.

## 2. Datasets

All inputs are public and require no credentials. `data/raw/` and
`data/processed/` are git-ignored (rebuilt per machine); only the distilled
`app/data/` bundle is committed.

| Dataset | File(s) | Provides | Source |
|---------|---------|----------|--------|
| Berkeley Earth surface temperature (Kaggle) | `GlobalLandTemperaturesByCity.csv` (~8.6M rows) | Monthly city temperatures 1743–2013 | [kaggle.com/datasets/berkeleyearth/…](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data) |
| OWID CO₂ & GHG | `owid-co2-data.csv` | Annual production CO₂ + population per country | [github.com/owid/co2-data](https://github.com/owid/co2-data) (fetched fresh; the Kaggle CO₂ snapshot lacks population) |
| OWID continents | `continents.csv` | Continent labels (fixed effects / Theil groups) | OWID |
| World Bank income groups | `world-bank-income-groups.csv` | Low/lower-mid/upper-mid/high class | World Bank |
| ETOPO 2022 (NOAA) | `ETOPO_2022_v1_60s…​.nc` | 60″ global elevation | [ngdc.noaa.gov](https://www.ngdc.noaa.gov) |
| Köppen–Geiger (Beck et al. 2018) | `Beck_KG_V1.zip` → `koppen_present_0p5.nc` | Present-day climate class at 0.5° | [gloh2o.org/koppen](https://www.gloh2o.org/koppen/) |
| Natural Earth 110m | `ne_110m_land.zip` | Land polygons (surface mask) + coastlines (coast distance) | [naturalearthdata.com](https://www.naturalearthdata.com/) |

`uv run python -c "from src.data_io import download_raw_data; download_raw_data()"`
fetches the Kaggle + OWID inputs; the ETOPO/Köppen/Natural Earth grids are
downloaded lazily by `src.explain` on first use.

## 3. The pipeline, stage by stage

Each module is runnable as `python -m src.<module>` and prints its own sanity
checks. Inputs flow strictly left to right.

| Stage | Module | Output | Key parameters | Sanity check |
|-------|--------|--------|----------------|--------------|
| Data layer | `src.data_io` | `climate.duckdb` | hemisphere-suffixed coords → signed floats; keyed on (City, Country, Lat, Lon) | 3,510 city-locations |
| Trends | `src.trends` | `city_trends.parquet` | baseline 1951-01…1980-12; analysis 1950-01…2013-09; ≥90% monthly coverage; Theil–Sen + 95% CI | land mean ≈ 0.146 °C/decade; >60°N ≈ 0.228 (ratio ≈ 1.56×) |
| Interpolation | `src.interpolate` | `trend_surface.html` | IDW vs ordinary kriging; k=30 neighbors; 2.0° grid; land-masked; leave-*location*-out CV | IDW wins LOO-CV 0.0083 vs 0.0099 °C/decade |
| Emissions | `src.emissions` | `country_inequality.parquet` | cumulative production CO₂ through 2013 ÷ 2013 population; Spearman + OLS (pooled, continent-FE), HC1 SEs | 157 countries; ρ ≈ +0.36; FE coef ≈ +0.029 |
| Features | `src.explain` | `city_features.parquet`, `explain_*` | abs-latitude, ETOPO elevation, coast distance, Köppen class, station density; city + country OLS | latitude baseline R²; Moran's I on residuals |
| Validation | `src.validation` | `validation_*` | extend series post-2013 vs the stored Theil–Sen line; overlap gate r ≥ 0.80 | stored lines underpredict (acceleration) |
| Inequality | `src.inequality` | `inequality_summary.json` | Gini, Theil-T (+ between/within continents), CV, variance; unweighted | Gini ≈ 0.175; Theil-T ≈ 0.050 |
| Decomposition | `src.decomposition` | `decomposition_summary.json` | group LMG/Shapley over `SCHEMA_V1` (available features); complete-case | n = 154; R² ≈ 0.63; geography ≈ 0.46, emissions ≈ 0.08, residual ≈ 0.37 |
| Stability | `src.stability` | `stability_summary.json` | country bootstrap (B = 2000, seeded); leave-one-country-out influence; Moran's I (k = 8) | geography `p_largest` = 1.0; Moran's I ≈ 0.33 (p = 0.005) |
| Projections | `src.pcs`, `src.projections` | `docs/pcs_v1.yaml`, `projections_v1.parquet` | PCS v1 identity binding; emits `Country` + 2 projections only | sha256 hashes match; mirror `git diff`-clean |
| Coupling (L3) | `src.coupling` | `coupling.parquet`, `coupling_summary.json` | DESC rank, z (ddof=0), rank/z-gap, Lorenz, Gini coeff, Spearman | n = 157; ρ ≈ +0.36; inequality ≈ 0.563 |
| Bundle | `src.app_assets` | `app/data/` | recomputes assets + folds in inequality, decomposition, coupling & stability | byte-stable; build-time slope check |

`src.app_assets` is the **single bundle builder**: it recomputes the
trends/interpolation/inequality assets and, when `city_features.parquet` and the
income table exist, also writes `inequality_summary.json` and
`decomposition_summary.json` (`build_decomposition_summaries`). One
`python -m src.app_assets` therefore regenerates the entire committed bundle; if
the Phase-7 feature inputs are absent it warns and skips the decomposition,
and the dashboard page renders its explicit "not built yet" state. It also builds
the Layer-3 `coupling.parquet` + `coupling_summary.json`
(`build_coupling_summary_asset`, unconditional) and folds in
`stability_summary.json` when present.

## 4. Preprocessing & assumptions

- **Location identity.** (City, Country) is **not** unique — 18 same-named pairs
  sit at 2–3 grid-snapped coordinates. Every stage keys on the full (City,
  Country, Latitude, Longitude) tuple; the interpolation CV holds out whole
  coordinate groups so grid-snapped duplicates cannot leak across folds.
- **Deseasonalization.** Monthly anomalies are taken against each location's own
  1951–1980 monthly climatology, gated on ≥90% non-null coverage in the window.
- **Trend estimator.** Per-location **Theil–Sen** slope (robust to outliers),
  reported in °C/decade with a 95% CI; an OLS slope is stored alongside for
  comparison only.
- **Analysis window ends Sept 2013** (the Kaggle snapshot's end). All trends are
  therefore *backcasts*; `src.validation` tests them against post-2013 gridded
  Berkeley Earth.
- **Researcher degrees of freedom** (each a documented choice, several
  earmarked for the stability layer's construction-sensitivity block —
  `docs/stability_roadmap.md` §6, roadmapped; the shipped stability layer covers
  bootstrap share CIs, influence and residual spatial structure):
  unweighted vs population/area-weighted country means; mean vs median
  aggregation of city slopes; the 2013 cutoff; per-capita vs total emissions;
  ~1° grid-snapped sampling of Köppen/elevation.

## 5. Country aggregation

- **Warming** per country = the **unweighted mean** of its city-location
  Theil–Sen slopes. Unweighted because the project datasets carry **no city
  populations**; this is *station-weighted*, not area- or population-weighted,
  and is the single largest validity caveat (see the design memo §7 and the
  README limitations).
- **Geography features** are aggregated city→country as means (abs-latitude,
  elevation, continentality, station density) and **modal class** (Köppen,
  hemisphere) — `aggregate_city_features_to_country`.
- **Emissions responsibility** = cumulative production-based CO₂ summed through
  2013, divided by 2013 population (t/person), plus absolute cumulative CO₂ (Mt)
  as a scale proxy.
- 157 countries match the temperature × OWID join (Puerto Rico and Réunion drop
  out — no OWID series); the decomposition runs on the **154** with complete
  cases over all used `SCHEMA_V1` features.

## 6. Inequality metrics and decomposition

### Inequality (`src.inequality`, descriptive)

On the country mean-warming distribution (strictly positive, so all are well
defined):

- **Variance** (population, ddof=0) and **coefficient of variation** (scale-free).
- **Gini** from the weighted mean absolute difference, `G = MAD / (2·mean)`.
- **Theil-T** (GE(1)), `T = Σ pᵢ (xᵢ/μ) ln(xᵢ/μ)`, with the exact
  **between/within-continent** decomposition `T = T_between + T_within` as a
  model-free cross-check on the regression decomposition.
- **Lorenz** curve points for the dashboard.

Headline: Gini ≈ 0.175, Theil-T ≈ 0.050, of which ≈18% is *between* continents.

### Decomposition (`src.decomposition`, descriptive)

Group-level **LMG / Shapley-Owen R²** attribution: each axis's share is its
incremental R² averaged over **every ordering** in which the axes could enter the
model, so correlated axes (emissions ↔ geography) split their shared variance
fairly instead of by entry order. The four `SCHEMA_V1` axes — emissions,
geography, socioeconomic, population — plus a named **residual** (`1 − R²`) sum to
exactly 1 (`check_sums_to_one`). Categoricals (income, Köppen, hemisphere,
spatial block) enter as a grouped fixed-effect block and are attributed to their
group as a unit. Heavy-tailed positive magnitudes (`cum_co2_per_capita`,
`cum_co2_total`, `population`) enter as log₁₀; `status="proposed"` features
(`gdp_per_capita`, `urbanization_rate`, `co2_intensity_gdp`) are excluded from
the live design until their data source is wired.

Headline (n = 154, R² = 0.63): geography 0.455, emissions 0.084, socioeconomic
0.056, population 0.036, residual 0.369. Detail in `docs/key_findings.md`.

## 7. Built-in reproducibility & integrity checks

These run inside the pipeline and **fail loud** rather than publish a
stale/inconsistent artifact:

- **Build-time trend consistency.** `app_assets.theil_sen_intercepts` refits
  Theil–Sen on the rebuilt anomalies and asserts the slopes match the stored
  parquet to `SLOPE_CONSISTENCY_ATOL = 1e-8`; `attach_city_ids` asserts the
  anomaly locations map one-to-one to trends rows with identical `n_obs`. A
  stale `city_trends.parquet` raises `RuntimeError`.
- **Schema enforcement.** `feature_schema.validate_design_matrix` rejects any
  design-matrix column outside `SCHEMA_V1` (ids/outcome must be whitelisted);
  `assert_groups_disjoint` guarantees the groups partition the matrix so Shapley
  shares cannot double-count.
- **Bundle column checks.** `app_assets._copy_findings_parquet` and
  `loaders._read_bundle_parquet` fail loudly on schema drift, at build and at
  load, instead of breaking a page far downstream.
- **Deterministic JSON.** `data_io.round_floats` rounds the inequality /
  decomposition summaries to 12 significant figures at serialization, absorbing
  the last-bit `numpy.linalg.lstsq`/BLAS noise that otherwise makes the
  committed JSON environment-dependent. The summaries reproduce **byte-for-byte**
  across platforms (regression-tested in
  `tests/test_app_assets.py::TestDecompositionSummaries` and
  `tests/test_data_io.py::TestRoundFloats`).
- **Schema mirror.** `python -m src.feature_schema` regenerates
  `docs/feature_schema_v1.yaml` from the dataclasses; it is `git diff`-clean,
  so the human-readable contract cannot silently drift from the code.
- **Disclaimer travels with the numbers.** Every summary JSON carries
  `feature_schema.INTERPRETATION_NOTE`, so the variance-attribution-only boundary
  cannot be dropped by a downstream consumer.

To verify reproducibility yourself after a pipeline run:

```bash
uv run python -m src.app_assets            # rebuild the committed bundle
git diff --stat app/data/                  # expect: no change (deterministic)
uv run pytest -q                           # 362 passing, synthetic fixtures
uv run ruff check src tests app
```

## 8. Interpretation limits

The shares and metrics are **descriptive variance attribution, not causal
climate attribution**. In particular the pipeline **cannot** support: causal
claims about emissions→warming (CO₂ is well-mixed); policy counterfactuals;
impact/exposure inequality (the outcome is mean annual-trend warming, which
*understates* climate injustice — impacts concentrate in low-emitting tropical
countries); or within-country inequality (ecological aggregation). The dominant
threats to validity are **station sampling bias** and **axis collinearity**
(Shapley handles the latter *fairly* but cannot *separate* genuinely entangled
axes). The full "can/cannot support" and threats list is
`docs/decomposition_design_memo.md` §6–§7; the limitations are summarized in the
README and `docs/key_findings.md`.

## 9. Clean-clone quickstart

```bash
git clone https://github.com/it-malek/climate-inequality.git
cd climate-inequality
uv sync --extra dev
uv run pytest -q                              # all green, no data needed
uv run streamlit run app/streamlit_app.py     # dashboard from the committed bundle

# Optional full rebuild (downloads ~500 MB of public data):
uv run python -c "from src.data_io import download_raw_data; download_raw_data()"
uv run python -c "from src.data_io import load_city_temperatures, city_csv_path; load_city_temperatures(city_csv_path())"
uv run python -m src.trends && uv run python -m src.emissions && uv run python -m src.interpolate
uv run python -m src.validation && uv run python -m src.explain
uv run python -m src.app_assets               # regenerates app/data/ deterministically
```
