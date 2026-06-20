# Climate Inequality: Mapping Where Warming Hits Hardest

> Which regions are warming fastest — and is warming proportional to
> emissions responsibility?

**Status:** ✅ Complete and **deployed:
[climate-inequality.streamlit.app](https://climate-inequality.streamlit.app/)**.
The pipeline runs from the raw data through trend fitting, spatial
interpolation, the emissions join, out-of-sample validation and explanatory
variables to the headline **cross-country variance decomposition** — a
group-level Shapley/LMG attribution over the frozen `SCHEMA_V1` feature
contract. Design and scope boundary:
[`docs/decomposition_design_memo.md`](docs/decomposition_design_memo.md);
end-to-end reproduction: [`docs/reproducibility.md`](docs/reproducibility.md);
results: [`docs/key_findings.md`](docs/key_findings.md) /
[`docs/research_summary.md`](docs/research_summary.md); next-phase roadmap:
[`docs/stability_roadmap.md`](docs/stability_roadmap.md) and
[`docs/future_work.md`](docs/future_work.md).

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
- **Variance decomposition (the headline).** The cross-country variance in
  country-mean warming trend is attributed to four *structural axes* —
  emissions, geography, socioeconomic development, population — plus an
  explicit residual, using a group-level LMG / Shapley-Owen R² decomposition
  (each axis's incremental R² averaged over every ordering, so correlated
  axes split their shared variance fairly rather than by entry order). The
  axes are fixed by a frozen feature contract (`src/feature_schema.py`,
  `SCHEMA_V1`); nothing outside the contract may enter a model. The shares are
  **descriptive variance attribution, not causal climate attribution** — there
  is deliberately no physical/driver layer (see the design memo's scope
  boundary). This demotes the earlier single "emissions coefficient" to one
  axis among several.

## Findings

**The headline: cross-country warming inequality is overwhelmingly geographic.**
A group-level Shapley/LMG decomposition of the variance in country-mean warming
trend (n = 154 countries, total R² = 0.63) attributes **46% of total variance
to physical geography** (latitude foremost), **8% to emissions
responsibility**, 6% to socioeconomic development and 4% to
population/urbanization, leaving a **37% residual** that none of the four axes
explains. Of the *explained* variance, geography is ≈72% and emissions ≈13%.
Emissions' standalone R² is 0.13, but most of that overlaps geography once both
are present — so the decomposition turns the latitude-vs-emissions confound
into a *measured overlap* rather than a caveat. The shares are a descriptive
variance attribution, **not** causal climate attribution (CO₂ is well-mixed; a
country's emissions do not preferentially heat its own territory). Full numbers
and interpretation: [`docs/key_findings.md`](docs/key_findings.md).

**Warming is universal across the sample, but far from uniform.** Every one
of the 3,510 city-locations shows a positive 1950–2013 trend, and for 99.1%
of them the entire Theil–Sen 95% confidence interval sits above zero. The
land mean is **0.146 °C/decade**; the middle half of locations spans
0.11–0.18, and the tails differ by a factor of four (5th percentile 0.06,
95th percentile 0.25 °C/decade).

**The fastest warming is at high northern latitudes — with a notable
mid-latitude hotspot.** Mean trends rise with northern latitude: 0.10
°C/decade in the northern tropics, 0.15 at 23.5–45°N, 0.21 at 45–60°N, and
**0.23 °C/decade above 60°N — 1.56× the land mean**, the expected
Arctic-amplification signature. (The textbook ~2× ratio is measured against
the *land+ocean* global mean; a land-only baseline is itself elevated, which
compresses the ratio, and only 25 stations sit above 60°N.) The single
fastest-warming cluster is not Arctic, though: the Iranian plateau and
Central Asia (Mashhad, Herat, Ashgabat — up to 0.34 °C/decade), alongside
Siberia's Norilsk (0.33). The slowest warming is along the southern coast
of China (~0.02–0.03 °C/decade).

**Countries with greater historical emissions responsibility warm faster.**
Across 157 countries, country-mean warming correlates positively with
cumulative per-capita CO₂ through 2013 (Spearman ρ = +0.36, p ≈ 4×10⁻⁶).
The top emissions quartile warmed 0.196 °C/decade on average against 0.149
in the bottom quartile. Regressing warming on log₁₀ emissions gives
**+0.021 °C/decade per tenfold increase in cumulative per-capita CO₂**
(95% CI [+0.013, +0.029], HC1 robust SEs) pooled, and **+0.029** (95% CI
[+0.014, +0.045], p = 2×10⁻⁴, R² = 0.27) with continent fixed effects —
the relationship holds *within* continents, not just between them.

**Interpretation: this is geography, not local retribution.** CO₂ is
well-mixed, so a country's own emissions do not preferentially heat its own
territory. The association arises mostly because historically high-emitting
industrialized countries sit at the mid-to-high northern latitudes where
amplification is strongest, while the lowest cumulative emitters (Burundi:
0.9 t/person, vs ~1,180 t/person for the US and UK — a 1,300-fold range)
cluster in the slower-warming tropics. Continent fixed effects absorb part,
but not all, of that latitude confound. The honest headline is therefore:
**the countries most responsible for cumulative emissions are not escaping
warming — if anything their territories warm faster in annual-mean terms —
yet the temperature trend alone understates climate inequality**, because
impacts scale with heat exposure, vulnerability, and adaptive capacity,
which concentrate in low-emitting tropical countries (see Limitations and
`docs/future_work.md`).

**A methodological finding worth keeping:** naive leave-one-out CV reverses
the interpolation-method ranking. Berkeley Earth's grid-snapped coordinates
put 2,821 of the 3,510 locations into 677 shared-coordinate groups; with
plain row-wise folds, a held-out point's coordinate twin stays in the
training set and leaks a near-exact answer (IDW RMSE 0.0057 vs kriging
0.0160 °C/decade — a property of the duplicates, not the methods). Holding
out entire coordinate groups removes the leak and most of the gap (0.0083
vs 0.0099); IDW still wins, but narrowly and for honest reasons.

## Dashboard

**Live at [climate-inequality.streamlit.app](https://climate-inequality.streamlit.app/).**

Streamlit app in two sections. **Decomposition** (the headline): the
inequality-decomposition page — Gini/Theil of country warming, the Shapley
variance-share bar, and an emissions-only / geography-only / full-model
explorer; the country warming map; a "How confident are we?" panel that
scaffolds the forthcoming stability diagnostics (it renders an explicit pending
state until the stability layer ships — see `docs/stability_roadmap.md`); and a
**Responsibility vs impact** page (Layer 3) comparing each country's emissions
responsibility against its warming exposure — Spearman ρ, the Lorenz/Gini
inequality coefficient, and the mismatch leaders.
**Foundations** (how the inputs were built): the interpolated warming surface
with a city-station toggle; a city explorer (any location's anomaly series and
fitted trend); the emissions-vs-warming scatter; an out-of-sample validation
page (did the 1950–2013 trends hold post-2013?); and a drivers page (what
explains where warming is fastest?). A fixed interpretation banner on every
page states the variance-attribution-only boundary.

```bash
uv run streamlit run app/streamlit_app.py
```

The app reads only the committed `app/data/` bundle (~5 MB, built by
`uv run python -m src.app_assets`), so it runs identically on a fresh
clone and on Streamlit Community Cloud — no raw-data download needed.

**Deployment:** the app is deployed from `it-malek/climate-inequality`,
branch `main`, main file path `app/streamlit_app.py`, Python 3.11. The
cloud environment installs `app/requirements.txt` (the dependency file in
the entrypoint's directory takes precedence over root-level files) and
redeploys automatically on push to `main`.

## Limitations

- **Station sampling bias.** Trends exist only where Berkeley Earth has
  city series: dense in populous mid-latitudes, sparse over the Arctic
  (25 locations >60°N), the Sahara, Amazonia, and Siberia. Country means
  are unweighted across city-locations — station-weighted, not area- or
  population-weighted — because the project datasets carry no city
  populations.
- **Land-only, ending September 2013 (analysis window).** Ocean warming is
  absent (which is also why the Arctic ratio reads 1.56× rather than the
  canonical ~2×). Post-2013 Berkeley Earth gridded data is used in Phase 6
  to validate the fitted trends; warming has accelerated beyond what the
  1950–2013 lines forecast (see the "Did the trends hold?" dashboard page).
- **Grid-snapped coordinates.** Multiple cities share identical lat/lon,
  and (City, Country) is not a unique key; the pipeline keys on the full
  coordinate identity and the CV holds out whole coordinate groups, but
  the underlying location precision is still ~1°.
- **Measurement uncertainty not propagated.** The dataset's
  `AverageTemperatureUncertainty` column is not used to weight the trend
  fits; early-window months are noisier than recent ones.
- **Emissions accounting choices.** Responsibility = cumulative
  *production-based, territorial* CO₂ — no consumption-based correction
  for traded goods, no land-use-change CO₂, no non-CO₂ gases — divided by
  *2013* population, which understates the historical responsibility of
  countries whose populations have since grown slowly (and vice versa).
  Puerto Rico and Réunion drop out (no OWID series; 157 of 159 countries
  matched).
- **Correlation, not attribution.** Warming at any location is driven by
  global forcing plus regional dynamics, not by that country's own
  emissions; the latitude/industrialization confound is only partially
  absorbed by continent fixed effects. The regression describes who
  experiences faster warming, not what caused it locally.
- **Smooth surfaces.** The IDW surface (k=30, 2° grid) understates local
  variability between stations, and the CV RMSE measures how well the
  *trend field* interpolates — not the accuracy of the trends themselves.

## Future work

A prioritized roadmap lives in [`docs/future_work.md`](docs/future_work.md):
switching to Berkeley Earth's gridded product, population-weighted
exposure, consumption-based emissions and Lorenz/Gini inequality framings,
gradient-boosted trees + SHAP vs the Phase 7 OLS, and extreme-heat
(rather than mean) warming inequality using GHCN-Daily data.

## Reproducing

```bash
uv sync --extra dev
uv run pytest    # no data needed — tests run on synthetic fixtures
uv run ruff check src tests app

# Build the data layer (public Kaggle datasets, no credentials needed):
uv run python -c "from src.data_io import download_raw_data; download_raw_data()"
uv run python -c "from src.data_io import load_city_temperatures, city_csv_path; load_city_temperatures(city_csv_path())"

# Pipeline modules (each prints its sanity checks):
uv run python -m src.trends         # city_trends.parquet
uv run python -m src.interpolate    # outputs/trend_surface.html
uv run python -m src.emissions      # country_inequality.parquet + scatter
uv run python -m src.validation     # data/processed/validation_*.{json,parquet}
uv run python -m src.explain        # city_features.parquet + explain_*.{json,parquet}
uv run python -m src.feature_schema # regenerate docs/feature_schema_v1.yaml mirror
uv run python -m src.inequality     # inequality_summary.json (Gini/Theil)
uv run python -m src.decomposition  # decomposition_summary.json (Shapley/LMG)
uv run python -m src.pcs            # regenerate docs/pcs_v1.yaml (PCS v1 registry mirror)
uv run python -m src.projections    # projections_v1.parquet (PCS v1 instantiation)
uv run python -m src.coupling       # coupling_summary.json + coupling.parquet (Layer 3)
uv run python -m src.app_assets     # app/data/ bundle — folds in ALL of the above

uv run streamlit run app/streamlit_app.py
```

`src.app_assets` is the single bundle builder: it recomputes the trends,
interpolation and inequality assets *and* (when the upstream `city_features`
and income inputs exist) the headline `inequality_summary.json` and
`decomposition_summary.json`, so one `python -m src.app_assets` regenerates the
entire committed `app/data/` bundle.

The dashboard bundle is committed, so the app (and its tests) work without
any of the above data steps. On a fresh machine, start with
`docs/new_machine_setup.md`.

## Data

- [Berkeley Earth — Climate Change: Earth Surface Temperature Data](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
- [Our World in Data — CO₂ and Greenhouse Gas Emissions](https://github.com/owid/co2-data)
- [Natural Earth — 110m land polygons](https://www.naturalearthdata.com/downloads/110m-physical-vectors/) (land mask)
