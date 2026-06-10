# Climate Inequality: Mapping Where Warming Hits Hardest

> Which regions are warming fastest — and is warming proportional to
> emissions responsibility?

**Status:** 🚧 In progress — Phase 1 (data layer). See `README` for the
full project plan.

*Originally proposed as an undergraduate research fellowship project (2022);
rebuilt and substantially upgraded in 2026.*

## Question

Global warming is not geographically uniform. This project (1) quantifies
per-city warming rates from 1950–2013 using the Berkeley Earth surface
temperature dataset, (2) interpolates those rates into continuous spatial
surfaces, and (3) tests whether the countries warming fastest are the ones
most responsible for cumulative CO₂ emissions.

## Methods

<!-- Filled in as phases complete. Outline: -->
- **Trends:** monthly temperature anomalies (relative to per-city 1951–1980
  climatology), per-city Theil–Sen slopes in °C/decade.
- **Interpolation:** IDW baseline vs ordinary kriging, evaluated with
  leave-one-out cross-validation.
- **Inequality analysis:** country-level warming rates joined against Our
  World in Data cumulative per-capita CO₂.

## Findings

<!-- TODO: headline result with effect size and uncertainty -->

## Dashboard

<!-- TODO: Streamlit Community Cloud link -->

## Limitations

<!-- TODO: station coverage bias, dataset ends Sept 2013, land-only
interpolation, grid-snapped coordinates -->

## Reproducing

```bash
uv sync          # or: pip install -e ".[dev]"
# Kaggle credentials required for data download — see docs/setup.md
pytest
```

## Data

- [Berkeley Earth — Climate Change: Earth Surface Temperature Data](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
- [Our World in Data — CO₂ and Greenhouse Gas Emissions](https://github.com/owid/co2-data)
