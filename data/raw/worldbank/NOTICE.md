# World Bank income classification — vendored copy

`world-bank-income-groups.csv` is Our World in Data's mirror of the World Bank
income classification (one row per country-year, 1987–2024; columns `Entity`,
`Code`, `Year`, `World Bank's income classification`).

- Source: <https://ourworldindata.org/grapher/world-bank-income-groups>
- Retrieved: 2026-06
- Licence: Our World in Data publishes its grapher data under CC BY 4.0; the
  underlying classification is the World Bank's.

The pipeline uses each country's most recent classification. The file is
committed (the rest of `data/raw/` is gitignored) so that the socioeconomic
axis of the decomposition and the income stratification are pinned to one
vintage; `src.explain.main()` only downloads it when it is absent.
