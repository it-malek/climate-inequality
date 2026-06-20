# Historical local setup

Commands and reported values here belong to this historical revision.
Personal machine-recovery and session instructions have been omitted.

## Fresh clone (WSL / Linux / macOS, any normal filesystem)

In WSL, keep the repo in the Linux filesystem (e.g. `~/projects/`),
never under `/mnt/c/...`, the Windows mount is slow and breaks the
filesystem semantics dev tools expect.

On the Mac, clone somewhere iCloud does not touch, `~/projects/` is
safe; `~/Documents/` and `~/Desktop/` are NOT if "Desktop & Documents"
iCloud sync is on.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv (macOS: brew install uv)
git clone https://github.com/it-malek/climate-inequality.git
cd climate-inequality
uv sync --extra dev      # uv installs Python 3.11+ itself if needed
uv run pytest -q         # expect all green (362 tests; synthetic fixtures, no data)
uv run ruff check src tests app
```

The dashboard works immediately on a fresh clone, it reads only the
committed `app/data/` bundle, no data rebuild required:

```bash
uv run streamlit run app/streamlit_app.py
```

Rebuild the gitignored data layer, public Kaggle datasets plus OWID
pulls, no credentials needed:

```bash
uv run python -c "from src.data_io import download_raw_data; download_raw_data()"
uv run python -c "from src.data_io import load_city_temperatures, city_csv_path; print(load_city_temperatures(city_csv_path()))"
uv run python -m src.trends       # phase-2 artifact: city_trends.parquet
uv run python -m src.emissions    # phase-4 artifacts: country_inequality.parquet + scatter
uv run python -m src.interpolate  # phase-3 artifact: trend_surface.html (~20 s)
uv run python -m src.app_assets   # phase-5: regenerates the committed
                                  # app/data/ bundle, only needed after a
                                  # pipeline change (commit the result)
```

Expected (README sanity checks): `src.trends` → ~3,510
city-locations, global mean ≈ 0.146 °C/decade, >60°N ≈ 0.228 (ratio
≈ 1.56×). `src.emissions` → 157 countries, Spearman ρ ≈ +0.36,
continent-FE OLS ≈ +0.029 °C/decade per 10× emissions.
`src.interpolate` → IDW wins leave-location-out CV 0.0083 vs 0.0099.
The Kaggle download is ~500 MB and the DuckDB load takes a few
minutes; don't kill slow-looking steps.

Line endings are normalized to LF by `.gitattributes`; no autocrlf
configuration is needed on any platform.

