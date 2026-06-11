# Historical local setup

Commands and reported values here belong to this historical revision.
Personal machine-recovery and session instructions have been omitted.

## Fresh clone (WSL / Linux / any normal filesystem)

In WSL, keep the repo in the Linux filesystem (e.g. `~/projects/`),
never under `/mnt/c/...`, the Windows mount is slow and breaks the
filesystem semantics dev tools expect.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv (macOS: brew install uv)
git clone https://github.com/it-malek/climate-inequality.git
cd climate-inequality
uv sync --extra dev      # uv installs Python 3.11+ itself if needed
uv run pytest -q         # expect all green (49 tests as of phase-2)
uv run ruff check src tests
```

Rebuild the gitignored data layer, public Kaggle datasets, no
credentials needed:

```bash
uv run python -c "from src.data_io import download_raw_data; download_raw_data()"
uv run python -c "from src.data_io import load_city_temperatures, city_csv_path; print(load_city_temperatures(city_csv_path()))"
uv run python -m src.trends
```

Expected from the last command (README sanity checks): ~3,510
city-locations; global mean ≈ 0.146 °C/decade; >60°N ≈ 0.228
(ratio ≈ 1.56×).

Line endings are normalized to LF by `.gitattributes`; no autocrlf
configuration is needed on any platform.

