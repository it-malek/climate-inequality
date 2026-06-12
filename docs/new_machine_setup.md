# New machine bootstrap

GitHub is the single sync channel between machines:
`https://github.com/it-malek/climate-inequality`

Both machines work from normal git clones:

- **PC — WSL: a clone in the Linux filesystem.** Section A.
- **Mac — a clone in a local folder outside iCloud.** Section A.
  (The old iCloud-synced copy is retired; Section B is kept only in
  case it ever needs recovering.)

## A. Fresh clone (WSL / Linux / macOS — any normal filesystem)

In WSL, keep the repo in the Linux filesystem (e.g. `~/projects/`),
never under `/mnt/c/...` — the Windows mount is slow and breaks the
filesystem semantics dev tools expect.

On the Mac, clone somewhere iCloud does not touch — `~/projects/` is
safe; `~/Documents/` and `~/Desktop/` are NOT if "Desktop & Documents"
iCloud sync is on.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv (macOS: brew install uv)
git clone https://github.com/it-malek/climate-inequality.git
cd climate-inequality
uv sync --extra dev      # uv installs Python 3.11+ itself if needed
uv run pytest -q         # expect all green (92 tests as of phase-4)
uv run ruff check src tests
```

Rebuild the gitignored data layer — public Kaggle datasets plus OWID
pulls, no credentials needed:

```bash
uv run python -c "from src.data_io import download_raw_data; download_raw_data()"
uv run python -c "from src.data_io import load_city_temperatures, city_csv_path; print(load_city_temperatures(city_csv_path()))"
uv run python -m src.trends       # phase-2 artifact: city_trends.parquet
uv run python -m src.emissions    # phase-4 artifacts: country_inequality.parquet + scatter
uv run python -m src.interpolate  # phase-3 artifact: trend_surface.html (~20 s)
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

## B. The retired iCloud copy (recovery notes only)

The Mac previously worked from a copy inside iCloud Drive, which
corrupted `.git` more than once (HEAD/config/index/objects silently
vanished) and made the first run of the day glacial while evicted
files re-downloaded. That copy is retired — do not run sessions in it.

If git reports corruption or "not a git repository" even though `.git/`
exists, do NOT `git init` or `reset --hard`. Recovery that worked
before (never touches the working tree):

```bash
printf 'ref: refs/heads/main\n' > .git/HEAD
printf '[core]\n\trepositoryformatversion = 0\n\tbare = false\n' > .git/config
git remote add origin https://github.com/it-malek/climate-inequality.git
git fetch origin
git update-ref refs/heads/main origin/main
git branch --set-upstream-to=origin/main main
git reset --quiet
```

Optional hardening for the Mac: keep the venv out of iCloud by adding
`export UV_PROJECT_ENVIRONMENT=~/.venvs/climate-inequality` to the
shell profile (then `uv sync --extra dev` and delete the in-repo
`.venv/`).

## C. Two-machine discipline

- The working trees do NOT sync with each other — only GitHub connects
  them. Start every session with `git pull`; end with commit + push.
  Unpushed work exists on exactly one machine.
- Never run working sessions on both machines at once.
- `data/` is gitignored and machine-local — rebuild it per machine
  (section A); never copy it between machines.
