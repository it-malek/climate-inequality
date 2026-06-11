# New machine bootstrap

GitHub is the single sync channel between machines:
`https://github.com/it-malek/climate-inequality`

Two working copies exist:

- **PC — WSL (recommended pattern): a normal git clone.** Section A.
- **Mac — the iCloud-synced copy.** Needs special care. Section B.

## A. Fresh clone (WSL / Linux / any normal filesystem)

In WSL, keep the repo in the Linux filesystem (e.g. `~/projects/`),
never under `/mnt/c/...` — the Windows mount is slow and breaks the
filesystem semantics dev tools expect.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv (macOS: brew install uv)
git clone https://github.com/it-malek/climate-inequality.git
cd climate-inequality
uv sync --extra dev      # uv installs Python 3.11+ itself if needed
uv run pytest -q         # expect all green (49 tests as of phase-2)
uv run ruff check src tests
```

Rebuild the gitignored data layer — public Kaggle datasets, no
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

## B. The Mac's iCloud copy (special care)

The Mac working copy lives inside iCloud Drive, which has corrupted
`.git` here before (HEAD/config/index/objects silently vanished) and
makes the first run of the day glacial while evicted files re-download
— don't kill slow-looking commands.

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
