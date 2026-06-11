# New machine bootstrap

For any fresh machine (e.g., the Windows PC) opening this repo from the
iCloud-synced folder. **GitHub (`origin`) is the source of truth — iCloud
is just a folder, and it has corrupted `.git` here before.** Do all of
this before any phase work.

## 1. Trust git, not iCloud

```
git fetch origin
git status -sb
git log --oneline -3
```

Expect a clean tree on `main` in sync with `origin/main` (history:
`Initial commit` → `phase-1: ...` → `phase-2: ...`). If git reports
corruption or "not a git repository" even though `.git/` exists, iCloud
ate it again. Recovery that worked before (never touches the working
tree — do NOT `git init` or `reset --hard`):

```
printf 'ref: refs/heads/main\n' > .git/HEAD
printf '[core]\n\trepositoryformatversion = 0\n\tbare = false\n' > .git/config
git remote add origin https://github.com/it-malek/climate-inequality.git
git fetch origin
git update-ref refs/heads/main origin/main
git branch --set-upstream-to=origin/main main
git reset --quiet
```

(Or simply re-clone to a non-iCloud path and copy uncommitted work over.)

## 2. Windows only: stop line-ending rewrites

```
git config core.autocrlf false
```

Without this, every file shows as modified and CRLF churn syncs back to
the Mac. Line-ending-only diffs in `docs/original_proposal.txt` from
before this rule are safe to discard.

## 3. Install uv

- Windows (PowerShell):
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- macOS: `brew install uv`

## 4. Keep the virtualenv OUT of iCloud

A `.venv/` may exist in the repo — it belongs to another machine's OS and
must be neither used nor deleted from here. Point uv at a local,
non-synced path instead:

- Windows (PowerShell, then open a new terminal):
  `setx UV_PROJECT_ENVIRONMENT "$env:USERPROFILE\.venvs\climate-inequality"`
- macOS (shell profile):
  `export UV_PROJECT_ENVIRONMENT=~/.venvs/climate-inequality`

Then from the repo root (uv auto-installs Python 3.11+ if needed):

```
uv sync --extra dev
```

## 5. Verify the suite

```
uv run pytest -q          # expect: all green (49 tests as of phase-2)
uv run ruff check src tests
```

## 6. Rebuild data locally — don't wait for iCloud

`data/` is gitignored; re-creating ~700MB locally is faster and more
reliable than iCloud materializing it. Kaggle download needs no
credentials. From the repo root:

```
uv run python -c "from src.data_io import download_raw_data; download_raw_data()"
uv run python -c "from src.data_io import load_city_temperatures, city_csv_path; print(load_city_temperatures(city_csv_path()))"
uv run python -m src.trends
```

Expected from the last command (README sanity checks):
~3,510 city-locations; global mean ≈ 0.146 °C/decade; >60°N ≈ 0.228
(ratio ≈ 1.56×).

## 7. Two-machine discipline

- Never run working sessions on both machines at the same time — iCloud
  resolves concurrent edits by making "conflicted copy" duplicates.
- Start every session with `git fetch` + `git status`; end every session
  with commit + push. The remote is the only durable copy.
