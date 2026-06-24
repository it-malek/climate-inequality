# Autonomous decision log

Running log of decisions made without live sign-off during autonomous sessions:
what was chosen, why, and which alternatives were passed over. The intent is a fast
audit trail for review — not a design doc (those live elsewhere in `docs/`).

## 2026-06-24 — Exposure × vulnerability ("triple inequality") lens

Branch `feat/exposure-vulnerability-lens` (off `feat/era5-cross-check`). The three
design forks (per-person headline, permutation-p significance, all-three lenses)
and the branch base were decided *with* the user before the overnight run; the
items below are the smaller calls made solo under the "executive autonomy"
standing instruction.

| # | Decision | Why | Alternatives passed over |
|---|----------|-----|--------------------------|
| 1 | Log lives at `docs/ai_decisions.md`, not `doc/ai_decisions.md` as literally requested. | The repo keeps every markdown doc under `docs/`; `doc/` does not exist. Following the existing convention beats creating a one-off directory. | Literal `doc/` path (would fragment the docs tree). |
| 2 | New module `src/vulnerability.py` mirrors `src/era5_validation.py` (pure `compute_*` + best-effort `build_*` writing a `round_floats` JSON + a slim parquet), **not** the registry-bound `src/coupling.py` comparator. | Income is an ordinal *stratifier*, not a projection; PCS forbids composite projections and must stay frozen. era5_validation is the established "cross-check artifact, not a PCS projection" pattern. | A new PCS_V2 projection (violates the frozen-registry invariant); folding into `coupling.py` (would force it through `validate_projection_frame`). |
| 3 | Triple-inequality "verdict" is **computed** from the two gradients (responsibility rises significantly with income *and* area-warming is weaker + non-significant), not a hardcoded narrative string. | Keeps the summary descriptive and reproducible; no editorial claim baked into an artifact. | A prose conclusion (fragile, non-reproducible, drifts from the numbers). |
| 4 | Significance = scipy asymptotic Spearman p **plus** a deterministic label-permutation p (seeded, 999 perms), reusing the `morans_i` permutation idiom. Conley/spatial-HAC deliberately **not** built. | (Confirmed with user.) Spatial-HAC corrects a continuous OLS slope; the lens's object is a rank statistic over four ordinal strata, where a label permutation is the honest small-n tool. Conley belongs to the regression modules as its own PR. | Asymptotic p alone (optimistic at small n); building Conley first (out of scope, wrong target for a rank stat). |
| 5 | Per-tier warming reported as **population-weighted (per-person)** headline *and* per-country mean/median; the pop weight is a flat **count** weight, never cos(lat). | (Headline confirmed with user.) "Who suffers" = people. The intensive/extensive mirror: cos(lat) already lives inside the area lens at grid level; averaging country trends across a tier is an extensive people-count average (the GPW count rule). | cos(lat) on the tier average (double-counts meridian convergence, corrupts the science); per-country only (loses the per-person framing). |
| 6 | Added an `inequality_path` parameter to `build_vulnerability_asset`. | Lets the bundle-asset wiring be tested fully hermetically (synthetic income CSV + synthetic inequality), satisfying the "synthetic fixtures only" TDD rule. | Hardcoding `DEFAULT_INEQUALITY_PATH` (forces the wiring test to read the committed real parquet). |
| 7 | Two new chart builders added to `app/charts.py` (`income_gradient_chart`, `income_strata_box`) rather than inline plotly in the view. | Matches the repo split — `charts.py` holds all unit-testable `go.Figure` builders; views hold only Streamlit. | Inline plotly in the view (untestable, breaks the established separation). |
| 8 | Page placed in the **Decomposition** nav group, directly after "Responsibility vs impact". | It is the narrative successor ("who suffers, not who warms"); the upstream data pages stay under "Foundations". | A new top-level group (premature); under Foundations (buries the headline). |
| 9 | Built the real `vulnerability_summary.json` + `vulnerability_strata.parquet` into the committed `app/data/` bundle, computed off `app/data/country_inequality.parquet`. | The income CSV is in-repo, so the lens is always buildable; the committed bundle already ships the other optional lens artifacts (e.g. era5). Built from the bundle's own inequality copy for self-consistency. JSON is `round_floats` byte-stable; the parquet is on-platform deterministic (same status as the committed `coupling.parquet`). | Leaving the bundle without the lens (deployed dashboard would show only the pending state); full `python -m src.app_assets` rebuild (rewrites `stats.json` timestamp + risks drift in unrelated committed files — noisy diff). |

**Result on real data (157 countries):** responsibility vs income ρ = +0.885
(perm p = 0.001); area-weighted warming vs income ρ = −0.145 (perm p = 0.06, n.s.);
per-person, low-income countries warm the *most* (+0.205 vs +0.182 °C/decade for
high-income). The triple inequality **holds** under the de-artifacted lens.
