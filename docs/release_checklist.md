# Release checklist — v1.0

**Verdict: release-candidate ready.** The repository is reproducible, tested,
documented, and deployed, with a clearly-scoped roadmap for the next research
phase. The remaining items are non-blocking polish/infrastructure, listed below.

Assessed 2026-06-19 on branch `schema/x-schema-v1`.

---

**v1.1 update (2026-06-20).** `v1.0.0` is tagged on the PR #3 merge. Two additive,
in-bounds layers have since shipped on `main` and are released as **v1.1.0**: the
**stability diagnostics** page (bootstrap CIs on the Shapley shares,
leave-one-country-out influence, Moran's I on the residual) and the **Layer-3 PCS
responsibility–impact coupling** page. No `SCHEMA_V1` / PCS change; `pytest` 362
green; the `round_floats`-hardened JSON summaries (`inequality`, `decomposition`,
`coupling_summary`) rebuild byte-for-byte and the bundle is deterministic on a
fixed platform. **Cross-platform caveat:** the unrounded `coupling.parquet`
(content identical to ~1e-14) and the residual Moran's I (kNN tie-breaking) drift
between Linux and macOS — a hardening follow-up (item 5 in Remaining issues). See
the v1.1 sign-off steps at the end.

## Readiness gates (all green)

| Gate | Status | Evidence |
|------|:---:|----------|
| Tests pass | ✅ | `pytest` 362 passed (synthetic fixtures, no data, no network) |
| Lint clean | ✅ | `ruff check src tests app` clean |
| Builds from clean clone | ✅ | dashboard reads committed `app/data/`; tests need no data — `docs/reproducibility.md` §1 |
| Bundle is reproducible | ✅ | `python -m src.app_assets` regenerates the full bundle **deterministically** (byte-stable JSON via `round_floats`) |
| Single build command | ✅ | `src.app_assets` now folds in the headline inequality + decomposition summaries (was a manual copy) |
| CI configured | ✅ (new) | `.github/workflows/ci.yml` runs ruff + pytest on push/PR, pinned to the 3.11 deploy target |
| Scope boundary enforced in code | ✅ | `SCHEMA_V1` contract + `validate_design_matrix`; `INTERPRETATION_NOTE` travels with every summary |
| Deployed & live | ✅ | climate-inequality.streamlit.app, auto-redeploys on push to `main` |

## Publication-readiness assessment

**Ready, as a descriptive cross-sectional study.** Strengths:

- The scientific framing is honest and enforced: a structural *variance
  attribution*, never causal/physical attribution, with the boundary stated in
  the design memo, the dashboard banner, and every JSON artifact.
- Methods are reproducible end to end (`docs/reproducibility.md`) with built-in
  integrity checks (build-time trend consistency, schema enforcement,
  deterministic serialization).
- The narrative is complete: `docs/research_summary.md` (the study),
  `docs/key_findings.md` (the numbers), `docs/decomposition_design_memo.md`
  (rationale + pre-registration), `docs/project_audit.md` (this pass).
- Pre-registered hypotheses (H1–H5) are stated *and* the results reported against
  them — geography dominant, emissions small-but-nonzero, residual substantial.

The honest framing of what the project **cannot** claim (impact/injustice
inequality, causality, within-country structure) is a publication strength, not a
weakness — it is stated prominently rather than buried.

## Dashboard-readiness assessment

**Ready.** The landing (decomposition) page now answers all four first-visit
questions above the fold — *what was measured*, *how unequal*, *what explains it*,
and *what it can/cannot claim* (a dedicated panel) — grounded in the real
country-level spread. Strengths: centralized colorblind-safe visual language
(`app/theme.py`), an always-on interpretation banner, rich chart hovers and
per-chart captions, graceful "not built yet" states for deferred pages, and a
two-section nav (Decomposition / Foundations) with sidebar orientation. Streamlit
API usage is consistent (`width="stretch"` throughout). The "How confident are
we?" page now renders the shipped stability diagnostics — bootstrap CIs on the
Shapley shares (geography largest in 100% of resamples), leave-one-country-out
influence per share, and Moran's I on the residual (I ≈ 0.33, p = 0.005); the
remaining legacy-coefficient blocks stay roadmapped in `docs/stability_roadmap.md`.

## Remaining issues (non-blocking)

| # | Item | Severity | Notes |
|---|------|:---:|-------|
| 1 | ~~Verify the new CI workflow's first remote run~~ | — | ✅ Confirmed green on PR #3 (push + pull_request, ~30s). Pin `astral-sh/setup-uv` if the major tag ever drifts |
| 2 | Plotly `locationmode="ISO-3"` migration | Low | `app/charts.py:117` emits a `country names` deprecation warning; needs a vetted country→ISO-3 map (mind the Réunion/Puerto Rico edge cases) |
| 3 | Notebook outputs are committed | Low | Now documented as exploratory (`notebooks/README.md`); optionally strip with `nbstripout` |
| 4 | `app/requirements.txt` ↔ `uv.lock` drift | Low | Keep the pinned cloud stack in step with the lock on dependency bumps (already noted in the file) |
| 5 | Cross-platform bundle byte-stability | Low–Med | The `round_floats` JSON summaries reproduce byte-for-byte across platforms; the unrounded `coupling.parquet` (data identical to ~1e-14) and the residual Moran's I (platform-dependent kNN tie-breaking, ~1e-4) do not — confirmed deterministic *on-platform* (Linux-built commit vs macOS rebuild). Fix: extend `round_floats` to the `coupling.parquet` float columns and make kNN neighbor selection deterministic (sort by distance, then stable id); or pin bundle builds to the Linux/CI target. Mirrors audit finding #2. |

None blocks a v1.0 tag.

## Recommended future work

- **Stability layer — ✅ built** (`src/stability.py`, `app/data/stability_summary.json`,
  `docs/stability_roadmap.md`): bootstrap + block-bootstrap CIs on the Shapley shares
  (geography largest in 100% of resamples; P(emissions > 0) = 1.0), leave-one-country-out
  influence, and Moran's I on the country residual (I ≈ 0.33, p = 0.005), shipped with the
  sensitivity dashboard page.
- **Also built since v1.0** — Layer 1 physical-drivers model; the Layer 3 PCS v2 Wide
  Registry lenses (**consumption-based** responsibility, **people-weighted exposure** on
  SEDAC GPW v4) and the Lorenz/Gini injustice framing.

- **⏭️ Next research phase — area-weighted gridded warming (v1.2).** The project's #1
  external-validity gap is *station sampling bias*: country means are station-weighted, so
  dense mid-latitude clusters dominate and the Arctic/Sahara/Amazonia are under-sampled.
  Replace it with TRUE **area-weighted** country means from the Berkeley Earth 1°×1° gridded
  product — already in-repo (`data/raw/berkeley_gridded/Complete_TAVG_LatLong1.nc`, read by
  `src/validation.py`), **cos(latitude)**-weighted (required here: temperature is *intensive*,
  the mirror of the GPW population-count rule), with cells assigned to countries via the
  GPW v4 National Identifier Grid band (no new polygon dataset). Add it as a third L3 impact
  lens (`impact_index_area_weighted`) in the PCS v2 Wide Registry and re-test whether the
  climate-inequality conclusions (ρ ≈ 0.36, Gini ≈ 0.56, the Central-Asia mismatch leaders)
  survive when every km² counts equally. Reuses `src/population.py`
  `latitude_area_weights` / `area_weighted_mean`. See `docs/future_work.md` §2.
- **Then — extreme-heat (vs mean) inequality** (`docs/future_work.md` §4): extreme-heat
  *days* via GHCN-Daily, a new outcome dimension — but the data is **not in-repo** and is
  large, so scope download/storage first (lower priority than the zero-friction gridded epic).
- **Other open items** (`docs/future_work.md`): ERA5 cross-check, exposure × vulnerability
  (ND-GAIN), era-weighted responsibility, gradient-boosted trees + SHAP vs the Phase 7 OLS.

## Sign-off steps for a tagged v1.0

1. ✅ Merge this branch's reproducibility + docs + dashboard work to `main`
   (PR #3).
2. ✅ CI run green on the PR (push + pull_request).
3. ✅ Confirm the Streamlit Cloud redeploy from `main` renders the updated
   landing page.
4. ✅ Tag `v1.0` once #3 is confirmed — `v1.0.0` annotated on the PR #3 merge.

## Sign-off steps for a tagged v1.1

1. ⏳ Merge the stability + L3-coupling doc-alignment to `main` (this branch).
2. ⏳ CI green on the PR (push + pull_request).
3. ⏳ Confirm an **on-platform** deterministic rebuild — `git diff app/data/`
   clean apart from the `stats.json` timestamp and the known cross-platform
   `coupling.parquet` / Moran's I drift (item 5); a fully clean cross-platform
   rebuild requires the Linux/CI build target.
4. ⏳ Tag `v1.1.0` (additive: stability diagnostics + L3 coupling pages; no
   `SCHEMA_V1` / PCS change).
