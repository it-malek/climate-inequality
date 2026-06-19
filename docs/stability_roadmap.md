# Stability layer — design blueprint

**Status:** design document for the next major research phase. **No new
statistical models, outcomes, or causal/physical layer** — the committed Option B
architecture (single-layer, strictly cross-sectional, descriptive) is unchanged.
This blueprint specifies a `src/stability.py` module that re-estimates the
*existing* decomposition under perturbations and reports how stable the shares
are. It implements nothing; it is the spec the implementation phase builds
against.

It pairs with: `src/decomposition.py` (what is being stress-tested),
`docs/decomposition_design_memo.md` §7–§8 (the threats and pre-registered
priors this layer must probe), and `app/views/sensitivity.py` (the dashboard
page that already renders a `stability_summary.json` and shows a pending state
until this layer ships).

## 1. The question this layer answers

The decomposition reports point shares (geography 0.46, emissions 0.08, residual
0.37). The stability layer asks: **are those shares an artifact of estimator,
sampling, construction choices, or a few influential countries — or do they
hold?** The object of inference is the **stability of the Shapley shares**, not
the statistical significance of any single coefficient.

Three claims must survive for the headline to stand (pre-registered as the
acceptance criteria, §8):

- **C1.** Geography remains the **largest** share under every perturbation.
- **C2.** The emissions share stays **small and positive** (no perturbation
  inflates it to rival geography, none drives it to zero).
- **C3.** The residual stays **substantial** and is shown to be **spatially
  structured** rather than white noise.

## 2. `src/stability.py` — contract

- **Input:** the schema-named country design from
  `decomposition.build_country_design(...)` — exactly what the live decomposition
  consumes, so the layer never sees a feature outside `SCHEMA_V1`.
- **Core dependency:** `decomposition.group_lmg_shares(...)` is called repeatedly
  on perturbed/resampled designs. The layer adds **no new estimator** — it wraps
  the existing one.
- **Output:** `data/processed/stability_summary.json`, merged into the bundle by
  `app_assets` (a new optional-findings branch, mirroring validation/explain),
  and rendered by the existing sensitivity page.
- **Conventions (match the codebase):** seeded RNG (`numpy.random.default_rng`)
  for reproducible resamples; all floats passed through
  `data_io.round_floats` so the committed summary is byte-stable; a frozen
  result dataclass with an invariant check; synthetic-fixture tests (no real
  data), following `tests/conftest.py`.

### Proposed `stability_summary.json` schema

The page already reads three blocks — **keep them exactly** so no app change is
needed to light them up:

```jsonc
{
  "interpretation": "<INTERPRETATION_NOTE>",      // disclaimer travels, as everywhere
  "n_boot": 2000,
  "seed": 0,

  // --- already rendered by app/views/sensitivity.py (legacy-coefficient story) ---
  "df_sensitivity": [ {"df": 4, "coef": .., "ci_low": .., "ci_high": ..}, ... ],
  "uncertainty":    [ {"method": "HC1", "coef": .., "ci_low": .., "ci_high": ..},
                      {"method": "Conley HAC", "coef": .., "ci_low": .., "ci_high": ..} ],
  "influence": { "spec": "lat_continent", "top_dfbeta": [["United States", ..], ...] },

  // --- NEW: the decomposition's own confidence story (the centerpiece) ---
  "share_stability": {
    "method": "country_bootstrap",
    "groups": {
      "geography":     {"point": 0.455, "ci_low": .., "ci_high": .., "p_largest": 0.99},
      "emissions":     {"point": 0.084, "ci_low": .., "ci_high": ..},
      "socioeconomic": {"point": 0.056, "ci_low": .., "ci_high": ..},
      "population":    {"point": 0.036, "ci_low": .., "ci_high": ..},
      "residual":      {"point": 0.369, "ci_low": .., "ci_high": ..}
    }
  },
  "construction_sensitivity": [
    {"choice": "median aggregation", "geography": .., "emissions": .., "residual": ..},
    {"choice": "cumulative total CO2", ...}, ...
  ],
  "residual_spatial": {"morans_i": .., "p_value": .., "method": "centroid kNN"}
}
```

(A v1 implementation may ship `share_stability` + `residual_spatial` first and
backfill the legacy three; the page renders whichever blocks are present.)

## 3. Decomposition-share stability (the core)

### 3a. Nonparametric country bootstrap
Resample the 154 countries **with replacement** (the country is the natural
sampling unit), recompute the full LMG decomposition on each resample, and report
the bootstrap distribution of every group share. From `B` resamples:
- **Percentile CIs** (2.5/97.5) per share.
- **P(largest)** — the fraction of resamples in which geography is the top share
  (directly tests C1).
- **P(emissions > X)** and **P(emissions ≈ 0)** for a pre-set X (tests C2).

### 3b. Jackknife / leave-one-out influence
Drop each country in turn, recompute the shares, and report the range each share
moves over the 154 leave-one-out fits, plus the countries whose removal moves the
**emissions** or **geography** share most. This is the share-level analogue of
the DFBETA panel already on the page; a headline that rides on three high-leverage
countries is fragile.

### 3c. Leave-one-continent-out
Recompute with each continent held out (6 fits). If the geography share collapses
when, say, the Arctic-adjacent continent is removed, that localizes the result.

## 4. Bootstrap options (and their trade-offs)

| Option | Respects… | Use when |
|--------|-----------|----------|
| **Nonparametric country bootstrap** (default) | the i.i.d.-country assumption | the baseline; simplest, matches the unit of analysis |
| **Block bootstrap by `spatial_block`/continent** | spatial correlation between neighboring countries | residuals are spatially dependent (they are — §5) → the default CIs are likely too narrow; resample whole blocks |
| **Cluster/Bayesian-bootstrap weights** | smoother small-n behavior | n = 154 is small; Dirichlet weights avoid empty-cell resamples in the categorical axes |

Recommendation: ship the **country bootstrap** as the headline CI and the
**continent block bootstrap** as the spatially-honest comparison, and report
both — the gap between them *is* the spatial-dependence correction for the shares.
`B ≥ 2000`, seeded; categorical axes (income, Köppen, hemisphere, spatial block)
need a guard for resamples that drop a level (degenerate dummy block → skip or
reweight, logged).

## 5. Spatial robustness

The residual is 37% of total variance; whether it is **structured** decides
whether C3 is a finding or a shrug.

- **Moran's I on country residuals** using a centroid-based spatial weight
  (k-nearest-neighbor or distance-band). A significant positive I means the
  unexplained warming is regionally clustered — unobserved spatial processes, not
  noise — exactly hypothesis H5. (Moran's I is already computed on *city*-level
  residuals in `src.explain`; reuse that machinery at the country level.)
- **Conley (spatial HAC) standard errors** on the legacy emissions coefficient,
  with distance-decay between country centroids — the page's `uncertainty` block.
  Expect a wider CI than HC1; report the ratio.
- **`spatial_block` leverage.** Re-run the decomposition with `spatial_block`
  removed from the geography axis to bound how much of geography's dominance is
  the coarse regional dummy versus latitude/elevation/Köppen physics.

## 6. Construction sensitivity (researcher degrees of freedom)

Each is a single re-run of the existing decomposition with one input swapped
(design memo §7, hypothesis H6). Report the share vector for each, so a reader
sees the shares move (or not):

- **Aggregation:** mean → **median** of city slopes per country.
- **Emissions construction:** per-capita → **cumulative total**; 2013 cutoff →
  alternative cutoff years.
- **Country weighting:** unweighted → **population/area-weighted** means (the
  movement expected to be largest, per H6 — flagged, needs a population grid join
  from `docs/future_work.md`).
- **Transforms:** log₁₀ → identity for the heavy-tailed magnitudes.
- **Latitude flexibility:** linear → **GAM B-spline** latitude control at df ∈
  {4,6,8} (the page's `df_sensitivity` block) — confirms geography's share is not
  an artifact of a too-rigid linear latitude term.

## 7. What "stable" looks like (expected, pre-registered)

Per design-memo H6: shares should move **modestly** under estimator and
construction perturbations, and **most** under population/area-weighting and
Arctic sampling. The expected outcome is that **C1–C3 all hold** — geography
stays dominant, emissions stays small-but-nonzero, the residual stays large and
tests as spatially structured. A perturbation that overturns any of C1–C3 is
itself a **finding**, to be reported, not suppressed.

## 8. Acceptance criteria

The layer is "done" when it produces, deterministically and from synthetic
fixtures in tests:
1. Bootstrap CIs for all five shares + P(geography largest).
2. Leave-one-out and leave-one-continent-out share ranges + the most influential
   countries.
3. A country-block bootstrap CI alongside the i.i.d. one.
4. Moran's I on country residuals with a p-value.
5. The construction-sensitivity table for the swaps in §6.
6. A `stability_summary.json` that the **existing** sensitivity page renders with
   no app changes for the legacy blocks, plus a small page addition to draw the
   new `share_stability` CIs and `residual_spatial` result.

## 9. Explicit non-goals

- No new outcome variable (no exposure/extreme-heat outcome — that is
  `docs/future_work.md`, a different study).
- No causal or physical/driver layer (the committed scope boundary).
- No change to `SCHEMA_V1` group membership; the layer perturbs *estimation and
  sampling*, not the contract. (A schema revision is a separate, versioned event —
  `SCHEMA_V2` — never an in-place edit.)
- No new heavy runtime dependency in the deployed app: the layer runs offline in
  the pipeline and ships only the small JSON summary into the bundle.

## 10. Sequencing

1. `src/stability.py` skeleton + result dataclass + `stability_summary.json`
   writer (seeded, rounded) + synthetic-fixture tests.
2. Country bootstrap → `share_stability` (the headline; lights up the page).
3. Moran's I on country residuals → `residual_spatial`.
4. Construction-sensitivity swaps → `construction_sensitivity`.
5. Block bootstrap + leave-one-continent-out.
6. Backfill the legacy coefficient blocks (`df_sensitivity`, `uncertainty`,
   `influence`) the page already scaffolds.
7. `app_assets` merge branch + the small sensitivity-page addition for the new
   blocks.
