# Key findings

Headline results of the cross-country warming-inequality decomposition. All
numbers are **descriptive variance attributions, not causal climate
attribution** (see [interpretation](#interpretation) and the design memo's scope
boundary). Reproduced by `python -m src.inequality` and
`python -m src.decomposition`; see `docs/reproducibility.md`.

## 1. Warming is universal but unequal

Across **157 countries**, every country's 1950–2013 mean warming trend is
positive, but the rate is uneven:

| Metric | Value | Reading |
|--------|-------|---------|
| Mean country warming | 0.161 °C/decade | — |
| 5th–95th percentile | 0.086 – 0.254 °C/decade | the fastest-warming countries warmed ≈ 3× the slowest |
| Gini of warming | **0.175** | uneven, but far below income-style inequality |
| Theil-T | 0.050 | of which **≈18% is *between* continents**, ≈82% within |
| Coefficient of variation | 0.314 | — |

The inequality is real and measurable, but moderate — consistent with the
pre-registered prior (design memo H1). Most of it sits *within* continents, not
between them.

## 2. The inequality is overwhelmingly geographic

A group-level **LMG / Shapley** decomposition attributes the cross-country
variance in warming trend (**n = 154** complete cases, **total R² = 0.63**) to
four structural axes plus an explicit residual:

| Axis | Standalone R² | Shapley share (of total) | Share of *explained* |
|------|:---:|:---:|:---:|
| **Geography** (latitude, elevation, continentality, Köppen, hemisphere, spatial block) | 0.580 | **0.455** | **72%** |
| **Emissions** (cumulative CO₂ per-capita & total) | 0.132 | 0.084 | 13% |
| **Socioeconomic** (income group) | 0.080 | 0.056 | 9% |
| **Population** (population, station density) | 0.049 | 0.036 | 6% |
| **Residual** (unexplained, 1 − R²) | — | **0.369** | — |

The named axes plus the residual sum to 1 by construction. **Physical geography
is the dominant structuring axis** — about 72% of the explainable cross-country
warming variance — with latitude (polar amplification) foremost. This confirms
pre-registered hypothesis H2.

## 3. The emissions share is real but small, and mostly overlaps geography

Emissions responsibility has a non-trivial **standalone** R² of 0.132 — the raw
ranking of warming does track the ranking of cumulative per-capita CO₂. But its
**Shapley share is only 0.084** (≈13% of explained variance): once the overlap
with geography is shared out fairly across orderings, most of that standalone
association is *absorbed* by geography. The gap between 0.132 and 0.084 is the
**measured emissions∩geography overlap** — the latitude/industrialization
confound turned into a number rather than a caveat (hypothesis H3 confirmed).

This mirrors the legacy single-coefficient analysis (the "Foundations" pages):
warming rises **+0.029 °C/decade per 10× cumulative per-capita CO₂** within
continents (HC1 SEs), but the coefficient **halves to an insignificant +0.012
once a latitude control is added**. The decomposition generalizes that fragility
into a stable, order-independent share.

## 4. A substantial residual remains

**37% of the total** cross-country warming variance is left unexplained by all
four axes (hypothesis H5, partly). This is not noise to be dismissed: country
warming residuals are expected to be **spatially structured** (regional
processes the four axes do not capture), which is precisely what the deferred
stability layer is scoped to test (Moran's I / Conley HAC — see
`docs/stability_roadmap.md`). The residual is a named, testable object, not a
modeling afterthought.

## 5. Supporting spatial findings (city level)

From the underlying per-station analysis (the "Foundations" layer):

- **Land mean 0.146 °C/decade**; 99.1% of 3,510 city-locations have an entire
  Theil–Sen 95% CI above zero.
- **Arctic amplification:** >60°N warms at 0.23 °C/decade, **1.56×** the land
  mean (compressed from the textbook ~2× because the baseline here is land-only,
  and only 25 stations sit above 60°N).
- **Fastest cluster:** the Iranian plateau / Central Asia (Mashhad, Herat,
  Ashgabat — up to 0.34 °C/decade), a documented semi-arid amplification signal,
  not the Arctic.
- **Out-of-sample:** the 1950–2013 trends **systematically underpredict**
  post-2013 observations — warming accelerated beyond the fitted lines.

## Interpretation

**What the findings support.** Cross-country warming inequality is real and is
**overwhelmingly structured by physical geography** (latitude above all). What
looks like an emissions-responsibility signal is largely that historically
high-emitting industrialized countries **sit at the mid-to-high northern
latitudes** where amplification is strongest — a confound the decomposition
*measures* (as the emissions∩geography overlap) rather than merely flags.

**What they do not support.** None of this is causal. CO₂ is well-mixed, so a
country's own emissions do not preferentially heat its territory; a large
geography share is alignment, not mechanism. Crucially, **mean warming in
°C/decade is not a measure of climate *impact* or *injustice*** — a small
emissions share on *this* outcome is **not** evidence that climate inequality is
small. Impacts scale with heat exposure, vulnerability and adaptive capacity,
which concentrate in the low-emitting tropical countries that warm *less* in
mean-temperature terms. That exposure-and-vulnerability inequality is a
different, policy-relevant outcome this project explicitly does not measure (see
`docs/future_work.md`).

The dominant threats to these numbers are **station sampling bias** (dense
mid-latitudes, sparse Arctic/Sahara/Amazonia) and **axis collinearity** (Shapley
handles it *fairly* but cannot *separate* genuinely entangled axes). Both are
detailed in `docs/decomposition_design_memo.md` §7 and are the agenda for the
stability layer.
