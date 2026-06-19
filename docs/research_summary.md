# Research summary

A self-contained technical summary of the project for an educated audience.
Pairs with `docs/key_findings.md` (the headline numbers), the
`docs/decomposition_design_memo.md` (scientific rationale and pre-registration),
and `docs/reproducibility.md` (how to rebuild every figure). Throughout, the
analysis is **descriptive** — a structural decomposition of *observed* warming,
never a causal or physical climate-attribution claim.

## Abstract

Observed land warming over 1950–2013 is universal across countries but uneven.
We quantify *how unequal* that warming is (Gini, Theil) and, more importantly,
*how the inequality is structured*, by decomposing the cross-country variance in
mean warming trend into Shapley/LMG shares across four fixed axes — emissions
responsibility, physical geography, socioeconomic development, and
population/urbanization — plus an explicit residual. Across 154 countries the
model explains 63% of the variance; **physical geography takes the dominant
share (≈72% of the explained variance, ≈46% of the total)**, the emissions axis
a smaller but non-zero share (≈13% explained), and a substantial residual (37%
of total) remains. The emissions and geography axes overlap heavily, so the
project's headline contribution is turning the latitude-versus-emissions
confound into a *measured overlap* rather than a caveat.

## 1. Question

> How unequally is observed 1950–2013 warming distributed across countries, and
> to what extent is that inequality structured by emissions responsibility,
> physical geography, socioeconomic development, and population/urbanization?

This is a question about the **structure of an inequality**, not a causal effect.
We do not ask whether a country's CO₂ caused its own warming (it cannot,
materially — CO₂ is well-mixed); we ask how much of the cross-country spread in
warming coincides with each kind of measurable structure. An earlier framing — a
single "emissions coefficient" — is deliberately demoted to one axis among
several (see §6).

## 2. Data and scale

| Input | Scale / resolution |
|-------|--------------------|
| Berkeley Earth by-city temperatures (Kaggle) | ~8.6M monthly rows → **3,510 city-locations** |
| Analysis window | 1950-01 … 2013-09 (763 months); 1951–1980 climatology baseline |
| OWID CO₂ + population | annual, per country, cumulative through 2013 |
| Geography grids | ETOPO 2022 elevation (60″), Köppen–Geiger 0.5° (Beck et al. 2018), Natural Earth coastlines/land |
| Socioeconomic | World Bank income groups |
| Countries in the decomposition | 157 matched → **154 complete cases** |

The temperature data's defining quirk: `(City, Country)` is **not** a unique key
(18 same-named pairs at multiple grid-snapped coordinates), so every stage keys
on the full `(City, Country, Latitude, Longitude)` identity.

## 3. Methodology

The pipeline is a sequence of small, individually tested modules (full
parameters in `docs/reproducibility.md`):

1. **Trend estimation.** Monthly anomalies against each location's 1951–1980
   climatology (≥90% coverage gate), then a per-location **Theil–Sen** slope
   (°C/decade) with a 95% CI — robust to outliers, with OLS retained for
   comparison.
2. **Spatial interpolation.** Inverse-distance weighting vs ordinary kriging,
   compared by **leave-*location*-out** cross-validation that holds out whole
   grid-snapped coordinate groups (naïve row-wise CV leaks bit-identical
   duplicates and *reverses* the method ranking — a methodological finding in
   its own right). IDW wins narrowly and renders the published land-masked
   surface.
3. **Country aggregation.** Per-country warming is the **unweighted mean** of its
   city-location slopes (no city populations exist in the data — the project's
   largest caveat). Emissions responsibility is cumulative production-based CO₂
   through 2013 per 2013 resident.
4. **Feature assembly.** Each location is joined to elevation (ETOPO), coast
   distance (Natural Earth), Köppen class (Beck et al.), and a local
   station-density proxy; categoricals aggregate to a country's modal class.
5. **Inequality quantification.** Gini, Theil-T (with an exact
   between-/within-continent split), CV and variance on the country warming
   distribution — model-free descriptive measures.
6. **Variance decomposition.** Group-level **LMG / Shapley-Owen R²** over the
   frozen `SCHEMA_V1` feature contract: each axis's share is its incremental R²
   averaged over **all orderings**, so correlated axes split shared variance
   fairly rather than by entry order; the four axes plus a named residual sum to
   1. The contract is enforced in code — no feature outside `SCHEMA_V1` may enter
   any model, and the groups must partition the design matrix.
7. **Out-of-sample validation.** The 1950–2013 trends are tested against
   post-2013 gridded Berkeley Earth observations.

Why a decomposition rather than a regression coefficient: the axes are
empirically collinear (high emitters tend to be high-latitude, high-income), so a
single coefficient is fragile, while a Shapley share is order-independent and
makes the overlap itself reportable.

## 4. Results

(Full tables in `docs/key_findings.md`.)

- **Inequality.** Gini 0.175, Theil-T 0.050 (≈18% between continents); mean
  0.161 °C/decade, with the 5th–95th percentile spanning 0.086–0.254 (≈3×).
- **Decomposition** (n = 154, R² = 0.63): geography **0.455**, emissions 0.084,
  socioeconomic 0.056, population 0.036, residual **0.369**. Of the explained
  variance: geography ≈72%, emissions ≈13%.
- **Emissions overlap.** Emissions' standalone R² (0.132) exceeds its Shapley
  share (0.084); the difference is the variance it shares with geography. The
  legacy coefficient (+0.029 °C/decade per 10× CO₂ within continents) halves to
  an insignificant +0.012 under a latitude control — the same fragility the
  decomposition captures stably.
- **Spatial signals.** Arctic amplification 1.56× (land-only baseline); the
  fastest-warming cluster is the semi-arid Iranian plateau / Central Asia; the
  fitted trends underpredict post-2013 observations (acceleration).

## 5. Limitations

The binding constraints (full list: design memo §7, README):

- **Station sampling bias** — trends exist only where Berkeley Earth has city
  series (dense mid-latitudes, sparse Arctic/Sahara/Amazonia); country means are
  station-weighted, not area- or population-weighted. The single largest threat.
- **Axis collinearity** — Shapley splits shared variance *fairly* but cannot
  *separate* genuinely entangled axes; a dominant geography share could partly
  reflect that high emitters simply *are* high-latitude.
- **Ecological aggregation** — country means can mask or invert within-country
  relationships; conclusions apply to countries-as-units only.
- **Outcome ≠ impact** — mean annual-trend warming, land-only, ending Sept 2013;
  excludes ocean warming, post-2013 acceleration, and extreme-heat days.
- **Construction choices** — unweighted vs weighted means, cutoff year,
  per-capita vs total emissions, ~1° grid-snapped feature sampling — each a
  researcher degree of freedom the stability layer will perturb.
- **Spatial autocorrelation** in residuals makes naïve standard errors
  optimistic; R²-share inference is less affected than coefficient inference, but
  this must be measured before any share is reported as *precise*.

## 6. Interpretation and relationship to prior framing

The honest headline is that **cross-country warming inequality is predominantly a
geographic (latitudinal) phenomenon**, and the apparent emissions-responsibility
signal is largely that historically high-emitting countries happen to sit where
amplification is strongest. The decomposition's value is making that confound a
*measured quantity*.

This reframes — but does not contradict — the project's earlier emissions-vs-
warming analysis. That work found a real positive association (Spearman ρ ≈
+0.36; +0.029 °C/decade per 10× CO₂ within continents) that weakened under a
latitude control. Rather than report a fragile single coefficient, the project
was re-oriented to decompose the *variance* of warming, demoting emissions to one
responsibility axis among four structural axes. The result is descriptively
stronger and interpretively safer.

**The decomposition cannot speak to climate impact or injustice.** Mean warming
in °C understates the burden on low-emitting tropical countries that are most
exposed to heat and least able to adapt; a small emissions share on this outcome
is not evidence that climate inequality is small. Quantifying exposure- and
vulnerability-weighted inequality is the principal external-validity gap and a
priority in `docs/future_work.md`.
