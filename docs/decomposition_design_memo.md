# Design memo: variance attribution of global warming inequality

**Status:** design document. Pairs with the frozen feature contract in
`src/feature_schema.py` (`SCHEMA_V1`, *candidate* maturity) and governs the
`src/inequality.py` and `src/decomposition.py` modules (now implemented).
**Audience:** anyone reading or extending the decomposition. **Author/date:**
2026-06-18.

---

## 0. Scope boundary: structural decomposition, not climate attribution

A project like this could host **two distinct model layers**:

1. **Physical driver model** (time-series / panel): explain *temperature change
   over time* from forcings — CO₂, CH₄, N₂O, aerosols, ENSO, volcanic — with lag
   structure and confounding controls. Output: an estimated climate-response
   signal. This is mechanistic *climate attribution*.
2. **Inequality decomposition** (cross-sectional): explain *how observed warming
   is distributed across countries* via Shapley/LMG variance attribution on
   country warming trends. Output: variance shares + residual. Descriptive.

**Architectural decision (committed):** this project is **layer 2 only**. The
physical-driver / climate-attribution layer is an **explicit non-goal** and is
out of scope. The two layers must stay conceptually and statistically separated,
and the decomposition is *never* to be read as physical causality.

Two reasons make this the right boundary, not merely a convenient one:

- **The layers do not statistically compose.** A physical model explains
  *temporal* variance in a *global* temperature series; its output is a single
  response signal that is **spatially constant** across countries. A constant
  contributes **zero** to *cross-country* variance, so "physical model →
  inequality decomposition" is not a pipeline — the arrow carries no
  information. They answer orthogonal questions.
- **The data for a credible physical model is absent.** The repo has CO₂/CH₄/N₂O
  annual series (OWID) and a global temperature series (`GlobalTemperatures.csv`),
  but **no aerosol, ENSO, or volcanic-forcing** data. A driver model missing the
  cooling forcings and internal variability would be under-identified and
  causally fraught — exactly the failure mode this project was re-oriented to
  avoid.

**Hard rule.** Shapley/LMG group shares are a **variance attribution only**. They
quantify how the cross-country warming ranking *aligns* with each structural
axis, never a physical or causal effect. This rule is enforced in code by the
canonical disclaimer `src.feature_schema.INTERPRETATION_NOTE`, which every model
output (`inequality_summary.json`, `decomposition_summary.json`) carries. See
§6 (can/cannot support) and §7 (threats) for the downstream consequences.

---

## 1. The primary scientific question

> **How unequally is observed 1950–2013 warming distributed across countries,
> and to what extent is that inequality structured by emissions responsibility,
> physical geography, socioeconomic development, and population/urbanization
> structure?**

This is a question about the **structure of an inequality**, not about a causal
effect. We are not asking "does a country's CO₂ cause its own warming" (it
cannot, materially — CO₂ is well-mixed). We are asking: warming is demonstrably
uneven across countries (the land mean is 0.146 °C/decade, but the tails differ
roughly fourfold); when we line countries up by how fast they warmed, *how much
of that spread coincides with each kind of structure we can measure*?

The earlier framing — a single "emissions coefficient" (+0.029 °C/decade per
10× cumulative per-capita CO₂, which halves to an insignificant +0.012 once a
latitude control is added) — answered a narrower, more fragile question. This
memo deliberately demotes that coefficient to **one input among several** and
reorients the project around decomposing the *variance* of warming.

## 2. Outcome variable(s) under consideration

The candidate outcomes, and why we rank them:

1. **Country-mean warming trend** (`warming_trend`, °C/decade) — the unweighted
   mean of a country's city-location Theil–Sen slopes. **Preferred** (§3).
2. **City-location warming trend** — the 3,510 per-location slopes themselves.
   Higher resolution, but socioeconomic/emissions/population axes are *not
   defined* at the city level in our data, so a four-axis decomposition is
   impossible here. Retained only as a city-level *geography-only* descriptive
   layer and as an input to the country aggregate.
3. **An inequality statistic of the trends** (Gini / Theil / variance of
   country trends) — this is the *quantity whose magnitude we report* in
   `inequality.py`, but it is a scalar summary, not a per-country outcome a
   regression can decompose. The Theil index additionally supports an exact
   between-/within-continent split, which we will report alongside the
   regression-based decomposition as an independent cross-check.
4. **Population-weighted heat *exposure* or extreme-heat trends** — the
   policy-relevant impact outcomes. **Out of scope** for v1: we lack city
   populations and daily data. Flagged in §7 as the main external-validity gap.

So two outcomes coexist by design: a **scalar inequality measure** of the
country trends (the headline "how unequal"), and the **per-country trend** as
the regression target (the "how is it structured").

## 3. Why country-level warming trend is the preferred outcome

- **It is the only level at which all four explanatory axes exist.** Emissions
  responsibility, income classification, GDP, and national population are
  country attributes. A decomposition that includes them *requires* a country
  unit of analysis (`SCHEMA_V1.unit_of_analysis == "country"`).
- **It matches the question.** "Climate inequality across countries" is a
  cross-national statement; the country is the natural unit of moral and policy
  interest (responsibility, capacity, exposure are all nationally framed).
- **It is robust to the data's worst weakness.** Station sampling is wildly
  uneven (dense mid-latitudes, ~25 locations above 60°N). A single dense city
  cluster perturbs a city-level analysis far more than a country mean, which
  averages within national borders. (It does *not* fix the bias — see §7 — only
  dampens its leverage.)
- **It keeps the sample size honest for a variance decomposition.** 157
  countries is small but workable for a 4-group LMG/Shapley partition; 3,510
  cities would over-fit the geography axis and starve the others of meaning.

The cost is **ecological aggregation** (a country mean discards within-country
variance and can invert individual-level relationships). We accept it as
intrinsic to the question and document it as a validity threat (§7).

## 4. Why emissions, geography, socioeconomic development, and population are *distinct* axes

The four groups in `SCHEMA_V1` are not an arbitrary partition; each names a
*different mechanism* by which warming inequality could be structured. They are
empirically correlated (high emitters tend to be high-latitude, high-income,
and industrialized) — which is precisely why a *decomposition* rather than a
single regression coefficient is the right tool. The groups are kept distinct
because conflating any two would change the meaning of every share:

- **Emissions / industrial responsibility** (`cum_co2_per_capita`,
  `cum_co2_total`, `co2_intensity_gdp`). A *responsibility* axis: who built the
  industrial base historically. It is a moral/attribution construct, not a
  local physical forcing.
- **Geography / physical constraints** (`abs_latitude`, `elevation`,
  `continentality`, `climate_zone`, `hemisphere`, `spatial_block`). The
  *physical* axis: where a country sits on the planet. This is the dominant
  known driver of the spatial warming pattern (polar amplification, land/ocean
  asymmetry) and is a first-class object of study, **not** a nuisance control to
  be partialled out.
- **Socioeconomic development** (`income_group`, `gdp_per_capita`). Development
  stage / adaptive capacity. Distinct from emissions: two countries with equal
  cumulative CO₂ can differ sharply in current development, and from population:
  scale ≠ wealth.
- **Population / urbanization** (`population`, `urbanization_rate`,
  `station_density`). Demographic scale and urban concentration — including
  urban-heat-island and sampling-density effects that are mechanistically
  separate from development level.

Merging emissions into geography would hide the entire research question;
merging socioeconomic into population would erase the wealth-vs-scale
distinction. Keeping them apart is what lets Shapley report *how much each kind
of structure* aligns with the warming spread, and — critically — how much
**overlap** there is between, say, the emissions and geography axes.

## 5. What a Shapley (LMG) share actually means here

We attribute the model R² to feature *groups* using the LMG / Shapley-Owen
method: each group's share is its **incremental R², averaged over all orderings
in which the groups could enter the model.**

Concretely, a reported share `share_geography = 0.55` means: *of the
cross-country variance in warming trend that our features can explain at all,
55% is attributable to the geography group once its contribution is averaged
fairly against every other ordering of the axes.* The shares of the four named
groups plus the **`residual`** group (the unexplained `1 − R²`) sum to exactly
1 by construction.

Three properties make this the right tool and bound its interpretation:

- **Order-independence / fair overlap-splitting.** When two axes are correlated
  (emissions and latitude are, strongly), the variance they *jointly* explain
  is split evenly across orderings rather than handed entirely to whichever
  enters first. This is why a single regression coefficient was fragile and a
  Shapley share is stable.
- **It is a decomposition of *explained variance*, not of *cause*.** A large
  emissions share does **not** mean emissions caused warming; it means the
  warming ranking and the emissions ranking are aligned in a way the other axes
  do not absorb. See §6.
- **Groups are atomic and fixed by the contract.** Membership is part of
  `SCHEMA_V1`, not a model-time choice; categoricals (income, Köppen,
  hemisphere, spatial block) attribute to their group as a block, never per
  dummy level.

## 6. What the decomposition can and cannot support

**Can support:**
- A quantitative statement of *how unequal* warming is across countries
  (Gini/Theil/variance — `inequality.py`), independent of any model.
- A statement of *how that inequality is structured*: e.g. "geography accounts
  for the majority of explainable cross-country warming variance; the emissions
  axis adds little once its overlap with geography is shared out" — if that is
  what the data show.
- A measure of how much the emissions and geography axes *overlap*, making the
  latitude/industrialization confound a measured quantity rather than a caveat.
- The size of the **residual** share, i.e. how much country warming inequality
  our four axes leave unexplained — and (via the stability layer) whether that
  residual is spatially structured.

**Cannot support:**
- **Causal claims.** Nothing here identifies a causal effect of emissions,
  income, or population on warming. CO₂ is well-mixed; a country's emissions do
  not preferentially heat its own territory. Shares are descriptive alignment,
  not mechanism.
- **Policy counterfactuals** ("if country X had emitted less, it would warm
  less"). The design cannot speak to interventions.
- **Impact or exposure inequality.** The outcome is mean annual-trend warming,
  which understates climate *injustice*: impacts scale with heat exposure,
  vulnerability, and adaptive capacity, which concentrate in low-emitting
  tropical countries. A small emissions share on *this* outcome is **not**
  evidence that climate inequality is small.
- **Within-country inequality**, by the ecological-aggregation construction.

## 7. Threats to validity and alternative explanations

1. **Station sampling bias.** Trends exist only where Berkeley Earth has city
   series — dense mid-latitudes, sparse Arctic/Sahara/Amazonia/Siberia. Country
   means are station-weighted, not area- or population-weighted. This can bias
   both the inequality magnitude and every axis share; it is the single largest
   threat.
2. **Collinearity of the axes.** Emissions, latitude, and income are strongly
   correlated. Shapley handles this *fairly* (it is the reason we use it) but
   cannot *separate* what is genuinely entangled: a dominant geography share
   could reflect that high emitters simply *are* high-latitude. The overlap is
   reportable but not resolvable into distinct causes.
3. **Ecological inference.** Country aggregation can mask or invert
   within-country relationships; conclusions apply to countries-as-units only.
4. **Outcome ≠ impact.** Land-only, annual-mean, ending Sept 2013. Ocean
   warming, post-2013 acceleration, and extreme-heat days are excluded; the
   land-only baseline also compresses the Arctic ratio to 1.56×.
5. **Temporal mismatch in emissions.** Cumulative emissions are divided by
   *2013* population, understating responsibility for slow-growing populations.
   `cum_co2_total` is offered alongside per-capita to bound this.
6. **Residual spatial autocorrelation.** Country residuals are likely spatially
   dependent; naive SEs overstate precision. The decomposition (R² shares) is
   less affected than coefficient inference, but the stability layer must
   measure this (Moran's I / Conley HAC) before any share is trusted as
   precise.
7. **Construction choices.** Unweighted vs population/area-weighted country
   means; mean vs median aggregation of city slopes; cutoff year; Köppen and
   elevation sampled at ~1° grid-snapped coordinates. Each is a researcher
   degree of freedom the stability layer will perturb; a share that moves under
   these is a finding, not a nuisance.

**Leading alternative explanation to keep in view:** the entire cross-country
warming inequality may be *physically* structured (latitude/land/ocean), with
emissions, income, and population aligning only because industrialized,
wealthy, populous countries happen to sit at high northern latitudes. The
decomposition is designed to *measure* how much of the apparent emissions
structure is in fact this geographic confound — and to report it honestly even
when (especially when) the emissions share turns out to be small.

## 8. Expected findings / hypotheses

These are *pre-registered priors*, stated before running the decomposition so
that a result matching them is not mistaken for a result engineered to match
them. They are predictions about a **descriptive** decomposition; none is a
causal hypothesis. We will report where the data confirm, contradict, or leave
them undetermined.

- **H1 — warming inequality is real but moderate.** The Gini of country-mean
  warming trends is well above zero but far below income-style inequality
  (rough prior: Gini ≈ 0.15–0.30), consistent with the README's roughly
  fourfold spread between the 5th and 95th trend percentiles. Warming is
  universal (every country positive) but uneven.
- **H2 — geography dominates the explainable share.** The geography axis
  (latitude foremost) takes the **largest** Shapley share — plausibly a
  majority of explained variance — because polar amplification is the strongest
  known structuring mechanism and latitude is the cleanest measured feature.
- **H3 — the emissions share is real but largely overlaps geography.** Emissions
  carries a non-trivial *raw* (univariate) association (the README's
  ρ = +0.36), but its **Shapley share shrinks** once geography's overlap is
  shared out — mirroring how the +0.029 coefficient halved to an insignificant
  +0.012 under a latitude control. Predicted standalone emissions share: small
  (order ~5–15% of explained variance), with a large emissions∩geography
  overlap.
- **H4 — socioeconomic and population are the smallest named shares.** Income
  and population/urbanization add little *beyond* emissions and geography, with
  which they are collinear; we expect their incremental Shapley shares to be the
  smallest of the four, though non-zero (urban-heat / development signals).
- **H5 — the residual is substantial and spatially structured.** A large
  fraction of country warming inequality (prior: ≥ 40% of total variance) is
  left unexplained by all four axes, and that residual is **not** white noise —
  the stability layer should detect significant spatial autocorrelation
  (Moran's I) in it, pointing to unobserved regional processes.
- **H6 — shares are stable to construction choices but sensitive to sampling.**
  Mean-vs-median aggregation, cutoff year, and robust estimators should move the
  shares only modestly; the largest movement is expected from
  population/area-weighting and from the Arctic's sparse sampling, per §7.

If H2–H3 hold, the honest headline is: *cross-country warming inequality is
overwhelmingly a geographic (latitudinal) phenomenon, and what looks like an
emissions-responsibility signal is mostly that the historically high-emitting
countries happen to sit at high northern latitudes* — a confound the
decomposition **measures** rather than merely cautions about.

---

### One-paragraph summary

We will report **how unequal** 1950–2013 country warming is (a Gini/Theil/
variance scalar) and **how that inequality is structured** by decomposing the
explained variance of country-mean warming trend into Shapley shares across four
fixed axes — emissions, geography, socioeconomic, population — plus an explicit
residual. The shares are descriptive variance attributions, not causal effects;
their headline value is making the latitude-vs-emissions confound a *measured
overlap* rather than a caveat. The dominant risks are station sampling bias and
axis collinearity, and the outcome (mean warming) is explicitly *not* a measure
of climate impact or injustice.
