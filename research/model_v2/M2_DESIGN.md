# M2 design — one joint latitude functional-form candidate

**Recorded 2026-09-16. Design and pre-registration only: no M2 model has been fitted, scored or
cross-validated, and no M2 coefficient exists.**
* **Authorization.** The owner authorized M2 design, but not execution, on 2026-09-16. The rules are in
  [`M2_EVALUATION_CONTRACT.md`](M2_EVALUATION_CONTRACT.md).
* **Provenance.** The historical basis and its limits are in
  [`M2_PROVENANCE_AUDIT.md`](M2_PROVENANCE_AUDIT.md).
* **Feasibility.** Predictor-only structural evidence, run from pushed commit `fbaa3ff`, is in
  [`M2_FEASIBILITY_AUDIT.md`](M2_FEASIBILITY_AUDIT.md) and `outputs/m2_feasibility/`.

## 1. The candidate

```
eta_i(M2) = eta_i(M0*) + theta_1 N_1(a_i) + theta_2 N_2(a_i) + gamma I(SH_i) a_i
```

* `eta_i(M0*)` is the frozen M0\* linear predictor, unchanged. It keeps its intercept, its linear
  `abs_latitude` term, its `hemisphere` main effect and every other column.
* `a_i` is M0\*'s station-based `abs_latitude`, in degrees.
* `I(SH_i) = 1` exactly when M0\*'s `hemisphere` label is `"S"`, else 0. It is the same indicator M0\*
  already encodes as `hemisphere=S`.
* `N_1` and `N_2` are the two added natural cubic basis functions (§3). Together with the retained
  linear term they give latitude exactly three non-constant dimensions.
* `gamma` is one southern-hemisphere latitude-slope difference (§4), coded with centre c = 0°.

**Parameter budget.** The candidate has **+3 coefficients** relative to M0\*, all in the existing
`geography` group. The full-sample rank goes from **20 to 23**, as the committed feasibility record
shows.

**What this represents.** It is common latitude curvature plus a hemisphere-specific linear slope
difference. It does **not** identify NH snow-albedo feedback, and it cannot test an onset of curvature
near 50°N (§6). An NH-only spline would be a new specification, not the historical global
`abs_latitude` spline, and is not substituted.

## 2. Alternatives considered, and why one joint candidate

| Option | Added coefficients | Status |
|---|---:|---|
| H1 only: `I(SH)·a` | +1 | Not a candidate |
| Spline only: two added natural cubic columns | +2 | Not a candidate |
| **Joint H1 + H5: both** | **+3** | **The single M2 candidate** |

* **Reason for the joint candidate.** The `af97bd2` roadmap lists (a) and (b) together as M2's terms,
  "at most three". The owner fixed one joint comparison on 2026-09-16. Nothing had been fitted, so this
  is not a performance-based choice.
* **The partial options are never scored.** **No H1-only, spline-only or other latitude-form variant is
  scored on this outcome dataset in this research line, before or after the M2 result, and the M2
  result is not cited as motivation for one.**
* **What a result would mean.** A successful joint candidate shows only that the pre-registered richer
  latitude form improves transferable prediction relative to M0\*. On its own that establishes neither
  hemispheric slope asymmetry, nor curvature, nor any mechanism.

## 3. The natural cubic latitude basis (`m2-latitude-basis-v1`)

The reference implementation is [`m2_latitude_basis.py`](m2_latitude_basis.py), committed at
`fbaa3ff`. Any later evaluator must import it.

### 3.1 Formula

The basis is the truncated-power form of the natural cubic spline (Hastie, Tibshirani & Friedman, *ESL*,
2nd ed., eqs. 5.4–5.5). For knots ξ₁ < ξ₂ < ξ₃ < ξ₄:

```
d_k(a) = [ (a − ξ_k)₊³ − (a − ξ₄)₊³ ] / (ξ₄ − ξ_k),   k = 1, 2, 3
N_1(a) = d_1(a) − d_3(a)
N_2(a) = d_2(a) − d_3(a)
```

* **Evaluation.** `(t)₊ = max(t, 0)`, and cubes are computed as the IEEE product `t·t·t`, never through
  a platform `pow` kernel.
* **Span.** {1, a, N₁, N₂} spans the four-dimensional space of natural cubic splines on the four
  knots: cubic between knots, C² at every knot, and linear below ξ₁ and above ξ₄.
* **"Three df".** This means three **non-constant** dimensions, with the intercept excluded. M0\*
  already carries `a`, so exactly **two columns are added**.
* **Relation to R.** For identical, non-degenerate knots, the function space equals that of R's
  `ns(a, df = 3)` plus an intercept; only the parameterization differs. It is not a bit-level
  equivalence:
  * the frozen knots are defined by `numpy.quantile(method="linear")` under the `uv.lock`-pinned numpy,
    not by R's type-7 arithmetic, which can differ in the last bits;
  * R's shoving of interior knots that coincide with a boundary knot is not part of this design. Here a
    coincidence raises (§3.2).
* **Tests** (synthetic only) check:
  * the formula, value for value;
  * that scipy's independent natural interpolating spline lies in the span to 1e-9;
  * C² continuity and exactly linear tails;
  * the quantile convention;
  * that deserialization refuses bad state.

### 3.2 Knots and state

* **Interior knots.** ξ₂ and ξ₃ are `numpy.quantile(a_train, [1/3, 2/3], method="linear")`, which is
  Hyndman–Fan type 7 by definition. The probabilities are the IEEE doubles of Python `1/3` and `2/3`.
* **Boundary knots.** ξ₁ = `min(a_train)` and ξ₄ = `max(a_train)`.
* **Ties.** They are permitted; only the resulting knots matter.
* **Degeneracy.** The state is self-validating. It raises `DegenerateLatitudeBasis` if:
  * any consecutive knot gap is ≤ **1e-6 degrees**;
  * any knot is non-finite or negative;
  * there are fewer than four training values;
  * the version differs.

  **There is no fallback:**
  * no df reduction;
  * no knot merging or moving;
  * no alternative quantile rule;
  * no regularization;
  * no substitute basis.

  A degenerate required fit is a structural failure (§5).
* **Centering and scaling.** Neither latitude nor any added column is centred, standardized or
  rescaled. Coefficients come from the frozen V1 estimator (`numpy.linalg.lstsq`, SVD, `rcond=None`),
  whose fitted values do not depend on column scale. Conditioning was audited, not tuned.

### 3.3 Train-only state (resolves "sample tertiles")

The historical phrase "knots at the sample tertiles" is resolved as **the tertiles of each fit's own
training predictors**. This follows the `af97bd2` nested-validation principle: any data-derived quantity
is "chosen inside each training fold … or fixed a priori" (`SPATIAL_CV_PROTOCOL.md`).

1. **Every cross-validated fit** derives (ξ₁…ξ₄) from its **training rows' `abs_latitude` only**. This
   covers each primary 500 km training fit, each M49 training fit and each random-reference fit.
2. That fit freezes **one immutable state** (`LatitudeState`: four knots, the training n and the basis
   version).
3. The identical state transforms the fit's training rows and its held-out rows. **Held-out predictors
   never influence knots or boundaries.**
4. The **full-data fit** uses the full-data predictor state. Every in-sample calculation reuses it: the
   fitted values, all 16 LMG coalitions and residual Moran's I.
5. **Every later refit** derives one immutable state from that refit's own rows and uses it for all of
   that refit's coalitions, with degeneracy recorded and no fallback. This covers a bootstrap or block
   bootstrap resample at M4, and a replication arm.
6. **The M0\* comparator does not use the state.** It has no latitude basis to transform.

**Serialization.** The state is JSON with:
* the four knots in decimal and as exact `float.hex` strings, the training n and the version;
* the quantile method, probabilities, boundary and tail conventions.

Deserialization refuses any foreign convention, and refuses decimal knots that disagree with their hex.
**The hex knots are authoritative.** Exact state comparisons use
`LatitudeState.from_json(summary["full_sample"]["station/<representation>"]["latitude_state"])` from the
committed feasibility record, never the decimal CSV columns.

### 3.4 Extrapolation

Rows outside a fit's [ξ₁, ξ₄] are transformed with the same formula, which is linear there. No row is
clamped or dropped. The feasibility record counts every held-out extrapolation:
* **Primary protocol:** Gabon below its training minimum (1.337°) and Iceland above its training maximum
  (61.556°).
* **M49 protocol:** when Middle Africa is held out, Gabon lies below the training minimum; when Northern
  Europe is held out, 8 of its 10 countries lie above the training maximum (53.749°).
* **Random-reference folds:** fold 5 has 1 row below and fold 6 has 4 rows above.

These are properties of the frozen design, disclosed in advance. They are not tuned away.

## 4. The hemisphere interaction

* **Column.** `hemisphere=S:abs_latitude = I(SH) · (a − 0)`, which is uncentred.
* **Meaning of γ.** γ is the southern-minus-northern difference in the latitude slope, constant across
  latitude, given the common curvature.
* **Hierarchy.** The `hemisphere` main effect and the linear `abs_latitude` term stay in M0\*, so the
  interaction adds exactly one identifiable dimension. The fixed centre changes only the hemisphere main
  effect's coefficient, never γ, the fitted values or the rank.
* **Labels.** Labels outside {`N`, `S`} raise. The column is built from the label directly and not
  through training-level dummy coding.
* **Support.** Both levels are present in every required training fit: at least 18 southern countries,
  spanning at least 32°.
* **No history to override.** No centering or coding convention exists anywhere in the project's
  history (`M2_PROVENANCE_AUDIT.md` §1).
* **Not introduced:**
  * fold-specific, mean or outcome-based centering;
  * separate per-hemisphere slope columns;
  * any other parameterization.

## 5. Structural feasibility requirement (verified before freezing)

Every required fit must satisfy all of the following in both registered representations:
* M0\* rank equals its column count;
* the candidate's rank equals its column count;
* the candidate adds **exactly three identifiable dimensions**;
* the state is non-degenerate.

Categorical levels legitimately absent from a training fit reduce both designs' column counts equally.

The required fits are:
* the full sample;
* all 151 primary training fits;
* all 20 M49 training fits;
* the 10 random-reference fits.

A failure anywhere is a structural failure, with no repair. The canonical record (`fbaa3ff`) shows every
required fit passing, and the conditional M1a-measurement arm passing as well.

## 6. Shape diagnostic (predeclared, descriptive, never a verdict condition)

**Raw basis coefficients are never interpreted.** The ns1, ns2, linear `abs_latitude` and `hemisphere=S`
coefficients carry no invariant meaning in this parameterization, and their fold summaries are not
stability evidence:
* ns1 and ns2 multiply different functions in each fit;
* the linear coefficient is the slope below ξ₁;
* `hemisphere=S` is the southern offset at 0°.

**One shape quantity is predeclared: γ** (°C/decade per degree of latitude). It is the constant
southern-minus-northern latitude-slope difference, which is invariant to the basis parameterization and
to the fixed centre.
* **Reported:** the full-fit value, the fraction of the 151 primary training fits with γ < 0, and the
  median and range.
* **Historical expectation:** negative. H1: "NH slope positive …, SH slope negative or zero".
* **Not independent:** that expectation derives from the same outcome's M0 residuals.
* **Never a verdict condition.**

**No curvature contrast is predeclared.** On [ξ₃, ξ₄] a natural cubic spline is a single cubic with
f″ linear and f″(ξ₄) = 0. So for any latitudes ξ₃ ≤ u < v ≤ ξ₄:

f′(v) − f′(u) = f″(ξ₃)(v − u)(ξ₄ − (u + v)/2)/(ξ₄ − ξ₃)

That is a fixed positive multiple of the fitted curvature at ξ₃.
* **Full fit and every primary training fit.** ξ₃ < 40° and ξ₄ > 60° (ξ₃ ≤ 39.599°, ξ₄ ≥ 61.556°), so
  f′(60°) − f′(40°) = 20·f″(ξ₃)(ξ₄ − 50)/(ξ₄ − ξ₃).
* **Two reference fits.** When M49 Northern Europe is held out (ξ₄ = 53.749°) and in random fold 6
  (ξ₄ = 58.66°), 60° lies beyond the upper boundary knot, where the slope is constant.

ξ₃ lies near 30–40°, a band that includes southern countries, and the basis forces curvature to fall to
zero at the training maximum. So this pooled, tertile-knot spline **cannot represent or test H5's
"steepening above ~50°N"**. A slope contrast there would describe curvature near 36°, not the historical
hypothesis. Any NH-specific or 50°N-knot form would be a new, residual-shaped design and is not
introduced.

**M2 is therefore reported primarily as a predictive functional-form test.** Curvature enters only
through the predictive comparison. H5 remains the historical motivation for including curvature, not a
separately tested shape. Its residual evidence was measured on land-area-centroid |latitude|, while M2
uses station `abs_latitude` (`M2_PROVENANCE_AUDIT.md` §4).

## 7. Decomposition semantics

* **Groups.** There are four named groups: `emissions` (responsibility), `geography`, `socioeconomic`
  and `population`. That gives **16 coalitions**; the intercept is common to all.
* **Geography.** It holds linear `abs_latitude`, `abs_latitude_ns1`, `abs_latitude_ns2`,
  `hemisphere=S:abs_latitude`, the `hemisphere` main effect and every other existing geography
  predictor. These stay together in **every** coalition that contains geography.
* **No new group.** There is no nonlinear-latitude, interaction or H1/H5 group, and no allocation is
  split across groups.
* **State.** Every coalition uses the full-data latitude state (§3.3).
* **Estimator and definition.** M2 is linear in its parameters and fitted by OLS, so the existing
  R²-based LMG/Shapley definition applies unchanged. `OPEN_DECISIONS.md` item 10, the definition for
  non-OLS stages, stays open for M3.
* **Units.** Shares are fractions of total outcome variance. Named shares sum to the in-sample R², and
  named plus residual sum to 1.
* **Meaning.** LMG remains a descriptive allocation among correlated groups. It is not causal
  attribution.

## 8. Implementation form (for the later evaluator)

M2 is a **research-only schema extension** in the manner of `SCHEMA_M1B`.
* The three columns are declared in `geography`.
* The extension reproduces M0\* exactly when the columns are absent.
* `V2_PREDICTOR_CONTRACT.md` is unchanged; nonlinear terms are not part of it.
* Predictors come through the same verified scoring-input path as M1b, without the C2 package. The
  evaluator never uses the rounded numeric columns of `m0_countries.csv`.

Before fitting any candidate, the evaluator must assert all of the following:
* its full-sample M0\* encoded design equals the `M0star` rows of the frozen `m1b_design_matrices.csv`
  value for value, **after aligning columns by name**;
* `lb.fit_state(full-sample abs_latitude)` equals the committed record's hex state exactly (§3.3);
* M0\* reproduces its frozen `drop_per_capita` and `drop_total` cards to 1e-10.
