# M0.5 rank-deficiency audit and proposed V2 predictor contract

## Conclusion

The exact redundancy leaves predictions unchanged but makes the four-group
variance allocation representation-dependent. Removing any one of the three
redundant log representations produces rank 20 of 20, identical M0 fitted values
and identical corrected territorial-CV predictions to numerical precision.
The narrow responsibility conclusion survives: its Shapley/LMG contribution
ranges from **1.313% to 2.219% of national-outcome variance** across these
full-rank forms. It is not meaningful to interpret the redundant model's
individual log-emissions or log-population coefficient signs.

Recommend **log cumulative total CO₂ in responsibility; log population and the
unchanged station-density feature in population**, dropping log per-capita CO₂.
Register **log per-capita cumulative CO₂ plus log population** (drop log total)
as the sensitivity. This recommendation separates national historical emissions
scale from demographic scale; it is not a choice based on fit or the smaller
share. There is no fit-based preference: the projections are identical.
The owner must approve the V2 contract before M1a; V1 is not modified.

## 1. Exact source identity and estimator

Tagged source `src/emissions.py::cumulative_emissions_per_capita` sums each
country's available annual production-based CO₂ through the cutoff (2013),
takes population in that year and constructs

`cum_co2_t_per_capita = cumulative_co2_mt × 1,000,000 / population`.

There is no universal common start date imposed on countries' historical
records. `src/decomposition.py::build_country_design` maps these to
`cum_co2_per_capita`, `cum_co2_total` and `population`.
`feature_block` takes log10 of all three. Thus in real arithmetic,

`log10(per capita) − log10(total Mt) + log10(population) − 6×intercept = 0`.

This is a **mathematical construction identity**, not merely high empirical
correlation. Observed maximum floating deviation is 1.7763568394002505e-15.
Per-capita and total CO₂ belong to **emissions/responsibility**; population
belongs to **population**. The intercept supplies the conversion constant.
The entire emissions block alone consequently spans log population, even before
the population group joins a coalition. The latter also contains station density,
so the two complete groups are not identical.

The 151 × 21 matrix has rank 20. Its smallest singular value is
3.2512855668902603e-15, compared with the NumPy rank threshold
3.348786142542778e-10. The next singular values are about 0.81935 and 1.22756;
there is one exact null direction. Complete column/group lists and every singular
value are in `outputs/rank_audit.json`.

Both V1 `src/decomposition.py::_r2` and the frozen CV encoder use
`numpy.linalg.lstsq(..., rcond=None)`: a rank-truncated least-squares solve
returning the minimum-Euclidean-norm coefficient representative. It is not a
ridge penalty, a scientific identification constraint, or a dropped-column
choice recorded by the model. For null vector

`v = {intercept: −6, per_capita: +1, total: −1, population: +1}`,

all `beta + t*v` give the same fitted vector. The returned beta depends on
column scale/units. Expressing total emissions in tonnes rather than Mt changes
the minimum-norm coefficients and reverses some signs without changing fit
(maximum fitted difference 1.00e-14). Adding `1000*v` changes coefficients by
thousands while altering numerical fitted values by at most 2.33e-12.

The fitted vector is the unique orthogonal projection onto the design's column
space even though beta is not unique. Each minimal deletion retains exactly
that column space. This also holds in each training fold; the same identity
holds in held-out rows and the same training-only categorical encoding and
unseen-level rule are retained. The largest CV prediction discrepancy among
these forms is 4.29e-14 °C/decade. Rank-only changes do not remove residual
spatial structure or improve transfer.

## 2. Controlled scorecard

Freeze the original 151 complete cases **before** deleting a feature: no sample
expansion. All rows use the corrected 500 km GPW-footprint territorial LOCO
from `4263429`, not the legacy 0.2120 CV result. Units of errors: °C/decade.
Residual Moran's I below uses the frozen station-centroid kNN8 weights.

| Representation | Columns / rank | In-sample R² | Corrected CV R² | CV RMSE | Residual I | OOF residual I |
|---|---:|---:|---:|---:|---:|---:|
| M0, all three logs | 21 / 20 | .636446 | .153080 | .042873 | .268689 | .321547 |
| Drop per-capita CO₂ (proposed primary) | 20 / 20 | .636446 | .153080 | .042873 | .268689 | .321547 |
| Drop total CO₂ (registered sensitivity proposal) | 20 / 20 | .636446 | .153080 | .042873 | .268689 | .321547 |
| Drop population | 20 / 20 | .636446 | .153080 | .042873 | .268689 | .321547 |

| Representation | Geography | Responsibility | Socioeconomic | Population | Residual | Responsibility standalone R² | Population standalone R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | .513033 | .021393 | .069323 | .032697 | .363554 | .002740 | .059705 |
| Drop per-capita | .518837 | .013134 | .062821 | .041653 | .363554 | .001017 | .059705 |
| Drop total | .513520 | .020435 | .069282 | .033208 | .363554 | .00000516 | .059705 |
| Drop population | .513276 | .022190 | .069339 | .031642 | .363554 | .002740 | .056877 |

All shares are fractions of total outcome variance, not fractions of explained
variance. Full precision, every coalition R², the full frozen scorecard and
all country predictions are saved in `rank_audit*.json/csv`. Each row has
primary MAE .034465, absolute-error IQR .043031, one unseen-category row,
M49 secondary R² .302970 / RMSE .038894, and unchanged random-reference results.
Paired RMSE differences versus corrected M0 are numerical zero, not substantive
performance gains. The full-rank unscaled condition numbers are ~17,827,
13,727 and 12,232: the exact defect is removed, but these numbers still depend
on heterogeneous feature units and are not evidence that all correlations vanish.

## 3. Why Shapley shares change

LMG averages a group's increase in in-sample R² over every order in which the
four conceptual groups can enter. Removing a cross-group redundant column
preserves the **full coalition** but changes some **partial coalitions**.
For example, dropping per-capita CO₂ reduces responsibility-plus-income R²
from .187507 to .106570, even though responsibility-plus-population spans the
same space in both versions. This changes allocations to geography and income
as well as to responsibility and population.

Relative to V1, dropping per-capita changes total-variance shares by approximately:

- Geography: +0.580 percentage points.
- Responsibility: −0.826 percentage points (about −38.6% of its original share).
- Socioeconomic: −0.650 percentage points.
- Population: +0.896 percentage points.

The relative responsibility shift is material for its precise numeric meaning,
although its absolute share remains small. V1's .021393 is correct for its
frozen grouped representation; do not restate V1 using a V2 allocation.
For the primary and registered-sensitivity proposals, the responsibility group's
last-entry unique increment is the same .008682625 in R² (full .636446003 minus
geography+income+population .627763378). This is distinct from the Shapley share,
which includes shared/suppression structure across other entry orders. Near-zero
standalone responsibility R² does not imply its multivariate contribution must
also be zero.

## 4. Coefficients and meaningful instability

Coefficients below are changes per unit log10 predictor (all other columns
held fixed), not causal effects. In the redundant fit those three 'held fixed'
changes cannot be independently realized by the construction identity.

| Representation | Per-capita coefficient | Total coefficient | Population coefficient |
|---|---:|---:|---:|
| M0 minimum-norm representative | +.038508 | −.022736 | +.025640 |
| Drop per-capita | — | +.015772 | −.012868 |
| Drop total | +.015772 | — | +.002904 |
| Drop population | +.012868 | +.002904 | — |

These transformations agree with algebraically translating M0 beta along its
null vector, with maximum coefficient discrepancy 1.55e-14. Other columns retain
their full-model coefficients to numerical precision. The responsibility
coefficient in the primary/sensitivity proposals is positive in all 151 primary
training fits, range +.000777 to +.023139. Population's sign agrees with its
full-fit sign in 150/151 fits in each proposal, but that full-fit sign changes
between proposals: holding total emissions fixed and holding emissions per
person fixed are different conditional questions. Apparent sign stability in
the redundant least-norm fit is not identification. Full fold coefficient
ranges, signs and materialized-column counts are in the JSON and coefficient CSV;
'Iceland's own dummy' is absent from its training fit, not estimated from its
held-out outcome. These are diagnostics, not retention criteria in this audit.

## 5. Proposed contract and alternatives considered

### Primary proposal: total responsibility plus population

- Responsibility: `log10(cum_co2_total)` only, existing Mt units and historical
  construction unchanged.
- Population: `log10(population)` and existing untransformed `station_density`.
- Geography (all six features) and socioeconomic income categories unchanged.
- Drop `cum_co2_per_capita` from the V2 primary design; keep the recorded source
  value for sensitivity reporting. Do not mutate `SCHEMA_V1` or public artifacts.

National historical emissions scale is explicitly named as responsibility;
population remains in its declared group. This preserves information because
per-capita emissions can be recovered from the two logs and the constant.
Total emissions can remain correlated with population; full rank does not imply
conceptual/statistical orthogonality. The retained station-density feature also
means 'population group' is still V1's mixed demographic/sampling group, not a
new pure-population concept. Moving it is outside M0.5/M1a scope.

### Registered sensitivity proposal: per-person responsibility plus population

Replace the responsibility feature with `log10(cum_co2_per_capita)` and drop total;
retain population and every other predictor. This expresses a different
responsibility convention (historical emissions per cutoff-year resident).
Its denominator creates a population relationship but no exact rank redundancy
with the remaining matrix. Report both conventions together; do not select one
by its share or apparent coefficient sign. This is a recommendation pending the
owner's next-session contract approval, not an implemented V2 schema.

### Other representations

Dropping population is mathematically sound but keeps recoverable demographic
scale inside the emissions group and leaves only station density in the group
named population. It is the least clean conceptual separation and is diagnostic,
not the registered sensitivity. Rescaling/centering the redundant columns cannot
restore rank. An SVD/QR basis or orthogonalizing emissions against population
can span the same space but mixes group meanings or privileges an entry order;
these do not identify a uniquely correct Shapley allocation. Combining the two
groups avoids cross-group ambiguity by changing the scientific question, so it
is not a replacement for this four-group contract. The three minimal deletions
exhaust the two-of-three choices retaining the current named representations.

## 6. Interpretation and verification

The conclusion supported here is narrow: historical national responsibility
provides little explanatory variance for **cross-country differences in the
area-weighted national warming rate** under these descriptive grouped models.
The allocation is associational, not causal. It does **not** say greenhouse-gas
emissions do not cause global warming, nor that emissions' global warming effect
is measured by this country-level decomposition. Rank repair does not address
product differences, station-derived geography, omitted physical processes or
within-country heterogeneity.

Recovered from the interrupted run: all four completed scorecards, full/fold
coefficients, country predictions and deterministic-repeat checks. The audit
was not restarted under a different specification. Focused tests independently
verify exact-null behavior, equal fitted spaces and a hand-derived synthetic
Shapley redistribution; saved real-data outputs are validated on resume.
Reproduce with `python -m research.model_v2.rank_audit`. Corrected distance files
are read-only; the runner writes only `outputs/rank_audit*`.
