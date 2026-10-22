# Stability of the decomposition

The decomposition allocates cross-country warming variance among groups of
attributes. Resampling and single-country deletion describe how those shares
change when the observed country sample is perturbed. They do not identify
causes, correct missingness bias or establish inference for excluded countries.

## Population and method

The point estimates use 151 complete countries under both area-weighted and
station-weighted warming. The revised country bootstrap samples those same
151 countries with replacement 2,000 times, using seed 0 and the same country
order for both outcomes. Every draw refits the unchanged OLS coalitions and
Shapley/LMG allocation. Failures are recorded without replacement; both new
country-bootstrap arms completed all 2,000 draws with zero failures.

The historical implementation instead sampled 157 area-outcome rows or 154
station-outcome rows, then dropped incomplete rows inside each fit. Its retained
sample size therefore varied across draws. Point estimates already used 151
complete cases. This difference changes the bootstrap population, rather than
the estimator: the released 2.1.0 intervals condition on the observed complete
countries. They make no adjustment for why other countries are missing.

The area design excludes Gambia, Mauritius and Guyana for missing climate
classes, and Bahrain, Hong Kong and Singapore for missing area-weighted warming.
The historical station bootstrap starts from 154 rows and excludes the same
three countries with missing climate classes. Both revised samples therefore
contain the same 151 countries.

The continent bootstrap draws six whole continents with replacement, 2,000
times using seed 1. All retained country identities, multiplicities and estimator
inputs were checked across all 2,000 draws before reusing the completed block
summaries. This is a sensitivity to one coarse block scheme. With only six
heterogeneous continents, its intervals are not a calibrated spatial-dependence
correction and need not always be wider than country-bootstrap intervals.

Leave-one-country-out influence and residual diagnostics continue to use the
151-country point-estimate population. Residual Moran's I uses row-standardised
k-nearest-neighbour weights, k = 8, on mean retained source coordinates, with
199 label permutations and seed 0.

## Revised legacy-decomposition results

These tables retain the original grouped representation, including total and
per-capita responsibility alongside population. The displayed intervals use the
new complete-case country bootstrap; block intervals retain exactly verified
completed calculations.

Area-weighted warming, 151 complete countries:

| Share | Point | Country-bootstrap 95% interval | Continent-bootstrap interval |
|---|:-:|:-:|:-:|
| Geography | 0.513 | 0.445–0.624 | 0.384–0.577 |
| Emissions | 0.021 | 0.009–0.054 | 0.011–0.197 |
| Socioeconomic | 0.069 | 0.040–0.134 | 0.040–0.145 |
| Population | 0.033 | 0.013–0.078 | 0.004–0.109 |
| Residual | 0.364 | 0.233–0.407 | 0.189–0.504 |

Station-weighted warming on the same 151 countries:

| Share | Point | Country-bootstrap 95% interval | Continent-bootstrap interval |
|---|:-:|:-:|:-:|
| Geography | 0.458 | 0.396–0.567 | 0.417–0.577 |
| Emissions | 0.085 | 0.051–0.132 | 0.028–0.189 |
| Socioeconomic | 0.062 | 0.037–0.121 | 0.031–0.123 |
| Population | 0.029 | 0.010–0.067 | 0.003–0.080 |
| Residual | 0.365 | 0.248–0.396 | 0.176–0.432 |

Geography is the largest named group in all 2,000 country draws under each
outcome. That is a finite resampling result, not a probability that geography
caused the warming. Likewise, LMG shares cannot be negative, so the fraction of
positive emissions shares is not a significance test. Bootstrap means differ
from the point estimates when rows are duplicated; percentile intervals here
describe the chosen perturbation rather than guaranteed sampling coverage.

The largest single-country deletion changes the geography share by 0.0181
(Egypt) and the emissions share by 0.0066 (Syria). Residual Moran's I is 0.269
for area warming and 0.314 for station warming, with p = 0.005 in each recorded
199-permutation test. This supports descriptive residual clustering; it does
not distinguish omitted physical processes from correlated product structure.

## Relation to the retained full-rank model

The separate V2 analysis retains M0*, which removes the algebraically redundant
per-capita emissions column. Its fitted values and residual share match the
legacy model, but its named allocations differ because the groups no longer
share that redundant column relationship. Its own fresh 151-country M4 bootstrap
uses 2,000 country draws and 2,000 continent draws in each of two responsibility
representations. It is separate from the legacy bootstrap-population revision.

In the primary total-CO₂ representation, geography accounts for 0.519 of total
variance (country-bootstrap interval 0.452–0.630), responsibility 0.013
(0.006–0.035), and the residual 0.364 (0.233–0.407). Area remeasurement of
geography remains a measurement sensitivity. Baseline dryness and the richer
latitude specification do not meet their registered support criteria; the
latitude improvement interval narrowly includes zero. These negative and
inconclusive findings are retained, rather than interpreted as proof that the
added features have no association.

Both spatial error and spatial lag models meet the fixed Berkeley Earth
prediction criteria. Neither is uniquely selected, and their predictive gains
do not reproduce under the aligned ERA5 outcomes. Their covariance accounting
is not a physical explanation for the residual.

## Shared source locations and neighbour ties

The 3,510 city-location records occupy 1,366 distinct coordinate pairs; 2,821
records belong to shared-coordinate groups. The original source already has
rounded coordinates and, within shared locations, identical monthly values.
Official Berkeley Earth location documentation supports shared source-location
assignment; these are not 3,510 independent weather-station measurements.
No duplicated identity-month records or processing-introduced coordinate
rounding were found. Every city record remains included.

Finite row-order perturbations changed historical neighbour graphs but left
the tested significance decisions unchanged. The
`identity-distance-ties-v1` policy made graphs, Moran statistics and permutation
results invariant across the tested orders. This provides reproducibility,
not independent observations or proof of permutation exchangeability. The
replacement interpolation comparison also keeps coincident locations together
and fits the variogram within each training fold; the old full-field-variogram
comparison remains historical.

## Limits and evidence

The 2.1.0 analysis follows the frozen bootstrap-population comparison and
accepted October 2026 validation records. Exact draws, failure counts, population
exclusions and block-equivalence checks are retained with that run. The public
bundle's `stability_summary.json` carries the revised legacy intervals;
`residual_structure_summary.json` carries the separate full-rank results.

Trend uncertainty and predictor measurement error are not propagated into
these shares. Country weighting is equal, within-country variation is outside
this analysis, and dependence can extend beyond the territorial exclusion.
Several follow-up hypotheses were motivated by these same outcomes. The
national decomposition and the global conditional hindcast address different
questions and are not combined into a causal attribution.
