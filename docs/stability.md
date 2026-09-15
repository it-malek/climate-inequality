# Stability of the decomposition

`src/stability.py` re-runs the Shapley/LMG decomposition
(`src/decomposition.py`) under resampling and single-country deletion and
tests the spatial structure of its residual. It adds no estimator: every number
is the same decomposition on a perturbed sample, so the layer speaks only to
how stable the variance shares are, never to significance or causation. The
output is `stability_summary.json`, rendered on the dashboard's
"How confident are we?" page.

The primary run is on the area-weighted outcome; the station-weighted outcome
on the same countries gets the bootstrap and Moran's I blocks under a
`sensitivity` key, so the reader can see how much of each interval depends on
how national warming was constructed.

## What is computed

- **Country bootstrap.** Resample the 151 complete-case countries with
  replacement 2,000 times (seed 0), recompute the full decomposition each time,
  and report the mean, standard deviation and 2.5/97.5 percentile of every
  share including the residual; the fraction of resamples in which geography is
  the largest named axis; and the fraction in which the emissions share is
  positive (a Shapley share of R² cannot be negative, so this is 1 by
  construction and is reported for completeness). A resample that makes the
  design unusable (for example a categorical level vanishing) is skipped and
  counted, never substituted.
- **Continent block bootstrap.** The same, but resampling whole continents
  (seed 1) so that spatial correlation within a continent is respected. Its
  intervals are wider; the gap between the two is the spatial-dependence
  correction for the shares.
- **Leave-one-country-out influence.** Drop each country in turn and record
  how far every share moves; report the ten most influential countries per
  share. Primary outcome only.
- **Moran's I on the residual.** Row-standardised k-nearest-neighbour weights
  (k = 8) on country centroids (the mean of each country's station
  coordinates), a two-sided permutation p-value from 199 label permutations
  (seed 0).

## Result

Area-weighted outcome (primary), 151 countries:

| Share | Point | Bootstrap 95% interval | Block-bootstrap interval |
|---|:-:|:-:|:-:|
| Geography | 0.513 | 0.447–0.623 | 0.384–0.577 |
| Emissions | 0.021 | 0.009–0.054 | 0.011–0.197 |
| Socioeconomic | 0.069 | 0.040–0.134 | 0.040–0.145 |
| Population | 0.033 | 0.013–0.074 | 0.004–0.109 |
| Residual | 0.364 | 0.235–0.407 | 0.189–0.504 |

Station-weighted outcome on the same 151 countries (sensitivity):

| Share | Point | Bootstrap 95% interval | Block-bootstrap interval |
|---|:-:|:-:|:-:|
| Geography | 0.458 | 0.396–0.572 | 0.417–0.577 |
| Emissions | 0.085 | 0.052–0.133 | 0.028–0.189 |
| Socioeconomic | 0.062 | 0.039–0.118 | 0.031–0.123 |
| Population | 0.029 | 0.010–0.066 | 0.003–0.080 |
| Residual | 0.365 | 0.249–0.394 | 0.176–0.432 |

Geography is the largest axis in 100% of country resamples under both
outcomes, with overlapping intervals. The emissions share's country-bootstrap
intervals (0.01–0.05 area-weighted, 0.05–0.13 station-weighted) touch only at
the boundary; its block-bootstrap interval is wide under both because with six
continents a resample can omit Europe or duplicate Asia. The bootstrap means
sit above the point estimates for every named axis (and below for the
residual) because the decomposition overfits slightly on a resample with
duplicated rows; the intervals are the honest quantity. Leave-one-out
influence on the primary outcome is small: no single country moves the
geography share by more than 0.018 (Egypt, Brazil, Canada) or the emissions
share by more than 0.007 (Syria).

Moran's I on the full-model residual is 0.269 (p = 0.005, n = 151) under the
area-weighted outcome and 0.314 under the station-weighted one: the 36% of
variance the four axes do not explain is regionally clustered, pointing to
regional processes (or regionally correlated observational structure) that
none of the features capture.

## Not done

- A construction-sensitivity table beyond the outcome weighting (median
  instead of mean aggregation of city slopes, alternative cutoff years, total
  instead of per-capita emissions). The population-weighted country mean is
  handled as a lens in the coupling stage only.
- Conley (spatial HAC) standard errors on the legacy emissions coefficient, and
  a spline latitude control. These target the single coefficient rather than
  the shares and were dropped when the project moved to the decomposition.
- Leave-one-continent-out fits.
