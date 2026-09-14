# Stability of the decomposition

`src/stability.py` re-runs the Shapley/LMG decomposition
(`src/decomposition.py`) under resampling and single-country deletion and
tests the spatial structure of its residual. It adds no estimator: every number
is the same decomposition on a perturbed sample, so the layer speaks only to
how stable the variance shares are, never to significance or causation. The
output is `stability_summary.json`, rendered on the dashboard's
"How confident are we?" page.

## What is computed

- **Country bootstrap.** Resample the 154 complete-case countries with
  replacement 2,000 times (seed 0), recompute the full decomposition each time,
  and report the mean, standard deviation and 2.5/97.5 percentile of every
  share including the residual; the fraction of resamples in which geography is
  the largest named axis; and the fraction in which the emissions share is
  positive. A resample that makes the design unusable (for example a
  categorical level vanishing) is skipped and counted, never substituted.
- **Continent block bootstrap.** The same, but resampling whole continents
  (seed 1) so that spatial correlation within a continent is respected. Its
  intervals are wider; the gap between the two is the spatial-dependence
  correction for the shares.
- **Leave-one-country-out influence.** Drop each country in turn and record
  how far every share moves; report the ten most influential countries per
  share.
- **Moran's I on the residual.** Row-standardised k-nearest-neighbour weights
  (k = 8) on country centroids (the mean of each country's station
  coordinates), a two-sided permutation p-value from 199 label permutations
  (seed 0).

## Result

| Share | Point | Bootstrap 95% interval | Block-bootstrap interval |
|---|:-:|:-:|:-:|
| Geography | 0.455 | 0.392–0.563 | 0.414–0.573 |
| Emissions | 0.084 | 0.051–0.132 | 0.028–0.189 |
| Socioeconomic | 0.056 | 0.034–0.114 | 0.029–0.123 |
| Population | 0.036 | 0.013–0.076 | 0.003–0.080 |
| Residual | 0.369 | 0.251–0.404 | 0.181–0.433 |

Geography is the largest axis in 100% of country resamples and the emissions
share is positive in 100%. The bootstrap means sit above the point estimates
for every named axis (and below for the residual) because the decomposition
overfits slightly on a resample with duplicated rows; the intervals are the
honest quantity. Leave-one-out influence is small: no single country moves the
geography share by more than 0.018 (Mongolia, Botswana) or the emissions share
by more than 0.008 (Afghanistan).

Moran's I on the full-model residual is 0.327 (p = 0.005, n = 154): the 37% of
variance the four axes do not explain is regionally clustered, pointing to
regional processes (or regionally correlated station sampling) that none of
the features capture.

## Not done

- A construction-sensitivity table (median instead of mean aggregation of city
  slopes, alternative cutoff years, total instead of per-capita emissions).
  Two of the largest construction choices, population- and area-weighted
  country means, are instead handled as separate lenses in the coupling stage.
- Conley (spatial HAC) standard errors on the legacy emissions coefficient, and
  a spline latitude control. These target the single coefficient rather than
  the shares and were dropped when the project moved to the decomposition.
- Leave-one-continent-out fits.
