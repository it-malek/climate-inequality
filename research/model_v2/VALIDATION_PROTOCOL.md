# Validation protocol for the residual-structure investigation

Every candidate extension of the static model was judged under one fixed
protocol, chosen before any predictor was measured and never revised after a
result was seen. This page states that protocol, the full-rank representation
of the static model that every comparison uses, and the decision rules.

## 1. The question the protocol answers

Can a model explain warming differences in countries whose neighbourhood it has
not seen? Random K-fold cross-validation cannot answer this, because
neighbouring countries share the residual process: the static model's residual
is positively autocorrelated out to about 500 km between territories
([`RESIDUAL_DIAGNOSTICS.md`](RESIDUAL_DIAGNOSTICS.md) §3), so a held-out
country whose neighbours are in training is predicted partly from that shared
process rather than from transferable structure. Under random 10-fold the
static model's R² is 0.428; under the primary protocol below it is 0.153, and
the gap is the part of the in-sample fit that does not transfer.

## 2. Primary protocol: leave-one-country-out with a 500 km territorial exclusion

**Folds.** 151 folds, one per country. For the fold that holds out country *i*,
the training set is every country *j* whose minimum territory-to-territory
distance from *i* exceeds 500 km. Every bordering and near-bordering country is
therefore out of training for every fold. Training sets hold 128–150 countries
(mean 141.3).

**Territorial distance.** A country's territory is the union of the 15
arc-minute GPW v4 rev11 national-identifier cell footprints (band 11, the same
lookup the public pipeline uses), including islands, disconnected components
and separately coded units; nothing is aggregated by sovereignty. Clearance
between two countries is certified conservatively: cell centres are converted
to WGS84 Earth-centred coordinates, the minimum centre-to-centre chord over all
cell pairs is found with an exact KD-tree query, and the lower bound

    distance >= min_chord - 2r - 1 mm,   r = (a^2 / b) * sqrt(2) * h = 19.745 km

(with *h* half a cell width in radians and *a*, *b* the WGS84 semi-axes)
accounts for any point inside either footprint. A pair is retained only when
the bound exceeds 500 km; uncertain pairs are excluded. The smallest retained
bound is 500.68 km. The certificate cannot falsely admit a pair within 500 km
on the adopted geometry; it can exclude some pairs whose true separation is
slightly above 500 km (nineteen unordered pairs have an upper bound above
500 km). The geometry, bounds and every directed fold membership are committed
(`outputs/territory_*.csv`, `outputs/territory_gpw_footprints.npz`).

An earlier implementation measured distance between 1° cell centres and did
not guarantee the 500 km clearance; its scores (CV R² 0.212, RMSE 0.0414) are
kept in `outputs/m0_scorecard.json` and `outputs/m0_cv_errors_primary.csv` as
a superseded approximation. The corrected scores (CV R² 0.153, RMSE 0.0429)
are the baseline for every comparison; 100 of 151 folds changed and 109
unordered pairs were newly excluded.

**Why 500 km.** On the territorial metric the residual correlogram is positive
only in the 0–500 km ring (I = 0.33, p = 0.001), indistinguishable from zero at
500–1000 km (I = 0.03) and negative beyond 1000 km. A centroid-distance buffer
would leave the neighbours of large countries in training (the centroids of
Russia, Canada, the United States and Mexico are more than 1500 km from every
other centroid), which is why the metric is territorial.

**Unseen categorical levels.** A held-out country whose level of a categorical
feature is absent from its training fold receives the mean of the training
level effects. Under the primary protocol this affects one row (Iceland, the
only Köppen E country).

**Determinism.** The fold set is the country set, the threshold is fixed on a
committed distance table, and the estimator is a least-squares solve. Runs are
byte-identical with `PYTHONHASHSEED=0` (the group Shapley routine iterates a
`frozenset`, which otherwise perturbs the last bit of the shares).

## 3. Secondary and reference protocols

* **Leave-one-UN-M49-sub-region-out** (20 folds of 1–15 countries): holds out a
  contiguous region and asks whether the model transfers at a larger scale. It
  cannot make a candidate pass; it can veto an improvement that appears only
  under the primary protocol. South America is both a sub-region and a
  continent, so its countries are predicted with an unseen continent level.
* **Random 10-fold** (seed 0): the non-spatial reference that shows the optimism
  gap. Never a criterion.
* **Exclusion-width sensitivities**: a 1000 km territorial exclusion and a
  1500 km land-centroid exclusion, reported for the retained models only.

Designs evaluated and rejected before the protocol was fixed: leave-one-continent-out
(the continent effect is unidentifiable for the held-out block), k-means
centroid clusters (fold sizes 1–38 and a score that swings with k),
absolute-latitude bands (tests extrapolation along the main gradient),
unbuffered leave-one-country-out (median nearest training country 470 km, so
neighbours leak), and region hold-outs with an added buffer.

## 4. Scorecard

Every stage reports the same quantities on the pooled out-of-fold predictions:
CV R², RMSE and MAE under the primary protocol; the M49 R² and RMSE; the random
reference; the worst eligible sub-region (n >= 3) RMSE; fold RMSE median and
maximum; the IQR of absolute out-of-fold error; calibration slope and
intercept; Moran's I of the in-sample and out-of-fold residuals on the fixed
station-centroid kNN8 weights (row-standardised, 999 permutations); the number
of held-out rows with an unseen categorical level; in-sample R² and the group
Shapley shares; the sign stability of each coefficient across training fits.
All errors are in °C/decade.

**Paired comparison.** Two models are compared on the same folds and the same
countries. The difference in out-of-fold RMSE is given a 95% interval from a
country bootstrap of the paired per-country errors (2,000 resamples, seed 0,
no refitting). This interval treats countries as exchangeable; it does not
correct for the spatial dependence between overlapping folds, and the
protocol's geographic safeguards are the 500 km exclusion, the M49 hold-out
and the worst-region veto. For the static model the pooled RMSE has an
interval of roughly ±0.004 °C/decade, so a paired comparison resolves
differences of about 0.002–0.003.

## 5. Decision rules

A candidate demonstrates **predictive support** when its paired RMSE change
relative to the comparator is negative and the interval's upper end is below
zero. A physical covariate must in addition carry its pre-stated coefficient
sign in the full fit and in at least 80% of the primary training fits;
improvement with the wrong sign is not support for the mechanism.

A supported candidate is **promoted** to replace the comparator only when the
change is at most −0.002 °C/decade, the M49 RMSE does not rise by more than
+0.001, the worst eligible sub-region RMSE does not rise by more than +0.001,
and every required fit is structurally valid (full column rank, no degenerate
basis). A candidate that is not supported is closed; no alternative window,
transform, dataset, threshold, knot placement or partial specification is
tried in response to its result.

For the spatial models the same five conditions define **qualification for
prediction** (paired change below zero with the interval below zero; M49 and
worst-region vetoes; computable in every required fit). Two qualifying
families are reported side by side; no rule selects between them.

A **static stopping indicator** was defined on the point estimates: it fires
when the best primary RMSE change over all static candidates is worse than
−0.002 °C/decade, which would have declared the residual not recoverable with
country-level static covariates. It did not fire (best change −0.00295), so
the narrower statement in [`RESULTS.md`](RESULTS.md) applies.

A **material change** to the public conclusion would be any of: geography no
longer the largest named group at the point estimate; the country- or
block-bootstrap interval failing to establish it as largest; or the
responsibility share exceeding 0.10. None occurred.

Diagnostics that are reported but never decide anything: the residual Moran's
I change (a change smaller than 0.05 is "essentially unchanged"), overfitting
flags (in-sample gain exceeding the CV gain by more than 0.05; worse CV RMSE
with better in-sample RMSE; a degrees-of-freedom rise), calibration, and
per-country and per-region error changes.

## 6. The full-rank representation of the static model

The public model's design has 21 columns (with the intercept) but rank 20:
cumulative CO₂ per capita is the cumulative total divided by the same 2013
population that enters the population group, so after the log transforms

    log10(per capita) - log10(total, Mt) + log10(population) = 6

holds for every country (maximum deviation 1.8e-15). The least-norm solve
returns the projection R², so every public number is unaffected, but the group
Shapley allocation depends on which representation of the redundant direction
is used (`outputs/rank_audit.json`; `rank_audit.py`).

Every comparison in this investigation therefore uses **M0\***: the same
countries, outcome, folds and estimator, with the per-capita column dropped so
that the responsibility group holds log cumulative total CO₂ and the
population group holds log population and station density. Fitted values,
predictions, residuals and CV scores are identical to the public model to
numerical precision (differences at 1e-14); only the allocation differs:

| Share of total variance | Public representation (rank 20 of 21) | M0\* (rank 20 of 20) |
|---|---:|---:|
| Geography | 0.5130 | 0.5188 |
| Historical responsibility | 0.0214 | 0.0131 |
| Socioeconomic | 0.0693 | 0.0628 |
| Population | 0.0327 | 0.0417 |
| Residual | 0.3636 | 0.3636 |

The representation that keeps per-capita CO₂ and drops the total (geography
0.5135, responsibility 0.0204) is carried as a registered sensitivity
throughout; its predictions agree with M0\* to 1e-13 in every stage, and its
verdicts are always identical. Shares under the two representations are never
mixed in one table.

## 7. Provenance guards

Outcome-facing evaluators refuse to start unless `PYTHONHASHSEED=0` is set, the
transitive closure of local modules they import is committed and identical to
the head of the remote branch, the manifests of every upstream stage they read
verify against their digests, and the output directory does not exist. Each
run writes a result manifest with the SHA-256 of every deterministic artifact
and a provenance file listing the code closure and input digests. These guards
are engineering controls on reproducibility; they do not change any number.


## Diagnostic tail convention

Frozen global and local Moran permutation p-values compare absolute statistics,
`abs(I_permuted) >= abs(I_observed)`, with the plus-one Monte Carlo correction.
They do not compare distances from the global expectation `-1/(n-1)` or the
row-specific conditional expectation. This is an absolute-statistic randomization
tail, not a symmetric two-sided tail around the null expectation. Local maps
and their FDR labels are exploratory; their dependence and same-data hypothesis
use prevent reading them as independently confirmed regional mechanisms.
No diagnostic values or labels were recalculated in the repository audit.
