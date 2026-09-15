# Spatial cross-validation protocol (frozen)

> **Implementation correction:** the original 1° distance description and scores
> below are preserved as history. The current implementation is specified in
> `TERRITORIAL_CV_CORRECTION.md`; the approved territory ontology is at the end
> of this document. The intended 500 km protocol is unchanged.

The validation design was chosen before any predictor work and is fixed for
every stage M1–M4. It was chosen on design criteria; the M0 scores under each
candidate were computed to check feasibility and noise, not to pick a winner.
Code: `research/model_v2/cv.py` (folds, estimator wrapper, scorecard),
`borders.py` (the buffer metric), `run_cv_design.py` (the candidate comparison,
`outputs/cv_design_candidates.csv` and `.json`, fold memberships in
`outputs/cv_fold_membership.csv`).

## The question the protocol must answer

Can a model explain warming differences in countries or regions that were not
effectively represented by nearby training observations? Random K-fold cannot,
because neighbouring countries share the residual process: the M0 residual is
positively autocorrelated to about 1500 km between land centroids and to about
500 km between territories (`M0_RESIDUAL_DIAGNOSTICS.md` §3), so a held-out
country whose neighbours are in training is predicted partly from that shared
process rather than from transferable structure.

## Primary protocol: leave-one-country-out with a 500 km territorial buffer

**Fold construction.** 151 folds, one per country. For fold *i* the training
set is every country *j* whose minimum inter-territory distance to *i* exceeds
500 km, where the inter-territory distance is the smallest great-circle
distance between any 1° land cell assigned to *i* and any assigned to *j*
(`outputs/country_border_distance_km.csv`; adjacent countries are 40–160 km
apart at this resolution). The model is fit on the training set and predicts
country *i*; the 151 out-of-fold predictions are pooled for the scores.

**Why the territorial metric.** A buffer on centroid distance leaves the
neighbours of large countries in training: the centroids of Russia, Canada,
the United States and Mexico are more than 1500 km from every other centroid,
so a 1500 km centroid buffer removes nothing for them (Canada would be
predicted with the United States in training). On the territorial metric the
residual correlogram is positive only in the 0–500 km ring (I = 0.33,
p = 0.001), indistinguishable from zero at 500–1000 km (I = 0.03, p = 0.52),
and negative beyond 1000 km; on the centroid metric the positive range extends
to 1500 km because centroid distances are inflated for large countries. The
width is therefore the first territorial ring with no positive dependence:
500 km. It is frozen; a later stage with a different residual range does not
change it.

**Leakage prevention.** Every bordering and near-bordering country is out of
training for every fold (the nearest training territory is at least 500 km
away, median 597 km). The only remaining shared information is through
countries beyond 500 km, which the correlogram shows carry no positive
residual dependence.

**Unseen categorical levels.** A test country whose level of a categorical
feature is absent from its training fold receives the mean of the training
level effects (`unseen="mean_effect"`). Under the primary protocol this affects
one row: Iceland, the only Köppen E country, whose in-sample residual is zero
by construction and whose out-of-fold error is therefore the honest one.

**Determinism.** No random element: the fold set is the country set, the
buffer is a fixed threshold on a committed distance table, ties are impossible
(strict inequality on continuous distances), and the estimator is a
least-squares solve. Two runs produce identical scores.

**Training-set sizes.** 128–150 countries (mean 142.7): the estimator's
variance stays close to the full-sample fit, so differences between models
reflect transferable structure rather than fold-size noise. Countries with the
smallest training sets are in Europe and around the Black Sea (dense
neighbourhoods); New Zealand loses no training row.

**M0 under the primary protocol** (`outputs/m0_scorecard.json`,
`outputs/m0_cv_errors_primary.csv`):

| Metric | M0 |
|---|---|
| Spatial-CV R² (pooled) | 0.212 |
| Spatial-CV RMSE | 0.0414 °C/decade |
| Spatial-CV MAE | 0.0332 |
| Calibration slope / intercept | 0.66 / +0.060 |
| Out-of-fold residual Moran's I (V1 weights) | 0.327 |
| Worst sub-region (n ≥ 3) | Central Asia, RMSE 0.069 |
| In-sample R² for comparison | 0.636 |

The gap between 0.636 and 0.212 is the amount of M0's fit that does not
transfer to a country whose neighbourhood is unseen; under random 10-fold the
R² is 0.428, so roughly half of the optimism is spatial.

**Weaknesses, stated.** (1) Difficulty is heterogeneous: an isolated country
(New Zealand, Iceland, Sri Lanka) loses few training rows and is effectively a
leave-one-out case, a European country loses twenty. This is inherent to the
question (isolated countries have no near neighbours to leak from) and is
reported through the per-region table. (2) Single-country folds have no
per-fold R²; pooled metrics and per-region RMSE are used instead. (3) Training
sets overlap heavily, so per-country errors are dependent; uncertainty on the
pooled metrics comes from a country bootstrap of the per-country errors, not
from fold-to-fold variance. (4) The width is derived from M0's residual;
freezing it means later stages are tested at M0's correlation length, which is
the conservative choice. (5) Negative residual dependence at 1000–4000 km means
that distant training countries are, for M0, slightly anti-informative; the
protocol does not correct for it and no model is credited for it.

## Secondary stress test: leave-one-UN-M49-sub-region-out

20 folds (sizes 1–15; `regions.py`). Holds out a whole contiguous region and
asks whether the model explains a region it has never seen, with an
interpretable region-level error table. Kept because it tests transfer at a
larger scale than the primary and because its errors map onto the clusters in
the residual diagnostics. Caveats: South America is both a sub-region and a
continent, so its 11 countries are predicted with an unseen `spatial_block`
level (mean-effect rule); Melanesia has one country; fold sizes are uneven.
M0: R² 0.303, RMSE 0.0389, worst region (n ≥ 3) Central America 0.062. The
secondary cannot make a model pass; it can veto an improvement that appears
only under the primary.

## Reference and sensitivities (reported, never criteria)

* Random 10-fold, seed 0 (M0: R² 0.428, RMSE 0.0352): the non-spatial
  reference that shows the optimism gap.
* Border buffer 1000 km (M0: R² 0.072, RMSE 0.0449) and centroid buffer 1500 km
  (M0: R² 0.225, RMSE 0.0410): to show that a model ranking does not depend on
  the exact width or metric.

## Designs evaluated and not adopted

| Design | M0 R² | Why not |
|---|---|---|
| Leave-one-continent-out (6 folds) | −0.40 | V1's continent effect is unidentifiable for the held-out block (every test row has an unseen level); it tests the fold design, not the model |
| k-means clusters of centroids, k = 6–15 | 0.08–0.25 | fold sizes 1–38, 4–29 rows with unseen levels, and the score swings with an arbitrary k |
| Absolute-latitude bands (5 folds) | 0.24 | removes whole regimes and tests extrapolation along the main gradient, which is not the question |
| Leave-one-country-out, no buffer | 0.485 | median nearest training country 470 km: neighbours leak, and the score sits next to random K-fold's |
| Region hold-outs with an added buffer | 0.28–0.29 | the buffer adds little to a contiguous hold-out and complicates the region table |

## Nested validation and model selection

Any quantity tuned to data (a spline's degrees of freedom, a penalty, a
spatial range) must be chosen inside each training fold (inner buffered
leave-one-out on the training set) or fixed a priori; M2 fixes its degrees of
freedom a priori to avoid nesting. Decisions to retain a covariate or stage are
made on the primary protocol, with the secondary as a veto; the sensitivities
are for the report. The protocol is not revised after M1–M3 results are seen.

## Is n = 151 enough?

For M0 the pooled RMSE has a country-bootstrap 95% interval of roughly
±0.004 °C/decade, so a paired comparison (same folds, same countries) resolves
differences of about 0.002–0.003. Improvements smaller than that are reported
as point estimates and not called improvements. The per-region table is
descriptive only; regions have 1–15 countries.

## Approved implementation correction: frozen analytical territory units

Recorded before corrected M0 scores are known. Territory means the frozen
GPW/ISO country unit used for the V1 outcome. All represented land components,
including islands and disconnected components, count; separately coded units
remain separate. FRA does not absorb GUF. This is analytical-unit consistency,
not a statement about sovereignty or political status. A sovereign-state
exclusion scheme asks a different question and is not a Model V2 sensitivity.

The preceding Session 1 numeric results and 1° centre-distance description are
**legacy results of a defective approximation**, not evidence of guaranteed
500 km territorial clearance. Their files are preserved. The intended protocol
is still LOCO with <=500 km territory exclusion, unchanged M49 secondary,
train-only mean-level-effect fallback and unchanged scorecard definitions.

The correction uses the full footprints of the WGS84 GPW v4 rev11 15′ national
identifier cells (band 11, the same lookup as V1), with a conservative geometric
clearance certificate. It does not call finer cell centres exact boundaries.
Method and refreshed results will be recorded in `TERRITORIAL_CV_CORRECTION.md`.
