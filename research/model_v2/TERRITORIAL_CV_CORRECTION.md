# Correcting territorial-distance enforcement in spatial CV

## Decision and scope (fixed before corrected scores)

The intended protocol is unchanged: leave one country out and exclude countries
with territory <=500 km from the test country's territory. Primary threshold
500 km; M49 secondary, random reference role, unseen-category handling, scorecard
and M0 outcome/predictors unchanged. This corrects implementation, not model fit.
The owner approved preserving V1's GPW/ISO analytical units: separately coded
territories are not aggregated by sovereignty. This is not a political-status
claim. No sovereign-state sensitivity is introduced.

## Geometry and certified conservative approximation

Source: the existing GPW v4 rev11 15-arc-minute national identifier band 11 and
lookup used by V1. The file declares `+proj=longlat +datum=WGS84 +ellps=WGS84`
and creation date 2018-11-16. Input SHA-256, lookup SHA-256 and a compact derived
snapshot of **all** assigned cells for the 151 units are recorded in
`outputs/m0_scorecard_territory_corrected.json`. The geometry is the union of
closed 0.25° latitude-longitude **cell footprints**, not their centre points.
Implicit footprints avoid topology loss from polygon simplification. All assigned
cells participate; no outcome coverage, climate trend or station filter is used.

At the equator a cell is approximately 27.8 km north–south by 27.8 km east–west;
east–west width shrinks with latitude. This is a finite-resolution analytical
territory representation, not exact shoreline geometry. GPW's sub-grid omissions,
classification and country assignment remain source limitations: there is no
claim to certify unrepresented real-world islands or borders. Within the adopted
GPW footprint geometry, discretization cannot leave a <500 km pair in training.
No additional geometry-fidelity ambiguity is concealed by this claim.

For each cell centre, convert geodetic latitude/longitude to WGS84 Earth-centred
Earth-fixed (ECEF) Cartesian coordinates. Let `a,b` be ellipsoid semiaxes and
`h=0.125°` in radians. The WGS84 meridional and normal curvature radii are both
bounded above by `Rmax=a²/b`. A path linear in unwrapped longitude/latitude from a
cell centre to any cell point has surface length at most

`r = Rmax × sqrt(2) × h`, approximately **19.745 km**.

This follows from `ds²=M² dφ² + N² cos²φ dλ²` and `M,N <= Rmax`.
ECEF chord distance is no greater than surface geodesic distance. For any two
points in any footprints of two units, the Euclidean triangle inequality gives

`geodesic distance >= minimum ECEF centre chord − 2r`.

The global minimum centre chord is found deterministically with exact (eps=0)
3D KD-tree queries over every assigned cell. The retained-pair certificate is

`lower = max(0, min_chord − 2r − 0.000001 km) > 500 km`.

One millimetre is subtracted for numerical roundoff. Bounds are stored at full
float precision, not rounded before thresholding. Equality and uncertain cases
are excluded. This is an explicitly conservative approximation: the two-cell
radius allowance is about 39.49 km; chord versus surface length adds further
conservatism (about 0.2 km for a ~540 km chord, not a global error bound).
It can exclude some pairs whose true footprint separation exceeds 500 km; it
cannot falsely certify clearance for a pair within 500 km on the adopted geometry.
The threshold has not been increased or tuned: uncertain distance estimates are
resolved toward exclusion, as authorized.

For audit, a WGS84 geodesic between the nearest-chord centre pair provides a
valid upper bound on minimum footprint distance. It is a witness, not necessarily
the nearest geodesic centre pair. The pyproj `Geod(ellps='WGS84')` inverse is used.
Closed cells that share an edge/corner have distance zero. All disconnected
components and represented enclaves/exclaves participate without dissolving
country codes. ECEF and wrapped-longitude overlap tests handle the antimeridian;
no planar longitude-distance shortcut or country centroid is used.

### Verification and limits

A retained pair must have the certified lower bound >500 km. This verifies the
entire implicit footprint union, not only a sampled boundary. The saved upper
bound can distinguish definite-neighbour exclusions from conservatively uncertain
ones; their count is reported. No exact polygon-boundary distances are claimed.
This representation was chosen from the existing environment for analytical-unit
fidelity and its provable clearance property before corrected scores were known.

## Reproducibility and audit trail

Run `.venv/bin/python -m research.model_v2.run_territory_correction`.
The runner repeats geometry and primary predictions and requires bitwise equality.
It recomputes legacy primary and M49 scores and checks existing numeric metrics
within 1e-10. New artifacts are separate:

- `territory_gpw_footprints.npz`: all centre coordinates and implicit 0.25° footprints.
- `territory_distance_lower_km.csv`, `territory_distance_upper_km.csv`: bounds.
- `territory_cv_membership_comparison.csv`: every old and new directed membership,
  including self-exclusion (151 × 151 rows).
- `territory_cv_fold_changes.csv`: changed folds, training counts, added/removed
  neighbours, per-fold clearance certificates.
- `territory_cv_changed_pairs.csv`: unordered changed pairs and distance evidence.
- `m0_cv_errors_territory_corrected.csv`: corrected per-country predictions/errors.
- `m0_scorecard_territory_corrected.json`: complete metrics and comparison.

All Session 1 outputs, including `m0_scorecard.json`, `m0_cv_errors_primary.csv`,
`country_border_distance_km.csv`, candidate fold memberships and correlograms,
remain unchanged as **legacy 1° approximation** results. They do not satisfy the
intended territorial-clearance guarantee. Historical candidate-design code is
not the current primary scorer. `run_territory_correction` is the corrected entry.
M49 and random reference membership are unchanged; their nearest-distance
ancillary fields retain the legacy metric to reproduce the historical scorecard.
The corrected primary's `nearest_train_km` fields are explicitly **lower bounds**.

The runner supplements the existing scorecard implementation with already-defined
metrics (effective rank, permutation p-values, fold error IQR, sign stability,
paired RMSE bootstrap); it changes no score definition. Baseline-vs-itself paired
RMSE is zero. Legacy-vs-corrected paired differences are descriptive correction
effects, not an improvement-selection exercise.

## Corrected M0 results

| Metric | Legacy 1° approximation | Corrected footprint certificate |
|---|---:|---:|
| cv_r2 | 0.211980511 | 0.153079913 |
| cv_rmse | 0.0413548918 | 0.0428725807 |
| cv_mae | 0.0332032072 | 0.0344645875 |
| fold_error_iqr | 0.0419797291 | 0.0430306433 |
| fold_rmse_median | 0.029367841 | 0.0302126784 |
| fold_rmse_min | 0.000122568473 | 5.20551845e-05 |
| fold_rmse_max | 0.0959523138 | 0.114734551 |
| mean_n_train | 142.715232 | 141.271523 |
| rows_with_unseen_level | 1 | 1 |
| calibration_slope | 0.660266955 | 0.609411497 |
| calibration_intercept | 0.0600581796 | 0.0689358527 |
| residual_morans_i_cv | 0.326598571 | 0.321547168 |

Worst single-country fold: Iran → Brazil. Worst eligible region remains Central Asia (RMSE 0.069319946 → 0.069364378).
All errors are °C/decade. Fold dispersion is IQR of absolute country errors.

100/151 folds changed, with 218 newly excluded directed relations (109 unordered pairs), no newly retained pairs. All 88 preflight witness pairs are excluded. The geometry contains 249,801 assigned cells. Nineteen excluded unordered pairs have an upper bound above 500 km: these are explicitly conservative/uncertain exclusions, not claimed exact <=500 km neighbours.

Smallest retained clearance lower bound: 500.682949 km. Both independently rebuilt geometry matrices and both primary prediction arrays are bitwise identical. M49 secondary reproduces every historical numeric field within 1e-10: R²=0.302970020, RMSE=0.038894137, 12 rows with unseen levels. Primary unseen handling remains one row. The current implementation supplements the scorecard without changing those definitions.

Corrected minus legacy RMSE: +0.001517689; paired country-bootstrap 95% interval [0.000674982, 0.002595330] (2,000 resamples, seed 0). This interval is descriptive and conditional on the highly overlapping folds; country resampling does not remove their spatial dependence. The change is recorded as an enforcement correction, not performance optimization.

V1 in-sample values and all Session 1 output bytes remain unchanged. No new predictor or outcome was introduced. The correction is independent of the later rank and product audits.

## Validation completed

Full project + research suite: **617 passed, 5 skipped, 3 warnings** (20.13 s).
Ruff and `git diff --check` pass. Independent code review found no material
issues in the bound or score comparability; its requested source-identity test
now verifies every saved cell against GPW, FRA/GUF separation and the actual
Nicaragua–Colombia island witness. The focused territorial suite has 13 tests.
Production paths and every tracked Session 1 output compare byte-for-byte to
`af97bd2`; the `v1.3.0` tag still resolves to `a6733cb78292b9b27466e7ebfe7c4fdc6b59592f`.
