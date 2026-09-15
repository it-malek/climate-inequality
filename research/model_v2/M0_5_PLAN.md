# Validation correction and M0.5 execution plan

Approved scope: preserve the frozen GPW/ISO units; correct enforcement of 500 km
territorial LOCO without tuning the threshold, estimator, unseen-level handling,
M49 secondary, or scorecard. No V1 changes, new predictors or M1a implementation.

## Commit 1: territorial CV enforcement

- [x] Record ontology and geometry before corrected outcomes are examined.
- [x] Add tests for shared borders, disconnected islands, long countries,
  antimeridian, threshold clearance, deterministic folds and NIC–COL regression.
- [x] Implement `territory.py`: implicit unions of every GPW assigned 0.25° cell
  footprint on WGS84, centre ECEF trees, conservative footprint-distance bounds.
  Retain only pairs whose lower bound exceeds 500 km. No sampled centres are
  reported as exact territory distance. Missing source components remain a source
  limitation, not an unacknowledged numerical guarantee.
- [x] Create a separate corrected-scorecard runner and new artifacts: old/new
  training membership, fold changes, pair witnesses, certified clearance, full
  corrected scorecard and complete legacy/corrected comparison.
- [x] Repeat geometry/fold/prediction computation for determinism; compare M49
  numeric metrics. Run research/full tests and ruff; verify legacy/production
  hashes. Commit only the correction, its tests/results/docs and existing evidence.

## Commit 2: substantive M0.5

- [ ] Rank audit on exact M0: three deletions of the redundant log representation,
  coefficient-null-space evidence, full scorecards and group standalone R²;
  recommend a primary and registered sensitivity based on conceptual separation.
- [ ] Product audit on unchanged M0 predictors: exact source/temporal comparability,
  matched-country agreement, residual/global/local spatial diagnostics, region and
  country tables; label robust/sensitive/unresolved patterns without adding inputs.
- [ ] Specify area-consistent measurements for every current geography feature;
  sources, weighting, masks, multipart land, missingness, bathymetry and leakage.
- [ ] Focused tests, full/research tests, ruff, reproducibility and production/legacy
  integrity checks; research-only commit. Do not push without research authorization.
- [ ] Report results and stop before M1a.
