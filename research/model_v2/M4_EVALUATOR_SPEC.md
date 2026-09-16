# M4 evaluator specification — frozen before the first M4 computation

**Recorded 2026-09-16. Committed and pushed before any M4 computation on the warming outcome.** The method,
scope and completion criteria are [`DOWNSTREAM_COMPLETION_SPEC.md`](DOWNSTREAM_COMPLETION_SPEC.md) §3 and
[`M4_EVIDENCE_INVENTORY.md`](M4_EVIDENCE_INVENTORY.md), frozen before any M2 fit (`4dc2fad`; Amendment A1
`ef04958`). This file fixes execution order and artifacts only. Implementation: `m4_evaluate.py`;
reconstruction: `m4_independent_check.py`.

## 1. Execution boundary and prerequisites

`m4_evaluate.main()` refuses unless: `PYTHONHASHSEED=0`; a fresh output root; a clean, pushed code closure
(transitive local imports, this file, the downstream specification, the inventory, `pyproject.toml`,
`uv.lock`); the M2 primary result and conditional disposition, the M3 station package and the M3
land-centroid package all verify against their manifests and are committed and pushed; the station
package's final naming agrees with the M2 branch; the legacy ERA5 file matches `5a18ccba…` and the M0.5
audit record; the aligned ERA5 file matches its summary digest; and the M2 scoring-input gate passes.

## 2. Order

1. **Static share uncertainty** for S (both representations): country bootstrap (`default_rng(0)`,
   `integers(0, 151, 151)` per draw), then continent block bootstrap (`default_rng(1)`, `choice` over
   `pandas.unique` of `spatial_block` in canonical order), 2,000 draws each. Each draw re-encodes its own levels,
   re-fits the latitude state in branch B and computes the 16-coalition projection-R² LMG. Degenerate draws are
   counted and skipped; rank-deficient usable draws are counted.
2. **Buffer sensitivities**: territorial 1000 km (the `4263429` lower bounds) and land-centroid 1500 km
   (`m0_countries.csv` centroids, round-trip parsed), for S, each qualifying extension and, in branch B, M0\*.
   M0\*'s CV R², RMSE and MAE must reproduce the committed M0 sensitivity cards to 1e-9 (refusal otherwise).
3. **Product check**: aligned ERA5 then legacy ERA5, both representations, for S and each qualifying
   extension, with the §3.5 anchors (branch A: S in-sample R² equals the M0.5 `r2.era5` to 1e-10; branch B:
   the aligned M2 card reproduces the committed M2 conditional arm, if computed, to 1e-10).
4. **Consolidation** of committed stage evidence; **inventory status**; **final specification record**.

## 3. Artifacts (deterministic unless stated)

| File | Contents |
|---|---|
| `m4_bootstrap_draws.csv` | every draw: representation, bootstrap kind, draw, rows, usable, failure, rank-deficient flag, four named shares and residual |
| `m4_bootstrap_summary.json` | per representation and kind: point, mean, SD, percentile interval per share; `p_geography_largest`; geography-margin 2.5th percentile; counts; material-change flags |
| `m4_sensitivities.json`, `m4_sensitivity_predictions.csv` | per protocol × representation: scores, anchors, paired intervals, dispositions; predictions |
| `m4_products.json`, `m4_product_predictions.csv` | per product × representation: S card, point materiality, residual agreement, anchors; extensions' cards, accounting and `arm_conditions`; `product_sensitive`; predictions |
| `m4_consolidated_table.json` / `.csv` | one row per stage and arm with status, scores, paired interval, worst region, OOF Moran and the registered allocation quantities; legacy block and separations |
| `m4_inventory_status.json` | final status of every inventory row |
| `m4_final_specification.json` | the three naming fields, M2 label, stopping indicator, M3 dispositions, three numbers per retained stage, material change, commits, residual description |
| `m4_provenance.json` | code closure digests and commit, inputs, prerequisite manifests, software |
| `m4_result_manifest.json` | SHA-256 of every deterministic artifact; inventory completeness; final naming |
| `m4_run_metadata.json` | not deterministic |

## 4. Verification

`m4_independent_check.py` (no import of `m4_evaluate`, `m2_evaluate` or `src.decomposition`) recomputes every
bootstrap draw (the draws are regenerated from the same seeds and refitted by QR projections with permutation
Shapley), the
percentile summaries and material-change flags, the sensitivity scores and paired intervals from saved
predictions, the M0\* anchors, the product-check S in-sample R² and residual agreement, and the consistency of
the final record with the committed M2 and M3 manifests. Negative controls: a perturbed draw share, a shifted
percentile, a changed naming field. One unchanged-code rerun must be byte-identical.
