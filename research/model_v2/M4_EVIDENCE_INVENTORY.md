# M4 requirement-to-evidence inventory (frozen before any M2 fit)

**Recorded 2026-09-16, pre-M2.** Governing rules: [`DOWNSTREAM_COMPLETION_SPEC.md`](DOWNSTREAM_COMPLETION_SPEC.md)
§3. S is the retained static specification (branch A: M0\*; branch B: M2). "Ext." means a qualifying
station-weight spatial extension. Every row must end in verified evidence or its permitted disposition;
missing work is never "not applicable". The M4 evaluator writes the final status of every row to
`outputs/m4_final/m4_inventory_status.json`.

**Reuse rule.** Committed evidence is reused only for an identical estimand (sample and order, outcome
values, estimator, representation, groups, memberships, weights, method), by artifact SHA-256.

| ID | Requirement | Provenance | Applies to | Compatible committed evidence | Missing work | Permitted non-executed / not-applicable disposition |
|---|---|---|---|---|---|---|
| I1 | One scorecard table M0, M1a, M1(b), M2, M3 | `af97bd2` plan M4; `SCORECARD.md` | all stages | `m0_scorecard_territory_corrected.json`, `rank_audit.json` (M0\* cards), `m1a_scorecard.json`, `m1b_primary/m1b_scorecard.json`; M2 and M3 results when committed | assemble only | a cell a stage's frozen evaluator never computed is labelled "not recorded by the frozen <stage> evaluator"; rejected stages get no refit to fill it |
| I2 | Legacy 1° CV kept separate | `TERRITORIAL_CV_CORRECTION.md` (`4263429`) | M0 | `m0_scorecard.json` | label only | — |
| I3 | V1 allocations separate from V2 full-rank allocations | `V2_PREDICTOR_CONTRACT.md` | M0 vs M0\* | V1 bundle (`app/data/decomposition_summary.json`, `stability_summary.json`); rank-audit cards | label only | — |
| I4 | Paired ΔRMSE and intervals vs registered comparator | plan M4; `SCORECARD.md` metric 6 | M1a, M1b, M2, M3 families | M1a, M1b committed; M2 result | M3 (spec §2.9) | — |
| I5 | Per-region RMSE and bias, primary and secondary | `SCORECARD.md` | all stages | recorded in M0\*, M1a, M1b cards (primary and `secondary`); M2 result | M3 | as I1 |
| I6 | OOF residual Moran's I | plan M4 ("three numbers"); `SCORECARD.md` metric 9 | all stages | committed cards | M3 | — |
| I7 | Effective degrees of freedom | `SCORECARD.md` metric 13 | all stages | committed cards | M3 (rank + 1) | — |
| I8 | Random 10-fold reference | `SCORECARD.md` | all stages | committed cards | M3 | — |
| I9 | Per-capita representation alongside every score and share | `V2_PREDICTOR_CONTRACT.md` | all V2 rows | committed per stage | every new computation | — |
| I10 | Bootstrap share intervals: country (2,000, seed 0) and continent block (seed 1) | plan M4; `SCORECARD.md` | **S** only, both representations | none (V1 intervals are M0's rank-deficient grouping) | compute (spec §3.2) | M1a, M1b, non-promoted M2, M0\* in branch B, spatial families: **not applicable** (not a retained OLS stage / no historically defined share) |
| I11 | Material-change evaluation of the retained final model | plan "Material change" | S (point + country-bootstrap interval; block reported) | — | compute (spec §3.2) | — |
| I12 | Stopping rule | plan M4; `M2_EVALUATION_CONTRACT.md` §11; owner 2026-09-16 | M1a, M1b, M2 | M1a +0.002343, M1b +0.001723 committed; M2 result | report | — |
| I13 | M3 SEM and SAR, station-centroid kNN8: scores, qualification, dependence parameter, accounting, non-spatial-part group contributions | plan M3 | both families on S | none | M3 station package | family **non-computable** under spec §2.8 (no fallback); accounting still reported iff the full-sample fit is valid |
| I14 | Spatial range | plan M3 | both families | — | report graph-distance summaries | range **not applicable** (fixed kNN8 has no estimated range) |
| I15 | Spatial uncertainty | spec §3.3 | computable families | — | model-based se and Wald interval of θ̂; training-fit θ̂ distribution | bootstrap of accounting parts or filtered shares **not applicable / not historically defined** |
| I16 | Land-centroid kNN8 weight sensitivity | plan M3 | both families on S | none | M3 land-centroid package (always run) | non-computable disposition per family |
| I17 | 1000 km territorial buffer sensitivity | `SPATIAL_CV_PROTOCOL.md`; `4263429` | S; each Ext.; M0\* in branch B | M0 card in `m0_scorecard_territory_corrected.json` (anchor, not a substitute) | compute (spec §3.4) | non-qualifying families and rejected candidates: **not executed** (not retained); failed cell: non-computable |
| I18 | 1500 km land-centroid buffer sensitivity | same | same | same | compute | same |
| I19 | Product check of the eventual model, aligned ERA5 (preferred) | `9fb56e1` audit Decision and §7.3; register M0.5 status | S; each Ext. | M0.5 `r2.era5` anchor (branch A); M2 conditional aligned arm anchor (branch B) | compute (spec §3.5) | arm non-computable if any outcome is non-finite; rejected candidates and non-qualifying families not executed |
| I20 | Product check, legacy ERA5 (preprocessing sensitivity) | `9fb56e1` §7.3 ("using both constructions") | S; each Ext. | M0.5 legacy `r2.era5` anchor (branch A) | compute | legacy M2-vs-M0\* paired replication **not executed** (M2 product replication frozen aligned-only, contract §10) |
| I21 | Two-product-mean outcome | `af97bd2` plan M4, conditional on owner approval | — | — | none | **not executed**: never authorized; excluded by owner 2026-09-16 |
| I22 | Measurement (geography remeasurement) sensitivity | `M1A_REPORT.md` §13; `M2_EVALUATION_CONTRACT.md` §10 | S | branch A: `m1a_scorecard.json`; branch B: M2 conditional M1a arm | none | spatial extensions: **not executed** (no registered M3 geography remeasurement) |
| I23 | Coefficient sign stability (recorded, not criteria) | `SCORECARD.md` | covariate stages | M1b C2 summary; M2 γ summary | M3 θ̂ same-sign fraction (descriptive) | basis coefficients: not interpreted (M2 design §6) |
| I24 | Residual share description and limitations | plan M4 | final report | — | closure | never "unknown climate cause" |
| I25 | Final naming fields | owner 2026-09-16; spec §2.14 | — | — | M4 record | — |
| I26 | Decadal-variability (H8), unit-of-analysis (H9), station-density (H10) diagnostics | `HYPOTHESIS_REGISTER.md` | — | — | none | **not pursued** (register) |
| I27 | Optional geostatistical family; C1, C3; alternative latitude forms | plan M3; M1b design; M2 contract | — | — | none | **not pursued**, not experimentally rejected |
