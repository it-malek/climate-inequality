# M2 predictor-only structural feasibility audit

**Recorded 2026-09-16, before any M2 fit. No outcome was read, nothing was estimated, and no residual,
RMSE, R² or coefficient was computed.** The audit shows that the frozen M2 candidate (`M2_DESIGN.md`)
is structurally executable under the frozen protocols. It does not show that the candidate is useful.

## 1. What was run

```
uv run python -m research.model_v2.m2_feasibility
```

* **Where it ran.** From audit code commit **`fbaa3ff`**, which was clean and already pushed; the audit
  refuses to write its canonical record otherwise. The output went to `outputs/m2_feasibility/`.
* **Code coverage.** The record's code digests cover the audit module, the basis module, and every local
  module they import: `cv.py`, `src/feature_schema.py`, `src/decomposition.py` and `src/stability.py`.
* **Dry-run agreement.** An earlier uncommitted dry run produced byte-identical fit and coalition
  tables.

**Inputs.** All four are digest-pinned, and every read is an explicit column allow-list:

| Input (pin) | Columns read | Why |
|---|---|---|
| `outputs/m0_countries.csv` (`9e99b379…`, the M1b scoring pin) | `Country`, `iso3`, the M49 subregion label and the four categorical predictor labels. Seven numeric columns are read separately, only to measure their rounding. | categorical labels |
| `outputs/m1b_primary/m1b_design_matrices.csv` (`bbdc8b1d…`) | the six artifact columns. All rows are loaded, including the M1b rows' C2 values; M0\* rows are selected in memory. The file holds no outcome. | exact full-sample M0\* scoring values of the numeric predictors, log10 already applied |
| `outputs/m1b_primary/m1b_cv_folds.csv` (`6a9e3976…`) | identifiers, primary training memberships, M49 and random-10 fold ids | frozen fold memberships |
| `outputs/m1a_geography_features.csv` (`a62d04d4…`, the M1b scoring pin) | `iso3`, the five remeasured features and `spatial_block` | the conditional M1a-measurement arm |

**No outcome.**
* Every allow-list is screened for outcome and model-derived tokens, and so are the encoded design's
  `M0star` column names.
* The audit module contains no estimator call. Tests enforce this at source level, and a test replaces
  `V1Design.fit` and `numpy.linalg.lstsq` with raising stubs.

**A finding about inputs.** The numeric predictor columns of `m0_countries.csv` are **rounded copies**.
They differ from the exact scoring values by up to 4.6e-8 relative, and latitude by up to 4.8e-7°. They
must not be used as M2 predictors, so the audit takes numeric values from the frozen encoded design.

**Encoder and identity.**
* The encoder is M0\*'s: V1 feature order, train-only categorical levels and drop-first coding. Tests
  show it equals `cv.V1Design` bit for bit.
* The full-sample encoding matches the frozen `M0star` design value for value, **after aligning columns
  by name**. The frozen artifact is in schema order; the V1 encoder puts numerics before dummies.
* The numeric part of that identity holds by construction. The independent evidence is that the labels
  rebuild **every frozen dummy column bit for bit**, the column and group sets agree, and the inputs are
  digest-pinned.

## 2. Structural result: **pass** in every required fit

### 2.1 Station design (the M2 candidate)

| Representation / protocol | Fits | n_train | M0\* columns | Ranks = columns | Added identifiable dims | SH in training |
|---|---:|---|---|---|---|---|
| total CO₂ / full sample | 1 | 151 | 20 | 20 → 23 | 3 | 29 |
| total CO₂ / primary 500 km LOCO | 151 | 124–150 | 19–20 | all | 3 in every fit | 18–29 |
| total CO₂ / M49 leave-one-subregion-out | 20 | 136–150 | 19–20 | all | 3 in every fit | 20–29 |
| total CO₂ / random 10-fold (reference) | 10 | 135–136 | 19–20 | all | 3 in every fit | 24–28 |
| per capita / all of the above | 182 | same | same | all | 3 in every fit | same |

* **Column counts.** A count of 19 reflects a categorical level legitimately absent from that training
  fit; the candidate always adds exactly three columns and three rank.
* **Southern latitude support.** In every training fit, southern latitudes span at least 32.08°.
* **Identical across representations.** Ranks, knots and hemisphere counts are identical between the
  two representations.
* **Coalitions.** In all 16 group coalitions the candidate's rank exceeds M0\*'s by exactly 3 when
  `geography` is present and by 0 otherwise. For example, geography alone 14 → 17 and all four groups
  20 → 23. Every coalition matrix has full column rank.

### 2.2 Conditional M1a area-measurement arm

M2_EVALUATION_CONTRACT.md §10 defines this arm. The five remeasured geography features replace the
station ones, and the latitude state is recomputed per fit.

It also passes everywhere: full sample 20 → 23, and +3 identifiable dimensions in every primary, M49 and
random fit in both representations. Differences from the station design:
* **Knots.** Full-sample knots are 1.2511 / 15.9297 / 36.3154 / 64.9723°.
* **Southern range.** Southern latitudes reach 41.65°.
* **Hemisphere.** Gabon and Kenya change hemisphere.
* **Columns.** Primary training fits all have 20 M0\* columns.

## 3. Knots (station design)

| Protocol | ξ₁ (lower boundary) | ξ₂ (1/3 quantile) | ξ₃ (2/3 quantile) | ξ₄ (upper boundary) | Smallest gap |
|---|---|---|---|---|---:|
| full sample | 0.8 | 14.948 | 35.745829 | 65.09 | 14.148° |
| primary (range over 151 fits) | 0.8 – 1.791 | 12.598 – 19.017 | 30.529 – 39.599 | 61.556 – 65.09 | 11.798° |
| M49 (range over 20 fits) | 0.8 – 1.337 | 13.889 – 18.849 | 34.114 – 39.226 | 53.749 – 65.09 | 13.089° |
| random 10-fold (range) | 0.8 – 1.337 | 14.465 – 16.87 | 34.56 – 38.007 | 58.66 – 65.09 | 13.665° |

* **No degeneracy.** The smallest consecutive gap anywhere is 11.8°, far above the 1e-6° tolerance.
* **Ties.** The exact scoring latitudes take 130 distinct values across 151 countries. No repeated knot
  results.
* **Serialization.** The full-sample state is recorded exactly, as hex knots, in
  `m2_feasibility_summary.json` → `full_sample` → `station/<representation>` → `latitude_state`. That
  field, not the decimal CSV columns, is the reference for any later exact comparison.

## 4. Held-out extrapolation (linear tails, disclosed in advance)

| Protocol | Fit | Rows below ξ₁ | Rows above ξ₄ | Training range | Held-out range |
|---|---|---:|---:|---|---|
| primary | Gabon held out | 1 | 0 | 1.337 – 65.09 | 0.8 |
| primary | Iceland held out | 0 | 1 | 0.8 – 61.556 | 65.09 |
| M49 | Middle Africa held out (fold 8) | 1 | 0 | 1.337 – 65.09 | 0.8 – 12.32 |
| M49 | Northern Europe held out (fold 11) | 0 | 8 | 0.8 – 53.749 | 52.80 – 65.09 |
| random 10-fold (reference) | fold 5 | 1 | 0 | 1.337 – 65.09 | 0.8 – 49.99 |
| random 10-fold (reference) | fold 6 | 0 | 4 | 0.8 – 58.66 | 6.16 – 65.09 |

**Unseen categorical levels in held-out rows** (inherited behaviour): primary 1 row, M49 12 rows,
random 1 row.

## 5. Conditioning (audited, not tuned)

Condition numbers are given two ways. Raw means the unscaled matrix; column-equilibrated means every
column is scaled to unit norm, which is independent of units.

| Maximum across fits | M0\* equilibrated | Candidate equilibrated | M0\* raw | Candidate raw |
|---|---:|---:|---:|---:|
| total CO₂, primary | 110 | 164 | 1.91e4 | 2.91e4 |
| total CO₂, M49 | 106 | 163 | 1.88e4 | 2.77e4 |
| per capita, primary | 58 | 161 | 1.48e4 | 2.21e4 |
| M1a arm, total CO₂, primary | 110 | 151 | 2.02e4 | 3.11e4 |

* **Identification margin.** Scale the three added columns to unit norm and project out the M0\* column
  space. The smallest singular value of the remainder has a minimum of **0.018589** over all fits, reached
  in station primary fits: the added columns are identified with a clear margin.
* **No transformation needed.** Condition numbers of order 1e4 are harmless in double precision for SVD
  least squares and for `matrix_rank`'s default tolerance. No centering, scaling or other transformation
  was chosen or needed.
* **Platform scope.**
  * The basis values themselves use IEEE products, and knots come from `numpy.quantile` under the
    `uv.lock`-pinned numpy.
  * Singular values and condition numbers are platform-scoped in their last bits.
  * The structural evidence is the ranks, knots and counts.

## 6. Decision

**Structural feasibility holds for the frozen candidate in every required fit, in both representations,
and for the conditional M1a arm.** The contract may be frozen as specified. No df, knot, basis,
centering or column change was considered or needed.

## 7. Artifacts

`outputs/m2_feasibility/`:
* **`m2_feasibility_fits.csv`**: one row per measurement × representation × protocol × fit (728 rows),
  with every field above and the serialized latitude state.
* **`m2_feasibility_coalitions.csv`**: 64 coalition rank rows.
* **`m2_feasibility_summary.json`**. It holds:
  * inputs, pins and columns read, and the numeric source;
  * the code commit, push state and digests;
  * encoded identity and full-sample states;
  * per-protocol summaries, including the extrapolating fits;
  * checks and verdicts, and software versions.
* **`m2_feasibility_manifest.json`**: SHA-256 of the three files.

`test_m2_feasibility.py` checks the committed record against the committed code digests, the input
pins, its own manifest and its key sets. It fails if this document is committed without the record. The
test suite never re-runs the audit.
