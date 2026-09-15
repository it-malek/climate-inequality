# M1b amended C2 build — stopped at a pre-outcome hard stop

**Recorded 2026-09-15. PRE-OUTCOME GATE FAILED — the warming outcome is still untouched.**

## Status

* **Stop.** Both builds under `M1B_EVALUATION_CONTRACT.md` Amendment 1 (build commit `f27d0a0`) stopped at hard stop A1.7, "non-structural invalid support cell", in one Peru cell. The contract sends this stop to owner review.
* **Not done.** No redundancy diagnostic was run. No warming outcome, residual or score was loaded for M1b, and nothing was fitted.
* **Not frozen.** C2 is not frozen. The build outputs are kept below as evidence only, and they are not the M1b predictor.
* **No new rule.** No rule, threshold or definition was changed after the stop, and no further amendment was made.

## 1. The builds

* **Determinism.** Two builds from a clean tree at `f27d0a0` were byte-identical in all five deterministic outputs. Build A wrote to `outputs/` and build B to a separate root, each in its own process with a different `PYTHONHASHSEED`.
  * Wall time was 17.9 s and 16.8 s, and peak resident memory about 3.6–3.8 GB.
  * The manifest records `code_path_matches_commit: true`, and the builder sha256 `92ba87657a8d0ef1b72334f9d8cc78219c183b65c42099d65d3f6fdfec3ae300` equals the blob at `f27d0a0`.
* **Invariants.** Every check against the preserved original-rule build (`c5ff9b5`) holds exactly (`m1b_amendment1_stop_record.json`):
  * total and native-valid areas;
  * native fraction equals the original coverage;
  * native-only C2 equals the original C2, bit for bit;
  * C2 is unchanged in every country with no harmonized cell;
  * native P̄/Ē means and all §2.5 station-support columns;
  * the support checkpoint is byte-identical;
  * non-native cell counts equal the original invalid-cell counts, and there are 66,501 native-valid cells globally.

## 2. What passed

**Coverage.** All 151 countries reach ≥ 0.98. The minimum is Bahamas at 0.98875, and no other country is below 0.99.

| ISO3 | Native fraction (original coverage) | Harmonized fraction | Unresolved fraction | Amended coverage |
|---|---:|---:|---:|---:|
| BHS | 0.8865 | 0.1023 | 0.0112 | 0.9888 |
| PAN | 0.9728 | 0.0272 | 0.0000 | 1.0000 |
| PHL | 0.9761 | 0.0224 | 0.0015 | 0.9985 |

**Harmonization** (distinct cells unless stated):
* **Targets.** 2,752 structural-mask target cells:
  * 2,408 harmonized, covering 122,615 km² summed over country rows;
  * 344 unresolved (no native one-ring donor), covering 6,576 km².
* **Variables masked** in the harmonized cells: both variables in 1,889; PET only in 519.
* **Donor distance** (WGS84): minimum 6.56 km, median 52.24 km, mean 45.11 km, 95th percentile 72.46 km, maximum 78.45 km.
* **Cross-border.** 56 cross-border (country, cell) rows cover 2,188 km². The 2,408 harmonized cells draw on 2,072 distinct donor cells, of which 4 support none of the 151 countries.
* **Largest harmonized fractions:** BHS 0.1023, PAN 0.0272, PHL 0.0224, NLD 0.0175, GRC 0.0136, CUB 0.0130, NZL 0.0098, IDN 0.0083, KOR 0.0081, PNG 0.0079.
* **Largest unresolved fractions:** BHS 0.0112, KOR 0.0015, PHL 0.0015, JPN 0.0009.

**Size of the harmonization change.** This is QA only, not a scored quantity. Of 151 countries, 98 contain a harmonized cell. Among them the median |Δ C2| (harmonized minus native-only) is 8.5e−5. The largest changes are:

| ISO3 | Harmonized fraction | Δ C2 |
|---|---:|---:|
| BHS | 0.1023 | −0.00522 |
| ECU | 0.0054 | −0.00204 |
| GRC | 0.0136 | −0.00167 |
| CHL | 0.0027 | +0.00143 |
| IDN | 0.0083 | −0.00092 |

**Station support.** Unchanged: computed over native cells, and the checkpoint is byte-identical to the original. The five lowest PRE station-supported shares are SAU 0.276, YEM 0.288, HTI 0.453, OMN 0.510 and TCD 0.547, the registered sensitivity subset of Amendment 2. No threshold was added.

## 3. The stop

| Item | Value |
|---|---|
| CRU cell | row 164, column 200; centre 7.75°S, 79.75°W (north coast of Peru) |
| Country and land | Peru, 23.01 km² (1.79e−5 of Peru's terrestrial area) |
| PET | fill value in all 1,500 months of the record (CRU's fixed PET mask) |
| PRE | finite in all months, and exactly 0.0 in every one of the 1,500 months; P̄ = 0 mm/yr |
| PRE station support | ≥ 1 contributing station in 240 of the 360 window months (maximum 5) |
| State (A1.2) | PRE `nonpositive_mean`, PET `masked` → non-structural |
| Peru coverage | 0.999982 (the cell is unresolved) |

* **Why it is non-structural under A1.2.** The cell has no valid C2 for two independent reasons: PET is masked, and precipitation is identically zero, so log10(P̄/Ē) is undefined even with PET. It therefore does not lack C2 "solely because" of the fixed mask, which is the owner rule's condition for harmonization.
* **Why it stops.** A1.7 halts on any non-structural invalid support cell, for owner review, even though Peru's coverage passes.
* **Diagnosis: a methodological ambiguity, not an implementation bug.** The builder applies A1.2 and A1.7 as written, and the invariants above confirm the native construction.

**Decision needed from the owner.** Nothing has been chosen, and each option would be a new, dated, pre-outcome amendment:
* keep the cell unresolved and lift the A1.7 non-structural stop for it;
* treat a PET-masked cell whose PRE is identically zero across the whole record as structural, and harmonize it by the one-ring rule;
* another resolution.

Either named choice moves Peru's C2 by at most 1.79e−5 × |x_donor − C2_PER|.

## 4. Erratum to the feasibility audit and to Amendment 1's evidence statement

Three committed statements are wrong for exactly this one cell:
* `M1B_FEASIBILITY_AUDIT.md` §5 item 3 ("none has P̄ ≤ 0 or Ē ≤ 0");
* Amendment 1's "Why it was needed" ("no invalid support cell has … a non-positive climatological mean");
* A1.2's closing sentence ("the audit found … no such support cell").

**Cause.** The audit script (`feasibility/m1b_original_coverage.py`) put a cell in its non-positive-mean class only when both PRE and PET were complete. This cell has PET masked, so the audit counted it as "PET-only masked, PRE valid", which is true of its finite PRE values but not of its mean. The first review of Amendment 1 flagged exactly this gap. The final A1.2 rule closes it, and that is why the amended build found the cell.

The contract text is not edited after C2 values exist; this record is the correction.

## 5. Artifacts and reproduction

`outputs/m1b_amendment1_stopped_build/`:
* `m1b_hydroclimate_features.csv`, `m1b_hydroclimate_qa.csv`, `m1b_harmonization_cells.csv`, `m1b_support_checkpoint.json` and `m1b_measurement_manifest.json`: build A's outputs, byte-identical to build B;
* `m1b_build_run.json` and `m1b_build_run_b.json`: wall-clock metadata;
* `m1b_amendment1_stop_record.json`: determinism hashes, invariants, coverage, harmonization and the raw facts of the stop cell.

Reproduce as follows.
1. Run both builds from a checkout of `f27d0a0`. The manifest records the build commit, so building at a later commit changes that one field.
2. Run the record script, committed in `bd518a0`, against those build outputs.

```
python -m research.model_v2.m1b_hydroclimate            # at f27d0a0
python -m research.model_v2.m1b_hydroclimate BUILD_B    # at f27d0a0
python -m research.model_v2.feasibility.m1b_amendment1_stop BUILD_B
```
