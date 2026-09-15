# M1b feasibility audit under the original coverage rule

**Recorded 2026-09-15. This is a measurement-feasibility audit, not a model result.**

## Status at the time of this audit

* **C2 was built deterministically.** Three independent builds with the committed builder are byte-identical in all four outputs.
* **The original 98% terrestrial-coverage rule failed in three countries:** Bahamas, Panama and the Philippines.
* **The cause is a structural mismatch** between CRU TS's fixed 0.5° land mask and the finer frozen M1a terrestrial support.
* **No warming outcome had been accessed.** The builder and this audit read only country identifiers, the M1a land support, the GPW country grid and the CRU PRE/PET files.
* **No redundancy diagnostic had been run, with or without the outcome.** No redundancy output exists in the working tree or in any commit, and `m1b_redundancy.py` refuses to start while measurement hard stops fail.
* **No M1b model score existed.** No scorecard or CV-error output exists, and nothing was fitted.
* **No threshold was relaxed.** No country was dropped, filled, imputed or given a neighbour value.

The build stopped at the frozen gate (`M1B_EVALUATION_CONTRACT.md` §2.4), as the contract requires. The owner has approved a pre-outcome coverage amendment, which is recorded and committed separately. Nothing in this audit applies it.

## 1. What was built

* **Contract:** `M1B_EVALUATION_CONTRACT.md` at `60111ae` (pushed before acquisition).
* **Builder:** `m1b_hydroclimate.py` at commit `c6ef919`, sha256 `57cf5b64ad87fab6f6e7554ce97b163567c4b32372b513212d21f94f1f70bfee`. This is the implementation commit `ef09196` plus the station-count validation fix `c6ef919`.
* **The fix in `c6ef919`.** The first build attempt stopped before writing any output or computing any C2 value:
  * the station-count check had been applied to the whole global raster, where CRU stores `stn` fill values over non-land cells;
  * contract §2.5 defines the audit over valid cells;
  * the check now runs over valid cells and still stops on any missing or out-of-range count there.

  This was an implementation defect under the frozen contract, not a rule change.
* **Formula (unchanged):** for each CRU cell, `x_j = log10(P̄_j / Ē_j)` over 1920-01…1949-12, with PET multiplied by calendar days. C2 is the M1a-terrestrial-area-weighted mean of `x_j` over valid cells.

## 2. Source verification

| File | Bytes | SHA-256 |
|---|---:|---|
| `cru_ts4.10.1901.2025.pre.dat.nc.gz` | 698,006,393 | `b7e7a3b74e9887db74d8db62227800708ff89ebbabeed5b11f302c99a551edaa` |
| `cru_ts4.10.1901.2025.pet.dat.nc.gz` | 72,951,284 | `fcc792b090a651d9a19187acf105cd726d680171d01ee48630f0a1a10816f013` |
| `cru_ts4.10.1901.2025.pre.dat.nc` | 6,220,812,084 | `efe27c453101fd8b98b0544a198f457b872cf983a9cadb791ecd8e9cddbb3eae` |
| `cru_ts4.10.1901.2025.pet.dat.nc` | 1,555,211,692 | `0bc9a1319f76d93a9f8fb8517ce82399b8c5f6e59dfcf26d802f0dbbd1e00715` |

* **Sizes and server metadata.** The compressed sizes equal the contract's pinned HTTP Content-Length. A HEAD request during this audit returned the same Content-Length and the same Last-Modified dates as recorded in contract §1.1.
* **Hashes.** They were recomputed during this audit and equal the build manifest.
* **Gzip integrity.** Both `.gz` streams decompress with a valid CRC to bytes whose SHA-256 equals the on-disk `.nc` files.
* **Release notes.** `Release_Notes_CRU_TS_4.10.txt` sha256 `fb3c8cb5…` matches the contract.
* **Inside-file checks** (all as contracted):
  * both files are NETCDF3_CLASSIC;
  * variables `pre` (with `stn`, `mae`, `maea`) and `pet`;
  * units `mm/month` and `mm/day`;
  * `time` in days since 1900-1-1 on the Gregorian calendar, 1,500 contiguous unique months 1901-01…2025-12;
  * latitude −89.75…89.75 and longitude −179.75…179.75, ascending at 0.5°;
  * `_FillValue` = `missing_value` = 9.96921e36 (`stn`: −999);
  * titles "CRU TS4.10 Precipitation" and "CRU TS4.10 Potential Evapotranspiration", run ID 2604091129.
* **Provenance disclosure.** Contract §1.2 makes the SHA-256 values the pin and says they are committed before any C2 value is examined.
  * The full hashes were first written by the build manifest, in the same run that constructed C2.
  * They are committed for the first time with this audit.
  * The only artifacts produced between acquisition and this commit are the four build outputs preserved here: features, coverage/support QA, manifest and support checkpoint. No artifact relating C2 to any other quantity exists.

## 3. Determinism

Three builds with the unmodified builder produced byte-identical files:
* the preserved original build;
* rebuild A;
* rebuild B.

The SHA-256 values are in `outputs/m1b_feasibility_original_coverage/m1b_original_coverage_determinism.json`.

| File | SHA-256 |
|---|---|
| `m1b_hydroclimate_features.csv` | `a90bbcdb81fbc951ce875555aa599626bef155506db9aa2b81e5e174f5b2584f` |
| `m1b_hydroclimate_qa.csv` | `747da75cef0667cafbcbf613b561640238eecdf8476186cb5cfb214efdbd7b91` |
| `m1b_measurement_manifest.json` | `0714524f8fae3526cf09be491b61ddaa4b10aac7a1d9280aca9a19bb3a329063` |
| `m1b_support_checkpoint.json` | `405e5bdd63e84d1e74f18401f15f898dec64079aa79096e5cc25a585d792307e` |

Each rebuild took about 11 s wall time with warm file caches, with a peak resident memory of about 4.7 GB. Environment: Python 3.11.12, NumPy 2.4.6, pandas 3.0.3, xarray 2026.4.0, macOS 26.6.2 arm64.

**Country assignment and area.** Per-country terrestrial area summed over CRU cells reproduces the M1a `terrestrial_area_km2` to a maximum relative error of 1.1e−12. That sum does not depend on which CRU cell a GPW cell is mapped to, so the audit also checks the grid geometry explicitly:
* GPW is stored north-to-south, and the builder flips it to ascending latitude;
* the flipped 0.25° centres are −89.875 + 0.25·r and −179.875 + 0.25·c;
* every GPW cell's edges lie inside CRU cell (r // 2, c // 2).

## 4. The coverage failures

Coverage is the M1a terrestrial area in valid CRU cells divided by total M1a terrestrial area. The rule requires ≥ 0.98. ✗ marks a failure. Countries at or above 0.99 are not listed; the full 151-row table is `m1b_original_coverage_by_country.csv`.

| ISO3 | Country | Terrestrial km² | Valid km² | Missing km² | Coverage | Shortfall | km² short of 0.98 | Support cells | Invalid cells | Both masked km² | PET-only masked km² | Temporal/non-positive km² | Mean land fraction of invalid cells (max) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BHS | Bahamas | 13,426 | 11,902 | 1,524 | 0.8865 ✗ | 0.0935 | 1,256 | 79 | 50 | 993 | 532 | 0 | 0.030 (0.067) |
| PAN | Panama | 75,970 | 73,905 | 2,065 | 0.9728 ✗ | 0.0072 | 546 | 54 | 17 | 1,901 | 164 | 0 | 0.086 (0.123) |
| PHL | Philippines | 296,328 | 289,237 | 7,091 | 0.9761 ✗ | 0.0039 | 1,164 | 269 | 80 | 6,666 | 425 | 0 | 0.068 (0.169) |
| NLD | Netherlands | 34,324 | 33,724 | 600 | 0.9825 | — | — | 33 | 3 | 600 | 0 | 0 | 0.283 (0.301) |
| GRC | Greece | 132,580 | 130,774 | 1,805 | 0.9864 | — | — | 155 | 37 | 412 | 1,394 | 0 | 0.057 (0.116) |
| CUB | Cuba | 111,481 | 110,029 | 1,452 | 0.9870 | — | — | 89 | 22 | 1,163 | 290 | 0 | 0.042 (0.063) |
| NZL | New Zealand | 265,218 | 262,552 | 2,666 | 0.9899 | — | — | 214 | 49 | 2,422 | 244 | 0 | 0.068 (0.140) |

Column definitions:
* **Both masked:** PRE and PET are fill in all 360 window months.
* **PET-only masked:** PRE is valid in all 360 months, PET is fill in all 360.
* **Land fraction:** the M1a terrestrial area of the 151 countries in the CRU cell divided by the cell's spherical area.

**Overall:** 148 of 151 countries pass; 144 have coverage ≥ 0.99, 52 have exactly 1, and the median is 0.99973.

## 5. Cause: structural CRU land-mask mismatch

From `m1b_original_coverage_structural_mask.json`, read from the raw files:

1. **The CRU masks are fixed in time.**
   * PRE: 67,420 cells are valid in every one of the 1,500 months and 191,780 are fill in every month; no cell is partially missing, either in the window or over the full record.
   * PET: 66,501 cells are valid throughout, 192,699 are fill throughout, and none is partially missing.
2. **PET's mask is a strict subset of PRE's.** 919 cells are PRE land but PET fill; no cell is PET land but PRE fill. All 66,501 PET land cells are valid C2 cells.
3. **Every invalid support cell is a masked cell.**
   * The 151 countries' M1a support touches 65,335 CRU cells; 2,753 of them are invalid:
     * 2,182 are masked in both variables;
     * 571 are masked in PET only.
   * None has partial temporal missingness, and none has P̄ ≤ 0 or Ē ≤ 0.
   * All 129,214 km² of invalid support is fill in all 1,500 months. That is 0.100% of the 151-country support.
   * The builder's reason label "non-finite month" therefore always means "fill in every month" here.
4. **The missing land lies in cells that CRU treats as sea.** Their M1a terrestrial fractions are small:
   * area-weighted mean 0.03 (Bahamas), 0.09 (Panama) and 0.07 (Philippines);
   * no failing-country invalid cell exceeds 0.17.

   These are 0.5° cells holding small fragments of land that the GSHHG-based M1a support resolves but CRU's fixed 0.5° land mask classifies as non-land.

**What the cause is not:**
* **not missing monthly observations** (no partial cells exist);
* **not invalid climatological means** (no non-positive P̄ or Ē);
* **not a code defect** (three identical builds; the valid-cell rule reproduces the contract);
* **not a country-assignment defect** (area reproduces M1a to 1.1e−12, and the grid nesting is verified).

## 6. Pre-outcome station-support checkpoint (unchanged, no threshold)

Computed over valid cells, weighted by terrestrial area, from `m1b_support_checkpoint.json`:

* **Median country:** 99.8% of land-months have ≥ 1 PRE station; mean PRE station count is 7.31.
* **Lowest station-supported share:**
  * Saudi Arabia 0.276;
  * Yemen 0.288;
  * Haiti 0.453;
  * Oman 0.510;
  * Chad 0.547;
  * then United Arab Emirates 0.557.
* **Lowest mean station count:** Yemen 0.44, United Arab Emirates 0.57, Oman 0.59, Saudi Arabia 0.72.
* **Highest pure-climatology share** (terrestrial area with `stn` = 0 in all 360 months): Yemen 0.446, Saudi Arabia 0.399, Mali 0.116.

No country is pathological under the contract's definitional stop: no country has a pure-climatology share of 1, and no `stn` value is invalid in a valid cell. PET has no station-count field, and its support stays an unquantified limitation. C2 is a **CRU-reconstructed** 1920–1949 baseline, not an independently observed pre-1950 state.

## 7. What was not done

* The coverage threshold was not relaxed.
* No cell was filled from a neighbour or nearest cell, and no source was changed.
* No country was removed.
* The redundancy diagnostic was not run.
* The warming outcome (Berkeley or ERA5), M0 or M1a residuals, and any C2–outcome relationship were not loaded or examined.
* No model was fitted or scored.

## 8. Artifacts and reproduction

`outputs/m1b_feasibility_original_coverage/`:
* **Preserved original build outputs:** `m1b_hydroclimate_features.csv`, `m1b_hydroclimate_qa.csv`, `m1b_measurement_manifest.json` (source hashes, code hash, hard stops) and `m1b_support_checkpoint.json`.
* **`m1b_original_coverage_by_country.csv`:** 151 rows. Coverage, missing area by mask class, invalid-cell counts and land fractions.
* **`m1b_original_coverage_structural_mask.json`:** global and support-level mask statistics, and the grid-alignment checks.
* **`m1b_original_coverage_determinism.json`:** SHA-256 of the preserved outputs and two rebuilds.

Reproduce from a checkout of `c5ff9b5`, where the builder is byte-identical to `c6ef919`. Later builder commits change the manifest's code hash and remove the constant the audit script reads.

```
python -m research.model_v2.m1b_hydroclimate BUILD_A
python -m research.model_v2.m1b_hydroclimate BUILD_B
python -m research.model_v2.feasibility.m1b_original_coverage BUILD_A BUILD_B
```

The audit script (`feasibility/m1b_original_coverage.py`):
* asserts that both rebuilds are byte-identical to the preserved outputs;
* recomputes coverage and asserts that it matches the preserved QA;
* writes the three audit files.

A second run of the script reproduced its own three outputs byte-for-byte.

## Addendum (2026-09-15, recorded with contract Amendment 1): the scope of the outcome statements

The statements above that no warming outcome was accessed mean the following precisely.

* **The original-rule build.** Builder `c6ef919` read `outputs/m0_countries.csv` with `usecols=['Country', 'iso3']` and hashed the whole file for its manifest.
  * That file also carries warming, fitted, residual and ERA5 trend columns. They were never parsed into a data frame or used.
  * The audit script at `c5ff9b5` reads the file the same way, so reproducing from that checkout opens it, with identifier columns only.
  * From `81c5dca` onwards, the builder takes identifiers from the outcome-free M1a support record instead.
* **The test suite.** This protocol runs the repository's unit-test suite. Some of its existing tests load the frozen V1 design, which contains the warming outcome, to verify M0/M0\* reproduction and the M1b evaluator's schema. None of them involves a C2 value.
* **C2 itself.** No C2 value has been joined with, compared with or correlated with any warming outcome, fitted value, residual or score.
* **Owner approval.** The status section's statement that the owner approved an amendment "which is recorded and committed separately" referred to approval in principle. The amendment text was committed afterwards, as `M1B_EVALUATION_CONTRACT.md` Amendment 1.
