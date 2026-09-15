# M1b C2 predictor package — frozen before any outcome relationship

**Recorded 2026-09-15.** C2 is frozen under `M1B_EVALUATION_CONTRACT.md` Amendments 1–3.
* No redundancy diagnostic had been run when this package was frozen.
* No M1b model was fitted or scored, and C2 was not joined with any warming outcome.
* Nothing may change in this package because of a later redundancy or score result.

C2 is **CRU-reconstructed 1920–1949 baseline hydroclimatic dryness**: the terrestrial-area-weighted mean of cell log10(P̄/Ē) over *resolved* support. It is not a predictor built only from information available before 1950. Passing coverage establishes neither predictive value nor a causal mechanism, and the 98% gate does not prove the absence of bias from missing support.

## Provenance

| Item | Value |
|---|---|
| Build commit (Amendment 3 frozen and pushed first) | `58e64bfbe6a6b12503a8b08cea51fecd5415e1a5` |
| Gate version | `M1B_EVALUATION_CONTRACT.md Amendment 3` |
| Measurement manifest SHA-256 | `1974686199b9c7960ead2b105e6225f34b9b6fb33627858d2bc15239d7de4c9c` |
| PRE `.gz` / `.nc` SHA-256 | `b7e7a3b7…551edaa` / `efe27c45…b3eae` |
| PET `.gz` / `.nc` SHA-256 | `fcc792b0…816f013` / `0bc9a131…0715` |
| GPW grid / national-ID lookup | `e15f6228…dab85d` / `bbe7b135…62e089`, asserted against records at `4263429`, `ff67545`, `a7dea34`, `9fb56e1`, `bd518a0` |
| M1a support | land-area npz `642cbe5a…66cf` and QA `bdc4f1ca…25c8` at `a7dea34` |
| Software | Python 3.11.12, NumPy 2.4.6, pandas 3.0.3, xarray 2026.4.0, netCDF4 1.7.4, pyproj 3.7.2 / PROJ 9.5.1, macOS 26.6.2 arm64 |

**Package artifacts** (`outputs/`; each digest is recorded in the manifest and checked by `verify_package`):

| File | SHA-256 |
|---|---|
| `m1b_hydroclimate_features.csv` | `cf33ba7cfb87c2184a94df5e5686f0951adf692819af8de27329ea698d72e1f4` |
| `m1b_hydroclimate_qa.csv` | `e00cbb619d7ad1e1bac0f732c8d030bc49dd0329c2820c38e83e226c65ff54a7` |
| `m1b_harmonization_cells.csv` | `bd40fa76a48842495ab76f8d3d2d67138e16426098536d4031e6d4d64a23ebfe` |
| `m1b_unresolved_support.csv` | `5f8e202f4dc99deeb42e22ff1dd98b0d0d4c02a4bd5ecae317d30512a256f317` |
| `m1b_support_checkpoint.json` | `405e5bdd63e84d1e74f18401f15f898dec64079aa79096e5cc25a585d792307e` |
| `m1b_measurement_manifest.json` | `1974686199b9c7960ead2b105e6225f34b9b6fb33627858d2bc15239d7de4c9c` |

Evidence: `m1b_package_record.json` (comparison evidence), `m1b_build_run.json` and `m1b_build_run_b.json` (wall-clock run metadata; not byte-compared).

## Determinism

Two builds ran at the build commit from a clean tree:
* independent processes (`PYTHONHASHSEED` 0 and 73);
* separate fresh roots;
* 15.2 s and 14.3 s internal wall time; about 17.5 s and 4.9–5.1 GB peak memory including interpreter start.

All six deterministic outputs were byte-identical. Only run metadata differs. The installed files equal build A's bytes, and `verify_package(outputs/)` passes.

## Gates

* **All hard stops pass,** with an empty `hard_stops` list, for the ordered 151-country frozen population.
* **Coverage:** every country has coverage ≥ 0.98. The minimum is Bahamas at 0.9887528054237732, the only country below 0.99. Panama is 1.0, the Philippines 0.9985162289023632 and Peru 0.999982140608819.
* **Peru (Amendment 3):** CRU cell (164, 200) holds 23.01253170656061 km² of Peru.
  * PRE state `nonpositive_mean` (zero in every month); PET state `masked`.
  * Status unresolved, reason `non_structural_invalid`; all donor fields and the value are NA.
  * It appears in `m1b_unresolved_support.csv` and counts against Peru's coverage. It is the only non-structural cell.
* **Unresolved support:** 346 (country, cell) rows covering 6,598.686 km²:
  * structural cells with no native one-ring donor: 6,575.673 km², in 344 distinct cells;
  * non-structural: 23.013 km².
* **Harmonization, unchanged from the stopped build:**
  * 2,752 structural target cells, of which 2,408 are harmonized (122,615.27 km²);
  * donor distance (WGS84): median 52.24 km, 95th percentile 72.46 km, maximum 78.45 km;
  * 56 cross-border (country, cell) rows.

## Invariance

**Against the stopped Amendment 1 build (`bd518a0`):**
* byte-identical: features, including the reference `cf33ba7c…`; QA; harmonization cells, so every donor and transferred value; and the station-support checkpoint;
* hence every country C2 value, area and coverage is unchanged.

The manifest differs only in the declared keys:
* `amendment` → `amendments`, `gate_version`, `hard_stop_rules`, `unresolved_support` and `artifact_sha256`;
* the pass flag and stop list;
* `code_sha256` and `contract_sha256`;
* `git` (commit and amendment commit);
* `support_inputs`, which gains only `gpw_and_lookup_pins_asserted`.

**Against the original-rule build (`c5ff9b5`):** exact equality of:
* native-only C2 and native fraction (the original C2 and coverage);
* native and total areas;
* C2 wherever nothing was harmonized;
* all station-support columns;
* the checkpoint bytes.

## Station support (native cells; unchanged)

* **Lowest PRE station-supported share:** Saudi Arabia 0.276, Yemen 0.288, Haiti 0.453, Oman 0.510, Chad 0.547. These form the Amendment 2 conditional five-country replication set.
* **Highest pure-climatology share:** Yemen 0.446, Saudi Arabia 0.399, Mali 0.116.
* **No threshold** was defined.

## Reproduce

At `58e64bf`:

```
python -m research.model_v2.m1b_hydroclimate BUILD_A      # PYTHONHASHSEED=0
python -m research.model_v2.m1b_hydroclimate BUILD_B      # PYTHONHASHSEED=73
python -m research.model_v2.feasibility.m1b_amendment3_package 58e64bf… BUILD_A BUILD_B --verify-only
```

Without `--verify-only`, the script installs build A into `outputs/` (refusing to overwrite) and writes `m1b_package_record.json`. Historical scripts reproduce only at their own commits.
