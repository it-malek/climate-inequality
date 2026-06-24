# ND-GAIN Country Index — vendored data NOTICE

`ndgain_latest.csv` is a slim, derived extract (one row per country: ISO3 +
latest-available-year vulnerability / readiness / gain scores) of the **ND-GAIN
Country Index**.

- **Source:** Notre Dame Global Adaptation Initiative (ND-GAIN),
  University of Notre Dame — <https://gain.nd.edu/our-work/country-index/download-data/>
- **Original file:** `ndgain_countryindex_2026.zip` (members
  `resources/{vulnerability,readiness,gain}/*.csv`, wide ISO3 × year).
- **License:** distributed by ND-GAIN under an open Creative Commons license.
- **Suggested citation:** Notre Dame Global Adaptation Initiative Country Index
  (ND-GAIN), University of Notre Dame.
- **Derivation:** `scripts/fetch_ndgain.py` →
  `src.vulnerability.derive_ndgain_latest` (latest non-null year per country).
- **Retrieved:** 2026-06 (latest data year in this extract: 2023).

This slim CSV is force-added (the `data/raw/` tree is otherwise gitignored),
mirroring the vendored World Bank income-groups CSV.
