# Notebooks (exploratory / archival)

These notebooks are **exploratory and archival**, not part of the reproducible
pipeline. Every published number comes from the tested modules in `src/`; the
notebooks are kept for provenance and scratch analysis only and may contain
stale outputs.

| Notebook | Role |
|----------|------|
| `00_original_2022_cleaning_archive.ipynb` | Archive of the original 2022 undergraduate-fellowship cleaning code, kept for provenance. |
| `01_data_quality.ipynb` | Exploratory data-quality checks on the Berkeley Earth city data. |
| `02_explanatory.ipynb` | Exploratory explanatory-variable analysis (a precursor to `src/explain.py`). |

**Source of truth:** `src/` (modules) + `tests/` (synthetic-fixture tests) +
`docs/reproducibility.md`. If a notebook and the pipeline disagree, the pipeline
is correct.
