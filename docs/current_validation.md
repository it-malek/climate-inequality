# Released 2.1.0 analysis: reconstruction and validation

Version 2.1.0 reconstructs the analysis in a separate, versioned run. The
commands below document that reconstruction. The historical commands in
`reproducibility.md` describe 2.0.0; separate output directories preserve both
generations and distinguish fresh fits from documented reuse.
Run modules from the repository root with Python 3.11.

## Inputs and fixed settings

Set `TASK_RAW_ROOT` to the directory of verified source inputs and
`TASK_NDGAIN_ARCHIVE` to the official 2026 release_data_v2 ZIP with SHA-256
`e854c7fe6ccef8c5d24a5ef7f370b7f61d1b8c6816da19fda4cd892ca6cdf0de`.
Set `TASK_RUN_ROOT` to a new absolute direct-child directory under
`outputs/prospective/prospective-corrections-2026-10-22/`. Existing runs are
immutable; use a new name or numbered retry. The historical prepared ND-GAIN
extract remains required for vintage comparison and is preserved separately.

The method templates are `current_validation_spec.json`,
`bootstrap_population_spec.json` and `ndgain_updated_source_spec.json`.
These pre-execution specifications retain their original status and scope
wording as historical evidence; that wording does not describe the release
status of 2.1.0.
The initializer freezes actual local paths before execution; stages record
source, inputs, package versions and hashes before running and check for drift
on completion. Failed attempts remain separate. These commands build local
outputs and leave retained historical outputs intact.

## Dependency order

```bash
uv sync --locked --extra dev
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --locked python -m scripts.init_current_validation --run-root "$TASK_RUN_ROOT" --raw-root "$TASK_RAW_ROOT"
for stage in inputs-koppen preprocessing forcings physical explain comparisons core city-validation era5 vulnerability stability rolling uncertainty; do
  uv run --locked python -m scripts.run_current_validation "$stage" --run-root "$TASK_RUN_ROOT" --raw-root "$TASK_RAW_ROOT" || break
done
uv run --locked python -m scripts.run_spatial_validation pilot --run-root "$TASK_RUN_ROOT" --raw-root "$TASK_RAW_ROOT"
uv run --locked python -m scripts.run_spatial_validation full --run-root "$TASK_RUN_ROOT" --raw-root "$TASK_RAW_ROOT"
uv run --locked python -m scripts.run_spatial_validation neighbor --run-root "$TASK_RUN_ROOT" --raw-root "$TASK_RAW_ROOT"
uv run --locked python -m scripts.run_bootstrap_population_audit freeze --run-root "$TASK_RUN_ROOT" --raw-root "$TASK_RAW_ROOT"
uv run --locked python -m scripts.run_bootstrap_population_audit run --run-root "$TASK_RUN_ROOT" --raw-root "$TASK_RAW_ROOT"
for stage in freeze extract calculate; do
  uv run --locked python -m scripts.run_ndgain_update "$stage" --run-root "$TASK_RUN_ROOT" --raw-root "$TASK_RAW_ROOT" --archive "$TASK_NDGAIN_ARCHIVE" || break
done
```

Inspect completion receipts after each stage. A failed command stops its loop;
resolve it before continuing. The bootstrap comparison requires the legacy
157/154-row pre-filter stability checkpoint, then adopts the fixed complete-case
151-country policy. Optional previous-run reuse checks exact ordered inputs,
missingness, estimator and runtime; tiny floating-point differences trigger refits.
The updated ND-GAIN analysis fixes all components to 2023, without older-year fallback.

For V2, first run and inspect each representative pilot, then its full stage:

```bash
uv run --locked python -m research.model_v2.prospective_refit --run "$TASK_RUN_ROOT" --raw "$TASK_RAW_ROOT" --stage measurement-pilot
uv run --locked python -m research.model_v2.prospective_refit --run "$TASK_RUN_ROOT" --raw "$TASK_RAW_ROOT" --stage measurement
uv run --locked python -m research.model_v2.prospective_refit --run "$TASK_RUN_ROOT" --raw "$TASK_RAW_ROOT" --stage static
uv run --locked python -m research.model_v2.prospective_refit --run "$TASK_RUN_ROOT" --raw "$TASK_RAW_ROOT" --stage spatial-pilot
uv run --locked python -m research.model_v2.prospective_refit --run "$TASK_RUN_ROOT" --raw "$TASK_RAW_ROOT" --stage spatial
uv run --locked python -m research.model_v2.prospective_refit --run "$TASK_RUN_ROOT" --raw "$TASK_RAW_ROOT" --stage final-checks
uv run --locked python -m research.model_v2.prospective_refit --run "$TASK_RUN_ROOT" --raw "$TASK_RAW_ROOT" --stage product-lisa
uv run --locked python -m research.model_v2.prospective_refit --run "$TASK_RUN_ROOT" --raw "$TASK_RAW_ROOT" --stage buffer-predictions
uv run --locked python -m research.model_v2.prospective_assets --run "$TASK_RUN_ROOT"
uv run --locked python -m scripts.build_validation_candidate --run-root "$TASK_RUN_ROOT" --physical-stage physical
```

The recorded run selected successful `physical_attempt2` and `v2_static_attempt2`
receipts; a new clean run uses canonical names above. The public reproduction
scripts were checked against the execution drivers used for the recorded run. They have not been independently executed as a
second complete scientific reconstruction.

## Accepted interpretation

Training-only kriging has lower descriptive RMSE than IDW under both fixed
protocols; the fixed IDW map remains a visual aid. Identity-based tie handling
stabilizes residual diagnostics while preserving coincident city records and
the separate fixed V2 graphs. Rolling-origin intervals are conditional on
observed future drivers and fitted parameters, with 24 dependent observations
per horizon. Neither nonsignificant rank tests nor coverage counts establish
equivalence or calibration. The static geography result, negative static
extensions and failed ERA5 spatial replication remain findings.

The original ND-GAIN archive and original M1a measurement-specification text
were not recovered. Updated ND-GAIN is explicitly labeled and pinned. M1a
uses the retained implementation, manifests, raw pins and restored available
gates; the unavailable document text is not claimed to have been reconstructed.

## Released bundle and verification

The bundle has 27 scientific assets. Compact source records, a separate current
validation index and source hashes accompany it. Raw archives, bulk draws,
bulk execution receipts stay outside the public tree. Compact historical
records remain available under `research/model_v2/`.
The historical research checker and tests retain their existing semantics.
Several released JSON assets retain `computation_status: prospective-development`.
That field records the isolated execution stage under which they were built;
it is not the publication status. Their bytes and the scientific bundle hash
remain preserved. `app/data/release.json` identifies the released version, and
`reports/publication_manifest.json` identifies the separate editorial metadata.
Run `python -m src.research_audit` to verify retained evidence and the
canonical index. The optional `--write` mode regenerates only that index after
a documented source change. Verification establishes evidence bytes, not
scientific truth.

```bash
uv run --locked ruff check src tests app scripts research
uv run --locked pytest -q -rs
uv run --locked python -m src.research_audit
uv run --locked python -m unittest discover -s app/tests -v
```

Saved-output verification is distinct from independent numerical
recomputation. The released bundle contains the corrected estimates; the
frozen historical records retain the earlier results and execution constraints.
