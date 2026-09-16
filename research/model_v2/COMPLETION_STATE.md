# Model V2 completion state

Compact, durable pointer record for the owner-authorized completion (2026-09-16). **Git and the committed
artifacts are the source of truth**: a checkpoint's own commit hash is resolved through Git
(`git log --oneline -- research/model_v2/COMPLETION_STATE.md`), never claimed here before it exists.

## Checkpoints

| # | Checkpoint | State | Identity (resolve through Git) |
|---|---|---|---|
| 1 | Downstream M3/M4 specification and pre-score resolutions | committed with this file's first version | spec `DOWNSTREAM_COMPLETION_SPEC.md`; M3 tooling `c27c7fd`, boundary disposition `99f2c3d`; feasibility record from `99f2c3d` |
| 2 | M2 evaluator freeze | committed with `M2_EVALUATOR_SPEC.md`, `m2_evaluate.py`, `m2_conditional.py`, `m2_independent_check.py` | downstream Amendment A1 `ef04958` precedes it |
| 3 | Primary M2 result, report, verification | not started | — |
| 4 | M2 conditional robustness or non-execution | not started | — |
| 5 | M3 evaluator freeze | not started | — |
| 6 | Primary M3 result and verification | not started | — |
| 7 | M3 land-centroid weight sensitivity | not started | — |
| 8 | M4 final evidence | not started | — |
| 9 | Scientific closure | not started | — |
| 10 | Verified integration into `main` | not started | — |

## First real scores

| Stage | First outcome-facing fit has occurred? |
|---|---|
| M2 candidate | **no** |
| M3 spatial families | **no** |
| M4 new computations | **no** |

Disclosed baseline-only calculations: the committed M1b test and the (uncommitted at this checkpoint) M2
evaluator anchor test score M0\* on real inputs against its frozen rank-audit cards; no M2 candidate is fitted
in either (M2 fits raise in the anchor test).

## Outcome-activated branches

* Branch (A: retained static M0\*; B: retained static M2): **not yet selected**.
* M2 conditional arms: **not yet authorized or refused**.
* M3 qualifying set / final naming: **not yet determined**.

## Evidence paths (this checkpoint)

* `research/model_v2/DOWNSTREAM_COMPLETION_SPEC.md`, `M2_PRE_SCORE_RESOLUTIONS.md` (M2 contract Amendment 1),
  `M4_EVIDENCE_INVENTORY.md`, `M3_FEASIBILITY_AUDIT.md`, `outputs/m3_feasibility/`.
* Dated correction note in `M2_DESIGN.md` §6; Amendment 1 pointer in `M2_EVALUATION_CONTRACT.md`.

## Verification at this checkpoint (focused)

* `uv run python -m pytest -q research/model_v2/tests/test_m3_spatial.py research/model_v2/tests/test_m3_independent_math.py research/model_v2/tests/test_m3_feasibility.py research/model_v2/tests/test_m2_feasibility.py research/model_v2/tests/test_m2_latitude_basis.py`
* `uv run ruff check research/model_v2`
* Full suite and isolation checks run at the M2 evaluator freeze (checkpoint 2).

## Next executable action

Score M2 once from the exact pushed evaluator-freeze commit:
`PYTHONHASHSEED=0 uv run python -m research.model_v2.m2_evaluate` (output `research/model_v2/outputs/m2_primary`),
then the independent check and one unchanged-code rerun into a fresh root.

Full suite before this freeze: 1137 passed, 6 skipped (`uv run python -m pytest -q tests research/model_v2/tests`);
`ruff check src tests app scripts research/model_v2` clean; app unittest OK; production outside `research/`
identical to `v1.3.0`.

## Blockers

None.
