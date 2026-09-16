# Model V2 completion state

Compact, durable pointer record for the owner-authorized completion (2026-09-16). **Git and the committed
artifacts are the source of truth**: a checkpoint's own commit hash is resolved through Git
(`git log --oneline -- research/model_v2/COMPLETION_STATE.md`), never claimed here before it exists.

## Checkpoints

| # | Checkpoint | State | Identity (resolve through Git) |
|---|---|---|---|
| 1 | Downstream M3/M4 specification and pre-score resolutions | committed with this file's first version | spec `DOWNSTREAM_COMPLETION_SPEC.md`; M3 tooling `c27c7fd`, boundary disposition `99f2c3d`; feasibility record from `99f2c3d` |
| 2 | M2 evaluator freeze | committed with `M2_EVALUATOR_SPEC.md`, `m2_evaluate.py`, `m2_conditional.py`, `m2_independent_check.py` | downstream Amendment A1 `ef04958` precedes it |
| 3 | Primary M2 result, report, verification | committed with `M2_REPORT.md` and `outputs/m2_primary/` | evaluator `b13bb04`; label `not supported`; retained static M0\*; stopping indicator does not fire |
| 4 | M2 conditional robustness or non-execution | committed with `outputs/m2_conditional/m2_conditional_not_run.json` | refusal returned before any arm frame was constructed; no arm fitted |
| 5 | M3 evaluator freeze | committed with `M3_EVALUATOR_SPEC.md`, `m3_evaluate.py`, `m3_independent_check.py` | method frozen at `4dc2fad` + `ef04958`; full suite 1200 passed, 6 skipped before this freeze |
| 6 | Primary M3 result and verification | committed with `M3_REPORT.md` and `outputs/m3_station/` | evaluator `a6df380`; SEM and SAR both qualify; final primary predictive model null (non-unique) |
| 7 | M3 land-centroid weight sensitivity | committed with `outputs/m3_land_centroid/` and the dated §8 addition to `M3_REPORT.md` | same frozen code `a6df380`; descriptive: SAR conditions hold, SEM fails the M49 veto |
| 8 | M4 final evidence | committed with `M4_REPORT.md` and `outputs/m4_final/` | evaluator `0366281`; inventory complete; no material change to the V1 conclusion; both spatial extensions product-sensitive |
| 9 | Scientific closure | committed with `V2_FINAL_REPORT.md` and `outputs/v2_final/v2_final_record.json` | dated status updates to plan, decisions, README, register, scorecard |
| 10 | Verified integration into `main` | fast-forward of `main` to the commit adding this line; integrated tree re-verified before push | resolve through `git log origin/main` |

## First real scores

| Stage | First outcome-facing fit has occurred? |
|---|---|
| M2 candidate | **yes**: once, from `b13bb04` (2026-09-16), plus one unchanged-code rerun |
| M3 spatial families | **yes**: station arm once from `a6df380`, plus one unchanged-code rerun |
| M4 new computations | **yes**: once from `0366281`, plus one unchanged-code rerun |

Disclosed baseline-only calculations: the committed M1b test and the (uncommitted at this checkpoint) M2
evaluator anchor test score M0\* on real inputs against its frozen rank-audit cards; no M2 candidate is fitted
in either (M2 fits raise in the anchor test).

## Outcome-activated branches

* Branch: **A** (retained static specification M0\*), mechanically from the M2 label `not supported`.
* M2 conditional arms: **not executed** (no predictive support); verified non-execution record committed.
* M3 (station arm): qualifying set **[M3-SEM(M0\*), M3-SAR(M0\*)]**; `final_primary_predictive_model` **null**.

## Evidence paths (this checkpoint)

* `research/model_v2/DOWNSTREAM_COMPLETION_SPEC.md`, `M2_PRE_SCORE_RESOLUTIONS.md` (M2 contract Amendment 1),
  `M4_EVIDENCE_INVENTORY.md`, `M3_FEASIBILITY_AUDIT.md`, `outputs/m3_feasibility/`.
* Dated correction note in `M2_DESIGN.md` §6; Amendment 1 pointer in `M2_EVALUATION_CONTRACT.md`.

## Verification at this checkpoint (focused)

* `uv run python -m pytest -q research/model_v2/tests/test_m3_spatial.py research/model_v2/tests/test_m3_independent_math.py research/model_v2/tests/test_m3_feasibility.py research/model_v2/tests/test_m2_feasibility.py research/model_v2/tests/test_m2_latitude_basis.py`
* `uv run ruff check research/model_v2`
* Full suite and isolation checks run at the M2 evaluator freeze (checkpoint 2).

## Branch verification before integration (at `8802a4b`)

* `uv run python -m pytest -q tests research/model_v2/tests`: 1258 passed, 6 skipped.
* `uv run ruff check src tests app scripts research/model_v2`: clean. `uv run python -m unittest discover -s app/tests`: OK.
* `git diff --check a6733cb HEAD`: clean. `git diff v1.3.0 HEAD -- . ':!research'`: empty (production, bundle and app unchanged).
* No historical output modified or deleted since `23d31f4`; historical documents received dated additions only.
* GitHub CI on `8802a4b`: success. `main` is unprotected and `origin/main` (`a6733cb`) is an ancestor.

## Next executable action

None within this research line after integration. `main` is fast-forwarded to the commit that adds this
section, re-verified on the integrated tree, and pushed; confirm with `git ls-remote origin refs/heads/main`.

## Blockers

None.
