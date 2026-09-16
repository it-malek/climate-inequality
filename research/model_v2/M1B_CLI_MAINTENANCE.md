# Post-M1b auxiliary CLI maintenance: `--compare` exit status

**Recorded 2026-09-16, after the M1b result and its closure. Maintenance only: no scoring rule,
threshold, algorithm, input pin, contract or saved artifact changes, and M1b was not re-scored.**

## Defect

Disclosed in [`M1B_REPORT.md`](M1B_REPORT.md) §8. `m1b_evaluate.main()` returns a scoring result
(carrying `integrity_passed`) or, under `--compare RUN_A RUN_B`, a reproducibility record (carrying
`byte_identical`). The frozen `__main__` line read `outcome['integrity_passed']` in both modes, so a
completed comparison printed its record and then raised `KeyError: 'integrity_passed'` (exit 1).

Reproduced before the fix through the real entry point, over two synthetic directories of placeholder
bytes (no scoring input is read on this path):

```
uv run python -m research.model_v2.m1b_evaluate --compare <synthetic_a> <synthetic_b>
{"byte_identical": true, "differing": []}   ->   KeyError: 'integrity_passed', exit 1
```

## Change

Only the exit mapping. A new `cli_exit_code(outcome)` is called from `__main__`:

| Outcome shape | Exit |
|---|---|
| scoring, `integrity_passed is True` | 0 (unchanged) |
| scoring, `integrity_passed is False` | 3 (unchanged) |
| `--compare`, `byte_identical is True` | 0 |
| `--compare`, `byte_identical is False` | 4 |
| neither key, both keys, a non-boolean flag, or not a mapping | raises `RuntimeError` (nonzero) |

## Evidence

* **Tests first** (`tests/test_m1b_cli_exit.py`). Written before the change, 31 of the 32 initial tests
  failed on the frozen code. The one that passed only characterizes the comparison record's shape.
  All pass after the change. They cover direct mapping, including every malformed shape; a subprocess run of
  `main()` + `cli_exit_code()` over synthetic identical, differing and metadata-only-differing runs;
  and an AST check that `__main__` is exactly `sys.exit(cli_exit_code(main()))`.
* **The comparison path cannot score.** The subprocess driver replaces `require_hash_seed`,
  `verify_scoring_inputs`, `build_frozen`, `evaluate`, `build_arms`, `score_model`, `support_context`
  and `redundancy_context_from` with stubs that raise. A negative control confirms that a
  non-comparison invocation hits a stub and exits nonzero without creating its output directory.
* **Scoring definitions are unchanged** (AST comparison against `618c8b8`):
  * 119 of 120 original top-level nodes are identical, including 45 functions and classes and
    42 module-level assignments (every constant, threshold, tolerance and input pin);
  * the only changed node is the `__main__` guard, and the only added node is `cli_exit_code`,
    which no pre-existing definition calls;
  * the other 17 `CODE_PATH` files are byte-identical to `618c8b8`.

  This check is kept as a regression test. Negative controls confirm that it fires on a changed
  constant, an edited function body or an extra helper.
* **No artifact changed.** Digests of all 133 files under `research/model_v2/outputs`, `outputs` and
  `app/data` were identical before and after the full suite ran.
* **Full suite.** 886 passed, 6 skipped with the patch applied, before the M2 design files existed.
  `ruff check` is clean.

## Provenance consequence (deliberately not hidden)

`m1b_evaluate.py` is on the M1b `CODE_PATH`, so its digest changes:

* frozen: `233a1a896a7d6e880e33bff9a433b4eac3e20489a786accec0810dd9ff43b027`;
* after this maintenance commit: `ba23865eef702f8e8fc9349696e788958f56781705dbef036895c152fe4bcae4`.

**The M1b result remains the product of `618c8b8`, and historical reproduction remains pinned there**
(check out `618c8b8`, then run the command in `M1B_REPORT.md` §7). A run from any later commit would
record a different code-path digest, so its `m1b_provenance.json` would not be byte-identical to the
saved one. That is the intended behaviour of the provenance guard. The guard was not weakened, and the
evaluator commit recorded in the saved result is not relabelled.
