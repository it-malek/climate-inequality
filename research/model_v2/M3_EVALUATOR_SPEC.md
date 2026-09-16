# M3 evaluator specification — frozen before the first outcome-facing M3 fit

**Recorded 2026-09-16. Committed and pushed before any spatial model is fitted to the warming outcome.**
The method is [`DOWNSTREAM_COMPLETION_SPEC.md`](DOWNSTREAM_COMPLETION_SPEC.md) §2, frozen before any M2 fit
(commit `4dc2fad`). This file fixes only execution order, identities and artifacts; it changes no equation,
estimator, gate, estimand or naming rule. Implementation: `m3_evaluate.py` (both weights arms, run
separately), `m3_spatial.py` (the estimator library), `m3_independent_check.py` (post-result reconstruction).

## 1. Execution boundary and prerequisites

`m3_evaluate.main(--weights station|land)` refuses, before any spatial fit, unless:
* `PYTHONHASHSEED=0`; the output root is fresh;
* the code closure (transitive local imports of `m3_evaluate.py`, this file, `DOWNSTREAM_COMPLETION_SPEC.md`,
  `M3_FEASIBILITY_AUDIT.md`, `pyproject.toml`, `uv.lock`) is tracked, identical to `HEAD` and contained in the
  live remote head;
* the M2 primary result verifies against its manifest, passed integrity, and is committed and pushed; the M2
  conditional disposition (`outputs/m2_conditional/`: a manifest with its artifacts, or the not-run record) is
  committed and pushed;
* `--weights land` additionally requires the station package (`outputs/m3_station/`) to verify against its
  manifest and be committed and pushed;
* the M2 scoring-input gate (`m2_evaluate.verify_pins`, `build_frozen`) passes again.

**Branch.** S is read from the committed M2 manifest `retained_static_specification` (`M0*` or `M2`).

## 2. Order of computation

1. **S reproduction (refusal on failure).** For both representations: S (and, in branch B, M0\*) is refitted
   with the frozen M2 functions for the full sample and every primary, M49 and random fit; in-sample and
   out-of-fold predictions must match `m2_country_predictions.csv` to 1e-10, and S's card metrics on the M2
   `REFERENCE_METRICS` must match `m2_scorecard.json` to 1e-10.
2. **Graph identity.** The full-sample kNN8 matrix of the arm equals the M1b identity digest.
3. **Spatial fits.** For each family (SEM, SAR) and representation: the full-sample fit, then every primary,
   M49 and random fit in fold order, each with its own training encoder, latitude state (branch B), training
   graph and sink attachments (spec §2.2, §2.7). A `SpatialFitFailure` marks that family non-computable for
   the arm and representation; no later fit of that family is attempted and nothing is repaired.
4. **Cards, paired comparisons, qualification** (station arm: Q1–Q5 decide; land arm: `arm_conditions` only),
   accounting, filtered shares, dependence-parameter summaries, graph descriptors, overfitting diagnostics.
5. **Representation identity** (θ̂ full and every primary training fit 1e-5; predictions, non-allocation scores
   and accounting parts 1e-6; `DOWNSTREAM_COMPLETION_SPEC.md` Amendment A1). A family computable in one
   representation only is an integrity failure.
6. **Final naming** (station arm only), spec §2.14.

## 3. Artifacts (deterministic unless stated)

| File | Contents |
|---|---|
| `m3_scorecard.json` | branch, S reproduction record, per family × representation: card, status and first failure, paired comparisons, Q1–Q5 or `arm_conditions`, accounting, filtered shares, dependence parameter, graph descriptors, diagnostics; representation identity; final naming (station arm); integrity |
| `m3_country_predictions.csv` | arm × family × representation × protocol (`in_sample` one-step, `primary_loco`, `m49_subregion_lo`, `random10`) × country: observed, prediction, error, trend (in-sample rows), fold, n_train, nearest training km, unseen levels, adjustment a_o |
| `m3_fits.csv` | every spatial fit: family, representation, protocol, fit label, n, p, θ̂ (repr), ℓc(θ̂), σ̂², at_domain_bound, grid local maxima, evaluations, columns, β̂ (JSON), training ISO3 list, encoder levels (JSON), latitude state (JSON or empty) |
| `m3_graphs.csv` | every fit: training-node neighbour lists (ISO3, nearest first) and held-out attachment lists |
| `m3_full_sample_grid.csv` | ℓc on the 199-point grid for each full-sample fit |
| `m3_provenance.json` | code closure digests and commit, M2 result digests, input identities, weights digest, software |
| `m3_result_manifest.json` | SHA-256 of every deterministic artifact; evaluator commit; arm; branch; family statuses; qualifying set and final naming (station arm); integrity |
| `m3_run_metadata.json` | not deterministic |

## 4. Independent reconstruction and reproducibility

`m3_independent_check.py` imports neither `m3_spatial.py` nor `m3_evaluate.py`. From the saved artifacts and
the pinned inputs it rebuilds each fit's design and graph from the saved lists, recomputes ℓc through the
eigenvalues of W (`m3_independent_math.py`), checks that θ̂ is a local maximum on a fine grid (one-sided at
the domain bound) and in the global grid basin, rebuilds β̂ by QR, recomputes every prediction by explicit
loops, and recomputes metrics, paired intervals, Q1–Q5, accounting parts, filtered shares by permutation
averaging, and the final naming. Negative controls (perturbed θ̂, a swapped neighbour, a perturbed
prediction, a shifted veto threshold) must be detected and are saved. One unchanged-code rerun into a fresh
root must reproduce every deterministic artifact byte for byte.
