# M1b closure: verification from saved evidence and dated documentation corrections

**Recorded 2026-09-16, after the M1b result (`1680d85`) and the conditional non-execution record
(`067bc98`). This is post-result documentation only.**
* M1b was not rerun. The evaluator was not run in scoring mode, and no conditional runner, independent
  check, redundancy or sensitivity script was executed.
* Two things did run, for the separate CLI maintenance (`M1B_CLI_MAINTENANCE.md`):
  * the evaluator's auxiliary `--compare` mode, once, on synthetic placeholder files;
  * the full test suite. It includes a committed test that scores the two M0\* comparators on real
    inputs to check their frozen cards, and it never scores a C2 candidate.
* All 133 output files hashed identically before and after, and no artifact was regenerated.
* Every saved M1b scorecard, prediction, fold, coefficient, design, coalition, provenance, manifest,
  bootstrap, reproducibility, independent-check and refusal record is byte-unchanged.
* The original report is preserved in Git: `M1B_REPORT.md` blob `7d08ed26` at `1680d85`.

## 1. Status after closure

| Line | Status |
|---|---|
| **M1b** (M0\* + C2 baseline hydroclimatic dryness) | **Completed negative experiment, not promoted.** Verdict `not supported`: generalization worsened under the frozen procedure. |
| **M1a** (area-consistent geography remeasurement) | Not promoted. Remains the registered area-consistent geography **measurement sensitivity**. |
| **M0\*** | Remains the station-geography **development baseline** and the comparator for any later stage. |
| Follow-ups | **None automatic.** There is no nonlinear, saturating or interacted C2, no alternative C2 definition, window, dataset or coverage rule, no C1 or C3 acquisition, no conditional replication and no M3. |

Any later M2 work rests on the `af97bd2` roadmap and a separate owner authorization, never on this
result. Its provenance is audited in a separate, later commit.

## 2. What was verified

**Method.** Six read-only verification dimensions were checked:
* artifact digests and code-path provenance;
* headline numbers and the verdict;
* representation equivalence and allocation;
* the grounding of the conditional refusal;
* a contamination sweep;
* report chronology.

Each dimension had an independent verifier and a separate adversarial re-check told to refute it.
The load-bearing claims were then re-derived directly in the closure session.

### 2.1 Repository and artifacts

* **Start state.** At the start, `HEAD` = `origin/research/model-v2-residual-structure` =
  `25732fbed7a3cf40885013a3da466887430dcaf0`. `git diff --exit-code v1.3.0 HEAD -- . ':!research'` and
  `git diff --check` were both clean.
* **Artifact digests.** All nine `m1b_result_manifest.json` digests recompute exactly. The manifest
  records evaluator commit `618c8b8`, verdict `not supported`, `association_supported: false` and
  `integrity_passed: true`.
* **Code-path provenance.** All 18 `CODE_PATH` digests in `m1b_provenance.json` equal the file blobs
  at `618c8b8` and at `25732fb`. `CODE_PATH` is a literal tuple containing no test file, so the
  test-only commit `25732fb` did not touch the scored code path.

### 2.2 Headline result (saved values, primary total-CO₂ representation, n = 151)

| Quantity | M0\* | M1b |
|---|---:|---:|
| In-sample R² | 0.6364460027547281 | 0.6364565668953286 |
| Primary spatial-CV R² | 0.15307991296259638 | 0.08363912905523074 |
| Primary spatial-CV RMSE (°C/decade) | 0.04287258067948669 | 0.044595565735706416 |
| M49 RMSE | 0.03889413690932386 | 0.03969289536306859 |
| Worst eligible region (Central Asia, n = 4) RMSE | 0.06936437825894065 | 0.07349481277418388 |

* **Paired comparison.** ΔRMSE = **+0.0017229850562197266**. The paired country-resampling interval
  (2,000 resamples, seed 0) is **[+0.0008914057822278219, +0.0028494442025791093]**.
* **`baseline_dryness`.** The full-fit coefficient is **+0.0004740731807010154**. It is strictly
  negative in **68 of 151** primary training fits, and none is exactly zero.
* **Countries.** **44** improved and **107** worsened, with no ties.
* **Verdict Booleans as stored.** S1, S2, A1, A2, A4 and A5 are false; **A3 is true**.
  `worsened_generalization` is true and `improvement_without_prestated_sign` is false. Each Boolean
  matches the contract §4.4 thresholds when they are re-applied by hand to the saved numbers.
* **Reference values supplied to the closure verification** (not a repository document). Several
  differ from the saved artifacts in the last binary digits, by at most 2.3e-16:
  * M1b in-sample R² …287 (saved …286);
  * M0\* CV R² 0.1530799129625966 (saved 0.15307991296259638);
  * some share values, which match the independent check's recomputed floats rather than the
    scorecard's.

  * The saved M0\* values are anchored by the frozen pre-M1b rank-audit cards and predictions
    (`9fb56e1`).
  * The saved M1b values reproduce from the saved predictions and coalition R² values (the independent
    check) and from the committed two-run reproducibility record.

  These gaps are far below every contract tolerance, and they are not artifact drift.

### 2.3 Representation equivalence (unconditional per-capita arm, A3.6)

* Maximum prediction difference **4.4242e-14**, against a tolerance of 1e-9.
* ΔRMSE difference **1.2490e-16**; interval bounds differ by at most 3.9e-16.
* Full-fit C2 coefficient difference **1.6263e-17**; at most 2.56e-16 across the 151 training fits,
  against a tolerance of 1e-8.

### 2.4 Allocation (in-sample LMG/Shapley, primary representation)

| Group | M0\* share | M1b share | Shift |
|---|---:|---:|---:|
| geography | 0.51883745 | 0.48399974 | **−0.03483771385643075** |
| population | 0.04165336 | 0.03622326 | **−0.005430103864027112** |
| socioeconomic | 0.06282147 | 0.05933838 | **−0.0034830934768674826** |
| responsibility (emissions) | 0.01313372 | 0.01106872 | **−0.002064998689786929** |
| hydroclimate | — | 0.04582647 | **+0.04582647402771282** |

The shifts are differences of the saved scorecard shares. The independent check's recomputed floats
agree to within 8.3e-17: for example geography −0.034837713856430697 and hydroclimate
+0.045826474027712845.

* **The shifts net to the full-model gain.** They sum to **+0.0000105641406005**; the scorecard
  `in_sample_r2_gain` is 1.056414060052191e-05. This netting holds by construction, because LMG shares
  sum to R², and the residual share falls by the same amount.
* **Geography's allocation falls by about 76% of the total displaced.** The four negative shifts total
  0.045815909887; geography's 0.034837713856 is **76.04%** of that. This split is not an identity: it
  reflects how strongly C2 correlates with each group.
* **Per-capita hydroclimate share.** **0.04492079275334753** in the scorecard; the independent check
  gives …756.

**The shares show how LMG splits variance that C2 shares with correlated groups. They do not show where
warming physically comes from.**
* LMG averages each group's marginal R² contribution over every order in which the groups can enter.
  A group correlated with groups already present can therefore receive a large allocation even when
  its last-entry improvement is negligible.
* Here C2 adds 0.0000106 to the full-model R², yet receives a 0.0458 share, about 4,300 times its
  last-entry increment. The pre-outcome redundancy diagnostic had found 65% of C2's cross-country
  variance inside the M0\* design space.
* The shifts identify no physical source of warming. They do not show that dryness "takes"
  explanatory power from geography in any causal sense.

### 2.5 Conditional analyses

* **The refusal comes before any arm is built or fitted.** `m1b_sensitivity.run` first calls
  `verified_primary`. That function digest-verifies the committed, pushed primary result and reads
  its verdict.
  * On `association_supported: false` the runner writes the not-run record and returns.
  * That return comes before `verify_scoring_inputs`, `build_frozen` and `run_arms`, the only path
    to any arm builder, design matrix, fit or CV call.
* **The record is grounded in the committed result.** `m1b_conditional_not_run.json` cites the nine
  primary digests, which match the manifest, and a verdict byte-identical to the committed scorecard.
* **The rule is the contract's.** Contract §5 and Amendment 2 A2.1 make the replications conditional
  on the linear association being supported.

### 2.6 Contamination sweep

Tracked, untracked and ignored files under `research/` were searched by directory, file name, file
type and content tokens. The sweep found:
* no conditional scorecard;
* no M2 or M3 result;
* no fitted spline or hemisphere-interaction coefficient, CV score, RMSE or R².

The only M2-related files were this session's untracked design drafts, which are predictor-only and
read no outcome.

## 3. Evidence limits recorded at closure

These do not change the result. They bound what the saved evidence establishes.

1. **The reproducibility rerun can no longer be re-inspected.** Its second run lived in a session
   scratch directory that no longer exists. The two-run byte identity now rests on the committed
   `m1b_reproducibility.json`; the run-A digests still equal the manifest.
2. **The independent check's negative controls are not recorded.** `M1B_REPORT.md` §5 says negative
   controls were run for every family and all fired. Neither the committed
   `m1b_independent_check.json` nor the module records them, so this cannot be verified from saved
   evidence.
3. **The independent check is independent in code, not in time.** It shares no code with the
   evaluator but was written after the evaluator run and committed with the result (`1680d85`). It
   validates aggregation of saved predictions, coefficients and coalition R², not the regressions
   themselves.
4. **The result manifest and the provenance record are self-attesting.** One process wrote both, and
   the manifest records the provenance digest (not the reverse). Tamper evidence after the fact comes
   from the pushed Git history, not from the digests alone.
5. **`CODE_PATH` is a declared list, not the full import closure.** A static import closure of the
   evaluator at `618c8b8` has 31 local files, 16 of them not listed. Examples: `geometry.py`,
   `m1a_geography.py`, `src/area_weighting.py`. All are unchanged from `618c8b8` to `25732fb`, and the
   production tree is unchanged since `v1.3.0`, but the recorded digests do not cover them.
6. **The scored C2 values depend on the CSV parser.** The evaluator read the C2 package with pandas'
   default parser. The scored values equal that parse exactly, but differ from a round-trip-exact
   parse by at most 4.4e-16 in 88 countries. The difference is immaterial.
7. **Real-data comparator scoring happened before the freeze, from uncommitted code.** Before freeze
   commit `618c8b8` (committed 00:53:58Z, pushed 00:54:02Z), the evaluator's `verify_scoring_inputs`,
   `build_frozen` and `score_model` were called directly on real inputs from the uncommitted working
   tree. That bypassed the `main()` provenance guard (`M1B_EVALUATOR_SPEC.md` §1).
   * **Main session.**
     * 00:04Z: both M0\* comparators (total-CO₂ and per-capita) against their `rank_audit.json` cards.
     * 00:22Z: the M1a-geography and aligned-ERA5 arm comparators, in both representations.
     * 00:40Z: one timing run of the primary comparator.
     * None of these runs called `build_arms`.
   * **Review subagents in the same session.**
     * 00:29Z: both primary comparators.
     * 00:31Z: the M1a-arm comparators, and full-sample aligned-ERA5 comparator fits.
     * 00:40Z: construction, without scoring, of the arm frames, including the 146-country arm.
     * 00:45Z: the primary comparator again.
   * **Committed test.** `test_m1b_scoring_gate.py::test_the_comparators_reproduce_the_frozen_rank_audit_cards`,
     committed at `618c8b8`, scores both M0\* comparators on real inputs whenever the suite runs.
   * **Candidate frames.** Three review-subagent runs (00:29Z, 00:31Z, 00:31Z) called `build_arms`,
     which constructs candidate frames containing the real, already-frozen C2. **Only the comparator
     frames were scored.**
   * **Not all anchors were frozen.** The M0\* and M1a-arm comparators were checked against frozen
     cards on all reference metrics. The aligned-ERA5 arm has a frozen reference only for in-sample R².
     Its spatial-CV values were new, uncommitted numbers, for example M0\* CV RMSE 0.0672321.
   * **No model containing C2 was fitted to any outcome before the primary score** (00:54:12Z). No
     conditional replication in the contract's sense (candidate against comparator within an arm) was
     ever computed.

   **Evidence.**
   * The pushed freeze commit message records the anchor checks: "Verified before the freeze, without
     fitting C2 to any outcome".
   * The printed values, the subagent runs and the timing run rest on local session transcripts, which
     are mutable records rather than repository artifacts. The session began at 23:05:20Z, after the
     redundancy diagnostic (`80cd73b`, 22:39:50Z). Its scratch directory has been deleted.
8. **The `067bc98` commit message is inaccurate.** It says the runner checked "the scoring-input gate"
   before refusing. In the code that ran, the refusal returns before `verify_scoring_inputs`; the
   provenance guard and the primary result's digests were checked instead. The committed record itself
   makes no such claim. Commit messages are immutable, so the correction is recorded here.
9. **The report committed a statement about the refusal record before that record existed.**
   `M1B_REPORT.md` says "the runner records the non-execution". The report was committed at
   01:09:13Z, 16 s before the runner ran and wrote the record that `067bc98` committed. The statement
   turned out to be accurate.
10. **Some refusal-path tests postdate the run.** The tests exercising `verified_primary`'s Git and
    verdict checks were added in `25732fb`, after the refusal run, against the identical runner
    bytes.
11. **Push timing rests on local evidence.** The remote exposes no push times. The local
    remote-tracking reflog records the `618c8b8` push at 00:54:02Z, 10 s before the run started
    (00:54:12Z), and the run metadata records upstream commit `618c8b8`. It records the `1680d85` push
    at 01:09:15Z, before the conditional runner (01:09:29Z). Both are local, mutable records.
12. **CRU source bytes were not re-hashed at closure.** The chronology in §4 uses three sources:
    * commit times;
    * the local remote-tracking reflog (push times);
    * CRU file birth and modification times, which are consistent but not tamper-proof.

    The recorded hashes were first committed at `c5ff9b5`. The 6.2 GB and 1.6 GB decompressed CRU files
    were not re-hashed during closure.

## 4. Dated corrections to `M1B_REPORT.md` (post-result, 2026-09-16)

The report is edited in place with dated markers. Each passage it originally contained is quoted
verbatim below. No number, verdict or artifact changes.

### C1. Chronology of the contract amendments and CRU acquisition

**Original (opening paragraph):** "The rules were frozen in
[`M1B_EVALUATION_CONTRACT.md`](M1B_EVALUATION_CONTRACT.md) (Amendments 1–3) before any CRU data were
acquired, and the operational detail in [`M1B_EVALUATOR_SPEC.md`](M1B_EVALUATOR_SPEC.md) before this
run."

**Why it is wrong.** Only the original contract preceded acquisition. All three amendments came after
acquisition and after C2 values existed under the original coverage rule. Exact order (CDT,
2026-09-15), from commit times, the local push reflog and CRU file times:

| Time | Commit | Event |
|---|---|---|
| 13:09:37 (pushed 13:09:38) | `60111ae` | Original measurement and evaluation contract frozen and pushed "before CRU acquisition". It amends the unpushed local draft `02154db`, authored 13:00:30. |
| 13:09:54–13:23:08 | — | CRU TS v4.10 PRE/PET `.gz` downloaded (file birth and modification times) |
| 13:24:22 | `ef09196` | C2 builder, redundancy-diagnostic script and first evaluator draft. No C2 value exists yet and no diagnostic has run. |
| 13:24:43–13:25:11 | — | Decompressed `.nc` files created. The first build attempt (`M1B_FEASIBILITY_AUDIT.md` §1) ran after this. |
| 13:27:03 | `c6ef919` | Station-count validation fix |
| 14:33:13 | `c5ff9b5` | Feasibility audit: C2 built under the original coverage rule (three countries fail); source hashes first committed |
| 15:57:15 | `5514a86` | **Amendment 1** (structural land-mask harmonization) |
| 15:59:09 | `f27d0a0` | **Amendment 2** (station-support sensitivity; outcome-free redundancy inputs) |
| 16:04:07 | `bd518a0` | Amendment 1 build stopped on one non-structural Peru cell. Its C2 outputs are preserved. |
| 17:20:43 | `58e64bf` | **Amendment 3** (non-structural support counts against coverage; integrity gates; A3.6 per-capita) |
| 17:23:52 | `222e2c1` | Frozen Amendment 3 package. Its C2 equals the stopped Amendment 1 build in all 151 countries. |
| 17:39:50 | `80cd73b` | Pre-outcome redundancy diagnostic |
| 19:53:58 | `618c8b8` | Evaluator frozen |
| 20:09:13 | `1680d85` | First and only M1b score recorded |

Every amendment preceded the redundancy diagnostic and any outcome-facing fit involving C2.

**Corrected text** (links omitted): "The original measurement and evaluation contract,
`M1B_EVALUATION_CONTRACT.md`, was frozen and pushed before any CRU data were acquired (`60111ae`).
Amendments 1–3 were adopted after acquisition and after C2 values existed under the original coverage
rule, but before the redundancy diagnostic and before any outcome-facing fit involving C2. The
operational detail in `M1B_EVALUATOR_SPEC.md` was frozen before this run."

### C2. Where the hydroclimate allocation comes from

**Original (§3, group shares):** "The hydroclimate share is drawn almost entirely from geography
(−0.0348), which is what the pre-outcome redundancy diagnostic predicted: …"

**Why it is wrong.** Geography's allocation accounts for about 76% of the displaced total, not almost
all of it, and the sentence reads as identifying a source.

**Corrected text:** "Relative to M0\*, the named-group allocations shift by: geography −0.03484,
population −0.00543, socioeconomic −0.00348, responsibility −0.00206 and hydroclimate +0.04583. The net
change equals the in-sample R² gain (+0.0000106) by construction, because LMG shares sum to R².
Geography accounts for about 76% of the displaced allocation. That split reflects C2's correlation with
each group under LMG's averaging over coalitions, and it does not identify a physical source. It is
consistent with the pre-outcome redundancy diagnostic: …"

### C3. Coverage-rule sentence (identified during closure verification)

**Original (§6):** "No alternative window, transform, dataset, threshold, aggregation order or coverage
rule was tried, before or after."

**Why it is inaccurate.** The coverage rule was amended twice (Amendments 1 and 3) before the redundancy
diagnostic and before any outcome-facing fit involving C2. Those amendments are recorded in the
contract, and their C2 values are identical, but "coverage rule … before" is literally false.

**Corrected text:** "No alternative window, transform, dataset, threshold or aggregation order was
tried, before or after. The coverage rule changed only through the recorded pre-outcome contract
Amendments 1 and 3, and no coverage rule was tried in response to this result."

## 5. The auxiliary `--compare` defect

`M1B_REPORT.md` §8 is unchanged by this closure. The defect is addressed separately, as dated
post-M1b maintenance, in its own commit. The M1b result remains the product of `618c8b8`, and
historical reproduction remains pinned there.
