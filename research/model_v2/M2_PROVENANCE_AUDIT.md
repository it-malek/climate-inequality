# M2 historical provenance audit

**Recorded 2026-09-16, after the M1a and M1b results. Design-only: no M2 fit, score, cross-validated
prediction or coefficient exists.** This audit reconstructs from Git, not from current prose, what was
proposed for M2, when, and how specifically, so the M2 contract admits only what the historical record
supports.

**Method.**
* **Readers.** Six read-only readers each quoted one source verbatim:
  * `af97bd2` `EXPERIMENTAL_PLAN.md`;
  * `af97bd2` `HYPOTHESIS_REGISTER.md`;
  * `da4455d` `M1B_HYPOTHESIS_REGISTER.md`;
  * the full `git log -p` evolution of the plan and register;
  * the current decision, contract and protocol documents;
  * the M1a/M1b/M0.5 carry-over rules.
* **Classifiers.** Three classifiers (strict, skeptical and archivist) worked from that corpus. Load-bearing
  quotes were then re-read directly with `git show`.
* **Constraints.** No evaluator was run, and no residual value was read.

## 1. What the record contains

**The M2 section of `EXPERIMENTAL_PLAN.md` first appears in `af97bd2`** (2026-09-14 23:17:25 −0500),
before any M1a or M1b outcome, and is byte-identical at `25732fb`:

> **Change.** (a) `hemisphere × abs_latitude` (H1); (b) a natural cubic spline in `abs_latitude` with
> three degrees of freedom and knots at the sample tertiles (H5), fixed a priori, no df search; (c) at
> most one interaction between a retained M1 covariate and continentality or latitude, chosen before
> fitting and written into the register. Nothing else: no GAM over all features, no automated term
> selection.
>
> **Gates.** Retention as in M1, with the overfitting signal checked explicitly (effective degrees of
> freedom is recorded).

The section heading is "pre-specified nonlinear terms (H1, H5; at most three)". The register summary
in the same commit reads "M2 = H1, H5 (spline)".

The same commit's `SPATIAL_CV_PROTOCOL.md` ("Nested validation and model selection") adds:

> Any quantity tuned to data (a spline's degrees of freedom, a penalty, a spatial range) must be chosen
> inside each training fold (inner buffered leave-one-out on the training set) or fixed a priori; M2
> fixes its degrees of freedom a priori to avoid nesting.

`SCORECARD.md` defines effective degrees of freedom for spline stages and lists AIC/BIC as deliberately
not recorded.

**No centering or coding convention for the hemisphere interaction appears anywhere in the history.**
The owner's 2026-09-16 coding choice, `I(SH) · abs_latitude` with centre 0°, therefore overrides no
historical convention.

## 2. Classification

* **A.** A pre-existing hypothesis with an identifiable specification.
* **B.** A prior mention whose adoption now would be a post-result rescue. Ineligible.
* **C.** Previously contemplated but insufficiently specified.

| # | Proposal | First commit | Class | Eligible for M2? |
|---|---|---|---|---|
| 1 | H1 / plan M2(a): `hemisphere × abs_latitude` ("or separate latitude slopes per hemisphere") | `af97bd2` | **A** | Yes, after freezing the coding, measurement and rules (§3) and with the disclosures in §4 |
| 2 | Plan M2(b): natural cubic `abs_latitude` spline, three df, knots at the sample tertiles, no df search | `af97bd2` (plan only; the register says just "fixed degrees of freedom") | **A** | Yes, in exactly this pooled form, after freezing the details in §3 |
| 3 | H5: NH high-latitude snow/ice-albedo curvature, "steepening above ~50°N" | `af97bd2` | **C** | Only as the historical motivation for #2. An NH-only spline, a ~50°N knot or a convexity pass/fail rule would be new, residual-shaped choices. |
| 4 | Plan M2(c): at most one interaction of a *retained* M1 covariate with continentality or latitude | `af97bd2` | **C, void** | No. No M1 covariate was retained (M1a not promoted; M1b not supported). The slot authorizes nothing and **does not reintroduce C2** in any form: linear, saturating, spline or interacted. |
| 5 | H3 item 7: aridity effect "plausibly saturating in hyper-arid regimes (hence a candidate for M2 curvature)" | `af97bd2` | **B** | No. It predates M1b, so it is not invented after the result. But it fixes no form, and reviving it immediately after linear C2 failed would be a rescue. Pre-score contract `60111ae` §4.6: nonlinear dryness belongs to M2 "only if independently justified later, and may not be introduced in response to this score". |
| 6 | `M1B_HYPOTHESIS_REGISTER.md` C2 row: "No saturation or spline term: curvature is M2's question." | `da4455d` (after M1a, before M1b) | **B** | No. It is an M1b scope boundary, not a registration. No M2 text ever carried it forward. |
| 7 | M1 → M2 admission gate: SH Spearman \|ρ\| > 0.4, or OOF Moran's I > 0.2, or ≥ 10 countries in significant LISA clusters | `af97bd2` | **C** (exact thresholds, but its object no longer exists) | Yes, with an explicit owner disposition (§5). |
| 8 | M2 retention rule: "retention as in M1, with the overfitting signal checked explicitly" | `af97bd2` | **C** | Yes, with freezing. Spline basis coefficients have no invariant sign, but γ does, and H1 pre-states it as negative. The owner's 2026-09-16 removal of every sign criterion is a disclosed loosening (§4 item 9). |
| 9 | H6 item 7: SH ocean influence "possibly SH-specific (an interaction with hemisphere, hence paired with H1)"; C1 "kept for possible later M2 nonlinear or interaction work" | `af97bd2`; `da4455d` | **C** | No. It needs a never-measured, held physical covariate outside M2's list ("Nothing else"). |
| 10 | Elevation-dependent warming "subsumed by C3 or M2" | `da4455d` | **C** | No. |

**Context, not a proposal.** Flexible latitude forms predate V2, and neither of the following may
justify any M2 df, knot, basis or interaction choice:
* **V1 robustness phase** (`0ea0999`, 2026-06-18). A B-spline latitude term was fitted to the country
  warming trend as a control for the emissions coefficient: statsmodels `GLMGam`, `df = 4`, degree 3,
  default `alpha = 0`, so unpenalized; legacy station outcome.
* **V1 phase 7** (`43a795c`, 2026-06-15). City-level `abs_latitude:C(koppen)` interactions were fitted.

## 3. What history fixes, and what must be frozen now

**Fixed by `af97bd2`:**
* df fixed a priori, with no nesting (`SPATIAL_CV_PROTOCOL.md`);
* the pooled `abs_latitude` variable;
* a natural cubic spline;
* three degrees of freedom;
* knots at tertiles;
* no df search;
* the product term `hemisphere × abs_latitude`;
* at most three terms;
* "Nothing else".

**"Three df" has one internally consistent reading.** Two tertile cut points plus two boundary knots
give four knots. A natural cubic spline on four knots spans four dimensions including the constant, so
it has three non-constant dimensions. "Three df" therefore excludes the intercept. This is R's
`ns(x, df = 3)` convention. The intercept-included reading contradicts "tertiles".

**Not fixed historically, frozen in `M2_DESIGN.md` as dated post-M1b decisions:**
1. **"Sample" tertiles.** Whether the full sample or each training fit. Frozen as training predictors
   only. This follows the `af97bd2` nested-validation principle, although that text does not name knots
   explicitly.
2. **Quantile algorithm and ties.**
3. **Boundary knots and the tail convention.**
4. **The basis parameterization.** It does not change the fitted space.
5. **Whether the spline replaces or keeps M0\*'s linear latitude term.** Nested: kept, with two added
   columns.
6. **Interaction coding and centre.** Owner: `I(SH) · a`, c = 0°.
7. **Measurement.** M0\*'s station `abs_latitude` and `hemisphere`; M1a's area measurements only as a
   later sensitivity.
8. **Joint versus separate testing.** Owner: one joint candidate.
9. **Retention and sign rules.** Owner: no coefficient-sign criterion; out-of-fold gates only.
10. **Identification and degeneracy rules.**

## 4. Honesty disclosures that travel with M2

1. **H1 and H5 predate every M1 outcome, but they are not outcome-naive.** The `af97bd2` register opens
   "Hypotheses generated from the M0 residual". It was written in the commit that computed that
   residual:
   * H1 cites the in-sample SH residual gradient (ρ = −0.68, n = 29);
   * H5 cites the under-predicted NH 50–70° band.

   "Pre-M1b" does not mean untouched data or independent external confirmation. An eventual M2 test is
   a **prospectively frozen follow-up within the same outcome dataset**.
2. **H1's directional expectation is not an independent prediction.** Its NH slope (~+0.001 °C/decade
   per degree, "as now") is M0's fitted coefficient, and "SH slope negative or zero" repeats the
   residual pattern. Only out-of-fold transfer criteria can carry evidential weight.
3. **The evidence status of H1 and H5 cannot carry M2's admission.**
   * `da4455d` withdrew the Session 1 Berkeley residual evidence for H3, H5 and H6 ("not relied on").
     It never mentioned H1, and H1 was never re-based on an independent rationale.
   * The M0.5 product audit (`9fb56e1`) classifies H1's SH gradient as "not directly classified
     (unresolved)". Under `PRODUCT_STABILITY_AUDIT.md` §7.2, unresolved patterns "must not be the
     justification for an M1 predictor".
   * H5's evidence is partly robust (Canada, Estonia) and otherwise unresolved or unclassified (Latvia,
     Northern Europe, Russia, Finland).
   * **Neither pattern justifies M2.** M2's admission rests on the `af97bd2` roadmap alone.
   * §7.2 is scoped to M1 predictors. Reading it as not governing pre-registered functional forms of
     existing M0\* features is a recorded interpretation (`M2_EVALUATION_CONTRACT.md` §12).
4. **The executable terms differ from the evidence that motivated them.**
   * **Shape.** H5's mechanism is NH-specific and centred near 50°N, but M2(b) pools both hemispheres
     with tertile knots. A natural cubic spline on those knots forces curvature to zero at the training
     maximum, so it cannot represent an onset near 50°N (`M2_DESIGN.md` §6).
   * **Measurement.** H1's ρ = −0.68 and H5's 50–70° band were computed in `af97bd2` `diagnostics.py`
     on **land-area-centroid |latitude|** (`abs_centroid_lat`), with station hemisphere. M2 uses
     **station `abs_latitude`**. For example, the NH 50–70° count is 17 on centroid latitude, matching
     H5's n = 17, but 16 on station latitude.
   * The M2(b) wording is kept. "Correcting" it now would be a new, residual-informed design.
5. **The design was written after both negative results.**
   * The M2 design and its code were written on 2026-09-16, about 15 h after the M1b result (`1680d85`,
     2026-09-15 20:09), whose report ends "no … M2 or M3 work follows from this result".
   * The M1b result is not the justification. M2 rests on the `af97bd2` roadmap, resumed by the
     owner's 2026-09-16 design-only authorization.
   * The designers had already seen the M0/M0\* residual diagnostics, the M0\* out-of-fold Moran's I,
     and the M1a and M1b scorecards.
6. **Stopping-rule pressure.** The M4 stopping rule covers "the best of M1a–M2".
   * Neither M1a nor M1b improved on M0\*'s primary RMSE. M1a's point estimate was worse, with an
     interval covering zero; M1b worsened, with an interval above zero.
   * M2 alone now decides whether the rule fires. This creates an incentive toward an M2 success, and it
     is the reason every detail, including the rule's operational form (contract §11), is frozen before
     any fit.
7. **Overlap with C2.** A latitude spline or hemisphere slope can represent subtropical dry-belt
   structure: C2 had R² 0.650 on the M0\* design, geography alone 0.6215. Any M2 result is a **geography
   functional-form result**. It is not support for hydroclimate or dryness, and not a rescue of C2.
8. **Some text reached the branch through amended commits.**
   * The C1 "possible later M2 nonlinear or interaction work" line was added in the `6e23539` → `da4455d`
     amend (chain `b9ebafd` → `6e23539` → `da4455d`).
   * The §4.6 "independently justified later" clause was added in the `02154db` → `60111ae` amend.
   * The pre-amend commits lack both passages and are reachable only from the local reflog.
   * `af97bd2` is a message-only amend of `caac037`, with an identical tree.
   * All of these amends predate the M1b score.
9. **The coefficient-sign rule was removed by the owner.** Under "retention as in M1", H1's γ, which has
   an invariant, pre-stated negative direction, would have faced the covariate sign rule. The owner
   removed every coefficient-sign criterion for M2 on 2026-09-16, after the M1b result. This is a
   disclosed loosening relative to the `af97bd2` rule and to M1b S2/A5.

## 5. The admission gate

* **The gate's object is gone.** The plan's gate is written for "the retained M1 model's out-of-fold
  errors". No M1 covariate was retained, so the retained model is **M0\*** itself. The M1a gate had
  already sent development back to M0\*'s measurements (`M1A_REPORT.md` §13).
* **M0\* passes condition (ii) on a pre-M1 value.** Its out-of-fold residual Moran's I is
  **0.3215471676927579** (station-centroid kNN8; `outputs/rank_audit.json`, card `drop_per_capita`,
  committed at `9fb56e1` on 2026-09-15 10:42, before the M1a score). That exceeds 0.2.
* **The other two conditions are not computed.** Condition (i), the SH Spearman gradient, and
  condition (iii), LISA counts, would need new residual analysis. **No M1b residual is used.**
* **The owner decided on 2026-09-16** that this frozen M0\* value satisfies the historical gate.

**The gate is procedural, not evidence for M2:**
* its thresholds were written with M0 values already visible and above all three (in-sample ρ −0.68,
  in-sample I 0.269, 29 LISA countries);
* since `da4455d` its section has been labelled "superseded by M1b";
* two of its three conditions measure spatial clustering, not nonlinearity;
* residual autocorrelation does not by itself establish nonlinear misspecification.

## 6. Carried-forward product and measurement caveats (M0.5)

* **ERA5 stays out of M2 model choice.** `PRODUCT_STABILITY_AUDIT.md` §7 keeps ERA5 out of predictor
  choice, transforms, model ranking, threshold tuning and CV design.
  * Its role is external robustness after a specification is frozen, never an unseen validation sample.
  * The aligned construction is the preferred arm.
  * No ERA5 relationship is used to choose M2's form or knots.
* **§7.2, quoted exactly:** "Product-sensitive and unresolved patterns may stay in the hypothesis
  register as observations. They must not be the justification for an M1 predictor." Its application to
  M2 is recorded in §4 item 3.
* **M1a stays a measurement sensitivity only.** It is replicated only as the contract specifies, and
  the station and area measurement sets are never searched jointly (`M1A_REPORT.md` §13). Under M1a's
  area measurements some countries change hemisphere (for example Gabon and Kenya).
