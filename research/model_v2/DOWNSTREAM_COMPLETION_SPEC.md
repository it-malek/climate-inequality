# Downstream completion specification: M3 and M4 (frozen before any M2 fit)

**Recorded 2026-09-16, before any M2 candidate fit or score, and before any M3 or M4 computation on
the warming outcome.** This file completes, prospectively, the executable mathematics of the two stages
that the `af97bd2` roadmap requires after M2. It is binding for both possible M2 outcomes. Observing the
M2 result may only select branch A or branch B (§1); it may not change any rule below.

* **Authority.** The owner's completion instruction of 2026-09-16 (recorded in
  [`M2_PRE_SCORE_RESOLUTIONS.md`](M2_PRE_SCORE_RESOLUTIONS.md)) authorizes M2 execution, the required M3
  accounting and the M4 final assessment, and asks that every result-sensitive M3/M4 choice be frozen
  here first.
* **Evidence inventory.** [`M4_EVIDENCE_INVENTORY.md`](M4_EVIDENCE_INVENTORY.md) maps every M4 requirement
  to its source, applicable model, reusable evidence, missing work and permitted disposition.
* **State.** [`COMPLETION_STATE.md`](COMPLETION_STATE.md).
* **Structural feasibility.** Predictor-only graph structure and synthetic-outcome estimator checks are in
  [`M3_FEASIBILITY_AUDIT.md`](M3_FEASIBILITY_AUDIT.md) and `outputs/m3_feasibility/`. No warming outcome
  was read for them.

## 0. Historical scope versus the executable operationalization

A stage being historically required is not the same as its method having been historically executable.
The table separates the two. Historical files are not edited to make a method look older than this file.

| Item | Historical source (what it establishes) | Executable then? | Frozen here |
|---|---|---|---|
| M3 families | `af97bd2` `EXPERIMENTAL_PLAN.md` M3: "a spatial error model and a spatial lag model on the retained M2 (or M1, or M1a) specification with the V1 weights (station-centroid kNN, k = 8), and with land-centroid kNN k = 8 as a sensitivity" | No: no equation, estimator, domain or prediction rule | §2 |
| Optional third family | same: "optionally a geostatistical residual" | No | **Not pursued** (owner, 2026-09-16); not rejected |
| Held-out graph | same: "Out-of-fold predictions use only training-fold neighbours; … a spatial term can improve the primary score only through whatever dependence survives beyond the buffer" | Principle only | §2.2, §2.7 |
| M2 → M3 | same: "**M2 → M3** always proceeds, because M3 is an accounting step, not a model-selection step" | Yes | Every integrity-valid M2 path runs M3 |
| M3 reportables | same: "the estimated spatial range and dependence parameter; how much of the residual share is absorbed; whether the group contributions of the non-spatial part change materially when the spatial term is present; and whether the spatial term generalises under the primary protocol at all" | Labels only | §2.11–§2.12 |
| M3 retention | same: "A spatial term is retained for prediction only under the improvement rule; its accounting value is reported regardless" + plan "Genuine improvement" + `SCORECARD.md` ("metrics 11 and 12 can veto it") + `e3e5901` `M1A_EVALUATION_CONTRACT.md` §4.1 (numeric vetoes, "the plan's genuine-improvement definition with its vetoes") | Yes, once the comparator is named | §2.10 |
| Non-OLS group shares | `af97bd2` `OPEN_DECISIONS.md` item 10 (a Session 1 recommendation, never approved; still open at `23d31f4`) | No | §2.11: transparent multi-part accounting, no forced "spatial share" (owner-permitted disposition) |
| Nesting | `af97bd2` `SPATIAL_CV_PROTOCOL.md`: data-tuned quantities "chosen inside each training fold … or fixed a priori" | Yes | k = 8 fixed a priori; θ estimated inside each training fit |
| Likelihood criteria | `af97bd2` `SCORECARD.md`: AIC/BIC or likelihood-based criteria "deliberately not recorded" | Yes | No likelihood criterion is used or compared (§2.5) |
| M4 table and bootstraps | `af97bd2` plan M4 and `SCORECARD.md` ("bootstrap intervals for the shares of the retained stages"; V1 country bootstrap 2,000 seed 0 and continent block bootstrap seed 1 "computed at M4 for the retained stages") | Procedure yes; its object (the retained stage) no | §3.2 |
| M4 sensitivities | `af97bd2` `SPATIAL_CV_PROTOCOL.md` (1000 km border buffer; 1500 km centroid buffer, "to show that a model ranking does not depend on the exact width or metric"); corrected implementation `4263429` | Yes | §3.4 |
| Product check | `af97bd2` plan M4 (H7 arm, conditional on owner approval, with a two-product mean); `9fb56e1` `PRODUCT_STABILITY_AUDIT.md` Decision ("A post-freeze ERA5 check of the eventual model remains necessary") and §7.3 ("final post-freeze ERA5 check, using both constructions"); `9fb56e1` `HYPOTHESIS_REGISTER.md` ("Repeat product robustness after V2 is frozen") | No model, metric or scope defined | §3.5 (no two-product mean: owner, 2026-09-16) |
| Final model naming | No historical unique-family selection rule exists (searched: plan, register, scorecard, protocol, decisions, M1a/M1b/M2 contracts) | — | §2.14 (owner disposition table) |

## 1. Branches selected mechanically by the M2 result

| M2 primary label (`M2_EVALUATION_CONTRACT.md` §5) | Branch | Retained static specification S |
|---|---|---|
| `supported and promoted` | **B** | **M2** (M0\* + `abs_latitude_ns1` + `abs_latitude_ns2` + `hemisphere=S:abs_latitude`) |
| `predictive support, not promoted` | **A** | **M0\*** |
| `not supported` | **A** | **M0\*** |
| no label (integrity failure) | none | Stop policy; M3 and M4 do not start |

The branch is read by the M3 evaluator from the committed, pushed, digest-verified M2 result. Nothing else
about the M2 result enters M3 or M4 as a rule.

**Static stopping indicator** (reported with M2 and carried into M4): with d the primary total-CO₂
Berkeley paired ΔRMSE of M2 against M0\*, the indicator **fires iff d > −0.002** (equality does not fire
it). M1a (+0.002343) and M1b (+0.001723) already have positive point deltas, so d decides it. It is
independent of promotion and of branch selection, and no branch authorizes further static-model search.

## 2. M3: explicit spatial structure

### 2.1 Data and design

* **Countries, order, outcome.** The frozen 151 countries in canonical order; the Berkeley area-weighted
  trend 1950-01…2013-09 (°C/decade), bound to the frozen ordered reference exactly as the M2 evaluator
  binds it.
* **Static design X for a fit.** S's encoded design built from that fit's training rows only: intercept;
  exact numeric predictors (the pinned `M0star` rows of `m1b_design_matrices.csv`, round-trip parsed,
  already transformed); train-only drop-first categorical coding from the pinned labels; in branch B, the
  M2 latitude block computed with the fit's own training `LatitudeState` (`m2_latitude_basis.py`,
  unchanged). p = number of columns; every required fit must have rank(X) = p (inherited).
* **Representations.** Primary total CO₂ decides everything. The per-capita representation is fitted and
  reported unconditionally with identical folds, graphs, states and optimizer; it decides nothing.
* **Groups.** The four named groups `emissions`, `geography`, `socioeconomic`, `population`; in branch B
  the latitude additions stay in `geography`. No spatial column is added to any group.

### 2.2 Weights and graphs

* **Station-centroid distances (primary).** `station_lon`, `station_lat` of the pinned
  `outputs/m0_countries.csv` (`9e99b379…`, read with `float_precision="round_trip"`), great-circle distances from `spatial.haversine_matrix`
  (radius 6371.0088 km), one 151 × 151 matrix D. The full-sample row-standardised kNN8 matrix built from it
  must equal, by SHA-256 of its float64 bytes, the M1b identity `station_weights_sha256`
  (`m1b_provenance.json`).
* **Land-centroid distances (sensitivity, §2.13).** `centroid_lon`, `centroid_lat` of the pinned
  `outputs/country_geometry.csv` (`c8d72866…`), indexed by ISO3 in canonical order; the full-sample kNN8
  matrix must equal the M1b identity `area_weights_sha256`.
* **kNN8 on a node set N** (N listed in canonical order). For i ∈ N its neighbours are the first 8
  elements of N \ {i} ordered by (D[i, j], canonical index of j) ascending; w_ij = 1/8 for those, 0
  otherwise; zero diagonal. This is `spatial.knn_weights` applied to D[N, N], whose local order is the
  canonical order, so the tie rule (distance, then index) is unchanged. Every row sums to exactly 1.
* **Training graph.** For a fit with training set T: W = kNN8(T). The training graph never contains a
  held-out or buffer-excluded node.
* **Held-out attachment (sink rule).** Each held-out node o receives links to the first 8 elements of T
  ordered by (D[o, j], canonical index), each with weight 1/8. Held-out nodes send no links to training
  nodes or to other held-out nodes. The augmented matrix over (T, O) is W\* = [[W_TT, 0], [W_OT, 0]].
  Because the training block of W\* equals the estimation graph, the training likelihood is unchanged by
  the held-out nodes.
* **Size.** Every required fit must have |T| ≥ 9; otherwise the fit fails structurally (§2.8).
* **Full sample.** W = kNN8(all 151). This station matrix is the V1 Moran matrix.

### 2.3 Spatial error model (SEM)

```
y = X β + u,    u = λ W u + ε,    ε ~ N(0, σ² I_n)
```

### 2.4 Spatial lag model (SAR)

```
y = ρ W y + X β + ε,    ε ~ N(0, σ² I_n)
```

θ denotes λ (SEM) or ρ (SAR).

### 2.5 Estimator: Gaussian maximum likelihood by an explicit in-repo algorithm

No spatial-econometrics library is added; the algorithm is implemented in
`research/model_v2/m3_spatial.py` with NumPy only, under the `uv.lock`-pinned NumPy.

**Parameter domain.** Θ = [−0.99, 0.99]. For a row-standardised W the spectral radius is 1, so
I − θW is non-singular on the open interval (−1, 1); the closed subset keeps I − θW well conditioned. The
estimate is the **constrained** maximum-likelihood estimate over Θ, following the established convention of
bounding spatial ML to (−1, 1) (PySAL `spreg` `ML_Lag` and `ML_Error`, bounded scalar search).

**Concentrated log-likelihood at θ.** M(θ) = I_n − θW; `(sign, logdet) = numpy.linalg.slogdet(M(θ))`.

| Family | Transformed regression | β̂(θ) | e(θ) |
|---|---|---|---|
| SEM | X̃ = M X, ỹ = M y | `numpy.linalg.lstsq(X̃, ỹ, rcond=None)` | ỹ − X̃ β̂ |
| SAR | ỹ = M y | `numpy.linalg.lstsq(X, ỹ, rcond=None)` | ỹ − X β̂ |

σ̂²(θ) = e·e / n, and

```
ℓc(θ) = −(n/2)(ln 2π + 1) − (n/2) ln σ̂²(θ) + logdet.
```

**An evaluation fails** if sign ≠ +1, logdet is non-finite, the rank returned by that `lstsq` call ≠ p,
σ̂² is not finite and positive, or ℓc is non-finite.

**Maximization (deterministic, fixed iteration counts).**
1. Grid G = `numpy.linspace(-0.99, 0.99, 199)`; evaluate ℓc at every grid point in order. Any failed
   evaluation fails the fit.
2. k\* = the first index attaining the maximum grid value. Bracket a = G[max(k\* − 1, 0)],
   b = G[min(k\* + 1, 198)].
3. Golden-section search with φ = (√5 − 1)/2 (`(math.sqrt(5.0) - 1.0) / 2.0`):
   c = b − φ(b − a), d = a + φ(b − a), evaluate c then d; then exactly 60 iterations of: if
   ℓc(c) ≥ ℓc(d) then (b, d) ← (d, c), c ← b − φ(b − a), evaluate c; else (a, c) ← (c, d),
   d ← a + φ(b − a), evaluate d. Any failed evaluation fails the fit.
4. θ̂ = the evaluated point (grid or golden) with the largest ℓc; ties go to the earliest evaluation.
5. **Boundary estimates.** If |θ̂| > 0.99 − 1e-6 the estimate is valid, flagged `at_domain_bound`, counted
   and reported; no alternative domain is tried, and its asymptotic interval is not applicable (below).
   *Why not a failure* (decided 2026-09-16 before any M2 or M3 fit, from synthetic evidence only,
   `M3_FEASIBILITY_AUDIT.md`): with the frozen static designs, whose continent and climate dummies absorb
   spatially smooth error, the constrained SEM estimate lies on the lower bound in 14 of 100 (M0\* design)
   and 13 of 100 (M2 design) synthetic draws generated with no spatial dependence. Treating a constrained boundary estimate as a structural failure
   would make a family non-computable because of the conventional bound rather than any failure of
   estimation.
6. β̂ = β̂(θ̂), σ̂² = σ̂²(θ̂).

**Recorded per fit:** θ̂, ℓc(θ̂) (optimization evidence only), β̂ with column names, σ̂², n, p, the number
of strict local maxima of the grid sequence (endpoints compared one-sided), and for the full-sample fit the
whole grid of ℓc values. **The maximized log-likelihood is never used as, or compared as, a model
criterion** (no AIC, BIC, likelihood ratio or pseudo-R² ranking), across or within families.

**Model-based parameter uncertainty (full-sample fit, reported, never a gate).** Expected information at
(β̂, σ̂², θ̂) (Anselin 1988, ch. 6):
* SEM, with B = M(λ̂), W_B = W B⁻¹: I_ββ = X′B′BX/σ̂², I_βσ² = 0, I_βλ = 0, I_σ²σ² = n/(2σ̂⁴),
  I_σ²λ = tr(W_B)/σ̂², I_λλ = tr(W_B W_B) + tr(W_B′W_B).
* SAR, with A = M(ρ̂), W_A = W A⁻¹, m = W_A X β̂: I_ββ = X′X/σ̂², I_βρ = X′m/σ̂², I_βσ² = 0,
  I_σ²σ² = n/(2σ̂⁴), I_σ²ρ = tr(W_A)/σ̂², I_ρρ = tr(W_A W_A) + tr(W_A′W_A) + m′m/σ̂².

For an interior estimate, se(θ̂) = √[(I⁻¹)_θθ] of the full (p + 2) × (p + 2) matrix, inverted with
`numpy.linalg.inv`; Wald
interval θ̂ ± 1.959963984540054 · se. For a boundary estimate, or if `numpy.linalg.inv` raises, or the
variance is non-finite or ≤ 0, se and the interval are recorded as `not computable`; the fit stays valid. This interval is asymptotic, conditional on W and on
the Gaussian model; it is not spatially robust and not design-based.

### 2.6 Fitted values, residuals and innovations

For a fitted (θ̂, β̂) on rows with design X, outcome y and graph W:

| Quantity | SEM | SAR | Meaning |
|---|---|---|---|
| trend t | X β̂ | M(ρ̂)⁻¹ X β̂ (`numpy.linalg.solve`) | the model's mean given the covariates only |
| structural residual | û = y − X β̂ | y − t | deviation from the trend; spatially dependent by construction |
| one-step fitted value ŷ | X β̂ + λ̂ W û | ρ̂ W y + X β̂ | conditional prediction of each node from its neighbours' observed values (zero diagonal: never its own) |
| innovation ε̂ | y − ŷ = M(λ̂) û | y − ŷ = M(ρ̂) y − X β̂ | the model's in-sample one-step residual |

Checks per fit: the two expressions for ε̂ agree to 1e-10; mean(ε̂²) equals σ̂² to a relative 1e-10.

### 2.7 Held-out prediction

For held-out node o in a fit with training set T, training estimates (θ̂, β̂), training structural
residuals û_T = y_T − X_T β̂, and attachment weights w_o· (§2.2):

```
SEM:  ŷ_o = x_o′β̂ + a_o + λ̂ Σ_j w_oj û_j
SAR:  ŷ_o = x_o′β̂ + a_o + ρ̂ Σ_j w_oj y_j
```

* x_o is built with the fit's training encoder and, in branch B, the training latitude state.
* a_o is the inherited unseen-level adjustment applied to the spatial β̂ exactly as
  `cv.V1Design.predict` applies it: for each categorical feature whose held-out level is absent from the
  training levels, add the mean of the list [0, β̂ of each non-reference training level] (repeats kept).
* **Numerical order:** p = X_o β̂ + θ̂ · Σ_j w_oj s_j (s = û_T or y_T), then each categorical's adjustment is
  added in encoder order.
* Under W\*, u_T (SEM) and y_T (SAR) depend only on ε_T, and ε_o is independent of them, so
  x_o′β̂ + θ̂ Σ_j w_oj s_j is the exact conditional expectation E[y_o | y_T, X] of the augmented model given the
  estimates. Where a_o ≠ 0 the prediction additionally carries the inherited protocol rule for a level the
  training fit never saw.
* **Only training outcomes enter.** Held-out outcomes and buffer-excluded outcomes enter no estimate, lag,
  residual or prediction. Tests must perturb held-out and excluded outcomes and show bit-identical
  predictions.
* No quantity is tuned against a CV score: k = 8 is fixed a priori and θ is estimated inside each training
  fit by §2.5.

### 2.8 Protocols, required fits and failure rules

**Required fits per family, weights arm and representation:** the full sample; the 151 primary 500 km
footprint-certified LOCO training fits (memberships identical to the frozen corrected memberships); the 20
M49 leave-one-subregion-out fits; the 10 random-reference fits (seed 0). Protocol definitions, memberships
and M49/random identities are verified exactly as in the M2 evaluator.

**Structural non-computability (a scientific disposition, not an integrity failure).** A family under a
weights arm is **non-computable** when, in any required fit of the primary representation: rank(X) ≠ p;
|T| < 9; any likelihood evaluation fails (§2.5); β̂, σ̂², a trend solve or any prediction is non-finite. A
boundary estimate (§2.5 step 5) is not a failure. The first failing fit and reason are recorded. **No fallback**: no other domain,
optimizer, k, weights, estimator, dropped fit, regularization or imputation. A non-computable family does
not qualify (§2.10). Its accounting (§2.11) is still reported if and only if its full-sample fit is valid.

**Integrity failures (the line stops; no qualification labels):**
1. any input pin, code-path, push-state, output-root or `PYTHONHASHSEED` refusal (before any spatial fit
   these write nothing and may be retried from identical frozen bytes);
2. the committed M2 result fails its manifest or the branch cannot be read;
3. S's recomputed predictions (in-sample and all three protocols) differ from the committed M2 result by
   more than 1e-10, or S's committed card metrics are not reproduced to 1e-10;
4. weight-matrix digests or fold identities differ from the frozen records;
5. a family is non-computable in one representation but not the other, or the representation identity
   fails: θ̂ (full and every primary training fit) differs by more than **1e-5**, or predictions,
   non-allocation scores or accounting parts by more than **1e-6** (Amendment A1 below);
6. the accounting identity (§2.11) fails by more than 1e-9;
7. any gate input is non-finite.

### 2.9 Reported per family (both weights arms, both representations)

* **Scores** (inherited definitions): in-sample **one-step** R² = 1 − Σε̂²/TSS (labelled as such; it
  uses neighbours' observed outcomes and is not an OLS R²), trend R² = 1 − Σ(y − t)²/TSS; CV R², RMSE, MAE
  for primary, M49 and random; fold RMSE median/min/max and worst fold country; IQR of |OOF error|;
  calibration slope and intercept (OLS of y on the OOF prediction); per-M49-region RMSE, MAE and bias and
  each model's own worst eligible region (n ≥ 3); rows with an unseen level; mean training size; nearest
  training distance (territorial lower bound).
* **Residual structure:** Moran's I of the innovations ε̂ (in-sample) and of the primary OOF errors, V1
  station weights, 999 permutations, seed 0; the same statistics with land-centroid weights recorded; the
  Moran's I of the structural residual recorded.
* **Effective degrees of freedom:** rank(X) + 1 (`SCORECARD.md` metric 13: "parameter count plus the
  spatial parameter"); σ² is not counted.
* **Dependence parameter:** full-sample θ̂ with §2.5 se and Wald interval (or its not-applicable reason);
  over the 151 primary training fits the median, minimum, maximum, the count strictly above zero and the
  count at the domain bound; boundary counts for the M49 and random fits (descriptive; never a gate).
* **Spatial range:** **not applicable.** A fixed kNN8 graph has no estimated range. Reported instead: the
  full-sample graph's 1st- and 8th-neighbour great-circle distance (median, maximum), and per protocol the
  held-out attachment distances (1st and 8th neighbour: minimum, median, maximum).
* **Paired comparisons (primary protocol):** ΔRMSE against S with the frozen
  `run_territory_correction.paired_rmse_interval` (2,000 country resamples, seed 0, identical indices,
  percentile 2.5/97.5); M49 and random RMSE differences; worst(f) − worst(S); per-country change in
  |OOF error| with counts improved/worsened/tied. In branch B the same against M0\*, descriptive.
* **Overfitting diagnostics (never gates):** (i) one-step in-sample R² gain over S minus primary CV R² gain
  over S > 0.05; (ii) primary OOF RMSE worsens while in-sample RMSE (√mean ε̂²) improves; (iii) θ̂ same-sign
  fraction over the 151 primary training fits, reported without a cutoff (as for γ,
  `M2_PRE_SCORE_RESOLUTIONS.md` §5); (iv) effective df rise (= 1) > 5 with an RMSE gain < 0.002 —
  evaluated, never triggered by construction. The summary Boolean uses (i), (ii) and (iv).

### 2.10 Qualification: retention for prediction (station weights, primary representation)

Per family f, with d_f = RMSE_primary(f) − RMSE_primary(S) and [lo_f, hi_f] its frozen paired interval:

| # | Condition | Comparison | Source |
|---|---|---|---|
| Q1 | d_f < 0 | strict | plan "Genuine improvement" |
| Q2 | hi_f < 0 | strict (with Q1, the interval excludes zero) | plan "Genuine improvement" |
| Q3 | M49 RMSE(f) − M49 RMSE(S) ≤ +0.001 | inclusive | plan secondary veto; `SCORECARD.md` metric 11 |
| Q4 | worst(f) − worst(S) ≤ +0.001, each model's own worst eligible M49 region (n ≥ 3) | inclusive | `SCORECARD.md` metric 12 veto as operationalized in `e3e5901` §4.1, applied unchanged in M1b and M2 |
| Q5 | computable in every required fit (§2.8) | — | structural |

**f qualifies iff Q1–Q5 hold.** Families are judged independently. Not applied: the 0.002 practical
threshold (it belongs to the M1a–M2 static stopping rule and has no independent M3 source); any sign
criterion on θ or β; any likelihood criterion; any overfitting veto. `worsened_generalization` is reported
(d_f > 0 and lo_f > 0). The per-capita representation emits the same Booleans descriptively.

### 2.11 Accounting (full sample; reported for every computable full-sample fit, qualifying or not)

**Estimands.** With TSS = Σ(y − ȳ)², RSS_S = Σ(y − X β̂_OLS)² from S's full-sample OLS, and
RSS_ε = Σ ε̂² from the family's full-sample fit:

| Part | Estimand | Meaning |
|---|---|---|
| 1. static component | A_static = 1 − RSS_S/TSS | S's in-sample R²; **exactly the sum of S's four OLS LMG shares** (checked to 1e-9). S's LMG shares are the descriptive grouping of this part and are not recomputed. |
| 2. represented spatial dependence | A_dependence = (RSS_S − RSS_ε)/TSS | in-sample reduction in squared one-step error when S's OLS mean is replaced by the family's ML fit and its neighbour-conditional predictor; it includes the change from β̂_OLS to β̂_ML and is not separable additively |
| 3. remaining innovation | A_innovation = RSS_ε/TSS | squared in-sample innovations as a fraction of TSS |
| 4. held-out performance | §2.9 primary/M49/random scores | the only evidence about transfer |

**Identity:** A_static + A_dependence + A_innovation = 1 (checked to 1e-9). Also reported:
`absorbed_fraction_of_static_residual` = (RSS_S − RSS_ε)/RSS_S, the trend R², and the innovation Moran's I.

**What these are not.** A_dependence is not an LMG share, not a variance component of y, not a
likelihood improvement or pseudo-R² ranking, and not a mechanism. It is never added to, folded into or
compared as a share with geography or any named group. In-sample fit improvement from a spatial term is
not evidence of transferable structure; part 4 is.

**Why no Shapley "spatial share".** A coalition game mixing OLS coalitions (mean predictors) with ML spatial
coalitions (neighbour-conditional one-step predictors whose estimator does not minimize squared error)
does not give an outcome-variance allocation comparable with OLS LMG; forcing one would count spatial
dependence on a different scale from the groups. `OPEN_DECISIONS.md` item 10's Session 1 proposal is
therefore resolved by this transparent multi-part accounting (owner-permitted disposition).

**Group contributions of the non-spatial part** (the plan's "whether the group contributions of the
non-spatial part change materially"). With θ̂ fixed at the full-sample estimate and, in branch B, the
full-sample latitude state:
* filtered outcome y\* = M(θ̂) y;
* SEM coalition design: intercept + M(λ̂) X_g for the coalition's groups g (M(λ̂)·1 = (1 − λ̂)·1, so the
  intercept is unchanged); SAR coalition design: intercept + X_g;
* coalition R²\* = 1 − RSS/TSS\* from `numpy.linalg.lstsq(rcond=None)`, TSS\* = Σ(y\* − ȳ\*)²; 16
  coalitions, with each coalition's columns stacked in the fixed group order (emissions, geography,
  socioeconomic, population); Shapley weights as `src.decomposition`; filtered shares sum to R²\*_full and
  shares + residual\* = 1 to 1e-9, with residual\* = 1 − R²\*_full;
* the full coalition's fitted values must equal the ML fit's transformed fitted values to 1e-10 (it
  reproduces β̂_ML up to the intercept scaling);
* **reported as fractions of the filtered outcome's variance, a different denominator from S's shares.**
  They are never summed with, subtracted from or substituted for S's shares or the accounting parts.
* **Material change of the non-spatial part** (plan definition applied to these shares): geography is not
  strictly the largest named filtered share, or the emissions filtered share exceeds 0.10. The composition
  (share ÷ R²) of S and of the filtered decomposition is reported side by side.

### 2.12 Allocation and interpretation rules

* Static LMG shares remain the descriptive grouping of S; A_dependence and A_innovation sit outside it.
* A fall in residual Moran's I or a rise in in-sample fit from a spatial term accounts for spatial
  covariance; it does not explain a mechanism and is never described as one (`af97bd2` interpretation
  rule).
* Qualification is retention for prediction only. It does not reopen S, the groups or any M1/M2 decision.

### 2.13 Land-centroid kNN8 weight sensitivity (historically required accounting; always run)

After the station-weight M3 package is committed and pushed, both families are refitted with land-centroid
distances in §2.2 (graphs and held-out attachments), everything else identical: folds and territorial
buffers, S, representations, optimizer, failure rules, scores, accounting. Moran's I is still reported
with the V1 station weights (the scorecard statistic), with land-centroid weights recorded. Q1–Q5 are
emitted under `arm_conditions`, explicitly descriptive: this arm cannot change the qualifying set, the
final naming or any primary decision, and the better geometry is never selected. A structural failure is
recorded as non-computable, without fallback.

### 2.14 Final model naming (frozen for every branch)

Names: `M3-SEM(M0*)`, `M3-SAR(M0*)` in branch A; `M3-SEM(M2)`, `M3-SAR(M2)` in branch B (station-centroid
kNN8). Three separate machine-readable fields are always written:

| Qualifying station-weight spatial extensions | `retained_static_specification` | `qualifying_spatial_extensions` | `final_primary_predictive_model` |
|---|---|---|---|
| none (including non-computable families) | S | `[]` | **S** |
| exactly one | S | `[that family]` | **that family** |
| both | S | `[SEM, SAR]` (listed in that fixed order, not a ranking) | **`null`**, with reason "two qualifying spatial extensions; no historical unique-family selection rule" |

**Why "exactly one" is designated.** The roadmap explicitly licenses it: `af97bd2` M3, "A spatial term is
**retained for prediction** only under the improvement rule", and M4, which assesses "retained stages"
and "the retained final model". A single family satisfying the improvement rule is the unique model retained
for prediction at the last stage. The plan's "M3 is an accounting step, not a model-selection step" explains
why M3 always runs; it does not withdraw the explicit retention clause. When both qualify there is no
historical rule to choose, so no winner is chosen: a non-unique result is a valid conclusion. In every
case the descriptive decomposition conclusion rests on S's OLS LMG shares (§3.2), because no historically
defined share exists for a spatial model.

## 3. M4: final spatial out-of-sample assessment (changes no model)

### 3.1 Evidence inventory and reuse rules

`M4_EVIDENCE_INVENTORY.md` lists every requirement. **Reuse rule:** committed evidence is reused only for
an identical estimand: same sample and order, outcome values, estimator, representation, group definition,
protocol memberships, weights and method. Reuse is by SHA-256 of the committed artifact; values are copied,
not recomputed, unless §3.4–§3.5 state a recomputation with an anchor check. Original rank-deficient V1
intervals are M0's and never stand in for M0\* or M2. Defective legacy 1° CV rows stay in a separate
`legacy` block; V1 allocations stay separate from V2 full-rank allocations.

### 3.2 Static share uncertainty for S (both representations)

The V1 procedures of `src/stability.py`, applied to S's exact scoring design:
* **Country bootstrap:** `rng = numpy.random.default_rng(0)`; 2,000 draws, each
  `idx = rng.integers(0, 151, 151)`, rows taken in drawn order.
* **Continent block bootstrap:** `rng = numpy.random.default_rng(1)`;
  `blocks = pandas.unique(spatial_block labels in canonical order)`; per draw
  `chosen = rng.choice(blocks, size=len(blocks), replace=True)` and the rows of each chosen block, in
  canonical order within the block, concatenated in chosen order; 2,000 draws.
* **Per draw:** categorical dummies rebuilt from the draw's own levels (sorted, drop-first, as
  `src.decomposition.feature_block`); exact numerics; in branch B a new `LatitudeState` from the draw's
  `abs_latitude` (ties permitted) applied to the draw; the 16-coalition projection R² LMG of
  `src.decomposition` (lstsq, `rcond=None`).
* **Degenerate draw:** `DegenerateLatitudeBasis` (branch B), a `ValueError` or `LinAlgError`, or any
  non-finite share. It is counted and skipped, never replaced. Usable draws with a rank-deficient full
  design are kept (projection R² is defined, as in V1) and counted separately.
* **Reported per group and residual:** point (full-sample LMG), mean, sample SD, 2.5/97.5 percentiles
  (`numpy.percentile`, linear) for both bootstraps; `p_geography_largest` as V1 (named order emissions,
  geography, socioeconomic, population; ties to the earlier group); usable, degenerate and
  rank-deficient-usable counts.
* **Material change** (plan definition), per representation, reported as it falls:
  * `point_geography_not_largest`: geography's point share is not strictly larger than every other named
    point share;
  * `interval_geography_not_established` (country bootstrap; the block-bootstrap version is reported
    alongside): the 2.5th percentile over usable draws of D_b = geography_b − max(other named)_b is ≤ 0;
  * `responsibility_exceeds_0_10`: emissions point share > 0.10 (its 97.5th percentile > 0.10 reported
    descriptively);
  * `material_change_to_v1_conclusion` = any of the three (country bootstrap).
* If branch B: the bootstrap applies to M2 only. M0\* is then the comparator, not the retained
  specification; it receives no bootstrap.

### 3.3 Spatial uncertainty

* **Applicable:** the §2.5 model-based se and Wald interval of θ̂ for each computable family, and the
  descriptive distribution of θ̂ over the 151 primary training fits.
* **Not applicable / not historically defined:** a country or block bootstrap of the accounting parts or of
  filtered shares. The historical bootstrap is defined for "the shares of the retained stages" (OLS LMG);
  resampling duplicates nodes in a kNN graph, which has no defined construction, and none is invented.
* The paired ΔRMSE intervals of §2.9 are the predictive uncertainty (country-resampling, not spatially
  corrected).

### 3.4 Buffer sensitivities (reporting only; never criteria)

Protocols: corrected territorial buffer **1000 km** (the `4263429` lower bounds, training excludes
distance ≤ 1000) and land-centroid buffer **1500 km** (`spatial.haversine_matrix` of the
`m0_countries.csv` `centroid_lon`/`centroid_lat`, excluding ≤ 1500), both leave-one-country-out, as in
`run_territory_correction.py`.

Computed for **retained specifications**: S; each qualifying station-weight spatial extension; and, in
branch B, M0\* as S's comparator (the protocol purpose is to show whether "a model ranking" survives).
Both representations. Reported: CV R², RMSE, MAE; paired ΔRMSE with the frozen interval for (S − M0\*)
in branch B and (extension − S). Anchor: M0\*'s sensitivity CV R², RMSE and MAE (computed in either
branch whenever M0\* is in scope) must match the committed M0 sensitivity cards in
`m0_scorecard_territory_corrected.json` to 1e-9 (same column space in every training fit).
Structural failure in a sensitivity fit: recorded non-computable for that cell, no fallback.
Non-qualifying families and rejected M1a/M1b/M2 candidates: **not executed** (not retained).

### 3.5 Product check of the eventual model

**Requirement:** `9fb56e1` ("A post-freeze ERA5 check of the eventual model remains necessary"; §7.3 "using
both constructions"). **Excluded:** the two-product-mean training outcome (owner, 2026-09-16); ERA5 never
selects anything.

* **Outcomes.** Aligned ERA5 (**preferred** for interpretation):
  `outputs/product_stability_aligned_era5_trends.csv`, column `trend_c_per_decade_era5_area`, SHA-256
  `23c96aea…`. Legacy ERA5 (**preprocessing sensitivity**, required by §7.3): `app/data/era5_area_trends.parquet`,
  column `trend_c_per_decade_era5_area`, SHA-256 `5a18ccba…`. Both mapped by ISO3 to the canonical 151; all
  must be finite, else the product arm is non-computable.
* **Eventual models:** S and each qualifying station-weight spatial extension. Both representations.
* **Per product × model**, identical predictors, folds, states (the predictors do not change) and graphs:
  in-sample R² (one-step and trend R² for spatial), S's LMG shares with the §3.2 point materiality flags,
  primary/M49/random CV R² and RMSE, residual Moran's I in-sample and OOF, Pearson and Spearman correlation
  and sign agreement between the product's in-sample residuals (innovations for spatial) and Berkeley's.
  For a spatial extension also θ̂, the §2.11 accounting parts, paired ΔRMSE against S under the same
  product with the frozen interval, and Q1–Q5 as descriptive `arm_conditions`; `product_sensitive` is set
  when the extension qualifies on Berkeley but Q1–Q5 do not all hold under aligned ERA5.
* **Anchors.** S's in-sample R² must equal the M0.5 summaries' `r2.era5` (aligned
  0.4653656626320404, legacy 0.5003406042520144) to 1e-10 in branch A (same column space as M0). In branch B,
  the aligned M2 card is additionally checked to 1e-10 against the committed M2 conditional aligned-ERA5 arm,
  which remains a separate record.
* **Not executed:** product refits of rejected M1a/M1b/M2 candidates or non-qualifying spatial families; a
  legacy-ERA5 M2-versus-M0\* paired replication (M2's conditional product replication is frozen aligned-only
  by `M2_EVALUATION_CONTRACT.md` §10 and is not reopened); bootstraps under ERA5.
* **Interpretation.** Product disagreement is reported as such; it is not an identified amount or cause of
  observational error, and it does not "bound" an observational share.

### 3.6 Measurement sensitivity

* Branch A: reuse the committed `m1a_scorecard.json` (C0 = M0\* to 1e-10, and M1a), both representations.
* Branch B: reuse the committed M2 conditional M1a-geography arm (or its recorded non-computable disposition).
* Spatial extensions: **not executed.** M3's registered alternative geography concerns spatial weights
  (§2.13); no M3 geography remeasurement is registered.

### 3.7 Consolidated table

One machine-readable table and one prose table, from committed evidence plus §3.2–§3.6: corrected M0 (and
the legacy 1° row in a separate block), M0\* (both representations), M1a, M1b, M2, and both M3 families
under both weights. Per row: status (frozen baseline / approved baseline / not promoted / not supported /
promoted / qualifying / non-qualifying / non-computable / not executed), primary, secondary and reference
scores, paired ΔRMSE and interval against its registered comparator, worst eligible region, OOF Moran's I,
the registered allocation quantities (OLS LMG shares; for spatial rows the accounting parts and filtered
shares), and each missing cell's disposition. **The three M4 numbers per retained stage:** primary RMSE
relative to M0\*, OOF Moran's I, and the geography share with its §3.2 interval (for a spatial extension:
the filtered geography share, interval not applicable).

### 3.8 Stopping rule and residual description

* The static stopping indicator (§1) is reported as it fell. If it fires, M4 states that the residual is not
  recoverable **with the tested country-level static specifications** (not with all possible covariates);
  M3 then quantifies its spatial covariance, which M4 reports as the finding.
* The residual share of S (and the innovation part of any spatial extension) is described as variance in
  cross-country warming differences not captured at country resolution by the tested models with these
  observations, never as an unknown climate cause.

### 3.9 Scientific completion criteria

M4 is complete when, and only when:
1. every row of `M4_EVIDENCE_INVENTORY.md` ends in verified evidence or its listed permitted disposition;
2. the final record `m4_final/m4_final_specification.json` carries `retained_static_specification`,
   `qualifying_spatial_extensions` and `final_primary_predictive_model` per §2.14, the M2 label, the static
   stopping indicator, each M3 family's disposition and the exact spec, evaluator and result commits;
3. its numbers are independently reconstructed (§4) and one unchanged-code reproducibility run matches;
4. prose, scorecards and the machine-readable record agree on model identities, verdicts and limitations.

## 4. Execution controls for M3 and M4

* **Freeze before first real fit.** Each evaluator (`m3_evaluate.py`, `m4_evaluate.py`) with its tests is
  committed and pushed before its first outcome-facing run, and its method must match this file.
* **Guard.** Refuse, before reading any outcome, unless: `PYTHONHASHSEED=0`; every file of the evaluator's
  **transitive local import closure** (computed from the AST at run time over `research.*` and `src.*`),
  this specification, the governing contracts, `pyproject.toml` and `uv.lock` is tracked and identical to
  `HEAD`; `HEAD` is contained in the **live** remote branch head (`git ls-remote`, not only the cached
  tracking ref); the output root does not exist; the prerequisite result packages verify against their
  committed manifests and are committed and pushed.
* **Prerequisites.** M3 station arm: committed M2 primary result and committed M2 conditional disposition.
  M3 land-centroid arm: committed M3 station package. M4: committed M3 station and land-centroid packages.
* **Outputs.** Deterministic JSON (`json.dumps(indent=2, allow_nan=False)` + newline), CSV floats written
  with round-trip `repr`; a result manifest with SHA-256 of every deterministic artifact; a separate
  non-deterministic run-metadata file; provenance with code-closure digests, input pins, fold and weight
  identities, software versions and platform.
* **Verification per result package:** an independent numerical reconstruction (separate code path: SEM/SAR
  likelihood through eigenvalues of W instead of `slogdet`, a local fine-grid optimality check of θ̂,
  predictions rebuilt from saved β̂, θ̂, neighbour lists and designs, metrics, intervals, gates, accounting;
  M4 bootstrap shares through QR instead of lstsq) with recorded negative controls for the critical checks;
  one unchanged-code reproducibility run into a fresh root, comparing every deterministic artifact byte for
  byte and listing permitted metadata differences. Scores are platform-scoped in their last bits.
* **After a result exists** no method in this file changes. A reproducible post-result defect with possible
  numerical consequences stops the line.

## Amendment A1 (2026-09-16, before any M2 fit and before any M3 fit): representation-identity tolerances

**Original text (§2.8 item 5, as committed at `4dc2fad`):** "predictions > 1e-9, θ̂ (full and every primary
training fit) > 1e-8, non-allocation scores and accounting parts > 1e-9".

**Why it changed.** Those values were carried over from the OLS stages, where both representations solve the
same projection and agree to rounding. An ML estimate is the argmax of a smooth concentrated likelihood; a
perturbation ε in ℓc from rounding (the two representations parameterize the same column space differently)
moves the argmax by about √(2ε / c), with c = −ℓc″(θ̂). With |ℓc| of order 10², ε ≈ 1e-13 and c ≈ 25
(se ≈ 0.2), the attainable agreement is about 1e-7 in θ̂ and about 1e-8 in predictions. On synthetic
identity data (n = 60, the M2 evaluator's synthetic frame), before any real fit, the per-capita representation
reproduced θ̂ to 1.35e-7 (SEM) and 7.4e-8 (SAR) and out-of-fold predictions to 3.3e-9. So 1e-8 cannot be met
even by a correct implementation.

**Replacement.** θ̂ 1e-5 and predictions, non-allocation scores and accounting parts 1e-6: about 50 times the
derived resolution, and far below any difference a representation-specific defect (a wrong column, group or
graph) would produce. The OLS tolerances of the static stages are unchanged. This is an operational tolerance
set from synthetic evidence before any M2 or M3 number existed; no gate, estimand or rule changes.

## 5. Not pursued (not experimentally rejected)

Geostatistical residual model; alternative k, weights, estimators or domains; spatial Durbin or combined
SAC models; spatial terms on M1a/M1b; spatial-model bootstraps; C1, C3 and alternative latitude forms;
the H7 two-product-mean outcome; H8–H10 diagnostics.
