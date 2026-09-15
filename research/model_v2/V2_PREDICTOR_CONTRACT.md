# V2 predictor contract — approved

**Status: approved by the project owner on 2026-09-15** (after the M0.5 review,
commit `9fb56e1`). This contract applies to Model V2 research only. V1 and its
original decomposition are unchanged; V1 shares are never restated under this
parameterization. Scientific rationale and full sensitivity scorecards:
[RANK_DEFICIENCY_AUDIT.md](RANK_DEFICIENCY_AUDIT.md).

| Group | Primary (approved) | Registered representation sensitivity (approved) |
|---|---|---|
| Historical responsibility | log10 cumulative production CO₂ total, Mt through 2013 (`cum_co2_total`) | log10 cumulative production CO₂ per 2013 resident, t/person (`cum_co2_per_capita`) |
| Population | log10 population 2013; existing `station_density` | Same |
| Geography | Existing six M0 features; approved M1a measurements when that stage is evaluated | Same |
| Socioeconomic | Existing income-group categorical block | Same |

The primary design drops `cum_co2_per_capita`; the sensitivity drops
`cum_co2_total`. Both have 20 columns and rank 20 on the 151 complete cases.
Both give the same M0 projection and the same corrected CV predictions.

## Approved rationale

The rationale is conceptual, not performance-based:

* total cumulative national emissions is the cleaner primary representation of
  historical national responsibility;
* population remains explicitly represented in its declared group;
* the exact cross-group rank deficiency is removed;
* no information needed by the full model is lost;
* the per-capita formulation is a legitimate alternative conception of
  responsibility, so it is the registered sensitivity.

Neither representation was selected for its Shapley share, coefficient sign,
CV score or preferred scientific conclusion.

## Rules from this point forward

* Every Model V2 comparison uses the primary contract. The per-capita contract is
  reported alongside it as the registered representation sensitivity wherever
  shares or scores are reported.
* The representation that gives a better result is never promoted to primary.
* Stage comparisons are like with like: a changed stage is compared with the
  rank-only M0 equivalent under the **same** representation. Share changes caused
  by dropping a redundant log column are never credited to a later stage.
* No feature-group move, new physical concept, nonlinear term or categorical
  regrouping is part of this contract.
* Frozen elsewhere and unchanged:
  * 500 km GPW/ISO footprint-certified LOCO, M49 secondary, random CV as
    reference only;
  * train-only encoders and the mean-training-level-effect rule;
  * the 151-country sample and the area-weighted Berkeley outcome.
* Responsibility shares remain descriptive explained variance in country
  outcomes. They are never global climate-forcing or causal-attribution shares.
