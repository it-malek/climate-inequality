# Proposed V2 predictor contract — approval pending

The rank audit recommends the following registered pair of research
representations. Neither changes V1 or implements M1a. Scientific rationale and
full sensitivity scorecards: [RANK_DEFICIENCY_AUDIT.md](RANK_DEFICIENCY_AUDIT.md).

| Group | Primary proposal | Registered sensitivity proposal |
|---|---|---|
| Responsibility | log10 cumulative production CO₂ total, Mt through 2013 | log10 cumulative production CO₂ per 2013 resident, t/person |
| Population | log10 population 2013; existing station-density feature | Same |
| Geography | Existing six M0 features; future approved M1a measurements only | Same |
| Socioeconomic | Existing income-group categorical block | Same |

Delete per-capita from the primary matrix; delete total from the sensitivity.
Both retain 20 columns/rank 20 on the present 151 complete cases, with identical
M0 projection and corrected CV predictions. Neither representation is selected
by CV performance or the size/sign of its group share. No feature-group move,
new physical concept, nonlinear term or categorical-level regrouping is allowed.

For M1a, compare changed geography measurements to the rank-only M0 equivalent
**using the same approved representation and group partition**. Report V1's
original decomposition separately: differences caused by changing the log
representation must not be credited to measurement alignment.

Use the frozen 500 km GPW/ISO footprint-certified LOCO, M49 secondary, random
reference only, train-only encoders and mean-training-level-effect rule. Preserve
the 151-country sample and original area-weighted Berkeley outcome. Report the
full frozen scorecard under both conventions if the owner approves this pair.
Responsibility shares remain descriptive explained country-outcome variance,
never global climate-forcing or causal-attribution shares.
