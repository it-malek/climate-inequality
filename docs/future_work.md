# Open questions

Items are ordered roughly by value for effort. None requires restructuring the
pipeline; each extends it.

## Data

- **ERA5 cell-level disagreement.** The Berkeley and ERA5 area-weighted country
  rankings agree only at ρ = 0.62. Mapping where the two products disagree at
  the cell level would locate station inhomogeneities and reanalysis biases;
  the per-cell slopes already exist in memory during the ERA5 cross-check.
- **Extreme heat, not mean warming.** Monthly means hide heat extremes, and
  tropical countries sit closer to physiological thresholds. GHCN-Daily would
  support an extreme-heat-days outcome; it is large and not in the repository,
  so storage and download need scoping first.
- **Urban heat island.** These are city series. Comparing against Berkeley
  Earth's rural-only flagged stations would bound how much of the
  0.146 °C/decade station mean is urban rather than climatic.
- **Measurement uncertainty.** The `AverageTemperatureUncertainty` column is
  unused. Weighted fits, or a check that early-window noise does not bias the
  Theil–Sen slopes, would close a documented limitation.

## Method

- **The unexplained third.** About 36% of the cross-country variance in
  area-weighted warming lies outside the decomposition and its residual is
  spatially clustered (Moran's I 0.27). Whether that is recoverable physical
  structure (aridity, precipitation regime, snow cover, albedo, land cover),
  model form (nonlinearity, interactions), spatial dependence to be modelled
  rather than absorbed, or irreducible at country resolution is the next
  research phase, to be judged by spatially held-out validation rather than
  in-sample R².
- **Feature taxonomy.** Station density sits in the population group but
  partly encodes observational sampling structure; a schema revision that
  treats it as a nuisance term would be defensible under either outcome.
- **Construction sensitivity for the decomposition.** Median instead of mean
  aggregation of city slopes, alternative cutoff years, total instead of
  per-capita emissions, and a second gridded temperature product as the
  outcome (the ERA5 cross-check exists only for the coupling comparator).
- **Spatially honest inference for the legacy coefficient.** Conley standard
  errors and a spline latitude control were prototyped on an abandoned branch
  but never shipped; the decomposition sidestepped the need, but a reader
  comparing with the single-coefficient literature would want them.
- **Era-weighted responsibility.** Cumulative emissions divided by the
  population of the emitting era rather than by 2013 population.
- **Within-country inequality.** The per-city trends allow a within- versus
  between-country variance split; which countries contain both fast- and
  slow-warming regions?
- **Kriging refinements.** Anisotropic variograms and a nugget fixed from
  measurement uncertainty would make the interpolation comparison fairer
  before IDW is declared the winner.
- **Rolling-window trends.** Piecewise or 30-year rolling Theil–Sen fits would
  map where warming is accelerating, not just where it is fast.

## Physical model

- Alternative forcing vintages and an explicit test of the fixed one-year lag.
- A two-box energy-balance variant as a separate artifact, if a physical
  simulator is ever wanted alongside the regression.
