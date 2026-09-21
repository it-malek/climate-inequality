# Limitations and open questions

The project is complete. This page states what its results rest on and the
scientific questions they leave open. None of the questions is a planned
extension; each would need its own design.

## What the results rest on

- **Country aggregation.** Every country is one unit. A country mean discards
  within-country variance and can invert relationships that hold at finer
  scales; seven large countries have more internal spread in cell trends than
  the decomposition residual.
- **Sample.** The decomposition uses the 151 countries with an area-weighted
  rate and complete features; the responsibility comparison uses 154 to 157.
  Every country carries equal weight.
- **Products.** Berkeley Earth interpolates across data-sparse interiors from
  the same stations that define the station-weighted rate; ERA5 has a
  distinct reanalysis and data-assimilation error structure. Their
  area-weighted national trends agree only at ρ about 0.6, and they disagree
  by more than the decomposition residual.
- **Window.** The trends end in September 2013 and describe a period slower
  than the present; warming after 2013 ran ahead of the fitted lines.
- **Uncertainty.** Trend uncertainty at the cell and station level and
  measurement error in the predictors are not propagated into the shares.
  Resampling intervals treat countries as exchangeable and are not spatially
  corrected; spatial dependence persists beyond the 500 km exclusion used for
  validation, and the spatial models' held-out predictions rely on the
  observed outcomes of neighbouring countries.
- **Hypothesis generation.** Several residual hypotheses were motivated by the
  same outcome data they were tested on, so a consistent result is not
  independent confirmation and a rejected specification does not disprove its
  mechanism.
- **No causal identification.** Shares are allocations of explained
  cross-country variance; correlations describe alignment; nothing estimates
  the effect of a country's emissions on its own climate.

## Open questions

- **A multi-product outcome.** A latent national warming estimate treating
  Berkeley Earth, ERA5 and other products as noisy measurements would separate
  product disagreement from climate signal before any decomposition, and would
  say whether the spatial models' Berkeley-specific gain is a property of one
  field or of the climate.
- **Time.** Rolling or piecewise decompositions would show whether the
  geography share and the residual's spatial pattern are stable across decades
  or are the imprint of particular variability phases; a diagnostic that
  regresses each country's cell-mean series on the Pacific and Atlantic decadal
  indices would bound how much of the residual is unforced variability.
- **Sub-national units.** The per-cell trends allow a within-country versus
  between-country split and a hierarchical model of cells within countries, at
  the cost of a different estimand.
- **Independently motivated residual covariates.** Regional aerosol
  histories, baseline snow-cover duration and synoptic-scale ocean exposure
  are physically motivated candidates that were recorded but not pursued; each
  would need its own declared group and a pre-specified sign, and aerosol
  histories in particular are correlated with the responsibility axis.
- **Physically meaningful spatial graphs.** A neighbour structure derived from
  circulation, coastline geometry or climate regime rather than centroid
  distance would make the spatial accounting interpretable.
- **Other definitions of responsibility.** Consumption-based accounting is
  provided for 114 countries and reorders responsibility only slightly;
  era-weighted populations, non-CO₂ gases and land-use emissions would change
  who is responsible and could change the alignment.
- **Impact outcomes.** Extreme-heat days, exposure-weighted degree days or
  damages would measure burden rather than mean warming, which understates the
  burden on tropical low emitters.
- **Measurement.** The Berkeley Earth uncertainty column is unused; weighted
  trend fits and a rural-station comparison would bound measurement and
  urban-heat effects on the station mean.
- **Detection and attribution.** Whether observed regional warming is
  consistent with the forced response is a distinct question, answered with
  climate-model fingerprints rather than cross-country regression.
