# Hypothesis register

## M0.5 review status

The entries below preserve the Session 1 hypotheses; registration is not approval
to test new predictors. The product audit in `PRODUCT_STABILITY_AUDIT.md` now
controls which residual patterns are eligible for later physical-hypothesis
generation. Product-sensitive/unresolved patterns need additional evidence;
none is a demonstrated mechanism. ERA5 is an external robustness outcome,
not a model-selection criterion. Repeat product robustness after V2 is frozen.
M1a's reviewed measurement contract is `M1A_MEASUREMENT_SPEC.md`; no implementation
or new predictor has been authorized in M0.5.

The table below gives the product status of the residual evidence cited in the
Session 1 entries. Classes come from
`outputs/product_stability_pattern_classification.csv`, where a pattern counts as
robust only if both ERA5 constructions agree. Patterns not listed there were not
classified and remain unresolved.

| Entry | Cited evidence → M0.5 product status |
|---|---|
| H1 | SH latitude gradient not directly classified (unresolved). Its ingredients split: Argentina extreme and Southern Africa region robust; New Zealand sensitive; Botswana and South Africa unresolved. |
| H3 | Central Asia and Western Africa regional patterns sensitive; Iran extreme sensitive; Turkmenistan and Mauritania unresolved; UAE HH unresolved. The Gulf's large positive residuals are ERA5-only (sensitive). |
| H4 | Levant LL clusters (Cyprus, Jordan, Lebanon) sensitive; Egypt, Libya, Israel, Mexico, Myanmar, Romania, Slovakia extremes sensitive; Central America and Eastern Europe regions sensitive; India unresolved; South Korea extreme robust. The cluster premise is not product-robust. |
| H5 | Canada and Estonia extremes robust; Latvia unresolved; Northern Europe region unresolved (preprocessing-dependent); Russia and Finland not classified. |
| H6 | Argentina extreme and Southern Africa region robust; New Zealand sensitive; South Africa and Botswana unresolved. |
| H7 | Superseded by the audit. Legacy numbers reproduce, but "LL clusters coincide with the largest disagreement" is too strong. Aligning preprocessing halves the mean bias without improving agreement. |

These statuses govern only which evidence may *motivate* a hypothesis. They do
not approve any H-entry for testing.
**M1b (2026-09-15):** H3, H5 and H6 are re-specified in `M1B_HYPOTHESIS_REGISTER.md` on independent physical rationale. Their Session 1 residual evidence was Berkeley-derived and is not relied on. H4 is not advanced.

**Closure status (2026-09-16, `V2_FINAL_REPORT.md` §5).**
* **H1 and H5:** tested only jointly, as M2, and **not supported**. γ was negative in all training fits, which
  is descriptive only.
* **H2:** not supported (M1a).
* **H3:** not supported, as C2 (M1b).
* **H4:** not advanced.
* **H6:** not pursued.
* **H7:** eventual-model product check run in M4; the spatial gains are product-sensitive.
* **H8–H10:** not pursued.

The historical entries below are unchanged.

The binding guardrail, which requires product-robust evidence and/or a strong
independent physical hypothesis before any new covariate and forbids post-hoc
searches against Berkeley residuals, is in `M1A_EVALUATION_CONTRACT.md` §6.

Hypotheses generated from the M0 residual (`M0_RESIDUAL_DIAGNOSTICS.md`), written
before any candidate variable was fetched or tested. Each entry states the
residual evidence, the mechanism, the candidate measurement, its timing, the
mediator risk, sources, expected form, collinearity with the existing geography
features, and whether it changes the meaning of a feature group. "Stage" is
where it belongs in the M0 → M4 sequence; "not pursued" means recorded and
deliberately left alone.

Expected directions are pre-stated so that a covariate can be rejected for the
wrong sign even when it improves fit.

---

## H1. The latitudinal gradient differs by hemisphere — stage M2 (functional form)

1. **Residual pattern.** SH residual falls with latitude (ρ = −0.68, p = 5×10⁻⁵,
   n = 29; SH tropics +0.015, SH subtropics/mid-latitudes −0.024); NH flat.
2. **Mechanism.** Over 1950–2013 land warming amplified toward the NH high
   latitudes (snow/ice–albedo, land-dominated hemisphere); SH mid-latitude land
   (30–50°S) is surrounded by a slowly warming, heat-absorbing Southern Ocean and
   warmed little. One slope for both hemispheres averages a positive NH gradient
   with a negative SH one.
3. **Candidate quantity.** No new data: the interaction `hemisphere × abs_latitude`
   (or separate latitude slopes per hemisphere).
4. **Timing.** Static.
5. **Mediator risk.** None.
6. **Sources.** None needed.
7. **Expected form.** NH slope positive (as now, ~+0.001 °C/decade per degree),
   SH slope negative or zero; possibly a convex NH shape (H5).
8. **Collinearity.** By construction with `hemisphere` and `abs_latitude`; the
   interaction is identified because 29 SH countries span 1–47°.
9. **Group meaning.** Unchanged; it stays inside geography.

## H2. Geography remeasured over terrestrial area — stage M1a, evaluated

**Result (`M1A_REPORT.md`):** not supported as a model improvement. Area-consistent remeasurement (under pre-result Amendment 1) did not improve transfer or reduce residual spatial autocorrelation; the group conclusion is unchanged.

1. **Verified evidence.** 26/151 station-modal versus area-dominant Köppen
   differences; 17 negative country-mean elevation values, not a complete
   independently masked bathymetry inventory; Canada's mean station |latitude|
   is 14.1872° below its absolute land-vector-centroid latitude. The latter is
   not yet the proposed mean absolute land latitude. See the measurement spec
   for definitions, country lists and within/between-country distinctions.
2. **Hypothesis.** Measuring existing geography on a support consistent with
   the area-weighted outcome may change residual structure. This does not
   uniquely identify station siting as a cause.
3. **Specification.** `M1A_MEASUREMENT_SPEC.md`: mean absolute land latitude,
   independently land-masked elevation retaining genuine negative dry land,
   area-mean coast distance, one area-dominant Köppen category, majority-area
   hemisphere; unchanged OWID continent category. No new physical concepts.
4. **Sources and timing.** Exact source versions, area integration, masks,
   coverage gates, terrain/coast caveats and contemporary climate-classification
   leakage limitations are in the specification. No new data fetched here.
5. **Comparison.** Use the same owner-approved full-rank contract in M0 and M1a,
   the frozen outcome/151 countries and corrected CV folds. No measurement
   gain has been estimated in this session.


## H3. Baseline aridity and land–atmosphere coupling — stage M1 (physical)

1. **Residual pattern.** High–high clusters in the Atlantic Sahel, the Gulf and
   Central Asia/Iran (Turkmenistan +0.062, Iran +0.063, Mauritania +0.051);
   arid (B) countries have the widest residual spread; \|e\| rises with the arid
   share (ρ = +0.19).
2. **Mechanism.** In soil-moisture-limited regimes evaporative cooling cannot
   increase with warming, so the surface warms more per unit forcing; drying
   trends amplify this (land–atmosphere feedback). Against it: the Levant, Egypt
   and Libya are arid and strongly over-predicted, so aridity alone cannot sort
   the residual; the hypothesis is that aridity interacts with something else
   (H4, H7) or acts through its baseline level rather than its class.
3. **Candidate quantity.** Baseline aridity index P/PET (climatology), or
   baseline annual precipitation; area-weighted over the land.
4. **Timing.** Before or at the start of the window (a 1951–1980 or earlier
   climatology), never a within-window trend.
5. **Mediator risk.** High if measured as a trend (drying is partly a
   consequence of warming); low for a baseline climatology.
6. **Sources.** CRU TS 4 (0.5°, 1901–), CGIAR-CSI Global Aridity Index (Zomer
   et al., WorldClim-based), TerraClimate; all gridded, area-averaging is the
   same operation as the outcome's.
7. **Expected form.** Positive effect of aridity on the warming rate, plausibly
   saturating in hyper-arid regimes (hence a candidate for M2 curvature).
8. **Collinearity.** Substantial with Köppen B, continentality and latitude
   20–35°; moderate with elevation.
9. **Group meaning.** Adds a "surface-climate regime" element to geography;
   acceptable if declared. It must not be read as climate attribution.

## H4. Regional aerosol forcing histories — stage M1, conditional (physical, guarded)

1. **Residual pattern.** The low–low clusters (eastern Mediterranean/Levant,
   south-eastern Europe, mainland South and South-east Asia, Central
   America/Mexico, South Korea) are regions of heavy anthropogenic aerosol
   loading over 1950–2013; the high–high Sahel/Central Asia clusters are not
   industrial aerosol regions.
2. **Mechanism.** Sulphate and carbonaceous aerosols cool the surface
   regionally; where emissions rose through most of the window (South Asia,
   Middle East, parts of East Asia and Central America) the 64-year trend is
   damped; where they peaked around 1980 and fell (Europe) the second half of
   the window carries brightening. Over a single linear slope the net effect
   depends on the history, not just the level.
3. **Candidate quantity.** Area-mean aerosol optical depth or SO₂/BC emission
   trajectory over the window, summarised as (a) the mean loading and (b) the
   change between the first and second halves.
4. **Timing.** During the window, legitimately: it is a forcing, not a response.
5. **Mediator risk.** Low physically; high interpretively (see 9).
6. **Sources.** CEDS gridded emissions (Hoesly et al. 2018), MACv2-SP aerosol
   climatology, MERRA-2 AOD (1980–), CMIP6 input4MIPs forcing fields.
7. **Expected form.** Negative relation between the rise in loading over the
   window and the trend; positive for a fall.
8. **Collinearity.** Strong with income, population and the emissions group
   (industrialisation co-emits CO₂ and aerosol precursors).
9. **Group meaning.** This is the one hypothesis that touches the V1 guardrail:
   an aerosol term is a regional forcing, not responsibility, but it is
   correlated with the responsibility axis, and adding it to any existing group
   would change what that group's share means. If admitted it must be its own
   declared group ("regional forcing"), with wording that keeps the
   responsibility result separate. Owner decision required.

## H5. Snow/ice–albedo amplification at NH high latitudes — stage M2 (curvature) or M1

1. **Residual pattern.** NH 50–70° band under-predicted (+0.013, n = 17;
   Canada +0.064, Estonia +0.045, Latvia +0.040, Russia +0.025, Finland +0.024).
2. **Mechanism.** Retreating seasonal snow cover lowers surface albedo; the
   feedback is concentrated at 50–70°N, making the latitudinal gradient convex
   rather than linear.
3. **Candidate quantity.** Either curvature in latitude (a pre-specified
   natural spline with fixed degrees of freedom) or a baseline snow-cover
   climatology (mean annual snow-cover duration), area-weighted.
4. **Timing.** Baseline climatology (1967–1990 NOAA/Rutgers, or ERA5 snow
   cover 1950–1980).
5. **Mediator risk.** High for snow-cover trends (a response); low for a
   baseline duration.
6. **Sources.** Rutgers Global Snow Lab (1966–), ERA5 snow depth/cover.
7. **Expected form.** Positive; steepening above ~50°N.
8. **Collinearity.** Very high with latitude and Köppen D/E; the spline version
   avoids a new variable.
9. **Group meaning.** Geography, unchanged.

## H6. Large-scale ocean influence on SH mid-latitude land — stage M1 (physical), paired with H1

1. **Residual pattern.** SH 25–50°S over-predicted (Argentina −0.052, Chile
   −0.032, New Zealand −0.046, South Africa −0.021, Botswana −0.056).
2. **Mechanism.** The Southern Ocean took up heat and warmed slowly over the
   window; land within its atmospheric reach warmed less. V1 continentality
   (distance to the nearest coast) measures exposure to *any* ocean at ~100 km
   scale, not exposure to a slowly warming ocean at synoptic scale.
3. **Candidate quantity.** Ocean fraction within 1000–2000 km of the land
   centroid ("oceanicity" at synoptic scale), optionally weighted by the
   surrounding SST trend (the latter is a during-window quantity).
4. **Timing.** Static for the ocean fraction; during-window for an SST-trend
   weighting (then not a baseline predictor).
5. **Mediator risk.** None for the fraction; the SST-trend variant is a
   co-response and belongs to a diagnostic, not to M1.
6. **Sources.** Land mask (in repository); HadISST or ERSST for the variant.
7. **Expected form.** Negative; possibly SH-specific (an interaction with
   hemisphere, hence paired with H1).
8. **Collinearity.** With hemisphere, coastal fraction and continentality.
9. **Group meaning.** Geography, unchanged.

## H7. Observational-product structure — robustness arm, not a predictor (M4, owner decision)

1. **Residual pattern.** Berkeley–ERA5 country disagreement sd 0.063 > outcome
   sd 0.047, spatially clustered (I = 0.18), and the low–low clusters coincide
   with the regions where Berkeley warms far less than ERA5 (Egypt, Mexico,
   Guatemala, Libya, Saudi Arabia, Iraq).
2. **Mechanism.** In station-sparse deserts and mountains Berkeley's kriged
   field is smoothed toward the regional mean and depends on which stations
   exist; ERA5 assimilates other observations and has its own biases. Part of
   the residual may be a property of the product, not of the climate.
3. **Candidate treatment.** Not a covariate. Refit every stage on the ERA5
   area-weighted outcome (same operator, same window) and on the two-product
   mean; the component of the residual that changes sign between products is
   observational uncertainty.
4. **Timing.** Not applicable.
5. **Mediator risk.** Not applicable.
6. **Sources.** In repository (`era5_area_trends.parquet`, ERA5 grid).
7. **Expected form.** Stages that capture physics should improve under both
   products; a gain confined to one product is suspect.
8. **Collinearity.** Not applicable.
9. **Group meaning.** None, but it changes the temperature product, which the
   V1 baseline and this session's rules forbid as a primary; it can only run as
   an explicitly approved robustness arm.

## H8. Internal decadal variability aliased into the linear trend — not pursued as a predictor

1. **Residual pattern.** Pacific-facing negatives (Mexico, Central America,
   Chile, Argentina, New Zealand, Korea) and Atlantic-facing positives (West
   Africa, Caribbean, Brazil, north-eastern Europe/Canada); negative correlogram
   at 2000–4000 km (dipoles).
2. **Mechanism.** Pacific and Atlantic decadal SST modes changed phase inside
   the window (PDO 1976–77 and ~1999; AMO ~1965 and ~1995); their regional land
   imprint survives in a 64-year slope.
3. **Candidate quantity.** Not a cross-sectional predictor. A time-series
   diagnostic (regress each country's cell-mean monthly series on the mode
   indices and re-estimate the trend with the modes removed) would quantify the
   component; it changes the outcome definition and is out of scope for M1–M3.
4–9. Recorded so that a residual that remains after M3 is not called
   "unexplained physics" when part of it is unforced variability at this
   resolution. Owner decision whether to open a diagnostic arm.

## H9. The unit of analysis — not pursued (estimand)

Seven large countries have more internal trend variance than the residual;
the model gives each included country the same regression weight. An area-weighted regression or a
cell-level hierarchical model would answer a different question (how land
warms) rather than the V1 question (how countries as units differ). Recorded as
an open decision, not a stage.

## H10. Station density as observational structure — not pursued in this line

`station_density` sits in the population group, is orthogonal to the residual
by construction, and gives Japan leverage 0.47. Moving it to an "observational"
group is a schema revision (forbidden here) and would change every share's
meaning; recorded for the owner.

---

Summary by stage: M1a = H2; M1 = H3, H6 (and H4 only with an owner decision);
M2 = H1, H5 (spline); M3 = accounting of whatever remains; M4 robustness arm =
H7; not pursued = H8, H9, H10.

### Future aerosol guardrail (no data fetched in M0.5)

If aerosol history is later tested, predeclare a separate regional/external
forcing concept, not geography. Before fitting, specify exact observable, physical
rationale, provenance, time aggregation, forcing/inventory/proxy meaning, overlap
with national responsibility, socioeconomic confounding and interpretation of its
decomposition share. This record does not authorize aerosol work or any M1 input.
