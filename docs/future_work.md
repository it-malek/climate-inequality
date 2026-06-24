# Future work

Concrete next steps, roughly in order of value-for-effort. Each item names
the data it needs and the question it answers; none requires restructuring
the existing pipeline — they extend it.

## 1. Validate the fitted trends out of sample (Phase 6, highest value)

✅ **built** — `src/validation.py` + the "Did the trends hold?" dashboard page.
Result: the 1950–2013 lines *underpredict* — full-record land slope **0.200** vs the
fitted **0.147 °C/decade** (Δ +0.053, tight CI), mean post-2013 residual **+0.48 °C**;
warming accelerated, exactly as anticipated below.

The dataset ends September 2013, so every trend here is a *backcast*. The
strongest possible upgrade is to test them against the twelve years of
observations that now exist:

- **Data:** Berkeley Earth's current station/city series
  ([berkeleyearth.org/data](https://berkeleyearth.org/data/), updated
  monthly) or NOAA GHCN-M v4. Both overlap the Kaggle snapshot's cities.
- **Method:** extend each city's anomaly series through ~2025 (same
  1951–1980 climatology), then ask two questions per location:
  (a) *Does the 1950–2013 Theil–Sen line predict 2014–2025 anomalies?*
  (forecast residuals vs the fit's CI); (b) *Has the trend accelerated?*
  (refit on the full window, compare slopes; 2015–16 and 2023–24 El Niño
  years will dominate, so report both with and without ENSO adjustment).
- **Expected story:** global land warming has accelerated post-2013, so the
  fitted lines should systematically *underpredict* — a finding in itself,
  and an honest stress test of the whole pipeline.

## 2. Better warming data

- **Berkeley Earth 1°×1° gridded product — ⏭️ NEXT EPIC (v1.2).** The project's
  #1 external-validity gap is station sampling bias: country means are
  *station-weighted*, so dense mid-latitude clusters dominate. This replaces it
  with TRUE **area-weighted** country means. Zero data friction — the grid is
  already in-repo (`data/raw/berkeley_gridded/Complete_TAVG_LatLong1.nc`, read by
  `src/validation.py`). Plan: per-cell Theil–Sen trend (same operator/window as
  the station pipeline) → assign cells to countries via the GPW v4 **National
  Identifier Grid** band (no new polygon dataset) → **cos(latitude)** area-weighted
  mean per country. *cos(lat) is REQUIRED here* — temperature is an *intensive*
  field, the exact mirror of the GPW population-**count** rule (where cos-lat is
  forbidden); reuse `src/population.py` `latitude_area_weights` / `area_weighted_mean`.
  Ships as a third L3 impact lens (`impact_index_area_weighted`) in the PCS v2 Wide
  Registry, then re-tests whether the climate-inequality conclusions (ρ≈0.36,
  Gini≈0.56, the Central-Asia mismatch leaders) survive when every km² counts equally.
- **ERA5 reanalysis 2m temperature** ✅ **built** (independent cross-check of the
  v1.2 area-weighted finding). v1.2's headline — area-weighting collapses the
  warming↔responsibility coupling (ρ +0.36 → +0.01) — rested on a single gridded
  product, so this recomputes the *same* area-weighted country warming off ERA5,
  a model-assimilated field with no station-sampling gaps. **Same operator, window
  and cos(lat) weighting** as `src.area_weighting` (the streamed Theil–Sen
  `cell_trends` was generalized with `var`/`decode_times`/`time_to_months` knobs,
  defaults unchanged); only the data source differs. ERA5-specific handling:
  absolute Kelvin (irrelevant for a *slope*), CF datetime axis, 0–360 longitudes
  normalized for the GPW band-11 ISO3 sampler (`src/era5_weighting.py`). The
  comparator (`src/era5_validation.py`) reports the world-land sanity mean and the
  station/Berkeley/ERA5 coupling reproduction (Spearman ρ + Gini, via the reused
  `src.coupling` helpers) side-by-side on one common country set — kept as a
  **cross-check artifact, not a new PCS projection** (`PCS_V2` stays frozen at six).
  The ERA5 grid is not in-repo (~200 MB, gitignored); fetch it once via
  `scripts/fetch_era5.py` (needs a Copernicus CDS account + the `era5` optional
  dependency), then `uv run python -m src.era5_validation`. If ERA5 reproduces the
  collapse the finding is robust to the data source; if not, it was
  Berkeley-specific — equally publishable. Where the two products disagree at the
  *cell* level, station inhomogeneities are suspect.
- **Population weighting:** ✅ **built** (Layer 3 exposure lens). City-locations
  are joined against the SEDAC **GPW v4.11** 15-arc-minute population-*count*
  grid by coordinates (no name matching, `src/population.py`), turning the
  station-weighted country mean into a *people-weighted* one ("warming
  experienced by the average resident"). Counts are used directly — **no
  cos(latitude) weighting**: a count grid already embeds meridian convergence,
  and cos-weighting would understate fast-warming high-latitude residents
  (cos(lat) is for intensive/density fields only). Loading is native-lazy via
  `xarray`/`netcdf4` (only the sampled cells are read; **dask is intentionally
  not a dependency**).

## 3. Better inequality metrics

The current metric (country warming vs cumulative per-capita CO₂) measures
*who warms*, not *who suffers*. Variants that sharpen the question:

- **Consumption-based emissions** ✅ **built** (Layer 3 consumption lens, PCS v2):
  OWID's `consumption_co2` reallocates traded-goods emissions to the consuming
  country; the lens tests whether the production/consumption distinction changes
  the responsibility ranking, with both cumulatives summed over each country's
  shared consumption-available window to avoid a window confound.
- **Lorenz/Gini framing:** ✅ **built** — the cumulative share of warming
  exposure (station- or **people-weighted**) is plotted against the cumulative
  share of emissions responsibility, with the Gini-style inequality coefficient,
  on the Layer 3 dashboard page.
- **Exposure × vulnerability:** join ND-GAIN (vulnerability/readiness) or
  income groups; the hypothesis worth testing is a *triple* inequality —
  low emitters warm somewhat less in °C but are far more exposed
  (outdoor labor, agriculture) and less able to adapt.
- **Era-weighted responsibility:** divide cumulative emissions by
  population *of the emitting era* (e.g., population-year-weighted), not
  2013 population — fairer to fast-growing countries.
- **Degrees per benefit:** warming experienced per unit of cumulative GDP
  generated by fossil energy — the "who got the upside" framing.

## 4. New questions this data can spotlight

- **Extremes vs means — follow-on epic (after the gridded v1.2).** Monthly means
  hide heat extremes. With daily data (GHCN-Daily), is inequality in *extreme-heat
  days* larger than in mean warming? (Almost certainly yes — tropical countries sit
  closer to physiological thresholds.) NOTE: GHCN-Daily is **not in-repo** and is
  large — scope the download/storage first; lower priority than the zero-friction
  gridded epic above.
- **Within-country inequality:** the per-city trends already exist —
  variance decomposition of warming within vs between countries; which
  countries contain both fast- and slow-warming regions?
- **The Iranian-plateau hotspot:** the fastest-warming cluster in the data
  (≈0.32–0.34 °C/decade around 31–38°N in Iran/Central Asia) is a real,
  documented semi-arid amplification signal — worth a focused writeup
  (aridity feedback, Caspian influence, station quality?).
- **Urban heat island contamination:** these are *city* series. Compare
  city trends against Berkeley Earth's rural-only flagged series to bound
  how much of the 0.146 °C/decade is UHI rather than climate.
- **Trend acceleration:** piecewise or rolling-window Theil–Sen (e.g.,
  30-year windows) to map *where* warming is accelerating, not just where
  it is fast.

## 5. Method upgrades

- **Uncertainty-weighted fits:** use `AverageTemperatureUncertainty` as
  weights (weighted least squares alongside Theil–Sen) — currently a
  documented limitation.
- **Spatially-honest regression:** country warming observations are
  spatially correlated, so HC1 SEs are optimistic; add a latitude term and
  Conley (spatial HAC) standard errors, or fit a spatial error model.
- **Temporal surfaces:** the original 2022 proposal imagined interpolated
  temperature *over time*; an animated decade-by-decade anomaly surface
  (same IDW machinery, one frame per decade) would close that loop and
  make a striking dashboard page.
- **Kriging refinements:** anisotropic variograms (E–W correlation lengths
  exceed N–S in temperature fields) and a nugget fixed from measurement
  uncertainty would make the kriging comparison fairer before declaring
  IDW the winner.

## 6. Layer 1 — physical-drivers model (a new layer)

✅ **built** — `src/physical_model.py` + `src/forcings.py` produce `forcings.parquet`,
`physical_trajectory.parquet` and `physical_summary.json`, wired into the app bundle
and dashboard (train R² 0.91, hindcast band coverage 91%, AR(1) ρ ≈ 0).

A genuinely *physical* layer: global mean temperature as a response to radiative
forcings (CO₂/CH₄/N₂O/aerosol/volcanic/solar) plus ENSO, fit as a closed-form
Bayesian linear state-space model with AR(1) inertia and hindcast validation. It is
specified in the `climate-inequality-instructions` repo (`03-models.md`,
`07-data-schemas.md`) and *extends* rather than
restructures the pipeline (layers never merge; artifact-only communication). The
toolchain decision for that build — **Python (NumPy/SciPy) at the core, statsmodels as a
test-time cross-check, Wolfram as a design-time symbolic oracle; Julia and the
climate-emulator stack rejected for the specified model** — is recorded in
[`docs/l1_toolchain_survey.md`](l1_toolchain_survey.md).
