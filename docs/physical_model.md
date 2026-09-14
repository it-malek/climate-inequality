# Global temperature versus radiative forcing

`src/forcings.py` assembles an annual table of the global mean temperature
anomaly and the effective radiative forcings; `src/physical_model.py` fits a
closed-form Bayesian regression with AR(1) errors and scores it out of sample.
The module is kept apart from the cross-country analysis: it explains the
global trajectory over time, which has no cross-country variance, and it never
feeds the decomposition or the coupling stage.

## Data

| Series | Source | Use |
|---|---|---|
| `temp_anomaly` | NASA GISTEMP v4 global land–ocean annual mean (`J-D`), re-centred on 1951–1980 | outcome, 1951–2024 |
| `temp_uncertainty` | Berkeley Earth land–ocean monthly uncertainty, RMS within year | schema column, not used by the fit |
| `erf_co2`, `erf_ch4`, `erf_n2o`, `erf_aerosol`, `erf_volcanic`, `erf_solar` | Forster et al., *Indicators of Global Climate Change*, AR6-consistent ERF (W/m²) | drivers |
| `oni` | NOAA CPC Oceanic Niño Index, annual mean of the overlapping seasons | ENSO regressor |

Three checks run before the table is written: the years must be contiguous,
the estimator columns must have no gaps, and GISTEMP must correlate with the
Berkeley Earth annual series at r ≥ 0.95 over their overlap (a guard against a
silently changed upstream file). A magnitude guard (mean anomaly over 2010–2020
between 0.3 and 2.0 °C) catches a unit change that a correlation would miss.

## Model

```
T(t) = b₀ + Σᵢ bᵢ Fᵢ(t − ℓᵢ) + εₜ,   εₜ = ρ εₜ₋₁ + uₜ,   uₜ ~ N(0, σ²)
```

- Lags ℓᵢ are fixed: one year for the slow, ocean-mediated CO₂, CH₄, N₂O and
  aerosol forcings; zero for volcanic, solar and ENSO.
- Drivers are standardised on the training-period moments; an intercept column
  is prepended.
- The design and outcome are whitened with the Prais–Winsten AR(1) transform
  at the current ρ (the first row is scaled by √(1 − ρ²) and kept).
- A Normal-Inverse-Gamma conjugate prior gives the posterior in closed form:
  a ridge (Gaussian) prior with precision λ on the slopes and a near-flat prior
  on the intercept, and an Inverse-Gamma(10⁻³, 10⁻³) prior on σ². λ is chosen
  by maximising the log marginal likelihood over log₁₀ λ ∈ [−6, 6] with
  bounded Brent search.
- ρ is re-estimated from the lag-1 autocorrelation of the un-whitened training
  residuals and the whitening repeated to convergence (Cochrane–Orcutt style
  iteration, starting at ρ = 0).
- Predictions are Student-t with 2αₙ degrees of freedom. In-sample years use
  the stationary AR(1) variance factor 1/(1 − ρ²); an out-of-sample year at
  horizon h adds ρʰ times the last training residual to the mean and uses the
  h-step forecast-error factor (1 − ρ²ʰ)/(1 − ρ²). Parameter uncertainty enters
  through xᵀVₙx.

No sampler is used anywhere, so the fit is deterministic; the summary records
the SHA-256 of the forcings table it was fit on.

## Fit

Training window 1951–2013 (63 years), test window 2014–2024 (11 years).

| Quantity | Value |
|---|:-:|
| Train R² | 0.915 |
| Test RMSE | 0.103 °C |
| Test 95% band coverage | 10 of 11 years |
| AR(1) ρ | −0.034 |
| Ridge λ (empirical Bayes) | 0.97 |

| Driver | Sensitivity (°C per W/m²) | 95% interval |
|---|:-:|:-:|
| CO₂ | 0.37 | 0.06 to 0.68 |
| CH₄ | 0.06 | −1.00 to 1.12 |
| N₂O | 3.22 | −0.11 to 6.56 |
| Aerosol | 0.07 | −0.16 to 0.31 |
| Volcanic | 0.08 | 0.02 to 0.14 |
| Solar | 0.40 | −0.05 to 0.84 |
| ONI (per index unit) | 0.064 | 0.028 to 0.100 |

The greenhouse-gas forcings rise together over the period, so their individual
coefficients are only partly identified; the ridge prior stabilises the joint
fit without resolving the split. The estimated ρ is close to zero, which means
that once the forcings are in the model there is little year-to-year
persistence left in the residual at annual resolution. The volcanic term
reproduces the cooling after Agung (1963), El Chichón (1982) and Pinatubo
(1991).

These are model sensitivities under a linear AR(1) surrogate, not measured
climate sensitivities and not a detection-and-attribution result.

## Why this estimator

The fit is a 63-row, 8-column linear model; speed is irrelevant. The estimator
was kept closed-form on NumPy and SciPy so that the committed artifact is
byte-reproducible and adds no dependency to the pipeline; statsmodels' GLSAR
serves as a test-time cross-check. Forward climate emulators (FaIR, Hector,
MAGICC, two-layer energy-balance models) solve the inverse of this problem,
producing temperature from forcing, and would belong in a separate artifact if
the project ever needed a physical simulator; MCMC frameworks were rejected
because a conjugate posterior needs no sampler.
