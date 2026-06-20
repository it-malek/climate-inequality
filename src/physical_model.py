"""Layer 1: physical climate driver model (closed-form Bayesian ridge-GLS).

Fits annual global mean temperature anomaly ``T(t)`` to effective radiative
forcings (CO2, CH4, N2O, aerosol, volcanic, solar) plus the ENSO ONI index:

    T(t) = b0 + sum_i b_i * F_i(t - l_i) + eps_t,
    eps_t = rho * eps_{t-1} + u_t,   u_t ~ N(0, sigma^2)

The estimator is the spec's closed-form recipe (``03-models.md`` Layer 1), with
**no MCMC and no sampling seed**:

  1. Lag each forcing by its fixed physical lag and standardize on the training
     moments (an intercept column is prepended afterwards).
  2. Whiten the design with the Prais-Winsten AR(1) transform at the current rho.
  3. Place a Normal-Inverse-Gamma conjugate prior on ``(b, sigma^2)`` -- a ridge
     (Gaussian) prior with precision ``lambda`` on the slopes, near-flat on the
     intercept -- and compute the closed-form NIG posterior on the whitened design.
  4. Choose ``lambda`` by empirical Bayes (maximize the closed-form log evidence).
  5. Re-estimate rho from the lag-1 autocorrelation of the un-whitened residuals
     and iterate 2-5 to convergence (Cochrane-Orcutt / Prais-Winsten).

Outputs (the layer contract, see ``07-data-schemas.md``): per-driver sensitivity
(posterior mean +/- sd and 95% credible interval, deg C per W/m^2), a predicted
trajectory with Student-t predictive bands, and hindcast skill (train year<=2013,
test year>2013: out-of-sample RMSE, band coverage, AR(1) rho). Two artifacts are
written: ``physical_summary.json`` and ``physical_trajectory.parquet``.

**Framing.** L1 is a *predictive association* of the temperature trajectory with
forcing proxies, validated by out-of-sample hindcast -- it is **not** formal
detection-and-attribution and makes no causal claim. The ``interpretation`` field
in the summary carries this disclaimer.

Determinism: the fit is entirely closed-form (numpy/scipy, float64); ``lambda`` is
optimized over a fixed bracket with bounded Brent; linear systems are solved by
Cholesky (``cho_factor``/``cho_solve``), never an explicit ``inv``. Committed JSON
is float-rounded (:func:`src.data_io.round_floats`) and the parquet is written
through :func:`src.data_io.write_typed_parquet` for on-platform byte-stability.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from math import lgamma, log, pi

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t

from src.data_io import PROCESSED_DIR, round_floats, write_typed_parquet

logger = logging.getLogger(__name__)

# Driver order is fixed; the first six are effective radiative forcings (W/m^2)
# read as `erf_<driver>`, ONI is the dimensionless ENSO regressor read as `oni`.
DRIVERS: tuple[str, ...] = ("co2", "ch4", "n2o", "aerosol", "volcanic", "solar", "oni")
ERF_DRIVERS: tuple[str, ...] = DRIVERS[:6]

# Fixed physical lags (years): slow ocean-mediated GHG/aerosol forcings lag one
# year; fast volcanic/solar/ENSO effects are contemporaneous (spec 03-models.md).
DEFAULT_LAGS: dict[str, int] = {
    "co2": 1,
    "ch4": 1,
    "n2o": 1,
    "aerosol": 1,
    "volcanic": 0,
    "solar": 0,
    "oni": 0,
}

TRAIN_END = 2013                  # train on year <= TRAIN_END, test on year > it
DELTA = 1e-6                      # rho clamp keeps the AR(1) strictly stationary
V0_INTERCEPT = 1e8                # near-flat prior variance on the intercept
ALPHA0 = 1e-3                     # NIG inverse-gamma shape (weakly informative)
BETA0 = 1e-3                      # NIG inverse-gamma scale (weakly informative)
LAMBDA_LOG10_BOUNDS = (-6.0, 6.0)  # empirical-Bayes search bracket for log10(lambda)
MAX_ITERS = 100                   # Prais-Winsten / Cochrane-Orcutt iteration cap
RHO_TOL = 1e-8                    # convergence tolerance on |delta rho|

YEAR_COL = "year"
TEMP_COL = "temp_anomaly"

INTERPRETATION = (
    "Descriptive predictive association between the global mean temperature "
    "anomaly and effective-radiative-forcing proxies, validated by out-of-sample "
    "hindcast skill. The per-driver sensitivities are model coefficients (deg C "
    "per W/m^2), NOT formal detection-and-attribution: this layer makes no "
    "deterministic causal claim about any individual forcing."
)

# On-disk schema of physical_trajectory.parquet (DuckDB types), in order.
TRAJECTORY_SCHEMA: dict[str, str] = {
    "year": "BIGINT",
    "observed": "DOUBLE",
    "predicted_mean": "DOUBLE",
    "lower95": "DOUBLE",
    "upper95": "DOUBLE",
    "in_sample": "BOOLEAN",
}
TRAJECTORY_COLUMNS = tuple(TRAJECTORY_SCHEMA)

DEFAULT_FORCINGS_PATH = PROCESSED_DIR / "forcings.parquet"
DEFAULT_SUMMARY_PATH = PROCESSED_DIR / "physical_summary.json"
DEFAULT_TRAJECTORY_PATH = PROCESSED_DIR / "physical_trajectory.parquet"


def driver_column(driver: str) -> str:
    """Source column in ``forcings.parquet`` for `driver` (`erf_*` or `oni`)."""
    return "oni" if driver == "oni" else f"erf_{driver}"


# ---------------------------------------------------------------------
# Design assembly and standardization (spec 03-models.md, eqn for T(t))
# ---------------------------------------------------------------------


def build_design(
    forcings: pd.DataFrame,
    lags: dict[str, int] = DEFAULT_LAGS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lag-align the forcings into a natural-unit design matrix.

    Sorts by year, requires the input years to be contiguous (the Prais-Winsten
    transform assumes consecutive annual observations -- an internal gap would
    silently mis-lag the AR(1) structure), shifts each driver column by its
    integer lag so row ``t`` carries ``F_i(t - l_i)``, then keeps only
    complete-case rows over all drivers and the outcome.

    Args:
        forcings: one row per year with ``year``, ``temp_anomaly`` and the driver
            columns (``erf_<driver>`` for the six forcings, ``oni`` for ENSO).
        lags: per-driver integer lag in years.

    Returns:
        ``(years, X_raw, y, train_mask)`` -- ``years`` (n,), ``X_raw`` (n, p) in
        natural units with **no** intercept column, ``y`` (n,) the anomaly, and
        ``train_mask`` (n,) bool of ``years <= TRAIN_END``.

    Raises:
        ValueError: on a missing column or an internal year gap.
    """
    missing = [c for c in (YEAR_COL, TEMP_COL) if c not in forcings.columns]
    missing += [driver_column(d) for d in DRIVERS if driver_column(d) not in forcings.columns]
    if missing:
        raise ValueError(f"forcings frame missing column(s): {sorted(set(missing))}")

    frame = forcings.sort_values(YEAR_COL).reset_index(drop=True)
    raw_years = frame[YEAR_COL].to_numpy()
    if np.any(np.diff(raw_years) != 1):
        gaps = raw_years[1:][np.diff(raw_years) != 1]
        raise ValueError(f"forcings years are not contiguous (gap before {gaps.tolist()})")

    columns = []
    for driver in DRIVERS:
        series = frame[driver_column(driver)].astype(float)
        columns.append(series.shift(lags[driver]).to_numpy())
    x_raw = np.column_stack(columns)
    y = frame[TEMP_COL].to_numpy(dtype=float)

    complete = ~np.isnan(x_raw).any(axis=1) & ~np.isnan(y)
    years = raw_years[complete].astype(np.int64)
    x_raw = x_raw[complete]
    y = y[complete]
    # Leading lag NaNs drop contiguously, but an interior NaN in a driver/outcome
    # column would leave a gap here even when the raw years were contiguous --
    # re-check so the AR(1) whitening never silently treats a multi-year jump as
    # a single annual step.
    if np.any(np.diff(years) != 1):
        gaps = years[1:][np.diff(years) != 1]
        raise ValueError(
            f"years not contiguous after complete-case drop (gap before {gaps.tolist()}); "
            "an interior NaN in a driver or the outcome column breaks the AR(1) lag structure"
        )
    train_mask = years <= TRAIN_END
    if not train_mask.any():
        raise ValueError(f"no training rows (year <= {TRAIN_END}) after lag alignment")
    return years, x_raw, y, train_mask


def standardize(
    x_raw: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns on the **training** moments and prepend an intercept.

    Standardizing on the training mean/sd (``ddof=0``) keeps the ridge penalty
    scale-free and prevents the held-out years from leaking into the fit. The
    leading column of ones is the intercept (whose prior is near-flat).

    Returns:
        ``(X_std, mu, s)`` -- ``X_std`` (n, p+1) with column 0 all ones, and the
        per-driver training means ``mu`` (p,) and std-devs ``s`` (p,).
    """
    train = x_raw[train_mask]
    mu = train.mean(axis=0)
    s = train.std(axis=0, ddof=0)
    if np.any(s == 0.0):
        raise ValueError("a driver has zero training variance; cannot standardize")
    x_std = (x_raw - mu) / s
    intercept = np.ones((x_raw.shape[0], 1))
    return np.hstack([intercept, x_std]), mu, s


# ---------------------------------------------------------------------
# Prais-Winsten AR(1) whitening (spec 03-models.md, "AR(1)-whitened")
# ---------------------------------------------------------------------


def prais_winsten_whiten(matrix: np.ndarray, rho: float) -> np.ndarray:
    """Apply the Prais-Winsten AR(1) transform along the first (time) axis.

    Row 0 is scaled by ``sqrt(1 - rho^2)`` (the exact factor that gives the
    transformed first observation the same variance as the later innovations, so
    it is retained rather than dropped as in Cochrane-Orcutt); every later row
    ``t`` becomes ``M[t] - rho * M[t-1]``. Works on a 1-D vector (``y``) or a 2-D
    design (``X``) since both operations index the leading axis.

    Args:
        matrix: array whose first axis is time.
        rho: AR(1) coefficient (assumed ``|rho| < 1``).

    Returns:
        The whitened array, same shape as `matrix`.
    """
    m = np.asarray(matrix, dtype=float)
    whitened = np.empty_like(m)
    whitened[0] = m[0] * np.sqrt(1.0 - rho**2)
    whitened[1:] = m[1:] - rho * m[:-1]
    return whitened


# ---------------------------------------------------------------------
# Normal-Inverse-Gamma posterior + evidence (spec 03-models.md, conjugate NIG)
# ---------------------------------------------------------------------


def _prior_precision(p1: int, lam: float, v0: float) -> np.ndarray:
    """Diagonal NIG prior precision: ``1/v0`` on intercept, ``lam`` on each slope."""
    prec = np.full(p1, lam, dtype=float)
    prec[0] = 1.0 / v0
    return prec


def nig_posterior(
    xw: np.ndarray,
    yw: np.ndarray,
    lam: float,
    v0: float = V0_INTERCEPT,
    alpha0: float = ALPHA0,
    beta0: float = BETA0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Closed-form Normal-Inverse-Gamma posterior on the whitened design.

    With prior ``b ~ N(0, sigma^2 Lambda0^{-1})``, ``sigma^2 ~ InvGamma(alpha0,
    beta0)`` and ``Lambda0 = diag(1/v0, lam, ..., lam)``:

        Lambda_n = Lambda0 + Xw^T Xw
        m_n      = Lambda_n^{-1} Xw^T yw
        alpha_n  = alpha0 + N/2
        beta_n   = beta0 + 0.5 (yw^T yw - m_n^T Xw^T yw)

    The posterior covariance scale ``V_n = Lambda_n^{-1}`` and ``m_n`` are obtained
    by Cholesky solve (``cho_factor``/``cho_solve``) -- never an explicit ``inv`` --
    for numerical stability on the correlated forcing design.

    Returns:
        ``(m_n, V_n, alpha_n, beta_n)``.
    """
    n, p1 = xw.shape
    prior_prec = _prior_precision(p1, lam, v0)
    lambda_n = np.diag(prior_prec) + xw.T @ xw
    cho = cho_factor(lambda_n, lower=True)
    xty = xw.T @ yw
    m_n = cho_solve(cho, xty)
    v_n = cho_solve(cho, np.eye(p1))
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + 0.5 * float(yw @ yw - m_n @ xty)
    return m_n, v_n, alpha_n, beta_n


def log_evidence(
    xw: np.ndarray,
    yw: np.ndarray,
    lam: float,
    v0: float = V0_INTERCEPT,
    alpha0: float = ALPHA0,
    beta0: float = BETA0,
) -> float:
    """Closed-form NIG log marginal likelihood (the empirical-Bayes objective).

        log p(y) = -N/2 log(2 pi)
                   + 1/2 logdet(Lambda0) - 1/2 logdet(Lambda_n)
                   + alpha0 log(beta0) - alpha_n log(beta_n)
                   + lgamma(alpha_n) - lgamma(alpha0)

    Log-determinants come from the Cholesky factor (sum of log diagonal),
    consistent with :func:`nig_posterior`'s stable solve.
    """
    n, p1 = xw.shape
    prior_prec = _prior_precision(p1, lam, v0)
    lambda_n = np.diag(prior_prec) + xw.T @ xw
    cho_low, _ = cho_factor(lambda_n, lower=True)
    logdet_lambda_n = 2.0 * float(np.sum(np.log(np.diag(cho_low))))
    logdet_lambda0 = float(np.sum(np.log(prior_prec)))

    xty = xw.T @ yw
    m_n = cho_solve((cho_low, True), xty)
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + 0.5 * float(yw @ yw - m_n @ xty)
    return (
        -0.5 * n * log(2.0 * pi)
        + 0.5 * logdet_lambda0
        - 0.5 * logdet_lambda_n
        + alpha0 * log(beta0)
        - alpha_n * log(beta_n)
        + lgamma(alpha_n)
        - lgamma(alpha0)
    )


def select_lambda(
    xw: np.ndarray,
    yw: np.ndarray,
    v0: float = V0_INTERCEPT,
    alpha0: float = ALPHA0,
    beta0: float = BETA0,
) -> float:
    """Empirical-Bayes ridge precision: maximize the log evidence over log10(lambda).

    Bounded Brent over ``log10(lambda) in LAMBDA_LOG10_BOUNDS`` -- deterministic
    (fixed bracket + tolerance) and reconciles the spec's "Brent's method" with
    its search bracket.
    """

    def neg_log_evidence(log10_lam: float) -> float:
        return -log_evidence(xw, yw, 10.0**log10_lam, v0, alpha0, beta0)

    res = minimize_scalar(neg_log_evidence, bounds=LAMBDA_LOG10_BOUNDS, method="bounded")
    return float(10.0**res.x)


def estimate_rho(residuals: np.ndarray) -> float:
    """Lag-1 autocorrelation of `residuals`, clamped strictly inside the unit root.

    ``rho = sum_{t>=1} e_t e_{t-1} / sum_t e_t^2``, clipped to
    ``[-1 + DELTA, 1 - DELTA]`` so the AR(1) stays stationary even on a near-unit-
    root series.
    """
    e = np.asarray(residuals, dtype=float)
    denom = float(e @ e)
    if denom == 0.0:
        return 0.0
    rho = float(e[1:] @ e[:-1] / denom)
    return float(np.clip(rho, -1.0 + DELTA, 1.0 - DELTA))


# ---------------------------------------------------------------------
# Iterative Prais-Winsten / NIG fit (spec 03-models.md, "estimated by iteration")
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FitState:
    """Converged NIG posterior + AR(1) parameters on the training rows."""

    m_n: np.ndarray
    v_n: np.ndarray
    alpha_n: float
    beta_n: float
    rho: float
    lam: float
    mu: np.ndarray
    s: np.ndarray
    n_train: int


def fit_nig_ar1(
    x_std: np.ndarray, y: np.ndarray, train_mask: np.ndarray
) -> FitState:
    """Iterate Prais-Winsten whitening + NIG fit + rho update to convergence.

    Starting from ``rho = 0``: whiten the training design and outcome, select
    ``lambda`` by evidence, compute the NIG posterior, re-estimate ``rho`` from the
    **un-whitened** training residuals, and repeat until ``|delta rho| < RHO_TOL``
    (or ``MAX_ITERS``). A final whiten+fit at the converged ``rho`` guarantees the
    stored posterior is exactly consistent with ``FitState.rho``.
    """
    x_tr = x_std[train_mask]
    y_tr = y[train_mask]

    rho = 0.0
    for _ in range(MAX_ITERS):
        xw = prais_winsten_whiten(x_tr, rho)
        yw = prais_winsten_whiten(y_tr, rho)
        lam = select_lambda(xw, yw)
        m_n, _, _, _ = nig_posterior(xw, yw, lam)
        residuals = y_tr - x_tr @ m_n
        rho_new = estimate_rho(residuals)
        converged = abs(rho_new - rho) < RHO_TOL
        rho = rho_new
        if converged:
            break

    # Final refit at the converged rho so (m_n, V_n, alpha_n, beta_n, lam) match.
    xw = prais_winsten_whiten(x_tr, rho)
    yw = prais_winsten_whiten(y_tr, rho)
    lam = select_lambda(xw, yw)
    m_n, v_n, alpha_n, beta_n = nig_posterior(xw, yw, lam)
    return FitState(
        m_n=m_n,
        v_n=v_n,
        alpha_n=alpha_n,
        beta_n=beta_n,
        rho=rho,
        lam=lam,
        mu=np.zeros(0),  # filled by caller after standardize
        s=np.ones(0),
        n_train=int(train_mask.sum()),
    )


# ---------------------------------------------------------------------
# Back-transform to natural units + predictive trajectory (spec Outputs)
# ---------------------------------------------------------------------


def back_transform(
    m_n: np.ndarray,
    v_n: np.ndarray,
    alpha_n: float,
    beta_n: float,
    mu: np.ndarray,
    s: np.ndarray,
) -> tuple[dict[str, dict[str, float]], float]:
    """Map standardized posterior coefficients to natural-unit sensitivities.

    The marginal posterior of each coefficient is Student-t with ``df = 2 alpha_n``
    and scale ``sqrt(beta_n/alpha_n * V_n[kk])``; its standard deviation is
    ``sqrt(beta_n/(alpha_n - 1) * V_n[kk])``. Dividing by the driver's training sd
    ``s_i`` converts a standardized slope to deg C per natural unit (W/m^2, or
    dimensionless for ONI). The natural-unit intercept ``b0`` is returned for
    recovery checks (the trajectory itself uses the standardized ``m_n`` directly).

    Returns:
        ``(sensitivity, intercept)`` -- ``sensitivity[driver] = {mean, sd, ci_low,
        ci_high}`` keyed by :data:`DRIVERS`.
    """
    df = 2.0 * alpha_n
    t_crit = float(student_t.ppf(0.975, df))
    diag_v = np.diag(v_n)

    sensitivity: dict[str, dict[str, float]] = {}
    for i, driver in enumerate(DRIVERS):
        k = i + 1  # column 0 is the intercept
        mean = float(m_n[k] / s[i])
        sd = float(np.sqrt(beta_n / (alpha_n - 1.0) * diag_v[k]) / s[i])
        half = t_crit * float(np.sqrt(beta_n / alpha_n * diag_v[k]) / s[i])
        sensitivity[driver] = {
            "mean": mean,
            "sd": sd,
            "ci_low": mean - half,
            "ci_high": mean + half,
        }

    slopes_natural = m_n[1:] / s
    intercept = float(m_n[0] - np.sum(slopes_natural * mu))
    return sensitivity, intercept


def predict_trajectory(
    fit: FitState,
    x_std: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    train_mask: np.ndarray,
    train_end: int = TRAIN_END,
) -> pd.DataFrame:
    """Posterior-predictive mean + 95% Student-t bands for every year.

    In-sample years use the stationary AR(1) noise factor ``c = 1/(1 - rho^2)``.
    For a test year at horizon ``h = year - train_end`` the conditional forecast
    adds the decayed last-training residual ``rho^h * eps_t0`` to the mean and uses
    the ``h``-step forecast-error factor ``c = (1 - rho^{2h})/(1 - rho^2)``. The
    band half-width is ``t_crit * sqrt(beta_n/alpha_n * (c + x_t^T V_n x_t))``,
    combining innovation and parameter uncertainty under the Student-t predictive.
    """
    rho = fit.rho
    m_n, v_n = fit.m_n, fit.v_n
    df = 2.0 * fit.alpha_n
    t_crit = float(student_t.ppf(0.975, df))
    sigma2 = fit.beta_n / fit.alpha_n
    one_minus_rho2 = 1.0 - rho**2

    # Last training residual (un-whitened) anchors the AR(1) forecast.
    x_tr = x_std[train_mask]
    eps_t0 = float((y[train_mask] - x_tr @ m_n)[-1])

    rows = []
    for idx in range(len(years)):
        x_t = x_std[idx]
        mean_param = float(x_t @ m_n)
        if train_mask[idx]:
            mean = mean_param
            c = 1.0 / one_minus_rho2
        else:
            h = int(years[idx] - train_end)
            mean = mean_param + rho**h * eps_t0
            c = (1.0 - rho ** (2 * h)) / one_minus_rho2
        var = sigma2 * (c + float(x_t @ v_n @ x_t))
        half = t_crit * np.sqrt(var)
        rows.append(
            {
                "year": int(years[idx]),
                "observed": float(y[idx]),
                "predicted_mean": mean,
                "lower95": mean - half,
                "upper95": mean + half,
                "in_sample": bool(train_mask[idx]),
            }
        )
    return pd.DataFrame(rows, columns=list(TRAJECTORY_COLUMNS))


# ---------------------------------------------------------------------
# Result + orchestration (mirrors src/coupling.py, src/decomposition.py)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalModelResult:
    """The L1 layer contract: AR(1) rho, ridge lambda, sensitivities, hindcast."""

    outcome: str
    n_years: int
    train_end: int
    ar1_rho: float
    lambda_: float
    sigma2_mean: float
    lags: dict[str, int] = field(default_factory=dict)
    sensitivity: dict[str, dict[str, float]] = field(default_factory=dict)
    hindcast: dict[str, float] = field(default_factory=dict)

    def check(self, atol: float = 1e-9) -> None:
        """Assert stationarity, a bounded coverage, finite variance, full keys."""
        if not abs(self.ar1_rho) < 1.0:
            raise AssertionError(f"ar1_rho {self.ar1_rho!r} not strictly stationary")
        coverage = self.hindcast.get("test_band_coverage", 0.0)
        if not (0.0 - atol <= coverage <= 1.0 + atol):
            raise AssertionError(f"test_band_coverage {coverage!r} outside [0, 1]")
        if not (np.isfinite(self.sigma2_mean) and self.sigma2_mean > 0.0):
            raise AssertionError(f"sigma2_mean {self.sigma2_mean!r} not finite positive")
        if set(self.sensitivity) != set(DRIVERS):
            raise AssertionError(f"sensitivity keys {set(self.sensitivity)} != DRIVERS")
        if set(self.lags) != set(DRIVERS):
            raise AssertionError(f"lags keys {set(self.lags)} != DRIVERS")


def compute_physical_model(
    forcings: pd.DataFrame,
    lags: dict[str, int] = DEFAULT_LAGS,
    train_end: int = TRAIN_END,
) -> tuple[pd.DataFrame, PhysicalModelResult]:
    """Full pure pipeline: design -> iterative NIG fit -> trajectory + hindcast.

    Args:
        forcings: ``forcings.parquet`` rows (one per year; see module docstring).
        lags: per-driver integer lags (default :data:`DEFAULT_LAGS`).
        train_end: last in-sample year (default :data:`TRAIN_END`).

    Returns:
        ``(trajectory_df, result)`` -- the per-year trajectory table and the
        :class:`PhysicalModelResult` summary. Calls ``result.check()``.
    """
    years, x_raw, y, _ = build_design(forcings, lags)
    train_mask = years <= train_end
    if not train_mask.any():
        raise ValueError(f"no training rows (year <= {train_end}) after lag alignment")

    x_std, mu, s = standardize(x_raw, train_mask)
    fit = fit_nig_ar1(x_std, y, train_mask)
    fit = FitState(
        m_n=fit.m_n, v_n=fit.v_n, alpha_n=fit.alpha_n, beta_n=fit.beta_n,
        rho=fit.rho, lam=fit.lam, mu=mu, s=s, n_train=fit.n_train,
    )

    sensitivity, _intercept = back_transform(
        fit.m_n, fit.v_n, fit.alpha_n, fit.beta_n, mu, s
    )
    trajectory = predict_trajectory(fit, x_std, y, years, train_mask, train_end)

    # Hindcast metrics in natural deg C (mean function for train R^2; conditional
    # forecasts for the test block).
    test_mask = ~train_mask
    obs = trajectory["observed"].to_numpy()
    pred = trajectory["predicted_mean"].to_numpy()
    y_tr = obs[train_mask]
    ss_tot = float(np.sum((y_tr - y_tr.mean()) ** 2))
    ss_res = float(np.sum((y_tr - pred[train_mask]) ** 2))
    train_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    n_test = int(test_mask.sum())
    if n_test:
        test_resid = obs[test_mask] - pred[test_mask]
        test_rmse = float(np.sqrt(np.mean(test_resid**2)))
        inside = (
            (obs[test_mask] >= trajectory["lower95"].to_numpy()[test_mask])
            & (obs[test_mask] <= trajectory["upper95"].to_numpy()[test_mask])
        )
        test_band_coverage = float(np.mean(inside))
    else:
        test_rmse = 0.0
        test_band_coverage = 0.0

    result = PhysicalModelResult(
        outcome="global_temp_anomaly",
        n_years=int(len(years)),
        train_end=int(train_end),
        ar1_rho=float(fit.rho),
        lambda_=float(fit.lam),
        sigma2_mean=float(fit.beta_n / (fit.alpha_n - 1.0)),
        lags=dict(lags),
        sensitivity=sensitivity,
        hindcast={
            "train_r2": train_r2,
            "test_rmse": test_rmse,
            "test_band_coverage": test_band_coverage,
            "n_test": n_test,
        },
    )
    result.check()
    return trajectory, result


def summary_payload(result: PhysicalModelResult) -> dict:
    """JSON payload with the non-causal disclaimer, float-rounded for byte-stability."""
    return round_floats({"interpretation": INTERPRETATION, **asdict(result)})


def build_physical_model(
    forcings_path=DEFAULT_FORCINGS_PATH,
    summary_path=DEFAULT_SUMMARY_PATH,
    trajectory_path=DEFAULT_TRAJECTORY_PATH,
) -> dict:
    """Read forcings, fit the model, and write the two L1 artifacts.

    Returns:
        Dict with ``trajectory`` (DataFrame), ``result``
        (:class:`PhysicalModelResult`), ``summary_path`` and ``trajectory_path``.
    """
    forcings = pd.read_parquet(forcings_path)
    trajectory, result = compute_physical_model(forcings)
    write_typed_parquet(trajectory, trajectory_path, TRAJECTORY_SCHEMA, order_by=("year",))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary_payload(result), indent=2) + "\n", encoding="utf-8"
    )
    logger.info("wrote %s and %s", trajectory_path, summary_path)
    return {
        "trajectory": trajectory,
        "result": result,
        "summary_path": summary_path,
        "trajectory_path": trajectory_path,
    }


def main() -> None:
    """Fit on the real forcings table and print the hindcast headline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = build_physical_model()
    r = out["result"]
    h = r.hindcast
    print(
        f"L1 physical driver model ({r.outcome}, n={r.n_years}, "
        f"train<= {r.train_end}, deterministic / predictive-association)"
    )
    print(f"  AR(1) rho          : {r.ar1_rho:+.3f}")
    print(f"  ridge lambda       : {r.lambda_:.3g}")
    print(f"  train R^2          : {h['train_r2']:.3f}")
    print(f"  test RMSE (deg C)  : {h['test_rmse']:.3f}")
    print(f"  test band coverage : {h['test_band_coverage']:.0%}  (n_test={h['n_test']})")
    print("  sensitivities (deg C per W/m^2; ONI dimensionless):")
    for driver in DRIVERS:
        sens = r.sensitivity[driver]
        print(
            f"    {driver:<9s} {sens['mean']:+.4f}  "
            f"[{sens['ci_low']:+.4f}, {sens['ci_high']:+.4f}]"
        )
    print(f"  trajectory: {out['trajectory_path']}")
    print(f"  summary:    {out['summary_path']}")


if __name__ == "__main__":
    main()
