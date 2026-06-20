"""Tests for src.physical_model -- the Layer 1 physical driver estimator.

Encodes the acceptance gate from the spec (03-models.md Layer 1): the
Prais-Winsten whitener, the closed-form NIG posterior + evidence-lambda, recovery
of known synthetic parameters, near-unit-root stationarity, out-of-sample hindcast
coverage, deterministic byte-stable artifacts, the trajectory/summary schemas, and
a statsmodels cross-check of the AR(1) GLS fit.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_io import write_typed_parquet
from src.physical_model import (
    ALPHA0,
    DEFAULT_LAGS,
    DRIVERS,
    TRAIN_END,
    TRAJECTORY_SCHEMA,
    V0_INTERCEPT,
    FitState,
    PhysicalModelResult,
    back_transform,
    build_design,
    compute_physical_model,
    driver_column,
    fit_nig_ar1,
    log_evidence,
    nig_posterior,
    prais_winsten_whiten,
    select_lambda,
    standardize,
    summary_payload,
)

REQUIRED_SUMMARY_KEYS = {
    "interpretation",
    "outcome",
    "n_years",
    "train_end",
    "ar1_rho",
    "lambda_",
    "sigma2_mean",
    "lags",
    "sensitivity",
    "hindcast",
}
HINDCAST_KEYS = {"train_r2", "test_rmse", "test_band_coverage", "n_test"}

# A fixed, physically-plausible truth used across recovery / hindcast tests.
TRUE_BETA = {
    "co2": 0.9,
    "ch4": 0.3,
    "n2o": 0.2,
    "aerosol": -0.5,
    "volcanic": -0.4,
    "solar": 0.1,
    "oni": 0.08,
}
TRUE_INTERCEPT = -0.2


def make_synthetic_forcings(
    n_years: int = 175,
    beta: dict[str, float] = TRUE_BETA,
    intercept: float = TRUE_INTERCEPT,
    rho: float = 0.6,
    sigma: float = 0.05,
    lags: dict[str, int] = DEFAULT_LAGS,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic annual forcings + temperature from the exact generative model.

    Builds smooth trending ERF series (1850..) and a stationary ONI, draws AR(1)
    innovations via ``np.random.default_rng(seed)``, and forms ``temp_anomaly =
    intercept + sum_i beta_i F_i(t - l_i) + eps_t``. The first ``max(lag)`` years
    carry leading NaNs in the lagged design and are complete-case-dropped by
    :func:`build_design`. Mirrors the fixture style in tests/test_coupling.py.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(1850, 1850 + n_years)
    norm = (years - years[0]) / (years[-1] - years[0])  # 0..1 ramp

    # Smooth, monotone-ish ERF proxies; aerosol negative; volcanic spiky; oni osc.
    # Independent per-driver noise breaks the otherwise near-perfect collinearity
    # of the monotone GHG ramps so the AR(1) rho and slopes are identifiable.
    def jitter(scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, n_years)

    forcing = {
        "co2": 0.5 + 2.5 * norm**1.5 + jitter(0.20),
        "ch4": 0.2 + 0.6 * norm + jitter(0.10),
        "n2o": 0.05 + 0.25 * norm + jitter(0.05),
        "aerosol": -0.3 - 0.9 * norm + jitter(0.15),
        "volcanic": -0.2 * (np.sin(2 * np.pi * 3 * norm) ** 2) + jitter(0.05),
        "solar": 0.05 * np.sin(2 * np.pi * 5 * norm) + jitter(0.03),
        "oni": np.sin(2 * np.pi * 9 * norm) + jitter(0.15),
    }

    # AR(1) errors with stationary initial variance.
    u = rng.normal(0.0, sigma, n_years)
    eps = np.empty(n_years)
    eps[0] = u[0] / np.sqrt(1.0 - rho**2)
    for t in range(1, n_years):
        eps[t] = rho * eps[t - 1] + u[t]

    temp = np.full(n_years, float(intercept))
    for driver, b in beta.items():
        lagged = pd.Series(forcing[driver]).shift(lags[driver]).to_numpy()
        temp = temp + b * np.nan_to_num(lagged, nan=0.0)
    temp = temp + eps
    # Years that depended on a pre-sample lag are not valid observations.
    max_lag = max(lags.values())
    temp[:max_lag] = np.nan

    data = {"year": years, "temp_anomaly": temp, "temp_uncertainty": 0.05}
    erf = ("co2", "ch4", "n2o", "aerosol", "volcanic", "solar")
    for driver in erf:
        data[f"erf_{driver}"] = forcing[driver]
    data["erf_total"] = sum(forcing[d] for d in erf)
    data["oni"] = forcing["oni"]
    return pd.DataFrame(data)


class TestWhitener:
    def test_first_row_variance_scaling(self):
        m = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        rho = 0.7
        w = prais_winsten_whiten(m, rho)
        assert w[0] == pytest.approx(3.0 * np.sqrt(1.0 - rho**2))
        assert w[1] == pytest.approx(1.0 - rho * 3.0)

    def test_works_on_2d_rowwise(self):
        m = np.arange(12.0).reshape(6, 2)
        rho = 0.4
        w = prais_winsten_whiten(m, rho)
        assert w.shape == m.shape
        np.testing.assert_allclose(w[0], m[0] * np.sqrt(1.0 - rho**2))
        np.testing.assert_allclose(w[2], m[2] - rho * m[1])

    def test_whitened_ar1_is_approximately_white(self):
        rng = np.random.default_rng(1)
        rho = 0.8
        n = 4000
        u = rng.normal(0, 1, n)
        e = np.empty(n)
        e[0] = u[0] / np.sqrt(1 - rho**2)
        for t in range(1, n):
            e[t] = rho * e[t - 1] + u[t]
        w = prais_winsten_whiten(e, rho)
        # lag-1 autocorrelation of the whitened series should be near zero
        r1 = (w[1:] @ w[:-1]) / (w @ w)
        assert abs(r1) < 0.05


class TestNIGPosterior:
    def _design(self, n=50, p=3, seed=0):
        rng = np.random.default_rng(seed)
        x = np.hstack([np.ones((n, 1)), rng.normal(size=(n, p))])
        y = rng.normal(size=n)
        return x, y

    def test_lambda_n_identity(self):
        x, y = self._design()
        lam = 2.5
        m_n, v_n, alpha_n, beta_n = nig_posterior(x, y, lam)
        p1 = x.shape[1]
        prior_prec = np.full(p1, lam)
        prior_prec[0] = 1.0 / V0_INTERCEPT
        lambda_n_expected = np.diag(prior_prec) + x.T @ x
        # V_n is the inverse of Lambda_n: V_n @ Lambda_n == I
        np.testing.assert_allclose(v_n @ lambda_n_expected, np.eye(p1), atol=1e-9)

    def test_shapes_and_alpha(self):
        x, y = self._design(n=40, p=4)
        m_n, v_n, alpha_n, beta_n = nig_posterior(x, y, 1.0)
        assert m_n.shape == (5,)
        assert v_n.shape == (5, 5)
        assert alpha_n == pytest.approx(ALPHA0 + 40 / 2.0)
        assert beta_n > 0.0

    def test_mean_solves_ridge_normal_equations(self):
        x, y = self._design(n=60, p=3, seed=2)
        lam = 3.0
        m_n, _, _, _ = nig_posterior(x, y, lam)
        prior_prec = np.full(x.shape[1], lam)
        prior_prec[0] = 1.0 / V0_INTERCEPT
        # (Lambda0 + X'X) m_n == X'y
        lhs = (np.diag(prior_prec) + x.T @ x) @ m_n
        np.testing.assert_allclose(lhs, x.T @ y, atol=1e-8)


class TestEvidenceLambda:
    def test_log_evidence_finite(self):
        df = make_synthetic_forcings(seed=3)
        years, x_raw, y, train_mask = build_design(df)
        x_std, _, _ = standardize(x_raw, train_mask)
        xw = prais_winsten_whiten(x_std[train_mask], 0.5)
        yw = prais_winsten_whiten(y[train_mask], 0.5)
        assert np.isfinite(log_evidence(xw, yw, 1.0))

    def test_select_lambda_in_bounds_and_deterministic(self):
        df = make_synthetic_forcings(seed=4)
        years, x_raw, y, train_mask = build_design(df)
        x_std, _, _ = standardize(x_raw, train_mask)
        xw = prais_winsten_whiten(x_std[train_mask], 0.3)
        yw = prais_winsten_whiten(y[train_mask], 0.3)
        lam1 = select_lambda(xw, yw)
        lam2 = select_lambda(xw, yw)
        assert 10**-6 <= lam1 <= 10**6
        assert lam1 == lam2  # byte-identical across repeated calls


class TestRecovery:
    def test_recovers_rho_slopes_and_intercept(self):
        # Large sample, small noise -> tight recovery of the generative truth.
        df = make_synthetic_forcings(n_years=400, rho=0.5, sigma=0.02, seed=7)
        years, x_raw, y, train_mask = build_design(df)
        x_std, mu, s = standardize(x_raw, train_mask)
        fit = fit_nig_ar1(x_std, y, train_mask)
        assert fit.rho == pytest.approx(0.5, abs=0.1)

        sensitivity, intercept = back_transform(
            fit.m_n, fit.v_n, fit.alpha_n, fit.beta_n, mu, s
        )
        for driver, true_b in TRUE_BETA.items():
            sens = sensitivity[driver]
            # within a few posterior sd of truth
            assert abs(sens["mean"] - true_b) <= 5.0 * sens["sd"] + 0.05
        assert intercept == pytest.approx(TRUE_INTERCEPT, abs=0.15)


class TestStationarity:
    def test_near_unit_root_stays_stationary(self):
        df = make_synthetic_forcings(n_years=300, rho=0.999, sigma=0.03, seed=9)
        _, result = compute_physical_model(df)
        assert abs(result.ar1_rho) < 1.0


class TestHindcast:
    def test_metrics_well_formed(self):
        df = make_synthetic_forcings(seed=11)
        trajectory, result = compute_physical_model(df)
        h = result.hindcast
        assert h["n_test"] == int((trajectory["year"] > TRAIN_END).sum())
        assert h["n_test"] > 0
        assert np.isfinite(h["test_rmse"]) and h["test_rmse"] > 0.0
        assert 0.0 <= h["test_band_coverage"] <= 1.0

    def test_mean_coverage_near_nominal(self):
        coverages = []
        for seed in range(8):
            df = make_synthetic_forcings(n_years=220, rho=0.5, sigma=0.05, seed=seed)
            _, result = compute_physical_model(df)
            coverages.append(result.hindcast["test_band_coverage"])
        assert np.mean(coverages) >= 0.90  # nominal 0.95, deterministic per seed


class TestDeterminism:
    def test_summary_json_byte_identical(self):
        df = make_synthetic_forcings(seed=5)
        _, r1 = compute_physical_model(df)
        _, r2 = compute_physical_model(df)
        s1 = json.dumps(summary_payload(r1), indent=2) + "\n"
        s2 = json.dumps(summary_payload(r2), indent=2) + "\n"
        assert s1 == s2

    def test_trajectory_arrays_identical(self):
        df = make_synthetic_forcings(seed=6)
        t1, _ = compute_physical_model(df)
        t2, _ = compute_physical_model(df)
        pd.testing.assert_frame_equal(t1, t2)


class TestSchemas:
    def test_trajectory_round_trips_typed_parquet(self, tmp_path):
        df = make_synthetic_forcings(seed=12)
        trajectory, _ = compute_physical_model(df)
        path = tmp_path / "physical_trajectory.parquet"
        write_typed_parquet(trajectory, path, TRAJECTORY_SCHEMA, order_by=("year",))
        loaded = pd.read_parquet(path)
        assert list(loaded.columns) == list(TRAJECTORY_SCHEMA)
        assert str(loaded["year"].dtype).startswith("int")
        assert str(loaded["predicted_mean"].dtype) == "float64"
        assert loaded["in_sample"].dtype == bool

    def test_summary_keys_and_interpretation(self):
        df = make_synthetic_forcings(seed=13)
        _, result = compute_physical_model(df)
        payload = summary_payload(result)
        assert REQUIRED_SUMMARY_KEYS <= set(payload)
        assert "predictive association" in payload["interpretation"].lower()
        assert set(payload["sensitivity"]) == set(DRIVERS)
        assert set(payload["lags"]) == set(DRIVERS)
        assert set(payload["hindcast"]) == HINDCAST_KEYS
        for sens in payload["sensitivity"].values():
            assert set(sens) == {"mean", "sd", "ci_low", "ci_high"}


class TestCrossCheck:
    def test_glsar_recovers_rho_and_slopes(self):
        sm = pytest.importorskip("statsmodels.api")
        df = make_synthetic_forcings(n_years=400, rho=0.5, sigma=0.02, seed=21)
        years, x_raw, y, train_mask = build_design(df)
        x_std, mu, s = standardize(x_raw, train_mask)
        fit = fit_nig_ar1(x_std, y, train_mask)

        # statsmodels GLSAR on the same standardized training design.
        model = sm.GLSAR(y[train_mask], x_std[train_mask], rho=1)
        res = model.iterative_fit(maxiter=50)
        assert float(res.model.rho[0]) == pytest.approx(fit.rho, abs=0.1)
        # MLE (GLSAR) vs ridge-NIG posterior mean: loose agreement on slopes.
        np.testing.assert_allclose(res.params, fit.m_n, atol=0.15)


class TestResultCheck:
    def _result(self, **overrides):
        base = dict(
            outcome="global_temp_anomaly",
            n_years=100,
            train_end=TRAIN_END,
            ar1_rho=0.5,
            lambda_=1.0,
            sigma2_mean=0.01,
            lags=dict(DEFAULT_LAGS),
            sensitivity={d: {"mean": 0.0, "sd": 1.0, "ci_low": -1.0, "ci_high": 1.0} for d in DRIVERS},
            hindcast={"train_r2": 0.9, "test_rmse": 0.1, "test_band_coverage": 0.95, "n_test": 11},
        )
        base.update(overrides)
        return PhysicalModelResult(**base)

    def test_accepts_valid(self):
        self._result().check()

    def test_rejects_coverage_above_one(self):
        bad = self._result(hindcast={"train_r2": 0.9, "test_rmse": 0.1, "test_band_coverage": 1.5, "n_test": 11})
        with pytest.raises(AssertionError, match="coverage"):
            bad.check()

    def test_rejects_nonstationary_rho(self):
        with pytest.raises(AssertionError, match="stationary"):
            self._result(ar1_rho=1.0).check()


class TestBuildDesign:
    def test_rejects_year_gap(self):
        df = make_synthetic_forcings(seed=1)
        df = df.drop(index=100).reset_index(drop=True)  # punch an internal gap
        with pytest.raises(ValueError, match="contiguous"):
            build_design(df)

    def test_rejects_interior_nan_induced_gap(self):
        # Years stay contiguous, but a lone interior NaN in a driver column drops
        # one row in complete-case filtering -- which must still fail loudly so the
        # AR(1) whitening never treats the resulting 2-year jump as one annual step.
        df = make_synthetic_forcings(seed=1)
        df.loc[100, "erf_aerosol"] = np.nan
        with pytest.raises(ValueError, match="contiguous"):
            build_design(df)

    def test_driver_column_mapping(self):
        assert driver_column("co2") == "erf_co2"
        assert driver_column("oni") == "oni"

    def test_fitstate_is_frozen(self):
        fs = FitState(
            m_n=np.zeros(8), v_n=np.eye(8), alpha_n=10.0, beta_n=1.0,
            rho=0.5, lam=1.0, mu=np.zeros(7), s=np.ones(7), n_train=100,
        )
        with pytest.raises(Exception):
            fs.rho = 0.9  # type: ignore[misc]
