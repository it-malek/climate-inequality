"""Synthetic boundary regressions and independent fixed-rho predictive checks."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from src.explain import morans_i
from src.physical_model import FitState, predict_trajectory
from src.vulnerability import (
    _direct_triple_verdict,
    _spearman_gradient,
    _triple_inequality_verdict,
)


@pytest.mark.parametrize('verdict,sign', [
    (_triple_inequality_verdict, 1), (_direct_triple_verdict, -1),
])
@pytest.mark.parametrize('leg', ['area', 'responsibility'])
@pytest.mark.parametrize('key', ['rho', 'p_permutation'])
@pytest.mark.parametrize('bad', [None, np.nan, np.inf, -np.inf])
def test_missing_evidence_is_not_support(verdict, sign, leg, key, bad):
    resp = {'rho': sign * .9, 'p_permutation': .01, 'n': 12}
    area = {'rho': .1, 'p_permutation': .7, 'n': 12}
    (area if leg == 'area' else resp)[key] = bad
    result = verdict(resp, area)
    assert result['holds'] is False
    assert result['status'] == 'insufficient_evidence'
    assert any(leg in reason and key in reason for reason in result['reasons'])


@pytest.mark.parametrize('verdict,sign', [
    (_triple_inequality_verdict, 1), (_direct_triple_verdict, -1),
])
def test_finite_verdict_controls(verdict, sign):
    resp = {'rho': sign * .9, 'p_permutation': .01}
    area = {'rho': .1, 'p_permutation': .7}
    assert verdict(resp, area)['holds'] is True
    resp['p_permutation'] = .8
    assert verdict(resp, area)['holds'] is False


@pytest.mark.parametrize('values,n', [
    ([np.nan] * 5, 0), ([1, 2, np.nan, np.inf, -np.inf], 2), ([2] * 5, 5),
])
def test_gradient_unavailable_has_finite_pair_count_and_reason(values, n):
    result = _spearman_gradient(pd.Series(values), pd.Series(range(5)), 19)
    assert result['rho'] is None and result['p_permutation'] is None
    assert result['n'] == n
    assert result['reason']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('resid,coords,k', [
    (np.ones(5), np.arange(5.), 2),
    (np.array([1.]), np.array([0.]), 1),
    (np.array([]), np.array([]), 1),
    (np.arange(5.), np.arange(5.), 0),
    (np.arange(5.), np.arange(5.), -1),
    (np.array([1., np.nan, 3.]), np.arange(3.), 1),
    (np.arange(3.), np.array([1., np.inf, 3.]), 1),
])
def test_undefined_moran_has_no_numerical_p(resid, coords, k):
    result = morans_i(resid, coords, coords, k=k, n_permutations=19)
    statistic, p_value = result
    assert statistic is None and p_value is None
    assert result.reason and result.status == 'unavailable'


def test_moran_valid_case_independent_dense_weight_control():
    # Unique distances, one-dimensional equatorial geometry; exact old RNG order.
    lon = np.array([0., 1., 4., 10., 22.])
    residuals = np.array([2., -1., 4., 0., -5.])
    distances = abs(lon[:, None] - lon[None, :])
    np.fill_diagonal(distances, np.inf)
    w = np.zeros((5, 5))
    w[np.arange(5)[:, None], np.argsort(distances, axis=1)[:, :2]] = .5
    z = residuals - residuals.mean()
    observed = z @ w @ z / (z @ z)
    rng = np.random.default_rng(13)
    permuted = [rng.permutation(z) for _ in range(39)]
    expected_p = (1 + sum(abs(p @ w @ p / (p @ p)) >= abs(observed)
                          for p in permuted)) / 40
    actual, p = morans_i(residuals, lon, np.zeros(5), k=2,
                         n_permutations=39, seed=13)
    assert actual == pytest.approx(observed)
    assert p == expected_p


def forecast_case(rho, future, horizons=(1, 2, 7)):
    fit = FitState(np.array([.3, -.2]), np.array([[.2, .04], [.04, .4]]),
                   6., 5., rho, 1., np.zeros(0), np.ones(0), 1)
    x = np.array([[1., 2.], *[future for _ in horizons]])
    years = np.array([2000, *[2000 + h for h in horizons]])
    y = np.array([.8, *[0. for _ in horizons]])
    mask = np.array([True, *[False for _ in horizons]])
    return fit, x, y, years, mask


@pytest.mark.parametrize('rho', [0., .8, -.8, .999999])
@pytest.mark.parametrize('future', [[1., 2.], [1., -2.]])
def test_forecast_scale_against_expanded_covariance(rho, future):
    fit, x, y, years, mask = forecast_case(rho, future)
    rows = predict_trajectory(fit, x, y, years, mask, train_end=2000)
    for i, h in enumerate([0, 1, 2, 7]):
        if i == 0:
            c = 1 / (1 - rho**2)
            param = x[0] @ fit.v_n @ x[0]
            mean = x[0] @ fit.m_n
        else:
            # Expanded covariance of x_h beta + rho^h (y_T - x_T beta).
            a = rho**h
            param = (x[i] @ fit.v_n @ x[i]
                     + a*a * (x[0] @ fit.v_n @ x[0])
                     - 2*a * (x[i] @ fit.v_n @ x[0]))
            c = sum(rho**(2*j) for j in range(h))
            mean = x[i] @ fit.m_n + a * (y[0] - x[0] @ fit.m_n)
        scale2 = fit.beta_n / fit.alpha_n * (c + param)
        half = (rows.upper95[i] - rows.lower95[i]) / 2
        assert (half / t.ppf(.975, 2*fit.alpha_n))**2 == pytest.approx(scale2)
        assert rows.predicted_mean[i] == pytest.approx(mean)


@pytest.mark.parametrize('rho,future', [(.8, [1., 2.]), (.8, [1., -2.]),
                                      (-.6, [1., 2.]), (0., [1., 2.])])
def test_forecast_matches_hierarchical_simulation(rho, future):
    fit, x, y, years, mask = forecast_case(rho, future, horizons=(1, 4))
    rows = predict_trajectory(fit, x, y, years, mask, train_end=2000)
    rng = np.random.default_rng(291)
    draws = 100_000
    sigma2 = 1 / rng.gamma(fit.alpha_n, 1 / fit.beta_n, draws)
    beta = fit.m_n + (rng.normal(size=(draws, 2)) @ np.linalg.cholesky(fit.v_n).T
                     * np.sqrt(sigma2[:, None]))
    residual = y[0] - beta @ x[0]
    for h in range(1, 5):
        # Recursion in the generative AR(1), independent of the closed form.
        residual = rho * residual + rng.normal(size=draws) * np.sqrt(sigma2)
        if h not in (1, 4):
            continue
        i = 1 if h == 1 else 2
        sample = beta @ x[i] + residual
        scale2 = ((rows.upper95[i] - rows.lower95[i]) /
                  (2*t.ppf(.975, 2*fit.alpha_n)))**2
        variance = scale2 * fit.alpha_n / (fit.alpha_n - 1)
        assert np.var(sample) == pytest.approx(variance, rel=.025)
        assert np.mean((sample >= rows.lower95[i]) &
                       (sample <= rows.upper95[i])) == pytest.approx(.95, abs=.003)


# Reuse only synthetic frames; neither fixture loads data files.
pytest_plugins = ["tests.test_vulnerability"]


@pytest.mark.parametrize('leg', ['area', 'responsibility'])
@pytest.mark.parametrize('values,n', [
    ([np.nan] * 8, 0), ([.1, .2] + [np.nan] * 6, 2),
    ([.2] * 8, 8), ([np.inf] * 8, 0),
])
def test_evidence_state_reaches_summary_and_rendered_view(scenario, ndgain, leg, values, n):
    from streamlit.testing.v1 import AppTest
    from src.vulnerability import AREA_COL, RESPONSIBILITY_COL, compute_vulnerability

    frame, income = scenario
    frame = frame.copy()
    frame[AREA_COL if leg == 'area' else RESPONSIBILITY_COL] = values
    gain, iso = ndgain
    _, summary = compute_vulnerability(frame, income, gain, iso, n_permutations=19)
    summary['available'] = True
    summary = json.loads(json.dumps(summary, allow_nan=False))
    for verdict in (summary['triple_inequality'], summary['ndgain']['triple_inequality_direct']):
        assert verdict['status'] == 'insufficient_evidence'
        assert verdict['holds'] is False
        assert verdict[f'{leg}_warming_gradient' if leg == 'area'
                       else 'responsibility_gradient']['n'] == n

    def page(summary):
        from unittest.mock import patch
        from app.views.vulnerability import render
        with patch('app.loaders.load_vulnerability_summary', return_value=summary), \
             patch('app.loaders.load_vulnerability_strata', return_value=None):
            render()

    app = AppTest.from_function(page, args=(summary,)).run()
    assert not app.exception
    assert sum(metric.value == 'Insufficient evidence' for metric in app.metric) == 2
    assert len(app.warning) == 2
    assert all('Insufficient evidence' in warning.value for warning in app.warning)


def test_moran_rejects_unusable_neighbor_matrix(monkeypatch):
    monkeypatch.setattr('src.explain.knn_indices', lambda *a, **k: np.zeros((4, 3), int))
    result = morans_i(np.arange(4.), np.arange(4.), np.zeros(4), k=2)
    assert tuple(result) == (None, None)
    assert result.reason == 'unusable_neighbors'


def test_constant_moran_state_reaches_country_summary(monkeypatch):
    from src.feature_schema import SCHEMA_V1
    from src.stability import _residual_spatial
    monkeypatch.setattr('src.stability._full_model_residuals',
                        lambda *args: (np.array(['A', 'B', 'C', 'D']), np.ones(4)))
    centroids = pd.DataFrame({'Country': ['A', 'B', 'C', 'D'],
                             'Longitude': [0., 1., 3., 8.], 'Latitude': [0.] * 4})
    result = _residual_spatial(pd.DataFrame(), centroids, SCHEMA_V1, 'ok',
                               k=2, n_permutations=19, seed=0)
    assert result['morans_i'] is None and result['p_value'] is None
    assert result['reason'] == 'constant_residuals'
    assert result['status'] == 'unavailable'
    json.dumps(result, allow_nan=False)


def test_unavailable_city_moran_survives_serialization():
    from src.explain import CityModelResult, build_explain_summary
    result = CityModelResult('full', 'y ~ x', 5, 1., moran_reason='constant_residuals')
    summary = build_explain_summary([result], [])['city_model']
    assert summary['moran_i_full'] is None and summary['moran_p_full'] is None
    assert summary['moran_reason_full'] == 'constant_residuals'
    assert summary['specs'][0]['moran_status'] == 'unavailable'


def test_vulnerability_legacy_summary_remains_readable():
    from app.views.vulnerability import _verdict_label
    assert _verdict_label({'holds': True}) == 'Criterion met'
    assert _verdict_label({'holds': False}) == 'Not supported'


def test_bundle_cli_reports_insufficient_evidence_without_building(scenario):
    from src.app_assets import vulnerability_report
    from src.vulnerability import AREA_COL, compute_vulnerability
    frame, income = scenario
    frame = frame.copy()
    frame[AREA_COL] = np.nan
    _, summary = compute_vulnerability(frame, income, n_permutations=19)
    text = vulnerability_report(summary)
    assert 'insufficient_evidence' in text
    assert 'insufficient_finite_pairs' in text
    assert 'n=0' in text


def test_station_comparison_preserves_unavailable_moran_reason():
    from app.views.sensitivity import _comparison_table
    primary = {'share_stability': {'groups': {}},
               'residual_spatial': {'morans_i': .2}}
    station = {'share_stability': {'groups': {}},
               'residual_spatial': {'morans_i': None, 'reason': 'constant_residuals'}}
    table = _comparison_table(primary, station)
    assert table.loc['Moran availability'].iloc[1] == 'constant_residuals'
    primary['residual_spatial']['morans_i'] = None
    table = _comparison_table(primary, station)
    assert table.loc['Moran availability'].iloc[0] == 'unavailable (reason not recorded)'


@pytest.mark.parametrize('alpha', [.75, 6.])
def test_conditional_design_cancellation_leaves_only_innovations(alpha):
    from dataclasses import replace
    fit, x, y, years, mask = forecast_case(.8, [0., 0.], horizons=(3,))
    fit = replace(fit, alpha_n=alpha)
    x[1] = fit.rho**3 * x[0]
    rows = predict_trajectory(fit, x, y, years, mask, train_end=2000)
    half = (rows.upper95[1] - rows.lower95[1]) / 2
    scale2 = fit.beta_n / alpha * sum(fit.rho**(2*j) for j in range(3))
    assert half == pytest.approx(t.ppf(.975, 2*alpha) * np.sqrt(scale2))


def test_forecast_horizon_limit():
    fit, x, y, years, mask = forecast_case(.5, [1., -2.], horizons=(100,))
    rows = predict_trajectory(fit, x, y, years, mask, train_end=2000)
    scale2 = fit.beta_n / fit.alpha_n * (1/(1-fit.rho**2) + x[1] @ fit.v_n @ x[1])
    assert rows.predicted_mean[1] == pytest.approx(x[1] @ fit.m_n)
    assert rows.upper95[1] - rows.predicted_mean[1] == pytest.approx(
        t.ppf(.975, 2*fit.alpha_n) * np.sqrt(scale2))
