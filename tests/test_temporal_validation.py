import numpy as np
import pandas as pd
from src import physical_model as pm
from src.temporal_validation import fit_with_trace, rolling_origin, rank_uncertainty


def forcing_frame():
    rng = np.random.default_rng(17)
    years = np.arange(1950, 2025)
    frame = pd.DataFrame({'year': years, 'temp_anomaly': rng.normal(size=len(years))})
    for driver in pm.DRIVERS:
        frame[pm.driver_column(driver)] = rng.normal(size=len(years))
    return frame


def test_trace_matches_fixed_estimator_and_future_response_cannot_change_fit():
    frame = forcing_frame()
    years, raw, y, _ = pm.build_design(frame)
    mask = years <= 1990
    x, _, _ = pm.standardize(raw, mask)
    fit, trace = fit_with_trace(x, y, mask)
    old = pm.fit_nig_ar1(x, y, mask)
    np.testing.assert_array_equal(fit.m_n, old.m_n)
    np.testing.assert_array_equal(fit.v_n, old.v_n)
    assert fit.rho == old.rho and fit.lam == old.lam
    assert len(trace['iterations']) <= pm.MAX_ITERS
    y[~mask] = 1e9
    changed, changed_trace = fit_with_trace(x, y, mask)
    np.testing.assert_array_equal(fit.m_n, changed.m_n)
    assert trace == changed_trace


def test_rolling_future_changes_do_not_affect_origin_fit():
    frame = forcing_frame()
    predictions, fits, summary = rolling_origin(frame, origins=[1990], max_horizon=3)
    changed = frame.copy()
    changed.loc[changed.year > 1990, 'temp_anomaly'] += 100
    other, otherfits, _ = rolling_origin(changed, origins=[1990], max_horizon=3)
    assert fits == otherfits
    np.testing.assert_array_equal(predictions.predicted_mean, other.predicted_mean)
    assert predictions.horizon.tolist() == [1, 2, 3]
    assert [s['n'] for s in summary] == [1, 1, 1]


def test_pair_specific_count_and_undefined_draws_preserved():
    frame = pd.DataFrame({'Country': list('abcde'), 'continent': list('XXYYZ'),
                          'x': [1, 2, 3, 4, np.nan], 'y': [4, 3, 2, 1, 0]})
    result, draws = rank_uncertainty(frame, 'x', 'y', n=40)
    assert result['n'] == 4 and result['blocks'] == 2 and result['rho'] == -1
    assert len(draws) == 80
    for record in result['bootstrap'].values():
        assert record['valid'] + record['undefined'] == 40
