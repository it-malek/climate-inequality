"""M1b C2 construction, redundancy stops and the two-level verdict on synthetic fixtures."""
import numpy as np
import pandas as pd
import pytest

from research.model_v2 import m1b_hydroclimate as h


def test_window_is_exactly_the_contract_months_with_calendar_days():
    times = pd.date_range('1901-01-16', '2025-12-16', freq='MS') + pd.Timedelta(days=15)
    idx, days = h.window_months(times.to_numpy())
    assert len(idx) == 360 and pd.DatetimeIndex(times[idx])[0].strftime('%Y-%m') == '1920-01'
    assert pd.DatetimeIndex(times[idx])[-1].strftime('%Y-%m') == '1949-12'
    assert days.sum() == 30 * 365 + 8  # leap Februaries 1920..1948
    with pytest.raises(ValueError):
        h.window_months(times[:300].to_numpy())


def test_cell_climatology_formula_validity_and_no_floor():
    days = np.asarray(pd.period_range('1920-01', '1949-12', freq='M').days_in_month, dtype=float)
    pre = np.full((360, 2, 2), 50.0)
    pet = np.full((360, 2, 2), 2.0)
    pre[:, 0, 1] = 0.0          # zero precipitation: invalid, never floored
    pet[5, 1, 0] = np.nan       # one missing month: invalid
    p_bar, e_bar, x, valid, reason = h.cell_climatology(pre, pet, days)
    assert p_bar[0, 0] == pytest.approx(600.0)
    assert e_bar[0, 0] == pytest.approx(2.0 * (30 * 365 + 8) / 30)
    assert x[0, 0] == pytest.approx(np.log10(600.0 / e_bar[0, 0]))
    assert valid.tolist() == [[True, False], [False, True]]
    assert reason[0, 1] == 'P_bar <= 0' and reason[1, 0] == 'non-finite month'


def test_country_value_is_log_of_area_weighted_geometric_mean_and_coverage():
    land = np.zeros((720, 1440))
    country = np.full((720, 1440), -1, dtype=np.int16)
    land[0:2, 0:2] = [[1.0, 1.0], [1.0, 1.0]]      # CRU cell (0, 0): area 4
    land[0:2, 2:4] = [[3.0, 0.0], [0.0, 0.0]]      # CRU cell (0, 1): area 3
    land[2, 0] = 1.0                               # CRU cell (1, 0): area 1, invalid below
    country[0:3, 0:4] = 0
    cells = h.country_cells(['AAA'], land, country)
    x = np.full((360, 720), np.nan)
    valid = np.zeros((360, 720), bool)
    ai = {(0, 0): 0.5, (0, 1): 2.0}
    for (i, k), value in ai.items():
        x[i, k], valid[i, k] = np.log10(value), True
    reason = np.full((360, 720), 'non-finite month', dtype=object)
    zeros = np.zeros((360, 720))
    qa = h.aggregate(cells, x, valid, reason, zeros + 1, zeros + 1, zeros + 1, zeros + 2, np.zeros((360, 720), bool), 1)
    geometric = np.exp((4 * np.log(0.5) + 3 * np.log(2.0)) / 7)
    assert qa.baseline_dryness[0] == pytest.approx(np.log10(geometric))
    assert qa.coverage[0] == pytest.approx(7 / 8)
    assert qa.terrestrial_area_km2[0] == pytest.approx(8.0)


def test_station_support_rejects_invalid_counts_and_flags_pure_climatology():
    stn = np.zeros((360, 1, 2))
    stn[:10, 0, 1] = 3
    supported, mean_stn, pure = h.support_quantities(stn)
    assert pure.tolist() == [[True, False]] and supported[0, 1] == pytest.approx(10 / 360)
    for bad in (9, -1, 0.5, np.nan):
        broken = stn.copy()
        broken[0, 0, 0] = bad
        with pytest.raises(ValueError):
            h.support_quantities(broken)


def test_redundancy_diagnostic_refuses_outcome_and_stops_on_exact_redundancy():
    from research.model_v2.m1b_redundancy import diagnose
    rng = np.random.default_rng(1)
    n = 60
    frame = pd.DataFrame({
        'Country': [f'C{i}' for i in range(n)], 'cum_co2_total': 10 ** rng.normal(3, 1, n),
        'abs_latitude': rng.uniform(0, 60, n), 'elevation': rng.uniform(0, 2000, n),
        'continentality': rng.uniform(0, 900, n), 'climate_zone': rng.choice(list('ABCD'), n),
        'hemisphere': rng.choice(list('NS'), n), 'spatial_block': rng.choice(['Africa', 'Asia', 'Europe'], n),
        'income_group': rng.choice(['High', 'Low'], n), 'population': 10 ** rng.normal(7, 1, n),
        'station_density': rng.uniform(0, 5, n)})
    with pytest.raises(AssertionError):
        diagnose(frame.assign(warming_trend=0.1), rng.normal(size=n))
    ok = diagnose(frame, rng.normal(size=n))
    assert ok['hard_stops'] == [] and 0 <= ok['r2_full_m0star'] < 1
    exact = diagnose(frame, 2 * frame.abs_latitude.to_numpy() - 0.01 * frame.elevation.to_numpy())
    assert any('exact rank redundancy' in s for s in exact['hard_stops'])
    assert 'zero-variance C2' in diagnose(frame, np.ones(n))['hard_stops']


def card(delta, lo, hi, coef, frac, sec=0.0389, region=0.0694):
    return {'paired_delta_vs_m0': {'delta_rmse': delta, 'country_bootstrap_95_interval': [lo, hi]},
            'coefficients': {'baseline_dryness': {'full': coef, 'same_sign_fraction': frac}},
            'secondary': {'cv_rmse': sec}, 'worst_region': {'rmse': region}}


def test_two_level_verdict():
    from research.model_v2.m1b_evaluate import verdict
    base = card(0, 0, 0, 0, 1)
    assert verdict(base, card(-0.003, -0.005, -0.001, -0.02, 0.9))['verdict'] == 'supported and promoted'
    sub = verdict(base, card(-0.001, -0.002, -0.0002, -0.02, 0.9))
    assert sub['verdict'] == 'supported but sub-material / not promoted' and not sub['A2_practical_le_minus_0.002']
    vetoed = verdict(base, card(-0.003, -0.005, -0.001, -0.02, 0.9, sec=0.0405))
    assert vetoed['verdict'] == 'supported but sub-material / not promoted' and not vetoed['A3_m49_veto_passed']
    wrong_sign = verdict(base, card(-0.003, -0.005, -0.001, 0.02, 0.9))
    assert wrong_sign['verdict'] == 'not supported' and wrong_sign['improvement_without_prestated_sign']
    unstable = verdict(base, card(-0.003, -0.005, -0.001, -0.02, 0.79))
    assert unstable['verdict'] == 'not supported'
    covers = verdict(base, card(-0.003, -0.006, 0.0001, -0.02, 0.9))
    assert covers['verdict'] == 'not supported'
    assert verdict(base, card(0.003, 0.001, 0.005, -0.02, 0.9))['worsened_generalization']


def test_extended_schema_reproduces_m0star_design_when_c2_absent():
    from research.model_v2 import cv, m0
    from research.model_v2.m1b_evaluate import M1bDesign, design_matrix
    if not (m0.DEFAULT_INEQUALITY_PATH.exists() and m0.DEFAULT_FEATURES_PATH.exists()):
        pytest.skip('requires local frozen inputs')
    frame = m0.m0_complete_design(*m0.load_inputs()).drop(columns='cum_co2_per_capita')
    np.testing.assert_array_equal(design_matrix(frame), m0.design_matrix(frame)[0])
    np.testing.assert_allclose(M1bDesign().fit(frame).predict(frame), cv.V1Design().fit(frame).predict(frame),
                               rtol=0, atol=1e-13)
    with_c2 = frame.assign(baseline_dryness=np.linspace(-1, 1, len(frame)))
    assert design_matrix(with_c2).shape[1] == design_matrix(frame).shape[1] + 1
