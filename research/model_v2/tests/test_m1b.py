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


def test_station_support_audits_valid_cells_only():
    stn = np.zeros((360, 1, 3))
    stn[:10, 0, 1] = 3
    stn[:, 0, 2] = np.nan                      # CRU non-land cell: fill value, invalid for C2
    valid = np.array([[True, True, False]])
    supported, mean_stn, pure = h.support_quantities(stn, valid)
    assert pure.tolist() == [[True, False, False]] and supported[0, 1] == pytest.approx(10 / 360)
    assert mean_stn[0, 1] == pytest.approx(30 / 360) and np.isnan(supported[0, 2])
    for bad in (9, -1, 0.5, np.nan):
        broken = stn.copy()
        broken[0, 0, 0] = bad                  # inside a valid cell: must stop
        with pytest.raises(ValueError):
            h.support_quantities(broken, valid)


def test_support_nesting_accepts_the_gpw_grid_and_rejects_flips_and_shifts():
    lat_north_to_south = 89.875 - 0.25 * np.arange(720)       # GPW storage order
    lon = -179.875 + 0.25 * np.arange(1440)
    h.check_support_nesting(lat_north_to_south, lon)
    for bad_lat, bad_lon in [(lat_north_to_south[::-1], lon), (lat_north_to_south + 0.125, lon),
                             (lat_north_to_south, lon + 0.25), (lat_north_to_south[:-1], lon)]:
        with pytest.raises(ValueError):
            h.check_support_nesting(bad_lat, bad_lon)


def test_pinned_source_fails_closed_on_size_or_hash_mismatch(tmp_path, monkeypatch):
    import gzip
    import hashlib
    payload = b'not really netcdf'
    gz = tmp_path / 'x.nc.gz'
    with gzip.GzipFile(gz, 'wb', mtime=0) as f:
        f.write(payload)
    monkeypatch.setattr(h, 'CRU_DIR', tmp_path)
    spec = {'path': 'var/x.nc.gz', 'bytes': gz.stat().st_size, 'gz_sha256': hashlib.sha256(gz.read_bytes()).hexdigest(),
            'nc_bytes': len(payload), 'nc_sha256': hashlib.sha256(payload).hexdigest()}
    nc, record = h.pinned_source(spec)
    assert nc.read_bytes() == payload and record['nc_sha256'] == spec['nc_sha256']
    for key, wrong in [('bytes', spec['bytes'] + 1), ('gz_sha256', '0' * 64), ('nc_bytes', 1), ('nc_sha256', '0' * 64)]:
        with pytest.raises(ValueError):
            h.pinned_source({**spec, key: wrong})


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
