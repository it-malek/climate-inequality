"""Regression checks for fresh-input V2 routing and numerical encoding."""
import numpy as np
import pandas as pd
import pytest

from research.model_v2 import cv, m2_evaluate as ev, m2_feasibility as feas
from research.model_v2 import prospective_refit as pr


def frame(n=40):
    rng = np.random.default_rng(8)
    return pd.DataFrame({'iso3': [f'X{i}' for i in range(n)],
                         'cum_co2_total': np.exp(rng.normal(10, 1, n)),
                         'population': np.exp(rng.normal(8, 1, n)),
                         'station_density': np.exp(rng.normal(0, 1, n)),
                         'abs_latitude': rng.uniform(1, 70, n),
                         'elevation': rng.uniform(1, 4000, n),
                         'continentality': rng.uniform(1, 2000, n),
                         'climate_zone': np.resize(['A', 'B', 'C'], n),
                         'hemisphere': np.resize(['N', 'S'], n),
                         'income_group': np.resize(['high', 'low'], n),
                         'spatial_block': np.resize(['Asia', 'Europe'], n),
                         'warming_trend': rng.normal(0.2, 0.03, n)})


def test_fresh_logged_design_matches_raw_encoder():
    raw = frame()
    numeric = feas.transform_numerics(raw)
    fitted = ev.fit_model(ev.COMPARATOR, numeric.iloc[:30])
    reference = cv.V1Design().fit(raw.iloc[:30])
    np.testing.assert_allclose(ev.predict(fitted, numeric.iloc[30:]), reference.predict(raw.iloc[30:]), atol=1e-12)
    assert fitted.matrix_columns == reference._matrix(raw.iloc[:30])[0].shape[1]


def test_input_routing_restored_after_failure():
    original = pr.geo.KOPPEN_PATH
    with pytest.raises(RuntimeError), pr.routed(pr.geo, KOPPEN_PATH='prospective'):
        assert pr.geo.KOPPEN_PATH == 'prospective'
        raise RuntimeError('controlled failure')
    assert pr.geo.KOPPEN_PATH == original


def test_snapshot_refuses_to_overwrite_evidence(tmp_path, monkeypatch):
    (tmp_path / 'control').mkdir()
    path = tmp_path / 'control/v2_static_pre_execution.json'
    path.write_text('original attempt')
    with pytest.raises(FileExistsError):
        pr.snapshot(tmp_path, 'static')
    assert path.read_text() == 'original attempt'


def test_national_asset_retains_distinct_missingness():
    from research.model_v2.prospective_assets import national_warming

    inequality = pd.DataFrame({'Country': ['B', 'A'], 'owid_country': ['B', 'A'],
        'continent': ['Europe', 'Asia'], 'cum_co2_t_per_capita': [2., 1.],
        'trend_c_per_decade': [.2, .1], 'trend_c_per_decade_pop_weighted': [.3, .15],
        'trend_c_per_decade_area_weighted': [np.nan, .11]})
    design = pd.DataFrame({'Country': ['A'], 'iso3': ['AAA']})
    era = pd.DataFrame({'iso3': ['AAA'], 'trend_c_per_decade_era5_area': [.12]})
    result = national_warming(inequality, design, era)
    assert result.Country.tolist() == ['A', 'B']
    assert result.trend_station.notna().sum() == 2
    assert result.trend_area.notna().sum() == result.trend_era5_aligned.notna().sum() == 1
    assert result.loc[0, 'trend_era5_aligned'] == .12


def test_national_asset_includes_outcomes_outside_model_complete_cases():
    from research.model_v2.prospective_assets import national_warming

    inequality = pd.DataFrame({'Country': ['A', 'B'], 'owid_country': ['A', 'B'],
        'continent': ['Asia', 'Europe'], 'cum_co2_t_per_capita': [1., 2.],
        'trend_c_per_decade': [.1, .2], 'trend_c_per_decade_pop_weighted': [.15, .3],
        'trend_c_per_decade_area_weighted': [.11, np.nan]})
    full_identity = pd.DataFrame({'Country': ['A', 'B'], 'iso3': ['AAA', 'BBB']})
    era = pd.DataFrame({'iso3': ['AAA', 'BBB'], 'trend_c_per_decade_era5_area': [.12, .22]})
    result = national_warming(inequality, full_identity, era)
    assert result.trend_area.notna().sum() == 1
    assert result.trend_era5_aligned.notna().sum() == 2
    assert result.loc[result.Country == 'B', 'trend_era5_aligned'].item() == .22


def test_area_design_keeps_metadata_outside_strict_schema():
    from research.model_v2 import m1a_evaluate as area, m1b_evaluate as hydro

    complete = frame().drop(columns='iso3').assign(Country=[f'C{i}' for i in range(40)])
    complete['cum_co2_per_capita'] = complete.cum_co2_total / complete.population * 1e6
    metadata = complete.assign(iso3=[f'X{i}' for i in range(40)], centroid_lat=1., centroid_lon=2.,
                               m49_subregion='fixture', station_lat=3., station_lon=4.)
    features = complete[['Country', *area.REMEASURED]].copy()
    features['elevation'] *= 1.1
    result = pr.area_design(complete, features, 'primary_total_co2')
    # Exercise the exact strict validation path that rejected the real static attempt.
    shares = hydro.group_lmg_shares(result, schema=hydro.SCHEMA_M1B)
    assert np.isfinite(list(shares.shares.values())).all()
    expected = area.frames(complete, features)['primary']['M1a']
    pd.testing.assert_frame_equal(result, expected)
    assert 'centroid_lat' in metadata and 'centroid_lat' not in result
    with pytest.raises(ValueError, match='not in schema'):
        pr.area_design(metadata, features, 'primary_total_co2')
