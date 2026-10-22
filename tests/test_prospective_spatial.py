"""Synthetic undefined-statistic checks for V2 diagnostic primitives, never scoring."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from research.model_v2.spatial import gearys_c, local_morans, morans_i


@pytest.mark.parametrize('values,w,reason', [
    (np.ones(5), np.ones((5, 5)) - np.eye(5), 'constant_residuals'),
    (np.full(5, .1), np.ones((5, 5)) - np.eye(5), 'constant_residuals'),
    (np.arange(2.), np.ones((2, 2)) - np.eye(2), 'insufficient_samples'),
    (np.array([]), np.zeros((0, 0)), 'insufficient_samples'),
    (np.arange(5.), np.zeros((5, 5)), 'unusable_neighbors'),
    (np.arange(5.), np.full((5, 5), np.nan), 'unusable_neighbors'),
    (np.array([0., np.nan, 2.]), np.ones((3, 3)) - np.eye(3), 'nonfinite_values'),
])
def test_undefined_global_and_local_moran(values, w, reason):
    for func in (morans_i, gearys_c):
        result = func(values, w, 19, 0)
        assert result.statistic is None and result.p_value is None
        assert result.reason == reason and result.status == 'unavailable'
        json.dumps(asdict(result), allow_nan=False)
    local = local_morans(values, w, 19, 0)
    assert local.p_value.isna().all() and local.local_i.isna().all()
    assert (local.label == 'unavailable').all()
    assert (local.reason == reason).all()
    assert local.attrs['reason'] == reason


def test_local_isolate_does_not_become_nonsignificant():
    w = np.ones((5, 5)) - np.eye(5)
    w[0] = 0
    local = local_morans(np.arange(5.), w, 19, 0)
    assert pd.isna(local.loc[0, 'p_value'])
    assert pd.isna(local.loc[0, 'local_i'])
    assert local.loc[0, 'reason'] == 'unusable_neighbors'
    assert local.loc[0, 'label'] == 'unavailable'
    assert (local.loc[1:, 'status'] == 'available').all()
    # A global statistic can still be defined on a graph with an isolate.
    assert morans_i(np.arange(5.), w, 19, 0).status == 'available'


def test_valid_v2_diagnostics_match_original_values_exactly():
    # Recorded by executing the original Mac implementation on this synthetic case.
    values = np.array([2., -1., 4., 0., -5.])
    w = np.array([[0,.5,.5,0,0], [.5,0,.5,0,0], [0,.5,0,.5,0],
                  [0,0,.5,0,.5], [0,0,.5,.5,0]])
    result = morans_i(values, w, 39, 13)
    assert (result.statistic, result.expected, result.p_value, result.z_score) == (
        -0.2608695652173913, -.25, .625, -0.12073386164640734)
    geary = gearys_c(values, w, 39, 13)
    assert (geary.statistic, geary.p_value, geary.z_score) == (
        1.0217391304347827, 1., 0.12210313946005465)
    local = local_morans(values, w, 39, 13)
    assert list(local.p_value) == [.775, .275, 1., 1., .325]
    assert list(local.local_i) == [0.32608695652173914, -0.32608695652173914,
                                  -0.2173913043478261, -0., -1.0869565217391306]
    assert list(local.quadrant) == ['HH', 'LH', 'HL', 'HL', 'LH']
    assert (local.label == 'ns').all()


def diagnostic_table():
    return pd.DataFrame({'Country': list('ABCDE'), 'iso3': list('ABCDE'),
                         'station_lon': [0., 1., 4., 8., 12.], 'station_lat': [0.] * 5,
                         'centroid_lon': [0., 1., 4., 8., 12.], 'centroid_lat': [0.] * 5,
                         'residual': np.ones(5), 'm49_subregion': ['synthetic'] * 5})


def test_v2_diagnostic_summaries_propagate_unavailable_without_false_clusters(monkeypatch):
    from research.model_v2 import diagnostics as d
    from research.model_v2.borders import border_correlogram
    from research.model_v2.spatial import haversine_matrix
    table = diagnostic_table()
    monkeypatch.setattr(d, 'N_PERMUTATIONS', 19)
    monkeypatch.setattr(d.pd, 'read_csv', lambda *a, **kw: pd.DataFrame(columns=['iso3_a', 'iso3_b']))
    weights = d.weights_sensitivity(table, np.ones(5))
    assert weights['range_of_morans_i']['min'] is None
    assert all(row.get('p') is None for row in weights.values())
    assert weights['gearys_c_knn_k8_area']['reason'] == 'constant_residuals'
    local, summary = d.local_structure(table, np.ones(5))
    assert summary['n_significant'] == 0
    assert summary['n_available_paired'] == 0
    assert summary['label_agreement_station_vs_area_weights'] is None
    assert summary['unavailable_reasons'] == ['constant_residuals']
    assert (local.lisa_v1_status == 'unavailable').all()
    rings = d.correlogram(table, np.ones(5))['rings']
    border = border_correlogram(np.ones(5), haversine_matrix(table.station_lon, table.station_lat))
    assert rings and border
    for row in rings + border:
        assert row['morans_i'] is None and row['p'] is None
        assert row['reason'] == 'constant_residuals'


def test_fdr_preserves_unavailable_and_available_local_states():
    from research.model_v2.product_stability import local_moran_fdr
    frame = pd.DataFrame({'p_value': [np.nan, .001, .9], 'quadrant': ['unavailable', 'HH', 'LH'],
                          'status': ['unavailable', 'available', 'available'],
                          'reason': ['constant_residuals', None, None]})
    result = local_moran_fdr(frame)
    assert list(result.fdr_label) == ['unavailable', 'HH', 'ns']
    assert pd.isna(result.q_value[0]) and result.reason[0] == 'constant_residuals'
    assert result.q_value[1] == .002


def test_influence_no_permutation_statistics_carry_unavailable_reasons():
    from research.model_v2.diagnostics import moran_influence, _moran_no_perm
    table = diagnostic_table().assign(lisa_v1_label='unavailable')
    result = moran_influence(table, np.ones(5))
    assert result['full_sample_knn8_station'] is None
    assert result['jackknife_max_abs_delta'] is None
    assert result['jackknife_top10'] == []
    assert result['unavailable_reasons']['full_sample_knn8_station'] == 'constant_residuals'
    assert result['unavailable_reasons']['drop_top_5_abs_residual'] == 'insufficient_samples'
    json.dumps(result, allow_nan=False)
    # Independent complete-graph identity for nonconstant centered values.
    value, reason = _moran_no_perm(np.arange(5.), np.ones((5, 5)) - np.eye(5))
    assert value == -.25 and reason is None
