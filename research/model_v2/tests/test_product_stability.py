"""Paired-product audit invariants, independent of the full spatial run."""
import numpy as np
import pandas as pd

from research.model_v2.product_stability import bh_adjust, paired_refit, classify_region


def test_paired_refit_restricts_before_fitting_and_preserves_projection():
    x = np.column_stack([np.ones(5), np.arange(5)])
    b = np.array([1., 4., 2., 5., 99.])
    e = np.array([2., 3., 4., 8., np.nan])
    mask, rb, re = paired_refit(x, b, e)
    assert mask.tolist() == [True, True, True, True, False]
    xc = x[mask]
    np.testing.assert_allclose(xc.T @ rb, 0, atol=1e-12)
    np.testing.assert_allclose(xc.T @ re, 0, atol=1e-12)
    gap = b[mask] - e[mask]
    np.testing.assert_allclose(rb - re, gap - xc @ np.linalg.lstsq(xc, gap, rcond=None)[0])


def test_bh_adjust_restores_order_and_handles_missing():
    np.testing.assert_allclose(bh_adjust([.04, .001, .03, np.nan]), [.04, .003, .04, np.nan])


def test_region_criteria_distinguish_reversal_from_small_uncertainty():
    assert classify_region(pd.Series(dict(n=5, b_mean=.03, e_mean=.02, sign_agreement=.8)), .01, .01) == 'product-robust'
    assert classify_region(pd.Series(dict(n=5, b_mean=.03, e_mean=-.02, sign_agreement=.2)), .01, .01) == 'product-sensitive'
    assert classify_region(pd.Series(dict(n=2, b_mean=.03, e_mean=.02, sign_agreement=1.)), .01, .01) == 'unresolved'


def test_region_classification_is_symmetric_for_one_material_product():
    row = pd.Series(dict(n=5, b_mean=.001, e_mean=.03, sign_agreement=.8))
    assert classify_region(row, .01, .02) == 'product-sensitive'
    swapped = pd.Series(dict(n=5, b_mean=.03, e_mean=.001, sign_agreement=.8))
    assert classify_region(swapped, .02, .01) == 'product-sensitive'


def test_lisa_overlap_keeps_outliers_separate_and_handles_empty_union():
    from research.model_v2.product_stability import lisa_overlap
    data = {'iso3': ['AAA', 'BBB']}
    for w in ['station_knn8', 'area_knn8']:
        for inference in ['label', 'fdr_label']:
            data[f'{w}_b_residual_{inference}'] = ['HH', 'HL']
            data[f'{w}_e_residual_{inference}'] = ['HH', 'LH']
    result = lisa_overlap(pd.DataFrame(data))
    assert result['station_knn8_label_HH']['intersection'] == ['AAA']
    assert result['station_knn8_label_HL']['n_overlap'] == 0
    assert result['station_knn8_label_LL']['jaccard'] is None


def test_monthly_normalization_changes_seasonal_sen_but_constant_offset_does_not():
    import xarray as xr
    from scipy.stats import theilslopes
    from research.model_v2.product_stability import monthly_anomalies
    from src.cleaning import to_decimal_decades

    dates = pd.date_range('1950-01-01', '2013-09-01', freq='MS')
    t = to_decimal_decades(pd.Series(dates)).to_numpy()
    y = .2 * t + 20 * np.sin(2 * np.pi * (dates.month.to_numpy() - 1) / 12)
    y += np.random.default_rng(37).normal(0, .2, len(y))
    data = xr.DataArray(y, dims='time', coords={'time': dates})
    aligned = monthly_anomalies(data)
    baseline = aligned.sel(time=slice('1951', '1980')).groupby('time.month').mean()
    np.testing.assert_allclose(baseline, 0, atol=1e-12)
    raw_slope = theilslopes(y, t).slope
    assert abs(theilslopes(y - 273.15, t).slope - raw_slope) < 1e-12
    assert abs(theilslopes(aligned.to_numpy(), t).slope - raw_slope) > 1e-5
    assert abs(theilslopes(aligned.to_numpy(), t).slope - .2) < .01


def test_pattern_evidence_requires_strong_replication_and_agreeing_arms():
    from research.model_v2.product_stability import combine_arms, evidence_class

    assert evidence_class('strong', 'strong') == 'product-robust'
    assert evidence_class('strong', 'none') == evidence_class('none', 'strong') == 'product-sensitive'
    assert evidence_class('strong', 'weak') == evidence_class('weak', 'weak') == 'unresolved'
    assert combine_arms('product-robust', 'product-robust') == 'product-robust'
    assert combine_arms('product-sensitive', 'product-sensitive') == 'product-sensitive'
    assert combine_arms('product-robust', 'product-sensitive') == 'unresolved'
    assert combine_arms('unresolved', 'product-sensitive') == 'unresolved'


def test_tail_level_uses_same_stable_order_as_tail_lists():
    from research.model_v2.product_stability import TAIL_K, TAIL_NEAR_K, tail_level

    scores = np.linspace(1, 0, 40)
    assert tail_level(scores, 0) == tail_level(scores, TAIL_K - 1) == 'strong'
    assert tail_level(scores, TAIL_K) == tail_level(scores, TAIL_NEAR_K - 1) == 'weak'
    assert tail_level(scores, TAIL_NEAR_K) == 'none'


def test_classify_patterns_combines_arms_for_local_and_tail_entries():
    from research.model_v2.product_stability import classify_patterns

    iso = [f'C{i:02d}' for i in range(40)]
    table = pd.DataFrame({'iso3': iso, 'b_residual': np.linspace(1, -1, 40)})
    for w in ['station_knn8', 'area_knn8']:
        for p in 'be':
            table[f'{w}_{p}_residual_label'] = 'ns'
            table[f'{w}_{p}_residual_fdr_label'] = 'ns'
    table.loc[0, ['station_knn8_b_residual_label', 'station_knn8_b_residual_fdr_label']] = 'HH'
    absolute, aligned = table.copy(), table.copy()
    absolute['e_residual'] = absolute.b_residual
    aligned['e_residual'] = -aligned.b_residual
    absolute.loc[0, ['station_knn8_e_residual_label', 'station_knn8_e_residual_fdr_label']] = 'HH'
    moran = {'statistic': .2, 'p_value': .01}
    summary = {'spatial': {w: {'b_residual': moran, 'e_residual': moran} for w in ['station_knn8', 'area_knn8']},
               'tail_overlap': {k: {'n_overlap': 15} for k in ['positive', 'negative', 'absolute']}}
    region = pd.DataFrame({'n': [5], 'b_mean': [.03], 'e_mean': [.03], 'sign_agreement': [1.],
                           'classification': ['product-robust']}, index=['R'])
    result = classify_patterns({'absolute': (absolute, summary, region), 'aligned': (aligned, summary, region)})
    local = result.set_index(['weights', 'pattern']).loc[('station_knn8', 'C00 HH')]
    assert (local.class_absolute_arm, local.class_aligned_arm, local.classification) == (
        'product-robust', 'product-sensitive', 'unresolved')
    tail = result.set_index('pattern').loc['C00 positive']
    assert (tail.class_absolute_arm, tail.class_aligned_arm, tail.classification) == (
        'product-robust', 'product-sensitive', 'unresolved')
    assert result.set_index('pattern').loc['R', 'classification'] == 'product-robust'
