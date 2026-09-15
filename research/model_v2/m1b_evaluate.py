"""M1b scoring under the frozen ``M1B_EVALUATION_CONTRACT.md`` (pushed ``60111ae``).

M1b = M0* + ``baseline_dryness`` in a research-only ``hydroclimate`` group. Before scoring, it
refuses to run unless measurement and redundancy hard stops pass and M0* reproduces the rank-audit
``drop_per_capita`` card to 1e-10 through this module's schema extension.
Run with ``PYTHONHASHSEED=0 python -m research.model_v2.m1b_evaluate``.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

from research.model_v2 import cv, territory
from research.model_v2.m0 import OUTPUT_DIR, load_inputs, m0_complete_design
from research.model_v2.run_m0_scorecard import MIN_REGION_N, PRIMARY_BUFFER_KM
from research.model_v2.run_territory_correction import paired_rmse_interval
from research.model_v2.spatial import haversine_matrix, knn_weights, morans_i
from src import feature_schema as fs
from src.decomposition import CATEGORICAL_FEATURES, feature_block, group_lmg_shares
from src.stability import _used_features

FEATURE = 'baseline_dryness'
HYDROCLIMATE = fs.FeatureGroup(
    key='hydroclimate', title='Baseline hydroclimate (research-only, M1b)',
    rationale='Pre-outcome-window (1920-1949) CRU TS v4.10 local log aridity over terrestrial area; '
              'a physical state concept, not geography and not attribution.',
    features=(fs.FeatureSpec(FEATURE, 'Area-weighted mean of cell log10(P/PET), 1920-1949 climatology.',
                             'log10(dimensionless)'),),
)
SCHEMA_M1B = dataclasses.replace(
    fs.SCHEMA_V1, version='v1+m1b-hydroclimate-research',
    groups=tuple(g for g in fs.SCHEMA_V1.groups if g.key != 'residual') + (HYDROCLIMATE,)
    + tuple(g for g in fs.SCHEMA_V1.groups if g.key == 'residual'))
PRACTICAL = -0.002
VETO = 0.001
MORAN_CHANGE = 0.05
OVERFIT_GAP = 0.05
MATERIAL_RESPONSIBILITY = 0.10
NAMED_GROUPS = ('geography', 'emissions', 'socioeconomic', 'population', 'hydroclimate')
SIGN_STABILITY = 0.80


class M1bDesign(cv.V1Design):
    def _features(self, columns):
        used = _used_features(columns, SCHEMA_M1B, fs.STATUS_AVAILABLE)
        return [name for names in used.values() for name in names]


def fit_predict(train, test):
    return M1bDesign().fit(train).predict(test)


def design_matrix(frame):
    used = _used_features(frame.columns, SCHEMA_M1B, fs.STATUS_AVAILABLE)
    blocks = [feature_block(frame, f)[0] for names in used.values() for f in names]
    return np.hstack([np.ones((len(frame), 1)), *blocks])


def complete_score(frame, table, result, *, reference=None):
    """`run_territory_correction.complete_score` with the extended schema (same metric definitions)."""
    fitted = M1bDesign().fit(frame).predict(frame)
    w = cv.v1_weights(table.station_lon.to_numpy(), table.station_lat.to_numpy())
    card = cv.scorecard(frame, fitted, result, w, shares=False)
    lmg = group_lmg_shares(frame, schema=SCHEMA_M1B)
    card.update({f'share_{k}': float(v) for k, v in lmg.shares.items()})
    card['residual_share'] = float(lmg.residual_share)
    y = frame.warming_trend.to_numpy()
    errors = np.abs(y - result.yhat)
    card['fold_error_iqr'] = float(np.subtract(*np.quantile(errors, [0.75, 0.25])))
    card['effective_degrees_of_freedom'] = int(np.linalg.matrix_rank(design_matrix(frame)))
    card['residual_moran_in_sample'] = asdict(morans_i(y - fitted, w))
    card['residual_moran_cv'] = asdict(morans_i(y - result.yhat, w))
    regions = cv.region_errors(result, y, table.m49_subregion)
    eligible = regions[regions.n >= MIN_REGION_N]
    card['worst_region'] = {'region': str(eligible.index[0]), 'rmse': float(eligible.iloc[0].rmse),
                            'n': int(eligible.iloc[0].n), 'min_region_n': MIN_REGION_N}
    card['region_rmse'] = regions.rmse.to_dict()
    card['region_bias'] = regions.bias.to_dict()
    card['worst_fold_country'] = str(table.iloc[card['worst_fold']].Country)
    card['paired_delta_vs_m0'] = paired_rmse_interval(y, result.yhat, result.yhat if reference is None else reference)
    return card


def score(frame, table, distance, area_w, reference=None):
    n = len(frame)
    folds = np.arange(n)
    primary = cv.cross_validate(frame, folds, fit_predict, dist=distance, buffer_km=PRIMARY_BUFFER_KM)
    card = complete_score(frame, table, primary, reference=reference)
    secondary = cv.cross_validate(frame, cv.folds_from_labels(table.m49_subregion), fit_predict, dist=distance)
    card['secondary'] = complete_score(frame, table, secondary)
    card['random_reference_only'] = complete_score(frame, table, cv.cross_validate(frame, cv.folds_random(n, 10), fit_predict, dist=distance))
    fitted = M1bDesign().fit(frame).predict(frame)
    y = frame.warming_trend.to_numpy()
    card['recorded_area_centroid_moran_in_sample'] = morans_i(y - fitted, area_w).statistic
    card['recorded_area_centroid_moran_cv'] = morans_i(y - primary.yhat, area_w).statistic
    x = design_matrix(frame)
    card['matrix_columns'], card['matrix_rank'] = x.shape[1], int(np.linalg.matrix_rank(x))
    card['unseen_rows_by_categorical'] = {
        name: cv.design_descriptors(frame, f, distance, b)['test_rows_with_unseen_level_by_categorical']
        for name, f, b in [('primary', folds, PRIMARY_BUFFER_KM), ('secondary', cv.folds_from_labels(table.m49_subregion), None)]}
    full = M1bDesign().fit(frame)
    stab = {c: [] for c in full.columns}
    for _, _, train in cv.iter_folds(folds, distance, PRIMARY_BUFFER_KM):
        model = M1bDesign().fit(frame[train])
        for c, v in zip(model.columns, model.coef):
            stab.setdefault(c, []).append(float(v))
    card['coefficients'] = {c: {'full': float(v), 'fold_values_n': len(stab[c]),
                                'same_sign_fraction': float(np.mean(np.sign(stab[c]) == np.sign(v)))}
                            for c, v in zip(full.columns, full.coef)}
    card['levels_present'] = {c: sorted(frame[c].astype(str).unique().tolist())
                              for c in sorted(CATEGORICAL_FEATURES) if c in frame}
    return card, primary


def verdict(c0, m1b):
    """Contract §4.4 two-level verdict (mechanical)."""
    d = m1b['paired_delta_vs_m0']
    lo, hi = d['country_bootstrap_95_interval']
    coef = m1b['coefficients'][FEATURE]
    s1 = d['delta_rmse'] < 0 and hi < 0
    s2 = coef['full'] < 0 and coef['same_sign_fraction'] >= SIGN_STABILITY
    a2 = d['delta_rmse'] <= PRACTICAL
    a3 = m1b['secondary']['cv_rmse'] - c0['secondary']['cv_rmse'] <= VETO
    a4 = m1b['worst_region']['rmse'] - c0['worst_region']['rmse'] <= VETO
    if s1 and s2 and a2 and a3 and a4:
        label = 'supported and promoted'
    elif s1 and s2:
        label = 'supported but sub-material / not promoted'
    else:
        label = 'not supported'
    return {'verdict': label, 'S1_interval_below_zero': bool(s1), 'S2_sign_negative_and_stable': bool(s2),
            'A1': bool(s1), 'A2_practical_le_minus_0.002': bool(a2), 'A3_m49_veto_passed': bool(a3),
            'A4_worst_region_veto_passed': bool(a4), 'A5': bool(s2),
            'delta_rmse': d['delta_rmse'], 'interval': [lo, hi],
            'coefficient_full': coef['full'], 'coefficient_same_sign_fraction': coef['same_sign_fraction'],
            'improvement_without_prestated_sign': bool(d['delta_rmse'] < 0 and hi < 0 and not s2),
            'worsened_generalization': bool(d['delta_rmse'] > 0 and lo > 0),
            'm49_rmse_change': m1b['secondary']['cv_rmse'] - c0['secondary']['cv_rmse'],
            'worst_region_rmse_change': m1b['worst_region']['rmse'] - c0['worst_region']['rmse']}


def diagnostics(c0, m1b):
    """Contract §4.5 diagnostics (not verdict conditions)."""
    gain_in, gain_cv = m1b['in_sample_r2'] - c0['in_sample_r2'], m1b['cv_r2'] - c0['cv_r2']
    d_in = m1b['residual_morans_i_in_sample'] - c0['residual_morans_i_in_sample']
    d_cv = m1b['residual_morans_i_cv'] - c0['residual_morans_i_cv']
    spatial = ('reduced' if d_in <= -MORAN_CHANGE and d_cv <= -MORAN_CHANGE else
               'increased' if d_in >= MORAN_CHANGE and d_cv >= MORAN_CHANGE else 'essentially unchanged')
    shares = {g: m1b.get(f'share_{g}', 0.0) for g in NAMED_GROUPS}
    largest = max(shares, key=shares.get)
    material = largest != 'geography' or shares['emissions'] > MATERIAL_RESPONSIBILITY
    return {'overfitting_signal': bool(gain_in - gain_cv > OVERFIT_GAP or
                                       (m1b['cv_rmse'] > c0['cv_rmse'] and m1b['in_sample_rmse'] < c0['in_sample_rmse'])),
            'in_sample_r2_gain': gain_in, 'cv_r2_gain': gain_cv,
            'residual_spatial_structure': f'residual spatial autocorrelation {spatial}',
            'delta_moran_in_sample': d_in, 'delta_moran_cv': d_cv,
            'largest_named_group': largest, 'material_change_to_conclusion': bool(material),
            'scientific_stability': 'material change' if material else 'stable', 'shares': shares}


def main():
    logging.getLogger('src.feature_schema').setLevel(logging.ERROR)
    for name in ['m1b_measurement_manifest.json', 'm1b_redundancy_diagnostic.json']:
        stops = json.loads((OUTPUT_DIR / name).read_text()).get('hard_stops')
        if stops:
            raise RuntimeError(f'{name} hard stops: {stops}')
    design = m0_complete_design(*load_inputs())
    table = pd.read_csv(OUTPUT_DIR / 'm0_countries.csv')
    assert design.Country.tolist() == table.Country.tolist()
    c2 = pd.read_csv(OUTPUT_DIR / 'm1b_hydroclimate_features.csv')
    assert c2.iso3.tolist() == table.iso3.tolist()
    m0star = design.drop(columns='cum_co2_per_capita').reset_index(drop=True)
    m1b = m0star.copy()
    m1b[FEATURE] = c2.baseline_dryness.to_numpy()
    distance = territory.load_bounds(table.iso3.tolist())
    geom = pd.read_csv(OUTPUT_DIR / 'country_geometry.csv').set_index('iso3').loc[table.iso3]
    area_w = knn_weights(haversine_matrix(geom.centroid_lon.to_numpy(), geom.centroid_lat.to_numpy()), 8)
    c0, c0_cv = score(m0star, table, distance, area_w)
    audit = json.loads((OUTPUT_DIR / 'rank_audit.json').read_text())['cards']['drop_per_capita']
    for k in ['in_sample_r2', 'cv_r2', 'cv_rmse', 'share_geography', 'share_emissions', 'share_socioeconomic',
              'share_population', 'residual_morans_i_in_sample']:
        if abs(c0[k] - audit[k]) > 1e-10:
            raise AssertionError(f'M0* does not reproduce the rank audit through the M1b schema: {k}')
    card, card_cv = score(m1b, table, distance, area_w, reference=c0_cv.yhat)
    y = m0star.warming_trend.to_numpy()
    errors = pd.DataFrame({'iso3': table.iso3, 'Country': table.Country, 'm49_subregion': table.m49_subregion,
                           'observed': y, 'M0star_cv_prediction': c0_cv.yhat, 'M1b_cv_prediction': card_cv.yhat,
                           'M0star_abs_error': np.abs(y - c0_cv.yhat), 'M1b_abs_error': np.abs(y - card_cv.yhat)})
    errors['abs_error_change'] = errors.M1b_abs_error - errors.M0star_abs_error
    errors = errors.set_index('iso3')
    result = {'contract': 'research/model_v2/M1B_EVALUATION_CONTRACT.md', 'M0star': c0, 'M1b': card,
              'verdict': verdict(c0, card), 'diagnostics': diagnostics(c0, card),
              'share_hydroclimate': card['share_hydroclimate'],
              'countries_improved': int((errors.abs_error_change < 0).sum()),
              'countries_worsened': int((errors.abs_error_change > 0).sum())}
    (OUTPUT_DIR / 'm1b_scorecard.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=float) + '\n')
    errors.reset_index().to_csv(OUTPUT_DIR / 'm1b_country_cv_errors.csv', index=False)
    print(json.dumps(result['verdict'], indent=2, default=float))
    return result


if __name__ == '__main__':
    main()
