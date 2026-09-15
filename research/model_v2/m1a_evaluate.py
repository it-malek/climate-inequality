"""M1a scoring under the frozen ``M1A_EVALUATION_CONTRACT.md``.

Run ``python -m research.model_v2.m1a_evaluate`` only after
``m1a_geography`` has produced a manifest in which every frozen country passes
every gate. Otherwise this refuses to score. Writes only ``outputs/m1a_*``.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from research.model_v2 import cv, territory
from research.model_v2.m0 import OUTPUT_DIR, design_matrix, load_inputs, m0_complete_design
from research.model_v2.run_m0_scorecard import PRIMARY_BUFFER_KM
from research.model_v2.run_territory_correction import complete_score
from research.model_v2.spatial import haversine_matrix, knn_weights, morans_i
from src.decomposition import CATEGORICAL_FEATURES

REMEASURED = ('abs_latitude', 'elevation', 'continentality', 'climate_zone', 'hemisphere')
PAIRS = {'primary': ('cum_co2_per_capita', 'drop_per_capita'),
         'percapita_sensitivity': ('cum_co2_total', 'drop_total')}
LARGE_COUNTRIES = ('CAN', 'BRA', 'RUS', 'DZA')
SECONDARY_VETO = WORST_REGION_VETO = 0.001  # degC/decade
MORAN_CHANGE = 0.05
OVERFIT_GAP = 0.05
MATERIAL_RESPONSIBILITY = 0.10


def load_features(design, table):
    manifest = json.loads((OUTPUT_DIR / 'm1a_measurement_manifest.json').read_text())
    if not manifest['all_features_pass_gates']:
        raise RuntimeError('M1a measurement gates failed; primary scoring is ineligible pending an owner decision: '
                           f"{manifest['gate_failures']} {manifest['quadrature_failures']}")
    frozen = json.loads((OUTPUT_DIR / 'm0_scorecard_territory_corrected.json').read_text())['geometry']
    gpw = manifest['sources']['gpw_national_identifier']
    if (gpw['sha256'], gpw['lookup_sha256'], manifest['sources']['territory_snapshot_sha256']) != (
            frozen['source_sha256'], frozen['lookup_sha256'], frozen['snapshot_sha256']):
        raise ValueError('M1a GPW sources differ from the corrected CV manifest')
    snapshot = np.load(OUTPUT_DIR / 'territory_gpw_footprints.npz')
    qa = pd.read_csv(OUTPUT_DIR / 'm1a_geography_qa.csv').set_index('iso3')
    if any(qa.loc[code, 'gpw_cells'] != len(snapshot[code]) for code in table.iso3):
        raise ValueError('M1a support cells differ from the frozen territorial footprints')
    features = pd.read_csv(OUTPUT_DIR / 'm1a_geography_features.csv')
    if features.iso3.tolist() != table.iso3.tolist() or features.Country.tolist() != design.Country.tolist():
        raise ValueError('M1a feature rows do not match the frozen country order')
    if not (features.spatial_block.to_numpy() == design.spatial_block.to_numpy()).all():
        raise ValueError('spatial_block must be copied unchanged')
    if features[list(REMEASURED)].isna().any().any():
        raise ValueError('missing remeasured feature')
    return features, manifest


def frames(design, features):
    """Comparator and M1a frames per representation; only the five columns differ."""
    out = {}
    for pair, (dropped, _) in PAIRS.items():
        comparator = design.drop(columns=dropped).reset_index(drop=True)
        m1a = comparator.copy()
        for f in REMEASURED:
            m1a[f] = features[f].to_numpy()
        same = [c for c in comparator.columns if c not in REMEASURED]
        pd.testing.assert_frame_equal(comparator[same], m1a[same])
        out[pair] = {'C0': comparator, 'M1a': m1a}
    return out


def sign_stability(frame, fold_id, distance):
    full = cv.V1Design().fit(frame)
    signs = {c: [] for c in full.columns}
    for _, _, train in cv.iter_folds(fold_id, distance, PRIMARY_BUFFER_KM):
        model = cv.V1Design().fit(frame[train])
        for c, value in zip(model.columns, model.coef):
            if c in signs:
                signs[c].append(np.sign(value))
    return {c: {'full': float(v), 'same_sign_fraction': float(np.mean(np.array(signs[c]) == np.sign(v)))}
            for c, v in zip(full.columns, full.coef)}


def score(frame, table, distance, area_w, reference=None):
    n = len(frame)
    primary = cv.cross_validate(frame, np.arange(n), dist=distance, buffer_km=PRIMARY_BUFFER_KM)
    card = complete_score(frame, table, primary, reference=reference)
    secondary = cv.cross_validate(frame, cv.folds_from_labels(table.m49_subregion), dist=distance)
    card['secondary'] = complete_score(frame, table, secondary)
    card['random_reference_only'] = complete_score(frame, table, cv.cross_validate(frame, cv.folds_random(n, 10), dist=distance))
    fitted = cv.V1Design().fit(frame).predict(frame)
    y = frame.warming_trend.to_numpy()
    card['recorded_area_centroid_moran_in_sample'] = morans_i(y - fitted, area_w).statistic
    card['recorded_area_centroid_moran_cv'] = morans_i(y - primary.yhat, area_w).statistic
    x, columns, _ = design_matrix(frame)
    card['encoded_columns'] = columns
    card['matrix_columns'], card['matrix_rank'] = x.shape[1], int(np.linalg.matrix_rank(x))
    card['levels_present'] = {c: sorted(frame[c].astype(str).unique().tolist())
                              for c in sorted(CATEGORICAL_FEATURES) if c in frame}
    unseen = {}
    for name, folds, buffer in [('primary', np.arange(n), PRIMARY_BUFFER_KM),
                                ('secondary', cv.folds_from_labels(table.m49_subregion), None)]:
        unseen[name] = cv.design_descriptors(frame, folds, distance, buffer)['test_rows_with_unseen_level_by_categorical']
    card['unseen_rows_by_categorical'] = unseen
    card['coefficient_sign_stability_recorded'] = sign_stability(frame, np.arange(n), distance)
    return card, primary


def classify(c0, m1a, errors):
    """Contract §4, applied mechanically to one comparator/M1a pair."""
    delta = m1a['paired_delta_vs_m0']
    lo, hi = delta['country_bootstrap_95_interval']
    secondary_ok = m1a['secondary']['cv_rmse'] - c0['secondary']['cv_rmse'] <= SECONDARY_VETO
    region_ok = m1a['worst_region']['rmse'] - c0['worst_region']['rmse'] <= WORST_REGION_VETO
    if delta['delta_rmse'] < 0 and hi < 0 and secondary_ok and region_ok:
        transfer = 'improved generalization'
    elif delta['delta_rmse'] > 0 and lo > 0:
        transfer = 'worsened generalization'
    else:
        transfer = 'no detectable change in transfer'
    d_in = m1a['residual_morans_i_in_sample'] - c0['residual_morans_i_in_sample']
    d_cv = m1a['residual_morans_i_cv'] - c0['residual_morans_i_cv']
    if d_in <= -MORAN_CHANGE and d_cv <= -MORAN_CHANGE:
        spatial = 'reduced residual spatial autocorrelation'
    elif d_in >= MORAN_CHANGE and d_cv >= MORAN_CHANGE:
        spatial = 'increased residual spatial autocorrelation'
    else:
        spatial = 'residual spatial autocorrelation essentially unchanged'
    shares = {g: m1a[f'share_{g}'] for g in ['geography', 'emissions', 'socioeconomic', 'population']}
    material = max(shares, key=shares.get) != 'geography' or shares['emissions'] > MATERIAL_RESPONSIBILITY
    gain_in = m1a['in_sample_r2'] - c0['in_sample_r2']
    gain_cv = m1a['cv_r2'] - c0['cv_r2']
    overfit = (gain_in - gain_cv > OVERFIT_GAP) or (
        m1a['cv_rmse'] > c0['cv_rmse'] and m1a['in_sample_rmse'] < c0['in_sample_rmse'])
    improved_t, improved_s = transfer == 'improved generalization', spatial.startswith('reduced')
    summary = ('improved generalization and spatial specification' if improved_t and improved_s else
               'improved generalization only' if improved_t else
               'improved spatial specification only' if improved_s else 'neither')
    not_worse = lo <= 0 <= hi
    large_before = errors.loc[list(LARGE_COUNTRIES), 'C0_abs_error']
    large_after = errors.loc[list(LARGE_COUNTRIES), 'M1a_abs_error']
    return {
        'transfer': transfer, 'secondary_veto_passed': bool(secondary_ok), 'worst_region_veto_passed': bool(region_ok),
        'spatial_structure': spatial, 'delta_moran_in_sample': d_in, 'delta_moran_cv': d_cv,
        'material_change_to_conclusion': bool(material), 'scientific_stability': 'material change' if material else 'stable',
        'overfitting_signal': bool(overfit), 'in_sample_r2_gain': gain_in, 'cv_r2_gain': gain_cv,
        'summary': summary,
        'gate_m1a_to_m1': {'genuine_improvement': improved_t, 'not_worse': bool(not_worse),
                           'large_country_abs_errors_before': large_before.to_dict(),
                           'large_country_abs_errors_after': large_after.to_dict(),
                           'large_country_errors_all_reduced': bool((large_after < large_before).all())},
    }


def main():
    logging.getLogger('src.feature_schema').setLevel(logging.ERROR)
    design = m0_complete_design(*load_inputs())
    table = pd.read_csv(OUTPUT_DIR / 'm0_countries.csv')
    assert design.Country.tolist() == table.Country.tolist()
    features, manifest = load_features(design, table)
    distance = territory.load_bounds(table.iso3.tolist())
    geom = pd.read_csv(OUTPUT_DIR / 'country_geometry.csv').set_index('iso3').loc[table.iso3]
    area_w = knn_weights(haversine_matrix(geom.centroid_lon.to_numpy(), geom.centroid_lat.to_numpy()), 8)
    rank = json.loads((OUTPUT_DIR / 'rank_audit.json').read_text())['cards']
    result, rows, countries = {'manifest_code_sha256': manifest['code_sha256']}, [], []
    for pair, stage in frames(design, features).items():
        c0, c0_cv = score(stage['C0'], table, distance, area_w)
        audit = rank[PAIRS[pair][1]]
        for k in ['in_sample_r2', 'cv_r2', 'cv_rmse', 'share_geography', 'share_emissions',
                  'share_socioeconomic', 'share_population', 'residual_morans_i_in_sample']:
            if abs(c0[k] - audit[k]) > 1e-10:
                raise AssertionError(f'{pair} comparator does not reproduce rank audit {k}')
        m1a, m1a_cv = score(stage['M1a'], table, distance, area_w, reference=c0_cv.yhat)
        y = stage['C0'].warming_trend.to_numpy()
        errors = pd.DataFrame({'iso3': table.iso3, 'Country': table.Country, 'm49_subregion': table.m49_subregion,
                               'observed': y, 'C0_cv_prediction': c0_cv.yhat, 'M1a_cv_prediction': m1a_cv.yhat,
                               'C0_abs_error': np.abs(y - c0_cv.yhat), 'M1a_abs_error': np.abs(y - m1a_cv.yhat),
                               'C0_unseen_levels': c0_cv.unseen_levels, 'M1a_unseen_levels': m1a_cv.unseen_levels})
        errors['abs_error_change'] = errors.M1a_abs_error - errors.C0_abs_error
        errors = errors.set_index('iso3')
        verdict = classify(c0, m1a, errors)
        verdict['countries_improved'] = int((errors.abs_error_change < 0).sum())
        verdict['countries_worsened'] = int((errors.abs_error_change > 0).sum())
        result[pair] = {'C0': c0, 'M1a': m1a, 'classification': verdict}
        for name, card in [('C0', c0), ('M1a', m1a)]:
            rows.append({'pair': pair, 'row': name, **{k: card[k] for k in [
                'n', 'matrix_columns', 'matrix_rank', 'in_sample_r2', 'cv_r2', 'cv_rmse', 'cv_mae',
                'fold_rmse_median', 'fold_rmse_min', 'fold_rmse_max', 'fold_error_iqr',
                'residual_morans_i_in_sample', 'residual_morans_i_cv', 'share_geography', 'share_emissions',
                'share_socioeconomic', 'share_population', 'residual_share', 'rows_with_unseen_level']},
                'worst_fold_country': card['worst_fold_country'], 'worst_region': card['worst_region']['region'],
                'worst_region_rmse': card['worst_region']['rmse'],
                'secondary_cv_r2': card['secondary']['cv_r2'], 'secondary_cv_rmse': card['secondary']['cv_rmse'],
                'random_reference_cv_r2': card['random_reference_only']['cv_r2'],
                'delta_rmse_vs_comparator': card['paired_delta_vs_m0']['delta_rmse'],
                'delta_rmse_95_low': card['paired_delta_vs_m0']['country_bootstrap_95_interval'][0],
                'delta_rmse_95_high': card['paired_delta_vs_m0']['country_bootstrap_95_interval'][1]})
        countries.append(errors.reset_index().assign(pair=pair))
    result['contract'] = 'research/model_v2/M1A_EVALUATION_CONTRACT.md'
    (OUTPUT_DIR / 'm1a_scorecard.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=float) + '\n')
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'm1a_scorecard_summary.csv', index=False)
    pd.concat(countries).to_csv(OUTPUT_DIR / 'm1a_country_cv_errors.csv', index=False)
    print(pd.DataFrame(rows).T.to_string())
    print(json.dumps({p: result[p]['classification'] for p in PAIRS}, indent=2, default=float))
    return result


if __name__ == '__main__':
    main()
