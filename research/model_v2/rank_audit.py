"""M0.5: controlled full-rank representations of the fixed M0 column space.

No feature expansion, selection or production/schema mutation. The three
minimal deletions span the choices of two out of three redundant log columns.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from research.model_v2 import cv, territory
from research.model_v2.m0 import (
    OUTPUT_DIR, design_matrix, load_inputs, m0_complete_design, verify_against_bundle,
)
from research.model_v2.run_territory_correction import complete_score
from src.decomposition import group_lmg_shares

VARIANTS = {
    'M0_legacy_representation': None,
    'drop_per_capita': 'cum_co2_per_capita',
    'drop_total': 'cum_co2_total',
    'drop_population': 'population',
}
NULL_VECTOR = {'intercept': -6., 'cum_co2_per_capita': 1.,
               'cum_co2_total': -1., 'population': 1.}


def parameterizations(design):
    """Fix the sample first; remove only the named representation."""
    return {name: design.copy() if column is None else design.drop(columns=column).copy()
            for name, column in VARIANTS.items()}


def audit_identity(design):
    x, columns, groups = design_matrix(design)
    error = (np.log10(design.cum_co2_per_capita)
             - np.log10(design.cum_co2_total) + np.log10(design.population) - 6)
    maximum = float(np.max(np.abs(error)))
    if maximum > 1e-12:
        raise ValueError('structural log identity does not hold to floating precision')
    singular = np.linalg.svd(x, compute_uv=False)
    beta, _, rank, _ = np.linalg.lstsq(x, design.warming_trend, rcond=None)
    null = np.array([NULL_VECTOR.get(c, 0) for c in columns])
    # Same data expressed in tonnes rather than Mt changes the arbitrary
    # least-norm coefficient representative, not its projection.
    tonnes = x.copy()
    tonnes[:, columns.index('cum_co2_total')] += 6
    beta_t = np.linalg.lstsq(tonnes, design.warming_trend, rcond=None)[0]
    return {
        'identity_constant': 6., 'max_abs_identity_error': maximum,
        'columns': columns, 'groups': groups, 'rank': int(rank),
        'n_columns': x.shape[1], 'null_vector': NULL_VECTOR,
        'singular_values': singular.tolist(),
        'rank_tolerance': float(singular[0] * max(x.shape) * np.finfo(float).eps),
        'full_coefficients': dict(zip(columns, beta.tolist())),
        'null_shift_1000_max_fitted_difference': float(np.max(np.abs(x @ (beta + 1000*null) - x @ beta))),
        'tonnes_unit_change_coefficients': dict(zip(columns, beta_t.tolist())),
        'tonnes_unit_change_max_fitted_difference': float(np.max(np.abs(tonnes @ beta_t - x @ beta))),
    }


def main():
    # Missing schema features are deliberate registered deletions, not data loss.
    logging.getLogger('src.feature_schema').setLevel(logging.ERROR)
    design = m0_complete_design(*load_inputs())
    table = pd.read_csv(OUTPUT_DIR / 'm0_countries.csv')
    assert design.Country.tolist() == table.Country.tolist()
    distance = territory.load_bounds(table.iso3.tolist())
    n = len(design)
    fold_id = np.arange(n)
    baseline_cv = cv.cross_validate(design, fold_id, dist=distance, buffer_km=500)
    baseline_fitted = cv.V1Design().fit(design).predict(design)
    identity = audit_identity(design)
    expected_m0 = json.loads((OUTPUT_DIR / 'm0_scorecard_territory_corrected.json').read_text())['corrected_primary']
    cards, summary, coefficients, country_errors = {}, [], [], []
    all_coalitions = {}
    for name, frame in parameterizations(design).items():
        x, columns, groups = design_matrix(frame)
        enc = cv.V1Design().fit(frame)
        fitted = enc.predict(frame)
        result = cv.cross_validate(frame, fold_id, dist=distance, buffer_km=500)
        repeated = cv.cross_validate(frame, fold_id, dist=distance, buffer_km=500)
        assert np.array_equal(result.yhat, repeated.yhat)
        card = complete_score(frame, table, result, reference=baseline_cv.yhat)
        secondary = cv.cross_validate(frame, cv.folds_from_labels(table.m49_subregion), dist=distance)
        card['secondary'] = complete_score(frame, table, secondary)
        reference = cv.cross_validate(frame, cv.folds_random(n, 10), dist=distance)
        card['random_reference_only'] = complete_score(frame, table, reference)
        lmg = group_lmg_shares(frame)
        card['standalone_r2'] = lmg.univariate_r2
        card['columns_by_group'] = groups
        card['matrix_columns'] = x.shape[1]
        card['matrix_rank'] = int(np.linalg.matrix_rank(x))
        s = np.linalg.svd(x, compute_uv=False)
        card['smallest_singular_value'] = float(s[-1])
        card['condition_number'] = float(s[0] / s[-1])
        card['max_full_prediction_change_vs_m0'] = float(np.max(np.abs(fitted-baseline_fitted)))
        card['max_cv_prediction_change_vs_m0'] = float(np.max(np.abs(result.yhat-baseline_cv.yhat)))
        card['deterministic_repeat_identical'] = True
        if name != 'M0_legacy_representation':
            assert card['matrix_rank'] == card['matrix_columns']
        assert card['max_full_prediction_change_vs_m0'] < 1e-9
        assert card['max_cv_prediction_change_vs_m0'] < 1e-9
        # Expose every coalition to make changes in the Shapley game auditable.
        from itertools import combinations
        from src.decomposition import _r2, feature_block
        group_names = list(lmg.group_features)
        coalition = {}
        for size in range(len(group_names)+1):
            for subset in combinations(group_names, size):
                blocks = [feature_block(frame, f)[0] for g in subset for f in lmg.group_features[g]]
                coalition['+'.join(subset) or 'intercept_only'] = _r2(blocks, frame.warming_trend.to_numpy())
        all_coalitions[name] = coalition
        fold_coefs = []
        for fold, _, train in cv.iter_folds(fold_id, distance, 500):
            model = cv.V1Design().fit(frame[train])
            fold_coefs.append(pd.Series(model.coef, index=model.columns))
            for c, value in zip(model.columns, model.coef):
                coefficients.append(dict(variant=name, fit='primary_training', fold=fold, column=c, coefficient=value))
        byfold = pd.DataFrame(fold_coefs)
        card['coefficient_stability'] = {}
        for c, value in zip(enc.columns, enc.coef):
            values = byfold[c].dropna()
            coefficients.append(dict(variant=name, fit='full_sample', fold=-1, column=c, coefficient=value))
            card['coefficient_stability'][c] = {
                'full': float(value), 'n_identified_folds': int(len(values)),
                'fold_min': float(values.min()), 'fold_median': float(values.median()),
                'fold_max': float(values.max()),
                'same_sign_fraction': float((np.sign(values) == np.sign(value)).mean()),
            }
        # Algebraic translation of the minimum-norm M0 vector into each reduced basis.
        beta0 = identity['full_coefficients']
        removed = VARIANTS[name]
        shift = 0 if removed is None else -beta0[removed] / NULL_VECTOR[removed]
        translated = {c: beta0[c]+shift*NULL_VECTOR.get(c,0) for c in columns}
        actual = dict(zip(enc.columns, enc.coef))
        card['max_coefficient_translation_error'] = float(max(abs(actual[c]-translated[c]) for c in columns))
        cards[name] = card
        row = dict(variant=name, removed=removed or 'none', n=n,
                   columns=x.shape[1], rank=card['matrix_rank'])
        for k in ['in_sample_r2','cv_r2','cv_rmse','cv_mae','fold_error_iqr',
                  'residual_morans_i_in_sample','residual_morans_i_cv',
                  'share_geography','share_emissions','share_socioeconomic','share_population',
                  'residual_share','rows_with_unseen_level']:
            row[k] = card[k]
        row['standalone_emissions_r2'] = lmg.univariate_r2['emissions']
        row['standalone_population_r2'] = lmg.univariate_r2['population']
        row['secondary_cv_r2'] = card['secondary']['cv_r2']
        row['secondary_cv_rmse'] = card['secondary']['cv_rmse']
        summary.append(row)
        for i, r in table.iterrows():
            country_errors.append(dict(variant=name, iso3=r.iso3, Country=r.Country,
                                       observed=frame.iloc[i].warming_trend, fitted=fitted[i],
                                       cv_prediction=result.yhat[i], unseen_levels=result.unseen_levels[i]))
    for k in ['cv_r2','cv_rmse','in_sample_r2','share_emissions']:
        assert abs(cards['M0_legacy_representation'][k]-expected_m0[k]) < 1e-10
    out = dict(baseline_verification=verify_against_bundle(), identity=identity,
               protocol='unchanged corrected 500km GPW-footprint LOCO',
               cards=cards, coalitions=all_coalitions,
               interpretation='Descriptive/associational explained country-warming variance, not causal attribution.')
    (OUTPUT_DIR / 'rank_audit.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / 'rank_audit_scorecard.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUTPUT_DIR / 'rank_audit_coefficients.csv', index=False)
    pd.DataFrame(country_errors).to_csv(OUTPUT_DIR / 'rank_audit_country_predictions.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    return out


if __name__ == '__main__':
    main()
