"""Pre-outcome redundancy diagnostic for C2 (``M1B_EVALUATION_CONTRACT.md`` §3 and Amendment 2).

The M0* predictors are assembled from explicit allow-lists of predictor source columns for the
151 frozen countries, so no warming outcome, residual or score is ever read. The outcome column is
also asserted absent before any computation.
Run with ``python -m research.model_v2.m1b_redundancy``.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from research.model_v2.m0 import OUTPUT_DIR, design_matrix
from src.decomposition import OUTCOME_COL, aggregate_city_features_to_country, feature_block
from src.emissions import DEFAULT_INEQUALITY_PATH
from src.explain import DEFAULT_FEATURES_PATH, INCOME_PATH, load_income_groups

GEOGRAPHY = ['abs_latitude', 'elevation', 'continentality', 'climate_zone', 'hemisphere', 'spatial_block']
GROUPS = {'responsibility': ['cum_co2_total'], 'socioeconomic': ['income_group'],
          'population': ['population', 'station_density']}
NEAR_REDUNDANT_R2 = 0.9
EXACT_R2 = 1 - 1e-10
# Amendment 2: the only source columns ever read; none holds a warming trend, fit, residual or score.
INEQUALITY_COLUMNS = ['Country', 'owid_country', 'continent', 'cumulative_co2_mt', 'population']
CITY_COLUMNS = ['Country', 'abs_latitude', 'hemisphere', 'coast_km', 'elevation_m', 'koppen', 'station_density']
PREDICTOR_COLUMNS = ['Country', 'cum_co2_total', 'abs_latitude', 'elevation', 'continentality', 'climate_zone',
                     'hemisphere', 'spatial_block', 'income_group', 'population', 'station_density']
SUPPORT_RECORD = OUTPUT_DIR / 'm1a_geography_qa.csv'  # frozen, outcome-free identifiers and row order


def m0star_predictors(countries):
    """The M0* predictor frame (primary representation, no per-capita CO2) for ``countries``, in order.

    Mirrors ``src.decomposition.build_country_design`` column for column, without its outcome column,
    and restricts to the frozen countries instead of an outcome-dependent complete-case filter.
    """
    inequality = pd.read_parquet(DEFAULT_INEQUALITY_PATH, columns=INEQUALITY_COLUMNS)
    city = pd.read_parquet(DEFAULT_FEATURES_PATH, columns=CITY_COLUMNS)
    table = inequality.merge(aggregate_city_features_to_country(city), on='Country', how='left')
    income = load_income_groups(INCOME_PATH).set_index('owid_country')['income_group']
    frame = pd.DataFrame({
        'Country': table['Country'], 'cum_co2_total': table['cumulative_co2_mt'],
        'abs_latitude': table['abs_latitude'], 'elevation': table['elevation'],
        'continentality': table['continentality'], 'climate_zone': table['climate_zone'],
        'hemisphere': table['hemisphere'], 'spatial_block': table['continent'],
        'income_group': table['owid_country'].map(income), 'population': table['population'],
        'station_density': table['station_density'],
    })
    frame = frame.set_index('Country').loc[list(countries)].reset_index()
    if frame.columns.tolist() != PREDICTOR_COLUMNS or OUTCOME_COL in frame.columns:
        raise AssertionError('predictor frame columns differ from the allow-list')
    if frame.drop(columns='Country').isna().any().any():
        raise ValueError('a frozen country has a missing M0* predictor')
    return frame


def r2_on(blocks, target):
    x = np.hstack([np.ones((len(target), 1)), *blocks])
    beta = np.linalg.lstsq(x, target, rcond=None)[0]
    resid = target - x @ beta
    return float(1 - resid @ resid / np.sum((target - target.mean()) ** 2)), x, resid


def condition_numbers(x):
    """2-norm condition number with raw columns and with columns scaled to unit length."""
    return {'raw': float(np.linalg.cond(x)), 'unit_column_scaled': float(np.linalg.cond(x / np.linalg.norm(x, axis=0)))}


def diagnose(predictors, c2):
    """Redundancy of C2 against the M0* predictor frame (no outcome column allowed)."""
    if OUTCOME_COL in predictors.columns:
        raise AssertionError('outcome column must be removed before the redundancy diagnostic')
    hard_stops = []
    if not np.isfinite(c2).all():
        hard_stops.append('non-finite C2')
    if np.nanstd(c2) == 0:
        hard_stops.append('zero-variance C2')
    if hard_stops:  # data defect: no statistic is meaningful; stop for review
        return {'n': int(len(c2)), 'hard_stops': hard_stops}
    x, columns, _ = design_matrix(predictors)
    beta = np.linalg.lstsq(x, c2, rcond=None)[0]
    resid = c2 - x @ beta
    r2 = float(1 - resid @ resid / np.sum((c2 - c2.mean()) ** 2))
    n, p = x.shape
    x_with_c2 = np.column_stack([x, c2])
    rank_x = int(np.linalg.matrix_rank(x))
    rank_xc = int(np.linalg.matrix_rank(x_with_c2))
    if rank_xc == rank_x or r2 >= EXACT_R2:
        hard_stops.append('exact rank redundancy of C2 with the M0* design')
    geography_blocks = [feature_block(predictors, f)[0] for f in GEOGRAPHY]
    group_r2 = {'geography': r2_on(geography_blocks, c2)[0]}
    for group, features in GROUPS.items():
        group_r2[group] = r2_on([feature_block(predictors, f)[0] for f in features], c2)[0]
    numeric = {'abs_latitude': predictors.abs_latitude, 'elevation': predictors.elevation,
               'continentality': predictors.continentality, 'log10_cum_co2_total': np.log10(predictors.cum_co2_total),
               'log10_population': np.log10(predictors.population), 'station_density': predictors.station_density}
    correlations = {k: {'pearson': float(pearsonr(v, c2).statistic), 'spearman': float(spearmanr(v, c2).statistic)}
                    for k, v in numeric.items()}
    categories = {}
    for col in ['climate_zone', 'hemisphere', 'spatial_block', 'income_group']:
        g = pd.Series(c2).groupby(predictors[col].astype(str).to_numpy())
        categories[col] = {level: {'n': int(s.size), 'mean': float(s.mean()),
                                   'sd': float(s.std(ddof=1)) if s.size > 1 else None} for level, s in g}
    return {
        'n': n, 'design_columns': columns, 'design_column_count': p, 'design_rank': rank_x, 'rank_with_c2': rank_xc,
        'condition_number_design': condition_numbers(x), 'condition_number_design_with_c2': condition_numbers(x_with_c2),
        'r2_full_m0star': r2, 'adjusted_r2_full_m0star': float(1 - (1 - r2) * (n - 1) / (n - rank_x)),
        'vif': float(1 / (1 - r2)) if r2 < 1 else None, 'residual_sd_of_c2': float(resid.std(ddof=rank_x)),
        'c2_mean': float(c2.mean()), 'c2_sd': float(c2.std(ddof=1)),
        'r2_by_group_alone': group_r2, 'correlations': correlations, 'category_summaries': categories,
        'near_redundant_flag': r2 > NEAR_REDUNDANT_R2, 'hard_stops': hard_stops,
    }


def main():
    logging.getLogger('src.feature_schema').setLevel(logging.ERROR)
    manifest = json.loads((OUTPUT_DIR / 'm1b_measurement_manifest.json').read_text())
    if not manifest['all_hard_stops_pass']:
        raise RuntimeError(f"measurement hard stops: {manifest['hard_stops']}")
    features = pd.read_csv(OUTPUT_DIR / 'm1b_hydroclimate_features.csv')
    frozen = pd.read_csv(SUPPORT_RECORD, usecols=['iso3', 'Country'])
    if features.iso3.tolist() != frozen.iso3.tolist() or features.Country.tolist() != frozen.Country.tolist():
        raise ValueError('row order mismatch between C2 and the frozen country record')
    predictors = m0star_predictors(features.Country)
    if OUTCOME_COL in predictors.columns or predictors.columns.tolist() != PREDICTOR_COLUMNS:
        raise AssertionError('outcome or unexpected column in the redundancy inputs')
    result = diagnose(predictors, features.baseline_dryness.to_numpy(float))
    result['inputs'] = {'inequality_columns_read': INEQUALITY_COLUMNS, 'city_feature_columns_read': CITY_COLUMNS,
                        'predictor_columns': PREDICTOR_COLUMNS, 'outcome_column_present': False,
                        'c2_manifest_code_sha256': manifest['code_sha256'], 'c2_git_commit': manifest['git']['commit']}
    (OUTPUT_DIR / 'm1b_redundancy_diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result.get(k) for k in ['r2_full_m0star', 'adjusted_r2_full_m0star', 'vif', 'design_rank',
                                             'rank_with_c2', 'condition_number_design_with_c2', 'r2_by_group_alone',
                                             'near_redundant_flag', 'hard_stops']}, indent=2))
    return result


if __name__ == '__main__':
    main()
