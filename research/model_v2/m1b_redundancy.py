"""Pre-outcome redundancy diagnostic for C2 (``M1B_EVALUATION_CONTRACT.md`` §3).

The frozen design loader returns the outcome column. It is dropped before any computation and
asserted absent, so no warming value enters this diagnostic.
Run with ``python -m research.model_v2.m1b_redundancy``.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from research.model_v2.m0 import OUTPUT_DIR, design_matrix, load_inputs, m0_complete_design
from src.decomposition import OUTCOME_COL, feature_block

GEOGRAPHY = ['abs_latitude', 'elevation', 'continentality', 'climate_zone', 'hemisphere', 'spatial_block']
GROUPS = {'responsibility': ['cum_co2_total'], 'socioeconomic': ['income_group'],
          'population': ['population', 'station_density']}
NEAR_REDUNDANT_R2 = 0.9
EXACT_R2 = 1 - 1e-10


def r2_on(blocks, target):
    x = np.hstack([np.ones((len(target), 1)), *blocks])
    beta = np.linalg.lstsq(x, target, rcond=None)[0]
    resid = target - x @ beta
    return float(1 - resid @ resid / np.sum((target - target.mean()) ** 2)), x, resid


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
    rank_x = int(np.linalg.matrix_rank(x))
    rank_xc = int(np.linalg.matrix_rank(np.column_stack([x, c2])))
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
        'n': n, 'design_columns': columns, 'design_rank': rank_x, 'rank_with_c2': rank_xc,
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
    design = m0_complete_design(*load_inputs())
    predictors = design.drop(columns=[OUTCOME_COL, 'cum_co2_per_capita']).reset_index(drop=True)
    del design  # the outcome is never used below
    table = pd.read_csv(OUTPUT_DIR / 'm0_countries.csv', usecols=['Country', 'iso3'])
    features = pd.read_csv(OUTPUT_DIR / 'm1b_hydroclimate_features.csv')
    if features.iso3.tolist() != table.iso3.tolist() or predictors.Country.tolist() != table.Country.tolist():
        raise ValueError('row order mismatch')
    result = diagnose(predictors, features.baseline_dryness.to_numpy(float))
    result['outcome_column_present'] = OUTCOME_COL in predictors.columns
    (OUTPUT_DIR / 'm1b_redundancy_diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result.get(k) for k in ['r2_full_m0star', 'adjusted_r2_full_m0star', 'vif', 'r2_by_group_alone',
                                             'near_redundant_flag', 'hard_stops']}, indent=2))
    return result


if __name__ == '__main__':
    main()
