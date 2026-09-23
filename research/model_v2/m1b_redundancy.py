"""Pre-outcome redundancy diagnostic for C2 against the M0* predictors.

The M0* predictors are assembled from explicit allow-lists of predictor source columns for the
151 fixed countries, so no warming outcome, fitted value, residual or score column is ever read;
selected predictor columns are read from parquet containers that also hold outcome columns. The
outcome column is asserted absent before any computation. The measurement package is verified
against its manifest digests before C2 is read, the predictor inputs against their recorded SHA-256
digests, and the output records both, plus the ordered predictor frame and design identity.
Run with ``python -m research.model_v2.m1b_redundancy``; the process exits with status 2 on a hard stop.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from research.model_v2 import m1b_hydroclimate as h
from research.model_v2.m0 import OUTPUT_DIR, ROOT, design_matrix
from src.decomposition import OUTCOME_COL, aggregate_city_features_to_country, feature_block
from src.emissions import DEFAULT_INEQUALITY_PATH
from src.explain import DEFAULT_FEATURES_PATH, INCOME_PATH, load_income_groups

GEOGRAPHY = ['abs_latitude', 'elevation', 'continentality', 'climate_zone', 'hemisphere', 'spatial_block']
GROUPS = {'responsibility': ['cum_co2_total'], 'socioeconomic': ['income_group'],
          'population': ['population', 'station_density']}
NEAR_REDUNDANT_R2 = 0.9
EXACT_R2 = 1 - 1e-10
# The only source columns ever read; none holds a warming trend, fit, residual or score.
INEQUALITY_COLUMNS = ['Country', 'owid_country', 'continent', 'cumulative_co2_mt', 'population']
CITY_COLUMNS = ['Country', 'abs_latitude', 'hemisphere', 'coast_km', 'elevation_m', 'koppen', 'station_density']
PREDICTOR_COLUMNS = ['Country', 'cum_co2_total', 'abs_latitude', 'elevation', 'continentality', 'climate_zone',
                     'hemisphere', 'spatial_block', 'income_group', 'population', 'station_density']
SUPPORT_RECORD = OUTPUT_DIR / 'm1a_geography_qa.csv'  # outcome-free identifiers and row order
# Predictor input digests as recorded by the product-stability audit (outputs/product_stability_summary.json
# "files") and the M1a support record; asserted before any read.
PREDICTOR_INPUT_SHA256 = {
    DEFAULT_INEQUALITY_PATH: '74bbad767106b589b486995b2373e49ba17ae32897e5e0b0cc678205460567f4',
    DEFAULT_FEATURES_PATH: '7d4a95fa624d2c1811073a157f9ffd145b97bbb5d69fa2f4bd97c1d666289d64',
    INCOME_PATH: '79569499049031746c4c56019e430d02c3c182ad7e9011247fb82183227d41a5',
    SUPPORT_RECORD: 'bdc4f1caff740fea7949b636c367afb6c68b423eccebb07c9391b357358425c8',
}
RECORD = 'm1b_redundancy_diagnostic.json'


def m0star_predictors(countries):
    """The M0* predictor frame (primary representation, no per-capita CO2) for ``countries``, in order.

    Mirrors ``src.decomposition.build_country_design`` column for column, without its outcome column,
    and restricts to the fixed country list instead of an outcome-dependent complete-case filter.
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
        raise ValueError('a country in the fixed list has a missing M0* predictor')
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
    if hard_stops:  # data defect: no statistic is meaningful; stop
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


def json_safe(value):
    """JSON-serialisable copy in which non-finite floats become the strings 'inf', '-inf' or 'nan'."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else ('nan' if math.isnan(v) else ('inf' if v > 0 else '-inf'))
    return value


def check_predictor_inputs(pins=None):
    """Fail closed unless every predictor input matches its recorded SHA-256 digest; returns the verified digests."""
    verified = {}
    for path, pin in (PREDICTOR_INPUT_SHA256 if pins is None else pins).items():
        digest = h.sha256(path)
        if digest != pin:
            raise ValueError(f'{Path(path).name}: sha256 {digest} differs from the recorded digest {pin}')
        verified[str(Path(path).resolve().relative_to(ROOT)) if Path(path).resolve().is_relative_to(ROOT) else str(path)] = digest
    return verified


def frame_sha256(frame):
    return hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode('utf-8')).hexdigest()


def verify_record(out_dir, manifest_sha256, features_sha256):
    """Fail closed unless the redundancy record passes and was computed from exactly this measurement package."""
    record = json.loads((Path(out_dir) / RECORD).read_bytes())
    if not isinstance(record, dict) or record.get('status') != 'pass' or record.get('all_hard_stops_pass') is not True \
            or record.get('hard_stops') != []:
        raise ValueError('redundancy record does not record a passing diagnostic with an empty hard_stops list')
    provenance = record.get('provenance')
    if not isinstance(provenance, dict) or provenance.get('measurement_manifest_sha256') != manifest_sha256 \
            or provenance.get('c2_features_sha256') != features_sha256:
        raise ValueError('redundancy record was not computed from the current measurement package')
    return record


def main(out_dir=OUTPUT_DIR):
    logging.getLogger('src.feature_schema').setLevel(logging.ERROR)
    out_dir = Path(out_dir)
    manifest, manifest_sha256 = h.verify_package(out_dir)          # before any C2 value is read
    input_sha256 = check_predictor_inputs()
    features_bytes = h.read_verified(out_dir, h.FEATURES, manifest)
    features = pd.read_csv(io.BytesIO(features_bytes))
    frozen = pd.read_csv(SUPPORT_RECORD, usecols=['iso3', 'Country'])
    if features.columns.tolist() != ['iso3', 'Country', 'baseline_dryness']:
        raise ValueError('C2 features do not have the expected columns')
    if features.iso3.tolist() != frozen.iso3.tolist() or features.Country.tolist() != frozen.Country.tolist():
        raise ValueError('row order mismatch between C2 and the country record')
    country_list = hashlib.sha256(('\n'.join(features.iso3) + '\n').encode()).hexdigest()
    if country_list != manifest.get('support_inputs', {}).get('country_list_sha256'):
        raise ValueError('C2 country list differs from the measurement manifest')
    predictors = m0star_predictors(features.Country)
    if OUTCOME_COL in predictors.columns or predictors.columns.tolist() != PREDICTOR_COLUMNS:
        raise AssertionError('outcome or unexpected column in the redundancy inputs')
    c2 = features.baseline_dryness.to_numpy(float)
    x, columns, _ = design_matrix(predictors)
    result = diagnose(predictors, c2)
    result['status'] = 'pass' if not result['hard_stops'] else 'hard_stop'
    result['all_hard_stops_pass'] = not result['hard_stops']
    result['provenance'] = {
        'measurement_manifest_sha256': manifest_sha256,
        'c2_features_sha256': hashlib.sha256(features_bytes).hexdigest(),
        'measurement_git_commit': manifest['git']['commit'], 'measurement_code_sha256': manifest.get('code_sha256'),
        'measurement_gate_version': manifest['gate_version'],
        'predictor_input_sha256': input_sha256,
        'inequality_columns_read': INEQUALITY_COLUMNS, 'city_feature_columns_read': CITY_COLUMNS,
        'predictor_columns': PREDICTOR_COLUMNS, 'outcome_column_present': OUTCOME_COL in predictors.columns,
        'country_list_sha256': country_list,
        'predictor_frame_sha256': frame_sha256(predictors),
        'design_columns': columns,
        'design_matrix_sha256': hashlib.sha256(np.ascontiguousarray(x, dtype='<f8').tobytes()).hexdigest(),
        'c2_vector_sha256': hashlib.sha256(np.ascontiguousarray(c2, dtype='<f8').tobytes()).hexdigest(),
        'redundancy_code_sha256': h.sha256(Path(__file__)),
        'non_finite_statistics_encoding': "non-finite floats are written as the strings 'inf', '-inf' or 'nan'",
    }
    (out_dir / RECORD).write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(json_safe({k: result.get(k) for k in ['status', 'r2_full_m0star', 'adjusted_r2_full_m0star', 'vif',
                                                         'design_rank', 'rank_with_c2', 'condition_number_design',
                                                         'condition_number_design_with_c2', 'r2_by_group_alone',
                                                         'near_redundant_flag', 'hard_stops']}), indent=2))
    return result


if __name__ == '__main__':
    sys.exit(0 if main()['all_hard_stops_pass'] else 2)
