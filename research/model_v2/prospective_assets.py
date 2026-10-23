"""Assemble dashboard V2 assets exclusively from fresh accepted stage receipts."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

from research.model_v2 import m0, m1b_evaluate as b
from research.model_v2.prospective_refit import digest, write_json
from src.data_io import round_floats
from src.residual_assets import responsibility_comparison

PRIMARY = 'primary_total_co2'
GROUPS = ('geography', 'socioeconomic', 'population', 'emissions', 'residual')
FAMILIES = {'spatial_error': 'sem', 'spatial_lag': 'sar'}


def national_warming(inequality, design, era5):
    """Keep the different complete-case samples visible in the national table."""
    columns = ['Country', 'owid_country', 'continent', 'cum_co2_t_per_capita',
               'trend_c_per_decade', 'trend_c_per_decade_pop_weighted', 'trend_c_per_decade_area_weighted']
    out = inequality[columns].rename(columns={'trend_c_per_decade': 'trend_station',
        'trend_c_per_decade_pop_weighted': 'trend_population', 'trend_c_per_decade_area_weighted': 'trend_area'})
    out = out.merge(design[['Country', 'iso3']], on='Country', how='left', validate='one_to_one')
    out = out.merge(era5[['iso3', 'trend_c_per_decade_era5_area']].rename(
        columns={'trend_c_per_decade_era5_area': 'trend_era5_aligned'}), on='iso3', how='left', validate='many_to_one')
    return out.sort_values('Country', kind='stable').reset_index(drop=True)


def allocation(card):
    named = {key.removeprefix('share_'): value for key, value in card.items() if key.startswith('share_') and isinstance(value, (int, float))}
    result = {**named, 'residual': card['residual_share']}
    if abs(sum(result.values()) - 1.0) > b.SHARE_ATOL:
        raise ValueError('Public named allocation plus residual does not sum to one')
    return result


def paired_summary(paired):
    return {'delta_rmse': paired['delta_rmse'], 'interval': paired['country_bootstrap_95_interval']}


def summary(national, static, spatial, boot, sensitivities, products):
    baseline = static[f'{PRIMARY}/M0star']
    main_boot, pc_boot = boot[PRIMARY], boot['per_capita']
    s = {'n': int(baseline['n']), 'in_sample_r2': baseline['in_sample_r2'],
         'residual_share': baseline['residual_share'], 'primary_cv_r2': baseline['cv_r2'],
         'primary_cv_rmse': baseline['cv_rmse'], 'primary_cv_mae': baseline['cv_mae'],
         'm49_cv_rmse': baseline['secondary']['cv_rmse'], 'm49_cv_r2': baseline['secondary']['cv_r2'],
         'random_cv_rmse': baseline['random_reference_only']['cv_rmse'],
         'random_cv_r2': baseline['random_reference_only']['cv_r2'], 'worst_region': baseline['worst_region'],
         'residual_morans_i_in_sample': baseline['residual_morans_i_in_sample'],
         'residual_morans_i_out_of_fold': baseline['residual_morans_i_cv'],
         'calibration_slope': baseline['calibration_slope'],
         'shares': main_boot['country']['groups'], 'block_shares': main_boot['block']['groups'],
         'p_geography_largest': {kind: main_boot[kind]['p_geography_largest'] for kind in ['country', 'block']},
         'n_bootstrap': main_boot['country']['n_draws'],
         'per_capita_representation_shares': {g: pc_boot['country']['groups'][g]['point'] for g in GROUPS}}
    investigations = []
    for model, key, label in [('M1a', 'geography_measurement', 'Area-consistent geography measurement'),
                               ('M1b', 'baseline_dryness', 'Baseline hydroclimatic dryness'),
                               ('M2', 'latitude_form', 'Flexible latitude functional form')]:
        card = static[f'{PRIMARY}/{model}']
        paired = card['paired_vs_M0star']
        verdict = (card['registered_classification']['summary'] if model == 'M1a' else card['registered_decision']['verdict'])
        investigations.append({'key': key, 'label': label, 'verdict': verdict,
            'primary_cv_r2': card['cv_r2'], 'primary_cv_rmse': card['cv_rmse'],
            'm49_cv_rmse': card['secondary']['cv_rmse'], 'oof_morans_i': card['residual_morans_i_cv'],
            **paired_summary(paired), 'in_sample_r2': card['in_sample_r2'], 'allocation': allocation(card),
            **({'registered_classification': card['registered_classification'], 'historical_decision': 'not promoted; measurement sensitivity'} if model == 'M1a' else {})})
    public_spatial = {}
    for key, family in FAMILIES.items():
        arms = {}
        for weight, public_weight in [('station', 'station'), ('land', 'land_centroid')]:
            arm = spatial[f'{weight}/{PRIMARY}/{family}']
            if arm['status'] != 'computable':
                raise ValueError(f'{weight}/{family} unavailable; public candidate requires explicit scope revision')
            conditions = arm['qualification']
            entry = {**paired_summary(arm['paired']['vs_static']),
                     'm49_rmse_change': conditions['m49_rmse_change'],
                     'worst_region_rmse_change': conditions['worst_region_rmse_change'],
                     'theta': arm['theta'], 'accounting': {k: arm['accounting']['estimands'][k] for k in ['a_static', 'a_dependence', 'a_innovation']},
                     'oof_morans_i': arm['card']['residual_morans_i_cv']}
            entry['qualifies' if weight == 'station' else 'conditions_hold'] = conditions['qualifies']
            if weight == 'station':
                entry['primary_cv_r2'] = arm['card']['cv_r2']
                investigations.append({'key': key, 'label': 'Spatial error model' if family == 'sem' else 'Spatial lag model',
                    'verdict': 'qualifies for prediction' if conditions['qualifies'] else 'does not meet qualification conditions',
                    'primary_cv_r2': arm['card']['cv_r2'], 'primary_cv_rmse': arm['card']['cv_rmse'],
                    'm49_cv_rmse': arm['card']['secondary']['cv_rmse'], 'oof_morans_i': arm['card']['residual_morans_i_cv'],
                    **paired_summary(arm['paired']['vs_static']), 'in_sample_r2': arm['card']['in_sample_one_step_r2'],
                    'allocation': None})
            arms[public_weight] = entry
        public_spatial[key] = {'family': f'M3-{family.upper()}(M0*)', **arms}
    robustness = []
    for weight, public_weight in [('station', 'station'), ('land', 'land_centroid')]:
        robustness.append({'key': f'berkeley_{weight}', 'product': 'Berkeley Earth',
            'label': f'Berkeley Earth, {weight} centroid graph, 500 km',
            **{key: {k: public_spatial[key][public_weight][k] for k in ['delta_rmse', 'interval', 'theta']} for key in FAMILIES}})
    for protocol, key, label in [('buffer_territory_1000km', 'berkeley_1000km', '1000 km territorial exclusion'),
                                  ('buffer_centroid_1500km', 'berkeley_1500km', '1500 km centroid exclusion')]:
        robustness.append({'key': key, 'product': 'Berkeley Earth', 'label': f'Berkeley Earth, {label}',
            **{key: paired_summary(sensitivities[f'{protocol}/{PRIMARY}/{family}']['paired']) for key, family in FAMILIES.items()}})
    product_static = {}
    for arm, key, label in [('aligned_era5', 'era5_aligned', 'ERA5, anomaly-aligned construction'),
                             ('legacy_era5', 'era5_absolute', 'ERA5, absolute-temperature construction')]:
        product = products[f'{arm}/{PRIMARY}']
        card = product['M0star']
        robustness.append({'key': key, 'product': 'ERA5', 'label': label,
            **{key: {**paired_summary(product[family]['paired']['vs_static']), 'theta': product[family]['theta']} for key, family in FAMILIES.items()}})
        shares = allocation(card)
        product_static[key] = {'in_sample_r2': card['in_sample_r2'], 'primary_cv_r2': card['cv_r2'],
            'primary_cv_rmse': card['cv_rmse'], **{f'share_{g}': shares[g] for g in GROUPS if g != 'residual'},
            'residual_agreement_with_berkeley': product['residual_agreement_with_berkeley'],
            'material_change': {'point_geography_not_largest': shares['geography'] <= max(shares[g] for g in GROUPS if g not in ['geography', 'residual']),
                                'responsibility_exceeds_0_10': shares['emissions'] > .10}}
    best = min(static[f'{PRIMARY}/{model}']['paired_vs_M0star']['delta_rmse'] for model in ['M1a', 'M1b', 'M2'])
    return {'note': 'Fresh reconstruction from verified inputs under fixed specifications. Shares describe cross-country variance; spatial accounting describes covariance. Neither establishes causation.',
        'static': s, 'investigations': investigations, 'spatial': public_spatial, 'robustness': robustness,
        'product_static': product_static, 'responsibility_comparison': responsibility_comparison(national),
        'final': {'retained_static_specification': 'M0*',
            'qualifying_spatial_extensions': [f'M3-{family.upper()}(M0*)' for key, family in FAMILIES.items() if public_spatial[key]['station']['qualifies']],
            'historical_qualifying_spatial_extensions': ['M3-SEM(M0*)', 'M3-SAR(M0*)'], 'final_primary_predictive_model': None,
            'selection_status': 'Historical fixed decisions retained; fresh qualification conditions reported alongside.',
            'static_stopping_indicator_fires': best > -.002, 'static_stopping_best_delta': best,
            'static_stopping_threshold': -.002,
            'material_change_to_conclusion': main_boot['material_change']['material_change_to_v1_conclusion']}}


def build(run, out, *, static_stage='v2_static'):
    run, out = Path(run), Path(out)
    if Path(static_stage).name != static_stage:
        raise ValueError('Static stage must be one completed receipt name')
    def load(path):
        return json.loads((run / path).read_text())
    sources = [f'{static_stage}/scorecards.json', 'v2_spatial/scorecards.json', 'v2_final_checks/bootstrap_summary.json',
               'v2_final_checks/sensitivities.json', 'v2_final_checks/products.json',
               'preprocessing/country_inequality.parquet', f'{static_stage}/fresh_country_design.csv', 'era5/aligned_era5_trends.csv']
    inequality = pd.read_parquet(run / sources[5])
    raw = Path(json.loads((run / 'VALIDATION_SPEC.json').read_text())['inputs']['raw_root'])
    owid = raw / 'owid/owid-co2-data.csv'
    identity = inequality[['Country']].copy()
    identity['iso3'] = identity.Country.map(m0.iso3_bridge(inequality, owid))
    national = national_warming(inequality, identity, pd.read_csv(run / sources[7]))
    result = summary(national, *[load(p) for p in sources[:5]])
    result['sources'] = {p.replace('/', '_'): {'path': p, 'sha256': digest(run / p)} for p in sources}
    result['sources']['owid_iso_bridge'] = {'path': 'data/raw/owid/owid-co2-data.csv', 'sha256': digest(owid)}
    result['national_comparison_scope'] = 'Each warming lens uses every finite country pair, independent of model-predictor missingness. V2 fits retain their fixed151-country sample.'
    out.mkdir(parents=True, exist_ok=True)
    national.to_parquet(out / 'national_warming.parquet', index=False, compression='zstd')
    write_json(out / 'residual_structure_summary.json', round_floats(result))
    return {'assets': ['national_warming.parquet', 'residual_structure_summary.json'], 'countries': len(national)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage-name', default='v2_public', help='New immutable public-adapter receipt name')
    parser.add_argument('--static-stage', default='v2_static', help='Accepted completed static receipt')
    args = parser.parse_args(argv)
    if Path(args.stage_name).name != args.stage_name or args.stage_name in ('.', '..'):
        raise ValueError('Public stage must be one confined name')
    if Path(args.static_stage).name != args.static_stage:
        raise ValueError('Static stage must be one completed receipt name')
    spec = importlib.util.spec_from_file_location('validation_runlib', args.run / 'control/runlib.py')
    runlib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runlib)
    sources = [args.run / p for p in [f'{args.static_stage}/scorecards.json', 'v2_spatial/scorecards.json',
        'v2_final_checks/bootstrap_summary.json', 'v2_final_checks/sensitivities.json', 'v2_final_checks/products.json',
        'preprocessing/country_inequality.parquet', f'{args.static_stage}/fresh_country_design.csv', 'era5/aligned_era5_trends.csv']]
    sources += [args.run / 'VALIDATION_SPEC.json', runlib.RAW / 'owid/owid-co2-data.csv']
    runlib.stage(args.stage_name, sources, lambda out: build(args.run, out, static_stage=args.static_stage), runner=Path(__file__),
                 dependencies=[args.static_stage, 'v2_spatial', 'v2_final_checks'])


if __name__ == '__main__':
    main()
