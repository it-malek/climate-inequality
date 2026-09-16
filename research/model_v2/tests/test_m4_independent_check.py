"""M4 independent check: synthetic miniature M4 packages only.

Every artifact below is generated in ``tmp_path`` by a small writer in this file that follows the saved M4
artifact formats and uses a different numerical route from the checker (per-draw pandas frames, SVD least
squares, ``matrix_rank``, subset-weight Shapley shares, ``scipy.stats.spearmanr``). No repository data file is
read, no outcome is real and no evaluator code is imported. Out-of-fold predictions are fabricated: the checker
takes them as given and recomputes only what follows from them.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import io
import itertools
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from research.model_v2 import m4_independent_check as ic

N = 40
N_BOOT = 30
BLOCKS = ('Europe', 'Africa', 'Oceania', 'Asia', 'Americas')    # first-appearance order, deliberately unsorted
INCOME = ('High-income countries', 'Low-income countries', 'Upper-middle-income countries')
CATS = ('climate_zone', 'hemisphere', 'spatial_block', 'income_group')
NUMERIC = {'primary_total_co2': ['cum_co2_total', 'abs_latitude', 'elevation', 'continentality', 'population',
                                 'station_density'],
           'per_capita': ['cum_co2_per_capita', 'abs_latitude', 'elevation', 'continentality', 'population',
                          'station_density']}
GROUP_OF = {'cum_co2_total': 'emissions', 'cum_co2_per_capita': 'emissions', 'abs_latitude': 'geography',
            'elevation': 'geography', 'continentality': 'geography', 'climate_zone': 'geography',
            'hemisphere': 'geography', 'spatial_block': 'geography', 'income_group': 'socioeconomic',
            'population': 'population', 'station_density': 'population'}
GROUPS = ('emissions', 'geography', 'socioeconomic', 'population')
SHARES = (*GROUPS, 'residual')
ADDED = ['abs_latitude_ns1', 'abs_latitude_ns2', 'hemisphere=S:abs_latitude']
STATIC_MODEL = {'M0*': 'M0star', 'M2': 'M2'}
SENSITIVITY = ('buffer_territory_1000km', 'buffer_centroid_1500km')
SKIPPED_EXTENSION_CELL = ('buffer_centroid_1500km', 'per_capita')     # a recorded non-computable sensitivity cell
PROTOCOLS = ('primary_loco', 'm49_subregion_lo', 'random10')
CARD_KEY = {'primary_loco': None, 'm49_subregion_lo': 'secondary', 'random10': 'random_reference_only'}
ARTIFACTS = ('m4_bootstrap_draws.csv', 'm4_bootstrap_summary.json', 'm4_sensitivities.json',
             'm4_sensitivity_predictions.csv', 'm4_products.json', 'm4_product_predictions.csv',
             'm4_consolidated_table.json', 'm4_consolidated_table.csv', 'm4_inventory_status.json',
             'm4_final_specification.json', 'm4_provenance.json')
CONTROLS = ('a_perturbed_draw_share', 'b_shifted_percentile', 'c_changed_naming_field',
            'd_perturbed_sensitivity_prediction')


# ---------------------------------------------------------------------
# A synthetic world and a miniature evaluator on a different route
# ---------------------------------------------------------------------


def world(seed, latitude_ties):
    rng = np.random.default_rng(seed)
    block = np.array([*BLOCKS, *rng.choice(['Europe', 'Africa', 'Asia', 'Americas'], N - 7), 'Oceania', 'Oceania'])
    income = np.empty(N, dtype=object)
    for b in BLOCKS:                      # every block carries at least two income levels
        members = np.flatnonzero(block == b)
        income[members] = [INCOME[k % 3] for k in range(len(members))]
    oceania = np.flatnonzero(block == 'Oceania')
    climate = rng.choice(['A', 'B', 'C'], N).astype(object)
    climate[oceania[0]] = 'D'             # a one-country level: rank-deficient country draws
    hemisphere = np.array(['N'] * N, dtype=object)
    hemisphere[oceania] = 'S'
    hemisphere[np.flatnonzero(block == 'Americas')[:3]] = 'S'
    a = rng.uniform(2.0, 66.0, N)
    if latitude_ties:                     # twelve identical minima: some draws have a repeated knot
        a[rng.choice(N, 12, replace=False)] = 1.5
    w = {'iso3': np.array([f'S{i:02d}' for i in range(N)]), 'Country': np.array([f'Country {i}' for i in range(N)]),
         'climate_zone': climate, 'hemisphere': hemisphere, 'spatial_block': block, 'income_group': income,
         'cum_co2_total': rng.uniform(0.5, 3.5, N), 'abs_latitude': a, 'elevation': rng.uniform(0, 2500, N),
         'continentality': rng.uniform(0, 900, N), 'population': rng.uniform(5.5, 9.2, N),
         'station_density': rng.uniform(0, 2.5, N)}
    w['cum_co2_per_capita'] = w['cum_co2_total'] - w['population'] + 6.0
    w['y'] = (0.15 + 0.004 * a + 5e-5 * (a - 30.0) ** 2 + 0.01 * w['cum_co2_total']
              + 0.02 * (block == 'Asia') + rng.normal(0, 0.03, N))
    return w


def frame_of(w, representation, outcome):
    frame = pd.DataFrame({c: w[c] for c in ('iso3', *CATS, *NUMERIC[representation])})
    frame['warming_trend'] = outcome
    return frame


def encode(frame, representation, model):
    names, blocks = ['intercept'], [np.ones(len(frame))]
    for c in NUMERIC[representation]:
        names.append(c)
        blocks.append(frame[c].to_numpy(float))
    for c in CATS:
        values = frame[c].astype(str).to_numpy()
        for level in sorted(frame[c].astype(str).unique().tolist())[1:]:
            names.append(f'{c}={level}')
            blocks.append((values == level).astype(float))
    if model == 'M2':
        a = frame.abs_latitude.to_numpy(float)
        lower, upper = np.quantile(a, [1 / 3, 2 / 3], method='linear')
        knots = (float(a.min()), float(lower), float(upper), float(a.max()))
        if not all(b - k > 1e-6 for k, b in zip(knots, knots[1:])):
            raise ValueError(f'knots are not strictly increasing by more than 1e-06 degrees: {knots}')

        def d(k):
            t, u = np.maximum(a - knots[k], 0.0), np.maximum(a - knots[3], 0.0)
            return (t * t * t - u * u * u) / (knots[3] - knots[k])
        blocks += [d(0) - d(2), d(1) - d(2), (frame.hemisphere.to_numpy() == 'S').astype(float) * a]
        names += ADDED
    groups = np.array(['intercept' if n == 'intercept' else 'geography' if n in ADDED else GROUP_OF[n.split('=')[0]]
                       for n in names])
    return np.column_stack(blocks), names, groups


def lmg(frame, representation, model):
    x, _, groups = encode(frame, representation, model)
    y = frame.warming_trend.to_numpy(float)
    tss = float(np.sum((y - y.mean()) ** 2))
    present = [g for g in GROUPS if (groups == g).any()]
    r2 = {}
    for size in range(len(present) + 1):
        for coalition in itertools.combinations(present, size):
            keep = (groups == 'intercept') | np.isin(groups, coalition)
            beta = np.linalg.lstsq(x[:, keep], y, rcond=None)[0]
            r2[frozenset(coalition)] = float(1 - np.sum((y - x[:, keep] @ beta) ** 2) / tss)
    k = len(present)
    shares = dict.fromkeys(GROUPS, 0.0)
    for g in present:
        others = [h for h in present if h != g]
        shares[g] = sum(math.factorial(s) * math.factorial(k - s - 1) / math.factorial(k)
                        * (r2[frozenset(sub) | {g}] - r2[frozenset(sub)])
                        for s in range(k) for sub in itertools.combinations(others, s))
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return {'shares': shares, 'total': r2[frozenset(present)], 'fitted': x @ beta,
            'rank_deficient': bool(np.linalg.matrix_rank(x) < x.shape[1]), 'columns': x.shape[1]}


def bootstrap(frame, representation, model, kind):
    rng = np.random.default_rng(0 if kind == 'country' else 1)
    blocks = pd.unique(frame['spatial_block'])
    rows = []
    for b in range(N_BOOT):
        if kind == 'country':
            draw = frame.iloc[rng.integers(0, len(frame), len(frame))].reset_index(drop=True)
        else:
            chosen = rng.choice(blocks, size=len(blocks), replace=True)
            draw = pd.concat([frame[frame['spatial_block'] == c] for c in chosen], ignore_index=True)
        row = {'bootstrap': kind, 'draw': b, 'n_rows': len(draw)}
        try:
            result = lmg(draw, representation, model)
            values = [result['shares'][g] for g in GROUPS] + [1.0 - result['total']]
            if not all(np.isfinite(values)):
                raise ValueError('non-finite share')
            row.update({'usable': True, 'failure': '', 'rank_deficient': result['rank_deficient'],
                        **dict(zip(SHARES, values))})
        except ValueError as error:
            row.update({'usable': False, 'failure': f'DegenerateLatitudeBasis: {error}', 'rank_deficient': '',
                        **dict.fromkeys(SHARES, np.nan)})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(point, draws):
    out = {}
    for kind, frame in draws.groupby('bootstrap', sort=False):
        usable = frame[frame.usable.astype(bool)]
        groups = {}
        for g in SHARES:
            values = usable[g].to_numpy(float)
            lo, hi = np.percentile(values, [2.5, 97.5])
            groups[g] = {'point': point[g], 'mean': float(values.mean()), 'sd': float(values.std(ddof=1)),
                         'ci_low': float(lo), 'ci_high': float(hi)}
        margin = usable['geography'].to_numpy(float) - usable[['emissions', 'socioeconomic', 'population']].max(axis=1)
        out[kind] = {'n_draws': len(frame), 'usable': len(usable), 'degenerate': int((~frame.usable.astype(bool)).sum()),
                     'rank_deficient_usable': int(usable.rank_deficient.astype(bool).sum()), 'groups': groups,
                     'p_geography_largest': float(np.mean([max(GROUPS, key=lambda g: r[g]) == 'geography'
                                                           for _, r in usable.iterrows()])),
                     'geography_margin_percentile_2_5': float(np.percentile(margin.to_numpy(float), 2.5)),
                     'seed': 0 if kind == 'country' else 1}
    others = max(point[g] for g in GROUPS if g != 'geography')
    material = {'point_geography_not_largest': bool(not point['geography'] > others),
                'interval_geography_not_established_country': out['country']['geography_margin_percentile_2_5'] <= 0,
                'interval_geography_not_established_block': out['block']['geography_margin_percentile_2_5'] <= 0,
                'responsibility_exceeds_0_10': bool(point['emissions'] > 0.10),
                'responsibility_upper_country_ci_exceeds_0_10': out['country']['groups']['emissions']['ci_high'] > 0.10}
    material['material_change_to_v1_conclusion'] = bool(material['point_geography_not_largest']
                                                         or material['interval_geography_not_established_country']
                                                         or material['responsibility_exceeds_0_10'])
    return {**out, 'material_change': material, 'rule': 'synthetic'}


def metrics(y, yhat):
    e = y - yhat
    return {'cv_r2': float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), 'cv_rmse': float(np.sqrt(np.mean(e ** 2))),
            'cv_mae': float(np.mean(np.abs(e)))}


def paired(y, new, reference):
    index = np.random.default_rng(0).integers(0, len(y), size=(2000, len(y)))
    delta = np.sqrt(np.mean((y - new)[index] ** 2, axis=1)) - np.sqrt(np.mean((y - reference)[index] ** 2, axis=1))
    return {'delta_rmse': float(np.sqrt(np.mean((y - new) ** 2)) - np.sqrt(np.mean((y - reference) ** 2))),
            'country_bootstrap_95_interval': np.quantile(delta, [0.025, 0.975]).tolist(), 'resamples': 2000, 'seed': 0}


def csv_text(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator='\n')
    return buffer.getvalue()


def write_json(payload, path):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(package):
    final = json.loads((package / 'm4_final_specification.json').read_text())
    manifest = {'result': 'synthetic M4 final assessment', 'evaluator_commit': 'synthetic', 'inventory_complete': True,
                'final': {k: final[k] for k in ('retained_static_specification', 'qualifying_spatial_extensions',
                                                'final_primary_predictive_model')},
                'integrity_passed': True, 'artifact_sha256': {n: sha(package / n) for n in ARTIFACTS},
                'non_deterministic_artifacts': ['m4_run_metadata.json']}
    write_json(manifest, package / 'm4_result_manifest.json')


# ---------------------------------------------------------------------
# The synthetic package and its inputs
# ---------------------------------------------------------------------


def write_package(root: Path, static='M0*', qualifying=('M3-SEM(M0*)',), seed=11, latitude_ties=False):
    w = world(seed, latitude_ties)
    rng = np.random.default_rng(seed + 100)
    y = w['y']
    model = STATIC_MODEL[static]
    qualifying = list(qualifying)
    inputs_dir, package = root / 'inputs', root / 'm4_final'
    inputs_dir.mkdir(parents=True)
    package.mkdir()

    # Inputs: rounded numerics in the country table (never used for scoring), exact ones in the design file.
    countries = pd.DataFrame({'Country': w['Country'], 'warming_trend': np.round(y, 3), 'iso3': w['iso3'],
                              'abs_latitude': np.round(w['abs_latitude'], 2), **{c: w[c] for c in CATS}})
    (inputs_dir / 'm0_countries.csv').write_text(csv_text(countries))
    design_rows = []
    for representation in NUMERIC:
        x, names, groups = encode(frame_of(w, representation, y), representation, 'M0*')
        for j, (column, group) in enumerate(zip(names, groups)):
            design_rows += [{'representation': representation, 'model': 'M0star', 'iso3': w['iso3'][i],
                             'column': column, 'group': group, 'value': float(x[i, j])} for i in range(N)]
            design_rows += [{'representation': representation, 'model': 'M1b', 'iso3': w['iso3'][i],
                             'column': column, 'group': group, 'value': float(x[i, j]) + 1.0} for i in range(N)]
    (inputs_dir / 'design.csv').write_text(csv_text(pd.DataFrame(design_rows)))
    outcome = pd.concat([pd.DataFrame({'variant': variant, 'iso3': w['iso3'], 'Country': w['Country'],
                                       'observed': y + shift, 'fitted': 0.0})
                         for variant, shift in (('M0_legacy_representation', 1.0), ('drop_per_capita', 0.0))])
    (inputs_dir / 'outcome.csv').write_text(csv_text(outcome))

    # Branch-dependent committed records.
    label = 'supported and promoted' if static == 'M2' else 'not supported'
    stopping = {'point_delta_rmse_vs_m0star': {'M1a': 0.0023, 'M1b': 0.0017, 'M2': -0.001}, 'best': -0.001,
                'threshold': -0.002, 'fires': True, 'm2_decides': True}
    write_json({'retained_static_specification': static, 'verdict': label, 'integrity_passed': True},
               inputs_dir / 'm2_result_manifest.json')
    write_json({'verdict': {'verdict': label}, 'static_stopping_indicator': stopping}, inputs_dir / 'm2_scorecard.json')
    final_model = static if not qualifying else qualifying[0] if len(qualifying) == 1 else None
    naming = {'retained_static_specification': static, 'qualifying_spatial_extensions': qualifying,
              'final_primary_predictive_model': final_model, 'reason': 'synthetic'}
    write_json({'weights': 'station', 'final_naming': naming, 'integrity_passed': True},
               inputs_dir / 'm3_station_manifest.json')

    # A. Bootstrap draws and summaries.
    summaries, draw_frames = {}, []
    for representation in NUMERIC:
        frame = frame_of(w, representation, y)
        full = lmg(frame, representation, model)
        point = {**full['shares'], 'residual': 1.0 - full['total']}
        draws = pd.concat([bootstrap(frame, representation, model, 'country'),
                           bootstrap(frame, representation, model, 'block')], ignore_index=True)
        summaries[representation] = summarize(point, draws)
        draw_frames.append(draws.assign(representation=representation, model=model))
    (package / 'm4_bootstrap_draws.csv').write_text(csv_text(pd.concat(draw_frames, ignore_index=True)))
    write_json(summaries, package / 'm4_bootstrap_summary.json')

    # B. Sensitivities from fabricated predictions (M0* identical across representations, as in the real design).
    models = [model] + (['M0star'] if model == 'M2' else []) + qualifying
    sensitivities, sensitivity_rows, anchors = {}, [], {}
    noise = {(p, m): rng.normal(0, 0.035, N) for p in SENSITIVITY for m in models}
    for protocol in SENSITIVITY:
        anchors[protocol] = metrics(y, y + noise[(protocol, 'M0star')]) if 'M0star' in models else None
        for representation in NUMERIC:
            entry, predictions = {}, {}
            for name in models:
                if qualifying and name == qualifying[0] and (protocol, representation) == SKIPPED_EXTENSION_CELL:
                    entry[name] = {'status': 'non_computable', 'failure': 'synthetic structural failure'}
                    continue
                shift = 0.0 if name == 'M0star' or representation == 'primary_total_co2' else 1e-4
                predictions[name] = y + noise[(protocol, name)] + shift
                entry[name] = {'status': 'computed', **metrics(y, predictions[name])}
            if model == 'M2' and 'M2' in predictions and 'M0star' in predictions:
                entry['paired_S_minus_M0star'] = paired(y, predictions['M2'], predictions['M0star'])
            for name in qualifying:
                if name in predictions and model in predictions:
                    entry[f'paired_{name}_minus_S'] = paired(y, predictions[name], predictions[model])
            sensitivities[f'{protocol}/{representation}'] = entry
            for name, values in predictions.items():
                sensitivity_rows += [{'protocol': protocol, 'representation': representation, 'model': name,
                                      'iso3': w['iso3'][i], 'observed': y[i], 'prediction': values[i]} for i in range(N)]
    sensitivities['scope'] = 'synthetic'
    m0_sensitivity = {'sensitivity_territory_1000km': anchors['buffer_territory_1000km'] or {},
                      'sensitivity_centroid_1500km': anchors['buffer_centroid_1500km'] or {}}
    write_json(m0_sensitivity, inputs_dir / 'm0_sensitivity.json')
    write_json(sensitivities, package / 'm4_sensitivities.json')
    (package / 'm4_sensitivity_predictions.csv').write_text(csv_text(pd.DataFrame(sensitivity_rows)))

    # C. Products: ERA5 outcomes, M0* in-sample R2 anchors, S cards and fabricated CV predictions.
    era5 = {'aligned_era5': y + rng.normal(0, 0.02, N), 'legacy_era5': y + rng.normal(0, 0.025, N)}
    shuffled = rng.permutation(N)
    aligned = pd.DataFrame({'iso3': w['iso3'][shuffled], 'trend_c_per_decade_era5_area': era5['aligned_era5'][shuffled],
                            'era5_cell_coverage': 1.0})
    (inputs_dir / 'aligned.csv').write_text(csv_text(aligned))
    pd.DataFrame({'iso3': w['iso3'][shuffled[::-1]],
                  'trend_c_per_decade_era5_area': era5['legacy_era5'][shuffled[::-1]]}).to_parquet(
        inputs_dir / 'legacy.parquet', index=False)
    for product, name in (('aligned_era5', 'aligned_summary.json'), ('legacy_era5', 'legacy_summary.json')):
        m0 = lmg(frame_of(w, 'primary_total_co2', era5[product]), 'primary_total_co2', 'M0*')
        write_json({'r2': {'berkeley': 0.6, 'era5': m0['total']}}, inputs_dir / name)
    products, product_rows = {}, []
    for product, outcome_p in era5.items():
        for representation in NUMERIC:
            result = lmg(frame_of(w, representation, outcome_p), representation, model)
            berkeley = lmg(frame_of(w, representation, y), representation, model)
            yhat = {p: outcome_p + rng.normal(0, 0.03, N) for p in PROTOCOLS}
            card = {'in_sample_r2': float(1 - np.sum((outcome_p - result['fitted']) ** 2)
                                          / np.sum((outcome_p - outcome_p.mean()) ** 2)),
                    **metrics(outcome_p, yhat['primary_loco']),
                    'secondary': metrics(outcome_p, yhat['m49_subregion_lo']),
                    'random_reference_only': metrics(outcome_p, yhat['random10']),
                    **{f'share_{g}': float(v) for g, v in result['shares'].items()},
                    'residual_share': 1.0 - result['total']}
            shares = result['shares']
            a, b = outcome_p - result['fitted'], y - berkeley['fitted']
            entry = {'S': {'model': model, 'card': card,
                           'material_change_point': {
                               'point_geography_not_largest': bool(not shares['geography'] > max(
                                   shares[g] for g in GROUPS if g != 'geography')),
                               'responsibility_exceeds_0_10': bool(shares['emissions'] > 0.10)},
                           'residual_agreement_with_berkeley': {
                               'pearson': float(np.corrcoef(a, b)[0, 1]),
                               'spearman': float(stats.spearmanr(a, b).statistic),
                               'sign_agreement_n': int(np.sum(np.sign(a) == np.sign(b)))},
                           'anchor': 'synthetic'}}
            for k, name in enumerate(qualifying):
                ext = {p: outcome_p + rng.normal(0, 0.025, N) for p in PROTOCOLS}
                entry[name] = {'status': 'computed', 'theta': 0.3,
                               'card': {**metrics(outcome_p, ext['primary_loco']),
                                        'secondary': metrics(outcome_p, ext['m49_subregion_lo']),
                                        'random_reference_only': metrics(outcome_p, ext['random10'])},
                               'arm_conditions': {'qualifies': k == 0, 'descriptive_only': True},
                               'residual_agreement_with_berkeley': None}
                product_rows += [{'product': product, 'representation': representation, 'model': name,
                                  'protocol': p, 'iso3': w['iso3'][i], 'observed': outcome_p[i],
                                  'prediction': ext[p][i]} for p in PROTOCOLS for i in range(N)]
            product_rows += [{'product': product, 'representation': representation, 'model': model, 'protocol': p,
                              'iso3': w['iso3'][i], 'observed': outcome_p[i], 'prediction': yhat[p][i]}
                             for p in PROTOCOLS for i in range(N)]
            products[f'{product}/{representation}'] = entry
    for name in qualifying:
        products[f'product_sensitive/{name}'] = not products['aligned_era5/primary_total_co2'][name][
            'arm_conditions']['qualifies']
    products['interpretation'] = 'synthetic'
    write_json(products, package / 'm4_products.json')
    (package / 'm4_product_predictions.csv').write_text(csv_text(pd.DataFrame(product_rows)))

    # D. Consolidation, inventory, final record, provenance.
    rows = [{'stage': 'M0*', 'status': 'approved', 'primary_cv_rmse': 0.04}, {'stage': 'M2', 'status': label,
                                                                             'primary_cv_rmse': 0.041}]
    write_json({'rows': rows, 'legacy_block': {}, 'separations': []}, package / 'm4_consolidated_table.json')
    (package / 'm4_consolidated_table.csv').write_text(csv_text(pd.json_normalize(rows)))
    permitted = {'I14', 'I21', 'I24', 'I26', 'I27'}
    write_json({'items': [{'id': f'I{i}', 'status': 'permitted disposition' if f'I{i}' in permitted
                           else 'verified evidence', 'evidence': 'synthetic'} for i in range(1, 28)],
                'retained_static_specification': static, 'qualifying_spatial_extensions': qualifying},
               package / 'm4_inventory_status.json')
    geography = summaries['primary_total_co2']['country']['groups']['geography']
    three = {static: {'primary_rmse_minus_m0star': 0.0, 'oof_morans_i': 0.1, 'geography_share': geography['point'],
                      'geography_share_country_bootstrap_95': [geography['ci_low'], geography['ci_high']]}}
    three.update({name: {'primary_rmse_minus_m0star': -0.003, 'oof_morans_i': 0.02,
                         'geography_share_filtered_trend': 0.2, 'geography_share_interval': 'not applicable (spec §3.3)'}
                  for name in qualifying})
    final = {k: naming[k] for k in ('retained_static_specification', 'qualifying_spatial_extensions',
                                    'final_primary_predictive_model')}
    final.update({'final_naming_reason': naming['reason'], 'm2_label': label,
             'static_stopping_indicator': stopping, 'm3_family_dispositions': {},
             'three_numbers_per_retained_stage': three,
             'material_change_to_v1_conclusion': {r: summaries[r]['material_change'] for r in NUMERIC},
             'commits': {}, 'residual_description': 'synthetic'})
    write_json(final, package / 'm4_final_specification.json')
    write_json({'synthetic': True}, package / 'm4_provenance.json')
    write_manifest(package)

    inputs = ic.Inputs(countries=inputs_dir / 'm0_countries.csv', design=inputs_dir / 'design.csv',
                       outcome=inputs_dir / 'outcome.csv', m0_sensitivity=inputs_dir / 'm0_sensitivity.json',
                       aligned_era5=inputs_dir / 'aligned.csv', aligned_summary=inputs_dir / 'aligned_summary.json',
                       legacy_era5=inputs_dir / 'legacy.parquet', legacy_summary=inputs_dir / 'legacy_summary.json',
                       m2_manifest=inputs_dir / 'm2_result_manifest.json', m2_scorecard=inputs_dir / 'm2_scorecard.json',
                       m3_station_manifest=inputs_dir / 'm3_station_manifest.json')
    x_m0, _, _ = encode(frame_of(w, 'primary_total_co2', y), 'primary_total_co2', 'M0*')
    expected = {'n': N, 'n_boot': N_BOOT, 'full_columns': {'M0*': x_m0.shape[1], 'M2': x_m0.shape[1] + 3}}
    return {'root': root, 'package': package, 'inputs': inputs, 'expected': expected, 'world': w,
            'draws': pd.concat(draw_frames, ignore_index=True), 'static': static, 'qualifying': qualifying}


def check(result, package=None):
    return ic.run(package or result['package'], result['inputs'], result['expected'])


def copy_package(result, tmp_path):
    target = tmp_path / 'm4_final'
    shutil.copytree(result['package'], target)
    return target


@pytest.fixture(scope='module')
def branch_a(tmp_path_factory):
    result = write_package(tmp_path_factory.mktemp('m4_branch_a'))
    return result, check(result)


@pytest.fixture(scope='module')
def branch_b(tmp_path_factory):
    result = write_package(tmp_path_factory.mktemp('m4_branch_b'), static='M2', qualifying=(), seed=3,
                           latitude_ties=True)
    return result, check(result)


@pytest.fixture(scope='module')
def two_extensions(tmp_path_factory):
    result = write_package(tmp_path_factory.mktemp('m4_two'), qualifying=('M3-SEM(M0*)', 'M3-SAR(M0*)'), seed=7)
    return result, check(result)


# ---------------------------------------------------------------------
# The reconstruction on consistent packages
# ---------------------------------------------------------------------


@pytest.mark.parametrize('scenario', ['branch_a', 'branch_b', 'two_extensions'])
def test_consistent_synthetic_package_passes_every_section(scenario, request):
    result, report = request.getfixturevalue(scenario)
    assert report['failed_checks'] == []
    assert report['all_checks_pass'] is True
    assert all(block['passes'] for block in report['section_summary'].values())
    assert report['negative_controls_all_detected'] is True
    assert report['bootstrap_recomputation']['draws_recomputed'] == N_BOOT * 2 * 2
    assert report['retained_static_specification'] == result['static']


def test_fixtures_exercise_degenerate_rank_deficient_and_non_computable_cases(branch_a, branch_b, two_extensions):
    a, b = branch_a[0]['draws'], branch_b[0]['draws']
    assert (~b.usable.astype(bool)).any() and b.usable.astype(bool).any()     # repeated knots in branch B
    assert (a.usable.astype(bool)).all()
    assert (a.rank_deficient == True).any()                                   # noqa: E712
    sensitivities = json.loads((branch_a[0]['package'] / 'm4_sensitivities.json').read_text())
    assert sensitivities['buffer_centroid_1500km/per_capita']['M3-SEM(M0*)']['status'] == 'non_computable'
    assert 'paired_S_minus_M0star' in json.loads((branch_b[0]['package'] / 'm4_sensitivities.json').read_text())[
        'buffer_territory_1000km/primary_total_co2']
    assert two_extensions[1]['sections']['6_final_record']['passes']
    final = json.loads((two_extensions[0]['package'] / 'm4_final_specification.json').read_text())
    assert final['final_primary_predictive_model'] is None


def test_every_draw_is_recomputed_and_counts_agree(branch_b):
    result, report = branch_b
    draws = result['draws']
    for representation in NUMERIC:
        for kind in ('country', 'block'):
            saved = draws[(draws.representation == representation) & (draws.bootstrap == kind)]
            counts = report['bootstrap_recomputation']['recomputed_counts'][f'{representation}/{kind}']
            assert counts['usable'] == int(saved.usable.astype(bool).sum())
    checks = {r['check']: r for r in report['sections']['2_bootstrap_draws']['comparisons']}
    assert checks['primary_total_co2/country/shares/geography']['n'] == int(
        draws[(draws.representation == 'primary_total_co2') & (draws.bootstrap == 'country')].usable.astype(bool).sum())


@pytest.mark.parametrize('name', CONTROLS)
def test_each_negative_control_is_detected_from_a_clean_baseline(branch_a, name):
    control = branch_a[1]['negative_controls'][name]
    assert control['detected'] is True
    assert control['baseline_failed_checks'] == []
    assert control['failed_checks']


def test_controls_fail_the_specific_comparisons(branch_a):
    controls = branch_a[1]['negative_controls']
    assert 'primary_total_co2/country/geography/mean' in controls['a_perturbed_draw_share']['also_failed_summary_checks']
    assert controls['b_shifted_percentile']['failed_checks'] == ['primary_total_co2/country/geography/ci_low']
    assert 'final/final_primary_predictive_model_vs_m3_station_naming' in controls['c_changed_naming_field'][
        'failed_checks']
    assert any('M0star_anchor' in c for c in controls['d_perturbed_sensitivity_prediction']['failed_checks'])


def test_json_report_is_written_and_exit_zero(branch_a, tmp_path):
    result = branch_a[0]
    out = tmp_path / 'verification' / 'm4_independent_check.json'
    code = ic.main(['--package', str(result['package']), '--out', str(out)], inputs=result['inputs'],
                   expected=result['expected'])
    written = json.loads(out.read_text())
    assert code == 0
    assert written['all_checks_pass'] is True and written['negative_controls_all_detected'] is True
    assert set(written['negative_controls']) == set(CONTROLS)
    assert {'independent', 'not_independent'} <= set(written['scope'])
    assert all({'passes', 'max_abs_diff'} <= set(block) for block in written['section_summary'].values())


def test_real_structure_expectation_fails_on_the_synthetic_sizes(branch_a, tmp_path):
    result = branch_a[0]
    code = ic.main(['--package', str(result['package']), '--out', str(tmp_path / 'out.json')], inputs=result['inputs'],
                   expected={**result['expected'], 'n': 151})
    assert code == 3


# ---------------------------------------------------------------------
# Tampering
# ---------------------------------------------------------------------


def test_manifest_digest_tampering_is_refused(branch_a, tmp_path):
    result = branch_a[0]
    package = copy_package(result, tmp_path)
    path = package / 'm4_sensitivities.json'
    path.write_text(path.read_text().replace('synthetic', 'synthetic ', 1))
    with pytest.raises(ic.DigestRefused):
        check(result, package)
    out = tmp_path / 'refused.json'
    code = ic.main(['--package', str(package), '--out', str(out)], inputs=result['inputs'], expected=result['expected'])
    assert code == 3
    assert 'refused' in json.loads(out.read_text())


def test_unlisted_artifact_is_refused(branch_a, tmp_path):
    result = branch_a[0]
    package = copy_package(result, tmp_path)
    manifest = json.loads((package / 'm4_result_manifest.json').read_text())
    del manifest['artifact_sha256']['m4_bootstrap_draws.csv']
    (package / 'm4_result_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ic.DigestRefused):
        check(result, package)


def rewrite_draws(package, edit):
    frame = pd.read_csv(package / 'm4_bootstrap_draws.csv', dtype=str, keep_default_na=False)
    edit(frame)
    (package / 'm4_bootstrap_draws.csv').write_text(csv_text(frame))
    write_manifest(package)


def test_consistently_rehashed_wrong_draw_share_is_caught(branch_a, tmp_path):
    result = branch_a[0]
    package = copy_package(result, tmp_path)

    def edit(frame):
        row = frame.index[(frame.representation == 'per_capita') & (frame.bootstrap == 'block')][4]
        frame.loc[row, 'population'] = repr(float(frame.loc[row, 'population']) + 1e-6)
    rewrite_draws(package, edit)
    failed = check(result, package)
    assert 'per_capita/block/shares/population' in failed['sections']['2_bootstrap_draws']['failed_checks']
    assert not failed['section_summary']['3_bootstrap_summaries_and_material_change']['passes']


def test_consistently_rehashed_flipped_flags_are_caught(branch_b, tmp_path):
    result = branch_b[0]
    package = copy_package(result, tmp_path)

    def edit(frame):
        rows = frame.index[(frame.representation == 'primary_total_co2') & (frame.bootstrap == 'country')]
        usable = next(r for r in rows if frame.loc[r, 'usable'] == 'True')
        frame.loc[usable, 'rank_deficient'] = 'False' if frame.loc[usable, 'rank_deficient'] == 'True' else 'True'
        degenerate = next(r for r in rows if frame.loc[r, 'usable'] == 'False')
        frame.loc[degenerate, 'usable'] = 'True'
    rewrite_draws(package, edit)
    failed = check(result, package)['sections']['2_bootstrap_draws']['failed_checks']
    assert {'primary_total_co2/country/rank_deficient', 'primary_total_co2/country/usable'} <= set(failed)


def test_consistently_rehashed_wrong_naming_and_inventory_are_caught(branch_a, tmp_path):
    result = branch_a[0]
    package = copy_package(result, tmp_path)
    final = json.loads((package / 'm4_final_specification.json').read_text())
    final['final_primary_predictive_model'] = 'M0*'
    write_json(final, package / 'm4_final_specification.json')
    inventory = json.loads((package / 'm4_inventory_status.json').read_text())
    inventory['items'][9]['status'] = 'missing'
    write_json(inventory, package / 'm4_inventory_status.json')
    write_manifest(package)
    failed = set(check(result, package)['sections']['6_final_record']['failed_checks'])
    assert {'final/naming_rule', 'final/final_primary_predictive_model_vs_m3_station_naming',
            'inventory/statuses_permitted'} <= failed


def test_wrong_product_anchor_is_caught(branch_a, tmp_path):
    result = branch_a[0]
    summary = json.loads(Path(result['inputs'].aligned_summary).read_text())
    summary['r2']['era5'] += 1e-8
    path = tmp_path / 'aligned_summary.json'
    write_json(summary, path)
    failed = ic.run(result['package'], dataclasses.replace(result['inputs'], aligned_summary=path), result['expected'])
    assert 'aligned_era5/primary_total_co2/S/in_sample_r2_vs_m0_5_r2_era5' in failed['sections']['5_product_check'][
        'failed_checks']


# ---------------------------------------------------------------------
# The independent mathematics
# ---------------------------------------------------------------------


def test_projection_r2_matches_least_squares_including_rank_deficiency():
    rng = np.random.default_rng(4)
    x = np.column_stack([np.ones(30), rng.normal(size=(30, 4)) * [1, 100, 1e-2, 5]])
    deficient = np.column_stack([x, x[:, 1] + x[:, 2], np.zeros(30)])
    y = rng.normal(size=30)
    tss = float(np.sum((y - y.mean()) ** 2))
    for z in (x, deficient):
        beta = np.linalg.lstsq(z, y, rcond=None)[0]
        expected = 1 - np.sum((y - z @ beta) ** 2) / tss
        got = ic.projection(z, y, tss)
        assert abs(got['r2'] - expected) <= 1e-12
        assert got['rank'] == np.linalg.matrix_rank(z)


def test_permutation_shares_equal_subset_weight_shares_with_a_null_group():
    rng = np.random.default_rng(9)
    x = np.column_stack([np.ones(25), rng.normal(size=(25, 4))])
    groups = ['intercept', 'emissions', 'geography', 'geography', 'population']
    y = rng.normal(size=25)
    shares = ic.group_allocation(x, groups, y)['shares']
    assert shares['socioeconomic'] == 0.0
    frame_groups = np.array(groups)
    tss = float(np.sum((y - y.mean()) ** 2))
    present = ['emissions', 'geography', 'population']
    r2 = {}
    for size in range(4):
        for coalition in itertools.combinations(present, size):
            keep = (frame_groups == 'intercept') | np.isin(frame_groups, coalition)
            beta = np.linalg.lstsq(x[:, keep], y, rcond=None)[0]
            r2[frozenset(coalition)] = 1 - np.sum((y - x[:, keep] @ beta) ** 2) / tss
    for g in present:
        others = [h for h in present if h != g]
        weighted = sum(math.factorial(s) * math.factorial(2 - s) / 6 * (r2[frozenset(sub) | {g}] - r2[frozenset(sub)])
                       for s in range(3) for sub in itertools.combinations(others, s))
        assert abs(shares[g] - weighted) <= 1e-12


def test_draw_rows_regenerate_the_pandas_construction(branch_a):
    result = branch_a[0]
    context = ic.load_context(result['inputs'])
    w = result['world']
    frame = pd.DataFrame({'spatial_block': w['spatial_block'], 'row': np.arange(N)})
    rng = np.random.default_rng(1)
    blocks = pd.unique(frame['spatial_block'])
    assert list(blocks) == list(BLOCKS)
    for rows in ic.draw_row_sets(context, 'block', 10):
        chosen = rng.choice(blocks, size=len(blocks), replace=True)
        expected = pd.concat([frame[frame['spatial_block'] == c] for c in chosen], ignore_index=True).row.to_numpy()
        assert np.array_equal(rows, expected)
    rng = np.random.default_rng(0)
    for rows in ic.draw_row_sets(context, 'country', 10):
        assert np.array_equal(rows, rng.integers(0, N, N))


def test_natural_cubic_columns_and_repeated_knots():
    knots = (10.0, 20.0, 40.0, 60.0)
    a = np.array([5.0, 30.0, 50.0, 70.0])
    d = lambda k, v: (max(v - knots[k], 0.0) ** 3 - max(v - knots[3], 0.0) ** 3) / (knots[3] - knots[k])  # noqa: E731
    assert np.allclose(ic.natural_cubic_columns(knots, a), [[d(0, v) - d(2, v), d(1, v) - d(2, v)] for v in a],
                       rtol=1e-15, atol=0)
    with pytest.raises(ic.DegenerateDraw):
        ic.draw_knots(np.array([1.0] * 10 + [5.0, 9.0]))
    with pytest.raises(ic.DegenerateDraw):
        ic.draw_knots(np.array([1.0, 2.0, 3.0]))
    assert ic.draw_knots(np.array([1.0, 2.0, 3.0, 4.0])) == (1.0, 2.0, 3.0, 4.0)


def test_average_ranks_and_pearson_match_scipy():
    rng = np.random.default_rng(12)
    a = np.round(rng.normal(size=60), 1)          # ties
    b = rng.normal(size=60)
    assert np.array_equal(ic.average_ranks(a), stats.rankdata(a))
    assert abs(ic.pearson(ic.average_ranks(a), ic.average_ranks(b)) - stats.spearmanr(a, b).statistic) <= 1e-12
    assert abs(ic.pearson(a, b) - np.corrcoef(a, b)[0, 1]) <= 1e-12


def test_sign_agreement_is_exact_except_at_numerically_zero_residuals():
    a = np.array([0.02, -0.01, 3e-18, 0.05])
    b = np.array([0.01, 0.02, -2e-17, -0.04])
    assert ic.sign_agreement('s', a, b, 1)['passes'] and ic.sign_agreement('s', a, b, 2)['passes']
    assert not ic.sign_agreement('s', a, b, 0)['passes'] and not ic.sign_agreement('s', a, b, 3)['passes']
    assert not ic.sign_agreement('s', a, b, True)['passes']
    assert ic.sign_agreement('s', np.array([0.1, -0.2]), np.array([0.3, -0.1]), 2)['recomputed'][
        'numerically_zero_positions'] == 0


def test_non_finite_product_outcome_requires_a_non_computable_arm(branch_a, tmp_path):
    result = branch_a[0]
    aligned = pd.read_csv(result['inputs'].aligned_era5, float_precision='round_trip')
    aligned.loc[aligned.iso3 == 'S05', 'trend_c_per_decade_era5_area'] = np.nan
    path = tmp_path / 'aligned.csv'
    path.write_text(csv_text(aligned))
    report = ic.run(result['package'], dataclasses.replace(result['inputs'], aligned_era5=path), result['expected'])
    failed = report['sections']['5_product_check']['failed_checks']
    assert {'aligned_era5/non_computable', 'aligned_era5/no_predictions'} <= set(failed)
    assert not any(f.startswith('legacy_era5') for f in failed)


def test_naming_rule():
    assert ic.naming_rule('M0*', []) == 'M0*'
    assert ic.naming_rule('M2', ['M3-SAR(M2)']) == 'M3-SAR(M2)'
    assert ic.naming_rule('M0*', ['M3-SEM(M0*)', 'M3-SAR(M0*)']) is None


def test_exact_comparison_does_not_confuse_booleans_and_integers():
    assert ic.exact('x', True, 1)['passes'] is False
    assert ic.exact('x', {'a': [False]}, {'a': [0]})['passes'] is False
    assert ic.exact('x', {'a': [None, 2]}, {'a': [None, 2]})['passes'] is True


FORBIDDEN = ('m4_evaluate', 'm3_evaluate', 'm3_spatial', 'm3_feasibility', 'm2_evaluate', 'm2_conditional',
             'm2_feasibility', 'm2_latitude_basis', 'm2_independent_check', 'cv', 'spatial', 'run_territory_correction',
             'decomposition', 'stability', 'v2_provenance')


def test_module_imports_only_numpy_pandas_stdlib_and_the_independent_math():
    source = Path(ic.__file__).read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or '')
            imported |= {f'{node.module}.{alias.name}' for alias in node.names}
    top = {name.split('.')[0] for name in imported}
    assert top <= {'__future__', 'argparse', 'copy', 'dataclasses', 'hashlib', 'itertools', 'json', 'math', 'sys',
                   'time', 'pathlib', 'typing', 'numpy', 'pandas', 'research'}
    local = {name for name in imported if name.startswith('research')}
    assert all(name.startswith('research.model_v2.m3_independent_math') or name == 'research.model_v2' for name in local)
    assert local & {'research.model_v2.m3_independent_math'}
    assert not any(part in FORBIDDEN or part.startswith(('m4_evaluate', 'm3_evaluate', 'm3_spatial', 'm2_'))
                   for name in imported for part in name.split('.'))
    assert 'importlib' not in source and '__import__' not in source and 'scipy' not in source


def test_no_forbidden_module_is_loaded_transitively():
    probe = ('import sys; import research.model_v2.m4_independent_check; '
             'print("\\n".join(sorted(m for m in sys.modules if m.startswith(("research", "src", "scipy")))))')
    root = Path(ic.__file__).resolve().parents[2]
    loaded = subprocess.run([sys.executable, '-c', probe], cwd=root, capture_output=True, text=True, check=True)
    modules = set(loaded.stdout.split())
    assert 'research.model_v2.m4_independent_check' in modules
    assert modules <= {'research', 'research.model_v2', 'research.model_v2.m3_independent_math',
                       'research.model_v2.m4_independent_check'}
