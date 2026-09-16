"""M2 independent check: synthetic miniature primary results only.

Every artifact below is generated in ``tmp_path`` by a small writer in this file that follows the saved M2
artifact formats and uses a different numerical route from the checker (SVD least squares, ``numpy.quantile``
knots, product cubes, subset-weight Shapley shares). No repository data file is read, no outcome is real and
no M2 evaluator code is imported.
"""
from __future__ import annotations

import ast
import hashlib
import io
import itertools
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.model_v2 import m2_independent_check as ic

N = 36
REGIONS = ('Region A', 'Region B', 'Region C')
CATS = ('climate_zone', 'hemisphere', 'spatial_block', 'income_group')
GROUP_OF = {'cum_co2_total': 'emissions', 'cum_co2_per_capita': 'emissions', 'abs_latitude': 'geography',
            'elevation': 'geography', 'continentality': 'geography', 'climate_zone': 'geography',
            'hemisphere': 'geography', 'spatial_block': 'geography', 'income_group': 'socioeconomic',
            'population': 'population', 'station_density': 'population'}
NUMERIC = {'primary_total_co2': ['cum_co2_total', 'abs_latitude', 'elevation', 'continentality', 'population',
                                 'station_density'],
           'per_capita': ['cum_co2_per_capita', 'abs_latitude', 'elevation', 'continentality', 'population',
                          'station_density']}
MODELS, PROTOCOLS = ('M0star', 'M2'), ('primary_loco', 'm49_subregion_lo', 'random10')
ADDED = ['abs_latitude_ns1', 'abs_latitude_ns2', 'hemisphere=S:abs_latitude']
GROUPS = ('emissions', 'geography', 'socioeconomic', 'population')
STAGES = {'M1a': 0.002342838562091845, 'M1b': 0.0017229850562197266}
DETERMINISTIC = ('m2_scorecard.json', 'm2_country_predictions.csv', 'm2_primary_country_comparison.csv',
                 'm2_cv_folds.csv', 'm2_fit_states.csv', 'm2_coefficients.csv', 'm2_coefficient_summary.json',
                 'm2_shapley_coalitions.csv', 'm2_design_matrices.csv', 'm2_provenance.json')


# ---------------------------------------------------------------------
# A miniature M2 result in the saved formats
# ---------------------------------------------------------------------


def world(seed=5, strong=False):
    rng = np.random.default_rng(seed)
    region = np.array([REGIONS[i % 3] for i in range(N)])
    block = rng.choice(['Africa', 'Asia', 'Europe'], N)
    block[np.flatnonzero(region == 'Region C')[:3]] = 'Oceania'   # unseen when Region C is held out
    hemisphere = np.where(np.arange(N) % 4 == 1, 'S', 'N')
    a = rng.uniform(1.0, 68.0, N)
    w = {'iso3': np.array([f'S{i:02d}' for i in range(N)]), 'Country': np.array([f'Country {i}' for i in range(N)]),
         'm49_subregion': region, 'climate_zone': rng.choice(list('ABC'), N), 'hemisphere': hemisphere,
         'spatial_block': block, 'income_group': rng.choice(['High-income countries', 'Low-income countries',
                                                             'Upper-middle-income countries'], N),
         'cum_co2_total': rng.uniform(0.5, 3.5, N), 'abs_latitude': a, 'elevation': rng.uniform(0, 2500, N),
         'continentality': rng.uniform(0, 900, N), 'population': rng.uniform(5.5, 9.2, N),
         'station_density': rng.uniform(0, 2.5, N)}
    w['cum_co2_per_capita'] = w['cum_co2_total'] - w['population'] + 6.0
    w['y'] = (0.12 + 0.004 * a + 6e-5 * (a - 30.0) ** 2 - 0.002 * a * (hemisphere == 'S')
              + 0.01 * w['cum_co2_total'] + rng.normal(0, 0.03, N))
    if strong:   # a large southern slope difference and curvature: M2 is promoted
        w['y'] = (0.12 + 0.002 * a + 4e-4 * np.clip(a - 35.0, 0, None) ** 2 - 0.004 * a * (hemisphere == 'S')
                  + rng.normal(0, 0.01, N))
    return w


def state_json(knots, n_train):
    return json.dumps({'version': 'm2-latitude-basis-v1', 'n_train': n_train, 'knots': list(knots),
                       'knots_hex': [k.hex() for k in knots], 'quantile_method': 'linear',
                       'probabilities': ['1/3', '2/3'], 'boundary_knots': 'training minimum and maximum',
                       'tails': 'linear beyond the boundary knots'}, sort_keys=True)


def design(w, representation, model, rows, levels, knots):
    names, blocks = ['intercept', *NUMERIC[representation]], [np.ones(len(rows))]
    blocks += [w[c][rows] for c in NUMERIC[representation]]
    for c in CATS:
        for level in levels[c][1:]:
            blocks.append((w[c][rows] == level).astype(float))
            names.append(f'{c}={level}')
    if model == 'M2':
        a, last = w['abs_latitude'][rows], knots[3]

        def d(k):
            t, u = np.maximum(a - knots[k], 0.0), np.maximum(a - last, 0.0)
            return (t * t * t - u * u * u) / (last - knots[k])
        blocks += [d(0) - d(2), d(1) - d(2), (w['hemisphere'][rows] == 'S').astype(float) * a]
        names += ADDED
    return np.column_stack(blocks), names


def fit(w, representation, model, train, test):
    levels = {c: sorted(set(w[c][train].tolist())) for c in CATS}
    a = w['abs_latitude'][train]
    knots = None
    if model == 'M2':
        lower, upper = np.quantile(a, [1 / 3, 2 / 3], method='linear')
        knots = (float(a.min()), float(lower), float(upper), float(a.max()))
    x, names = design(w, representation, model, train, levels, knots)
    beta = np.linalg.lstsq(x, w['y'][train], rcond=None)[0]
    rows = train if len(test) == 0 else test
    xr, _ = design(w, representation, model, rows, levels, knots)
    yhat = xr @ beta
    for c in CATS:
        effects = np.array([0.0] + [beta[names.index(f'{c}={lv}')] for lv in levels[c][1:]])
        yhat = yhat + ~np.isin(w[c][rows], levels[c]) * effects.mean()
    base = x[:, :len(names) - (3 if model == 'M2' else 0)]
    return {'levels': levels, 'knots': knots, 'names': names, 'beta': beta, 'rows': rows, 'yhat': yhat, 'x': x,
            'rank': int(np.linalg.matrix_rank(x)), 'base_columns': base.shape[1],
            'base_rank': int(np.linalg.matrix_rank(base))}


def folds():
    training = {i: [j for j in range(N) if min(abs(i - j), N - abs(i - j)) > 1] for i in range(N)}
    m49 = pd.factorize(pd.Series([REGIONS[i % 3] for i in range(N)]), sort=True)[0]
    random3 = np.random.default_rng(0).permutation(N) % 3
    return training, m49, random3


def csv_text(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator='\n')
    return buffer.getvalue()


def rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))


def metrics(y, yhat):
    e = y - yhat
    return {'cv_r2': float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), 'cv_rmse': rmse(e),
            'cv_mae': float(np.mean(np.abs(e)))}


def shares_by_subsets(x, groups, y):
    tss, r2, rows = float(np.sum((y - y.mean()) ** 2)), {}, []
    for size in range(5):
        for coalition in itertools.combinations(GROUPS, size):
            keep = (groups == 'intercept') | np.isin(groups, coalition)
            beta = np.linalg.lstsq(x[:, keep], y, rcond=None)[0]
            r2[frozenset(coalition)] = float(1 - np.sum((y - x[:, keep] @ beta) ** 2) / tss)
            rows.append({'coalition': '+'.join(coalition) or 'intercept_only', 'n_groups': size,
                         'r2': r2[frozenset(coalition)]})
    shares = {}
    for g in GROUPS:
        others = [h for h in GROUPS if h != g]
        shares[g] = sum(math.factorial(s) * math.factorial(3 - s) / 24 * (r2[frozenset(sub) | {g}] - r2[frozenset(sub)])
                        for s in range(4) for sub in itertools.combinations(others, s))
    return shares, r2[frozenset(GROUPS)], rows


def write_result(root: Path, seed=5, strong=False):
    w = world(seed, strong)
    training, m49, random3 = folds()
    ids = {'m49_subregion_lo': m49, 'random10': random3}
    everyone = np.arange(N)
    root.mkdir(parents=True, exist_ok=True)
    primary = root / 'm2_primary'
    primary.mkdir()
    countries = pd.DataFrame({'Country': w['Country'], 'warming_trend': w['y'], 'iso3': w['iso3'],
                              **{c: w[c] for c in CATS}, 'm49_subregion': w['m49_subregion']})
    (root / 'm0_countries.csv').write_text(csv_text(countries))
    (root / 'm1a_scorecard.json').write_text(json.dumps({'primary': {'M1a': {'paired_delta_vs_m0': {
        'delta_rmse': STAGES['M1a']}}}}))
    (root / 'm1b_scorecard.json').write_text(json.dumps({'verdict': {'delta_rmse': STAGES['M1b']}}))

    predictions, states, coefficients, designs, coalitions = [], [], [], [], []
    cards, oof, gammas, latitude = {}, {}, {}, {}
    for rep, model in itertools.product(NUMERIC, MODELS):
        fits = [('full_sample', 'all', everyone, np.array([], int))]
        fits += [('primary_loco', w['iso3'][i], np.array(training[i]), np.array([i])) for i in range(N)]
        fits += [(p, str(f), everyone[ids[p] != f], everyone[ids[p] == f]) for p in ids for f in sorted(set(ids[p]))]
        yhat = {p: np.full(N, np.nan) for p in ('in_sample', *PROTOCOLS)}
        for protocol, label, train, test in fits:
            result = fit(w, rep, model, train, test)
            assert result['rank'] == len(result['names']) and result['base_rank'] == result['base_columns']
            name = 'in_sample' if protocol == 'full_sample' else protocol
            yhat[name][result['rows']] = result['yhat']
            k = result['knots']
            held = w['abs_latitude'][test]
            states.append({'representation': rep, 'model': model, 'protocol': protocol, 'fit': label,
                           'held_out_iso3': '|'.join(w['iso3'][test]), 'n_train': len(train), 'n_test': len(test),
                           'levels': json.dumps(result['levels'], sort_keys=True), 'columns': '|'.join(result['names']),
                           'matrix_columns': len(result['names']), 'matrix_rank': result['rank'],
                           'baseline_columns': result['base_columns'], 'baseline_rank': result['base_rank'],
                           'n_train_sh': int((w['hemisphere'][train] == 'S').sum()),
                           'latitude_state': state_json(k, len(train)) if k else '',
                           'test_below_lower_boundary': int((held < k[0]).sum()) if k and len(test) else '',
                           'test_above_upper_boundary': int((held > k[3]).sum()) if k and len(test) else ''})
            coefficients += [{'representation': rep, 'model': model, 'protocol': protocol, 'fit': label,
                              'column': c, 'coefficient': float(b)} for c, b in zip(result['names'], result['beta'])]
            if model == 'M2':
                for c in ('abs_latitude_ns1', 'abs_latitude_ns2', ADDED[2], 'abs_latitude'):
                    latitude.setdefault((rep, c, protocol), {})[label] = float(result['beta'][result['names'].index(c)])
            if protocol == 'full_sample':
                full = result
            for i, value in zip(result['rows'], result['yhat']):
                a = w['abs_latitude'][i]
                flag = ('' if model == 'M0star' or name == 'in_sample' else
                        'below' if a < k[0] else 'above' if a > k[3] else '')
                predictions.append({'representation': rep, 'model': model, 'protocol': name, 'iso3': w['iso3'][i],
                                    'Country': w['Country'][i], 'm49_subregion': w['m49_subregion'][i],
                                    'observed': w['y'][i], 'prediction': value, 'error': w['y'][i] - value,
                                    'fold_id': (-1 if name == 'in_sample' else
                                                i if name == 'primary_loco' else int(label)),
                                    'n_train': len(train), 'nearest_train_km': np.nan if name == 'in_sample' else 100.0,
                                    'unseen_levels': sum(w[c][i] not in result['levels'][c] for c in CATS),
                                    'extrapolation': flag,
                                    '_order': (list(NUMERIC).index(rep), MODELS.index(model),
                                               ('in_sample', *PROTOCOLS).index(name), int(i))})
        groups = np.array(['intercept' if n == 'intercept' else 'geography' if n in ADDED else GROUP_OF[n.split('=')[0]]
                           for n in full['names']])
        for j, (column, group) in enumerate(zip(full['names'], groups)):
            designs += [{'representation': rep, 'model': model, 'iso3': w['iso3'][i], 'column': column, 'group': group,
                         'value': float(full['x'][i, j])} for i in range(N)]
        shares, total, rows = shares_by_subsets(full['x'], groups, w['y'])
        coalitions += [{'representation': rep, 'model': model, **r} for r in rows]
        y = w['y']
        frame = pd.DataFrame({'region': w['m49_subregion'], 'e': y - yhat['primary_loco']})
        regions = frame.groupby('region').e.agg(n='size', rmse=lambda v: float(np.sqrt(np.mean(v ** 2))))
        regions = regions.sort_values('rmse', ascending=False)
        card = {'n': N, 'in_sample_r2': metrics(y, yhat['in_sample'])['cv_r2'],
                'in_sample_rmse': rmse(y - yhat['in_sample']),
                **metrics(y, yhat['primary_loco']), 'secondary': metrics(y, yhat['m49_subregion_lo']),
                'random_reference_only': metrics(y, yhat['random10']),
                'worst_region': {'region': str(regions.index[0]), 'rmse': float(regions.iloc[0].rmse),
                                 'n': int(regions.iloc[0].n), 'min_region_n': 3},
                'region_rmse': {str(k): float(v) for k, v in regions.rmse.items()},
                **{f'share_{g}': float(v) for g, v in shares.items()}, 'residual_share': 1.0 - total,
                'matrix_columns': len(full['names']), 'matrix_rank': full['rank']}
        if model == 'M2':
            values = np.array([latitude[(rep, ADDED[2], 'primary_loco')][w['iso3'][i]] for i in range(N)])
            g = latitude[(rep, ADDED[2], 'full_sample')]['all']
            card['gamma'] = {'full': g, 'n_primary_training_fits': N, 'strictly_negative_n': int((values < 0).sum()),
                             'strictly_negative_fraction': float((values < 0).mean()),
                             'zero_n': int((values == 0).sum()),
                             'same_sign_as_full_fraction': float(np.mean(np.sign(values) == np.sign(g))),
                             'median': float(np.median(values)), 'min': float(values.min()), 'max': float(values.max()),
                             'cutoff': None}
            card['latitude_state_full_sample'] = json.loads(state_json(full['knots'], N))
        cards.setdefault(rep, {})[model] = card
        oof[(rep, model)] = yhat
        gammas[rep] = full['knots']

    y = w['y']
    representations, equivalence = {}, {'predictions': {}, 'coefficients': {}, 'scores': {}}
    for rep in NUMERIC:
        new, ref = oof[(rep, 'M2')]['primary_loco'], oof[(rep, 'M0star')]['primary_loco']
        index = np.random.default_rng(0).integers(0, N, size=(2000, N))
        delta_b = np.sqrt(np.mean((y - new)[index] ** 2, axis=1)) - np.sqrt(np.mean((y - ref)[index] ** 2, axis=1))
        paired = {'delta_rmse': float(rmse(y - new) - rmse(y - ref)),
                  'country_bootstrap_95_interval': np.quantile(delta_b, [0.025, 0.975]).tolist(),
                  'resamples': 2000, 'seed': 0}
        d, (lo, hi) = paired['delta_rmse'], paired['country_bootstrap_95_interval']
        m49_change = cards[rep]['M2']['secondary']['cv_rmse'] - cards[rep]['M0star']['secondary']['cv_rmse']
        worst_change = cards[rep]['M2']['worst_region']['rmse'] - cards[rep]['M0star']['worst_region']['rmse']
        p1, p2, r1, r2, r3 = d < 0, hi < 0, d <= -0.002, m49_change <= 0.001, worst_change <= 0.001
        label = ('supported and promoted' if p1 and p2 and r1 and r2 and r3 else
                 'predictive support, not promoted' if p1 and p2 else 'not supported')
        verdict = {'verdict': label, 'predictive_support': bool(p1 and p2),
                   'promoted': label == 'supported and promoted',
                   'P1_delta_below_zero': bool(p1), 'P2_interval_upper_below_zero': bool(p2),
                   'R1_delta_le_minus_0.002': bool(r1), 'R2_m49_veto_passed': bool(r2),
                   'R3_worst_region_veto_passed': bool(r3), 'R4_structural_integrity': True, 'delta_rmse': d,
                   'interval': [lo, hi], 'm49_rmse_change': m49_change, 'worst_region_rmse_change': worst_change,
                   'worsened_generalization': bool(d > 0 and lo > 0)}
        entry = {'decides_verdict': rep == 'primary_total_co2', 'M0star': cards[rep]['M0star'], 'M2': cards[rep]['M2'],
                 'paired_primary_comparison': paired}
        if rep == 'primary_total_co2':
            entry['verdict'] = verdict
        else:
            conditions = dict(verdict)
            entry['representation_conditions'] = {**conditions, 'label_descriptive_only': conditions.pop('verdict')}
            entry['representation_conditions'].pop('verdict', None)
        representations[rep] = entry
    a, b = representations['primary_total_co2'], representations['per_capita']
    for model in MODELS:
        equivalence['predictions'][model] = {p: float(np.max(np.abs(oof[('primary_total_co2', model)][p]
                                                                    - oof[('per_capita', model)][p])))
                                             for p in ('in_sample', *PROTOCOLS)}
    for column in ('abs_latitude_ns1', 'abs_latitude_ns2', ADDED[2], 'abs_latitude'):
        full_gap = abs(latitude[('primary_total_co2', column, 'full_sample')]['all']
                       - latitude[('per_capita', column, 'full_sample')]['all'])
        folds_gap = [abs(latitude[('primary_total_co2', column, 'primary_loco')][k]
                         - latitude[('per_capita', column, 'primary_loco')][k]) for k in w['iso3']]
        equivalence['coefficients'][column] = {'full': float(full_gap), 'primary_training_max': float(max(folds_gap)),
                                               'training_fits_compared': N}
    equivalence['scores']['paired'] = {
        'delta_rmse': abs(a['paired_primary_comparison']['delta_rmse'] - b['paired_primary_comparison']['delta_rmse']),
        'interval': [abs(u - v) for u, v in zip(a['paired_primary_comparison']['country_bootstrap_95_interval'],
                                                b['paired_primary_comparison']['country_bootstrap_95_interval'])]}
    equivalence['passes'] = True
    deltas = {**STAGES, 'M2': a['paired_primary_comparison']['delta_rmse']}
    verdict = a['verdict']
    scorecard = {'primary_representation': 'primary_total_co2', 'representations': representations,
                 'static_stopping_indicator': {'point_delta_rmse_vs_m0star': deltas, 'best': min(deltas.values()),
                                               'threshold': -0.002, 'fires': bool(min(deltas.values()) > -0.002),
                                               'm2_decides': bool(STAGES['M1a'] > 0 and STAGES['M1b'] > 0)},
                 'representation_equivalence': equivalence, 'integrity_failures': [], 'integrity_passed': True,
                 'verdict': verdict,
                 'retained_static_specification': 'M2' if verdict['verdict'] == 'supported and promoted' else 'M0*'}

    training_iso = ['|'.join(w['iso3'][training[i]]) for i in range(N)]
    frames = {
        'm2_country_predictions.csv': pd.DataFrame(sorted(predictions, key=lambda r: r['_order']))
        .drop(columns='_order'),
        'm2_fit_states.csv': pd.DataFrame(states), 'm2_coefficients.csv': pd.DataFrame(coefficients),
        'm2_design_matrices.csv': pd.DataFrame(designs), 'm2_shapley_coalitions.csv': pd.DataFrame(coalitions),
        'm2_cv_folds.csv': pd.DataFrame({'iso3': w['iso3'], 'Country': w['Country'], 'primary_fold': everyone,
                                         'primary_n_train': [len(training[i]) for i in range(N)],
                                         'primary_training_iso3': training_iso, 'm49_subregion': w['m49_subregion'],
                                         'm49_fold': m49, 'random10_fold': random3}),
        'm2_primary_country_comparison.csv': pd.DataFrame({'iso3': w['iso3']}),
    }
    for name, frame in frames.items():
        (primary / name).write_text(csv_text(frame))
    bound = (('m0_countries.csv', 'm0_countries.csv'), ('m1a_scorecard.json', 'm1a_scorecard.json'),
             ('m1b_primary/m1b_scorecard.json', 'm1b_scorecard.json'))
    provenance = {'inputs': {f'research/model_v2/outputs/{p}': {'sha256': sha(root / f), 'frozen_at': 'synthetic'}
                             for p, f in bound}}
    for name, payload in (('m2_scorecard.json', scorecard), ('m2_provenance.json', provenance),
                          ('m2_coefficient_summary.json', {})):
        (primary / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    write_manifest(primary)
    full_columns = {m: cards['primary_total_co2'][m]['matrix_columns'] for m in MODELS}
    return {'root': root, 'primary': primary, 'countries': root / 'm0_countries.csv',
            'm1a': root / 'm1a_scorecard.json', 'm1b': root / 'm1b_scorecard.json',
            'expected': {'n': N, 'full_columns': full_columns}, 'scorecard': scorecard}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(primary):
    scorecard = json.loads((primary / 'm2_scorecard.json').read_text())
    verdict = scorecard['verdict']
    manifest = {'result': 'synthetic M2 primary', 'verdict': verdict['verdict'],
                'predictive_support': verdict['predictive_support'],
                'retained_static_specification': scorecard['retained_static_specification'],
                'static_stopping_indicator_fires': scorecard['static_stopping_indicator']['fires'],
                'integrity_passed': True, 'artifact_sha256': {n: sha(primary / n) for n in DETERMINISTIC}}
    (primary / 'm2_result_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def check(result, primary=None):
    return ic.run(primary or result['primary'], result['countries'], result['m1a'], result['m1b'], result['expected'])


@pytest.fixture(scope='module')
def result(tmp_path_factory):
    return write_result(tmp_path_factory.mktemp('m2_synthetic'))


@pytest.fixture(scope='module')
def report(result):
    return check(result)


@pytest.fixture(scope='module')
def promoted(tmp_path_factory):
    result = write_result(tmp_path_factory.mktemp('m2_synthetic_promoted'), strong=True)
    return result, check(result)


def copy_primary(result, tmp_path):
    target = tmp_path / 'm2_primary'
    shutil.copytree(result['primary'], target)
    return target


# ---------------------------------------------------------------------
# The reconstruction on consistent artifacts
# ---------------------------------------------------------------------


def test_consistent_synthetic_result_passes_every_section(report):
    assert report['failed_checks'] == []
    assert report['all_checks_pass'] is True
    assert all(block['passes'] for block in report['section_summary'].values())
    assert report['fits_reconstructed'] == 2 * 2 * (1 + N + 3 + 3)
    assert report['negative_controls_all_detected'] is True


@pytest.mark.parametrize('name', ['a_perturbed_prediction', 'b_shifted_promotion_threshold', 'c_altered_knot',
                                  'd_swapped_training_membership', 'e_altered_coefficient'])
def test_each_negative_control_is_detected_from_a_clean_baseline(report, name):
    control = report['negative_controls'][name]
    assert control['detected'] is True
    if 'baseline_failed_checks' in control:
        assert control['baseline_failed_checks'] == []


def test_promoted_regime_passes_and_the_shifted_threshold_changes_the_label(promoted):
    result, report = promoted
    assert result['scorecard']['verdict']['verdict'] == 'supported and promoted'
    assert result['scorecard']['static_stopping_indicator']['fires'] is False
    assert report['all_checks_pass'] is True and report['negative_controls_all_detected'] is True
    variants = report['negative_controls']['b_shifted_promotion_threshold']['variants']
    assert all(v['label_changed'] and v['label_under_shift'] == 'predictive support, not promoted'
               for v in variants.values())


def test_controls_fail_the_specific_comparisons(report):
    controls = report['negative_controls']
    assert 'predictions' in controls['a_perturbed_prediction']['failed_checks']
    assert all(v['R1_comparison_failed'] for v in controls['b_shifted_promotion_threshold']['variants'].values())
    assert controls['c_altered_knot']['channels_failed']['full_design_added_columns']
    assert controls['c_altered_knot']['channels_failed']['knots_from_training_latitudes']
    assert 'coefficients' in controls['d_swapped_training_membership']['failed_checks']
    assert controls['e_altered_coefficient']['failed_checks'] == ['coefficients']


def test_fixture_exercises_unseen_levels_extrapolation_and_buffer(result):
    predictions = pd.read_csv(result['primary'] / 'm2_country_predictions.csv', keep_default_na=False)
    assert (predictions.unseen_levels[predictions.protocol == 'm49_subregion_lo'] > 0).any()
    assert set(predictions.extrapolation[(predictions.model == 'M2') & (predictions.protocol != 'in_sample')]) >= {
        '', 'below', 'above'}
    folds = pd.read_csv(result['primary'] / 'm2_cv_folds.csv')
    assert (folds.primary_n_train < N - 1).all()


def test_json_report_is_written_and_exit_zero(result, tmp_path):
    out = tmp_path / 'verification' / 'm2_independent_check.json'
    code = ic.main(['--primary', str(result['primary']), '--out', str(out), '--countries', str(result['countries']),
                    '--m1a-scorecard', str(result['m1a']), '--m1b-scorecard', str(result['m1b'])],
                   expected=result['expected'])
    written = json.loads(out.read_text())
    assert code == 0
    assert written['all_checks_pass'] is True and written['negative_controls_all_detected'] is True
    assert set(written['negative_controls']) == {'a_perturbed_prediction', 'b_shifted_promotion_threshold',
                                                 'c_altered_knot', 'd_swapped_training_membership',
                                                 'e_altered_coefficient'}
    assert {'independent', 'not_independent'} <= set(written['scope'])


def test_real_structure_expectation_fails_on_the_synthetic_sizes(result, tmp_path):
    code = ic.main(['--primary', str(result['primary']), '--out', str(tmp_path / 'out.json'),
                    '--countries', str(result['countries']), '--m1a-scorecard', str(result['m1a']),
                    '--m1b-scorecard', str(result['m1b'])])
    assert code == 3


# ---------------------------------------------------------------------
# Tampering
# ---------------------------------------------------------------------


def test_manifest_digest_tampering_is_refused(result, tmp_path):
    primary = copy_primary(result, tmp_path)
    path = primary / 'm2_coefficients.csv'
    path.write_text(path.read_text().replace('primary_total_co2', 'primary_total_co2 ', 1))
    with pytest.raises(ic.DigestRefused):
        check(result, primary)
    out = tmp_path / 'refused.json'
    code = ic.main(['--primary', str(primary), '--out', str(out), '--countries', str(result['countries']),
                    '--m1a-scorecard', str(result['m1a']), '--m1b-scorecard', str(result['m1b'])],
                   expected=result['expected'])
    assert code == 3
    assert 'refused' in json.loads(out.read_text())


def test_unlisted_artifact_is_refused(result, tmp_path):
    primary = copy_primary(result, tmp_path)
    manifest = json.loads((primary / 'm2_result_manifest.json').read_text())
    del manifest['artifact_sha256']['m2_fit_states.csv']
    (primary / 'm2_result_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ic.DigestRefused):
        check(result, primary)


def test_consistently_rehashed_wrong_prediction_is_caught(result, tmp_path):
    primary = copy_primary(result, tmp_path)
    frame = pd.read_csv(primary / 'm2_country_predictions.csv', float_precision='round_trip', keep_default_na=False)
    row = frame.index[(frame.model == 'M2') & (frame.protocol == 'm49_subregion_lo')][3]
    frame.loc[row, 'prediction'] += 1e-6
    frame.loc[row, 'error'] = frame.loc[row, 'observed'] - frame.loc[row, 'prediction']
    (primary / 'm2_country_predictions.csv').write_text(csv_text(frame))
    write_manifest(primary)
    failed = check(result, primary)
    assert failed['all_checks_pass'] is False
    assert not failed['section_summary']['5_predictions']['passes']
    assert not failed['section_summary']['6_metrics']['passes']


def test_consistently_rehashed_wrong_label_is_caught(result, tmp_path):
    primary = copy_primary(result, tmp_path)
    scorecard = json.loads((primary / 'm2_scorecard.json').read_text())
    labels = ['supported and promoted', 'predictive support, not promoted', 'not supported']
    wrong = next(label for label in labels if label != scorecard['verdict']['verdict'])
    scorecard['verdict']['verdict'] = scorecard['representations']['primary_total_co2']['verdict']['verdict'] = wrong
    (primary / 'm2_scorecard.json').write_text(json.dumps(scorecard, indent=2) + '\n')
    write_manifest(primary)
    failed = check(result, primary)
    assert 'primary_total_co2/verdict/label' in failed['sections']['8_verdict_and_stopping']['failed_checks']


# ---------------------------------------------------------------------
# The independent mathematics
# ---------------------------------------------------------------------


def test_verdict_label_boundaries():
    promoted = ic.verdict_conditions(-0.002, -0.01, -1e-9, 0.001, 0.001, True)
    assert promoted['label'] == 'supported and promoted' and promoted['R1'] and promoted['R2'] and promoted['R3']
    assert ic.verdict_conditions(-0.002, -0.01, 0.0, 0.0, 0.0, True)['label'] == 'not supported'
    assert ic.verdict_conditions(-0.0019, -0.01, -1e-4, 0.0, 0.0, True)['label'] == 'predictive support, not promoted'
    assert ic.verdict_conditions(-0.003, -0.01, -1e-4, 0.0011, 0.0, True)['label'] == 'predictive support, not promoted'
    assert ic.verdict_conditions(0.001, 1e-5, 0.002, 0.0, 0.0, True)['worsened_generalization'] is True
    assert ic.verdict_conditions(0.001, 0.0, 0.002, 0.0, 0.0, True)['worsened_generalization'] is False
    assert ic.verdict_conditions(-0.003, -0.01, -1e-4, 0.0, 0.0, False)['label'] is None
    assert ic.verdict_conditions(float('nan'), -0.01, -1e-4, 0.0, 0.0, True)['label'] is None
    shifted = ic.verdict_conditions(-0.0025, -0.01, -1e-4, 0.0, 0.0, True, practical=-0.003)
    assert shifted['label'] == 'predictive support, not promoted' and shifted['R1'] is False


def test_natural_cubic_columns_hand_values_and_linear_tails():
    knots = (10.0, 20.0, 40.0, 60.0)
    a = np.array([5.0, 30.0, 50.0])
    d = lambda k, x: (max(x - knots[k], 0.0) ** 3 - max(x - knots[3], 0.0) ** 3) / (knots[3] - knots[k])  # noqa: E731
    expected = [[d(0, x) - d(2, x), d(1, x) - d(2, x)] for x in a]
    assert np.allclose(ic.natural_cubic_columns(knots, a), expected, rtol=1e-15, atol=0)
    tail = ic.natural_cubic_columns(knots, np.array([61.0, 63.0, 65.0]))
    assert np.allclose(np.diff(tail, 2, axis=0), 0.0, atol=1e-9)
    assert np.all(ic.natural_cubic_columns(knots, np.array([0.0, 9.9])) == 0.0)


def test_linear_quantile_and_knots_match_numpy():
    rng = np.random.default_rng(3)
    for n in (4, 5, 17, 124, 151):
        values = rng.uniform(0, 70, n)
        lower, upper = np.quantile(values, [1 / 3, 2 / 3], method='linear')
        knots = ic.knots_from_training(values)
        assert max(abs(knots[1] - lower), abs(knots[2] - upper)) <= 1e-12
        assert knots[0] == values.min() and knots[3] == values.max()


def test_qr_coefficients_match_svd_least_squares():
    rng = np.random.default_rng(4)
    x = np.column_stack([np.ones(40), rng.normal(size=(40, 6)) * [1, 100, 1e-2, 5, 1, 30]])
    y = rng.normal(size=40)
    assert np.allclose(ic.qr_coefficients(x, y), np.linalg.lstsq(x, y, rcond=None)[0], rtol=1e-10, atol=1e-12)


def test_permutation_shapley_equals_subset_weights():
    rng = np.random.default_rng(6)
    r2 = {frozenset(c): (0.0 if not c else float(rng.uniform(0, 1)))
          for s in range(5) for c in itertools.combinations(GROUPS, s)}
    by_permutation = ic.shapley_by_permutations(r2, GROUPS)
    for g in GROUPS:
        others = [h for h in GROUPS if h != g]
        weighted = sum(math.factorial(s) * math.factorial(3 - s) / 24 * (r2[frozenset(sub) | {g}] - r2[frozenset(sub)])
                       for s in range(4) for sub in itertools.combinations(others, s))
        assert abs(by_permutation[g] - weighted) <= 1e-15
    assert abs(sum(by_permutation.values()) - r2[frozenset(GROUPS)]) <= 1e-15


def test_paired_interval_is_the_contract_resampling():
    rng = np.random.default_rng(8)
    y, new, old = rng.normal(size=25), rng.normal(size=25), rng.normal(size=25)
    index = np.random.default_rng(0).integers(0, 25, size=(2000, 25))
    statistic = np.sqrt(np.mean((y - new)[index] ** 2, axis=1)) - np.sqrt(np.mean((y - old)[index] ** 2, axis=1))
    interval = ic.paired_interval(y, new, old)
    assert [interval['low'], interval['high']] == np.quantile(statistic, [0.025, 0.975]).tolist()


FORBIDDEN = ('m2_evaluate', 'm2_conditional', 'm2_feasibility', 'm2_latitude_basis', 'cv', 'spatial',
             'run_territory_correction', 'decomposition', 'stability', 'm3_')


def test_module_imports_only_numpy_pandas_and_the_standard_library():
    tree = ast.parse(Path(ic.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or '')
            imported |= {f'{node.module}.{alias.name}' for alias in node.names}
    top = {name.split('.')[0] for name in imported}
    assert top <= {'__future__', 'argparse', 'dataclasses', 'hashlib', 'itertools', 'json', 'math', 'sys', 'pathlib',
                   'typing', 'numpy', 'pandas'}
    assert not any(part.startswith(FORBIDDEN) for name in imported for part in name.split('.'))
    assert 'importlib' not in Path(ic.__file__).read_text() and '__import__' not in Path(ic.__file__).read_text()
