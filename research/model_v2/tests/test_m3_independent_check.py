"""M3 independent check: synthetic miniature packages only.

Every artifact below is written in ``tmp_path`` by a small builder in this file that follows the saved M3 and M2
artifact formats and takes a different numerical route from the checker (``slogdet`` and SVD least squares for
the likelihood, dense matrix products for lags and predictions, ``lexsort`` kNN, power cubes and
``numpy.quantile`` knots, subset-weight Shapley shares, ``polyfit`` calibration). The frozen search itself is
borrowed from ``m3_independent_math.maximize`` with that separate objective. No repository data file is read, no
outcome is real, and no evaluator module is imported.
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

from research.model_v2 import m3_independent_check as ic
from research.model_v2 import m3_independent_math as mi

N = 36
CLUSTERS = (('Region A', 10.0, 45.0), ('Region B', 30.0, 0.0), ('Region C', 100.0, 30.0), ('Region D', -60.0, -20.0))
BUFFER_KM = 300.0
R_EARTH = 6371.0088
CATS = ('climate_zone', 'hemisphere', 'spatial_block', 'income_group')
NUMERIC = {'primary_total_co2': ['cum_co2_total', 'abs_latitude', 'elevation', 'continentality', 'population',
                                 'station_density'],
           'per_capita': ['cum_co2_per_capita', 'abs_latitude', 'elevation', 'continentality', 'population',
                          'station_density']}
REPS = ('primary_total_co2', 'per_capita')
FAMS = ('sem', 'sar')
PROTOCOLS = ('primary_loco', 'm49_subregion_lo', 'random10')
CARD_PATH = {'primary_loco': (), 'm49_subregion_lo': ('secondary',), 'random10': ('random_reference_only',)}
ADDED = ['abs_latitude_ns1', 'abs_latitude_ns2', 'hemisphere=S:abs_latitude']
GROUPS = ('emissions', 'geography', 'socioeconomic', 'population')
GROUP_OF = {'cum_co2_total': 'emissions', 'cum_co2_per_capita': 'emissions', 'abs_latitude': 'geography',
            'elevation': 'geography', 'continentality': 'geography', 'climate_zone': 'geography',
            'hemisphere': 'geography', 'spatial_block': 'geography', 'income_group': 'socioeconomic',
            'population': 'population', 'station_density': 'population'}
EXPECTED = {'n': N, 'folds': {'primary_loco': N, 'm49_subregion_lo': 4, 'random10': 4}}
M3_ARTIFACTS = ('m3_scorecard.json', 'm3_country_predictions.csv', 'm3_fits.csv', 'm3_graphs.csv',
                'm3_full_sample_grid.csv', 'm3_provenance.json')
PINS = {'countries': 'research/model_v2/outputs/m0_countries.csv',
        'design': 'research/model_v2/outputs/m1b_primary/m1b_design_matrices.csv',
        'geometry': 'research/model_v2/outputs/country_geometry.csv',
        'rank_audit': 'research/model_v2/outputs/rank_audit_country_predictions.csv'}


# ---------------------------------------------------------------------
# A synthetic world
# ---------------------------------------------------------------------


def world(seed=3):
    rng = np.random.default_rng(seed)
    cluster = np.arange(N) % 4
    lon = np.array([CLUSTERS[c][1] for c in cluster]) + rng.uniform(-8, 8, N)
    lat = np.array([CLUSTERS[c][2] for c in cluster]) + rng.uniform(-8, 8, N)
    lon[4], lat[4] = lon[0], lat[0]          # identical station centroids: exact distance ties everywhere
    block = rng.choice(['Africa', 'Asia', 'Europe'], N).astype(object)
    block[[3, 7]] = 'Oceania'                # only in Region D: unseen when Region D is held out
    block[11] = 'North America'              # a single country: unseen in its own primary fit
    w = {'iso3': [f'S{i:02d}' for i in range(N)], 'Country': [f'Country {i}' for i in range(N)],
         'm49_subregion': np.array([CLUSTERS[c][0] for c in cluster]),
         'station_lon': lon, 'station_lat': lat,
         'centroid_lon': lon + rng.uniform(-1.5, 1.5, N), 'centroid_lat': lat + rng.uniform(-1.5, 1.5, N),
         'climate_zone': rng.choice(['A', 'B', 'C'], N), 'hemisphere': np.where(lat < 0, 'S', 'N'),
         'spatial_block': block.astype(str),
         'income_group': rng.choice(['High-income countries', 'Low-income countries', 'Upper-middle-income countries'], N),
         'cum_co2_total': rng.uniform(0.5, 3.5, N), 'abs_latitude': np.abs(lat), 'elevation': rng.uniform(0, 2500, N),
         'continentality': rng.uniform(0, 900, N), 'population': rng.uniform(5.5, 9.2, N),
         'station_density': rng.uniform(0, 2.5, N)}
    w['cum_co2_per_capita'] = w['cum_co2_total'] - w['population'] + 6.0
    d = haversine(lon, lat)
    wf = knn_dense(d, np.arange(N))
    u = np.linalg.solve(np.eye(N) - 0.6 * wf, rng.normal(0, 0.03, N))
    w['y'] = 0.1 + 0.004 * w['abs_latitude'] + 0.01 * w['cum_co2_total'] + u
    return w


def haversine(lon, lat):
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    dphi, dlam = phi[:, None] - phi[None, :], lam[:, None] - lam[None, :]
    a = np.sin(dphi / 2) ** 2 + np.cos(phi[:, None]) * np.cos(phi[None, :]) * np.sin(dlam / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def first_eight(row, candidates, exclude=None):
    candidates = np.asarray([c for c in candidates if c != exclude])
    return candidates[np.lexsort((candidates, row[candidates]))][:8]


def knn_lists(d, nodes):
    return [first_eight(d[i], nodes, exclude=i) for i in nodes]


def knn_dense(d, nodes):
    nodes = np.asarray(nodes)
    local = {int(v): k for k, v in enumerate(nodes)}
    w = np.zeros((len(nodes), len(nodes)))
    for k, nb in enumerate(knn_lists(d, nodes)):
        w[k, [local[int(j)] for j in nb]] = 1.0 / 8
    return w


def fold_plan(w):
    station = haversine(w['station_lon'], w['station_lat'])
    ids = {'primary_loco': np.arange(N), 'm49_subregion_lo': pd.factorize(pd.Series(w['m49_subregion']), sort=True)[0],
           'random10': np.random.default_rng(0).permutation(N) % 4}
    plan = {}
    for protocol, fold_ids in ids.items():
        fits = []
        for fold in sorted(set(fold_ids.tolist())):
            test = np.flatnonzero(fold_ids == fold)
            if protocol == 'primary_loco':
                train = np.array([j for j in range(N) if j != fold and station[fold, j] > BUFFER_KM])
            else:
                train = np.flatnonzero(fold_ids != fold)
            label = w['iso3'][fold] if protocol == 'primary_loco' else str(fold)
            fits.append((int(fold), label, train, test))
        plan[protocol] = (fold_ids, fits)
    return plan


def state_json(knots, n_train):
    return json.dumps({'version': 'm2-latitude-basis-v1', 'n_train': n_train, 'knots': list(knots),
                       'knots_hex': [k.hex() for k in knots], 'quantile_method': 'linear',
                       'probabilities': ['1/3', '2/3'], 'boundary_knots': 'training minimum and maximum',
                       'tails': 'linear beyond the boundary knots'}, sort_keys=True)


def knots_of(a):
    lower, upper = np.quantile(a, [1 / 3, 2 / 3], method='linear')
    return (float(a.min()), float(lower), float(upper), float(a.max()))


def design(w, rep, model, rows, levels, knots):
    columns = ['intercept', *NUMERIC[rep]]
    blocks = [np.ones(len(rows))] + [np.asarray(w[c], float)[rows] for c in NUMERIC[rep]]
    for c in CATS:
        for level in levels[c][1:]:
            columns.append(f'{c}={level}')
            blocks.append((np.asarray(w[c])[rows] == level).astype(float))
    if model == 'M2':
        a = w['abs_latitude'][rows]

        def d(k):
            return (np.maximum(a - knots[k], 0) ** 3 - np.maximum(a - knots[3], 0) ** 3) / (knots[3] - knots[k])
        blocks += [d(0) - d(2), d(1) - d(2), (np.asarray(w['hemisphere'])[rows] == 'S') * a]
        columns += ADDED
    return np.column_stack(blocks), columns


def levels_of(w, rows):
    return {c: sorted(set(np.asarray(w[c])[rows].tolist())) for c in CATS}


def adjustment(w, beta, columns, levels, rows):
    out = np.zeros(len(rows))
    for c in CATS:
        effects = [0.0] + [beta[columns.index(f'{c}={lv}')] for lv in levels[c][1:]]
        unseen = ~np.isin(np.asarray(w[c])[rows], levels[c])
        out = out + unseen * np.mean(effects)
    return out


def unseen_count(w, levels, rows):
    return np.array([sum(np.asarray(w[c])[i] not in levels[c] for c in CATS) for i in rows])


# ---------------------------------------------------------------------
# Static (M2-result) and spatial (M3-result) fits by a separate route
# ---------------------------------------------------------------------


def ols_predict(w, rep, model, train, rows):
    levels = levels_of(w, train)
    knots = knots_of(w['abs_latitude'][train]) if model == 'M2' else None
    x, columns = design(w, rep, model, train, levels, knots)
    beta = np.linalg.lstsq(x, w['y'][train], rcond=None)[0]
    xo, _ = design(w, rep, model, rows, levels, knots)
    return xo @ beta + adjustment(w, beta, columns, levels, rows), x, columns, beta


def concentrated(family, y, x, w, theta):
    n = len(y)
    m = np.eye(n) - theta * w
    xt = m @ x if family == 'sem' else x
    beta = np.linalg.lstsq(xt, m @ y, rcond=None)[0]
    e = m @ y - xt @ beta
    s2 = float(e @ e / n)
    return -(n / 2) * (math.log(2 * math.pi) + 1) - (n / 2) * math.log(s2) + np.linalg.slogdet(m)[1], beta, s2


def local_maxima(values):
    v = list(values)
    return sum(1 for i in range(len(v)) if (i == 0 or v[i] > v[i - 1]) and (i == len(v) - 1 or v[i] > v[i + 1]))


def spatial_fit(family, y, x, w):
    search = mi.maximize(lambda t: concentrated(family, y, x, w, t)[0])
    loglik, beta, s2 = concentrated(family, y, x, w, search.theta)
    return {'theta': search.theta, 'loglik': loglik, 'beta': beta, 'sigma2': s2, 'grid': list(search.grid),
            'maxima': local_maxima(search.grid), 'evaluations': len(search.evaluations),
            'at_bound': bool(abs(search.theta) > 0.99 - 1e-6)}


def shapley_subsets(r2):
    shares = {}
    for g in GROUPS:
        others = [h for h in GROUPS if h != g]
        total = 0.0
        for size in range(len(GROUPS)):
            weight = math.factorial(size) * math.factorial(len(GROUPS) - size - 1) / math.factorial(len(GROUPS))
            for subset in itertools.combinations(others, size):
                total += weight * (r2[frozenset(subset) | {g}] - r2[frozenset(subset)])
        shares[g] = total
    return shares


def coalition_r2(target, intercept, blocks):
    r2 = {}
    tss = float(np.sum((target - target.mean()) ** 2))
    for size in range(len(GROUPS) + 1):
        for coalition in itertools.combinations(GROUPS, size):
            z = np.column_stack([intercept] + [blocks[g] for g in coalition])
            e = target - z @ np.linalg.lstsq(z, target, rcond=None)[0]
            r2[frozenset(coalition)] = float(1 - e @ e / tss)
    return r2


def group_blocks(x, columns):
    owner = [GROUP_OF.get(c.split('=')[0], 'geography') if c != 'intercept' else 'intercept' for c in columns]
    owner = ['geography' if c in ADDED else o for c, o in zip(columns, owner)]
    return {g: x[:, [j for j, o in enumerate(owner) if o == g]] for g in GROUPS}


def moran(v, w):
    z = v - v.mean()
    return float(len(z) / w.sum() * (z @ w @ z) / (z @ z))


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def card(y, pred, fold, n_train, unseen, regions, one_step, sw, lw):
    e = y - pred
    folds = np.unique(fold)
    fold_rmse = np.array([rmse(e[fold == f]) for f in folds])
    slope, intercept = np.polyfit(pred, y, 1)
    table = (pd.DataFrame({'region': regions, 'e': e}).groupby('region')['e']
             .agg(n='size', rmse=lambda v: rmse(v.to_numpy()), mae=lambda v: float(np.mean(np.abs(v))), bias='mean')
             .sort_values('rmse', ascending=False))
    eligible = table[table.n >= 3]
    tss = float(np.sum((y - y.mean()) ** 2))
    return {'n': len(y), 'cv_r2': float(1 - np.sum(e ** 2) / tss), 'cv_rmse': rmse(e), 'cv_mae': float(np.mean(np.abs(e))),
            'in_sample_one_step_r2': float(1 - np.sum((y - one_step) ** 2) / tss),
            'in_sample_one_step_rmse': rmse(y - one_step), 'n_folds': len(folds),
            'fold_rmse_median': float(np.median(fold_rmse)), 'fold_rmse_min': float(fold_rmse.min()),
            'fold_rmse_max': float(fold_rmse.max()), 'worst_fold': int(folds[np.argmax(fold_rmse)]),
            'fold_error_iqr': float(np.subtract(*np.quantile(np.abs(e), [0.75, 0.25]))),
            'calibration_slope': float(slope), 'calibration_intercept': float(intercept),
            'mean_n_train': float(np.mean(n_train)), 'rows_with_unseen_level': int((unseen > 0).sum()),
            'residual_morans_i_in_sample': moran(y - one_step, sw), 'residual_morans_i_cv': moran(e, sw),
            'recorded_land_centroid_moran_cv': moran(e, lw),
            'worst_region': {'region': str(eligible.index[0]), 'rmse': float(eligible.iloc[0].rmse),
                             'n': int(eligible.iloc[0].n), 'min_region_n': 3},
            'region_rmse': {str(k): float(v) for k, v in table.rmse.items()},
            'region_bias': {str(k): float(v) for k, v in table.bias.items()},
            'region_mae': {str(k): float(v) for k, v in table.mae.items()}}


def paired(y, new, reference):
    idx = np.random.default_rng(0).integers(0, len(y), size=(2000, len(y)))
    delta = np.sqrt(np.mean((y - new)[idx] ** 2, axis=1)) - np.sqrt(np.mean((y - reference)[idx] ** 2, axis=1))
    return {'delta_rmse': float(np.sqrt(np.mean((y - new) ** 2)) - np.sqrt(np.mean((y - reference) ** 2))),
            'country_bootstrap_95_interval': np.quantile(delta, [0.025, 0.975]).tolist(), 'resamples': 2000, 'seed': 0}


# ---------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------


def write_csv(frame, path):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator='\n')
    Path(path).write_text(buffer.getvalue())


def write_json(payload, path):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_digest(w):
    return hashlib.sha256(np.ascontiguousarray(w, dtype='<f8').tobytes()).hexdigest()


def write_inputs(root, w):
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({k: w[k] for k in ('Country', 'iso3', 'm49_subregion', 'climate_zone', 'hemisphere',
                                            'spatial_block', 'income_group', 'station_lon', 'station_lat')})
    frame['warming_trend'] = w['y']          # an extra column the checker must not read
    write_csv(frame, root / 'm0_countries.csv')
    geometry = pd.DataFrame({'iso3': ['ZZZ', *w['iso3'][::-1]], 'centroid_lon': [0.0, *w['centroid_lon'][::-1]],
                             'centroid_lat': [0.0, *w['centroid_lat'][::-1]]})
    write_csv(geometry, root / 'country_geometry.csv')
    rows = []
    for rep in REPS:
        levels = levels_of(w, np.arange(N))
        x, columns = design(w, rep, 'M0star', np.arange(N), levels, None)
        for model in ('M0star', 'M1b'):
            for i in range(N):
                for j, c in enumerate(columns):
                    rows.append({'representation': rep, 'model': model, 'iso3': w['iso3'][i], 'column': c,
                                 'group': 'x', 'value': float(x[i, j]) + (1.0 if model == 'M1b' else 0.0)})
    write_csv(pd.DataFrame(rows), root / 'm1b_design_matrices.csv')
    audit = pd.DataFrame({'variant': ['drop_per_capita'] * N + ['other'] * N, 'iso3': w['iso3'] * 2,
                          'observed': list(w['y']) + list(w['y'] + 1)})
    write_csv(audit, root / 'rank_audit_country_predictions.csv')
    return {'countries': root / 'm0_countries.csv', 'design': root / 'm1b_design_matrices.csv',
            'geometry': root / 'country_geometry.csv', 'rank_audit': root / 'rank_audit_country_predictions.csv'}


def write_m2(root, w, plan, static):
    root.mkdir(parents=True, exist_ok=True)
    rows, cards, static_fits = [], {}, {}
    for rep in REPS:
        cards[rep] = {}
        for model in ('M0star', 'M2'):
            full_pred, x, columns, _ = ols_predict(w, rep, model, np.arange(N), np.arange(N))
            r2 = coalition_r2(w['y'], np.ones(N), group_blocks(x, columns))
            cards[rep][model] = {f'share_{g}': v for g, v in shapley_subsets(r2).items()}
            if model == 'M2':
                cards[rep][model]['latitude_state_full_sample'] = json.loads(state_json(knots_of(w['abs_latitude']), N))
            base = {'representation': rep, 'model': model}
            for i in range(N):
                rows.append({**base, 'protocol': 'in_sample', 'iso3': w['iso3'][i], 'Country': w['Country'][i],
                             'm49_subregion': w['m49_subregion'][i], 'observed': w['y'][i], 'prediction': full_pred[i],
                             'error': w['y'][i] - full_pred[i], 'fold_id': -1, 'n_train': N,
                             'nearest_train_km': np.nan, 'unseen_levels': 0, 'extrapolation': ''})
            for protocol in PROTOCOLS:
                ids, fits = plan[protocol]
                pred, n_train = np.zeros(N), np.zeros(N, int)
                for _fold, _label, train, test in fits:
                    pred[test] = ols_predict(w, rep, model, train, test)[0]
                    n_train[test] = len(train)
                static_fits[(rep, model, protocol)] = pred
                for i in range(N):
                    rows.append({**base, 'protocol': protocol, 'iso3': w['iso3'][i], 'Country': w['Country'][i],
                                 'm49_subregion': w['m49_subregion'][i], 'observed': w['y'][i],
                                 'prediction': pred[i], 'error': w['y'][i] - pred[i], 'fold_id': int(ids[i]),
                                 'n_train': int(n_train[i]), 'nearest_train_km': 500.0, 'unseen_levels': 0,
                                 'extrapolation': ''})
    write_csv(pd.DataFrame(rows), root / 'm2_country_predictions.csv')
    write_json({'representations': cards}, root / 'm2_scorecard.json')
    write_json({'artifact_sha256': {n: sha(root / n) for n in ('m2_scorecard.json', 'm2_country_predictions.csv')},
                'retained_static_specification': static, 'integrity_passed': True}, root / 'm2_result_manifest.json')
    return cards, static_fits


def run_family(w, rep, family, model, plan, d):
    y = w['y']
    everyone = np.arange(N)
    levels = levels_of(w, everyone)
    knots = knots_of(w['abs_latitude']) if model == 'M2' else None
    x, columns = design(w, rep, model, everyone, levels, knots)
    wf = knn_dense(d, everyone)
    full = spatial_fit(family, y, x, wf)
    m = np.eye(N) - full['theta'] * wf
    mean = x @ full['beta']
    if family == 'sem':
        trend, one_step = mean, mean + full['theta'] * wf @ (y - mean)
    else:
        trend, one_step = np.linalg.solve(m, mean), full['theta'] * wf @ y + mean
    run = {'full': {**full, 'columns': columns, 'x': x, 'w': wf, 'trend': trend, 'one_step': one_step,
                    'train': everyone, 'levels': '', 'state': ''}, 'protocols': {}}
    for protocol in PROTOCOLS:
        ids, fits = plan[protocol]
        pred, adjust, unseen, n_train = np.zeros(N), np.zeros(N), np.zeros(N, int), np.zeros(N, int)
        records = []
        for fold, label, train, test in fits:
            lv = levels_of(w, train)
            kn = knots_of(w['abs_latitude'][train]) if model == 'M2' else None
            xt, cols = design(w, rep, model, train, lv, kn)
            assert np.linalg.matrix_rank(xt) == xt.shape[1], (protocol, label)
            wt = knn_dense(d, train)
            fit = spatial_fit(family, y[train], xt, wt)
            attach = [first_eight(d[o], train) for o in test]
            local = {int(v): k for k, v in enumerate(train)}
            w_ot = np.zeros((len(test), len(train)))
            for k, nb in enumerate(attach):
                w_ot[k, [local[int(j)] for j in nb]] = 1.0 / 8
            source = y[train] - xt @ fit['beta'] if family == 'sem' else y[train]
            xo, _ = design(w, rep, model, test, lv, kn)
            base = xo @ fit['beta'] + fit['theta'] * (w_ot @ source)
            final = base + adjustment(w, fit['beta'], cols, lv, test)
            pred[test], adjust[test] = final, final - base
            unseen[test], n_train[test] = unseen_count(w, lv, test), len(train)
            records.append({**fit, 'fold': fold, 'label': label, 'train': train, 'test': test, 'columns': cols,
                            'levels': json.dumps(lv, sort_keys=True),
                            'state': state_json(kn, len(train)) if kn else '', 'attach': attach,
                            'neighbours': knn_lists(d, train)})
        run['protocols'][protocol] = {'ids': ids, 'pred': pred, 'adjust': adjust, 'unseen': unseen,
                                      'n_train': n_train, 'records': records}
    return run


def write_package(root, w, plan, *, arm, static, m2_dir, m2_cards, static_fits, inputs):
    root.mkdir(parents=True, exist_ok=True)
    model = {'M0*': 'M0star', 'M2': 'M2'}[static]
    station_d = haversine(w['station_lon'], w['station_lat'])
    land_d = haversine(w['centroid_lon'], w['centroid_lat'])
    d = station_d if arm == 'station' else land_d
    sw, lw = knn_dense(station_d, np.arange(N)), knn_dense(land_d, np.arange(N))
    y, iso3 = w['y'], np.array(w['iso3'])
    runs = {(rep, fam): run_family(w, rep, fam, model, plan, d) for rep in REPS for fam in FAMS}
    prediction_rows, fit_rows, graph_rows, grid_rows, seen = [], [], [], [], set()
    entries = {}
    for (rep, fam), run in runs.items():
        full = run['full']
        base = {'weights': arm, 'family': fam, 'representation': rep}
        for i in range(N):
            prediction_rows.append({**base, 'protocol': 'in_sample', 'iso3': iso3[i], 'm49_subregion': w['m49_subregion'][i],
                                    'observed': y[i], 'prediction': full['one_step'][i],
                                    'error': y[i] - full['one_step'][i], 'trend': full['trend'][i], 'fold_id': -1,
                                    'n_train': N, 'nearest_train_km': np.nan, 'unseen_levels': 0, 'adjustment': 0.0})
        for protocol in PROTOCOLS:
            block = run['protocols'][protocol]
            for i in range(N):
                prediction_rows.append({**base, 'protocol': protocol, 'iso3': iso3[i],
                                        'm49_subregion': w['m49_subregion'][i], 'observed': y[i],
                                        'prediction': block['pred'][i], 'error': y[i] - block['pred'][i], 'trend': np.nan,
                                        'fold_id': int(block['ids'][i]), 'n_train': int(block['n_train'][i]),
                                        'nearest_train_km': 400.0, 'unseen_levels': int(block['unseen'][i]),
                                        'adjustment': float(block['adjust'][i])})
        fits = [('full_sample', 'all', full, None)] + [(p, r['label'], r, r) for p in PROTOCOLS
                                                        for r in run['protocols'][p]['records']]
        for protocol, label, fit, record in fits:
            fit_rows.append({**base, 'protocol': protocol, 'fit': label, 'n': len(fit['train']), 'p': len(fit['columns']),
                             'theta': fit['theta'], 'loglik': fit['loglik'], 'sigma2': fit['sigma2'],
                             'at_domain_bound': fit['at_bound'], 'grid_local_maxima': fit['maxima'],
                             'evaluations': fit['evaluations'], 'columns': '|'.join(fit['columns']),
                             'beta': json.dumps([float(v) for v in fit['beta']]),
                             'training_iso3': '|'.join(iso3[fit['train']]), 'levels': fit['levels'],
                             'latitude_state': fit['state']})
            if rep != 'primary_total_co2' or (protocol, label) in seen:
                continue
            seen.add((protocol, label))
            lists = knn_lists(d, fit['train']) if record is None else record['neighbours']
            for node_, nb in zip(fit['train'], lists):
                graph_rows.append({'weights': arm, 'protocol': protocol, 'fit': label, 'role': 'training',
                                   'iso3': iso3[node_], 'neighbours': '|'.join(iso3[nb])})
            if record is not None:
                for node_, nb in zip(record['test'], record['attach']):
                    graph_rows.append({'weights': arm, 'protocol': protocol, 'fit': label, 'role': 'held_out',
                                       'iso3': iso3[node_], 'neighbours': '|'.join(iso3[nb])})
        for theta, value in zip(np.linspace(-0.99, 0.99, 199), full['grid']):
            grid_rows.append({**base, 'theta': float(theta), 'loglik': float(value)})
        entries[(rep, fam)] = score_entry(w, rep, fam, run, arm, static, model, static_fits, m2_cards, sw, lw, d)

    identity = representation_identity(runs, entries)
    integrity = identity['passes']
    naming = None
    if arm == 'station' and integrity:
        qualifying = [f'M3-{f.upper()}({static})' for f in FAMS
                      if entries[('primary_total_co2', f)]['qualification']['qualifies']]
        naming = {'retained_static_specification': static, 'qualifying_spatial_extensions': qualifying,
                  'final_primary_predictive_model': static if not qualifying else qualifying[0]
                  if len(qualifying) == 1 else None, 'reason': 'synthetic', 'order_note': 'synthetic'}
    scorecard = {'specification': 'synthetic', 'weights': arm, 'branch': 'B' if static == 'M2' else 'A',
                 'retained_static_specification': static, 'static_reproduction': {},
                 'families': {f'{r}/{f}': e for (r, f), e in entries.items()}, 'representation_identity': identity,
                 'final_naming': naming, 'integrity_failures': [] if integrity else ['identity'],
                 'integrity_passed': integrity}
    write_json(scorecard, root / 'm3_scorecard.json')
    write_csv(pd.DataFrame(prediction_rows), root / 'm3_country_predictions.csv')
    write_csv(pd.DataFrame(fit_rows), root / 'm3_fits.csv')
    write_csv(pd.DataFrame(graph_rows), root / 'm3_graphs.csv')
    write_csv(pd.DataFrame(grid_rows), root / 'm3_full_sample_grid.csv')
    write_json({'inputs': {PINS[k]: {'sha256': sha(p), 'frozen_at': 'synthetic'} for k, p in inputs.items()},
                'm2_result_manifest_sha256': sha(m2_dir / 'm2_result_manifest.json'),
                'identity': {'station_weights_sha256': weights_digest(sw), 'area_weights_sha256': weights_digest(lw)},
                'weights_distance_source': arm}, root / 'm3_provenance.json')
    write_manifest(root, scorecard)
    return scorecard


def write_manifest(root, scorecard):
    write_json({'result': 'synthetic', 'weights': scorecard['weights'],
                'retained_static_specification': scorecard['retained_static_specification'],
                'family_status': {k: v['status'] for k, v in scorecard['families'].items()},
                'final_naming': scorecard['final_naming'], 'integrity_passed': scorecard['integrity_passed'],
                'artifact_sha256': {n: sha(root / n) for n in M3_ARTIFACTS},
                'non_deterministic_artifacts': ['m3_run_metadata.json']}, root / 'm3_result_manifest.json')


def score_entry(w, rep, fam, run, arm, static, model, static_fits, m2_cards, sw, lw, d):
    y, full = w['y'], run['full']
    regions = np.asarray(w['m49_subregion'])
    blocks = run['protocols']
    cards = {p: card(y, blocks[p]['pred'], blocks[p]['ids'], blocks[p]['n_train'], blocks[p]['unseen'], regions,
                     full['one_step'], sw, lw) for p in PROTOCOLS}
    top = {**cards['primary_loco'], 'secondary': cards['m49_subregion_lo'], 'random_reference_only': cards['random10']}
    tss = float(np.sum((y - y.mean()) ** 2))
    top['trend_r2'] = float(1 - np.sum((y - full['trend']) ** 2) / tss)
    top['structural_residual_moran_in_sample'] = moran(y - full['trend'], sw)
    top['effective_degrees_of_freedom'] = len(full['columns']) + 1
    top['sigma2'] = full['sigma2']
    static = {p: static_fits[(rep, model, p)] for p in PROTOCOLS}
    pair = paired(y, blocks['primary_loco']['pred'], static['primary_loco'])
    change = np.abs(y - blocks['primary_loco']['pred']) - np.abs(y - static['primary_loco'])
    paired_block = {'vs_static': pair, 'countries': {'improved': int((change < 0).sum()), 'worsened': int((change > 0).sum()),
                                                     'tied': int((change == 0).sum())}}
    for p in ('m49_subregion_lo', 'random10'):
        paired_block[f'{p}_rmse_change_vs_static'] = rmse(y - blocks[p]['pred']) - rmse(y - static[p])
    if model == 'M2':
        paired_block['vs_m0star_descriptive'] = paired(y, blocks['primary_loco']['pred'],
                                                       static_fits[(rep, 'M0star', 'primary_loco')])
    static_regions = card(y, static['primary_loco'], blocks['primary_loco']['ids'], blocks['primary_loco']['n_train'],
                          blocks['primary_loco']['unseen'], regions, full['one_step'], sw, lw)['worst_region']['rmse']
    delta, (lo, hi) = pair['delta_rmse'], pair['country_bootstrap_95_interval']
    m49 = cards['m49_subregion_lo']['cv_rmse'] - rmse(y - static['m49_subregion_lo'])
    region = cards['primary_loco']['worst_region']['rmse'] - static_regions
    q = {'Q1_delta_below_zero': delta < 0, 'Q2_interval_upper_below_zero': hi < 0, 'Q3_m49_veto_passed': m49 <= 0.001,
         'Q4_worst_region_veto_passed': region <= 0.001, 'Q5_computable': True}
    conditions = {**q, 'qualifies': all(q.values()), 'delta_rmse': delta, 'interval': [lo, hi], 'm49_rmse_change': m49,
                  'worst_region_rmse_change': region, 'veto_threshold': 0.001, 'worsened_generalization': delta > 0 and lo > 0}

    x, wf, theta = full['x'], full['w'], full['theta']
    shares = {g: m2_cards[rep][model][f'share_{g}'] for g in GROUPS}
    rss_s = float(np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2))
    rss_e = float(np.sum((y - full['one_step']) ** 2))
    m = np.eye(N) - theta * wf
    target = m @ y
    groups = group_blocks(x, full['columns'])
    blocks_f = {g: (m @ b if fam == 'sem' else b) for g, b in groups.items()}
    r2 = coalition_r2(target, np.ones(N), blocks_f)
    filtered = shapley_subsets(r2)
    z = np.column_stack([np.ones(N)] + [blocks_f[g] for g in GROUPS])
    full_fit = z @ np.linalg.lstsq(z, target, rcond=None)[0]
    ml_fit = (m @ x if fam == 'sem' else x) @ full['beta']
    r2_full = r2[frozenset(GROUPS)]
    accounting = {
        'estimands': {'a_static': 1 - rss_s / tss, 'a_dependence': (rss_s - rss_e) / tss, 'a_innovation': rss_e / tss,
                      'absorbed_fraction_of_static_residual': (rss_s - rss_e) / rss_s,
                      'identity_sum': (1 - rss_s / tss) + (rss_s - rss_e) / tss + rss_e / tss, 'tss': tss,
                      'rss_static': rss_s, 'rss_innovation': rss_e, 'one_step_r2': 1 - rss_e / tss,
                      'trend_r2': top['trend_r2']},
        'static_lmg_shares': shares, 'static_lmg_share_sum_equals_a_static': True,
        'filtered_trend_shares': {'shares': filtered, 'r2_full': r2_full, 'residual_share': 1 - r2_full, 'n_coalitions': 16,
                                  'full_coalition_max_gap': float(np.max(np.abs(full_fit - ml_fit)))},
        'composition': {'static': {g: v / sum(shares.values()) for g, v in shares.items()},
                        'filtered': {g: v / r2_full for g, v in filtered.items()}},
        'material_change_of_non_spatial_part': bool(not all(filtered['geography'] > v for g, v in filtered.items()
                                                            if g != 'geography') or filtered['emissions'] > 0.10)}

    dependence = {'full_sample': {'theta': theta, 'at_domain_bound': full['at_bound']}}
    for p in PROTOCOLS:
        thetas = np.array([r['theta'] for r in blocks[p]['records']])
        dependence[p] = {'n_fits': len(thetas), 'median': float(np.median(thetas)), 'min': float(thetas.min()),
                         'max': float(thetas.max()), 'strictly_positive_n': int((thetas > 0).sum()),
                         'at_domain_bound_n': int(sum(r['at_bound'] for r in blocks[p]['records']))}
    primary = np.array([r['theta'] for r in blocks['primary_loco']['records']])
    dependence['same_sign_as_full_fraction_primary'] = float(np.mean(np.sign(primary) == np.sign(theta)))
    neighbours = knn_lists(d, np.arange(N))
    near = np.array([[d[i, nb[0]], d[i, nb[-1]]] for i, nb in enumerate(neighbours)])
    descriptors = {'full_sample': {'first_neighbour_km_median': float(np.median(near[:, 0])),
                                   'first_neighbour_km_max': float(near[:, 0].max()),
                                   'eighth_neighbour_km_median': float(np.median(near[:, 1])),
                                   'eighth_neighbour_km_max': float(near[:, 1].max())}}
    for p in PROTOCOLS:
        att = np.array([[d[o, nb[0]], d[o, nb[-1]]] for r in blocks[p]['records'] for o, nb in zip(r['test'], r['attach'])])
        descriptors[p] = {f'attachment_{s}_km_{name}': float(fn(att[:, k])) for k, s in enumerate(('first', 'eighth'))
                          for name, fn in (('min', np.min), ('median', np.median), ('max', np.max))}
    entry = {'status': 'computable', 'failure': None, 'name': f'M3-{fam.upper()}({static})', 'weights': arm,
             'accounting': accounting,
             'dependence_parameter_full_sample': {'theta': theta, 'at_domain_bound': full['at_bound'], 'se': 0.1},
             'card': top, 'paired': paired_block, 'dependence_parameter': dependence, 'graph_descriptors': descriptors}
    if arm == 'station' and rep == 'primary_total_co2':
        entry['qualification'] = conditions
    else:
        entry['arm_conditions'] = {**conditions, 'descriptive_only': True, 'note': 'synthetic'}
    return json.loads(json.dumps(entry, default=lambda v: v.item() if isinstance(v, np.generic) else v))


def representation_identity(runs, entries):
    report, passes = {}, True
    for fam in FAMS:
        a, b = runs[('primary_total_co2', fam)], runs[('per_capita', fam)]
        predictions = {'in_sample_one_step': float(np.max(np.abs(a['full']['one_step'] - b['full']['one_step'])))}
        for p in PROTOCOLS:
            predictions[p] = float(np.max(np.abs(a['protocols'][p]['pred'] - b['protocols'][p]['pred'])))
        thetas = [abs(a['full']['theta'] - b['full']['theta'])] + [
            abs(ra['theta'] - rb['theta']) for ra, rb in zip(a['protocols']['primary_loco']['records'],
                                                            b['protocols']['primary_loco']['records'])]
        ea, eb = entries[('primary_total_co2', fam)], entries[('per_capita', fam)]

        def metric(entry, name):
            value = entry['card']
            for part in name.split('.'):
                value = value[part]
            return value
        scores = {name: abs(metric(ea, name) - metric(eb, name)) for name in ic.EQUIVALENT_SCORES}
        scores['delta_rmse'] = abs(ea['paired']['vs_static']['delta_rmse'] - eb['paired']['vs_static']['delta_rmse'])
        scores['interval'] = max(abs(x - z) for x, z in zip(ea['paired']['vs_static']['country_bootstrap_95_interval'],
                                                            eb['paired']['vs_static']['country_bootstrap_95_interval']))
        parts = {k: abs(ea['accounting']['estimands'][k] - eb['accounting']['estimands'][k]) for k in ic.ACCOUNTING_PARTS}
        ok = (all(v <= 1e-6 for v in predictions.values()) and max(thetas) <= 1e-5
              and all(v <= 1e-6 for v in scores.values()) and all(v <= 1e-6 for v in parts.values()))
        report[fam] = {'predictions': predictions, 'theta_max': float(max(thetas)), 'scores': scores,
                       'accounting_parts': parts, 'passes': bool(ok)}
        passes &= ok
    report['tolerances'] = {'predictions': 1e-6, 'theta': 1e-5, 'scores_and_accounting': 1e-6}
    report['passes'] = bool(passes)
    return report


def build(root, *, arm, static, seed=3):
    w = world(seed)
    plan = fold_plan(w)
    inputs = write_inputs(root / 'inputs', w)
    cards, static_fits = write_m2(root / 'm2_primary', w, plan, static)
    package = root / ('m3_station' if arm == 'station' else 'm3_land_centroid')
    scorecard = write_package(package, w, plan, arm=arm, static=static, m2_dir=root / 'm2_primary', m2_cards=cards,
                              static_fits=static_fits, inputs=inputs)
    return {'root': root, 'package': package, 'm2': root / 'm2_primary', 'inputs': inputs, 'scorecard': scorecard}


def check(built, package=None, controls=True):
    return ic.run(package or built['package'], built['m2'], expected=EXPECTED, controls=controls, **built['inputs'])


def cli_args(built, package=None, out=None):
    args = ['--package', str(package or built['package']), '--m2', str(built['m2'])]
    for flag_name, key in (('--countries', 'countries'), ('--design', 'design'), ('--geometry', 'geometry'),
                           ('--rank-audit', 'rank_audit')):
        args += [flag_name, str(built['inputs'][key])]
    return args + (['--out', str(out)] if out else [])


@pytest.fixture(scope='module')
def station(tmp_path_factory):
    built = build(tmp_path_factory.mktemp('station'), arm='station', static='M2')
    code = ic.main(cli_args(built), expected=EXPECTED)
    report = json.loads(ic.default_out(built['package']).read_text())
    return {**built, 'code': code, 'report': report}


@pytest.fixture(scope='module')
def land(tmp_path_factory):
    built = build(tmp_path_factory.mktemp('land'), arm='land', static='M0*')
    return {**built, 'report': check(built)}


def copy_built(built, tmp_path):
    root = tmp_path / 'copy'
    shutil.copytree(built['root'], root)
    moved = {**built, 'root': root, 'package': root / built['package'].name, 'm2': root / 'm2_primary',
             'inputs': {k: root / 'inputs' / Path(v).name for k, v in built['inputs'].items()}}
    return moved


def failed_details(report):
    return {name: [c for c in block['checks'] if not c['passes']] for name, block in report['sections'].items()
            if not block['passes']}


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_fixture_exercises_both_integrity_paths_unseen_levels_ties_and_buffers(station):
    scorecard = station['scorecard']
    assert scorecard['integrity_passed'] and scorecard['final_naming'] is not None
    fits = pd.read_csv(station['package'] / 'm3_fits.csv', keep_default_na=False, dtype={'fit': str})
    assert set(fits.protocol) == {'full_sample', *PROTOCOLS} and len(fits) == 4 * (1 + N + 4 + 4)
    primary = fits[(fits.protocol == 'primary_loco') & (fits.representation == 'primary_total_co2')]
    assert primary.n.min() < N - 1                                   # the synthetic buffer excludes neighbours
    predictions = pd.read_csv(station['package'] / 'm3_country_predictions.csv')
    assert (predictions.unseen_levels > 0).any() and (predictions.adjustment != 0).any()
    assert set(fits[fits.protocol == 'full_sample'].levels) == {''}  # the evaluator writes no full-sample state
    assert fits.latitude_state.str.len().gt(0).sum() == 4 * (N + 8)


def test_consistent_station_package_passes_every_section_and_exits_zero(station):
    report = station['report']
    assert report['all_checks_pass'], failed_details(report)
    assert report['negative_controls_all_detected'], report['negative_controls']
    assert station['code'] == 0
    assert report['fits_checked'] == 4 * (1 + N + 4 + 4)
    assert all(block['n_comparisons'] > 0 for block in report['section_summary'].values())
    assert report['arm'] == 'station' and report['retained_static_specification'] == 'M2'


def test_consistent_land_package_in_branch_a_passes(land):
    report = land['report']
    assert report['all_checks_pass'], failed_details(report)
    assert report['negative_controls_all_detected'], report['negative_controls']
    naming = report['sections']['10_qualification_and_naming']['checks']
    assert any(c['check'] == 'final_naming_is_null' and c['passes'] for c in naming)


@pytest.mark.parametrize('name,channel', [
    ('a_perturbed_theta', 'loglik'), ('b_swapped_neighbour', 'training_neighbour_lists_equal_knn8_recomputation'),
    ('c_perturbed_prediction', 'held_out_prediction'), ('d_shifted_veto_threshold', None),
    ('e_perturbed_beta', 'beta_qr_at_saved_theta')])
def test_each_negative_control_is_detected_from_a_clean_baseline(station, name, channel):
    control = station['report']['negative_controls'][name]
    assert control['detected'] is True
    assert control['baseline_failed_checks'] == []
    if channel is not None:
        assert channel in control['failed_checks']
    else:
        assert all(v['flip_flagged'] for v in control['variants'].values())


def test_perturbed_theta_also_fails_optimality_and_predictions(station):
    failed = station['report']['negative_controls']['a_perturbed_theta']['failed_checks']
    assert {'local_maximum_on_fine_grid', 'beta_qr_at_saved_theta', 'held_out_prediction'} <= set(failed)


def test_perturbed_prediction_also_moves_the_card_metrics(station):
    metric_failed = station['report']['negative_controls']['c_perturbed_prediction']['also_failed_metric_checks']
    assert 'card.cv_rmse' in metric_failed


def test_manifest_digest_tampering_is_refused(station, tmp_path):
    built = copy_built(station, tmp_path)
    path = built['package'] / 'm3_fits.csv'
    path.write_text(path.read_text().replace('primary_loco', 'primary_loco', 1) + '\n')
    with pytest.raises(ic.DigestRefused, match='manifest verification failed'):
        check(built, controls=False)
    out = tmp_path / 'refused.json'
    assert ic.main(cli_args(built, out=out), expected=EXPECTED) == 3
    record = json.loads(out.read_text())
    assert record['all_checks_pass'] is False and 'refused' in record


def test_m2_manifest_tampering_is_refused(station, tmp_path):
    built = copy_built(station, tmp_path)
    path = built['m2'] / 'm2_country_predictions.csv'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ic.DigestRefused):
        check(built, controls=False)


def test_unlisted_required_artifact_is_refused(station, tmp_path):
    built = copy_built(station, tmp_path)
    manifest = json.loads((built['package'] / 'm3_result_manifest.json').read_text())
    del manifest['artifact_sha256']['m3_graphs.csv']
    write_json(manifest, built['package'] / 'm3_result_manifest.json')
    with pytest.raises(ic.DigestRefused):
        check(built, controls=False)


def test_consistently_rehashed_wrong_values_are_caught(station, tmp_path):
    built = copy_built(station, tmp_path)
    fits = pd.read_csv(built['package'] / 'm3_fits.csv', keep_default_na=False, dtype=str)
    row = fits.index[(fits.protocol == 'm49_subregion_lo') & (fits.family == 'sar')
                     & (fits.representation == 'per_capita')][0]
    fits.loc[row, 'loglik'] = repr(float(fits.loc[row, 'loglik']) + 1e-6)
    broken = fits.index[(fits.protocol == 'random10') & (fits.family == 'sem') & (fits.representation == 'per_capita')][0]
    fits.loc[broken, 'beta'] = '[1.0'
    write_csv(fits, built['package'] / 'm3_fits.csv')
    scorecard = json.loads((built['package'] / 'm3_scorecard.json').read_text())
    scorecard['families']['primary_total_co2/sem']['accounting']['estimands']['a_dependence'] += 1e-8
    scorecard['families']['per_capita/sar']['card']['secondary']['worst_region']['region'] = 'Region Z'
    write_json(scorecard, built['package'] / 'm3_scorecard.json')
    write_manifest(built['package'], scorecard)
    report = check(built, controls=False)
    assert not report['all_checks_pass']
    failed = set(report['failed_checks'])
    assert '4_likelihood: loglik' in failed
    assert '11_accounting_and_filtered_shares: accounting.estimands.a_dependence' in failed
    assert '9_metrics_and_paired: card.worst_region.region' in failed
    assert '2_fit_structure_designs_states: fit_row_parses' in failed
    assert '9_metrics_and_paired: status_equals_recomputed_computability' in failed
    assert '3_graphs: training_neighbour_lists_equal_knn8_recomputation' not in failed


def test_a_wrong_saved_qualification_or_naming_is_caught(station, tmp_path):
    built = copy_built(station, tmp_path)
    scorecard = json.loads((built['package'] / 'm3_scorecard.json').read_text())
    entry = scorecard['families']['primary_total_co2/sar']['qualification']
    entry['Q3_m49_veto_passed'] = not entry['Q3_m49_veto_passed']
    scorecard['final_naming']['final_primary_predictive_model'] = 'M3-SAR(M2)-wrong'
    write_json(scorecard, built['package'] / 'm3_scorecard.json')
    write_manifest(built['package'], scorecard)
    failed = set(check(built, controls=False)['failed_checks'])
    assert '10_qualification_and_naming: qualification.Q3_m49_veto_passed' in failed
    assert '10_qualification_and_naming: final_naming.final_primary_predictive_model' in failed


def test_a_non_computable_family_disposition_is_verified_and_a_false_status_is_caught(station, tmp_path):
    built = copy_built(station, tmp_path)
    package = built['package']
    fits = pd.read_csv(package / 'm3_fits.csv', keep_default_na=False, dtype=str)
    first = fits[(fits.family == 'sar') & (fits.protocol == 'primary_loco')].fit.unique()[:5]
    keep = (fits.family != 'sar') | (fits.protocol == 'full_sample') | ((fits.protocol == 'primary_loco') & fits.fit.isin(first))
    write_csv(fits[keep], package / 'm3_fits.csv')
    predictions = pd.read_csv(package / 'm3_country_predictions.csv', keep_default_na=False, dtype=str)
    write_csv(predictions[predictions.family != 'sar'], package / 'm3_country_predictions.csv')
    scorecard = json.loads((package / 'm3_scorecard.json').read_text())
    for rep in REPS:
        old = scorecard['families'][f'{rep}/sar']
        entry = {k: old[k] for k in ('name', 'weights', 'accounting', 'dependence_parameter_full_sample')}
        entry.update({'status': 'non_computable', 'failure': {'protocol': 'primary_loco', 'fit': 'S05',
                                                              'reason': 'evaluation-failed', 'detail': 'synthetic'}})
        conditions = {'qualifies': False, 'Q5_computable': False, 'reason': 'non-computable family'}
        if rep == 'primary_total_co2':
            entry['qualification'] = conditions
        else:
            entry['arm_conditions'] = {**conditions, 'descriptive_only': True, 'note': 'synthetic'}
        scorecard['families'][f'{rep}/sar'] = entry
    scorecard['representation_identity']['sar'] = {'status': 'non_computable'}
    scorecard['representation_identity']['passes'] = scorecard['representation_identity']['sem']['passes']
    write_json(scorecard, package / 'm3_scorecard.json')
    write_manifest(package, scorecard)
    report = check(built, controls=False)
    assert report['all_checks_pass'], failed_details(report)

    scorecard['families']['per_capita/sar']['status'] = 'computable'
    write_json(scorecard, package / 'm3_scorecard.json')
    write_manifest(package, scorecard)
    failed = set(check(built, controls=False)['failed_checks'])
    assert '9_metrics_and_paired: status_equals_recomputed_computability' in failed


def test_final_naming_table():
    assert ic.final_naming('M0*', {'sem': False, 'sar': False})['final_primary_predictive_model'] == 'M0*'
    one = ic.final_naming('M2', {'sem': False, 'sar': True})
    assert one['final_primary_predictive_model'] == 'M3-SAR(M2)' and one['qualifying_spatial_extensions'] == ['M3-SAR(M2)']
    both = ic.final_naming('M0*', {'sem': True, 'sar': True})
    assert both['final_primary_predictive_model'] is None
    assert both['qualifying_spatial_extensions'] == ['M3-SEM(M0*)', 'M3-SAR(M0*)']


@pytest.mark.parametrize('delta,lo,hi,m49,worst,qualifies', [
    (-0.001, -0.002, -0.0001, 0.001, 0.001, True), (-0.001, -0.002, 0.0, 0.0, 0.0, False),
    (0.0, -0.001, -0.0001, 0.0, 0.0, False), (-0.001, -0.002, -0.0001, 0.0011, 0.0, False),
    (-0.001, -0.002, -0.0001, 0.0, 0.0011, False)])
def test_qualification_boundaries(delta, lo, hi, m49, worst, qualifies):
    assert ic.qualification(delta, lo, hi, m49, worst, True)['qualifies'] is qualifies
    assert ic.qualification(delta, lo, hi, m49, worst, False) == {'qualifies': False, 'Q5_computable': False}
    assert ic.qualification(math.nan, lo, hi, m49, worst, True) == {'qualifies': None}


def test_natural_cubic_columns_and_knots_match_direct_formulas():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 70, 25)
    knots = ic.knots_from(a)
    np.testing.assert_allclose(knots, knots_of(a), rtol=0, atol=1e-12)
    ours = ic.natural_cubic(knots, a)

    def d(k):
        return (np.maximum(a - knots[k], 0) ** 3 - np.maximum(a - knots[3], 0) ** 3) / (knots[3] - knots[k])
    np.testing.assert_allclose(ours, np.column_stack([d(0) - d(2), d(1) - d(2)]), rtol=1e-12, atol=1e-9)
    beyond = ic.natural_cubic(knots, np.array([knots[3] + 1, knots[3] + 2, knots[3] + 3]))
    np.testing.assert_allclose(np.diff(beyond, 2, axis=0), 0, atol=1e-6)      # linear above the last knot


def test_haversine_and_knn_follow_the_tie_rule():
    lon, lat = np.array([0.0, 1.0, 1.0, -1.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    d = ic.haversine_km(lon, lat)
    np.testing.assert_allclose(d, haversine(lon, lat), rtol=1e-13, atol=1e-9)
    assert d[1, 2] == 0.0 and d[0, 1] == d[0, 2] == d[0, 3]
    assert d[0, 1] == pytest.approx(2 * math.pi * R_EARTH / 360, rel=1e-12)


def test_column_groups_follow_the_named_prefixes():
    names = ['intercept', 'cum_co2_total', 'abs_latitude', 'abs_latitude_ns1', 'hemisphere=S:abs_latitude',
             'hemisphere=S', 'climate_zone=B', 'spatial_block=Asia', 'elevation', 'continentality',
             'income_group=Low-income countries', 'population', 'station_density']
    assert [ic.column_group(n) for n in names] == ['intercept', 'emissions', 'geography', 'geography', 'geography',
                                                  'geography', 'geography', 'geography', 'geography', 'geography',
                                                  'socioeconomic', 'population', 'population']


def test_module_imports_none_of_the_evaluator_modules():
    tree = ast.parse(Path(ic.__file__).read_text())
    imported = set()
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Import):
            imported |= {alias.name for alias in statement.names}
        elif isinstance(statement, ast.ImportFrom):
            imported |= {f'{statement.module}.{alias.name}' for alias in statement.names} | {statement.module}
    forbidden = ('m3_spatial', 'm3_evaluate', 'm2_evaluate', 'm2_feasibility', 'm2_latitude_basis', 'cv', 'spatial',
                 'run_territory_correction', 'decomposition', 'stability', 'v2_provenance', 'm2_independent_check')
    for name in imported:
        assert not any(part in forbidden for part in name.split('.')), name
    local = {name for name in imported if name.startswith(('research', 'src'))}
    assert local <= {'research.model_v2', 'research.model_v2.m3_independent_math'}, local
    third_party = {name.split('.')[0] for name in imported} - {'research', '__future__'}
    assert third_party <= {'argparse', 'copy', 'dataclasses', 'hashlib', 'json', 'math', 'sys', 'pathlib', 'typing',
                           'numpy', 'pandas'}, third_party
