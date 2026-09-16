"""M3 structural feasibility audit: graphs, log-determinants and the estimator on synthetic outcomes only.

Before any M2 score, this checks that ``DOWNSTREAM_COMPLETION_SPEC.md`` §2 is executable on the frozen
sample, for both possible retained static specifications (M0* and M2), without reading any warming outcome:

* **Graphs** for both weights arms (station centroid, land centroid) and every fit the spec requires or
  reports (full sample; primary 500 km LOCO; M49; random 10-fold; the M4 1000 km territorial and 1500 km
  land-centroid buffers): training-set size, kNN8 row sums and diagonal, distance ties at the eighth/ninth
  neighbour boundary for training nodes and held-out attachments, attachment distances, the sign and
  finiteness of log|I - theta W| over the whole frozen grid, and the spectrum's extreme values.
* **Estimator executability** on the real encoded designs with a **synthetic** outcome drawn from each family
  (fixed seeds): every station-weight fit of the full sample and the primary, M49 and random protocols, for
  both families and both static designs, through the frozen maximizer; failures, interior status, local
  maxima, timing and recovery of the synthetic dependence parameter.

The canonical record is written only from committed, pushed code:
``uv run python -m research.model_v2.m3_feasibility``.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.model_v2 import cv, territory
from research.model_v2 import m2_feasibility as feas
from research.model_v2 import m2_latitude_basis as lb
from research.model_v2 import m3_spatial as ms
from research.model_v2 import v2_provenance as prov
from research.model_v2.run_m0_scorecard import PRIMARY_BUFFER_KM, SENSITIVITY_BORDER_BUFFER_KM
from research.model_v2.run_m0_scorecard import SENSITIVITY_CENTROID_BUFFER_KM
from research.model_v2.spatial import haversine_matrix, knn_weights

ROOT = prov.ROOT
OUTPUTS = 'research/model_v2/outputs'
OUT = ROOT / OUTPUTS / 'm3_feasibility'
COUNTRY_TABLE = f'{OUTPUTS}/m0_countries.csv'
GEOMETRY = f'{OUTPUTS}/country_geometry.csv'
FOLDS = f'{OUTPUTS}/m1b_primary/m1b_cv_folds.csv'
ENCODED_DESIGN = f'{OUTPUTS}/m1b_primary/m1b_design_matrices.csv'
M1B_PROVENANCE = f'{OUTPUTS}/m1b_primary/m1b_provenance.json'
PINS = {
    COUNTRY_TABLE: '9e99b3798e91f2b7e40250921edd5e5a788e4191df3e9027dceb98fe76b43536',
    GEOMETRY: 'c8d72866d97d13288b3024781cd262f378ce08749d5166e0d4805bea01903e3a',
    FOLDS: '6a9e3976f51e60a66c6b70fd65cc56626df1eec28a41e774c39260d1a2a459f1',
    ENCODED_DESIGN: 'bbdc8b1df1314b55c409f6ff59db03d39a9109d0a6827bef9ec58ce8a4caa982',
    M1B_PROVENANCE: '415a7c6437fa54f2013f894721c6c9dce4e255e173a67b4b3cd85e2650ea1b19',
    f'{OUTPUTS}/territory_distance_lower_km.csv': '79a20f007ebe08044833753956cdf859c27d505a2c10747b5e4df3469ee2c6e4',
}
TABLE_COLUMNS = ['Country', 'iso3', 'm49_subregion', 'station_lon', 'station_lat', 'centroid_lon', 'centroid_lat',
                 'climate_zone', 'hemisphere', 'spatial_block', 'income_group']
RECORD_FILES = ('m3_feasibility_graphs.csv', 'm3_feasibility_synthetic_fits.csv', 'm3_feasibility_summary.json')
SYNTHETIC_THETA = {'sem': 0.4, 'sar': 0.4}
SYNTHETIC_SIGMA = 0.03
SYNTHETIC_SEED = 20260916
CODE = ('research/model_v2/m3_feasibility.py',)


def inputs():
    for rel, pin in PINS.items():
        if prov.sha256(ROOT / rel) != pin:
            raise prov.ExecutionRefused(f'{rel} does not match its pinned digest')
    table = pd.read_csv(ROOT / COUNTRY_TABLE, usecols=TABLE_COLUMNS, float_precision='round_trip')[TABLE_COLUMNS]
    geometry = (pd.read_csv(ROOT / GEOMETRY, usecols=['iso3', 'centroid_lon', 'centroid_lat'],
                            float_precision='round_trip').set_index('iso3').loc[table.iso3])
    encoded = pd.read_csv(ROOT / ENCODED_DESIGN, float_precision='round_trip')
    guard = [c for c in encoded.columns if c not in feas.ENCODED_COLUMNS]
    if guard:
        raise ValueError(f'unexpected encoded-design columns {guard}')
    return table, geometry, encoded


def frame_for(table, encoded, representation):
    rows = encoded[(encoded.representation == representation) & (encoded.model == 'M0star')]
    wide = rows.pivot(index='iso3', columns='column', values='value')
    numerics = wide.loc[table.iso3, feas.NUMERIC_COLUMNS[representation]].reset_index()
    return table.drop(columns=['centroid_lon', 'centroid_lat']).merge(numerics, on='iso3', validate='one_to_one')


def plans(table, bounds, centroid_distance):
    n = len(table)
    out = {'full_sample': [('all', np.arange(n), np.array([], int))]}
    specs = {'primary_loco': (np.arange(n), bounds, PRIMARY_BUFFER_KM),
             'm49_subregion_lo': (cv.folds_from_labels(table.m49_subregion), None, None),
             'random10': (cv.folds_random(n, 10), None, None),
             'buffer_territory_1000km': (np.arange(n), bounds, SENSITIVITY_BORDER_BUFFER_KM),
             'buffer_centroid_1500km': (np.arange(n), centroid_distance, SENSITIVITY_CENTROID_BUFFER_KM)}
    for protocol, (ids, dist, buffer) in specs.items():
        fits = []
        for fold, test, train in cv.iter_folds(ids, dist, buffer):
            label = str(table.iso3.iloc[int(fold)]) if len(np.unique(ids)) == n else str(int(fold))
            fits.append((label, np.flatnonzero(train), np.flatnonzero(test)))
        out[protocol] = fits
    return out


def boundary_ties(dist, rows, candidates, exclude_self):
    """Rows whose eighth and ninth nearest candidates are at exactly equal distance."""
    ties = 0
    for i in rows:
        others = candidates[candidates != i] if exclude_self else candidates
        d = np.sort(dist[i, others])
        if len(d) > ms.K and d[ms.K - 1] == d[ms.K]:
            ties += 1
    return ties


def graph_record(arm, protocol, label, dist, train, test):
    record = {'weights': arm, 'protocol': protocol, 'fit': label, 'n_train': len(train), 'n_test': len(test)}
    try:
        w = ms.training_graph(dist, train)
    except ms.SpatialFitFailure as failure:
        return {**record, 'structural_pass': False, 'failure': failure.reason}
    record['training_boundary_ties'] = boundary_ties(dist, train, train, True)
    record['row_sums_exactly_one'] = bool((w.sum(axis=1) == 1.0).all())
    signs, finite = [], []
    for theta in ms.theta_grid():
        sign, logdet = np.linalg.slogdet(ms.spatial_filter(theta, w))
        signs.append(sign)
        finite.append(np.isfinite(logdet))
    record['logdet_sign_positive_on_grid'] = bool(all(s == 1.0 for s in signs))
    record['logdet_finite_on_grid'] = bool(all(finite))
    eig = np.linalg.eigvals(w)
    record['spectral_radius'] = float(np.max(np.abs(eig)))
    record['min_real_eigenvalue'] = float(np.min(eig.real))
    first, eighth = [], []
    for i in train:
        d = np.sort(dist[i, train[train != i]])
        first.append(d[0])
        eighth.append(d[ms.K - 1])
    record['training_first_neighbour_km_median'] = float(np.median(first))
    record['training_eighth_neighbour_km_median'] = float(np.median(eighth))
    record['training_eighth_neighbour_km_max'] = float(np.max(eighth))
    if len(test):
        neighbours, weights = ms.attachment(dist, train, test)
        record['attachment_boundary_ties'] = boundary_ties(dist, test, train, False)
        near = dist[test[:, None], neighbours]
        record['attachment_first_km_min'] = float(near[:, 0].min())
        record['attachment_eighth_km_max'] = float(near[:, -1].max())
        record['attachment_weights_sum_one'] = bool((weights.sum(axis=1) == 1.0).all())
    record['structural_pass'] = bool(record['row_sums_exactly_one'] and record['logdet_sign_positive_on_grid']
                                     and record['logdet_finite_on_grid'] and len(train) >= ms.MIN_TRAINING_NODES)
    record['failure'] = ''
    return record


def static_design(model, rows, encoder, state):
    matrix, names = feas.encode(rows, encoder)
    if model == 'M2':
        matrix = np.hstack([matrix, lb.added_block(state, rows.abs_latitude.to_numpy(float), rows.hemisphere.to_numpy())])
        names = [*names, *lb.ADDED_COLUMNS]
    return matrix, names


def synthetic_outcome(family, x, w, rng):
    beta = rng.normal(0.0, 0.01, x.shape[1])
    beta[0] = 0.18
    eps = rng.normal(0.0, SYNTHETIC_SIGMA, len(x))
    m = ms.spatial_filter(SYNTHETIC_THETA[family], w)
    if family == 'sem':
        return x @ beta + np.linalg.solve(m, eps)
    return np.linalg.solve(m, x @ beta + eps)


def synthetic_fits(frame, dist, protocol_plans):
    """Executability of the frozen estimator on real designs with a synthetic outcome (no warming data)."""
    rows = []
    rng = np.random.default_rng(SYNTHETIC_SEED)
    for model in ('M0star', 'M2'):
        full_state = lb.fit_state(frame.abs_latitude.to_numpy(float))
        x_full, _ = static_design(model, frame, feas.baseline_encoder(frame), full_state)
        w_full = ms.training_graph(dist, np.arange(len(frame)))
        for family in ms.FAMILIES:
            y = synthetic_outcome(family, x_full, w_full, rng)
            for protocol in ('full_sample', 'primary_loco', 'm49_subregion_lo', 'random10'):
                for label, train, test in protocol_plans[protocol]:
                    started = time.perf_counter()
                    record = {'model': model, 'family': family, 'protocol': protocol, 'fit': label,
                              'n_train': len(train)}
                    tr = frame.iloc[train]
                    encoder = feas.baseline_encoder(tr)
                    state = lb.fit_state(tr.abs_latitude.to_numpy(float)) if model == 'M2' else None
                    x_train, names = static_design(model, tr, encoder, state)
                    try:
                        w = ms.training_graph(dist, train)
                        fit = ms.fit_spatial(family, y[train], x_train, w, names)
                        ms.fitted_quantities(fit, y[train], x_train, w)
                        if len(test):
                            neighbours, weights = ms.attachment(dist, train, test)
                            x_test, _ = static_design(model, frame.iloc[test], encoder, state)
                            prediction = ms.predict_held_out(fit, y[train], x_train, x_test,
                                                             ms.local_positions(train, neighbours), weights)
                            perturbed = y.copy()
                            perturbed[test] += 1e3
                            again = ms.predict_held_out(fit, perturbed[train], x_train, x_test,
                                                        ms.local_positions(train, neighbours), weights)
                            record['held_out_invariance'] = bool(np.array_equal(prediction, again))
                        record.update({'status': 'computable', 'theta': fit.theta, 'true_theta': SYNTHETIC_THETA[family],
                                       'interior': True, 'n_grid_local_maxima': fit.n_grid_local_maxima,
                                       'n_evaluations': fit.n_evaluations, 'rank': int(np.linalg.matrix_rank(x_train)),
                                       'columns': x_train.shape[1], 'failure': ''})
                        if protocol == 'full_sample':
                            interval = ms.dependence_interval(fit, x_train, w)
                            record.update({'se': interval.se, 'wald_lower': interval.lower, 'wald_upper': interval.upper})
                    except ms.SpatialFitFailure as failure:
                        record.update({'status': 'non_computable', 'failure': failure.reason})
                    record['seconds'] = time.perf_counter() - started
                    rows.append(record)
    return rows


def synthetic_shrinkage(frame, dist, draws=40, theta=0.4):
    """Monte Carlo of the full-sample estimator on synthetic outcomes (intercept-only mean, true theta 0.4)
    for three designs, to document before any result how far the frozen designs pull theta-hat toward zero."""
    rng = np.random.default_rng(SYNTHETIC_SEED + 1)
    w = ms.training_graph(dist, np.arange(len(frame)))
    state = lb.fit_state(frame.abs_latitude.to_numpy(float))
    designs = {'intercept_only': np.ones((len(frame), 1)),
               'M0star': static_design('M0star', frame, feas.baseline_encoder(frame), None)[0],
               'M2': static_design('M2', frame, feas.baseline_encoder(frame), state)[0]}
    rows = []
    for name, x in designs.items():
        for family in ms.FAMILIES:
            estimates = []
            for _ in range(draws):
                eps = rng.normal(0.0, SYNTHETIC_SIGMA, len(frame))
                mean = np.full(len(frame), 0.18)
                m = ms.spatial_filter(theta, w)
                y = mean + np.linalg.solve(m, eps) if family == 'sem' else np.linalg.solve(m, mean + eps)
                estimates.append(ms.fit_spatial(family, y, x, w).theta)
            rows.append({'design': name, 'family': family, 'true_theta': theta, 'draws': draws,
                         'mean_theta_hat': float(np.mean(estimates)), 'sd_theta_hat': float(np.std(estimates, ddof=1))})
    return rows


def code_state():
    files = sorted({*prov.import_closure(CODE), *prov.LOCK_FILES})
    try:
        return {**prov.frozen_code_state(files), 'canonical_eligible': True}
    except prov.ExecutionRefused as refusal:
        return {'canonical_eligible': False, 'refusal': str(refusal),
                'code_closure_sha256': {p: prov.sha256(ROOT / p) for p in files}}


def run(out_dir=OUT):
    out_dir = Path(out_dir)
    canonical = out_dir.resolve() == OUT.resolve()
    prov.require_fresh(out_dir)
    code = code_state()
    if canonical and not code['canonical_eligible']:
        raise prov.ExecutionRefused(f"the canonical record needs committed, pushed code: {code['refusal']}")
    table, geometry, encoded = inputs()
    station = haversine_matrix(table.station_lon.to_numpy(), table.station_lat.to_numpy())
    land = haversine_matrix(geometry.centroid_lon.to_numpy(), geometry.centroid_lat.to_numpy())
    centroid_buffer = haversine_matrix(table.centroid_lon.to_numpy(), table.centroid_lat.to_numpy())
    identity = json.loads((ROOT / M1B_PROVENANCE).read_text())['identity']
    digests = {'station': prov.array_digest(knn_weights(station, 8)), 'land': prov.array_digest(knn_weights(land, 8))}
    if digests['station'] != identity['station_weights_sha256'] or digests['land'] != identity['area_weights_sha256']:
        raise ValueError('full-sample kNN8 weights differ from the frozen M1b identities')
    bounds = territory.load_bounds(table.iso3.tolist())
    protocol_plans = plans(table, bounds, centroid_buffer)
    folds = pd.read_csv(ROOT / FOLDS)
    for (label, train, _test), row in zip(protocol_plans['primary_loco'], folds.itertuples()):
        if label != row.iso3 or sorted(table.iso3.iloc[train]) != sorted(row.primary_training_iso3.split('|')):
            raise ValueError(f'primary membership of {row.iso3} differs from the frozen fold artifact')
    graphs = [graph_record(arm, protocol, label, dist, train, test)
              for arm, dist in (('station', station), ('land', land))
              for protocol, fits in protocol_plans.items() for label, train, test in fits]
    frame = frame_for(table, encoded, 'primary_total_co2')
    started = time.perf_counter()
    synthetic = synthetic_fits(frame, station, protocol_plans)
    synthetic_seconds = time.perf_counter() - started
    shrinkage = synthetic_shrinkage(frame, station)
    graph_frame, fit_frame = pd.DataFrame(graphs), pd.DataFrame(synthetic)
    summary = {
        'audit': 'M3 structural feasibility: graphs and the frozen estimator on synthetic outcomes; no warming outcome read',
        'specification': 'research/model_v2/DOWNSTREAM_COMPLETION_SPEC.md', 'estimator_version': ms.ESTIMATOR_VERSION,
        'canonical': canonical, 'code': code, 'inputs': {p: prov.sha256(ROOT / p) for p in PINS},
        'full_sample_weight_digests_match_m1b_identity': True,
        'graphs': {
            'fits': len(graph_frame), 'all_structural_pass': bool(graph_frame.structural_pass.all()),
            'min_n_train': int(graph_frame.n_train.min()),
            'by_weights_protocol': {f'{a}/{p}': {
                'fits': int(len(g)), 'n_train_min': int(g.n_train.min()), 'n_train_max': int(g.n_train.max()),
                'training_boundary_ties_total': int(g.training_boundary_ties.sum()),
                'attachment_boundary_ties_total': int(g.attachment_boundary_ties.fillna(0).sum())
                if 'attachment_boundary_ties' in g else 0,
                'attachment_first_km_min': float(g.attachment_first_km_min.min()) if g.n_test.max() > 0 else None,
                'attachment_eighth_km_max': float(g.attachment_eighth_km_max.max()) if g.n_test.max() > 0 else None,
                'spectral_radius_max': float(g.spectral_radius.max()),
                'min_real_eigenvalue_min': float(g.min_real_eigenvalue.min()),
                'logdet_sign_positive_all': bool(g.logdet_sign_positive_on_grid.all())}
                for (a, p), g in graph_frame.groupby(['weights', 'protocol'], sort=False)},
        },
        'synthetic_estimator': {
            'outcome': f'synthetic draws from each family with theta={SYNTHETIC_THETA}, sigma={SYNTHETIC_SIGMA}, '
                       f'seed {SYNTHETIC_SEED}; real encoded designs and station graphs; no warming outcome',
            'fits': len(fit_frame), 'non_computable': int((fit_frame.status != 'computable').sum()),
            'failures': fit_frame.loc[fit_frame.status != 'computable', ['model', 'family', 'protocol', 'fit', 'failure']]
            .to_dict('records'),
            'held_out_invariance_all': bool(fit_frame.held_out_invariance.dropna().astype(bool).all()),
            'max_grid_local_maxima': int(fit_frame.n_grid_local_maxima.max()),
            'full_sample': fit_frame[fit_frame.protocol == 'full_sample'][
                ['model', 'family', 'theta', 'true_theta', 'se', 'wald_lower', 'wald_upper', 'n_grid_local_maxima']]
            .to_dict('records'),
            'seconds_total': synthetic_seconds, 'seconds_per_fit_median': float(fit_frame.seconds.median()),
        },
        'synthetic_shrinkage_monte_carlo': {
            'rows': shrinkage,
            'reading': 'synthetic draws with an intercept-only mean and true theta 0.4 on the full-sample station graph; '
                       'the regional and climate dummies of the frozen static designs absorb spatially smooth error, '
                       'so ML theta-hat is pulled toward zero. This is recorded before any M3 result to inform how '
                       'dependence parameters are read; it changes no rule.'},
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'platform': platform.platform(), 'machine': platform.machine()},
        'numerical_note': 'eigenvalues, log-determinants, timings and synthetic estimates are platform-scoped in their '
                          'last bits; the structural evidence is sizes, ties, signs and failure counts',
    }
    out_dir.mkdir(parents=True)
    prov.write_csv(graph_frame, out_dir / RECORD_FILES[0])
    prov.write_csv(fit_frame.drop(columns=['seconds']), out_dir / RECORD_FILES[1])
    prov.write_json(summary, out_dir / RECORD_FILES[2])
    prov.write_json({name: prov.sha256(out_dir / name) for name in RECORD_FILES}, out_dir / 'm3_feasibility_manifest.json')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='M3 structural feasibility audit (no outcome).')
    parser.add_argument('--out', default=str(OUT))
    summary = run(parser.parse_args(argv).out)
    print(prov.json_text({'graphs_all_pass': summary['graphs']['all_structural_pass'],
                          'synthetic_non_computable': summary['synthetic_estimator']['non_computable'],
                          'canonical': summary['canonical']}))
    return summary


if __name__ == '__main__':
    result = main()
    sys.exit(0 if result['graphs']['all_structural_pass'] else 5)
