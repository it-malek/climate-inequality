"""Run fixed spatial validation against an explicitly selected private run root.

The run root must supply its frozen VALIDATION_SPEC.json, verified preprocessing
checkpoint, control/runlib.py stage recorder, and the recorded interpolation
parameter-handoff resolution. Evidence stays under that selected run root.
"""
import argparse
import json
import importlib.util
import resource
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import spatial_validation as sv  # noqa: E402 - support direct script execution
from src.explain import CITY_MODEL_SPECS, load_income_groups  # noqa: E402 - support direct script execution
from src.feature_schema import SCHEMA_V1, STATUS_AVAILABLE  # noqa: E402 - support direct script execution
from src.stability import _full_model_residuals, country_centroids, outcome_designs  # noqa: E402 - support direct script execution



def fold_action(d, names):
    cities, folds = sv.interpolation_folds(pd.read_parquet(PRE / 'city_trends.parquet'))
    cities.assign(**folds).to_parquet(d / 'fold_assignment.parquet', index=False)
    records, scores = [], []
    for name, fold in names:
        start = time.monotonic()
        train, test = sv.interpolation_masks(cities, folds[name], fold, 250 if 'spatial' in name else 0)
        record = {'protocol': name, 'fold': fold, 'n_train': int(train.sum()), 'n_test': int(test.sum()), 'train_coordinate_groups': len(cities.loc[train, sv.COORDINATES].drop_duplicates())}
        prediction = cities.loc[test].copy()
        prediction['protocol'], prediction['fold'] = name, fold
        try:
            values, parameters, variance = sv.predict_interpolation_fold(cities, train, test)
            prediction = prediction.assign(**values, kriging_variance=variance)
            record.update(status='completed', variogram_parameters=parameters, nonfinite_predictions={key: int((~np.isfinite(v)).sum()) for key, v in values.items()})
            scores.append({'protocol': name, 'fold': fold, 'scores': sv.interpolation_scores(prediction)})
        except Exception as exc:
            record.update(status='failed', error=repr(exc))
            prediction = prediction.assign(idw=np.nan, kriging=np.nan, kriging_variance=np.nan)
        record.update(seconds=time.monotonic()-start, process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        records.append(record)
        prediction.to_parquet(d / f'{name}-{fold}.parquet', index=False)
        dump(d / 'fold_records.json', records)
        print('FOLD', record, flush=True)
    dump(d / 'fold_scores.json', scores)
    return {'fold_records': records, 'failures': sum(r['status'] == 'failed' for r in records)}


def pilot(d):
    result = fold_action(d, [('coordinate_group_5fold', 0)])
    result['projected_full_seconds'] = result['fold_records'][0]['seconds'] * 15
    dump(d / 'resource_pilot.json', result)
    return result


def full(d):
    pilot_record = json.loads((RUN / 'interpolation-pilot/resource_pilot.json').read_text())
    if pilot_record['projected_full_seconds'] > 4*3600 or pilot_record['fold_records'][0]['process_peak_rss_bytes'] > 8*1024**3:
        raise RuntimeError('Frozen resource threshold exceeded; equivalent-computation review required')
    result = fold_action(d, [('coordinate_group_5fold', f) for f in range(1, 5)] + [('spatial_region_10fold_250km', f) for f in range(10)])
    files = [RUN / 'interpolation-pilot/coordinate_group_5fold-0.parquet', *sorted(d.glob('*fold-*.parquet')), *sorted(d.glob('*250km-*.parquet'))]
    predictions = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    predictions.to_parquet(d / 'all_predictions.parquet', index=False)
    summary = {name: sv.interpolation_scores(part) for name, part in predictions.groupby('protocol')}
    paired = predictions[sv.COORDINATES + ['protocol', 'fold']].copy()
    for loss, transform in [('squared', lambda x: x**2), ('absolute', abs)]:
        paired[loss + '_idw_minus_kriging'] = transform(predictions.idw-predictions.slope_c_per_decade) - transform(predictions.kriging-predictions.slope_c_per_decade)
    paired.groupby(['protocol', 'fold', *sv.COORDINATES]).mean().reset_index().to_parquet(d / 'paired_group_losses.parquet', index=False)
    dump(d / 'summary.json', summary)
    return {**result, 'summary': summary, 'pilot_reused': str(files[0])}


def source_lineage(d, features):
    # Full raw key/month duplicates distinguish repeated monthly observations from duplicate records.
    con = duckdb.connect()
    con.execute('SET threads=1')
    con.execute("SET memory_limit='1GB'")
    con.execute("SET temp_directory='" + str(d / 'duckdb-temp').replace("'", "''") + "'")
    path = str(RAW_CITY).replace("'", "''")
    con.execute(f"CREATE VIEW raw_city AS SELECT * FROM read_csv_auto('{path}')")
    n, distinct = con.execute('SELECT count(*), count(DISTINCT (dt,City,Country,Latitude,Longitude)) FROM raw_city').fetchone()
    locations = con.execute('SELECT DISTINCT City,Country,Latitude,Longitude FROM raw_city').df()
    monthly = con.execute("SELECT count(*), count(*) FILTER (WHERE lo != hi) FROM (SELECT dt,Latitude,Longitude,min(AverageTemperature) AS lo,max(AverageTemperature) AS hi FROM raw_city GROUP BY dt,Latitude,Longitude HAVING count(DISTINCT (Country,City)) > 1)").fetchone()
    con.close()
    for col, negative in [('Latitude', 'S'), ('Longitude', 'W')]:
        locations[col + '_source'] = locations[col]
        locations[col] = locations[col].str[:-1].astype(float) * np.where(locations[col].str.endswith(negative), -1, 1)
    locations.to_parquet(d / 'raw_source_locations.parquet', index=False)
    matched = features[sv.IDENTITY].merge(locations, on=sv.IDENTITY, how='left', indicator=True)
    matched.to_parquet(d / 'source_lineage.parquet', index=False)
    counts = features.groupby(sv.COORDINATES).size()
    repeated = features.merge(counts.rename('coordinate_group_size'), on=sv.COORDINATES)
    repeated[repeated.coordinate_group_size > 1].to_parquet(d / 'colocated_city_records.parquet', index=False)
    lon, lat = features.Longitude.to_numpy(), features.Latitude.to_numpy()
    xyz = sv._to_unit_sphere(lon, lat)
    distances, _ = sv.cKDTree(xyz).query(xyz, k=10)
    tie_rows = int((abs(distances[:, 8] - distances[:, 9]) <= 1e-12).sum())
    return {'raw_monthly_rows': n, 'duplicate_source_identity_month_rows': n-distinct, 'shared_coordinate_month_groups': monthly[0], 'shared_coordinate_month_groups_with_differing_temperatures': monthly[1], 'raw_unique_locations': len(locations), 'raw_duplicate_full_location_keys': int(locations.duplicated(sv.IDENTITY).sum()), 'feature_locations': len(features), 'unmatched_source_locations': int((matched['_merge'] != 'both').sum()), 'shared_coordinate_groups': int((counts > 1).sum()), 'city_records_at_shared_coordinates': int(counts[counts > 1].sum()), 'rows_with_k8_boundary_ties_including_self_query': tie_rows, 'max_coordinate_fractional_digits': {col: int(locations[col+'_source'].str[:-1].str.split('.').str[-1].str.len().max()) for col in sv.COORDINATES}, 'interpretation': 'Source location strings are rounded coordinate/grid representations; multiple distinct city identities share exact signed source-coordinate pairs. Repeated months are expected time-series rows; duplicate identity-month rows are separately counted. Source strings alone do not establish original unrounded station positions or the upstream gridding algorithm.'}


def neighbor(d):
    features = pd.read_parquet(PRE / 'city_features.parquet')
    lineage = source_lineage(d, features)
    dump(d / 'source_lineage.json', lineage)
    samples = {}
    for name, formula in CITY_MODEL_SPECS.items():
        fit = smf.ols(formula, data=features, missing='drop').fit()
        sample = features.loc[fit.model.data.row_labels].reset_index(drop=True)
        samples['city_' + name] = (sample, fit.resid.to_numpy())
    centroids = country_centroids(features).set_index('Country')
    for name, design in outcome_designs(pd.read_parquet(PRE/'country_inequality.parquet'), features, load_income_groups(INCOME)).items():
        countries, residuals = _full_model_residuals(design, SCHEMA_V1, STATUS_AVAILABLE)
        sample = centroids.loc[countries].reset_index().assign(City='country-centroid')
        samples['country_' + name] = (sample, residuals)
    summaries = {}
    for name, (sample, residuals) in samples.items():
        sample.assign(residual=residuals).to_parquet(d / (name + '-residuals.parquet'), index=False)
        records, graphs, acceptance = sv.neighbor_order_validation(sample, residuals)
        records.to_csv(d / (name + '-orders.csv'), index=False)
        np.savez_compressed(d / (name + '-graphs.npz'), **graphs)
        summaries[name] = {'n': len(sample), 'acceptance': acceptance, 'ranges': []}
        for (policy, randomization), part in records.groupby(['policy', 'randomization']):
            summaries[name]['ranges'].append({'policy': policy, 'randomization': randomization, 'I_min': part.I.min(), 'I_max': part.I.max(), 'p_min': part.p.min(), 'p_max': part.p.max(), 'alpha_decisions': sorted(part.reject_alpha_005.unique().tolist()), 'max_changed_rows': int(part.changed_rows.max()), 'min_mean_overlap': part.mean_neighbor_overlap.min()})
        print('NEIGHBOR', name, acceptance, flush=True)
    dump(d / 'summary.json', summaries)
    return {'samples': summaries, 'source_lineage': lineage, 'accepted': all(s['acceptance']['accepted'] for s in summaries.values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pilot', 'full', 'neighbor'])
    parser.add_argument('--run-root', type=Path, required=True, help='Explicit private validation run directory with frozen specification and control/runlib.py')
    parser.add_argument('--raw-root', type=Path, required=True, help='Verified read-only raw source directory')
    args = parser.parse_args()
    RUN = args.run_root.resolve(strict=True)
    RAW = args.raw_root.resolve(strict=True)
    spec = importlib.util.spec_from_file_location('spatial_validation_runlib', RUN / 'control/runlib.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load the selected run stage recorder')
    runlib = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runlib
    spec.loader.exec_module(runlib)
    # Resolve all scientific inputs and evidence destinations from explicit roots.
    runlib.ROOT, runlib.RUN, runlib.RAW = ROOT, RUN, RAW
    dump, stage = runlib.dump, runlib.stage
    PRE = RUN / 'preprocessing'
    RAW_CITY = RAW / 'climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv'
    INCOME = RAW / 'worldbank/world-bank-income-groups.csv'
    sources = [ROOT/'src/spatial_validation.py', ROOT/'research/model_v2/cv.py', ROOT/'src/interpolate.py', ROOT/'src/explain.py', ROOT/'src/stability.py', ROOT/'src/decomposition.py', ROOT/'src/feature_schema.py']
    if args.mode == 'neighbor':
        stage('neighbor-order', [PRE/'city_features.parquet', PRE/'country_inequality.parquet', INCOME, RAW_CITY], neighbor, runner=__file__, dependencies=['preprocessing'], sources=sources)
    elif args.mode == 'pilot':
        stage('interpolation-pilot', [PRE/'city_trends.parquet', RUN/'control/interpolation-implementation-resolution.json'], pilot, runner=__file__, dependencies=['preprocessing'], sources=sources)
    else:
        stage('interpolation-full', [PRE/'city_trends.parquet', RUN/'interpolation-pilot/completed.json', RUN/'control/interpolation-implementation-resolution.json'], full, runner=__file__, dependencies=['preprocessing', 'interpolation-pilot'], sources=sources)
