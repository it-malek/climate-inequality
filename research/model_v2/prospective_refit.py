"""Prospective V2 refits from fresh inputs, using fixed numerical kernels.

Saved geometry and fixed region definitions may be reused; saved predictions,
encoded predictor tables and scores never enter the reconstructed designs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from research.model_v2 import m0, m1a_evaluate as a, m1a_geography as geo
from research.model_v2 import m1b_evaluate as b, m1b_hydroclimate as hydro
from research.model_v2 import m2_evaluate as ev, m2_feasibility as feas
from research.model_v2 import m3_evaluate as m3, m4_evaluate as m4, territory, geometry
from research.model_v2 import prospective_gates as gates, m1b_redundancy as red
from research.model_v2.regions import m49_table
from research.model_v2.spatial import haversine_matrix, knn_weights
from src.area_weighting import load_national_id_lookup
from src.decomposition import OUTCOME_COL
from src.explain import load_income_groups
from src.stability import country_centroids

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    def default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=default, allow_nan=False) + '\n')


@contextmanager
def routed(module, **values):
    """Explicit input routing in this single-process runner only."""
    old = {key: getattr(module, key) for key in values}
    try:
        for key, value in values.items():
            setattr(module, key, value)
        yield
    finally:
        for key, value in old.items():
            setattr(module, key, value)


def context(run, raw):
    run, raw = Path(run), Path(raw)
    inequality = pd.read_parquet(run / 'preprocessing/country_inequality.parquet')
    city = pd.read_parquet(run / 'preprocessing/city_features.parquet')
    income = load_income_groups(raw / m0.INCOME_PATH.relative_to(ROOT / 'data/raw'))
    complete = m0.m0_complete_design(inequality, city, income).reset_index(drop=True)
    table = complete.copy()
    owid = raw / m0.OWID_CO2_PATH.relative_to(ROOT / 'data/raw')
    table['iso3'] = table.Country.map(m0.iso3_bridge(inequality, owid))
    if table.iso3.isna().any() or not table.iso3.is_unique:
        raise ValueError('Fresh ISO3 mapping incomplete or ambiguous')
    table = table.merge(m49_table(), on='iso3', how='left', validate='one_to_one')
    if table.m49_subregion.isna().any():
        raise ValueError('Missing fixed M49 label')
    station = country_centroids(city).set_index('Country').loc[table.Country]
    table['station_lon'], table['station_lat'] = station.Longitude.to_numpy(), station.Latitude.to_numpy()
    geometry = pd.read_csv(ROOT / ev.GEOMETRY, usecols=['iso3', 'centroid_lon', 'centroid_lat']).set_index('iso3').loc[table.iso3]
    table['centroid_lon'], table['centroid_lat'] = geometry.centroid_lon.to_numpy(), geometry.centroid_lat.to_numpy()
    distance = territory.load_bounds(table.iso3.tolist())
    frames = {r: feas.transform_numerics(feas.representation_frame(table, r)) for r in ev.REPRESENTATIONS}
    distances = {'station': haversine_matrix(table.station_lon.to_numpy(), table.station_lat.to_numpy()),
                 'land': haversine_matrix(table.centroid_lon.to_numpy(), table.centroid_lat.to_numpy())}
    sw, aw = knn_weights(distances['station'], 8), knn_weights(distances['land'], 8)
    identity = {'source': 'fresh pipeline complete-case design', 'n': len(table),
                'geometry_sha256': digest(ROOT / ev.GEOMETRY),
                'fixed_territory_lower_bounds_sha256': digest(territory.LOWER_PATH),
                'fixed_territory_footprints_sha256': digest(territory.SNAPSHOT_PATH),
                'fixed_geometry_reuse': 'Outcome-free GPW analytical-unit geometry and territorial clearance; fixed fold definitions reconstructed',
                'raw_design_sources_sha256': {str(p): digest(p) for p in [owid, raw / m0.INCOME_PATH.relative_to(ROOT / 'data/raw')]},
                'station_weights_sha256': ev.prov.array_digest(sw), 'area_weights_sha256': ev.prov.array_digest(aw),
                'territory_distance_sha256': ev.prov.array_digest(distance)}
    folds = ev.fold_plan(table, len(table))
    identity['fold_id_sha256'] = {p: ev.prov.array_digest(ids) for p, (ids, _) in folds.items()}
    identity.update(gates.context_identity(complete, table, frames, distance, folds))
    frozen = ev.Frozen(table, frames, table[OUTCOME_COL].to_numpy(float), distance, sw, aw,
                       folds, pd.DataFrame(), identity)
    return complete, frozen, (inequality, city, income), distances


_ORIGINAL_GSHHG = geo.load_gshhg
_ORIGINAL_GRID = geo.load_national_grid


def national_grid(path):
    lookup = Path(path).parent / geo.GPW_NATID_LOOKUP_PATH.name
    with routed(geometry, load_national_id_lookup=lambda: load_national_id_lookup(lookup)):
        return _ORIGINAL_GRID(path)


def fresh_gpw(codes, path):
    with routed(geo, load_national_grid=lambda: national_grid(path)):
        return geo.gpw_country_grid(codes)


def measurement(run, raw, *, out_dir=None):
    run, raw = Path(run), Path(raw)
    out = Path(out_dir) if out_dir is not None else run / 'v2_measurement'
    if out_dir is None:
        out.mkdir(exist_ok=False)
    _, frozen, inputs, _ = context(run, raw)
    table_path = out / 'fresh_countries.csv'
    frozen.table.to_csv(table_path, index=False)
    def raw_path(path):
        return raw / Path(path).relative_to(ROOT / 'data/raw')
    gpw = raw_path(geo.GPW_PATH)
    raw_gates = gates.raw_measurement_sources(raw, run / 'inputs-koppen/koppen_present_0p5.nc')
    write_json(out / 'm1a_source_gates.json', raw_gates)
    source_paths = [gpw, raw_path(hydro.GPW_NATID_LOOKUP_PATH), raw_path(geo.ETOPO_PATH),
                    raw_path(geo.LAND_ZIP_PATH), run / 'inputs-koppen/koppen_present_0p5.nc',
                    raw / 'gshhg/gshhg-shp-2.3.7.zip']
    source_paths += sorted((raw / 'gshhg/GSHHS_shp/f').glob('GSHHS_f_L[1-4].*'))
    for spec in hydro.FILES.values():
        compressed = raw / 'cru_ts_4.10' / Path(spec['path']).name
        source_paths.extend([compressed, compressed.with_suffix('')])
    write_json(out / 'raw_inputs_pre_execution.json', {'paths_and_sha256': {str(p): digest(p) for p in source_paths},
                'routing': 'Explicit paths; unchanged numerical kernels; restored module routing after execution'})
    def manifest(stats, repaired, coast, attrs, passed, qa, code_hash):
        return {'prospective': True, 'all_features_pass_gates': passed,
                'gate_failures': qa.loc[qa.missing_reasons != '{}', ['iso3', 'missing_reasons']].to_dict('records'),
                'quadrature_failures': qa.loc[~qa.continentality_quadrature_ok, 'iso3'].tolist(),
                'walk_stats': stats, 'repaired_polygons': repaired, 'etopo_attrs': attrs,
                'kernel_sha256': code_hash, 'sources': raw_gates}
    def stations(coast, features):
        city = inputs[1][inputs[1].Country.isin(features.Country)].copy()
        city['coast_km_exact'] = coast.distance_km(city.Longitude.to_numpy(float), city.Latitude.to_numpy(float))
        return city.groupby('Country', observed=True).agg(continentality_station_v1=('coast_km', 'mean'),
                                continentality_station_exact_arc=('coast_km_exact', 'mean'))
    with routed(m0, load_inputs=lambda: inputs), routed(geo,
            OUTPUT_DIR_TABLE=table_path, ETOPO_PATH=raw_path(geo.ETOPO_PATH),
            KOPPEN_PATH=run / 'inputs-koppen/koppen_present_0p5.nc', LAND_ZIP_PATH=raw_path(geo.LAND_ZIP_PATH),
            load_gshhg=lambda: _ORIGINAL_GSHHG(raw / 'gshhg'),
            load_national_grid=lambda: national_grid(gpw), station_continentality=stations,
            manifest_payload=manifest):
        _, _, record = geo.build(out)
    if not record['all_features_pass_gates']:
        raise ValueError('Fresh M1a measurement gates failed; evidence preserved')
    _, support_gates = gates.m1a_support(out, frozen.table, frozen.table)
    write_json(out / 'm1a_support_gates.json', support_gates)
    pins = {out / 'm1a_cell_land_area_km2.npz': digest(out / 'm1a_cell_land_area_km2.npz'),
            out / 'm1a_geography_qa.csv': digest(out / 'm1a_geography_qa.csv'),
            gpw: hydro.FROZEN_INPUT_SHA256[hydro.GPW_PATH],
            raw_path(hydro.GPW_NATID_LOOKUP_PATH): hydro.FROZEN_INPUT_SHA256[hydro.GPW_NATID_LOOKUP_PATH]}
    with routed(m0, ROOT=Path(os.path.commonpath([raw, out]))), routed(hydro,
            CRU_DIR=raw / 'cru_ts_4.10', GPW_PATH=gpw, CONTRACT_PATH=gates.HYDRO_CONTRACT,
            GPW_NATID_LOOKUP_PATH=raw_path(hydro.GPW_NATID_LOOKUP_PATH),
            LAND_AREA_NPZ=out / 'm1a_cell_land_area_km2.npz', M1A_QA=out / 'm1a_geography_qa.csv',
            FROZEN_INPUT_SHA256=pins, gpw_country_grid=lambda codes: fresh_gpw(codes, gpw)):
        _, hrecord = hydro.build(out)
    hrecord['prospective'] = True
    hrecord['support_inputs']['historical_land_rule_reference_commit'] = hrecord['support_inputs'].pop('land_support_commit')
    hrecord['support_inputs']['land_support_pins_asserted'] = False
    hrecord['support_inputs']['prospective_fresh_support_hashes_verified'] = True
    hrecord['execution_receipt'] = {'started_sha256': digest(out / 'started.json'),
        'historical_commit_stamp_replaced': True,
        'identity_basis': 'Actual archived current-run code and preexecution input hashes; not the historical build commit'}
    write_json(out / hydro.MANIFEST, hrecord)
    if not hrecord['all_hard_stops_pass']:
        raise ValueError('Fresh M1b measurement gates failed; evidence preserved')
    _, package_gates = gates.m1b_package(out)
    write_json(out / 'm1b_package_gates.json', package_gates)


def add_predictions(rows, frozen, scored, representation, model, **extra):
    for protocol, values in [('in_sample', scored.fitted), *[(p, r.yhat) for p, r in scored.results.items()]]:
        rows.extend({**extra, 'representation': representation, 'model': model, 'protocol': protocol,
                     'iso3': code, 'observed': float(y), 'prediction': float(pred)}
                    for code, y, pred in zip(frozen.table.iso3, frozen.y, values))


def area_design(complete, features, representation):
    """Original raw M1a design; model metadata stays outside the schema."""
    gates.require(features.Country.tolist() == complete.Country.tolist(), 'M1a feature country order mismatch')
    frame = feas.representation_frame(complete, representation)
    for column in a.REMEASURED:
        frame[column] = features[column].to_numpy()
    b.fs.validate_design_matrix(frame.columns, b.SCHEMA_M1B, allow=(OUTCOME_COL, 'Country'))
    return frame


def static(run, raw, *, out_dir=None, measurement_dir=None):
    run, raw = Path(run), Path(raw)
    out = Path(out_dir) if out_dir is not None else run / 'v2_static'
    if out_dir is None:
        out.mkdir(exist_ok=False)
    measurement_dir = Path(measurement_dir) if measurement_dir is not None else run / 'v2_measurement'
    # Package and predictor-only checks precede opening any target columns here.
    c2_features, package_gates = gates.m1b_package(measurement_dir)
    with routed(red, DEFAULT_INEQUALITY_PATH=run / 'preprocessing/country_inequality.parquet',
                DEFAULT_FEATURES_PATH=run / 'preprocessing/city_features.parquet',
                INCOME_PATH=raw / m0.INCOME_PATH.relative_to(ROOT / 'data/raw')):
        predictors = red.m0star_predictors(c2_features.Country)
    redundancy = gates.redundancy(predictors, c2_features, package_gates)
    write_json(out / red.RECORD, red.json_safe(redundancy))
    complete, frozen, _, _ = context(run, raw)
    fresh_predictors = complete.drop(columns=[OUTCOME_COL, 'cum_co2_per_capita'])
    gates.require(red.frame_sha256(predictors) == red.frame_sha256(fresh_predictors), 'Scored predictor values differ from predictor-only redundancy inputs')
    features, support_gates = gates.m1a_support(measurement_dir, frozen.table, complete)
    c2 = c2_features[b.FEATURE].to_numpy(float)
    gates.require(c2_features.iso3.tolist() == frozen.table.iso3.tolist(), 'C2 scored identity/order mismatch')
    features = features.set_index('iso3')
    frozen.table.to_csv(out / 'fresh_country_design.csv', index=False)
    write_json(out / 'input_identity.json', frozen.identity)
    pd.DataFrame({'iso3': frozen.table.iso3, **{p: ids for p, (ids, _) in frozen.folds.items()}}).to_csv(out / 'fold_ids.csv', index=False)
    bf = b.Frozen(complete, frozen.table, c2, frozen.distance, frozen.station_w,
                  frozen.area_w, frozen.table[['centroid_lon', 'centroid_lat']].to_numpy(), frozen.folds, frozen.identity)
    feasibility = ev.check_against_feasibility(frozen)
    b_arms, integrity = b.score_representations(bf, reference_cards=None)
    gates.require(not integrity, f'M1b scoring integrity failures: {integrity}')
    for arm in b_arms.values():
        gates.scored_integrity(arm['scored'][b.COMPARATOR], expected_columns=20)
        gates.scored_integrity(arm['scored'][b.CANDIDATE], expected_columns=21, c2=True)
    b_equivalence = gates.equivalent(b_arms)
    cards, rows, m2_arms, classifications = {}, [], {}, {}
    for representation in ev.REPRESENTATIONS:
        frame = frozen.frames[representation]
        fitted_models = {}
        for model in (ev.COMPARATOR, ev.CANDIDATE):
            scored = ev.score_model(model, frame, frozen)
            gates.m2_integrity(scored)
            fitted_models[model] = scored
            cards[f'{representation}/{model}'] = scored.card
            add_predictions(rows, frozen, scored, representation, model)
        baseline = fitted_models[ev.COMPARATOR]
        b_baseline = b_arms[representation]['scored'][b.COMPARATOR]
        for protocol in ev.PROTOCOLS:
            gates.require(np.max(np.abs(baseline.results[protocol].yhat - b_baseline.results[protocol].yhat)) <= b.PREDICTION_ATOL,
                          'Fresh raw/logged comparator implementations disagree')
        area_frame = area_design(complete, features, representation)
        area_scored = b.score_model(area_frame, bf)
        gates.scored_integrity(area_scored)
        fitted_models['M1a'] = area_scored
        fitted_models['M1b'] = b_arms[representation]['scored'][b.CANDIDATE]
        for model in ['M1a', 'M1b']:
            current = fitted_models[model]
            cards[f'{representation}/{model}'] = current.card
            add_predictions(rows, frozen, current, representation, model)
        for model in ['M1a', 'M1b', ev.CANDIDATE]:
            current = fitted_models[model]
            paired = ev.paired_rmse_interval(frozen.y, current.results[ev.PRIMARY].yhat, baseline.results[ev.PRIMARY].yhat)
            current.card['paired_vs_M0star'] = paired
            current.card['decides_verdict'] = representation == 'primary_total_co2'
            if model == 'M1a':
                current.card['paired_delta_vs_m0'] = paired
                errors = pd.DataFrame({'C0_abs_error': abs(frozen.y - baseline.results[ev.PRIMARY].yhat),
                    'M1a_abs_error': abs(frozen.y - current.results[ev.PRIMARY].yhat)}, index=frozen.table.iso3)
                classified = a.classify(baseline.card, current.card, errors)
                classifications[representation] = classified
                current.card['registered_classification'] = classified
            elif model == 'M1b':
                decision = b_arms[representation]['verdict']
                current.card['registered_decision' if representation == 'primary_total_co2' else 'representation_conditions'] = (
                    decision if representation == 'primary_total_co2' else b.non_deciding_conditions(decision))
            else:
                decision = ev.verdict(baseline.card, current.card, paired, True)
                current.card['registered_decision' if representation == 'primary_total_co2' else 'representation_conditions'] = (
                    decision if representation == 'primary_total_co2' else ev.non_deciding_conditions(decision))
                m2_arms[representation] = {'scored': {ev.COMPARATOR: baseline, ev.CANDIDATE: current},
                    'cards': {ev.COMPARATOR: baseline.card, ev.CANDIDATE: current.card}, 'paired': paired}
    m2_equivalence = ev.equivalence(m2_arms)
    gates.require(m2_equivalence['passes'] is True, 'M0/M2 original representation equivalence failed')
    fit = m0.fit_m0(complete)
    write_json(out / 'm0_full_fit.json', {'r2': fit.r2, 'n': len(complete)})
    write_json(out / 'scorecards.json', cards)
    write_json(out / 'm1a_classification.json', classifications)
    write_json(out / 'scientific_gates.json', {'all_passed': True, 'm1a_support': support_gates,
        'm1b_package': package_gates, 'm1b_representation_equivalence': b_equivalence,
        'm0_m2_representation_equivalence': m2_equivalence,
        'fresh_recomputed_states_vs_historical_predictor_only_feasibility': feasibility,
        'primary_representation': 'primary_total_co2', 'per_capita_decides_verdict': False})
    pd.DataFrame(rows).to_csv(out / 'predictions.csv', index=False)
    return cards


def spatial(run, raw, *, pilot=False, out_dir=None):
    run = Path(run)
    out = Path(out_dir) if out_dir is not None else run / ('v2_spatial_pilot' if pilot else 'v2_spatial')
    if out_dir is None:
        out.mkdir(exist_ok=False)
    _, frozen, _, distances = context(run, raw)
    cards = {}
    for weights, dist in distances.items():
        runs = {}
        for representation in ev.REPRESENTATIONS:
            frame = frozen.frames[representation]
            reference = None if pilot else ev.score_model(ev.COMPARATOR, frame, frozen)
            for family in m3.FAMILIES:
                started = time.perf_counter()
                key = f'{weights}/{representation}/{family}'
                if pilot:
                    fits = []
                    for fold, test, train in list(ev.cv.iter_folds(frozen.folds[ev.PRIMARY][0], frozen.distance, 500))[:3]:
                        value = m3.fit_one(family, ev.COMPARATOR, frame, dist, np.flatnonzero(train), np.flatnonzero(test))
                        fits.append({'fold': int(fold), 'theta': value['fit'].theta})
                    cards[key] = {'fits': fits, 'wall_seconds': time.perf_counter() - started}
                else:
                    result = m3.run_family(family, ev.COMPARATOR, frame, frozen, dist, representation)
                    entry = {'status': result.status, 'failure': result.failure, 'wall_seconds': time.perf_counter() - started}
                    if result.status == 'computable':
                        entry['theta'] = result.full.theta
                        entry['card'] = m3.family_card(result, frame, frozen, 'M0*', frozen.station_w, frozen.area_w, dist)
                        entry['paired'] = m3.paired_block(frozen.y, result, reference)
                        entry['qualification'] = m3.qualification(entry['card'], reference.card, entry['paired']['vs_static'], True)
                        entry['accounting'] = m3.accounting_block(result, frame, reference)
                    cards[key] = entry
                    runs[(representation, family)] = result
                write_json(out / 'scorecards.json', cards)
        if not pilot:
            m3.prediction_rows(runs, frozen, weights).to_csv(out / f'{weights}_predictions.csv', index=False)
            m3.fit_rows(runs, frozen, weights).to_csv(out / f'{weights}_fits.csv', index=False)
    return cards


def final_checks(run, raw, *, out_dir=None):
    """Existing retained M0*, qualifying SEM/SAR only; no new model selection."""
    run = Path(run)
    out = Path(out_dir) if out_dir is not None else run / 'v2_final_checks'
    if out_dir is None:
        out.mkdir(exist_ok=False)
    _, frozen, _, distances = context(run, raw)
    # Fixed historical retained set. New qualification is reported separately;
    # it does not authorize changing the model selection history.
    qualifying = ('sem', 'sar')
    sensitivity, rows = {}, []
    for protocol, buffer in m4.SENSITIVITY_PROTOCOLS.items():
        distance = frozen.distance if 'territory' in protocol else distances['land']
        sf = replace(frozen, distance=distance, folds={protocol: (np.arange(frozen.n), buffer)})
        for representation, frame in frozen.frames.items():
            baseline_result, _ = ev.run_protocol(ev.COMPARATOR, frame, sf, protocol)
            result = baseline_result
            sensitivity[f'{protocol}/{representation}/M0star'] = m4.cv_scores(frozen.y, result.yhat)
            for family in qualifying:
                result = m3.run_family(family, ev.COMPARATOR, frame, sf, distances['station'], representation, protocols=(protocol,))
                entry = {'status': result.status, 'failure': result.failure}
                if result.status == 'computable':
                    entry.update(m4.cv_scores(frozen.y, result.results[protocol].yhat))
                    entry['paired'] = ev.paired_rmse_interval(frozen.y, result.results[protocol].yhat, baseline_result.yhat)
                    entry['theta'] = result.full.theta
                sensitivity[f'{protocol}/{representation}/{family}'] = entry
    write_json(out / 'sensitivities.json', sensitivity)
    products = {}
    for name, path in [('aligned_era5', run / 'era5/aligned_era5_trends.csv'), ('legacy_era5', run / 'era5/era5_area_trends.parquet')]:
        data = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
        y = data.set_index('iso3').loc[frozen.table.iso3, m4.ERA5_COLUMN].to_numpy(float)
        for representation, base in frozen.frames.items():
            frame = base.assign(**{OUTCOME_COL: y})
            pf = replace(frozen, y=y, frames={**frozen.frames, representation: frame})
            scored = ev.score_model(ev.COMPARATOR, frame, pf)
            reference = ev.score_model(ev.COMPARATOR, base, frozen)
            entry = {'M0star': scored.card, 'residual_agreement_with_berkeley': m4.residual_agreement(y - scored.fitted, frozen.y - reference.fitted)}
            add_predictions(rows, pf, scored, representation, ev.COMPARATOR, product=name)
            for family in qualifying:
                result = m3.run_family(family, ev.COMPARATOR, frame, pf, distances['station'], representation)
                arm = {'status': result.status, 'failure': result.failure}
                if result.status == 'computable':
                    arm['theta'] = result.full.theta
                    arm['card'] = m3.family_card(result, frame, pf, 'M0*', pf.station_w, pf.area_w, distances['station'])
                    arm['paired'] = m3.paired_block(y, result, scored)
                    arm['conditions'] = m3.qualification(arm['card'], scored.card, arm['paired']['vs_static'], True)
                    m3.prediction_rows({(representation, family): result}, pf, 'station').assign(product=name).to_csv(
                        out / f'{name}_{representation}_{family}_predictions.csv', index=False)
                entry[family] = arm
            products[f'{name}/{representation}'] = entry
            write_json(out / 'products.json', products)
    pd.DataFrame(rows).to_csv(out / 'product_static_predictions.csv', index=False)
    summary, draws = m4.static_uncertainty(frozen, ev.COMPARATOR)
    write_json(out / 'bootstrap_summary.json', summary)
    draws.to_csv(out / 'bootstrap_draws.csv', index=False)
    return products


def snapshot(run, stage):
    """Record actual local source hashes before any stage reads outcomes."""
    out = Path(run) / 'control' / f'v2_{stage}_pre_execution.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f'Use a distinct attempt directory; refusing overwrite: {out}')
    sources = sorted({*ev.prov.import_closure(['research/model_v2/prospective_refit.py']), *ev.prov.LOCK_FILES})
    write_json(out, {'stage': stage, 'source_sha256': {p: digest(ROOT / p) for p in sources},
                     'input_sha256': {str(p): digest(p) for folder in ['preprocessing', 'inputs-koppen', 'era5']
                                      for p in sorted((Path(run) / folder).glob('*')) if p.is_file()}})


def measurement_pilot(run, raw, *, out_dir=None):
    """Three fixed 16-degree tiles, unchanged integration; never a result sample."""
    import pyogrio
    import shapely
    import xarray as xr

    run, raw = Path(run), Path(raw)
    out = Path(out_dir) if out_dir is not None else run / 'v2_measurement_pilot'
    if out_dir is None:
        out.mkdir(exist_ok=False)
    _, frozen, _, _ = context(run, raw)
    def raw_path(path):
        return raw / Path(path).relative_to(ROOT / 'data/raw')
    paths = [raw_path(geo.ETOPO_PATH), raw_path(geo.LAND_ZIP_PATH), raw_path(geo.GPW_PATH),
             raw / 'gshhg/gshhg-shp-2.3.7.zip', run / 'inputs-koppen/koppen_present_0p5.nc']
    write_json(out / 'raw_inputs_pre_execution.json', {str(p): digest(p) for p in paths})
    levels, _ = _ORIGINAL_GSHHG(raw / 'gshhg')
    coast = geo.Coast.build(shapely.union_all(pyogrio.read_dataframe(f'zip://{raw_path(geo.LAND_ZIP_PATH)}').geometry.to_numpy()))
    countries, _, _ = fresh_gpw(frozen.table.iso3.tolist(), raw_path(geo.GPW_PATH))
    with xr.open_dataset(raw_path(geo.ETOPO_PATH)) as ds:
        z = ds[geo.ETOPO_VAR].to_numpy()
    with xr.open_dataset(run / 'inputs-koppen/koppen_present_0p5.nc') as ds:
        koppen = geo.koppen_group_index(ds[geo.KOPPEN_VAR].to_numpy())[::-1].copy()
    results = []
    for name, r, c in [('Europe', 496, 704), ('South_Asia', 416, 960), ('Maritime_Asia', 320, 1120)]:
        sums, batch = geo.Sums(frozen.n), [geo.Records()]
        def flush():
            geo.accumulate(batch[0], sums, countries, lambda i, j: z[i, j].astype(float), koppen, coast, geo.pixel_area_rows())
            batch[0] = geo.Records()
        def emit(kind, r0, r1, c0, c1, geom):
            geo.emit_into(batch[0], kind, r0, r1, c0, c1, geom)
            if batch[0].size >= 1_000_000:
                flush()
        started = time.perf_counter()
        stats = geo.walk(levels, countries, emit, r, r + 64, c, c + 64)
        flush()
        results.append({'tile': name, 'r0': r, 'c0': c, 'walk_stats': stats,
                        'pixels': int(sums.pixels.sum()), 'wall_seconds': time.perf_counter() - started})
        write_json(out / 'timing.json', {'pilot_only': True, 'tiles': results,
                   'scope': 'Integration and coastline distance timing; no country output or scoring'})
    return results


def product_lisa(run, raw, *, out_dir=None):
    """Fresh paired M0 product residuals and fixed 9,999-permutation inference."""
    from dataclasses import asdict
    from research.model_v2 import product_stability as ps, spatial as sp

    run = Path(run)
    out = Path(out_dir) if out_dir is not None else run / 'v2_product_stability'
    if out_dir is None:
        out.mkdir(exist_ok=False)
    complete, frozen, _, _ = context(run, raw)
    x, names, groups = m0.design_matrix(complete)
    arms = {}
    for arm, path in [('absolute', run / 'era5/era5_area_trends.parquet'), ('aligned', run / 'era5/aligned_era5_trends.csv')]:
        era = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
        b = complete[OUTCOME_COL].to_numpy(float)
        e = frozen.table.iso3.map(era.set_index('iso3')[m4.ERA5_COLUMN]).to_numpy(float)
        keep, rb, re = ps.paired_refit(x, b, e)
        table = frozen.table.loc[keep, ['Country', 'iso3', 'm49_subregion']].reset_index(drop=True)
        table = table.assign(berkeley=b[keep], era5=e[keep], b_residual=rb, e_residual=re)
        table['b_minus_e'] = table.berkeley - table.era5
        table['b_sign'] = np.sign(np.where(abs(rb) < 1e-10, 0, rb)).astype(int)
        table['e_sign'] = np.sign(np.where(abs(re) < 1e-10, 0, re)).astype(int)
        table['sign_agreement'] = table.b_sign == table.e_sign
        gap = table.b_minus_e.to_numpy()
        table['squared_gap_share'] = gap**2 / np.sum(gap**2)
        results = {}
        # Rebuild each graph on the exact paired complete-case sample.
        geom = frozen.table.loc[keep]
        weights = {'station_knn8': knn_weights(haversine_matrix(geom.station_lon.to_numpy(), geom.station_lat.to_numpy()), 8),
                   'area_knn8': knn_weights(haversine_matrix(geom.centroid_lon.to_numpy(), geom.centroid_lat.to_numpy()), 8)}
        for weight, w in weights.items():
            results[weight] = {}
            for col in ['b_minus_e', 'b_residual', 'e_residual']:
                results[weight][col] = asdict(sp.morans_i(table[col].to_numpy(), w, 9999, 0))
                local = ps.local_moran_fdr(sp.local_morans(table[col].to_numpy(), w, 9999, 0))
                for field in ['local_i', 'p_value', 'q_value', 'quadrant', 'label', 'fdr_label', 'status', 'reason']:
                    table[f'{weight}_{col}_{field}'] = local[field].to_numpy()
        region = table.groupby('m49_subregion', sort=True).agg(
            n=('iso3', 'size'), b_mean=('b_residual', 'mean'), e_mean=('e_residual', 'mean'),
            difference_mean=('b_minus_e', 'mean'), difference_median=('b_minus_e', 'median'),
            difference_sd=('b_minus_e', 'std'), squared_gap_share=('squared_gap_share', 'sum'),
            sign_agreement=('sign_agreement', 'mean'))
        bt, et = .5 * rb.std(ddof=1), .5 * re.std(ddof=1)
        region['classification'] = region.apply(ps.classify_region, axis=1, args=(bt, et))
        tails = {}
        for kind in ['positive', 'negative', 'absolute']:
            scores = [v if kind == 'positive' else -v if kind == 'negative' else abs(v) for v in [rb, re]]
            aa, bb = [set(table.iloc[np.argsort(-v, kind='stable')[:15]].iso3) for v in scores]
            tails[kind] = {'k': 15, 'berkeley': sorted(aa), 'era5': sorted(bb), 'intersection': sorted(aa & bb),
                           'n_overlap': len(aa & bb), 'jaccard': len(aa & bb) / len(aa | bb)}
        summary = {'prospective': True, 'common_n': int(keep.sum()), 'm0_n': len(complete), 'design_columns': names,
                   'groups': groups, 'design_rank': int(np.linalg.matrix_rank(x[keep])),
                   'r2': {key: float(1 - np.sum(r**2) / np.sum((v - v.mean())**2))
                          for key, r, v in [('berkeley', rb, b[keep]), ('era5', re, e[keep])]},
                   'residual_correlations': ps.correlations(rb, re), 'outcome_correlations': ps.correlations(b[keep], e[keep]),
                   'sign_agreement_n': int(table.sign_agreement.sum()), 'tail_overlap': tails,
                   'region_material_threshold': {'berkeley': bt, 'era5': et}, 'spatial': results,
                   'permutations': 9999, 'seed': 0, 'lisa_overlap': ps.lisa_overlap(table),
                   'lisa_fdr_counts': {col: table[col].value_counts().to_dict() for col in table if col.endswith('fdr_label')},
                   'outcome_input_sha256': digest(path), 'fresh_design_identity': frozen.identity}
        table.to_csv(out / f'{arm}_countries.csv', index=False)
        region.to_csv(out / f'{arm}_regions.csv')
        write_json(out / f'{arm}_summary.json', summary)
        arms[arm] = table, summary, region
    ps.classify_patterns(arms).to_csv(out / 'pattern_classification.csv', index=False)
    return {arm: value[1] for arm, value in arms.items()}


def buffer_predictions(run, raw, *, out_dir=None):
    """Supplement completed buffer scores with fresh fixed-fit row evidence."""
    run = Path(run)
    out = Path(out_dir) if out_dir is not None else run / 'v2_buffer_predictions'
    if out_dir is None:
        out.mkdir(exist_ok=False)
    _, frozen, _, distances = context(run, raw)
    previous = json.loads((run / 'v2_final_checks/sensitivities.json').read_text())
    rows, checks = [], []
    for protocol, buffer in m4.SENSITIVITY_PROTOCOLS.items():
        distance = frozen.distance if 'territory' in protocol else distances['land']
        sf = replace(frozen, distance=distance, folds={protocol: (np.arange(frozen.n), buffer)})
        for representation, frame in frozen.frames.items():
            baseline, fits = ev.run_protocol(ev.COMPARATOR, frame, sf, protocol)
            gates.require(not any(f['failures'] for f in fits), 'Buffer M0* structural fit failure')
            models = [(ev.COMPARATOR, baseline, {int(f['fold']): f['train_index'] for f in fits}, None)]
            for family in ['sem', 'sar']:
                run_family = m3.run_family(family, ev.COMPARATOR, frame, sf, distances['station'], representation, protocols=(protocol,))
                gates.require(run_family.status == 'computable', f'Buffer {family} noncomputable')
                models.append((family, run_family.results[protocol],
                    {int(f['fold']): f['train'] for f in run_family.records[protocol]}, run_family.full.theta))
            for model, result, training, theta in models:
                gates.require(np.isfinite(result.yhat).all(), 'Non-finite buffer predictions')
                error = frozen.y - result.yhat
                # Independently calculate metrics directly from retained row vectors.
                metrics = {'cv_rmse': float(np.sqrt(np.mean(error**2))), 'cv_mae': float(np.mean(abs(error))),
                           'cv_r2': float(1 - np.sum(error**2) / np.sum((frozen.y - frozen.y.mean())**2))}
                key = f'{protocol}/{representation}/{model}'
                differences = {metric: abs(value - previous[key][metric]) for metric, value in metrics.items()}
                gates.require(all(v <= m4.SENSITIVITY_ANCHOR_ATOL for v in differences.values()),
                              f'Supplemental buffer metrics differ from sealed comparison: {key}')
                checks.append({'key': key, 'metrics': metrics, 'absolute_differences_vs_sealed': differences,
                               'comparison_tolerance': m4.SENSITIVITY_ANCHOR_ATOL})
                for i, code in enumerate(frozen.table.iso3):
                    train = training[int(result.fold_id[i])]
                    rows.append({'protocol': protocol, 'representation': representation, 'model': model,
                        'iso3': code, 'observed': float(frozen.y[i]), 'prediction': float(result.yhat[i]),
                        'error': float(error[i]), 'fold_id': int(result.fold_id[i]), 'n_train': int(result.n_train[i]),
                        'nearest_train_km': float(result.nearest_train_km[i]), 'unseen_levels': int(result.unseen_levels[i]),
                        'training_iso3': '|'.join(frozen.table.iso3.iloc[train]), 'theta_full_sample': theta})
    predictions = pd.DataFrame(rows)
    gates.require(len(predictions) == 2 * 2 * 3 * frozen.n, 'Incomplete supplemental buffer prediction set')
    predictions.to_csv(out / 'predictions.csv', index=False)
    write_json(out / 'independent_score_checks.json', {'all_passed': True, 'checks': checks,
        'scope': 'New fixed-specification fits; metrics calculated directly from saved row vectors and compared with sealed previous aggregates'})
    return {'prediction_rows': len(predictions), 'score_checks': len(checks)}


def stage_inputs(run, raw, stage):
    """Explicit per-stage scientific dependencies, hashed before fitting."""
    run, raw = Path(run), Path(raw)
    paths = [run / 'preprocessing/country_inequality.parquet', run / 'preprocessing/city_features.parquet',
             raw / m0.INCOME_PATH.relative_to(ROOT / 'data/raw'),
             raw / m0.OWID_CO2_PATH.relative_to(ROOT / 'data/raw'), ROOT / ev.GEOMETRY,
             territory.LOWER_PATH, territory.SNAPSHOT_PATH]
    paths += [ROOT / p for p in [b.COUNTRY_TABLE, b.FOLD_MEMBERSHIP, b.TERRITORY_MEMBERSHIP, b.CORRECTED_ERRORS, ev.M1B_FOLDS, ev.FEASIBILITY_FITS, ev.FEASIBILITY_SUMMARY]]
    paths += [ROOT / 'research/model_v2/outputs/m1a_measurement_manifest.json', gates.HYDRO_CONTRACT]
    if stage in ['measurement', 'measurement-pilot']:
        paths += [raw / Path(p).relative_to(ROOT / 'data/raw') for p in
                  [geo.GPW_PATH, geo.ETOPO_PATH, geo.LAND_ZIP_PATH, geo.KOPPEN_PATH, hydro.GPW_NATID_LOOKUP_PATH]]
        paths += [run / 'inputs-koppen/koppen_present_0p5.nc', raw / 'gshhg/gshhg-shp-2.3.7.zip']
        paths += sorted((raw / 'gshhg/GSHHS_shp/f').glob('GSHHS_f_L[1-4].*'))
        if stage == 'measurement':
            for spec in hydro.FILES.values():
                compressed = raw / 'cru_ts_4.10' / Path(spec['path']).name
                paths.extend([compressed, compressed.with_suffix('')])
    if stage == 'static':
        paths += [run / 'v2_measurement' / name for name in
                  ['started.json', 'completed.json', 'm1a_geography_features.csv', 'm1a_measurement_manifest.json', 'm1a_geography_qa.csv', 'm1a_cell_land_area_km2.npz',
                   *hydro.PACKAGE_ARTIFACTS, hydro.MANIFEST]]
    if stage == 'buffer-predictions':
        paths.append(run / 'v2_final_checks/sensitivities.json')
    if stage in ['final-checks', 'product-lisa']:
        paths += [run / 'era5/era5_area_trends.parquet', run / 'era5/aligned_era5_trends.csv']
    return sorted(set(paths))


def main(argv=None):
    import importlib.util

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--stage', choices=['measurement', 'measurement-pilot', 'static', 'spatial', 'spatial-pilot', 'final-checks', 'product-lisa', 'buffer-predictions'], required=True)
    parser.add_argument('--attempt', type=int, default=1, help='Fresh numbered attempt; earlier evidence is preserved')
    parser.add_argument('--measurement-stage', default='v2_measurement', help='Accepted completed measurement receipt')
    args = parser.parse_args(argv)
    if args.attempt < 1 or Path(args.measurement_stage).name != args.measurement_stage:
        raise ValueError('Invalid attempt or measurement receipt name')
    spec = importlib.util.spec_from_file_location('validation_runlib', args.run / 'control/runlib.py')
    runlib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runlib)
    if runlib.RUN.resolve() != args.run.resolve():
        raise ValueError('Receipt runner belongs to a different run')
    name = 'v2_' + args.stage.replace('-', '_')
    if args.attempt > 1:
        name += f'_attempt{args.attempt}'
    sources = [ROOT / p for p in ev.prov.import_closure(['research/model_v2/prospective_refit.py'])] + [gates.HYDRO_CONTRACT]
    def action(out):
        functions = {'measurement': measurement, 'measurement-pilot': measurement_pilot, 'static': static,
                     'spatial': spatial, 'final-checks': final_checks, 'product-lisa': product_lisa, 'buffer-predictions': buffer_predictions}
        if args.stage == 'spatial-pilot':
            spatial(args.run, args.raw, pilot=True, out_dir=out)
        elif args.stage == 'static':
            static(args.run, args.raw, out_dir=out, measurement_dir=args.run / args.measurement_stage)
        else:
            functions[args.stage](args.run, args.raw, out_dir=out)
        return {'scientific_output_directory': str(out), 'stage': args.stage}
    dependencies = ['preprocessing']
    if args.stage == 'static':
        dependencies.append(args.measurement_stage)
    if args.stage == 'buffer-predictions':
        dependencies.append('v2_final_checks')
    inputs = stage_inputs(args.run, args.raw, args.stage)
    if args.stage == 'static' and args.measurement_stage != 'v2_measurement':
        inputs = [args.run / args.measurement_stage / p.name if p.parent == args.run / 'v2_measurement' else p for p in inputs]
    runlib.stage(name, inputs, action,
                 runner=Path(__file__), dependencies=dependencies, sources=sources)


if __name__ == '__main__':
    main()
