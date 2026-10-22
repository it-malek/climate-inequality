"""Failure-rejection coverage for the prospective scientific acceptance gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from research.model_v2 import prospective_gates as g
from research.model_v2.prospective_assets import allocation


def test_gshhg_pin_and_extracted_member_drift_fail_closed(tmp_path):
    archive = tmp_path / 'gshhg-shp-2.3.7.zip'
    member = 'GSHHS_shp/f/GSHHS_f_L1.shp'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr(member, b'fixed polygon bytes')
    path = tmp_path / member
    path.parent.mkdir(parents=True)
    path.write_bytes(b'fixed polygon bytes')
    pin = g.sha(archive)
    assert g.gshhg_sources(tmp_path, pin)['archive_sha256'] == pin
    with pytest.raises(ValueError, match='Pinned source mismatch'):
        g.gshhg_sources(tmp_path, '0' * 64)
    path.write_bytes(b'changed extracted bytes')
    with pytest.raises(ValueError, match='extracted members'):
        g.gshhg_sources(tmp_path, pin)


def context_frames():
    n = 151
    total, population = np.linspace(10, 200, n), np.linspace(100, 2000, n)
    complete = pd.DataFrame({'Country': [f'C{i}' for i in range(n)], 'cum_co2_total': total,
        'cum_co2_per_capita': total / population * 1e6, 'population': population,
        'abs_latitude': np.linspace(1, 60, n), 'elevation': np.ones(n), 'continentality': np.ones(n),
        'station_density': np.ones(n), 'warming_trend': np.linspace(.1, .3, n)})
    table = complete.assign(iso3=[f'X{i}' for i in range(n)])
    frames = {rep: g.ev.feas.transform_numerics(g.ev.feas.representation_frame(table, rep)) for rep in g.ev.REPRESENTATIONS}
    folds = {'primary_loco': (np.arange(n), 500), 'm49_subregion_lo': (np.arange(n) % 20, None), 'random10': (np.arange(n) % 10, None)}
    return complete, table, frames, folds


@pytest.mark.parametrize('defect,match', [('order', 'identity/order'), ('identity', 'log identity'),
                                         ('nonfinite', 'Non-finite'), ('folds', 'fold counts')])
def test_context_identity_rejects_design_drift(monkeypatch, defect, match):
    complete, table, frames, folds = context_frames()
    labels = table[['Country', 'iso3']].copy()
    monkeypatch.setattr(g, 'fixed_inputs', lambda: {})
    monkeypatch.setattr(g.pd, 'read_csv', lambda *args, **kwargs: labels)
    monkeypatch.setattr(g.ev, 'fold_identities', lambda *args: {'checked': True})
    if defect == 'order':
        table = table.iloc[::-1].reset_index(drop=True)
    elif defect == 'identity':
        complete.loc[0, 'cum_co2_per_capita'] *= 2
    elif defect == 'nonfinite':
        complete.loc[0, 'warming_trend'] = np.nan
    else:
        folds['random10'] = (np.zeros(151, int), None)
    with pytest.raises(ValueError, match=match):
        g.context_identity(complete, table, frames, np.zeros((151, 151)), folds)


def passing_card():
    return {'matrix_columns': 21, 'matrix_rank': 21,
            'training_fits': {'all_full_rank': True, 'feature_identified_in_all': True},
            'share_accounting': {'passes': True}, 'share_geography': .5, 'share_hydroclimate': .1,
            'residual_share': .4}


@pytest.mark.parametrize('defect', ['rank', 'training', 'c2', 'accounting', 'numeric_accounting'])
def test_scoring_gate_rejects_every_integrity_failure(defect):
    card = passing_card()
    if defect == 'rank':
        card['matrix_rank'] = 20
    elif defect == 'training':
        card['training_fits']['all_full_rank'] = False
    elif defect == 'c2':
        card['training_fits']['feature_identified_in_all'] = False
    elif defect == 'accounting':
        card['share_accounting']['passes'] = False
    else:
        card['residual_share'] = .5
    with pytest.raises(ValueError):
        g.scored_integrity(SimpleNamespace(card=card), expected_columns=21, c2=True)


def test_original_representation_equivalence_refusal_propagates(monkeypatch):
    monkeypatch.setattr(g.b, 'equivalence', lambda arms: {'passes': False})
    with pytest.raises(ValueError, match='original tolerances'):
        g.equivalent({})


def package(tmp_path):
    ids = pd.DataFrame({'iso3': ['AAA'], 'Country': ['A']})
    features = ids.assign(baseline_dryness=-.2)
    features.to_csv(tmp_path / g.hydro.FEATURES, index=False)
    ids.to_csv(tmp_path / 'm1a_geography_qa.csv', index=False)
    (tmp_path / 'm1a_cell_land_area_km2.npz').write_bytes(b'fresh support fixture')
    qa = ids.assign(total_land_area_km2=1., native_valid_area_km2=1., harmonized_area_km2=0.,
        unresolved_area_km2=0., unresolved_structural_area_km2=0., non_structural_invalid_area_km2=0.,
        coverage=1., pre_pure_climatology_share=0., c2_harmonized=-.2)
    qa.to_csv(tmp_path / 'm1b_hydroclimate_qa.csv', index=False)
    for name in g.hydro.PACKAGE_ARTIFACTS:
        if not (tmp_path / name).exists():
            (tmp_path / name).write_text('{}\n')
    kernel = Path(g.hydro.__file__).resolve()
    archived = tmp_path / 'execution-source' / kernel.relative_to(g.ROOT)
    archived.parent.mkdir(parents=True)
    archived.write_bytes(kernel.read_bytes())
    (tmp_path / 'started.json').write_text(json.dumps({'code_sha256': {str(kernel): g.sha(kernel)}}))
    manifest = {'gate_version': g.hydro.GATE_VERSION, 'all_hard_stops_pass': True, 'hard_stops': [],
        'git': {'commit': None, 'code_path_matches_commit': None}, 'code_sha256': g.sha(kernel),
        'contract_freeze_commit': g.hydro.CONTRACT_COMMIT, 'contract_sha256': g.sha(g.HYDRO_CONTRACT),
        'support_inputs': {'m1a_cell_land_area_sha256': g.sha(tmp_path / 'm1a_cell_land_area_km2.npz'),
            'm1a_geography_qa_sha256': g.sha(tmp_path / 'm1a_geography_qa.csv'),
            'country_list_sha256': hashlib.sha256(b'AAA\n').hexdigest()},
        'artifact_sha256': {p: g.sha(tmp_path / p) for p in g.hydro.PACKAGE_ARTIFACTS},
        'window': list(g.hydro.WINDOW), 'n_months': g.hydro.N_MONTHS, 'coverage_min': g.hydro.COVERAGE_MIN,
        'amendments': list(g.hydro.AMENDMENTS), 'formula': 'fixed source kernel',
        'cru_version': g.hydro.CRU_VERSION, 'cru_run_id': g.hydro.CRU_RUN_ID,
        'sources': {var: {**{k: spec[k] for k in ['gz_sha256', 'nc_sha256', 'nc_bytes', 'units']},
                          'gz_bytes': spec['bytes']} for var, spec in g.hydro.FILES.items()},
        'execution_receipt': {'started_sha256': g.sha(tmp_path / 'started.json'), 'historical_commit_stamp_replaced': True}}
    return manifest


def test_current_package_accepts_honest_receipt_without_fake_historical_git(tmp_path):
    manifest = package(tmp_path)
    (tmp_path / g.hydro.MANIFEST).write_text(json.dumps(manifest))
    features, result = g.m1b_package(tmp_path)
    assert result['all_passed'] and features.iso3.tolist() == ['AAA']


@pytest.mark.parametrize('defect', ['window', 'bool_months', 'coverage', 'version', 'artifact', 'archive', 'receipt', 'c2'])
def test_current_package_rejects_scientific_and_provenance_drift(tmp_path, defect):
    manifest = package(tmp_path)
    if defect == 'window':
        manifest['window'] = ['1950-01', '1979-12']
    elif defect == 'bool_months':
        manifest['n_months'] = True
    elif defect == 'coverage':
        manifest['coverage_min'] = .9
    elif defect == 'version':
        manifest['gate_version'] = 'incorrect'
    elif defect == 'artifact':
        (tmp_path / g.hydro.FEATURES).write_text('tampered')
    elif defect == 'archive':
        archived = tmp_path / 'execution-source' / Path(g.hydro.__file__).resolve().relative_to(g.ROOT)
        archived.write_text('changed code')
    elif defect == 'receipt':
        manifest['execution_receipt']['started_sha256'] = '0' * 64
    else:
        features = pd.read_csv(tmp_path / g.hydro.FEATURES)
        features['baseline_dryness'] = np.nan
        features.to_csv(tmp_path / g.hydro.FEATURES, index=False)
        manifest['artifact_sha256'][g.hydro.FEATURES] = g.sha(tmp_path / g.hydro.FEATURES)
    (tmp_path / g.hydro.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        g.m1b_package(tmp_path)


def test_dynamic_allocation_preserves_hydroclimate_and_rejects_missing_share():
    card = {'share_geography': .5, 'share_emissions': .02, 'share_socioeconomic': .08,
            'share_population': .03, 'share_hydroclimate': .07, 'residual_share': .3}
    result = allocation(card)
    assert result['hydroclimate'] == .07 and sum(result.values()) == 1.
    altered = deepcopy(card)
    del altered['share_hydroclimate']
    with pytest.raises(ValueError, match='does not sum to one'):
        allocation(altered)


def m1a_package(tmp_path, monkeypatch):
    root = tmp_path / 'reference'
    reference = root / 'research/model_v2/outputs/m1a_measurement_manifest.json'
    reference.parent.mkdir(parents=True)
    snapshot = tmp_path / 'footprints.npz'
    np.savez(snapshot, AAA=np.zeros((2, 2)), BBB=np.zeros((1, 2)))
    monkeypatch.setattr(g, 'ROOT', root)
    monkeypatch.setattr(g.territory, 'SNAPSHOT_PATH', snapshot)
    sources = {'gpw_national_identifier': {'sha256': 'gpw-pin', 'lookup_sha256': 'lookup-pin'},
               'territory_snapshot_sha256': g.sha(snapshot)}
    reference.write_text(json.dumps({'sources': sources}))
    out = tmp_path / 'measurement'
    out.mkdir()
    table = pd.DataFrame({'iso3': ['AAA', 'BBB'], 'Country': ['A', 'B'], 'spatial_block': ['Asia', 'Europe']})
    features = table.assign(abs_latitude=[10., 20.], elevation=[100., 200.], continentality=[3., 5.],
                            climate_zone=['A', 'B'], hemisphere=['N', 'S'])
    qa = table.assign(gpw_cells=[2, 1], continentality_quadrature_ok=True)
    features.to_csv(out / 'm1a_geography_features.csv', index=False)
    qa.to_csv(out / 'm1a_geography_qa.csv', index=False)
    (out / 'm1a_measurement_manifest.json').write_text(json.dumps({'all_features_pass_gates': True,
        'gate_failures': [], 'quadrature_failures': [], 'sources': sources}))
    return out, table, features, qa


@pytest.mark.parametrize('defect', ['order', 'block', 'missing', 'counts', 'source'])
def test_m1a_support_rejects_each_registered_support_defect(tmp_path, monkeypatch, defect):
    out, table, features, qa = m1a_package(tmp_path, monkeypatch)
    _, passing = g.m1a_support(out, table, table)
    assert passing['all_passed']
    if defect == 'order':
        features = features.iloc[::-1]
    elif defect == 'block':
        features.loc[0, 'spatial_block'] = 'Europe'
    elif defect == 'missing':
        features.loc[0, 'elevation'] = np.nan
    elif defect == 'counts':
        qa.loc[0, 'gpw_cells'] = 3
        qa.to_csv(out / 'm1a_geography_qa.csv', index=False)
    else:
        path = out / 'm1a_measurement_manifest.json'
        record = json.loads(path.read_text())
        record['sources']['gpw_national_identifier']['lookup_sha256'] = 'different-source'
        path.write_text(json.dumps(record))
    features.to_csv(out / 'm1a_geography_features.csv', index=False)
    with pytest.raises(ValueError):
        g.m1a_support(out, table, table)


def full_rank_predictors(n=151):
    rng = np.random.default_rng(414)
    values = pd.DataFrame({'Country': [f'C{i}' for i in range(n)],
        'cum_co2_total': np.exp(rng.normal(10, 1, n)), 'abs_latitude': rng.uniform(1, 70, n),
        'elevation': rng.uniform(1, 2000, n), 'continentality': rng.uniform(1, 1000, n),
        'climate_zone': np.resize(list('ABCDE'), n), 'hemisphere': np.resize(['N', 'S'], n),
        'spatial_block': np.resize(['Africa', 'Asia', 'Europe', 'NA', 'SA', 'Oceania'], n),
        'income_group': np.resize(['low', 'lower_mid', 'upper_mid', 'high'], n),
        'population': np.exp(rng.normal(8, 1, n)), 'station_density': rng.uniform(.1, 10, n)})
    for column in ['climate_zone', 'hemisphere', 'spatial_block', 'income_group']:
        values[column] = rng.permutation(values[column].to_numpy())
    return values[g.red.PREDICTOR_COLUMNS], rng.normal(-.2, .3, n)


def test_predictor_only_redundancy_preserves_exact_rank_hard_stop():
    predictors, c2 = full_rank_predictors()
    features = pd.DataFrame({'Country': predictors.Country, 'baseline_dryness': c2})
    result = g.redundancy(predictors, features, {'manifest_sha256': 'fixture'})
    assert result['design_rank'] == 20 and result['rank_with_c2'] == 21
    features['baseline_dryness'] = np.log10(predictors.cum_co2_total)
    with pytest.raises(ValueError, match='redundancy hard stop'):
        g.redundancy(predictors, features, {'manifest_sha256': 'fixture'})
    with pytest.raises(ValueError, match='predictor-only'):
        g.redundancy(predictors.assign(warming_trend=0.), features, {'manifest_sha256': 'fixture'})
