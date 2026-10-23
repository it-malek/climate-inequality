"""Fail-closed scientific gates for fresh-input V2 reconstruction.

Historical files below supply only pinned raw provenance, fixed geometry and fold
identity. Fresh packages use their actual archived preexecution receipt identity.
No historical fitted predictions, observed outcomes or scores are used here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

from research.model_v2 import m1a_geography as geo
from research.model_v2 import m1b_evaluate as b, m1b_hydroclimate as hydro, m1b_redundancy as red
from research.model_v2 import m2_evaluate as ev, territory
from src.decomposition import LOG10_FEATURES, OUTCOME_COL

ROOT = Path(__file__).resolve().parents[2]
HYDRO_CONTRACT = ROOT / 'research/model_v2/evidence/M1B_EVALUATION_CONTRACT.md'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hydro.sha256(path)


def pinned(path, expected):
    actual = sha(path)
    require(actual == expected, f'Pinned source mismatch: {path}')
    return actual


def gshhg_sources(directory, archive_pin=geo.GSHHG_ZIP_SHA256):
    """Original ZIP pin and exact extracted full-resolution level 1-4 members."""
    directory = Path(directory)
    archive = directory / 'gshhg-shp-2.3.7.zip'
    archive_sha = pinned(archive, archive_pin)
    with zipfile.ZipFile(archive) as zipped:
        members = {Path(name).name: hashlib.sha256(zipped.read(name)).hexdigest()
                   for name in zipped.namelist()
                   if name.startswith('GSHHS_shp/f/GSHHS_f_L') and name[-5] in '1234'}
    extracted = {p.name: sha(p) for p in sorted((directory / 'GSHHS_shp/f').glob('GSHHS_f_L[1-4].*'))}
    require(bool(members) and members == extracted, 'GSHHG extracted members differ from pinned archive')
    return {'archive_sha256': archive_sha, 'member_sha256': extracted}


def fixed_inputs():
    """Only geometry/identity files, with their original fixed byte pins."""
    names = [b.COUNTRY_TABLE, b.GEOMETRY, b.FOLD_MEMBERSHIP, b.TERRITORY_MEMBERSHIP,
             str(territory.LOWER_PATH.relative_to(ROOT)), b.CORRECTED_ERRORS]
    result = {name: pinned(ROOT / name, b.FROZEN_SCORING_INPUTS[name][0]) for name in names}
    for name in [ev.M1B_FOLDS, ev.FEASIBILITY_FITS, ev.FEASIBILITY_SUMMARY]:
        result[name] = pinned(ROOT / name, ev.ALL_PINS[name][0])
    source = json.loads((ROOT / 'research/model_v2/outputs/m1a_measurement_manifest.json').read_text())['sources']
    result[str(territory.SNAPSHOT_PATH.relative_to(ROOT))] = pinned(territory.SNAPSHOT_PATH, source['territory_snapshot_sha256'])
    return result


def raw_measurement_sources(raw, koppen):
    """Preserve all raw M1a pins, plus explicit fresh Köppen conversion equality."""
    raw, koppen = Path(raw), Path(koppen)
    historical = json.loads((ROOT / 'research/model_v2/outputs/m1a_measurement_manifest.json').read_text())['sources']
    sources = {'gshhg': gshhg_sources(raw / 'gshhg')}
    original_manifest = json.loads((ROOT / 'research/model_v2/outputs/m1a_measurement_manifest.json').read_text())
    sources['historical_measurement_spec'] = {**original_manifest['spec'], 'available_in_current_checkout': False,
        'current_execution_authority': 'Frozen current validation specification and unchanged archived measurement kernel'}
    for name in ['etopo', 'natural_earth_land', 'gpw_national_identifier', 'koppen']:
        spec = historical[name]
        path = raw / Path(spec['path']).relative_to('data/raw')
        sources[name] = {**spec, 'path': str(path), 'sha256': pinned(path, spec['sha256'])}
    lookup = raw / hydro.GPW_NATID_LOOKUP_PATH.relative_to(ROOT / 'data/raw')
    pinned(lookup, historical['gpw_national_identifier']['lookup_sha256'])
    old_koppen = Path(sources['koppen']['path'])
    with xr.open_dataset(old_koppen) as reference, xr.open_dataset(koppen) as fresh:
        for field in ['lat', 'lon', geo.KOPPEN_VAR]:
            require(np.array_equal(reference[field].to_numpy(), fresh[field].to_numpy()),
                    f'Fresh Köppen conversion differs from fixed source values: {field}')
    sources['koppen']['historical_reference_sha256'] = sources['koppen']['sha256']
    sources['koppen']['sha256'], sources['koppen']['path'] = sha(koppen), str(koppen)
    sources['koppen']['fresh_conversion_value_and_coordinate_equality'] = True
    sources['territory_snapshot_sha256'] = pinned(territory.SNAPSHOT_PATH, historical['territory_snapshot_sha256'])
    require(sources['gpw_national_identifier']['sha256'] == hydro.FROZEN_INPUT_SHA256[hydro.GPW_PATH], 'GPW support pin disagreement')
    return sources


def context_identity(complete, table, frames, distance, folds):
    """Fixed population/folds/counts, finite numerics and exact log representation."""
    pins = fixed_inputs()
    labels = pd.read_csv(ROOT / b.COUNTRY_TABLE, usecols=['Country', 'iso3'])
    require(len(table) == len(complete) == 151 and table.iso3.is_unique, 'Fixed 151-country population')
    require(table.Country.tolist() == labels.Country.tolist() and table.iso3.tolist() == labels.iso3.tolist(),
            'Fresh country identity/order differs from fixed analytical population')
    for name in LOG10_FEATURES & set(complete):
        require(np.isfinite(complete[name].to_numpy(float)).all() and (complete[name] > 0).all(), f'Invalid positive/log predictor {name}')
    require(np.isfinite(complete[OUTCOME_COL].to_numpy(float)).all(), 'Non-finite fresh outcome')
    for representation, frame in frames.items():
        numeric = ev.feas.NUMERIC_COLUMNS[representation]
        require(np.isfinite(frame[numeric].to_numpy(float)).all(), f'Non-finite transformed predictors: {representation}')
    error = float(np.max(np.abs(np.log10(complete.cum_co2_per_capita.to_numpy(float))
                               - np.log10(complete.cum_co2_total.to_numpy(float))
                               + np.log10(complete.population.to_numpy(float)) - 6)))
    require(error <= b.IDENTITY_ATOL, 'Representation log identity failed')
    memberships = ev.fold_identities(table, distance, folds)
    require(len(np.unique(folds['m49_subregion_lo'][0])) == 20 and len(np.unique(folds['random10'][0])) == 10,
            'Fixed fold counts changed')
    return {'fixed_input_sha256': pins, 'representation_identity_max_abs_error': error, **memberships}


def m1a_support(directory, table, design):
    directory = Path(directory)
    manifest = json.loads((directory / 'm1a_measurement_manifest.json').read_text())
    require(manifest.get('all_features_pass_gates') is True, 'M1a measurement gates failed')
    require(manifest.get('gate_failures') == [] and manifest.get('quadrature_failures') == [], 'M1a unresolved feature/quadrature gates')
    features = pd.read_csv(directory / 'm1a_geography_features.csv', float_precision='round_trip')
    qa = pd.read_csv(directory / 'm1a_geography_qa.csv', float_precision='round_trip')
    for frame in [features, qa]:
        require(frame.iso3.tolist() == table.iso3.tolist() and frame.Country.tolist() == table.Country.tolist(), 'M1a exact identity/order mismatch')
    require(features.spatial_block.tolist() == design.spatial_block.tolist(), 'M1a changed spatial blocks')
    require(not features[list(geo.FEATURES)].isna().any().any(), 'Missing M1a features')
    require(np.isfinite(features[['abs_latitude', 'elevation', 'continentality']].to_numpy(float)).all(), 'Non-finite M1a features')
    sources = manifest['sources']
    fixed = json.loads((ROOT / 'research/model_v2/outputs/m1a_measurement_manifest.json').read_text())['sources']
    for key in ['sha256', 'lookup_sha256']:
        require(sources['gpw_national_identifier'][key] == fixed['gpw_national_identifier'][key], 'M1a GPW/lookup support provenance mismatch')
    require(sources['territory_snapshot_sha256'] == sha(territory.SNAPSHOT_PATH) == fixed['territory_snapshot_sha256'], 'M1a footprint snapshot provenance mismatch')
    with np.load(territory.SNAPSHOT_PATH) as snapshot:
        require(all(int(count) == len(snapshot[code]) for code, count in zip(qa.iso3, qa.gpw_cells)), 'M1a GPW support cell counts differ from fixed footprints')
    require(bool(qa.continentality_quadrature_ok.all()), 'M1a distance quadrature gate failed')
    return features, {'all_passed': True, 'country_n': len(table), 'manifest_sha256': sha(directory / 'm1a_measurement_manifest.json')}


def receipt_identity(directory, manifest):
    """Current archived execution, replacing only the historical Git-output stamp."""
    directory = Path(directory)
    started = json.loads((directory / 'started.json').read_text())
    identity = manifest.get('execution_receipt')
    require(isinstance(identity, dict) and identity.get('started_sha256') == sha(directory / 'started.json'), 'Missing or mismatched current preexecution receipt')
    source = str(Path(hydro.__file__).resolve())
    require(started['code_sha256'].get(source) == manifest['code_sha256'], 'Measurement kernel is not bound to archived preexecution code')
    archived = directory / 'execution-source' / Path(source).relative_to(ROOT)
    require(archived.is_file() and sha(archived) == manifest['code_sha256'], 'Archived measurement kernel bytes mismatch')
    require(identity.get('historical_commit_stamp_replaced') is True, 'Current receipt identity must explicitly replace historical commit stamp')
    return {'started_sha256': identity['started_sha256'], 'kernel_sha256': manifest['code_sha256']}


def m1b_package(directory):
    """Original manifest/package gates with honest current execution identity."""
    directory = Path(directory)
    manifest = json.loads((directory / hydro.MANIFEST).read_text())
    types = {'gate_version': str, 'all_hard_stops_pass': bool, 'hard_stops': list, 'git': dict,
             'code_sha256': str, 'contract_freeze_commit': str, 'support_inputs': dict,
             'artifact_sha256': dict, 'window': list, 'n_months': int, 'coverage_min': float,
             'amendments': list, 'formula': str, 'execution_receipt': dict}
    for key, kind in types.items():
        require(type(manifest.get(key)) is kind, f'M1b manifest type mismatch: {key}')
    require(manifest['gate_version'] == hydro.GATE_VERSION and manifest['contract_freeze_commit'] == hydro.CONTRACT_COMMIT, 'M1b rule version mismatch')
    require(manifest['all_hard_stops_pass'] is True and manifest['hard_stops'] == [], 'M1b measurement hard stop')
    require(manifest['window'] == list(hydro.WINDOW) and manifest['n_months'] == hydro.N_MONTHS, 'M1b fixed window mismatch')
    require(manifest['coverage_min'] == hydro.COVERAGE_MIN and manifest['amendments'] == list(hydro.AMENDMENTS), 'M1b coverage/amendments mismatch')
    require(manifest['contract_sha256'] == sha(HYDRO_CONTRACT), 'M1b contract hash mismatch')
    require(manifest.get('cru_version') == hydro.CRU_VERSION and manifest.get('cru_run_id') == hydro.CRU_RUN_ID, 'M1b CRU release/run mismatch')
    for variable, spec in hydro.FILES.items():
        source = manifest.get('sources', {}).get(variable, {})
        for key in ['gz_sha256', 'nc_sha256', 'nc_bytes', 'units']:
            require(source.get(key) == spec[key], f'M1b source metadata mismatch: {variable}/{key}')
        require(source.get('gz_bytes') == spec['bytes'], f'M1b source compressed size mismatch: {variable}')
    require(sorted(manifest['artifact_sha256']) == sorted(hydro.PACKAGE_ARTIFACTS), 'M1b exact package artifact set mismatch')
    for name in hydro.PACKAGE_ARTIFACTS:
        hydro.read_verified(directory, name, manifest)
    require(manifest['support_inputs']['m1a_cell_land_area_sha256'] == sha(directory / 'm1a_cell_land_area_km2.npz')
            and manifest['support_inputs']['m1a_geography_qa_sha256'] == sha(directory / 'm1a_geography_qa.csv'), 'M1b fresh M1a support digests mismatch')
    features = pd.read_csv(directory / hydro.FEATURES, float_precision='round_trip')
    support = pd.read_csv(directory / 'm1a_geography_qa.csv', usecols=['iso3', 'Country'])
    require(features.columns.tolist() == ['iso3', 'Country', b.FEATURE], 'M1b feature columns mismatch')
    require(features[['iso3', 'Country']].equals(support), 'M1b exact country identity/order mismatch')
    country_digest = hashlib.sha256(('\n'.join(features.iso3) + '\n').encode()).hexdigest()
    require(country_digest == manifest['support_inputs']['country_list_sha256'], 'M1b ordered country digest mismatch')
    require(np.isfinite(features[b.FEATURE].to_numpy(float)).all(), 'M1b non-finite C2')
    qa = pd.read_csv(directory / 'm1b_hydroclimate_qa.csv', float_precision='round_trip')
    require(hydro.hard_stops(qa, features.iso3.tolist()) == [], 'Fresh M1b QA hard stop')
    record = receipt_identity(directory, manifest)
    return features, {'all_passed': True, 'manifest_sha256': sha(directory / hydro.MANIFEST), **record}


def redundancy(predictors, features, package_record):
    require(predictors.columns.tolist() == red.PREDICTOR_COLUMNS and OUTCOME_COL not in predictors, 'Redundancy inputs must be predictor-only allowlist')
    require(predictors.Country.tolist() == features.Country.tolist(), 'Redundancy row identity mismatch')
    c2 = features[b.FEATURE].to_numpy(float)
    result = red.diagnose(predictors, c2)
    require(result['hard_stops'] == [], 'Predictor-only C2 redundancy hard stop')
    require(result['design_rank'] == 20 and result['rank_with_c2'] == 21, 'M1b predictor design must have ranks 20/21')
    result['provenance'] = {'measurement_manifest_sha256': package_record['manifest_sha256'],
                            'predictor_frame_sha256': red.frame_sha256(predictors),
                            'c2_vector_sha256': b.array_digest(c2),
                            'predictor_columns': red.PREDICTOR_COLUMNS, 'outcome_column_present': False}
    result['status'], result['all_hard_stops_pass'] = 'pass', True
    return result


def scored_integrity(scored, *, expected_columns=None, c2=False):
    card = scored.card
    require(card['matrix_columns'] == card['matrix_rank'], 'Full-sample design rank deficiency')
    if expected_columns is not None:
        require(card['matrix_columns'] == expected_columns, 'Unexpected fixed full-sample rank')
    require(card['training_fits']['all_full_rank'] is True, 'Primary training-fit rank deficiency')
    if c2:
        require(card['training_fits']['feature_identified_in_all'] is True, 'C2 not identified in every primary training fit')
    require(card['share_accounting']['passes'] is True, 'Named-share accounting failed')
    shares = [value for key, value in card.items() if key.startswith('share_') and isinstance(value, (int, float))]
    require(np.isfinite([*shares, card['residual_share']]).all() and abs(sum(shares) + card['residual_share'] - 1) <= b.SHARE_ATOL,
            'Recomputed named-share accounting failed')


def m2_integrity(scored):
    require(scored.card['structural_integrity']['passes'] is True, 'M0/M2 structural integrity failed')
    require(scored.card['share_accounting']['passes'] is True, 'M0/M2 share accounting failed')


def equivalent(arms):
    result = b.equivalence(arms)
    require(result['passes'] is True, 'M1b representation equivalence failed at original tolerances')
    return result
