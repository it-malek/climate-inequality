"""Record of the stopped one-ring harmonization C2 build (hard stop: a non-structural invalid support cell).

Read-only. Compares two independent harmonized builds byte-for-byte, checks exact invariants against
the preserved original-rule build, summarises coverage and harmonization QA, and reads the raw CRU
values of every non-structural support cell. Reads only C2 build outputs and the pinned CRU files;
no warming outcome, residual or score is read, and no alternative rule is computed.

Run from the build commit, after build A (into outputs/) and build B (separate root)::

    python -m research.model_v2.feasibility.m1b_amendment1_stop BUILD_B

Moves build A's outputs into ``outputs/m1b_amendment1_stopped_build/`` and writes the stop record there.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from research.model_v2 import m1b_hydroclimate as h

ORIGINAL = h.OUT / 'm1b_feasibility_original_coverage'
STOPPED = h.OUT / 'm1b_amendment1_stopped_build'
FAILED_ORIGINALLY = ['BHS', 'PAN', 'PHL']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(a, b):
    return bool(np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True))


def raw_cell(i, k):
    """Full-record raw PRE, stn and PET facts for one CRU cell, and its neighbours' variable masks."""
    out = {'lat': float(h.lat_centre(i)), 'lon': float(h.lon_centre(k))}
    with netCDF4.Dataset(h.CRU_DIR / 'cru_ts4.10.1901.2025.pre.dat.nc') as nc:
        pre, stn = nc['pre'], nc['stn']
        pre.set_auto_maskandscale(False)
        stn.set_auto_maskandscale(False)
        record, counts = pre[:, i, k].astype(float), stn[228:588, i, k]
        window = record[228:588]
        out['pre'] = {'record_months': int(record.size), 'record_min': float(record.min()), 'record_max': float(record.max()),
                      'record_months_positive': int((record > 0).sum()), 'window_sum_mm': float(window.sum()),
                      'window_p_bar_mm_yr': float(window.sum() / 30), 'window_months_with_station': int((counts >= 1).sum()),
                      'window_max_station_count': int(counts.max())}
    with netCDF4.Dataset(h.CRU_DIR / 'cru_ts4.10.1901.2025.pet.dat.nc') as nc:
        pet = nc['pet']
        pet.set_auto_maskandscale(False)
        out['pet'] = {'record_fill_months': int((pet[:, i, k] == pet.getncattr('_FillValue')).sum()), 'record_months': int(pet.shape[0])}
    return out


def main(build_b):
    build_a = h.OUT
    files = {name: {'build_a': digest(build_a / name), 'build_b': digest(build_b / name)} for name in h.DETERMINISTIC_OUTPUTS}
    for record in files.values():
        record['byte_identical'] = record['build_a'] == record['build_b']
    if not all(r['byte_identical'] for r in files.values()):
        raise AssertionError(f'harmonized builds differ: {files}')
    manifest = json.loads((build_a / 'm1b_measurement_manifest.json').read_text())
    if manifest['all_hard_stops_pass']:
        raise AssertionError('this record is only for a stopped build')
    qa = pd.read_csv(build_a / 'm1b_hydroclimate_qa.csv')
    cells = pd.read_csv(build_a / 'm1b_harmonization_cells.csv', keep_default_na=False)
    original = pd.read_csv(ORIGINAL / 'm1b_hydroclimate_qa.csv')
    if qa.iso3.tolist() != original.iso3.tolist():
        raise AssertionError('row order differs from the original-rule build')
    untouched = qa.n_harmonized_cells.to_numpy() == 0
    invariants = {
        'total_area_equals_original': exact(qa.total_land_area_km2, original.terrestrial_area_km2),
        'native_area_equals_original_valid_area': exact(qa.native_valid_area_km2, original.valid_area_km2),
        'native_fraction_equals_original_coverage': exact(qa.native_fraction, original.coverage),
        'native_only_c2_equals_original_c2': exact(qa.c2_native_support_only, original.baseline_dryness),
        'c2_unchanged_where_nothing_harmonized': exact(qa.c2_harmonized[untouched], original.baseline_dryness[untouched]),
        'native_p_bar_and_e_bar_equal_original': exact(qa.p_bar_native_area_mean_mm_yr, original.p_bar_area_mean_mm_yr)
                                                 and exact(qa.e_bar_native_area_mean_mm_yr, original.e_bar_area_mean_mm_yr),
        'station_support_equals_original': all(exact(qa[c], original[c]) for c in
                                               ['pre_station_supported_share', 'pre_mean_station_count', 'pre_pure_climatology_share']),
        'support_checkpoint_byte_identical_to_original': digest(build_a / 'm1b_support_checkpoint.json')
                                                          == digest(ORIGINAL / 'm1b_support_checkpoint.json'),
        'non_native_cells_equal_original_invalid_cells': bool(np.array_equal(qa.n_harmonized_cells + qa.n_unresolved_cells,
                                                                             original.n_cru_cells - original.n_valid_cru_cells)),
        'native_valid_cells_global_equals_original': manifest['cells_native_valid_global']
                                                     == json.loads((ORIGINAL / 'm1b_measurement_manifest.json').read_text())['cells_valid_global'],
        'git_code_path_matches_build_commit': manifest['git']['code_path_matches_commit'],
    }
    if not all(invariants.values()):
        raise AssertionError(f'invariant failed: {invariants}')

    bad = cells[cells.reason == h.NON_STRUCTURAL]
    stop_cells = []
    for r in bad.drop_duplicates(['target_lat_index', 'target_lon_index']).itertuples():
        rows = bad[(bad.target_lat_index == r.target_lat_index) & (bad.target_lon_index == r.target_lon_index)]
        country = qa.set_index('iso3').loc[rows.iso3.tolist()]
        stop_cells.append({
            'lat_index': int(r.target_lat_index), 'lon_index': int(r.target_lon_index),
            'pre_state': r.pre_state, 'pet_state': r.pet_state,
            'countries': {c: {'land_area_km2': float(a), 'share_of_country_terrestrial_area': float(a / country.loc[c, 'total_land_area_km2']),
                              'country_coverage': float(country.loc[c, 'coverage'])}
                          for c, a in zip(rows.iso3, rows.target_land_area_km2.astype(float))},
            'raw': raw_cell(int(r.target_lat_index), int(r.target_lon_index)),
        })
    by_harmonized = qa.nlargest(10, 'harmonized_fraction')
    record = {
        'build_commit': manifest['git']['commit'], 'builder_sha256': manifest['code_sha256'],
        'hard_stops': manifest['hard_stops'],
        'determinism': {'files': files, 'runs': {name: json.loads((d / h.RUN_METADATA).read_text())
                                                 for name, d in (('build_a', build_a), ('build_b', build_b))}},
        'invariants_against_original_rule_build': invariants,
        'coverage': {'min': float(qa.coverage.min()), 'min_iso3': qa.iso3[qa.coverage.idxmin()],
                     'countries_below_0_98': qa.iso3[qa.coverage < h.COVERAGE_MIN].tolist(),
                     'countries_below_0_99': qa.loc[qa.coverage < 0.99, ['iso3', 'coverage']].values.tolist(),
                     'originally_failing': qa.set_index('iso3').loc[FAILED_ORIGINALLY, ['native_fraction', 'harmonized_fraction',
                                                                                         'unresolved_fraction', 'coverage']].to_dict('index'),
                     'largest_harmonized_fractions': by_harmonized[['iso3', 'harmonized_fraction']].values.tolist()},
        'harmonization': manifest['harmonization'],
        'variable_states_in_support': manifest['variable_states_in_support'],
        'non_structural_stop_cells': stop_cells,
        'warming_outcome_read': False,
    }
    STOPPED.mkdir(exist_ok=True)
    (STOPPED / 'm1b_amendment1_stop_record.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    for name in (*h.DETERMINISTIC_OUTPUTS, h.RUN_METADATA):
        shutil.move(str(build_a / name), str(STOPPED / name))
    shutil.copyfile(build_b / h.RUN_METADATA, STOPPED / 'm1b_build_run_b.json')
    print(json.dumps({k: record[k] for k in ['hard_stops', 'invariants_against_original_rule_build', 'coverage',
                                             'non_structural_stop_cells']}, indent=1))
    return record


if __name__ == '__main__':
    main(Path(sys.argv[1]))
