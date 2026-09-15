"""Verify two independent Amendment 3 C2 builds and freeze the predictor package (pre-outcome).

Reads only C2 build outputs, the preserved historical evidence directories and Git; no warming outcome,
residual or score is read and no C2 value is computed. Fails closed on any difference.

    python -m research.model_v2.feasibility.m1b_amendment3_package BUILD_COMMIT BUILD_A BUILD_B [--verify-only]

1. Checks both builds: passing gates, the build commit, byte-identical deterministic outputs, package digests.
2. Checks exact invariance against the stopped Amendment 1 build (bd518a0) and the original-rule build (c5ff9b5).
3. Copies build A's package into ``outputs/`` (refusing to overwrite anything) and writes
   ``outputs/m1b_package_record.json`` with the comparison evidence and both builds' run metadata.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.model_v2 import m1b_hydroclimate as h

OUT = h.OUT
STOPPED = OUT / 'm1b_amendment1_stopped_build'
ORIGINAL = OUT / 'm1b_feasibility_original_coverage'
RECORD = OUT / 'm1b_package_record.json'
REFERENCE_STOPPED_FEATURES_SHA256 = 'cf33ba7cfb87c2184a94df5e5686f0951adf692819af8de27329ea698d72e1f4'
REFERENCE_COVERAGE = {'BHS': 0.9887528054237732, 'PAN': 1.0, 'PHL': 0.9985162289023632, 'PER': 0.999982140608819}
PERU_CELL = (164, 200)
UNCHANGED_BYTES = ('m1b_hydroclimate_features.csv', 'm1b_hydroclimate_qa.csv', 'm1b_harmonization_cells.csv',
                   'm1b_support_checkpoint.json')
DECLARED_MANIFEST_CHANGES = {'amendment', 'amendments', 'gate_version', 'git', 'contract_sha256', 'support_inputs',
                             'hard_stop_rules', 'unresolved_support', 'hard_stops', 'all_hard_stops_pass',
                             'artifact_sha256', 'code_sha256'}
SUPPORT_INPUT_ADDED_KEYS = {'gpw_and_lookup_pins_asserted'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(a, b):
    return bool(np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_build(root, build_commit):
    manifest, manifest_sha = h.verify_package(root)
    require(manifest['git']['commit'] == build_commit, f'{root}: build commit {manifest["git"]["commit"]} != {build_commit}')
    require(manifest['code_sha256'] == hashlib.sha256(subprocess.run(
        ['git', 'show', f'{build_commit}:research/model_v2/m1b_hydroclimate.py'], cwd=h.m0.ROOT, capture_output=True,
        check=True).stdout).hexdigest(), f'{root}: builder hash differs from the build commit')
    for var, spec in h.FILES.items():
        src = manifest['sources'][var]
        require(src['gz_sha256'] == spec['gz_sha256'] and src['nc_sha256'] == spec['nc_sha256'], f'{root}: {var} source hash')
    require(manifest['support_inputs']['gpw_sha256'] == h.FROZEN_INPUT_SHA256[h.GPW_PATH], 'GPW hash')
    require(manifest['support_inputs']['gpw_national_identifier_lookup_sha256'] == h.FROZEN_INPUT_SHA256[h.GPW_NATID_LOOKUP_PATH],
            'lookup hash')
    return manifest, manifest_sha


def main(build_commit, build_a, build_b, install=True):
    build_a, build_b = Path(build_a), Path(build_b)
    require(subprocess.run(['git', 'merge-base', '--is-ancestor', build_commit, 'HEAD'], cwd=h.m0.ROOT).returncode == 0,
            'build commit is not an ancestor of HEAD')
    manifest, manifest_sha = check_build(build_a, build_commit)
    check_build(build_b, build_commit)
    files = {n: {'build_a': digest(build_a / n), 'build_b': digest(build_b / n)} for n in h.DETERMINISTIC_OUTPUTS}
    for r in files.values():
        r['byte_identical'] = r['build_a'] == r['build_b']
    require(all(r['byte_identical'] for r in files.values()), f'builds differ: {files}')
    runs = {name: json.loads((root / h.RUN_METADATA).read_text()) for name, root in (('build_a', build_a), ('build_b', build_b))}
    for name, run in runs.items():
        require(run['deterministic_output_sha256'] == {n: r[name] for n, r in files.items()}, f'{name} run metadata digests')

    qa = pd.read_csv(build_a / 'm1b_hydroclimate_qa.csv')
    features = pd.read_csv(build_a / h.FEATURES)
    cells = pd.read_csv(build_a / 'm1b_harmonization_cells.csv', keep_default_na=False)
    unresolved = pd.read_csv(build_a / 'm1b_unresolved_support.csv', keep_default_na=False)
    frozen = pd.read_csv(h.M1A_QA, usecols=['iso3', 'Country'])
    stopped_qa = pd.read_csv(STOPPED / 'm1b_hydroclimate_qa.csv')
    stopped_manifest = json.loads((STOPPED / h.MANIFEST).read_text())
    original = pd.read_csv(ORIGINAL / 'm1b_hydroclimate_qa.csv')

    population = {'ordered_151_countries_equal_frozen_record': features.iso3.tolist() == frozen.iso3.tolist()
                  and features.Country.tolist() == frozen.Country.tolist() and len(features) == 151}
    gates = {'all_hard_stops_pass': manifest['all_hard_stops_pass'] is True, 'hard_stops_empty': manifest['hard_stops'] == [],
             'min_coverage': float(qa.coverage.min()), 'min_coverage_iso3': qa.iso3[qa.coverage.idxmin()],
             'all_coverage_ge_0_98': bool((qa.coverage >= h.COVERAGE_MIN).all()),
             'reference_coverage_exact': {c: float(qa.set_index('iso3').loc[c, 'coverage']) for c in REFERENCE_COVERAGE}}
    require(all(population.values()) and gates['all_hard_stops_pass'] and gates['hard_stops_empty'] and gates['all_coverage_ge_0_98'],
            f'gates or population failed: {gates} {population}')
    for c, ref in REFERENCE_COVERAGE.items():
        require(abs(gates['reference_coverage_exact'][c] - ref) <= (0 if c != 'PER' else 1e-15), f'{c} coverage {gates["reference_coverage_exact"][c]}')

    peru_cells = cells[(cells.target_lat_index == PERU_CELL[0]) & (cells.target_lon_index == PERU_CELL[1])]
    peru_unresolved = unresolved[(unresolved.target_lat_index == PERU_CELL[0]) & (unresolved.target_lon_index == PERU_CELL[1])]
    na_fields = ['donor_lat_index', 'donor_lon_index', 'donor_lat', 'donor_lon', 'donor_distance_km', 'donor_c2',
                 'donor_countries', 'cross_border']
    peru = {'rows': int(len(peru_cells)), 'iso3': peru_cells.iso3.tolist(), 'status': peru_cells.status.tolist(),
            'reason': peru_cells.reason.tolist(), 'pre_state': peru_cells.pre_state.tolist(), 'pet_state': peru_cells.pet_state.tolist(),
            'donor_fields_all_na': bool((peru_cells[na_fields] == 'NA').all().all()),
            'land_area_km2': peru_cells.target_land_area_km2.astype(float).tolist(),
            'in_unresolved_support_table': int(len(peru_unresolved)),
            'n_non_structural_rows_globally': int((cells.reason == h.NON_STRUCTURAL).sum())}
    require(peru['rows'] == 1 and peru['iso3'] == ['PER'] and peru['status'] == ['unresolved'] and peru['reason'] == [h.NON_STRUCTURAL]
            and peru['pre_state'] == ['nonpositive_mean'] and peru['pet_state'] == ['masked'] and peru['donor_fields_all_na']
            and peru['in_unresolved_support_table'] == 1, f'Peru cell state changed: {peru}')

    stopped = {name: digest(build_a / name) == digest(STOPPED / name) for name in UNCHANGED_BYTES}
    stopped['stopped_features_reference_sha256'] = digest(STOPPED / h.FEATURES) == REFERENCE_STOPPED_FEATURES_SHA256
    stopped['qa_named_columns_exact'] = all(exact(qa[c], stopped_qa[c]) for c in stopped_qa.columns if c not in ('iso3', 'Country'))
    changed = {k for k in set(manifest) | set(stopped_manifest) if manifest.get(k) != stopped_manifest.get(k)}
    stopped['manifest_changes_within_declared_set'] = changed <= DECLARED_MANIFEST_CHANGES
    stopped['manifest_changed_keys'] = sorted(changed)
    support_new, support_old = manifest['support_inputs'], stopped_manifest['support_inputs']
    stopped['support_inputs_only_added_declared_keys'] = (set(support_new) - set(support_old) == SUPPORT_INPUT_ADDED_KEYS
                                                         and all(support_new[k] == support_old[k] for k in support_old))
    stopped['git_changes_limited_to_commit_and_amendment'] = {k for k in manifest['git'] if manifest['git'][k] != stopped_manifest['git'].get(k)} <= {'commit', 'contract_amendment_commit'}
    require(all(v for k, v in stopped.items() if k != 'manifest_changed_keys'), f'invariance vs stopped build failed: {stopped}')

    untouched = qa.n_harmonized_cells.to_numpy() == 0
    versus_original = {
        'native_only_c2_equals_original_c2': exact(qa.c2_native_support_only, original.baseline_dryness),
        'native_fraction_equals_original_coverage': exact(qa.native_fraction, original.coverage),
        'native_area_equals_original_valid_area': exact(qa.native_valid_area_km2, original.valid_area_km2),
        'total_area_equals_original': exact(qa.total_land_area_km2, original.terrestrial_area_km2),
        'c2_unchanged_where_nothing_harmonized': exact(qa.c2_harmonized[untouched], original.baseline_dryness[untouched]),
        'station_support_equals_original': all(exact(qa[c], original[c]) for c in
                                               ['pre_station_supported_share', 'pre_mean_station_count', 'pre_pure_climatology_share']),
        'support_checkpoint_byte_identical_to_original': digest(build_a / 'm1b_support_checkpoint.json') == digest(ORIGINAL / 'm1b_support_checkpoint.json'),
    }
    require(all(versus_original.values()), f'invariance vs original build failed: {versus_original}')

    unresolved_totals = {'rows': int(len(unresolved)),
                         'area_km2': float(unresolved.target_land_area_km2.astype(float).sum()),
                         'structural_area_km2': float(unresolved[unresolved.reason == h.STRUCTURAL].target_land_area_km2.astype(float).sum()),
                         'non_structural_area_km2': float(unresolved[unresolved.reason == h.NON_STRUCTURAL].target_land_area_km2.astype(float).sum()),
                         'manifest': manifest['unresolved_support']}
    require(unresolved_totals['rows'] == int((cells.status == h.UNRESOLVED).sum()) == manifest['unresolved_support']['rows'],
            'unresolved-support row count mismatch')
    require(np.isclose(unresolved_totals['area_km2'], qa.unresolved_area_km2.sum(), rtol=1e-12, atol=0), 'unresolved area mismatch')

    if not install:
        print(json.dumps({'gates': gates, 'population': population, 'peru': peru, 'unresolved_support': unresolved_totals,
                          'invariance_vs_stopped_amendment1_build': stopped,
                          'invariance_vs_original_rule_build': versus_original, 'files': files}, indent=1))
        return None
    for name in (*h.DETERMINISTIC_OUTPUTS, h.RUN_METADATA, 'm1b_build_run_b.json', RECORD.name):
        require(not (OUT / name).exists(), f'refusing to overwrite outputs/{name}')
    for name in h.DETERMINISTIC_OUTPUTS:
        shutil.copyfile(build_a / name, OUT / name)
    shutil.copyfile(build_a / h.RUN_METADATA, OUT / h.RUN_METADATA)
    shutil.copyfile(build_b / h.RUN_METADATA, OUT / 'm1b_build_run_b.json')
    installed = {n: digest(OUT / n) == files[n]['build_a'] for n in h.DETERMINISTIC_OUTPUTS}
    require(all(installed.values()), f'installed bytes differ: {installed}')
    h.verify_package(OUT)

    record = {'build_commit': build_commit, 'measurement_manifest_sha256': manifest_sha, 'gate_version': manifest['gate_version'],
              'source_sha256': {v: {'gz': s['gz_sha256'], 'nc': s['nc_sha256']} for v, s in manifest['sources'].items()},
              'artifact_sha256': manifest['artifact_sha256'],
              'determinism': {'files': files, 'runs': runs, 'independent_processes': True},
              'population': population, 'gates': gates, 'peru': peru, 'unresolved_support': unresolved_totals,
              'invariance_vs_stopped_amendment1_build': stopped, 'invariance_vs_original_rule_build': versus_original,
              'harmonization': manifest['harmonization'], 'installed_bytes_match_build_a': installed,
              'warming_outcome_read': False}
    RECORD.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: record[k] for k in ['gates', 'peru', 'unresolved_support', 'invariance_vs_stopped_amendment1_build',
                                             'invariance_vs_original_rule_build']}, indent=1))
    return record


if __name__ == '__main__':
    main(*sys.argv[1:4], install='--verify-only' not in sys.argv[4:])
