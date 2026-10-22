"""Assemble one public bundle from accepted prospective stage outputs.

Private receipts and bulk predictions remain outside app/data. This command does
not edit the historical bundle, push, or deploy anything.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pandas as pd

from src import snapshot
from src.data_io import round_floats
from src.validation_receipts import dump, sha

STAGES = ('forcings', 'core', 'city-validation', 'explain', 'comparisons', 'bootstrap-population',
          'era5', 'physical_attempt2', 'rolling', 'neighbor-order', 'interpolation-full',
          'ndgain-updated-input', 'vulnerability-updated', 'v2_public')


def load(path):
    return json.loads(Path(path).read_text())


def public_metadata(value, run):
    """Remove machine-local roots without modifying scientific values."""
    if isinstance(value, dict):
        return {key: public_metadata(item, run) for key, item in value.items()}
    if isinstance(value, list):
        return [public_metadata(item, run) for item in value]
    if isinstance(value, str):
        return value.replace(str(run) + '/', '')
    return value


def deterministic_diagnostics(explain, stability, run):
    explain, stability = deepcopy(explain), deepcopy(stability)
    acceptance = load(run / 'neighbor-order/summary.json')
    metadata = {'policy_id': 'identity-distance-ties-v1', 'neighbors': 8,
                'boundary_chord_tolerance': 1e-12, 'permutations': 199, 'seed': 0,
                'randomization': 'identity-indexed', 'zero_distance_records': 'retained',
                'scope': 'explain and stability residual diagnostics; fixed V2 model graphs unchanged'}

    def select(sample):
        if not acceptance[sample]['acceptance']['accepted']:
            raise ValueError('Neighbor invariance not accepted: ' + sample)
        table = pd.read_csv(run / 'neighbor-order' / f'{sample}-orders.csv')
        selected = table.loc[(table.policy != 'historical') & (table.order == 'identity_sorted')
                             & (table.randomization == 'identity')]
        if len(selected) != 1:
            raise ValueError('Ambiguous neighbor-policy result')
        return float(selected.iloc[0].I), float(selected.iloc[0].p)

    for spec in explain['city_model']['specs']:
        spec['moran_i'], spec['moran_p'] = select('city_' + spec['spec_name'])
        if spec['spec_name'] == 'full':
            # The page reads these convenience scalars as well as the spec rows.
            explain['city_model']['moran_i_full'] = spec['moran_i']
            explain['city_model']['moran_p_full'] = spec['moran_p']
    for name, summary in [('area_weighted', stability),
                          ('station_weighted', stability['sensitivity']['station_weighted'])]:
        spatial = summary['residual_spatial']
        spatial['morans_i'], spatial['p_value'] = select('country_' + name)
        spatial['method'] = 'centroid kNN, identity-distance-ties-v1'
        spatial['neighbor_policy'] = metadata
    explain['neighbor_policy'] = metadata
    stability['neighbor_policy'] = metadata
    return explain, stability


def accepted_stages(physical_stage='physical_attempt2', bootstrap_stage='bootstrap-population', v2_stage='v2_public'):
    for name in [physical_stage, bootstrap_stage, v2_stage]:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('accepted stage names must be simple confined names')
    return tuple(physical_stage if name == 'physical_attempt2' else bootstrap_stage if name == 'bootstrap-population' else v2_stage if name == 'v2_public' else name for name in STAGES)


def assemble(run, out, *, physical_stage='physical_attempt2', bootstrap_stage='bootstrap-population', v2_stage='v2_public'):
    accepted_stages(physical_stage, bootstrap_stage, v2_stage)
    run, out = Path(run), Path(out)
    bundle = out / 'app/data'
    bundle.mkdir(parents=True, exist_ok=False)
    mapping = {name: 'core/' + name for name in ['city_trends.parquet', 'city_anomalies.parquet',
                'trend_surface.parquet', 'country_inequality.parquet', 'stats.json']}
    mapping.update({name: 'city-validation/' + name for name in ['validation.parquet', 'validation_global.parquet']})
    mapping['explain_features.parquet'] = 'explain/explain_features.parquet'
    mapping.update({name: 'comparisons/' + name for name in [
        'inequality_summary.json', 'decomposition_summary.json',
        'coupling.parquet', 'coupling_summary.json', 'coupling_consumption.parquet',
        'coupling_consumption_summary.json', 'coupling_exposure.parquet',
        'coupling_exposure_summary.json', 'coupling_area.parquet', 'coupling_area_summary.json']})
    mapping['stability_summary.json'] = bootstrap_stage + '/stability_summary_complete_case.json'
    mapping.update({name: 'era5/' + name for name in ['era5_area_trends.parquet', 'era5_validation_summary.json']})
    mapping.update({name: 'vulnerability-updated/' + name for name in ['vulnerability_strata.parquet', 'vulnerability_summary.json']})
    mapping.update({name: physical_stage + '/' + name for name in ['physical_trajectory.parquet', 'physical_summary.json']})
    mapping.update({name: v2_stage + '/' + name for name in ['national_warming.parquet', 'residual_structure_summary.json']})
    if len(mapping) != 27:
        raise ValueError('Public bundle must have exactly 27 scientific assets')
    provenance = {}
    for name, relative in mapping.items():
        source = run / relative
        shutil.copy2(source, bundle / name)
        provenance[name] = {'source': relative, 'source_sha256': sha(source),
                            'mode': 'fresh calculation from verified inputs', 'derived_in_assembly': False}
    provenance['stability_summary.json']['mode'] = (
        'fresh complete-case country bootstrap; exactly equivalent completed block draws reused; '
        'fresh point/influence/residual fits retained')
    stats = load(bundle / 'stats.json')
    stats['validation'] = load(run / 'city-validation/validation_summary.json')
    explain, stability = deterministic_diagnostics(load(run / 'explain/explain_summary.json'),
                                                  load(bundle / 'stability_summary.json'), run)
    stats['explain'] = explain
    stats['interpolation']['evaluation_status'] = 'completed training-only evaluation'
    stats['interpolation']['surface_method'] = stats['interpolation'].pop('winner')
    stats['interpolation']['descriptive_lower_rmse'] = 'kriging in both fixed protocols'
    stats['interpolation']['validation'] = load(run / 'interpolation-full/summary.json')
    stats['interpolation']['interpretation'] = (
        'Matched training-only grouped and buffered spatial evaluation on previously examined data; '
        'explicit partial-sill parameter routing. Fixed IDW display surface was chosen before scores.')
    stats['source_location_note'] = (
        '3,510 city records occupy 1,366 source-coordinate groups. They are not independent '
        'weather-station measurements; source coordinates are shared and rounded.')
    physical = load(bundle / 'physical_summary.json')
    forcings_path = run / 'forcings/forcings.parquet'
    fitted_inputs = load(run / physical_stage / 'started.json')['input_sha256']
    forcings_hash = sha(forcings_path)
    if fitted_inputs.get(str(forcings_path.resolve())) != forcings_hash:
        raise ValueError('Forcings do not match the accepted physical fit input')
    physical['forcings_hash'] = forcings_hash
    physical['rolling_origin_validation'] = load(run / 'rolling/summary.json')
    vulnerability = load(bundle / 'vulnerability_summary.json')
    uncertainty = load(run / 'vulnerability-updated/uncertainty.json')
    # Country identities already ship in the strata table; private draw inventories stay private.
    for record in uncertainty['evaluations']:
        record.pop('countries', None)
    vulnerability['rank_uncertainty'] = uncertainty
    source = load(run / 'ndgain-updated-input/source-record.json')
    vulnerability['updated_source'] = {
        'analysis_id': source['analysis_id'], 'release_label': '2026 release_data_v2',
        'selected_observation_year': 2023, 'download_date': '2026-10-22',
        'archive_sha256': source['release']['sha256'],
        'source_page': source['release']['official_page'],
        'interpretation': 'Updated-source analysis at fixed observation year; not reconstruction of the unavailable historical archive.'}
    comparison = load(run / 'vulnerability-updated/vintage-comparison.json')
    vulnerability['updated_source_comparison'] = {key: comparison[key] for key in
        ['matched_complete_n', 'same_country_correlations']}
    vulnerability['criterion_interpretation'] = (
        'The retained Boolean criteria describe specified signs and permutation tests. '
        'Legacy field names containing not_lower are not equivalence or no-lower-warming evidence.')
    residual = load(bundle / 'residual_structure_summary.json')
    # Ship only the compact records needed to inspect the fresh V2 derivation.
    # Raw acquisition pins are input provenance, not supposedly bundled files.
    residual['input_sources'] = {'owid_iso_bridge': residual['sources'].pop('owid_iso_bridge')}
    for entry in residual['sources'].values():
        relative = entry['path']
        source_path = run / relative
        destination = Path('reports/current_validation') / relative
        if sha(source_path) != entry['sha256']:
            raise ValueError('V2 source digest mismatch: ' + relative)
        (out / destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, out / destination)
        entry['path'] = str(destination)
    rewrites = {'residual_structure_summary.json': residual, 'stats.json': stats, 'stability_summary.json': stability,
                'physical_summary.json': physical, 'vulnerability_summary.json': vulnerability}
    for name, value in rewrites.items():
        dump(bundle / name, round_floats(public_metadata(value, run)))
        provenance[name]['derived_in_assembly'] = True
    for name in mapping:
        path = bundle / name
        if path.suffix == '.json':
            text = path.read_text()
            if '/Users/' in text or '/private/tmp/' in text:
                raise ValueError('Private machine path in public asset: ' + name)
        provenance[name]['public_sha256'] = sha(path)
    release = snapshot.write_manifest(bundle, version='2.1.0')
    dump(out / 'asset-provenance.json', provenance)
    dump(out / 'candidate-status.json', {'status': 'local proposed release; not published',
        'version': '2.1.0', 'assets': 27, 'bundle_sha256': release['bundle_sha256'],
        'scientific_validation': 'current-public-scientific-validation-2026-10-22-v1',
        'source_update': 'ndgain-2026-v2-observation-2023-update-v1',
        'bootstrap_population': 'complete-case-country-bootstrap-v1'})
    return {'version': '2.1.0', 'bundle_sha256': release['bundle_sha256'], 'assets': 27}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--stage-name', default='candidate-bundle')
    parser.add_argument('--physical-stage', default='physical_attempt2')
    parser.add_argument('--v2-stage', default='v2_public')
    parser.add_argument('--bootstrap-stage', default='bootstrap-population')
    args = parser.parse_args()
    stages = accepted_stages(args.physical_stage, args.bootstrap_stage, args.v2_stage)
    run = args.run_root.resolve(strict=True)
    spec = importlib.util.spec_from_file_location('runlib', run / 'control/runlib.py')
    runlib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runlib)
    for name in stages:
        runlib.verified_stage(name)
    inputs = [run / name / 'completed.json' for name in stages]
    inputs.extend(path for name in stages for path in (run / name).glob('*')
                  if path.is_file() and path.suffix in {'.json', '.parquet', '.csv'})
    runlib.stage(args.stage_name, inputs, lambda out: assemble(run, out, physical_stage=args.physical_stage, bootstrap_stage=args.bootstrap_stage, v2_stage=args.v2_stage),
                 runner=__file__, dependencies=stages)


if __name__ == '__main__':
    main()
