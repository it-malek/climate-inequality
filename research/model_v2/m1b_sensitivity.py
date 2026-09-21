"""Conditional M1b robustness arms.

Three registered replications, each refitting **both** models under that arm's own inputs and sample
with the exact country C2 values of the primary run, and each reporting both representations:

* ``m1a_area_geography``   - the M1a area-consistent geography in place of station geography;
* ``aligned_era5_outcome`` - the committed preprocessing-aligned ERA5 outcome;
* ``station_support_146``  - exactly Saudi Arabia, Yemen, Haiti, Oman and Chad removed.

Execution, not merely publication, is conditional: :func:`verified_primary` refuses unless the
committed, pushed primary result verifies against its own manifest and records association support.
Results are descriptive and cannot change either verdict level. Run with
``PYTHONHASHSEED=0 python -m research.model_v2.m1b_sensitivity --primary DIR --out DIR``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from research.model_v2 import cv
from research.model_v2 import v2_provenance as prov
from research.model_v2 import m1b_evaluate as ev
from research.model_v2 import m1b_hydroclimate as hydro
from research.model_v2.m0 import OUTPUT_DIR, ROOT
from research.model_v2.spatial import haversine_matrix, knn_weights
from src.decomposition import OUTCOME_COL

M1A_FEATURES = 'research/model_v2/outputs/m1a_geography_features.csv'
M1A_SCORECARD = 'research/model_v2/outputs/m1a_scorecard.json'
M1A_MANIFEST = 'research/model_v2/outputs/m1a_measurement_manifest.json'
ERA5_TRENDS = 'research/model_v2/outputs/product_stability_aligned_era5_trends.csv'
ERA5_SUMMARY = 'research/model_v2/outputs/product_stability_aligned_summary.json'
ERA5_COLUMN = 'trend_c_per_decade_era5_area'
REMEASURED = ('abs_latitude', 'elevation', 'continentality', 'climate_zone', 'hemisphere')
EXCLUDED_ISO3 = ('SAU', 'YEM', 'HTI', 'OMN', 'TCD')  # the five lowest PRE station-supported shares
ARMS = ('m1a_area_geography', 'aligned_era5_outcome', 'station_support_146')
DEFAULT_PRIMARY = ev.DEFAULT_OUT
DEFAULT_OUT = OUTPUT_DIR / 'm1b_conditional'
DETERMINISTIC_ARTIFACTS = ('m1b_conditional_scorecard.json', 'm1b_conditional_predictions.csv',
                           'm1b_conditional_coefficients.csv', 'm1b_conditional_provenance.json')
RESULT_MANIFEST = 'm1b_conditional_manifest.json'
RUN_METADATA = 'm1b_conditional_run_metadata.json'
NOT_RUN = 'm1b_conditional_not_run.json'


# ---------------------------------------------------------------------
# Precondition: a verified, committed, pushed primary result
# ---------------------------------------------------------------------


def verified_primary(primary_dir):
    """Fail closed unless the primary result is committed, pushed and digest-verified."""
    primary_dir = Path(primary_dir)
    manifest_path = primary_dir / ev.RESULT_MANIFEST
    if not manifest_path.exists():
        raise RuntimeError(f'no primary result manifest at {manifest_path}')
    manifest = json.loads(manifest_path.read_bytes())
    recorded = manifest.get('artifact_sha256')
    if not isinstance(recorded, dict) or sorted(recorded) != sorted(ev.DETERMINISTIC_ARTIFACTS):
        raise RuntimeError('the primary result manifest does not record exactly the result artifacts')
    for name, digest in recorded.items():
        actual = hydro.sha256(primary_dir / name)
        if actual != digest:
            raise RuntimeError(f'primary artifact {name} does not match its recorded digest')
    if manifest.get('integrity_passed') is not True:
        raise RuntimeError('the primary result did not pass its own integrity checks')
    try:
        paths = [str((primary_dir / name).resolve().relative_to(ROOT))
                 for name in (*ev.DETERMINISTIC_ARTIFACTS, ev.RESULT_MANIFEST)]
    except ValueError:
        raise RuntimeError(f'the primary result at {primary_dir} is outside the repository, '
                           'so it cannot be a committed result') from None
    if ev._git('ls-files', '--error-unmatch', '--', *paths, check=False).returncode != 0:
        raise RuntimeError('the primary result is not committed to git')
    if ev._git('diff', '--quiet', 'HEAD', '--', *paths, check=False).returncode != 0:
        raise RuntimeError('the committed primary result differs from the files on disk')
    head = ev._git('rev-parse', 'HEAD').stdout.strip()
    upstream = ev._git('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}').stdout.strip()
    if ev._git('merge-base', '--is-ancestor', head, upstream, check=False).returncode != 0:
        raise RuntimeError(f'HEAD {head} is not contained in {upstream}; push the primary result first')
    evaluator_commit = manifest.get('evaluator_commit')
    if not isinstance(evaluator_commit, str) or len(evaluator_commit) != 40:
        raise RuntimeError('the primary result does not record its evaluator commit')
    if ev._git('diff', '--quiet', evaluator_commit, 'HEAD', '--', *ev.CODE_PATH,
               check=False).returncode != 0:
        raise RuntimeError('the evaluator code path changed after the primary result was produced')
    scorecard = json.loads((primary_dir / 'm1b_scorecard.json').read_bytes())
    provenance = json.loads((primary_dir / 'm1b_provenance.json').read_bytes())
    if provenance.get('code', {}).get('commit') != evaluator_commit:
        raise RuntimeError('the primary provenance commit differs from the result manifest')
    verdict = scorecard['representations']['primary_total_co2']['verdict']
    if verdict.get('verdict') != manifest.get('verdict'):
        raise RuntimeError('the primary manifest verdict differs from the scorecard verdict')
    if not isinstance(verdict.get('association_supported'), bool):
        raise RuntimeError('the primary verdict does not record association support')
    return {'primary_dir': str(primary_dir), 'manifest': manifest, 'verdict': verdict,
            'evaluator_commit': evaluator_commit, 'head': head, 'upstream': upstream,
            'primary_scorecard_sha256': hydro.sha256(primary_dir / 'm1b_scorecard.json'),
            'primary_manifest_sha256': hydro.sha256(manifest_path)}


# ---------------------------------------------------------------------
# Arm construction: the primary run's C2 values are never rebuilt or renormalized
# ---------------------------------------------------------------------


def _replace(frozen, **changes):
    return dataclasses.replace(frozen, **changes)


def arm_identity(arm, base, *, arm_name, **extra):
    """Identity digests of the arm's OWN frame, so no arm inherits the primary run's record."""
    design = arm.design
    predictors = design.drop(columns=[OUTCOME_COL, 'cum_co2_per_capita']).reset_index(drop=True)
    return {
        'arm': arm_name, 'n': arm.n,
        'inherited_from_primary': {k: base.get(k) for k in
                                   ('country_list_sha256', 'outcome_source_column')},
        'outcome_vector_sha256': ev.array_digest(design[OUTCOME_COL].to_numpy(float)),
        'c2_vector_sha256': ev.array_digest(arm.c2),
        'predictor_frame_sha256': ev.red.frame_sha256(predictors),
        'iso3_sha256': ev.hashlib.sha256(('\n'.join(arm.table.iso3) + '\n').encode()).hexdigest(),
        'primary_membership_sha256': ev.hashlib.sha256(np.array(
            [train for _f, _t, train in ev.cv.iter_folds(arm.folds[ev.PRIMARY][0], arm.distance,
                                                         ev.PRIMARY_BUFFER_KM)]).tobytes()).hexdigest(),
        'm49_fold_sha256': ev.hashlib.sha256(arm.folds['m49_subregion_lo'][0].tobytes()).hexdigest(),
        'random10_fold_sha256': ev.hashlib.sha256(arm.folds['random10'][0].tobytes()).hexdigest(),
        'station_weights_sha256': ev.array_digest(arm.station_w),
        'area_weights_sha256': ev.array_digest(arm.area_w),
        **extra,
    }


def m1a_geography_arm(frozen, package_dir=OUTPUT_DIR):
    """M1a area-consistent geography replaces station geography; nothing else moves."""
    features = pd.read_csv(ROOT / M1A_FEATURES)
    ev._require(features.iso3.tolist() == frozen.table.iso3.tolist()
                and features.Country.tolist() == frozen.table.Country.tolist(),
                'M1a feature rows do not match the fixed country order')
    manifest = json.loads((ROOT / M1A_MANIFEST).read_text())
    ev._require(manifest.get('all_features_pass_gates') is True,
                'the M1a measurement manifest does not record passing gates')
    ev._require((features.spatial_block.to_numpy() == frozen.design.spatial_block.to_numpy()).all(),
                'the M1a measurements carry a different spatial_block; it must be copied unchanged')
    design = frozen.design.copy()
    for feature in REMEASURED:
        ev._require(features[feature].notna().all(), f'missing M1a {feature}')
        design[feature] = features[feature].to_numpy()
    replaced = [f for f in REMEASURED if not design[f].equals(frozen.design[f])]
    ev._require(replaced, 'the M1a arm changed no geography column; the measurements are not in use')
    same = [c for c in frozen.design.columns if c not in REMEASURED]
    pd.testing.assert_frame_equal(design[same], frozen.design[same])
    cards = json.loads((ROOT / M1A_SCORECARD).read_text())
    reference = {'primary_total_co2': {'card': cards['primary']['M1a'], 'metrics': ev.REFERENCE_METRICS},
                 'per_capita': {'card': cards['percapita_sensitivity']['M1a'],
                                'metrics': ev.REFERENCE_METRICS}}
    arm = _replace(frozen, design=design)
    arm = _replace(arm, identity=arm_identity(arm, frozen.identity, arm_name='m1a_area_geography',
                                              m1a_features_sha256=hydro.sha256(ROOT / M1A_FEATURES)))
    return arm, reference, {
        'replaced_features': list(REMEASURED), 'source': M1A_FEATURES,
        'source_sha256': hydro.sha256(ROOT / M1A_FEATURES),
        'comparator_reference': 'm1a_scorecard.json M1a cards (both representations)'}


def aligned_era5_arm(frozen, package_dir=OUTPUT_DIR):
    """The committed preprocessing-aligned ERA5 outcome replaces Berkeley for both models."""
    trends = pd.read_csv(ROOT / ERA5_TRENDS)
    summary = json.loads((ROOT / ERA5_SUMMARY).read_text())
    digest = hydro.sha256(ROOT / ERA5_TRENDS)
    ev._require(summary.get('aligned_outcome_sha256') == digest,
                'the aligned ERA5 outcome does not match the digest recorded by the aligned summary')
    ev._require(not trends.iso3.duplicated().any(), 'duplicate ISO3 in the aligned ERA5 outcome')
    values = trends.set_index('iso3')[ERA5_COLUMN].reindex(frozen.table.iso3).to_numpy(float)
    ev._require(np.isfinite(values).all(),
                'a country has no finite aligned ERA5 outcome; no missing-country rule is created')
    design = frozen.design.copy()
    design[OUTCOME_COL] = values
    reference_r2 = float(summary['r2']['era5'])
    reference = {r: {'card': {'in_sample_r2': reference_r2}, 'metrics': ('in_sample_r2',)}
                 for r in ev.REPRESENTATIONS}
    arm = _replace(frozen, design=design)
    arm = _replace(arm, identity=arm_identity(arm, frozen.identity, arm_name='aligned_era5_outcome',
                                              outcome_source_column=ERA5_COLUMN,
                                              aligned_outcome_sha256=digest))
    return arm, reference, {
        'outcome_source': ERA5_TRENDS, 'outcome_column': ERA5_COLUMN, 'outcome_sha256': digest,
        'outcome_vector_sha256': ev.array_digest(values),
        'comparator_reference': 'product_stability_aligned_summary.json r2.era5',
        'note': 'committed aligned measurements, not a new reconstruction; every other input is held fixed'}


def station_support_arm(frozen, package_dir=OUTPUT_DIR):
    """Exactly the five lowest-station-support countries are removed; both models refit at n = 146."""
    checkpoint = json.loads((Path(package_dir) / 'm1b_support_checkpoint.json').read_text())
    lowest = [row[0] for row in checkpoint['extremes_top10']['lowest_station_supported_share'][:5]]
    ev._require(tuple(lowest) == EXCLUDED_ISO3,
                f'the five excluded countries {EXCLUDED_ISO3} are not the verified lowest support set {lowest}')
    keep = ~frozen.table.iso3.isin(EXCLUDED_ISO3).to_numpy()
    ev._require(int((~keep).sum()) == len(EXCLUDED_ISO3), 'the excluded ISO3 codes are not all present')
    index = np.flatnonzero(keep)
    table = frozen.table.loc[keep].reset_index(drop=True)
    design = frozen.design.loc[keep].reset_index(drop=True)
    distance = frozen.distance[np.ix_(index, index)]
    centroids = frozen.area_centroids[index]
    arm = _replace(
        frozen, design=design, table=table, c2=frozen.c2[keep], distance=distance,
        station_w=cv.v1_weights(table.station_lon.to_numpy(), table.station_lat.to_numpy()),
        area_centroids=centroids,
        area_w=knn_weights(haversine_matrix(centroids[:, 0], centroids[:, 1]), 8),
        folds=ev.fold_plan(table, distance, len(table)))
    arm = _replace(arm, identity=arm_identity(arm, frozen.identity, arm_name='station_support_146',
                                              excluded_iso3=list(EXCLUDED_ISO3)))
    return arm, None, {
        'excluded_iso3': list(EXCLUDED_ISO3), 'n': len(table),
        'source': 'm1b_support_checkpoint.json lowest PRE station-supported share (Amendment 2 A2.1)',
        'comparator': 'the 146-country comparator, never the 151-country comparator',
        'fold_construction': 'the same algorithm re-applied to the 146 rows; '
                             'paired resampling at n = 146, 2,000 resamples, seed 0'}


ARM_BUILDERS = {'m1a_area_geography': m1a_geography_arm,
                'aligned_era5_outcome': aligned_era5_arm,
                'station_support_146': station_support_arm}


# ---------------------------------------------------------------------
# Running the arms
# ---------------------------------------------------------------------


def arm_conditions(arm):
    """The same mechanical Booleans as the primary verdict, explicitly descriptive."""
    conditions = dict(arm['verdict'])
    label = conditions.pop('verdict', None)
    return {**conditions, 'arm_label_descriptive_only': label,
            'note': 'descriptive replication conditions; they cannot change either verdict level'}


def run_arms(frozen, names=ARMS, package_dir=OUTPUT_DIR):
    out, integrity = {}, []
    for name in names:
        arm_frozen, reference, description = ARM_BUILDERS[name](frozen, package_dir)
        arms, failures = ev.score_representations(
            arm_frozen, reference_cards=reference,
            expected_columns=ev.EXPECTED_COLUMNS if name != 'm1a_area_geography' else None)
        integrity.extend(f'{name}: {failure}' for failure in failures)
        equivalence = ev.equivalence(arms)
        if not equivalence['passes']:
            integrity.append(f'{name}: representation equivalence outside the prespecified tolerance')
        out[name] = {
            'description': description, 'n': arm_frozen.n, 'identity': arm_frozen.identity,
            'representations': {
                representation: {
                    'comparator_reference': entry['reference'],
                    'M0star': entry['cards'][ev.COMPARATOR], 'M1b': entry['cards'][ev.CANDIDATE],
                    'paired_primary_comparison': entry['paired'],
                    'arm_conditions': arm_conditions(entry), 'diagnostics': entry['diagnostics'],
                } for representation, entry in arms.items()},
            'frames': arms, 'frozen': arm_frozen,
        }
    return out, integrity


def product_sensitivity(primary_verdict, results):
    """A gain confined to Berkeley is flagged product-sensitive."""
    era5 = results.get('aligned_era5_outcome')
    if era5 is None:
        return None
    conditions = (era5.get('representations', {}).get('primary_total_co2', {})
                  .get('arm_conditions', {}))
    replicated = conditions.get('S1_interval_below_zero')
    if not isinstance(replicated, bool):
        return {'primary_association_supported': bool(primary_verdict['association_supported']),
                'aligned_era5_S1_replicated': None, 'product_sensitive': None,
                'reason': 'the aligned-ERA5 arm did not produce evaluable conditions'}
    return {'primary_association_supported': bool(primary_verdict['association_supported']),
            'aligned_era5_S1_replicated': replicated,
            'product_sensitive': bool(primary_verdict['association_supported'] and not replicated),
            'definition': 'the primary association holds on Berkeley while the aligned-ERA5 arm does '
                          'not satisfy S1 (delta RMSE < 0 with the paired interval entirely below zero)'}


def evidence_frames(results):
    predictions, coefficients = [], []
    for name, result in results.items():
        frozen = result['frozen']
        predictions.append(ev.prediction_rows(result['frames'], frozen).assign(arm=name))
        coefficients.append(ev.coefficient_rows(result['frames']).assign(arm=name))
    return pd.concat(predictions, ignore_index=True), pd.concat(coefficients, ignore_index=True)


def run(primary_dir=DEFAULT_PRIMARY, out_dir=DEFAULT_OUT, *, package_dir=OUTPUT_DIR):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise RuntimeError(f'{out_dir} already exists; a run never overwrites a prior run')
    authorization = verified_primary(primary_dir)
    verdict = authorization['verdict']
    if not verdict['association_supported']:
        out_dir.mkdir(parents=True, exist_ok=False)
        record = {
            'status': 'not run under the contract',
            'reason': 'contract §5 and A2.1 run the registered replications only if the linear '
                      'hydroclimate association is supported (S1 and S2); the committed primary '
                      'result does not record association support',
            'primary_verdict': verdict, 'authorization': authorization,
            'arms_not_run': list(ARMS),
            'no_favourable_sensitivity_search': True,
        }
        (out_dir / NOT_RUN).write_text(ev.json_text(record), encoding='utf-8')
        return record
    inputs = ev.verify_scoring_inputs(package_dir)
    frozen = ev.build_frozen(inputs)
    results, integrity = run_arms(frozen, package_dir=package_dir)
    predictions, coefficients = evidence_frames(results)
    out_dir.mkdir(parents=True, exist_ok=False)     # nothing is created before the arms have run
    scorecard = {
        'contract': ev.CONTRACT_PATH, 'specification': ev.SPEC_PATH,
        'status': 'conditional robustness arms executed after the committed primary result',
        'authorization': {k: v for k, v in authorization.items() if k != 'manifest'},
        'primary_verdict': verdict,
        'cannot_change_verdict': 'no sensitivity or representation changes either verdict level',
        'arms': {name: {k: v for k, v in result.items() if k not in ('frames', 'frozen')}
                 for name, result in results.items()},
        'product_sensitivity': product_sensitivity(verdict, results),
        'integrity_failures': integrity, 'integrity_passed': not integrity,
    }
    (out_dir / 'm1b_conditional_scorecard.json').write_text(ev.json_text(scorecard), encoding='utf-8')
    ev.write_csv(predictions, out_dir / 'm1b_conditional_predictions.csv')
    ev.write_csv(coefficients, out_dir / 'm1b_conditional_coefficients.csv')
    (out_dir / 'm1b_conditional_provenance.json').write_text(ev.json_text({
        'contract_sha256': prov.sha256(ROOT / ev.CONTRACT_PATH),
        'specification_sha256': prov.sha256(ROOT / ev.SPEC_PATH),
        'code': ev.frozen_code_state(), 'inputs': inputs.digests,
        'primary_result': {k: authorization[k] for k in
                           ('primary_dir', 'evaluator_commit', 'primary_scorecard_sha256',
                            'primary_manifest_sha256')},
        'arm_identities': {name: result['identity'] for name, result in results.items()},
        'software': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__,
                     'platform': platform.platform(), 'machine': platform.machine()},
    }), encoding='utf-8')
    (out_dir / RESULT_MANIFEST).write_text(ev.json_text({
        'result': 'M1b conditional robustness arms', 'arms': list(results),
        'evaluator_commit': authorization['evaluator_commit'],
        'primary_manifest_sha256': authorization['primary_manifest_sha256'],
        'integrity_passed': scorecard['integrity_passed'],
        'artifact_sha256': {name: hydro.sha256(out_dir / name) for name in DETERMINISTIC_ARTIFACTS},
        'non_deterministic_artifacts': [RUN_METADATA],
    }), encoding='utf-8')
    return scorecard


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the registered conditional M1b robustness arms.')
    parser.add_argument('--primary', default=str(DEFAULT_PRIMARY), help='committed primary result directory')
    parser.add_argument('--out', default=str(DEFAULT_OUT), help='fresh output root (never overwritten)')
    args = parser.parse_args(argv)
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat(timespec='seconds')
    ev.require_hash_seed()
    ev.frozen_code_state()
    logging.getLogger('src.feature_schema').setLevel(logging.ERROR)
    result = run(Path(args.primary), Path(args.out))
    (Path(args.out) / RUN_METADATA).write_text(ev.json_text({
        'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'wall_seconds': time.perf_counter() - started, 'host': platform.node(),
        'executable': sys.executable, 'argv': sys.argv, 'output_root': str(args.out),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
    }), encoding='utf-8')
    print(ev.json_text({'status': result.get('status'),
                        'integrity_failures': result.get('integrity_failures', []),
                        'product_sensitivity': result.get('product_sensitivity')}))
    return result


if __name__ == '__main__':
    outcome = main()
    sys.exit(0 if outcome.get('integrity_passed', True) else 3)
