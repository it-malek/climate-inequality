"""Conditional M2 robustness arms, or their verified non-execution (``M2_EVALUATOR_SPEC.md`` §8).

Authorized only by a committed, pushed, digest-verified primary M2 result that records predictive support
(promoted or not). Two arms, each refitting M0* and M2 in both representations under every protocol:

* ``m1a_area_geography``   - the frozen M1a area measurements replace the five station geography features;
  every fit's latitude state is recomputed from M1a ``abs_latitude`` and the interaction uses M1a hemisphere;
* ``aligned_era5_outcome`` - the committed preprocessing-aligned ERA5 outcome replaces Berkeley; predictors
  are unchanged, so every fit's latitude state must equal the primary run's.

Without predictive support the runner writes its non-execution record and returns before any arm frame is
constructed or any model is fitted. Arms are descriptive and cannot change the primary label.

Run with ``PYTHONHASHSEED=0 uv run python -m research.model_v2.m2_conditional [--primary DIR] [--out DIR]``.
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

from research.model_v2 import m2_evaluate as ev
from research.model_v2 import m2_feasibility as feas
from research.model_v2 import m2_latitude_basis as lb
from research.model_v2 import v2_provenance as prov
from src.decomposition import OUTCOME_COL

ROOT = prov.ROOT
M1A_FEATURES = f'{ev.OUTPUTS}/m1a_geography_features.csv'
M1A_MANIFEST = f'{ev.OUTPUTS}/m1a_measurement_manifest.json'
ERA5_TRENDS = f'{ev.OUTPUTS}/product_stability_aligned_era5_trends.csv'
ERA5_SUMMARY = f'{ev.OUTPUTS}/product_stability_aligned_summary.json'
ERA5_COLUMN = 'trend_c_per_decade_era5_area'
ARMS = ('m1a_area_geography', 'aligned_era5_outcome')
DEFAULT_PRIMARY = ev.DEFAULT_OUT
DEFAULT_OUT = ROOT / ev.OUTPUTS / 'm2_conditional'
NOT_RUN = 'm2_conditional_not_run.json'
DETERMINISTIC_ARTIFACTS = ('m2_conditional_scorecard.json', 'm2_conditional_predictions.csv',
                           'm2_conditional_fit_states.csv', 'm2_conditional_coefficients.csv',
                           'm2_conditional_provenance.json')
RESULT_MANIFEST = 'm2_conditional_manifest.json'
RUN_METADATA = 'm2_conditional_run_metadata.json'


def verified_primary(primary_dir) -> dict:
    """Fail closed unless the primary result is committed, pushed, digest-verified and integrity-valid."""
    primary_dir = Path(primary_dir)
    manifest = prov.verify_manifest(primary_dir, ev.RESULT_MANIFEST)
    if sorted(manifest['artifact_sha256']) != sorted(ev.DETERMINISTIC_ARTIFACTS):
        raise prov.ExecutionRefused('the primary manifest does not record exactly the result artifacts')
    if manifest.get('integrity_passed') is not True:
        raise prov.ExecutionRefused('the primary result did not pass its integrity checks')
    try:
        paths = [str((primary_dir / n).resolve().relative_to(ROOT))
                 for n in (*ev.DETERMINISTIC_ARTIFACTS, ev.RESULT_MANIFEST, ev.RUN_METADATA)]
    except ValueError:
        raise prov.ExecutionRefused('the primary result is outside the repository') from None
    pushed = prov.tracked_and_pushed(paths)
    evaluator_commit = manifest.get('evaluator_commit')
    provenance = json.loads((primary_dir / 'm2_provenance.json').read_text())
    if not isinstance(evaluator_commit, str) or provenance['code']['commit'] != evaluator_commit:
        raise prov.ExecutionRefused('the primary result does not record a consistent evaluator commit')
    recorded = provenance['code']['code_closure_sha256']
    current = {p: prov.sha256(ROOT / p) for p in ev.code_files()}
    if recorded != current:
        changed = sorted(set(recorded.items()) ^ set(current.items()))
        raise prov.ExecutionRefused(f'the M2 code closure changed after the primary result: {changed}')
    scorecard = json.loads((primary_dir / 'm2_scorecard.json').read_text())
    verdict = scorecard['representations']['primary_total_co2']['verdict']
    if verdict.get('verdict') != manifest.get('verdict') or verdict.get('verdict') is None:
        raise prov.ExecutionRefused('the primary verdict is missing or differs from its manifest')
    return {'primary_dir': str(primary_dir.relative_to(ROOT) if primary_dir.is_absolute() else primary_dir),
            'verdict': verdict, 'evaluator_commit': evaluator_commit, 'head': pushed['commit'],
            'predictive_support': verdict['verdict'] in ev.SUPPORT_LABELS,
            'artifact_sha256': manifest['artifact_sha256'],
            'manifest_sha256': prov.sha256(primary_dir / ev.RESULT_MANIFEST)}


def not_run_record(primary) -> dict:
    return {'status': 'not executed', 'reason': f"primary label {primary['verdict']['verdict']!r} does not record "
                                                'predictive support (P1 and P2)',
            'rule': 'M2_EVALUATION_CONTRACT.md §10 and M2_PRE_SCORE_RESOLUTIONS.md §8: the M1a-geography and '
                    'aligned-ERA5 replications run only under predictive support; no arm was constructed or fitted',
            'primary_result': primary}


def m1a_frames(frozen):
    features = feas.read_table_columns(ROOT / M1A_FEATURES, None, feas.M1A_COLUMNS)
    manifest = json.loads((ROOT / M1A_MANIFEST).read_text())
    ev._require(manifest.get('all_features_pass_gates') is True, 'the M1a manifest does not record passing gates')
    frames = {}
    for representation, frame in frozen.frames.items():
        arm = feas.m1a_frame(frame, features)
        frames[representation] = {ev.COMPARATOR: arm, ev.CANDIDATE: arm}
    return frames


def era5_outcome(frozen):
    summary = json.loads((ROOT / ERA5_SUMMARY).read_text())
    ev._require(prov.sha256(ROOT / ERA5_TRENDS) == summary['aligned_outcome_sha256'],
                'the aligned ERA5 file differs from the aligned summary digest')
    trends = pd.read_csv(ROOT / ERA5_TRENDS, float_precision='round_trip').set_index('iso3')
    y = trends.loc[frozen.table.iso3, ERA5_COLUMN].to_numpy(float)
    ev._require(np.isfinite(y).all(), 'a country has a non-finite aligned ERA5 outcome')
    return y, float(summary['r2']['era5'])


def arm_states_equal_primary(arm_scored, primary_dir) -> dict:
    primary = pd.read_csv(Path(primary_dir) / 'm2_fit_states.csv', dtype={'fit': str, 'latitude_state': str},
                          keep_default_na=False)
    primary = primary[primary.model == ev.CANDIDATE].set_index(['representation', 'protocol', 'fit'])
    compared = 0
    for representation, scored in arm_scored.items():
        fits = [('full_sample', 'all', scored.full)]
        fits += [(r['protocol'], r['fit'], r['fitted']) for p in ev.PROTOCOLS for r in scored.records[p]]
        for protocol, fit, fitted in fits:
            saved = lb.LatitudeState.from_json(primary.loc[(representation, protocol, fit), 'latitude_state'])
            ev._require(saved == fitted.state, f'{representation} {protocol} {fit}: ERA5 state differs from primary')
            compared += 1
    return {'fits_compared': compared, 'identical_to_primary_states': True}


class ArmIntegrityFailure(RuntimeError):
    """A software, reference or equivalence defect: not the allowed non-computable disposition; it stops the run."""


def _structural(arms):
    return [f'{r} {m}: structural integrity fails' for r, a in arms.items() for m, card in a['cards'].items()
            if not card['structural_integrity']['passes']]


def run_arm(name, frozen, frames, *, outcome=None, reference=None, primary_dir=None):
    """Score one arm. Only a structural failure is recorded as non-computable; other defects raise."""
    arm_frozen = frozen if outcome is None else dataclasses.replace(frozen, y=outcome)
    if outcome is not None:
        frames = {r: {m: f.assign(**{OUTCOME_COL: outcome}) for m, f in by.items()} for r, by in frames.items()}
    non_computable = {'arm': name, 'status': 'non_computable',
                      'rule': 'a contract-defined structural failure in a conditional arm is recorded without repair '
                              'and cannot affect the primary label'}
    arms = {}
    try:
        for representation in ev.REPRESENTATIONS:
            scored = ev.score_model(ev.COMPARATOR, frames[representation][ev.COMPARATOR], arm_frozen)
            arms[representation] = {'frames': frames[representation], 'scored': {ev.COMPARATOR: scored},
                                    'cards': {ev.COMPARATOR: scored.card}}
        if _structural(arms):
            return {**non_computable, 'reason': _structural(arms)}, None
        for representation, arm in arms.items():
            try:
                arm['reference'] = ev.reference_check(arm['cards'][ev.COMPARATOR], reference[representation]['card'],
                                                      f'{name} {representation} M0*', reference[representation]['metrics'])
            except prov.ExecutionRefused as error:
                raise ArmIntegrityFailure(str(error)) from None
        integrity = ev.score_candidates(arm_frozen, arms)
    except lb.DegenerateLatitudeBasis as error:
        return {**non_computable, 'reason': f'DegenerateLatitudeBasis: {error}'}, None
    if _structural(arms):
        return {**non_computable, 'reason': _structural(arms)}, None
    if integrity:
        raise ArmIntegrityFailure(f'{name}: {integrity}')
    states = (arm_states_equal_primary({r: a['scored'][ev.CANDIDATE] for r, a in arms.items()}, primary_dir)
              if outcome is not None else None)
    equiv = ev.equivalence(arms)
    if not equiv['passes']:
        raise ArmIntegrityFailure(f'{name}: representation equivalence outside the prespecified tolerances')
    entry = {'arm': name, 'status': 'computed', 'representation_equivalence': equiv,
             'latitude_states_vs_primary': states,
             'representations': {r: {'comparator_reference': a['reference'], 'M0star': a['cards'][ev.COMPARATOR],
                                     'M2': a['cards'][ev.CANDIDATE], 'paired_primary_comparison': a['paired'],
                                     'arm_conditions': {**ev.non_deciding_conditions(a['verdict']),
                                                        'note': 'descriptive; cannot change the primary label'},
                                     'diagnostics': a['diagnostics']} for r, a in arms.items()},
             'identity': {'outcome_vector_sha256': prov.array_digest(arm_frozen.y),
                          'predictor_frames_sha256': {r: prov.array_digest(
                              frames[r][ev.CANDIDATE][feas.NUMERIC_COLUMNS[r]].to_numpy(float)) for r in frames}}}
    return entry, arms


def main(argv=None):
    parser = argparse.ArgumentParser(description='Conditional M2 robustness arms or their non-execution.')
    parser.add_argument('--primary', default=str(DEFAULT_PRIMARY))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat(timespec='seconds')
    prov.require_hash_seed()
    code = prov.frozen_code_state(ev.code_files())
    out_dir = prov.require_fresh(args.out)
    primary = verified_primary(Path(args.primary))
    if not primary['predictive_support']:
        out_dir.mkdir(parents=True)
        prov.write_json(not_run_record(primary), out_dir / NOT_RUN)
        prov.write_json({'started_utc': started_utc, 'code_commit': code['commit'],
                         'live_remote_commit': code['live_remote_commit'], 'executable': sys.executable,
                         'argv': sys.argv}, out_dir / RUN_METADATA)
        print(prov.json_text({'status': 'not executed', 'label': primary['verdict']['verdict']}))
        return {'integrity_passed': True, 'status': 'not executed'}
    logging.getLogger('src.feature_schema').setLevel(logging.ERROR)
    digests = ev.verify_pins()
    frozen = ev.build_frozen()
    m1a_cards = json.loads((ROOT / ev.M1A_SCORECARD).read_text())
    m1a_reference = {'primary_total_co2': {'card': m1a_cards['primary']['M1a'], 'metrics': ev.REFERENCE_METRICS},
                     'per_capita': {'card': m1a_cards['percapita_sensitivity']['M1a'], 'metrics': ev.REFERENCE_METRICS}}
    y_era5, r2_era5 = era5_outcome(frozen)
    era5_reference = {r: {'card': {'in_sample_r2': r2_era5}, 'metrics': ('in_sample_r2',)} for r in ev.REPRESENTATIONS}
    entries, all_arms = {}, {}
    entries['m1a_area_geography'], all_arms['m1a_area_geography'] = run_arm(
        'm1a_area_geography', frozen, m1a_frames(frozen), reference=m1a_reference)
    entries['aligned_era5_outcome'], all_arms['aligned_era5_outcome'] = run_arm(
        'aligned_era5_outcome', frozen, ev.representation_frames(frozen), outcome=y_era5,
        reference=era5_reference, primary_dir=args.primary)
    era5 = entries['aligned_era5_outcome']
    primary_support = bool(primary['verdict']['predictive_support'])
    era5_support = (era5['status'] == 'computed'
                    and bool(era5['representations']['primary_total_co2']['arm_conditions']['predictive_support']))
    result = {'primary_result': primary, 'arms': entries,
              'product_sensitive': bool(primary_support and not era5_support) if era5['status'] == 'computed' else None,
              'rule': 'descriptive replications; neither arm changes the primary label',
              'integrity_passed': True}
    predictions, states, coefficients = [], [], []
    for name, arms in all_arms.items():
        if arms is None or entries[name]['status'] != 'computed':
            continue
        arm_frozen = frozen if name != 'aligned_era5_outcome' else dataclasses.replace(frozen, y=y_era5)
        predictions.append(ev.prediction_rows(arms, arm_frozen).assign(arm=name))
        states.append(ev.state_rows(arms).assign(arm=name))
        coefficients.append(ev.coefficient_rows(arms).assign(arm=name))
    empty = pd.DataFrame({'arm': []})
    out_dir.mkdir(parents=True)
    prov.write_json(result, out_dir / 'm2_conditional_scorecard.json')
    prov.write_csv(pd.concat(predictions, ignore_index=True) if predictions else empty,
                   out_dir / 'm2_conditional_predictions.csv')
    prov.write_csv(pd.concat(states, ignore_index=True) if states else empty, out_dir / 'm2_conditional_fit_states.csv')
    prov.write_csv(pd.concat(coefficients, ignore_index=True) if coefficients else empty,
                   out_dir / 'm2_conditional_coefficients.csv')
    prov.write_json({'code': {k: v for k, v in code.items() if k != 'live_remote_commit'}, 'inputs': digests,
                     'identity': frozen.identity, 'software': ev.software()},
                    out_dir / 'm2_conditional_provenance.json')
    prov.write_json({'result': 'M2 conditional robustness arms', 'evaluator_commit': code['commit'],
                     'primary_manifest_sha256': primary['manifest_sha256'],
                     'arm_status': {n: e['status'] for n, e in entries.items()},
                     'artifact_sha256': {n: prov.sha256(out_dir / n) for n in DETERMINISTIC_ARTIFACTS},
                     'non_deterministic_artifacts': [RUN_METADATA]}, out_dir / RESULT_MANIFEST)
    prov.write_json({'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                     'wall_seconds': time.perf_counter() - started, 'host': platform.node(),
                     'executable': sys.executable, 'argv': sys.argv, 'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
                     'live_remote_commit': code['live_remote_commit']}, out_dir / RUN_METADATA)
    print(prov.json_text({'arms': {n: e['status'] for n, e in entries.items()},
                          'product_sensitive': result['product_sensitive']}))
    return result


if __name__ == '__main__':
    outcome = main()
    sys.exit(0 if outcome.get('integrity_passed') is True else 3)
