"""Conditional M1b robustness arms: authorization, arm construction and descriptive reporting.

Execution is deferred, not merely publication: every refusal below must fire before any arm is
scored. Scored fixtures are synthetic except where a test explicitly uses a committed frozen artifact.
"""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from research.model_v2 import m1b_evaluate as ev
from research.model_v2 import m1b_hydroclimate as hydro
from research.model_v2 import m1b_sensitivity as sens
from research.model_v2.m0 import DEFAULT_FEATURES_PATH, DEFAULT_INEQUALITY_PATH, ROOT
from research.model_v2.tests.test_m1b_evaluator import synthetic_frozen

REAL_INPUTS = DEFAULT_INEQUALITY_PATH.exists() and DEFAULT_FEATURES_PATH.exists()
needs_inputs = pytest.mark.skipif(not REAL_INPUTS, reason='requires the local frozen inputs')


def fake_primary(tmp_path, *, association=True, promoted=False, integrity=True):
    """A digest-consistent primary result directory (never committed, so authorization must refuse)."""
    primary = tmp_path / 'm1b_primary'
    primary.mkdir(parents=True)
    verdict = {'verdict': 'supported and promoted' if promoted else
               ('supported but sub-material / not promoted' if association else 'not supported'),
               'association_supported': association, 'promoted': promoted,
               'S1_interval_below_zero': association, 'S2_sign_negative_and_stable': association}
    scorecard = {'representations': {'primary_total_co2': {'verdict': verdict}}}
    (primary / 'm1b_scorecard.json').write_text(json.dumps(scorecard) + '\n')
    (primary / 'm1b_provenance.json').write_text(json.dumps({'code': {'commit': 'a' * 40}}) + '\n')
    for name in ev.DETERMINISTIC_ARTIFACTS:
        path = primary / name
        if not path.exists():
            path.write_text('placeholder\n')
    manifest = {'evaluator_commit': 'a' * 40, 'verdict': verdict['verdict'],
                'association_supported': association, 'integrity_passed': integrity,
                'artifact_sha256': {name: hydro.sha256(primary / name)
                                    for name in ev.DETERMINISTIC_ARTIFACTS}}
    (primary / ev.RESULT_MANIFEST).write_text(json.dumps(manifest) + '\n')
    return primary, manifest


# --- authorization ------------------------------------------------------------------------------

def test_refuses_without_a_primary_result(tmp_path):
    with pytest.raises(RuntimeError, match='no primary result manifest'):
        sens.verified_primary(tmp_path / 'missing')


def test_refuses_a_tampered_primary_artifact(tmp_path):
    primary, _ = fake_primary(tmp_path)
    (primary / 'm1b_country_predictions.csv').write_text('tampered\n')
    with pytest.raises(RuntimeError, match='does not match its recorded digest'):
        sens.verified_primary(primary)


def test_refuses_a_manifest_that_does_not_cover_every_artifact(tmp_path):
    primary, manifest = fake_primary(tmp_path)
    manifest['artifact_sha256'].pop('m1b_cv_folds.csv')
    (primary / ev.RESULT_MANIFEST).write_text(json.dumps(manifest) + '\n')
    with pytest.raises(RuntimeError, match='exactly the result artifacts'):
        sens.verified_primary(primary)


def test_refuses_a_primary_result_that_failed_its_own_integrity_checks(tmp_path):
    primary, _ = fake_primary(tmp_path, integrity=False)
    with pytest.raises(RuntimeError, match='integrity'):
        sens.verified_primary(primary)


def test_refuses_an_uncommitted_primary_result(tmp_path):
    primary, _ = fake_primary(tmp_path)
    with pytest.raises(RuntimeError, match='outside the repository|not committed'):
        sens.verified_primary(primary)


def test_no_conditional_runs_when_association_support_fails(tmp_path, monkeypatch):
    primary, _ = fake_primary(tmp_path, association=False)
    monkeypatch.setattr(sens, 'verified_primary', lambda _dir: {
        'primary_dir': str(primary), 'manifest': {}, 'evaluator_commit': 'a' * 40,
        'verdict': json.loads((primary / 'm1b_scorecard.json').read_text())
        ['representations']['primary_total_co2']['verdict']})

    def forbidden(*args, **kwargs):
        raise AssertionError('a conditional arm was scored without association support')
    monkeypatch.setattr(sens, 'run_arms', forbidden)
    monkeypatch.setattr(ev, 'verify_scoring_inputs', forbidden)
    monkeypatch.setattr(ev, 'build_frozen', forbidden)
    out = tmp_path / 'conditional'
    record = sens.run(primary, out)
    assert record['status'] == 'not run under the contract'
    assert record['arms_not_run'] == list(sens.ARMS) and record['no_favourable_sensitivity_search']
    assert (out / sens.NOT_RUN).exists()
    assert not (out / 'm1b_conditional_scorecard.json').exists()


def test_conditional_runs_for_association_support_even_without_promotion(tmp_path, monkeypatch):
    primary, _ = fake_primary(tmp_path, association=True, promoted=False)
    verdict = json.loads((primary / 'm1b_scorecard.json').read_text())[
        'representations']['primary_total_co2']['verdict']
    monkeypatch.setattr(sens, 'verified_primary', lambda _dir: {
        'primary_dir': str(primary), 'manifest': {}, 'evaluator_commit': 'a' * 40,
        'verdict': verdict, 'head': 'h', 'upstream': 'u',
        'primary_scorecard_sha256': 'x', 'primary_manifest_sha256': 'y'})
    calls = []
    monkeypatch.setattr(ev, 'verify_scoring_inputs', lambda *a, **k: ev.ScoringInputs(
        {'git': {'commit': 'z'}}, 'm', {}, b'', {'digests': True}))
    monkeypatch.setattr(ev, 'build_frozen', lambda _inputs: synthetic_frozen(n=24, seed=5))
    monkeypatch.setattr(sens, 'run_arms', lambda frozen, **kwargs: (calls.append(frozen) or ({}, [])))
    monkeypatch.setattr(sens, 'evidence_frames', lambda _results: (pd.DataFrame({'a': [1]}),
                                                                   pd.DataFrame({'b': [2]})))
    monkeypatch.setattr(ev, 'frozen_code_state', lambda *a, **k: {'commit': 'a' * 40})
    out = tmp_path / 'conditional'
    scorecard = sens.run(primary, out)
    assert calls, 'the arms were not run although association support holds'
    assert scorecard['primary_verdict']['promoted'] is False
    assert (out / 'm1b_conditional_scorecard.json').exists()
    assert json.loads((out / sens.RESULT_MANIFEST).read_text())['artifact_sha256']


def test_a_run_never_overwrites_a_prior_conditional_run(tmp_path, monkeypatch):
    primary, _ = fake_primary(tmp_path, association=False)
    monkeypatch.setattr(sens, 'verified_primary', lambda _dir: {
        'primary_dir': str(primary), 'manifest': {}, 'evaluator_commit': 'a' * 40,
        'verdict': {'association_supported': False}})
    out = tmp_path / 'conditional'
    sens.run(primary, out)
    with pytest.raises(RuntimeError, match='never overwrites'):
        sens.run(primary, out)


def test_conditional_execution_does_not_touch_the_primary_result(tmp_path, monkeypatch):
    primary, _ = fake_primary(tmp_path, association=False)
    before = {p.name: p.read_bytes() for p in primary.iterdir()}
    monkeypatch.setattr(sens, 'verified_primary', lambda _dir: {
        'primary_dir': str(primary), 'manifest': {}, 'evaluator_commit': 'a' * 40,
        'verdict': {'association_supported': False}})
    sens.run(primary, tmp_path / 'conditional')
    assert {p.name: p.read_bytes() for p in primary.iterdir()} == before


# --- arm construction ----------------------------------------------------------------------------

def frozen_with_excluded():
    codes = list(sens.EXCLUDED_ISO3) + [f'X{i:02d}' for i in range(35)]
    return synthetic_frozen(n=len(codes), seed=11, iso3=codes)


def test_station_support_arm_removes_exactly_the_five_countries(tmp_path):
    frozen = frozen_with_excluded()
    (tmp_path / 'm1b_support_checkpoint.json').write_text(json.dumps({'extremes_top10': {
        'lowest_station_supported_share': [[c, 0.1] for c in sens.EXCLUDED_ISO3]}}))
    arm, reference, description = sens.station_support_arm(frozen, package_dir=tmp_path)
    assert reference is None and arm.n == frozen.n - 5
    assert not set(sens.EXCLUDED_ISO3) & set(arm.table.iso3)
    keep = ~frozen.table.iso3.isin(sens.EXCLUDED_ISO3).to_numpy()
    np.testing.assert_array_equal(arm.c2, frozen.c2[keep])
    index = np.flatnonzero(keep)
    np.testing.assert_array_equal(arm.distance, frozen.distance[np.ix_(index, index)])
    assert arm.station_w.shape == (arm.n, arm.n) and arm.area_w.shape == (arm.n, arm.n)
    assert len(arm.folds[ev.PRIMARY][0]) == arm.n
    assert np.array_equal(arm.folds['random10'][0], ev.cv.folds_random(arm.n, 10))
    assert description['excluded_iso3'] == list(sens.EXCLUDED_ISO3) and arm.identity['n'] == arm.n


def test_station_support_arm_refuses_a_different_five(tmp_path):
    frozen = frozen_with_excluded()
    (tmp_path / 'm1b_support_checkpoint.json').write_text(json.dumps({'extremes_top10': {
        'lowest_station_supported_share': [[c, 0.1] for c in ['SAU', 'YEM', 'HTI', 'OMN', 'PER']]}}))
    with pytest.raises(ValueError, match='lowest support set'):
        sens.station_support_arm(frozen, package_dir=tmp_path)


def test_station_support_arm_compares_against_its_own_comparator(tmp_path):
    frozen = frozen_with_excluded()
    (tmp_path / 'm1b_support_checkpoint.json').write_text(json.dumps({'extremes_top10': {
        'lowest_station_supported_share': [[c, 0.1] for c in sens.EXCLUDED_ISO3]}}))
    arm, _reference, _description = sens.station_support_arm(frozen, package_dir=tmp_path)
    arms, integrity = ev.score_representations(arm, reference_cards=None)
    assert integrity == []
    entry = arms['primary_total_co2']
    y = arm.design.warming_trend.to_numpy()
    expected = ev.paired_rmse_interval(y, entry['scored'][ev.CANDIDATE].results[ev.PRIMARY].yhat,
                                       entry['scored'][ev.COMPARATOR].results[ev.PRIMARY].yhat)
    assert entry['paired'] == expected
    assert entry['cards'][ev.COMPARATOR]['n'] == entry['cards'][ev.CANDIDATE]['n'] == arm.n


def test_era5_arm_refuses_a_country_without_a_finite_aligned_outcome():
    with pytest.raises(ValueError, match='no finite aligned ERA5 outcome'):
        sens.aligned_era5_arm(synthetic_frozen(n=12, seed=2))


@needs_inputs
def test_era5_arm_replaces_only_the_outcome_with_the_committed_aligned_measurements():
    inputs = ev.verify_scoring_inputs()
    frozen = ev.build_frozen(inputs)
    arm, reference, description = sens.aligned_era5_arm(frozen)
    assert description['outcome_sha256'] == hydro.sha256(ROOT / sens.ERA5_TRENDS)
    assert reference['primary_total_co2']['metrics'] == ('in_sample_r2',)
    changed = [c for c in frozen.design.columns
               if not frozen.design[c].equals(arm.design[c])]
    assert changed == ['warming_trend']
    assert np.isfinite(arm.design.warming_trend.to_numpy()).all()
    assert not np.array_equal(arm.design.warming_trend.to_numpy(),
                              frozen.design.warming_trend.to_numpy())


@needs_inputs
def test_m1a_arm_replaces_exactly_the_five_remeasured_features():
    inputs = ev.verify_scoring_inputs()
    frozen = ev.build_frozen(inputs)
    arm, reference, description = sens.m1a_geography_arm(frozen)
    changed = [c for c in frozen.design.columns if not frozen.design[c].equals(arm.design[c])]
    assert set(changed) <= set(sens.REMEASURED) and 'spatial_block' not in changed
    assert description['replaced_features'] == list(sens.REMEASURED)
    assert reference['per_capita']['metrics'] == ev.REFERENCE_METRICS
    assert np.array_equal(arm.c2, frozen.c2) and arm.n == frozen.n


# --- descriptive reporting -------------------------------------------------------------------------

def test_arm_conditions_are_descriptive_and_never_a_verdict():
    entry = {'verdict': {'verdict': 'supported and promoted', 'S1_interval_below_zero': True,
                         'association_supported': True}}
    conditions = sens.arm_conditions(entry)
    assert 'verdict' not in conditions
    assert conditions['arm_label_descriptive_only'] == 'supported and promoted'
    assert 'cannot change either verdict level' in conditions['note']


@pytest.mark.parametrize('replicated, sensitive', [(True, False), (False, True)])
def test_product_sensitivity_flag(replicated, sensitive):
    results = {'aligned_era5_outcome': {'representations': {'primary_total_co2': {
        'arm_conditions': {'S1_interval_below_zero': replicated}}}}}
    flag = sens.product_sensitivity({'association_supported': True}, results)
    assert flag['product_sensitive'] is sensitive
    assert flag['aligned_era5_S1_replicated'] is replicated
    assert sens.product_sensitivity({'association_supported': True}, {}) is None


def test_arms_are_the_three_registered_replications_and_are_never_combined():
    assert sens.ARMS == ('m1a_area_geography', 'aligned_era5_outcome', 'station_support_146')
    assert sorted(sens.ARM_BUILDERS) == sorted(sens.ARMS)
    assert sens.EXCLUDED_ISO3 == ('SAU', 'YEM', 'HTI', 'OMN', 'TCD')


def test_each_arm_records_its_own_identity_not_the_primary_run_s(tmp_path):
    frozen = frozen_with_excluded()
    frozen = dataclasses.replace(frozen, identity={**frozen.identity,
                                                   'country_list_sha256': 'primary-list',
                                                   'outcome_source_column': 'berkeley'})
    (tmp_path / 'm1b_support_checkpoint.json').write_text(json.dumps({'extremes_top10': {
        'lowest_station_supported_share': [[c, 0.1] for c in sens.EXCLUDED_ISO3]}}))
    arm, _reference, _description = sens.station_support_arm(frozen, package_dir=tmp_path)
    identity = arm.identity
    assert identity['arm'] == 'station_support_146' and identity['n'] == arm.n
    assert identity['excluded_iso3'] == list(sens.EXCLUDED_ISO3)
    assert identity['iso3_sha256'] != ev.hashlib.sha256(
        ('\n'.join(frozen.table.iso3) + '\n').encode()).hexdigest()
    primary = ev.hashlib.sha256(np.array(
        [t for _f, _x, t in ev.cv.iter_folds(frozen.folds[ev.PRIMARY][0], frozen.distance,
                                             ev.PRIMARY_BUFFER_KM)]).tobytes()).hexdigest()
    assert identity['primary_membership_sha256'] != primary
    assert identity['station_weights_sha256'] != ev.array_digest(frozen.station_w)
    assert identity['inherited_from_primary']['country_list_sha256'] == 'primary-list'


def test_product_sensitivity_survives_a_refused_arm_verdict():
    results = {'aligned_era5_outcome': {'representations': {'primary_total_co2': {
        'arm_conditions': {'refused': 'a coefficient was not finite',
                           'arm_label_descriptive_only': None}}}}}
    flag = sens.product_sensitivity({'association_supported': True}, results)
    assert flag['product_sensitive'] is None and flag['aligned_era5_S1_replicated'] is None
    assert 'did not produce evaluable conditions' in flag['reason']
