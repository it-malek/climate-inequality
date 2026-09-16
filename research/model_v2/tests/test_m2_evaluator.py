"""M2 evaluator: gates, verdict boundaries, estimator semantics, equivalence and the conditional runner.

Scored fixtures are synthetic. Real inputs are used only for digest gates, predictor-only structure and one
disclosed baseline-only anchor (M0* against its frozen rank-audit cards), in which any attempt to fit M2 raises.
No test fits M2 to a real outcome.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.model_v2 import cv
from research.model_v2 import m1b_evaluate as m1b
from research.model_v2 import m2_conditional as cond
from research.model_v2 import m2_evaluate as ev
from research.model_v2 import m2_latitude_basis as lb
from research.model_v2 import v2_provenance as prov
from research.model_v2.spatial import haversine_matrix, knn_weights
from src.decomposition import OUTCOME_COL, group_lmg_shares

ZONES, BLOCKS = list('ABCD'), ['Africa', 'Asia', 'Europe', 'South America']
INCOMES = ['High-income countries', 'Low-income countries', 'Upper-middle-income countries']
REGIONS = ['R1', 'R2', 'R3', 'R4', 'R5']


def _levels(rng, levels, n):
    base = list(levels) * 3
    return list(rng.permutation(base + list(rng.choice(levels, n - len(base)))))


def synthetic_frame(n=60, seed=0, representation='primary_total_co2'):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-50, 65, n)
    frame = pd.DataFrame({
        'Country': [f'C{i}' for i in range(n)], 'iso3': [f'X{i:02d}' for i in range(n)],
        'm49_subregion': _levels(rng, REGIONS, n), 'station_lon': rng.uniform(-170, 170, n), 'station_lat': lat,
        'climate_zone': _levels(rng, ZONES, n), 'hemisphere': np.where(lat < 0, 'S', 'N'),
        'spatial_block': _levels(rng, BLOCKS, n), 'income_group': _levels(rng, INCOMES, n),
        'cum_co2_total': rng.uniform(1.0, 4.0, n), 'abs_latitude': np.abs(lat) + 0.1,
        'elevation': rng.uniform(0, 2000, n), 'continentality': rng.uniform(0, 800, n),
        'population': rng.uniform(5.5, 9.0, n), 'station_density': rng.uniform(0, 3, n)})
    if representation == 'per_capita':
        frame = frame.rename(columns={'cum_co2_total': 'cum_co2_per_capita'})
        frame['cum_co2_per_capita'] = frame['cum_co2_per_capita'] - frame['population'] + 6.0
    frame[OUTCOME_COL] = (0.2 + 0.001 * frame.abs_latitude + 0.00002 * frame.abs_latitude ** 2
                          + rng.normal(0, 0.02, n))
    return frame


def synthetic_frozen(n=60, seed=0):
    total = synthetic_frame(n, seed)
    per_capita = synthetic_frame(n, seed, 'per_capita')
    table = total[ev.TABLE_COLUMNS]
    distance = haversine_matrix(total.station_lon.to_numpy(), total.station_lat.to_numpy())
    w = knn_weights(distance, 8)
    return ev.Frozen(table=table, frames={'primary_total_co2': total, 'per_capita': per_capita},
                     y=total[OUTCOME_COL].to_numpy(), distance=distance, station_w=w, area_w=w,
                     folds=ev.fold_plan(table, n), encoded=None, identity={})


# --- pins and the execution boundary ---------------------------------------------------------------

def test_the_m1b_scoring_input_pins_are_inherited_unchanged():
    assert ev.FROZEN_SCORING_INPUTS == m1b.FROZEN_SCORING_INPUTS


def test_every_pin_matches_the_committed_bytes():
    digests = ev.verify_pins()
    assert set(digests) == set(ev.ALL_PINS)


def test_an_altered_pin_refuses(monkeypatch):
    pins = dict(ev.ALL_PINS)
    key = ev.ENCODED_DESIGN
    pins[key] = ('0' * 64, pins[key][1])
    monkeypatch.setattr(ev, 'ALL_PINS', pins)
    with pytest.raises(prov.ExecutionRefused, match='differs from the record frozen'):
        ev.verify_pins()


def test_the_hash_seed_is_required(monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', '1')
    with pytest.raises(prov.ExecutionRefused, match='PYTHONHASHSEED'):
        prov.require_hash_seed()


def test_an_existing_output_root_refuses(tmp_path):
    with pytest.raises(prov.ExecutionRefused, match='already exists'):
        prov.require_fresh(tmp_path)


def test_the_code_closure_is_computed_transitively():
    closure = set(ev.code_files())
    for path in ('research/model_v2/m2_evaluate.py', 'research/model_v2/m2_conditional.py',
                 'research/model_v2/m2_latitude_basis.py', 'research/model_v2/m2_feasibility.py',
                 'research/model_v2/cv.py', 'research/model_v2/spatial.py', 'research/model_v2/territory.py',
                 'research/model_v2/v2_provenance.py', 'src/decomposition.py', 'src/feature_schema.py',
                 'research/model_v2/M2_EVALUATOR_SPEC.md', 'research/model_v2/DOWNSTREAM_COMPLETION_SPEC.md',
                 'uv.lock', 'pyproject.toml'):
        assert path in closure


def test_a_dirty_or_unpushed_closure_refuses(monkeypatch):
    calls = []

    def fake_git(*args, check=True):
        calls.append(args)
        done = type('Done', (), {})()
        done.stdout, done.stderr = 'abc\n', ''
        done.returncode = 1 if args[0] == 'diff' else 0
        if args[:2] == ('diff', '--name-only'):
            done.returncode, done.stdout = 0, 'research/model_v2/cv.py\n'
        return done

    monkeypatch.setattr(prov, '_git', fake_git)
    with pytest.raises(prov.ExecutionRefused, match='differs from HEAD'):
        prov.frozen_code_state(['research/model_v2/cv.py'])

    def unpushed(*args, check=True):
        done = type('Done', (), {})()
        done.stderr, done.returncode = '', 0
        done.stdout = {'rev-parse': 'aaaa\n', 'ls-remote': 'bbbb\trefs/heads/x\n'}.get(args[0], '')
        if args[0] == 'rev-parse' and '@{upstream}' in args:
            done.stdout = 'origin/x\n'
        if args[0] == 'merge-base':
            done.returncode = 1
        return done

    monkeypatch.setattr(prov, '_git', unpushed)
    with pytest.raises(prov.ExecutionRefused, match='not contained in the live remote head'):
        prov.frozen_code_state(['research/model_v2/cv.py'])


def test_a_live_remote_head_missing_locally_refuses(monkeypatch):
    def missing(*args, check=True):
        done = type('Done', (), {})()
        done.stderr, done.returncode = '', 0
        done.stdout = {'rev-parse': 'aaaa\n', 'ls-remote': 'bbbb\trefs/heads/x\n'}.get(args[0], '')
        if args[0] == 'rev-parse' and '@{upstream}' in args:
            done.stdout = 'origin/x\n'
        if args[0] == 'cat-file':
            done.returncode = 1
        return done

    monkeypatch.setattr(prov, '_git', missing)
    with pytest.raises(prov.ExecutionRefused, match='fetch before running'):
        prov.frozen_code_state(['research/model_v2/cv.py'])


# --- verdict boundaries ------------------------------------------------------------------------------

def _cards(m49=0.0, region=0.0):
    comparator = {'secondary': {'cv_rmse': 0.0}, 'worst_region': {'rmse': 0.0, 'region': 'A'}}
    candidate = {'secondary': {'cv_rmse': m49}, 'worst_region': {'rmse': region, 'region': 'B'}}
    return comparator, candidate


def _paired(delta, lo, hi):
    return {'delta_rmse': delta, 'country_bootstrap_95_interval': [lo, hi], 'resamples': 2000, 'seed': 0}


@pytest.mark.parametrize('delta,lo,hi,m49,region,label', [
    (-0.002, -0.003, -0.001, 0.001, 0.001, 'supported and promoted'),            # all inclusive at equality
    (-0.0019, -0.003, -0.001, 0.0, 0.0, 'predictive support, not promoted'),    # R1 fails
    (-0.003, -0.004, -0.001, 0.002, 0.0, 'predictive support, not promoted'),   # R2 fails
    (-0.003, -0.004, -0.001, 0.0, 0.002, 'predictive support, not promoted'),   # R3 fails
    (-0.003, -0.004, 0.0, 0.0, 0.0, 'not supported'),                           # P2 strict at zero
    (0.0, -0.001, 0.001, 0.0, 0.0, 'not supported'),                            # P1 strict at zero
    (0.001, 0.0005, 0.002, 0.0, 0.0, 'not supported'),
])
def test_verdict_boundaries(delta, lo, hi, m49, region, label):
    comparator, candidate = _cards(m49, region)
    result = ev.verdict(comparator, candidate, _paired(delta, lo, hi), True)
    assert result['verdict'] == label
    assert result['predictive_support'] == (label != 'not supported')


def test_an_integrity_failure_gives_no_label():
    comparator, candidate = _cards()
    result = ev.verdict(comparator, candidate, _paired(-0.01, -0.02, -0.005), False)
    assert result['verdict'] is None and result['predictive_support'] is False


@pytest.mark.parametrize('delta,lo,expected', [(0.001, 0.0005, True), (0.001, 0.0, False), (0.0, 0.001, False)])
def test_worsened_generalization_is_strict_on_both_parts(delta, lo, expected):
    comparator, candidate = _cards()
    assert ev.verdict(comparator, candidate, _paired(delta, lo, 0.01), True)['worsened_generalization'] is expected


@pytest.mark.parametrize('field', ['delta', 'lo', 'hi', 'm49', 'region'])
def test_non_finite_verdict_inputs_refuse(field):
    values = {'delta': -0.01, 'lo': -0.02, 'hi': -0.005, 'm49': 0.0, 'region': 0.0}
    values[field] = np.nan
    comparator, candidate = _cards(values['m49'], values['region'])
    with pytest.raises(ValueError, match='not finite'):
        ev.verdict(comparator, candidate, _paired(values['delta'], values['lo'], values['hi']), True)


@pytest.mark.parametrize('delta,fires', [(-0.002, False), (-0.0019999, True), (-0.003, False), (0.001, True)])
def test_the_static_stopping_indicator(delta, fires):
    result = ev.stopping_indicator(delta)
    assert result['fires'] is fires
    assert result['m2_decides'] is True
    assert result['point_delta_rmse_vs_m0star']['M1b'] == pytest.approx(0.0017229850562197266, abs=0)


# --- estimator semantics (synthetic) ----------------------------------------------------------------------

def test_m0star_equals_the_frozen_v1_estimator_on_raw_values():
    frame = synthetic_frame()
    train, test = frame.iloc[:45], frame.iloc[45:].copy()
    test.loc[test.index[0], 'climate_zone'] = 'Z'          # an unseen level
    raw = frame.copy()
    for column in ('cum_co2_total', 'population'):
        raw[column] = 10.0 ** raw[column]
    expected = cv.V1Design().fit(raw.iloc[:45]).predict(raw.loc[test.index].assign(climate_zone=test.climate_zone))
    got = ev.predict(ev.fit_model(ev.COMPARATOR, train), test)
    assert np.max(np.abs(got - expected)) < 1e-12


def test_m2_state_and_coefficients_use_training_rows_only():
    frame = synthetic_frame()
    train, test = frame.iloc[:45], frame.iloc[45:]
    fitted = ev.fit_model(ev.CANDIDATE, train)
    assert fitted.state == lb.fit_state(train.abs_latitude.to_numpy())
    assert fitted.state != lb.fit_state(frame.abs_latitude.to_numpy())
    assert fitted.columns[-3:] == list(lb.ADDED_COLUMNS)
    assert np.array_equal(ev.predict(fitted, test), ev.predict(fitted, test.assign(**{OUTCOME_COL: -5.0})))
    far = test.assign(abs_latitude=test.abs_latitude + 100.0)
    beyond = ev.design(ev.CANDIDATE, far, fitted.encoder, fitted.state)[0][:, -3:-1]
    slope = ev.design(ev.CANDIDATE, far.assign(abs_latitude=far.abs_latitude + 1.0), fitted.encoder, fitted.state)[0][:, -3:-1]
    second = ev.design(ev.CANDIDATE, far.assign(abs_latitude=far.abs_latitude + 2.0), fitted.encoder, fitted.state)[0][:, -3:-1]
    assert np.allclose(second - slope, slope - beyond, rtol=0, atol=1e-6)   # linear tails beyond the upper knot


def test_cross_validated_predictions_ignore_held_out_and_buffered_outcomes():
    frozen = synthetic_frozen()
    frame = frozen.frames['primary_total_co2']
    result, records = ev.run_protocol(ev.CANDIDATE, frame, frozen, ev.PRIMARY)
    i = 7
    buffered = np.flatnonzero(frozen.distance[i] <= 500.0)
    changed = frame.copy()
    changed.loc[buffered, OUTCOME_COL] = 1e6
    again, _ = ev.run_protocol(ev.CANDIDATE, changed, frozen, ev.PRIMARY)
    assert again.yhat[i] == result.yhat[i]
    assert len(records) == len(frame) and all(r['fitted'].state is not None for r in records)


def test_encoded_lmg_equals_the_frozen_decomposition_on_raw_values():
    frame = synthetic_frame()
    raw = frame.drop(columns=['iso3', 'm49_subregion', 'station_lon', 'station_lat']).copy()
    for column in ('cum_co2_total', 'population'):
        raw[column] = 10.0 ** raw[column]
    frozen = group_lmg_shares(raw)
    ours = ev.lmg_shares(ev.COMPARATOR, frame)
    for group, value in frozen.shares.items():
        assert ours['shares'][group] == pytest.approx(value, abs=1e-12)
    assert ours['total_r2'] == pytest.approx(frozen.total_r2, abs=1e-12)
    assert len(ours['coalitions']) == 16


def test_m2_columns_stay_in_geography_and_shares_account():
    frame = synthetic_frame()
    result = ev.lmg_shares(ev.CANDIDATE, frame)
    matrix, names = ev.design(ev.CANDIDATE, frame, ev.feas.baseline_encoder(frame), result['state'])
    groups = ev.column_groups(frame, names)
    assert [g for n, g in zip(names, groups) if n in lb.ADDED_COLUMNS] == ['geography'] * 3
    assert sum(result['shares'].values()) == pytest.approx(result['total_r2'], abs=1e-12)


def test_degenerate_latitudes_raise_without_fallback():
    frame = synthetic_frame().assign(abs_latitude=10.0)
    with pytest.raises(lb.DegenerateLatitudeBasis):
        ev.fit_model(ev.CANDIDATE, frame)


def test_a_rank_deficient_design_is_a_structural_failure():
    frame = synthetic_frame()
    frame['elevation'] = frame['continentality'] * 2.0
    fitted = ev.fit_model(ev.CANDIDATE, frame)
    assert 'M0* design is rank deficient' in ev.structural_failures(fitted)


def test_score_model_on_a_synthetic_frame_records_every_fit_and_gamma():
    frozen = synthetic_frozen()
    scored = ev.score_model(ev.CANDIDATE, frozen.frames['primary_total_co2'], frozen, expected_full=False)
    assert scored.card['structural_integrity']['fits_checked'] == {ev.PRIMARY: 60, 'm49_subregion_lo': 5,
                                                                    'random10': 10}
    gamma = scored.full.coef[scored.full.columns.index(ev.GAMMA)]
    assert scored.card['coefficients'][ev.GAMMA]['full'] == gamma
    assert scored.card['coefficients']['abs_latitude_ns1']['interpretation'].startswith('basis')


def test_representations_are_equivalent_on_synthetic_identity_data():
    frozen = synthetic_frozen()
    arms = {}
    for representation in ev.REPRESENTATIONS:
        frame = frozen.frames[representation]
        comparator = ev.score_model(ev.COMPARATOR, frame, frozen, expected_full=False)
        candidate = ev.score_model(ev.CANDIDATE, frame, frozen, expected_full=False)
        y = frame[OUTCOME_COL].to_numpy()
        arms[representation] = {
            'scored': {ev.COMPARATOR: comparator, ev.CANDIDATE: candidate},
            'cards': {ev.COMPARATOR: comparator.card, ev.CANDIDATE: candidate.card},
            'paired': ev.paired_rmse_interval(y, candidate.results[ev.PRIMARY].yhat,
                                              comparator.results[ev.PRIMARY].yhat)}
    report = ev.equivalence(arms)
    assert report['passes'], report


def test_a_broken_representation_identity_fails_equivalence():
    frozen = synthetic_frozen()
    broken = frozen.frames['per_capita'].copy()
    broken['cum_co2_per_capita'] = np.random.default_rng(3).permutation(broken['cum_co2_per_capita'].to_numpy())
    frozen.frames['per_capita'] = broken
    arms = {}
    for representation in ev.REPRESENTATIONS:
        frame = frozen.frames[representation]
        comparator = ev.score_model(ev.COMPARATOR, frame, frozen, expected_full=False)
        candidate = ev.score_model(ev.CANDIDATE, frame, frozen, expected_full=False)
        y = frame[OUTCOME_COL].to_numpy()
        arms[representation] = {
            'scored': {ev.COMPARATOR: comparator, ev.CANDIDATE: candidate},
            'cards': {ev.COMPARATOR: comparator.card, ev.CANDIDATE: candidate.card},
            'paired': ev.paired_rmse_interval(y, candidate.results[ev.PRIMARY].yhat,
                                              comparator.results[ev.PRIMARY].yhat)}
    assert not ev.equivalence(arms)['passes']


def test_reference_check_is_strict_and_rejects_non_finite_values():
    card = {'cv_rmse': 0.1, 'secondary': {'cv_rmse': 0.2}}
    assert ev.reference_check(card, card, 'x', ('cv_rmse', 'secondary.cv_rmse'))['reproduced']
    with pytest.raises(prov.ExecutionRefused, match='does not reproduce'):
        ev.reference_check(card, {'cv_rmse': 0.1 + 2e-10, 'secondary': {'cv_rmse': 0.2}}, 'x', ('cv_rmse',))
    with pytest.raises(prov.ExecutionRefused, match='not finite'):
        ev.reference_check({'cv_rmse': np.nan}, card, 'x', ('cv_rmse',))


def test_cli_exit_codes():
    assert ev.cli_exit_code({'integrity_passed': True}) == 0
    assert ev.cli_exit_code({'integrity_passed': False}) == 3
    assert ev.cli_exit_code({'byte_identical': False}) == 4
    for bad in ({}, {'integrity_passed': True, 'byte_identical': True}, {'integrity_passed': 1}, []):
        with pytest.raises(RuntimeError):
            ev.cli_exit_code(bad)


# --- real inputs: identities, predictor-only structure and the baseline-only anchor ---------------------

@pytest.fixture(scope='module')
def real_frozen():
    original = np.linalg.lstsq

    def refuse(*args, **kwargs):
        raise AssertionError('building the frozen frame must not fit anything')

    np.linalg.lstsq = refuse
    try:
        return ev.build_frozen()
    finally:
        np.linalg.lstsq = original


def test_the_real_frozen_frame_passes_every_identity(real_frozen):
    identity = real_frozen.identity
    assert identity['n'] == 151
    assert set(identity['encoded_identity']) == set(ev.REPRESENTATIONS)
    assert identity['representation_identity_max_abs_error'] <= ev.IDENTITY_ATOL


def test_every_fit_reproduces_the_committed_feasibility_record(real_frozen, monkeypatch):
    monkeypatch.setattr(np.linalg, 'lstsq', lambda *a, **k: (_ for _ in ()).throw(AssertionError('no fits')))
    result = ev.check_against_feasibility(real_frozen)
    assert result['fits_compared'] == 364


def test_an_altered_feasibility_state_refuses(real_frozen, monkeypatch):
    original = ev.predictor_only_states

    def shifted(frozen, frame, representation):
        rows = original(frozen, frame, representation)
        state = lb.LatitudeState.from_json(rows[5]['latitude_state'])
        rows[5]['latitude_state'] = lb.LatitudeState(knots=(state.knots[0], state.knots[1] + 1e-9, *state.knots[2:]),
                                                     n_train=state.n_train).to_json()
        return rows

    monkeypatch.setattr(ev, 'predictor_only_states', shifted)
    with pytest.raises(prov.ExecutionRefused, match='latitude state differs'):
        ev.check_against_feasibility(real_frozen)


def test_baseline_only_anchor_m0star_reproduces_its_frozen_cards(real_frozen, monkeypatch):
    """Disclosed baseline-only anchor on real inputs; any M2 fit raises (M2_EVALUATOR_SPEC.md §3)."""
    original = ev.fit_model

    def comparator_only(model, train, outcome=OUTCOME_COL):
        assert model == ev.COMPARATOR, 'no M2 candidate may be fitted in tests'
        return original(model, train, outcome)

    monkeypatch.setattr(ev, 'fit_model', comparator_only)
    arms = ev.score_comparators(real_frozen, ev.representation_frames(real_frozen), ev.reference_cards())
    for arm in arms.values():
        assert arm['reference']['reproduced']


def test_a_pre_candidate_refusal_writes_nothing(real_frozen, monkeypatch, tmp_path):
    def refuse(frozen):
        raise prov.ExecutionRefused('synthetic refusal')

    monkeypatch.setattr(ev, 'check_against_feasibility', refuse)
    out = tmp_path / 'run'
    with pytest.raises(prov.ExecutionRefused):
        ev.evaluate(real_frozen, out, provenance={}, reference_cards=ev.reference_cards())
    assert not out.exists()


def test_the_disclosed_extrapolation_counts_match_the_feasibility_record():
    fits = pd.read_csv(ev.ROOT / ev.FEASIBILITY_FITS)
    station = fits[(fits.measurement == 'station') & (fits.representation == 'primary_total_co2')]
    expected = {ev.PRIMARY: (1, 1), 'm49_subregion_lo': (1, 8), 'random10': (1, 4)}
    for protocol, (below, above) in expected.items():
        rows = station[station.protocol == protocol]
        assert (int(rows.test_below_lower_boundary.sum()), int(rows.test_above_upper_boundary.sum())) == (below, above)


# --- the conditional runner ----------------------------------------------------------------------------------

def _primary_package(tmp_path, label):
    directory = tmp_path / 'primary'
    directory.mkdir()
    scorecard = {'representations': {'primary_total_co2': {'verdict': {'verdict': label,
                                                                       'predictive_support': label in ev.SUPPORT_LABELS}}}}
    for name in ev.DETERMINISTIC_ARTIFACTS:
        (directory / name).write_text('{}\n')
    (directory / 'm2_scorecard.json').write_text(json.dumps(scorecard))
    manifest = {'artifact_sha256': {n: prov.sha256(directory / n) for n in ev.DETERMINISTIC_ARTIFACTS},
                'integrity_passed': True, 'verdict': label, 'evaluator_commit': 'a' * 40}
    (directory / ev.RESULT_MANIFEST).write_text(json.dumps(manifest))
    return directory


def test_the_conditional_runner_refuses_a_tampered_primary(tmp_path):
    directory = _primary_package(tmp_path, 'not supported')
    (directory / 'm2_scorecard.json').write_text('{"tampered": true}')
    with pytest.raises(prov.ExecutionRefused, match='differs from its manifest'):
        cond.verified_primary(directory)


def test_without_support_nothing_is_constructed_or_fitted(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    monkeypatch.setattr(prov, 'frozen_code_state', lambda files: {'commit': 'c', 'live_remote_commit': 'c'})
    primary = {'verdict': {'verdict': 'not supported', 'predictive_support': False}, 'predictive_support': False}
    monkeypatch.setattr(cond, 'verified_primary', lambda directory: primary)
    for name in ('m1a_frames', 'era5_outcome', 'run_arm'):
        monkeypatch.setattr(cond, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError('constructed an arm')))
    for name in ('verify_pins', 'build_frozen', 'score_model', 'fit_model'):
        monkeypatch.setattr(ev, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError('touched inputs')))
    out = tmp_path / 'conditional'
    result = cond.main(['--primary', str(tmp_path), '--out', str(out)])
    assert result['status'] == 'not executed'
    record = json.loads((out / cond.NOT_RUN).read_text())
    assert record['status'] == 'not executed' and 'no arm was constructed or fitted' in record['rule']


def test_a_structural_arm_failure_is_non_computable_but_a_reference_failure_stops(monkeypatch):
    frozen = synthetic_frozen()
    frames = ev.representation_frames(frozen)
    degenerate = {r: {m: f.assign(abs_latitude=10.0) for m, f in by.items()} for r, by in frames.items()}
    loose = {r: {'card': {}, 'metrics': ()} for r in ev.REPRESENTATIONS}
    monkeypatch.setattr(ev, 'EXPECTED_FULL', {ev.COMPARATOR: 16, ev.CANDIDATE: 19})
    entry, arms = cond.run_arm('synthetic', frozen, degenerate, reference=loose)
    assert entry['status'] == 'non_computable' and arms is None
    strict = {r: {'card': {'in_sample_r2': -1.0}, 'metrics': ('in_sample_r2',)} for r in ev.REPRESENTATIONS}
    with pytest.raises(cond.ArmIntegrityFailure):
        cond.run_arm('synthetic', frozen, frames, reference=strict)


def test_the_conditional_code_is_in_the_evaluator_closure():
    assert 'research/model_v2/m2_conditional.py' in ev.code_files()
    assert Path(ev.ROOT / 'research/model_v2/m2_conditional.py').exists()
