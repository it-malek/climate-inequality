"""M3 evaluator: qualification boundaries, final naming, held-out invariance, failure dispositions, accounting.

Every fitted fixture is synthetic (the synthetic scoring frame of the M2 evaluator tests). No test fits a
spatial model to the real warming outcome, and no test reads an M2 or M3 result.
"""
import json

import numpy as np
import pytest

from research.model_v2 import m2_evaluate as ev
from research.model_v2 import m3_evaluate as m3
from research.model_v2 import m3_spatial as ms
from research.model_v2 import v2_provenance as prov
from research.model_v2.tests.test_m2_evaluator import synthetic_scoring
from src.decomposition import OUTCOME_COL


def _card(secondary=0.0, worst=0.0):
    return {'secondary': {'cv_rmse': secondary}, 'worst_region': {'rmse': worst}}


def _paired(delta, lo, hi):
    return {'delta_rmse': delta, 'country_bootstrap_95_interval': [lo, hi]}


@pytest.mark.parametrize('delta,lo,hi,m49,worst,qualifies', [
    (-0.001, -0.002, -0.0001, 0.001, 0.001, True),      # vetoes inclusive at equality; no 0.002 threshold
    (-0.001, -0.002, 0.0, 0.0, 0.0, False),             # Q2 strict
    (0.0, -0.001, -0.0001, 0.0, 0.0, False),            # Q1 strict
    (-0.001, -0.002, -0.0001, 0.002, 0.0, False),       # Q3
    (-0.001, -0.002, -0.0001, 0.0, 0.002, False),       # Q4
])
def test_qualification_boundaries(delta, lo, hi, m49, worst, qualifies):
    result = m3.qualification(_card(m49, worst), _card(), _paired(delta, lo, hi), True)
    assert result['qualifies'] is qualifies


def test_a_non_computable_family_never_qualifies():
    assert m3.qualification(_card(), _card(), _paired(-0.01, -0.02, -0.005), False)['qualifies'] is False


def test_non_finite_qualification_inputs_refuse():
    with pytest.raises(ValueError, match='not finite'):
        m3.qualification(_card(np.nan), _card(), _paired(-0.01, -0.02, -0.005), True)


@pytest.mark.parametrize('flags,expected_final,expected_set', [
    ((False, False), 'M0*', []),
    ((True, False), 'M3-SEM(M0*)', ['M3-SEM(M0*)']),
    ((False, True), 'M3-SAR(M0*)', ['M3-SAR(M0*)']),
    ((True, True), None, ['M3-SEM(M0*)', 'M3-SAR(M0*)']),
])
def test_final_naming_table(flags, expected_final, expected_set):
    entries = {('primary_total_co2', f): {'qualification': {'qualifies': q}} for f, q in zip(m3.FAMILIES, flags)}
    naming = m3.final_naming('M0*', entries)
    assert naming['final_primary_predictive_model'] == expected_final
    assert naming['qualifying_spatial_extensions'] == expected_set
    assert naming['retained_static_specification'] == 'M0*'


def test_branch_b_names_carry_the_static_specification():
    entries = {('primary_total_co2', 'sem'): {'qualification': {'qualifies': True}},
               ('primary_total_co2', 'sar'): {'qualification': {'qualifies': False}}}
    assert m3.final_naming('M2', entries)['final_primary_predictive_model'] == 'M3-SEM(M2)'


@pytest.fixture(scope='module')
def synthetic():
    scoring = synthetic_scoring(n=60, seed=4)
    dist = np.array(scoring.distance)
    return scoring, dist


@pytest.mark.parametrize('static_model', [ev.COMPARATOR, ev.CANDIDATE])
@pytest.mark.parametrize('family', m3.FAMILIES)
def test_held_out_and_buffered_outcomes_never_enter_predictions(synthetic, family, static_model):
    scoring, dist = synthetic
    frame = scoring.frames['primary_total_co2']
    run = m3.run_family(family, static_model, frame, scoring, dist, 'primary_total_co2')
    assert run.status == 'computable', run.failure
    i = 11
    excluded = np.flatnonzero(scoring.distance[i] <= 500.0)          # i itself and its buffer
    changed = frame.copy()
    changed.loc[excluded, OUTCOME_COL] = 50.0
    records = [r for r in run.records[ev.PRIMARY] if i in r['test']]
    again = m3.fit_one(family, static_model, changed, dist, records[0]['train'], records[0]['test'])
    assert again['prediction'][0] == run.results[ev.PRIMARY].yhat[i]
    assert again['fit'].theta == records[0]['fit'].theta


def test_a_spatial_fit_failure_makes_the_family_non_computable_without_fallback(synthetic, monkeypatch):
    scoring, dist = synthetic
    frame = scoring.frames['primary_total_co2']
    original, calls = ms.fit_spatial, []

    def failing(family, y, x, w, columns=None):
        calls.append(len(y))
        if len(calls) == 5:
            raise ms.SpatialFitFailure('rank-deficient', 'injected')
        return original(family, y, x, w, columns)

    monkeypatch.setattr(ms, 'fit_spatial', failing)
    run = m3.run_family('sem', ev.COMPARATOR, frame, scoring, dist, 'primary_total_co2')
    assert run.status == 'non_computable' and run.failure['reason'] == 'rank-deficient'
    assert len(calls) == 5 and run.full is not None                # nothing attempted after the failure


def test_a_boundary_estimate_is_computable_and_flagged(synthetic, monkeypatch):
    scoring, dist = synthetic
    frame = scoring.frames['primary_total_co2']
    original = ms.maximize

    def to_bound(objective):
        result = original(lambda t: objective(t) + 1e6 * (-t))
        return result

    monkeypatch.setattr(ms, 'maximize', to_bound)
    run = m3.run_family('sar', ev.COMPARATOR, frame, scoring, dist, 'primary_total_co2')
    assert run.status == 'computable'
    assert run.full.at_domain_bound and run.interval.se is None


def test_accounting_identity_and_filtered_shares_on_synthetic_data(synthetic):
    scoring, dist = synthetic
    frame = scoring.frames['primary_total_co2']
    static = ev.score_model(ev.COMPARATOR, frame, scoring, expected_full=False)
    run = m3.run_family('sem', ev.COMPARATOR, frame, scoring, dist, 'primary_total_co2')
    block = m3.accounting_block(run, frame, static)
    parts = block['estimands']
    assert parts['identity_sum'] == pytest.approx(1.0, abs=1e-12)
    assert parts['a_static'] == pytest.approx(static.card['in_sample_r2'], abs=1e-12)
    filtered = block['filtered_trend_shares']
    assert sum(filtered['shares'].values()) == pytest.approx(filtered['r2_full'], abs=1e-9)
    assert filtered['n_coalitions'] == 16


def test_representation_identity_holds_for_spatial_families_on_synthetic_identity_data(synthetic):
    scoring, dist = synthetic
    runs, entries = {}, {}
    for representation in ev.REPRESENTATIONS:
        frame = scoring.frames[representation]
        y = frame[OUTCOME_COL].to_numpy(float)
        static = ev.score_model(ev.COMPARATOR, frame, scoring, expected_full=False)
        for family in m3.FAMILIES:
            run = m3.run_family(family, ev.COMPARATOR, frame, scoring, dist, representation)
            runs[(representation, family)] = run
            entries[(representation, family)] = {
                'card': m3.family_card(run, frame, scoring, 'M0*', scoring.station_w, scoring.area_w, dist),
                'paired': m3.paired_block(y, run, static), 'accounting': m3.accounting_block(run, frame, static)}
    report = m3.equivalence(runs, entries)
    assert report['passes'], json.dumps(report, default=str)[:2000]


def test_a_family_computable_in_one_representation_only_fails_identity():
    ok, bad = m3.FamilyRun('sem', 'primary_total_co2'), m3.FamilyRun('sem', 'per_capita', status='non_computable')
    runs = {('primary_total_co2', 'sem'): ok, ('per_capita', 'sem'): bad,
            ('primary_total_co2', 'sar'): m3.FamilyRun('sar', 'primary_total_co2', status='non_computable'),
            ('per_capita', 'sar'): m3.FamilyRun('sar', 'per_capita', status='non_computable')}
    assert m3.equivalence(runs, {})['passes'] is False


def test_the_code_closure_covers_the_estimator_the_static_path_and_the_lock_file():
    closure = set(m3.code_files())
    for path in ('research/model_v2/m3_evaluate.py', 'research/model_v2/m3_spatial.py',
                 'research/model_v2/m2_evaluate.py', 'research/model_v2/m2_latitude_basis.py',
                 'research/model_v2/v2_provenance.py',
                 'research/model_v2/DOWNSTREAM_COMPLETION_SPEC.md',
                 'research/model_v2/M3_EVALUATOR_SPEC.md', 'uv.lock'):
        assert path in closure


def test_prerequisites_refuse_without_a_committed_m2_result(tmp_path):
    with pytest.raises((prov.ExecutionRefused, FileNotFoundError)):
        m3.verified_m2(primary=tmp_path / 'missing', conditional=tmp_path / 'missing_conditional')
    with pytest.raises((prov.ExecutionRefused, FileNotFoundError)):
        m3.verified_station(tmp_path / 'missing_station')


def test_the_retained_static_specification_must_agree_with_the_label(tmp_path, monkeypatch):
    primary = tmp_path / 'primary'
    primary.mkdir()
    for name in ev.DETERMINISTIC_ARTIFACTS:
        (primary / name).write_text('{}\n')
    manifest = {'artifact_sha256': {n: prov.sha256(primary / n) for n in ev.DETERMINISTIC_ARTIFACTS},
                'integrity_passed': True, 'verdict': 'not supported', 'retained_static_specification': 'M2'}
    (primary / ev.RESULT_MANIFEST).write_text(json.dumps(manifest))
    conditional = tmp_path / 'conditional'
    conditional.mkdir()
    (conditional / 'm2_conditional_not_run.json').write_text('{}')
    monkeypatch.setattr(prov, 'tracked_and_pushed', lambda paths: {'commit': 'c'})
    monkeypatch.setattr(m3, '_repo_paths', lambda directory, names: list(names))
    with pytest.raises(prov.ExecutionRefused, match='inconsistent with the label'):
        m3.verified_m2(primary=primary, conditional=conditional)


# --- end-to-end on synthetic data, and real-input prerequisites without any spatial fit -------------------

@pytest.mark.parametrize('static', ['M0*', 'M2'])
@pytest.mark.parametrize('weights', ['station', 'land'])
def test_evaluate_end_to_end_on_synthetic_data(synthetic, monkeypatch, tmp_path, static, weights):
    scoring, dist = synthetic
    models = [ev.COMPARATOR] + ([ev.CANDIDATE] if static == 'M2' else [])
    scored = {(r, m): ev.score_model(m, scoring.frames[r], scoring, expected_full=False)
              for r in ev.REPRESENTATIONS for m in models}
    monkeypatch.setattr(m3, 'reproduce_static', lambda scoring, models, primary=None: {'record': {}, 'scored': scored})
    monkeypatch.setattr(m3, 'arm_distance', lambda scoring, weights: dist)
    result = m3.evaluate(scoring, tmp_path / 'out', weights=weights, static=static, provenance={'code': {'commit': 'x'}})
    assert result['integrity_passed'], result['integrity_failures']
    manifest = json.loads((tmp_path / 'out' / m3.RESULT_MANIFEST).read_text())
    assert sorted(manifest['artifact_sha256']) == sorted(m3.DETERMINISTIC_ARTIFACTS)
    for name, digest in manifest['artifact_sha256'].items():
        assert prov.sha256(tmp_path / 'out' / name) == digest
    if weights == 'station':
        naming = result['final_naming']
        assert naming['retained_static_specification'] == static
        assert set(naming) >= {'qualifying_spatial_extensions', 'final_primary_predictive_model'}
        assert 'qualification' in result['families']['primary_total_co2/sem']
    else:
        assert result['final_naming'] is None
        assert 'arm_conditions' in result['families']['primary_total_co2/sem']
    graphs = __import__('pandas').read_csv(tmp_path / 'out' / 'm3_graphs.csv')
    assert set(graphs.role) == {'training', 'held_out'}


def test_the_committed_m2_result_satisfies_the_m3_prerequisites(monkeypatch):
    """Manifests, label and branch of the committed M2 result. Push state is a run-time guard (recorded in each
    result's provenance), so here only tracking is checked: the test must not depend on whether the checked-out
    branch has been pushed."""
    import subprocess

    def tracked_only(paths):
        done = subprocess.run(['git', 'ls-files', '--error-unmatch', *paths], cwd=prov.ROOT, capture_output=True)
        assert done.returncode == 0, done.stderr
        return {'commit': 'not checked here'}

    monkeypatch.setattr(prov, 'tracked_and_pushed', tracked_only)
    m2 = m3.verified_m2()
    assert m2['retained_static_specification'] == 'M0*'
    assert m2['conditional_disposition'] == 'not executed'


def test_real_static_reproduction_and_graph_identities_without_any_spatial_fit(monkeypatch):
    from research.model_v2 import m0

    required = (m0.DEFAULT_INEQUALITY_PATH, m0.DEFAULT_FEATURES_PATH, m0.INCOME_PATH)
    if not all(path.is_file() for path in required):
        pytest.skip('integration check requires local pipeline country/features/income inputs')
    """Baseline-only: refits the committed retained static model (OLS) and checks both weight identities."""
    monkeypatch.setattr(ms, 'fit_spatial', lambda *a, **k: (_ for _ in ()).throw(AssertionError('no spatial fit')))
    scoring = ev.build_frozen()
    reproduction = m3.reproduce_static(scoring, [ev.COMPARATOR])
    assert all(v['max_prediction_difference'] <= m3.REFERENCE_ATOL for v in reproduction['record'].values())
    for weights in ('station', 'land'):
        assert m3.arm_distance(scoring, weights).shape == (151, 151)
