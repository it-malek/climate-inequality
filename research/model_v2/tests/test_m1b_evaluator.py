"""M1b evaluator: gates, arms, verdict boundaries, equivalence, evidence and the conditional runners.

Every scored fixture here is synthetic. The real C2 package is used only through digest checks and
read-only gate functions; no test fits the real C2 against a real outcome.
"""
import dataclasses
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from research.model_v2 import m1b_evaluate as ev
from research.model_v2 import m1b_hydroclimate as hydro
from research.model_v2.m0 import ROOT
from research.model_v2.spatial import haversine_matrix, knn_weights

ZONES, HEMIS = list('ABCDE'), list('NS')
BLOCKS = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
INCOMES = ['High-income countries', 'Low-income countries', 'Lower-middle-income countries',
           'Upper-middle-income countries']
REGIONS = ['Northern Africa', 'Western Asia', 'Northern Europe', 'South America',
           'Eastern Asia', 'Western Africa']


def _levels(rng, levels, n):
    """Every level present at least twice, otherwise random: cyclic labels would be collinear."""
    base = list(levels) * 2
    return list(rng.permutation(base + list(rng.choice(levels, max(n - len(base), 0)))))[:n]


def synthetic_frozen(n=40, seed=0, effect=-0.04, iso3=None):
    """A frozen-shaped scoring frame with a synthetic outcome and synthetic C2."""
    rng = np.random.default_rng(seed)
    codes = iso3 or [f'X{i:02d}' for i in range(n)]
    countries = [f'Country {i}' for i in range(n)]
    lon, lat = rng.uniform(-170, 170, n), rng.uniform(-55, 65, n)
    total = 10 ** rng.uniform(1.0, 4.0, n)
    population = 10 ** rng.uniform(5.5, 9.0, n)
    c2 = rng.normal(-0.2, 0.5, n)
    design = pd.DataFrame({
        'Country': countries,
        'warming_trend': (0.15 + 0.01 * np.log10(total) + 0.0005 * np.abs(lat)
                          + effect * c2 + rng.normal(0, 0.01, n)),
        'cum_co2_per_capita': total * 1e6 / population,
        'cum_co2_total': total,
        'abs_latitude': np.abs(lat), 'elevation': rng.uniform(0, 2000, n),
        'continentality': rng.uniform(0, 900, n),
        'climate_zone': _levels(rng, ZONES, n), 'hemisphere': _levels(rng, HEMIS, n),
        'spatial_block': _levels(rng, BLOCKS, n), 'income_group': _levels(rng, INCOMES, n),
        'population': population, 'station_density': rng.uniform(0, 5, n),
    })
    table = pd.DataFrame({'Country': countries, 'iso3': codes,
                          'm49_subregion': [REGIONS[i % len(REGIONS)] for i in range(n)],
                          'station_lon': lon, 'station_lat': lat})
    distance = haversine_matrix(lon, lat)
    np.fill_diagonal(distance, 0.0)
    weights = knn_weights(distance, 8)
    return ev.Frozen(design=design, table=table, c2=c2, distance=distance, station_w=weights,
                     area_w=weights, area_centroids=np.column_stack([lon, lat]),
                     folds=ev.fold_plan(table, distance, n), identity={'n': n, 'synthetic': True})


def reference_from(frozen, expected=None):
    """Score each comparator once so the reference-reproduction path is exercised for real."""
    arms = ev.build_arms(frozen)
    return {r: {'card': ev.score_model(arm['frames'][ev.COMPARATOR], frozen,
                                       expected_columns=expected).card,
                'metrics': ev.REFERENCE_METRICS} for r, arm in arms.items()}


@pytest.fixture(scope='module')
def frozen():
    return synthetic_frozen()


@pytest.fixture(scope='module')
def reference(frozen):
    return reference_from(frozen, expected=ev.EXPECTED_COLUMNS['comparator'])


@pytest.fixture(scope='module')
def evaluated(tmp_path_factory, frozen, reference):
    out = tmp_path_factory.mktemp('run') / 'primary'
    result = ev.evaluate(frozen, out, provenance={'code': {'commit': 'a' * 40}},
                         reference_cards=reference)
    return result, out


# --- arms and their construction -------------------------------------------------------------

def test_arms_are_the_two_contracted_representations_with_the_identical_c2(frozen):
    arms = ev.build_arms(frozen)
    assert sorted(arms) == ['per_capita', 'primary_total_co2']
    for representation, arm in arms.items():
        comparator, candidate = arm['frames'][ev.COMPARATOR], arm['frames'][ev.CANDIDATE]
        dropped = ev.REPRESENTATIONS[representation]['drop']
        assert dropped not in comparator.columns and dropped not in candidate.columns
        assert ev.FEATURE not in comparator.columns
        np.testing.assert_array_equal(candidate[ev.FEATURE].to_numpy(), frozen.c2)
        pd.testing.assert_frame_equal(candidate.drop(columns=ev.FEATURE), comparator)
    primary = arms['primary_total_co2']['frames'][ev.CANDIDATE]
    per_capita = arms['per_capita']['frames'][ev.CANDIDATE]
    np.testing.assert_array_equal(primary.warming_trend.to_numpy(), per_capita.warming_trend.to_numpy())
    np.testing.assert_array_equal(primary[ev.FEATURE].to_numpy(), per_capita[ev.FEATURE].to_numpy())


def test_full_sample_rank_expectations_hold_for_both_representations(frozen, evaluated):
    result, _ = evaluated
    for representation in ev.REPRESENTATIONS:
        entry = result['representations'][representation]
        assert (entry['M0star']['matrix_columns'], entry['M0star']['matrix_rank']) == (20, 20)
        assert (entry['M1b']['matrix_columns'], entry['M1b']['matrix_rank']) == (21, 21)


def test_rank_expectation_is_not_imposed_on_training_folds(evaluated):
    result, _ = evaluated
    fits = result['representations']['primary_total_co2']['M1b']['training_fits']
    assert fits['all_full_rank'] and fits['feature_identified_in_all']
    assert max(fits['column_counts']) == 21


def test_a_singleton_level_gives_a_reduced_column_training_fold_and_an_unseen_level_row():
    frozen = synthetic_frozen(n=36, seed=13)
    zones = ['A'] * 12 + ['B'] * 12 + ['C'] * 11 + ['E']      # 'E' has exactly one member
    frozen = dataclasses.replace(frozen, design=frozen.design.assign(climate_zone=zones))
    candidate = frozen.design.drop(columns='cum_co2_per_capita').reset_index(drop=True)
    candidate[ev.FEATURE] = frozen.c2
    scored = ev.score_model(candidate, frozen, expected_columns=None)
    assert scored.card['rows_with_unseen_level'] >= 1
    counts = scored.card['training_fits']['column_counts']
    assert min(counts) < max(counts), 'the singleton level must drop a column in its own fold'
    singleton = next(f for f in scored.fits if f['held_out_iso3'] == frozen.table.iso3.iloc[-1])
    assert 'climate_zone=E' not in singleton['columns']
    assert singleton['feature_identified'] and singleton['matrix_rank'] == singleton['matrix_columns']
    assert np.isfinite(scored.results[ev.PRIMARY].yhat).all()
    assert scored.card['unseen_rows_by_categorical'][ev.PRIMARY]['climate_zone'] >= 1


def test_both_comparators_are_scored_before_either_candidate(frozen, reference, monkeypatch):
    order, real = [], ev.score_model

    def spy(frame, *args, **kwargs):
        order.append(ev.FEATURE in frame.columns)
        return real(frame, *args, **kwargs)
    monkeypatch.setattr(ev, 'score_model', spy)
    ev.score_representations(frozen, reference_cards=reference)
    assert order == [False, False, True, True]


def test_a_comparator_that_misses_its_reference_card_refuses(frozen, reference):
    broken = {r: {'card': {**entry['card'], 'cv_rmse': entry['card']['cv_rmse'] + 1e-9},
                  'metrics': entry['metrics']} for r, entry in reference.items()}
    with pytest.raises(ValueError, match='does not reproduce the frozen card'):
        ev.score_representations(frozen, reference_cards=broken)


def test_reference_check_rejects_non_finite_values(reference):
    card = reference['primary_total_co2']['card']
    with pytest.raises(ValueError, match='not finite'):
        ev.reference_check({**card, 'cv_rmse': float('nan')}, card, 'x')


# --- representation equivalence and allocation differences ------------------------------------

def test_representation_predictions_and_non_allocation_scores_agree(evaluated):
    result, _ = evaluated
    equivalence = result['representation_equivalence']
    assert equivalence['passes']
    for model in (ev.COMPARATOR, ev.CANDIDATE):
        assert max(equivalence['predictions'][model].values()) <= ev.PREDICTION_ATOL
        assert max(equivalence['scores'][model].values()) <= ev.PREDICTION_ATOL
    assert equivalence['coefficients']['baseline_dryness_full'] <= ev.COEF_ATOL
    assert equivalence['coefficients']['baseline_dryness_training_max'] <= ev.COEF_ATOL
    assert equivalence['coefficients']['training_fits_compared'] > 0


def test_distinct_shapley_allocations_are_permitted_and_reported(evaluated):
    result, _ = evaluated
    shares = result['representation_equivalence']['allocations_may_differ'][ev.CANDIDATE]
    assert set(shares) <= set(ev.NAMED_GROUPS) and 'hydroclimate' in shares
    assert any(abs(entry['difference']) > 0 for entry in shares.values())
    assert result['representation_equivalence']['passes']


def test_named_shares_and_residual_accounting(evaluated):
    result, _ = evaluated
    for representation in ev.REPRESENTATIONS:
        for model in (ev.COMPARATOR, ev.CANDIDATE):
            accounting = result['representations'][representation][model]['share_accounting']
            assert accounting['passes']
            assert abs(accounting['named_share_sum_minus_in_sample_r2']) <= ev.SHARE_ATOL
            assert abs(accounting['named_plus_residual_minus_one']) <= ev.SHARE_ATOL


def test_diagnostics_report_the_three_dimension_summary_and_context(evaluated):
    result, _ = evaluated
    for representation in ev.REPRESENTATIONS:
        summary = result['representations'][representation]['diagnostics']['three_dimension_summary']
        assert set(summary) >= {'transfer', 'spatial_specification', 'stability'}
        assert summary['transfer'] in {'improved generalization', 'worsened generalization',
                                       'no detectable change in transfer'}
        assert summary['stability'] in {'stable', 'material change'}
    assert 'not a predictor built only from information' in result['provenance_limitation']
    assert result['redundancy_context'] is None          # supplied by main() from the verified record


def test_redundancy_context_is_read_from_the_verified_record():
    inputs_record = json.loads((hydro.OUT / 'm1b_redundancy_diagnostic.json').read_text())
    context = ev.redundancy_context_from(inputs_record)
    assert context['r2_full_m0star'] == ev.ACCEPTED_REDUNDANCY['r2_full_m0star']
    assert context['vif'] == ev.ACCEPTED_REDUNDANCY['vif']
    assert context['rank_without_c2'] == 20 and context['rank_with_c2'] == 21
    assert context['near_redundant_flag'] is False
    assert context['r2_by_group_alone']['geography'] == pytest.approx(0.6215, abs=1e-4)


def test_support_quality_context_comes_from_the_verified_checkpoint():
    context = ev.support_context()
    lowest = [row[0] for row in context['lowest_pre_station_supported_share']]
    assert lowest == ['SAU', 'YEM', 'HTI', 'OMN', 'TCD']
    assert set(context['pre_station_support_distribution']) == {
        'pre_station_supported_share', 'pre_mean_station_count', 'pre_pure_climatology_share'}
    assert 'no station-count threshold' in context['no_exclusion_threshold']


def test_secondary_and_random_cards_carry_no_self_paired_field(evaluated):
    result, _ = evaluated
    for representation in ev.REPRESENTATIONS:
        for model in (ev.COMPARATOR, ev.CANDIDATE):
            card = result['representations'][representation][model]
            assert 'paired_delta_vs_m0' not in card['secondary']
            assert 'paired_delta_vs_m0' not in card['random_reference_only']
        paired = result['representations'][representation]['paired_primary_comparison']
        assert paired['resamples'] == 2000 and paired['seed'] == 0
        assert paired['delta_rmse'] != 0.0


# --- the verdict ------------------------------------------------------------------------------

def cards(m49_comparator=0.0389, m49_candidate=0.0389, region_comparator=0.0694, region_candidate=0.0694):
    comparator = {'secondary': {'cv_rmse': m49_comparator}, 'worst_region': {'rmse': region_comparator}}
    candidate = {'secondary': {'cv_rmse': m49_candidate}, 'worst_region': {'rmse': region_candidate}}
    return comparator, candidate


def paired(delta, lo, hi):
    return {'delta_rmse': delta, 'country_bootstrap_95_interval': [lo, hi], 'resamples': 2000, 'seed': 0}


def sign(full=-0.02, negative_n=140, n_fits=151, zero_n=0):
    from fractions import Fraction
    fraction = Fraction(negative_n, n_fits)
    return {'full': full, 'negative_n': negative_n, 'n_fits': n_fits,
            'negative_fraction': float(fraction), 'zero_n': zero_n,
            'negative_and_stable': bool(full < 0 and fraction >= ev.SIGN_STABILITY)}


def test_two_level_verdict_labels():
    comparator, candidate = cards()
    promoted = ev.verdict(comparator, candidate, paired(-0.003, -0.005, -0.001), sign())
    assert promoted['verdict'] == 'supported and promoted' and promoted['promoted']
    sub = ev.verdict(comparator, candidate, paired(-0.001, -0.002, -0.0002), sign())
    assert sub['verdict'] == 'supported but sub-material / not promoted'
    assert sub['association_supported'] and not sub['A2_practical_le_minus_0.002']
    vetoed = ev.verdict(*cards(m49_candidate=0.0405), paired(-0.003, -0.005, -0.001), sign())
    assert vetoed['verdict'] == 'supported but sub-material / not promoted'
    assert not vetoed['A3_m49_veto_passed'] and vetoed['association_supported']
    wrong_sign = ev.verdict(comparator, candidate, paired(-0.003, -0.005, -0.001), sign(full=0.02))
    assert wrong_sign['verdict'] == 'not supported' and wrong_sign['improvement_without_prestated_sign']
    unstable = ev.verdict(comparator, candidate, paired(-0.003, -0.005, -0.001), sign(negative_n=120))
    assert unstable['verdict'] == 'not supported' and not unstable['S2_sign_negative_and_stable']
    covers_zero = ev.verdict(comparator, candidate, paired(-0.003, -0.006, 0.0001), sign())
    assert covers_zero['verdict'] == 'not supported'
    worse = ev.verdict(comparator, candidate, paired(0.003, 0.001, 0.005), sign())
    assert worse['worsened_generalization'] and worse['verdict'] == 'not supported'


@pytest.mark.parametrize('negative_n, supported', [(120, False), (121, True)])
def test_sign_stability_boundary_at_eighty_percent_of_151(negative_n, supported):
    comparator, candidate = cards()
    out = ev.verdict(comparator, candidate, paired(-0.003, -0.005, -0.001), sign(negative_n=negative_n))
    assert out['S2_sign_negative_and_stable'] is supported
    assert ev.verdict(comparator, candidate, paired(-0.003, -0.005, -0.001),
                      sign(negative_n=4, n_fits=5))['S2_sign_negative_and_stable'] is True
    assert ev.verdict(comparator, candidate, paired(-0.003, -0.005, -0.001),
                      sign(negative_n=3, n_fits=5))['S2_sign_negative_and_stable'] is False


@pytest.mark.parametrize('delta, practical', [(-0.002, True), (-0.0019999999, False)])
def test_practical_threshold_boundary(delta, practical):
    out = ev.verdict(*cards(), paired(delta, -0.005, -0.001), sign())
    assert out['A2_practical_le_minus_0.002'] is practical


@pytest.mark.parametrize('change, passes', [(0.001, True), (0.0010000001, False)])
def test_veto_boundaries_at_one_thousandth(change, passes):
    # Differences are built exactly, so the boundary itself is tested rather than float noise.
    m49 = ev.verdict(*cards(m49_comparator=0.0, m49_candidate=change),
                     paired(-0.003, -0.005, -0.001), sign())
    assert m49['A3_m49_veto_passed'] is passes and m49['m49_rmse_change'] == change
    region = ev.verdict(*cards(region_comparator=0.0, region_candidate=change),
                        paired(-0.003, -0.005, -0.001), sign())
    assert region['A4_worst_region_veto_passed'] is passes


@pytest.mark.parametrize('delta, hi, s1', [(-1e-12, -1e-12, True), (0.0, -0.001, False), (-0.001, 0.0, False)])
def test_s1_requires_a_strictly_negative_delta_and_interval(delta, hi, s1):
    assert ev.verdict(*cards(), paired(delta, -0.01, hi), sign())['S1_interval_below_zero'] is s1


def test_verdict_refuses_non_finite_inputs():
    for bad in [paired(float('nan'), -0.005, -0.001), paired(-0.003, -0.005, float('inf'))]:
        with pytest.raises(ValueError, match='not finite'):
            ev.verdict(*cards(), bad, sign())
    with pytest.raises(ValueError, match='not finite'):
        ev.verdict(*cards(), paired(-0.003, -0.005, -0.001), sign(full=float('nan')))


def test_sign_condition_counts_strictly_negative_and_requires_every_fit():
    summary = {ev.FEATURE: {'full': -0.05, 'fold_values_n': 5, 'fold_values_finite_n': 5,
                            'negative_n': 4, 'zero_n': 1}}
    out = ev.sign_condition(summary, 5)
    assert out['negative_n'] == 4 and out['zero_n'] == 1 and out['negative_and_stable'] is True
    with pytest.raises(ValueError, match='exactly 6 finite'):
        ev.sign_condition(summary, 6)
    missing = {ev.FEATURE: {**summary[ev.FEATURE], 'fold_values_finite_n': 4}}
    with pytest.raises(ValueError, match='exactly 5 finite'):
        ev.sign_condition(missing, 5)


def test_zero_is_not_negative_in_the_coefficient_summary():
    full = type('Fit', (), {'columns': [ev.FEATURE], 'coef': np.array([-0.1])})()
    fits = [{'columns': [ev.FEATURE], 'coef': [v]} for v in (-1.0, 0.0, 1.0, -2.0)]
    summary = ev.coefficient_summary(full, fits)[ev.FEATURE]
    assert summary['negative_n'] == 2 and summary['zero_n'] == 1
    assert summary['negative_fraction'] == 0.5 and summary['fold_values_finite_n'] == 4


def test_the_verdict_is_a_pure_function_of_the_primary_representation(evaluated):
    result, _ = evaluated
    primary = result['representations']['primary_total_co2']
    recomputed = ev.verdict(primary['M0star'], primary['M1b'], primary['paired_primary_comparison'],
                            ev.sign_condition(primary['M1b']['coefficients'], result['verdict']['coefficient_n_fits']))
    assert recomputed == primary['verdict'] == result['verdict']
    # Exactly one contract verdict is reported; the per-capita arm carries a descriptive label only.
    per_capita = result['representations']['per_capita']
    assert 'verdict' not in per_capita and not ev.REPRESENTATIONS['per_capita']['decides_verdict']
    conditions = per_capita['representation_conditions']
    assert conditions['label_descriptive_only'] in {'supported and promoted', 'not supported',
                                                    'supported but sub-material / not promoted'}
    assert 'never determines acceptance' in conditions['note']
    assert conditions['S1_interval_below_zero'] in (True, False)


def test_per_capita_arms_are_materialized_even_when_the_association_fails(tmp_path):
    frozen = synthetic_frozen(n=40, seed=3, effect=+0.05)     # a positive, wrong-signed effect
    reference = reference_from(frozen, expected=ev.EXPECTED_COLUMNS['comparator'])
    result = ev.evaluate(frozen, tmp_path / 'run', provenance={'code': {'commit': 'b' * 40}},
                         reference_cards=reference)
    assert result['verdict']['verdict'] == 'not supported'
    assert set(result['representations']) == {'primary_total_co2', 'per_capita'}
    for representation in ev.REPRESENTATIONS:
        entry = result['representations'][representation]
        assert entry['M1b']['coefficients'][ev.FEATURE]['full'] > 0
        assert entry['paired_primary_comparison']['resamples'] == 2000


# --- evidence, determinism and refusals --------------------------------------------------------

def test_every_deterministic_artifact_is_written_and_digested(evaluated):
    result, out = evaluated
    manifest = json.loads((out / ev.RESULT_MANIFEST).read_text())
    assert sorted(manifest['artifact_sha256']) == sorted(ev.DETERMINISTIC_ARTIFACTS)
    for name, digest in manifest['artifact_sha256'].items():
        assert hydro.sha256(out / name) == digest
    assert manifest['verdict'] == result['verdict']['verdict']
    assert manifest['association_supported'] == result['verdict']['association_supported']


def test_saved_predictions_and_coefficients_support_independent_recomputation(evaluated, frozen):
    _result, out = evaluated
    predictions = pd.read_csv(out / 'm1b_country_predictions.csv')
    assert set(predictions.protocol) == {'in_sample', *ev.PROTOCOLS}
    assert len(predictions) == 2 * 2 * 4 * frozen.n
    primary = predictions[(predictions.representation == 'primary_total_co2')
                          & (predictions.model == ev.CANDIDATE)
                          & (predictions.protocol == ev.PRIMARY)]
    y = frozen.design.warming_trend.to_numpy()
    rmse = float(np.sqrt(np.mean((y - primary.prediction.to_numpy()) ** 2)))
    card = json.loads((out / 'm1b_scorecard.json').read_text())
    assert abs(rmse - card['representations']['primary_total_co2']['M1b']['cv_rmse']) < 1e-12
    coefficients = pd.read_csv(out / 'm1b_coefficients.csv')
    training = coefficients[(coefficients.fit == 'primary_training')
                            & (coefficients.column == ev.FEATURE)
                            & (coefficients.representation == 'primary_total_co2')]
    assert len(training) == frozen.n and training.held_out_iso3.nunique() == frozen.n
    folds = pd.read_csv(out / 'm1b_cv_folds.csv')
    assert len(folds) == frozen.n and folds.primary_training_iso3.str.len().min() > 0
    designs = pd.read_csv(out / 'm1b_design_matrices.csv')
    assert set(designs.group) >= {'intercept', 'geography', 'hydroclimate'}
    coalitions = pd.read_csv(out / 'm1b_shapley_coalitions.csv')
    candidate = coalitions[(coalitions.model == ev.CANDIDATE)
                           & (coalitions.representation == 'primary_total_co2')]
    assert len(candidate) == 2 ** 5 and candidate.r2.max() <= 1.0


def test_two_runs_of_unchanged_code_are_byte_identical(tmp_path):
    frozen = synthetic_frozen(n=28, seed=7)
    reference = reference_from(frozen, expected=ev.EXPECTED_COLUMNS['comparator'])
    kwargs = {'provenance': {'code': {'commit': 'c' * 40}}, 'reference_cards': reference}
    ev.evaluate(frozen, tmp_path / 'a', **kwargs)
    ev.evaluate(frozen, tmp_path / 'b', **kwargs)
    for name in ev.DETERMINISTIC_ARTIFACTS:
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()


def test_a_run_never_overwrites_a_prior_run(tmp_path, frozen, reference):
    ev.evaluate(frozen, tmp_path / 'once', provenance={'code': {'commit': 'd' * 40}},
                reference_cards=reference)
    with pytest.raises(FileExistsError):
        ev.evaluate(frozen, tmp_path / 'once', provenance={'code': {'commit': 'd' * 40}},
                    reference_cards=reference)


def test_deterministic_paired_resampling():
    y = np.linspace(0.1, 0.3, 40)
    new, old = y + 0.01, y + 0.02
    first = ev.paired_rmse_interval(y, new, old)
    assert first == ev.paired_rmse_interval(y, new, old)
    assert first['resamples'] == 2000 and first['seed'] == 0 and first['delta_rmse'] < 0


# --- the execution boundary --------------------------------------------------------------------

def test_main_refuses_before_reading_any_gate_or_input_when_the_code_is_not_frozen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('scoring touched an input before the provenance guard')
    for name in ('verify_scoring_inputs', 'build_frozen', 'evaluate'):
        monkeypatch.setattr(ev, name, forbidden)
    monkeypatch.setenv('PYTHONHASHSEED', '0')

    def dirty(*args, check=True):
        if args[0] == 'ls-files':
            return subprocess.CompletedProcess(args, 0, '', '')
        if args[0] == 'diff' and '--quiet' in args:
            return subprocess.CompletedProcess(args, 1, '', '')
        if args[0] == 'diff':
            return subprocess.CompletedProcess(args, 0, 'research/model_v2/m1b_evaluate.py\n', '')
        return subprocess.CompletedProcess(args, 0, 'deadbeef\n', '')
    monkeypatch.setattr(ev, '_git', dirty)
    with pytest.raises(RuntimeError, match='differs from HEAD'):
        ev.main([])


def test_main_refuses_an_unpushed_commit(monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', '0')

    def unpushed(*args, check=True):
        if args[0] == 'merge-base':
            return subprocess.CompletedProcess(args, 1, '', '')
        if args[0] == 'diff' or args[0] == 'ls-files':
            return subprocess.CompletedProcess(args, 0, '', '')
        return subprocess.CompletedProcess(args, 0, 'deadbeef\n', '')
    monkeypatch.setattr(ev, '_git', unpushed)
    with pytest.raises(RuntimeError, match='is not contained in'):
        ev.frozen_code_state()


def test_main_refuses_untracked_code_path_files(monkeypatch):
    def untracked(*args, check=True):
        if args[0] == 'ls-files':
            return subprocess.CompletedProcess(args, 0, 'research/model_v2/m1b_evaluate.py\n', '')
        return subprocess.CompletedProcess(args, 0, 'deadbeef\n', '')
    monkeypatch.setattr(ev, '_git', untracked)
    with pytest.raises(RuntimeError, match='untracked'):
        ev.frozen_code_state()


def test_scoring_requires_pythonhashseed_zero(monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', '73')
    with pytest.raises(RuntimeError, match='PYTHONHASHSEED=0'):
        ev.require_hash_seed()
    monkeypatch.delenv('PYTHONHASHSEED', raising=False)
    with pytest.raises(RuntimeError, match='PYTHONHASHSEED=0'):
        ev.require_hash_seed()


def test_the_live_code_path_is_committed_and_the_pins_match_the_repository():
    for path in ev.CODE_PATH:
        assert (ROOT / path).exists(), path
    for relative, (pin, _commit) in ev.FROZEN_SCORING_INPUTS.items():
        assert hydro.sha256(ROOT / relative) == pin, relative
    for name, pin in ev.ACCEPTED_PACKAGE.items():
        assert hydro.sha256(hydro.OUT / name) == pin, name


def test_a_refused_pre_scoring_check_leaves_no_output_directory(tmp_path, frozen, reference):
    broken = {r: {'card': {**entry['card'], 'in_sample_r2': entry['card']['in_sample_r2'] + 1e-6},
                  'metrics': entry['metrics']} for r, entry in reference.items()}
    out = tmp_path / 'refused'
    with pytest.raises(ValueError, match='does not reproduce the frozen card'):
        ev.evaluate(frozen, out, provenance={'code': {'commit': 'e' * 40}}, reference_cards=broken)
    assert not out.exists(), 'a refusal must not poison the output path for a later retry'


def test_reproducibility_record_compares_deterministic_artifacts_only(tmp_path, frozen, reference):
    kwargs = {'provenance': {'code': {'commit': 'f' * 40}}, 'reference_cards': reference}
    ev.evaluate(frozen, tmp_path / 'a', **kwargs)
    ev.evaluate(frozen, tmp_path / 'b', **kwargs)
    (tmp_path / 'a' / ev.RUN_METADATA).write_text('{"host": "one"}\n')
    (tmp_path / 'b' / ev.RUN_METADATA).write_text('{"host": "two"}\n')
    record = ev.reproducibility_record(tmp_path / 'a', tmp_path / 'b')
    assert record['byte_identical'] and ev.RUN_METADATA in record['excluded_as_run_metadata']
    assert set(record['artifacts']) == {*ev.DETERMINISTIC_ARTIFACTS, ev.RESULT_MANIFEST}
    (tmp_path / 'b' / 'm1b_cv_folds.csv').write_text('changed\n')
    differing = ev.reproducibility_record(tmp_path / 'a', tmp_path / 'b')
    assert not differing['byte_identical']
    assert not differing['artifacts']['m1b_cv_folds.csv']['identical']
