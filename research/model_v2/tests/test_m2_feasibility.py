"""M2 predictor-only feasibility audit: structure, identity pins and the no-outcome boundary.

Every frame here is synthetic. The committed audit output is checked read-only; the real audit is not
re-run by the test suite.
"""
import ast
import inspect
import itertools

import numpy as np
import pandas as pd
import pytest

from research.model_v2 import cv
from research.model_v2 import m2_feasibility as fe
from research.model_v2 import m2_latitude_basis as lb

ZONES, BLOCKS = list('ABCDE'), ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
INCOMES = ['High-income countries', 'Low-income countries', 'Lower-middle-income countries',
           'Upper-middle-income countries']


def synthetic_raw(n=48, seed=0):
    rng = np.random.default_rng(seed)
    hemisphere = np.where(np.arange(n) % 4 == 0, 'S', 'N')
    lat = np.where(hemisphere == 'S', rng.uniform(1, 45, n), rng.uniform(1, 66, n))
    total = 10 ** rng.uniform(1, 4, n)
    population = 10 ** rng.uniform(5.5, 9, n)
    cyc = lambda levels: [levels[i % len(levels)] for i in rng.permutation(n)]  # noqa: E731
    return pd.DataFrame({
        'Country': [f'Country {i}' for i in range(n)], 'iso3': [f'X{i:02d}' for i in range(n)],
        'cum_co2_per_capita': total * 1e6 / population, 'cum_co2_total': total,
        'abs_latitude': lat, 'elevation': rng.uniform(0, 2000, n), 'continentality': rng.uniform(0, 900, n),
        'climate_zone': cyc(ZONES), 'hemisphere': hemisphere, 'spatial_block': cyc(BLOCKS),
        'income_group': cyc(INCOMES), 'population': population, 'station_density': rng.uniform(0, 5, n),
        'm49_subregion': [f'R{i % 6}' for i in range(n)],
    })


def synthetic_predictors(n=48, seed=0):
    """Transformed numerics, as the audit reads them from the committed encoded design."""
    return fe.transform_numerics(synthetic_raw(n, seed))


# --- the no-outcome boundary --------------------------------------------------------------------

def test_the_allow_lists_hold_no_outcome_or_model_derived_column():
    forbidden = {'warming_trend', 'fitted', 'residual', 'leverage', 'studentized', 'cooks_d',
                 'station_trend', 'people_trend', 'era5_trend', 'abs_residual', 'sq_share',
                 'era5_minus_berkeley', 'station_minus_area', 'lisa_v1_local_i', 'lisa_v1_p',
                 'lisa_v1_quadrant', 'lisa_v1_label', 'lisa_area_label', 'cell_mean', 'cell_sd'}
    for columns in (fe.LABEL_COLUMNS, fe.TABLE_NUMERIC_COLUMNS, fe.FOLD_COLUMNS, fe.ENCODED_COLUMNS, fe.M1A_COLUMNS):
        assert not forbidden & set(columns)
        assert not any('trend' in c or 'resid' in c or 'lisa' in c for c in columns)


def test_the_audit_module_never_fits_or_names_an_outcome():
    source = inspect.getsource(fe)
    for token in ('lstsq', 'warming_trend', 'OUTCOME_COL', '.fit(', 'fit_predict', 'm0_complete_design',
                  'load_inputs', 'cross_validate', '.solve(', 'linalg.solve', 'pinv'):
        assert token not in source, token
    imported = {alias.name for node in ast.walk(ast.parse(source))
                if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not imported & {'m1b_evaluate', 'm0_complete_design', 'fit_m0', 'm1b_sensitivity'}


def test_a_read_with_an_outcome_column_requested_is_refused(tmp_path, monkeypatch):
    path = tmp_path / 'countries.csv'
    synthetic_raw().assign(warming_trend=0.1).to_csv(path, index=False)
    with pytest.raises(ValueError, match='outside the predictor allow-list'):
        fe.read_table_columns(path, None, [*fe.LABEL_COLUMNS, 'warming_trend'])


def test_a_pinned_input_with_the_wrong_digest_is_refused(tmp_path):
    path = tmp_path / 'countries.csv'
    synthetic_raw().to_csv(path, index=False)
    with pytest.raises(ValueError, match='digest'):
        fe.read_table_columns(path, '0' * 64, fe.LABEL_COLUMNS)


def test_the_audit_pins_equal_the_m1b_scoring_input_pins():
    from research.model_v2 import m1b_evaluate as ev
    sha, _commit = ev.FROZEN_SCORING_INPUTS[fe.COUNTRY_TABLE]
    assert fe.PINS[fe.COUNTRY_TABLE] == sha
    import json
    manifest = json.loads((fe.ROOT / fe.M1B_MANIFEST).read_text())
    assert fe.PINS[fe.FOLDS] == manifest['artifact_sha256']['m1b_cv_folds.csv']
    assert fe.PINS[fe.ENCODED_DESIGN] == manifest['artifact_sha256']['m1b_design_matrices.csv']
    assert fe.PINS[fe.M1A_FEATURES] == ev.FROZEN_SCORING_INPUTS[fe.M1A_FEATURES][0]


def test_encoded_design_column_values_are_screened_before_use():
    encoded = pd.DataFrame({'representation': ['primary_total_co2'] * 2, 'model': ['M0star'] * 2,
                            'iso3': ['X00', 'X00'], 'column': ['abs_latitude', 'warming_trend'],
                            'group': ['geography', 'outcome'], 'value': [1.0, 2.0]})
    with pytest.raises(ValueError, match='outside the predictor allow-list'):
        fe.encoded_numerics(encoded, 'primary_total_co2', ['X00'])


def test_encoding_never_calls_the_estimator(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('the audit fitted a model')
    monkeypatch.setattr(cv.V1Design, 'fit', forbidden)
    monkeypatch.setattr(np.linalg, 'lstsq', forbidden)
    frame = fe.representation_frame(synthetic_predictors(), 'primary_total_co2')
    record = fe.fit_record(frame, np.arange(40), np.arange(40, 48))
    assert record['added_identifiable_dimensions'] == 3


# --- the encoder is M0*'s encoder ---------------------------------------------------------------

@pytest.mark.parametrize('representation', ['primary_total_co2', 'per_capita'])
def test_the_encoder_reproduces_v1design_bit_for_bit_on_transformed_numerics(representation):
    raw = fe.representation_frame(synthetic_raw(), representation)
    transformed = fe.representation_frame(synthetic_predictors(), representation)
    manual = fe.baseline_encoder(transformed.iloc[:40])
    fitted = cv.V1Design().fit(raw.iloc[:40].assign(warming_trend=np.linspace(0, 1, 40)))  # synthetic only
    assert (manual.numeric, manual.categorical, manual.levels) == (fitted.numeric, fitted.categorical,
                                                                    fitted.levels)
    matrix, names = fe.encode(transformed, manual)
    expected, expected_names = fitted._matrix(raw)
    assert names == expected_names
    np.testing.assert_array_equal(matrix, expected)


@pytest.mark.parametrize('representation, dropped', [('primary_total_co2', 'cum_co2_per_capita'),
                                                     ('per_capita', 'cum_co2_total')])
def test_representation_frames_drop_exactly_the_other_responsibility_column(representation, dropped):
    frame = fe.representation_frame(synthetic_predictors(), representation)
    assert dropped not in frame.columns and set(frame.columns) == set(synthetic_raw().columns) - {dropped}
    assert set(fe.NUMERIC_COLUMNS[representation]) <= set(frame.columns)


# --- per-fit structure --------------------------------------------------------------------------

def test_a_well_supported_fit_adds_exactly_three_identifiable_dimensions():
    frame = fe.representation_frame(synthetic_predictors(), 'primary_total_co2')
    record = fe.fit_record(frame, np.arange(36), np.arange(36, 48))
    assert record['baseline_rank'] == record['baseline_columns']
    assert record['candidate_columns'] == record['baseline_columns'] + 3
    assert record['candidate_rank'] == record['candidate_columns']
    assert record['structural_pass'] and record['failure'] == ''
    state = lb.fit_state(frame.abs_latitude.iloc[:36])
    assert (record['knot_lower_boundary'], record['knot_interior_1'], record['knot_interior_2'],
            record['knot_upper_boundary']) == state.knots
    assert record['n_train_sh'] + record['n_train_nh'] == 36
    assert record['added_min_singular_value_unit_columns'] > 0


def test_a_training_set_without_southern_countries_fails_structurally_without_fallback():
    predictors = synthetic_predictors()
    north = np.flatnonzero(predictors.hemisphere == 'N')
    frame = fe.representation_frame(predictors, 'primary_total_co2')
    record = fe.fit_record(frame, north, np.flatnonzero(predictors.hemisphere == 'S'))
    assert not record['structural_pass']
    assert record['added_identifiable_dimensions'] < 3
    assert record['candidate_columns'] == record['baseline_columns'] + 3, 'no column was dropped'


def test_repeated_knots_are_recorded_as_a_structural_failure_not_repaired():
    predictors = synthetic_predictors()
    predictors.loc[:, 'abs_latitude'] = np.where(np.arange(len(predictors)) < 40, 12.0, 30.0)
    frame = fe.representation_frame(predictors, 'primary_total_co2')
    record = fe.fit_record(frame, np.arange(48), np.arange(0))
    assert not record['structural_pass'] and 'knots are not strictly increasing' in record['failure']


def test_held_out_extrapolation_is_counted_against_the_training_boundaries():
    predictors = synthetic_predictors()
    order = np.argsort(predictors.abs_latitude.to_numpy())
    frame = fe.representation_frame(predictors, 'primary_total_co2')
    test = np.array([order[0], order[-1]])
    record = fe.fit_record(frame, order[1:-1], test)
    assert (record['test_below_lower_boundary'], record['test_above_upper_boundary']) == (1, 1)


def test_coalitions_cover_all_sixteen_subsets_and_only_geography_coalitions_gain_three():
    frame = fe.representation_frame(synthetic_predictors(), 'primary_total_co2')
    rows = fe.coalition_records(frame)
    assert len(rows) == 16 == 2 ** len(fe.GROUPS)
    subsets = {tuple(r['groups'].split('+')) if r['groups'] else () for r in rows}
    assert subsets == {c for k in range(5) for c in itertools.combinations(fe.GROUPS, k)}
    for r in rows:
        gain = r['candidate_rank'] - r['baseline_rank']
        assert gain == (3 if 'geography' in r['groups'].split('+') else 0), r


# --- failures are recorded, never repaired ------------------------------------------------------

def _degenerate_frame():
    predictors = synthetic_predictors()
    predictors.loc[:, 'abs_latitude'] = np.where(np.arange(len(predictors)) < 40, 12.0, 30.0)
    return fe.representation_frame(predictors, 'primary_total_co2')


def test_a_degenerate_fit_is_summarized_as_a_failure_without_crashing():
    good = fe.representation_frame(synthetic_predictors(), 'primary_total_co2')
    rows = [{'measurement': 'station', 'representation': 'primary_total_co2', 'protocol': 'primary_loco',
             'fit': 'GOOD', **fe.fit_record(good, np.arange(40), np.arange(40, 48))},
            {'measurement': 'station', 'representation': 'primary_total_co2', 'protocol': 'primary_loco',
             'fit': 'BAD', **fe.fit_record(_degenerate_frame(), np.arange(48), np.arange(0))}]
    summary = fe._summary(pd.DataFrame(rows))['station/primary_total_co2/primary_loco']
    assert summary['all_structural_pass'] is False
    assert list(summary['failing_fits']) == ['BAD'] and 'degenerate' in summary['failing_fits']['BAD']
    assert summary['candidate_rank_equals_columns_in_all'] is False


def test_a_degenerate_full_sample_writes_failed_coalition_rows_instead_of_raising():
    rows = fe.coalition_records(_degenerate_frame())
    assert len(rows) == 16 and all(r['candidate_rank'] == -1 and 'degenerate' in r['failure'] for r in rows)


def test_the_m1a_arm_replaces_exactly_the_five_remeasured_features():
    frame = fe.representation_frame(synthetic_predictors(), 'primary_total_co2')
    m1a = frame[['iso3', *fe.M1A_REMEASURED, 'spatial_block']].copy()
    m1a['abs_latitude'] = m1a['abs_latitude'] + 1.0
    m1a['hemisphere'] = np.where(m1a['hemisphere'] == 'S', 'N', 'S')
    arm = fe.m1a_frame(frame, m1a)
    changed = [c for c in frame.columns if not frame[c].equals(arm[c])]
    assert set(changed) <= set(fe.M1A_REMEASURED) and {'abs_latitude', 'hemisphere'} <= set(changed)
    broken = m1a.assign(spatial_block='Elsewhere')
    with pytest.raises(ValueError, match='spatial_block'):
        fe.m1a_frame(frame, broken)


def test_the_canonical_record_refuses_uncommitted_or_unpushed_code_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, 'OUT', tmp_path / 'canonical')
    for state in ({'tracked': True, 'clean': False, 'pushed': True},
                  {'tracked': True, 'clean': True, 'pushed': False},
                  {'tracked': False, 'clean': False, 'pushed': False}):
        monkeypatch.setattr(fe, 'code_state', lambda state=state: {'commit': 'x', **state})
        with pytest.raises(RuntimeError, match='committed, unmodified, pushed'):
            fe.run(tmp_path / 'canonical')
        assert not (tmp_path / 'canonical').exists()


def test_the_code_path_covers_every_local_module_the_audit_depends_on():
    local = set()
    for node in ast.walk(ast.parse(inspect.getsource(fe))):
        if not isinstance(node, ast.ImportFrom) or not node.module or node.module.split('.')[0] not in ('src', 'research'):
            continue
        if node.module in ('src', 'research.model_v2'):
            local |= {f"{node.module.replace('.', '/')}/{alias.name}.py" for alias in node.names}
        else:
            local.add(f"{node.module.replace('.', '/')}.py")
    assert local == {'research/model_v2/cv.py', 'research/model_v2/m2_latitude_basis.py', 'src/feature_schema.py',
                     'src/decomposition.py', 'src/stability.py'}
    assert local | {'research/model_v2/m2_feasibility.py'} == set(fe.CODE)


# --- the committed record (read-only; the audit itself is never re-run by the suite) -----------------

def test_the_committed_feasibility_record_matches_its_code_and_its_manifest():
    import json
    summary_path = fe.OUT / 'm2_feasibility_summary.json'
    if not summary_path.exists():
        pytest.skip('the feasibility record is not committed')
    summary = json.loads(summary_path.read_text())
    assert summary['canonical'] is True
    assert summary['structural_feasibility'] == 'pass' and all(summary['checks'].values())
    code = summary['code']
    assert code['tracked'] is code['clean'] is code['pushed'] is True
    # The recorded digests describe the module bytes at the time of the run; the
    # public snapshot's modules differ from them only in comments and docstrings.
    assert set(code['sha256']) == set(fe.CODE)
    for path in code['sha256']:
        assert (fe.ROOT / path).exists(), path
    assert set(summary['inputs']) == set(fe.PINS)
    for path, entry in summary['inputs'].items():
        assert entry['sha256'] == entry['pinned'] == fe.PINS[path] == fe.sha256(fe.ROOT / path)
    manifest = json.loads((fe.OUT / 'm2_feasibility_manifest.json').read_text())
    assert set(manifest) == set(fe.RECORD_FILES)
    for name, digest in manifest.items():
        assert fe.sha256(fe.OUT / name) == digest, name
    full = summary['full_sample']['station/primary_total_co2']
    assert (full['baseline_rank'], full['candidate_rank']) == (20, 23)
    state = lb.LatitudeState.from_json(full['latitude_state'])
    assert state == lb.LatitudeState.from_json(summary['full_sample']['station/per_capita']['latitude_state'])
