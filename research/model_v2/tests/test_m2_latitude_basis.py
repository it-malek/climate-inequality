"""M2 latitude functional form: the frozen natural cubic basis and the H1 interaction (M2_DESIGN.md).

Synthetic reference calculations only. No outcome exists anywhere in this module or its tests.
"""
import json

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from research.model_v2 import m2_latitude_basis as lb


@pytest.fixture
def latitudes():
    return np.random.default_rng(0).uniform(0.5, 66.0, 140)


# --- the frozen state ---------------------------------------------------------------------------

def test_knots_are_training_extrema_and_linear_method_tertiles(latitudes):
    state = lb.fit_state(latitudes)
    q1, q2 = np.quantile(latitudes, [1 / 3, 2 / 3], method='linear')
    assert state.knots == (float(latitudes.min()), float(q1), float(q2), float(latitudes.max()))
    assert state.n_train == 140 and state.version == lb.BASIS_VERSION


def test_the_quantile_convention_is_numpy_linear_which_is_hyndman_fan_type_7():
    x = np.array([0.0, 1.0, 2.0, 10.0])
    state = lb.fit_state(x)
    # Type 7: h = (n - 1) p; 3 * 1/3 = 1 -> exactly x[1]; 3 * 2/3 = 2 -> exactly x[2].
    assert state.knots == (0.0, 1.0, 2.0, 10.0)
    y = np.array([0.0, 3.0, 4.0, 5.0, 9.0])
    # h = 4/3 -> 3 + (4 - 3)/3; h = 8/3 -> 4 + (5 - 4) * 2/3
    assert lb.fit_state(y).knots[1:3] == pytest.approx((3 + 1 / 3, 4 + 2 / 3), abs=1e-15)


def test_the_state_depends_on_training_values_only_not_their_order(latitudes):
    shuffled = np.random.default_rng(1).permutation(latitudes)
    assert lb.fit_state(shuffled) == lb.fit_state(latitudes)


def test_ties_are_permitted_when_the_four_knots_stay_distinct():
    x = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 9.0, 12.0, 12.0, 20.0])
    state = lb.fit_state(x)
    assert all(b - a > lb.KNOT_GAP_ATOL for a, b in zip(state.knots, state.knots[1:]))


@pytest.mark.parametrize('values', [
    [3.0, 3.0, 3.0, 3.0, 3.0],                  # constant
    [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 9.0],        # both tertiles on one tied value
    [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 9.0],        # lower tertile on the lower boundary
    [1.0, 4.0, 9.0, 9.0, 9.0, 9.0, 9.0],        # upper tertile on the upper boundary
])
def test_repeated_knots_raise_and_nothing_falls_back(values):
    with pytest.raises(lb.DegenerateLatitudeBasis, match='knots are not strictly increasing'):
        lb.fit_state(np.array(values))


@pytest.mark.parametrize('bad', [[1.0, np.nan, 3.0, 4.0, 5.0], [1.0, np.inf, 3.0, 4.0, 5.0],
                                 [-1.0, 2.0, 3.0, 4.0, 5.0], [[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0, 3.0]])
def test_invalid_training_latitudes_raise(bad):
    with pytest.raises(lb.DegenerateLatitudeBasis):
        lb.fit_state(np.array(bad, dtype=float))


def test_the_state_round_trips_through_its_serialization_exactly(latitudes):
    state = lb.fit_state(latitudes)
    text = state.to_json()
    assert lb.LatitudeState.from_json(text) == state
    record = json.loads(text)
    assert record['knots_hex'] == [float(k).hex() for k in state.knots]
    assert record['quantile_method'] == 'linear' and record['probabilities'] == ['1/3', '2/3']


@pytest.mark.parametrize('field, value', [('version', 'other'), ('quantile_method', 'nearest'),
                                          ('probabilities', ['1/4', '3/4']), ('boundary_knots', 'padded'),
                                          ('tails', 'cubic')])
def test_a_serialized_state_with_a_foreign_convention_is_refused(latitudes, field, value):
    record = json.loads(lb.fit_state(latitudes).to_json())
    with pytest.raises(ValueError, match=field):
        lb.LatitudeState.from_json(json.dumps({**record, field: value}))


def test_a_serialized_state_with_an_edited_knot_is_refused(latitudes):
    record = json.loads(lb.fit_state(latitudes).to_json())
    edited = {**record, 'knots': [record['knots'][0] + 1e-9, *record['knots'][1:]]}
    with pytest.raises(ValueError, match='hex'):
        lb.LatitudeState.from_json(json.dumps(edited))


@pytest.mark.parametrize('knots, n_train', [((5.0, 3.0, 3.0, 1.0), 10), ((0.0, 1.0, 1.0, 2.0), 10),
                                            ((0.0, 1.0, 2.0, float('nan')), 10), ((-1.0, 1.0, 2.0, 3.0), 10),
                                            ((0.0, 1.0, 2.0), 10), ((0.0, 1.0, 2.0, 3.0), 3)])
def test_a_state_cannot_be_built_around_the_degeneracy_rules(knots, n_train):
    with pytest.raises(lb.DegenerateLatitudeBasis):
        lb.LatitudeState(knots=knots, n_train=n_train)


def test_a_valid_state_serialized_by_hand_with_bad_knots_is_refused(latitudes):
    record = json.loads(lb.fit_state(latitudes).to_json())
    bad = [5.0, 3.0, 3.0, 1.0]
    with pytest.raises(lb.DegenerateLatitudeBasis):
        lb.LatitudeState.from_json(json.dumps({**record, 'knots': bad, 'knots_hex': [k.hex() for k in bad]}))


def test_the_state_is_immutable(latitudes):
    state = lb.fit_state(latitudes)
    with pytest.raises(AttributeError):
        state.knots = (0.0, 1.0, 2.0, 3.0)


# --- the basis ----------------------------------------------------------------------------------

def test_the_basis_is_the_esl_truncated_power_natural_spline_formula():
    state = lb.LatitudeState(knots=(0.0, 10.0, 25.0, 60.0), n_train=4)
    x = np.array([-5.0, 0.0, 5.0, 17.0, 40.0, 60.0, 75.0])

    def cube(t):
        return t * t * t

    def d(k):
        xi, last = state.knots[k], state.knots[-1]
        return (cube(np.maximum(x - xi, 0.0)) - cube(np.maximum(x - last, 0.0))) / (last - xi)
    expected = np.column_stack([d(0) - d(2), d(1) - d(2)])
    np.testing.assert_array_equal(lb.nonlinear_columns(state, x), expected)


def test_cubes_use_ieee_products_not_a_platform_pow_kernel():
    import inspect
    source = inspect.getsource(lb)
    assert '** 3' not in source and 'np.power' not in source and 'math.pow' not in source
    assert lb._cube(np.array([1.1]))[0] == 1.1 * 1.1 * 1.1


def test_with_the_intercept_and_linear_term_the_basis_spans_every_natural_cubic_spline_on_the_knots():
    """Independent reference: scipy's natural interpolating spline lies in span{1, a, N1, N2}."""
    state = lb.LatitudeState(knots=(0.8, 14.2, 33.7, 65.1), n_train=100)
    grid = np.linspace(0.8, 65.1, 400)
    basis = np.column_stack([np.ones_like(grid), grid, lb.nonlinear_columns(state, grid)])
    rng = np.random.default_rng(3)
    for _ in range(5):
        spline = CubicSpline(state.knots, rng.normal(size=4), bc_type='natural')
        target = spline(grid)
        coef, *_ = np.linalg.lstsq(basis, target, rcond=None)
        assert np.max(np.abs(basis @ coef - target)) < 1e-9
    assert np.linalg.matrix_rank(basis) == 4


def test_the_basis_is_twice_continuously_differentiable_at_every_knot():
    state = lb.LatitudeState(knots=(2.0, 11.0, 29.0, 58.0), n_train=10)
    h = 1e-4
    for knot in state.knots:
        left = lb.nonlinear_columns(state, np.array([knot - 2 * h, knot - h, knot]))
        right = lb.nonlinear_columns(state, np.array([knot, knot + h, knot + 2 * h]))
        second_left = (left[2] - 2 * left[1] + left[0]) / h ** 2
        second_right = (right[2] - 2 * right[1] + right[0]) / h ** 2
        np.testing.assert_allclose(second_left, second_right, atol=1e-2)
        np.testing.assert_allclose(left[2], right[0], rtol=0, atol=0)


@pytest.mark.parametrize('side', ['below', 'above'])
def test_the_tails_beyond_the_boundary_knots_are_exactly_linear(side):
    state = lb.LatitudeState(knots=(5.0, 12.0, 30.0, 50.0), n_train=10)
    x = np.array([0.0, 1.0, 2.0, 3.0]) if side == 'below' else np.array([50.0, 60.0, 70.0, 80.0])
    columns = lb.nonlinear_columns(state, x)
    np.testing.assert_allclose(np.diff(columns, n=2, axis=0), 0.0, atol=1e-8)
    if side == 'below':
        np.testing.assert_array_equal(columns, 0.0)


def test_held_out_rows_never_move_the_state_and_are_transformed_with_the_training_state(latitudes):
    train, test = latitudes[:100], np.array([0.1, 70.0, 30.0])
    state = lb.fit_state(train)
    alone = lb.nonlinear_columns(state, test)
    together = lb.nonlinear_columns(state, np.concatenate([train, test]))[-3:]
    np.testing.assert_array_equal(alone, together)
    assert lb.fit_state(train) == state
    assert lb.extrapolation(state, test) == {'below': 1, 'above': 1, 'min': 0.1, 'max': 70.0}


def test_the_hemisphere_interaction_is_the_southern_indicator_times_uncentred_latitude():
    a = np.array([10.0, 20.0, 30.0, 40.0])
    hemisphere = np.array(['N', 'S', 'S', 'N'])
    np.testing.assert_array_equal(lb.sh_interaction(hemisphere, a), [0.0, 20.0, 30.0, 0.0])
    assert lb.INTERACTION_CENTRE_DEGREES == 0.0


@pytest.mark.parametrize('labels', [['N', 'South', 'S'], ['n', 's', 'S'], ['N', None, 'S']])
def test_hemisphere_labels_outside_the_frozen_coding_raise(labels):
    with pytest.raises(ValueError, match='hemisphere'):
        lb.sh_interaction(np.array(labels, dtype=object), np.array([1.0, 2.0, 3.0]))


def test_added_columns_are_named_and_grouped_in_geography(latitudes):
    state = lb.fit_state(latitudes)
    names = lb.ADDED_COLUMNS
    assert names == ('abs_latitude_ns1', 'abs_latitude_ns2', 'hemisphere=S:abs_latitude')
    block = lb.added_block(state, latitudes, np.where(latitudes > 30, 'S', 'N'))
    assert block.shape == (140, 3) and lb.ADDED_GROUP == 'geography'


def test_the_module_never_touches_an_outcome():
    import ast
    import inspect
    source = inspect.getsource(lb)
    for token in ('warming', 'OUTCOME', 'lstsq', 'trend', 'residual', 'm0_complete_design', 'load_inputs'):
        assert token not in source, token
    names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
    assert not names & {'y', 'yhat', 'fit_predict'}
