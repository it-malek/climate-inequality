"""M3 explicit spatial structure: graphs, SEM/SAR maximum likelihood, prediction, uncertainty, accounting.

Synthetic reference calculations only: random point clouds, designs and outcomes drawn in the tests with
``numpy.random.default_rng`` and fixed seeds. No repository data file is read.
"""
import ast
import dataclasses
import inspect
import math
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import pytest

from research.model_v2 import m3_spatial as m3
from research.model_v2.spatial import haversine_matrix, knn_weights

BETA = np.array([0.2, 1.0, -0.5, 0.3])
COLUMNS = ('intercept', 'a', 'b', 'c')
GRID = np.linspace(-0.99, 0.99, 199)


def _cloud(n, seed):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180.0, 180.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-0.9, 0.95, n)))
    return haversine_matrix(lon, lat), lon, lat


def _design(n, seed, columns=3):
    return np.column_stack([np.ones(n), np.random.default_rng(seed).normal(size=(n, columns))])


def _simulate(family, theta, w, x, beta, rng):
    m_inv = np.linalg.inv(np.eye(len(x)) - theta * w)
    eps = rng.normal(size=len(x))
    return x @ beta + m_inv @ eps if family == 'sem' else m_inv @ (x @ beta + eps)


def _reason(call):
    with pytest.raises(m3.SpatialFitFailure) as info:
        call()
    return info.value.reason


@pytest.fixture(scope='module')
def cloud():
    dist, _, _ = _cloud(150, 0)
    return dist, m3.training_graph(dist, np.arange(150)), _design(150, 1)


@pytest.fixture(scope='module')
def fits(cloud):
    _, w, x = cloud
    out = {}
    for family, theta, seed in (('sem', 0.5, 2), ('sar', -0.3, 3)):
        y = _simulate(family, theta, w, x, BETA, np.random.default_rng(seed))
        out[family] = (y, m3.fit_spatial(family, y, x, w, columns=COLUMNS))
    return out


@pytest.fixture
def tied_distances():
    rng = np.random.default_rng(6)
    d = rng.integers(1, 4, size=(40, 40)).astype(float)
    d = np.minimum(d, d.T)
    np.fill_diagonal(d, 0.0)
    return d


# --- graphs -------------------------------------------------------------------------------------

def _brute_force_neighbours(dist, candidates, node):
    return sorted((int(j) for j in candidates if j != node), key=lambda j: (dist[node, j], j))[:8]


def test_the_training_graph_is_knn_weights_on_the_canonical_submatrix():
    dist, _, _ = _cloud(60, 4)
    train = np.sort(np.random.default_rng(5).choice(60, 40, replace=False))
    w = m3.training_graph(dist, train)
    np.testing.assert_array_equal(w, knn_weights(dist[np.ix_(train, train)], 8))
    assert (w.sum(axis=1) == 1.0).all() and (np.diagonal(w) == 0.0).all()
    assert ((w > 0).sum(axis=1) == 8).all() and set(np.unique(w)) == {0.0, 0.125}
    assert m3.K == 8 and m3.MIN_TRAINING_NODES == 9


def test_training_neighbours_break_distance_ties_by_canonical_index_and_match_the_graph(tied_distances):
    d = tied_distances
    train = np.sort(np.random.default_rng(7).choice(40, 25, replace=False))
    neighbours = m3.training_neighbours(d, train)
    assert neighbours.shape == (25, 8)
    for row, i in zip(neighbours, train):
        assert row.tolist() == _brute_force_neighbours(d, train, i)
    w = m3.training_graph(d, train)
    local = m3.local_positions(train, neighbours)
    for r in range(len(train)):
        assert set(np.flatnonzero(w[r]).tolist()) == set(local[r].tolist())


def test_held_out_nodes_attach_as_sinks_to_the_first_eight_training_nodes(tied_distances):
    d = tied_distances
    rng = np.random.default_rng(8)
    train = np.sort(rng.choice(40, 25, replace=False))
    test = np.setdiff1d(np.arange(40), train)
    neighbours, weights = m3.attachment(d, train, test)
    assert neighbours.shape == weights.shape == (15, 8)
    for row, o in zip(neighbours, test):
        assert row.tolist() == _brute_force_neighbours(d, train, o)
    assert (weights == 0.125).all() and (weights.sum(axis=1) == 1.0).all()
    w_star = m3.augmented_graph(m3.training_graph(d, train), m3.local_positions(train, neighbours), weights)
    np.testing.assert_array_equal(w_star[:25, :25], m3.training_graph(d, train))
    assert (w_star[:25, 25:] == 0).all() and (w_star[25:, 25:] == 0).all()
    assert (w_star[25:].sum(axis=1) == 1.0).all()


def test_the_training_likelihood_is_unchanged_by_the_sink_nodes(cloud):
    dist, _, _ = cloud
    train, test = np.arange(0, 120), np.arange(120, 150)
    w = m3.training_graph(dist, train)
    nb, wt = m3.attachment(dist, train, test)
    w_star = m3.augmented_graph(w, m3.local_positions(train, nb), wt)
    for theta in (-0.7, 0.2, 0.9):
        _, small = np.linalg.slogdet(np.eye(120) - theta * w)
        _, large = np.linalg.slogdet(np.eye(150) - theta * w_star)
        assert large == pytest.approx(small, abs=1e-10)


@pytest.mark.parametrize('size', [1, 5, 8])
def test_fewer_than_nine_training_nodes_fail_structurally(size):
    dist, _, _ = _cloud(20, 9)
    train = np.arange(size)
    assert _reason(lambda: m3.training_graph(dist, train)) == 'too-few-training-nodes'
    assert _reason(lambda: m3.training_neighbours(dist, train)) == 'too-few-training-nodes'
    assert _reason(lambda: m3.attachment(dist, train, np.array([19]))) == 'too-few-training-nodes'
    w = knn_weights(dist[:size, :size], 8)
    y, x = np.arange(size, dtype=float), np.ones((size, 1))
    assert _reason(lambda: m3.fit_spatial('sem', y, x, w)) == 'too-few-training-nodes'


def test_nine_training_nodes_are_enough_and_give_the_complete_graph():
    dist, _, _ = _cloud(20, 9)
    w = m3.training_graph(dist, np.arange(9))
    np.testing.assert_array_equal(w, (np.ones((9, 9)) - np.eye(9)) / 8)


@pytest.mark.parametrize('train, test', [([3, 1, 2, 4, 5, 6, 7, 8, 9, 10], [0]), ([0, 1, 1, 2, 3, 4, 5, 6, 7, 8], [19]),
                                         (list(range(10)), [5]), (list(range(10, 21)), [0]),
                                         ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [19])])
def test_non_canonical_or_overlapping_node_sets_are_refused(train, test):
    dist, _, _ = _cloud(20, 9)
    with pytest.raises(ValueError):
        m3.attachment(dist, np.array(train), np.array(test))


@pytest.mark.parametrize('edit, reason', [('row', 'graph-row-sum'), ('diagonal', 'graph-diagonal')])
def test_a_graph_that_is_not_row_standardised_with_zero_diagonal_fails(cloud, edit, reason):
    _, w, _ = cloud
    bad = w.copy()
    if edit == 'row':
        bad[3] *= 0.9
    else:
        bad[3, 3] = 0.125
    assert _reason(lambda: m3.validate_graph(bad)) == reason
    y = np.random.default_rng(0).normal(size=150)
    assert _reason(lambda: m3.fit_spatial('sar', y, np.ones((150, 1)), bad)) == reason


# --- concentrated likelihood --------------------------------------------------------------------

def _full_loglik(family, y, x, w, beta, sigma2, theta):
    """Unconcentrated Gaussian log-likelihood, determinant through the eigenvalues of W."""
    n = len(y)
    mu = np.linalg.eigvals(w).astype(complex)
    logdet = float(np.sum(np.log(1.0 - theta * mu)).real)
    m = np.eye(n) - theta * w
    e = m @ (y - x @ beta) if family == 'sem' else m @ y - x @ beta
    return -n / 2 * math.log(2 * math.pi) - n / 2 * math.log(sigma2) + logdet - float(e @ e) / (2 * sigma2)


@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_the_concentrated_loglik_is_the_full_loglik_at_the_profile_estimates(cloud, fits, family):
    _, w, x = cloud
    y, _ = fits[family]
    n = len(y)
    for theta in (-0.8, -0.1, 0.35, 0.95):
        ev = m3.concentrated_loglik(family, theta, y, x, w)
        m = np.eye(n) - theta * w
        xt = m @ x if family == 'sem' else x
        gls = np.linalg.solve(xt.T @ xt, xt.T @ (m @ y))
        np.testing.assert_allclose(ev.beta, gls, rtol=0, atol=1e-10)
        e = m @ y - xt @ gls
        assert ev.sigma2 == pytest.approx(float(e @ e) / n, rel=1e-12)
        full = _full_loglik(family, y, x, w, ev.beta, ev.sigma2, theta)
        assert ev.loglik == pytest.approx(full, abs=1e-8)
        assert ev.loglik >= _full_loglik(family, y, x, w, ev.beta + 0.01, ev.sigma2, theta)
        assert ev.loglik >= _full_loglik(family, y, x, w, ev.beta, ev.sigma2 * 1.05, theta)


def test_a_negative_determinant_sign_fails_the_evaluation():
    w = (np.ones((10, 10)) - np.eye(10)) / 9
    y = np.random.default_rng(0).normal(size=10)
    # det(I - 1.5 W) = (1 - 1.5)(1 + 1.5/9)^9 < 0
    assert _reason(lambda: m3.concentrated_loglik('sem', 1.5, y, np.ones((10, 1)), w)) == 'slogdet-sign'


def test_a_non_finite_logdet_fails_the_evaluation(cloud, fits, monkeypatch):
    _, w, x = cloud
    y, _ = fits['sem']
    monkeypatch.setattr(m3.np.linalg, 'slogdet', lambda m: (1.0, -np.inf))
    assert _reason(lambda: m3.concentrated_loglik('sem', 0.1, y, x, w)) == 'logdet-non-finite'


@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_a_rank_deficient_design_fails_every_evaluation_and_the_fit(cloud, fits, family):
    _, w, x = cloud
    y, _ = fits[family]
    collinear = np.column_stack([x, x[:, 1] - 2.0 * x[:, 2]])
    assert _reason(lambda: m3.concentrated_loglik(family, 0.0, y, collinear, w)) == 'rank-deficient'
    assert _reason(lambda: m3.fit_spatial(family, y, collinear, w)) == 'rank-deficient'


def test_a_zero_innovation_variance_fails_the_evaluation(cloud):
    _, w, x = cloud
    assert _reason(lambda: m3.concentrated_loglik('sar', 0.3, np.zeros(150), x, w)) == 'sigma2-invalid'


def test_unknown_family_and_non_finite_inputs_are_refused(cloud, fits):
    _, w, x = cloud
    y, _ = fits['sem']
    with pytest.raises(ValueError, match='family'):
        m3.fit_spatial('sdm', y, x, w)
    with pytest.raises(ValueError, match='finite'):
        m3.fit_spatial('sem', np.where(np.arange(150) == 4, np.nan, y), x, w)


# --- the maximizer ------------------------------------------------------------------------------

def _replay_golden(objective, k):
    """Independent spelling of the golden-section stage; returns its points in evaluation order."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(GRID[max(k - 1, 0)]), float(GRID[min(k + 1, 198)])
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    points = [c, d]
    for _ in range(60):
        if objective(c) >= objective(d):
            b, d = d, c
            c = b - phi * (b - a)
            points.append(c)
        else:
            a, c = c, d
            d = a + phi * (b - a)
            points.append(d)
    return points


def test_the_grid_is_199_points_on_the_closed_domain():
    grid = m3.theta_grid()
    np.testing.assert_array_equal(grid, GRID)
    assert grid[0] == -0.99 and grid[-1] == 0.99 and m3.PHI == (math.sqrt(5.0) - 1.0) / 2.0


@pytest.mark.parametrize('peak', [0.3137, -0.87654, 0.0049])
def test_the_maximizer_evaluates_the_grid_then_c_and_d_then_exactly_sixty_golden_points(peak):
    def objective(t):
        return -(t - peak) ** 2
    calls = []
    result = m3.maximize(lambda t: calls.append(t) or objective(t))
    assert result.n_evaluations == len(calls) == 199 + 2 + 60
    assert calls[:199] == [float(g) for g in GRID]
    k = int(np.argmax([objective(float(g)) for g in GRID]))
    assert calls[199:] == _replay_golden(objective, k)
    assert abs(result.theta - peak) < 1e-7
    assert result.value == max(objective(t) for t in calls)
    np.testing.assert_array_equal(result.grid_values, [objective(float(g)) for g in GRID])


def test_evaluation_ties_go_to_the_earliest_evaluated_point():
    def plateau(t):
        return -max(abs(t - 0.2) - 0.05, 0.0)
    result = m3.maximize(plateau)
    values = [plateau(float(g)) for g in GRID]
    k = values.index(max(values))
    assert result.theta == float(GRID[k]) and result.value == 0.0


def test_a_golden_tie_between_c_and_d_keeps_the_left_subinterval():
    def left_shelf(t):
        return 0.0 if t <= -0.5 else -(t + 0.5)
    calls = []
    result = m3.maximize(lambda t: calls.append(t) or left_shelf(t))
    assert result.at_domain_bound and result.theta == -0.99
    golden = calls[199:]
    assert golden == _replay_golden(left_shelf, 0)
    # every tie takes (b, d) <- (d, c): the search walks monotonically towards the left end
    assert all(later < earlier for earlier, later in zip(golden[2:], golden[3:]))


def test_grid_ties_bracket_the_first_maximizing_index():
    left, right = float(GRID[49]), float(GRID[149])

    def bimodal(t):
        return -min(abs(t - left), abs(t - right))
    calls = []
    result = m3.maximize(lambda t: calls.append(t) or bimodal(t))
    assert result.theta == left
    assert all(float(GRID[48]) <= t <= float(GRID[50]) for t in calls[199:])
    assert m3.grid_local_maxima(result.grid_values) == 2


@pytest.mark.parametrize('values, count', [([1, 2, 1, 3, 3, 1], 1), ([3, 1, 2], 2), ([1, 2, 3], 1),
                                           ([2, 2, 2], 0), ([5.0], 1), ([0, 1, 0, 1, 0, 1, 0], 3)])
def test_strict_local_maxima_compare_endpoints_one_sided(values, count):
    assert m3.grid_local_maxima(values) == count


@pytest.mark.parametrize('peak', [0.995, 1.3, -0.9999, -2.0, 0.9899995])
def test_a_maximum_outside_the_interior_margin_is_a_flagged_constrained_estimate(peak):
    result = m3.maximize(lambda t: -(t - peak) ** 2)
    assert result.at_domain_bound and abs(result.theta) <= 0.99


@pytest.mark.parametrize('peak', [0.989998, -0.989998])
def test_a_maximum_just_inside_the_interior_margin_is_not_flagged(peak):
    result = m3.maximize(lambda t: -(t - peak) ** 2)
    assert abs(result.theta - peak) < 1e-7 and not result.at_domain_bound


@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_a_smooth_field_gives_a_flagged_boundary_estimate_without_an_asymptotic_interval(family):
    dist, _, lat = _cloud(150, 0)
    w = m3.training_graph(dist, np.arange(150))
    fit = m3.fit_spatial(family, lat, np.ones((150, 1)), w)
    assert fit.at_domain_bound and abs(fit.theta) <= 0.99
    interval = m3.dependence_interval(fit, np.ones((150, 1)), w)
    assert interval.se is None and 'boundary estimate' in interval.reason


def test_any_failed_or_non_finite_evaluation_fails_the_maximization():
    def failing(t):
        if len(seen) > 199:
            raise m3.SpatialFitFailure('slogdet-sign', 'injected')
        seen.append(t)
        return -t * t
    seen = []
    assert _reason(lambda: m3.maximize(failing)) == 'slogdet-sign'
    assert _reason(lambda: m3.maximize(lambda t: math.nan if t > 0.5 else -t * t)) == 'objective-non-finite'


# --- fits ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('family, theta', [('sem', 0.5), ('sar', -0.3)])
def test_the_estimator_recovers_the_dependence_parameter(cloud, fits, family, theta):
    _, w, x = cloud
    y, fit = fits[family]
    assert abs(fit.theta - theta) < 0.2
    np.testing.assert_allclose(fit.beta, BETA, atol=0.35)
    assert fit.sigma2 == pytest.approx(1.0, rel=0.3)


@pytest.mark.parametrize('family, theta', [('sem', 0.5), ('sar', -0.3)])
def test_over_replications_the_estimator_is_centred_and_its_expected_information_se_matches_the_spread(
        cloud, family, theta):
    _, w, x = cloud
    rng = np.random.default_rng(10)
    thetas, ses = [], []
    for _ in range(60):
        y = _simulate(family, theta, w, x, BETA, rng)
        fit = m3.fit_spatial(family, y, x, w)
        thetas.append(fit.theta)
        ses.append(m3.dependence_interval(fit, x, w).se)
    assert abs(np.mean(thetas) - theta) < 0.07
    assert np.std(thetas, ddof=1) == pytest.approx(np.mean(ses), rel=0.25)


@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_a_fit_records_the_grid_and_the_final_evaluation(cloud, fits, family):
    _, w, x = cloud
    y, fit = fits[family]
    assert (fit.family, fit.n, fit.p, fit.columns, fit.n_evaluations) == (family, 150, 4, COLUMNS, 261)
    assert fit.version == m3.ESTIMATOR_VERSION and fit.grid_loglik.shape == (199,)
    for i in (0, 57, 120, 198):
        assert fit.grid_loglik[i] == m3.concentrated_loglik(family, float(GRID[i]), y, x, w).loglik
    assert fit.loglik >= fit.grid_loglik.max()
    assert fit.n_grid_local_maxima == m3.grid_local_maxima(fit.grid_loglik) == 1
    final = m3.concentrated_loglik(family, fit.theta, y, x, w)
    np.testing.assert_array_equal(fit.beta, final.beta)
    assert (fit.sigma2, fit.loglik) == (final.sigma2, final.loglik)
    # local optimality on a fine grid around theta-hat
    for step in (1e-3, 1e-4):
        for t in (fit.theta - step, fit.theta + step):
            assert m3.concentrated_loglik(family, t, y, x, w).loglik <= fit.loglik


def test_a_fit_is_immutable(fits):
    _, fit = fits['sem']
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.theta = 0.0
    with pytest.raises(ValueError):
        fit.beta[0] = 1.0
    with pytest.raises(ValueError):
        fit.grid_loglik[0] = 1.0


def test_the_fit_is_deterministic(cloud, fits):
    _, w, x = cloud
    y, fit = fits['sar']
    again = m3.fit_spatial('sar', y, x, w, columns=COLUMNS)
    assert again.theta == fit.theta and again.loglik == fit.loglik
    np.testing.assert_array_equal(again.beta, fit.beta)
    np.testing.assert_array_equal(again.grid_loglik, fit.grid_loglik)


# --- fitted quantities --------------------------------------------------------------------------

@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_fitted_quantities_follow_their_definitions(cloud, fits, family):
    _, w, x = cloud
    y, fit = fits[family]
    q = m3.fitted_quantities(fit, y, x, w)
    mean = x @ fit.beta
    m_inv = np.linalg.inv(np.eye(150) - fit.theta * w)
    trend = mean if family == 'sem' else m_inv @ mean
    one_step = mean + fit.theta * (w @ (y - mean)) if family == 'sem' else fit.theta * (w @ y) + mean
    np.testing.assert_allclose(q.trend, trend, rtol=0, atol=1e-10)
    np.testing.assert_allclose(q.structural_residual, y - trend, rtol=0, atol=1e-10)
    np.testing.assert_allclose(q.one_step, one_step, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(q.innovation, y - q.one_step)
    assert abs(np.mean(q.innovation ** 2) - fit.sigma2) <= 1e-10 * fit.sigma2


@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_the_innovation_checks_fail_on_violation(cloud, fits, family):
    _, w, x = cloud
    y, fit = fits[family]
    # at 1e13 scale the two innovation expressions round differently by far more than 1e-10
    assert _reason(lambda: m3.fitted_quantities(fit, y * 1e13, x, w)) == 'innovation-identity'
    shifted = dataclasses.replace(fit, sigma2=fit.sigma2 * (1 + 1e-8))
    assert _reason(lambda: m3.fitted_quantities(shifted, y, x, w)) == 'innovation-variance'


# --- held-out prediction ------------------------------------------------------------------------

def _split(dist, n_test, seed, buffer_km=None):
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(len(dist), n_test, replace=False))
    excluded = np.zeros(len(dist), dtype=bool)
    if buffer_km is not None:
        excluded = (dist[test] <= buffer_km).any(axis=0)
    excluded[test] = False
    held = np.zeros(len(dist), dtype=bool)
    held[test] = True
    train = np.flatnonzero(~held & ~excluded)
    return train, test, np.flatnonzero(excluded)


def _pipeline(family, dist, y, x, train, test):
    """Everything from graph to prediction, handed the full outcome vector."""
    w = m3.training_graph(dist, train)
    fit = m3.fit_spatial(family, y[train], x[train], w, columns=COLUMNS)
    neighbours, weights = m3.attachment(dist, train, test)
    local = m3.local_positions(train, neighbours)
    return fit, m3.predict_held_out(fit, y[train], x[train], x[test], local, weights), local, weights


@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_held_out_predictions_are_the_conditional_expectations_of_the_augmented_model(cloud, fits, family):
    dist, _, x = cloud
    y, _ = fits[family]
    train, test, _ = _split(dist, 30, 11)
    fit, prediction, local, weights = _pipeline(family, dist, y, x, train, test)
    w_star = m3.augmented_graph(m3.training_graph(dist, train), local, weights)
    x_star = np.vstack([x[train], x[test]])
    n_t = len(train)
    b_inv = np.linalg.inv(np.eye(len(x_star)) - fit.theta * w_star)
    cov = fit.sigma2 * b_inv @ b_inv.T
    mu = x_star @ fit.beta if family == 'sem' else b_inv @ (x_star @ fit.beta)
    y_t = y[train]
    conditional = mu[n_t:] + cov[n_t:, :n_t] @ np.linalg.solve(cov[:n_t, :n_t], y_t - mu[:n_t])
    np.testing.assert_allclose(prediction, conditional, rtol=0, atol=1e-9)
    # the same rows of the augmented one-step predictor, whatever the held-out outcomes are
    junk = np.random.default_rng(12).normal(size=len(test)) * 100
    y_star = np.concatenate([y_t, junk])
    if family == 'sem':
        u_star = y_star - x_star @ fit.beta
        one_step = x_star @ fit.beta + fit.theta * (w_star @ u_star)
    else:
        one_step = fit.theta * (w_star @ y_star) + x_star @ fit.beta
    np.testing.assert_allclose(prediction, one_step[n_t:], rtol=0, atol=1e-12)
    if family == 'sem':
        u_t = y_t - x[train] @ fit.beta
        np.testing.assert_allclose(cov[n_t:, :n_t] @ np.linalg.solve(cov[:n_t, :n_t], u_t),
                                   fit.theta * (w_star[n_t:, :n_t] @ u_t), rtol=0, atol=1e-9)


@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_held_out_and_buffer_excluded_outcomes_never_move_a_prediction(cloud, fits, family):
    dist, _, x = cloud
    y, _ = fits[family]
    train, test, excluded = _split(dist, 6, 13, buffer_km=1500.0)
    assert len(excluded) > 0 and len(train) >= 9
    fit, prediction, _, _ = _pipeline(family, dist, y, x, train, test)
    perturbed = y.copy()
    rng = np.random.default_rng(14)
    perturbed[test] += rng.normal(size=len(test)) * 1e3
    perturbed[excluded] = rng.normal(size=len(excluded)) * 1e6
    fit2, prediction2, _, _ = _pipeline(family, dist, perturbed, x, train, test)
    np.testing.assert_array_equal(prediction2, prediction)
    assert (fit2.theta, fit2.loglik, fit2.sigma2) == (fit.theta, fit.loglik, fit.sigma2)
    np.testing.assert_array_equal(fit2.beta, fit.beta)


def test_held_out_predictions_use_the_training_lag_formula_directly(cloud, fits):
    dist, _, x = cloud
    for family in ('sem', 'sar'):
        y, _ = fits[family]
        train, test, _ = _split(dist, 10, 15)
        fit, prediction, local, _ = _pipeline(family, dist, y, x, train, test)
        source = y[train] - x[train] @ fit.beta if family == 'sem' else y[train]
        expected = x[test] @ fit.beta + fit.theta * source[local].mean(axis=1)
        np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-13)


def test_attachments_are_validated(cloud, fits):
    dist, _, x = cloud
    y, _ = fits['sar']
    train, test, _ = _split(dist, 5, 16)
    fit, _, local, weights = _pipeline('sar', dist, y, x, train, test)
    bad = weights.copy()
    bad[0, 0] = 0.2
    assert _reason(lambda: m3.predict_held_out(fit, y[train], x[train], x[test], local, bad)) == 'graph-row-sum'
    with pytest.raises(ValueError):
        m3.predict_held_out(fit, y[train], x[train], x[test], local + len(train), weights)
    with pytest.raises(ValueError):
        m3.predict_held_out(fit, y[train], x[train], x[test], local[:, :7], weights[:, :7])
    with pytest.raises(ValueError):
        m3.local_positions(train, test)


def test_the_unseen_level_adjustment_mirrors_the_v1_design_rule():
    from research.model_v2.cv import V1Design
    names = ['intercept', 'a', 'koppen=B', 'koppen=C', 'region=Y']
    beta = np.array([0.1, 0.7, -0.4, 0.9, 0.35])
    levels = {'koppen': ['A', 'B', 'C'], 'region': ['X', 'Y']}
    design = V1Design(numeric=['a'], categorical=['koppen', 'region'], levels=levels, coef=beta, columns=names)
    test = pd.DataFrame({'a': [0.5, -1.0, 2.0, 0.0], 'koppen': ['B', 'D', 'A', 'E'], 'region': ['Z', 'X', 'Y', 'W']})
    fit = m3.SpatialFit(family='sem', theta=0.0, loglik=0.0, beta=beta, sigma2=1.0, n=10, p=5, columns=names,
                        n_grid_local_maxima=1, grid_loglik=np.zeros(199), n_evaluations=261)
    x_test, _ = design._matrix(test)
    labels = {c: test[c].tolist() for c in levels}
    adjusted = m3.apply_unseen_level_adjustment(x_test @ beta, fit, levels, labels)
    np.testing.assert_array_equal(adjusted, design.predict(test))
    koppen_mean, region_mean = np.mean([0.0, -0.4, 0.9]), np.mean([0.0, 0.35])
    np.testing.assert_allclose(m3.unseen_level_adjustment(fit, levels, labels, 4),
                               [region_mean, koppen_mean, 0.0, koppen_mean + region_mean], rtol=0, atol=1e-15)
    seen = {'koppen': ['A', 'C', 'B', 'A'], 'region': ['X', 'Y', 'Y', 'X']}
    np.testing.assert_array_equal(m3.apply_unseen_level_adjustment(x_test @ beta, fit, levels, seen), x_test @ beta)


def test_the_unseen_level_adjustment_needs_column_names(fits):
    _, fit = fits['sem']
    anonymous = dataclasses.replace(fit, columns=None)
    with pytest.raises(ValueError, match='column names'):
        m3.unseen_level_adjustment(anonymous, {'region': ['X', 'Y']}, {'region': ['Z']}, 1)


# --- expected information -----------------------------------------------------------------------

def _numerical_hessian(f, point, h=1e-4):
    k = len(point)
    hess = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            hess[i, j] = (f(point + ei + ej) - f(point + ei - ej) - f(point - ei + ej) + f(point - ei - ej)) / (4 * h * h)
    return hess


@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_the_expected_information_follows_the_anselin_formulas(cloud, fits, family):
    _, w, x = cloud
    _, fit = fits[family]
    info = m3.expected_information(fit, x, w)
    p, s2 = 4, fit.sigma2
    assert info.shape == (6, 6)
    np.testing.assert_allclose(info, info.T, rtol=1e-14, atol=0)
    w_m = w @ np.linalg.inv(np.eye(150) - fit.theta * w)
    assert info[p, p] == pytest.approx(150 / (2 * s2 ** 2), rel=1e-14)
    assert info[p, p + 1] == pytest.approx(np.trace(w_m) / s2, rel=1e-12)
    assert (info[:p, p] == 0).all()
    frobenius = float(np.sum(w_m ** 2))
    if family == 'sem':
        bx = (np.eye(150) - fit.theta * w) @ x
        np.testing.assert_allclose(info[:p, :p], bx.T @ bx / s2, rtol=1e-12)
        assert (info[:p, p + 1] == 0).all()
        assert info[p + 1, p + 1] == pytest.approx(np.sum(w_m * w_m.T) + frobenius, rel=1e-12)
    else:
        lagged = w_m @ x @ fit.beta
        np.testing.assert_allclose(info[:p, :p], x.T @ x / s2, rtol=1e-12)
        np.testing.assert_allclose(info[:p, p + 1], x.T @ lagged / s2, rtol=1e-12)
        assert info[p + 1, p + 1] == pytest.approx(np.sum(w_m * w_m.T) + frobenius + lagged @ lagged / s2, rel=1e-12)


@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_the_expected_information_se_is_close_to_the_observed_information_se(cloud, fits, family):
    _, w, x = cloud
    y, fit = fits[family]
    interval = m3.dependence_interval(fit, x, w)
    point = np.concatenate([fit.beta, [fit.sigma2, fit.theta]])
    hess = _numerical_hessian(lambda v: _full_loglik(family, y, x, w, v[:4], v[4], v[5]), point)
    observed_se = math.sqrt(np.linalg.inv(-hess)[-1, -1])
    assert interval.se == pytest.approx(observed_se, rel=0.3)
    assert interval.reason is None
    assert interval.lower == fit.theta - 1.959963984540054 * interval.se
    assert interval.upper == fit.theta + 1.959963984540054 * interval.se


@pytest.mark.parametrize('sigma2', [math.inf, math.nan])
def test_a_variance_that_cannot_be_computed_is_recorded_not_raised(cloud, fits, sigma2):
    _, w, x = cloud
    _, fit = fits['sem']
    interval = m3.dependence_interval(dataclasses.replace(fit, sigma2=sigma2), x, w)
    assert (interval.se, interval.lower, interval.upper) == (None, None, None)
    assert interval.reason.startswith('not computable')


# --- accounting ---------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def grouped(cloud):
    _, w, _ = cloud
    x = _design(150, 21, columns=5)
    beta = np.array([0.1, 0.8, -0.6, 0.4, 0.2, -0.3])
    names = ('intercept', 'co2', 'lat', 'elev', 'gdp', 'pop')
    blocks = {'emissions': x[:, [1]], 'geography': x[:, [2, 3]], 'socioeconomic': x[:, [4]], 'population': x[:, [5]]}
    out = {}
    for family, theta, seed in (('sem', 0.6, 22), ('sar', 0.4, 23)):
        y = _simulate(family, theta, w, x, beta, np.random.default_rng(seed))
        out[family] = (y, m3.fit_spatial(family, y, x, w, columns=names))
    return w, x, blocks, out


@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_the_three_accounting_parts_sum_to_one(grouped, family):
    w, x, _, out = grouped
    y, fit = out[family]
    q = m3.fitted_quantities(fit, y, x, w)
    rss_s = m3.static_rss(y, x)
    ols = np.linalg.solve(x.T @ x, x.T @ y)
    assert rss_s == pytest.approx(float(np.sum((y - x @ ols) ** 2)), rel=1e-10)
    acc = m3.accounting(y, rss_s, q.innovation, trend=q.trend)
    tss = float(np.sum((y - y.mean()) ** 2))
    rss_e = float(np.sum(q.innovation ** 2))
    assert acc.a_static + acc.a_dependence + acc.a_innovation == pytest.approx(1.0, abs=1e-12)
    assert acc.a_static == pytest.approx(1 - rss_s / tss, abs=1e-15)
    assert acc.a_dependence == pytest.approx((rss_s - rss_e) / tss, abs=1e-15)
    assert acc.absorbed_fraction_of_static_residual == pytest.approx((rss_s - rss_e) / rss_s, abs=1e-15)
    assert acc.one_step_r2 == pytest.approx(1 - acc.a_innovation, abs=1e-15)
    assert acc.trend_r2 == pytest.approx(1 - float(np.sum((y - q.trend) ** 2)) / tss, abs=1e-15)
    assert acc.a_dependence > 0


def test_the_accounting_identity_and_the_static_share_sum_are_enforced(grouped):
    w, x, blocks, out = grouped
    y, fit = out['sem']
    q = m3.fitted_quantities(fit, y, x, w)
    rss_s = m3.static_rss(y, x)
    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    static = dataclasses.replace(fit, theta=0.0, beta=ols)
    shares = m3.filtered_group_shares(static, y, x, w, blocks).shares
    m3.accounting(y, rss_s, q.innovation, static_shares=shares)
    with pytest.raises(m3.IntegrityFailure, match='LMG'):
        m3.accounting(y, rss_s, q.innovation, static_shares={**shares, 'population': shares['population'] + 1e-6})
    with pytest.raises(m3.IntegrityFailure, match='sum to'):
        m3.accounting(y, math.inf, q.innovation)


def _permutation_lmg(y_star, intercept, columns, groups):
    """LMG as the average over all orderings of sequential R2 gains (independent of Shapley weights)."""
    tss = float(np.sum((y_star - y_star.mean()) ** 2))

    def r2(members):
        design = np.hstack([intercept, *(columns[g] for g in members)])
        coef = np.linalg.lstsq(design, y_star, rcond=None)[0]
        return 1 - float(np.sum((y_star - design @ coef) ** 2)) / tss
    shares = dict.fromkeys(groups, 0.0)
    orders = list(permutations(groups))
    for order in orders:
        for i, g in enumerate(order):
            shares[g] += (r2(order[:i + 1]) - r2(order[:i])) / len(orders)
    return shares, r2(groups)


@pytest.mark.parametrize('family', ['sem', 'sar'])
def test_filtered_group_shares_sum_to_the_filtered_r2_and_match_an_ordering_average(grouped, family):
    w, x, blocks, out = grouped
    y, fit = out[family]
    result = m3.filtered_group_shares(fit, y, x, w, blocks)
    assert result.groups == ('emissions', 'geography', 'socioeconomic', 'population')
    assert result.n_coalitions == 16 and result.full_coalition_max_gap <= 1e-10
    assert sum(result.shares.values()) == pytest.approx(result.r2_full, abs=1e-12)
    assert result.residual_share == 1 - result.r2_full
    m = np.eye(150) - fit.theta * w
    y_star = m @ y
    assert result.tss_filtered == pytest.approx(float(np.sum((y_star - y_star.mean()) ** 2)), rel=1e-14)
    # SEM: the reference uses M 1 instead of the plain intercept (the same column space)
    intercept = (m @ np.ones(150)).reshape(-1, 1) if family == 'sem' else np.ones((150, 1))
    columns = {g: (m @ b if family == 'sem' else b) for g, b in blocks.items()}
    reference, r2_full = _permutation_lmg(y_star, intercept, columns, result.groups)
    for g in result.groups:
        assert result.shares[g] == pytest.approx(reference[g], abs=1e-10)
    assert result.r2_full == pytest.approx(r2_full, abs=1e-12)
    assert result.composition == {g: s / result.r2_full for g, s in result.shares.items()}


def test_sem_filtering_rescales_the_intercept_without_leaving_its_span(grouped):
    w, _, _, out = grouped
    lam = out['sem'][1].theta
    ones = np.ones(150)
    np.testing.assert_allclose((np.eye(150) - lam * w) @ ones, (1 - lam) * ones, rtol=0, atol=1e-15)


def test_the_full_coalition_must_reproduce_the_ml_transformed_fit(grouped):
    w, x, blocks, out = grouped
    for family in ('sem', 'sar'):
        y, fit = out[family]
        m = np.eye(150) - fit.theta * w
        design = np.hstack([np.ones((150, 1)), *((m @ b) if family == 'sem' else b for b in blocks.values())])
        coef = np.linalg.lstsq(design, m @ y, rcond=None)[0]
        ml = (m @ x) @ fit.beta if family == 'sem' else x @ fit.beta
        assert np.max(np.abs(design @ coef - ml)) <= 1e-10
        wrong = dataclasses.replace(fit, beta=fit.beta * 1.01)
        with pytest.raises(m3.IntegrityFailure, match='full coalition'):
            m3.filtered_group_shares(wrong, y, x, w, blocks)
        partial = {g: b for g, b in blocks.items() if g != 'population'}
        with pytest.raises(m3.IntegrityFailure, match='full coalition'):
            m3.filtered_group_shares(fit, y, x, w, partial)


def test_at_theta_zero_the_filtered_shares_are_the_ols_lmg_shares_of_src_decomposition(grouped):
    from src.decomposition import _r2, _shapley_weight
    w, x, blocks, out = grouped
    y, fit = out['sar']
    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    for family in ('sem', 'sar'):
        static = dataclasses.replace(fit, family=family, theta=0.0, beta=ols)
        result = m3.filtered_group_shares(static, y, x, w, blocks)
        groups = list(blocks)
        for g in groups:
            others = [h for h in groups if h != g]
            expected = sum(_shapley_weight(len(s), 4) * (_r2([blocks[h] for h in (*s, g)], y) - _r2([blocks[h] for h in s], y))
                           for size in range(4) for s in combinations(others, size))
            assert result.shares[g] == pytest.approx(expected, abs=1e-12)
    assert all(m3.shapley_weight(s, 4) == _shapley_weight(s, 4) for s in range(4))


@pytest.mark.parametrize('shares, changed', [
    ({'emissions': 0.02, 'geography': 0.30, 'socioeconomic': 0.10, 'population': 0.01}, False),
    ({'emissions': 0.02, 'geography': 0.30, 'socioeconomic': 0.30, 'population': 0.01}, True),
    ({'emissions': 0.02, 'geography': 0.20, 'socioeconomic': 0.25, 'population': 0.01}, True),
    ({'emissions': 0.11, 'geography': 0.30, 'socioeconomic': 0.10, 'population': 0.01}, True),
    ({'emissions': 0.10, 'geography': 0.30, 'socioeconomic': 0.10, 'population': 0.01}, False),
])
def test_material_change_of_the_non_spatial_part(shares, changed):
    assert m3.material_change(shares) is changed


# --- module hygiene -----------------------------------------------------------------------------

def test_the_module_reads_no_file_and_imports_numpy_only():
    source = inspect.getsource(m3)
    for token in ('open(', 'read_csv', 'Path(', 'outputs', 'app/data', 'warming', 'pandas', 'scipy'):
        assert token not in source, token
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {'__future__', 'math', 'collections.abc', 'dataclasses', 'itertools', 'typing', 'numpy',
                        'research.model_v2.spatial'}
