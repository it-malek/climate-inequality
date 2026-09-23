"""M3 independent reference mathematics: graphs, likelihood, prediction, metrics, accounting, information.

Synthetic reference calculations only: every graph, design and outcome below is generated here with
``numpy.random.default_rng`` and fixed seeds. No repository data file is read.
"""
from __future__ import annotations

import ast
import inspect
import math
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import linregress, multivariate_normal

from research.model_v2 import m3_independent_math as im
from research.model_v2.spatial import haversine_matrix

FAMILIES = ('SEM', 'SAR')


def _distances(n, seed):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180.0, 180.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-0.9, 0.9, n)))
    return haversine_matrix(lon, lat)


def _graph(dist, nodes):
    local = im.local_positions(im.knn_neighbours(dist, nodes), nodes)
    return local, im.dense_weights(local, len(nodes))


def _simulate(family, n=80, theta=0.5, seed=0, extra_columns=0):
    rng = np.random.default_rng(seed)
    dist = _distances(n, seed + 100)
    nodes = list(range(n))
    local, w = _graph(dist, nodes)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2 + extra_columns))])
    beta = np.concatenate([[0.2, 0.5, -0.3], rng.normal(scale=0.4, size=extra_columns)])
    eps = rng.normal(scale=0.3, size=n)
    m = np.eye(n) - theta * w
    y = x @ beta + np.linalg.solve(m, eps) if family == 'SEM' else np.linalg.solve(m, x @ beta + eps)
    return SimpleNamespace(dist=dist, local=local, w=w, x=x, y=y, beta=beta, n=n)


def _dense_concentrated(family, y, x, w, theta):
    n = len(y)
    m = np.eye(n) - theta * w
    x_t = m @ x if family == 'SEM' else x
    beta, *_ = np.linalg.lstsq(x_t, m @ y, rcond=None)
    e = m @ y - x_t @ beta
    sigma2 = e @ e / n
    loglik = -(n / 2) * (math.log(2 * math.pi) + 1) - (n / 2) * math.log(sigma2) + np.linalg.slogdet(m)[1]
    return loglik, beta, sigma2


def _dense_optimum(family, data):
    result = minimize_scalar(lambda t: -_dense_concentrated(family, data.y, data.x, data.w, t)[0],
                             bounds=im.DOMAIN, method='bounded', options={'xatol': 1e-11})
    return float(result.x)


# --- graphs -------------------------------------------------------------------------------------

def _reference_first_eight(dist, node, candidates):
    candidates = np.array(sorted(candidates))
    order = np.argsort(dist[node, candidates], kind='stable')  # stable: ties keep canonical order
    return tuple(int(c) for c in candidates[order][:8])


def test_knn_neighbours_order_by_distance_then_canonical_index_on_heavily_tied_distances():
    dist = np.round(_distances(40, 1) / 2500.0) * 2500.0
    subset = [i for i in range(40) if i % 3 != 1]
    for nodes in (list(range(40)), subset):
        neighbours = im.knn_neighbours(dist, nodes)
        assert len(neighbours) == len(nodes)
        for node, row in zip(nodes, neighbours):
            assert row == _reference_first_eight(dist, node, [j for j in nodes if j != node])
            assert node not in row and set(row) <= set(nodes)


def test_all_equal_distances_give_the_first_eight_other_canonical_indices():
    dist = np.ones((10, 10)) - np.eye(10)
    neighbours = im.knn_neighbours(dist, range(10))
    assert neighbours[0] == (1, 2, 3, 4, 5, 6, 7, 8)
    assert neighbours[4] == (0, 1, 2, 3, 5, 6, 7, 8)
    assert neighbours[9] == (0, 1, 2, 3, 4, 5, 6, 7)


def test_held_out_attachments_link_only_to_training_nodes_under_the_same_tie_rule():
    dist = np.round(_distances(40, 2) / 2500.0) * 2500.0
    held_out = [2, 11, 30]
    train = [i for i in range(40) if i not in held_out and i % 7 != 0]
    attachments = im.attachment_neighbours(dist, train, held_out)
    for node, row in zip(held_out, attachments):
        assert row == _reference_first_eight(dist, node, train)
        assert set(row) <= set(train)


def test_weights_are_one_eighth_rows_sum_exactly_to_one_and_the_diagonal_is_zero():
    dist = _distances(30, 3)
    _, w = _graph(dist, list(range(30)))
    assert set(np.unique(w)) == {0.0, 0.125}
    assert (w.sum(axis=1) == 1.0).all() and (np.diag(w) == 0.0).all()


def test_fewer_than_nine_training_nodes_is_a_structural_failure_and_bad_node_sets_raise():
    dist = _distances(20, 4)
    with pytest.raises(im.StructuralFailure):
        im.knn_neighbours(dist, range(8))
    with pytest.raises(im.StructuralFailure):
        im.attachment_neighbours(dist, range(8), [15])
    assert len(im.knn_neighbours(dist, range(9))) == 9
    with pytest.raises(ValueError, match='strictly increasing'):
        im.knn_neighbours(dist, [0, 2, 1, 3, 4, 5, 6, 7, 8, 9])
    with pytest.raises(ValueError, match='training node'):
        im.attachment_neighbours(dist, range(12), [5])


# --- likelihood ---------------------------------------------------------------------------------

@pytest.mark.parametrize('seed', [0, 1, 2])
def test_eigen_log_determinant_equals_slogdet_on_random_row_standardised_knn_graphs(seed):
    n = 60 + 30 * seed
    dist = _distances(n, 10 + seed)
    _, w = _graph(dist, list(range(n)))
    eigenvalues = im.weight_eigenvalues(w)
    signs = []
    for theta in (-0.99, -0.6, -0.01, 0.0, 0.3, 0.77, 0.99, 1.01, 1.4, -2.5, 3.0):
        sign, logdet = np.linalg.slogdet(np.eye(n) - theta * w)
        got_sign, got_logdet = im.eigen_logdet(eigenvalues, theta)
        assert got_sign == sign
        assert got_logdet == pytest.approx(logdet, abs=1e-9)
        signs.append(got_sign)
    assert seed == 0 or -1.0 in signs  # a negative determinant outside the domain is exercised


@pytest.mark.parametrize('family', FAMILIES)
def test_qr_concentrated_loglik_equals_the_dense_gaussian_loglik_at_the_profiled_beta_and_sigma2(family):
    data = _simulate(family, seed=5)
    n = data.n
    for theta in (-0.7, 0.1, 0.55, 0.95):
        result = im.profile(family, data.y, data.x, data.w, theta)
        assert result.valid and result.rank == 3
        m = np.eye(n) - theta * data.w
        _, beta, sigma2 = _dense_concentrated(family, data.y, data.x, data.w, theta)
        np.testing.assert_allclose(result.beta, beta, rtol=0, atol=1e-10)
        assert result.sigma2 == pytest.approx(sigma2, rel=1e-12)
        e = m @ (data.y - data.x @ beta) if family == 'SEM' else m @ data.y - data.x @ beta
        full = -(n / 2) * math.log(2 * math.pi * sigma2) + np.linalg.slogdet(m)[1] - e @ e / (2 * sigma2)
        assert result.loglik == pytest.approx(full, abs=1e-9)
        precision_root = np.linalg.inv(m)
        cov = sigma2 * precision_root @ precision_root.T
        mean = data.x @ beta if family == 'SEM' else precision_root @ data.x @ beta
        assert result.loglik == pytest.approx(multivariate_normal(mean, cov).logpdf(data.y), abs=1e-7)


def test_evaluation_failures_are_named_in_a_fixed_order():
    data = _simulate('SEM', n=40, seed=6)
    duplicated = np.column_stack([data.x, data.x[:, 1]])
    assert im.profile('SEM', data.y, duplicated, data.w, 0.2).failure == 'rank 3 != p 4'
    assert im.profile('SAR', data.y, duplicated, data.w, 0.2).failure == 'rank 3 != p 4'
    outside = im.profile('SEM', data.y, data.x, data.w, 1.01)
    assert np.linalg.slogdet(np.eye(40) - 1.01 * data.w)[0] == -1.0
    assert outside.sign == -1.0 and outside.failure == 'determinant sign -1' and math.isnan(outside.loglik)
    with pytest.raises(im.EvaluationFailure, match='rank'):
        im.beta_given_theta('SAR', data.y, duplicated, data.w, 0.2)
    with pytest.raises(ValueError, match='family'):
        im.profile('SDM', data.y, data.x, data.w, 0.2)


def test_the_search_is_the_199_point_grid_then_sixty_golden_iterations():
    target = 0.3137

    def objective(t):
        return -(t - target) ** 2

    result = im.maximize(objective)
    grid = np.linspace(-0.99, 0.99, 199)
    assert len(result.grid) == 199 and len(result.evaluations) == 199 + 2 + 60
    assert [t for t, _ in result.evaluations[:199]] == [float(g) for g in grid]
    k = int(np.argmax([objective(float(g)) for g in grid]))
    a, b = float(grid[k - 1]), float(grid[k + 1])
    assert result.evaluations[199][0] == b - im.PHI * (b - a)
    assert result.evaluations[200][0] == a + im.PHI * (b - a)
    assert all(a <= t <= b for t, _ in result.evaluations[199:])
    assert result.theta == pytest.approx(target, abs=1e-9) and result.failure is None
    assert result.strict_local_maxima == 1 and im.PHI == (math.sqrt(5.0) - 1.0) / 2.0


def test_ties_go_to_the_earliest_evaluation_and_boundary_estimates_are_flagged_not_failed():
    flat = im.maximize(lambda t: 1.0)
    assert flat.theta == -0.99 and flat.failure is None and flat.at_domain_bound
    # l(c) >= l(d) on equality keeps [a, d]: the bracket collapses onto the lower grid point.
    assert flat.evaluations[-1][0] < -0.9899 and flat.strict_local_maxima == 0
    rising = im.maximize(lambda t: t)
    assert rising.theta == pytest.approx(0.99, abs=1e-12) and rising.failure is None and rising.at_domain_bound
    near_edge = im.maximize(lambda t: -(t - (0.99 - 5e-7)) ** 2)
    assert near_edge.at_domain_bound
    inside = im.maximize(lambda t: -(t - (0.99 - 5e-6)) ** 2)
    assert inside.failure is None and not inside.at_domain_bound


def test_strict_local_maxima_compare_endpoints_one_sided():
    assert im.maximize(lambda t: t * t).strict_local_maxima == 2
    assert im.strict_local_maxima([3.0, 1.0, 2.0]) == 2
    assert im.strict_local_maxima([1.0, 1.0, 1.0]) == 0
    assert im.strict_local_maxima([0.0, 2.0, 2.0, 0.0, 1.0, 0.0]) == 1


def test_any_failed_evaluation_fails_the_search():
    def objective(t):
        if t > 0.5:
            raise im.EvaluationFailure('synthetic')
        return -t * t

    assert im.maximize(objective).failure.startswith('evaluation failed')
    assert im.maximize(lambda t: math.nan if t > 0.9 else 0.0).failure.startswith('evaluation failed')


@pytest.mark.parametrize('family', FAMILIES)
def test_the_reference_fit_matches_a_bounded_scipy_optimum_of_the_dense_likelihood(family):
    data = _simulate(family, seed=7)
    fit = im.reference_fit(family, data.y, data.x, data.w)
    optimum = _dense_optimum(family, data)
    assert fit.failure is None and fit.n == 80 and fit.p == 3 and len(fit.grid) == 199
    assert fit.theta == pytest.approx(optimum, abs=1e-6)
    loglik, beta, sigma2 = _dense_concentrated(family, data.y, data.x, data.w, fit.theta)
    np.testing.assert_allclose(fit.beta, beta, rtol=0, atol=1e-10)
    assert fit.sigma2 == pytest.approx(sigma2, rel=1e-10) and fit.loglik == pytest.approx(loglik, abs=1e-9)
    np.testing.assert_allclose(im.beta_given_theta(family, data.y, data.x, data.w, fit.theta), fit.beta,
                               rtol=0, atol=0)


@pytest.mark.parametrize('family', FAMILIES)
def test_local_optimality_passes_at_the_optimum_and_detects_a_shifted_theta(family):
    data = _simulate(family, seed=8)
    optimum = _dense_optimum(family, data)
    check = im.local_optimality_check(family, data.y, data.x, data.w, optimum)
    assert check['passes'] and check['shortfall'] <= 1e-8 and check['points'] == 41
    assert check['grid_max'] - check['grid_min'] == pytest.approx(2e-3, abs=1e-12)
    for shift in (5e-4, -5e-4):
        negative = im.local_optimality_check(family, data.y, data.x, data.w, optimum + shift)
        assert not negative['passes'] and negative['shortfall'] > 1e-6
        assert abs(negative['best_grid_theta'] - optimum) < abs(shift)


def test_the_local_grid_is_clipped_to_the_domain():
    data = _simulate('SEM', n=40, seed=9)
    check = im.local_optimality_check('SEM', data.y, data.x, data.w, 0.9895)
    assert check['grid_max'] == 0.99 and not check['failed_points']


@pytest.mark.parametrize('family', FAMILIES)
def test_the_global_scan_places_the_estimate_in_the_global_basin_and_rejects_a_distant_theta(family):
    data = _simulate(family, seed=10)
    fit = im.reference_fit(family, data.y, data.x, data.w)
    scan = im.global_scan(family, data.y, data.x, data.w, fit.theta)
    assert scan['points'] == 397 and not scan['failed_points'] and scan['in_global_basin']
    assert scan['basin'][0] <= fit.theta <= scan['basin'][1]
    assert scan['loglik_at_estimate'] >= scan['loglik_max']
    assert not im.global_scan(family, data.y, data.x, data.w, fit.theta - 0.3)['in_global_basin']


# --- fitted quantities and held-out prediction --------------------------------------------------

def test_sem_one_step_innovation_equals_m_times_the_structural_residual():
    data = _simulate('SEM', seed=11)
    fit = im.reference_fit('SEM', data.y, data.x, data.w)
    q = im.in_sample_quantities('SEM', data.y, data.x, data.local, fit.theta, fit.beta, fit.sigma2)
    m = np.eye(data.n) - fit.theta * data.w
    np.testing.assert_allclose(q['trend'], data.x @ fit.beta, rtol=0, atol=1e-14)
    np.testing.assert_allclose(q['innovation'], m @ (data.y - data.x @ fit.beta), rtol=0, atol=1e-12)
    lagged = data.w @ q['structural_residual']
    np.testing.assert_allclose(q['one_step'], data.x @ fit.beta + fit.theta * lagged,
                               rtol=0, atol=1e-12)
    assert q['innovation_expression_gap'] < 1e-10 and q['sigma2_relative_gap'] < 1e-10


def test_sar_trend_solves_the_lag_system_and_the_innovation_is_m_y_minus_x_beta():
    data = _simulate('SAR', seed=12)
    fit = im.reference_fit('SAR', data.y, data.x, data.w)
    q = im.in_sample_quantities('SAR', data.y, data.x, data.local, fit.theta, fit.beta, fit.sigma2)
    m = np.eye(data.n) - fit.theta * data.w
    np.testing.assert_allclose(m @ q['trend'], data.x @ fit.beta, rtol=0, atol=1e-12)
    np.testing.assert_allclose(q['structural_residual'], data.y - q['trend'], rtol=0, atol=0)
    np.testing.assert_allclose(q['innovation'], m @ data.y - data.x @ fit.beta, rtol=0, atol=1e-12)
    assert q['innovation_expression_gap'] < 1e-10 and q['sigma2_relative_gap'] < 1e-10


def _split(n_total=45, seed=13, held_out=(4, 19, 33), excluded=(5, 20)):
    dist = _distances(n_total, seed)
    train = [i for i in range(n_total) if i not in held_out and i not in excluded]
    local = im.local_positions(im.knn_neighbours(dist, train), train)
    attach = im.local_positions(im.attachment_neighbours(dist, train, list(held_out)), train)
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n_total), rng.normal(size=(n_total, 2))])
    return SimpleNamespace(train=train, held_out=list(held_out), excluded=list(excluded), local=local,
                           attach=attach, x=x, y=rng.normal(size=n_total), beta=np.array([0.1, -0.4, 0.7]))


@pytest.mark.parametrize('family', FAMILIES)
def test_held_out_loops_equal_the_dense_matrix_computation(family):
    s = _split()
    w_ot = im.dense_weights(s.attach, len(s.train))
    x_t, y_t, x_o = s.x[s.train], s.y[s.train], s.x[s.held_out]
    theta = 0.37
    predictions = im.held_out_predictions(family, theta, s.beta, x_t, y_t, x_o, s.attach)
    source = y_t - x_t @ s.beta if family == 'SEM' else y_t
    np.testing.assert_allclose(predictions, x_o @ s.beta + theta * w_ot @ source, rtol=0, atol=1e-14)
    np.testing.assert_allclose(im.lag(source, s.attach), w_ot @ source, rtol=0, atol=1e-15)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('theta', [0.45, -0.6])
def test_held_out_predictions_are_the_conditional_expectations_of_the_sink_augmented_model(family, theta):
    """Joint Gaussian over (T, O) under W* = [[W_TT, 0], [W_OT, 0]]; E[y_O | y_T] from its covariance."""
    s = _split(seed=14)
    n_t, n_o = len(s.train), len(s.held_out)
    w_star = np.zeros((n_t + n_o, n_t + n_o))
    w_star[:n_t, :n_t] = im.dense_weights(s.local, n_t)
    w_star[n_t:, :n_t] = im.dense_weights(s.attach, n_t)
    x_star = np.vstack([s.x[s.train], s.x[s.held_out]])
    sigma2 = 0.2
    a_inv = np.linalg.inv(np.eye(n_t + n_o) - theta * w_star)
    mean = x_star @ s.beta if family == 'SEM' else a_inv @ x_star @ s.beta
    cov = sigma2 * a_inv @ a_inv.T
    y_t = s.y[s.train]
    conditional = mean[n_t:] + cov[n_t:, :n_t] @ np.linalg.solve(cov[:n_t, :n_t], y_t - mean[:n_t])
    predictions = im.held_out_predictions(family, theta, s.beta, s.x[s.train], y_t, s.x[s.held_out], s.attach)
    np.testing.assert_allclose(predictions, conditional, rtol=0, atol=1e-9)


@pytest.mark.parametrize('family', FAMILIES)
def test_held_out_and_excluded_outcomes_never_move_a_prediction(family):
    s = _split(seed=15)
    perturbed = s.y.copy()
    perturbed[s.held_out + s.excluded] += np.random.default_rng(0).normal(scale=50.0, size=5)

    def predict(y):
        return im.held_out_predictions(family, 0.4, s.beta, s.x[s.train], y[s.train], s.x[s.held_out],
                                       s.attach)

    assert np.array_equal(predict(s.y), predict(perturbed))


def test_the_unseen_level_adjustment_is_the_v1_mean_effect_rule():
    names = ['intercept', 'x', 'kind=q', 'region=b', 'region=c']
    beta = np.array([1.0, 2.0, 0.3, -0.6, 0.9])
    training = {'region': ['c', 'a', 'b', 'a'], 'kind': ['q', 'p', 'p']}
    test = {'region': ['a', 'd', 'c', 'z'], 'kind': ['p', 'p', 'r', 's']}
    region_mean = (0.0 - 0.6 + 0.9) / 3
    kind_mean = (0.0 + 0.3) / 2
    adjustment = im.unseen_level_adjustment(beta, names, training, test, 4)
    np.testing.assert_allclose(adjustment, [0.0, region_mean, kind_mean, region_mean + kind_mean],
                               rtol=0, atol=1e-15)
    s = _split(seed=16)
    base = im.held_out_predictions('SAR', 0.3, s.beta, s.x[s.train], s.y[s.train], s.x[s.held_out], s.attach)
    a = np.array([0.0, 0.25, -0.5])
    shifted = im.held_out_predictions('SAR', 0.3, s.beta, s.x[s.train], s.y[s.train], s.x[s.held_out],
                                      s.attach, a)
    np.testing.assert_allclose(shifted - base, a, rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match='drop-first'):
        im.unseen_level_adjustment(beta, names[:-1], training, test, 4)


# --- metrics ------------------------------------------------------------------------------------

def test_point_metrics_and_calibration_match_direct_formulas():
    rng = np.random.default_rng(17)
    prediction = rng.normal(0.2, 0.1, 151)
    y = 0.05 + 0.8 * prediction + rng.normal(scale=0.05, size=151)
    e = y - prediction
    assert im.rmse(y, prediction) == pytest.approx(np.sqrt(np.mean(e ** 2)), rel=1e-14)
    assert im.mae(y, prediction) == pytest.approx(np.mean(np.abs(e)), rel=1e-14)
    assert im.r2(y, prediction) == pytest.approx(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), rel=1e-13)
    calibration = im.calibration(y, prediction)
    reference = linregress(prediction, y)
    assert calibration['slope'] == pytest.approx(reference.slope, rel=1e-12)
    assert calibration['intercept'] == pytest.approx(reference.intercept, abs=1e-13)


def test_the_paired_rmse_interval_is_reproducible_ordered_and_matches_a_per_draw_loop():
    rng = np.random.default_rng(18)
    n = 60
    y = rng.normal(size=n)
    reference = y + rng.normal(scale=0.5, size=n)
    new = y + rng.normal(scale=0.45, size=n)
    first, second = im.paired_rmse_interval(y, new, reference), im.paired_rmse_interval(y, new, reference)
    assert first == second and first['resamples'] == 2000 and first['seed'] == 0
    lower, upper = first['interval']
    assert lower <= upper
    idx = np.random.default_rng(0).integers(0, n, size=(2000, n))
    deltas = [math.sqrt(sum((y[i] - new[i]) ** 2 for i in row) / n)
              - math.sqrt(sum((y[i] - reference[i]) ** 2 for i in row) / n) for row in idx]
    np.testing.assert_allclose(first['interval'], np.quantile(deltas, [0.025, 0.975]), rtol=0, atol=1e-12)
    assert first['delta_rmse'] == pytest.approx(im.rmse(y, new) - im.rmse(y, reference), abs=0)
    assert im.paired_rmse_interval(y, reference, reference)['interval'] == [0.0, 0.0]
    better = im.paired_rmse_interval(y, y + 0.5 * (reference - y), reference)
    assert better['delta_rmse'] < 0 and better['interval'][1] < 0


def test_moran_statistic_is_n_over_s0_times_zwz_over_zz():
    dist = _distances(50, 19)
    _, w = _graph(dist, list(range(50)))
    values = np.random.default_rng(19).normal(size=50)
    z = values - values.mean()
    loop = sum(z[i] * w[i, j] * z[j] for i in range(50) for j in range(50))
    assert im.moran_statistic(values, w) == pytest.approx(loop / np.sum(z * z), abs=1e-14)
    symmetric = np.exp(-dist / 3000.0)
    np.fill_diagonal(symmetric, 0.0)
    expected = 50 / symmetric.sum() * (z @ symmetric @ z) / (z @ z)
    assert im.moran_statistic(values, symmetric) == pytest.approx(expected, rel=1e-12)


# --- accounting ---------------------------------------------------------------------------------

def _lstsq_r2(z, target):
    coef, *_ = np.linalg.lstsq(z, target, rcond=None)
    return 1 - np.sum((target - z @ coef) ** 2) / np.sum((target - target.mean()) ** 2)


def _subset_weight_shapley(players, value):
    g = len(players)
    shares = {}
    for player in players:
        others = [p for p in players if p != player]
        shares[player] = sum(math.factorial(size) * math.factorial(g - size - 1) / math.factorial(g)
                             * (value(frozenset(combo) | {player}) - value(frozenset(combo)))
                             for size in range(g) for combo in combinations(others, size))
    return shares


def test_permutation_shapley_equals_subset_weight_shapley_on_a_random_four_group_problem():
    rng = np.random.default_rng(20)
    n = 120
    blocks = {g: rng.normal(size=(n, k)) for g, k in zip('abcd', (1, 3, 2, 2))}
    blocks['c'][:, 0] += 0.8 * blocks['a'][:, 0]
    y = sum(block @ rng.normal(size=block.shape[1]) for block in blocks.values()) + rng.normal(size=n)

    def value(coalition):
        return _lstsq_r2(np.hstack([np.ones((n, 1))] + [blocks[g] for g in 'abcd' if g in coalition]), y)

    permutation = im.permutation_shapley(list('abcd'), value)
    subset = _subset_weight_shapley(list('abcd'), value)
    for g in 'abcd':
        assert permutation[g] == pytest.approx(subset[g], abs=1e-13)
    assert math.fsum(permutation.values()) == pytest.approx(value(frozenset('abcd')) - value(frozenset()),
                                                            abs=1e-13)


def test_projection_r2_trims_rank_deficient_columns_to_the_lstsq_projection():
    rng = np.random.default_rng(21)
    n = 50
    base = rng.normal(size=(n, 2))
    z = np.column_stack([np.ones(n), base[:, 0], base[:, 0], base[:, 1], base[:, 0] - 2.0 * base[:, 1]])
    y = rng.normal(size=n)
    value, fitted = im.projection_r2(z, y)
    _, keep = im.column_space_basis(z)
    assert keep == [0, 1, 3]
    assert value == pytest.approx(_lstsq_r2(z, y), abs=1e-13)
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
    np.testing.assert_allclose(fitted, z @ coef, rtol=0, atol=1e-13)


def test_accounting_parts_close_the_identity_and_the_static_part_is_s_ols_r2():
    data = _simulate('SEM', seed=22, extra_columns=2)
    fit = im.reference_fit('SEM', data.y, data.x, data.w)
    q = im.in_sample_quantities('SEM', data.y, data.x, data.local, fit.theta, fit.beta)
    groups = {'emissions': data.x[:, [1]], 'geography': data.x[:, [2, 3]], 'population': data.x[:, [4]]}

    def value(coalition):
        return _lstsq_r2(np.hstack([np.ones((data.n, 1))] + [groups[g] for g in groups if g in coalition]),
                         data.y)

    shares = _subset_weight_shapley(list(groups), value)
    parts = im.accounting_parts(data.y, data.x, q['innovation'], shares)
    tss = np.sum((data.y - data.y.mean()) ** 2)
    rss_static = np.sum((data.y - data.x @ np.linalg.lstsq(data.x, data.y, rcond=None)[0]) ** 2)
    rss_innovation = np.sum(q['innovation'] ** 2)
    assert parts['static_component'] == pytest.approx(1 - rss_static / tss, abs=1e-13)
    assert parts['represented_spatial_dependence'] == pytest.approx((rss_static - rss_innovation) / tss,
                                                                    abs=1e-13)
    assert parts['remaining_innovation'] == pytest.approx(rss_innovation / tss, abs=1e-13)
    assert parts['absorbed_fraction_of_static_residual'] == pytest.approx(
        (rss_static - rss_innovation) / rss_static, abs=1e-13)
    assert abs(parts['identity_gap']) < 1e-12 and abs(parts['static_shares_sum_gap']) < 1e-12


@pytest.mark.parametrize('family', FAMILIES)
def test_the_full_filtered_coalition_reproduces_the_ml_transformed_fit(family):
    data = _simulate(family, seed=23, extra_columns=3)
    fit = im.reference_fit(family, data.y, data.x, data.w)
    groups = {'emissions': data.x[:, 1], 'geography': data.x[:, [2, 3]], 'socioeconomic': data.x[:, 4],
              'population': data.x[:, [5]]}
    result = im.filtered_decomposition(family, data.y, groups, data.w, fit.theta)
    m = np.eye(data.n) - fit.theta * data.w
    transformed_fit = (m @ data.x if family == 'SEM' else data.x) @ fit.beta
    np.testing.assert_allclose(result['full_fitted'], transformed_fit, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result['filtered_outcome'], m @ data.y, rtol=0, atol=0)
    np.testing.assert_allclose(m @ np.ones(data.n), (1.0 - fit.theta) * np.ones(data.n), rtol=0, atol=1e-15)
    assert abs(result['shares_sum_gap']) < 1e-12 and result['residual'] == 1.0 - result['r2_full']
    assert set(result['shares']) == set(groups) and 'material_change' in result


def test_material_change_flags_follow_the_v1_definition():
    rng = np.random.default_rng(24)
    dist = _distances(90, 24)
    _, w = _graph(dist, list(range(90)))
    geo, emissions, other = rng.normal(size=(90, 2)), rng.normal(size=90), rng.normal(size=90)
    groups = {'emissions': emissions, 'geography': geo, 'socioeconomic': other}
    noise = rng.normal(scale=0.3, size=90)
    stable = im.filtered_decomposition('SAR', 2.0 * geo[:, 0] + 0.05 * emissions + noise, groups, w, 0.2)
    assert stable['material_change'] == {'geography_not_strictly_largest': False,
                                         'emissions_exceeds_0_10': False, 'material_change': False}
    changed = im.filtered_decomposition('SAR', 0.1 * geo[:, 0] + 2.0 * emissions + noise, groups, w, 0.2)
    assert changed['material_change'] == {'geography_not_strictly_largest': True,
                                          'emissions_exceeds_0_10': True, 'material_change': True}


# --- expected information -----------------------------------------------------------------------

def _gaussian_fisher(family, x, w, theta, beta, sigma2, h=1e-5):
    """Fisher information of y ~ N(mu(q), Sigma(q)) from central differences of mu and Sigma."""
    n, p = x.shape

    def moments(q):
        a_inv = np.linalg.inv(np.eye(n) - q[p + 1] * w)
        mean = x @ q[:p] if family == 'SEM' else a_inv @ x @ q[:p]
        return mean, q[p] * a_inv @ a_inv.T

    q0 = np.concatenate([beta, [sigma2, theta]])
    precision = np.linalg.inv(moments(q0)[1])
    d_mean, d_cov = [], []
    for i in range(p + 2):
        step = h * max(1.0, abs(q0[i]))
        up, down = q0.copy(), q0.copy()
        up[i] += step
        down[i] -= step
        (mean_up, cov_up), (mean_down, cov_down) = moments(up), moments(down)
        d_mean.append((mean_up - mean_down) / (2 * step))
        d_cov.append((cov_up - cov_down) / (2 * step))
    return np.array([[d_mean[i] @ precision @ d_mean[j]
                      + 0.5 * np.trace(precision @ d_cov[i] @ precision @ d_cov[j]) for j in range(p + 2)]
                     for i in range(p + 2)])


@pytest.mark.parametrize('family', FAMILIES)
def test_expected_information_equals_the_gaussian_fisher_information(family):
    data = _simulate(family, n=40, seed=25)
    theta, beta, sigma2 = 0.42, np.array([0.3, -0.8, 0.5]), 0.15
    analytic = im.expected_information(family, data.x, data.w, theta, beta, sigma2)
    numeric = _gaussian_fisher(family, data.x, data.w, theta, beta, sigma2)
    assert analytic.shape == (5, 5)
    np.testing.assert_allclose(analytic, analytic.T, rtol=0, atol=1e-10)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-5)
    if family == 'SEM':
        assert (analytic[:3, 3:] == 0).all()


@pytest.mark.parametrize('family', FAMILIES)
def test_theta_se_is_the_inverse_information_element_with_the_normal_wald_quantile(family):
    data = _simulate(family, seed=26)
    fit = im.reference_fit(family, data.y, data.x, data.w)
    result = im.theta_standard_error(family, data.x, data.w, fit.theta, fit.beta, fit.sigma2)
    info = im.expected_information(family, data.x, data.w, fit.theta, fit.beta, fit.sigma2)
    se = math.sqrt(np.linalg.inv(info)[-1, -1])
    assert result['se'] == pytest.approx(se, rel=1e-10)
    assert result['wald_95'] == pytest.approx([fit.theta - 1.959963984540054 * se,
                                               fit.theta + 1.959963984540054 * se], rel=1e-10)


def test_a_non_finite_information_matrix_is_not_computable():
    data = _simulate('SAR', n=30, seed=27)
    result = im.theta_standard_error('SAR', data.x, data.w, 0.3, data.beta, math.nan)
    assert result['se'] == 'not computable' and result['wald_95'] == 'not computable'


# --- independence -------------------------------------------------------------------------------

def test_the_module_is_a_separate_numerical_path():
    tree = ast.parse(inspect.getsource(im))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {'__future__', 'math', 'collections.abc', 'dataclasses', 'itertools', 'numpy'}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {'lstsq', 'slogdet', 'det', 'inv', 'pinv', 'polyfit', 'knn_weights'}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert 'open' not in names
