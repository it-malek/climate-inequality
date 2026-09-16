"""M3 independent reference mathematics: a second numerical route to every saved spatial quantity.

A verification path for ``DOWNSTREAM_COMPLETION_SPEC.md`` §2.2-§2.7 and §2.11, written without reading or
importing the M3 estimator module. Given saved theta, beta, neighbour lists and designs, it rebuilds the
graphs, likelihood, optimum, fitted quantities, held-out predictions, metrics, intervals and accounting by
different calculations so the two routes can be compared. NumPy and the standard library only; it reads
no file and never sees an outcome other than the arrays it is handed.

Graphs (§2.2). For a node set N in canonical order, node i's kNN8 neighbours are the first 8 of N \\ {i}
under Python ``sorted`` of the tuples (D[i, j], j); a held-out node o takes the first 8 of the training set
T under (D[o, j], j). Each link weighs 1/8, so the augmented matrix is W* = [[W_TT, 0], [W_OT, 0]] (sink
rule) and every |T| < 9 is a structural failure.

Likelihood (§2.5). With M(theta) = I - theta W and w_k the complex eigenvalues of W,

    ln|det M(theta)| = sum_k ln|1 - theta w_k|,     sign det M(theta) = sign cos(sum_k arg(1 - theta w_k)),

(conjugate pairs cancel in the phase; a negative real factor contributes pi). The transformed regression
(SEM: X~ = M X, y~ = M y; SAR: X~ = X, y~ = M y) is solved by a reduced QR, beta = R^-1 Q' y~,
e = y~ - X~ beta, sigma2 = e'e / n and

    l_c(theta) = -(n/2)(ln 2 pi + 1) - (n/2) ln sigma2 + ln|det M(theta)|.

The rank condition uses ``numpy.linalg.matrix_rank``, whose SVD cut-off s_max * max(n, p) * eps is the
``lstsq(rcond=None)`` cut-off. ``maximize`` re-runs the frozen optimizer (199-point grid, first maximal
index, golden section with exactly 60 iterations, earliest-evaluation ties, |theta| <= 0.99 - 1e-6).

Fitted quantities (§2.6-§2.7). Spatial lags are explicit loops over neighbour lists, never products with
W: SEM one-step X beta + lambda lag(u), SAR one-step rho lag(y) + X beta; held-out x_o'beta + a_o +
theta lag_o. a_o is the V1 mean-effect adjustment: the mean of the multiset [0] + [beta('c=level') for the
sorted training levels after the first], added once per categorical whose held-out level is unseen. (The
spec writes the set {0} U {beta}; ``cv.V1Design.predict``, which it says to apply exactly, averages the
array, so equal effects are not merged here.)

Accounting (§2.11). Coalition R2 is a projection R2. Rank deficiency is handled by greedy column trimming:
a column is kept only if it raises ``matrix_rank`` of the kept columns; the reduced QR of the kept columns
gives the projection Q Q' y. For full-rank coalitions this is exactly the lstsq projection. Shapley shares
are averages of marginal R2 over all G! orderings (not the subset-weight formula).

Expected information (§2.5). The Anselin (1988) entries are assembled from explicit element sums
(``numpy.einsum``): tr(A) = sum_i A_ii, tr(AA) = sum_ij A_ij A_ji, tr(A'A) = sum_ij A_ij^2, with W M^-1
from ``numpy.linalg.solve(M, I)``; the (p + 2) matrix, ordered (beta, sigma2, theta), is inverted by
``numpy.linalg.solve`` against the identity. A singular matrix (LinAlgError) is recorded as not computable,
like a non-finite or non-positive variance.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations

import numpy as np

VERSION = 'm3-independent-math-v1'
FAMILIES = ('SEM', 'SAR')
K_NEIGHBOURS = 8
LINK_WEIGHT = 1.0 / K_NEIGHBOURS
MIN_TRAINING_NODES = 9
DOMAIN = (-0.99, 0.99)
GRID_POINTS = 199
GOLDEN_ITERATIONS = 60
PHI = (math.sqrt(5.0) - 1.0) / 2.0
INTERIOR_MARGIN = 1e-6
SCAN_POINTS = 397
LOCAL_HALF_WIDTH = 1e-3
LOCAL_POINTS = 41
OPTIMALITY_ATOL = 1e-8
Z_975 = 1.959963984540054
RESAMPLES = 2000
RESAMPLE_SEED = 0
MATERIAL_EMISSIONS_SHARE = 0.10
NOT_COMPUTABLE = 'not computable'


class StructuralFailure(ValueError):
    """A graph cannot be built under the frozen rules. There is no fallback."""


class EvaluationFailure(ValueError):
    """One likelihood evaluation failed a §2.5 condition."""


def _family(family: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f'family must be one of {FAMILIES}, not {family!r}')
    return family


def _fsum(values) -> float:
    return math.fsum(np.asarray(values, dtype=float).ravel().tolist())


# --- graphs (§2.2) ------------------------------------------------------------------------------

def _distances(dist) -> np.ndarray:
    d = np.asarray(dist, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or not np.isfinite(d).all():
        raise ValueError('distances must be a finite square matrix')
    return d


def _canonical(nodes, size: int) -> list[int]:
    members = [int(v) for v in nodes]
    if any(b <= a for a, b in zip(members, members[1:])) or any(v < 0 or v >= size for v in members):
        raise ValueError('a node set must be strictly increasing canonical indices of the distance matrix')
    return members


def _first_eight(row: np.ndarray, candidates: Iterable[int]) -> tuple[int, ...]:
    ranked = sorted((float(row[j]), j) for j in candidates)
    return tuple(j for _, j in ranked[:K_NEIGHBOURS])


def knn_neighbours(dist, nodes) -> list[tuple[int, ...]]:
    """kNN8 on N: per node of N (in order), its 8 canonical neighbours ordered by (distance, index)."""
    d = _distances(dist)
    members = _canonical(nodes, len(d))
    if len(members) < MIN_TRAINING_NODES:
        raise StructuralFailure(f'|T| = {len(members)} < {MIN_TRAINING_NODES}')
    return [_first_eight(d[i], (j for j in members if j != i)) for i in members]


def attachment_neighbours(dist, train_nodes, held_out_nodes) -> list[tuple[int, ...]]:
    """Sink rule: each held-out node links to the first 8 training nodes by (distance, index)."""
    d = _distances(dist)
    train = _canonical(train_nodes, len(d))
    held_out = _canonical(held_out_nodes, len(d))
    if len(train) < MIN_TRAINING_NODES:
        raise StructuralFailure(f'|T| = {len(train)} < {MIN_TRAINING_NODES}')
    if set(train) & set(held_out):
        raise ValueError('a held-out node cannot also be a training node')
    return [_first_eight(d[o], train) for o in held_out]


def local_positions(neighbours: Sequence[Sequence[int]], nodes: Sequence[int]) -> list[tuple[int, ...]]:
    """Canonical neighbour indices -> positions inside the (canonically ordered) node set."""
    position = {int(v): i for i, v in enumerate(nodes)}
    return [tuple(position[int(j)] for j in row) for row in neighbours]


def _check_lists(local_neighbours, n_columns: int) -> None:
    for row in local_neighbours:
        if len(row) != K_NEIGHBOURS or len(set(row)) != K_NEIGHBOURS or any(j < 0 or j >= n_columns
                                                                           for j in row):
            raise ValueError(f'each neighbour list needs {K_NEIGHBOURS} distinct positions below {n_columns}')


def dense_weights(local_neighbours: Sequence[Sequence[int]], n_columns: int) -> np.ndarray:
    """Rows of 1/8 weights on the listed positions (W_TT when rows are T, W_OT when rows are held out)."""
    _check_lists(local_neighbours, n_columns)
    w = np.zeros((len(local_neighbours), n_columns))
    for i, row in enumerate(local_neighbours):
        for j in row:
            w[i, j] = LINK_WEIGHT
    return w


def lag(values, local_neighbours: Sequence[Sequence[int]]) -> np.ndarray:
    """sum_j w_ij v_j by an explicit loop over each neighbour list (no product with W)."""
    v = np.asarray(values, dtype=float)
    _check_lists(local_neighbours, len(v))
    out = np.empty(len(local_neighbours))
    for i, row in enumerate(local_neighbours):
        total = 0.0
        for j in row:
            total += LINK_WEIGHT * float(v[j])
        out[i] = total
    return out


# --- likelihood (§2.5) --------------------------------------------------------------------------

def weight_eigenvalues(w) -> np.ndarray:
    return np.linalg.eigvals(np.asarray(w, dtype=float)).astype(complex)


def eigen_logdet(eigenvalues, theta: float) -> tuple[float, float]:
    """(sign, ln|det(I - theta W)|) from the eigenvalues of W; (0, -inf) when a factor vanishes."""
    factors = 1.0 - float(theta) * np.asarray(eigenvalues, dtype=complex)
    moduli = np.abs(factors)
    if not (moduli > 0).all():
        return 0.0, -math.inf
    sign = 1.0 if math.cos(_fsum(np.angle(factors))) > 0 else -1.0
    return sign, _fsum(np.log(moduli))


@dataclass(frozen=True, eq=False)
class Profile:
    """One concentrated-likelihood evaluation; ``failure`` names the first §2.5 condition that failed."""

    family: str
    theta: float
    loglik: float
    beta: np.ndarray
    sigma2: float
    sign: float
    logdet: float
    rank: int
    failure: str | None

    @property
    def valid(self) -> bool:
        return self.failure is None


def _regression(y, x, w) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x, w = (np.asarray(a, dtype=float) for a in (y, x, w))
    n = len(y)
    if y.ndim != 1 or x.ndim != 2 or x.shape[0] != n or w.shape != (n, n):
        raise ValueError(f'shapes do not form a regression: y {y.shape}, X {x.shape}, W {w.shape}')
    if not (np.isfinite(y).all() and np.isfinite(x).all() and np.isfinite(w).all()):
        raise ValueError('y, X and W must be finite')
    return y, x, w


def _transformed(family: str, y, x, w, theta: float) -> tuple[np.ndarray, np.ndarray]:
    m = np.eye(len(y)) - theta * w
    return (m @ x if family == 'SEM' else x), m @ y


def _qr_coefficients(z: np.ndarray, target: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(z)
    return np.linalg.solve(r, q.T @ target)


def profile(family: str, y, x, w, theta: float, eigenvalues=None) -> Profile:
    """l_c(theta), beta(theta), sigma2(theta) through eigenvalues and QR, with the §2.5 failure rules."""
    family = _family(family)
    y, x, w = _regression(y, x, w)
    n, p = x.shape
    theta = float(theta)
    sign, logdet = eigen_logdet(weight_eigenvalues(w) if eigenvalues is None else eigenvalues, theta)
    x_t, y_t = _transformed(family, y, x, w, theta)
    rank = int(np.linalg.matrix_rank(x_t))
    failure = None
    if sign != 1.0:
        failure = f'determinant sign {sign:+g}'
    elif not math.isfinite(logdet):
        failure = 'log-determinant not finite'
    elif rank != p:
        failure = f'rank {rank} != p {p}'
    if failure is not None:
        return Profile(family, theta, math.nan, np.full(p, np.nan), math.nan, sign, logdet, rank, failure)
    beta = _qr_coefficients(x_t, y_t)
    e = y_t - x_t @ beta
    sigma2 = _fsum(e * e) / n
    if not (math.isfinite(sigma2) and sigma2 > 0):
        failure = 'sigma2 not finite and positive'
        return Profile(family, theta, math.nan, beta, sigma2, sign, logdet, rank, failure)
    loglik = -(n / 2) * (math.log(2 * math.pi) + 1) - (n / 2) * math.log(sigma2) + logdet
    failure = None if math.isfinite(loglik) else 'log-likelihood not finite'
    return Profile(family, theta, loglik, beta, sigma2, sign, logdet, rank, failure)


def beta_given_theta(family: str, y, x, w, theta: float) -> np.ndarray:
    """beta(theta) by QR on the transformed regression; raises if the evaluation fails."""
    result = profile(family, y, x, w, theta)
    if not result.valid:
        raise EvaluationFailure(f'theta={result.theta!r}: {result.failure}')
    return result.beta


def strict_local_maxima(values: Sequence[float]) -> int:
    """Strict local maxima of a sequence; the two endpoints are compared with their one neighbour."""
    v = [float(a) for a in values]
    last = len(v) - 1
    return sum(1 for i in range(len(v)) if (i == 0 or v[i] > v[i - 1]) and (i == last or v[i] > v[i + 1]))


def _earliest_max(pairs: Sequence[tuple[float, float]]) -> tuple[float, float]:
    best = pairs[0]
    for pair in pairs[1:]:
        if pair[1] > best[1]:
            best = pair
    return best


@dataclass(frozen=True, eq=False)
class Maximization:
    theta: float
    loglik: float
    grid: tuple[float, ...]
    strict_local_maxima: int | None
    evaluations: tuple[tuple[float, float], ...]
    failure: str | None
    at_domain_bound: bool = False


def maximize(objective: Callable[[float], float]) -> Maximization:
    """The frozen §2.5 search on any objective; the objective raises EvaluationFailure to fail a point."""
    evaluations: list[tuple[float, float]] = []

    def evaluate(theta: float) -> float:
        value = float(objective(theta))
        if not math.isfinite(value):
            raise EvaluationFailure(f'non-finite value at theta={theta!r}')
        evaluations.append((theta, value))
        return value

    grid = [float(t) for t in np.linspace(DOMAIN[0], DOMAIN[1], GRID_POINTS)]
    try:
        values = [evaluate(t) for t in grid]
        k = 0
        for i, value in enumerate(values):
            if value > values[k]:
                k = i
        a, b = grid[max(k - 1, 0)], grid[min(k + 1, GRID_POINTS - 1)]
        c, d = b - PHI * (b - a), a + PHI * (b - a)
        fc = evaluate(c)
        fd = evaluate(d)
        for _ in range(GOLDEN_ITERATIONS):
            if fc >= fd:
                b, d, fd = d, c, fc
                c = b - PHI * (b - a)
                fc = evaluate(c)
            else:
                a, c, fc = c, d, fd
                d = a + PHI * (b - a)
                fd = evaluate(d)
    except EvaluationFailure as exc:
        done = tuple(v for _, v in evaluations[:GRID_POINTS])
        return Maximization(math.nan, math.nan, done, None, tuple(evaluations), f'evaluation failed: {exc}')
    theta, loglik = _earliest_max(evaluations)
    failure = None
    at_bound = not abs(theta) <= DOMAIN[1] - INTERIOR_MARGIN
    return Maximization(theta, loglik, tuple(values), strict_local_maxima(values), tuple(evaluations),
                        failure, at_bound)


@dataclass(frozen=True, eq=False)
class ReferenceFit:
    family: str
    theta: float
    loglik: float
    beta: np.ndarray
    sigma2: float
    n: int
    p: int
    strict_local_maxima: int | None
    grid: tuple[float, ...]
    failure: str | None
    at_domain_bound: bool = False


def reference_fit(family: str, y, x, w) -> ReferenceFit:
    """The whole §2.5 fit re-derived on the eigenvalue/QR likelihood."""
    family = _family(family)
    y, x, w = _regression(y, x, w)
    eigenvalues = weight_eigenvalues(w)

    def objective(theta: float) -> float:
        result = profile(family, y, x, w, theta, eigenvalues)
        if not result.valid:
            raise EvaluationFailure(f'theta={theta!r}: {result.failure}')
        return result.loglik

    search = maximize(objective)
    n, p = x.shape
    if search.failure is not None:
        return ReferenceFit(family, search.theta, search.loglik, np.full(p, np.nan), math.nan, n, p,
                            search.strict_local_maxima, search.grid, search.failure)
    best = profile(family, y, x, w, search.theta, eigenvalues)
    return ReferenceFit(family, search.theta, best.loglik, best.beta, best.sigma2, n, p,
                        search.strict_local_maxima, search.grid, None, search.at_domain_bound)


def local_optimality_check(family: str, y, x, w, theta_hat: float, half_width: float = LOCAL_HALF_WIDTH,
                           points: int = LOCAL_POINTS, atol: float = OPTIMALITY_ATOL) -> dict:
    """Is l_c(theta_hat) >= every value on a fine grid around it (clipped to the domain) minus atol?"""
    y, x, w = _regression(y, x, w)
    eigenvalues = weight_eigenvalues(w)
    theta_hat = float(theta_hat)
    at = profile(family, y, x, w, theta_hat, eigenvalues)
    grid = np.clip(np.linspace(theta_hat - half_width, theta_hat + half_width, points), *DOMAIN)
    evaluated = [profile(family, y, x, w, float(t), eigenvalues) for t in grid]
    failed = [r.theta for r in evaluated if not r.valid]
    valid = [(r.theta, r.loglik) for r in evaluated if r.valid]
    best_theta, best_loglik = _earliest_max(valid) if valid else (math.nan, math.nan)
    passes = bool(at.valid and not failed and at.loglik >= best_loglik - atol)
    return {'theta_hat': theta_hat, 'loglik_at_estimate': at.loglik, 'best_grid_theta': best_theta,
            'best_grid_loglik': best_loglik, 'shortfall': best_loglik - at.loglik, 'half_width': half_width,
            'points': points, 'grid_min': float(grid.min()), 'grid_max': float(grid.max()), 'atol': atol,
            'failed_points': failed, 'passes': passes}


def global_scan(family: str, y, x, w, theta_hat: float | None = None, points: int = SCAN_POINTS,
                atol: float = OPTIMALITY_ATOL) -> dict:
    """Coarse scan of the whole domain; the estimate is in the global basin if it lies between the scan
    points either side of the scan argmax (and attains the scan maximum up to atol)."""
    y, x, w = _regression(y, x, w)
    eigenvalues = weight_eigenvalues(w)
    grid = [float(t) for t in np.linspace(DOMAIN[0], DOMAIN[1], points)]
    evaluated = [profile(family, y, x, w, t, eigenvalues) for t in grid]
    failed = [r.theta for r in evaluated if not r.valid]
    valid = [(i, r.loglik) for i, r in enumerate(evaluated) if r.valid]
    if not valid:
        return {'points': points, 'failed_points': failed, 'theta_argmax': math.nan}
    k, best = _earliest_max(valid)
    record = {'points': points, 'failed_points': failed, 'theta_argmax': grid[k], 'loglik_max': best,
              'basin': [grid[max(k - 1, 0)], grid[min(k + 1, points - 1)]],
              'strict_local_maxima': None if failed else strict_local_maxima([r.loglik for r in evaluated])}
    if theta_hat is not None:
        at = profile(family, y, x, w, float(theta_hat), eigenvalues)
        record['loglik_at_estimate'] = at.loglik
        record['in_global_basin'] = bool(record['basin'][0] <= theta_hat <= record['basin'][1]
                                         and at.valid and at.loglik >= best - atol)
    return record


# --- fitted quantities and held-out prediction (§2.6-§2.7) --------------------------------------

def _row_means(x, beta) -> np.ndarray:
    x, beta = np.asarray(x, dtype=float), np.asarray(beta, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(beta):
        raise ValueError(f'design {x.shape} does not match beta {beta.shape}')
    return np.array([math.fsum((row * beta).tolist()) for row in x])


def in_sample_quantities(family: str, y, x, local_neighbours, theta: float, beta, sigma2=None) -> dict:
    """Trend, structural residual, one-step fitted value and innovation (two expressions) for one fit."""
    family = _family(family)
    y = np.asarray(y, dtype=float)
    theta = float(theta)
    mean = _row_means(x, beta)
    m = np.eye(len(y)) - theta * dense_weights(local_neighbours, len(y))
    if family == 'SEM':
        trend = mean
        structural = y - trend
        one_step = mean + theta * lag(structural, local_neighbours)
        alternative = m @ structural
    else:
        trend = np.linalg.solve(m, mean)
        structural = y - trend
        one_step = theta * lag(y, local_neighbours) + mean
        alternative = m @ y - mean
    innovation = y - one_step
    record = {'trend': trend, 'structural_residual': structural, 'one_step': one_step,
              'innovation': innovation, 'innovation_alternative': alternative,
              'innovation_expression_gap': float(np.max(np.abs(innovation - alternative))),
              'mean_squared_innovation': _fsum(innovation * innovation) / len(y)}
    if sigma2 is not None:
        record['sigma2_relative_gap'] = abs(record['mean_squared_innovation'] - sigma2) / sigma2
    return record


def unseen_level_adjustment(beta, names: Sequence[str], training_values: Mapping[str, Iterable],
                            test_values: Mapping[str, Sequence], n_test: int) -> np.ndarray:
    """V1 mean-effect rule: per categorical c, add mean([0] + beta of c=level for sorted training
    levels[1:]) to every held-out row whose level of c is not a training level."""
    beta = np.asarray(beta, dtype=float)
    names = list(names)
    if set(training_values) != set(test_values):
        raise ValueError('training and held-out categorical features differ')
    out = np.zeros(n_test)
    for c in training_values:
        levels = sorted({str(v) for v in training_values[c]})
        expected = [f'{c}={level}' for level in levels[1:]]
        coded = [name for name in names if name.startswith(f'{c}=')]
        if sorted(coded) != sorted(expected):
            raise ValueError(f'coefficient names for {c!r} are not the drop-first training coding')
        effects = [0.0] + [float(beta[names.index(name)]) for name in expected]
        mean_effect = math.fsum(effects) / len(effects)
        rows = [str(v) for v in test_values[c]]
        if len(rows) != n_test:
            raise ValueError(f'{c!r} has {len(rows)} held-out rows, not {n_test}')
        known = set(levels)
        out += np.array([0.0 if v in known else mean_effect for v in rows])
    return out


def held_out_predictions(family: str, theta: float, beta, x_train, y_train, x_test, attachments,
                         adjustment=None) -> np.ndarray:
    """SEM: x_o'beta + a_o + lambda lag_o(y_T - X_T beta); SAR: x_o'beta + a_o + rho lag_o(y_T).

    ``attachments`` are positions inside the training rows. Only training outcomes are an input."""
    family = _family(family)
    y_train = np.asarray(y_train, dtype=float)
    source = y_train - _row_means(x_train, beta) if family == 'SEM' else y_train
    mean = _row_means(x_test, beta)
    a = np.zeros(len(mean)) if adjustment is None else np.asarray(adjustment, dtype=float)
    if len(attachments) != len(mean) or a.shape != mean.shape:
        raise ValueError('one attachment list and one adjustment per held-out row are required')
    return mean + a + float(theta) * lag(source, attachments)


# --- metrics ------------------------------------------------------------------------------------

def rmse(y, prediction) -> float:
    e = np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)
    return math.sqrt(_fsum(e * e) / len(e))


def mae(y, prediction) -> float:
    e = np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)
    return _fsum(np.abs(e)) / len(e)


def r2(y, prediction) -> float:
    y = np.asarray(y, dtype=float)
    e = y - np.asarray(prediction, dtype=float)
    centred = y - _fsum(y) / len(y)
    return 1.0 - _fsum(e * e) / _fsum(centred * centred)


def calibration(y, prediction) -> dict:
    """OLS of y on the prediction: the 2 x 2 normal equations [n, Sp; Sp, Spp](a, b) = (Sy, Spy) by
    Cramer's rule."""
    y, p = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    n, s_p, s_y, s_pp, s_py = len(y), _fsum(p), _fsum(y), _fsum(p * p), _fsum(p * y)
    det = n * s_pp - s_p * s_p
    return {'slope': (n * s_py - s_p * s_y) / det, 'intercept': (s_y * s_pp - s_p * s_py) / det}


def paired_rmse_interval(y, new, reference) -> dict:
    """RMSE(new) - RMSE(reference) with the 2,000-resample, seed-0, identical-index percentile interval."""
    y, new, reference = (np.asarray(a, dtype=float) for a in (y, new, reference))
    n = len(y)
    rng = np.random.default_rng(RESAMPLE_SEED)
    idx = rng.integers(0, n, size=(RESAMPLES, n))
    delta = (np.sqrt(np.mean((y - new)[idx] ** 2, axis=1))
             - np.sqrt(np.mean((y - reference)[idx] ** 2, axis=1)))
    lower, upper = np.quantile(delta, [0.025, 0.975])
    return {'delta_rmse': rmse(y, new) - rmse(y, reference), 'interval': [float(lower), float(upper)],
            'resamples': RESAMPLES, 'seed': RESAMPLE_SEED}


def moran_statistic(values, w) -> float:
    """(n / S0) z'Wz / z'z with z the centred values; no permutation inference."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(w, dtype=float)
    z = v - _fsum(v) / len(v)
    return (len(z) / _fsum(w)) * float(np.einsum('i,ij,j->', z, w, z)) / _fsum(z * z)


# --- accounting (§2.11) -------------------------------------------------------------------------

def accounting_parts(y, x_static, innovations, static_shares: Mapping[str, float] | None = None) -> dict:
    """Static, represented-dependence and innovation parts of TSS, with S's OLS by QR (rank p)."""
    y = np.asarray(y, dtype=float)
    x_static = np.asarray(x_static, dtype=float)
    innovations = np.asarray(innovations, dtype=float)
    if int(np.linalg.matrix_rank(x_static)) != x_static.shape[1]:
        raise ValueError('the static design must have full column rank')
    centred = y - _fsum(y) / len(y)
    tss = _fsum(centred * centred)
    residual = y - x_static @ _qr_coefficients(x_static, y)
    rss_static, rss_innovation = _fsum(residual * residual), _fsum(innovations * innovations)
    record = {'tss': tss, 'rss_static': rss_static, 'rss_innovation': rss_innovation,
              'static_component': 1.0 - rss_static / tss,
              'represented_spatial_dependence': (rss_static - rss_innovation) / tss,
              'remaining_innovation': rss_innovation / tss,
              'absorbed_fraction_of_static_residual': (rss_static - rss_innovation) / rss_static}
    record['identity_gap'] = (record['static_component'] + record['represented_spatial_dependence']
                              + record['remaining_innovation'] - 1.0)
    if static_shares is not None:
        record['static_shares_sum_gap'] = math.fsum(static_shares.values()) - record['static_component']
    return record


def column_space_basis(z) -> tuple[np.ndarray, list[int]]:
    """Orthonormal basis of span(z) from greedily rank-trimmed columns (see module docstring)."""
    z = np.asarray(z, dtype=float)
    keep: list[int] = []
    rank = 0
    for j in range(z.shape[1]):
        trial = int(np.linalg.matrix_rank(z[:, keep + [j]]))
        if trial > rank:
            keep, rank = keep + [j], trial
    q, _ = np.linalg.qr(z[:, keep])
    return q, keep


def projection_r2(z, target) -> tuple[float, np.ndarray]:
    """1 - RSS/TSS for the projection of target on span(z); z must contain the intercept."""
    target = np.asarray(target, dtype=float)
    centred = target - _fsum(target) / len(target)
    tss = _fsum(centred * centred)
    if not tss > 0:
        raise ValueError('the target has no variance')
    q, _ = column_space_basis(z)
    fitted = q @ (q.T @ target)
    e = target - fitted
    return 1.0 - _fsum(e * e) / tss, fitted


def permutation_shapley(players: Sequence[Hashable], value: Callable[[frozenset], float]) -> dict:
    """Shapley values as the average marginal contribution over all orderings of the players."""
    players = list(players)
    cache: dict[frozenset, float] = {}

    def v(coalition: frozenset) -> float:
        if coalition not in cache:
            cache[coalition] = float(value(coalition))
        return cache[coalition]

    marginals: dict[Hashable, list[float]] = {player: [] for player in players}
    for order in permutations(players):
        coalition: frozenset = frozenset()
        before = v(coalition)
        for player in order:
            coalition = coalition | {player}
            after = v(coalition)
            marginals[player].append(after - before)
            before = after
    orderings = math.factorial(len(players))
    return {player: math.fsum(m) / orderings for player, m in marginals.items()}


def filtered_decomposition(family: str, y, groups: Mapping[str, np.ndarray], w, theta: float) -> dict:
    """Filtered-outcome group shares at fixed theta: y* = M y on intercept + (M X_g for SEM, X_g for SAR)."""
    family = _family(family)
    y = np.asarray(y, dtype=float)
    theta = float(theta)
    m = np.eye(len(y)) - theta * np.asarray(w, dtype=float)
    y_star = m @ y
    names = list(groups)
    blocks = {g: (m @ np.asarray(groups[g], dtype=float).reshape(len(y), -1) if family == 'SEM'
                  else np.asarray(groups[g], dtype=float).reshape(len(y), -1)) for g in names}
    intercept = np.ones((len(y), 1))

    def design(coalition) -> np.ndarray:
        return np.hstack([intercept] + [blocks[g] for g in names if g in coalition])

    shares = permutation_shapley(names, lambda coalition: projection_r2(design(coalition), y_star)[0])
    full_r2, full_fitted = projection_r2(design(set(names)), y_star)
    record = {'family': family, 'theta': theta, 'shares': shares, 'r2_full': full_r2,
              'residual': 1.0 - full_r2, 'shares_sum_gap': math.fsum(shares.values()) - full_r2,
              'composition': {g: s / full_r2 for g, s in shares.items()}, 'full_fitted': full_fitted,
              'filtered_outcome': y_star}
    if 'geography' in shares and 'emissions' in shares:
        not_largest = not all(shares['geography'] > s for g, s in shares.items() if g != 'geography')
        exceeds = shares['emissions'] > MATERIAL_EMISSIONS_SHARE
        record['material_change'] = {'geography_not_strictly_largest': not_largest,
                                     'emissions_exceeds_0_10': exceeds,
                                     'material_change': not_largest or exceeds}
    return record


# --- expected information (§2.5) ---------------------------------------------------------------

def expected_information(family: str, x, w, theta: float, beta, sigma2: float) -> np.ndarray:
    """Anselin expected information at (beta, sigma2, theta), ordered (beta, sigma2, theta)."""
    family = _family(family)
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    beta = np.asarray(beta, dtype=float)
    n, p = x.shape
    theta, sigma2 = float(theta), float(sigma2)
    m = np.eye(n) - theta * w
    wm = np.einsum('ik,kj->ij', w, np.linalg.solve(m, np.eye(n)))
    trace = float(np.einsum('ii->', wm))
    trace_square = float(np.einsum('ij,ji->', wm, wm))
    trace_gram = float(np.einsum('ij,ij->', wm, wm))
    info = np.zeros((p + 2, p + 2))
    s, t = p, p + 1
    if family == 'SEM':
        mx = np.einsum('ik,kj->ij', m, x)
        info[:p, :p] = np.einsum('ki,kj->ij', mx, mx) / sigma2
        info[t, t] = trace_square + trace_gram
    else:
        lagged_mean = np.einsum('ik,kj,j->i', wm, x, beta)
        info[:p, :p] = np.einsum('ki,kj->ij', x, x) / sigma2
        info[:p, t] = info[t, :p] = np.einsum('ki,k->i', x, lagged_mean) / sigma2
        info[t, t] = trace_square + trace_gram + float(np.einsum('i,i->', lagged_mean, lagged_mean)) / sigma2
    info[s, s] = n / (2.0 * sigma2 * sigma2)
    info[s, t] = info[t, s] = trace / sigma2
    return info


def theta_standard_error(family: str, x, w, theta: float, beta, sigma2: float) -> dict:
    """se(theta) = sqrt((I^-1)_theta,theta) and the Wald 95% interval, or 'not computable'."""
    info = expected_information(family, x, w, theta, beta, sigma2)
    try:
        variance = float(np.linalg.solve(info, np.eye(len(info)))[-1, -1])
    except np.linalg.LinAlgError:
        variance = math.nan
    if not math.isfinite(variance) or variance <= 0:
        return {'variance': variance, 'se': NOT_COMPUTABLE, 'wald_95': NOT_COMPUTABLE}
    se = math.sqrt(variance)
    return {'variance': variance, 'se': se, 'wald_95': [float(theta) - Z_975 * se, float(theta) + Z_975 * se]}
