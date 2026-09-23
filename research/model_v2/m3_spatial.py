"""M3 explicit spatial structure: kNN8 graphs, SEM/SAR Gaussian ML, held-out prediction, accounting.

Versioned reference implementation of the M3 spatial rules, NumPy only (no spatial-econometrics
library). Every numerical rule is fixed: graph and tie rule, grid,
golden-section iteration count, tie rules of the maximizer, domain, interior rule, failure conditions,
information matrix, accounting identities. The module reads no file and never sees a named outcome.

Graphs. Nodes are canonical row indices of one full distance matrix D. kNN8 on a canonical-ordered
node set N gives each i the first 8 elements of N \\ {i} ordered by (D[i, j], canonical j), weight 1/8,
zero diagonal: ``spatial.knn_weights`` on D[N, N], whose local order is the canonical order. A held-out
node o is a sink: it links to the first 8 training nodes by (D[o, j], canonical j) with weight 1/8 and
sends nothing, so the augmented matrix over (T, O) is W* = [[W_TT, 0], [W_OT, 0]].

Models, with M(theta) = I - theta W and W row-standardised:

    SEM   y = X beta + u,   u = lambda W u + eps          SAR   y = rho W y + X beta + eps

Concentrated log-likelihood. SEM regresses M y on M X, SAR regresses M y on X, both with
``numpy.linalg.lstsq(rcond=None)``; e is the transformed residual, sigma2(theta) = e.e / n and

    lc(theta) = -(n/2)(ln 2 pi + 1) - (n/2) ln sigma2(theta) + logdet M(theta)      (numpy slogdet).

An evaluation fails when the slogdet sign is not +1, logdet is non-finite, the lstsq rank is not p,
sigma2 is not finite and positive, or lc is non-finite. Maximization: lc on
``linspace(-0.99, 0.99, 199)`` in order; bracket [G[k*-1], G[k*+1]] (clipped) around the first argmax
k*; golden section with phi = (sqrt 5 - 1)/2 evaluating c then d, then exactly 60 one-point iterations;
theta-hat is the best evaluated point with ties to the earliest evaluation. This is the constrained ML
estimate over the conventional domain (as PySAL ``spreg`` bounds ML_Lag and ML_Error to (-1, 1)): an
estimate with |theta-hat| > 0.99 - 1e-6 is valid, flagged ``at_domain_bound`` and counted, and its Wald
interval is not applicable. Any evaluation failure raises :class:`SpatialFitFailure`; there is no fallback.

Fitted quantities and prediction. SEM: trend X beta, one-step X beta + lambda W u with
u = y - X beta, held-out x_o'beta + lambda sum_j w_oj u_j. SAR: trend M^-1 X beta, one-step
rho W y + X beta, held-out x_o'beta + rho sum_j w_oj y_j. Under W* these are E[y_o | y_T] given the
estimates, and they use training outcomes only. The inherited unseen-level adjustment is a separate step.

Expected information (Anselin 1988, ch. 6), parameters ordered (beta, sigma2, theta):
    SEM, B = M(lambda), W_B = W B^-1: I_bb = X'B'BX/s2, I_ss = n/(2 s2^2), I_sl = tr W_B/s2,
        I_ll = tr(W_B W_B) + tr(W_B'W_B), all other cross terms 0;
    SAR, A = M(rho), W_A = W A^-1, m = W_A X beta: I_bb = X'X/s2, I_br = X'm/s2, I_ss = n/(2 s2^2),
        I_sr = tr W_A/s2, I_rr = tr(W_A W_A) + tr(W_A'W_A) + m'm/s2, I_bs = 0.
se(theta-hat) = sqrt((I^-1)_theta,theta) from ``numpy.linalg.inv`` of the whole matrix.

Accounting. A_static = 1 - RSS_S/TSS, A_dependence = (RSS_S - RSS_eps)/TSS,
A_innovation = RSS_eps/TSS, identity checked to 1e-9. Filtered-trend group shares: y* = M(theta-hat) y
regressed on intercept + M X_g (SEM; M 1 = (1 - lambda) 1, so the intercept column is unchanged) or
intercept + X_g (SAR) for every coalition, Shapley weights s!(G-s-1)!/G! as ``src.decomposition``.
Coalition columns are stacked in the given group order, never in hash order.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import NamedTuple

import numpy as np

from research.model_v2.spatial import knn_weights

ESTIMATOR_VERSION = 'm3-spatial-v1'
FAMILIES = ('sem', 'sar')
K = 8
NEIGHBOUR_WEIGHT = 1.0 / K
MIN_TRAINING_NODES = 9
DOMAIN_BOUND = 0.99
GRID_POINTS = 199
GOLDEN_ITERATIONS = 60
PHI = (math.sqrt(5.0) - 1.0) / 2.0
INTERIOR_MARGIN = 1e-6
INNOVATION_ATOL = 1e-10
INNOVATION_VARIANCE_RTOL = 1e-10
WALD_Z = 1.959963984540054
ACCOUNTING_ATOL = 1e-9
COALITION_FIT_ATOL = 1e-10
SHARE_SUM_ATOL = 1e-9
NOT_COMPUTABLE = 'not computable'


class SpatialFitFailure(ValueError):
    """A structural failure of an M3 fit: the fit is non-computable, never retried."""

    def __init__(self, reason: str, detail: str = ''):
        super().__init__(f'{reason}: {detail}' if detail else reason)
        self.reason = reason
        self.detail = detail


class IntegrityFailure(RuntimeError):
    """An accounting identity failed: an integrity failure, the run stops."""


# --- graphs ------------------------------------------------------------------------------

def _square(dist) -> np.ndarray:
    d = np.asarray(dist, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f'a distance matrix must be square, found shape {d.shape}')
    return d


def _canonical(indices, n: int, name: str) -> np.ndarray:
    idx = np.asarray(indices)
    if idx.ndim != 1 or (len(idx) and not np.issubdtype(idx.dtype, np.integer)):
        raise ValueError(f'{name} must be a 1-D integer array of canonical indices')
    if (np.diff(idx) <= 0).any() or (len(idx) and (idx[0] < 0 or idx[-1] >= n)):
        raise ValueError(f'{name} must be strictly increasing canonical indices in [0, {n})')
    return idx.astype(np.intp)


def _require_size(n: int) -> None:
    if n < MIN_TRAINING_NODES:
        raise SpatialFitFailure('too-few-training-nodes', f'|T| = {n} < {MIN_TRAINING_NODES}')


def validate_graph(w) -> None:
    """|T| >= 9, zero diagonal and every row summing to exactly 1, or a structural failure."""
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f'a graph must be a square matrix, found shape {w.shape}')
    _require_size(len(w))
    if not (np.diagonal(w) == 0.0).all():
        raise SpatialFitFailure('graph-diagonal', 'a node links to itself')
    if not (w.sum(axis=1) == 1.0).all():
        raise SpatialFitFailure('graph-row-sum', 'a row does not sum to exactly 1')


def training_graph(dist, train_idx) -> np.ndarray:
    """W = kNN8(T), rows and columns in canonical order of the training nodes."""
    d = _square(dist)
    train = _canonical(train_idx, len(d), 'train_idx')
    _require_size(len(train))
    w = knn_weights(d[np.ix_(train, train)], K)
    validate_graph(w)
    return w


def training_neighbours(dist, train_idx) -> np.ndarray:
    """(|T|, 8) canonical neighbour indices of each training node, nearest first (for artifacts)."""
    d = _square(dist)
    train = _canonical(train_idx, len(d), 'train_idx')
    _require_size(len(train))
    rows = np.empty((len(train), K), dtype=np.intp)
    for r, i in enumerate(train):
        order = train[np.lexsort((train, d[i, train]))]
        rows[r] = order[order != i][:K]
    return rows


def attachment(dist, train_idx, test_idx) -> tuple[np.ndarray, np.ndarray]:
    """Sink attachment of held-out nodes: (n_test, 8) canonical training neighbours and 1/8 weights."""
    d = _square(dist)
    train = _canonical(train_idx, len(d), 'train_idx')
    test = _canonical(test_idx, len(d), 'test_idx')
    _require_size(len(train))
    if np.intersect1d(train, test).size:
        raise ValueError('a held-out node cannot also be a training node')
    neighbours = np.empty((len(test), K), dtype=np.intp)
    for r, o in enumerate(test):
        neighbours[r] = train[np.lexsort((train, d[o, train]))[:K]]
    return neighbours, np.full((len(test), K), NEIGHBOUR_WEIGHT)


def local_positions(train_idx, canonical) -> np.ndarray:
    """Positions within the training rows of canonical training indices (any shape)."""
    train = np.asarray(train_idx)
    c = np.asarray(canonical)
    pos = np.searchsorted(train, c)
    if len(train) == 0 or (pos >= len(train)).any() or not (train[np.minimum(pos, len(train) - 1)] == c).all():
        raise ValueError('an index is not a training node')
    return pos.astype(np.intp)


def augmented_graph(w_train, neighbours, weights) -> np.ndarray:
    """Dense W* = [[W_TT, 0], [W_OT, 0]] from local neighbour positions (reference and checks)."""
    w_train = np.asarray(w_train, dtype=float)
    nb, wt = np.asarray(neighbours), np.asarray(weights, dtype=float)
    n_t, n_o = len(w_train), len(nb)
    out = np.zeros((n_t + n_o, n_t + n_o))
    out[:n_t, :n_t] = w_train
    for r in range(n_o):
        out[n_t + r, nb[r]] += wt[r]
    return out


# --- concentrated likelihood and its maximization --------------------------------------

class Evaluation(NamedTuple):
    loglik: float
    beta: np.ndarray
    sigma2: float


class Maximum(NamedTuple):
    theta: float
    value: float
    grid_values: np.ndarray
    n_evaluations: int
    at_domain_bound: bool


def _family(family: str) -> None:
    if family not in FAMILIES:
        raise ValueError(f'family must be one of {FAMILIES}, found {family!r}')


def _data(y, x, w=None):
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or x.shape[0] != len(y):
        raise ValueError(f'need y of shape (n,) and X of shape (n, p), found {y.shape} and {x.shape}')
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        raise ValueError('y and X must be finite')
    if w is None:
        return y, x
    w = np.asarray(w, dtype=float)
    if w.shape != (len(y), len(y)):
        raise ValueError(f'W must have shape {(len(y), len(y))}, found {w.shape}')
    return y, x, w


def spatial_filter(theta: float, w) -> np.ndarray:
    """M(theta) = I - theta W."""
    w = np.asarray(w, dtype=float)
    return np.eye(len(w)) - float(theta) * w


def concentrated_loglik(family: str, theta: float, y, x, w) -> Evaluation:
    """lc(theta) with beta-hat(theta) and sigma2-hat(theta); raises on every pre-specified failure condition."""
    _family(family)
    y, x, w = _data(y, x, w)
    n, p = x.shape
    m = spatial_filter(theta, w)
    sign, logdet = np.linalg.slogdet(m)
    if sign != 1.0:
        raise SpatialFitFailure('slogdet-sign', f'sign {sign} at theta={theta!r}')
    if not math.isfinite(logdet):
        raise SpatialFitFailure('logdet-non-finite', f'logdet {logdet} at theta={theta!r}')
    y_t = m @ y
    x_t = m @ x if family == 'sem' else x
    beta, _, rank, _ = np.linalg.lstsq(x_t, y_t, rcond=None)
    if rank != p:
        raise SpatialFitFailure('rank-deficient', f'lstsq rank {rank} != p = {p} at theta={theta!r}')
    e = y_t - x_t @ beta
    sigma2 = float(e @ e) / n
    if not (math.isfinite(sigma2) and sigma2 > 0.0):
        raise SpatialFitFailure('sigma2-invalid', f'sigma2 {sigma2} at theta={theta!r}')
    loglik = -(n / 2) * (math.log(2.0 * math.pi) + 1.0) - (n / 2) * math.log(sigma2) + float(logdet)
    if not math.isfinite(loglik):
        raise SpatialFitFailure('loglik-non-finite', f'loglik {loglik} at theta={theta!r}')
    return Evaluation(loglik, beta, sigma2)


def theta_grid() -> np.ndarray:
    return np.linspace(-DOMAIN_BOUND, DOMAIN_BOUND, GRID_POINTS)


def grid_local_maxima(values) -> int:
    """Strict local maxima of a sequence; an endpoint is compared with its one neighbour only."""
    v = np.asarray(values, dtype=float)
    last = len(v) - 1
    return sum(int((i == 0 or v[i] > v[i - 1]) and (i == last or v[i] > v[i + 1])) for i in range(len(v)))


def maximize(objective: Callable[[float], float]) -> Maximum:
    """The fixed grid + golden-section maximizer over [-0.99, 0.99]; flags a constrained boundary estimate."""
    points, values = [], []

    def evaluate(theta: float) -> float:
        value = float(objective(theta))
        if not math.isfinite(value):
            raise SpatialFitFailure('objective-non-finite', f'{value} at theta={theta!r}')
        points.append(theta)
        values.append(value)
        return value

    grid = theta_grid()
    grid_values = np.array([evaluate(float(g)) for g in grid])
    k = int(np.argmax(grid_values))
    a, b = float(grid[max(k - 1, 0)]), float(grid[min(k + 1, GRID_POINTS - 1)])
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
    best = 0
    for i, value in enumerate(values):
        if value > values[best]:
            best = i
    theta = points[best]
    grid_values.setflags(write=False)
    return Maximum(theta, values[best], grid_values, len(values), bool(abs(theta) > DOMAIN_BOUND - INTERIOR_MARGIN))


@dataclass(frozen=True, eq=False)
class SpatialFit:
    """One immutable M3 fit. The log-likelihood is optimization evidence only, never a criterion."""

    family: str
    theta: float
    loglik: float
    beta: np.ndarray
    sigma2: float
    n: int
    p: int
    columns: tuple[str, ...] | None
    n_grid_local_maxima: int
    grid_loglik: np.ndarray
    n_evaluations: int
    at_domain_bound: bool = False
    version: str = ESTIMATOR_VERSION

    def __post_init__(self):
        _family(self.family)
        for name in ('beta', 'grid_loglik'):
            array = np.array(getattr(self, name), dtype=float)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.beta.shape != (self.p,):
            raise ValueError(f'beta has shape {self.beta.shape}, expected ({self.p},)')
        if self.columns is not None:
            object.__setattr__(self, 'columns', tuple(str(c) for c in self.columns))
            if len(self.columns) != self.p:
                raise ValueError(f'{len(self.columns)} column names for p = {self.p}')


def fit_spatial(family: str, y, x, w, columns: Sequence[str] | None = None) -> SpatialFit:
    """Gaussian ML fit of SEM or SAR on one training graph; raises SpatialFitFailure."""
    _family(family)
    y, x, w = _data(y, x, w)
    validate_graph(w)
    trace = maximize(lambda theta: concentrated_loglik(family, theta, y, x, w).loglik)
    final = concentrated_loglik(family, trace.theta, y, x, w)
    if not np.isfinite(final.beta).all():
        raise SpatialFitFailure('beta-non-finite', f'at theta-hat {trace.theta!r}')
    return SpatialFit(family=family, theta=trace.theta, loglik=final.loglik, beta=final.beta,
                      sigma2=final.sigma2, n=len(y), p=x.shape[1], columns=columns,
                      n_grid_local_maxima=grid_local_maxima(trace.grid_values), grid_loglik=trace.grid_values,
                      n_evaluations=trace.n_evaluations, at_domain_bound=trace.at_domain_bound)


def _match(fit: SpatialFit, y: np.ndarray, x: np.ndarray) -> None:
    if len(y) != fit.n or x.shape[1] != fit.p:
        raise ValueError(f'rows ({len(y)}, {x.shape[1]}) do not match the fit ({fit.n}, {fit.p})')


# --- fitted values, residuals, innovations --------------------------------------------

@dataclass(frozen=True, eq=False)
class FittedQuantities:
    trend: np.ndarray
    structural_residual: np.ndarray
    one_step: np.ndarray
    innovation: np.ndarray


def fitted_quantities(fit: SpatialFit, y, x, w) -> FittedQuantities:
    """Trend, structural residual, one-step fitted value and innovation, with both identity checks."""
    y, x, w = _data(y, x, w)
    _match(fit, y, x)
    m = spatial_filter(fit.theta, w)
    mean = x @ fit.beta
    if fit.family == 'sem':
        trend = mean
        residual = y - trend
        one_step = mean + fit.theta * (w @ residual)
        other = m @ residual
    else:
        trend = np.linalg.solve(m, mean)
        if not np.isfinite(trend).all():
            raise SpatialFitFailure('trend-non-finite', 'M(rho)^-1 X beta')
        residual = y - trend
        one_step = fit.theta * (w @ y) + mean
        other = m @ y - mean
    innovation = y - one_step
    if not all(np.isfinite(v).all() for v in (residual, one_step, innovation, other)):
        raise SpatialFitFailure('fitted-non-finite', 'a fitted quantity is non-finite')
    gap = float(np.max(np.abs(innovation - other)))
    if not gap <= INNOVATION_ATOL:
        raise SpatialFitFailure('innovation-identity', f'the two innovation expressions differ by {gap!r}')
    variance = float(np.mean(innovation * innovation))
    if not abs(variance - fit.sigma2) <= INNOVATION_VARIANCE_RTOL * fit.sigma2:
        raise SpatialFitFailure('innovation-variance', f'mean(eps^2) {variance!r} != sigma2 {fit.sigma2!r}')
    return FittedQuantities(trend, residual, one_step, innovation)


# --- held-out prediction ----------------------------------------------------------------

def predict_held_out(fit: SpatialFit, y_train, x_train, x_test, neighbours, weights) -> np.ndarray:
    """x_o'beta + theta sum_j w_oj s_j with s = u_T (SEM) or y_T (SAR); no unseen-level adjustment.

    ``neighbours`` are positions within the training rows (see :func:`local_positions`).
    """
    y, x = _data(y_train, x_train)
    _match(fit, y, x)
    xo = np.asarray(x_test, dtype=float)
    nb, wt = np.asarray(neighbours), np.asarray(weights, dtype=float)
    if xo.ndim != 2 or xo.shape[1] != fit.p or not np.isfinite(xo).all():
        raise ValueError(f'held-out design must be finite with {fit.p} columns, found shape {xo.shape}')
    if nb.shape != (len(xo), K) or wt.shape != nb.shape or (nb.size and not np.issubdtype(nb.dtype, np.integer)):
        raise ValueError(f'attachments must be integer positions and weights of shape {(len(xo), K)}')
    if nb.size and (nb.min() < 0 or nb.max() >= fit.n):
        raise ValueError('an attachment position is not a training row')
    if not (wt.sum(axis=1) == 1.0).all():
        raise SpatialFitFailure('graph-row-sum', 'an attachment row does not sum to exactly 1')
    source = y - x @ fit.beta if fit.family == 'sem' else y
    prediction = xo @ fit.beta + fit.theta * np.sum(wt * source[nb], axis=1)
    if not np.isfinite(prediction).all():
        raise SpatialFitFailure('prediction-non-finite', 'a held-out prediction is non-finite')
    return prediction


def _mean_effects(fit: SpatialFit, levels: Mapping[str, Sequence[str]]) -> list[tuple[str, set, float]]:
    if fit.columns is None:
        raise ValueError('the unseen-level adjustment needs the fit to carry column names')
    names = list(fit.columns)
    out = []
    for c, known in levels.items():
        known = [str(v) for v in known]
        effects = np.array([0.0] + [fit.beta[names.index(f'{c}={level}')] for level in known[1:]])
        out.append((c, set(known), effects.mean()))
    return out


def apply_unseen_level_adjustment(prediction, fit: SpatialFit, levels: Mapping[str, Sequence[str]],
                                  test_labels: Mapping[str, Sequence]) -> np.ndarray:
    """Add the mean of {0} U {beta of every non-reference training level} where a level is unseen.

    Mirrors ``cv.V1Design.predict`` (``unseen='mean_effect'``): ``levels`` maps each categorical feature,
    in encoder order, to its training levels (reference first); additions happen in that order.
    """
    yhat = np.asarray(prediction, dtype=float)
    for c, known, mean_effect in _mean_effects(fit, levels):
        labels = [str(v) for v in test_labels[c]]
        if len(labels) != len(yhat):
            raise ValueError(f'{len(labels)} labels of {c!r} for {len(yhat)} predictions')
        unseen = np.array([label not in known for label in labels], dtype=bool)
        yhat = yhat + unseen * mean_effect
    return yhat


def unseen_level_adjustment(fit: SpatialFit, levels: Mapping[str, Sequence[str]],
                            test_labels: Mapping[str, Sequence], n_test: int) -> np.ndarray:
    """a_o alone: the adjustment applied to zeros."""
    return apply_unseen_level_adjustment(np.zeros(n_test), fit, levels, test_labels)


# --- model-based parameter uncertainty --------------------------------------------------

@dataclass(frozen=True)
class DependenceInterval:
    se: float | None
    lower: float | None
    upper: float | None
    reason: str | None


def expected_information(fit: SpatialFit, x, w) -> np.ndarray:
    """(p + 2) x (p + 2) expected information at (beta-hat, sigma2-hat, theta-hat), ordered (beta, s2, theta)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    n, p = x.shape
    if p != fit.p or w.shape != (n, n) or n != fit.n:
        raise ValueError('X and W do not match the fit')
    s2 = fit.sigma2
    m = spatial_filter(fit.theta, w)
    wm = w @ np.linalg.inv(m)
    info = np.zeros((p + 2, p + 2))
    traces = np.trace(wm @ wm) + np.trace(wm.T @ wm)
    if fit.family == 'sem':
        bx = m @ x
        info[:p, :p] = bx.T @ bx / s2
        info[p + 1, p + 1] = traces
    else:
        lagged_mean = wm @ (x @ fit.beta)
        info[:p, :p] = x.T @ x / s2
        info[:p, p + 1] = info[p + 1, :p] = x.T @ lagged_mean / s2
        info[p + 1, p + 1] = traces + float(lagged_mean @ lagged_mean) / s2
    info[p, p] = n / (2.0 * s2 * s2)
    info[p, p + 1] = info[p + 1, p] = np.trace(wm) / s2
    return info


def dependence_interval(fit: SpatialFit, x, w) -> DependenceInterval:
    """se(theta-hat) and the Wald interval theta-hat +- 1.959963984540054 se, or not computable."""
    if fit.at_domain_bound:
        return DependenceInterval(None, None, None, f'{NOT_COMPUTABLE}: boundary estimate, asymptotic interval not applicable')
    info = expected_information(fit, x, w)
    try:
        variance = float(np.linalg.inv(info)[-1, -1])
    except np.linalg.LinAlgError as error:
        return DependenceInterval(None, None, None, f'{NOT_COMPUTABLE}: information matrix inversion failed ({error})')
    if not (math.isfinite(variance) and variance > 0.0):
        return DependenceInterval(None, None, None, f'{NOT_COMPUTABLE}: variance {variance!r}')
    se = math.sqrt(variance)
    return DependenceInterval(se, fit.theta - WALD_Z * se, fit.theta + WALD_Z * se, None)


# --- accounting ------------------------------------------------------------------------

@dataclass(frozen=True)
class Accounting:
    tss: float
    rss_static: float
    rss_innovation: float
    a_static: float
    a_dependence: float
    a_innovation: float
    absorbed_fraction_of_static_residual: float
    one_step_r2: float
    trend_r2: float | None


def static_rss(y, x) -> float:
    """RSS_S = sum (y - X beta-OLS)^2 with ``lstsq(rcond=None)``."""
    y, x = _data(y, x)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coef
    return float(np.sum(residual ** 2))


def accounting(y, rss_static: float, innovations, trend=None,
               static_shares: Mapping[str, float] | None = None) -> Accounting:
    """The three-part in-sample accounting; identity (and S's share sum, if given) checked to 1e-9."""
    y = np.asarray(y, dtype=float)
    eps = np.asarray(innovations, dtype=float)
    if eps.shape != y.shape:
        raise ValueError('innovations must align with y')
    tss = float(np.sum((y - y.mean()) ** 2))
    rss_eps = float(np.sum(eps ** 2))
    rss_static = float(rss_static)
    a_static = 1.0 - rss_static / tss
    a_dependence = (rss_static - rss_eps) / tss
    a_innovation = rss_eps / tss
    total = a_static + a_dependence + a_innovation
    if not abs(total - 1.0) <= ACCOUNTING_ATOL:
        raise IntegrityFailure(f'accounting parts sum to {total!r}, not 1 within {ACCOUNTING_ATOL}')
    if static_shares is not None:
        share_sum = sum(float(v) for v in static_shares.values())
        if not abs(share_sum - a_static) <= ACCOUNTING_ATOL:
            raise IntegrityFailure(f"S's LMG shares sum to {share_sum!r}, A_static is {a_static!r}")
    trend_r2 = None if trend is None else 1.0 - float(np.sum((y - np.asarray(trend, dtype=float)) ** 2)) / tss
    return Accounting(tss=tss, rss_static=rss_static, rss_innovation=rss_eps, a_static=a_static,
                      a_dependence=a_dependence, a_innovation=a_innovation,
                      absorbed_fraction_of_static_residual=(rss_static - rss_eps) / rss_static,
                      one_step_r2=1.0 - rss_eps / tss, trend_r2=trend_r2)


def shapley_weight(subset_size: int, n_groups: int) -> float:
    """LMG/Shapley weight s!(G - s - 1)!/G!, as ``src.decomposition``."""
    return math.factorial(subset_size) * math.factorial(n_groups - subset_size - 1) / math.factorial(n_groups)


@dataclass(frozen=True)
class FilteredShares:
    """Group shares of the filtered outcome's variance: a different denominator from S's shares."""

    family: str
    theta: float
    groups: tuple[str, ...]
    shares: dict[str, float]
    r2_full: float
    residual_share: float
    tss_filtered: float
    n_coalitions: int
    full_coalition_max_gap: float

    @property
    def composition(self) -> dict[str, float]:
        return {g: s / self.r2_full for g, s in self.shares.items()}


def filtered_group_shares(fit: SpatialFit, y, x, w, blocks: Mapping[str, np.ndarray]) -> FilteredShares:
    """LMG shares of y* = M(theta-hat) y with theta-hat fixed; ``blocks`` maps group -> columns, in order.

    ``x`` is the fit's full design (intercept included); the blocks, without intercept, must span the
    same columns, which the full-coalition check (1e-10 against the ML transformed fit) enforces.
    """
    y, x, w = _data(y, x, w)
    _match(fit, y, x)
    groups = tuple(blocks)
    if not groups:
        raise ValueError('at least one group is required')
    m = spatial_filter(fit.theta, w)
    y_star = m @ y
    tss = float(np.sum((y_star - y_star.mean()) ** 2))
    if not (math.isfinite(tss) and tss > 0.0):
        raise IntegrityFailure(f'the filtered outcome has total sum of squares {tss!r}')
    columns = {}
    for g in groups:
        block = np.asarray(blocks[g], dtype=float)
        if block.ndim != 2 or len(block) != len(y):
            raise ValueError(f'group {g!r} must be a 2-D block with {len(y)} rows')
        columns[g] = m @ block if fit.family == 'sem' else block
    intercept = np.ones((len(y), 1))
    cache: dict[frozenset[str], tuple[float, np.ndarray]] = {}

    def coalition_r2(subset: frozenset[str]) -> float:
        if subset not in cache:
            design = np.hstack([intercept, *(columns[g] for g in groups if g in subset)])
            coef, *_ = np.linalg.lstsq(design, y_star, rcond=None)
            fitted = design @ coef
            cache[subset] = (1.0 - float(np.sum((y_star - fitted) ** 2)) / tss, fitted)
        return cache[subset][0]

    shares = {g: 0.0 for g in groups}
    for g in groups:
        others = [h for h in groups if h != g]
        for size in range(len(others) + 1):
            weight = shapley_weight(size, len(groups))
            for combo in combinations(others, size):
                s = frozenset(combo)
                shares[g] += weight * (coalition_r2(s | {g}) - coalition_r2(s))
    full = frozenset(groups)
    r2_full = coalition_r2(full)
    ml_fitted = (m @ x) @ fit.beta if fit.family == 'sem' else x @ fit.beta
    gap = float(np.max(np.abs(cache[full][1] - ml_fitted)))
    if not gap <= COALITION_FIT_ATOL:
        raise IntegrityFailure(f'the full coalition misses the ML transformed fit by {gap!r}')
    residual = 1.0 - r2_full
    if not abs(sum(shares.values()) + residual - 1.0) <= SHARE_SUM_ATOL:
        raise IntegrityFailure(f'filtered shares sum to {sum(shares.values())!r}, R2* is {r2_full!r}')
    return FilteredShares(family=fit.family, theta=fit.theta, groups=groups,
                          shares={g: float(v) for g, v in shares.items()}, r2_full=float(r2_full),
                          residual_share=float(residual), tss_filtered=tss, n_coalitions=len(cache),
                          full_coalition_max_gap=gap)


def material_change(shares: Mapping[str, float]) -> bool:
    """Geography not strictly the largest named share, or the emissions share above 0.10."""
    geography = shares['geography']
    return any(v >= geography for g, v in shares.items() if g != 'geography') or shares['emissions'] > 0.10
