"""M2 latitude functional form: the natural cubic basis and the H1 hemisphere interaction.

Versioned reference implementation of the M2 latitude design. It is predictor-only: it builds the two
added natural cubic columns and the southern-hemisphere slope column from ``abs_latitude`` and
``hemisphere`` alone, and it never sees an outcome, a fitted value or a model.

Basis (Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*, 2nd ed., eqs. 5.4-5.5).
For knots xi_1 < xi_2 < xi_3 < xi_4,

    d_k(a) = [(a - xi_k)_+^3 - (a - xi_4)_+^3] / (xi_4 - xi_k),     k = 1, 2, 3
    N_1(a) = d_1(a) - d_3(a),     N_2(a) = d_2(a) - d_3(a).

Together with the intercept and the linear ``abs_latitude`` column that M0* already carries,
{1, a, N_1, N_2} spans exactly the four-dimensional space of natural cubic splines on those knots:
cubic between knots, twice continuously differentiable, and linear below xi_1 and above xi_4. That is
three nonconstant latitude dimensions ("three df", intercept excluded), of which two are new. Cubes are
computed as ``t * t * t`` (IEEE basic operations), never through a platform ``pow`` kernel.

State. xi_1 and xi_4 are the training minimum and maximum; xi_2 and xi_3 are
``numpy.quantile(train, [1/3, 2/3], method='linear')``. The state is computed from training rows only,
is immutable and self-validating, and is applied unchanged to every row it transforms.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

BASIS_VERSION = 'm2-latitude-basis-v1'
QUANTILE_METHOD = 'linear'
PROBABILITIES = (1 / 3, 2 / 3)
PROBABILITY_LABELS = ['1/3', '2/3']
BOUNDARY_KNOTS = 'training minimum and maximum'
TAILS = 'linear beyond the boundary knots'
# Minimum separation, in degrees, between consecutive knots. Anything closer is a repeated knot.
KNOT_GAP_ATOL = 1e-6
MIN_TRAINING_VALUES = 4
# H1 is coded I(SH) * abs_latitude with a fixed centre of 0 degrees: no fold, mean or outcome centring.
INTERACTION_CENTRE_DEGREES = 0.0
HEMISPHERE_LEVELS = ('N', 'S')
SOUTHERN = 'S'
ADDED_COLUMNS = ('abs_latitude_ns1', 'abs_latitude_ns2', 'hemisphere=S:abs_latitude')
ADDED_GROUP = 'geography'


class DegenerateLatitudeBasis(ValueError):
    """The basis cannot be built on these values. There is no fallback."""


@dataclass(frozen=True)
class LatitudeState:
    """One fit's immutable latitude transformation: four knots and the training size behind them."""

    knots: tuple[float, float, float, float]
    n_train: int
    version: str = BASIS_VERSION

    def __post_init__(self):
        if self.version != BASIS_VERSION:
            raise DegenerateLatitudeBasis(f'latitude state version {self.version!r} is not {BASIS_VERSION!r}')
        if len(self.knots) != 4 or not all(isinstance(k, float) and math.isfinite(k) and k >= 0
                                           for k in self.knots):
            raise DegenerateLatitudeBasis(f'a state needs four finite, non-negative float knots: {self.knots}')
        if not all(b - a > KNOT_GAP_ATOL for a, b in zip(self.knots, self.knots[1:])):
            raise DegenerateLatitudeBasis(f'knots are not strictly increasing by more than {KNOT_GAP_ATOL} '
                                          f'degrees: {self.knots}')
        if not isinstance(self.n_train, int) or self.n_train < MIN_TRAINING_VALUES:
            raise DegenerateLatitudeBasis(f'a state needs at least {MIN_TRAINING_VALUES} training values')

    def to_json(self) -> str:
        record = {'version': self.version, 'n_train': self.n_train, 'knots': list(self.knots),
                  'knots_hex': [k.hex() for k in self.knots], 'quantile_method': QUANTILE_METHOD,
                  'probabilities': PROBABILITY_LABELS, 'boundary_knots': BOUNDARY_KNOTS, 'tails': TAILS}
        return json.dumps(record, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> LatitudeState:
        """Exact state from its serialization; the hex knots are authoritative."""
        record = json.loads(text)
        frozen = {'version': BASIS_VERSION, 'quantile_method': QUANTILE_METHOD,
                  'probabilities': PROBABILITY_LABELS, 'boundary_knots': BOUNDARY_KNOTS, 'tails': TAILS}
        for key, value in frozen.items():
            if record.get(key) != value:
                raise ValueError(f'serialized latitude state has {key}={record.get(key)!r}, not {value!r}')
        knots = tuple(float.fromhex(h) for h in record['knots_hex'])
        if list(knots) != [float(k) for k in record['knots']]:
            raise ValueError('decimal knots disagree with their exact hex serialization')
        return cls(knots=knots, n_train=int(record['n_train']))


def fit_state(abs_latitude_train) -> LatitudeState:
    """Knots from training latitudes only; raises on anything the fixed design cannot use."""
    a = np.asarray(abs_latitude_train, dtype=float)
    if a.ndim != 1 or len(a) < MIN_TRAINING_VALUES:
        raise DegenerateLatitudeBasis(f'need a 1-D array of at least {MIN_TRAINING_VALUES} training '
                                      f'latitudes, found shape {a.shape}')
    if not np.isfinite(a).all() or (a < 0).any():
        raise DegenerateLatitudeBasis('training latitudes must be finite, non-negative degrees')
    lower, upper = np.quantile(a, list(PROBABILITIES), method=QUANTILE_METHOD)
    return LatitudeState(knots=(float(a.min()), float(lower), float(upper), float(a.max())), n_train=len(a))


def _cube(t):
    return t * t * t


def nonlinear_columns(state: LatitudeState, abs_latitude) -> np.ndarray:
    """The two added natural cubic columns (n, 2) for any rows, under a fixed state."""
    x = np.asarray(abs_latitude, dtype=float)
    last = state.knots[-1]

    def d(k):
        xi = state.knots[k]
        return (_cube(np.maximum(x - xi, 0.0)) - _cube(np.maximum(x - last, 0.0))) / (last - xi)
    return np.column_stack([d(0) - d(2), d(1) - d(2)])


def sh_interaction(hemisphere, abs_latitude) -> np.ndarray:
    """I(hemisphere == 'S') * (abs_latitude - 0): one southern slope difference."""
    labels = np.asarray(hemisphere, dtype=object)
    if not all(isinstance(v, str) and v in HEMISPHERE_LEVELS for v in labels):
        raise ValueError(f'hemisphere labels must be one of {HEMISPHERE_LEVELS}')
    a = np.asarray(abs_latitude, dtype=float)
    return (labels == SOUTHERN).astype(float) * (a - INTERACTION_CENTRE_DEGREES)


def added_block(state: LatitudeState, abs_latitude, hemisphere) -> np.ndarray:
    """The three M2 columns in ``ADDED_COLUMNS`` order, all belonging to geography."""
    return np.column_stack([nonlinear_columns(state, abs_latitude), sh_interaction(hemisphere, abs_latitude)])


def extrapolation(state: LatitudeState, abs_latitude) -> dict:
    """How many rows fall outside the state's boundary knots, and the rows' latitude range."""
    x = np.asarray(abs_latitude, dtype=float)
    return {'below': int((x < state.knots[0]).sum()), 'above': int((x > state.knots[-1]).sum()),
            'min': float(x.min()), 'max': float(x.max())}
