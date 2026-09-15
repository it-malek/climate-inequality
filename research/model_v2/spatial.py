"""Spatial weights and autocorrelation statistics for the residual diagnostics.

Dense implementations (n = 151) of the constructions the M0 diagnostics compare:
k-nearest-neighbour, distance-band, inverse-distance and raster-contiguity weights,
all row-standardised; global Moran's I and Geary's C with permutation p-values; and
local Moran's I (LISA) with conditional permutation p-values.

With row-standardised weights, ``S0 = n`` and Moran's I reduces to
``sum_i z_i * (W z)_i / sum_i z_i^2``; for kNN weights this is exactly the
statistic ``src.explain.morans_i`` computes, so the k = 8 station-centroid case
reproduces the V1 bundle value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances (km) between points."""
    phi = np.radians(np.asarray(lat, dtype=float))
    lam = np.radians(np.asarray(lon, dtype=float))
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = np.sin(dphi / 2) ** 2 + np.cos(phi[:, None]) * np.cos(phi[None, :]) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _row_standardise(w: np.ndarray) -> np.ndarray:
    rows = w.sum(axis=1, keepdims=True)
    return np.divide(w, rows, out=np.zeros_like(w, dtype=float), where=rows > 0)


def knn_weights(dist: np.ndarray, k: int) -> np.ndarray:
    """Row-standardised kNN weights; ties broken by distance then by index (deterministic)."""
    n = dist.shape[0]
    w = np.zeros((n, n))
    for i in range(n):
        order = np.lexsort((np.arange(n), dist[i]))
        order = order[order != i][:k]
        w[i, order] = 1.0
    return _row_standardise(w)


def distance_band_weights(dist: np.ndarray, cutoff_km: float) -> tuple[np.ndarray, int]:
    """Row-standardised binary weights within ``cutoff_km``; isolates get their nearest neighbour.

    Returns ``(W, n_isolates)`` so the caller can report how many units had no
    neighbour inside the band.
    """
    w = (dist <= cutoff_km).astype(float)
    np.fill_diagonal(w, 0.0)
    isolates = np.flatnonzero(w.sum(axis=1) == 0)
    for i in isolates:
        d = dist[i].copy()
        d[i] = np.inf
        w[i, int(np.argmin(d))] = 1.0
    return _row_standardise(w), int(len(isolates))


def inverse_distance_weights(dist: np.ndarray, power: float = 1.0, cutoff_km: float | None = None) -> np.ndarray:
    """Row-standardised inverse-distance weights, optionally truncated at ``cutoff_km``."""
    with np.errstate(divide="ignore"):
        w = 1.0 / np.power(dist, power)
    np.fill_diagonal(w, 0.0)
    if cutoff_km is not None:
        w[dist > cutoff_km] = 0.0
    w[~np.isfinite(w)] = 0.0
    return _row_standardise(w)


def contiguity_weights(
    ids: np.ndarray, pairs: pd.DataFrame, dist: np.ndarray, fallback_k: int = 1
) -> tuple[np.ndarray, int]:
    """Row-standardised queen contiguity from ``pairs`` (iso3_a, iso3_b); islands fall back to kNN.

    Returns ``(W, n_fallback)``.
    """
    index = {code: i for i, code in enumerate(ids)}
    n = len(ids)
    w = np.zeros((n, n))
    for a, b in zip(pairs["iso3_a"], pairs["iso3_b"], strict=True):
        if a in index and b in index:
            w[index[a], index[b]] = 1.0
            w[index[b], index[a]] = 1.0
    isolates = np.flatnonzero(w.sum(axis=1) == 0)
    for i in isolates:
        order = np.lexsort((np.arange(n), dist[i]))
        order = order[order != i][:fallback_k]
        w[i, order] = 1.0
    return _row_standardise(w), int(len(isolates))


@dataclass(frozen=True)
class MoranResult:
    statistic: float
    expected: float
    p_value: float
    z_score: float
    n_permutations: int


def morans_i(values: np.ndarray, w: np.ndarray, n_permutations: int = 999, seed: int = 0) -> MoranResult:
    """Global Moran's I with a two-sided permutation p-value (V1 convention)."""
    z = np.asarray(values, dtype=float) - np.mean(values)
    n = len(z)
    s0 = w.sum()
    denom = float(np.sum(z**2))

    def stat(zz: np.ndarray) -> float:
        return float((n / s0) * (zz @ w @ zz) / denom)

    observed = stat(z)
    rng = np.random.default_rng(seed)
    perms = np.array([stat(rng.permutation(z)) for _ in range(n_permutations)])
    p = (np.sum(np.abs(perms) >= abs(observed)) + 1) / (n_permutations + 1)
    zscore = (observed - perms.mean()) / perms.std(ddof=1) if perms.std(ddof=1) > 0 else np.nan
    return MoranResult(observed, -1.0 / (n - 1), float(p), float(zscore), n_permutations)


def gearys_c(values: np.ndarray, w: np.ndarray, n_permutations: int = 999, seed: int = 0) -> MoranResult:
    """Geary's C (1 under no autocorrelation; < 1 positive) with a permutation p-value."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    s0 = w.sum()
    denom = 2.0 * s0 * float(np.sum((x - x.mean()) ** 2))

    def stat(xx: np.ndarray) -> float:
        diff = xx[:, None] - xx[None, :]
        return float((n - 1) * np.sum(w * diff**2) / denom)

    observed = stat(x)
    rng = np.random.default_rng(seed)
    perms = np.array([stat(rng.permutation(x)) for _ in range(n_permutations)])
    p = (np.sum(np.abs(perms - 1.0) >= abs(observed - 1.0)) + 1) / (n_permutations + 1)
    zscore = (observed - perms.mean()) / perms.std(ddof=1) if perms.std(ddof=1) > 0 else np.nan
    return MoranResult(observed, 1.0, float(p), float(zscore), n_permutations)


def local_morans(values: np.ndarray, w: np.ndarray, n_permutations: int = 999, seed: int = 0) -> pd.DataFrame:
    """Local Moran's I per unit with conditional-permutation p-values and quadrant labels.

    ``I_i = z_i * sum_j w_ij z_j / (sum z^2 / n)``. The p-value permutes the other
    units' values while holding unit i fixed (Anselin 1995). Quadrants: HH/LL are
    clusters of like values, HL/LH spatial outliers; ``label`` is the quadrant when
    ``p < 0.05`` and ``"ns"`` otherwise.
    """
    z = np.asarray(values, dtype=float) - np.mean(values)
    n = len(z)
    m2 = float(np.sum(z**2)) / n
    lag = w @ z
    local = z * lag / m2
    rng = np.random.default_rng(seed)
    p = np.empty(n)
    for i in range(n):
        others = np.delete(z, i)
        wi = np.delete(w[i], i)
        nz = np.flatnonzero(wi)
        if len(nz) == 0:
            p[i] = np.nan
            continue
        sims = np.empty(n_permutations)
        for b in range(n_permutations):
            draw = rng.choice(others, size=len(nz), replace=False)
            sims[b] = z[i] * float(wi[nz] @ draw) / m2
        p[i] = (np.sum(np.abs(sims) >= abs(local[i])) + 1) / (n_permutations + 1)
    quadrant = np.where(z >= 0, np.where(lag >= 0, "HH", "HL"), np.where(lag >= 0, "LH", "LL"))
    label = np.where(p < 0.05, quadrant, "ns")
    return pd.DataFrame({"z": z, "lag": lag, "local_i": local, "p_value": p, "quadrant": quadrant, "label": label})
