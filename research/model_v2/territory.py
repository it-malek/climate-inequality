"""Conservative WGS84 clearance for the frozen GPW/ISO analytical units.

Geometry is the union of CLOSED 0.25-degree footprints of EVERY assigned GPW
national-identifier cell, not centres alone. A lower bound, never an asserted
exact shoreline distance, drives exclusion. See TERRITORIAL_CV_CORRECTION.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.spatial import cKDTree

from research.model_v2.geometry import GPW_CELL_DEG, load_national_grid
from research.model_v2.m0 import OUTPUT_DIR

GEOD = Geod(ellps="WGS84")
A_KM = GEOD.a / 1000
B_KM = GEOD.b / 1000
MAX_CURVATURE_KM = A_KM**2 / B_KM
NUMERIC_ALLOWANCE_KM = 1e-6  # 1 mm, subtracted from lower bounds
LOWER_PATH = OUTPUT_DIR / "territory_distance_lower_km.csv"
UPPER_PATH = OUTPUT_DIR / "territory_distance_upper_km.csv"
SNAPSHOT_PATH = OUTPUT_DIR / "territory_gpw_footprints.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def radius_bound_km(cell_deg: float = GPW_CELL_DEG) -> float:
    """Upper bound on surface distance from centre to ANY footprint point.

    WGS84 ds² = M² dphi² + N² cos²(phi) dlambda²; M,N <= a²/b.
    A straight path in unwrapped latitude/longitude to any cell point therefore
    has length <= (a²/b)*sqrt(2)*radians(cell_deg/2). Geodesic <= path length.
    """
    if not np.isfinite(cell_deg) or not 0 <= cell_deg <= 1:
        raise ValueError("cell width must be in [0, 1] degrees")
    return float(MAX_CURVATURE_KM * np.sqrt(2) * np.radians(cell_deg / 2))


def ecef(points: np.ndarray) -> np.ndarray:
    """Latitude/longitude centres on the WGS84 ellipsoid to ECEF km."""
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
            or not np.isfinite(points).all() or (np.abs(points[:, 0]) > 90).any()):
        raise ValueError("nonempty finite (latitude, longitude) geometry required")
    lat, lon = np.radians(points).T
    e2 = 1 - (B_KM / A_KM)**2
    normal = A_KM / np.sqrt(1 - e2 * np.sin(lat)**2)
    return np.column_stack([
        normal * np.cos(lat) * np.cos(lon),
        normal * np.cos(lat) * np.sin(lon),
        normal * (1 - e2) * np.sin(lat),
    ])


def _bounds(a, b, xyz_a, tree_b, cell_deg):
    distances, indices = tree_b.query(xyz_a, eps=0, workers=1)
    i = int(np.argmin(distances))
    p, q = a[i], b[int(indices[i])]
    chord = float(distances[i])
    # For every p in footprint A and q in B, triangle inequality gives
    # |p-q| >= min centre chord - 2*r. Surface geodesic >= |p-q|.
    lower = max(0.0, chord - 2 * radius_bound_km(cell_deg) - NUMERIC_ALLOWANCE_KM)
    _, _, meters = GEOD.inv(p[1], p[0], q[1], q[0])
    upper = float(meters / 1000)
    lon_delta = abs((p[1] - q[1] + 180) % 360 - 180)
    shared = (abs(p[0] - q[0]) <= cell_deg + 1e-12
              and lon_delta <= cell_deg + 1e-12)
    pole = (p[0] * q[0] > 0 and
            min(abs(p[0]), abs(q[0])) + cell_deg / 2 >= 90)
    if shared or pole:
        return 0.0, 0.0
    return lower, upper


def pair_bounds(a: np.ndarray, b: np.ndarray, cell_deg=GPW_CELL_DEG):
    """Certified lower and witness upper bound for two footprint unions, km.

    A returned zero lower bound alone does not prove contact. Upper=0 does.
    The lower bound deliberately excludes geometrically uncertain threshold
    pairs; it never falsely certifies >500 km clearance for represented land.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    xa, xb = ecef(a), ecef(b)
    return _bounds(a, b, xa, cKDTree(xb), cell_deg)


def distance_bounds(units: dict[str, np.ndarray], cell_deg=GPW_CELL_DEG):
    """All-pair bounds in insertion order, deterministic and symmetric."""
    codes = list(units)
    xyz = {c: ecef(p) for c, p in units.items()}
    trees = {c: cKDTree(x) for c, x in xyz.items()}
    lower = np.zeros((len(codes), len(codes)))
    upper = lower.copy()
    for i, a in enumerate(codes):
        for j in range(i + 1, len(codes)):
            b = codes[j]
            small, large = (a, b) if len(xyz[a]) <= len(xyz[b]) else (b, a)
            lo, hi = _bounds(units[small], units[large], xyz[small], trees[large], cell_deg)
            lower[i, j] = lower[j, i] = lo
            upper[i, j] = upper[j, i] = hi
    return (pd.DataFrame(lower, index=codes, columns=codes),
            pd.DataFrame(upper, index=codes, columns=codes))


def gpw_units(codes: list[str]) -> dict[str, np.ndarray]:
    iso, lats, lons = load_national_grid()
    if not (np.allclose(np.abs(np.diff(lats)), GPW_CELL_DEG)
            and np.allclose(np.diff(lons), GPW_CELL_DEG)):
        raise ValueError("unexpected GPW grid resolution")
    out = {}
    for code in codes:
        r, c = np.where(iso == code)
        out[code] = np.column_stack([lats[r], lons[c]])
        ecef(out[code])  # fail closed for missing units
    return out


def load_bounds(codes: list[str], path: Path = LOWER_PATH) -> np.ndarray:
    frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
    if list(frame.index) != codes or list(frame.columns) != codes:
        raise ValueError("territory bounds must match both country axes exactly")
    d = frame.to_numpy()
    if (not np.isfinite(d).all() or (d < 0).any()
            or not np.array_equal(d, d.T) or np.any(np.diag(d) != 0)):
        raise ValueError("invalid territory bound matrix")
    return d
