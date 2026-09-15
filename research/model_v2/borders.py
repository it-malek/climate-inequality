"""LEGACY 1° cell-centre distances: not a territorial clearance guarantee.

Current Model V2 CV uses ``territory.py``. This module is retained solely to
reproduce Session 1 diagnostic/candidate artifacts, which must not be overwritten.

A buffer on centroid distance leaves the neighbours of a large country in its
training fold (the centroid of Russia, Canada, the United States or Mexico is more
than 1500 km from every other centroid). The buffer used by the primary protocol
is therefore defined on the **minimum great-circle distance between any 1° land
cell of one country and any of the other** (0 km for a shared border at this
resolution is impossible; adjacent cells are ~111 km apart, so a shared border
shows as roughly 80–160 km). Computed once from the cached cell field and written
to ``outputs/country_border_distance_km.csv``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from research.model_v2 import cells
from research.model_v2.diagnostics import CORRELOGRAM_EDGES_KM, N_PERMUTATIONS, SEED, COUNTRY_TABLE_PATH
from research.model_v2.m0 import OUTPUT_DIR
from research.model_v2.spatial import EARTH_RADIUS_KM, morans_i

BORDER_DISTANCE_PATH = OUTPUT_DIR / "country_border_distance_km.csv"
BORDER_CORRELOGRAM_PATH = OUTPUT_DIR / "border_distance_correlogram.json"


def border_distance_matrix(iso_order: list[str]) -> pd.DataFrame:
    """Symmetric matrix of minimum cell-to-cell great-circle distances (km)."""
    lats, lons, _, iso3 = cells.load_cell_field()
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    phi, lam = np.radians(lat_grid), np.radians(lon_grid)
    xyz = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)
    points = {}
    for code in iso_order:
        m = iso3 == code
        if not m.any():
            raise ValueError(f"no 1° cell assigned to {code}")
        points[code] = xyz[m]
    trees = {code: cKDTree(p) for code, p in points.items()}
    n = len(iso_order)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = iso_order[i], iso_order[j]
            small, large = (a, b) if len(points[a]) <= len(points[b]) else (b, a)
            chord = trees[large].query(points[small])[0].min()
            km = 2 * EARTH_RADIUS_KM * np.arcsin(min(chord / 2, 1.0))
            out[i, j] = out[j, i] = km
    return pd.DataFrame(out, index=iso_order, columns=iso_order)


def load_or_build(table: pd.DataFrame) -> np.ndarray:
    """Border-distance matrix in the row order of ``table`` (built and cached on first use)."""
    order = table["iso3"].tolist()
    if BORDER_DISTANCE_PATH.exists():
        frame = pd.read_csv(BORDER_DISTANCE_PATH, index_col=0)
        if list(frame.index) == order:
            return frame.to_numpy()
    frame = border_distance_matrix(order)
    frame.to_csv(BORDER_DISTANCE_PATH, float_format="%.1f")
    return frame.to_numpy()


def border_correlogram(resid: np.ndarray, dist: np.ndarray) -> list[dict]:
    """Moran's I of the residual by ring of minimum inter-territory distance."""
    z = resid - resid.mean()
    rings = []
    edges = CORRELOGRAM_EDGES_KM
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        w = ((dist >= lo) & (dist < hi)).astype(float)
        np.fill_diagonal(w, 0.0)
        n_pairs = int(w.sum() / 2)
        if n_pairs == 0:
            continue
        r = morans_i(z, w, N_PERMUTATIONS, SEED)
        rings.append({"lo_km": lo, "hi_km": hi, "n_pairs": n_pairs, "morans_i": r.statistic, "p": r.p_value, "z": r.z_score})
    return rings


def main() -> None:
    table = pd.read_csv(COUNTRY_TABLE_PATH)
    dist = load_or_build(table)
    resid = table["residual"].to_numpy()
    rings = border_correlogram(resid, dist)
    BORDER_CORRELOGRAM_PATH.write_text(json.dumps({"rings": rings}, indent=1) + "\n", encoding="utf-8")
    off = dist[np.triu_indices(len(dist), 1)]
    print(f"border distances: {len(dist)} countries; min {off.min():.0f} km, median {np.median(off):.0f} km; pairs < 200 km: {(off < 200).sum()}")
    for r in rings:
        print(f"  [{r['lo_km']:>5},{r['hi_km']:>5}) km  pairs {r['n_pairs']:>5}  I {r['morans_i']:+.3f}  p {r['p']:.3f}")


if __name__ == "__main__":
    main()
