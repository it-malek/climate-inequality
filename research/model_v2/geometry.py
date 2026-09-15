"""Country geometry from the GPW v4 national-identifier grid, ETOPO 2022 and Köppen-Geiger.

Descriptive quantities used by the M0 residual diagnostics and by the spatial
cross-validation design: land area, area-weighted centroid, latitude spread,
coastal-cell fraction, raster contiguity between countries, within-country
elevation statistics and area-weighted Köppen major-group shares. Nothing here is
a model input; the V1 design is untouched.

All grids are the ones V1 already relies on (``src.area_weighting`` uses the same
GPW band for country assignment; ``src.explain`` samples the same ETOPO and Köppen
files at station coordinates). Resolution: GPW 15 arc-minutes, ETOPO sampled every
fifth 60 arc-second row/column (5 arc-minutes), Köppen 0.5 degrees.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.area_weighting import GPW_NATID_BAND, load_national_id_lookup
from src.explain import ETOPO_PATH, ETOPO_VAR, KOPPEN_GROUPS, KOPPEN_PATH, KOPPEN_VAR
from src.population import GPW_ENGINE, GPW_FILL_FLOOR, GPW_PATH, GPW_POP_VAR, GPW_RASTER_DIM

EARTH_RADIUS_KM = 6371.0088
GPW_CELL_DEG = 0.25
ETOPO_STRIDE = 5  # every fifth 60" sample -> 5 arc-minutes
KOPPEN_CELL_DEG = 0.5
KOPPEN_CLASSES = ("A", "B", "C", "D", "E")

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
GEOMETRY_PATH = OUTPUT_DIR / "country_geometry.csv"
ADJACENCY_PATH = OUTPUT_DIR / "country_adjacency.csv"


def load_national_grid(path: Path = GPW_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ISO3 per 15' cell (``""`` where no country), with the grid's lat/lon centres."""
    ds = xr.open_dataset(path, engine=GPW_ENGINE)
    try:
        band = ds[GPW_POP_VAR].isel({GPW_RASTER_DIM: GPW_NATID_BAND - 1}).to_numpy()
        lats = ds["latitude"].to_numpy().astype(float)
        lons = ds["longitude"].to_numpy().astype(float)
    finally:
        ds.close()
    lookup = load_national_id_lookup()
    codes = np.round(band).astype(np.int64)
    valid = np.isfinite(band) & (band >= GPW_FILL_FLOOR)
    iso = np.full(band.shape, "", dtype=object)
    for code in np.unique(codes[valid]):
        name = lookup.get(int(code))
        if name is not None:
            iso[valid & (codes == code)] = name
    return iso, lats, lons


def cell_area_km2(lat_centres: np.ndarray, cell_deg: float = GPW_CELL_DEG) -> np.ndarray:
    """Exact spherical area of a ``cell_deg`` square cell centred at each latitude."""
    half = np.radians(cell_deg / 2.0)
    phi = np.radians(lat_centres)
    return (EARTH_RADIUS_KM**2) * np.radians(cell_deg) * (np.sin(phi + half) - np.sin(phi - half))


def _grid_index(lat: np.ndarray, lon: np.ndarray, cell_deg: float, n_lat: int, n_lon: int):
    """Row/column of the descending-latitude, ascending-longitude grid containing a point."""
    row = np.clip(np.floor((90.0 - lat) / cell_deg).astype(int), 0, n_lat - 1)
    col = np.clip(np.floor((lon + 180.0) / cell_deg).astype(int), 0, n_lon - 1)
    return row, col


def country_geometry(iso: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
    """Area, centroid, latitude spread and coastal fraction per ISO3 from the 15' grid."""
    n_lat, n_lon = iso.shape
    area_row = cell_area_km2(lats)
    area = np.broadcast_to(area_row[:, None], iso.shape)
    lat_grid = np.broadcast_to(lats[:, None], iso.shape)
    lon_grid = np.broadcast_to(lons[None, :], iso.shape)
    land = iso != ""

    # A land cell is coastal when any 4-neighbour is not land (longitude wraps).
    non_land = ~land
    coastal = np.zeros(iso.shape, dtype=bool)
    coastal[1:, :] |= non_land[:-1, :]
    coastal[:-1, :] |= non_land[1:, :]
    coastal[:, 1:] |= non_land[:, :-1]
    coastal[:, :-1] |= non_land[:, 1:]
    coastal[:, 0] |= non_land[:, -1]
    coastal[:, -1] |= non_land[:, 0]
    coastal[0, :] = True
    coastal[-1, :] = True
    coastal &= land

    phi = np.radians(lat_grid)
    lam = np.radians(lon_grid)
    xyz = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)

    rows = []
    for code in sorted(set(iso[land].tolist())):
        m = iso == code
        w = area[m]
        total = float(w.sum())
        vec = (xyz[m] * w[:, None]).sum(axis=0) / total
        norm = np.linalg.norm(vec)
        c_lat = float(np.degrees(np.arcsin(vec[2] / norm)))
        c_lon = float(np.degrees(np.arctan2(vec[1], vec[0])))
        lat_cells = lat_grid[m]
        lat_mean = float(np.dot(w, lat_cells) / total)
        lat_sd = float(np.sqrt(np.dot(w, (lat_cells - lat_mean) ** 2) / total))
        rows.append(
            {
                "iso3": code,
                "n_cells_15min": int(m.sum()),
                "land_area_km2": total,
                "centroid_lon": c_lon,
                "centroid_lat": c_lat,
                "abs_centroid_lat": abs(c_lat),
                "lat_min": float(lat_cells.min()),
                "lat_max": float(lat_cells.max()),
                "lat_sd_area_weighted": lat_sd,
                "coastal_cell_fraction": float(np.dot(w, coastal[m]) / total),
            }
        )
    return pd.DataFrame(rows)


def raster_adjacency(iso: np.ndarray) -> pd.DataFrame:
    """Queen contiguity between countries from shared 15' cell edges/corners.

    Returns one row per unordered pair with the number of shared cell edges (rook)
    and corners-only contacts (queen minus rook); longitude wraps at the dateline.
    """
    edges: dict[tuple[str, str], int] = defaultdict(int)
    corners: dict[tuple[str, str], int] = defaultdict(int)

    def _count(a: np.ndarray, b: np.ndarray, store: dict) -> None:
        both = (a != "") & (b != "") & (a != b)
        for x, y in zip(a[both], b[both], strict=True):
            store[tuple(sorted((x, y)))] += 1

    _count(iso[:, :-1], iso[:, 1:], edges)
    _count(iso[:, -1], iso[:, 0], edges)
    _count(iso[:-1, :], iso[1:, :], edges)
    _count(iso[:-1, :-1], iso[1:, 1:], corners)
    _count(iso[:-1, 1:], iso[1:, :-1], corners)
    _count(iso[:-1, -1], iso[1:, 0], corners)
    _count(iso[:-1, 0], iso[1:, -1], corners)
    pairs = sorted(set(edges) | set(corners))
    return pd.DataFrame(
        {
            "iso3_a": [p[0] for p in pairs],
            "iso3_b": [p[1] for p in pairs],
            "shared_edges": [edges.get(p, 0) for p in pairs],
            "shared_corners": [corners.get(p, 0) for p in pairs],
        }
    )


def elevation_stats(
    iso: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    etopo_path: Path = ETOPO_PATH, stride: int = ETOPO_STRIDE,
) -> pd.DataFrame:
    """Within-country elevation mean, spread and quantiles from ETOPO 2022 (5' sample, land only)."""
    ds = xr.open_dataset(etopo_path)
    try:
        z = ds[ETOPO_VAR].isel(lat=slice(None, None, stride), lon=slice(None, None, stride)).to_numpy()
        e_lat = ds["lat"].to_numpy()[::stride]
        e_lon = ds["lon"].to_numpy()[::stride]
    finally:
        ds.close()
    n_lat, n_lon = iso.shape
    row, col = _grid_index(e_lat, e_lon, GPW_CELL_DEG, n_lat, n_lon)
    code = iso[row[:, None], col[None, :]]
    weight = np.broadcast_to(np.cos(np.radians(e_lat))[:, None], z.shape)
    rows = []
    for name in sorted(set(code[code != ""].tolist())):
        m = code == name
        below = float(np.mean(z[m] < 0))
        # ETOPO "surface" carries bathymetry; a 15' country cell can straddle the
        # coast, so samples below sea level are treated as water and dropped (this
        # also drops the small areas of land below sea level, e.g. the Dead Sea shore).
        m &= z >= 0
        if not m.any():
            rows.append({"iso3": name, "elev_below_sea_fraction": below})
            continue
        zz = z[m].astype(float)
        w = weight[m]
        mean = float(np.dot(w, zz) / w.sum())
        sd = float(np.sqrt(np.dot(w, (zz - mean) ** 2) / w.sum()))
        rows.append(
            {
                "iso3": name,
                "elev_mean_m": mean,
                "elev_sd_m": sd,
                "elev_p10_m": float(np.percentile(zz, 10)),
                "elev_p90_m": float(np.percentile(zz, 90)),
                "elev_max_m": float(zz.max()),
                "elev_below_sea_fraction": below,
            }
        )
    return pd.DataFrame(rows)


def koppen_area_shares(
    iso: np.ndarray, lats: np.ndarray, lons: np.ndarray, koppen_path: Path = KOPPEN_PATH
) -> pd.DataFrame:
    """Area-weighted share of each Köppen major group per country, and the dominant one."""
    ds = xr.open_dataset(koppen_path)
    try:
        codes = ds[KOPPEN_VAR].to_numpy()
        k_lat = ds["lat"].to_numpy()
        k_lon = ds["lon"].to_numpy()
    finally:
        ds.close()
    # Look up the 0.5° Köppen cell that contains each 15' GPW cell centre.
    krow, kcol = _grid_index(lats, lons, KOPPEN_CELL_DEG, len(k_lat), len(k_lon))
    sampled = codes[krow[:, None], kcol[None, :]]
    group = np.full(iso.shape, "", dtype=object)
    for code, letter in KOPPEN_GROUPS.items():
        group[sampled == code] = letter
    area = np.broadcast_to(cell_area_km2(lats)[:, None], iso.shape)
    rows = []
    for name in sorted(set(iso[iso != ""].tolist())):
        m = iso == name
        classified = m & (group != "")
        total = float(area[classified].sum())
        shares = {
            f"koppen_share_{letter}": (float(area[classified & (group == letter)].sum()) / total if total else np.nan)
            for letter in KOPPEN_CLASSES
        }
        dominant = max(KOPPEN_CLASSES, key=lambda c: shares[f"koppen_share_{c}"]) if total else ""
        rows.append(
            {
                "iso3": name,
                "koppen_classified_fraction": float(area[classified].sum() / area[m].sum()),
                **shares,
                "koppen_area_dominant": dominant,
                "koppen_area_dominant_share": shares[f"koppen_share_{dominant}"] if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_all(out_dir: Path = OUTPUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and write ``country_geometry.csv`` and ``country_adjacency.csv``."""
    iso, lats, lons = load_national_grid()
    geometry = country_geometry(iso, lats, lons)
    geometry = geometry.merge(elevation_stats(iso, lats, lons), on="iso3", how="left")
    geometry = geometry.merge(koppen_area_shares(iso, lats, lons), on="iso3", how="left")
    adjacency = raster_adjacency(iso)
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry.to_csv(out_dir / GEOMETRY_PATH.name, index=False, float_format="%.6g")
    adjacency.to_csv(out_dir / ADJACENCY_PATH.name, index=False)
    return geometry, adjacency


if __name__ == "__main__":
    geometry, adjacency = build_all()
    print(f"{len(geometry)} countries, {len(adjacency)} adjacent pairs -> {OUTPUT_DIR}")
    print(geometry.describe().T.to_string())
