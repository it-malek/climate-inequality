"""Shared nearest-cell lookup and sampling for regular lat/lon grids.

:mod:`src.validation` (Phase 6, the Berkeley Earth gridded temperature
NetCDF), :mod:`src.explain` (Phase 7, the ETOPO elevation and Koeppen
climate-class NetCDFs) and :mod:`src.population` (the population grid) all map
city-location coordinates onto the nearest cell of a static ``(lat, lon)``
grid. This module factors out that idiom -- the nearest-cell lookup plus the
longitude-convention handling and NetCDF point sampling built on it -- so every
call site (and its determinism/boundary-case tests) shares one implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def nearest_cell_indices(
    lat_index: pd.Index,
    lon_index: pd.Index,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-cell positions in `lat_index`/`lon_index` for each point.

    Args:
        lat_index: Grid latitude coordinate values (any order).
        lon_index: Grid longitude coordinate values (any order).
        lats: Query latitudes.
        lons: Query longitudes.

    Returns:
        `(lat_pos, lon_pos)` integer position arrays into `lat_index` /
        `lon_index`. Points outside the grid's range clamp to the nearest
        edge cell (``pandas.Index.get_indexer`` with ``method="nearest"``).
    """
    lat_pos = lat_index.get_indexer(lats, method="nearest")
    lon_pos = lon_index.get_indexer(lons, method="nearest")
    return lat_pos, lon_pos


def check_coordinate_orientation(
    nc_path: Path, lat_dim: str = "lat", lon_dim: str = "lon"
) -> dict:
    """Summarize a static grid's coordinate conventions.

    Args:
        nc_path: Path to a NetCDF file with 1D `lat_dim`/`lon_dim`
            coordinate variables.
        lat_dim: Name of the latitude coordinate variable.
        lon_dim: Name of the longitude coordinate variable.

    Returns:
        Dict with ``lat_min``, ``lat_max``, ``lat_ascending``, ``lon_min``,
        ``lon_max``, ``lon_ascending`` and ``lon_convention`` (one of
        ``"-180-180"`` or ``"0-360"``, by whether any longitude exceeds 180).
    """
    ds = xr.open_dataset(nc_path)
    try:
        lat = ds[lat_dim].to_numpy().astype(float)
        lon = ds[lon_dim].to_numpy().astype(float)
    finally:
        ds.close()
    return {
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lat_ascending": bool(lat[0] < lat[-1]),
        "lon_min": float(lon.min()),
        "lon_max": float(lon.max()),
        "lon_ascending": bool(lon[0] < lon[-1]),
        "lon_convention": "0-360" if lon.max() > 180.0 else "-180-180",
    }


def normalize_longitudes(lons: np.ndarray, convention: str) -> np.ndarray:
    """Convert signed [-180,180] longitudes to a grid's convention.

    `city_trends.parquet` longitudes are signed floats in [-180, 180]
    (guaranteed by :func:`src.cleaning.parse_coordinate`). If the target
    grid instead uses [0, 360], shift negative values by +360 so
    :func:`nearest_cell_indices` compares like with like.

    Args:
        lons: Signed longitudes in degrees, range [-180, 180].
        convention: ``"-180-180"`` (no-op) or ``"0-360"``.

    Returns:
        Longitudes in `convention`.

    Raises:
        ValueError: if `convention` is not recognized.
    """
    lons = np.asarray(lons, dtype=float)
    if convention == "-180-180":
        return lons
    if convention == "0-360":
        return np.where(lons < 0.0, lons + 360.0, lons)
    raise ValueError(f"unknown longitude convention: {convention!r}")


def sample_static_grid(
    nc_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    var: str,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> np.ndarray:
    """Nearest-cell values of a static 2D (lat, lon) NetCDF variable.

    Shared by elevation (:func:`src.explain.add_elevation`), Koeppen class
    (:func:`src.explain.add_koppen`) and population
    (:func:`src.population.sample_population`) sampling. Longitudes are
    normalized to the grid's convention (:func:`normalize_longitudes`,
    :func:`check_coordinate_orientation`) before the nearest-cell lookup
    (:func:`nearest_cell_indices`), so a [-180,180]-vs-[0,360] mismatch cannot
    silently wrap to the wrong side of the globe.

    Args:
        nc_path: Path to a NetCDF file with a `var(lat_dim, lon_dim)`
            variable (or `(lon_dim, lat_dim)` -- dimension order is read
            from the variable itself).
        lats, lons: Query coordinates, in signed [-90,90]/[-180,180]
            degrees.
        var: Variable name to sample.
        lat_dim, lon_dim: Names of the latitude/longitude dimensions.

    Returns:
        1D float array, one value per query point. Values are read via
        pointwise (vectorized) indexing, so the full grid is never loaded
        into memory.
    """
    orientation = check_coordinate_orientation(nc_path, lat_dim, lon_dim)
    query_lons = normalize_longitudes(np.asarray(lons, dtype=float), orientation["lon_convention"])
    query_lats = np.asarray(lats, dtype=float)

    ds = xr.open_dataset(nc_path)
    try:
        lat_index = pd.Index(ds[lat_dim].to_numpy().astype(float))
        lon_index = pd.Index(ds[lon_dim].to_numpy().astype(float))
        lat_pos, lon_pos = nearest_cell_indices(lat_index, lon_index, query_lats, query_lons)
        da = ds[var]
        values = da.isel(
            {
                lat_dim: xr.DataArray(lat_pos, dims="_points"),
                lon_dim: xr.DataArray(lon_pos, dims="_points"),
            }
        ).to_numpy()
    finally:
        ds.close()
    return np.asarray(values, dtype=float)
