"""Shared helpers for regular lat/lon grids.

Several stages sample a static ``(lat, lon)`` grid at city coordinates
(:mod:`src.validation` for the Berkeley Earth gridded temperatures,
:mod:`src.explain` for ETOPO elevation and Koeppen classes,
:mod:`src.population` for the GPW population grid) and two decode Berkeley's
fractional-year time axis (:mod:`src.validation`, :mod:`src.area_weighting`).
This module holds those pieces once: the nearest-cell lookup, the
longitude-convention handling, lazy NetCDF point sampling, and the time decoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def decode_fractional_years(values: Sequence[float] | np.ndarray) -> pd.DatetimeIndex:
    """Convert Berkeley Earth fractional decimal years to month timestamps.

    The gridded files encode time as ``year + (month - 0.5) / 12`` (mid-month
    decimals, e.g. 2014.041666... = January 2014), the same convention as
    :func:`src.cleaning.to_decimal_decades` in years instead of decades. xarray's
    CF decoding cannot parse the ``"year A.D."`` unit, so the axis is decoded
    here to first-of-month timestamps matching the pipeline's ``dt`` column.

    Raises:
        ValueError: if any value does not sit on the mid-month grid (e.g. a
            start-of-month ``year + (month - 1) / 12`` axis).
    """
    arr = np.asarray(values, dtype=float)
    years = np.floor(arr)
    month_float = (arr - years) * 12.0 + 0.5
    months = np.rint(month_float)
    off_grid = np.abs(month_float - months) > 0.01
    if off_grid.any():
        raise ValueError(
            f"{int(off_grid.sum())} time value(s) are not mid-month decimal "
            f"years, e.g. {arr[off_grid][:3]}"
        )
    parts = pd.DataFrame(
        {"year": years.astype(int), "month": months.astype(int), "day": 1}
    )
    return pd.DatetimeIndex(pd.to_datetime(parts))


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
    nc_path: Path, lat_dim: str = "lat", lon_dim: str = "lon", *, engine: str | None = None
) -> dict:
    """Summarize a static grid's coordinate conventions.

    Args:
        nc_path: Path to a NetCDF file with 1D `lat_dim`/`lon_dim`
            coordinate variables.
        lat_dim: Name of the latitude coordinate variable.
        lon_dim: Name of the longitude coordinate variable.
        engine: optional xarray backend engine (e.g. ``"netcdf4"``); ``None``
            lets xarray auto-detect.

    Returns:
        Dict with ``lat_min``, ``lat_max``, ``lat_ascending``, ``lon_min``,
        ``lon_max``, ``lon_ascending`` and ``lon_convention`` (one of
        ``"-180-180"`` or ``"0-360"``, by whether any longitude exceeds 180).
    """
    ds = xr.open_dataset(nc_path, engine=engine)
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
    *,
    band_dim: str | None = None,
    band: int | None = None,
    engine: str | None = None,
) -> np.ndarray:
    """Nearest-cell values of a static (lat, lon) NetCDF variable.

    Shared by elevation (:func:`src.explain.add_elevation`), Koeppen class
    (:func:`src.explain.add_koppen`) and population
    (:func:`src.population.sample_population`) sampling. Longitudes are
    normalized to the grid's convention (:func:`normalize_longitudes`,
    :func:`check_coordinate_orientation`) before the nearest-cell lookup
    (:func:`nearest_cell_indices`), so a [-180,180]-vs-[0,360] mismatch cannot
    silently wrap to the wrong side of the globe.

    Sampling is lazy: the file is opened with xarray's native lazy backend and
    only the queried points are read via pointwise (vectorized) indexing, so the
    full grid is never materialized -- important for large rasters (e.g. the
    ~84 MB GPW v4 population grid).

    Args:
        nc_path: Path to a NetCDF file with a `var(lat_dim, lon_dim)` variable
            (optionally with a leading `band_dim`, e.g. GPW's ``raster``).
        lats, lons: Query coordinates, in signed [-90,90]/[-180,180] degrees.
        var: Variable name to sample.
        lat_dim, lon_dim: Names of the latitude/longitude dimensions.
        band_dim: optional extra dimension to select a single band from before
            sampling (e.g. ``"raster"``); ``None`` for a plain 2D variable.
        band: the `band_dim` coordinate *value* to select (via ``.sel``).
        engine: optional xarray backend engine (e.g. ``"netcdf4"``).

    Returns:
        1D float array, one value per query point.
    """
    orientation = check_coordinate_orientation(nc_path, lat_dim, lon_dim, engine=engine)
    query_lons = normalize_longitudes(np.asarray(lons, dtype=float), orientation["lon_convention"])
    query_lats = np.asarray(lats, dtype=float)

    ds = xr.open_dataset(nc_path, engine=engine)
    try:
        lat_index = pd.Index(ds[lat_dim].to_numpy().astype(float))
        lon_index = pd.Index(ds[lon_dim].to_numpy().astype(float))
        lat_pos, lon_pos = nearest_cell_indices(lat_index, lon_index, query_lats, query_lons)
        da = ds[var]
        if band_dim is not None:
            da = da.sel({band_dim: band})
        values = da.isel(
            {
                lat_dim: xr.DataArray(lat_pos, dims="_points"),
                lon_dim: xr.DataArray(lon_pos, dims="_points"),
            }
        ).to_numpy()
    finally:
        ds.close()
    return np.asarray(values, dtype=float)
