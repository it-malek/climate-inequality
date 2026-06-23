"""People-weighted warming exposure from the GPW v4 population grid.

``docs/future_work.md`` §2: the country warming mean
(:func:`src.emissions.aggregate_trends_by_country`) is deliberately
*station-weighted* -- it over-counts dense mid-latitude station clusters and
under-counts where people actually live. This module turns it into a
*people-weighted* mean ("the warming the average resident experiences") by
sampling a population grid at each city-location's coordinates (no fragile
city-name matching, reusing :func:`src.grids.sample_static_grid`) and weighting
each location's Theil-Sen slope by that population.

**Grid.** SEDAC GPW v4.11, UN WPP-adjusted **population count** at 15 arc-minutes
(``data/raw/gpw-v4-...totpop_15_min_nc/...adjusted_rev11_15_min.nc``; gitignored,
~84 MB). One variable, ``UN WPP-Adjusted Population Count ...``, dims
``(raster, latitude, longitude)``; raster bands 1-5 are the count for
2000/2005/2010/2015/2020 (see ``gpw_v4_netcdf_contents_rev11.csv``); units are
*Persons* and the 2020 band sums to ~7.76e9. Loading is lazy: sampling reads only
the queried cells, never the full array.

**No cos(latitude) weighting of the counts.** This is population *count*
(persons per cell), not density. Two reasons cos(lat) area-weighting is wrong
here: (a) a count grid already embeds meridian convergence -- cells shrink toward
the poles, so the count per cell already reflects the smaller area; the correct
*global* statistic is the sum (:func:`global_population_total`), not a cos-weighted
mean. (b) In the exposure weighting the population is a *per-person weight* on
warming slopes; cos-weighting would systematically down-weight high-latitude
residents -- exactly the Arctic-amplified, fastest-warming people -- biasing the
inequality we measure. cos(lat) area-weighting is correct for an *intensive* field
(temperature, population *density*); :func:`latitude_area_weights` /
:func:`area_weighted_mean` provide it for that use, and are deliberately *not*
applied to the counts in :func:`population_weighted_country_mean`.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data_io import RAW_DIR
from src.grids import sample_static_grid

logger = logging.getLogger(__name__)

# SEDAC GPW v4.11 UN WPP-adjusted population COUNT, 15 arc-minute (gitignored).
GPW_DIR = (
    RAW_DIR
    / "gpw-v4-population-count-adjusted-to-2015-unwpp-country-totals-rev11_totpop_15_min_nc"
)
GPW_PATH = GPW_DIR / "gpw_v4_population_count_adjusted_rev11_15_min.nc"
GPW_POP_VAR = (
    "UN WPP-Adjusted Population Count, v4.11 "
    "(2000, 2005, 2010, 2015, 2020): 15 arc-minutes"
)
GPW_LAT_DIM = "latitude"
GPW_LON_DIM = "longitude"
GPW_RASTER_DIM = "raster"
GPW_ENGINE = "netcdf4"
# raster band (coordinate value) carrying the population COUNT for each year, per
# gpw_v4_netcdf_contents_rev11.csv (bands 6-20 are quality/area/identifier layers).
GPW_COUNT_BAND_BY_YEAR: dict[int, int] = {2000: 1, 2005: 2, 2010: 3, 2015: 4, 2020: 5}
GPW_DEFAULT_YEAR = 2020
# GPW _FillValue is a large negative (~ -3.4e38) for ocean / no-data.
GPW_FILL_FLOOR = 0.0

# The production grid for the people-weighted exposure lens.
POP_GRID_PATH = GPW_PATH

ID_COL = "Country"
POP_WEIGHTED_COL = "trend_c_per_decade_pop_weighted"
POP_COVERAGE_COL = "pop_weight_coverage"


def _dask_available() -> bool:
    """Whether dask is importable (xarray needs it to honor ``chunks=``)."""
    return importlib.util.find_spec("dask") is not None


def _require_grid(path: Path) -> Path:
    """Validate the population grid exists, with one informative error message.

    Shared by every entry point (:func:`verify_population_grid`,
    :func:`sample_population`, :func:`global_population_total`) so a missing
    15-arc-minute NetCDF fails the same clear way everywhere rather than as a
    cryptic backend error from ``xarray``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"population grid not found: {path}; place the SEDAC GPW v4 "
            f"15-arc-minute population NetCDF there (the data/raw tree is gitignored)"
        )
    return path


def verify_population_grid(
    path: Path = GPW_PATH,
    var: str = GPW_POP_VAR,
    *,
    engine: str = GPW_ENGINE,
    print_summary: bool = True,
) -> dict:
    """Validate the population grid exists and report its structure.

    Memory-safe verification: opens the file lazily (``engine``, and chunked when
    dask is installed) and reads only metadata -- never the ~84 MB data array.
    Prints the data-variable name(s), dimensions, units and the year->raster-band
    mapping so the targeted population-count band can be checked against
    ``gpw_v4_netcdf_contents_rev11.csv``.

    Args:
        path: NetCDF path.
        var: expected population-count variable name.
        engine: xarray backend (``"netcdf4"`` for a stable connection).
        print_summary: print the human-readable summary.

    Returns:
        Dict with ``path``, ``data_vars``, ``dims``, ``units``,
        ``count_band_by_year`` and ``lazy_chunks``.

    Raises:
        FileNotFoundError: if `path` is absent.
    """
    path = _require_grid(path)
    chunks = {GPW_LAT_DIM: 180, GPW_LON_DIM: 360} if _dask_available() else None
    ds = xr.open_dataset(path, engine=engine, chunks=chunks)
    try:
        data_vars = list(ds.data_vars)
        sizes = dict(ds.sizes)
        units = ds[var].attrs.get("units") if var in ds.data_vars else None
    finally:
        ds.close()

    if print_summary:
        print(f"population grid: {path}")
        print(f"  data variables: {data_vars}")
        print(f"  dims: {sizes}   units: {units}")
        lazy = chunks if chunks else "native lazy (dask not installed; chunks skipped)"
        print(f"  lazy loading: {lazy}")
        print("  population-count bands (gpw_v4_netcdf_contents_rev11.csv):")
        for year, band in GPW_COUNT_BAND_BY_YEAR.items():
            print(f"    year {year} -> raster band {band}")
    return {
        "path": str(path),
        "data_vars": data_vars,
        "dims": sizes,
        "units": units,
        "count_band_by_year": dict(GPW_COUNT_BAND_BY_YEAR),
        "lazy_chunks": chunks,
    }


def sample_population(
    lats: np.ndarray,
    lons: np.ndarray,
    nc_path: Path = GPW_PATH,
    year: int = GPW_DEFAULT_YEAR,
    *,
    var: str = GPW_POP_VAR,
    lat_dim: str = GPW_LAT_DIM,
    lon_dim: str = GPW_LON_DIM,
    band_dim: str = GPW_RASTER_DIM,
    engine: str = GPW_ENGINE,
) -> np.ndarray:
    """Population count at each ``(lat, lon)`` via lazy nearest-cell sampling.

    Selects the population-count band for `year` from the GPW raster stack and
    samples it pointwise (:func:`src.grids.sample_static_grid` -- lazy, reads only
    the queried cells). GPW no-data / ocean cells (fill < 0) are returned as NaN.

    Args:
        lats, lons: Query coordinates, signed degrees.
        nc_path: GPW NetCDF path.
        year: population-count year (key of :data:`GPW_COUNT_BAND_BY_YEAR`).
        var, lat_dim, lon_dim, band_dim, engine: GPW grid parameters.

    Returns:
        1D float array of person-counts (NaN where no-data), aligned to inputs.
    """
    nc_path = _require_grid(nc_path)
    band = GPW_COUNT_BAND_BY_YEAR[year]
    values = sample_static_grid(
        nc_path,
        np.asarray(lats, dtype=float),
        np.asarray(lons, dtype=float),
        var,
        lat_dim,
        lon_dim,
        band_dim=band_dim,
        band=band,
        engine=engine,
    )
    return np.where(values < GPW_FILL_FLOOR, np.nan, values)


def population_weighted_country_mean(
    trends: pd.DataFrame,
    nc_path: Path = GPW_PATH,
    year: int = GPW_DEFAULT_YEAR,
    value_col: str = "slope_c_per_decade",
) -> pd.DataFrame:
    """People-weighted mean warming slope per country (no cos-lat weighting).

    Each city-location's slope is weighted by the population *count* sampled at
    its coordinates -- one person, one vote, regardless of latitude (see the
    module docstring on why cos(lat) is deliberately *not* applied to counts). A
    location with NaN or non-positive population gets zero weight; a country with
    no positive weights falls back to the unweighted mean (logged), so every
    country keeps a value.

    Args:
        trends: one row per city-location, with ``Country``, ``Latitude``,
            ``Longitude`` and `value_col`.
        nc_path: GPW NetCDF path.
        year: population-count year.
        value_col: per-location slope column to average.

    Returns:
        One row per ``Country`` with :data:`POP_WEIGHTED_COL` and
        :data:`POP_COVERAGE_COL` (fraction of locations with positive population).
    """
    pop = sample_population(
        trends["Latitude"].to_numpy(), trends["Longitude"].to_numpy(), nc_path, year
    )
    valid = np.isfinite(pop) & (pop > 0.0)
    work = trends[[ID_COL, value_col]].assign(
        _weight=np.where(valid, pop, 0.0), _valid=valid
    )

    rows = []
    for country, grp in work.groupby(ID_COL, observed=True, sort=True):
        slopes = grp[value_col].to_numpy(dtype=float)
        weights = grp["_weight"].to_numpy(dtype=float)
        total = float(weights.sum())
        if total > 0.0:
            weighted = float(np.dot(slopes, weights) / total)
        else:
            weighted = float(slopes.mean())
            logger.info(
                "population weighting: %s has no positive population weights; "
                "falling back to the unweighted mean",
                country,
            )
        rows.append(
            {
                ID_COL: country,
                POP_WEIGHTED_COL: weighted,
                POP_COVERAGE_COL: float(grp["_valid"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=[ID_COL, POP_WEIGHTED_COL, POP_COVERAGE_COL])


# ---------------------------------------------------------------------
# Area-aware diagnostics (correct statistics for the count grid)
# ---------------------------------------------------------------------


def global_population_total(
    path: Path = GPW_PATH,
    year: int = GPW_DEFAULT_YEAR,
    *,
    var: str = GPW_POP_VAR,
    engine: str = GPW_ENGINE,
) -> float:
    """Total people in the count band for `year` -- the correct global statistic.

    For a count grid the meaningful global number is the **sum** (not a
    cos-weighted mean): a ~7.8e9 sanity check that the right band is targeted.
    Reads a single ~4 MB band (not the full 84 MB file); ocean/no-data (fill < 0)
    is masked out.
    """
    path = _require_grid(path)
    band = GPW_COUNT_BAND_BY_YEAR[year]
    ds = xr.open_dataset(path, engine=engine)
    try:
        arr = ds[var].sel({GPW_RASTER_DIM: band}).to_numpy()
    finally:
        ds.close()
    keep = np.isfinite(arr) & (arr >= GPW_FILL_FLOOR)
    return float(np.where(keep, arr, 0.0).sum())


def latitude_area_weights(lats: np.ndarray) -> np.ndarray:
    """cos(latitude) cell-area weights for a regular lat/lon grid (normalized).

    Accounts for the convergence of meridians toward the poles. This is the
    correct weight for **area-averaging an intensive field** (temperature,
    population *density*). It is deliberately **not** used to weight population
    *counts* in :func:`population_weighted_country_mean` -- see the module
    docstring.
    """
    w = np.clip(np.cos(np.radians(np.asarray(lats, dtype=float))), 0.0, None)
    total = w.sum()
    return w / total if total > 0 else w


def area_weighted_mean(field: np.ndarray, lats: np.ndarray) -> float:
    """Latitude (cos)-area-weighted mean of a 2D ``(lat, lon)`` field.

    The correct spatial average for an **intensive** field (e.g. the GPW
    population *density* product), using :func:`latitude_area_weights` so
    pole-ward cells -- which cover less area on a regular grid -- count less.
    Non-finite and negative (GPW fill) cells are masked. Provided for that use;
    the exposure pipeline weights raw counts, not cos(lat).

    Args:
        field: 2D array indexed ``(lat, lon)``.
        lats: 1D latitudes aligned to ``field``'s first axis.

    Returns:
        The area-weighted mean, or NaN if no cell is valid.
    """
    field = np.asarray(field, dtype=float)
    lat_w = np.clip(np.cos(np.radians(np.asarray(lats, dtype=float))), 0.0, None)
    weights = np.broadcast_to(lat_w[:, None], field.shape).astype(float).copy()
    valid = np.isfinite(field) & (field >= 0.0)
    weights = np.where(valid, weights, 0.0)
    total = weights.sum()
    if total <= 0.0:
        return float("nan")
    return float((np.where(valid, field, 0.0) * weights).sum() / total)
