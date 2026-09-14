"""Area-weighted country warming from ERA5 reanalysis 2 m temperature.

The area-weighted lens (:mod:`src.area_weighting`) rests on one gridded product.
This module recomputes it on ERA5, ECMWF's model-assimilated reanalysis, which
has no station-sampling gaps, as an independent check of the area-weighting
result. The estimator, window, country assignment and cos(latitude) reduction are
imported unchanged from :mod:`src.area_weighting`; only ERA5's conventions are
handled here:

- absolute Kelvin rather than anomalies (irrelevant to a slope; 1 K = 1 °C per
  decade);
- a CF datetime axis, decoded by xarray and snapped to first-of-month;
- 0-360 longitudes, normalized to [-180, 180) only when sampling the GPW
  national-identifier grid, so the ISO3 mask stays aligned with the native
  column order of the data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.area_weighting import (
    DEFAULT_LAT_CHUNK,
    DEFAULT_MIN_COVERAGE,
    GPW_NATID_LOOKUP_PATH,
    ISO3_COL,
    grid_coords,
    assign_cell_iso3,
    cell_trends,
    land_mean_from_slopes,
    load_national_id_lookup,
    reduce_cells_to_country,
)
from src.cleaning import DEFAULT_END, DEFAULT_START
from src.data_io import RAW_DIR
from src.population import GPW_PATH

logger = logging.getLogger(__name__)

# ERA5 monthly-averaged 2 m temperature, 1950-2013, regridded server-side to 1deg
# (gitignored under data/raw/*; fetched once via scripts/fetch_era5.py).
ERA5_GRID_PATH = RAW_DIR / "era5" / "era5_t2m_monthly_1950_2013_1deg.nc"
ERA5_VAR = "t2m"

ERA5_AREA_COL = "trend_c_per_decade_era5_area"
ERA5_COVERAGE_COL = "era5_cell_coverage"
ERA5_COLUMNS = (ISO3_COL, ERA5_AREA_COL, ERA5_COVERAGE_COL)


def era5_time_to_months(values: np.ndarray) -> pd.DatetimeIndex:
    """CF-decoded ERA5 timestamps -> first-of-month ``DatetimeIndex``.

    ERA5 monthly means are stamped at the start (or middle) of each month; snapping
    to ``period("M").to_timestamp()`` yields the first-of-month axis the
    station/Berkeley pipeline uses, so the decimal-decade trend axis matches exactly
    (the only thing the slope depends on is the spacing, but aligning the convention
    keeps the window bounds and coverage count identical to the Berkeley path).
    """
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(values)))
    return idx.to_period("M").to_timestamp()


def normalize_longitudes(lons: np.ndarray) -> np.ndarray:
    """ERA5 ``0..360`` longitudes -> geographic ``[-180, 180)`` (no reordering)."""
    lons = np.asarray(lons, dtype=float)
    return ((lons + 180.0) % 360.0) - 180.0


def era5_cell_iso3(lats, lons, gpw_path=GPW_PATH, lookup=None) -> np.ndarray:
    """ISO3 mask for ERA5 cells: sample GPW at the *geographic* longitudes.

    ERA5 ships ``0..360`` longitudes; :func:`src.area_weighting.assign_cell_iso3`
    (the GPW band-11 sampler) expects ``[-180, 180)``. Normalizing only the sampled
    longitudes -- not the column order -- keeps the returned ``(n_lat, n_lon)`` mask
    positionally aligned to the native data grid that :func:`cell_trends` reads.
    """
    return assign_cell_iso3(lats, normalize_longitudes(lons), gpw_path, lookup)


def era5_cell_slopes(
    nc_path=ERA5_GRID_PATH,
    gpw_path=GPW_PATH,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    lat_chunk: int = DEFAULT_LAT_CHUNK,
    lookup_path=GPW_NATID_LOOKUP_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell ERA5 Theil-Sen slopes plus the ISO3 mask (one pass over the grid).

    The grid pass is the expensive step (a Theil-Sen fit per land cell), so it is
    done once here and both the country table and the world land mean are reduced
    from its result.

    Returns:
        ``(mask, lats, slopes)`` -- the ``(n_lat, n_lon)`` ISO3 assignment, the
        latitude axis and the ``(n_lat, n_lon)`` slope array (NaN where unfit).
    """
    lats, lons = grid_coords(nc_path)
    lookup = load_national_id_lookup(lookup_path)
    mask = era5_cell_iso3(lats, lons, gpw_path, lookup)
    lats, _, slopes = cell_trends(
        nc_path, mask, start=start, end=end,
        min_coverage=min_coverage, lat_chunk=lat_chunk,
        var=ERA5_VAR, decode_times=True, time_to_months=era5_time_to_months,
    )
    return mask, lats, slopes


def reduce_era5_slopes(mask: np.ndarray, lats: np.ndarray, slopes: np.ndarray) -> pd.DataFrame:
    """cos(lat) area-weighted ERA5 slope per country from a grid pass.

    Returns:
        One row per ISO3 (sorted), columns :data:`ERA5_COLUMNS`: ``iso3``,
        ``trend_c_per_decade_era5_area`` and ``era5_cell_coverage``.
    """
    return reduce_cells_to_country(
        mask, lats, slopes, value_col=ERA5_AREA_COL, coverage_col=ERA5_COVERAGE_COL,
    )


def era5_area_weighted_country_trends(nc_path=ERA5_GRID_PATH, gpw_path=GPW_PATH, **kwargs) -> pd.DataFrame:
    """cos(lat) area-weighted ERA5 warming slope per country.

    Same operator, window and reduction as
    :func:`src.area_weighting.area_weighted_country_trends`, on ERA5 ``t2m``.
    Countries with no successfully-fit cell are omitted. Accepts the keyword
    arguments of :func:`era5_cell_slopes`.
    """
    return reduce_era5_slopes(*era5_cell_slopes(nc_path, gpw_path, **kwargs))


def era5_world_land_mean(nc_path=ERA5_GRID_PATH, gpw_path=GPW_PATH, **kwargs) -> float:
    """cos(lat) area-weighted ERA5 world land-mean slope (the ingest sanity check).

    Should land near Berkeley Earth's ~0.19 °C/decade global-land trend. Accepts
    the keyword arguments of :func:`era5_cell_slopes`.
    """
    _, lats, slopes = era5_cell_slopes(nc_path, gpw_path, **kwargs)
    return land_mean_from_slopes(lats, slopes)
