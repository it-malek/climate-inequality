"""ERA5 area-weighted warming -- the independent reanalysis cross-check.

``docs/future_work.md`` §2: v1.2's area-weighted lens (:mod:`src.area_weighting`)
overturned the headline coupling (warming<->responsibility Spearman rho **+0.36 ->
+0.01**) by weighting every km^2 equally instead of every station. That result
rests on a single gridded product (Berkeley Earth). This module recomputes the
*same* area-weighted country warming off **ERA5 reanalysis 2 m temperature** -- a
model-assimilated field with no station-sampling gaps -- as a fully independent
check: if ERA5 reproduces the collapse it is robust to the data source; if it does
not, the collapse was Berkeley-specific (an equally publishable finding).

**Same operator, same window, same weighting** as :mod:`src.area_weighting` -- only
the data source differs. The reusable machinery (streamed Theil-Sen
:func:`~src.area_weighting.cell_trends`, GPW band-11 ISO3 assignment, the cos(lat)
country reduce) is imported from there; this module supplies only the ERA5-specific
quirks:

- **Absolute Kelvin, not anomalies** -- irrelevant for a *slope* (1 K = 1 degC per
  decade), so no baseline conversion is needed.
- **CF datetime axis** -- decoded by xarray (``decode_times=True``) and snapped to
  first-of-month, matching the station/Berkeley monthly convention so the
  decimal-decade trend axis is identical.
- **0-360 longitudes** -- normalized to ``[-180, 180)`` *only* for the GPW
  national-identifier sampler (which expects geographic longitudes); the data array
  keeps its native column order, so the slope grid and the ISO3 mask stay
  positionally aligned.
- **cos(lat) area-weighting is REQUIRED** (a warming trend is an intensive field),
  exactly as in :mod:`src.area_weighting` -- the mirror of the GPW population-count
  rule, never to be confused.
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
    _grid_coords,
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


def era5_area_weighted_country_trends(
    nc_path=ERA5_GRID_PATH,
    gpw_path=GPW_PATH,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    lat_chunk: int = DEFAULT_LAT_CHUNK,
    lookup_path=GPW_NATID_LOOKUP_PATH,
) -> pd.DataFrame:
    """cos(lat) area-weighted ERA5 warming slope per country (the cross-check).

    Mirrors :func:`src.area_weighting.area_weighted_country_trends` exactly --
    same operator, window and cos(lat) reduce -- but reads ERA5 ``t2m`` (CF time,
    0-360 longitudes). Countries with no successfully-fit cell are omitted.

    Returns:
        One row per ISO3 (sorted), columns :data:`ERA5_COLUMNS`: ``iso3``,
        ``trend_c_per_decade_era5_area`` and ``era5_cell_coverage``.
    """
    lats, lons = _grid_coords(nc_path)
    lookup = load_national_id_lookup(lookup_path)
    mask = era5_cell_iso3(lats, lons, gpw_path, lookup)
    lats, lons, slopes = cell_trends(
        nc_path, mask, start=start, end=end,
        min_coverage=min_coverage, lat_chunk=lat_chunk,
        var=ERA5_VAR, decode_times=True, time_to_months=era5_time_to_months,
    )
    return reduce_cells_to_country(
        mask, lats, slopes,
        value_col=ERA5_AREA_COL, coverage_col=ERA5_COVERAGE_COL,
    )


def era5_world_land_mean(
    nc_path=ERA5_GRID_PATH,
    gpw_path=GPW_PATH,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    lat_chunk: int = DEFAULT_LAT_CHUNK,
    lookup_path=GPW_NATID_LOOKUP_PATH,
) -> float:
    """cos(lat) area-weighted ERA5 world land-mean slope (the headline sanity check).

    Should land near Berkeley's ~0.19 degC/decade global-land trend; an independent
    match validates the whole ERA5 ingest the way Berkeley's 0.1926 did for v1.2.
    """
    lats, lons = _grid_coords(nc_path)
    mask = era5_cell_iso3(lats, lons, gpw_path, load_national_id_lookup(lookup_path))
    lats, lons, slopes = cell_trends(
        nc_path, mask, start=start, end=end,
        min_coverage=min_coverage, lat_chunk=lat_chunk,
        var=ERA5_VAR, decode_times=True, time_to_months=era5_time_to_months,
    )
    return land_mean_from_slopes(lats, slopes)
