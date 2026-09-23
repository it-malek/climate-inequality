"""People-weighted country warming from the GPW v4 population grid.

The station-based country mean over-counts wherever stations cluster. This
module weights each city-location's Theil-Sen slope by the population living in
its grid cell -- the warming the average resident experiences rather than the
average station -- by sampling the SEDAC GPW v4.11 UN-adjusted population
*count* grid (15 arc-minutes, ~84 MB, gitignored) at each location's
coordinates with :func:`src.grids.sample_static_grid`. Raster bands 1-5 hold
the counts for 2000/2005/2010/2015/2020; band 5 (2020) is used.

Counts are used as weights directly, with no cos(latitude) factor. A count is
extensive: cells shrink toward the poles and already hold proportionally fewer
people, so a further latitude correction would double-count the geometry and
down-weight exactly the high-latitude residents who warm fastest. The intensive
case (a trend or a density averaged over cells) is the opposite and needs
cos(latitude); :func:`latitude_area_weights` provides it for
:mod:`src.area_weighting`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

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

ID_COL = "Country"
POP_WEIGHTED_COL = "trend_c_per_decade_pop_weighted"
POP_COVERAGE_COL = "pop_weight_coverage"


def _require_grid(path: Path) -> Path:
    """Validate the population grid exists, with one informative error message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"population grid not found: {path}; place the SEDAC GPW v4 "
            f"15-arc-minute population NetCDF there (the data/raw tree is gitignored)"
        )
    return path


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
    """People-weighted mean warming slope per country.

    Each city-location's slope is weighted by the population count sampled at
    its coordinates (no latitude correction; see the module docstring). A
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


def latitude_area_weights(lats: np.ndarray) -> np.ndarray:
    """Normalized cos(latitude) area weights for cells of a regular lat/lon grid.

    The weight for averaging an intensive field (a trend, a density) over grid
    cells whose area shrinks toward the poles. Not applied to population counts
    (see the module docstring).
    """
    w = np.clip(np.cos(np.radians(np.asarray(lats, dtype=float))), 0.0, None)
    total = w.sum()
    return w / total if total > 0 else w
