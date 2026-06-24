"""Area-weighted gridded warming exposure from the Berkeley Earth grid.

``docs/future_work.md`` §2: the project's #1 external-validity gap is *station
sampling bias*. The country warming mean
(:func:`src.emissions.aggregate_trends_by_country`) is *station-weighted*, so
dense mid-latitude station clusters dominate and the Arctic / Sahara / Amazonia
are under-sampled. This module replaces that with a **true area-weighted**
country mean computed directly off the Berkeley Earth 1°×1° gridded field: a
per-cell Theil-Sen trend (the SAME operator and 1950-01..2013-09 window as the
station pipeline, so the only difference is the weighting), each land cell
assigned to a country via the GPW v4 **National Identifier Grid**, then reduced
to a country mean weighted by ``cos(latitude)``.

**cos(latitude) IS REQUIRED here.** A warming *trend* is an **intensive** field;
on a regular lat/lon grid pole-ward cells cover less area, so an honest country
mean weights each cell by ``cos(lat)`` (:func:`src.population.latitude_area_weights`).
This is the EXACT MIRROR of the GPW population-**count** rule in
:mod:`src.population`, where cos(lat) is *forbidden* because a count is
**extensive** (it already embeds meridian convergence). The two must never be
confused: intensive field → cos-weight; extensive count → never cos-weight.

**Country assignment.** Reuse band 11 of the already-committed GPW NetCDF (the
"National Identifier Grid", numeric ISO codes per cell) sampled at each Berkeley
cell center -- no new polygon dataset. The numeric codes resolve to ISO3 via the
sibling lookup table, the canonical bridge to OWID's ``iso_code``.

**Memory-safe.** The full 1950-2013 window over the global grid would be a
~190 MB float array, so the per-cell trends are computed by streaming the grid in
**latitude-band chunks** (native-lazy ``xarray``; peak memory is one band, never
the whole window). No dask dependency, matching the GPW decision.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from src.cleaning import DEFAULT_END, DEFAULT_START, to_decimal_decades
from src.data_io import RAW_DIR
from src.grids import sample_static_grid
from src.population import (
    GPW_DIR,
    GPW_ENGINE,
    GPW_FILL_FLOOR,
    GPW_PATH,
    GPW_POP_VAR,
    GPW_RASTER_DIM,
    latitude_area_weights,
)

logger = logging.getLogger(__name__)

# Berkeley Earth 1°×1° gridded monthly anomalies (gitignored, ~199 MB; the same
# file src.validation reads). Defined here -- not imported from src.validation --
# so this module stays out of the validation->app_assets->emissions import cycle.
BERKELEY_GRID_PATH = RAW_DIR / "berkeley_gridded" / "Complete_TAVG_LatLong1.nc"
GRID_VAR = "temperature"
GRID_DIMS = ("time", "latitude", "longitude")

# GPW "National Identifier Grid" raster band (numeric country codes), per
# gpw_v4_netcdf_contents_rev11.csv order 11; the lookup maps Value -> ISO3.
GPW_NATID_BAND = 11
GPW_NATID_LOOKUP_PATH = GPW_DIR / "gpw_v4_national_identifier_grid_rev11_lookup.txt"

# Per-cell coverage gate: fraction of the analysis-window months a cell must have
# finite data to be fit, matching src.trends.DEFAULT_MIN_COVERAGE.
DEFAULT_MIN_COVERAGE = 0.9
# Latitude rows read per streamed slab (memory/throughput trade-off; one slab is
# ~ chunk x n_lon x n_window_months float32 -- a few MB at 1° resolution).
DEFAULT_LAT_CHUNK = 30

ISO3_COL = "iso3"
AREA_WEIGHTED_COL = "trend_c_per_decade_area_weighted"
AREA_COVERAGE_COL = "area_cell_coverage"
AREA_COLUMNS = (ISO3_COL, AREA_WEIGHTED_COL, AREA_COVERAGE_COL)


def _decode_fractional_years(values: Sequence[float] | np.ndarray) -> pd.DatetimeIndex:
    """Berkeley fractional decimal years -> first-of-month timestamps.

    Mirrors :func:`src.validation.decode_fractional_years` (re-implemented here to
    avoid the validation->app_assets->emissions->area_weighting import cycle): the
    grid encodes time as ``year + (month - 0.5) / 12`` (mid-month).
    """
    arr = np.asarray(values, dtype=float)
    years = np.floor(arr)
    month_float = (arr - years) * 12.0 + 0.5
    months = np.rint(month_float)
    off_grid = np.abs(month_float - months) > 0.01
    if off_grid.any():
        raise ValueError(
            f"{int(off_grid.sum())} time value(s) are not mid-month decimal years, "
            f"e.g. {arr[off_grid][:3]}"
        )
    parts = pd.DataFrame(
        {"year": years.astype(int), "month": months.astype(int), "day": 1}
    )
    return pd.DatetimeIndex(pd.to_datetime(parts))


def load_national_id_lookup(path: Path = GPW_NATID_LOOKUP_PATH) -> dict[int, str]:
    """Map GPW national-identifier numeric codes to ISO3 alpha codes.

    Reads the tab-separated ``gpw_v4_national_identifier_grid_rev11_lookup.txt``
    (columns ``Value``/``ISOCODE``/...); rows with a blank ISO code (e.g.
    disputed territories) are skipped, so they assign to no country.

    Raises:
        FileNotFoundError: if `path` is absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"GPW national-identifier lookup not found: {path}; it ships beside the "
            f"GPW NetCDF in the (gitignored) data/raw tree"
        )
    table = pd.read_csv(path, sep="\t", usecols=["Value", "ISOCODE"])
    out: dict[int, str] = {}
    for value, iso in zip(table["Value"], table["ISOCODE"], strict=True):
        if isinstance(iso, str) and iso.strip():
            out[int(value)] = iso.strip()
    return out


def _grid_coords(nc_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the (latitude, longitude) coordinate arrays of a Berkeley grid."""
    ds = xr.open_dataset(nc_path, decode_times=False)
    try:
        lats = ds["latitude"].to_numpy().astype(float)
        lons = ds["longitude"].to_numpy().astype(float)
    finally:
        ds.close()
    return lats, lons


def assign_cell_iso3(
    lats: np.ndarray,
    lons: np.ndarray,
    gpw_path: Path = GPW_PATH,
    lookup: dict[int, str] | None = None,
) -> np.ndarray:
    """ISO3 country code for every ``(lat, lon)`` grid cell (None over ocean).

    Samples GPW band 11 (:data:`GPW_NATID_BAND`) at each cell center via the lazy
    :func:`src.grids.sample_static_grid` (reads only the queried cells), rounds the
    numeric code and resolves it through :func:`load_national_id_lookup`. Ocean /
    no-data cells (GPW fill < 0) and unmapped codes become ``None``.

    Args:
        lats: 1D grid latitudes (cell centers).
        lons: 1D grid longitudes (cell centers).
        gpw_path: GPW NetCDF path.
        lookup: numeric-code -> ISO3 map; loaded from disk when ``None``.

    Returns:
        Object array of shape ``(len(lats), len(lons))``; ISO3 string or ``None``.
    """
    if lookup is None:
        lookup = load_national_id_lookup()
    lon_grid, lat_grid = np.meshgrid(
        np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
    )
    codes = sample_static_grid(
        gpw_path,
        lat_grid.ravel(),
        lon_grid.ravel(),
        GPW_POP_VAR,
        "latitude",
        "longitude",
        band_dim=GPW_RASTER_DIM,
        band=GPW_NATID_BAND,
        engine=GPW_ENGINE,
    ).reshape(lat_grid.shape)

    out = np.full(codes.shape, None, dtype=object)
    land = np.isfinite(codes) & (codes >= GPW_FILL_FLOOR)
    for idx in zip(*np.nonzero(land), strict=True):
        iso = lookup.get(int(round(float(codes[idx]))))
        if iso is not None:
            out[idx] = iso
    return out


def cell_trends(
    nc_path: Path,
    country_mask: np.ndarray,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    lat_chunk: int = DEFAULT_LAT_CHUNK,
    var: str = GRID_VAR,
    decode_times: bool = False,
    time_to_months: Callable[[np.ndarray], pd.DatetimeIndex] = _decode_fractional_years,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell Theil-Sen warming slope over the analysis window (streamed).

    Streams the gridded temperature variable in `lat_chunk`-row latitude bands so
    peak memory is one band, never the ~190 MB full-window array. Each cell with a
    country assignment (``country_mask`` not ``None``) and at least
    ``min_coverage`` of the window's months finite is fit with
    :func:`scipy.stats.theilslopes` on the
    :func:`src.cleaning.to_decimal_decades` axis -- the identical operator, axis
    and window as :func:`src.trends.fit_city_trends`. Ocean / unassigned cells are
    skipped (left NaN), which also avoids fitting two-thirds of the globe.

    The defaults read the Berkeley grid (``temperature``, mid-month fractional-year
    time axis); the `var` / `decode_times` / `time_to_months` knobs let another
    intensive-field grid on the same ``(time, latitude, longitude)`` layout reuse
    the identical operator and window -- e.g. ERA5 ``t2m`` (CF datetime axis), where
    only the *data source* differs, not the trend. A trend slope is unit-offset
    invariant, so ERA5's absolute Kelvin needs no anomaly conversion (1 K = 1 °C
    per decade).

    Args:
        nc_path: gridded NetCDF (Berkeley, or a synthetic/ERA5 file with the same
            ``var(time, latitude, longitude)`` layout).
        country_mask: ``(n_lat, n_lon)`` object array from
            :func:`assign_cell_iso3`; only non-``None`` cells are fit.
        start, end: inclusive analysis-window bounds (the station window).
        min_coverage: minimum finite-month fraction for a cell to be fit.
        lat_chunk: latitude rows per streamed slab.
        var: gridded variable name (default ``temperature``; ERA5 uses ``t2m``).
        decode_times: passed to :func:`xarray.open_dataset`; Berkeley's axis is not
            CF-decodable (``False``), ERA5's is (``True``).
        time_to_months: maps the raw ``time`` values to a first-of-month
            ``DatetimeIndex`` (Berkeley fractional-year decoder by default).

    Returns:
        ``(lats, lons, slopes)`` -- the grid coordinate arrays and an
        ``(n_lat, n_lon)`` float array of °C/decade slopes (NaN where unfit).

    Raises:
        ValueError: if the file lacks ``var(time, latitude, longitude)``.
    """
    ds = xr.open_dataset(nc_path, decode_times=decode_times)
    try:
        if var not in ds.data_vars:
            raise ValueError(
                f"{nc_path} has no {var!r} variable (found {sorted(ds.data_vars)})"
            )
        da = ds[var]
        if tuple(da.dims) != GRID_DIMS:
            raise ValueError(
                f"{var} has dims {tuple(da.dims)}, expected {GRID_DIMS}"
            )
        months = time_to_months(ds["time"].to_numpy())
        lats = ds["latitude"].to_numpy().astype(float)
        lons = ds["longitude"].to_numpy().astype(float)
        lo = int(months.searchsorted(pd.Timestamp(start)))
        hi = int(months.searchsorted(pd.Timestamp(end), side="right"))
        win_months = months[lo:hi]
        decades = to_decimal_decades(win_months.to_series()).to_numpy()
        n_possible = len(pd.period_range(start, end, freq="M"))
        min_obs = int(np.ceil(min_coverage * n_possible))

        n_lat, n_lon = lats.size, lons.size
        slopes = np.full((n_lat, n_lon), np.nan, dtype=float)
        for b0 in range(0, n_lat, lat_chunk):
            b1 = min(b0 + lat_chunk, n_lat)
            band_mask = country_mask[b0:b1]
            if not band_mask.any():  # all-ocean band: skip the read entirely
                continue
            slab = da.isel(
                time=slice(lo, hi), latitude=slice(b0, b1)
            ).to_numpy()  # (n_window, b1-b0, n_lon)
            for i in range(b1 - b0):
                for j in range(n_lon):
                    if band_mask[i, j] is None:
                        continue
                    col = slab[:, i, j]
                    finite = np.isfinite(col)
                    if int(finite.sum()) < min_obs:
                        continue
                    slopes[b0 + i, j] = float(
                        stats.theilslopes(col[finite], decades[finite]).slope
                    )
    finally:
        ds.close()
    return lats, lons, slopes


def area_weighted_country_trends(
    nc_path: Path = BERKELEY_GRID_PATH,
    gpw_path: Path = GPW_PATH,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    lat_chunk: int = DEFAULT_LAT_CHUNK,
    lookup_path: Path = GPW_NATID_LOOKUP_PATH,
) -> pd.DataFrame:
    """cos(latitude) area-weighted warming slope per country, off the grid.

    Assigns every Berkeley land cell to a country (:func:`assign_cell_iso3`),
    fits a per-cell trend (:func:`cell_trends`), then reduces each country's cells
    to a single mean weighted by ``cos(lat)``
    (:func:`src.population.latitude_area_weights` -- the intensive-field weighting;
    see the module docstring on why this is the mirror of the GPW count rule).
    Countries with no successfully-fit cell are omitted.

    Args:
        nc_path: Berkeley gridded NetCDF.
        gpw_path: GPW NetCDF carrying the band-11 national identifier grid.
        start, end: analysis-window bounds (the station window).
        min_coverage: per-cell finite-month coverage gate.
        lat_chunk: latitude rows per streamed slab.
        lookup_path: GPW national-identifier lookup table.

    Returns:
        One row per ISO3 country, sorted by ISO3, with columns
        :data:`AREA_COLUMNS`: ``iso3``, ``trend_c_per_decade_area_weighted`` and
        ``area_cell_coverage`` (fraction of the country's assigned land cells that
        were successfully fit).
    """
    lats, lons = _grid_coords(nc_path)
    lookup = load_national_id_lookup(lookup_path)
    mask = assign_cell_iso3(lats, lons, gpw_path, lookup)
    lats, lons, slopes = cell_trends(
        nc_path, mask, start=start, end=end,
        min_coverage=min_coverage, lat_chunk=lat_chunk,
    )
    return reduce_cells_to_country(
        mask, lats, slopes,
        value_col=AREA_WEIGHTED_COL, coverage_col=AREA_COVERAGE_COL,
    )


def reduce_cells_to_country(
    mask: np.ndarray,
    lats: np.ndarray,
    slopes: np.ndarray,
    *,
    value_col: str,
    coverage_col: str,
) -> pd.DataFrame:
    """Reduce per-cell slopes to one cos(lat) area-weighted mean per ISO3 country.

    Shared by the Berkeley (:func:`area_weighted_country_trends`) and ERA5
    (:mod:`src.era5_weighting`) paths so both apply the identical intensive-field
    weighting (:func:`src.population.latitude_area_weights`) and coverage
    accounting. Countries with no successfully-fit cell are omitted.

    Args:
        mask: ``(n_lat, n_lon)`` ISO3 assignment from :func:`assign_cell_iso3`.
        lats: 1D grid latitudes aligned to ``slopes``' first axis.
        slopes: ``(n_lat, n_lon)`` per-cell slopes (NaN where unfit).
        value_col: name of the emitted area-weighted-slope column.
        coverage_col: name of the emitted cell-coverage-fraction column.

    Returns:
        One row per ISO3 (sorted), columns ``(iso3, value_col, coverage_col)``.
    """
    lat_grid = np.broadcast_to(np.asarray(lats, dtype=float)[:, None], slopes.shape)
    rows = []
    for iso in sorted({v for v in mask.ravel() if v is not None}):
        assigned = mask == iso
        valid = assigned & np.isfinite(slopes)
        n_assigned = int(assigned.sum())
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        weights = latitude_area_weights(lat_grid[valid])
        trend = float(np.dot(weights, slopes[valid]))
        rows.append(
            {ISO3_COL: iso, value_col: trend, coverage_col: n_valid / n_assigned}
        )
    return pd.DataFrame(rows, columns=[ISO3_COL, value_col, coverage_col])


def land_mean_from_slopes(lats: np.ndarray, slopes: np.ndarray) -> float:
    """cos(lat) area-weighted mean of all finite cells (no country grouping).

    Shared by :func:`world_land_mean` and the ERA5 world-land sanity helper.
    """
    lat_grid = np.broadcast_to(np.asarray(lats, dtype=float)[:, None], slopes.shape)
    valid = np.isfinite(slopes)
    if not valid.any():
        return float("nan")
    weights = latitude_area_weights(lat_grid[valid])
    return float(np.dot(weights, slopes[valid]))


def world_land_mean(nc_path: Path = BERKELEY_GRID_PATH, **kwargs) -> float:
    """cos(lat) area-weighted world land-mean slope (the headline sanity check).

    Fits every land cell and area-weights with no country grouping; the result
    should sit near the station global mean (~0.146 °C/decade) -- a modest
    sampling shift, not a different planet. Accepts the same keyword arguments as
    :func:`area_weighted_country_trends` (``gpw_path``, ``start``, ``end``,
    ``min_coverage``, ``lat_chunk``, ``lookup_path``).
    """
    gpw_path = kwargs.pop("gpw_path", GPW_PATH)
    lookup_path = kwargs.pop("lookup_path", GPW_NATID_LOOKUP_PATH)
    lats, lons = _grid_coords(nc_path)
    mask = assign_cell_iso3(lats, lons, gpw_path, load_national_id_lookup(lookup_path))
    lats, lons, slopes = cell_trends(nc_path, mask, **kwargs)
    return land_mean_from_slopes(lats, slopes)
