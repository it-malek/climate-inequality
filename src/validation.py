"""Out-of-sample validation of the stored trends (Phase 6).

Every trend in this project is fit on city data ending 2013-09 (the
analysis window stored in ``city_trends.parquet``). The Berkeley Earth
gridded land dataset -- same provider, monthly anomalies relative to the
same 1951-1980 baseline as :data:`src.trends.BASELINE_START` -- extends
past 2024, so the stored Theil-Sen lines can be scored against
observations they never saw:

1. **Overlap agreement** (:func:`overlap_agreement`): per city-location,
   Pearson r / RMSE between the pipeline anomalies and the nearest
   1°x1° grid cell over the analysis window. Cities whose cell does not
   track them (islands and coastlines, where a 1° land average is not
   the city's series) are gated out of the scoring aggregates.
2. **Forecast residuals** (:func:`forecast_residuals`): observed grid
   anomaly minus the stored line (slope x decades + intercept, both
   from the committed bundle -- never refit) from 2013-10 onward.
   Systematically positive residuals mean the 1950-2013 lines
   underpredict observed warming, i.e. acceleration.
3. **Acceleration** (:func:`acceleration`): Theil-Sen slope refit on
   the grid series through the present vs the stored 1950-2013 slope.

Inputs are the committed dashboard bundle (``app/data/``), whose
intercepts were slope-verified at build time by
:func:`src.app_assets.theil_sen_intercepts`, plus the gridded NetCDF in
``data/raw/berkeley_gridded/`` (downloaded on first run, ~190 MB).

Dataset quirk (do not rediscover): the NetCDF time axis is fractional
decimal years in Berkeley's mid-month convention (2014.041666... =
January 2014) with units ``"year A.D."``, which CF decoding cannot
parse -- see :func:`decode_fractional_years`. Layout:
``temperature(time, latitude, longitude)`` float32 anomalies in °C,
NaN over ocean; the separate ``climatology`` variable is unused because
the pipeline also works in anomaly space on the same baseline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from src.app_assets import ANOMALIES_ASSET, APP_DATA_DIR, TRENDS_ASSET
from src.cleaning import parse_window, to_decimal_decades
from src.data_io import (
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    download_file,
    write_typed_parquet,
)
from src.figures import render_residual_map, render_validation_series
from src.grids import nearest_cell_indices
from src.trends import CITY_KEYS

logger = logging.getLogger(__name__)

# Verified June 2026: this is the link on the berkeleyearth.org data page
# (Gridded -> Monthly Land -> 1° x 1°).
GRIDDED_URL = (
    "https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com"
    "/Global/Gridded/Complete_TAVG_LatLong1.nc"
)
DEFAULT_GRIDDED_PATH = RAW_DIR / "berkeley_gridded" / "Complete_TAVG_LatLong1.nc"
DEFAULT_VALIDATION_PATH = PROCESSED_DIR / "validation.parquet"

# First out-of-sample month: the stored analysis window ends 2013-09.
DEFAULT_FORECAST_START = "2013-10-01"
# Cities whose grid cell correlates below this over the analysis window
# are reported but excluded from the scoring aggregates.
DEFAULT_MIN_OVERLAP_R = 0.8
# The 2023-24 El Nino spike is large enough to dominate an ~11-year mean,
# so residuals are reported both with and without months from here on.
EL_NINO_START = "2023-01-01"

GRID_VAR = "temperature"
GRID_DIMS = ("time", "latitude", "longitude")
# One decompressed time slab (240 x 180 x 360 float32) is ~62 MB: the full
# variable is ~850 MB, which does not fit comfortably in RAM everywhere.
TIME_CHUNK_MONTHS = 240

RESIDUAL_MAP_FIGURE = "validation_residual_map.html"
GLOBAL_SERIES_FIGURE = "validation_global_series.html"

# On-disk schema of validation.parquet (DuckDB types), in column order.
VALIDATION_SCHEMA = {
    "City": "VARCHAR",
    "Country": "VARCHAR",
    "Latitude": "DOUBLE",
    "Longitude": "DOUBLE",
    "city_id": "INTEGER",
    "grid_lat": "DOUBLE",
    "grid_lon": "DOUBLE",
    "n_overlap": "BIGINT",
    "overlap_r": "DOUBLE",
    "overlap_rmse": "DOUBLE",
    "overlap_bias": "DOUBLE",
    "gate_pass": "BOOLEAN",
    "n_forecast": "BIGINT",
    "mean_residual": "DOUBLE",
    "mean_residual_pre2023": "DOUBLE",
    "slope_c_per_decade": "DOUBLE",
    "slope_full": "DOUBLE",
    "slope_full_ci_low": "DOUBLE",
    "slope_full_ci_high": "DOUBLE",
    "slope_overlap_grid": "DOUBLE",
    "slope_delta": "DOUBLE",
    "forecast_start": "VARCHAR",
    "record_end": "VARCHAR",
}
VALIDATION_COLUMNS = list(VALIDATION_SCHEMA)


def decode_fractional_years(values: Sequence[float] | np.ndarray) -> pd.DatetimeIndex:
    """Convert Berkeley Earth fractional decimal years to month timestamps.

    The gridded files encode time as ``year + (month - 0.5) / 12``
    (mid-month decimals, e.g. 2014.041666... = January 2014) -- the same
    convention as :func:`src.cleaning.to_decimal_decades`, in years
    instead of decades. xarray's CF decoding cannot parse it, so the
    axis is decoded manually to first-of-month timestamps matching the
    pipeline's ``dt`` column.

    Args:
        values: Fractional-year floats.

    Returns:
        DatetimeIndex of first-of-month timestamps.

    Raises:
        ValueError: if any value does not sit on the mid-month grid
            (e.g. a start-of-month ``year + (month - 1) / 12`` axis).
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


def sample_grid_series(
    nc_path: Path,
    trends: pd.DataFrame,
    start: str | None = None,
    chunk_months: int = TIME_CHUNK_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nearest-cell monthly grid anomaly series for each city-location.

    Reads the temperature variable in `chunk_months`-sized time slabs and
    keeps only the sampled city columns, so peak memory stays around one
    slab plus the (months x cities) float32 result -- the full variable
    decompressed would be ~850 MB.

    Args:
        nc_path: Path to ``Complete_TAVG_LatLong1.nc`` (or a synthetic
            file with the same layout).
        trends: One row per city-location with ``city_id``, ``Latitude``,
            ``Longitude``.
        start: If given, months before this date are skipped at read time
            (the grid reaches back to 1750; the pipeline needs 1950 on).
        chunk_months: Time-slab size per read.

    Returns:
        (wide, cells): `wide` is a float32 frame indexed by month
        (first-of-month, via :func:`decode_fractional_years`) with one
        column per ``city_id``; NaN where the cell has no data (ocean
        cells for island cities are all-NaN). `cells` maps ``city_id``
        to the matched ``grid_lat``/``grid_lon`` cell centers.

    Raises:
        ValueError: if the file lacks ``temperature(time, latitude,
            longitude)`` or a coordinate axis is not monotonic.
    """
    ds = xr.open_dataset(nc_path, decode_times=False)
    try:
        if GRID_VAR not in ds.data_vars:
            raise ValueError(
                f"{nc_path} has no {GRID_VAR!r} variable "
                f"(found {sorted(ds.data_vars)})"
            )
        da = ds[GRID_VAR]
        if tuple(da.dims) != GRID_DIMS:
            raise ValueError(
                f"{GRID_VAR} has dims {tuple(da.dims)}, expected {GRID_DIMS}"
            )
        months = decode_fractional_years(ds["time"].to_numpy())
        lat_index = pd.Index(ds["latitude"].to_numpy().astype(float))
        lon_index = pd.Index(ds["longitude"].to_numpy().astype(float))
        lat_pos, lon_pos = nearest_cell_indices(
            lat_index, lon_index,
            trends["Latitude"].to_numpy(), trends["Longitude"].to_numpy(),
        )

        first = int(months.searchsorted(pd.Timestamp(start))) if start else 0
        blocks = []
        for lo in range(first, len(months), chunk_months):
            slab = da.isel(time=slice(lo, lo + chunk_months)).to_numpy()
            blocks.append(slab[:, lat_pos, lon_pos].astype(np.float32, copy=False))
        if blocks:
            values = np.vstack(blocks)
        else:
            values = np.empty((0, len(trends)), dtype=np.float32)
    finally:
        ds.close()

    city_ids = trends["city_id"].to_numpy()
    wide = pd.DataFrame(values, index=months[first:], columns=city_ids)
    wide.index.name = "dt"
    wide.columns.name = "city_id"
    cells = pd.DataFrame(
        {
            "city_id": city_ids,
            "grid_lat": lat_index.to_numpy()[lat_pos],
            "grid_lon": lon_index.to_numpy()[lon_pos],
        }
    )
    logger.info(
        "sampled %d cells over %d months (%s..%s)",
        len(cells),
        len(wide),
        f"{wide.index.min():%Y-%m}" if len(wide) else "-",
        f"{wide.index.max():%Y-%m}" if len(wide) else "-",
    )
    return wide, cells


def _column_nanmean(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means over finite entries: (mean with NaN where empty, count)."""
    finite = np.isfinite(matrix)
    count = finite.sum(axis=0)
    total = np.where(finite, matrix, 0.0).sum(axis=0)
    mean = np.where(count > 0, total / np.where(count > 0, count, 1), np.nan)
    return mean, count


def overlap_agreement(
    pipeline_wide: pd.DataFrame,
    grid_wide: pd.DataFrame,
    start: str,
    end: str,
    min_r: float = DEFAULT_MIN_OVERLAP_R,
) -> pd.DataFrame:
    """Per-city agreement between pipeline and grid anomalies in a window.

    Both inputs are wide (month x city_id) frames; only months present in
    both within [start, end] and finite on both sides count. Pearson r is
    the gate statistic: it ignores levels, so a cell whose climatology
    sits slightly off the city's does not fail a city that tracks it --
    any level offset shows up in ``overlap_bias`` instead.

    Args:
        pipeline_wide: Bundle anomalies pivoted to month x city_id.
        grid_wide: Output of :func:`sample_grid_series`.
        start: Window start (inclusive), normally the analysis window.
        end: Window end (inclusive).
        min_r: Gate threshold on Pearson r.

    Returns:
        One row per city_id: n_overlap, overlap_r, overlap_rmse,
        overlap_bias (mean grid - pipeline), gate_pass (r >= min_r;
        an undefined r -- e.g. an all-ocean cell -- fails).
    """
    common = pipeline_wide.index.intersection(grid_wide.index)
    common = common[
        (common >= pd.Timestamp(start)) & (common <= pd.Timestamp(end))
    ]
    a = pipeline_wide.loc[common].to_numpy(dtype=float)
    b = grid_wide.reindex(index=common, columns=pipeline_wide.columns).to_numpy(
        dtype=float
    )

    valid = np.isfinite(a) & np.isfinite(b)
    n = valid.sum(axis=0)
    safe_n = np.where(n > 0, n, 1)
    a0 = np.where(valid, a, 0.0)
    b0 = np.where(valid, b, 0.0)
    mean_a = a0.sum(axis=0) / safe_n
    mean_b = b0.sum(axis=0) / safe_n
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (a0 * b0).sum(axis=0) / safe_n - mean_a * mean_b
        var_a = (a0**2).sum(axis=0) / safe_n - mean_a**2
        var_b = (b0**2).sum(axis=0) / safe_n - mean_b**2
        r = cov / np.sqrt(var_a * var_b)
    undefined = (n < 2) | (var_a <= 0) | (var_b <= 0)
    r = np.where(undefined, np.nan, r)
    rmse = np.sqrt(((a0 - b0) ** 2).sum(axis=0) / safe_n)
    bias = (b0 - a0).sum(axis=0) / safe_n
    rmse = np.where(n > 0, rmse, np.nan)
    bias = np.where(n > 0, bias, np.nan)

    return pd.DataFrame(
        {
            "city_id": pipeline_wide.columns.to_numpy(),
            "n_overlap": n.astype(np.int64),
            "overlap_r": r,
            "overlap_rmse": rmse,
            "overlap_bias": bias,
            "gate_pass": np.where(np.isnan(r), False, r >= min_r).astype(bool),
        }
    )


def forecast_residuals(
    trends: pd.DataFrame,
    grid_wide: pd.DataFrame,
    forecast_start: str = DEFAULT_FORECAST_START,
    el_nino_start: str = EL_NINO_START,
    gate: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the stored Theil-Sen lines against post-window observations.

    The predicted anomaly for every grid month is ``slope x decades +
    intercept`` on the :func:`src.cleaning.to_decimal_decades` axis,
    using the stored slope and the bundle's slope-verified intercept --
    never a refit. Residual = observed - predicted.

    Args:
        trends: Bundle trends with ``city_id``, ``slope_c_per_decade``,
            ``intercept``.
        grid_wide: Output of :func:`sample_grid_series`.
        forecast_start: First out-of-sample month (inclusive).
        el_nino_start: Months from here on are excluded from
            ``mean_residual_pre2023`` (El Nino sensitivity check).
        gate: Optional boolean Series indexed by city_id; the global
            monthly aggregate averages only gate-passing cities
            (per-city rows are always computed for every city).

    Returns:
        (per_city, global_monthly): `per_city` has city_id, n_forecast,
        mean_residual, mean_residual_pre2023. `global_monthly` has one
        row per sampled month over the whole record: dt, observed,
        predicted (means across scored cities, each restricted to its
        observed months so the two series average the same data), and
        n_cities.
    """
    fits = trends.set_index("city_id").reindex(grid_wide.columns)
    decades = to_decimal_decades(grid_wide.index.to_series()).to_numpy()
    predicted = (
        np.outer(decades, fits["slope_c_per_decade"].to_numpy())
        + fits["intercept"].to_numpy()
    )
    observed = grid_wide.to_numpy(dtype=float)
    residual = observed - predicted

    months = grid_wide.index
    in_forecast = np.asarray(months >= pd.Timestamp(forecast_start))
    pre_el_nino = in_forecast & np.asarray(months < pd.Timestamp(el_nino_start))
    mean_residual, n_forecast = _column_nanmean(residual[in_forecast])
    mean_residual_pre, _ = _column_nanmean(residual[pre_el_nino])
    per_city = pd.DataFrame(
        {
            "city_id": grid_wide.columns.to_numpy(),
            "n_forecast": n_forecast.astype(np.int64),
            "mean_residual": mean_residual,
            "mean_residual_pre2023": mean_residual_pre,
        }
    )

    if gate is None:
        scored = np.ones(len(grid_wide.columns), dtype=bool)
    else:
        scored = (
            gate.reindex(grid_wide.columns).fillna(False).to_numpy(dtype=bool)
        )
    finite = np.isfinite(observed[:, scored])
    n_cities = finite.sum(axis=1)
    safe_n = np.where(n_cities > 0, n_cities, 1)
    observed_mean = np.where(finite, observed[:, scored], 0.0).sum(axis=1) / safe_n
    predicted_mean = np.where(finite, predicted[:, scored], 0.0).sum(axis=1) / safe_n
    empty = n_cities == 0
    global_monthly = pd.DataFrame(
        {
            "dt": months,
            "observed": np.where(empty, np.nan, observed_mean),
            "predicted": np.where(empty, np.nan, predicted_mean),
            "n_cities": n_cities.astype(np.int64),
        }
    )
    return per_city, global_monthly


def acceleration(
    trends: pd.DataFrame,
    grid_wide: pd.DataFrame,
    start: str,
    end: str,
    min_obs: int = 120,
) -> pd.DataFrame:
    """Refit Theil-Sen on the grid series and compare to stored slopes.

    Two fits per city on its grid-cell anomalies (the theilslopes call
    pattern of :func:`src.trends.fit_city_trends`): the full record from
    `start` onward (``slope_full``, with 95% CI) and the stored analysis
    window [start, end] only (``slope_overlap_grid`` -- the
    apples-to-apples check that the cell reproduces the stored slope
    before the post-window years enter). ``slope_delta`` is
    ``slope_full`` minus the stored slope.

    Args:
        trends: Bundle trends with ``city_id`` and ``slope_c_per_decade``.
        grid_wide: Output of :func:`sample_grid_series`.
        start: Analysis window start (inclusive); earlier grid months
            are ignored so the comparison shares its starting point.
        end: Analysis window end (inclusive).
        min_obs: Minimum finite months required for each fit; below it
            the city's fits are NaN.

    Returns:
        One row per city_id: slope_full, slope_full_ci_low/high,
        slope_overlap_grid, slope_delta (°C/decade).
    """
    months = grid_wide.index
    decades = to_decimal_decades(months.to_series()).to_numpy()
    in_full = np.asarray(months >= pd.Timestamp(start))
    in_window = in_full & np.asarray(months <= pd.Timestamp(end))
    values = grid_wide.to_numpy(dtype=float)
    stored = (
        trends.set_index("city_id")["slope_c_per_decade"]
        .reindex(grid_wide.columns)
        .to_numpy()
    )

    rows = []
    for j in range(values.shape[1]):
        finite = np.isfinite(values[:, j])
        full = finite & in_full
        window = finite & in_window
        if full.sum() < min_obs or window.sum() < min_obs:
            rows.append((np.nan, np.nan, np.nan, np.nan))
            continue
        fit = stats.theilslopes(values[full, j], decades[full], alpha=0.95)
        overlap = stats.theilslopes(values[window, j], decades[window])
        rows.append((fit.slope, fit.low_slope, fit.high_slope, overlap.slope))
    out = pd.DataFrame(
        rows,
        columns=[
            "slope_full",
            "slope_full_ci_low",
            "slope_full_ci_high",
            "slope_overlap_grid",
        ],
    )
    out.insert(0, "city_id", grid_wide.columns.to_numpy())
    out["slope_delta"] = out["slope_full"] - stored
    return out


def run_validation(
    nc_path: Path = DEFAULT_GRIDDED_PATH,
    bundle_dir: Path = APP_DATA_DIR,
    out_path: Path = DEFAULT_VALIDATION_PATH,
    figures_dir: Path = OUTPUTS_DIR,
    forecast_start: str = DEFAULT_FORECAST_START,
    min_overlap_r: float = DEFAULT_MIN_OVERLAP_R,
) -> dict:
    """Run the Phase 6 validation end to end and write its outputs.

    Reads the committed bundle (trends with slope-verified intercepts,
    plus the per-city anomalies), samples the gridded NetCDF at each
    city-location, gates on overlap agreement, scores forecast residuals
    from the stored lines, refits full-record slopes, and writes
    ``validation.parquet`` plus the residual-map and global-series
    figures.

    Args:
        nc_path: The gridded NetCDF (see :data:`GRIDDED_URL`).
        bundle_dir: The committed ``app/data/`` bundle directory.
        out_path: Destination parquet (deterministically overwritten).
        figures_dir: Where the two HTML figures are written.
        forecast_start: First out-of-sample month (inclusive).
        min_overlap_r: Agreement gate threshold.

    Returns:
        Dict with `frame` (the parquet contents), `global_monthly`,
        `stats` (scoring aggregates over gate-passing cities), `paths`.

    Raises:
        ValueError: if the bundle lacks required columns (stale bundle)
            or the gridded record ends before `forecast_start`.
    """
    trends = pd.read_parquet(bundle_dir / TRENDS_ASSET)
    required = {"city_id", "intercept", "slope_c_per_decade", "analysis_window"}
    required.update(CITY_KEYS)
    missing = sorted(required - set(trends.columns))
    if missing:
        raise ValueError(
            f"{bundle_dir / TRENDS_ASSET} is missing column(s) {missing}; "
            "rebuild the bundle (python -m src.app_assets)"
        )
    anomalies = pd.read_parquet(bundle_dir / ANOMALIES_ASSET)
    start, end = parse_window(str(trends["analysis_window"].iloc[0]))

    logger.info("sampling %d city cells from %s ...", len(trends), nc_path.name)
    grid_wide, cells = sample_grid_series(nc_path, trends, start=start)
    has_data = grid_wide.notna().any(axis=1).to_numpy()
    if not has_data.any():
        raise ValueError(f"no finite grid data sampled from {nc_path}")
    record_end = grid_wide.index[has_data].max()
    if record_end <= pd.Timestamp(forecast_start):
        raise ValueError(
            f"gridded record ends {record_end:%Y-%m}, before the forecast "
            f"start {forecast_start}; nothing to validate against"
        )

    pipeline_wide = (
        anomalies.assign(dt=pd.to_datetime(anomalies["dt"]))
        .pivot(index="dt", columns="city_id", values="anomaly")
    )
    agreement = overlap_agreement(
        pipeline_wide, grid_wide, start, end, min_r=min_overlap_r
    )
    gate = agreement.set_index("city_id")["gate_pass"]
    logger.info(
        "overlap gate r>=%.2f: %d of %d pass (%d undefined r)",
        min_overlap_r,
        int(gate.sum()),
        len(gate),
        int(agreement["overlap_r"].isna().sum()),
    )

    per_city, global_monthly = forecast_residuals(
        trends, grid_wide, forecast_start=forecast_start, gate=gate
    )
    logger.info(
        "refitting Theil-Sen on %d grid series (two windows each) ...",
        len(trends),
    )
    accel = acceleration(trends, grid_wide, start, end)

    frame = (
        trends[[*CITY_KEYS, "city_id", "slope_c_per_decade"]]
        .merge(cells, on="city_id")
        .merge(agreement, on="city_id")
        .merge(per_city, on="city_id")
        .merge(accel, on="city_id")
        .assign(
            forecast_start=forecast_start,
            record_end=f"{record_end:%Y-%m-%d}",
        )
    )[VALIDATION_COLUMNS]
    write_typed_parquet(
        frame,
        out_path,
        VALIDATION_SCHEMA,
        order_by=("Country", "City", "Latitude", "Longitude"),
    )
    logger.info("wrote %d rows to %s", len(frame), out_path)

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "table": out_path,
        "residual_map": figures_dir / RESIDUAL_MAP_FIGURE,
        "global_series": figures_dir / GLOBAL_SERIES_FIGURE,
    }
    render_residual_map(frame).write_html(paths["residual_map"])
    render_validation_series(
        global_monthly, forecast_start=forecast_start
    ).write_html(paths["global_series"])

    scored = frame.loc[frame["gate_pass"]]
    delta = scored["slope_delta"].dropna()
    if len(delta) > 1:
        ci_half = 1.96 * float(delta.std(ddof=1)) / np.sqrt(len(delta))
    else:
        ci_half = float("nan")
    in_forecast = (grid_wide.index >= pd.Timestamp(forecast_start)) & (
        grid_wide.index <= record_end
    )
    stats_out = {
        "n_locations": int(len(frame)),
        "n_no_grid": int(frame["overlap_r"].isna().sum()),
        "n_gate_pass": int(frame["gate_pass"].sum()),
        "median_overlap_r": float(frame["overlap_r"].median()),
        "forecast_start": forecast_start,
        "record_end": f"{record_end:%Y-%m-%d}",
        "n_forecast_months": int(in_forecast.sum()),
        "mean_residual": float(scored["mean_residual"].mean()),
        "mean_residual_pre2023": float(scored["mean_residual_pre2023"].mean()),
        "mean_slope_stored": float(scored["slope_c_per_decade"].mean()),
        "mean_slope_overlap_grid": float(scored["slope_overlap_grid"].mean()),
        "mean_slope_full": float(scored["slope_full"].mean()),
        "mean_slope_delta": float(delta.mean()),
        "slope_delta_ci_low": float(delta.mean() - ci_half),
        "slope_delta_ci_high": float(delta.mean() + ci_half),
    }
    return {
        "frame": frame,
        "global_monthly": global_monthly,
        "stats": stats_out,
        "paths": paths,
    }


def main() -> None:
    """Run the default validation and print the checkpoint sanity block."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    download_file(GRIDDED_URL, DEFAULT_GRIDDED_PATH, timeout=300)
    out = run_validation()
    s = out["stats"]
    print(
        f"city-locations: {s['n_locations']} "
        f"({s['n_no_grid']} with no usable grid cell)"
    )
    print(
        f"overlap agreement: median r {s['median_overlap_r']:.3f}; "
        f"gate r>={DEFAULT_MIN_OVERLAP_R}: {s['n_gate_pass']} pass, "
        f"{s['n_locations'] - s['n_gate_pass']} excluded"
    )
    print(
        f"forecast window: {s['forecast_start'][:7]}..{s['record_end'][:7]} "
        f"({s['n_forecast_months']} months)"
    )
    print(
        f"mean forecast residual: {s['mean_residual']:+.3f} °C "
        f"(excl 2023+: {s['mean_residual_pre2023']:+.3f}) "
        "-- expect positive (acceleration)"
    )
    print(
        f"slopes °C/decade: stored {s['mean_slope_stored']:.3f}; grid, same "
        f"window {s['mean_slope_overlap_grid']:.3f}; grid, full record "
        f"{s['mean_slope_full']:.3f} (expect modestly above stored)"
    )
    print(
        f"acceleration: mean slope delta {s['mean_slope_delta']:+.4f} "
        f"[95% CI {s['slope_delta_ci_low']:+.4f}, "
        f"{s['slope_delta_ci_high']:+.4f}] °C/decade "
        "(city-level CI; spatial correlation makes it optimistic)"
    )
    print(f"table:   {out['paths']['table']}")
    print(f"figures: {out['paths']['residual_map']}")
    print(f"         {out['paths']['global_series']}")


if __name__ == "__main__":
    main()
