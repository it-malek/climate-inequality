"""Per-city warming trends from deseasonalized temperature anomalies.

Phase 2 pipeline: compute each city-location's monthly climatology over a
baseline window (DuckDB), subtract it from observed monthly means to get
anomalies, gate on observation coverage, and fit a robust Theil-Sen trend
(°C/decade, 95% CI) plus an OLS slope per location. Heavy filtering and
aggregation stay in DuckDB; only window-filtered, per-location results are
pulled into pandas.

City identity: (City, Country) is NOT unique in the Berkeley Earth data —
18 same-named pairs exist at 2-3 grid coordinates each — so all grouping
here uses the full (City, Country, Latitude, Longitude) key.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import duckdb
import pandas as pd
from scipy import stats

from src.cleaning import DEFAULT_END, DEFAULT_START, coverage_by_city
from src.data_io import CITY_TABLE, DEFAULT_DB_PATH, PROCESSED_DIR

logger = logging.getLogger(__name__)

# Climatology baseline — see README "Known dataset quirks"
BASELINE_START = "1951-01-01"
BASELINE_END = "1980-12-01"
DEFAULT_MIN_COVERAGE = 0.9
DEFAULT_TRENDS_PATH = PROCESSED_DIR / "city_trends.parquet"

# Full per-location identity; (City, Country) alone is ambiguous (see above).
CITY_KEYS = ["City", "Country", "Latitude", "Longitude"]

TRENDS_COLUMNS = [
    *CITY_KEYS,
    "n_obs",
    "coverage",
    "slope_c_per_decade",
    "ci_low",
    "ci_high",
    "ols_slope",
    "baseline_window",
    "analysis_window",
]


def _safe_table(table: str) -> str:
    """Validate `table` as a plain SQL identifier safe to interpolate."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(f"invalid table name: {table!r}")
    return table


def compute_climatology(
    con: duckdb.DuckDBPyConnection,
    table: str = CITY_TABLE,
    baseline_start: str = BASELINE_START,
    baseline_end: str = BASELINE_END,
) -> pd.DataFrame:
    """Per-location mean temperature for each calendar month of a baseline.

    Args:
        con: Connection to a database holding `table` (schema as produced
            by :func:`src.data_io.load_city_temperatures`).
        table: Source table name.
        baseline_start: First baseline month (inclusive, "YYYY-MM-DD").
        baseline_end: Last baseline month (inclusive).

    Returns:
        Tidy frame keyed on (City, Country, Latitude, Longitude, Month)
        with a `climatology` column (°C). Months with no non-null baseline
        observation for a location have no row.
    """
    sql = f"""
        SELECT
            City,
            Country,
            Latitude,
            Longitude,
            month(dt) AS Month,
            avg(AverageTemperature) AS climatology
        FROM {_safe_table(table)}
        WHERE dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          AND AverageTemperature IS NOT NULL
        GROUP BY City, Country, Latitude, Longitude, Month
        ORDER BY City, Country, Latitude, Longitude, Month
    """
    return con.execute(sql, [baseline_start, baseline_end]).df()


def compute_anomalies(
    con: duckdb.DuckDBPyConnection,
    table: str = CITY_TABLE,
    baseline_start: str = BASELINE_START,
    baseline_end: str = BASELINE_END,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> pd.DataFrame:
    """Monthly anomalies vs climatology for locations passing coverage.

    Anomaly = observed monthly mean minus the location's climatology for
    that calendar month (:func:`compute_climatology`). The coverage gate is
    :func:`src.cleaning.coverage_by_city` keyed on the full (City, Country,
    Latitude, Longitude) identity — (City, Country) alone would pool the 18
    same-named city pairs and double-count their coverage.

    Args:
        con: Connection to a database holding `table`.
        table: Source table name.
        baseline_start: Climatology baseline start (inclusive).
        baseline_end: Climatology baseline end (inclusive).
        start: Analysis window start (inclusive).
        end: Analysis window end (inclusive).
        min_coverage: Minimum fraction of window months with non-null
            observations for a location to be kept.

    Returns:
        One row per (City, Country, Latitude, Longitude, dt) with an
        `anomaly` column (°C), for kept locations only.
    """
    climatology = compute_climatology(
        con, table=table, baseline_start=baseline_start, baseline_end=baseline_end
    )
    sql = f"""
        SELECT
            t.City,
            t.Country,
            t.Latitude,
            t.Longitude,
            t.dt,
            t.AverageTemperature,
            t.AverageTemperature - c.climatology AS anomaly
        FROM {_safe_table(table)} t
        LEFT JOIN _climatology c
            ON t.City = c.City
            AND t.Country = c.Country
            AND t.Latitude = c.Latitude
            AND t.Longitude = c.Longitude
            AND month(t.dt) = c.Month
        WHERE t.dt BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          AND t.AverageTemperature IS NOT NULL
        ORDER BY t.City, t.Country, t.Latitude, t.Longitude, t.dt
    """
    con.register("_climatology", climatology)
    try:
        obs = con.execute(sql, [start, end]).df()
    finally:
        con.unregister("_climatology")

    # Two string columns dominate the multi-million-row pull; categoricals
    # cut that memory severalfold and speed up the groupbys below.
    for col in ("City", "Country"):
        obs[col] = obs[col].astype("category")

    coverage = coverage_by_city(
        obs,
        min_fraction=min_coverage,
        start=start,
        end=end,
        group_keys=tuple(CITY_KEYS),
    )
    n_total = len(coverage)
    n_kept = int(coverage["keep"].sum())
    logger.info(
        "coverage >= %.0f%%: kept %d of %d city-locations, excluded %d",
        min_coverage * 100,
        n_kept,
        n_total,
        n_total - n_kept,
    )

    kept = obs.merge(coverage.loc[coverage["keep"], CITY_KEYS], on=CITY_KEYS)
    n_no_climatology = int(kept["anomaly"].isna().sum())
    if n_no_climatology:
        logger.warning(
            "dropping %d observation(s) with no climatology for their month",
            n_no_climatology,
        )
        kept = kept.loc[kept["anomaly"].notna()]
    return kept[[*CITY_KEYS, "dt", "anomaly"]].reset_index(drop=True)


def fit_city_trends(
    anomalies: pd.DataFrame,
    date_col: str = "dt",
    value_col: str = "anomaly",
) -> pd.DataFrame:
    """Theil-Sen and OLS warming trends per city-location, in °C/decade.

    Time is expressed in decimal decades, so fitted slopes are directly in
    °C/decade; `ci_low`/`ci_high` are the Theil-Sen 95% confidence bounds.
    Each location's fit is independent, so this loops over per-location
    numpy arrays rather than vectorizing across locations.

    Args:
        anomalies: Output of :func:`compute_anomalies` (no null values).
        date_col: Observation date column.
        value_col: Anomaly column to fit.

    Returns:
        One row per (City, Country, Latitude, Longitude) with n_obs,
        slope_c_per_decade, ci_low, ci_high, ols_slope.
    """
    dates = pd.to_datetime(anomalies[date_col])
    decades = (dates.dt.year + (dates.dt.month - 0.5) / 12) / 10
    work = anomalies.assign(_decades=decades)

    rows = []
    for keys, grp in work.groupby(CITY_KEYS, observed=True, sort=True):
        x = grp["_decades"].to_numpy()
        y = grp[value_col].to_numpy()
        theil = stats.theilslopes(y, x, alpha=0.95)
        ols = stats.linregress(x, y)
        rows.append(
            (*keys, len(y), theil.slope, theil.low_slope, theil.high_slope, ols.slope)
        )
    return pd.DataFrame(
        rows,
        columns=[
            *CITY_KEYS,
            "n_obs",
            "slope_c_per_decade",
            "ci_low",
            "ci_high",
            "ols_slope",
        ],
    )


def _write_trends_parquet(trends: pd.DataFrame, out_path: Path) -> None:
    """Overwrite `out_path` with an explicitly-typed parquet file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("_trends", trends)
        # COPY targets cannot take prepared parameters; inline the path as
        # an escaped SQL string literal (same approach as data_io).
        out_literal = str(out_path).replace("'", "''")
        con.execute(
            f"""
            COPY (
                SELECT
                    City::VARCHAR AS City,
                    Country::VARCHAR AS Country,
                    Latitude::DOUBLE AS Latitude,
                    Longitude::DOUBLE AS Longitude,
                    n_obs::BIGINT AS n_obs,
                    coverage::DOUBLE AS coverage,
                    slope_c_per_decade::DOUBLE AS slope_c_per_decade,
                    ci_low::DOUBLE AS ci_low,
                    ci_high::DOUBLE AS ci_high,
                    ols_slope::DOUBLE AS ols_slope,
                    baseline_window::VARCHAR AS baseline_window,
                    analysis_window::VARCHAR AS analysis_window
                FROM _trends
                ORDER BY Country, City, Latitude, Longitude
            ) TO '{out_literal}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def build_city_trends(
    db_path: Path = DEFAULT_DB_PATH,
    out_path: Path = DEFAULT_TRENDS_PATH,
    table: str = CITY_TABLE,
    baseline_start: str = BASELINE_START,
    baseline_end: str = BASELINE_END,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> pd.DataFrame:
    """Run the Phase 2 pipeline end to end and write city_trends.parquet.

    Opens `db_path` read-only, computes anomalies for locations passing the
    coverage gate, fits per-location trends, and deterministically
    overwrites `out_path` (re-running replaces the file, never appends).

    Args:
        db_path: DuckDB database containing `table`.
        out_path: Destination parquet file.
        table: Source table name.
        baseline_start: Climatology baseline start (inclusive).
        baseline_end: Climatology baseline end (inclusive).
        start: Analysis window start (inclusive).
        end: Analysis window end (inclusive).
        min_coverage: Minimum per-location observation coverage.

    Returns:
        The trends frame as written, one row per kept city-location, with
        columns `TRENDS_COLUMNS`.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        anomalies = compute_anomalies(
            con,
            table=table,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            start=start,
            end=end,
            min_coverage=min_coverage,
        )
    finally:
        con.close()

    trends = fit_city_trends(anomalies)
    n_possible = len(pd.period_range(start, end, freq="M"))
    trends["coverage"] = trends["n_obs"] / n_possible
    trends["baseline_window"] = f"{baseline_start}..{baseline_end}"
    trends["analysis_window"] = f"{start}..{end}"
    trends = trends[TRENDS_COLUMNS]

    _write_trends_parquet(trends, out_path)
    logger.info("wrote %d city-location trends to %s", len(trends), out_path)
    return trends


def main() -> None:
    """Build the default trends table and print README sanity checks."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    trends = build_city_trends()
    global_mean = trends["slope_c_per_decade"].mean()
    arctic = trends.loc[trends["Latitude"] > 60.0, "slope_c_per_decade"]
    print(f"city-locations fit: {len(trends)}")
    print(f"global mean slope: {global_mean:.3f} °C/decade (expect ~0.1-0.2)")
    print(f">60°N mean slope ({len(arctic)} locations): {arctic.mean():.3f} °C/decade")
    print(f"Arctic amplification ratio: {arctic.mean() / global_mean:.2f}x (expect ~2)")


if __name__ == "__main__":
    main()
