"""Data acquisition and DuckDB ingestion for the climate-inequality project.

Two responsibilities, side effects only at these edges:
  1. Download the raw Kaggle datasets into ``data/raw/`` (kagglehub).
  2. Load ``GlobalLandTemperaturesByCity.csv`` into a DuckDB database,
     applying the coordinate parsing from :mod:`src.cleaning` during
     ingestion so downstream phases only ever see signed-float lat/lon.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import kagglehub
import pandas as pd

from src.cleaning import parse_coordinate

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DB_PATH = PROCESSED_DIR / "climate.duckdb"

TEMPERATURE_DATASET = "berkeleyearth/climate-change-earth-surface-temperature-data"
EMISSIONS_DATASET = "srikantsahu/co2-and-ghg-emission-data"
CITY_CSV_NAME = "GlobalLandTemperaturesByCity.csv"

CITY_TABLE = "city_temps"


@dataclass(frozen=True)
class IngestResult:
    """Summary of one DuckDB ingestion run."""

    db_path: Path
    table: str
    n_rows: int
    n_cities: int
    dt_min: date
    dt_max: date


def _sync_tree(src_dir: Path, dest_dir: Path) -> Path:
    """Copy files from `src_dir` into `dest_dir`, skipping up-to-date ones.

    A file is considered up to date when it exists at the destination with
    the same size; this keeps repeated download calls cheap.

    Args:
        src_dir: Directory to copy from (e.g. the kagglehub cache).
        dest_dir: Directory to copy into; created if missing.

    Returns:
        `dest_dir`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(p for p in src_dir.rglob("*") if p.is_file()):
        dest = dest_dir / src.relative_to(src_dir)
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            logger.info("up to date: %s", dest.name)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        logger.info("copied: %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest_dir


def download_raw_data(
    dest_dir: Path = RAW_DIR,
    handles: tuple[str, ...] = (TEMPERATURE_DATASET, EMISSIONS_DATASET),
) -> dict[str, Path]:
    """Download the project's Kaggle datasets into `dest_dir`.

    Uses kagglehub, which downloads into its own cache and works without
    credentials for public datasets; files are then copied into one
    subdirectory of `dest_dir` per dataset (skipping up-to-date files).

    Args:
        dest_dir: Target directory, normally ``data/raw/``.
        handles: Kaggle dataset handles (``owner/dataset-slug``).

    Returns:
        Mapping of dataset slug -> local directory containing its files.
    """
    out: dict[str, Path] = {}
    for handle in handles:
        logger.info("downloading %s ...", handle)
        cache_dir = Path(kagglehub.dataset_download(handle))
        slug = handle.rsplit("/", 1)[1]
        out[slug] = _sync_tree(cache_dir, dest_dir / slug)
    return out


def city_csv_path(raw_dir: Path = RAW_DIR) -> Path:
    """Locate ``GlobalLandTemperaturesByCity.csv`` under `raw_dir`.

    Raises:
        FileNotFoundError: if the file is absent (run
            :func:`download_raw_data` first).
    """
    matches = sorted(raw_dir.rglob(CITY_CSV_NAME))
    if not matches:
        raise FileNotFoundError(
            f"{CITY_CSV_NAME} not found under {raw_dir}; "
            "run download_raw_data() first"
        )
    return matches[0]


def _coordinate_mapping(values: list[str], bound: float) -> pd.DataFrame:
    """Parse distinct coordinate strings via cleaning.parse_coordinate.

    Berkeley Earth coordinates are grid-snapped, so the distinct strings
    number in the hundreds — parsing only those (then joining back in SQL)
    reuses the ported parser without a per-row Python UDF over ~8.6M rows.

    Args:
        values: Distinct raw coordinate strings (e.g. ``"32.95N"``).
        bound: Physical bound to validate against (90 for lat, 180 for lon).

    Returns:
        DataFrame with columns ``raw`` (str) and ``parsed`` (float).
    """
    parsed = [parse_coordinate(v) for v in values]
    bad = [v for v, p in zip(values, parsed) if abs(p) > bound]
    if bad:
        raise ValueError(f"coordinates exceed ±{bound}: {bad[:5]}")
    return pd.DataFrame({"raw": values, "parsed": parsed})


def load_city_temperatures(
    csv_path: Path,
    db_path: Path = DEFAULT_DB_PATH,
    table: str = CITY_TABLE,
) -> IngestResult:
    """Load the by-city temperature CSV into a DuckDB table.

    Replaces `table` if it already exists (idempotent). ``Latitude`` and
    ``Longitude`` are converted from hemisphere-suffixed strings to signed
    floats during ingestion; ``dt`` becomes a DATE. Null temperatures are
    preserved — windowing and coverage filters belong to later phases.

    Args:
        csv_path: Path to ``GlobalLandTemperaturesByCity.csv``.
        db_path: DuckDB database file, created (with parents) if missing.
        table: Destination table name.

    Returns:
        IngestResult with row/city counts and the date range.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"no such CSV: {csv_path}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(f"invalid table name: {table!r}")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        # DDL statements cannot take prepared parameters in DuckDB; inline
        # the path as an escaped SQL string literal instead.
        csv_literal = str(csv_path).replace("'", "''")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW _raw_city AS
            SELECT * FROM read_csv('{csv_literal}', header = true)
            """
        )
        lat_map = _coordinate_mapping(
            [r[0] for r in con.execute(
                "SELECT DISTINCT Latitude::VARCHAR FROM _raw_city"
            ).fetchall()],
            bound=90.0,
        )
        lon_map = _coordinate_mapping(
            [r[0] for r in con.execute(
                "SELECT DISTINCT Longitude::VARCHAR FROM _raw_city"
            ).fetchall()],
            bound=180.0,
        )
        con.register("_lat_map", lat_map)
        con.register("_lon_map", lon_map)

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT
                r.dt::DATE                                  AS dt,
                r.AverageTemperature::DOUBLE                AS AverageTemperature,
                r.AverageTemperatureUncertainty::DOUBLE     AS AverageTemperatureUncertainty,
                r.City::VARCHAR                             AS City,
                r.Country::VARCHAR                          AS Country,
                la.parsed                                   AS Latitude,
                lo.parsed                                   AS Longitude
            FROM _raw_city r
            JOIN _lat_map la ON r.Latitude::VARCHAR = la.raw
            JOIN _lon_map lo ON r.Longitude::VARCHAR = lo.raw
            """
        )

        n_raw = con.execute("SELECT count(*) FROM _raw_city").fetchone()[0]
        n_rows, n_cities, dt_min, dt_max = con.execute(
            f"""
            SELECT count(*),
                   count(DISTINCT (City, Country)),
                   min(dt),
                   max(dt)
            FROM {table}
            """
        ).fetchone()
        if n_rows != n_raw:
            raise RuntimeError(
                f"ingestion dropped rows: {n_raw} in CSV, {n_rows} in {table}"
            )
    finally:
        con.close()

    logger.info(
        "loaded %d rows / %d cities into %s::%s", n_rows, n_cities, db_path, table
    )
    return IngestResult(
        db_path=db_path,
        table=table,
        n_rows=n_rows,
        n_cities=n_cities,
        dt_min=dt_min,
        dt_max=dt_max,
    )
