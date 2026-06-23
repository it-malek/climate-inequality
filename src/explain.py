"""Phase 7: explanatory variables for warming trends.

Two questions, building on the per-city-location trends from Phase 2
(``data/processed/city_trends.parquet``, keyed on
:data:`src.trends.CITY_KEYS`):

1. **City level**: which geographic variables explain the spatial pattern
   of ``slope_c_per_decade``? Latitude (the tropics -> Arctic gradient) and
   aridity (the Iranian-plateau / Central-Asia hotspot) are the README's
   two candidates.
2. **Country level (the important one)**: does the stored
   +0.029 °C/decade-per-10x-emissions, continent-fixed-effects result
   (``app/data/stats.json``, README lines 80-82) survive a
   mean-|latitude| control?

This module introduces several new external grids and geometries (ETOPO
elevation, Koeppen climate classes, Natural Earth coastlines) feeding
straight into a regression whose result may revise a headline finding.
Before any of that is trusted, :func:`run_geo_preflight` -- a guardrail,
not new analysis -- verifies coordinate conventions, spot-checks three
known cities, checks grid-sampling determinism/NaN rates, and checks the
country-name joins. :func:`main` runs it first and aborts if it fails.

Sections, in execution order:
    1. Constants / paths.
    2. Coordinate-system helpers (shared by the preflight and the
       feature samplers).
    3. Feature assembly (-> ``city_features.parquet``).
    4. Geo + data integrity preflight (gates everything below).
    5. City-level model (3 specs + Moran's I diagnostic).
    6. Country-level coefficient-stability model (6 specs).
    7. ``main()``.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xarray as xr
from PIL import Image
from shapely import Geometry
from shapely.geometry import Point
from shapely.ops import nearest_points
from statsmodels.regression.linear_model import RegressionResultsWrapper

from src.data_io import PROCESSED_DIR, RAW_DIR, download_file, write_typed_parquet
from src.emissions import (
    BERKELEY_TO_OWID,
    DEFAULT_INEQUALITY_PATH,
)
# Grid samplers moved to src.grids (their natural home) so the lean
# emissions/population path can use them without importing this heavy module.
from src.grids import check_coordinate_orientation, sample_static_grid
from src.interpolate import (
    LAND_ZIP_PATH,
    _knn_indices,
    haversine_km,
    load_land_geometry,
)
from src.trends import CITY_KEYS, DEFAULT_TRENDS_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 1. Constants / paths
# ---------------------------------------------------------------------

# ETOPO 2022, 60 arc-second, global, ice-surface elevation NetCDF.
# Verified June 2026: the single global tile on NCEI's THREDDS server
# (15x15-degree tiles also exist, but this 60s product is one global file).
ETOPO_URL = (
    "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/"
    "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
)
ETOPO_PATH = RAW_DIR / "etopo" / "ETOPO_2022_v1_60s_N90W180_surface.nc"
ETOPO_VAR = "z"

# Beck et al. 2018 Koeppen-Geiger V1 maps (gloh2o.org/koppen/), present-day
# (1980-2016) 0.5-degree GeoTIFF -- 32 KB, the coarsest variant, distributed
# inside a 70 MB zip with the finer resolutions. GeoTIFF (not NetCDF/ASCII)
# is the only coarse present-day format available, so it is converted to a
# small NetCDF once (via Pillow's GeoTIFF tags, no rasterio dependency) by
# :func:`prepare_koppen_grid`.
KOPPEN_ZIP_URL = "https://ndownloader.figshare.com/files/12407516"
KOPPEN_ZIP_PATH = RAW_DIR / "koppen" / "Beck_KG_V1.zip"
KOPPEN_TIF_MEMBER = "Beck_KG_V1_present_0p5.tif"
KOPPEN_PATH = RAW_DIR / "koppen" / "koppen_present_0p5.nc"
KOPPEN_VAR = "koppen_code"

# OWID's mirror of the World Bank income classification (CSV, no XLSX/
# openpyxl dependency needed). One row per country-year; the most recent
# year per country is used.
INCOME_URL = (
    "https://ourworldindata.org/grapher/world-bank-income-groups.csv"
    "?v=1&csvType=full&useColumnShortNames=false"
)
INCOME_PATH = RAW_DIR / "worldbank" / "world-bank-income-groups.csv"

DEFAULT_FEATURES_PATH = PROCESSED_DIR / "city_features.parquet"

# Committed summary artifacts (small, merged into the app bundle by app_assets).
DEFAULT_EXPLAIN_SUMMARY_PATH = PROCESSED_DIR / "explain_summary.json"
DEFAULT_EXPLAIN_BUNDLE_PATH = PROCESSED_DIR / "explain_features.parquet"

# Slim city-features schema for the app bundle (float32 numerics). Latitude/
# Longitude are carried so the committed parquet can be ordered by the full
# (City, Country, Latitude, Longitude) identity -- same-named city pairs sit at
# multiple coordinates, so (City, Country) alone is not a deterministic key.
EXPLAIN_BUNDLE_SCHEMA = {
    "City": "VARCHAR",
    "Country": "VARCHAR",
    "Latitude": "FLOAT",
    "Longitude": "FLOAT",
    "abs_latitude": "FLOAT",
    "slope_c_per_decade": "FLOAT",
    "koppen": "VARCHAR",
}

# On-disk schema of city_features.parquet (DuckDB types), in column order.
FEATURES_SCHEMA = {
    "City": "VARCHAR",
    "Country": "VARCHAR",
    "Latitude": "DOUBLE",
    "Longitude": "DOUBLE",
    "slope_c_per_decade": "DOUBLE",
    "abs_latitude": "DOUBLE",
    "hemisphere": "VARCHAR",
    "coast_km": "DOUBLE",
    "elevation_m": "DOUBLE",
    "koppen": "VARCHAR",
    "station_density": "DOUBLE",
}
FEATURES_COLUMNS = list(FEATURES_SCHEMA)


# ---------------------------------------------------------------------
# 2. Coordinate-system helpers
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# 3. Feature assembly
# ---------------------------------------------------------------------


def add_latitude_features(trends: pd.DataFrame) -> pd.DataFrame:
    """Add `abs_latitude` and `hemisphere` columns.

    Free -- no external data. The first model spec
    (``slope_c_per_decade ~ abs_latitude``) uses only this.

    Args:
        trends: Frame with a `Latitude` column (signed degrees).

    Returns:
        Copy of `trends` with `abs_latitude` (float) and `hemisphere`
        (``"N"``/``"S"``) added.
    """
    return trends.assign(
        abs_latitude=trends["Latitude"].abs(),
        hemisphere=np.where(trends["Latitude"] >= 0, "N", "S"),
    )


def coast_distance_km(
    lon: np.ndarray, lat: np.ndarray, land: Geometry
) -> np.ndarray:
    """Great-circle distance from each (lon, lat) to the nearest coastline.

    Args:
        lon, lat: 1D arrays of point coordinates, in degrees.
        land: Land (multi)polygon, e.g. from
            :func:`src.interpolate.load_land_geometry`.

    Returns:
        1D array of distances in km. ``shapely.ops.nearest_points`` returns
        coordinates in degrees, so the result is converted via
        :func:`src.interpolate.haversine_km` -- a raw shapely distance would
        be in degrees, not km.
    """
    boundary = land.boundary
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    out = np.empty(len(lon), dtype=float)
    for i in range(len(lon)):
        point = Point(lon[i], lat[i])
        nearest_on_land, nearest_on_boundary = nearest_points(point, boundary)
        out[i] = haversine_km(
            nearest_on_land.x, nearest_on_land.y,
            nearest_on_boundary.x, nearest_on_boundary.y,
        )
    return out


def add_coast_distance(
    df: pd.DataFrame, land: Geometry | None = None
) -> pd.DataFrame:
    """Add `coast_km`, the distance from each location to the coastline.

    Args:
        df: Frame with `Longitude`/`Latitude` columns.
        land: Land geometry; defaults to
            :func:`src.interpolate.load_land_geometry` (cached Natural
            Earth 110m polygons).

    Returns:
        Copy of `df` with `coast_km` (float, km) added.
    """
    land = land if land is not None else load_land_geometry()
    coast_km = coast_distance_km(
        df["Longitude"].to_numpy(), df["Latitude"].to_numpy(), land
    )
    return df.assign(coast_km=coast_km)


def add_elevation(
    df: pd.DataFrame, nc_path: Path = ETOPO_PATH, var: str = ETOPO_VAR
) -> pd.DataFrame:
    """Add `elevation_m` from the ETOPO 2022 ice-surface grid.

    Downloads `nc_path` (~478 MB) via :func:`src.data_io.download_file` if
    missing. City coordinates are grid-snapped (~1 degree), so the sampled
    elevation is approximate -- fine for a regression feature.

    Args:
        df: Frame with `Longitude`/`Latitude` columns.
        nc_path: ETOPO NetCDF path.
        var: Elevation variable name.

    Returns:
        Copy of `df` with `elevation_m` (float, meters; negative = below
        sea level) added.
    """
    download_file(ETOPO_URL, nc_path, timeout=600)
    values = sample_static_grid(nc_path, df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), var)
    return df.assign(elevation_m=values)


# Beck et al. 2018 30-class legend (legend.txt in Beck_KG_V1.zip), collapsed
# to the 5 major Koeppen-Geiger groups. Code 0 (ocean / no data) is
# deliberately absent -> collapse_koppen maps it to NaN.
KOPPEN_GROUPS: dict[int, str] = {
    **{code: "A" for code in range(1, 4)},    # Af, Am, Aw
    **{code: "B" for code in range(4, 8)},    # BWh, BWk, BSh, BSk
    **{code: "C" for code in range(8, 17)},   # Csa..Cfc
    **{code: "D" for code in range(17, 29)},  # Dsa..Dfd
    **{code: "E" for code in (29, 30)},       # ET, EF
}


def collapse_koppen(
    codes: pd.Series, mapping: dict[int, str] = KOPPEN_GROUPS
) -> pd.Series:
    """Collapse 30-class Koeppen-Geiger codes to the 5 major groups A-E.

    Args:
        codes: Raw numeric class codes (0 = ocean/no-data, 1-30 = classes).
        mapping: code -> letter mapping; defaults to :data:`KOPPEN_GROUPS`.

    Returns:
        Series of single-letter strings, with unmapped codes (0, NaN, or
        anything outside 1-30) as NaN.
    """
    return codes.map(mapping)


def _geotiff_to_netcdf(tif_bytes: bytes, out_path: Path, var: str) -> Path:
    """Convert a single-band, geo-referenced GeoTIFF to a 2D NetCDF.

    Reads the raw pixel array with Pillow and the ``ModelPixelScaleTag``
    (33550) / ``ModelTiepointTag`` (33922) GeoTIFF tags for the pixel size
    and the (lon, lat) of the top-left pixel's corner, avoiding a rasterio
    dependency for what is otherwise a tiny grid.

    Args:
        tif_bytes: Raw GeoTIFF file contents.
        out_path: Destination NetCDF path; parents created if missing.
        var: Name to give the 2D data variable in the output dataset.

    Returns:
        `out_path`.
    """
    img = Image.open(io.BytesIO(tif_bytes))
    codes = np.array(img)
    n_rows, n_cols = codes.shape
    scale_x, scale_y = img.tag_v2[33550][0], img.tag_v2[33550][1]
    origin_lon, origin_lat = img.tag_v2[33922][3], img.tag_v2[33922][4]
    # GeoTIFF tiepoint gives the *corner* of the top-left pixel; coordinate
    # variables here use pixel centers.
    lon = origin_lon + (np.arange(n_cols) + 0.5) * scale_x
    lat = origin_lat - (np.arange(n_rows) + 0.5) * scale_y
    ds = xr.Dataset(
        {var: (("lat", "lon"), codes)},
        coords={"lat": lat, "lon": lon},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    return out_path


def prepare_koppen_grid(
    zip_path: Path = KOPPEN_ZIP_PATH,
    out_path: Path = KOPPEN_PATH,
    member: str = KOPPEN_TIF_MEMBER,
    url: str = KOPPEN_ZIP_URL,
    var: str = KOPPEN_VAR,
) -> Path:
    """Download Beck et al. 2018 V1 (if needed) and produce a NetCDF grid.

    Idempotent: if `out_path` already exists, returns it unchanged.

    Args:
        zip_path: Local path for the ~70 MB ``Beck_KG_V1.zip``.
        out_path: Destination NetCDF (one 0.5-degree present-day grid).
        member: Zip member to extract (the 0.5-degree present-day GeoTIFF).
        url: Source URL for `zip_path`.
        var: Output variable name (raw 0-30 class codes).

    Returns:
        `out_path`.
    """
    if out_path.exists():
        return out_path
    download_file(url, zip_path, timeout=120)
    with zipfile.ZipFile(zip_path) as zf:
        tif_bytes = zf.read(member)
    return _geotiff_to_netcdf(tif_bytes, out_path, var=var)


def add_koppen(
    df: pd.DataFrame, nc_path: Path = KOPPEN_PATH, var: str = KOPPEN_VAR
) -> pd.DataFrame:
    """Add `koppen`, the major Koeppen-Geiger group (A-E) at each location.

    Prepares `nc_path` via :func:`prepare_koppen_grid` if missing, samples
    raw class codes via :func:`sample_static_grid`, then collapses them via
    :func:`collapse_koppen`. Ocean/no-data cells (code 0) -> NaN; the count
    is logged.

    Args:
        df: Frame with `Longitude`/`Latitude` columns.
        nc_path: Koeppen grid NetCDF path (converted from GeoTIFF).
        var: Raw class-code variable name.

    Returns:
        Copy of `df` with `koppen` (single-letter string or NaN) added.
    """
    prepare_koppen_grid(out_path=nc_path, var=var)
    codes = sample_static_grid(nc_path, df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), var)
    koppen = collapse_koppen(pd.Series(codes, index=df.index))
    n_nan = int(koppen.isna().sum())
    if n_nan:
        logger.info("%d/%d locations have no Koeppen class (ocean cell)", n_nan, len(koppen))
    return df.assign(koppen=koppen)


def add_station_density(
    df: pd.DataFrame, radius_km: float = 100.0, k: int = 50
) -> pd.DataFrame:
    """Add `station_density`, the count of nearby city-locations.

    For each location, counts how many of its `k` nearest neighbors
    (by great-circle distance, via :func:`src.interpolate._knn_indices`)
    fall within `radius_km` (excluding itself). If a location's true
    neighbor count within `radius_km` exceeds `k`, this undercounts --
    an accepted approximation for a "first to drop" feature at
    city-location density (handdown Phase 7 table, feature 5).

    Args:
        df: Frame with `Longitude`/`Latitude` columns.
        radius_km: Neighborhood radius in km.
        k: Number of nearest neighbors to examine per location.

    Returns:
        Copy of `df` with `station_density` (float count) added.
    """
    lon = df["Longitude"].to_numpy()
    lat = df["Latitude"].to_numpy()
    n = len(df)
    k_query = min(k + 1, n)  # +1: the nearest "neighbor" is the point itself
    idx = _knn_indices(lon, lat, lon, lat, k=k_query)

    density = np.empty(n, dtype=float)
    for i in range(n):
        neighbors = idx[i][idx[i] != i][:k]
        dist = haversine_km(lon[i], lat[i], lon[neighbors], lat[neighbors])
        density[i] = float(np.sum(dist <= radius_km))
    return df.assign(station_density=density)


# Berkeley Earth -> World Bank income-table ("OWID-mirrored") country-name
# overrides. Empty: country_inequality.parquet's `owid_country` column
# (already overridden via BERKELEY_TO_OWID for the emissions join) matches
# the income table's `owid_country` for all 157 countries -- verified at
# session time. Kept for the same reason as BERKELEY_TO_OWID: a documented
# seam if a future income-table refresh renames a country.
BERKELEY_TO_WORLDBANK: dict[str, str] = {}


def load_income_groups(csv_path: Path = INCOME_PATH) -> pd.DataFrame:
    """Load the most recent World Bank income classification per country.

    Args:
        csv_path: OWID's mirror of the World Bank income groups
            (``Entity, Code, Year, World Bank's income classification``).

    Returns:
        One row per country: `owid_country`, `income_group` (each
        country's most recent available year -- some countries' series
        end before the table's latest year, e.g. Venezuela at 2019).

    Raises:
        FileNotFoundError: if `csv_path` is missing -- download it via
            :data:`INCOME_URL` first.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"no such file: {csv_path}; download it first")
    raw = pd.read_csv(csv_path)
    latest = raw.sort_values("Year").groupby("Entity", as_index=False).tail(1)
    return latest.rename(
        columns={"Entity": "owid_country", "World Bank's income classification": "income_group"}
    )[["owid_country", "income_group"]].reset_index(drop=True)


def attach_income_group(
    country_table: pd.DataFrame,
    income: pd.DataFrame,
    overrides: dict[str, str] = BERKELEY_TO_WORLDBANK,
) -> pd.DataFrame:
    """Join World Bank income groups onto a country table by `owid_country`.

    Args:
        country_table: Frame with an `owid_country` column.
        income: From :func:`load_income_groups`.
        overrides: `owid_country` -> income-table country-name overrides.

    Returns:
        Copy of `country_table` with `income_group` added (NaN where
        unmatched; unmatched names are logged).
    """
    income_map = income.set_index("owid_country")["income_group"]
    keys = country_table["owid_country"].map(lambda c: overrides.get(c, c))
    unmatched = sorted(set(keys) - set(income_map.index))
    if unmatched:
        logger.warning("no income group for: %s", unmatched)
    return country_table.assign(income_group=keys.map(income_map))


def build_city_features(
    trends_path: Path = DEFAULT_TRENDS_PATH,
    out_path: Path = DEFAULT_FEATURES_PATH,
    land: Geometry | None = None,
    etopo_path: Path = ETOPO_PATH,
    koppen_path: Path = KOPPEN_PATH,
    include_density: bool = True,
) -> pd.DataFrame:
    """Build the one-row-per-city-location feature table.

    Applies :func:`add_latitude_features`, :func:`add_coast_distance`,
    :func:`add_elevation`, :func:`add_koppen`, and (if `include_density`)
    :func:`add_station_density`, in that order, then writes
    `out_path` via :func:`src.data_io.write_typed_parquet`.

    Args:
        trends_path: :data:`src.trends.DEFAULT_TRENDS_PATH`.
        out_path: Destination parquet, :data:`DEFAULT_FEATURES_PATH`.
        land: Land geometry passed to :func:`add_coast_distance`; defaults
            to :func:`src.interpolate.load_land_geometry`.
        etopo_path: Passed to :func:`add_elevation`.
        koppen_path: Passed to :func:`add_koppen`.
        include_density: If False, `station_density` is all-NaN (the
            feature is excludable from model formulas with one flag, per
            the handdown's drop order).

    Returns:
        The feature DataFrame, columns :data:`FEATURES_COLUMNS`.
    """
    trends = pd.read_parquet(trends_path)[CITY_KEYS + ["slope_c_per_decade"]]
    df = add_latitude_features(trends)

    land = land if land is not None else load_land_geometry()
    df = add_coast_distance(df, land=land)
    df = add_elevation(df, nc_path=etopo_path)
    df = add_koppen(df, nc_path=koppen_path)
    if include_density:
        df = add_station_density(df)
    else:
        df = df.assign(station_density=np.nan)

    df = df[FEATURES_COLUMNS]
    write_typed_parquet(df, out_path, FEATURES_SCHEMA, order_by=CITY_KEYS)
    logger.info("wrote %d rows to %s", len(df), out_path)
    return df


# ---------------------------------------------------------------------
# 4. Geo + data integrity preflight (gates Sections 5-6)
# ---------------------------------------------------------------------

# (City, Country) pairs spot-checked by print_city_sanity_checks.
# "Reykjavík" matches city_trends.parquet's spelling (with the accent).
SANITY_CITIES: list[tuple[str, str]] = [
    ("New York", "United States"),
    ("Cairo", "Egypt"),
    ("Reykjavík", "Iceland"),
]

# Countries present in city_trends.parquet but absent from
# country_inequality.parquet (README limitations: 157/159 countries
# matched OWID -- Puerto Rico and Reunion have no OWID emissions series).
# Anything beyond this set is a new, unexplained join gap.
EXPECTED_COUNTRY_GAPS: set[str] = {"Puerto Rico", "Reunion"}

# Synthetic boundary points: no real city-location is near the poles or
# the antimeridian (max |lat|=69.92 Norilsk, lon range [-151.13, 176.95]),
# so these test that nearest-cell lookup neither wraps nor errors there.
BOUNDARY_POINTS = (np.array([89.99, -89.99, 0.0]), np.array([179.99, -179.99, 180.0]))


def print_city_sanity_checks(
    trends: pd.DataFrame,
    etopo_path: Path = ETOPO_PATH,
    koppen_path: Path = KOPPEN_PATH,
    land: Geometry | None = None,
    cities: list[tuple[str, str]] = SANITY_CITIES,
) -> pd.DataFrame:
    """Print and validate sampled features for `cities`.

    For each (City, Country) in `cities`, samples `elevation_m`, `koppen`,
    and `coast_km` and prints a small table. This is the guardrail's core
    spot-check: a coordinate-axis swap or grid misalignment would show up
    here as an implausible elevation, a wrong Koeppen class for Cairo, or
    a wrong coast-distance ordering -- well before it could be blamed on
    "noisy" regression coefficients.

    Args:
        trends: city_trends.parquet, has :data:`src.trends.CITY_KEYS`.
        etopo_path, koppen_path: grid paths, passed to
            :func:`sample_static_grid`.
        land: land geometry for :func:`add_coast_distance`; defaults to
            :func:`src.interpolate.load_land_geometry`.
        cities: (City, Country) pairs to check; defaults to
            :data:`SANITY_CITIES`.

    Returns:
        One row per matched (City, Country), columns `City`, `Country`,
        `Latitude`, `Longitude`, `elevation_m`, `koppen`, `coast_km`.

    Raises:
        AssertionError: if any `cities` entry is missing from `trends`,
            any `elevation_m` is outside [-100, 2000] m, any sanity
            city's `koppen` is NaN, Cairo's `koppen` is not "B" (arid),
            or `coast_km[Cairo]` is not greater than both New York's and
            Reykjavík's.
    """
    lookup = pd.DataFrame(cities, columns=["City", "Country"])
    sub = trends.merge(lookup, on=["City", "Country"], how="inner").reset_index(drop=True)
    found = set(zip(sub["City"], sub["Country"]))
    missing = set(cities) - found
    if missing:
        raise AssertionError(f"sanity cities not found in trends: {sorted(missing)}")

    land = land if land is not None else load_land_geometry()
    sub = add_coast_distance(sub, land=land)
    sub["elevation_m"] = sample_static_grid(
        etopo_path, sub["Latitude"].to_numpy(), sub["Longitude"].to_numpy(), ETOPO_VAR
    )
    codes = sample_static_grid(
        koppen_path, sub["Latitude"].to_numpy(), sub["Longitude"].to_numpy(), KOPPEN_VAR
    )
    sub["koppen"] = collapse_koppen(pd.Series(codes, index=sub.index))

    report = sub[["City", "Country", "Latitude", "Longitude", "elevation_m", "koppen", "coast_km"]]
    print(report.to_string(index=False))

    for _, row in report.iterrows():
        if not (-100.0 <= row["elevation_m"] <= 2000.0):
            raise AssertionError(
                f"{row['City']}: elevation_m={row['elevation_m']} outside plausible [-100, 2000] range"
            )
    if report["koppen"].isna().any():
        bad = report.loc[report["koppen"].isna(), "City"].tolist()
        raise AssertionError(f"sanity cities with no Koeppen class (ocean cell?): {bad}")

    by_city = report.set_index("City")
    cairo_koppen = by_city.loc["Cairo", "koppen"]
    if cairo_koppen != "B":
        raise AssertionError(f"Cairo Koeppen class = {cairo_koppen!r}, expected 'B' (arid)")

    cairo_coast = float(by_city.loc["Cairo", "coast_km"])
    for other in ("New York", "Reykjavík"):
        other_coast = float(by_city.loc[other, "coast_km"])
        if not (cairo_coast > other_coast):
            raise AssertionError(
                f"coast_km[Cairo]={cairo_coast:.1f} not > coast_km[{other}]={other_coast:.1f}"
            )
    return report


def check_sampling_determinism(
    nc_path: Path,
    var: str,
    lats: np.ndarray,
    lons: np.ndarray,
    n_repeats: int = 3,
) -> np.ndarray:
    """Assert repeated `sample_static_grid` calls are bit-identical.

    Catches nondeterminism from unsorted indices or hash-based operations
    that would otherwise silently make the feature table (and any model
    fit on it) unreproducible.

    Args:
        nc_path, var: passed to :func:`sample_static_grid`.
        lats, lons: query coordinates.
        n_repeats: number of repeated samplings to compare.

    Returns:
        The first run's sampled values, for reuse (e.g. by
        :func:`check_nan_rate`).

    Raises:
        AssertionError: if any repeat differs from the first
            (NaN-aware: NaN == NaN counts as equal).
    """
    runs = [sample_static_grid(nc_path, lats, lons, var) for _ in range(n_repeats)]
    for i, run in enumerate(runs[1:], start=1):
        if not np.array_equal(runs[0], run, equal_nan=True):
            raise AssertionError(f"{nc_path.name}: sampling run {i} differs from run 0 (nondeterministic)")
    return runs[0]


def check_nan_rate(values: np.ndarray | pd.Series, label: str, max_nan_fraction: float = 0.02) -> float:
    """Assert the NaN fraction in `values` is below `max_nan_fraction`.

    Args:
        values: sampled feature values (e.g. from
            :func:`check_sampling_determinism`, or post-
            :func:`collapse_koppen`).
        label: name used in the log message and error.
        max_nan_fraction: maximum acceptable fraction of NaNs. For
            elevation (every city-location is on land, ETOPO is global)
            this should be ~0; for Koeppen class, coastal cities whose
            grid-snapped coordinate's nearest 0.5-degree cell is ocean
            (code 0) give ~7% NaN on the real data -- a known, plausible
            rate, not a bug.

    Returns:
        The NaN fraction.

    Raises:
        AssertionError: if the fraction exceeds `max_nan_fraction`.
    """
    values = pd.Series(values)
    n_nan = int(values.isna().sum())
    nan_fraction = n_nan / len(values)
    logger.info("%s: %d/%d (%.1f%%) NaN", label, n_nan, len(values), 100 * nan_fraction)
    if nan_fraction > max_nan_fraction:
        raise AssertionError(
            f"{label}: NaN fraction {nan_fraction:.3f} exceeds max {max_nan_fraction}"
        )
    return nan_fraction


def check_country_join(
    trends: pd.DataFrame,
    country_inequality: pd.DataFrame,
    income: pd.DataFrame,
    overrides: dict[str, str] = BERKELEY_TO_OWID,
) -> dict[str, list[str]]:
    """Set-difference report across the three country-keyed tables.

    Args:
        trends: city_trends.parquet -- `Country` is Berkeley Earth-named.
        country_inequality: country_inequality.parquet -- `Country` is
            Berkeley-named, `owid_country` is OWID-named (post-
            :data:`src.emissions.BERKELEY_TO_OWID`).
        income: from :func:`load_income_groups` -- `owid_country` is
            OWID-named.
        overrides: Berkeley -> OWID name overrides, applied to
            `trends["Country"]` to check the override dict is internally
            consistent with `country_inequality["owid_country"]`.

    Returns:
        Dict of sorted set differences, each also logged:
        `trends_not_in_inequality`, `inequality_not_in_trends`,
        `trends_owid_not_in_inequality_owid`,
        `inequality_owid_not_in_income`. The third is restricted to
        countries that *do* appear in `country_inequality` -- countries
        entirely absent from it (e.g. Puerto Rico, Reunion) belong to
        `trends_not_in_inequality`, not here.
    """
    trends_countries = set(trends["Country"].unique())
    inequality_countries = set(country_inequality["Country"].unique())
    inequality_owid = set(country_inequality["owid_country"].unique())
    income_owid = set(income["owid_country"].unique())
    matched_owid = {overrides.get(c, c) for c in trends_countries & inequality_countries}

    report = {
        "trends_not_in_inequality": sorted(trends_countries - inequality_countries),
        "inequality_not_in_trends": sorted(inequality_countries - trends_countries),
        "trends_owid_not_in_inequality_owid": sorted(matched_owid - inequality_owid),
        "inequality_owid_not_in_income": sorted(inequality_owid - income_owid),
    }
    for key, vals in report.items():
        logger.info("%s (%d): %s", key, len(vals), vals if vals else "none")
    return report


def run_geo_preflight(
    trends_path: Path = DEFAULT_TRENDS_PATH,
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    income_path: Path = INCOME_PATH,
    etopo_path: Path = ETOPO_PATH,
    koppen_path: Path = KOPPEN_PATH,
    land: Geometry | None = None,
) -> bool:
    """Run the Phase 7 geo + data integrity preflight.

    A guardrail, not new analysis: confirms coordinate conventions, three
    known cities' sampled features, grid-sampling determinism/NaN rates,
    and country-name joins, *before* any feature is trusted for modeling.
    :func:`main` calls this first and aborts if it returns False.

    Args:
        trends_path: :data:`src.trends.DEFAULT_TRENDS_PATH`.
        inequality_path: :data:`src.emissions.DEFAULT_INEQUALITY_PATH`.
        income_path: :data:`INCOME_PATH`.
        etopo_path, koppen_path: grid paths.
        land: land geometry; defaults to
            :func:`src.interpolate.load_land_geometry`.

    Returns:
        True if every check passes; False if any fails (the failure is
        also printed/logged with its reason).
    """
    print("=== Geo + data integrity preflight ===")
    ok = True

    print("\n-- 1. Coordinate system integrity --")
    land = land if land is not None else load_land_geometry()
    gdf = gpd.read_file(f"zip://{LAND_ZIP_PATH}")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        print(f"  FAIL: Natural Earth land CRS = {gdf.crs}, expected EPSG:4326")
        ok = False
    else:
        print("  OK: Natural Earth land CRS = EPSG:4326 (lon, lat order)")

    for grid_name, path in (("ETOPO", etopo_path), ("Koeppen", koppen_path)):
        orient = check_coordinate_orientation(path)
        print(
            f"  {grid_name}: lat [{orient['lat_min']:.2f}, {orient['lat_max']:.2f}] "
            f"({'ascending' if orient['lat_ascending'] else 'descending'}), "
            f"lon [{orient['lon_min']:.2f}, {orient['lon_max']:.2f}] ({orient['lon_convention']})"
        )
        if orient["lon_convention"] != "-180-180":
            print(f"  NOTE: {grid_name} uses 0-360 longitudes; sample_static_grid will normalize")

    print("\n-- 2. Spatial alignment sanity checks --")
    trends = pd.read_parquet(trends_path)[CITY_KEYS + ["slope_c_per_decade"]]
    try:
        print_city_sanity_checks(trends, etopo_path=etopo_path, koppen_path=koppen_path, land=land)
    except AssertionError as exc:
        print(f"  FAIL: {exc}")
        ok = False

    print("\n-- 3. Grid sampling validation --")
    lats, lons = trends["Latitude"].to_numpy(), trends["Longitude"].to_numpy()
    try:
        elevation = check_sampling_determinism(etopo_path, ETOPO_VAR, lats, lons, n_repeats=2)
        check_nan_rate(elevation, "elevation_m (all city-locations)", max_nan_fraction=0.02)

        codes = check_sampling_determinism(koppen_path, KOPPEN_VAR, lats, lons, n_repeats=2)
        koppen = collapse_koppen(pd.Series(codes))
        check_nan_rate(koppen, "koppen (all city-locations)", max_nan_fraction=0.10)

        boundary_lats, boundary_lons = BOUNDARY_POINTS
        check_sampling_determinism(etopo_path, ETOPO_VAR, boundary_lats, boundary_lons)
        check_sampling_determinism(koppen_path, KOPPEN_VAR, boundary_lats, boundary_lons)
        print("  OK: elevation/Koeppen sampling deterministic, NaN rates within bounds")
    except AssertionError as exc:
        print(f"  FAIL: {exc}")
        ok = False

    print("\n-- 4. Country join integrity --")
    inequality = pd.read_parquet(inequality_path)
    income = load_income_groups(income_path)
    join_report = check_country_join(trends, inequality, income)
    for key, vals in join_report.items():
        print(f"  {key}: {vals if vals else 'none'}")
    unexpected = set(join_report["trends_not_in_inequality"]) - EXPECTED_COUNTRY_GAPS
    if unexpected:
        print(f"  FAIL: unexpected unmatched countries (trends vs. inequality): {sorted(unexpected)}")
        ok = False
    if join_report["trends_owid_not_in_inequality_owid"]:
        print(
            "  FAIL: BERKELEY_TO_OWID override produces names absent from country_inequality: "
            f"{join_report['trends_owid_not_in_inequality_owid']}"
        )
        ok = False

    print(f"\n=== Preflight {'PASSED' if ok else 'FAILED'} ===")
    return ok


# ---------------------------------------------------------------------
# 5. City-level model
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TermFit:
    """One regression term's effect size and uncertainty."""

    term: str
    coef: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float


def _extract_all_terms(fit: RegressionResultsWrapper) -> list[TermFit]:
    """Generalize :func:`src.emissions._extract_fit` to every term in `fit`.

    Args:
        fit: a fitted statsmodels OLS result.

    Returns:
        One :class:`TermFit` per row of `fit.params` (in that order),
        including the intercept.
    """
    ci = fit.conf_int()
    return [
        TermFit(
            term=term,
            coef=float(fit.params[term]),
            se=float(fit.bse[term]),
            ci_low=float(ci.loc[term, 0]),
            ci_high=float(ci.loc[term, 1]),
            p_value=float(fit.pvalues[term]),
        )
        for term in fit.params.index
    ]


def morans_i(
    residuals: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    k: int = 8,
    n_permutations: int = 199,
    seed: int = 0,
) -> tuple[float, float]:
    """Moran's I for spatial autocorrelation in `residuals`.

    Uses row-standardized k-nearest-neighbor weights (each point weighted
    ``1/k`` against its `k` nearest neighbors by great-circle distance, via
    :func:`src.interpolate._knn_indices`). With row-standardized weights
    ``S0 = n``, so

        I = sum_i(z_i * mean(z_neighbors_i)) / sum(z_i^2)

    where ``z = residuals - mean(residuals)``.

    Args:
        residuals: model residuals, one per point.
        lon, lat: coordinates, same length and order as `residuals`.
        k: number of nearest neighbors.
        n_permutations: number of label permutations for the p-value.
        seed: RNG seed for the permutations.

    Returns:
        `(I, p_value)`. `p_value` is the two-sided permutation p-value
        ``(count(|I_perm| >= |I_obs|) + 1) / (n_permutations + 1)``.
    """
    n = len(residuals)
    z = np.asarray(residuals, dtype=float) - np.mean(residuals)
    idx = _knn_indices(lon, lat, lon, lat, k=min(k + 1, n))
    neighbor_idx = np.array([row[row != i][:k] for i, row in enumerate(idx)])

    def _compute(zz: np.ndarray) -> float:
        neighbor_means = zz[neighbor_idx].mean(axis=1)
        return float(np.sum(zz * neighbor_means) / np.sum(zz**2))

    observed = _compute(z)
    rng = np.random.default_rng(seed)
    perm_stats = np.array([_compute(rng.permutation(z)) for _ in range(n_permutations)])
    p_value = (np.sum(np.abs(perm_stats) >= np.abs(observed)) + 1) / (n_permutations + 1)
    return observed, float(p_value)


def partial_r2_by_feature(
    formula: str, data: pd.DataFrame, groups: dict[str, list[str]]
) -> dict[str, float]:
    """Partial R^2 of each named group of terms.

    For each `(label, terms)` in `groups`, refits `formula` with every
    string in `terms` removed (each as `" + {term}"`), on the *same* rows
    used by the full-formula fit (`missing="drop"` applied once, to
    `formula`), and computes
    ``(R2_full - R2_reduced) / (1 - R2_reduced)``.

    Args:
        formula: the full model's patsy formula.
        data: input data; rows with missing values in `formula`'s columns
            are dropped before any fit.
        groups: human label -> list of formula terms to remove together,
            e.g. `{"koppen": ["C(koppen)", "abs_latitude:C(koppen)"]}` for
            an interaction spec.

    Returns:
        `{label: partial_r2}`.
    """
    full = smf.ols(formula, data=data, missing="drop").fit()
    used = data.loc[full.model.data.row_labels]
    r2_full = float(full.rsquared)

    out = {}
    for label, terms in groups.items():
        reduced_formula = formula
        for term in terms:
            reduced_formula = reduced_formula.replace(f" + {term}", "")
        r2_reduced = float(smf.ols(reduced_formula, data=used).fit().rsquared)
        out[label] = (r2_full - r2_reduced) / (1 - r2_reduced)
    return out


@dataclass(frozen=True)
class CityModelResult:
    """One city-level model spec's fit summary."""

    spec_name: str
    formula: str
    n: int
    r2: float
    terms: list[TermFit] = field(default_factory=list)
    partial_r2: dict[str, float] = field(default_factory=dict)
    moran_i: float | None = None
    moran_p: float | None = None


def fit_city_model(
    features: pd.DataFrame,
    spec_name: str,
    formula: str,
    cluster_col: str = "Country",
    partial_r2_groups: dict[str, list[str]] | None = None,
    morans_k: int | None = 8,
) -> tuple[CityModelResult, RegressionResultsWrapper]:
    """Fit one city-level OLS spec with country-clustered standard errors.

    Args:
        features: from :func:`build_city_features`.
        spec_name: label for :attr:`CityModelResult.spec_name`.
        formula: patsy formula, `"slope_c_per_decade ~ ..."`.
        cluster_col: column passed as `cov_kwds={"groups": ...}`.
        partial_r2_groups: passed to :func:`partial_r2_by_feature`; `None`
            skips it (e.g. for `baseline`, a single-term spec).
        morans_k: `k` for :func:`morans_i` on the fit's residuals; `None`
            skips it.

    Returns:
        `(CityModelResult, fit)` -- `fit` is the cluster-robust
        `RegressionResultsWrapper`, for callers that want more than the
        summary (e.g. residuals for plotting).
    """
    prelim = smf.ols(formula, data=features, missing="drop").fit()
    used = features.loc[prelim.model.data.row_labels]
    fit = smf.ols(formula, data=used).fit(
        cov_type="cluster", cov_kwds={"groups": used[cluster_col]}
    )

    partial_r2 = partial_r2_by_feature(formula, used, partial_r2_groups) if partial_r2_groups else {}

    moran_i_val: float | None = None
    moran_p: float | None = None
    if morans_k is not None:
        moran_i_val, moran_p = morans_i(
            fit.resid.to_numpy(), used["Longitude"].to_numpy(), used["Latitude"].to_numpy(), k=morans_k
        )

    result = CityModelResult(
        spec_name=spec_name,
        formula=formula,
        n=int(fit.nobs),
        r2=float(fit.rsquared),
        terms=_extract_all_terms(fit),
        partial_r2=partial_r2,
        moran_i=moran_i_val,
        moran_p=moran_p,
    )
    return result, fit


# Three city-level specs, in increasing complexity. "interaction" repeats
# "full" verbatim plus one interaction term -- partial_r2_by_feature relies
# on this textual nesting to strip terms cleanly (see
# CITY_PARTIAL_R2_GROUPS).
CITY_MODEL_SPECS: dict[str, str] = {
    "baseline": "slope_c_per_decade ~ abs_latitude",
    "full": (
        "slope_c_per_decade ~ abs_latitude + elevation_m + coast_km + C(koppen) + station_density"
    ),
    "interaction": (
        "slope_c_per_decade ~ abs_latitude + elevation_m + coast_km + C(koppen) + station_density"
        " + abs_latitude:C(koppen)"
    ),
}

# Per-spec partial-R^2 groupings (Step 4). "interaction" groups the Koeppen
# main effect with its interaction term, so dropping "koppen" removes both.
CITY_PARTIAL_R2_GROUPS: dict[str, dict[str, list[str]]] = {
    "full": {
        "elevation_m": ["elevation_m"],
        "coast_km": ["coast_km"],
        "koppen": ["C(koppen)"],
        "station_density": ["station_density"],
    },
    "interaction": {
        "elevation_m": ["elevation_m"],
        "coast_km": ["coast_km"],
        "koppen": ["C(koppen)", "abs_latitude:C(koppen)"],
        "station_density": ["station_density"],
    },
}


def compare_city_specs(
    features: pd.DataFrame,
    specs: dict[str, str] = CITY_MODEL_SPECS,
    partial_r2_groups: dict[str, dict[str, list[str]]] = CITY_PARTIAL_R2_GROUPS,
    cluster_col: str = "Country",
) -> list[CityModelResult]:
    """Fit every spec in `specs` and return the results side by side.

    Args:
        features: from :func:`build_city_features`.
        specs: name -> formula; defaults to :data:`CITY_MODEL_SPECS`.
        partial_r2_groups: spec name -> groups for
            :func:`partial_r2_by_feature`; specs absent here (e.g.
            `baseline`) skip it.
        cluster_col: passed to :func:`fit_city_model`.

    Returns:
        One :class:`CityModelResult` per spec, in `specs` order.
    """
    return [
        fit_city_model(
            features, name, formula,
            cluster_col=cluster_col,
            partial_r2_groups=partial_r2_groups.get(name),
        )[0]
        for name, formula in specs.items()
    ]


# ---------------------------------------------------------------------
# 6. Country-level coefficient-stability model
# ---------------------------------------------------------------------


def aggregate_features_by_country(features: pd.DataFrame) -> pd.DataFrame:
    """Mean |latitude| per country, from city-location features.

    Args:
        features: from :func:`build_city_features` -- has `Country` and
            `abs_latitude`.

    Returns:
        One row per `Country`, columns `Country` and `mean_abs_lat`.
    """
    return (
        features.groupby("Country", observed=True)["abs_latitude"]
        .mean()
        .reset_index()
        .rename(columns={"abs_latitude": "mean_abs_lat"})
    )


def build_country_table(
    features: pd.DataFrame,
    inequality: pd.DataFrame,
    income: pd.DataFrame,
    overrides: dict[str, str] = BERKELEY_TO_WORLDBANK,
) -> pd.DataFrame:
    """Assemble the country-level model input table.

    Joins :func:`aggregate_features_by_country` onto `inequality` on
    `Country`, attaches `income_group` via :func:`attach_income_group` on
    `owid_country`, then adds `log10_emissions = log10(cum_co2_t_per_capita)`
    -- dropping non-positive emissions first, the same rule as
    :func:`src.emissions.quantify_inequality`.

    Args:
        features: from :func:`build_city_features`.
        inequality: country_inequality.parquet
            (:data:`src.emissions.DEFAULT_INEQUALITY_PATH`).
        income: from :func:`load_income_groups`.
        overrides: passed to :func:`attach_income_group`.

    Returns:
        One row per country: `inequality`'s columns plus `mean_abs_lat`,
        `income_group`, `log10_emissions`.
    """
    mean_lat = aggregate_features_by_country(features)
    table = inequality.merge(mean_lat, on="Country", how="inner")
    table = attach_income_group(table, income, overrides=overrides)

    nonpositive = table["cum_co2_t_per_capita"] <= 0
    if nonpositive.any():
        logger.warning(
            "dropping %d countries with non-positive emissions before log10",
            int(nonpositive.sum()),
        )
        table = table.loc[~nonpositive]

    return table.assign(log10_emissions=np.log10(table["cum_co2_t_per_capita"]))


@dataclass(frozen=True)
class CountryModelResult:
    """One country-level model spec's fit summary."""

    spec_name: str
    formula: str
    n: int
    r2: float
    terms: list[TermFit] = field(default_factory=list)


# Six specs for the log10_emissions coefficient-stability table. "pooled"
# and "continent_fe" reproduce the stored README result (+0.021, +0.029
# [+0.014, +0.045]); "lat_continent" is the key spec -- does +0.029 survive
# a mean-|latitude| control within continents?
COUNTRY_MODEL_SPECS: dict[str, str] = {
    "pooled": "trend_c_per_decade ~ log10_emissions",
    "continent_fe": "trend_c_per_decade ~ log10_emissions + C(continent)",
    "lat": "trend_c_per_decade ~ log10_emissions + mean_abs_lat",
    "lat_continent": "trend_c_per_decade ~ log10_emissions + mean_abs_lat + C(continent)",
    "lat_income": (
        "trend_c_per_decade ~ log10_emissions + mean_abs_lat + C(continent) + C(income_group)"
    ),
    "interaction": (
        "trend_c_per_decade ~ log10_emissions + mean_abs_lat + C(continent)"
        " + log10_emissions:mean_abs_lat"
    ),
}


def fit_country_model(
    country_table: pd.DataFrame, specs: dict[str, str] = COUNTRY_MODEL_SPECS
) -> list[CountryModelResult]:
    """Fit every spec in `specs` with HC1-robust standard errors.

    Args:
        country_table: from :func:`build_country_table`.
        specs: name -> formula; defaults to :data:`COUNTRY_MODEL_SPECS`.

    Returns:
        One :class:`CountryModelResult` per spec, in `specs` order. The
        `log10_emissions` term's coefficient/CI across specs *is* the
        coefficient-stability table: compare `pooled` and `continent_fe`
        against the stored README result, then check whether
        `lat_continent` changes it.
    """
    results = []
    for name, formula in specs.items():
        fit = smf.ols(formula, data=country_table, missing="drop").fit(cov_type="HC1")
        results.append(
            CountryModelResult(
                spec_name=name,
                formula=formula,
                n=int(fit.nobs),
                r2=float(fit.rsquared),
                terms=_extract_all_terms(fit),
            )
        )
    return results


# ---------------------------------------------------------------------
# 7. Serialization helpers (for the app bundle)
# ---------------------------------------------------------------------


def term_to_dict(t: TermFit) -> dict:
    """Serialize a :class:`TermFit` to a JSON-safe dict."""
    return {
        "term": t.term,
        "coef": t.coef,
        "se": t.se,
        "ci_low": t.ci_low,
        "ci_high": t.ci_high,
        "p_value": t.p_value,
    }


def city_result_to_dict(r: CityModelResult) -> dict:
    """Serialize a :class:`CityModelResult` to a JSON-safe dict."""
    return {
        "spec_name": r.spec_name,
        "formula": r.formula,
        "n": r.n,
        "r2": r.r2,
        "terms": [term_to_dict(t) for t in r.terms],
        "partial_r2": dict(r.partial_r2),
        "moran_i": r.moran_i,
        "moran_p": r.moran_p,
    }


def country_result_to_dict(
    r: CountryModelResult, emissions_term: str = "log10_emissions"
) -> dict:
    """Serialize a :class:`CountryModelResult`, extracting the emissions term.

    The ``emissions`` key is a pre-shaped flat dict (coef, se, CI, p) so
    the coefficient-stability view does not need to hunt through ``terms``
    at render time.
    """
    emissions = next(
        (term_to_dict(t) for t in r.terms if t.term == emissions_term), None
    )
    return {
        "spec_name": r.spec_name,
        "formula": r.formula,
        "n": r.n,
        "r2": r.r2,
        "terms": [term_to_dict(t) for t in r.terms],
        "emissions": emissions,
    }


def build_explain_summary(
    city_results: list[CityModelResult],
    country_results: list[CountryModelResult],
) -> dict:
    """Build the ``explain`` stats sub-dict for ``stats.json``.

    Derives convenience scalars so the views do not need to search term
    lists at render time.

    Args:
        city_results: From :func:`compare_city_specs`.
        country_results: From :func:`fit_country_model`.

    Returns:
        Dict with ``city_model`` and ``country_model`` sub-structures.
    """
    city_specs = [city_result_to_dict(r) for r in city_results]

    baseline = next((r for r in city_results if r.spec_name == "baseline"), city_results[0])
    abs_lat_term = next((t for t in baseline.terms if t.term == "abs_latitude"), None)
    abs_lat_baseline = term_to_dict(abs_lat_term) if abs_lat_term else None

    full = next((r for r in city_results if r.spec_name == "full"), None)
    koppen_b: dict | None = None
    moran_i_full: float | None = None
    moran_p_full: float | None = None
    if full is not None:
        koppen_b_term = next((t for t in full.terms if t.term == "C(koppen)[T.B]"), None)
        koppen_b = term_to_dict(koppen_b_term) if koppen_b_term else None
        moran_i_full = full.moran_i
        moran_p_full = full.moran_p

    country_specs = [country_result_to_dict(r) for r in country_results]

    return {
        "city_model": {
            "specs": city_specs,
            "abs_latitude_baseline": abs_lat_baseline,
            "koppen_b_full": koppen_b,
            "moran_i_full": moran_i_full,
            "moran_p_full": moran_p_full,
        },
        "country_model": {
            "specs": country_specs,
        },
    }


def write_explain_summary(
    city_results: list[CityModelResult],
    country_results: list[CountryModelResult],
    features: pd.DataFrame,
    summary_path: Path = DEFAULT_EXPLAIN_SUMMARY_PATH,
    bundle_path: Path = DEFAULT_EXPLAIN_BUNDLE_PATH,
) -> dict[str, Path]:
    """Serialize model results and a slim feature parquet for the app bundle.

    The two output files are read by :mod:`src.app_assets` (if present)
    and merged into the committed ``app/data/`` bundle without requiring
    ETOPO or Koeppen downloads at bundle-build time.

    Args:
        city_results: From :func:`compare_city_specs`.
        country_results: From :func:`fit_country_model`.
        features: From :func:`build_city_features`.
        summary_path: Destination for the JSON stats blob.
        bundle_path: Destination for the 7-column features parquet.

    Returns:
        Dict mapping ``"summary"`` and ``"bundle"`` to their written paths.
    """
    summary = build_explain_summary(city_results, country_results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    slim = features[list(EXPLAIN_BUNDLE_SCHEMA)].copy()
    slim["Latitude"] = slim["Latitude"].astype("float32")
    slim["Longitude"] = slim["Longitude"].astype("float32")
    slim["abs_latitude"] = slim["abs_latitude"].astype("float32")
    slim["slope_c_per_decade"] = slim["slope_c_per_decade"].astype("float32")
    write_typed_parquet(
        slim, bundle_path, EXPLAIN_BUNDLE_SCHEMA, order_by=tuple(CITY_KEYS),
    )

    return {"summary": summary_path, "bundle": bundle_path}


# ---------------------------------------------------------------------
# 8. main()
# ---------------------------------------------------------------------


def main() -> None:
    """Run the Phase 7 pipeline: preflight, features, then both models.

    Aborts before building `city_features.parquet` or fitting any model
    if :func:`run_geo_preflight` fails (the execution-order rule from the
    geo + data integrity preflight).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Fetch every external input before the preflight reads them:
    # run_geo_preflight opens ETOPO + Koeppen (orientation/sampling) and the
    # World Bank income table (country-join check), but ETOPO/Koeppen are
    # otherwise downloaded only lazily inside build_city_features and the income
    # table not at all, so a clean machine would crash at the preflight.
    # (Mirrors validation.main()'s download-then-run pattern.)
    download_file(ETOPO_URL, ETOPO_PATH, timeout=600)
    prepare_koppen_grid()
    download_file(INCOME_URL, INCOME_PATH, timeout=120)

    if not run_geo_preflight():
        raise SystemExit("geo + data integrity preflight FAILED -- see report above")

    features = (
        pd.read_parquet(DEFAULT_FEATURES_PATH) if DEFAULT_FEATURES_PATH.exists() else build_city_features()
    )

    print("\n=== City-level model: slope_c_per_decade ~ geography ===")
    city_results = compare_city_specs(features)
    for r in city_results:
        print(f"\n-- {r.spec_name} (n={r.n}, R2={r.r2:.3f}) --")
        for t in r.terms:
            print(f"  {t.term:30s} {t.coef:+.5f} [{t.ci_low:+.5f}, {t.ci_high:+.5f}] p={t.p_value:.3g}")
        if r.partial_r2:
            print("  partial R2:", {k: round(v, 4) for k, v in r.partial_r2.items()})
        if r.moran_i is not None:
            print(f"  Moran's I = {r.moran_i:.4f} (p={r.moran_p:.3g})")

    baseline = city_results[0]
    abs_lat = next(t for t in baseline.terms if t.term == "abs_latitude")
    print(
        f"\nabs_latitude (baseline): {abs_lat.coef:+.5f} C/decade per degree |latitude| "
        f"[{abs_lat.ci_low:+.5f}, {abs_lat.ci_high:+.5f}]"
    )

    full = next(r for r in city_results if r.spec_name == "full")
    koppen_b = next((t for t in full.terms if t.term == "C(koppen)[T.B]"), None)
    if koppen_b is not None:
        verdict = "supports" if koppen_b.coef > 0 else "does not support"
        print(
            f"Koeppen-B (arid) vs reference (full): {koppen_b.coef:+.5f} "
            f"[{koppen_b.ci_low:+.5f}, {koppen_b.ci_high:+.5f}] p={koppen_b.p_value:.3g} "
            f"-- {verdict} the hotspot hypothesis"
        )
    if full.moran_i is not None:
        sig = "significant" if full.moran_p < 0.05 else "not significant"
        print(f"Moran's I (full residuals): {full.moran_i:.4f} (p={full.moran_p:.3g}) -- {sig}")

    print("\n=== Country-level model: trend_c_per_decade ~ emissions + geography ===")
    inequality = pd.read_parquet(DEFAULT_INEQUALITY_PATH)
    income = load_income_groups()
    country_table = build_country_table(features, inequality, income)
    country_results = fit_country_model(country_table)

    print(f"\n{'spec':<15s} {'n':>4s} {'R2':>6s}  log10_emissions [95% CI]")
    for r in country_results:
        term = next(t for t in r.terms if t.term == "log10_emissions")
        print(
            f"{r.spec_name:<15s} {r.n:4d} {r.r2:6.3f}  {term.coef:+.4f} "
            f"[{term.ci_low:+.4f}, {term.ci_high:+.4f}] p={term.p_value:.3g}"
        )

    print(f"\nwrote {DEFAULT_FEATURES_PATH}")
    summary_paths = write_explain_summary(city_results, country_results, features)
    print(f"summary: {summary_paths['summary']}")
    print(f"bundle:  {summary_paths['bundle']}")


if __name__ == "__main__":
    main()
