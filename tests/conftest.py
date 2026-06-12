"""Shared fixtures: a synthetic dashboard bundle built by the real builder.

The bundle drives both the builder tests (schemas, integrity) and the
AppTest page tests, so it goes through the genuine pipeline: synthetic
DuckDB -> build_city_trends -> build_app_assets, with a synthetic land
polygon instead of the Natural Earth download (no network, no real data).
"""

from pathlib import Path

import duckdb
import pytest
from shapely.geometry import box

from src.app_assets import build_app_assets
from src.trends import build_city_trends
from tests.test_emissions import make_inequality_frame
from tests.test_trends import make_city_frame, make_db

# Everything east of the prime meridian is "land" in the synthetic world,
# so the surface has both interpolated and ocean (NaN) cells.
SYNTHETIC_LAND = box(0.0, -60.0, 180.0, 85.0)

SYNTHETIC_CITIES = [
    # (city, country, lat, lon, slope °C/decade) — six locations, spread
    # out, one above 60°N, and one same-named pair at distinct coordinates.
    ("Alpha", "Kenya", 5.0, 35.0, 0.10),
    ("Beta", "Australia", -25.0, 135.0, 0.05),
    ("Arcticville", "Norway", 65.0, 20.0, 0.30),
    ("Springfield", "United States", 30.0, -90.0, 0.12),
    ("Springfield", "United States", 45.0, -90.0, 0.20),
    ("Gamma", "Brazil", -10.0, -55.0, 0.08),
]


def make_synthetic_inputs(root: Path) -> dict:
    """Synthetic DuckDB, trends parquet, and inequality parquet under `root`."""
    db_path = root / "climate.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        make_db(
            con,
            [
                make_city_frame(
                    city, country, lat, lon,
                    slope_per_decade=slope, noise_sd=0.05, seed=i,
                )
                for i, (city, country, lat, lon, slope) in enumerate(SYNTHETIC_CITIES)
            ],
        )
    finally:
        con.close()
    trends_path = root / "city_trends.parquet"
    build_city_trends(db_path, trends_path)
    inequality_path = root / "country_inequality.parquet"
    make_inequality_frame().to_parquet(inequality_path, index=False)
    return {
        "db_path": db_path,
        "trends_path": trends_path,
        "inequality_path": inequality_path,
    }


def build_synthetic_bundle(root: Path) -> dict:
    """Run build_app_assets end to end on synthetic inputs under `root`."""
    inputs = make_synthetic_inputs(root)
    result = build_app_assets(
        **inputs,
        out_dir=root / "bundle",
        surface_out_dir=root / "outputs",
        k=3,
        resolution=15.0,
        land=SYNTHETIC_LAND,
    )
    result["bundle_dir"] = root / "bundle"
    return result


@pytest.fixture(scope="session")
def synthetic_bundle(tmp_path_factory) -> dict:
    """One session-wide synthetic bundle (read-only for tests)."""
    return build_synthetic_bundle(tmp_path_factory.mktemp("app_bundle"))
