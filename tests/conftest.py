"""Shared fixtures: a synthetic dashboard bundle built by the real builder.

The bundle drives both the builder tests (schemas, integrity) and the
AppTest page tests, so it goes through the genuine pipeline: synthetic
DuckDB -> build_city_trends -> build_app_assets, with a synthetic land
polygon instead of the Natural Earth download (no network, no real data).

Phase 6/7 artifacts (validation_summary.json, explain_summary.json, etc.)
are written to a tmp ``processed/`` directory and passed to build_app_assets
so both new pages render under AppTest without touching real data/processed/.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.app_assets import build_app_assets
from src.decomposition import group_lmg_shares
from src.decomposition import summary_payload as decomp_payload
from src.explain import (
    compare_city_specs,
    fit_country_model,
    write_explain_summary,
    write_typed_parquet,
)
from src.inequality import country_warming_inequality
from src.inequality import summary_payload as ineq_payload
from src.trends import build_city_trends
from src.validation import VALIDATION_BUNDLE_SCHEMA, VALIDATION_GLOBAL_SCHEMA
from tests.test_decomposition import make_country_design
from tests.test_emissions import make_inequality_frame
from tests.test_explain import make_synthetic_city_features, make_synthetic_country_table
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


def _make_synthetic_validation_artifacts(processed: Path) -> dict[str, Path]:
    """Write minimal Phase 6 summary artifacts under `processed/`."""
    stats = {
        "n_locations": 6,
        "n_no_grid": 0,
        "n_gate_pass": 5,
        "median_overlap_r": 0.95,
        "forecast_start": "2013-10-01",
        "record_end": "2024-12-01",
        "n_forecast_months": 135,
        "mean_residual": 0.30,
        "mean_residual_pre2023": 0.25,
        "mean_slope_stored": 0.146,
        "mean_slope_overlap_grid": 0.142,
        "mean_slope_full": 0.200,
        "mean_slope_delta": 0.054,
        "slope_delta_ci_low": 0.040,
        "slope_delta_ci_high": 0.068,
    }
    processed.mkdir(parents=True, exist_ok=True)
    summary_path = processed / "validation_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    # Slim residual-map parquet (matches VALIDATION_BUNDLE_SCHEMA).
    cities = ["Alpha", "Beta", "Arcticville", "Springfield", "Gamma"]
    bundle_df = pd.DataFrame(
        {
            "City": cities,
            "Country": ["Kenya", "Australia", "Norway", "United States", "Brazil"],
            "Latitude": np.array([5.0, -25.0, 65.0, 30.0, -10.0], dtype="float32"),
            "Longitude": np.array([35.0, 135.0, 20.0, -90.0, -55.0], dtype="float32"),
            "mean_residual": np.array([0.3, 0.2, 0.5, 0.4, 0.1], dtype="float32"),
            "overlap_r": np.array([0.98, 0.97, 0.99, 0.96, 0.95], dtype="float32"),
            "gate_pass": [True, True, True, True, True],
        }
    )
    bundle_path = processed / "validation_bundle.parquet"
    write_typed_parquet(
        bundle_df, bundle_path, VALIDATION_BUNDLE_SCHEMA,
        order_by=("Country", "City", "Latitude", "Longitude"),
    )

    # Global monthly parquet.
    months = pd.date_range("1950-01-01", "2024-12-01", freq="MS")
    global_df = pd.DataFrame(
        {
            "dt": months,
            "observed": np.random.default_rng(0).uniform(-0.5, 1.5, len(months)).astype("float32"),
            "predicted": np.linspace(-0.3, 1.0, len(months)).astype("float32"),
        }
    )
    global_path = processed / "validation_global.parquet"
    write_typed_parquet(
        global_df, global_path, VALIDATION_GLOBAL_SCHEMA, order_by=("dt",),
    )

    return {
        "summary": summary_path,
        "bundle": bundle_path,
        "global": global_path,
    }


def _make_synthetic_explain_artifacts(processed: Path) -> dict[str, Path]:
    """Write minimal Phase 7 summary artifacts under `processed/`.

    Routes through the real write_explain_summary so the AppTest bundle
    exercises the production serializer (schema, float32 casts, full-key
    ordering) rather than a hand-rolled copy that could drift from it.
    """
    features = make_synthetic_city_features()
    country_table = make_synthetic_country_table()
    city_results = compare_city_specs(features)
    country_results = fit_country_model(country_table)

    processed.mkdir(parents=True, exist_ok=True)
    return write_explain_summary(
        city_results,
        country_results,
        features,
        summary_path=processed / "explain_summary.json",
        bundle_path=processed / "explain_features.parquet",
    )


def _make_synthetic_decomposition_artifacts(bundle_dir: Path) -> None:
    """Write synthetic inequality/decomposition summaries into the bundle.

    Routed through the production serializers (``summary_payload``) so the
    decomposition dashboard's AppTest exercises real bundle JSON, including the
    ``interpretation`` disclaimer, rather than a hand-rolled stand-in.
    """
    ineq = country_warming_inequality(make_inequality_frame())
    (bundle_dir / "inequality_summary.json").write_text(
        json.dumps(ineq_payload(ineq), indent=2) + "\n", encoding="utf-8"
    )
    decomp = group_lmg_shares(make_country_design(n=120))
    (bundle_dir / "decomposition_summary.json").write_text(
        json.dumps(decomp_payload(decomp), indent=2) + "\n", encoding="utf-8"
    )


def _make_synthetic_coupling_artifacts(bundle_dir: Path) -> None:
    """Write synthetic L3 coupling artifacts into the bundle via the real builders.

    Routed through the production ``build_coupling_summary_asset`` (v1
    responsibility-vs-impact) and ``build_coupling_consumption_asset`` (PCS v2
    consumption lens) so both pages' AppTests exercise real bundle artifacts.
    """
    from src.app_assets import (
        build_coupling_area_asset,
        build_coupling_consumption_asset,
        build_coupling_exposure_asset,
        build_coupling_summary_asset,
    )

    inequality_path = bundle_dir / "country_inequality.parquet"
    build_coupling_summary_asset(inequality_path=inequality_path, out_dir=bundle_dir)
    build_coupling_consumption_asset(inequality_path=inequality_path, out_dir=bundle_dir)
    build_coupling_exposure_asset(inequality_path=inequality_path, out_dir=bundle_dir)
    build_coupling_area_asset(inequality_path=inequality_path, out_dir=bundle_dir)


def _make_synthetic_physical_artifacts(bundle_dir: Path) -> None:
    """Write synthetic L1 physical-model artifacts into the bundle via the real builder.

    Uses the physical model's own generative fixture (varied, identifiable drivers
    spanning the 2013 train/test split) and routes it through the production
    ``build_physical_summary_asset`` so the physical-model page's AppTest and the
    shipped-bundle checks exercise real L1 artifacts (``physical_trajectory.parquet``
    + ``physical_summary.json``).
    """
    from src.app_assets import build_physical_summary_asset
    from tests.test_physical_model import make_synthetic_forcings

    forcings_path = bundle_dir / "forcings.parquet"
    make_synthetic_forcings().to_parquet(forcings_path, index=False)
    build_physical_summary_asset(forcings_path=forcings_path, out_dir=bundle_dir)


def build_synthetic_bundle(root: Path) -> dict:
    """Run build_app_assets end to end on synthetic inputs under `root`."""
    inputs = make_synthetic_inputs(root)
    processed = root / "processed"
    val_paths = _make_synthetic_validation_artifacts(processed)
    exp_paths = _make_synthetic_explain_artifacts(processed)
    result = build_app_assets(
        **inputs,
        out_dir=root / "bundle",
        surface_out_dir=root / "outputs",
        k=3,
        resolution=15.0,
        land=SYNTHETIC_LAND,
        validation_summary_path=val_paths["summary"],
        validation_bundle_path=val_paths["bundle"],
        validation_global_path=val_paths["global"],
        explain_summary_path=exp_paths["summary"],
        explain_bundle_path=exp_paths["bundle"],
    )
    _make_synthetic_decomposition_artifacts(root / "bundle")
    _make_synthetic_coupling_artifacts(root / "bundle")
    _make_synthetic_physical_artifacts(root / "bundle")
    result["bundle_dir"] = root / "bundle"
    return result


@pytest.fixture(scope="session")
def synthetic_bundle(tmp_path_factory) -> dict:
    """One session-wide synthetic bundle (read-only for tests)."""
    return build_synthetic_bundle(tmp_path_factory.mktemp("app_bundle"))
