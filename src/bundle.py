"""Names and location of the committed dashboard bundle (``app/data/``).

Dependency-free on purpose: the deployed Streamlit app imports this module to
find its inputs, and the pipeline imports it to know where to write them. The
builders live in :mod:`src.app_assets`; the readers in :mod:`app.loaders`.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DATA_DIR = PROJECT_ROOT / "app" / "data"

# Core assets, always written by src.app_assets.build_app_assets.
TRENDS_ASSET = "city_trends.parquet"
ANOMALIES_ASSET = "city_anomalies.parquet"
SURFACE_ASSET = "trend_surface.parquet"
INEQUALITY_ASSET = "country_inequality.parquet"
STATS_ASSET = "stats.json"

# Optional findings merged into stats.json / copied alongside it when the
# validation and explanatory stages have run.
VALIDATION_ASSET = "validation.parquet"
VALIDATION_GLOBAL_ASSET = "validation_global.parquet"
EXPLAIN_FEATURES_ASSET = "explain_features.parquet"

# Decomposition and its stability diagnostics.
INEQUALITY_SUMMARY_ASSET = "inequality_summary.json"
DECOMPOSITION_SUMMARY_ASSET = "decomposition_summary.json"
STABILITY_SUMMARY_ASSET = "stability_summary.json"

# Responsibility-impact coupling: the station-based comparison and the three
# alternative lenses.
COUPLING_ASSET = "coupling.parquet"
COUPLING_SUMMARY_ASSET = "coupling_summary.json"
COUPLING_CONSUMPTION_ASSET = "coupling_consumption.parquet"
COUPLING_CONSUMPTION_SUMMARY_ASSET = "coupling_consumption_summary.json"
COUPLING_EXPOSURE_ASSET = "coupling_exposure.parquet"
COUPLING_EXPOSURE_SUMMARY_ASSET = "coupling_exposure_summary.json"
COUPLING_AREA_ASSET = "coupling_area.parquet"
COUPLING_AREA_SUMMARY_ASSET = "coupling_area_summary.json"

# Cross-checks and stratifications.
ERA5_AREA_TRENDS_ASSET = "era5_area_trends.parquet"
ERA5_VALIDATION_SUMMARY_ASSET = "era5_validation_summary.json"
VULNERABILITY_STRATA_ASSET = "vulnerability_strata.parquet"
VULNERABILITY_SUMMARY_ASSET = "vulnerability_summary.json"

# Global temperature vs radiative forcing.
PHYSICAL_TRAJECTORY_ASSET = "physical_trajectory.parquet"
PHYSICAL_SUMMARY_ASSET = "physical_summary.json"

# Residual-structure investigation (copied from research/model_v2/outputs by
# src.residual_assets) and the national warming table it presents.
RESIDUAL_SUMMARY_ASSET = "residual_structure_summary.json"
NATIONAL_WARMING_ASSET = "national_warming.parquet"
