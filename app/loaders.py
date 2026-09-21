"""Cached read-only loaders for the committed ``app/data`` bundle.

Each loader parses one bundle file (built by :mod:`src.app_assets` or
:mod:`src.residual_assets`) and caches it with ``st.cache_data``, with verified content identity in every cache key. The app never touches ``data/`` or DuckDB; the
bundle is the single data source both locally and on Streamlit Community
Cloud, and nothing is downloaded or recomputed at run time.
"""

from __future__ import annotations

import json
from functools import wraps

import numpy as np
import pandas as pd
import streamlit as st

from src import bundle, snapshot
from src.bundle import APP_DATA_DIR  # noqa: F401  (tests point this at a synthetic bundle)
from src.cleaning import parse_window


@st.cache_data(show_spinner=False, max_entries=128)
def _cached_load(name, identity, args, kwargs, _loader):
    return _loader(*args, **kwargs)


def _snapshot_cached(loader):
    """Include verified file content in every loader's cache key."""
    @wraps(loader)
    def read(*args, **kwargs):
        identity = snapshot.fingerprint(APP_DATA_DIR)
        return _cached_load(loader.__name__, identity, args, kwargs, loader)
    return read


def _read_bundle_parquet(name: str, required: tuple[str, ...]) -> pd.DataFrame:
    """Read one bundle parquet, failing loudly if expected columns are gone.

    The pages dereference these columns far from the read; checking here
    turns a stale or hand-edited bundle into one clear error.
    """
    path = APP_DATA_DIR / name
    df = pd.read_parquet(path)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"{path} is missing column(s) {missing}; the bundle is stale, "
            "rebuild it (python -m src.app_assets)"
        )
    return df


def _load_optional_json(name: str) -> dict | None:
    """Read an optional bundle JSON, returning None when it is absent.

    Several summaries depend on inputs that may not have been built; a page
    then renders a pending state rather than erroring.
    """
    path = APP_DATA_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_parquet(name: str, required: tuple[str, ...]) -> pd.DataFrame | None:
    path = APP_DATA_DIR / name
    if not path.exists():
        return None
    return _read_bundle_parquet(name, required)


@_snapshot_cached
def load_city_trends() -> pd.DataFrame:
    """Per-city-location trends with ``city_id``, ``label``, ``intercept``."""
    return _read_bundle_parquet(
        bundle.TRENDS_ASSET,
        required=(
            "City", "Country", "Latitude", "Longitude", "n_obs", "coverage",
            "slope_c_per_decade", "ci_low", "ci_high", "ols_slope",
            "city_id", "label", "intercept",
        ),
    )


@_snapshot_cached
def load_anomalies() -> pd.DataFrame:
    """All monthly anomalies, keyed by ``city_id``."""
    return _read_bundle_parquet(
        bundle.ANOMALIES_ASSET, required=("city_id", "dt", "anomaly")
    )


@_snapshot_cached
def load_city_series(city_id: int) -> pd.DataFrame:
    """One city-location's (dt, anomaly) series, cached per city."""
    anomalies = load_anomalies()
    rows = anomalies.loc[anomalies["city_id"] == city_id, ["dt", "anomaly"]]
    return rows.reset_index(drop=True)


@_snapshot_cached
def load_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The interpolated surface as (grid_lon, grid_lat, values).

    Pivots the long-form parquet back to the 2D grid; NaN cells are ocean
    (land-masked upstream).
    """
    long_form = _read_bundle_parquet(
        bundle.SURFACE_ASSET, required=("lat", "lon", "value")
    )
    wide = (
        long_form.pivot(index="lat", columns="lon", values="value")
        .sort_index()
        .sort_index(axis=1)
    )
    return (
        wide.columns.to_numpy(dtype=float),
        wide.index.to_numpy(dtype=float),
        wide.to_numpy(dtype=float),
    )


@_snapshot_cached
def load_inequality() -> pd.DataFrame:
    """The country-level table (one row per matched country)."""
    return _read_bundle_parquet(
        bundle.INEQUALITY_ASSET,
        required=(
            "Country", "continent", "n_cities",
            "trend_c_per_decade", "cum_co2_t_per_capita",
        ),
    )


@_snapshot_cached
def load_coupling() -> pd.DataFrame:
    """The responsibility-warming comparison table (one row per country)."""
    return _read_bundle_parquet(
        bundle.COUPLING_ASSET,
        required=(
            "Country", "responsibility_index_v1", "impact_index_v1",
            "responsibility_rank", "impact_rank", "rank_gap", "z_gap",
        ),
    )


@_snapshot_cached
def load_stats() -> dict:
    """Headline statistics computed at bundle-build time (stats.json)."""
    return json.loads((APP_DATA_DIR / bundle.STATS_ASSET).read_text(encoding="utf-8"))


@_snapshot_cached
def load_inequality_summary() -> dict | None:
    """Descriptive warming-inequality metrics (``inequality_summary.json``)."""
    return _load_optional_json(bundle.INEQUALITY_SUMMARY_ASSET)


@_snapshot_cached
def load_decomposition_summary() -> dict | None:
    """Group Shapley variance shares (``decomposition_summary.json``)."""
    return _load_optional_json(bundle.DECOMPOSITION_SUMMARY_ASSET)


@_snapshot_cached
def load_stability_summary() -> dict | None:
    """Bootstrap, leave-one-out and Moran's I diagnostics, if present."""
    return _load_optional_json(bundle.STABILITY_SUMMARY_ASSET)


@_snapshot_cached
def load_physical_summary() -> dict | None:
    """Global forcing regression summary (``physical_summary.json``), if present."""
    return _load_optional_json(bundle.PHYSICAL_SUMMARY_ASSET)


@_snapshot_cached
def load_physical_trajectory() -> pd.DataFrame | None:
    """Observed-vs-predicted global temperature trajectory, if present."""
    return _load_optional_parquet(
        bundle.PHYSICAL_TRAJECTORY_ASSET,
        required=("year", "observed", "predicted_mean", "lower95", "upper95", "in_sample"),
    )


@_snapshot_cached
def load_coupling_summary() -> dict | None:
    """Responsibility-warming comparison metrics, if present."""
    return _load_optional_json(bundle.COUPLING_SUMMARY_ASSET)


@_snapshot_cached
def load_coupling_consumption_summary() -> dict | None:
    """Consumption-based responsibility comparison metrics, if present."""
    return _load_optional_json(bundle.COUPLING_CONSUMPTION_SUMMARY_ASSET)


@_snapshot_cached
def load_coupling_consumption() -> pd.DataFrame | None:
    """The consumption-lens diagnostic table, if present."""
    return _load_optional_parquet(
        bundle.COUPLING_CONSUMPTION_ASSET,
        required=(
            "Country", "impact_index_v1", "responsibility_index_consumption",
            "responsibility_index_production_matched", "production_matched_rank",
            "consumption_rank", "prod_to_cons_rank_gap", "prod_to_cons_z_gap",
            "impact_rank", "consumption_impact_z_gap",
        ),
    )


@_snapshot_cached
def load_coupling_exposure_summary() -> dict | None:
    """Population-weighted exposure comparison metrics, if present."""
    return _load_optional_json(bundle.COUPLING_EXPOSURE_SUMMARY_ASSET)


@_snapshot_cached
def load_coupling_exposure() -> pd.DataFrame | None:
    """The population-weighted exposure diagnostic table, if present."""
    return _load_optional_parquet(
        bundle.COUPLING_EXPOSURE_ASSET,
        required=(
            "Country", "responsibility_index_v1", "impact_index_v1",
            "impact_index_population_weighted", "station_rank", "people_rank",
            "station_to_people_rank_gap", "station_to_people_z_gap",
            "responsibility_rank", "people_responsibility_z_gap",
        ),
    )


@_snapshot_cached
def load_coupling_area_summary() -> dict | None:
    """Area-weighted exposure comparison metrics, if present."""
    return _load_optional_json(bundle.COUPLING_AREA_SUMMARY_ASSET)


@_snapshot_cached
def load_coupling_area() -> pd.DataFrame | None:
    """The area-weighted exposure diagnostic table, if present."""
    return _load_optional_parquet(
        bundle.COUPLING_AREA_ASSET,
        required=(
            "Country", "responsibility_index_v1", "impact_index_v1",
            "impact_index_area_weighted", "station_rank", "area_rank",
            "station_to_area_rank_gap", "station_to_area_z_gap",
            "responsibility_rank", "area_responsibility_z_gap",
        ),
    )


@_snapshot_cached
def load_era5_validation_summary() -> dict | None:
    """ERA5 versus Berkeley Earth area-weighted cross-check metrics, if present."""
    return _load_optional_json(bundle.ERA5_VALIDATION_SUMMARY_ASSET)


@_snapshot_cached
def load_vulnerability_summary() -> dict | None:
    """Income- and vulnerability-stratified metrics, if present."""
    return _load_optional_json(bundle.VULNERABILITY_SUMMARY_ASSET)


@_snapshot_cached
def load_vulnerability_strata() -> pd.DataFrame | None:
    """Per-country income-stratified warming and responsibility, if present."""
    return _load_optional_parquet(
        bundle.VULNERABILITY_STRATA_ASSET,
        required=(
            "owid_country", "continent", "income_group", "income_rank",
            "population", "cum_co2_t_per_capita", "trend_c_per_decade_area_weighted",
            "trend_c_per_decade_pop_weighted", "trend_c_per_decade",
        ),
    )


@_snapshot_cached
def load_residual_summary() -> dict | None:
    """The residual-structure investigation summary, if present."""
    return _load_optional_json(bundle.RESIDUAL_SUMMARY_ASSET)


@_snapshot_cached
def load_national_warming() -> pd.DataFrame | None:
    """National warming under four definitions plus responsibility, if present."""
    return _load_optional_parquet(
        bundle.NATIONAL_WARMING_ASSET,
        required=(
            "Country", "owid_country", "continent", "cum_co2_t_per_capita",
            "trend_station", "trend_population", "trend_area", "trend_era5_aligned",
        ),
    )


@_snapshot_cached
def load_country_latitudes() -> pd.DataFrame:
    """Mean signed latitude per country, aggregated from the city features.

    Derived in the app (no new statistical artifact) so the country map can
    show a latitude in its hover. One row per ``Country``.
    """
    features = load_explain_features()
    return (
        features.groupby("Country", as_index=False)["Latitude"]
        .mean()
        .rename(columns={"Latitude": "mean_latitude"})
    )


@_snapshot_cached
def load_validation_frame() -> pd.DataFrame:
    """Per-city residual-map data from the out-of-sample validation."""
    return _read_bundle_parquet(
        bundle.VALIDATION_ASSET,
        required=("City", "Country", "Latitude", "Longitude",
                  "mean_residual", "overlap_r", "gate_pass"),
    )


@_snapshot_cached
def load_validation_global() -> pd.DataFrame:
    """Monthly global observed vs predicted anomalies from the validation stage."""
    return _read_bundle_parquet(
        bundle.VALIDATION_GLOBAL_ASSET, required=("dt", "observed", "predicted")
    )


@_snapshot_cached
def load_explain_features() -> pd.DataFrame:
    """Slim city-features table (latitude, trend, climate class)."""
    return _read_bundle_parquet(
        bundle.EXPLAIN_FEATURES_ASSET,
        required=("City", "Country", "Latitude", "abs_latitude", "slope_c_per_decade", "koppen"),
    )


def window_years(window: str) -> str:
    """Render a stored window like ``1950-01-01..2013-09-01`` as ``1950-2013`` with an en dash."""
    start, end = parse_window(window)
    return f"{start[:4]}\u2013{end[:4]}"
