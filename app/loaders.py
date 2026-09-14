"""Cached read-only loaders for the committed ``app/data`` bundle.

Thin by design: each loader parses one bundle file (built by
:mod:`src.app_assets`) and caches it with ``st.cache_data``, so files are
read once per server process rather than once per interaction. The app
never touches ``data/`` or DuckDB -- the bundle is the single data source
both locally and on Streamlit Community Cloud.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src import bundle
from src.bundle import APP_DATA_DIR  # noqa: F401  (tests point this at a synthetic bundle)
from src.cleaning import parse_window


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
            f"{path} is missing column(s) {missing}; the bundle is stale — "
            "rebuild it (python -m src.app_assets)"
        )
    return df


@st.cache_data
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


@st.cache_data
def load_anomalies() -> pd.DataFrame:
    """All monthly anomalies, keyed by ``city_id``."""
    return _read_bundle_parquet(
        bundle.ANOMALIES_ASSET, required=("city_id", "dt", "anomaly")
    )


@st.cache_data
def load_city_series(city_id: int) -> pd.DataFrame:
    """One city-location's (dt, anomaly) series, cached per city."""
    anomalies = load_anomalies()
    rows = anomalies.loc[anomalies["city_id"] == city_id, ["dt", "anomaly"]]
    return rows.reset_index(drop=True)


@st.cache_data
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


@st.cache_data
def load_inequality() -> pd.DataFrame:
    """The country-level inequality table (one row per matched country)."""
    return _read_bundle_parquet(
        bundle.INEQUALITY_ASSET,
        required=(
            "Country", "continent", "n_cities",
            "trend_c_per_decade", "cum_co2_t_per_capita",
        ),
    )


@st.cache_data
def load_coupling() -> pd.DataFrame:
    """The responsibility-impact comparison table (one row per country)."""
    return _read_bundle_parquet(
        bundle.COUPLING_ASSET,
        required=(
            "Country", "responsibility_index_v1", "impact_index_v1",
            "responsibility_rank", "impact_rank", "rank_gap", "z_gap",
        ),
    )


@st.cache_data
def load_stats() -> dict:
    """Headline statistics computed at bundle-build time (stats.json)."""
    return json.loads((APP_DATA_DIR / bundle.STATS_ASSET).read_text(encoding="utf-8"))


def _load_optional_json(name: str) -> dict | None:
    """Read an optional bundle JSON, returning None when it is absent.

    Several summaries depend on inputs that may not have been built; a page
    degrades to a 'not built yet' state rather than erroring when its summary
    is missing from the bundle.
    """
    path = APP_DATA_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_inequality_summary() -> dict | None:
    """Descriptive warming-inequality metrics (``inequality_summary.json``)."""
    return _load_optional_json(bundle.INEQUALITY_SUMMARY_ASSET)


@st.cache_data
def load_decomposition_summary() -> dict | None:
    """Group LMG/Shapley variance shares (``decomposition_summary.json``)."""
    return _load_optional_json(bundle.DECOMPOSITION_SUMMARY_ASSET)


@st.cache_data
def load_stability_summary() -> dict | None:
    """Bootstrap / leave-one-out / Moran's I diagnostics, if present.

    Returns ``None`` when ``stability_summary.json`` is absent so the page
    renders a pending state rather than failing.
    """
    return _load_optional_json(bundle.STABILITY_SUMMARY_ASSET)


@st.cache_data
def load_physical_summary() -> dict | None:
    """Physical-model summary (``physical_summary.json``), if present.

    Returns ``None`` until the physical-model artifacts are added to the bundle (they require
    the network-derived ``forcings.parquet``), so the physical-model page renders an
    explicit pending state rather than failing.
    """
    return _load_optional_json(bundle.PHYSICAL_SUMMARY_ASSET)


@st.cache_data
def load_physical_trajectory() -> pd.DataFrame | None:
    """Observed-vs-predicted global temperature trajectory, if present.

    Returns ``None`` when ``physical_trajectory.parquet`` is absent from the bundle
    (the page degrades to a pending state). When present, the expected columns are
    validated so a stale bundle surfaces one clear error.
    """
    path = APP_DATA_DIR / bundle.PHYSICAL_TRAJECTORY_ASSET
    if not path.exists():
        return None
    return _read_bundle_parquet(
        bundle.PHYSICAL_TRAJECTORY_ASSET,
        required=("year", "observed", "predicted_mean", "lower95", "upper95", "in_sample"),
    )


@st.cache_data
def load_coupling_summary() -> dict | None:
    """Deterministic responsibility-impact comparator metrics, if present.

    Returns ``None`` until a ``coupling_summary.json`` is added to the bundle, so
    the responsibility-vs-impact page renders an explicit pending state rather
    than failing.
    """
    return _load_optional_json(bundle.COUPLING_SUMMARY_ASSET)


@st.cache_data
def load_coupling_consumption_summary() -> dict | None:
    """Two-pass consumption-lens comparator metrics (PCS v2), if present.

    Returns ``None`` until ``coupling_consumption_summary.json`` is added to the
    bundle (it needs the additive consumption columns in the country table), so
    the responsibility page's consumption comparison degrades gracefully.
    """
    return _load_optional_json(bundle.COUPLING_CONSUMPTION_SUMMARY_ASSET)


@st.cache_data
def load_coupling_consumption() -> pd.DataFrame | None:
    """The consumption-lens diagnostic table, if present.

    Returns ``None`` when ``coupling_consumption.parquet`` is absent; when present,
    the expected columns are validated so a stale bundle surfaces one clear error.
    """
    path = APP_DATA_DIR / bundle.COUPLING_CONSUMPTION_ASSET
    if not path.exists():
        return None
    return _read_bundle_parquet(
        bundle.COUPLING_CONSUMPTION_ASSET,
        required=(
            "Country", "impact_index_v1", "responsibility_index_consumption",
            "responsibility_index_production_matched", "production_matched_rank",
            "consumption_rank", "prod_to_cons_rank_gap", "prod_to_cons_z_gap",
            "impact_rank", "consumption_impact_z_gap",
        ),
    )


@st.cache_data
def load_coupling_exposure_summary() -> dict | None:
    """Two-pass people-weighted exposure comparator metrics (PCS v2), if present.

    Returns ``None`` until ``coupling_exposure_summary.json`` is in the bundle (it
    needs the people-weighted column, which needs the population grid at build
    time), so the responsibility page's exposure comparison degrades gracefully.
    """
    return _load_optional_json(bundle.COUPLING_EXPOSURE_SUMMARY_ASSET)


@st.cache_data
def load_coupling_exposure() -> pd.DataFrame | None:
    """The people-weighted exposure diagnostic table, if present."""
    path = APP_DATA_DIR / bundle.COUPLING_EXPOSURE_ASSET
    if not path.exists():
        return None
    return _read_bundle_parquet(
        bundle.COUPLING_EXPOSURE_ASSET,
        required=(
            "Country", "responsibility_index_v1", "impact_index_v1",
            "impact_index_population_weighted", "station_rank", "people_rank",
            "station_to_people_rank_gap", "station_to_people_z_gap",
            "responsibility_rank", "people_responsibility_z_gap",
        ),
    )


@st.cache_data
def load_coupling_area_summary() -> dict | None:
    """Two-pass area-weighted exposure comparator metrics (PCS v2), if present.

    Returns ``None`` until ``coupling_area_summary.json`` is in the bundle (it
    needs the area-weighted column, which needs the Berkeley grid at build time),
    so the responsibility page's area comparison degrades gracefully.
    """
    return _load_optional_json(bundle.COUPLING_AREA_SUMMARY_ASSET)


@st.cache_data
def load_coupling_area() -> pd.DataFrame | None:
    """The area-weighted exposure diagnostic table, if present."""
    path = APP_DATA_DIR / bundle.COUPLING_AREA_ASSET
    if not path.exists():
        return None
    return _read_bundle_parquet(
        bundle.COUPLING_AREA_ASSET,
        required=(
            "Country", "responsibility_index_v1", "impact_index_v1",
            "impact_index_area_weighted", "station_rank", "area_rank",
            "station_to_area_rank_gap", "station_to_area_z_gap",
            "responsibility_rank", "area_responsibility_z_gap",
        ),
    )


@st.cache_data
def load_era5_validation_summary() -> dict | None:
    """ERA5 vs Berkeley area-weighted cross-check metrics, if present.

    Returns ``None`` until ``era5_validation_summary.json`` is in the bundle (it
    needs the ERA5 grid fetched via ``scripts/fetch_era5.py`` at build time), so the
    validation page's cross-check panel degrades gracefully.
    """
    return _load_optional_json(bundle.ERA5_VALIDATION_SUMMARY_ASSET)


@st.cache_data
def load_vulnerability_summary() -> dict | None:
    """Income-stratified exposure x vulnerability lens metrics, if present.

    Returns ``None`` until ``vulnerability_summary.json`` is in the bundle (it needs
    the in-repo World Bank income CSV at build time), so the triple-inequality page
    renders an explicit pending state rather than failing.
    """
    return _load_optional_json(bundle.VULNERABILITY_SUMMARY_ASSET)


@st.cache_data
def load_vulnerability_strata() -> pd.DataFrame | None:
    """Per-country income-stratified warming/responsibility points, if present.

    Returns ``None`` when ``vulnerability_strata.parquet`` is absent from the bundle
    (the page degrades to a pending state). When present, the expected columns are
    validated so a stale bundle surfaces one clear error.
    """
    path = APP_DATA_DIR / bundle.VULNERABILITY_STRATA_ASSET
    if not path.exists():
        return None
    return _read_bundle_parquet(
        bundle.VULNERABILITY_STRATA_ASSET,
        required=(
            "owid_country", "continent", "income_group", "income_rank",
            "population", "cum_co2_t_per_capita", "trend_c_per_decade_area_weighted",
            "trend_c_per_decade_pop_weighted", "trend_c_per_decade",
        ),
    )


@st.cache_data
def load_country_latitudes() -> pd.DataFrame:
    """Mean signed latitude per country, aggregated from the city features.

    Derived in the app layer (no new statistical artifact) so the country map
    can show a latitude in its hover. One row per ``Country``.
    """
    features = load_explain_features()
    return (
        features.groupby("Country", as_index=False)["Latitude"]
        .mean()
        .rename(columns={"Latitude": "mean_latitude"})
    )


@st.cache_data
def load_validation_frame() -> pd.DataFrame:
    """Per-city residual-map data from the out-of-sample validation."""
    return _read_bundle_parquet(
        bundle.VALIDATION_ASSET,
        required=("City", "Country", "Latitude", "Longitude",
                  "mean_residual", "overlap_r", "gate_pass"),
    )


@st.cache_data
def load_validation_global() -> pd.DataFrame:
    """Monthly global observed vs predicted anomalies from the validation stage."""
    return _read_bundle_parquet(
        bundle.VALIDATION_GLOBAL_ASSET, required=("dt", "observed", "predicted")
    )


@st.cache_data
def load_explain_features() -> pd.DataFrame:
    """Slim city-features table (for the drivers scatter)."""
    return _read_bundle_parquet(
        bundle.EXPLAIN_FEATURES_ASSET,
        required=("City", "Country", "abs_latitude", "slope_c_per_decade", "koppen"),
    )


def window_years(window: str) -> str:
    """Render a stored window like ``1950-01-01..2013-09-01`` as ``1950–2013``."""
    start, end = parse_window(window)
    return f"{start[:4]}–{end[:4]}"
