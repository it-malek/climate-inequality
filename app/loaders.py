"""Cached read-only loaders for the committed ``app/data`` bundle.

Thin by design: each loader parses one bundle file (built by
:mod:`src.app_assets`) and caches it with ``st.cache_data``, so files are
read once per server process rather than once per interaction. The app
never touches ``data/`` or DuckDB -- the bundle is the single data source
both locally and on Streamlit Community Cloud.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# src.cleaning is pandas-only, safe for the deployed environment (the app
# already imports src.figures, which depends on it).
from src.cleaning import parse_window

# Must equal src.app_assets.APP_DATA_DIR (tests assert agreement); the app
# cannot import src.app_assets because its pipeline dependencies are absent
# from the deployed environment. Tests monkeypatch this to a synthetic
# bundle and clear the caches.
APP_DATA_DIR = Path(__file__).resolve().parent / "data"


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
        "city_trends.parquet",
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
        "city_anomalies.parquet", required=("city_id", "dt", "anomaly")
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
        "trend_surface.parquet", required=("lat", "lon", "value")
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
        "country_inequality.parquet",
        required=(
            "Country", "continent", "n_cities",
            "trend_c_per_decade", "cum_co2_t_per_capita",
        ),
    )


@st.cache_data
def load_stats() -> dict:
    """Headline statistics computed at bundle-build time (stats.json)."""
    return json.loads((APP_DATA_DIR / "stats.json").read_text(encoding="utf-8"))


@st.cache_data
def load_validation_frame() -> pd.DataFrame:
    """Per-city residual-map data from Phase 6 validation."""
    return _read_bundle_parquet(
        "validation.parquet",
        required=("City", "Country", "Latitude", "Longitude",
                  "mean_residual", "overlap_r", "gate_pass"),
    )


@st.cache_data
def load_validation_global() -> pd.DataFrame:
    """Monthly global observed vs predicted anomalies from Phase 6."""
    return _read_bundle_parquet(
        "validation_global.parquet", required=("dt", "observed", "predicted")
    )


@st.cache_data
def load_explain_features() -> pd.DataFrame:
    """Slim city-features table from Phase 7 (for the drivers scatter)."""
    return _read_bundle_parquet(
        "explain_features.parquet",
        required=("City", "Country", "abs_latitude", "slope_c_per_decade", "koppen"),
    )


def window_years(window: str) -> str:
    """Render a stored window like ``1950-01-01..2013-09-01`` as ``1950–2013``."""
    start, end = parse_window(window)
    return f"{start[:4]}–{end[:4]}"
