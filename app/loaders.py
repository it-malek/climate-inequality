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

# Must equal src.app_assets.APP_DATA_DIR (tests assert agreement); the app
# cannot import src.app_assets because its pipeline dependencies are absent
# from the deployed environment. Tests monkeypatch this to a synthetic
# bundle and clear the caches.
APP_DATA_DIR = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_city_trends() -> pd.DataFrame:
    """Per-city-location trends with ``city_id``, ``label``, ``intercept``."""
    return pd.read_parquet(APP_DATA_DIR / "city_trends.parquet")


@st.cache_data
def load_anomalies() -> pd.DataFrame:
    """All monthly anomalies, keyed by ``city_id``."""
    return pd.read_parquet(APP_DATA_DIR / "city_anomalies.parquet")


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
    long_form = pd.read_parquet(APP_DATA_DIR / "trend_surface.parquet")
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
    return pd.read_parquet(APP_DATA_DIR / "country_inequality.parquet")


@st.cache_data
def load_stats() -> dict:
    """Headline statistics computed at bundle-build time (stats.json)."""
    return json.loads((APP_DATA_DIR / "stats.json").read_text(encoding="utf-8"))


def window_years(window: str) -> str:
    """Render a stored window like ``1950-01-01..2013-09-01`` as ``1950–2013``."""
    start, end = window.split("..")
    return f"{start[:4]}–{end[:4]}"
