"""Streamlit entry point.

Run from the repository root, locally or on Streamlit Community Cloud
(main file path: ``app/streamlit_app.py``):

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts the script's own directory on sys.path, not the repository
# root; the ``app`` and ``src`` packages live at the root.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st  # noqa: E402

from app import loaders, theme  # noqa: E402
from src import snapshot  # noqa: E402
from app.views import (  # noqa: E402
    about,
    city_explorer,
    coupling,
    decomposition,
    explain,
    inequality,
    interpretation,
    overview,
    physical,
    residual,
    robustness,
    sensitivity,
    trend_map,
    validation,
    vulnerability,
    world_map,
)

st.set_page_config(
    page_title="Climate Inequality",
    layout="wide",
)

try:
    release_identity = snapshot.fingerprint(loaders.APP_DATA_DIR, require_manifest=True)
except (OSError, ValueError) as exc:
    st.error(f"The published result bundle is unavailable: {exc}")
    st.stop()

navigation = st.navigation(
    {
        "The study": [
            st.Page(overview.render, title="Overview", url_path="overview", default=True),
            st.Page(world_map.render, title="Geography of warming", url_path="warming"),
            st.Page(coupling.render, title="Responsibility and warming", url_path="responsibility"),
            st.Page(vulnerability.render, title="Income and vulnerability", url_path="vulnerability"),
            st.Page(decomposition.render, title="Variance decomposition", url_path="decomposition"),
            st.Page(sensitivity.render, title="Stability", url_path="stability"),
            st.Page(residual.render, title="Residual structure", url_path="residual"),
            st.Page(robustness.render, title="Robustness", url_path="robustness"),
            st.Page(interpretation.render, title="Interpretation and outlook", url_path="interpretation"),
        ],
        "Context": [
            st.Page(physical.render, title="Global forcing", url_path="forcing"),
            st.Page(validation.render, title="Post-2013 check", url_path="validation"),
        ],
        "Data": [
            st.Page(trend_map.render, title="Interpolated surface", url_path="surface"),
            st.Page(city_explorer.render, title="City explorer", url_path="cities"),
            st.Page(inequality.render, title="Emissions regression", url_path="emissions"),
            st.Page(explain.render, title="City-level drivers", url_path="drivers"),
            st.Page(about.render, title="Data and methods", url_path="about"),
        ],
    }
)
st.sidebar.caption(
    "Cross-country structure of observed land warming, 1950\u20132013. "
    "Berkeley Earth temperatures, Our World in Data emissions. "
    "Every page reads committed result records; nothing is recomputed on load."
)

st.sidebar.caption(f"Snapshot 2.0.0 | Bundle {release_identity[:16]}")

theme.interpretation_banner()
navigation.run()
