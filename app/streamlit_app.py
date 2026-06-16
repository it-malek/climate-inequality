"""Streamlit entry point.

Run from the repository root, locally or on Streamlit Community Cloud
(main file path: ``app/streamlit_app.py``):

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts the script's own directory on sys.path, not the repo
# root; the `app` and `src` packages live at the root.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st  # noqa: E402

from app.views import city_explorer, explain, inequality, trend_map, validation  # noqa: E402

st.set_page_config(
    page_title="Climate Inequality",
    page_icon="🌍",
    layout="wide",
)

navigation = st.navigation(
    [
        st.Page(trend_map.render, title="Warming map", icon="🗺️", default=True),
        st.Page(
            city_explorer.render, title="City explorer", icon="🏙️", url_path="cities"
        ),
        st.Page(
            inequality.render,
            title="Climate inequality",
            icon="⚖️",
            url_path="inequality",
        ),
        st.Page(
            validation.render,
            title="Did the trends hold?",
            icon="🔭",
            url_path="validation",
        ),
        st.Page(
            explain.render,
            title="What drives warming?",
            icon="🧭",
            url_path="drivers",
        ),
    ]
)
st.sidebar.caption(
    "Berkeley Earth city temperatures (1950–2013) × OWID cumulative CO₂. "
    "Reads the committed app/data bundle; methods in the README."
)
navigation.run()
