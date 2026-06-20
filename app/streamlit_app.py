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

from app import theme  # noqa: E402
from app.views import (  # noqa: E402
    city_explorer,
    coupling,
    decomposition,
    explain,
    inequality,
    sensitivity,
    trend_map,
    validation,
    world_map,
)

st.set_page_config(
    page_title="Climate Inequality",
    page_icon="🌍",
    layout="wide",
)

# Grouped navigation: the decomposition framework is the headline; the upstream
# data/methods pages are kept as "Foundations".
navigation = st.navigation(
    {
        "Decomposition": [
            st.Page(
                decomposition.render,
                title="Inequality decomposition",
                icon="⚖️",
                url_path="decomposition",
                default=True,
            ),
            st.Page(
                world_map.render, title="Warming map", icon="🗺️", url_path="map"
            ),
            st.Page(
                sensitivity.render,
                title="How confident are we?",
                icon="🎯",
                url_path="sensitivity",
            ),
            st.Page(
                coupling.render,
                title="Responsibility vs impact",
                icon="⚖️",
                url_path="coupling",
            ),
        ],
        "Foundations": [
            st.Page(
                trend_map.render, title="Interpolated surface", icon="🌍",
                url_path="surface",
            ),
            st.Page(
                city_explorer.render, title="City explorer", icon="🏙️",
                url_path="cities",
            ),
            st.Page(
                inequality.render, title="Emissions vs warming", icon="📈",
                url_path="emissions",
            ),
            st.Page(
                validation.render, title="Did the trends hold?", icon="🔭",
                url_path="validation",
            ),
            st.Page(
                explain.render, title="What drives warming?", icon="🧭",
                url_path="drivers",
            ),
        ],
    }
)
st.sidebar.caption(
    "**Decomposition** is the main story; **Foundations** shows how the "
    "warming, emissions and geography inputs were built.  \n"
    "Data: Berkeley Earth city temperatures (1950–2013) × OWID cumulative CO₂. "
    "Reads the committed `app/data` bundle; methods in the README."
)

# The interpretation boundary is shown at the top of every page (rendered here,
# before the selected page's body).
theme.interpretation_banner()
navigation.run()
