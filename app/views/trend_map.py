"""Warming map page: interpolated trend surface with a city-station layer."""

from __future__ import annotations

import streamlit as st

from app import loaders
from src.figures import build_trend_map

_MODES = {
    "Interpolated surface": "surface",
    "City stations": "cities",
    "Both": "both",
}


def render() -> None:
    """Render the warming-map page."""
    stats = loaders.load_stats()
    trends = loaders.load_city_trends()
    grid_lon, grid_lat, surface = loaders.load_surface()
    t = stats["trends"]
    interp = stats["interpolation"]

    st.title("Where has land warmed fastest?")
    st.markdown(
        f"Per-city warming trends, {loaders.window_years(t['analysis_window'])}, "
        f"fit on {t['n_locations']:,} Berkeley Earth city-locations and "
        "interpolated into a continuous land surface."
    )

    left, middle, right = st.columns(3)
    left.metric(
        "Global mean trend", f"{t['global_mean_c_per_decade']:.3f} °C/decade"
    )
    middle.metric(
        f">60°N mean ({t['n_arctic']} locations)",
        f"{t['arctic_mean_c_per_decade']:.3f} °C/decade",
    )
    right.metric("Arctic amplification", f"{t['arctic_ratio']:.2f}×")

    choice = st.radio("Map layer", list(_MODES), horizontal=True)
    fig = build_trend_map(
        grid_lon, grid_lat, surface, trends, mode=_MODES[choice], title=""
    )
    st.plotly_chart(fig, width="stretch")

    loo = {row["method"]: row["rmse"] for row in interp["cv_leave_location_out"]}
    st.caption(
        f"Theil–Sen slopes on monthly anomalies vs each location's "
        f"{loaders.window_years(t['baseline_window'])} climatology. Surface: "
        f"{interp['winner'].upper()} interpolation (k={interp['k']} neighbors, "
        f"{interp['resolution_deg']:g}° grid), masked to land; ocean cells are "
        f"blank because these are land stations. Leave-location-out CV RMSE: "
        f"IDW {loo['idw']:.4f} vs kriging {loo['kriging']:.4f} °C/decade."
    )
