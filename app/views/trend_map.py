"""Interpolated surface page: the city-location trends as a continuous land surface."""

from __future__ import annotations

import streamlit as st

from app import loaders, theme
from src.figures import build_trend_map

_MODES = {
    "Interpolated surface": "surface",
    "City locations": "cities",
    "Both": "both",
}


def render() -> None:
    """Render the interpolated-surface page."""
    stats = loaders.load_stats()
    trends = loaders.load_city_trends()
    grid_lon, grid_lat, surface = loaders.load_surface()
    t = stats["trends"]
    interp = stats["interpolation"]

    st.title("The interpolated warming surface")
    st.markdown(
        f"Per-city warming trends, {loaders.window_years(t['analysis_window'])}, "
        f"fitted on {t['n_locations']:,} Berkeley Earth city locations and "
        "interpolated into a continuous land surface. The surface shows the shape of the "
        "station sample; the national rates on the other pages come from the Berkeley "
        "Earth gridded product, not from this interpolation."
    )

    left, middle, right = st.columns(3)
    left.metric(
        "City-location mean", f"{t['global_mean_c_per_decade']:.3f} °C/decade",
        help="A sample mean over the city locations, not a global land mean.",
    )
    middle.metric(
        f"Mean above 60°N ({t['n_arctic']} locations)",
        f"{t['arctic_mean_c_per_decade']:.3f} °C/decade",
    )
    right.metric("Ratio to the city-location mean", f"{t['arctic_ratio']:.2f}")

    choice = st.radio("Map layer", list(_MODES), horizontal=True)
    fig = build_trend_map(
        grid_lon, grid_lat, surface, trends, mode=_MODES[choice], title=""
    )
    theme.plotly_chart(fig, width="stretch")

    loo = {row["method"]: row["rmse"] for row in interp["cv_leave_location_out"]}
    st.caption(
        f"Theil\u2013Sen slopes on monthly anomalies against each location's "
        f"{loaders.window_years(t['baseline_window'])} climatology. Surface: "
        f"{interp['winner'].upper()} interpolation (k = {interp['k']} neighbours, "
        f"{interp['resolution_deg']:g}° grid), masked to land; ocean cells are blank "
        "because these are land stations. Leave-location-out cross-validation RMSE: "
        f"IDW {loo['idw']:.4f} vs kriging {loo['kriging']:.4f} °C/decade."
    )
