"""Validation page: did the 1950–2013 trends hold out of sample?"""

from __future__ import annotations

import streamlit as st

from app import loaders
from src.figures import render_residual_map, render_validation_series

_NOT_BUILT = (
    "The validation bundle has not been built yet. "
    "Run `python -m src.validation` then `python -m src.app_assets` to populate this page."
)


def render() -> None:
    """Render the out-of-sample validation page."""
    stats = loaders.load_stats()
    if "validation" not in stats:
        st.title("Did the 1950–2013 trends hold out of sample?")
        st.info(_NOT_BUILT)
        return

    v = stats["validation"]
    forecast_window = f"{v['forecast_start'][:7]}–{v['record_end'][:7]}"

    st.title("Did the 1950–2013 trends hold out of sample?")
    st.markdown(
        f"Forecast window: **{forecast_window}** ({v['n_forecast_months']} months). "
        f"{v['n_gate_pass']} of {v['n_locations']} city-locations passed the "
        f"overlap-agreement gate (grid cell r ≥ 0.80 over 1950–2013)."
    )

    left, middle, right = st.columns(3)
    left.metric(
        "Overlap agreement (median r)",
        f"{v['median_overlap_r']:.2f}",
        help="Pearson r between pipeline and gridded Berkeley Earth anomalies "
             "over the 1950–2013 analysis window. Values near 1 mean the grid "
             "cell tracks the city series well.",
    )
    middle.metric(
        "Mean forecast residual",
        f"{v['mean_residual']:+.2f} °C",
        help=(
            "Observed grid anomaly minus the stored Theil–Sen prediction, "
            f"averaged over gate-passing cities and {forecast_window}. "
            f"Excluding the 2023–24 El Niño: {v['mean_residual_pre2023']:+.2f} °C. "
            "Positive = stored lines underpredict (acceleration)."
        ),
    )
    right.metric(
        "Slope: full record vs stored",
        f"{v['mean_slope_full']:.3f} vs {v['mean_slope_stored']:.3f} °C/decade",
        help=(
            f"Grid full-record slope {v['mean_slope_full']:.3f} °C/decade "
            f"vs the stored 1950–2013 slope {v['mean_slope_stored']:.3f} °C/decade. "
            f"Mean acceleration: {v['mean_slope_delta']:+.4f} "
            f"[{v['slope_delta_ci_low']:+.4f}, {v['slope_delta_ci_high']:+.4f}] "
            "°C/decade (city-level CI; spatial correlation makes it optimistic)."
        ),
    )

    global_df = loaders.load_validation_global()
    fig_series = render_validation_series(
        global_df,
        forecast_start=v["forecast_start"],
        title="",
    )
    st.plotly_chart(fig_series, use_container_width=True)
    st.caption(
        "Global mean land anomaly (gate-passing cities) vs the extrapolated "
        "stored Theil–Sen fit. The dotted line marks the out-of-sample boundary "
        "(October 2013). Observations pulling above the dashed prediction after "
        "2013 indicate that warming accelerated beyond what the 1950–2013 lines forecast."
    )

    frame = loaders.load_validation_frame()
    fig_map = render_residual_map(frame, title="")
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(
        "Per-city mean forecast residual (observed − predicted). "
        "Red = the stored trend underpredicted observed warming; "
        "blue = overpredicted. Grey × markers are gated-out cities "
        "(grid cell did not track the city series in the overlap period — "
        "typically islands and coastal locations where the 1° land average "
        "diverges from the city's series)."
    )

    with st.expander("Per-city validation table"):
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            frame.to_csv(index=False).encode("utf-8"),
            file_name="validation_residuals.csv",
            mime="text/csv",
        )
