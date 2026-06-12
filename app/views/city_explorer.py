"""City explorer page: one location's anomaly series and fitted trend."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import loaders
from src.figures import render_city_anomaly_series

_DEFAULT_COUNTRY = "United States"


def render() -> None:
    """Render the city-explorer page."""
    trends = loaders.load_city_trends()
    stats = loaders.load_stats()
    t = stats["trends"]

    st.title("City explorer")
    st.markdown(
        f"Monthly temperature anomalies and the fitted Theil–Sen trend for any "
        f"of the {t['n_locations']:,} city-locations, "
        f"{loaders.window_years(t['analysis_window'])}."
    )

    countries = sorted(trends["Country"].unique())
    default_index = (
        countries.index(_DEFAULT_COUNTRY) if _DEFAULT_COUNTRY in countries else 0
    )
    left, right = st.columns(2)
    country = left.selectbox("Country", countries, index=default_index)
    cities = trends.loc[trends["Country"] == country].sort_values("label")
    label = right.selectbox("City", cities["label"].tolist())
    row = cities.loc[cities["label"] == label].iloc[0]

    one, two, three, four = st.columns(4)
    one.metric("Theil–Sen trend", f"{row['slope_c_per_decade']:+.3f} °C/decade")
    two.metric("95% CI", f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")
    three.metric("Months observed", f"{int(row['n_obs']):,}")
    four.metric("Coverage", f"{row['coverage']:.0%}")

    series = loaders.load_city_series(int(row["city_id"]))
    fig = render_city_anomaly_series(
        series,
        slope=float(row["slope_c_per_decade"]),
        intercept=float(row["intercept"]),
        title=(
            f"{row['City']}, {row['Country']} "
            f"({row['Latitude']:.2f}°, {row['Longitude']:.2f}°)"
        ),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Anomalies vs this location's {loaders.window_years(t['baseline_window'])} "
        f"monthly climatology. OLS slope for comparison: "
        f"{row['ols_slope']:+.3f} °C/decade."
    )

    with st.expander("Location on map"):
        st.map(
            pd.DataFrame({"lat": [row["Latitude"]], "lon": [row["Longitude"]]}),
            zoom=4,
        )
