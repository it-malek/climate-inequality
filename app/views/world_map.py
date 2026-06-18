"""Country warming map: choropleth of warming trend with rich hover."""

from __future__ import annotations

import streamlit as st

from app import charts, loaders, theme


def render() -> None:
    """Render the interactive country warming-trend map."""
    st.title("Where warming is fastest")
    st.markdown(
        "Each country is colored by its mean 1950–2013 warming trend. Hover for "
        "the trend, cumulative per-capita CO₂ and mean latitude — the three "
        "quantities the decomposition relates."
    )

    table = loaders.load_inequality()
    latitudes = loaders.load_country_latitudes()

    location_col = "owid_country" if "owid_country" in table.columns else "Country"
    df = (
        table.merge(latitudes, on="Country", how="left")
        .rename(columns={location_col: "location", "trend_c_per_decade": "warming_trend"})
        .dropna(subset=["warming_trend"])
    )

    st.plotly_chart(charts.warming_choropleth(df), width="stretch")

    lo = df["warming_trend"].min()
    hi = df["warming_trend"].max()
    st.caption(
        f"{len(df)} countries · warming trend ranges {lo:.2f}–{hi:.2f} "
        f"{theme.TREND_UNIT} (sequential **{theme.TREND_COLORSCALE}** scale). "
        "Country means are the unweighted average of city-location trends "
        "(station-weighted, not area- or population-weighted). Color encodes "
        "*observed* warming only — not a modeled or causal quantity."
    )

    with st.expander("Country table (sortable)"):
        cols = ["location", "warming_trend", "cum_co2_t_per_capita", "mean_latitude", "continent"]
        st.dataframe(
            df[[c for c in cols if c in df.columns]].sort_values(
                "warming_trend", ascending=False
            ),
            width="stretch",
            hide_index=True,
        )
