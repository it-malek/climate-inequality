"""Geography of warming: national warming map, city trends by latitude, weighting schemes."""

from __future__ import annotations

import streamlit as st

from app import charts, loaders, theme

_DEFINITIONS: dict[str, tuple[str, str]] = {
    "Area-weighted (land area)": ("trend_area", "Area-weighted warming"),
    "Population-weighted (residents)": ("trend_population", "Population-weighted warming"),
    "Station-weighted (city locations)": ("trend_station", "Station-weighted warming"),
}


def render() -> None:
    """Render the geography-of-warming page."""
    stats = loaders.load_stats()
    t = stats["trends"]
    national = loaders.load_national_warming()
    latitudes = loaders.load_country_latitudes()
    era5 = loaders.load_era5_validation_summary()

    st.title("Geography of warming")
    st.markdown(
        f"Every one of the {t['n_locations']:,} city locations warmed over "
        f"{loaders.window_years(t['analysis_window'])}, but not evenly: warming rises "
        "with latitude, and the fastest cluster is the Iranian plateau and Central Asia "
        "rather than the Arctic. The map shows each country's warming rate under the "
        "chosen definition; the scatter shows the city locations themselves."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "City-location mean", f"{t['global_mean_c_per_decade']:.3f} {theme.TREND_UNIT}",
        help="The mean over the city locations: a sample mean of where the stations are, "
        "not a global land mean.",
    )
    c2.metric(
        f"Mean above 60°N ({t['n_arctic']} locations)",
        f"{t['arctic_mean_c_per_decade']:.3f} {theme.TREND_UNIT}",
        help=f"{t['arctic_ratio']:.2f} times the city-location mean.",
    )
    if era5 and era5.get("world_land_mean", {}).get("berkeley_area") is not None:
        c3.metric(
            "World land mean, area-weighted",
            f"{era5['world_land_mean']['berkeley_area']:.3f} {theme.TREND_UNIT}",
            help="cos-latitude mean over every fitted 1° land cell of the Berkeley Earth "
            "grid; Berkeley Earth's published global-land figure is about "
            f"{era5['world_land_mean']['berkeley_reference']}.",
        )

    st.subheader("National warming")
    if national is not None:
        choice = st.radio("Definition of national warming", list(_DEFINITIONS), horizontal=True)
        column, label = _DEFINITIONS[choice]
        df = (
            national.merge(latitudes, on="Country", how="left")
            .rename(columns={"owid_country": "location"})
            .dropna(subset=[column])
        )
        theme.plotly_chart(
            charts.warming_choropleth(df, value_col=column, value_label=label), width="stretch"
        )
        lo, hi = df[column].min(), df[column].max()
        st.caption(
            f"{len(df)} countries; {label.lower()} ranges {lo:.2f} to {hi:.2f} "
            f"{theme.TREND_UNIT}. Color encodes observed warming only."
        )
    else:
        table = loaders.load_inequality()
        location_col = "owid_country" if "owid_country" in table.columns else "Country"
        df = (
            table.merge(latitudes, on="Country", how="left")
            .rename(columns={location_col: "location", "trend_c_per_decade": "warming_trend"})
            .dropna(subset=["warming_trend"])
        )
        theme.plotly_chart(charts.warming_choropleth(df), width="stretch")
        st.caption(
            f"{len(df)} countries; station-weighted warming ranges "
            f"{df['warming_trend'].min():.2f} to {df['warming_trend'].max():.2f} {theme.TREND_UNIT}."
        )

    st.markdown(
        "**Three definitions answer three questions.** The station-weighted rate is the "
        "unweighted mean of a country's city-location trends: the warming of the average "
        "monitoring site. The population-weighted rate weights each location by the "
        "residents around it: population-weighted warming across the monitored locations. The area-weighted rate "
        "fits a trend to every 1° cell of the Berkeley Earth gridded field and averages "
        "over the country's land with cos-latitude weights: the warming of the average "
        "square kilometre. Differences are consistent with station-network geography, but "
        "gridded interpolation also changes the observational representation, so station "
        "siting is not isolated as the cause. The "
        "area-weighted rate is the primary cross-country outcome of this study; it is "
        "still built from the same station record, through Berkeley Earth's interpolation."
    )

    st.subheader("City locations by latitude")
    features = loaders.load_explain_features()
    theme.plotly_chart(charts.city_latitude_scatter(features), width="stretch")
    st.caption(
        "Each point is one city location. The line is the median trend within 10° latitude "
        "bands that hold at least ten locations; it summarizes the sample and is not a model. "
        "Warming steepens toward the northern high latitudes and is lowest in the "
        "10\u201330°N band that holds most of the world's stations."
    )

    with st.expander("Country table"):
        if national is not None:
            cols = ["Country", "continent", "trend_area", "trend_population",
                    "trend_station", "trend_era5_aligned", "cum_co2_t_per_capita"]
            st.dataframe(
                national[cols].sort_values("trend_area", ascending=False),
                width="stretch", hide_index=True,
            )
        else:
            st.dataframe(df, width="stretch", hide_index=True)
