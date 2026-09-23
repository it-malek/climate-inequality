"""Responsibility and warming: cumulative emissions against experienced warming.

Reads the committed coupling artifacts and the national warming table; renders
a pending state if the coupling summary is absent.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, loaders, theme

_NOT_BUILT = (
    "The responsibility comparison has not been built yet. Run `python -m "
    "src.projections` and `python -m src.coupling`, then rebuild the bundle "
    "(`python -m src.app_assets`)."
)


def _leaders_frame(rows: list) -> pd.DataFrame:
    """Two-column (Country, z_gap) frame for a leaders list."""
    return pd.DataFrame(rows, columns=["Country", "z_gap"])


def render() -> None:
    """Render the responsibility-and-warming page."""
    summary = loaders.load_coupling_summary()
    if summary is None:
        st.info(_NOT_BUILT)
        return
    table = loaders.load_coupling()
    national = loaders.load_national_warming()
    residual = loaders.load_residual_summary()

    st.title("Historical responsibility and experienced warming")
    st.markdown(
        "Responsibility is cumulative production-based CO₂ through 2013 per 2013 resident. "
        "Whether it lines up with how fast a country warmed depends on how national "
        "warming is defined: the alignment that appears under the station-weighted "
        "definition is weak once every unit of land area counts equally, under "
        "Berkeley Earth and under ERA5 alike."
    )

    if national is not None and residual is not None:
        comparison = residual["responsibility_comparison"]
        theme.plotly_chart(charts.responsibility_panels(national, comparison), width="stretch")
        rows = pd.DataFrame(comparison)[["label", "n", "spearman_rho", "spearman_p"]].rename(
            columns={"label": "Definition of national warming", "n": "Countries",
                     "spearman_rho": "Spearman ρ vs responsibility", "spearman_p": "p"}
        )
        st.dataframe(
            rows, width="stretch", hide_index=True,
            column_config={
                "Spearman ρ vs responsibility": st.column_config.NumberColumn(format="%+.3f"),
                "p": st.column_config.NumberColumn(format="%.2g"),
            },
        )
        st.caption(
            "Same responsibility axis in every panel; the sample changes with the definition "
            "(three countries have no 1° land cell, and the ERA5 comparison uses the 151 "
            "decomposition countries). The two area-weighted definitions come from different "
            "observational products and agree on a weak, non-significant rank "
            "correlation."
        )

    st.markdown(
        "**Why emissions do not stay local.** CO₂ is long-lived and mixes globally, "
        "so the forcing from any country's emissions is "
        "spread over the whole planet. Where warming is fastest is set by physical "
        "geography, regional feedbacks and variability, not by who emitted. A correlation "
        "between responsibility and warming may therefore reflect that high emitters "
        "happen to sit in fast-warming places, or that their stations do."
    )

    st.divider()
    st.subheader("Station-weighted comparison")
    left, middle, right = st.columns(3)
    left.metric(
        "Spearman ρ",
        f"{summary['spearman_rho']:+.2f}",
        help="Rank correlation between responsibility and station-weighted warming.",
    )
    middle.metric(
        "Inequality coefficient",
        f"{summary['inequality_coefficient']:.2f}",
        help="Twice the area between the cumulative-warming versus cumulative-responsibility "
        "curve and the diagonal (0 aligned, 1 maximally divergent).",
    )
    right.metric(
        "High warming, low responsibility",
        f"{summary['n_high_impact_low_responsibility']}",
        help="Countries whose standardized warming exceeds their standardized responsibility.",
    )

    theme.plotly_chart(charts.lorenz_chart(table), width="stretch")
    theme.plotly_chart(charts.mismatch_scatter(table), width="stretch")

    c1, c2 = st.columns(2)
    c1.subheader("Warmed most, emitted least")
    c1.dataframe(
        _leaders_frame(summary["top_suffer_least_cause"]),
        width="stretch",
        hide_index=True,
    )
    c2.subheader("Emitted most, warmed least")
    c2.dataframe(
        _leaders_frame(summary["top_cause_least_suffer"]),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Country table"):
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name="coupling.csv",
            mime="text/csv",
        )

    exposure = loaders.load_coupling_exposure_summary()
    area = loaders.load_coupling_area_summary()
    if exposure is not None or area is not None:
        _render_weighting_section(
            station_table=table,
            station_coeff=summary["inequality_coefficient"],
            exposure_summary=exposure,
            exposure_table=loaders.load_coupling_exposure(),
            area_summary=area,
            area_table=loaders.load_coupling_area(),
        )

    consumption = loaders.load_coupling_consumption_summary()
    if consumption is not None:
        _render_consumption_section(consumption, loaders.load_coupling_consumption())


def _render_consumption_section(summary: dict, table: pd.DataFrame | None) -> None:
    """Consumption-based responsibility against the window-matched production baseline."""
    window = summary["window"]
    shift = summary["production_to_consumption_shift"]
    cons_impact = summary["consumption_vs_impact"]

    st.divider()
    st.header("Counting emissions where goods are consumed")
    st.markdown(
        "Counting CO₂ where goods are consumed rather than produced moves responsibility "
        "from exporters to importers. Both cumulatives are summed over each country's "
        "consumption-available window (the consumption series starts around 1990), so the "
        f"comparison covers {window['n_countries']} countries with start years "
        f"{window['consumption_start_year_min']}\u2013{window['consumption_start_year_max']}."
    )

    left, right = st.columns(2)
    left.metric(
        "Production to consumption rank ρ",
        f"{shift['spearman_rho']:+.2f}",
        help="Spearman correlation between the production- and consumption-based "
        "responsibility rankings over the shared window (1 means the ranking is unchanged).",
    )
    right.metric(
        "Consumption inequality coefficient",
        f"{cons_impact['inequality_coefficient']:.2f}",
        help="The inequality coefficient of station-weighted warming against "
        "consumption-based responsibility.",
    )

    c1, c2 = st.columns(2)
    c1.subheader("More responsible under consumption")
    c1.caption("Net importers: rank rises when emissions follow consumption.")
    c1.dataframe(
        _leaders_frame(shift["top_suffer_least_cause"]),
        width="stretch",
        hide_index=True,
    )
    c2.subheader("Less responsible under consumption")
    c2.caption("Net exporters: rank falls when emissions follow consumption.")
    c2.dataframe(
        _leaders_frame(shift["top_cause_least_suffer"]),
        width="stretch",
        hide_index=True,
    )

    if table is not None:
        with st.expander("Consumption accounting: country table"):
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                "Download CSV",
                table.to_csv(index=False).encode("utf-8"),
                file_name="coupling_consumption.csv",
                mime="text/csv",
            )


def _render_weighting_section(
    *,
    station_table: pd.DataFrame,
    station_coeff: float,
    exposure_summary: dict | None,
    exposure_table: pd.DataFrame | None,
    area_summary: dict | None,
    area_table: pd.DataFrame | None,
) -> None:
    """Population- and area-weighted comparisons behind one basis toggle."""
    lenses: dict[str, dict] = {}
    if exposure_summary is not None and exposure_table is not None:
        lenses["Population-weighted (residents)"] = {
            "table": exposure_table,
            "impact_col": "impact_index_population_weighted",
            "z_col": "station_to_people_z_gap",
            "basis_label": "population-weighted",
            "shift": exposure_summary["station_vs_people"],
            "coeff": exposure_summary["people_weighted_inequality"]["inequality_coefficient"],
            "n_countries": exposure_summary["coverage"]["n_countries"],
            "rank_label": "Station to population rank ρ",
            "lorenz_title": "Cumulative population-weighted warming vs cumulative responsibility",
            "more_header": "Residents warm more than the stations suggest",
            "more_caption": "Population sits in the country's faster-warming regions.",
            "less_header": "Stations overstate residents' warming",
            "less_caption": "Monitoring over-samples fast-warming, sparsely populated areas.",
            "csv": "coupling_exposure.csv",
        }
    if area_summary is not None and area_table is not None:
        lenses["Area-weighted (land)"] = {
            "table": area_table,
            "impact_col": "impact_index_area_weighted",
            "z_col": "station_to_area_z_gap",
            "basis_label": "area-weighted",
            "shift": area_summary["station_vs_area"],
            "coeff": area_summary["area_weighted_inequality"]["inequality_coefficient"],
            "n_countries": area_summary["coverage"]["n_countries"],
            "rank_label": "Station to area rank ρ",
            "lorenz_title": "Cumulative area-weighted warming vs cumulative responsibility",
            "more_header": "Warm more once every square kilometre counts equally",
            "more_caption": "Fast-warming interiors are under-sampled by stations.",
            "less_header": "Stations overstate the land-average warming",
            "less_caption": "Stations cluster in the faster-warming regions.",
            "csv": "coupling_area.csv",
        }
    if not lenses:
        return

    st.divider()
    st.header("Changing the definition of national warming")
    st.markdown(
        "The station-weighted comparison above weights every monitoring location equally, so "
        "dense mid-latitude clusters dominate. Re-weighting by residents barely changes the "
        "picture, because people live near stations. Re-weighting by land area changes it a "
        "great deal: the rank correlation with responsibility collapses and the inequality "
        "coefficient rises."
    )

    basis = st.radio(
        "Inequality basis",
        ["Station-weighted", *lenses],
        horizontal=True,
        help="Switch the Lorenz curve and inequality coefficient between weighting every "
        "station equally, by population, and by land area.",
    )

    if basis == "Station-weighted":
        m1, *_ = st.columns(2)
        m1.metric(
            "Inequality coefficient",
            f"{station_coeff:.2f}",
            help="Cumulative station-weighted warming against cumulative responsibility.",
        )
        theme.plotly_chart(
            charts.lorenz_chart(station_table), width="stretch", key="weighting_lorenz"
        )
        return

    cfg = lenses[basis]
    table = cfg["table"]
    shift = cfg["shift"]

    m1, m2 = st.columns(2)
    m1.metric(
        "Inequality coefficient",
        f"{cfg['coeff']:.2f}",
        delta=f"{cfg['coeff'] - station_coeff:+.2f} vs station-weighted",
        delta_color="off",
        help="Cumulative warming against cumulative responsibility under the selected basis.",
    )
    m2.metric(
        cfg["rank_label"],
        f"{shift['spearman_rho']:+.2f}",
        help=f"How much the country ranking changes under the {cfg['basis_label']} basis "
        f"(1 means unchanged), across {cfg['n_countries']} countries.",
    )

    theme.plotly_chart(
        charts.lorenz_chart(table, impact_col=cfg["impact_col"], title=cfg["lorenz_title"]),
        width="stretch",
        key="weighting_lorenz",
    )
    theme.plotly_chart(
        charts.exposure_shift_scatter(
            table,
            impact_col=cfg["impact_col"],
            z_col=cfg["z_col"],
            basis_label=cfg["basis_label"],
        ),
        width="stretch",
        key="weighting_shift",
    )

    c1, c2 = st.columns(2)
    c1.subheader(cfg["more_header"])
    c1.caption(cfg["more_caption"])
    c1.dataframe(
        _leaders_frame(shift["top_suffer_least_cause"]),
        width="stretch",
        hide_index=True,
    )
    c2.subheader(cfg["less_header"])
    c2.caption(cfg["less_caption"])
    c2.dataframe(
        _leaders_frame(shift["top_cause_least_suffer"]),
        width="stretch",
        hide_index=True,
    )

    with st.expander(f"{cfg['basis_label'].capitalize()} basis: country table"):
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name=cfg["csv"],
            mime="text/csv",
        )
    st.caption(
        f"Units: {theme.TREND_UNIT} for warming, tonnes of CO₂ per 2013 resident for responsibility."
    )
