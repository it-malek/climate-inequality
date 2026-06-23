"""Responsibility vs impact page: the Layer 3 deterministic projection comparator.

Reads the committed ``coupling_summary.json`` and ``coupling.parquet`` bundle
artifacts; renders a pending state if the summary is absent.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, loaders

_NOT_BUILT = (
    "The coupling artifacts have not been built yet. Run `python -m "
    "src.projections` and `python -m src.coupling`, then rebuild the bundle "
    "(`python -m src.app_assets`)."
)


def _leaders_frame(rows: list) -> pd.DataFrame:
    """Two-column (Country, z_gap) frame for a leaders list."""
    return pd.DataFrame(rows, columns=["Country", "z_gap"])


def render() -> None:
    """Render the responsibility-impact comparator page."""
    summary = loaders.load_coupling_summary()
    if summary is None:
        st.info(_NOT_BUILT)
        return
    table = loaders.load_coupling()

    st.title("Responsibility vs impact")
    st.markdown(
        "Per-country comparison of two PCS projections — `responsibility_index_v1` "
        "(cumulative per-capita CO₂) and `impact_index_v1` (warming rate) — across "
        f"{len(table)} countries."
    )

    left, middle, right = st.columns(3)
    left.metric(
        "Spearman ρ",
        f"{summary['spearman_rho']:+.2f}",
        help="Rank correlation between responsibility and impact.",
    )
    middle.metric(
        "Inequality coefficient",
        f"{summary['inequality_coefficient']:.2f}",
        help="Gini-style divergence of cumulative impact from cumulative "
        "responsibility (0 = aligned, 1 = maximally divergent).",
    )
    right.metric(
        "High impact, low responsibility",
        f"{summary['n_high_impact_low_responsibility']}",
        help="Countries whose standardized impact exceeds their standardized "
        "responsibility (z-gap > 0).",
    )

    st.plotly_chart(charts.lorenz_chart(table), width="stretch")
    st.plotly_chart(charts.mismatch_scatter(table), width="stretch")

    c1, c2 = st.columns(2)
    c1.subheader("Suffer most, caused least")
    c1.dataframe(
        _leaders_frame(summary["top_suffer_least_cause"]),
        width="stretch",
        hide_index=True,
    )
    c2.subheader("Caused most, suffer least")
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

    # Consumption lens (PCS v2). Degrades silently when the artifacts are absent
    # (no extra st.info, so the pending-state contract stays exactly one info).
    consumption = loaders.load_coupling_consumption_summary()
    if consumption is not None:
        _render_consumption_section(consumption, loaders.load_coupling_consumption())

    # People-weighted exposure lens (PCS v2). Also degrades silently.
    exposure = loaders.load_coupling_exposure_summary()
    if exposure is not None:
        _render_exposure_section(
            exposure,
            loaders.load_coupling_exposure(),
            table,
            summary["inequality_coefficient"],
        )


def _render_consumption_section(summary: dict, table: pd.DataFrame | None) -> None:
    """Side-by-side consumption-based responsibility comparison (PCS v2)."""
    window = summary["window"]
    shift = summary["production_to_consumption_shift"]
    cons_impact = summary["consumption_vs_impact"]

    st.divider()
    st.header("Counting emissions where goods are consumed")
    st.markdown(
        "Re-counting CO₂ where goods are **consumed** rather than **produced** "
        "moves responsibility from exporters to importers. To keep the comparison "
        "honest, both cumulatives are summed over each country's "
        "consumption-available window (OWID's consumption series only starts "
        f"~1990), spanning {window['n_countries']} countries with start years "
        f"{window['consumption_start_year_min']}–{window['consumption_start_year_max']}."
    )

    left, right = st.columns(2)
    left.metric(
        "Production → consumption rank ρ",
        f"{shift['spearman_rho']:+.2f}",
        help="Spearman correlation between the production- and consumption-based "
        "responsibility rankings over the shared window (1 = ranking unchanged).",
    )
    right.metric(
        "Consumption inequality coefficient",
        f"{cons_impact['inequality_coefficient']:.2f}",
        help="Gini-style divergence of warming impact from consumption-based "
        "responsibility (the v1 lens, recomputed on consumption accounting).",
    )

    c1, c2 = st.columns(2)
    c1.subheader("More responsible under consumption")
    c1.caption("Net importers — rank rises when emissions follow consumption.")
    c1.dataframe(
        _leaders_frame(shift["top_suffer_least_cause"]),
        width="stretch",
        hide_index=True,
    )
    c2.subheader("Less responsible under consumption")
    c2.caption("Net exporters — rank falls when emissions follow consumption.")
    c2.dataframe(
        _leaders_frame(shift["top_cause_least_suffer"]),
        width="stretch",
        hide_index=True,
    )

    if table is not None:
        with st.expander("Consumption lens — country table"):
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                "Download CSV",
                table.to_csv(index=False).encode("utf-8"),
                file_name="coupling_consumption.csv",
                mime="text/csv",
            )


def _render_exposure_section(
    summary: dict,
    table: pd.DataFrame | None,
    station_table: pd.DataFrame,
    station_coeff: float,
) -> None:
    """People-weighted warming exposure comparison (PCS v2 exposure lens).

    All numbers are read from the pre-computed ``coupling_exposure.parquet`` /
    summary — no runtime recomputation, so the page stays snappy.
    """
    coverage = summary["coverage"]
    shift = summary["station_vs_people"]
    people_coeff = summary["people_weighted_inequality"]["inequality_coefficient"]

    st.divider()
    st.header("Warming the average resident actually feels")
    st.markdown(
        "Station means weight every monitoring location equally; **people-weighted** "
        "means weight each location by its population (GPW v4, 15-arc-minute count) — "
        "the warming experienced by the average *resident*, across "
        f"{coverage['n_countries']} countries. Counts are used directly (no "
        "cos-latitude weighting, which would understate fast-warming high-latitude "
        "populations)."
    )

    basis = st.radio(
        "Inequality basis",
        ["Station-based", "People-weighted (residents)"],
        horizontal=True,
        help="Switch the Lorenz curve and inequality coefficient between weighting "
        "every station equally and weighting by population.",
    )
    people_weighted = basis.startswith("People")
    if people_weighted and table is not None:
        coeff, other = people_coeff, station_coeff
        lorenz = charts.lorenz_chart(
            table,
            impact_col="impact_index_population_weighted",
            title="Cumulative people-weighted exposure vs cumulative responsibility",
        )
    else:
        coeff, other = station_coeff, people_coeff
        lorenz = charts.lorenz_chart(station_table)

    m1, m2 = st.columns(2)
    m1.metric(
        "Inequality coefficient",
        f"{coeff:.2f}",
        delta=f"{coeff - other:+.2f} vs other basis",
        delta_color="off",
        help="Gini-style divergence of cumulative warming exposure from cumulative "
        "responsibility, for the selected basis.",
    )
    m2.metric(
        "Station → people rank ρ",
        f"{shift['spearman_rho']:+.2f}",
        help="How much the exposure ranking changes when weighting by residents "
        "(1 = unchanged).",
    )

    st.plotly_chart(lorenz, width="stretch", key="exposure_lorenz")
    if table is not None:
        st.plotly_chart(
            charts.exposure_shift_scatter(table), width="stretch", key="exposure_shift"
        )

    c1, c2 = st.columns(2)
    c1.subheader("Residents more exposed than stations suggest")
    c1.caption("Population sits in the country's faster-warming regions.")
    c1.dataframe(
        _leaders_frame(shift["top_suffer_least_cause"]),
        width="stretch",
        hide_index=True,
    )
    c2.subheader("Stations overstate residents' exposure")
    c2.caption("Monitoring over-samples fast-warming, sparsely-populated areas.")
    c2.dataframe(
        _leaders_frame(shift["top_cause_least_suffer"]),
        width="stretch",
        hide_index=True,
    )

    if table is not None:
        with st.expander("Exposure lens — country table"):
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button(
                "Download CSV",
                table.to_csv(index=False).encode("utf-8"),
                file_name="coupling_exposure.csv",
                mime="text/csv",
            )
