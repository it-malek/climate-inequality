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

    # Alternative-weighting lenses (PCS v2): people-weighted and area-weighted
    # exposure share one Station / People / Area basis toggle. Degrades silently
    # to whichever lenses are present (or none).
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


def _render_weighting_section(
    *,
    station_table: pd.DataFrame,
    station_coeff: float,
    exposure_summary: dict | None,
    exposure_table: pd.DataFrame | None,
    area_summary: dict | None,
    area_table: pd.DataFrame | None,
) -> None:
    """Alternative-weighting exposure comparison: Station / People / Area (PCS v2).

    The station-weighted country mean over-samples dense mid-latitude monitoring
    clusters. Two pre-computed lenses re-weight it: **people-weighted** (by where
    residents live) and **area-weighted** (by land area, a per-cell gridded trend
    with cos-latitude weighting). All numbers are read from the pre-computed
    ``coupling_*`` artifacts — no runtime recomputation — and the section degrades
    to whichever lenses are present.
    """
    # Per-lens display config; only lenses with both summary and table are offered.
    lenses: dict[str, dict] = {}
    if exposure_summary is not None and exposure_table is not None:
        lenses["People-weighted (residents)"] = {
            "table": exposure_table,
            "impact_col": "impact_index_population_weighted",
            "z_col": "station_to_people_z_gap",
            "basis_label": "people-weighted",
            "shift": exposure_summary["station_vs_people"],
            "coeff": exposure_summary["people_weighted_inequality"][
                "inequality_coefficient"
            ],
            "n_countries": exposure_summary["coverage"]["n_countries"],
            "rank_label": "Station → people rank ρ",
            "lorenz_title": (
                "Cumulative people-weighted exposure vs cumulative responsibility"
            ),
            "more_header": "Residents more exposed than stations suggest",
            "more_caption": "Population sits in the country's faster-warming regions.",
            "less_header": "Stations overstate residents' exposure",
            "less_caption": "Monitoring over-samples fast-warming, sparse areas.",
            "csv": "coupling_exposure.csv",
        }
    if area_summary is not None and area_table is not None:
        lenses["Area-weighted (land)"] = {
            "table": area_table,
            "impact_col": "impact_index_area_weighted",
            "z_col": "station_to_area_z_gap",
            "basis_label": "area-weighted",
            "shift": area_summary["station_vs_area"],
            "coeff": area_summary["area_weighted_inequality"][
                "inequality_coefficient"
            ],
            "n_countries": area_summary["coverage"]["n_countries"],
            "rank_label": "Station → area rank ρ",
            "lorenz_title": (
                "Cumulative area-weighted exposure vs cumulative responsibility"
            ),
            "more_header": "More exposed once every km² counts equally",
            "more_caption": "Fast-warming interiors are under-sampled by stations.",
            "less_header": "Stations overstate area-average exposure",
            "less_caption": "Stations cluster in the faster-warming regions.",
            "csv": "coupling_area.csv",
        }
    if not lenses:
        return

    st.divider()
    st.header("Warming the average resident — and the average km² — actually feels")
    st.markdown(
        "Station means weight every monitoring location equally, so dense "
        "mid-latitude clusters dominate. Two re-weightings correct that: "
        "**people-weighted** (GPW v4 population count — counts used directly, no "
        "cos-latitude) and **area-weighted** (a per-grid-cell Theil–Sen trend off "
        "the Berkeley 1° field, **cos-latitude** weighted because a trend is an "
        "intensive field — the exact mirror of the population-count rule)."
    )

    basis = st.radio(
        "Inequality basis",
        ["Station-based", *lenses],
        horizontal=True,
        help="Switch the Lorenz curve and inequality coefficient between weighting "
        "every station equally, by population, and by land area.",
    )

    if basis == "Station-based":
        m1, *_ = st.columns(2)
        m1.metric(
            "Inequality coefficient",
            f"{station_coeff:.2f}",
            help="Gini-style divergence of cumulative warming exposure from "
            "cumulative responsibility, station-weighted.",
        )
        st.plotly_chart(
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
        delta=f"{cfg['coeff'] - station_coeff:+.2f} vs station-based",
        delta_color="off",
        help="Gini-style divergence of cumulative warming exposure from cumulative "
        "responsibility, for the selected basis.",
    )
    m2.metric(
        cfg["rank_label"],
        f"{shift['spearman_rho']:+.2f}",
        help=f"How much the exposure ranking changes under the {cfg['basis_label']} "
        "basis (1 = unchanged), across "
        f"{cfg['n_countries']} countries.",
    )

    st.plotly_chart(
        charts.lorenz_chart(
            table, impact_col=cfg["impact_col"], title=cfg["lorenz_title"]
        ),
        width="stretch",
        key="weighting_lorenz",
    )
    st.plotly_chart(
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

    with st.expander(f"{cfg['basis_label'].capitalize()} lens — country table"):
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name=cfg["csv"],
            mime="text/csv",
        )
