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
