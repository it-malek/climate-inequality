"""Stability page: how much do the decomposition's shares move under perturbation?

Renders the three blocks of ``stability_summary.json``: bootstrap 95% intervals
of the Shapley shares (with the continent block bootstrap alongside),
leave-one-country-out influence on each share, and Moran's I on the model
residual. The page computes nothing; every number is precomputed by
``src.stability``, and all of it describes the stability of a descriptive
variance decomposition, not significance or causation.
"""

from __future__ import annotations

import streamlit as st

from app import charts, loaders, theme

_NOT_BUILT = (
    "The stability diagnostics have not been built yet. They need the city "
    "features and income table; run `python -m src.explain` and then rebuild the "
    "bundle (`python -m src.app_assets`)."
)


def _pct(value: object) -> str:
    """Format a probability in [0, 1] as a percentage, or a dash if absent."""
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "—"


def _num(value: object, places: int = 3) -> str:
    """Format a float to `places` decimals, or a dash if absent."""
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "—"


def render() -> None:
    """Render the stability page."""
    st.title("How confident are we?")
    st.markdown(
        "How much the variance shares move under resampling, single-country "
        "deletion and spatial dependence. These diagnostics quantify *stability*, "
        "not significance."
    )

    summary = loaders.load_stability_summary()
    if summary is None:
        st.info(_NOT_BUILT)
        return

    share = summary.get("share_stability")
    if share and share.get("groups"):
        st.subheader("Stability of the variance shares")
        block = (share.get("block_bootstrap") or {}).get("groups")
        st.plotly_chart(
            charts.share_ci_chart(share["groups"], block_groups=block),
            width="stretch",
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("P(geography is the largest axis)", _pct(share.get("p_geography_largest")))
        c2.metric("P(emissions share > 0)", _pct(share.get("p_emissions_positive")))
        c3.metric("Bootstrap resamples", f"{share.get('n_boot', 0):,}")
        st.caption(
            "Country bootstrap: resample the countries with replacement and "
            "recompute the decomposition. Bars are 95% percentile intervals of "
            "each share; the faint grey band is the continent block bootstrap, "
            "which respects spatial correlation within continents, so the gap "
            "between the two is the spatial-dependence correction."
        )

    influence = summary.get("influence") or {}
    by_group = influence.get("by_group")
    if by_group:
        st.subheader("Most influential countries")
        st.caption(
            "Leave-one-country-out: how far each share moves when a single "
            "country is dropped. A share that rides on a few countries is fragile."
        )
        ordered = [k for k in theme.GROUP_ORDER if k in by_group]
        for i in range(0, len(ordered), 2):
            columns = st.columns(2)
            for col, key in zip(columns, ordered[i : i + 2]):
                col.plotly_chart(
                    charts.influence_bar(
                        [tuple(x) for x in by_group[key]],
                        title=f"{theme.GROUP_LABELS[key]} share",
                        color=theme.group_color(key),
                    ),
                    width="stretch",
                )

    spatial = summary.get("residual_spatial")
    if spatial and spatial.get("morans_i") is not None:
        st.subheader("Spatial structure of the residual")
        m1, m2, m3 = st.columns(3)
        m1.metric("Moran's I (residual)", _num(spatial.get("morans_i")))
        m2.metric("Permutation p-value", _num(spatial.get("p_value")))
        m3.metric("Neighbors (k)", str(spatial.get("k_neighbors", "—")))
        st.caption(
            "Moran's I on the full-model country residuals with k-nearest-"
            "neighbour weights on country centroids. A positive, significant I "
            "means the unexplained warming is regionally clustered rather than "
            "white noise: the residual share is structured, not measurement noise."
        )
