"""Stability page: how much the decomposition's shares move under perturbation.

Renders the blocks of ``stability_summary.json``: bootstrap 95% intervals of
the Shapley shares (with the continent block bootstrap alongside),
leave-one-country-out influence on each share, Moran's I on the model
residual, and the same intervals for the station-weighted outcome next to the
area-weighted primary. The page computes nothing; every number is precomputed
by ``src.stability``, and all of it describes the stability of a descriptive
variance decomposition, not significance or causation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, loaders, theme

_STATION_KEY = "station_weighted"

_NOT_BUILT = (
    "The stability diagnostics have not been built yet. They need the city "
    "features and income table; run `python -m src.explain` and then rebuild the "
    "bundle (`python -m src.app_assets`)."
)


def _pct(value: object) -> str:
    """Format a probability in [0, 1] as a percentage, or 'n/a' if absent."""
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: object, places: int = 3) -> str:
    """Format a float to ``places`` decimals, or 'n/a' if absent."""
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "n/a"


def render() -> None:
    """Render the stability page."""
    st.title("How stable are the shares?")

    summary = loaders.load_stability_summary()
    if summary is None:
        st.markdown(
            "How much the variance shares move under resampling, single-country "
            "deletion and spatial dependence. These diagnostics quantify "
            "*stability*, not significance."
        )
        st.info(_NOT_BUILT)
        return

    outcome = theme.outcome_label(summary.get("outcome_definition", "station_weighted"))
    st.markdown(
        f"How much the variance shares of **{outcome}** country warming move "
        "under resampling, single-country deletion and spatial dependence. "
        "These diagnostics quantify *stability*, not significance."
    )

    share = summary.get("share_stability")
    if share and share.get("groups"):
        st.subheader("Stability of the variance shares")
        block = (share.get("block_bootstrap") or {}).get("groups")
        theme.plotly_chart(
            charts.share_ci_chart(share["groups"], block_groups=block),
            width="stretch",
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("P(geography is the largest group)", _pct(share.get("p_geography_largest")))
        c2.metric("P(responsibility share > 0)", _pct(share.get("p_emissions_positive")))
        c3.metric("Bootstrap resamples", f"{share.get('n_boot', 0):,}")
        st.caption(
            "Country bootstrap: resample the countries with replacement and recompute the "
            "decomposition. Bars are 95% percentile intervals of each share; the faint band "
            "is the continent block bootstrap, which respects spatial correlation within "
            "continents, so the gap between the two is the spatial-dependence correction. "
            "These intervals were computed under the original representation of the model "
            "(responsibility measured by per-capita and total cumulative CO₂); the "
            "full-rank intervals on the decomposition page differ by under one percentage "
            "point."
        )

    influence = summary.get("influence") or {}
    by_group = influence.get("by_group")
    if by_group:
        st.subheader("Most influential countries")
        st.caption(
            "Leave-one-country-out: how far each share moves when a single country is "
            "dropped. A share that rides on a few countries is fragile; here no country "
            "moves any share by more than two percentage points."
        )
        ordered = [k for k in theme.GROUP_ORDER if k in by_group]
        for i in range(0, len(ordered), 2):
            columns = st.columns(2)
            for col, key in zip(columns, ordered[i : i + 2]):
                with col:
                    theme.plotly_chart(
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
        m3.metric("Neighbors (k)", str(spatial.get("k_neighbors", "n/a")))
        st.caption(
            "Moran's I on the full-model country residuals with k-nearest-neighbour "
            "weights on country centroids. A positive, significant value means the "
            "residual is regionally clustered rather than white noise: the unexplained "
            "share is structured, not measurement noise. The residual-structure page "
            "follows that thread."
        )

    station = (summary.get("sensitivity") or {}).get(_STATION_KEY)
    if station and share and share.get("groups"):
        st.subheader("Outcome sensitivity: station-weighted")
        st.markdown(
            "The same bootstrap on the same countries with each country's warming taken as "
            "the unweighted mean of its station trends. Point estimates with the 95% "
            "country-bootstrap interval; the last row is the residual Moran's I."
        )
        st.dataframe(_comparison_table(summary, station), width="stretch")
        st.caption(
            "Where the two intervals overlap, the share does not depend on how national "
            "warming was constructed; where they separate, it does. The gap is consistent "
            "with station-network geography, but the comparison also changes the "
            "observational product (a gridded field versus station means), so it does not "
            "isolate siting as the sole cause."
        )


def _interval(group: dict) -> str:
    """``point [low, high]`` as percentages of total variance."""
    return (
        f"{float(group['point']):.1%} "
        f"[{float(group['ci_low']):.1%}, {float(group['ci_high']):.1%}]"
    )


def _comparison_table(primary: dict, station: dict) -> pd.DataFrame:
    """Shares with bootstrap intervals for the two outcomes, plus Moran's I."""
    columns = [
        (f"{theme.outcome_label(primary.get('outcome_definition')).capitalize()} (primary)", primary),
        ("Station-weighted, same countries", station),
    ]
    groups = [
        (name, block["share_stability"]["groups"]) for name, block in columns
    ]
    keys = [k for k in theme.GROUP_ORDER if k in groups[0][1]]
    rows: dict[str, list[str]] = {}
    for key in keys:
        label = theme.GROUP_LABELS[key]
        rows[f"{label} share"] = [
            _interval(g[key]) if key in g else "n/a" for _, g in groups
        ]
    rows["Residual Moran's I"] = [
        _num((block.get("residual_spatial") or {}).get("morans_i"))
        for _, block in columns
    ]
    rows["Countries"] = [str(block.get("n_countries", "n/a")) for _, block in columns]
    return pd.DataFrame(rows, index=[name for name, _ in columns]).T
