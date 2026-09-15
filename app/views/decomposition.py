"""Decomposition dashboard: warming inequality and its variance structure.

Implements the main dashboard (inequality metrics + Shapley-share bar), the
interactive decomposition explorer (emissions-only / geography-only / full) and
the outcome-sensitivity table (area-weighted primary versus the station-weighted
construction). Reads the committed ``inequality_summary.json`` and
``decomposition_summary.json`` bundle artifacts; renders a pending state if
either is absent.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, loaders, theme

_NOT_BUILT = (
    "The decomposition summaries have not been built yet. They need the city "
    "features and income table; run `python -m src.explain` and then rebuild the "
    "bundle (`python -m src.app_assets`)."
)

_VIEW_OPTIONS = {
    "Emissions only": "emissions",
    "Geography only": "geography",
    "Full model": "full",
}

_VIEW_CAPTION = {
    "emissions": (
        "Emissions responsibility *alone* — its standalone R², before geography "
        "and the other axes are credited. How much it explains on its own depends "
        "on how national warming is constructed; see the outcome sensitivity below."
    ),
    "geography": (
        "Physical geography *alone* — its standalone R². The single largest "
        "structuring axis of cross-country warming under either outcome."
    ),
    "full": (
        "All axes together. Each segment is a Shapley share — variance "
        "attributed to that axis, with overlaps split fairly across orderings."
    ),
}

_OUTCOME_GLOSS = {
    "area_weighted": (
        "a Theil–Sen trend per 1° Berkeley Earth grid cell, averaged over each "
        "country's land area with cos-latitude weights, so that national "
        "aggregation is set by land area rather than by where the stations are"
    ),
    "station_weighted": (
        "the unweighted mean of the country's city-location trends from the "
        "Berkeley Earth station records"
    ),
}

_STATION_KEY = "station_weighted"
_STATION_ALL_KEY = "station_weighted_all_countries"


def render() -> None:
    """Render the warming-inequality decomposition dashboard."""
    st.title("Warming inequality decomposition")

    ineq = loaders.load_inequality_summary()
    decomp = loaders.load_decomposition_summary()
    if ineq is None or decomp is None:
        st.info(_NOT_BUILT)
        return

    outcome_key = str(decomp.get("outcome_definition", "station_weighted"))
    outcome = theme.outcome_label(outcome_key)
    gloss = _OUTCOME_GLOSS.get(outcome_key, "")
    st.markdown(
        "**What you're looking at.** Every country's land-warming rate over "
        f"1950–2013, **{outcome}**"
        + (f": {gloss}." if gloss else ".")
        + " Warming is universal — every country warmed — but uneven. This page "
        "answers two questions — *how unequal* is that warming (the metrics), "
        "and *how is the inequality structured* across historical emissions "
        "responsibility, physical geography, socioeconomic development and "
        "population (the bar), with an explicit unexplained residual."
    )

    # --- Headline inequality + fit metrics --------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Gini of warming",
        f"{ineq['gini']:.3f}",
        help="Gini coefficient of country mean warming trends (station-weighted "
        "country means, every matched country). 0 = every country warms "
        "identically; higher = more unequal.",
    )
    m2.metric(
        "Theil-T",
        f"{ineq['theil_t']:.3f}",
        help=(
            f"Theil-T index of the station-weighted country means; "
            f"{ineq['theil_between_share']:.0%} of it is *between* continents."
            if ineq.get("theil_between_share") is not None
            else "Theil-T index of country warming trends."
        ),
    )
    m3.metric(
        "Variance explained",
        f"{decomp['total_r2']:.0%}",
        help=f"Total R² of the full cross-sectional model of {outcome} warming over "
        f"{decomp['n']} countries.",
    )
    m4.metric(
        "Residual (unexplained)",
        f"{decomp['residual_share']:.0%}",
        help="Share of total cross-country warming variance no named axis "
        "explains.",
    )

    # --- Shapley share bar ------------------------------------------------
    st.subheader("How the inequality decomposes")
    st.plotly_chart(
        charts.shares_bar(decomp["shares"], decomp["residual_share"]),
        width="stretch",
    )
    st.caption(_headline_sentence(decomp, outcome))

    st.divider()

    # --- Interactive explorer --------------------------------------------
    st.subheader("Decomposition explorer")
    st.markdown(
        "Toggle between each axis on its own and the full model to see its "
        "contribution to variance explained."
    )
    choice = st.radio(
        "View",
        list(_VIEW_OPTIONS),
        horizontal=True,
        label_visibility="collapsed",
    )
    view = _VIEW_OPTIONS[choice]

    if view == "full":
        explained = decomp["total_r2"]
    else:
        explained = decomp["univariate_r2"][view]
    left, right = st.columns([1, 3])
    left.metric("Variance explained", f"{explained:.0%}")
    if view != "full":
        share = decomp["shares"].get(view)
        if share is not None:
            left.caption(
                f"Shapley share in the full model: **{share:.0%}** — the rest of "
                "its standalone R² is shared with other axes."
            )
    right.plotly_chart(
        charts.explorer_bar(decomp, view), width="stretch"
    )
    st.caption(_VIEW_CAPTION[view])

    # --- Outcome sensitivity ---------------------------------------------
    station = (decomp.get("sensitivity") or {}).get(_STATION_KEY)
    if station:
        st.divider()
        st.subheader("Outcome sensitivity: station-weighted")
        st.markdown(
            "The same decomposition with each country's warming taken as the "
            "unweighted mean of its station trends instead of the area-weighted "
            "rate. The *same countries* column changes nothing but the outcome; "
            "the *all countries* column adds the countries that have no "
            "area-weighted value."
        )
        station_all = (decomp.get("sensitivity") or {}).get(_STATION_ALL_KEY)
        st.dataframe(
            _sensitivity_table(decomp, station, station_all), width="stretch"
        )
        st.caption(_sensitivity_caption(decomp, station))

    with st.expander("What each axis contains"):
        for key in theme.GROUP_ORDER:
            if key == "residual" or key not in decomp["shares"]:
                continue
            st.markdown(
                f"- **{theme.GROUP_LABELS[key]}** — {theme.GROUP_GLOSS[key]} "
                f"· features used: `{', '.join(decomp['group_features'].get(key, []))}`"
            )

    with st.expander("What this can and cannot say"):
        st.markdown(
            "**It can say** how unequally observed warming is distributed across "
            "countries, and how much that inequality *aligns* with each kind of "
            "structure — here, that cross-country differences in land-warming "
            "rates are associated primarily with physical geography, while a "
            "country's historical emissions responsibility, income group and "
            "population add little independent explanatory variance. The "
            "station-weighted sensitivity shows how much the apparent emissions "
            "contribution depends on how national warming is constructed.\n\n"
            "**It cannot say** that emissions *caused* a country's warming, or "
            "that they did not: greenhouse gases drive global warming, but CO₂ "
            "is well-mixed, so a country's own emissions do not preferentially "
            "heat its territory, and this page asks a different question — why "
            "warming *rates differ among countries*. Nor is this a map of "
            "climate *impact* — the outcome is average warming in °C/decade, "
            "which understates the burden on low-emitting tropical countries "
            "that are most exposed to heat and least able to adapt. The shares "
            "are a **descriptive variance attribution, not causal climate "
            "attribution.**"
        )


def _headline_sentence(decomp: dict, outcome: str) -> str:
    """One-line plain-language summary of the dominant structure."""
    named = {k: v for k, v in decomp["shares"].items()}
    if not named:
        return ""
    top = max(named, key=named.get)
    total = decomp["total_r2"]
    of_explained = named[top] / total if total else float("nan")
    return (
        f"**{theme.GROUP_LABELS[top]}** is the largest structuring axis of "
        f"{outcome} warming — {named[top]:.0%} of total variance "
        f"({of_explained:.0%} of the variance the model explains). Residual "
        f"(unexplained) is {decomp['residual_share']:.0%}. Shares are a "
        "descriptive variance attribution, not causal effects."
    )


def _sensitivity_table(
    primary: dict, station: dict, station_all: dict | None
) -> pd.DataFrame:
    """Side-by-side shares (as percentages of total variance) per outcome."""
    columns: list[tuple[str, dict]] = [
        (f"{theme.outcome_label(primary.get('outcome_definition')).capitalize()} (primary)", primary),
        ("Station-weighted, same countries", station),
    ]
    if station_all:
        columns.append(("Station-weighted, all countries", station_all))
    keys = [k for k in theme.GROUP_ORDER if k != "residual" and k in primary["shares"]]
    rows: dict[str, list[str]] = {}
    for key in keys:
        rows[f"{theme.GROUP_LABELS[key]} share"] = [
            f"{block['shares'].get(key, float('nan')):.1%}" for _, block in columns
        ]
    rows["Residual (unexplained)"] = [
        f"{block['residual_share']:.1%}" for _, block in columns
    ]
    rows["Total R²"] = [f"{block['total_r2']:.3f}" for _, block in columns]
    rows["Countries"] = [f"{block['n']}" for _, block in columns]
    return pd.DataFrame(rows, index=[name for name, _ in columns]).T


def _sensitivity_caption(primary: dict, station: dict) -> str:
    """Plain-language reading of the outcome comparison, using the numbers shown."""
    geo_p = primary["shares"].get("geography")
    geo_s = station["shares"].get("geography")
    em_p = primary["shares"].get("emissions")
    em_s = station["shares"].get("emissions")
    parts = []
    if geo_p is not None and geo_s is not None:
        parts.append(
            f"Geography is the largest axis under both constructions "
            f"({geo_p:.1%} vs {geo_s:.1%} of total variance)"
        )
    if em_p is not None and em_s is not None:
        parts.append(
            f"the historical-emissions share is {em_p:.1%} vs {em_s:.1%}"
        )
    lead = "; ".join(parts) + ". " if parts else ""
    return (
        lead
        + "The difference is consistent with station-network geography (the "
        "coupling page shows the same shift in the rank correlation), but the "
        "area-weighted outcome also comes from a gridded product, so this "
        "comparison cannot isolate station siting as the sole cause. Neither "
        "outcome is independent of the station record: Berkeley Earth's grid is "
        "built from it."
    )
