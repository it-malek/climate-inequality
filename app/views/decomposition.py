"""Decomposition dashboard: warming inequality and its variance structure.

Implements the main dashboard (inequality metrics + Shapley-share bar) and the
interactive decomposition explorer (emissions-only / geography-only / full).
Reads the committed ``inequality_summary.json`` and ``decomposition_summary.json``
bundle artifacts; renders a pending state if either is absent.
"""

from __future__ import annotations

import streamlit as st

from app import charts, loaders, theme

_NOT_BUILT = (
    "The decomposition bundle has not been built yet. Run `python -m "
    "src.inequality` and `python -m src.decomposition`, then copy their "
    "`*_summary.json` into `app/data/`."
)

_VIEW_OPTIONS = {
    "Emissions only": "emissions",
    "Geography only": "geography",
    "Full model": "full",
}

_VIEW_CAPTION = {
    "emissions": (
        "Emissions responsibility *alone* — its standalone R². Compare with the "
        "full model: most of this overlaps geography once both are present."
    ),
    "geography": (
        "Physical geography *alone* — its standalone R². The single largest "
        "structuring axis of cross-country warming."
    ),
    "full": (
        "All axes together. Each segment is a Shapley share — variance "
        "attributed to that axis, with overlaps split fairly across orderings."
    ),
}


def render() -> None:
    """Render the warming-inequality decomposition dashboard."""
    st.title("Global Warming Inequality Decomposition")

    ineq = loaders.load_inequality_summary()
    decomp = loaders.load_decomposition_summary()
    if ineq is None or decomp is None:
        st.info(_NOT_BUILT)
        return

    st.markdown(
        "**What you're looking at.** Every country's average warming rate over "
        "1950–2013, measured from 3,510 Berkeley Earth city weather stations. "
        "Warming is universal — every country warmed — but uneven: the "
        "fastest-warming countries warmed about **three times faster** than the "
        "slowest. This page answers two questions — *how unequal* is that "
        "warming (the metrics), and *how is the inequality structured* across "
        "emissions, physical geography, socioeconomic development and population "
        "(the bar), with an explicit unexplained residual."
    )

    # --- Headline inequality + fit metrics --------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Gini of warming",
        f"{ineq['gini']:.3f}",
        help="Gini coefficient of country mean warming trends. 0 = every "
        "country warms identically; higher = more unequal.",
    )
    m2.metric(
        "Theil-T",
        f"{ineq['theil_t']:.3f}",
        help=(
            f"Theil-T index; {ineq['theil_between_share']:.0%} of it is "
            "*between* continents."
            if ineq.get("theil_between_share") is not None
            else "Theil-T index of country warming trends."
        ),
    )
    m3.metric(
        "Variance explained",
        f"{decomp['total_r2']:.0%}",
        help="Total R² of the full cross-sectional model over "
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
    st.caption(_headline_sentence(decomp))

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

    with st.expander("What each axis contains"):
        for key in theme.GROUP_ORDER:
            if key == "residual" or key not in decomp["shares"]:
                continue
            st.markdown(
                f"- **{theme.GROUP_LABELS[key]}** — {theme.GROUP_GLOSS[key]} "
                f"· features used: `{', '.join(decomp['group_features'].get(key, []))}`"
            )

    with st.expander("✅ What this can say · ❌ what it cannot"):
        st.markdown(
            "**It can say** how unequally observed warming is distributed across "
            "countries, and how much that inequality *aligns* with each kind of "
            "structure — here, that the warming ranking tracks physical "
            "geography far more than it tracks emissions, and that the two "
            "overlap heavily (so a raw emissions–warming correlation mostly "
            "reflects *where* high emitters sit).\n\n"
            "**It cannot say** that emissions *caused* a country's warming: CO₂ "
            "is well-mixed, so a country's own emissions do not preferentially "
            "heat its territory. Nor is this a map of climate *impact* — the "
            "outcome is average warming in °C/decade, which understates the "
            "burden on low-emitting tropical countries that are most exposed to "
            "heat and least able to adapt. The shares are a **descriptive "
            "variance attribution, not causal climate attribution.**"
        )


def _headline_sentence(decomp: dict) -> str:
    """One-line plain-language summary of the dominant structure."""
    named = {k: v for k, v in decomp["shares"].items()}
    if not named:
        return ""
    top = max(named, key=named.get)
    total = decomp["total_r2"]
    of_explained = named[top] / total if total else float("nan")
    return (
        f"**{theme.GROUP_LABELS[top]}** is the largest structuring axis — "
        f"{named[top]:.0%} of total variance ({of_explained:.0%} of the "
        f"explained part). Residual (unexplained) is {decomp['residual_share']:.0%}. "
        "Shares are a descriptive variance attribution, not causal effects."
    )
