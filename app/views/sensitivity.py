"""Sensitivity / robustness panel: how confident are we in the structure?

Renders three diagnostics when a ``stability_summary.json`` is present in the
bundle — GAM latitude-control df sensitivity, Conley-HAC vs HC1 uncertainty, and
the most influential countries by DFBETA. Until that summary is generated (the
stability layer is a deferred pipeline step), it shows an explicit pending state
describing what each diagnostic will report. This page adds no statistics; it
only visualizes a precomputed summary.
"""

from __future__ import annotations

import streamlit as st

from app import charts, loaders

_PENDING = (
    "These confidence diagnostics are **not built yet**. They are produced by "
    "the deferred *stability layer* (`src/stability.py`), which re-estimates the "
    "decomposition under alternative specifications and writes a "
    "`stability_summary.json` into the bundle. Until then, the panel below "
    "describes what it will show."
)

_PENDING_ITEMS = [
    (
        "Nonlinear latitude control (GAM df = 4, 6, 8)",
        "Re-estimates the emissions term with a B-spline latitude smoother of "
        "increasing flexibility. Stable estimates ⇒ the result is not an "
        "artifact of a too-rigid linear latitude control.",
    ),
    (
        "Spatial-dependence-robust uncertainty (Conley HAC vs HC1)",
        "Country residuals are spatially autocorrelated, so plain HC1 standard "
        "errors understate uncertainty. The Conley HAC CI widens them using "
        "distance-decay between country centroids.",
    ),
    (
        "Influence diagnostics (top DFBETA)",
        "Which individual countries most move the emissions estimate. A result "
        "riding on a few high-leverage emitters is fragile.",
    ),
]


def render() -> None:
    """Render the sensitivity / robustness panel."""
    st.title("How confident are we?")
    st.markdown(
        "Sensitivity of the structure to estimator, spatial dependence and "
        "influential countries. These diagnostics quantify *stability*, not "
        "significance — and remain a **descriptive** layer."
    )

    summary = loaders.load_stability_summary()
    if summary is None:
        st.info(_PENDING)
        for title, body in _PENDING_ITEMS:
            st.markdown(f"**{title}**  \n{body}")
        st.caption(
            "Note: these classic diagnostics target the legacy emissions "
            "*coefficient*. The decomposition's own confidence story is the "
            "*stability of the Shapley shares* (bootstrap / jackknife), which "
            "the same deferred layer will add."
        )
        return

    # --- 1. GAM df sensitivity -------------------------------------------
    df_rows = summary.get("df_sensitivity")
    if df_rows:
        st.subheader("Nonlinear latitude control")
        st.plotly_chart(
            charts.coef_ci_chart(
                df_rows, label_key="df", title="Emissions term vs GAM smoother df"
            ),
            width="stretch",
        )
        st.caption(
            "Each row re-estimates the emissions term with a latitude smoother "
            "of the given spline df. Overlapping CIs ⇒ insensitive to flexibility."
        )

    # --- 2. Conley HAC vs HC1 --------------------------------------------
    unc = summary.get("uncertainty")
    if unc:
        st.subheader("Spatial-dependence-robust uncertainty")
        st.plotly_chart(
            charts.coef_ci_chart(
                unc, label_key="method", title="HC1 vs Conley HAC 95% CI"
            ),
            width="stretch",
        )
        st.caption(
            "The Conley HAC interval accounts for spatial autocorrelation "
            "between country centroids; a much wider CI means HC1 was "
            "overconfident."
        )

    # --- 3. Influence / DFBETA -------------------------------------------
    influence = summary.get("influence") or {}
    top_dfbeta = influence.get("top_dfbeta")
    if top_dfbeta:
        st.subheader("Most influential countries")
        st.plotly_chart(
            charts.dfbeta_bar([tuple(x) for x in top_dfbeta]),
            width="stretch",
        )
        st.caption(
            "DFBETA on the emissions term"
            + (f" (spec: `{influence['spec']}`)." if influence.get("spec") else ".")
        )
