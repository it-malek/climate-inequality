"""Sensitivity / robustness panel: how confident are we in the structure?

Renders the decomposition's own confidence story when a ``stability_summary.json``
is present in the bundle: the **bootstrap 95% CIs of the Shapley shares** (with the
spatially-honest continent block bootstrap alongside), the **leave-one-country-out
influence** on each share, and **Moran's I** on the model residual. Until that
summary is generated (the stability layer is a deferred pipeline step), it shows an
explicit pending state. The page adds no statistics; it only visualizes a
precomputed summary, and everything here is the *stability of a descriptive
variance decomposition* — never a significance or causal claim.

Legacy coefficient diagnostics (GAM df sensitivity, Conley-HAC vs HC1, DFBETA) are
still rendered if their (optional) blocks are present, but they target the legacy
emissions *coefficient*, not the shares, and are not produced by this phase.
"""

from __future__ import annotations

import streamlit as st

from app import charts, loaders, theme

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


def _pct(value: object) -> str:
    """Format a probability in [0, 1] as a percentage, or em-dash if absent."""
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "—"


def _num(value: object, places: int = 3) -> str:
    """Format a float to `places` decimals, or em-dash if absent."""
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "—"


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

    # --- Decomposition stability: bootstrap CIs of the shares (centerpiece) ---
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
            "Country bootstrap — resample the countries with replacement and "
            "recompute the decomposition. Bars are 95% percentile intervals of "
            "each share; the faint grey band is the continent block bootstrap, "
            "which respects spatial correlation, so the gap is the spatial-"
            "dependence correction. This is the *stability of the shares* — a "
            "descriptive property of the variance decomposition, not a "
            "significance or causal claim."
        )

    # --- Leave-one-country-out influence on each share -------------------
    influence = summary.get("influence") or {}
    by_group = influence.get("by_group")
    if by_group:
        st.subheader("Most influential countries")
        st.caption(
            "Leave-one-country-out: how far each share moves when a single "
            "country is dropped. The ten countries whose removal moves each share "
            "most — a share that rides on a few countries is fragile."
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

    # --- Spatial structure of the residual (Moran's I) ------------------
    spatial = summary.get("residual_spatial")
    if spatial and spatial.get("morans_i") is not None:
        st.subheader("Spatial structure of the residual")
        m1, m2, m3 = st.columns(3)
        m1.metric("Moran's I (residual)", _num(spatial.get("morans_i")))
        m2.metric("Permutation p-value", _num(spatial.get("p_value")))
        m3.metric("Neighbors (k)", str(spatial.get("k_neighbors", "—")))
        st.caption(
            "Moran's I on the full-model country residuals (centroid k-nearest-"
            "neighbor weights). A positive, significant I means the unexplained "
            "warming is regionally clustered rather than white noise — so the "
            "large residual share is itself structured, not measurement noise. "
            "Descriptive only."
        )

    # --- 1. GAM df sensitivity (legacy coefficient diagnostic) ----------
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
