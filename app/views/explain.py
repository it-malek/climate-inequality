"""Drivers page: what explains where warming is fast?"""

from __future__ import annotations

import streamlit as st

from app import loaders
from src.figures import render_coefficient_stability, render_partial_effect_scatter

_NOT_BUILT = (
    "The explanatory-variables bundle has not been built yet. "
    "Run `python -m src.explain` then `python -m src.app_assets` to populate this page."
)


def render() -> None:
    """Render the explanatory variables (drivers) page."""
    stats = loaders.load_stats()
    if "explain" not in stats:
        st.title("What explains where warming is fast?")
        st.info(_NOT_BUILT)
        return

    e = stats["explain"]
    city = e["city_model"]
    country = e["country_model"]
    country_specs = country["specs"]

    st.title("What explains where warming is fast?")

    # ------------------------------------------------------------------
    # Section 1: Country-level coefficient-stability (the centerpiece)
    # ------------------------------------------------------------------
    st.subheader("Does the emissions–warming link survive a latitude control?")
    st.markdown(
        "The stored finding is **+0.029 °C/decade per 10× cumulative per-capita CO₂** "
        "within continents. Phase 7 tests whether that survives controlling for the "
        "fact that historically high-emitting countries also sit at higher latitudes — "
        "where Arctic amplification pushes warming up independently of emissions."
    )

    pooled = next((s for s in country_specs if s["spec_name"] == "pooled"), None)
    fe = next((s for s in country_specs if s["spec_name"] == "continent_fe"), None)
    lat_cont = next((s for s in country_specs if s["spec_name"] == "lat_continent"), None)

    col1, col2, col3 = st.columns(3)
    if pooled and pooled["emissions"]:
        em = pooled["emissions"]
        col1.metric(
            "Pooled effect",
            f"{em['coef']:+.3f}",
            help=(
                f"°C/decade per 10× CO₂, pooled OLS with HC1 SEs. "
                f"95% CI [{em['ci_low']:+.3f}, {em['ci_high']:+.3f}], "
                f"p = {em['p_value']:.2g}."
            ),
        )
    if fe and fe["emissions"]:
        em = fe["emissions"]
        col2.metric(
            "Within-continent (FE)",
            f"{em['coef']:+.3f}",
            help=(
                f"°C/decade per 10× CO₂, continent fixed effects + HC1 SEs. "
                f"95% CI [{em['ci_low']:+.3f}, {em['ci_high']:+.3f}], "
                f"p = {em['p_value']:.2g}. This is the README headline result."
            ),
        )
    if lat_cont and lat_cont["emissions"]:
        em = lat_cont["emissions"]
        col3.metric(
            "+ mean |latitude| control",
            f"{em['coef']:+.3f}",
            help=(
                f"°C/decade per 10× CO₂, with mean-|latitude| added within "
                f"continents + HC1 SEs. 95% CI [{em['ci_low']:+.3f}, "
                f"{em['ci_high']:+.3f}], p = {em['p_value']:.2g}. "
                "The coefficient halves and loses statistical significance — "
                "the within-continent link is explained by latitude geography, "
                "not by emissions responsibility per se."
            ),
        )

    fig_stability = render_coefficient_stability(country_specs, title="")
    st.plotly_chart(fig_stability, use_container_width=True)
    st.caption(
        "Each row is one model specification. The x-axis is the log₁₀(emissions) "
        "coefficient (°C/decade per 10× cumulative per-capita CO₂) with its 95% CI. "
        "Colored dots (CI excludes zero) vs grey dots (CI includes zero). "
        "The dotted line marks zero. Adding mean-|latitude| within continents "
        "(`lat_continent`) halves the coefficient and pushes the CI to include zero — "
        "the within-continent correlation is driven by latitude, not emissions directly."
    )

    # ------------------------------------------------------------------
    # Section 2: City-level geography
    # ------------------------------------------------------------------
    st.subheader("The spatial pattern is mostly latitude, with an arid hotspot")

    city_specs = city["specs"]
    baseline = next((s for s in city_specs if s["spec_name"] == "baseline"), None)
    full = next((s for s in city_specs if s["spec_name"] == "full"), None)
    interaction = next((s for s in city_specs if s["spec_name"] == "interaction"), None)

    c1, c2, c3 = st.columns(3)
    if baseline:
        c1.metric(
            "Baseline R² (|latitude| only)",
            f"{baseline['r2']:.3f}",
            help="OLS of warming trend on |latitude| alone, with country-clustered SEs.",
        )
    if full:
        c2.metric(
            "Full model R²",
            f"{full['r2']:.3f}",
            help=(
                "OLS adding elevation, coast distance, Köppen class, station density. "
                f"Moran's I on residuals = {city.get('moran_i_full', 'n/a'):.3f} "
                f"(p = {city.get('moran_p_full', float('nan')):.3g}) — "
                "significant spatial autocorrelation remains."
                if city.get("moran_i_full") is not None else
                "OLS adding elevation, coast distance, Köppen class, station density."
            ),
        )
    if interaction:
        c3.metric(
            "Interaction R²",
            f"{interaction['r2']:.3f}",
            help="Full model + abs_latitude × Köppen class interactions.",
        )

    features = loaders.load_explain_features()
    fig_scatter = render_partial_effect_scatter(features, title="")
    st.plotly_chart(fig_scatter, use_container_width=True)

    abs_lat_info = city.get("abs_latitude_baseline")
    koppen_b_info = city.get("koppen_b_full")
    caption_parts = [
        "Warming trend vs absolute latitude, colored by major Köppen climate class. "
        "Gray line: overall OLS trendline."
    ]
    if abs_lat_info:
        caption_parts.append(
            f"|Latitude| coefficient (baseline): {abs_lat_info['coef']:+.5f} "
            f"[{abs_lat_info['ci_low']:+.5f}, {abs_lat_info['ci_high']:+.5f}] "
            "°C/decade per degree."
        )
    if koppen_b_info:
        verdict = "supports" if koppen_b_info["coef"] > 0 else "does not support"
        caption_parts.append(
            f"Köppen-B (arid) vs reference class (full model): "
            f"{koppen_b_info['coef']:+.5f} "
            f"[{koppen_b_info['ci_low']:+.5f}, {koppen_b_info['ci_high']:+.5f}] "
            f"p = {koppen_b_info['p_value']:.3g} — {verdict} the arid-hotspot hypothesis."
        )
    if city.get("moran_i_full") is not None:
        caption_parts.append(
            f"Moran's I on full-model residuals = {city['moran_i_full']:.4f} "
            f"(p = {city['moran_p_full']:.3g}) — spatial autocorrelation persists "
            "even after controlling for geography."
        )
    st.caption(" ".join(caption_parts))

    with st.expander("Full city-model coefficients"):
        for spec in city_specs:
            st.markdown(f"**{spec['spec_name']}** (n={spec['n']}, R²={spec['r2']:.3f})")
            if spec["terms"]:
                import pandas as pd

                term_df = pd.DataFrame(
                    [
                        {
                            "term": t["term"],
                            "coef": t["coef"],
                            "ci_low": t["ci_low"],
                            "ci_high": t["ci_high"],
                            "p_value": t["p_value"],
                        }
                        for t in spec["terms"]
                    ]
                )
                st.dataframe(term_df, hide_index=True, use_container_width=True)
            if spec.get("partial_r2"):
                st.caption(
                    "Partial R²: "
                    + ", ".join(
                        f"{k}={v:.4f}" for k, v in spec["partial_r2"].items()
                    )
                )
