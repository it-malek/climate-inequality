"""Emissions regression page: station-weighted country warming against cumulative per-capita CO₂."""

from __future__ import annotations

import streamlit as st

from app import loaders, theme
from src.figures import render_inequality_scatter


def render() -> None:
    """Render the emissions-regression page."""
    table = loaders.load_inequality()
    stats = loaders.load_stats()
    ineq = stats["inequality"]
    fe = ineq["ols_fe"]
    pooled = ineq["ols_pooled"]

    st.title("Do historically high-emitting countries warm faster?")
    st.markdown(
        "The single-coefficient view that preceded the decomposition: station-weighted "
        f"country warming against cumulative per-capita CO₂ through {ineq['cutoff_year']}, "
        f"for {ineq['n_countries']} countries across {ineq['n_continents']} continents. "
        "Its coefficient depends on which correlated controls enter, which is why the "
        "main analysis reports variance shares instead."
    )

    left, middle, right = st.columns(3)
    left.metric(
        "Spearman ρ",
        f"{ineq['spearman_rho']:+.2f}",
        help=f"Rank correlation; p = {ineq['spearman_p']:.2g}.",
    )
    middle.metric(
        "Within-continent effect",
        f"{fe['coef']:+.3f}",
        help=(
            "°C/decade per tenfold cumulative per-capita CO₂, OLS with continent "
            f"fixed effects and HC1 robust standard errors. 95% interval [{fe['ci_low']:+.3f}, "
            f"{fe['ci_high']:+.3f}], p = {fe['p_value']:.2g}, R² = {fe['r2']:.2f}."
        ),
    )
    right.metric(
        "Pooled effect",
        f"{pooled['coef']:+.3f}",
        help=(
            "°C/decade per tenfold cumulative per-capita CO₂, pooled OLS with HC1 "
            f"robust standard errors. 95% interval [{pooled['ci_low']:+.3f}, "
            f"{pooled['ci_high']:+.3f}], p = {pooled['p_value']:.2g}."
        ),
    )

    fig = render_inequality_scatter(table, title="")
    theme.plotly_chart(fig, width="stretch")
    st.caption(
        f"Cumulative production-based CO₂ summed through {ineq['cutoff_year']} "
        f"and divided by {ineq['cutoff_year']} population (Our World in Data). Country "
        "warming is the unweighted mean of its city-location trends (station-weighted, "
        "not population- or area-weighted). Marker size: number of city locations behind "
        "each mean; grey line: pooled OLS fit."
    )

    with st.expander("Country table"):
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name="country_inequality.csv",
            mime="text/csv",
        )
