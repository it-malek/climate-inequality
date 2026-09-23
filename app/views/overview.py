"""Overview page: the research question, the scope of the study and its principal findings."""

from __future__ import annotations

import streamlit as st

from app import loaders


def _pct(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:.{digits}%}"


def render() -> None:
    """Render the overview page."""
    stats = loaders.load_stats()
    decomp = loaders.load_decomposition_summary()
    residual = loaders.load_residual_summary()
    coupling = loaders.load_coupling_summary()
    area = loaders.load_coupling_area_summary()

    st.title("Warming inequality across countries")
    st.markdown(
        "**The question.** How is cross-country variation in observed "
        "land-warming rates structured among physical geography, historical "
        "emissions responsibility, socioeconomic development and population "
        "characteristics, and what can be learned from the spatial structure "
        "that remains once those factors are accounted for?"
    )

    n_locations = stats["trends"]["n_locations"]
    n_countries = None
    r2 = geography = residual_share = None
    if residual is not None:
        static = residual["static"]
        n_countries = static["n"]
        r2 = static["in_sample_r2"]
        geography = static["shares"]["geography"]["point"]
        residual_share = static["shares"]["residual"]["point"]
    elif decomp is not None:
        n_countries = decomp["n"]
        r2 = decomp["total_r2"]
        geography = decomp["shares"].get("geography")
        residual_share = decomp["residual_share"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("City locations", f"{n_locations:,}", help="Berkeley Earth city records with a fitted 1950\u20132013 trend.")
    c2.metric("Countries decomposed", "n/a" if n_countries is None else f"{n_countries}",
              help="Complete cases in the cross-country variance decomposition.")
    c3.metric("Variance explained", _pct(r2),
              help="R squared of the static model of area-weighted national warming.")
    c4.metric("Geography share", _pct(geography),
              help="Shapley share of total cross-country variance allocated to physical geography.")

    st.subheader("Scope")
    st.markdown(
        f"- **Period.** Monthly land temperatures from January 1950 to September 2013; "
        "trends are Theil\u2013Sen slopes on anomalies against each location's 1951\u20131980 climatology.\n"
        f"- **Data.** Berkeley Earth city records ({n_locations:,} locations) and the Berkeley "
        "Earth 1° gridded land product; ERA5 reanalysis as an independent product check; "
        "Our World in Data emissions and population; World Bank income groups; ND-GAIN vulnerability.\n"
        "- **Unit.** The country. National warming is defined three ways (station-, "
        "population- and area-weighted); the area-weighted rate is the primary outcome of the "
        "decomposition, which uses the 151 countries with complete features.\n"
        "- **Design.** Cross-country and descriptive. Variance shares are allocations of "
        "explained cross-country variance; nothing here attributes warming to a cause or "
        "estimates the effect of a country's emissions on its own climate."
    )

    st.subheader("Principal findings")
    findings = []
    if coupling is not None and area is not None:
        findings.append(
            f"**Responsibility and experienced warming are different axes.** Ranked by "
            f"station-weighted warming, countries with more cumulative CO₂ per person warmed "
            f"faster (Spearman ρ {coupling['spearman_rho']:+.2f}); ranked by area-weighted "
            f"warming, the correlation is {area['area_weighted_inequality']['spearman_rho']:+.2f}. "
            "CO₂ is well mixed, so a country's emissions do not preferentially warm its own territory."
        )
    if geography is not None and r2 is not None:
        findings.append(
            f"**Geography structures most of what the static model explains.** Physical "
            f"geography accounts for {geography:.1%} of total cross-country variance, "
            f"about {geography / r2:.0%} of the {r2:.0%} the four groups explain together; "
            "socioeconomic development, population and historical responsibility add little "
            "independent variance. These are descriptive allocations, not causal contributions, "
            "and they say nothing about how much physical warming greenhouse gases cause."
        )
    if residual_share is not None:
        findings.append(
            f"**About {residual_share:.0%} of cross-country variance lies outside the static model, "
            "and it is spatially clustered.** Re-measuring geography over land area, a baseline "
            "dryness index and a richer latitude form did not demonstrate transferable improvement "
            "under geographically separated validation; spatial error and spatial lag models do "
            "improve it under Berkeley Earth and represent roughly 40% of the static residual as "
            "spatial covariance, but that gain does not reproduce with ERA5 outcomes."
        )
    findings.append(
        "**The countries least responsible warm no less and can adapt least.** Responsibility "
        "rises steeply with income and falls with ND-GAIN vulnerability, while area-weighted "
        "warming is flat across both."
    )
    st.markdown("\n".join(f"- {f}" for f in findings))

    st.subheader("How the site is organised")
    st.markdown(
        "1. **Geography of warming**: where land warmed fastest and how the definition of "
        "national warming changes the picture.\n"
        "2. **Responsibility and warming**: cumulative emissions against experienced warming "
        "under four national definitions, with the income and vulnerability strata.\n"
        "3. **Variance decomposition** and **Stability**: the four-group decomposition, its "
        "bootstrap intervals and its sensitivity to the outcome construction.\n"
        "4. **Residual structure** and **Robustness**: what remains after the static model, "
        "the four investigations of it, and how much survives a change of spatial graph, "
        "exclusion distance or temperature product.\n"
        "5. **Interpretation and outlook**: what the study establishes, its limitations, and "
        "the questions it leaves open.\n"
        "6. **Context**: the global forcing regression (a separate analysis of the global "
        "trajectory over time) and the post-2013 check of the fitted trends.\n"
        "7. **Data**: the interpolated surface, the city explorer, the regression views and "
        "the sources and methods."
    )
    st.caption(
        "Completed research snapshot. Every number on this site is read from committed "
        "result records; nothing is downloaded or re-estimated when a page loads."
    )
