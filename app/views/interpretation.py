"""Interpretation, limitations and outlook: what the completed study does and does not establish."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the interpretation page."""
    st.title("Interpretation, limitations and outlook")

    st.subheader("What the study establishes")
    st.markdown(
        "- Land warming over 1950\u20132013 was universal across the sampled locations and "
        "uneven: it rises toward the northern high latitudes, and the fastest-warming "
        "national rates are in Central Asia, the Iranian plateau and the Sahel rather "
        "than in the Arctic.\n"
        "- Historical emissions responsibility and experienced warming are different axes. "
        "The rank correlation between them that appears under a station-weighted definition "
        "of national warming disappears under an area-weighted one, under two independent "
        "temperature products.\n"
        "- Cross-country differences in area-weighted warming are associated primarily with "
        "physical geography, which accounts for about half of total variance and most of "
        "what a linear model of four attribute groups explains; historical responsibility, "
        "income and population add little independent variance. The allocation is stable "
        "under resampling, single-country deletion and a change of temperature product.\n"
        "- The countries least responsible warm no less and can adapt least: responsibility "
        "rises steeply with income and falls with vulnerability while area-weighted warming "
        "is flat across both.\n"
        "- Roughly a third of cross-country variance lies outside the static model and is "
        "spatially clustered. Re-measuring geography, a baseline dryness index and a richer "
        "latitude form did not demonstrate transferable improvement; explicit spatial "
        "dependence improves geographically separated prediction under Berkeley Earth and "
        "represents about 40% of the static residual as covariance, but that gain does not "
        "reproduce with ERA5 outcomes."
    )

    st.subheader("What the remaining variance may contain")
    st.markdown(
        "The residual is variance in cross-country warming differences that the tested "
        "country-level models do not capture with these observations. It is not an unknown "
        "climate cause, and the design cannot apportion it. It may contain, in unknown "
        "proportions:\n"
        "- omitted regional physical heterogeneity (soil-moisture regimes, regional aerosol "
        "histories, snow and albedo feedbacks, ocean influence on mid-latitude land);\n"
        "- internal climate variability aliased into a 64-year linear trend;\n"
        "- limitations of the linear, additive functional form;\n"
        "- differences between temperature products in station-sparse regions;\n"
        "- spatial dependence beyond what a fixed neighbour graph represents;\n"
        "- measurement error in the country-level predictors;\n"
        "- the aggregation of heterogeneous territories to one value per country;\n"
        "- finite-sample error at 151 countries."
    )

    st.subheader("Limitations")
    st.markdown(
        "- **Ecological aggregation.** Every country is one unit; nothing here describes "
        "variation within countries, and a country mean can invert relationships that hold "
        "at finer scales.\n"
        "- **Sample.** The decomposition uses the 151 countries with complete features; the "
        "responsibility comparison 154 to 157. Countries with no 1° land cell or no "
        "classified climate class are absent.\n"
        "- **Equal-country weighting.** The decomposition gives Luxembourg and Russia the same "
        "weight; an area- or population-weighted regression would answer a different question.\n"
        "- **Products.** Berkeley Earth interpolates across data-sparse interiors from the "
        "same stations that define the station-weighted rate, and ERA5's reanalysis has its "
        "own assimilation and model error structure. The two disagree on national trends by "
        "more than the residual.\n"
        "- **Window.** The analysis ends in September 2013. Warming after 2013 ran ahead of "
        "the fitted lines (see the post-2013 check), and the period sampled is slower than "
        "the present.\n"
        "- **Uncertainty propagation.** Trend uncertainty at the cell and station level and "
        "measurement error in the predictors are not propagated into the shares; resampling "
        "intervals treat countries as exchangeable and are not spatially corrected.\n"
        "- **Residual dependence.** Spatial dependence persists beyond the 500 km exclusion "
        "used for validation, and the spatial models' held-out predictions rely on the "
        "observed outcomes of neighbouring countries.\n"
        "- **Hypothesis generation.** Several follow-up hypotheses were motivated by the same "
        "outcome data they were tested on, so a consistent result is not independent "
        "confirmation and a rejected specification does not disprove its mechanism.\n"
        "- **No causal identification.** Every statistic is descriptive. Shares are "
        "allocations of explained cross-country variance; correlations describe alignment; "
        "nothing estimates the effect of a country's emissions on its own climate or "
        "attributes warming to its physical drivers."
    )

    st.subheader("Open questions")
    st.markdown(
        "Several questions follow naturally from these results and would each need a "
        "different design.\n"
        "- **A multi-product outcome.** A latent national warming estimate that treats "
        "Berkeley Earth, ERA5 and other products as noisy measurements would separate "
        "product disagreement from climate signal before any decomposition.\n"
        "- **Time.** Rolling or piecewise decompositions would show whether the geography "
        "share and the residual's spatial pattern are stable across decades or are the "
        "imprint of particular variability phases.\n"
        "- **Sub-national units.** The 1° cell trends allow a within-country versus "
        "between-country split and a hierarchical model of cells within countries, at the "
        "cost of a different estimand.\n"
        "- **Independently motivated residual covariates.** Regional aerosol histories, "
        "baseline snow cover and synoptic-scale ocean exposure are physically motivated "
        "candidates that were recorded but not pursued; each would need its own declared "
        "group and a pre-specified sign.\n"
        "- **Physically meaningful spatial graphs.** A neighbour structure derived from "
        "circulation or coastline geometry rather than centroid distance would make the "
        "spatial accounting interpretable rather than merely representative.\n"
        "- **Other definitions of responsibility.** Consumption-based accounting, era-weighted "
        "populations, non-CO₂ gases and land-use emissions change who is responsible and "
        "could change the alignment.\n"
        "- **Impact outcomes.** Extreme-heat days, exposure-weighted degree days or damages "
        "would measure burden rather than mean warming, which understates the burden on "
        "tropical low emitters.\n"
        "- **Detection and attribution.** Whether observed regional warming is consistent "
        "with forced response is a distinct question, answered with climate-model "
        "fingerprints rather than cross-country regression."
    )
