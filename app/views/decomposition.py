"""Variance decomposition page: how cross-country differences in warming are structured.

Reads the committed decomposition and inequality summaries and, when present,
the residual-structure summary that carries the full-rank shares with their
bootstrap intervals. Renders a pending state if the decomposition summaries are
absent.
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
    "Historical responsibility only": "emissions",
    "Geography only": "geography",
    "Full model": "full",
}

_VIEW_CAPTION = {
    "emissions": (
        "Historical responsibility alone: its standalone R squared, before geography and "
        "the other groups are credited. Under the area-weighted outcome it has almost no "
        "standalone association with warming; under the station-weighted outcome it does "
        "(see the sensitivity below)."
    ),
    "geography": (
        "Physical geography alone: its standalone R squared. It is the largest structuring "
        "group under either outcome definition."
    ),
    "full": (
        "All four groups together. Each segment is a Shapley share: variance credited to "
        "that group with overlaps split evenly across every order the groups could enter."
    ),
}

_OUTCOME_GLOSS = {
    "area_weighted": (
        "a Theil\u2013Sen trend per 1° Berkeley Earth grid cell, averaged with cos-latitude "
        "weights over cells assigned to the country at their centres. This approximates "
        "land-area weighting; coastlines, small islands and borders have coarse support"
    ),
    "station_weighted": (
        "the unweighted mean of the country's city-location trends from the Berkeley Earth "
        "station records"
    ),
}

_STATION_KEY = "station_weighted"
_STATION_ALL_KEY = "station_weighted_all_countries"


def render() -> None:
    """Render the variance decomposition page."""
    st.title("What structures cross-country differences in warming?")

    ineq = loaders.load_inequality_summary()
    decomp = loaders.load_decomposition_summary()
    if ineq is None or decomp is None:
        st.info(_NOT_BUILT)
        return
    residual = loaders.load_residual_summary()

    outcome_key = str(decomp.get("outcome_definition", "station_weighted"))
    outcome = theme.outcome_label(outcome_key)
    gloss = _OUTCOME_GLOSS.get(outcome_key, "")
    st.markdown(
        "Every country's land-warming rate over 1950\u20132013 is taken as its "
        f"**{outcome}** rate" + (f": {gloss}." if gloss else ".") + " A group-level Shapley "
        "decomposition then splits the cross-country variance of that rate among four fixed "
        "groups of country attributes, with overlaps shared evenly across the orders in which "
        "the groups could enter a linear model, and an explicit residual for what no group "
        "explains."
    )

    static = residual["static"] if residual is not None else None
    if static is not None:
        n = static["n"]
        total_r2 = static["in_sample_r2"]
        shares = {k: v["point"] for k, v in static["shares"].items() if k != "residual"}
        residual_share = static["shares"]["residual"]["point"]
    else:
        n = decomp["n"]
        total_r2 = decomp["total_r2"]
        shares = dict(decomp["shares"])
        residual_share = decomp["residual_share"]
    geography = shares.get("geography")
    emissions = shares.get("emissions")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Countries", f"{n}", help="Complete cases with an area-weighted rate and every feature.")
    m2.metric(
        "Variance explained", f"{total_r2:.1%}",
        help=f"R squared of the full linear model of {outcome} warming over {n} countries.",
    )
    m3.metric(
        "Geography share", "n/a" if geography is None else f"{geography:.1%}",
        help="Share of total cross-country variance; "
        + ("n/a" if geography is None else f"{geography / total_r2:.0%} of the explained variance."),
    )
    m4.metric(
        "Residual", f"{residual_share:.1%}",
        help="Share of total cross-country variance no named group explains.",
    )

    st.subheader("How the variance decomposes")
    if static is not None:
        theme.plotly_chart(
            charts.share_ci_chart(static["shares"], block_groups=static["block_shares"]),
            width="stretch",
        )
        p_country = static["p_geography_largest"]["country"]
        p_block = static["p_geography_largest"]["block"]
        st.caption(
            f"Points are the Shapley shares of total cross-country variance; bars are 95% "
            f"intervals from {static['n_bootstrap']:,} country-bootstrap resamples, and the "
            "faint bands are the wider continent-block bootstrap, which respects spatial "
            f"correlation within continents. Geography is the largest group in "
            f"{p_country:.0%} of country resamples and {p_block:.0%} of block resamples."
        )
    else:
        theme.plotly_chart(charts.shares_bar(shares, residual_share), width="stretch")

    st.markdown(
        _headline(geography, emissions, total_r2, residual_share, outcome)
    )

    st.divider()
    st.subheader("Decomposition explorer")
    st.markdown(
        "Each group on its own against the full model. A group's standalone R squared can "
        "be far larger than its Shapley share when the variance it explains overlaps with "
        "other groups."
    )
    choice = st.radio(
        "View",
        list(_VIEW_OPTIONS),
        horizontal=True,
        label_visibility="collapsed",
    )
    view = _VIEW_OPTIONS[choice]
    explorer_summary = {
        "total_r2": total_r2,
        "shares": shares,
        "residual_share": residual_share,
        "univariate_r2": decomp["univariate_r2"],
    }
    if view == "full":
        explained = total_r2
    else:
        explained = decomp["univariate_r2"][view]
    left, right = st.columns([1, 3])
    left.metric("Variance explained", f"{explained:.1%}")
    if view != "full" and shares.get(view) is not None:
        left.caption(
            f"Shapley share in the full model: **{shares[view]:.1%}**. The rest of the "
            "standalone R squared is shared with other groups."
        )
    with right:
        theme.plotly_chart(charts.explorer_bar(explorer_summary, view), width="stretch")
    st.caption(_VIEW_CAPTION[view])

    station = (decomp.get("sensitivity") or {}).get(_STATION_KEY)
    if station:
        st.divider()
        st.subheader("Outcome sensitivity: station-weighted")
        st.markdown(
            "The same decomposition with each country's warming taken as the unweighted "
            "mean of its station trends instead of the area-weighted rate. The *same "
            "countries* column changes nothing but the outcome; the *all countries* column "
            "adds the countries that have no area-weighted value."
        )
        station_all = (decomp.get("sensitivity") or {}).get(_STATION_ALL_KEY)
        st.dataframe(
            _sensitivity_table(decomp, station, station_all), width="stretch"
        )
        st.caption(_sensitivity_caption(decomp, station, static is not None))

    with st.expander("What each group contains"):
        for key in theme.GROUP_ORDER:
            if key == "residual" or key not in shares:
                continue
            features = decomp["group_features"].get(key, [])
            st.markdown(
                f"- **{theme.GROUP_LABELS[key]}**: {theme.GROUP_GLOSS[key]}. "
                f"Features: `{', '.join(features)}`."
            )
        if static is not None:
            st.markdown(
                "The headline shares use the full-rank representation of the model, in which "
                "the responsibility group holds cumulative total CO₂ and the population group "
                "holds population and station density. Cumulative CO₂ per capita is the total "
                "divided by the same population, so keeping all three columns leaves the fitted "
                "model unchanged but makes the allocation between those two groups "
                "representation-dependent; the per-capita representation gives geography "
                f"{static['per_capita_representation_shares']['geography']:.1%} and "
                f"responsibility {static['per_capita_representation_shares']['emissions']:.1%}."
            )

    with st.expander("What this can and cannot say"):
        st.markdown(
            "**It can say** how the cross-country variance of observed warming aligns with "
            "each kind of country attribute: here, that differences in land-warming rates "
            "are associated primarily with physical geography, while historical emissions "
            "responsibility, income group and population add little independent explanatory "
            "variance. The station-weighted sensitivity shows how much the apparent "
            "responsibility share depends on how national warming is constructed.\n\n"
            "**It cannot say** that emissions caused a country's warming, or that they did "
            "not. Greenhouse gases drive global warming, but CO₂ is well mixed, so a "
            "country's own emissions do not preferentially heat its territory; this page asks "
            "a different question, namely why warming rates differ among countries. A small "
            "responsibility share is not a statement that greenhouse gases are physically "
            "unimportant. Nor is this a map of climate impact: the outcome is the mean "
            "warming rate, which understates the burden on low-emitting tropical countries "
            "that are most exposed to heat and least able to adapt. The shares are a "
            "descriptive allocation of explained cross-country variance, not causal climate "
            "attribution."
        )


def _headline(
    geography: float | None, emissions: float | None, total_r2: float,
    residual_share: float, outcome: str,
) -> str:
    """Plain-language reading of the dominant structure."""
    if geography is None:
        return ""
    text = (
        f"**Geography** is the largest structuring group of {outcome} warming: "
        f"{geography:.1%} of total cross-country variance, or {geography / total_r2:.0%} of "
        f"the {total_r2:.0%} the model explains."
    )
    if emissions is not None:
        text += (
            f" **Historical responsibility** contributes {emissions:.1%} of total variance, "
            "very little independent explanatory variance once its overlap with the other "
            "groups is shared out."
        )
    text += (
        f" The residual is {residual_share:.0%}. Geography accounts for most of the "
        "*explained cross-country variance*; that is not a statement that it accounts for "
        "most physical warming."
    )
    return text


def _sensitivity_table(
    primary: dict, station: dict, station_all: dict | None
) -> pd.DataFrame:
    """Side-by-side shares (as percentages of total variance) per outcome."""
    columns: list[tuple[str, dict]] = [
        (f"{theme.outcome_label(primary.get('outcome_definition')).capitalize()} (original representation)", primary),
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
    rows["Residual"] = [f"{block['residual_share']:.1%}" for _, block in columns]
    rows["Responsibility standalone R squared"] = [
        f"{block['univariate_r2'].get('emissions', float('nan')):.3f}" for _, block in columns
    ]
    rows["Total R squared"] = [f"{block['total_r2']:.3f}" for _, block in columns]
    rows["Countries"] = [f"{block['n']}" for _, block in columns]
    return pd.DataFrame(rows, index=[name for name, _ in columns]).T


def _sensitivity_caption(primary: dict, station: dict, full_rank_headline: bool) -> str:
    """Plain-language reading of the outcome comparison, using the numbers shown."""
    geo_p = primary["shares"].get("geography")
    geo_s = station["shares"].get("geography")
    em_p = primary["shares"].get("emissions")
    em_s = station["shares"].get("emissions")
    parts = []
    if geo_p is not None and geo_s is not None:
        parts.append(
            f"Geography is the largest group under both constructions "
            f"({geo_p:.1%} vs {geo_s:.1%} of total variance)"
        )
    if em_p is not None and em_s is not None:
        parts.append(f"the responsibility share is {em_p:.1%} vs {em_s:.1%}")
    lead = "; ".join(parts) + ". " if parts else ""
    note = (
        "This comparison uses the original representation of the model (responsibility "
        "measured by both per-capita and total cumulative CO₂), so its area-weighted shares "
        "differ slightly from the full-rank headline above; the fitted model is identical. "
        if full_rank_headline else ""
    )
    return (
        lead + note
        + "The difference is consistent with station-network geography (the responsibility "
        "page shows the same shift in the rank correlation), but the area-weighted outcome "
        "also comes from a gridded product, so the comparison cannot isolate station siting "
        "as the sole cause. Neither outcome is independent of the station record: Berkeley "
        "Earth's grid is built from it."
    )
