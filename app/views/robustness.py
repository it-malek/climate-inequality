"""Robustness page: spatial graph, exclusion distance and temperature product.

Reads the committed ``residual_structure_summary.json`` and, when present, the
ERA5 cross-check summary of the responsibility comparison. Renders a pending
state if the residual summary is absent.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, loaders, theme

_NOT_BUILT = (
    "The residual-structure summary has not been built yet. It is copied from the "
    "research records by `python -m src.residual_assets`."
)


def render() -> None:
    """Render the robustness page."""
    st.title("Robustness and product dependence")
    summary = loaders.load_residual_summary()
    if summary is None:
        st.info(_NOT_BUILT)
        return

    static = summary["static"]
    products = summary["product_static"]
    settings = summary["robustness"]

    st.markdown(
        "Two questions decide how much weight the results can carry. Does the static "
        "decomposition change when the outcome comes from a different temperature product? "
        "And does the predictive advantage of the spatial models survive a different "
        "neighbour graph, a wider exclusion distance, or a different product?"
    )

    st.subheader("The static decomposition under ERA5")
    era5_rows = [
        ("Berkeley Earth (primary)", static["in_sample_r2"], static["primary_cv_rmse"],
         static["shares"]["geography"]["point"], static["shares"]["emissions"]["point"], None),
        ("ERA5, anomaly-aligned", products["era5_aligned"]["in_sample_r2"],
         products["era5_aligned"]["primary_cv_rmse"], products["era5_aligned"]["share_geography"],
         products["era5_aligned"]["share_emissions"],
         products["era5_aligned"]["residual_agreement_with_berkeley"]["pearson"]),
        ("ERA5, absolute temperature", products["era5_absolute"]["in_sample_r2"],
         products["era5_absolute"]["primary_cv_rmse"], products["era5_absolute"]["share_geography"],
         products["era5_absolute"]["share_emissions"],
         products["era5_absolute"]["residual_agreement_with_berkeley"]["pearson"]),
    ]
    st.dataframe(
        pd.DataFrame(era5_rows, columns=[
            "Outcome product", "In-sample R squared", "Out-of-fold RMSE",
            "Geography share", "Responsibility share", "Residual correlation with Berkeley Earth",
        ]),
        width="stretch", hide_index=True,
        column_config={
            "In-sample R squared": st.column_config.NumberColumn(format="%.3f"),
            "Out-of-fold RMSE": st.column_config.NumberColumn(format="%.4f"),
            "Geography share": st.column_config.NumberColumn(format="%.3f"),
            "Responsibility share": st.column_config.NumberColumn(format="%.3f"),
            "Residual correlation with Berkeley Earth": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "Same 151 countries, same features, same folds; only the outcome changes. The "
        "anomaly-aligned construction expresses ERA5 as anomalies against its own 1951\u20131980 "
        "monthly climatology, matching Berkeley Earth's convention; the absolute-temperature "
        "construction is the one behind the ERA5 cross-check below. Geography remains the "
        "largest group and the responsibility share stays far below 0.10 under both, so the "
        "static conclusion is comparatively stable. The fit is weaker and the two products' "
        "residuals agree only moderately, which is why the remaining variance cannot be "
        "read as a property of the climate alone."
    )

    st.subheader("The spatial models across graphs, distances and products")
    theme.plotly_chart(charts.robustness_forest(settings), width="stretch")
    st.caption(
        "Change in geographically separated error relative to the static model fitted on "
        "the same outcome, with 95% paired country-bootstrap intervals. Under Berkeley Earth "
        "the spatial lag model's advantage holds across the land-centroid graph and both "
        "wider exclusion distances; the spatial error model's interval covers zero at "
        "1000 km and its sub-region hold-out error rises under the land-centroid graph. "
        "Under either ERA5 construction neither model improves transfer, although positive "
        "dependence is still estimated."
    )
    st.markdown(
        "**Reading.** The static decomposition is comparatively stable, while the predictive "
        "advantage of the spatial models is product-sensitive and does not reproduce "
        "reliably under ERA5. This does not mean ERA5 disproves Berkeley Earth: the two "
        "products disagree on national trends by more than the residual itself (their "
        "area-weighted rankings agree at about ρ = 0.6), and product disagreement is not an "
        "identified amount or cause of observational error. What can be said is that the "
        "spatial gain belongs to the Berkeley Earth field as observed, and that ERA5 "
        "selected nothing."
    )

    era5 = loaders.load_era5_validation_summary()
    if era5 and era5.get("available"):
        _render_era5_crosscheck(era5)


def _render_era5_crosscheck(summary: dict) -> None:
    """The responsibility comparison repeated on the ERA5 area-weighted trend."""
    wl = summary["world_land_mean"]
    cc = summary["coupling_common"]
    ra = summary["rank_agreement"]

    st.divider()
    st.subheader("The responsibility comparison under ERA5")
    st.markdown(
        "The collapse of the responsibility correlation under area weighting rests on a "
        "gridded product interpolated from the same stations. Repeating the area-weighted "
        "computation on ERA5 (absolute-temperature construction) tests whether the collapse "
        "depends on the product."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "ERA5 world land mean",
        f"{wl['era5_area']:.3f} {theme.TREND_UNIT}",
        help=f"Berkeley Earth's published global-land reference is about {wl['berkeley_reference']}.",
    )
    if wl.get("berkeley_area") is not None:
        c2.metric("Berkeley Earth world land mean", f"{wl['berkeley_area']:.3f} {theme.TREND_UNIT}")
    c3.metric(
        "ERA5 vs Berkeley Earth rank ρ",
        f"{ra['era5_area_vs_berkeley_area']['rho']:+.3f}",
        help="Spearman ρ of the two products' area-weighted country trends.",
    )

    rows = [
        {"Definition": label,
         "Spearman ρ vs responsibility": cc[key]["spearman_vs_responsibility"]["rho"],
         "p": cc[key]["spearman_vs_responsibility"]["p"],
         "Inequality coefficient": cc[key]["gini"]}
        for key, label in (
            ("station", "Station-weighted"),
            ("berkeley_area", "Area-weighted (Berkeley Earth)"),
            ("era5_area", "Area-weighted (ERA5)"),
        )
    ]
    st.markdown(f"**Common {cc['n']}-country set** (every definition scored on identical countries):")
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Spearman ρ vs responsibility": st.column_config.NumberColumn(format="%+.3f"),
            "p": st.column_config.NumberColumn(format="%.2g"),
            "Inequality coefficient": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "The significant station-weighted correlation is not significant under either "
        "area-weighted product and the inequality coefficient rises under both. ERA5's "
        "point estimate keeps a faint positive trace, so the robust statement is that the "
        "correlation collapses to weak and non-significant, not that it vanishes."
    )
