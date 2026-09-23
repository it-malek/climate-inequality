"""Residual structure page: what remains after the static model, and the four investigations of it.

Reads the committed ``residual_structure_summary.json`` (copied from the
research records); renders a pending state if it is absent.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts, loaders, theme

_NOT_BUILT = (
    "The residual-structure summary has not been built yet. It is copied from the "
    "research records by `python -m src.residual_assets`."
)


def _interval(row: dict) -> str:
    lo, hi = row["interval"]
    return f"{row['delta_rmse']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def render() -> None:
    """Render the residual-structure page."""
    st.title("What remains after the static model?")
    summary = loaders.load_residual_summary()
    if summary is None:
        st.info(_NOT_BUILT)
        return

    static = summary["static"]
    rows = {r["key"]: r for r in summary["investigations"]}
    spatial = summary["spatial"]

    st.markdown(
        f"The static model leaves about **{static['shares']['residual']['point']:.0%}** of the "
        "cross-country variance in area-weighted warming outside its four groups, and that "
        "residual has spatial structure: countries that warmed faster than the model predicts sit "
        "next to each other (Central Asia, the Atlantic Sahel, the Gulf), and so do those "
        "that warmed more slowly (the Levant and Egypt, mainland South-east Asia, southern "
        "Africa, Central America). This page asks whether any of that structure can be "
        "recovered by a model that transfers to countries whose neighbourhood it has not "
        "seen."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Residual share", f"{static['shares']['residual']['point']:.1%}",
        help=f"95% country-bootstrap interval {static['shares']['residual']['ci_low']:.1%} to "
        f"{static['shares']['residual']['ci_high']:.1%}.",
    )
    c2.metric(
        "Residual Moran's I", f"{static['residual_morans_i_in_sample']:.3f}",
        help="In-sample residual, 8 nearest neighbours on station centroids; out of fold it is "
        f"{static['residual_morans_i_out_of_fold']:.3f}.",
    )
    c3.metric(
        "In-sample R squared", f"{static['in_sample_r2']:.3f}",
        help="Fit on all 151 countries.",
    )
    c4.metric(
        "Geographically separated R squared", f"{static['primary_cv_r2']:.3f}",
        help=f"Leave-one-country-out with every country within 500 km of the held-out "
        f"territory removed from training; RMSE {static['primary_cv_rmse']:.4f} "
        f"{theme.TREND_UNIT}. Random 10-fold gives {static['random_cv_r2']:.3f}.",
    )

    st.markdown(
        "**How transfer is measured.** Each country is predicted by a model fitted without "
        "it and without every country whose territory lies within 500 km of it, so bordering "
        "and near-bordering countries never inform a prediction. The gap between the "
        f"in-sample R squared ({static['in_sample_r2']:.2f}) and this geographically "
        f"separated R squared ({static['primary_cv_r2']:.2f}) is the part of the fit that "
        "does not transfer. Every candidate below is compared with the static model on the "
        "same folds; the change in out-of-fold error carries a 95% interval from a paired "
        "country bootstrap, which treats countries as exchangeable and is not spatially "
        "corrected."
    )

    st.subheader("Four investigations, one yardstick")
    order = ["geography_measurement", "baseline_dryness", "latitude_form", "spatial_error", "spatial_lag"]
    theme.plotly_chart(
        charts.delta_rmse_forest([rows[k] for k in order], static["primary_cv_rmse"]),
        width="stretch",
    )
    st.caption(
        "Blue: the whole interval is an improvement. Vermillion: the whole interval is a "
        "deterioration. Grey: the interval includes zero. A candidate demonstrates "
        "transferable improvement only when its interval lies entirely below zero."
    )

    st.markdown("#### Geographic measurement")
    g = rows["geography_measurement"]
    st.markdown(
        "The geography features of the static model are measured at station locations, "
        "while the outcome describes whole land areas. Re-measuring them over each "
        "country's territory changed the inputs a great deal (Canada's absolute latitude "
        "rises from 46.8° to 58.7°; 17 negative station-sampled elevations become positive; "
        "29 countries change climate class) but did not improve transfer: the paired change "
        f"in error is {_interval(g)} {theme.TREND_UNIT}, the sub-region hold-out and "
        "worst-region errors rose, and the out-of-fold residual Moran's I rose from "
        f"{static['residual_morans_i_out_of_fold']:.2f} to {g['oof_morans_i']:.2f}. "
        "Better-aligned measurement neither reduced the remaining spatial structure nor "
        "changed the group conclusion, so it was retained as a sensitivity rather than "
        "adopted."
    )

    st.markdown("#### Baseline hydroclimate")
    d = rows["baseline_dryness"]
    st.markdown(
        "One pre-specified physical covariate was added: a baseline dryness index (the "
        "area-weighted log ratio of precipitation to potential evapotranspiration over "
        "1920\u20131949), with the expectation that drier regimes warm faster. It added "
        "essentially no in-sample fit and made geographically separated prediction worse: "
        f"{_interval(d)} {theme.TREND_UNIT}, with the whole interval above zero, and the "
        "coefficient carried the opposite of the expected sign in the full fit and in most "
        "training fits. This does not show that hydroclimate is irrelevant to cross-country "
        "warming differences. It may reflect information the geography group already "
        "carries, the climatological anchoring of the pre-1950 fields, noise in the index, "
        "limited power at 151 countries, or a relationship that is not linear; no "
        "alternative window, transform or nonlinear term was tried in response."
    )

    st.markdown("#### Flexible latitude response")
    lat = rows["latitude_form"]
    st.markdown(
        "The residual falls with latitude in the Southern Hemisphere and rises at 50\u201370°N, "
        "which a single latitude slope cannot represent. Adding two natural-spline terms "
        "in latitude and a southern-hemisphere slope difference improved the point "
        f"estimate ({_interval(lat)} {theme.TREND_UNIT}; geographically separated R squared "
        f"{static['primary_cv_r2']:.3f} to {lat['primary_cv_r2']:.3f}) but the paired "
        "interval narrowly includes zero, so the richer form is not declared a transferable "
        "improvement, and the out-of-fold residual stayed spatially structured (Moran's I "
        f"{lat['oof_morans_i']:.2f}). The southern slope difference was negative in every "
        "training fit, which is consistent with the residual pattern that motivated the "
        "test; because the hypothesis came from the same outcome, that is not independent "
        "confirmation."
    )

    st.markdown("#### Explicit spatial dependence")
    sem = spatial["spatial_error"]["station"]
    sar = spatial["spatial_lag"]["station"]
    st.markdown(
        "Two spatial models were fitted on the unchanged static specification with a fixed "
        "8-nearest-neighbour graph on country centroids: a spatial error model (dependence "
        "in the errors) and a spatial lag model (dependence in the outcome). Both improve "
        "geographically separated prediction under Berkeley Earth: the spatial error model "
        f"by {_interval(sem)} and the spatial lag model by {_interval(sar)} "
        f"{theme.TREND_UNIT}, with lower sub-region hold-out error and out-of-fold Moran's I "
        f"falling to about {sem['oof_morans_i']:.2f}. Both meet the same pre-set conditions "
        "and neither is declared the single final model. Their held-out predictions use the "
        "observed outcomes of training countries at least 500 km away, so the gain depends "
        "on neighbouring observations being available; it is not a better static model."
    )
    theme.plotly_chart(charts.accounting_bar(spatial), width="stretch")
    acc_sem = sem["accounting"]
    acc_sar = sar["accounting"]
    st.caption(
        "In sample, the dependence part represents "
        f"{acc_sem['a_dependence'] / (1 - acc_sem['a_static']):.0%} (spatial error) and "
        f"{acc_sar['a_dependence'] / (1 - acc_sar['a_static']):.0%} (spatial lag) of the "
        "static residual. That is represented spatial covariance: not 40% of warming, not a "
        "causal share, not a mechanism, and not additive with the geography share of the "
        "static decomposition, whose group conclusion the non-spatial part of each model "
        "leaves unchanged."
    )

    st.subheader("Summary table")
    table = pd.DataFrame([
        {
            "Extension": r["label"],
            "Verdict": r["verdict"],
            "Out-of-fold RMSE": r["primary_cv_rmse"],
            "Sub-region hold-out RMSE": r["m49_cv_rmse"],
            "Out-of-fold Moran's I": r["oof_morans_i"],
            "Change vs static [95% interval]": _interval(r),
        }
        for r in [
            {"label": "Static model", "verdict": "comparator",
             "primary_cv_rmse": static["primary_cv_rmse"], "m49_cv_rmse": static["m49_cv_rmse"],
             "oof_morans_i": static["residual_morans_i_out_of_fold"],
             "delta_rmse": 0.0, "interval": [0.0, 0.0]},
            *[rows[k] for k in order],
        ]
    ])
    table.loc[0, "Change vs static [95% interval]"] = ""
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Out-of-fold RMSE": st.column_config.NumberColumn(format="%.4f"),
            "Sub-region hold-out RMSE": st.column_config.NumberColumn(format="%.4f"),
            "Out-of-fold Moran's I": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        f"Errors in {theme.TREND_UNIT} over the 151 decomposition countries. The full records, "
        "including the sub-region and worst-region conditions and the independent "
        "reconstruction of every result, are in the repository's research directory."
    )
