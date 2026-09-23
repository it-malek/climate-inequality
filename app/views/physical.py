"""Global forcing page: the global mean temperature as a response to radiative forcing.

A separate analysis from the cross-country pages: it explains the global
trajectory over time, which has no cross-country variance to explain. Reads
``physical_summary.json`` and ``physical_trajectory.parquet``; renders a
pending state if they are absent.
"""

from __future__ import annotations

import streamlit as st

from app import charts, loaders, theme

_NOT_BUILT = (
    "The forcing-regression artifacts have not been built yet. They need the "
    "network-derived `forcings.parquet`: run `python -m src.forcings` and then "
    "rebuild the bundle (`python -m src.app_assets`)."
)

# Presentation metadata: major low-latitude eruptions whose stratospheric
# aerosol drove a multi-year cooling the model captures through the volcanic term.
ERUPTIONS = [(1963, "Agung"), (1982, "El Chichón"), (1991, "Pinatubo")]


def render() -> None:
    """Render the global forcing page."""
    summary = loaders.load_physical_summary()
    trajectory = loaders.load_physical_trajectory()
    if summary is None or trajectory is None:
        st.info(_NOT_BUILT)
        return

    train_end = summary["train_end"]
    last_year = int(trajectory["year"].max())

    st.title("Global temperature and radiative forcing")
    st.markdown(
        "**A different statistical question.** The cross-country pages ask why warming rates "
        "differ among countries. This page asks what explains the global mean temperature "
        "through time: a regression of the annual anomaly on effective radiative forcings "
        "(CO₂, CH₄, N₂O, aerosol, volcanic, solar) and the ENSO state, with an AR(1) error, "
        f"fitted on data through **{train_end}** and asked to predict "
        f"**{train_end + 1}\u2013{last_year}**. A global signal is constant across countries, so "
        "it contributes nothing to cross-country variance; the two analyses are never combined."
    )
    st.caption(summary["interpretation"])

    theme.plotly_chart(
        charts.physical_trajectory_chart(trajectory, train_end, ERUPTIONS),
        width="stretch",
    )
    st.caption(
        "The dips after **Agung (1963)**, **El Chichón (1982)** and **Pinatubo (1991)** come "
        "from the volcanic forcing term: the regression reproduces transient shocks, not "
        "only a secular trend."
    )

    h = summary["hindcast"]
    st.subheader(f"Trained through {train_end}, tested on {train_end + 1}\u2013{last_year}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train R²", f"{h['train_r2']:.2f}", help="In-sample fit through the cutoff.")
    c2.metric("Test RMSE", f"{h['test_rmse']:.3f} °C", help="Out-of-sample error.")
    c3.metric(
        "Band coverage", f"{h['test_band_coverage']:.0%}",
        help=f"Share of the {h['n_test']} out-of-sample years inside the 95% band.",
    )
    c4.metric("AR(1) ρ", f"{summary['ar1_rho']:+.2f}", help="Estimated noise persistence.")

    left, right = st.columns(2)
    with left:
        theme.plotly_chart(charts.sensitivity_forest(summary["sensitivity"]), width="stretch")
        st.caption(
            "Read the intervals, not the points. The CO₂, CH₄ and N₂O forcings rise together "
            "over the period, so their individual coefficients are only partly identified; "
            "the N₂O estimate in particular is a large number with an interval that spans "
            "zero. The volcanic and ENSO terms, driven by sharp uncorrelated signals, are the "
            "tightly estimated ones."
        )
    with right:
        out = trajectory[trajectory["year"] > train_end]
        half_width = (out["upper95"] - out["lower95"]) / 2.0
        st.markdown("**Recent-year uncertainty**")
        st.markdown(
            "The 95% predictive band over the out-of-sample years spans roughly "
            f"±{half_width.mean():.2f} °C (±{half_width.max():.2f} °C at its widest): the "
            "model's uncertainty about warming it was not fitted on."
        )

    with st.expander("Trajectory table"):
        st.dataframe(trajectory, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV",
            trajectory.to_csv(index=False).encode("utf-8"),
            file_name="physical_trajectory.csv",
            mime="text/csv",
        )
        forcings_hash = summary.get("forcings_hash")
        if forcings_hash:
            st.caption(f"Forcings table SHA-256 `{forcings_hash[:12]}...`")
