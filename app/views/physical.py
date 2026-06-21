"""Physical climate-model page: the Layer 1 driver model (forcings → temperature).

Reads the committed ``physical_summary.json`` and ``physical_trajectory.parquet``
bundle artifacts; renders a pending state if they are absent. The page leads with the
*physical fingerprint* — the model reproduces the transient cooling after major
volcanic eruptions, not just the secular warming trend — then shows the out-of-sample
hindcast skill, with the driver sensitivities and recent-year uncertainty below the
fold. Everything here is a descriptive predictive association validated by hindcast
skill, never a causal detection-and-attribution claim.
"""

from __future__ import annotations

import streamlit as st

from app import charts, loaders

_NOT_BUILT = (
    "The physical-model artifacts have not been built yet. They need the "
    "network-derived `forcings.parquet`: run `python -m src.forcings` and then "
    "rebuild the bundle (`python -m src.app_assets`)."
)

# Presentation metadata (not from the data): major low-latitude eruptions whose
# stratospheric aerosol drove a multi-year cooling the model captures via erf_volcanic.
ERUPTIONS = [(1963, "Agung"), (1982, "El Chichón"), (1991, "Pinatubo")]


def render() -> None:
    """Render the Layer 1 physical-driver model page."""
    summary = loaders.load_physical_summary()
    trajectory = loaders.load_physical_trajectory()
    if summary is None or trajectory is None:
        st.info(_NOT_BUILT)
        return

    train_end = summary["train_end"]
    last_year = int(trajectory["year"].max())

    st.title("The physical climate engine")
    st.markdown(
        "A deterministic model of the global temperature anomaly as a response to "
        "effective radiative forcings (CO₂, CH₄, N₂O, aerosol, volcanic, solar) plus "
        "the ENSO state — fit with an AR(1) error structure on data through "
        f"**{train_end}**, then asked to predict **{train_end + 1}–{last_year}** it had "
        "never seen."
    )
    st.caption(summary["interpretation"])

    # --- HERO: the volcanic fingerprint -------------------------------------
    st.subheader("The model feels volcanic shocks")
    st.plotly_chart(
        charts.physical_trajectory_chart(trajectory, train_end, ERUPTIONS),
        width="stretch",
    )
    st.caption(
        "The sharp dips after **Agung (1963)**, **El Chichón (1982)** and "
        "**Pinatubo (1991)** come from the volcanic forcing term — the model "
        "reproduces transient physical shocks, not just a secular CO₂ trend."
    )

    # --- SECONDARY: out-of-sample hindcast skill ----------------------------
    h = summary["hindcast"]
    st.subheader(f"Trained ≤ {train_end}, tested on {train_end + 1}–{last_year}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train R²", f"{h['train_r2']:.2f}", help="In-sample fit through the cutoff.")
    c2.metric("Test RMSE", f"{h['test_rmse']:.3f} °C", help="Out-of-sample error.")
    c3.metric(
        "Band coverage", f"{h['test_band_coverage']:.0%}",
        help=f"Share of the {h['n_test']} out-of-sample years inside the 95% band.",
    )
    c4.metric("AR(1) ρ", f"{summary['ar1_rho']:+.2f}", help="Estimated noise persistence.")

    # --- BELOW THE FOLD: supporting context ---------------------------------
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.sensitivity_forest(summary["sensitivity"]), width="stretch")
    with right:
        out = trajectory[trajectory["year"] > train_end]
        half_width = (out["upper95"] - out["lower95"]) / 2.0
        st.markdown("**Recent-year uncertainty**")
        st.markdown(
            "The 95% predictive band over the out-of-sample tail spans roughly "
            f"±{half_width.mean():.2f} °C (±{half_width.max():.2f} °C at its widest), "
            "the model's honest confidence about warming it was not fit on."
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
            # Quiet provenance stamp: the SHA-256 of the forcings table this
            # trajectory was fit on, so the committed artifact is traceable.
            st.caption(f"forcings provenance · sha256 `{forcings_hash[:12]}…`")
