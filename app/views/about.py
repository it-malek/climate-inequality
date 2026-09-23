"""Data and methods page: sources, pipeline, reproduction and the snapshot record."""

from __future__ import annotations

import json

import streamlit as st

from app import loaders
from src import snapshot

SNAPSHOT_VERSION = "2.0.0"
SNAPSHOT_DATE = "September 2026"
REPOSITORY_URL = "https://github.com/it-malek/climate-inequality"


def render() -> None:
    """Render the data-and-methods page."""
    stats = loaders.load_stats()
    provenance = stats.get("provenance", {})

    st.title("Data and methods")
    st.markdown(
        f"This site is the interactive record of a completed, independent research project "
        f"(snapshot {SNAPSHOT_VERSION}, {SNAPSHOT_DATE}). Every page reads committed result "
        "records from the repository; no page downloads data or re-estimates a result. The "
        f"code, records and documentation are at [{REPOSITORY_URL}]({REPOSITORY_URL})."
    )

    identity = snapshot.fingerprint(loaders.APP_DATA_DIR, require_manifest=True)
    manifest = json.loads((loaders.APP_DATA_DIR / snapshot.MANIFEST).read_text())
    st.caption(f"Verified result bundle SHA-256: {identity}")
    st.download_button(
        "Download release manifest", json.dumps(manifest, indent=2),
        file_name="climate-inequality-2.0.0.json", mime="application/json",
    )

    st.subheader("Data sources")
    st.markdown(
        "- **Temperatures.** Berkeley Earth city records (Kaggle mirror, monthly means for "
        f"{stats['trends']['n_locations']:,} city locations) and the Berkeley Earth 1° gridded "
        "land product; ERA5 monthly 2 m temperature (Copernicus Climate Data Store) as an "
        "independent product.\n"
        "- **Emissions and population.** Our World in Data CO₂ and greenhouse-gas emissions "
        f"(pinned release, retrieved {provenance.get('owid_co2_retrieved', '2026')}) and "
        "population; Our World in Data continent classification.\n"
        "- **Geography.** ETOPO 2022 relief, the Köppen\u2013Geiger classification of Beck and "
        "colleagues (2018), Natural Earth land polygons, SEDAC GPW v4.11 population counts and "
        "national-identifier grids.\n"
        "- **Socioeconomic.** World Bank income classification (Our World in Data mirror); "
        "ND-GAIN Country Index vulnerability and readiness scores.\n"
        "- **Global forcing regression.** NASA GISTEMP v4, the AR6-consistent effective "
        "radiative forcing series of Forster and colleagues, and the NOAA Oceanic Niño Index.\n"
        "- **Residual investigation.** CRU TS v4.10 precipitation and potential "
        "evapotranspiration (baseline dryness) and GSHHG shorelines (area-consistent "
        "geography)."
    )

    st.subheader("Methods in brief")
    st.markdown(
        "- **Trends.** Theil\u2013Sen slopes on monthly anomalies against each location's "
        "1951\u20131980 climatology over January 1950 to September 2013, requiring 90% coverage.\n"
        "- **National warming.** Station-weighted (mean of city-location trends), "
        "population-weighted (GPW counts at each location) and area-weighted (a trend per 1° "
        "cell, averaged over the country's land with cos-latitude weights).\n"
        "- **Responsibility.** Cumulative production-based CO₂ through 2013 per 2013 resident; "
        "a window-matched consumption-based variant for the countries that have one.\n"
        "- **Decomposition.** Group-level Shapley (LMG) shares of the R squared of a linear "
        "model of area-weighted warming on four attribute groups over 151 countries, with "
        "country and continent-block bootstraps, leave-one-out influence and Moran's I.\n"
        "- **Residual investigation.** Leave-one-country-out validation with a 500 km "
        "territorial exclusion; paired country-bootstrap intervals on the change in "
        "out-of-fold error; spatial error and spatial lag models by maximum likelihood on a "
        "fixed 8-nearest-neighbour graph; every result independently reconstructed and "
        "rerun byte-identically.\n"
        "- **Global forcing regression.** A closed-form Bayesian regression of the annual "
        "global mean anomaly on lagged forcings and the ENSO index with AR(1) errors, trained "
        "through 2013 and scored on 2014\u20132024."
    )

    st.subheader("Reproduction")
    st.markdown(
        "The dashboard and the test suite run from a clean clone with no data download: "
        "the app reads only the committed bundle and the tests use synthetic fixtures."
    )
    st.code(
        "uv sync --extra dev\n"
        "uv run pytest -q\n"
        "uv run streamlit run app/streamlit_app.py",
        language="bash",
    )
    st.markdown(
        "Rebuilding every record from the raw sources takes several hours of downloads and "
        "computation; the stage-by-stage sequence, the parameters of every stage, and the "
        "integrity checks that keep a rebuild honest are documented in the repository's "
        "`docs/reproducibility.md`, and the residual investigation in "
        "`research/model_v2/README.md`."
    )

    st.subheader("About")
    st.markdown(
        "The project was designed, built and analysed by Malek Elaghel as an independent "
        "personal research project. It began as a 2022 undergraduate research proposal to "
        "map warming from the Berkeley Earth city data; the current pipeline and analyses "
        "were built in 2026. It received no external funding."
    )
    st.caption(
        "Berkeley Earth data are distributed under a Creative Commons licence; Our World in "
        "Data tables under CC BY 4.0; ND-GAIN data under an open Creative Commons licence; "
        "ERA5 under the Copernicus licence. Full notices accompany the vendored tables in the "
        "repository."
    )
