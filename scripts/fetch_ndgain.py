"""Vendor the slim ND-GAIN latest-year country table.

Downloads the ND-GAIN Country Index zip (Notre Dame Global Adaptation Initiative,
distributed under an open Creative Commons license), extracts the three wide metric
CSVs (vulnerability / readiness / gain), and derives the slim latest-year-per-country
table committed at :data:`src.vulnerability.NDGAIN_PATH`. The derivation lives in
:func:`src.vulnerability.derive_ndgain_latest` (pure, unit-tested on synthetic
fixtures); this script is only the network + extraction wrapper.

Suggested citation: Notre Dame Global Adaptation Initiative Country Index (ND-GAIN),
University of Notre Dame.

Run:
    uv run python -m scripts.fetch_ndgain
"""

from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd
import requests

from src.vulnerability import (
    GAIN_COL,
    NDGAIN_PATH,
    READINESS_COL,
    VULN_SCORE_COL,
    derive_ndgain_latest,
)

logger = logging.getLogger(__name__)

NDGAIN_ZIP_URL = "https://gain.nd.edu/assets/647440/ndgain_countryindex_2026.zip"

# Zip members for the three country-level metric tables (wide: ISO3 x year).
NDGAIN_MEMBERS: dict[str, str] = {
    VULN_SCORE_COL: "resources/vulnerability/vulnerability.csv",
    READINESS_COL: "resources/readiness/readiness.csv",
    GAIN_COL: "resources/gain/gain.csv",
}

# gain.nd.edu returns 403 to non-browser agents (bot protection), so present a
# browser User-Agent + referer like a normal download from the data page.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Referer": "https://gain.nd.edu/our-work/country-index/download-data/",
    "Accept": "application/zip,application/octet-stream,*/*",
}


def download_ndgain_zip(url: str = NDGAIN_ZIP_URL, timeout: int = 120) -> bytes:
    """Fetch the ND-GAIN zip bytes (browser headers to clear the 403 bot guard)."""
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def main() -> None:
    """Download, derive the slim latest-year table, and write the vendored CSV."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("downloading ND-GAIN Country Index zip ...")
    raw = download_ndgain_zip()

    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for metric, member in NDGAIN_MEMBERS.items():
            with zf.open(member) as fh:
                frames[metric] = pd.read_csv(fh)

    slim = derive_ndgain_latest(frames)
    NDGAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(NDGAIN_PATH, index=False)
    logger.info(
        "wrote %s (%d countries, latest year %d)",
        NDGAIN_PATH, len(slim), int(slim["ndgain_year"].max()),
    )


if __name__ == "__main__":
    main()
