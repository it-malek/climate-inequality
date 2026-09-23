"""ERA5 cross-check of the area-weighting result.

Moving from station-weighted to area-weighted country warming collapses the
warming-responsibility rank correlation (Spearman rho +0.36 -> +0.01) and raises
the inequality coefficient (0.56 -> 0.61). This module re-runs that comparison
with ERA5 reanalysis (:mod:`src.era5_weighting`) in place of the Berkeley Earth
grid and reports three things on one common country set: the world land-mean
slope (an ingest sanity check against Berkeley's ~0.19 deg C/decade), the rank
agreement between the two products, and the coupling of each lens with
responsibility (Spearman rho plus the coupling stage's inequality coefficient).

This preserves the historical absolute-temperature cross-check. The preferred
monthly-anomaly comparison is recorded in research/model_v2/outputs/m4_final/
m4_products.json; seasonal climatology subtraction can change Theil-Sen slopes.

It is a cross-check artifact, not a registered projection: it reuses the
coupling operators directly and writes ``era5_area_trends.parquet`` and
``era5_validation_summary.json``. When the (gitignored) ERA5 grid is absent it
writes an ``{"available": false}`` summary so a bundle build still succeeds.
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from scipy import stats

from src.area_weighting import (
    BERKELEY_GRID_PATH,
    GPW_NATID_LOOKUP_PATH,
    ISO3_COL,
    land_mean_from_slopes,
    world_land_mean,
)
from src.cleaning import DEFAULT_END, DEFAULT_START
from src.coupling import inequality_coefficient
from src.data_io import PROCESSED_DIR, round_floats, write_typed_parquet
from src.emissions import DEFAULT_INEQUALITY_PATH, OWID_CO2_PATH, load_owid_co2
from src.era5_weighting import (
    ERA5_AREA_COL,
    ERA5_COVERAGE_COL,
    ERA5_GRID_PATH,
    era5_cell_slopes,
    reduce_era5_slopes,
)
from src.population import GPW_PATH

logger = logging.getLogger(__name__)

# Committed country_inequality.parquet columns this cross-check reads (mirrors the
# stable head of src.emissions.INEQUALITY_COLUMNS).
OWID_COL = "owid_country"
STATION_COL = "trend_c_per_decade"
RESPONSIBILITY_COL = "cum_co2_t_per_capita"
BERKELEY_AREA_COL = "trend_c_per_decade_area_weighted"

# Berkeley Earth's published 1950-2013 global land trend, which the area-weighted
# Berkeley world-land mean (0.193) matches; ERA5's mean is compared to it too.
BERKELEY_GLOBAL_LAND_REFERENCE = 0.19

DEFAULT_ERA5_TRENDS_PATH = PROCESSED_DIR / "era5_area_trends.parquet"
DEFAULT_ERA5_SUMMARY_PATH = PROCESSED_DIR / "era5_validation_summary.json"

ERA5_TRENDS_SCHEMA: dict[str, str] = {
    ISO3_COL: "VARCHAR",
    ERA5_AREA_COL: "DOUBLE",
    ERA5_COVERAGE_COL: "DOUBLE",
}

_MIN_PAIRS = 3  # Spearman/Gini need at least a few countries to mean anything.


def _spearman(a: pd.Series, b: pd.Series) -> dict:
    """Spearman rho over the rows where both series are finite (rank agreement)."""
    pair = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(pair) < _MIN_PAIRS:
        return {"rho": None, "p": None, "n": int(len(pair))}
    rho, p = stats.spearmanr(pair["a"], pair["b"])
    return {"rho": float(rho), "p": float(p), "n": int(len(pair))}


def _gini(responsibility: pd.Series, impact: pd.Series) -> float | None:
    """The coupling stage's inequality coefficient, on the finite pairs."""
    pair = pd.concat(
        [responsibility.rename("r"), impact.rename("i")], axis=1
    ).dropna()
    if len(pair) < _MIN_PAIRS:
        return None
    return float(inequality_coefficient(pair["r"].to_numpy(), pair["i"].to_numpy()))


def compute_era5_validation(
    era5_trends: pd.DataFrame,
    inequality: pd.DataFrame,
    iso_by_owid: pd.Series,
) -> dict:
    """Country-level ERA5 cross-check metrics (pure; no I/O, no grids).

    Bridges the ISO3-keyed ERA5 trends onto the OWID-named country inequality table
    via `iso_by_owid` (the exact same map :func:`src.emissions.build_inequality_analysis`
    uses for the Berkeley-area merge), then computes rank agreement and the
    side-by-side coupling reproduction.

    Args:
        era5_trends: from :func:`src.era5_weighting.era5_area_weighted_country_trends`
            (columns ``iso3`` / ``trend_c_per_decade_era5_area`` / coverage).
        inequality: ``country_inequality.parquet`` (station, responsibility and
            Berkeley-area columns, keyed on ``owid_country``).
        iso_by_owid: OWID country-name -> ISO3 map.

    Returns:
        A JSON-ready dict (rounded by the caller) with ``coverage``,
        ``rank_agreement``, ``coupling_full`` and ``coupling_common`` blocks.
    """
    work = inequality.copy()
    work[ISO3_COL] = work[OWID_COL].map(iso_by_owid)
    merged = work.merge(
        era5_trends[[ISO3_COL, ERA5_AREA_COL]], on=ISO3_COL, how="left"
    )

    station = merged[STATION_COL]
    resp = merged[RESPONSIBILITY_COL]
    berk = merged[BERKELEY_AREA_COL]
    era5 = merged[ERA5_AREA_COL]

    # Apples-to-apples set: every lens defined, so the collapse is read off identical
    # countries (this is how the collapse was shown to be coverage-independent).
    common = station.notna() & resp.notna() & berk.notna() & era5.notna()
    cm = merged.loc[common]

    def _lens_block(impact_col: str) -> dict:
        return {
            "spearman_vs_responsibility": _spearman(
                cm[impact_col], cm[RESPONSIBILITY_COL]
            ),
            "gini": _gini(cm[RESPONSIBILITY_COL], cm[impact_col]),
        }

    return {
        "reference": {
            "berkeley_global_land_trend_c_per_decade": BERKELEY_GLOBAL_LAND_REFERENCE,
        },
        "coverage": {
            "n_inequality_countries": int(len(inequality)),
            "n_era5_countries": int(len(era5_trends)),
            "n_era5_joined": int(era5.notna().sum()),
            "n_common_all_lenses": int(common.sum()),
        },
        "rank_agreement": {
            "era5_area_vs_berkeley_area": _spearman(era5, berk),
            "era5_area_vs_station": _spearman(era5, station),
        },
        # Each lens vs responsibility on its own maximal coverage.
        "coupling_full": {
            "station_vs_responsibility": _spearman(station, resp),
            "berkeley_area_vs_responsibility": _spearman(berk, resp),
            "era5_area_vs_responsibility": _spearman(era5, resp),
        },
        # The headline side-by-side: all three lenses on the one common set.
        "coupling_common": {
            "n": int(common.sum()),
            "station": _lens_block(STATION_COL),
            "berkeley_area": _lens_block(BERKELEY_AREA_COL),
            "era5_area": _lens_block(ERA5_AREA_COL),
        },
    }


def build_era5_validation(
    era5_grid_path=ERA5_GRID_PATH,
    gpw_path=GPW_PATH,
    inequality_path=DEFAULT_INEQUALITY_PATH,
    co2_path=OWID_CO2_PATH,
    berkeley_grid_path=BERKELEY_GRID_PATH,
    lookup_path=GPW_NATID_LOOKUP_PATH,
    trends_path=DEFAULT_ERA5_TRENDS_PATH,
    summary_path=DEFAULT_ERA5_SUMMARY_PATH,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    write: bool = True,
) -> dict:
    """Compute ERA5 area trends, run the cross-check, write the artifacts.

    Best-effort/null: when the (gitignored, ~200 MB) ERA5 grid is absent the
    cross-check is skipped and a ``{"available": False}`` summary is written, so a
    no-grid build still succeeds -- mirroring the population/area null paths in
    :func:`src.emissions.build_inequality_analysis`.

    Returns:
        Dict with ``available`` and, when available, ``summary`` / ``era5_trends``
        / ``trends_path`` / ``summary_path``.
    """
    if not era5_grid_path.exists():
        logger.warning(
            "ERA5 grid absent (%s); cross-check skipped -- run scripts/fetch_era5.py "
            "to enable it",
            era5_grid_path,
        )
        summary = {"available": False, "reason": f"ERA5 grid absent: {era5_grid_path}"}
        if write:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return {"available": False, "summary": summary, "summary_path": summary_path}

    inequality = pd.read_parquet(inequality_path)
    owid = load_owid_co2(co2_path)
    iso_by_owid = (
        owid.dropna(subset=["iso_code"])
        .drop_duplicates("country")
        .set_index("country")["iso_code"]
    )
    # One pass over the ERA5 grid serves both the country table and the land mean.
    mask, lats, slopes = era5_cell_slopes(
        era5_grid_path, gpw_path, start=start, end=end, lookup_path=lookup_path
    )
    era5_trends = reduce_era5_slopes(mask, lats, slopes)
    summary = compute_era5_validation(era5_trends, inequality, iso_by_owid)
    summary["world_land_mean"] = {
        "era5_area": land_mean_from_slopes(lats, slopes),
        "berkeley_area": (
            world_land_mean(
                berkeley_grid_path, gpw_path=gpw_path,
                start=start, end=end, lookup_path=lookup_path,
            )
            if berkeley_grid_path.exists()
            else None
        ),
        "berkeley_reference": BERKELEY_GLOBAL_LAND_REFERENCE,
    }
    summary["available"] = True

    if write:
        write_typed_parquet(
            era5_trends, trends_path, ERA5_TRENDS_SCHEMA, order_by=(ISO3_COL,)
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(round_floats(summary), indent=2) + "\n", encoding="utf-8"
        )
        logger.info("wrote %s and %s", trends_path, summary_path)
    return {
        "available": True,
        "summary": summary,
        "era5_trends": era5_trends,
        "trends_path": trends_path,
        "summary_path": summary_path,
    }


def main() -> None:
    """Run the ERA5 cross-check and print the headline reproduction line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = build_era5_validation()
    if not out["available"]:
        print(out["summary"]["reason"])
        return
    s = out["summary"]
    wl = s["world_land_mean"]
    print(
        f"ERA5 world-land mean {wl['era5_area']:.4f} "
        f"(Berkeley {wl['berkeley_area']}, ref ~{wl['berkeley_reference']})"
    )
    cc = s["coupling_common"]
    print(f"common set n={cc['n']}  (lens vs responsibility, Spearman rho | Gini):")
    for lens in ("station", "berkeley_area", "era5_area"):
        rho = cc[lens]["spearman_vs_responsibility"]["rho"]
        print(f"  {lens:14s} rho={rho}  gini={cc[lens]['gini']}")


if __name__ == "__main__":
    main()
