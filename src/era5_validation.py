"""ERA5 vs Berkeley cross-check: does the area-weighted coupling collapse survive?

v1.2's headline (``src.area_weighting`` + ``src.coupling.compute_area_coupling``):
moving from *station-weighted* to **area-weighted** country warming collapses the
warming<->responsibility coupling from Spearman rho **+0.36 -> +0.01** and raises
inequality (Gini 0.563 -> 0.607). This module re-tests that on a fully independent
gridded product -- **ERA5 reanalysis** (:mod:`src.era5_weighting`) -- and reports
the three lenses side by side on one common country set:

- **world-land sanity:** ERA5's cos(lat) global-land mean should land near
  Berkeley's ~0.19 degC/decade -- an independent match validates the ERA5 ingest
  the way Berkeley's 0.1926 validated v1.2.
- **rank agreement:** Spearman rho of ERA5-area vs Berkeley-area (expected high if
  the two products agree) and vs station (expected lower).
- **coupling reproduction:** Spearman rho of each lens vs responsibility, plus the
  Gini, computed on the identical common set via the *same* operators as the L3
  comparator (:func:`src.coupling._inequality_coefficient` + scipy Spearman). If
  ERA5-area's rho is near zero like Berkeley-area's, the collapse is robust to the
  data source; if it is ~+0.36 like the station lens, the collapse was
  Berkeley-specific (an equally publishable finding).

This is a **cross-check artifact**, a sibling to :mod:`src.validation` -- NOT a new
PCS projection. ``PCS_V2`` stays frozen at six projections; the comparison reuses
the standalone coupling helpers directly rather than entering the registry-bound
comparator, so no semantic-registry growth is needed.
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
    world_land_mean,
)
from src.cleaning import DEFAULT_END, DEFAULT_START
from src.coupling import _inequality_coefficient
from src.data_io import PROCESSED_DIR, round_floats, write_typed_parquet
from src.emissions import DEFAULT_INEQUALITY_PATH, OWID_CO2_PATH, load_owid_co2
from src.era5_weighting import (
    ERA5_AREA_COL,
    ERA5_COVERAGE_COL,
    ERA5_GRID_PATH,
    era5_area_weighted_country_trends,
    era5_world_land_mean,
)
from src.population import GPW_PATH

logger = logging.getLogger(__name__)

# Committed country_inequality.parquet columns this cross-check reads (mirrors the
# stable head of src.emissions.INEQUALITY_COLUMNS).
OWID_COL = "owid_country"
STATION_COL = "trend_c_per_decade"
RESPONSIBILITY_COL = "cum_co2_t_per_capita"
BERKELEY_AREA_COL = "trend_c_per_decade_area_weighted"

# Berkeley Earth's documented 1950-2013 global *land* trend -- the anchor v1.2's
# world-land mean (0.1926) matched; ERA5's independent mean is compared to it.
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
    """Gini-style coupling coefficient via the L3 operator, on the finite pairs."""
    pair = pd.concat(
        [responsibility.rename("r"), impact.rename("i")], axis=1
    ).dropna()
    if len(pair) < _MIN_PAIRS:
        return None
    return float(_inequality_coefficient(pair["r"].to_numpy(), pair["i"].to_numpy()))


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
    # countries (the v1.2 finding was confirmed coverage-independent this way).
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
            "v1_2_station_vs_area_spearman": [0.359, 0.011],
            "v1_2_station_vs_area_gini": [0.563, 0.607],
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
    era5_trends = era5_area_weighted_country_trends(
        era5_grid_path, gpw_path, start=start, end=end, lookup_path=lookup_path
    )
    summary = compute_era5_validation(era5_trends, inequality, iso_by_owid)
    summary["world_land_mean"] = {
        "era5_area": era5_world_land_mean(
            era5_grid_path, gpw_path, start=start, end=end, lookup_path=lookup_path
        ),
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
