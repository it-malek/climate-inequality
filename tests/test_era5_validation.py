"""Tests for src.era5_validation -- the ERA5 vs Berkeley cross-check comparator.

Pure synthetic frames (no grids, no network): ``compute_era5_validation`` is
exercised against a hand-built inequality table + ERA5 trends + ISO bridge, with
expectations recomputed via the same operators it uses (scipy Spearman and the
reused :func:`src.coupling._inequality_coefficient`). The best-effort null path is
checked with an absent grid path.
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.coupling import _inequality_coefficient
from src.era5_validation import (
    BERKELEY_AREA_COL,
    ERA5_AREA_COL,
    OWID_COL,
    RESPONSIBILITY_COL,
    STATION_COL,
    build_era5_validation,
    compute_era5_validation,
)
from src.era5_weighting import ERA5_COVERAGE_COL
from src.area_weighting import ISO3_COL


@pytest.fixture
def scenario():
    """5 OWID countries; ERA5 covers 4 of them (E is missing) -> common set = 4."""
    inequality = pd.DataFrame({
        OWID_COL: ["A", "B", "C", "D", "E"],
        STATION_COL: [0.10, 0.20, 0.30, 0.40, 0.50],
        RESPONSIBILITY_COL: [1.0, 2.0, 4.0, 8.0, 16.0],
        BERKELEY_AREA_COL: [0.22, 0.18, 0.31, 0.27, 0.40],
    })
    iso_by_owid = pd.Series(
        {"A": "AAA", "B": "BBB", "C": "CCC", "D": "DDD", "E": "EEE"}
    )
    era5_trends = pd.DataFrame({
        ISO3_COL: ["AAA", "BBB", "CCC", "DDD"],
        ERA5_AREA_COL: [0.25, 0.19, 0.33, 0.24],
        ERA5_COVERAGE_COL: [1.0, 1.0, 0.95, 0.9],
    })
    return inequality, iso_by_owid, era5_trends


class TestComputeEra5Validation:
    def test_coverage_counts(self, scenario):
        inequality, iso_by_owid, era5 = scenario
        out = compute_era5_validation(era5, inequality, iso_by_owid)
        assert out["coverage"] == {
            "n_inequality_countries": 5,
            "n_era5_countries": 4,
            "n_era5_joined": 4,       # A..D matched via the ISO bridge
            "n_common_all_lenses": 4,  # E has no ERA5 trend
        }

    def test_rank_agreement_matches_scipy(self, scenario):
        inequality, iso_by_owid, era5 = scenario
        out = compute_era5_validation(era5, inequality, iso_by_owid)
        # ERA5 vs Berkeley over the 4 joined countries (A..D).
        e = [0.25, 0.19, 0.33, 0.24]
        b = [0.22, 0.18, 0.31, 0.27]
        rho, _ = stats.spearmanr(e, b)
        block = out["rank_agreement"]["era5_area_vs_berkeley_area"]
        assert block["rho"] == pytest.approx(rho)
        assert block["n"] == 4

    def test_coupling_common_reproduces_operators(self, scenario):
        inequality, iso_by_owid, era5 = scenario
        out = compute_era5_validation(era5, inequality, iso_by_owid)
        # Common set is A..D; responsibility is the same for every lens.
        resp = np.array([1.0, 2.0, 4.0, 8.0])
        era5_vals = np.array([0.25, 0.19, 0.33, 0.24])
        rho, _ = stats.spearmanr(era5_vals, resp)
        gini = _inequality_coefficient(resp, era5_vals)
        era5_block = out["coupling_common"]["era5_area"]
        assert out["coupling_common"]["n"] == 4
        assert era5_block["spearman_vs_responsibility"]["rho"] == pytest.approx(rho)
        assert era5_block["gini"] == pytest.approx(gini)
        # Station lens reported on the same common set for the side-by-side.
        assert "station" in out["coupling_common"]
        assert "berkeley_area" in out["coupling_common"]

    def test_full_coverage_uses_max_overlap(self, scenario):
        inequality, iso_by_owid, era5 = scenario
        out = compute_era5_validation(era5, inequality, iso_by_owid)
        # Station-vs-responsibility uses all 5 countries (no ERA5 needed).
        assert out["coupling_full"]["station_vs_responsibility"]["n"] == 5
        # ERA5-vs-responsibility uses the 4 ERA5-joined countries.
        assert out["coupling_full"]["era5_area_vs_responsibility"]["n"] == 4


class TestBuildEra5ValidationAbsentGrid:
    def test_absent_grid_writes_unavailable_summary(self, tmp_path):
        summary_path = tmp_path / "era5_validation_summary.json"
        out = build_era5_validation(
            era5_grid_path=tmp_path / "nope.nc", summary_path=summary_path
        )
        assert out["available"] is False
        written = json.loads(summary_path.read_text())
        assert written["available"] is False
        assert "ERA5 grid absent" in written["reason"]
