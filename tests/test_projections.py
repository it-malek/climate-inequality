"""Tests for src.projections -- the PCS v1 identity binding resolver (Layer 1 emit)."""

from __future__ import annotations

import pandas as pd
import pytest

import numpy as np

from src import pcs
from src.projections import (
    PCS_V1_BINDING,
    PCS_V2_BINDING,
    PROJECTIONS_COLUMNS,
    PROJECTIONS_CONSUMPTION_COLUMNS,
    consumption_window,
    resolve_consumption_projections,
    resolve_projections,
)


def make_country_table() -> pd.DataFrame:
    """Upstream country table with leakage columns the resolver must drop."""
    return pd.DataFrame(
        {
            "Country": ["A", "B", "C"],
            "continent": ["Africa", "Europe", "Asia"],
            "population": [1_000, 2_000, 3_000],
            "cum_co2_t_per_capita": [5.0, 50.0, 0.5],
            "trend_c_per_decade": [0.10, 0.20, 0.15],
            "consumption_start_year": [1990, 1995, np.nan],
            "cum_consumption_t_per_capita": [4.0, 60.0, np.nan],
            "cum_co2_window_t_per_capita": [3.0, 40.0, np.nan],
            "extra": [1, 2, 3],
        }
    )


class TestBinding:
    def test_binding_names_match_registry(self):
        assert set(PCS_V1_BINDING) == set(pcs.PROJECTION_NAMES)

    def test_emits_exactly_pcs_columns_no_leakage(self):
        out = resolve_projections(make_country_table())
        assert tuple(out.columns) == PROJECTIONS_COLUMNS
        assert set(out.columns) == {
            "Country", "responsibility_index_v1", "impact_index_v1",
        }

    def test_identity_binding_no_transformation(self):
        src = make_country_table()
        out = resolve_projections(src)
        assert out["responsibility_index_v1"].tolist() == (
            src["cum_co2_t_per_capita"].tolist()
        )
        assert out["impact_index_v1"].tolist() == src["trend_c_per_decade"].tolist()

    def test_missing_source_column_raises(self):
        bad = make_country_table().drop(columns=["trend_c_per_decade"])
        with pytest.raises(ValueError, match="trend_c_per_decade"):
            resolve_projections(bad)


class TestConsumptionBinding:
    def test_binding_names_match_v2_registry(self):
        assert set(PCS_V2_BINDING) == set(pcs.PROJECTION_NAMES_V2)

    def test_emits_exactly_wide_columns_no_leakage(self):
        out = resolve_consumption_projections(make_country_table())
        assert tuple(out.columns) == PROJECTIONS_CONSUMPTION_COLUMNS
        assert set(out.columns) == {
            "Country",
            "impact_index_v1",
            "responsibility_index_consumption",
            "responsibility_index_production_matched",
        }

    def test_identity_binding_no_transformation(self):
        src = make_country_table()
        out = resolve_consumption_projections(src).set_index("Country")
        # Country C has NULL consumption and is dropped; A and B survive.
        assert out.index.tolist() == ["A", "B"]
        assert out.loc["A", "responsibility_index_consumption"] == 4.0
        assert out.loc["B", "responsibility_index_production_matched"] == 40.0
        assert out.loc["A", "impact_index_v1"] == 0.10

    def test_drops_countries_without_window(self, caplog):
        with caplog.at_level("INFO", logger="src.projections"):
            out = resolve_consumption_projections(make_country_table())
        assert "C" not in out["Country"].tolist()
        assert len(out) == 2

    def test_missing_source_column_raises(self):
        bad = make_country_table().drop(columns=["cum_consumption_t_per_capita"])
        with pytest.raises(ValueError, match="cum_consumption_t_per_capita"):
            resolve_consumption_projections(bad)


class TestConsumptionWindow:
    def test_window_stats_over_covered_countries(self):
        window = consumption_window(make_country_table())
        # Only A (1990) and B (1995) have a start year; C is NaN.
        assert window["n_countries"] == 2
        assert window["consumption_start_year_min"] == 1990
        assert window["consumption_start_year_max"] == 1995
        assert window["consumption_start_year_median"] == pytest.approx(1992.5)

    def test_empty_is_safe(self):
        empty = make_country_table().assign(consumption_start_year=np.nan)
        window = consumption_window(empty)
        assert window["n_countries"] == 0
        assert window["consumption_start_year_min"] is None
