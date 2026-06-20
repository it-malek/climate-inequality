"""Tests for src.projections -- the PCS v1 identity binding resolver (Layer 1 emit)."""

from __future__ import annotations

import pandas as pd
import pytest

from src import pcs
from src.projections import (
    PCS_V1_BINDING,
    PROJECTIONS_COLUMNS,
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
