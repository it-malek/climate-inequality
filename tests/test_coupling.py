"""Tests for src.coupling -- the Layer 3 deterministic projection comparator.

Encodes the §9 acceptance gate: closure over the PCS projections, the closed
operator set, byte-stable schema-stable outputs, and a summary that carries only
the permitted fields (no interpretation).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.coupling import (
    COUPLING_CONSUMPTION_SCHEMA,
    COUPLING_SCHEMA,
    WIDE_ADMISSIBLE,
    CouplingResult,
    compute_consumption_coupling,
    compute_coupling,
    consumption_summary_payload,
    summary_payload,
    validate_projection_frame,
)
from src.data_io import write_typed_parquet

ALLOWED_SUMMARY_KEYS = {
    "spearman_rho",
    "n_high_impact_low_responsibility",
    "inequality_coefficient",
    "top_suffer_least_cause",
    "top_cause_least_suffer",
}


def make_projections(n: int = 12, seed: int = 0) -> pd.DataFrame:
    """Synthetic projection table: exactly Country + the two PCS projections."""
    rng = np.random.default_rng(seed)
    responsibility = np.abs(rng.normal(10.0, 5.0, n)) + 0.1
    impact = np.abs(rng.normal(0.15, 0.05, n)) + 0.01
    return pd.DataFrame(
        {
            "Country": [f"C{i}" for i in range(n)],
            "responsibility_index_v1": responsibility,
            "impact_index_v1": impact,
        }
    )


class TestClosureGuard:
    def test_rejects_non_pcs_column(self):
        frame = make_projections().assign(continent="X")
        with pytest.raises(ValueError, match="non-PCS"):
            validate_projection_frame(frame.columns)

    def test_rejects_missing_column(self):
        frame = make_projections().drop(columns=["impact_index_v1"])
        with pytest.raises(ValueError, match="missing"):
            validate_projection_frame(frame.columns)

    def test_compute_rejects_leaky_frame(self):
        frame = make_projections().assign(population=1)
        with pytest.raises(ValueError, match="non-PCS"):
            compute_coupling(frame)


class TestOperators:
    def test_rank_gap_sums_to_zero(self):
        table, _ = compute_coupling(make_projections())
        assert int(table["rank_gap"].sum()) == 0

    def test_z_gap_zero_mean(self):
        table, _ = compute_coupling(make_projections())
        assert table["z_gap"].mean() == pytest.approx(0.0, abs=1e-9)

    def test_z_gap_is_z_impact_minus_z_responsibility(self):
        proj = make_projections()
        table, _ = compute_coupling(proj)

        def z(series):
            x = series.to_numpy(dtype=float)
            return (x - x.mean()) / x.std(ddof=0)

        expected = z(proj["impact_index_v1"]) - z(proj["responsibility_index_v1"])
        assert table["z_gap"].to_numpy() == pytest.approx(expected)

    def test_spearman_matches_scipy(self):
        proj = make_projections()
        _, result = compute_coupling(proj)
        expected = float(
            stats.spearmanr(proj["responsibility_index_v1"], proj["impact_index_v1"])[0]
        )
        assert result.spearman_rho == pytest.approx(expected)


class TestInequalityCoefficient:
    def test_bounded_unit_interval(self):
        _, result = compute_coupling(make_projections(n=40, seed=3))
        assert 0.0 <= result.inequality_coefficient <= 1.0

    def test_zero_when_impact_proportional_to_responsibility(self):
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        proj = pd.DataFrame(
            {
                "Country": list("ABCDE"),
                "responsibility_index_v1": r,
                "impact_index_v1": 0.02 * r,
            }
        )
        _, result = compute_coupling(proj)
        assert result.inequality_coefficient == pytest.approx(0.0, abs=1e-9)

    def test_positive_under_divergence(self):
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        proj = pd.DataFrame(
            {
                "Country": list("ABCDE"),
                "responsibility_index_v1": r,
                "impact_index_v1": r[::-1].copy(),
            }
        )
        _, result = compute_coupling(proj)
        assert result.inequality_coefficient > 0.0


class TestLeaders:
    def test_suffer_least_cause_is_highest_z_gap(self):
        table, result = compute_coupling(make_projections(n=20, seed=1))
        assert result.top_suffer_least_cause[0][0] == (
            table.loc[table["z_gap"].idxmax(), "Country"]
        )

    def test_cause_least_suffer_is_lowest_z_gap(self):
        table, result = compute_coupling(make_projections(n=20, seed=1))
        assert result.top_cause_least_suffer[0][0] == (
            table.loc[table["z_gap"].idxmin(), "Country"]
        )

    def test_count_matches_positive_z_gap(self):
        table, result = compute_coupling(make_projections(n=25, seed=2))
        assert result.n_high_impact_low_responsibility == int(
            (table["z_gap"] > 0).sum()
        )


class TestSummaryPayload:
    def test_only_allowed_fields_and_no_interpretation(self):
        _, result = compute_coupling(make_projections())
        payload = summary_payload(result)
        assert set(payload) == ALLOWED_SUMMARY_KEYS
        assert "interpretation" not in payload

    def test_byte_stable(self):
        _, result = compute_coupling(make_projections())
        first = json.dumps(summary_payload(result), indent=2) + "\n"
        second = json.dumps(summary_payload(result), indent=2) + "\n"
        assert first == second


class TestParquetSchema:
    def test_round_trips_through_typed_parquet(self, tmp_path):
        table, _ = compute_coupling(make_projections())
        path = tmp_path / "coupling.parquet"
        write_typed_parquet(table, path, COUPLING_SCHEMA, order_by=("Country",))
        loaded = pd.read_parquet(path)
        assert list(loaded.columns) == list(COUPLING_SCHEMA)
        assert str(loaded["responsibility_rank"].dtype).startswith("int")
        assert str(loaded["z_gap"].dtype) == "float64"


def make_wide_projections(n: int = 12, seed: int = 0) -> pd.DataFrame:
    """Synthetic wide v2 frame: Country + the three registered v2 projections."""
    rng = np.random.default_rng(seed)
    production = np.abs(rng.normal(10.0, 5.0, n)) + 0.1
    # Consumption reshuffles responsibility relative to production (net trade).
    consumption = production * rng.uniform(0.4, 1.8, n)
    impact = np.abs(rng.normal(0.15, 0.05, n)) + 0.01
    return pd.DataFrame(
        {
            "Country": [f"C{i}" for i in range(n)],
            "impact_index_v1": impact,
            "responsibility_index_consumption": consumption,
            "responsibility_index_production_matched": production,
        }
    )


class TestParametrizedComparator:
    def test_defaults_reproduce_v1_schema(self):
        # Default x/y reproduce the v1 responsibility-vs-impact table exactly.
        table, _ = compute_coupling(make_projections())
        assert list(table.columns) == list(COUPLING_SCHEMA)

    def test_value_columns_named_by_chosen_pair(self):
        table, _ = compute_coupling(
            make_wide_projections(),
            x_col="responsibility_index_production_matched",
            y_col="responsibility_index_consumption",
            admissible=WIDE_ADMISSIBLE,
        )
        assert "responsibility_index_production_matched" in table.columns
        assert "responsibility_index_consumption" in table.columns

        # z_gap = z(y) - z(x): y is consumption, x is production-matched.
        def z(s):
            return (s - s.mean()) / s.std(ddof=0)

        proj = make_wide_projections()
        expected = (
            z(proj["responsibility_index_consumption"])
            - z(proj["responsibility_index_production_matched"])
        )
        assert table["z_gap"].to_numpy() == pytest.approx(expected.to_numpy())

    def test_wide_admissible_allows_registered_extra_column(self):
        # The wide frame carries a third registered column unused by this pair;
        # closure must allow it (registered) but reject an unregistered one.
        proj = make_wide_projections()
        validate_projection_frame(proj.columns, admissible=WIDE_ADMISSIBLE)
        with pytest.raises(ValueError, match="non-PCS"):
            validate_projection_frame(
                proj.assign(continent="X").columns, admissible=WIDE_ADMISSIBLE
            )

    def test_unregistered_pair_rejected(self):
        with pytest.raises(ValueError, match="not a registered projection"):
            compute_coupling(
                make_wide_projections(),
                x_col="nope",
                y_col="impact_index_v1",
                admissible=WIDE_ADMISSIBLE,
            )


class TestConsumptionCoupling:
    def test_wide_table_schema(self):
        table, _, _ = compute_consumption_coupling(make_wide_projections())
        assert list(table.columns) == list(COUPLING_CONSUMPTION_SCHEMA)

    def test_rank_shift_is_consumption_minus_production_rank(self):
        table, _, _ = compute_consumption_coupling(make_wide_projections(seed=4))
        expected = table["consumption_rank"] - table["production_matched_rank"]
        assert table["prod_to_cons_rank_gap"].tolist() == expected.tolist()

    def test_both_passes_bounded_inequality(self):
        _, cons_impact, prod_cons = compute_consumption_coupling(
            make_wide_projections(n=30, seed=5)
        )
        for result in (cons_impact, prod_cons):
            assert 0.0 <= result.inequality_coefficient <= 1.0

    def test_summary_payload_structure(self):
        _, cons_impact, prod_cons = compute_consumption_coupling(make_wide_projections())
        window = {
            "n_countries": 12,
            "consumption_start_year_min": 1990,
            "consumption_start_year_median": 1990.0,
            "consumption_start_year_max": 2003,
        }
        payload = consumption_summary_payload(cons_impact, prod_cons, window)
        assert set(payload) == {
            "window",
            "consumption_vs_impact",
            "production_to_consumption_shift",
        }
        assert payload["window"]["n_countries"] == 12
        assert "interpretation" not in payload
        assert "inequality_coefficient" in payload["consumption_vs_impact"]

    def test_summary_byte_stable(self):
        _, cons_impact, prod_cons = compute_consumption_coupling(make_wide_projections())
        window = {"n_countries": 12}
        first = json.dumps(consumption_summary_payload(cons_impact, prod_cons, window))
        second = json.dumps(consumption_summary_payload(cons_impact, prod_cons, window))
        assert first == second

    def test_wide_table_round_trips_through_typed_parquet(self, tmp_path):
        table, _, _ = compute_consumption_coupling(make_wide_projections())
        path = tmp_path / "coupling_consumption.parquet"
        write_typed_parquet(table, path, COUPLING_CONSUMPTION_SCHEMA, order_by=("Country",))
        loaded = pd.read_parquet(path)
        assert list(loaded.columns) == list(COUPLING_CONSUMPTION_SCHEMA)
        assert str(loaded["prod_to_cons_rank_gap"].dtype).startswith("int")


class TestResultCheck:
    def test_check_rejects_out_of_range(self):
        bad = CouplingResult(
            spearman_rho=0.0,
            n_high_impact_low_responsibility=0,
            inequality_coefficient=1.5,
        )
        with pytest.raises(AssertionError, match="outside"):
            bad.check()
