"""Tests for the descriptive warming-inequality metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_schema import INTERPRETATION_NOTE
from src.inequality import (
    coefficient_of_variation,
    country_warming_inequality,
    gini,
    lorenz_points,
    summarize_inequality,
    summary_payload,
    theil_decomposition,
    theil_t,
    variance,
    weighted_mean,
)


class TestGini:
    def test_perfect_equality_is_zero(self):
        assert gini([0.2, 0.2, 0.2, 0.2]) == pytest.approx(0.0)

    def test_known_values(self):
        # mean abs diff of {1,2,3,4} = 1.25, mean = 2.5 -> G = 0.25.
        assert gini([1, 2, 3, 4]) == pytest.approx(0.25)
        # {0, 1}: G = 0.5.
        assert gini([0.0, 1.0]) == pytest.approx(0.5)

    def test_weights_replicate_units(self):
        # Weighting [1, 2] by (2, 1) must equal the unweighted [1, 1, 2].
        assert gini([1.0, 2.0], weights=[2, 1]) == pytest.approx(gini([1.0, 1.0, 2.0]))

    def test_negative_values_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            gini([-0.1, 0.2])


class TestTheil:
    def test_perfect_equality_is_zero(self):
        assert theil_t([3.0, 3.0, 3.0]) == pytest.approx(0.0)

    def test_known_value(self):
        assert theil_t([1, 2, 3, 4]) == pytest.approx(0.106440, abs=1e-5)

    def test_non_positive_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            theil_t([0.0, 1.0])

    def test_between_plus_within_equals_total(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.05, 0.35, 40)
        g = np.array(["A", "B", "C", "D"] * 10)
        decomp = theil_decomposition(x, g)
        assert decomp.between + decomp.within == pytest.approx(decomp.total)
        assert decomp.total == pytest.approx(theil_t(x))
        assert sum(decomp.within_by_group.values()) == pytest.approx(decomp.within)

    def test_between_share_pure_between(self):
        # Within each group all values equal -> within = 0, all inequality between.
        x = np.array([0.1, 0.1, 0.3, 0.3])
        g = np.array(["low", "low", "high", "high"])
        decomp = theil_decomposition(x, g)
        assert decomp.within == pytest.approx(0.0)
        assert decomp.between_share == pytest.approx(1.0)


class TestSpread:
    def test_variance_and_cv(self):
        x = [0.1, 0.2, 0.3]
        assert variance(x) == pytest.approx(np.var(x))  # population (ddof=0)
        assert coefficient_of_variation(x) == pytest.approx(np.std(x) / np.mean(x))

    def test_weighted_mean(self):
        assert weighted_mean([1.0, 3.0], weights=[3, 1]) == pytest.approx(1.5)


class TestLorenz:
    def test_endpoints_and_monotone(self):
        pts = lorenz_points([0.3, 0.1, 0.2, 0.4])
        assert pts.iloc[0]["cum_unit_share"] == pytest.approx(0.0)
        assert pts.iloc[0]["cum_warming_share"] == pytest.approx(0.0)
        assert pts.iloc[-1]["cum_unit_share"] == pytest.approx(1.0)
        assert pts.iloc[-1]["cum_warming_share"] == pytest.approx(1.0)
        # Lorenz curve lies on/under the diagonal and is non-decreasing.
        assert (pts["cum_warming_share"].diff().dropna() >= -1e-12).all()
        assert (pts["cum_warming_share"] <= pts["cum_unit_share"] + 1e-12).all()


class TestSummary:
    def test_summary_without_groups_omits_theil_split(self):
        s = summarize_inequality([0.1, 0.2, 0.3])
        assert s.n == 3
        assert s.theil_between is None
        assert s.theil_within_by_group == {}

    def test_summary_with_groups_populates_split(self):
        x = [0.1, 0.12, 0.3, 0.33]
        g = ["A", "A", "B", "B"]
        s = summarize_inequality(x, groups=g)
        assert s.theil_between is not None
        assert s.theil_between + s.theil_within == pytest.approx(s.theil_t)

    def test_country_warming_inequality_reads_table(self):
        table = pd.DataFrame(
            {
                "Country": ["A", "B", "C", "D"],
                "continent": ["Africa", "Africa", "Europe", "Europe"],
                "trend_c_per_decade": [0.10, 0.15, 0.20, 0.25],
            }
        )
        s = country_warming_inequality(table)
        assert s.n == 4
        assert s.gini == pytest.approx(gini([0.10, 0.15, 0.20, 0.25]))
        assert s.theil_between_share is not None

    def test_country_warming_inequality_drops_nulls(self):
        table = pd.DataFrame(
            {
                "Country": ["A", "B", "C"],
                "continent": ["Africa", "Europe", "Europe"],
                "trend_c_per_decade": [0.10, np.nan, 0.20],
            }
        )
        s = country_warming_inequality(table, group_col=None)
        assert s.n == 2

    def test_summary_payload_carries_interpretation(self):
        payload = summary_payload(summarize_inequality([0.1, 0.2, 0.3]))
        assert payload["interpretation"] == INTERPRETATION_NOTE
        assert payload["gini"] == pytest.approx(gini([0.1, 0.2, 0.3]))
