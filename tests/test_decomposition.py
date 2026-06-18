"""Tests for the group-level LMG/Shapley warming-variance decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.decomposition import (
    aggregate_city_features_to_country,
    build_country_design,
    group_lmg_shares,
    summary_payload,
)
from src.feature_schema import INTERPRETATION_NOTE, SCHEMA_V1, feature_names


def make_country_design(
    n=120, seed=0, beta_emissions=0.0, beta_lat=0.03, noise_sd=0.01, extra=None
):
    """Synthetic schema-named country design with planted axis effects.

    Emissions (``cum_co2_per_capita``) and geography (``abs_latitude``,
    ``spatial_block``) are independent, so their LMG shares track their planted
    signal strength.
    """
    rng = np.random.default_rng(seed)
    log_em = rng.uniform(-1.0, 4.0, n)
    abs_lat = rng.uniform(0.0, 70.0, n)
    blocks = ["Africa", "Europe", "Asia", "Americas"]
    spatial_block = rng.choice(blocks, n)
    block_offset = {b: o for b, o in zip(blocks, [0.0, 0.02, 0.01, -0.01])}
    y = (
        0.10
        + beta_emissions * log_em
        + beta_lat * (abs_lat / 70.0)
        + np.array([block_offset[b] for b in spatial_block])
        + rng.normal(0.0, noise_sd, n)
    )
    df = pd.DataFrame(
        {
            "Country": [f"C{i}" for i in range(n)],
            "warming_trend": y,
            "cum_co2_per_capita": 10.0**log_em,  # strictly positive
            "abs_latitude": abs_lat,
            "spatial_block": spatial_block,
        }
    )
    if extra is not None:
        for name, vals in extra.items():
            df[name] = vals
    return df


def test_shares_plus_residual_sum_to_one():
    result = group_lmg_shares(make_country_design())
    result.check_sums_to_one()
    assert sum(result.shares.values()) == pytest.approx(result.total_r2)
    assert result.residual_share == pytest.approx(1.0 - result.total_r2)


def test_shares_are_non_negative_and_in_unit_interval():
    result = group_lmg_shares(make_country_design(seed=3))
    for key, share in result.shares.items():
        assert share >= -1e-9, key
    assert 0.0 <= result.total_r2 <= 1.0


def test_geography_dominates_when_it_carries_the_signal():
    # Geography drives y; emissions is independent noise.
    result = group_lmg_shares(
        make_country_design(beta_emissions=0.0, beta_lat=0.05, noise_sd=0.005)
    )
    assert "geography" in result.shares
    assert "emissions" in result.shares
    assert result.shares["geography"] > result.shares["emissions"]


def test_emissions_dominates_when_it_carries_the_signal():
    result = group_lmg_shares(
        make_country_design(beta_emissions=0.05, beta_lat=0.0, noise_sd=0.005)
    )
    assert result.shares["emissions"] > result.shares["geography"]


def test_single_group_share_equals_total_r2():
    df = pd.DataFrame(
        {
            "Country": ["A", "B", "C", "D", "E"],
            "warming_trend": [0.10, 0.15, 0.20, 0.25, 0.30],
            "abs_latitude": [5.0, 20.0, 35.0, 50.0, 65.0],
        }
    )
    result = group_lmg_shares(df)
    assert list(result.shares) == ["geography"]
    assert result.shares["geography"] == pytest.approx(result.total_r2)


def test_determinism():
    df = make_country_design(seed=7)
    a = group_lmg_shares(df)
    b = group_lmg_shares(df)
    assert a.shares == b.shares
    assert a.total_r2 == b.total_r2


def test_summary_payload_carries_interpretation():
    result = group_lmg_shares(make_country_design())
    payload = summary_payload(result)
    assert payload["interpretation"] == INTERPRETATION_NOTE
    assert payload["shares"] == result.shares


def test_out_of_schema_column_rejected():
    df = make_country_design()
    df["gdp_growth"] = 1.0  # not a schema feature
    with pytest.raises(ValueError, match="not in schema"):
        group_lmg_shares(df)


def test_proposed_feature_present_but_not_used():
    # gdp_per_capita is a schema feature (so it passes validation) but is
    # status="proposed", so it must not enter the live decomposition.
    df = make_country_design(extra={"gdp_per_capita": np.linspace(1e3, 5e4, 120)})
    result = group_lmg_shares(df)
    assert "socioeconomic" not in result.group_features
    assert "gdp_per_capita" not in {
        f for feats in result.group_features.values() for f in feats
    }


def test_categorical_block_contributes():
    # spatial_block (categorical) is the only signal -> geography share > 0.
    df = make_country_design(beta_emissions=0.0, beta_lat=0.0, noise_sd=0.002)
    result = group_lmg_shares(df)
    assert result.shares["geography"] > 0.0


class TestBuildCountryDesign:
    def _frames(self):
        inequality = pd.DataFrame(
            {
                "Country": ["Kenya", "Norway", "Brazil"],
                "owid_country": ["Kenya", "Norway", "Brazil"],
                "continent": ["Africa", "Europe", "South America"],
                "n_cities": [4, 3, 5],
                "trend_c_per_decade": [0.10, 0.30, 0.08],
                "cumulative_co2_mt": [50.0, 400.0, 600.0],
                "population": [50_000_000, 5_000_000, 210_000_000],
                "cum_co2_t_per_capita": [1.0, 80.0, 3.0],
            }
        )
        city_features = pd.DataFrame(
            {
                "Country": ["Kenya", "Kenya", "Norway", "Brazil", "Brazil"],
                "abs_latitude": [1.0, 3.0, 65.0, 10.0, 20.0],
                "elevation_m": [1600.0, 1700.0, 50.0, 800.0, 400.0],
                "coast_km": [400.0, 420.0, 30.0, 100.0, 1500.0],
                "station_density": [2.0, 3.0, 1.0, 5.0, 4.0],
                "koppen": ["A", "A", "D", "A", "C"],
                "hemisphere": ["N", "N", "N", "S", "S"],
            }
        )
        income = pd.DataFrame(
            {
                "owid_country": ["Kenya", "Norway", "Brazil"],
                "income_group": ["Low income", "High income", "Upper-middle income"],
            }
        )
        return inequality, city_features, income

    def test_aggregation_to_country(self):
        _, city_features, _ = self._frames()
        agg = aggregate_city_features_to_country(city_features)
        kenya = agg.set_index("Country").loc["Kenya"]
        assert kenya["abs_latitude"] == pytest.approx(2.0)
        assert kenya["climate_zone"] == "A"  # modal Koeppen class

    def test_design_has_only_schema_columns(self):
        inequality, city_features, income = self._frames()
        design = build_country_design(inequality, city_features, income)
        every_feature = set(feature_names(SCHEMA_V1, status=None))
        allowed = every_feature | {"Country", "warming_trend"}
        assert set(design.columns) <= allowed
        # Maps applied correctly.
        norway = design.set_index("Country").loc["Norway"]
        assert norway["warming_trend"] == pytest.approx(0.30)
        assert norway["cum_co2_per_capita"] == pytest.approx(80.0)
        assert norway["spatial_block"] == "Europe"
        assert norway["income_group"] == "High income"

    def test_design_decomposes(self):
        inequality, city_features, income = self._frames()
        design = build_country_design(inequality, city_features, income)
        result = group_lmg_shares(design)
        result.check_sums_to_one()
        assert result.n == 3
