"""Tests for src.forcings -- assembly of the Layer 1 ``forcings.parquet`` table.

Synthetic fixtures only (no network, no real data), per the reproducibility rule.
The raw frames mimic each source's on-disk shape -- GISTEMP ``Year``/``J-D`` with
``***`` gaps, Berkeley monthly land+ocean + uncertainty, the ERF aggregates' mid-year
index, and NOAA's seasonal ONI ``ANOM`` -- so the pure parsers, the inner-join span,
the uncertainty fallback, the fail-loud invariants and parquet byte-stability are all
exercised end to end without touching the wire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_io import write_typed_parquet
from src.forcings import (
    ESTIMATOR_COLUMNS,
    FORCINGS_COLUMNS,
    FORCINGS_SCHEMA,
    ForcingsCrossCheckError,
    ForcingsResult,
    assemble_forcings,
    compute_forcings,
    parse_berkeley,
    parse_erf,
    parse_gistemp,
    parse_oni,
)

_SEASONS = (
    "DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
    "JJA", "JAS", "ASO", "SON", "OND", "NDJ",
)


def _anomaly(year: int) -> float:
    """A smooth synthetic warming trend used for both temperature series."""
    return 0.012 * (year - 1962)


def _gistemp_raw(years: range) -> pd.DataFrame:
    return pd.DataFrame({"Year": list(years), "J-D": [round(_anomaly(y), 3) for y in years]})


def _berkeley_raw(years: range, unc: float = 0.05, anomaly=_anomaly) -> pd.DataFrame:
    rows = []
    for y in years:
        for m in range(1, 13):
            rows.append(
                {
                    "dt": f"{y}-{m:02d}-01",
                    "LandAndOceanAverageTemperature": 8.0 + anomaly(y),
                    "LandAndOceanAverageTemperatureUncertainty": unc,
                }
            )
    return pd.DataFrame(rows)


def _erf_raw(years: range) -> pd.DataFrame:
    rows = []
    for y in years:
        co2 = 0.02 * (y - 1900)
        rows.append(
            {
                "frac": y + 0.5,
                "CO2": co2,
                "CH4": 0.2,
                "N2O": 0.1,
                "aerosol": -0.5,
                "volcanic": 0.0,
                "solar": 0.05,
                "total": co2 - 0.5 + 0.2 + 0.1 + 0.05,
                "minor": 0.01,  # extra column the parser must ignore
            }
        )
    return pd.DataFrame(rows)


def _oni_raw(years: range) -> pd.DataFrame:
    rows = []
    for y in years:
        for i, seas in enumerate(_SEASONS):
            rows.append({"SEAS": seas, "YR": y, "TOTAL": 26.0, "ANOM": 0.1 * np.sin(i)})
    return pd.DataFrame(rows)


def make_raw_sources(
    gistemp_years: range = range(1945, 2025),
    berkeley_years: range = range(1945, 2016),
    erf_years: range = range(1930, 2025),
    oni_years: range = range(1945, 2025),
):
    """The four raw source frames; defaults give a contiguous 1945-2024 span."""
    return (
        _gistemp_raw(gistemp_years),
        _berkeley_raw(berkeley_years),
        _erf_raw(erf_years),
        _oni_raw(oni_years),
    )


class TestParsers:
    def test_gistemp_drops_missing_and_recentres_baseline(self):
        raw = _gistemp_raw(range(1945, 2001))
        raw["J-D"] = raw["J-D"].astype(object)  # real GISTEMP reads as object (has ***)
        raw.loc[raw["Year"] == 1990, "J-D"] = "***"  # missing month -> coerced out
        out = parse_gistemp(raw)
        assert 1990 not in out["year"].to_numpy()
        baseline = out[(out["year"] >= 1951) & (out["year"] <= 1980)]
        assert baseline["temp_anomaly"].mean() == pytest.approx(0.0, abs=1e-12)

    def test_berkeley_rms_uncertainty_and_full_year_filter(self):
        raw = _berkeley_raw(range(1951, 1990), unc=0.05)
        # drop one month from 1985 so it is no longer a complete year
        raw = raw[~((raw["dt"].str[:4] == "1985") & (raw["dt"].str[5:7] == "12"))]
        out = parse_berkeley(raw)
        assert 1985 not in out["year"].to_numpy()
        assert out["temp_uncertainty"].to_numpy() == pytest.approx(0.05)

    def test_erf_floors_year_and_renames(self):
        out = parse_erf(_erf_raw(range(1980, 1983)))
        assert list(out["year"]) == [1980, 1981, 1982]
        assert set(out.columns) == {
            "year", "erf_co2", "erf_ch4", "erf_n2o",
            "erf_aerosol", "erf_volcanic", "erf_solar", "erf_total",
        }
        assert out.loc[out["year"] == 1980, "erf_co2"].iloc[0] == pytest.approx(0.02 * 80)

    def test_oni_annual_mean(self):
        out = parse_oni(_oni_raw(range(1950, 1952)))
        expected = float(np.mean([0.1 * np.sin(i) for i in range(12)]))
        assert list(out["year"]) == [1950, 1951]
        assert out["oni"].to_numpy() == pytest.approx(expected)


class TestAssemble:
    def test_inner_join_span_and_uncertainty_fallback(self):
        g, b, e, o = make_raw_sources()
        frame, n_filled = assemble_forcings(
            parse_gistemp(g), parse_berkeley(b), parse_erf(e), parse_oni(o)
        )
        assert list(frame.columns) == list(FORCINGS_COLUMNS)
        assert frame["year"].min() == 1945 and frame["year"].max() == 2024
        # Berkeley ends 2015; 2016-2024 (9 years) get the fallback uncertainty.
        assert n_filled == 9
        assert not frame["temp_uncertainty"].isna().any()
        assert frame.loc[frame["year"] >= 2016, "temp_uncertainty"].to_numpy() == pytest.approx(0.05)


class TestComputeAndResult:
    def test_schema_and_provenance(self):
        frame, result = compute_forcings(*make_raw_sources())
        assert list(frame.columns) == list(FORCINGS_COLUMNS)
        assert (result.year_min, result.year_max, result.n_years) == (1945, 2024, 80)
        assert result.n_uncertainty_filled == 9
        assert result.n_estimator_nan == 0
        assert result.cross_check_corr == pytest.approx(1.0)
        # passing compute implies check() already held
        result.check()

    def test_no_nan_in_estimator_columns(self):
        frame, _ = compute_forcings(*make_raw_sources())
        assert not frame[list(ESTIMATOR_COLUMNS)].isna().to_numpy().any()

    def test_rejects_interior_year_gap(self):
        g, b, e, o = make_raw_sources()
        g = g[g["Year"] != 2000]  # hole inside the joined span
        with pytest.raises(AssertionError, match="contiguous"):
            compute_forcings(g, b, e, o)

    def test_rejects_low_cross_check_corr(self):
        g, b, e, o = make_raw_sources()
        # Berkeley anomaly oscillates, uncorrelated with the linear GISTEMP trend.
        b = _berkeley_raw(range(1945, 2016), anomaly=lambda y: 0.3 * np.sin(y * 1.7))
        with pytest.raises(ForcingsCrossCheckError, match="correlation"):
            compute_forcings(g, b, e, o)

    def test_check_rejects_implausible_oni(self):
        bad = ForcingsResult(
            year_min=1950, year_max=2024, n_years=75,
            n_uncertainty_filled=0, cross_check_corr=0.99,
            max_abs_oni=7.0, n_estimator_nan=0,
        )
        with pytest.raises(AssertionError, match="oni"):
            bad.check()

    def test_check_rejects_nan_corr(self):
        bad = ForcingsResult(
            year_min=1950, year_max=2024, n_years=75,
            n_uncertainty_filled=0, cross_check_corr=float("nan"),
            max_abs_oni=1.0, n_estimator_nan=0,
        )
        with pytest.raises(ForcingsCrossCheckError, match="correlation"):
            bad.check()


class TestByteStability:
    def test_round_trips_typed_parquet(self, tmp_path):
        frame, _ = compute_forcings(*make_raw_sources())
        path = tmp_path / "forcings.parquet"
        write_typed_parquet(frame, path, FORCINGS_SCHEMA, order_by=("year",))
        loaded = pd.read_parquet(path)
        assert list(loaded.columns) == list(FORCINGS_SCHEMA)
        assert str(loaded["year"].dtype).startswith("int")
        assert str(loaded["temp_anomaly"].dtype) == "float64"
        assert str(loaded["oni"].dtype) == "float64"

    def test_rebuild_is_byte_identical(self, tmp_path):
        frame, _ = compute_forcings(*make_raw_sources())
        a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
        write_typed_parquet(frame, a, FORCINGS_SCHEMA, order_by=("year",))
        write_typed_parquet(frame, b, FORCINGS_SCHEMA, order_by=("year",))
        assert a.read_bytes() == b.read_bytes()
