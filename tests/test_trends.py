"""Tests for src.trends — climatology, anomalies, coverage gate, trend fits.

No real dataset and no network: each test builds a small synthetic
city_temps table in DuckDB (in-memory or tmp file), mirroring the schema
produced by data_io.load_city_temperatures, with known injected seasonal
cycles and trends.
"""

import logging

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.trends import (
    BASELINE_END,
    BASELINE_START,
    TRENDS_COLUMNS,
    build_city_trends,
    compute_anomalies,
    compute_climatology,
    fit_city_trends,
)

BASE_TEMP = 10.0
AMPLITUDE = 8.0


def seasonal_cycle(months: pd.DatetimeIndex) -> np.ndarray:
    return AMPLITUDE * np.sin(2 * np.pi * (months.month - 1) / 12)


def make_city_frame(
    city,
    country,
    lat,
    lon,
    start="1950-01-01",
    end="2013-09-01",
    slope_per_decade=0.0,
    noise_sd=0.0,
    seed=0,
):
    """Monthly temps = base + seasonal cycle + linear trend + noise."""
    months = pd.date_range(start, end, freq="MS")
    rng = np.random.default_rng(seed)
    decades = (months.year + (months.month - 0.5) / 12) / 10
    temps = (
        BASE_TEMP
        + seasonal_cycle(months)
        + slope_per_decade * (decades - decades[0])
        + rng.normal(0.0, noise_sd, len(months))
    )
    return pd.DataFrame(
        {
            "dt": months,
            "AverageTemperature": temps,
            "AverageTemperatureUncertainty": 0.1,
            "City": city,
            "Country": country,
            "Latitude": lat,
            "Longitude": lon,
        }
    )


def make_anomaly_frame(
    city,
    country,
    lat,
    lon,
    slope_per_decade,
    start="1950-01-01",
    end="2013-09-01",
    noise_sd=0.0,
    seed=0,
):
    """Deseasonalized anomalies with a known injected trend."""
    months = pd.date_range(start, end, freq="MS")
    rng = np.random.default_rng(seed)
    decades = (months.year + (months.month - 0.5) / 12) / 10
    anomaly = slope_per_decade * (decades - decades[0]) + rng.normal(
        0.0, noise_sd, len(months)
    )
    return pd.DataFrame(
        {
            "City": city,
            "Country": country,
            "Latitude": lat,
            "Longitude": lon,
            "dt": months,
            "anomaly": anomaly,
        }
    )


def make_db(con, frames):
    """Create a city_temps table matching the real ingested schema."""
    synthetic = pd.concat(frames, ignore_index=True)
    con.register("_synthetic", synthetic)
    con.execute(
        """
        CREATE OR REPLACE TABLE city_temps AS
        SELECT
            dt::DATE AS dt,
            AverageTemperature,
            AverageTemperatureUncertainty,
            City,
            Country,
            Latitude,
            Longitude
        FROM _synthetic
        """
    )
    con.unregister("_synthetic")


@pytest.fixture
def con():
    connection = duckdb.connect()
    yield connection
    connection.close()


class TestComputeClimatology:
    def test_recovers_monthly_means(self, con):
        make_db(con, [make_city_frame("A", "X", 10.0, 20.0)])
        out = compute_climatology(con)
        assert len(out) == 12
        assert out["Month"].tolist() == list(range(1, 13))
        months = pd.date_range("2000-01-01", "2000-12-01", freq="MS")
        expected = BASE_TEMP + seasonal_cycle(months)
        assert out["climatology"].to_numpy() == pytest.approx(expected, abs=1e-9)

    def test_ignores_null_temperatures(self, con):
        frame = make_city_frame("A", "X", 10.0, 20.0)
        in_baseline = frame["dt"].between(BASELINE_START, BASELINE_END)
        januaries = frame["dt"].dt.month == 1
        frame.loc[in_baseline & januaries & (frame["dt"].dt.year % 2 == 0),
                  "AverageTemperature"] = np.nan
        make_db(con, [frame])
        out = compute_climatology(con)
        january = out.loc[out["Month"] == 1, "climatology"].iloc[0]
        # Remaining (odd-year) Januaries all share the same value.
        assert january == pytest.approx(BASE_TEMP, abs=1e-9)

    def test_same_named_cities_stay_separate(self, con):
        warm = make_city_frame("Springfield", "United States", 30.0, -90.0)
        cold = make_city_frame("Springfield", "United States", 45.0, -90.0)
        cold["AverageTemperature"] -= 15.0
        make_db(con, [warm, cold])
        out = compute_climatology(con)
        assert len(out) == 24
        by_lat = out.groupby("Latitude")["climatology"].mean()
        assert by_lat[30.0] - by_lat[45.0] == pytest.approx(15.0, abs=1e-9)

    def test_respects_baseline_window(self, con):
        frame = make_city_frame("A", "X", 10.0, 20.0)
        # A +10 °C shift after the baseline must not leak into it.
        frame.loc[frame["dt"] > BASELINE_END, "AverageTemperature"] += 10.0
        make_db(con, [frame])
        out = compute_climatology(con)
        months = pd.date_range("2000-01-01", "2000-12-01", freq="MS")
        expected = BASE_TEMP + seasonal_cycle(months)
        assert out["climatology"].to_numpy() == pytest.approx(expected, abs=1e-9)

    def test_bad_table_name_raises(self, con):
        with pytest.raises(ValueError, match="table name"):
            compute_climatology(con, table="bad; DROP TABLE x")


class TestComputeAnomalies:
    def test_removes_seasonal_cycle(self, con):
        make_db(con, [make_city_frame("A", "X", 10.0, 20.0)])
        out = compute_anomalies(con)
        # Pure seasonal signal: climatology subtraction leaves ~nothing.
        assert out["anomaly"].abs().max() == pytest.approx(0.0, abs=1e-9)
        assert len(out) == len(pd.period_range("1950-01", "2013-09", freq="M"))

    def test_baseline_monthly_means_are_zero(self, con):
        frame = make_city_frame(
            "A", "X", 10.0, 20.0, slope_per_decade=0.3, noise_sd=0.2, seed=1
        )
        make_db(con, [frame])
        out = compute_anomalies(con)
        baseline = out[out["dt"].between(BASELINE_START, BASELINE_END)]
        monthly_means = baseline.groupby(baseline["dt"].dt.month)["anomaly"].mean()
        assert monthly_means.abs().max() == pytest.approx(0.0, abs=1e-9)

    def test_excludes_low_coverage_city_and_logs(self, con, caplog):
        full = make_city_frame("A", "X", 10.0, 20.0)
        sparse = make_city_frame("B", "X", 11.0, 21.0, end="1980-12-01")
        make_db(con, [full, sparse])
        with caplog.at_level(logging.INFO, logger="src.trends"):
            out = compute_anomalies(con)
        assert set(out["City"].unique()) == {"A"}
        assert "excluded 1" in caplog.text

    def test_coverage_keyed_per_location_not_city_name(self, con):
        # Same (City, Country) at two coordinates: pooled coverage would
        # pass both; per-location coverage must drop the sparse one.
        full = make_city_frame("Springfield", "United States", 30.0, -90.0)
        sparse = make_city_frame(
            "Springfield", "United States", 45.0, -90.0, end="1980-12-01"
        )
        make_db(con, [full, sparse])
        out = compute_anomalies(con)
        assert out["Latitude"].unique().tolist() == [30.0]


class TestFitCityTrends:
    def test_recovers_injected_trend(self):
        anomalies = make_anomaly_frame(
            "A", "X", 10.0, 20.0, slope_per_decade=0.15, noise_sd=0.1, seed=2
        )
        out = fit_city_trends(anomalies)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["n_obs"] == len(anomalies)
        assert row["slope_c_per_decade"] == pytest.approx(0.15, abs=0.02)
        assert row["ols_slope"] == pytest.approx(0.15, abs=0.02)
        assert row["ci_low"] < 0.15 < row["ci_high"]

    def test_zero_trend_ci_brackets_zero(self):
        anomalies = make_anomaly_frame(
            "A", "X", 10.0, 20.0, slope_per_decade=0.0, noise_sd=0.1, seed=3
        )
        out = fit_city_trends(anomalies)
        row = out.iloc[0]
        assert row["slope_c_per_decade"] == pytest.approx(0.0, abs=0.02)
        assert row["ci_low"] < 0.0 < row["ci_high"]

    def test_distinguishes_same_named_locations(self):
        flat = make_anomaly_frame(
            "Springfield", "United States", 30.0, -90.0, slope_per_decade=0.0
        )
        warming = make_anomaly_frame(
            "Springfield", "United States", 45.0, -90.0, slope_per_decade=0.3
        )
        out = fit_city_trends(pd.concat([flat, warming], ignore_index=True))
        assert len(out) == 2
        by_lat = out.set_index("Latitude")["slope_c_per_decade"]
        assert by_lat[30.0] == pytest.approx(0.0, abs=1e-9)
        assert by_lat[45.0] == pytest.approx(0.3, abs=1e-9)


class TestBuildCityTrends:
    @pytest.fixture
    def db_path(self, tmp_path):
        path = tmp_path / "climate.duckdb"
        con = duckdb.connect(str(path))
        try:
            make_db(
                con,
                [
                    make_city_frame(
                        "A", "X", 65.0, 20.0,
                        slope_per_decade=0.2, noise_sd=0.1, seed=4,
                    ),
                    make_city_frame("B", "X", 11.0, 21.0, end="1980-12-01"),
                ],
            )
        finally:
            con.close()
        return path

    @pytest.fixture
    def out_path(self, tmp_path):
        return tmp_path / "city_trends.parquet"

    def read_parquet(self, path):
        con = duckdb.connect()
        try:
            return con.execute(
                "SELECT * FROM read_parquet(?)", [str(path)]
            ).df()
        finally:
            con.close()

    def test_schema_exclusion_and_recovered_slope(self, db_path, out_path):
        trends = build_city_trends(db_path, out_path)
        assert trends.columns.tolist() == TRENDS_COLUMNS
        assert trends["City"].tolist() == ["A"]  # B fails coverage
        row = trends.iloc[0]
        assert row["slope_c_per_decade"] == pytest.approx(0.2, abs=0.02)
        assert row["coverage"] == pytest.approx(1.0)
        assert row["baseline_window"] == f"{BASELINE_START}..{BASELINE_END}"
        assert row["analysis_window"] == "1950-01-01..2013-09-01"

    def test_parquet_matches_returned_frame(self, db_path, out_path):
        trends = build_city_trends(db_path, out_path)
        on_disk = self.read_parquet(out_path)
        assert on_disk.columns.tolist() == TRENDS_COLUMNS
        assert len(on_disk) == len(trends)
        assert on_disk["slope_c_per_decade"].tolist() == pytest.approx(
            trends["slope_c_per_decade"].tolist()
        )

    def test_rerun_is_idempotent(self, db_path, out_path):
        first = build_city_trends(db_path, out_path)
        second = build_city_trends(db_path, out_path)
        on_disk = self.read_parquet(out_path)
        assert len(first) == len(second) == len(on_disk) == 1

    def test_parquet_types_are_explicit(self, db_path, out_path):
        build_city_trends(db_path, out_path)
        con = duckdb.connect()
        try:
            schema = {
                row[0]: row[1]
                for row in con.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)", [str(out_path)]
                ).fetchall()
            }
        finally:
            con.close()
        assert schema["City"] == "VARCHAR"
        assert schema["n_obs"] == "BIGINT"
        assert schema["slope_c_per_decade"] == "DOUBLE"
        assert schema["analysis_window"] == "VARCHAR"
