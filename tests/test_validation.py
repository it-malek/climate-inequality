"""Tests for src.validation — synthetic NetCDF fixtures only, no network.

The fixtures mirror the real Berkeley Earth gridded layout:
``temperature(time, latitude, longitude)`` float32 anomalies with a
fractional decimal-year time axis (mid-month convention) and NaN where
there is no land. The end-to-end test drives the full sample → gate →
residual → acceleration path against the conftest synthetic bundle.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.cleaning import to_decimal_decades
from src.validation import (
    VALIDATION_BUNDLE_SCHEMA,
    VALIDATION_COLUMNS,
    VALIDATION_GLOBAL_SCHEMA,
    acceleration,
    decode_fractional_years,
    forecast_residuals,
    overlap_agreement,
    run_validation,
    sample_grid_series,
    write_validation_summary,
)


def fractional_axis(start, end):
    """(months, fractional-year floats) in Berkeley's mid-month convention."""
    months = pd.date_range(start, end, freq="MS")
    return months, (months.year + (months.month - 0.5) / 12).to_numpy()


def write_grid(path, lats, lons, t_frac, data, var="temperature"):
    """Write a synthetic NetCDF matching the Berkeley gridded layout."""
    ds = xr.Dataset(
        {var: (("time", "latitude", "longitude"), data.astype(np.float32))},
        coords={
            "time": np.asarray(t_frac, dtype=float),
            "latitude": np.asarray(lats, dtype=float),
            "longitude": np.asarray(lons, dtype=float),
        },
    )
    ds.to_netcdf(path)
    return path


def make_trends(rows):
    """Trends-like frame: (city_id, City, Country, lat, lon, slope, intercept)."""
    return pd.DataFrame(
        rows,
        columns=[
            "city_id",
            "City",
            "Country",
            "Latitude",
            "Longitude",
            "slope_c_per_decade",
            "intercept",
        ],
    )


def line(months, slope, intercept):
    """Evaluate slope x decades + intercept on the pipeline's time axis."""
    decades = to_decimal_decades(pd.Series(months)).to_numpy()
    return slope * decades + intercept


class TestDecodeFractionalYears:
    def test_january_is_year_plus_one_twentyfourth(self):
        out = decode_fractional_years([2014 + 0.5 / 12])
        assert out.tolist() == [pd.Timestamp("2014-01-01")]

    def test_december(self):
        out = decode_fractional_years([2013 + 11.5 / 12])
        assert out.tolist() == [pd.Timestamp("2013-12-01")]

    def test_round_trip_full_axis(self):
        months, t_frac = fractional_axis("1750-01-01", "2024-12-01")
        decoded = decode_fractional_years(t_frac)
        assert (decoded == months).all()
        # Same mid-month convention as the trend-fitting time axis.
        redecoded = to_decimal_decades(pd.Series(decoded)).to_numpy() * 10
        assert redecoded == pytest.approx(t_frac, abs=1e-9)

    def test_start_of_month_convention_raises(self):
        # year + (month - 1) / 12 lands between mid-month points.
        with pytest.raises(ValueError, match="mid-month"):
            decode_fractional_years([2014.0, 2014 + 1 / 12])


class TestSampleGridSeries:
    LATS = [10.5, 11.5]
    LONS = [20.5, 21.5]

    def make_file(self, tmp_path):
        months, t_frac = fractional_axis("1990-01-01", "1991-12-01")
        # One distinct constant per cell so sampling errors are visible.
        cell_values = np.array([[1.0, 2.0], [3.0, 4.0]])
        data = np.broadcast_to(
            cell_values, (len(months), 2, 2)
        ).copy()
        return write_grid(tmp_path / "grid.nc", self.LATS, self.LONS, t_frac, data)

    def make_two_cities(self):
        return make_trends(
            [
                (0, "A", "X", 10.6, 20.4, 0.1, 0.0),  # -> cell (10.5, 20.5) = 1
                (1, "B", "X", 11.4, 21.6, 0.1, 0.0),  # -> cell (11.5, 21.5) = 4
            ]
        )

    def test_nearest_cell_selection(self, tmp_path):
        wide, cells = sample_grid_series(self.make_file(tmp_path), self.make_two_cities())
        assert wide[0].unique().tolist() == [1.0]
        assert wide[1].unique().tolist() == [4.0]
        assert cells.set_index("city_id").loc[0].tolist() == [10.5, 20.5]
        assert cells.set_index("city_id").loc[1].tolist() == [11.5, 21.5]

    def test_index_is_decoded_months(self, tmp_path):
        wide, _ = sample_grid_series(self.make_file(tmp_path), self.make_two_cities())
        expected = pd.date_range("1990-01-01", "1991-12-01", freq="MS")
        assert (wide.index == expected).all()

    def test_chunked_read_matches_single_read(self, tmp_path):
        nc = self.make_file(tmp_path)
        trends = self.make_two_cities()
        one_shot, _ = sample_grid_series(nc, trends)
        chunked, _ = sample_grid_series(nc, trends, chunk_months=7)
        pd.testing.assert_frame_equal(one_shot, chunked)

    def test_start_skips_earlier_months(self, tmp_path):
        wide, _ = sample_grid_series(
            self.make_file(tmp_path), self.make_two_cities(), start="1991-01-01"
        )
        assert wide.index.min() == pd.Timestamp("1991-01-01")
        assert len(wide) == 12

    def test_missing_variable_raises(self, tmp_path):
        months, t_frac = fractional_axis("1990-01-01", "1990-12-01")
        nc = write_grid(
            tmp_path / "bad.nc",
            self.LATS,
            self.LONS,
            t_frac,
            np.zeros((len(months), 2, 2)),
            var="tavg",
        )
        with pytest.raises(ValueError, match="temperature"):
            sample_grid_series(nc, self.make_two_cities())

    def test_all_nan_cell_stays_nan(self, tmp_path):
        months, t_frac = fractional_axis("1990-01-01", "1990-12-01")
        data = np.full((len(months), 2, 2), np.nan)
        data[:, 1, 1] = 1.0
        nc = write_grid(tmp_path / "nan.nc", self.LATS, self.LONS, t_frac, data)
        wide, _ = sample_grid_series(nc, self.make_two_cities())
        assert wide[0].isna().all()  # ocean cell
        assert (wide[1] == 1.0).all()


class TestOverlapAgreement:
    START, END = "1950-01-01", "2013-09-01"

    def make_frames(self):
        months = pd.date_range(self.START, self.END, freq="MS")
        rng = np.random.default_rng(0)
        base = rng.normal(0.0, 1.0, (len(months), 2))
        pipeline = pd.DataFrame(
            np.column_stack([base, base]),
            index=months,
            columns=[0, 1, 2, 3],
        )
        # 0: identical; 1: constant offset; 2: pure noise; 3: all-NaN cell.
        grid = pipeline.copy()
        grid[1] = pipeline[1] + 0.5
        grid[2] = rng.normal(0.0, 1.0, len(months))
        grid[3] = np.nan
        return pipeline, grid

    def test_gate_and_statistics(self):
        pipeline, grid = self.make_frames()
        out = overlap_agreement(pipeline, grid, self.START, self.END).set_index(
            "city_id"
        )
        assert out["gate_pass"].tolist() == [True, True, False, False]
        assert out.loc[0, "overlap_r"] == pytest.approx(1.0)
        assert out.loc[0, "overlap_rmse"] == pytest.approx(0.0, abs=1e-12)
        assert out.loc[0, "overlap_bias"] == pytest.approx(0.0, abs=1e-12)
        n_months = len(pd.period_range(self.START, self.END, freq="M"))
        assert out["n_overlap"].tolist() == [n_months, n_months, n_months, 0]

    def test_offset_shows_in_bias_not_r(self):
        pipeline, grid = self.make_frames()
        out = overlap_agreement(pipeline, grid, self.START, self.END).set_index(
            "city_id"
        )
        assert out.loc[1, "overlap_r"] == pytest.approx(1.0)
        assert out.loc[1, "overlap_bias"] == pytest.approx(0.5)
        assert out.loc[1, "overlap_rmse"] == pytest.approx(0.5)

    def test_undefined_r_is_nan_and_fails_gate(self):
        pipeline, grid = self.make_frames()
        out = overlap_agreement(pipeline, grid, self.START, self.END).set_index(
            "city_id"
        )
        assert np.isnan(out.loc[3, "overlap_r"])
        assert not out.loc[3, "gate_pass"]

    def test_months_outside_window_ignored(self):
        pipeline, grid = self.make_frames()
        extra = pd.date_range("2013-10-01", "2024-12-01", freq="MS")
        garbage = pd.DataFrame(
            np.random.default_rng(1).normal(50.0, 10.0, (len(extra), 4)),
            index=extra,
            columns=grid.columns,
        )
        grid_long = pd.concat([grid, garbage])
        out = overlap_agreement(
            pipeline, grid_long, self.START, self.END
        ).set_index("city_id")
        assert out.loc[0, "overlap_r"] == pytest.approx(1.0)


class TestForecastResiduals:
    SLOPE = 0.2
    INTERCEPT = -40.0

    def make_inputs(self, end="2015-12-01", offset=0.3):
        months = pd.date_range("2010-01-01", end, freq="MS")
        trends = make_trends([(0, "A", "X", 10.0, 20.0, self.SLOPE, self.INTERCEPT)])
        observed = line(months, self.SLOPE, self.INTERCEPT) + offset
        grid = pd.DataFrame({0: observed}, index=months)
        return trends, grid

    def test_constant_offset_recovered(self):
        trends, grid = self.make_inputs()
        per_city, _ = forecast_residuals(trends, grid)
        row = per_city.iloc[0]
        # Forecast window 2013-10..2015-12 = 27 months.
        assert row["n_forecast"] == 27
        assert row["mean_residual"] == pytest.approx(0.3, abs=1e-9)
        assert row["mean_residual_pre2023"] == pytest.approx(0.3, abs=1e-9)

    def test_el_nino_split(self):
        trends, grid = self.make_inputs(end="2024-12-01", offset=0.0)
        post = grid.index >= pd.Timestamp("2023-01-01")
        grid.loc[~post, 0] += 0.1
        grid.loc[post, 0] += 0.5
        per_city, _ = forecast_residuals(trends, grid)
        row = per_city.iloc[0]
        assert row["mean_residual_pre2023"] == pytest.approx(0.1, abs=1e-9)
        n_pre, n_post = 111, 24  # 2013-10..2022-12, 2023-01..2024-12
        expected = (n_pre * 0.1 + n_post * 0.5) / (n_pre + n_post)
        assert row["n_forecast"] == n_pre + n_post
        assert row["mean_residual"] == pytest.approx(expected, abs=1e-9)

    def test_months_before_forecast_start_excluded(self):
        trends, grid = self.make_inputs()
        # Corrupt the in-sample months; the forecast stats must not move.
        grid.loc[grid.index < pd.Timestamp("2013-10-01"), 0] += 99.0
        per_city, _ = forecast_residuals(trends, grid)
        assert per_city.iloc[0]["mean_residual"] == pytest.approx(0.3, abs=1e-9)

    def test_global_series_respects_gate(self):
        months = pd.date_range("2010-01-01", "2015-12-01", freq="MS")
        trends = make_trends(
            [
                (0, "A", "X", 10.0, 20.0, self.SLOPE, self.INTERCEPT),
                (1, "B", "X", 11.0, 21.0, self.SLOPE, self.INTERCEPT),
            ]
        )
        good = line(months, self.SLOPE, self.INTERCEPT)
        grid = pd.DataFrame({0: good, 1: good + 100.0}, index=months)
        gate = pd.Series({0: True, 1: False})
        _, monthly = forecast_residuals(trends, grid, gate=gate)
        assert (monthly["n_cities"] == 1).all()
        assert monthly["observed"].to_numpy() == pytest.approx(good)
        assert monthly["predicted"].to_numpy() == pytest.approx(good)


class TestAcceleration:
    START, END = "1950-01-01", "2013-09-01"
    SLOPE = 0.15
    INTERCEPT = -29.0

    def test_pure_continuation_has_zero_delta(self):
        months = pd.date_range(self.START, "2024-12-01", freq="MS")
        trends = make_trends([(0, "A", "X", 10.0, 20.0, self.SLOPE, self.INTERCEPT)])
        grid = pd.DataFrame(
            {0: line(months, self.SLOPE, self.INTERCEPT)}, index=months
        )
        out = acceleration(trends, grid, self.START, self.END).iloc[0]
        assert out["slope_full"] == pytest.approx(self.SLOPE, abs=1e-9)
        assert out["slope_overlap_grid"] == pytest.approx(self.SLOPE, abs=1e-9)
        assert out["slope_delta"] == pytest.approx(0.0, abs=1e-9)
        assert out["slope_full_ci_low"] <= out["slope_full"] <= out["slope_full_ci_high"]

    def test_post_window_steepening_detected(self):
        months = pd.date_range(self.START, "2024-12-01", freq="MS")
        values = line(months, self.SLOPE, self.INTERCEPT)
        post = months > pd.Timestamp(self.END)
        decades = to_decimal_decades(pd.Series(months)).to_numpy()
        pivot = decades[~post][-1]
        values[post] = values[~post][-1] + 3 * self.SLOPE * (decades[post] - pivot)
        trends = make_trends([(0, "A", "X", 10.0, 20.0, self.SLOPE, self.INTERCEPT)])
        grid = pd.DataFrame({0: values}, index=months)
        out = acceleration(trends, grid, self.START, self.END).iloc[0]
        assert out["slope_overlap_grid"] == pytest.approx(self.SLOPE, abs=1e-9)
        assert out["slope_full"] > self.SLOPE
        assert out["slope_delta"] > 0

    def test_too_few_observations_gives_nan(self):
        months = pd.date_range(self.START, "2024-12-01", freq="MS")
        trends = make_trends([(0, "A", "X", 10.0, 20.0, self.SLOPE, self.INTERCEPT)])
        grid = pd.DataFrame({0: np.full(len(months), np.nan)}, index=months)
        out = acceleration(trends, grid, self.START, self.END).iloc[0]
        assert np.isnan(out["slope_full"])
        assert np.isnan(out["slope_delta"])


@pytest.fixture(scope="module")
def validation_run(synthetic_bundle, tmp_path_factory):
    """run_validation on the synthetic bundle + a matching synthetic grid.

    Every city's cell continues its stored fitted line through 2024
    (zero residual, zero acceleration by construction), except Gamma,
    whose cell is pure noise and must fail the agreement gate.
    """
    root = tmp_path_factory.mktemp("validation")
    trends = synthetic_bundle["trends"]
    months, t_frac = fractional_axis("1950-01-01", "2024-12-01")
    lats = sorted(set(np.floor(trends["Latitude"]) + 0.5))
    lons = sorted(set(np.floor(trends["Longitude"]) + 0.5))
    data = np.full((len(months), len(lats), len(lons)), np.nan)
    rng = np.random.default_rng(7)
    for row in trends.itertuples():
        i = int(np.argmin(np.abs(np.asarray(lats) - row.Latitude)))
        j = int(np.argmin(np.abs(np.asarray(lons) - row.Longitude)))
        if row.City == "Gamma":
            data[:, i, j] = rng.normal(0.0, 1.0, len(months))
        else:
            data[:, i, j] = line(months, row.slope_c_per_decade, row.intercept)
    nc = write_grid(root / "grid.nc", lats, lons, t_frac, data)
    out = run_validation(
        nc_path=nc,
        bundle_dir=synthetic_bundle["bundle_dir"],
        out_path=root / "validation.parquet",
        figures_dir=root / "figs",
    )
    out["out_path"] = root / "validation.parquet"
    out["figures_dir"] = root / "figs"
    return out


class TestRunValidation:
    def test_parquet_written_with_schema(self, validation_run):
        on_disk = pd.read_parquet(validation_run["out_path"])
        assert on_disk.columns.tolist() == VALIDATION_COLUMNS
        assert len(on_disk) == len(validation_run["frame"]) == 6

    def test_gate_excludes_only_the_noise_city(self, validation_run):
        frame = validation_run["frame"]
        assert frame.loc[~frame["gate_pass"], "City"].tolist() == ["Gamma"]
        assert validation_run["stats"]["n_gate_pass"] == 5

    def test_continuation_scores_zero(self, validation_run):
        scored = validation_run["frame"].query("gate_pass")
        # Grid cells were built exactly on the stored lines (float32).
        assert scored["mean_residual"].abs().max() < 1e-4
        assert scored["slope_delta"].abs().max() < 1e-3
        assert (scored["overlap_r"] > 0.8).all()

    def test_window_metadata(self, validation_run):
        frame = validation_run["frame"]
        assert (frame["forecast_start"] == "2013-10-01").all()
        assert (frame["record_end"] == "2024-12-01").all()
        assert validation_run["stats"]["n_forecast_months"] == 135

    def test_figures_written(self, validation_run):
        figs = validation_run["figures_dir"]
        assert (figs / "validation_residual_map.html").exists()
        assert (figs / "validation_global_series.html").exists()

    def test_stale_bundle_raises(self, synthetic_bundle, tmp_path):
        bundle = synthetic_bundle["bundle_dir"]
        stale = tmp_path / "bundle"
        stale.mkdir()
        pd.read_parquet(bundle / "city_trends.parquet").drop(
            columns=["intercept"]
        ).to_parquet(stale / "city_trends.parquet")
        with pytest.raises(ValueError, match="intercept"):
            run_validation(
                nc_path=tmp_path / "missing.nc",
                bundle_dir=stale,
                out_path=tmp_path / "out.parquet",
                figures_dir=tmp_path,
            )


def _make_stub_frame(n: int = 5) -> pd.DataFrame:
    """Minimal per-city validation frame matching VALIDATION_COLUMNS."""
    return pd.DataFrame(
        {
            "City": [f"City{i}" for i in range(n)],
            "Country": ["X"] * n,
            "Latitude": np.linspace(-10.0, 10.0, n),
            "Longitude": np.linspace(20.0, 30.0, n),
            "city_id": list(range(n)),
            "grid_lat": np.linspace(-10.0, 10.0, n),
            "grid_lon": np.linspace(20.0, 30.0, n),
            "n_overlap": [768] * n,
            "overlap_r": [0.95] * n,
            "overlap_rmse": [0.1] * n,
            "overlap_bias": [0.0] * n,
            "gate_pass": [True] * n,
            "n_forecast": [135] * n,
            "mean_residual": [0.3] * n,
            "mean_residual_pre2023": [0.25] * n,
            "slope_c_per_decade": [0.146] * n,
            "slope_full": [0.200] * n,
            "slope_full_ci_low": [0.180] * n,
            "slope_full_ci_high": [0.220] * n,
            "slope_overlap_grid": [0.142] * n,
            "slope_delta": [0.054] * n,
            "forecast_start": ["2013-10-01"] * n,
            "record_end": ["2024-12-01"] * n,
        }
    )


def _make_stub_global(n_months: int = 60) -> pd.DataFrame:
    months = pd.date_range("2013-10-01", periods=n_months, freq="MS")
    return pd.DataFrame(
        {
            "dt": months,
            "observed": np.linspace(0.5, 1.2, n_months),
            "predicted": np.linspace(0.4, 1.0, n_months),
        }
    )


_STUB_STATS = {
    "n_locations": 5,
    "n_no_grid": 0,
    "n_gate_pass": 5,
    "median_overlap_r": 0.95,
    "forecast_start": "2013-10-01",
    "record_end": "2024-12-01",
    "n_forecast_months": 135,
    "mean_residual": 0.30,
    "mean_residual_pre2023": 0.25,
    "mean_slope_stored": 0.146,
    "mean_slope_overlap_grid": 0.142,
    "mean_slope_full": 0.200,
    "mean_slope_delta": 0.054,
    "slope_delta_ci_low": 0.040,
    "slope_delta_ci_high": 0.068,
}


class TestWriteValidationSummary:
    def test_json_round_trips_stats(self, tmp_path):
        summary_path = tmp_path / "summary.json"
        write_validation_summary(
            _STUB_STATS,
            _make_stub_frame(),
            _make_stub_global(),
            summary_path=summary_path,
            bundle_path=tmp_path / "bundle.parquet",
            global_path=tmp_path / "global.parquet",
        )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload == _STUB_STATS
        assert payload["n_gate_pass"] == 5

    def test_bundle_parquet_schema(self, tmp_path):
        bundle_path = tmp_path / "bundle.parquet"
        write_validation_summary(
            _STUB_STATS,
            _make_stub_frame(),
            _make_stub_global(),
            summary_path=tmp_path / "summary.json",
            bundle_path=bundle_path,
            global_path=tmp_path / "global.parquet",
        )
        df = pd.read_parquet(bundle_path)
        assert list(df.columns) == list(VALIDATION_BUNDLE_SCHEMA)
        assert str(df["Latitude"].dtype) == "float32"
        assert str(df["gate_pass"].dtype) == "bool"

    def test_global_parquet_schema(self, tmp_path):
        global_path = tmp_path / "global.parquet"
        write_validation_summary(
            _STUB_STATS,
            _make_stub_frame(),
            _make_stub_global(),
            summary_path=tmp_path / "summary.json",
            bundle_path=tmp_path / "bundle.parquet",
            global_path=global_path,
        )
        df = pd.read_parquet(global_path)
        assert list(df.columns) == list(VALIDATION_GLOBAL_SCHEMA)
        assert str(df["observed"].dtype) == "float32"
        # DATE columns stored as Python date objects or datetime64 (parquet engine varies).
        assert df["dt"].dtype == object or pd.api.types.is_datetime64_any_dtype(df["dt"])

    def test_returns_path_dict(self, tmp_path):
        paths = write_validation_summary(
            _STUB_STATS,
            _make_stub_frame(),
            _make_stub_global(),
            summary_path=tmp_path / "summary.json",
            bundle_path=tmp_path / "bundle.parquet",
            global_path=tmp_path / "global.parquet",
        )
        assert set(paths) == {"summary", "bundle", "global"}
        assert all(p.exists() for p in paths.values())
