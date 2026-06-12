"""Tests for src.app_assets — the committed dashboard bundle builder.

Happy-path assertions run against the session-scoped synthetic bundle
(tests/conftest.py); the integrity-check tests rebuild from deliberately
tampered inputs. No real data and no network anywhere.
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.app_assets import (
    ANOMALIES_ASSET,
    APP_DATA_DIR,
    INEQUALITY_ASSET,
    STATS_ASSET,
    SURFACE_ASSET,
    TRENDS_ASSET,
    build_app_assets,
    disambiguate_labels,
    theil_sen_intercepts,
)
from src.cleaning import to_decimal_decades
from tests.conftest import SYNTHETIC_CITIES, SYNTHETIC_LAND, make_synthetic_inputs


class TestDisambiguateLabels:
    def test_appends_coordinates_only_for_same_named_pairs(self):
        trends = pd.DataFrame(
            {
                "City": ["Springfield", "Springfield", "Boston"],
                "Country": ["United States"] * 3,
                "Latitude": [30.0, 45.0, 42.36],
                "Longitude": [-90.0, -90.0, -71.06],
            }
        )
        labels = disambiguate_labels(trends)
        assert labels.tolist()[:2] == [
            "Springfield (30.00°, -90.00°)",
            "Springfield (45.00°, -90.00°)",
        ]
        assert labels.iloc[2] == "Boston"

    def test_same_name_in_different_countries_untouched(self):
        trends = pd.DataFrame(
            {
                "City": ["Springfield", "Springfield"],
                "Country": ["United States", "Canada"],
                "Latitude": [30.0, 45.0],
                "Longitude": [-90.0, -75.0],
            }
        )
        assert disambiguate_labels(trends).tolist() == ["Springfield"] * 2


class TestTheilSenIntercepts:
    @staticmethod
    def exact_inputs(slope=0.2, intercept=-39.0):
        months = pd.date_range("1950-01-01", "1959-01-01", freq="MS")  # odd count
        decades = to_decimal_decades(pd.Series(months))
        anomalies = pd.DataFrame(
            {
                "city_id": np.int32(0),
                "dt": months,
                "anomaly": slope * decades + intercept,
            }
        )
        trends = pd.DataFrame(
            {"city_id": [np.int32(0)], "City": ["A"], "slope_c_per_decade": [slope]}
        )
        return trends, anomalies

    def test_recovers_exact_intercept(self):
        trends, anomalies = self.exact_inputs()
        out = theil_sen_intercepts(trends, anomalies)
        assert out.loc[0] == pytest.approx(-39.0, abs=1e-9)

    def test_stale_slope_raises(self):
        trends, anomalies = self.exact_inputs()
        trends["slope_c_per_decade"] += 0.05
        with pytest.raises(RuntimeError, match="stale"):
            theil_sen_intercepts(trends, anomalies)


class TestBundleContents:
    def test_all_files_written(self, synthetic_bundle):
        for name in (
            TRENDS_ASSET, ANOMALIES_ASSET, SURFACE_ASSET, INEQUALITY_ASSET,
            STATS_ASSET,
        ):
            assert synthetic_bundle["paths"][name].exists()

    def test_trends_asset_ids_labels_intercepts(self, synthetic_bundle):
        trends = pd.read_parquet(synthetic_bundle["paths"][TRENDS_ASSET])
        assert trends["city_id"].tolist() == list(range(len(SYNTHETIC_CITIES)))
        assert str(trends["city_id"].dtype) == "int32"
        assert trends["intercept"].notna().all()
        springfields = trends.loc[trends["City"] == "Springfield", "label"]
        assert len(springfields) == 2
        assert springfields.nunique() == 2
        assert springfields.str.contains("°").all()
        others = trends.loc[trends["City"] != "Springfield", "label"]
        assert (others == trends.loc[others.index, "City"]).all()

    def test_intercept_matches_independent_refit(self, synthetic_bundle):
        trends = pd.read_parquet(synthetic_bundle["paths"][TRENDS_ASSET])
        anomalies = pd.read_parquet(synthetic_bundle["paths"][ANOMALIES_ASSET])
        city = anomalies.loc[anomalies["city_id"] == 0]
        fit = stats.theilslopes(
            city["anomaly"].to_numpy(),
            to_decimal_decades(city["dt"]).to_numpy(),
        )
        # float32 anomaly storage costs ~1e-7 of precision in the refit.
        assert trends.loc[0, "intercept"] == pytest.approx(fit.intercept, abs=1e-4)
        assert trends.loc[0, "slope_c_per_decade"] == pytest.approx(
            fit.slope, abs=1e-4
        )

    def test_anomaly_counts_match_n_obs(self, synthetic_bundle):
        trends = pd.read_parquet(synthetic_bundle["paths"][TRENDS_ASSET])
        anomalies = pd.read_parquet(synthetic_bundle["paths"][ANOMALIES_ASSET])
        assert str(anomalies["anomaly"].dtype) == "float32"
        counts = anomalies.groupby("city_id").size()
        expected = trends.set_index("city_id")["n_obs"]
        assert counts.sort_index().tolist() == expected.sort_index().tolist()

    def test_surface_long_form_covers_grid_with_ocean_nans(self, synthetic_bundle):
        surface = pd.read_parquet(synthetic_bundle["paths"][SURFACE_ASSET])
        grid = synthetic_bundle["surface"]
        assert len(surface) == len(grid["grid_lon"]) * len(grid["grid_lat"])
        west = surface.loc[surface["lon"] < 0.0, "value"]  # synthetic ocean
        east = surface.loc[surface["lon"] > 0.0, "value"]  # synthetic land
        assert west.isna().all()
        assert east.notna().any()

    def test_inequality_asset_round_trips(self, synthetic_bundle):
        out = pd.read_parquet(synthetic_bundle["paths"][INEQUALITY_ASSET])
        assert len(out) == 7  # make_inequality_frame: 4 Africa + 3 Europe
        assert {"Country", "continent", "cum_co2_t_per_capita"} <= set(out.columns)

    def test_stats_payload(self, synthetic_bundle):
        payload = json.loads(
            synthetic_bundle["paths"][STATS_ASSET].read_text(encoding="utf-8")
        )
        assert set(payload) == {
            "generated_at", "trends", "interpolation", "inequality",
        }
        trends = payload["trends"]
        assert trends["n_locations"] == len(SYNTHETIC_CITIES)
        assert trends["n_arctic"] == 1  # only Arcticville is above 60°N
        assert trends["arctic_ratio"] > 1.0  # it also warms fastest
        interp = payload["interpolation"]
        assert interp["winner"] in {"idw", "kriging"}
        assert len(interp["cv_leave_location_out"]) == 2
        assert all(row["rmse"] >= 0 for row in interp["cv_leave_location_out"])
        ineq = payload["inequality"]
        assert ineq["n_countries"] == 7
        assert ineq["cutoff_year"] == 2013
        assert {"coef", "se", "ci_low", "ci_high", "p_value", "r2"} == set(
            ineq["ols_fe"]
        )

    def test_data_dir_constant_matches_app_loaders(self):
        from app import loaders

        assert APP_DATA_DIR == loaders.APP_DATA_DIR


class TestIntegrityChecks:
    def test_stale_trends_parquet_refused(self, tmp_path):
        inputs = make_synthetic_inputs(tmp_path)
        trends = pd.read_parquet(inputs["trends_path"])
        trends["slope_c_per_decade"] += 0.05
        trends.to_parquet(inputs["trends_path"], index=False)
        with pytest.raises(RuntimeError, match="stale"):
            build_app_assets(
                **inputs,
                out_dir=tmp_path / "bundle",
                surface_out_dir=tmp_path / "outputs",
                k=3,
                resolution=15.0,
                land=SYNTHETIC_LAND,
            )

    def test_missing_trends_row_refused(self, tmp_path):
        inputs = make_synthetic_inputs(tmp_path)
        trends = pd.read_parquet(inputs["trends_path"])
        trends.iloc[:-1].to_parquet(inputs["trends_path"], index=False)
        with pytest.raises(RuntimeError, match="no trends row"):
            build_app_assets(
                **inputs,
                out_dir=tmp_path / "bundle",
                surface_out_dir=tmp_path / "outputs",
                k=3,
                resolution=15.0,
                land=SYNTHETIC_LAND,
            )
