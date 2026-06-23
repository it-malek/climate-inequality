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
    COUPLING_ASSET,
    COUPLING_CONSUMPTION_ASSET,
    COUPLING_CONSUMPTION_SUMMARY_ASSET,
    COUPLING_EXPOSURE_ASSET,
    COUPLING_EXPOSURE_SUMMARY_ASSET,
    COUPLING_SUMMARY_ASSET,
    DECOMPOSITION_SUMMARY_ASSET,
    EXPLAIN_FEATURES_ASSET,
    INEQUALITY_ASSET,
    INEQUALITY_SUMMARY_ASSET,
    STABILITY_SUMMARY_ASSET,
    STATS_ASSET,
    SURFACE_ASSET,
    TRENDS_ASSET,
    VALIDATION_ASSET,
    VALIDATION_GLOBAL_ASSET,
    _EXPLAIN_BUNDLE_PATH,
    _EXPLAIN_SUMMARY_PATH,
    _STABILITY_SUMMARY_PATH,
    _VALIDATION_BUNDLE_PATH,
    _VALIDATION_GLOBAL_PATH,
    _VALIDATION_SUMMARY_PATH,
    build_app_assets,
    build_coupling_consumption_asset,
    build_coupling_exposure_asset,
    build_coupling_summary_asset,
    build_decomposition_summaries,
    build_stability_summary_asset,
    disambiguate_labels,
    theil_sen_intercepts,
)
from src.cleaning import to_decimal_decades
from tests.conftest import SYNTHETIC_CITIES, SYNTHETIC_LAND, make_synthetic_inputs
from tests.test_emissions import make_inequality_frame


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
            "validation", "explain",
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


class TestOptionalFindingsConstants:
    """app_assets re-declares validation/explain path constants from PROCESSED_DIR
    to avoid a circular import; these tests verify that the re-declared copies
    stay in sync with the source-of-truth constants in validation.py / explain.py.
    """

    def test_validation_summary_path_matches_source(self):
        from src.validation import DEFAULT_VALIDATION_SUMMARY_PATH
        assert _VALIDATION_SUMMARY_PATH == DEFAULT_VALIDATION_SUMMARY_PATH

    def test_validation_bundle_path_matches_source(self):
        from src.validation import DEFAULT_VALIDATION_BUNDLE_PATH
        assert _VALIDATION_BUNDLE_PATH == DEFAULT_VALIDATION_BUNDLE_PATH

    def test_validation_global_path_matches_source(self):
        from src.validation import DEFAULT_VALIDATION_GLOBAL_PATH
        assert _VALIDATION_GLOBAL_PATH == DEFAULT_VALIDATION_GLOBAL_PATH

    def test_explain_summary_path_matches_source(self):
        from src.explain import DEFAULT_EXPLAIN_SUMMARY_PATH
        assert _EXPLAIN_SUMMARY_PATH == DEFAULT_EXPLAIN_SUMMARY_PATH

    def test_explain_bundle_path_matches_source(self):
        from src.explain import DEFAULT_EXPLAIN_BUNDLE_PATH
        assert _EXPLAIN_BUNDLE_PATH == DEFAULT_EXPLAIN_BUNDLE_PATH

    def test_stability_summary_path_matches_source(self):
        from src.stability import DEFAULT_SUMMARY_PATH
        assert _STABILITY_SUMMARY_PATH == DEFAULT_SUMMARY_PATH


class TestOptionalFindingsMerge:
    """_merge_optional_findings: absent warns+skips; half-present raises; present merges."""

    def _build_no_findings(self, root):
        inputs = make_synthetic_inputs(root)
        return build_app_assets(
            **inputs,
            out_dir=root / "bundle",
            surface_out_dir=root / "outputs",
            k=3,
            resolution=15.0,
            land=SYNTHETIC_LAND,
            validation_summary_path=root / "missing_summary.json",
            validation_bundle_path=root / "missing_bundle.parquet",
            validation_global_path=root / "missing_global.parquet",
            explain_summary_path=root / "missing_explain.json",
            explain_bundle_path=root / "missing_explain.parquet",
        )

    def test_absent_artifacts_skipped_gracefully(self, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="src.app_assets"):
            result = self._build_no_findings(tmp_path)
        payload = json.loads(
            result["paths"][STATS_ASSET].read_text(encoding="utf-8")
        )
        assert "validation" not in payload
        assert "explain" not in payload
        # Builder should have logged at least one warning about missing summaries.
        assert any("validation" in r.message.lower() or "explain" in r.message.lower()
                   for r in caplog.records)

    def test_half_present_raises(self, tmp_path):
        inputs = make_synthetic_inputs(tmp_path)
        summary_path = tmp_path / "validation_summary.json"
        summary_path.write_text(json.dumps({"n_locations": 1}) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="partial validation findings"):
            build_app_assets(
                **inputs,
                out_dir=tmp_path / "bundle",
                surface_out_dir=tmp_path / "outputs",
                k=3,
                resolution=15.0,
                land=SYNTHETIC_LAND,
                validation_summary_path=summary_path,
                validation_bundle_path=tmp_path / "missing_bundle.parquet",
                validation_global_path=tmp_path / "missing_global.parquet",
                explain_summary_path=tmp_path / "missing_explain.json",
                explain_bundle_path=tmp_path / "missing_explain.parquet",
            )

    def test_new_parquet_assets_written(self, synthetic_bundle):
        bundle_dir = synthetic_bundle["bundle_dir"]
        assert (bundle_dir / VALIDATION_ASSET).exists()
        assert (bundle_dir / VALIDATION_GLOBAL_ASSET).exists()
        assert (bundle_dir / EXPLAIN_FEATURES_ASSET).exists()

    def test_stale_schema_bundle_raises_at_build(self, tmp_path):
        # Summary + global present, but the residual-map parquet is missing a
        # required column (gate_pass) -- the build is the integrity checkpoint,
        # so this must fail here rather than silently copy and break the app.
        inputs = make_synthetic_inputs(tmp_path)
        summary_path = tmp_path / "validation_summary.json"
        summary_path.write_text(json.dumps({"n_locations": 1}) + "\n", encoding="utf-8")
        bad_bundle = tmp_path / "validation_bundle.parquet"
        pd.DataFrame(
            {
                "City": ["A"], "Country": ["X"],
                "Latitude": [1.0], "Longitude": [2.0],
                "mean_residual": [0.1], "overlap_r": [0.9],
                # gate_pass deliberately omitted
            }
        ).to_parquet(bad_bundle, index=False)
        global_path = tmp_path / "validation_global.parquet"
        pd.DataFrame(
            {
                "dt": pd.to_datetime(["2014-01-01"]),
                "observed": [0.1], "predicted": [0.1],
            }
        ).to_parquet(global_path, index=False)
        with pytest.raises(RuntimeError, match="missing required column"):
            build_app_assets(
                **inputs,
                out_dir=tmp_path / "bundle",
                surface_out_dir=tmp_path / "outputs",
                k=3,
                resolution=15.0,
                land=SYNTHETIC_LAND,
                validation_summary_path=summary_path,
                validation_bundle_path=bad_bundle,
                validation_global_path=global_path,
                explain_summary_path=tmp_path / "absent_explain.json",
                explain_bundle_path=tmp_path / "absent_explain.parquet",
            )


class TestDecompositionSummaries:
    """build_decomposition_summaries wires the headline page into the bundle."""

    def test_writes_inequality_summary_always(self, tmp_path):
        inequality_path = tmp_path / "country_inequality.parquet"
        make_inequality_frame().to_parquet(inequality_path, index=False)
        out_dir = tmp_path / "bundle"

        written = build_decomposition_summaries(
            inequality_path=inequality_path,
            out_dir=out_dir,
            city_features_path=tmp_path / "absent_features.parquet",
            income_path=tmp_path / "absent_income.csv",
        )

        # Inequality needs only the country table, so it is always produced.
        assert INEQUALITY_SUMMARY_ASSET in written
        # Decomposition inputs are absent: it degrades gracefully (the page
        # renders its 'not built yet' state), not an error.
        assert DECOMPOSITION_SUMMARY_ASSET not in written
        payload = json.loads(written[INEQUALITY_SUMMARY_ASSET].read_text("utf-8"))
        assert payload["interpretation"]  # the scope disclaimer travels along
        assert "gini" in payload and "theil_t" in payload

    def test_inequality_summary_json_is_byte_stable(self, tmp_path):
        # Rebuilding the same inputs must reproduce the file exactly (round_floats).
        inequality_path = tmp_path / "country_inequality.parquet"
        make_inequality_frame().to_parquet(inequality_path, index=False)
        first = build_decomposition_summaries(
            inequality_path=inequality_path, out_dir=tmp_path / "b1",
            city_features_path=tmp_path / "x", income_path=tmp_path / "y",
        )[INEQUALITY_SUMMARY_ASSET].read_text("utf-8")
        second = build_decomposition_summaries(
            inequality_path=inequality_path, out_dir=tmp_path / "b2",
            city_features_path=tmp_path / "x", income_path=tmp_path / "y",
        )[INEQUALITY_SUMMARY_ASSET].read_text("utf-8")
        assert first == second


class TestStabilitySummaryAsset:
    """build_stability_summary_asset wires the robustness layer into the bundle."""

    def test_absent_inputs_skipped_gracefully(self, tmp_path):
        # Without the Phase-7 city features / income table the stability summary
        # is omitted (the sensitivity page keeps its pending state), not an error.
        inequality_path = tmp_path / "country_inequality.parquet"
        make_inequality_frame().to_parquet(inequality_path, index=False)
        out_dir = tmp_path / "bundle"

        written = build_stability_summary_asset(
            inequality_path=inequality_path,
            out_dir=out_dir,
            city_features_path=tmp_path / "absent_features.parquet",
            income_path=tmp_path / "absent_income.csv",
        )

        assert written == {}
        assert not (out_dir / STABILITY_SUMMARY_ASSET).exists()


class TestCouplingAsset:
    """build_coupling_summary_asset wires the L3 comparator into the bundle."""

    def test_writes_both_artifacts(self, tmp_path):
        inequality_path = tmp_path / "country_inequality.parquet"
        make_inequality_frame().to_parquet(inequality_path, index=False)
        out_dir = tmp_path / "bundle"

        written = build_coupling_summary_asset(
            inequality_path=inequality_path, out_dir=out_dir
        )

        assert COUPLING_ASSET in written and COUPLING_SUMMARY_ASSET in written
        assert (out_dir / COUPLING_ASSET).exists()
        payload = json.loads((out_dir / COUPLING_SUMMARY_ASSET).read_text("utf-8"))
        assert set(payload) == {
            "spearman_rho", "n_high_impact_low_responsibility",
            "inequality_coefficient", "top_suffer_least_cause",
            "top_cause_least_suffer",
        }
        # §8: the comparator summary carries no interpretation/narrative.
        assert "interpretation" not in payload

    def test_coupling_summary_byte_stable(self, tmp_path):
        inequality_path = tmp_path / "country_inequality.parquet"
        make_inequality_frame().to_parquet(inequality_path, index=False)
        first = build_coupling_summary_asset(
            inequality_path=inequality_path, out_dir=tmp_path / "b1"
        )[COUPLING_SUMMARY_ASSET].read_text("utf-8")
        second = build_coupling_summary_asset(
            inequality_path=inequality_path, out_dir=tmp_path / "b2"
        )[COUPLING_SUMMARY_ASSET].read_text("utf-8")
        assert first == second


class TestCouplingConsumptionAsset:
    """build_coupling_consumption_asset wires the PCS v2 consumption lens in."""

    def test_writes_both_artifacts_with_window_and_two_passes(self, tmp_path):
        inequality_path = tmp_path / "country_inequality.parquet"
        make_inequality_frame().to_parquet(inequality_path, index=False)
        out_dir = tmp_path / "bundle"

        written = build_coupling_consumption_asset(
            inequality_path=inequality_path, out_dir=out_dir
        )

        assert COUPLING_CONSUMPTION_ASSET in written
        assert COUPLING_CONSUMPTION_SUMMARY_ASSET in written
        assert (out_dir / COUPLING_CONSUMPTION_ASSET).exists()
        payload = json.loads(
            (out_dir / COUPLING_CONSUMPTION_SUMMARY_ASSET).read_text("utf-8")
        )
        assert set(payload) == {
            "window", "consumption_vs_impact", "production_to_consumption_shift",
        }
        assert payload["window"]["n_countries"] > 0
        assert "interpretation" not in payload

    def test_byte_stable(self, tmp_path):
        inequality_path = tmp_path / "country_inequality.parquet"
        make_inequality_frame().to_parquet(inequality_path, index=False)
        first = build_coupling_consumption_asset(
            inequality_path=inequality_path, out_dir=tmp_path / "b1"
        )[COUPLING_CONSUMPTION_SUMMARY_ASSET].read_text("utf-8")
        second = build_coupling_consumption_asset(
            inequality_path=inequality_path, out_dir=tmp_path / "b2"
        )[COUPLING_CONSUMPTION_SUMMARY_ASSET].read_text("utf-8")
        assert first == second

    def test_skips_when_consumption_columns_absent(self, tmp_path):
        inequality_path = tmp_path / "country_inequality.parquet"
        # A pre-v2 country table with no consumption columns.
        frame = make_inequality_frame().drop(
            columns=[
                "consumption_start_year",
                "cum_consumption_t_per_capita",
                "cum_co2_window_t_per_capita",
            ]
        )
        frame.to_parquet(inequality_path, index=False)
        out_dir = tmp_path / "bundle"

        written = build_coupling_consumption_asset(
            inequality_path=inequality_path, out_dir=out_dir
        )
        assert written == {}
        assert not (out_dir / COUPLING_CONSUMPTION_ASSET).exists()


class TestCouplingExposureAsset:
    """build_coupling_exposure_asset wires the people-weighted exposure lens in."""

    def test_writes_both_artifacts_with_coverage_and_two_passes(self, tmp_path):
        inequality_path = tmp_path / "country_inequality.parquet"
        make_inequality_frame().to_parquet(inequality_path, index=False)
        out_dir = tmp_path / "bundle"

        written = build_coupling_exposure_asset(
            inequality_path=inequality_path, out_dir=out_dir
        )

        assert COUPLING_EXPOSURE_ASSET in written
        assert COUPLING_EXPOSURE_SUMMARY_ASSET in written
        assert (out_dir / COUPLING_EXPOSURE_ASSET).exists()
        payload = json.loads(
            (out_dir / COUPLING_EXPOSURE_SUMMARY_ASSET).read_text("utf-8")
        )
        assert set(payload) == {
            "coverage", "station_vs_people", "people_weighted_inequality",
        }
        assert payload["coverage"]["n_countries"] > 0
        assert "interpretation" not in payload

    def test_byte_stable(self, tmp_path):
        inequality_path = tmp_path / "country_inequality.parquet"
        make_inequality_frame().to_parquet(inequality_path, index=False)
        first = build_coupling_exposure_asset(
            inequality_path=inequality_path, out_dir=tmp_path / "b1"
        )[COUPLING_EXPOSURE_SUMMARY_ASSET].read_text("utf-8")
        second = build_coupling_exposure_asset(
            inequality_path=inequality_path, out_dir=tmp_path / "b2"
        )[COUPLING_EXPOSURE_SUMMARY_ASSET].read_text("utf-8")
        assert first == second

    def test_skips_when_population_weighting_absent(self, tmp_path):
        inequality_path = tmp_path / "country_inequality.parquet"
        frame = make_inequality_frame().drop(
            columns=["trend_c_per_decade_pop_weighted", "pop_weight_coverage"]
        )
        frame.to_parquet(inequality_path, index=False)
        out_dir = tmp_path / "bundle"

        written = build_coupling_exposure_asset(
            inequality_path=inequality_path, out_dir=out_dir
        )
        assert written == {}
        assert not (out_dir / COUPLING_EXPOSURE_ASSET).exists()
