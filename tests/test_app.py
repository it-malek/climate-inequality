"""Tests for the Streamlit app: loaders plus AppTest page smoke/interaction.

Pages run in-process via streamlit.testing.v1.AppTest against the
session-scoped synthetic bundle, with app.loaders.APP_DATA_DIR
monkeypatched and st.cache_data cleared around every test so nothing
leaks between bundles (or into the real committed one).
"""

import shutil

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import loaders


@pytest.fixture
def bundle_dir(synthetic_bundle, monkeypatch):
    """Point the loaders at the synthetic bundle, with clean caches."""
    directory = synthetic_bundle["bundle_dir"]
    monkeypatch.setattr(loaders, "APP_DATA_DIR", directory)
    st.cache_data.clear()
    yield directory
    st.cache_data.clear()


def _exec_view(module_name: str) -> None:
    # AppTest.from_function lifts this function's *source* into a fresh
    # module, so imports must happen inside the body; import_module returns
    # the already-imported (and monkeypatched) view from sys.modules.
    import importlib

    importlib.import_module(module_name).render()


def run_page(module_name: str) -> AppTest:
    at = AppTest.from_function(
        _exec_view, kwargs={"module_name": module_name}, default_timeout=10
    )
    at.run()
    assert not at.exception, at.exception
    return at


class TestLoaders:
    def test_surface_round_trips_through_long_form(self, bundle_dir, synthetic_bundle):
        grid_lon, grid_lat, values = loaders.load_surface()
        built = synthetic_bundle["surface"]
        assert grid_lon == pytest.approx(built["grid_lon"])
        assert grid_lat == pytest.approx(built["grid_lat"])
        assert values.shape == built["surface"].shape
        assert np.array_equal(np.isnan(values), np.isnan(built["surface"]))
        finite = ~np.isnan(values)
        # float32 storage: agree to ~1e-7 relative.
        assert values[finite] == pytest.approx(built["surface"][finite], rel=1e-5)

    def test_city_series_sorted_and_complete(self, bundle_dir):
        trends = loaders.load_city_trends()
        city_id = int(trends["city_id"].iloc[0])
        series = loaders.load_city_series(city_id)
        assert list(series.columns) == ["dt", "anomaly"]
        assert len(series) == int(trends["n_obs"].iloc[0])
        assert series["dt"].is_monotonic_increasing

    def test_era5_validation_summary_absent_returns_none(self, bundle_dir):
        # The synthetic bundle has no ERA5 cross-check (grid is never fetched in
        # CI), so the optional loader degrades to None and the panel is omitted.
        st.cache_data.clear()
        assert loaders.load_era5_validation_summary() is None

    def test_window_years(self):
        assert loaders.window_years("1950-01-01..2013-09-01") == "1950–2013"

    def test_window_years_rejects_malformed(self):
        with pytest.raises(ValueError):
            loaders.window_years("1950-01-01")

    def test_stale_bundle_missing_column_raises(
        self, bundle_dir, tmp_path, monkeypatch
    ):
        # The session bundle is read-only; break a copy of it instead.
        broken = tmp_path / "broken_bundle"
        shutil.copytree(bundle_dir, broken)
        trends = pd.read_parquet(broken / "city_trends.parquet")
        trends.drop(columns=["intercept"]).to_parquet(
            broken / "city_trends.parquet", index=False
        )
        monkeypatch.setattr(loaders, "APP_DATA_DIR", broken)
        st.cache_data.clear()
        with pytest.raises(ValueError, match="intercept"):
            loaders.load_city_trends()


class TestTrendMapPage:
    def test_renders_with_sanity_metrics(self, bundle_dir):
        at = run_page("app.views.trend_map")
        assert at.title[0].value == "Where has land warmed fastest?"
        assert len(at.metric) == 3
        assert "Arctic amplification" in [m.label for m in at.metric]
        assert at.radio[0].value == "Interpolated surface"

    @pytest.mark.parametrize("layer", ["City stations", "Both"])
    def test_layer_toggle(self, bundle_dir, layer):
        at = run_page("app.views.trend_map")
        at.radio[0].set_value(layer).run()
        assert not at.exception


class TestCityExplorerPage:
    def test_renders_default_country(self, bundle_dir):
        at = run_page("app.views.city_explorer")
        assert at.selectbox[0].value == "United States"
        # The same-named pair is disambiguated by coordinates in the picker.
        options = at.selectbox[1].options
        assert len(options) == 2
        assert all(option.startswith("Springfield (") for option in options)
        assert at.metric[0].value.endswith("°C/decade")

    def test_switching_country_updates_city_and_metrics(self, bundle_dir):
        at = run_page("app.views.city_explorer")
        at.selectbox[0].select("Norway").run()
        assert not at.exception
        assert at.selectbox[1].value == "Arcticville"
        # Injected trend for Arcticville is 0.30 °C/decade (plus noise).
        slope = float(at.metric[0].value.split(" ")[0])
        assert slope == pytest.approx(0.30, abs=0.05)


class TestInequalityPage:
    def test_renders_with_stats(self, bundle_dir):
        at = run_page("app.views.inequality")
        assert at.title[0].value == "Do historically high-emitting countries warm faster?"
        labels = [m.label for m in at.metric]
        assert labels == ["Spearman ρ", "Within-continent effect", "Pooled effect"]
        # The synthetic frame is exactly monotone, so rho is +1.00.
        assert at.metric[0].value == "+1.00"
        assert len(at.dataframe) == 1


class TestEntryPoint:
    def test_entry_runs_default_page(self, bundle_dir):
        at = AppTest.from_file("app/streamlit_app.py", default_timeout=10)
        at.run()
        assert not at.exception, at.exception
        # Default page is now the decomposition dashboard.
        assert at.title[0].value == "Global Warming Inequality Decomposition"

    def test_interpretation_banner_on_entry(self, bundle_dir):
        at = AppTest.from_file("app/streamlit_app.py", default_timeout=10)
        at.run()
        from app.theme import BANNER_TEXT

        assert any(BANNER_TEXT in (m.value or "") for m in at.markdown), (
            "interpretation banner must render on the entry page"
        )


class TestDecompositionPage:
    def test_renders_metrics_and_shares(self, bundle_dir):
        at = run_page("app.views.decomposition")
        assert at.title[0].value == "Global Warming Inequality Decomposition"
        labels = [m.label for m in at.metric]
        assert "Gini of warming" in labels
        assert "Residual (unexplained)" in labels
        subheaders = [s.value for s in at.subheader]
        assert "How the inequality decomposes" in subheaders
        assert "Decomposition explorer" in subheaders
        assert at.radio[0].value == "Emissions only"

    def test_explorer_toggle_switches_view(self, bundle_dir):
        at = run_page("app.views.decomposition")
        at.radio[0].set_value("Full model").run()
        assert not at.exception
        assert "Variance explained" in [m.label for m in at.metric]

    def test_pending_state_without_summaries(self, tmp_path, monkeypatch):
        # Point loaders at an empty dir: the page must degrade, not crash.
        monkeypatch.setattr(loaders, "APP_DATA_DIR", tmp_path)
        st.cache_data.clear()
        at = run_page("app.views.decomposition")
        assert not at.exception
        assert len(at.info) == 1
        st.cache_data.clear()


class TestWorldMapPage:
    def test_renders_choropleth(self, bundle_dir):
        at = run_page("app.views.world_map")
        assert at.title[0].value == "Where warming is fastest"
        # Caption reports the country count and trend range; presence ⇒ rendered.
        assert any("countries" in (c.value or "") for c in at.caption)


class TestCouplingPage:
    def test_renders_metrics_and_charts(self, bundle_dir):
        at = run_page("app.views.coupling")
        assert at.title[0].value == "Responsibility vs impact"
        labels = [m.label for m in at.metric]
        assert "Spearman ρ" in labels
        assert "Inequality coefficient" in labels
        assert "High impact, low responsibility" in labels

    def test_renders_consumption_section(self, bundle_dir):
        # The synthetic bundle carries the PCS v2 consumption artifacts.
        at = run_page("app.views.coupling")
        labels = [m.label for m in at.metric]
        assert "Production → consumption rank ρ" in labels
        assert "Consumption inequality coefficient" in labels
        headers = [h.value for h in at.header]
        assert any("consumed" in h.lower() for h in headers)

    def test_renders_exposure_section_with_toggle(self, bundle_dir):
        # The synthetic bundle carries the PCS v2 people-weighted AND area-weighted
        # exposure artifacts; both share one Station / People / Area basis toggle.
        at = run_page("app.views.coupling")
        headers = [h.value for h in at.header]
        assert any("resident" in h.lower() for h in headers)
        # the inequality-basis toggle exists; it defaults to Station-based.
        basis = next(r for r in at.radio if "basis" in r.label.lower())
        # Switching to people-weighted reveals its rank-shift metric, no exception.
        basis.set_value("People-weighted (residents)").run()
        assert "Station → people rank ρ" in [m.label for m in at.metric]
        assert not at.exception
        # Switching to area-weighted reveals the area rank-shift metric.
        basis = next(r for r in at.radio if "basis" in r.label.lower())
        basis.set_value("Area-weighted (land)").run()
        assert "Station → area rank ρ" in [m.label for m in at.metric]
        assert not at.exception

    def test_pending_state_without_summary(self, tmp_path, monkeypatch):
        # Point loaders at an empty dir: the page must degrade, not crash.
        monkeypatch.setattr(loaders, "APP_DATA_DIR", tmp_path)
        st.cache_data.clear()
        at = run_page("app.views.coupling")
        assert not at.exception
        assert len(at.info) == 1
        st.cache_data.clear()


class TestPhysicalPage:
    def test_renders_metrics_and_sections(self, bundle_dir):
        # The synthetic bundle now carries L1 artifacts (forcings -> physical model).
        at = run_page("app.views.physical")
        assert at.title[0].value == "The physical climate engine"
        labels = [m.label for m in at.metric]
        assert {"Train R²", "Test RMSE", "Band coverage", "AR(1) ρ"} <= set(labels)
        # Hero chart carries its own title (no Streamlit subheader); the hindcast
        # section header remains.
        subheaders = [s.value for s in at.subheader]
        assert any("Trained" in s for s in subheaders)

    def test_pending_state_without_artifacts(self, tmp_path, monkeypatch):
        # Point loaders at an empty dir: the page must degrade, not crash.
        monkeypatch.setattr(loaders, "APP_DATA_DIR", tmp_path)
        st.cache_data.clear()
        at = run_page("app.views.physical")
        assert not at.exception
        assert len(at.info) == 1
        st.cache_data.clear()


class TestSensitivityPage:
    def test_pending_state_by_default(self, bundle_dir):
        # No stability_summary.json in the synthetic bundle -> pending state.
        at = run_page("app.views.sensitivity")
        assert at.title[0].value == "How confident are we?"
        assert len(at.info) == 1
        assert len(at.subheader) == 0  # no diagnostic sections rendered

    def test_renders_when_summary_present(self, bundle_dir, tmp_path, monkeypatch):
        import json

        populated = tmp_path / "stab_bundle"
        shutil.copytree(bundle_dir, populated)
        summary = {
            "interpretation": "descriptive only",
            "df_sensitivity": [
                {"df": 4, "coef": 0.03, "ci_low": 0.01, "ci_high": 0.05},
                {"df": 6, "coef": 0.028, "ci_low": 0.008, "ci_high": 0.048},
                {"df": 8, "coef": 0.026, "ci_low": 0.004, "ci_high": 0.048},
            ],
            "uncertainty": [
                {"method": "HC1", "coef": 0.03, "ci_low": 0.015, "ci_high": 0.045},
                {"method": "Conley HAC", "coef": 0.03, "ci_low": -0.005, "ci_high": 0.065},
            ],
            "influence": {"spec": "lat_continent", "top_dfbeta": [["Russia", 0.012], ["Canada", -0.008]]},
        }
        (populated / "stability_summary.json").write_text(json.dumps(summary))
        monkeypatch.setattr(loaders, "APP_DATA_DIR", populated)
        st.cache_data.clear()
        at = run_page("app.views.sensitivity")
        assert not at.exception
        assert not at.info  # no pending banner
        assert len(at.subheader) == 3  # df-sensitivity, uncertainty, influence
        st.cache_data.clear()

    def test_renders_share_stability_blocks(self, bundle_dir, tmp_path, monkeypatch):
        # The stability layer's own schema (bootstrap share CIs, leave-one-out
        # influence, residual Moran's I) lights up the three new sections.
        import json

        populated = tmp_path / "share_bundle"
        shutil.copytree(bundle_dir, populated)
        groups = {
            "emissions": {"point": 0.08, "mean": 0.08, "std": 0.02, "ci_low": 0.03, "ci_high": 0.15},
            "geography": {"point": 0.45, "mean": 0.45, "std": 0.04, "ci_low": 0.35, "ci_high": 0.55, "p_largest": 0.99},
            "socioeconomic": {"point": 0.05, "mean": 0.05, "std": 0.01, "ci_low": 0.02, "ci_high": 0.09},
            "population": {"point": 0.04, "mean": 0.04, "std": 0.01, "ci_low": 0.01, "ci_high": 0.08},
            "residual": {"point": 0.38, "mean": 0.38, "std": 0.03, "ci_low": 0.30, "ci_high": 0.46},
        }
        summary = {
            "interpretation": "descriptive only",
            "share_stability": {
                "method": "country_bootstrap", "n_boot": 100, "n_failed": 0,
                "groups": groups,
                "p_geography_largest": 0.99, "p_emissions_positive": 0.97,
                "block_bootstrap": {
                    "by": "spatial_block", "n_boot": 100, "n_failed": 0,
                    "groups": {
                        "emissions": {"ci_low": 0.01, "ci_high": 0.18},
                        "geography": {"ci_low": 0.30, "ci_high": 0.60},
                        "residual": {"ci_low": 0.25, "ci_high": 0.50},
                    },
                },
            },
            "influence": {
                "method": "leave_one_country_out",
                "by_group": {
                    "emissions": [["China", 0.02], ["United States", -0.01]],
                    "geography": [["Russia", 0.03], ["Canada", -0.02]],
                    "socioeconomic": [["India", 0.01]],
                    "population": [["Brazil", 0.01]],
                },
            },
            "residual_spatial": {
                "morans_i": 0.21, "p_value": 0.01, "n_permutations": 199,
                "k_neighbors": 8, "method": "centroid kNN", "n": 150,
            },
        }
        (populated / "stability_summary.json").write_text(json.dumps(summary))
        monkeypatch.setattr(loaders, "APP_DATA_DIR", populated)
        st.cache_data.clear()
        at = run_page("app.views.sensitivity")
        assert not at.exception
        assert not at.info  # no pending banner
        subheaders = [s.value for s in at.subheader]
        assert subheaders == [
            "Stability of the variance shares",
            "Most influential countries",
            "Spatial structure of the residual",
        ]
        st.cache_data.clear()


class TestValidationPage:
    def test_renders_metrics_and_title(self, bundle_dir):
        at = run_page("app.views.validation")
        assert at.title[0].value == "Did the 1950–2013 trends hold out of sample?"
        v = loaders.load_stats()["validation"]
        metrics = {m.label: m.value for m in at.metric}
        # Values must match the underlying stats with the view's formatting --
        # guards against field mis-mapping (e.g. residual vs overlap r) or a
        # dropped unit/precision, which a label-only check would miss.
        assert metrics["Overlap agreement (median r)"] == f"{v['median_overlap_r']:.2f}"
        assert metrics["Mean forecast residual"] == f"{v['mean_residual']:+.2f} °C"
        assert metrics["Slope: full record vs stored"] == (
            f"{v['mean_slope_full']:.3f} vs {v['mean_slope_stored']:.3f} °C/decade"
        )

    def test_graceful_degradation_without_bundle(self, bundle_dir, tmp_path, monkeypatch):
        empty = tmp_path / "empty_bundle"
        import shutil
        shutil.copytree(bundle_dir, empty)
        # Write a stats.json without the "validation" key.
        import json
        stats_path = empty / "stats.json"
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        payload.pop("validation", None)
        stats_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(loaders, "APP_DATA_DIR", empty)
        st.cache_data.clear()
        at = run_page("app.views.validation")
        assert not at.exception
        assert at.info[0].value.startswith("The validation bundle has not been built")
        st.cache_data.clear()


class TestExplainPage:
    def test_renders_sections(self, bundle_dir):
        at = run_page("app.views.explain")
        assert at.title[0].value == "What explains where warming is fast?"
        # Two subheaders: coefficient stability + city geography.
        assert len(at.subheader) == 2
        # Three country metrics (pooled/continent_fe/lat_continent) + three
        # city R² metrics (baseline/full/interaction).
        assert len(at.metric) == 6

        e = loaders.load_stats()["explain"]
        metrics = {m.label: m.value for m in at.metric}
        country = {s["spec_name"]: s for s in e["country_model"]["specs"]}
        assert metrics["Pooled effect"] == f"{country['pooled']['emissions']['coef']:+.3f}"
        assert metrics["Within-continent (FE)"] == (
            f"{country['continent_fe']['emissions']['coef']:+.3f}"
        )
        assert metrics["+ mean |latitude| control"] == (
            f"{country['lat_continent']['emissions']['coef']:+.3f}"
        )
        city = {s["spec_name"]: s for s in e["city_model"]["specs"]}
        assert metrics["Baseline R² (|latitude| only)"] == f"{city['baseline']['r2']:.3f}"
        assert metrics["Full model R²"] == f"{city['full']['r2']:.3f}"
        assert metrics["Interaction R²"] == f"{city['interaction']['r2']:.3f}"

    def test_graceful_degradation_without_bundle(self, bundle_dir, tmp_path, monkeypatch):
        empty = tmp_path / "empty_explain_bundle"
        import shutil
        shutil.copytree(bundle_dir, empty)
        import json
        stats_path = empty / "stats.json"
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        payload.pop("explain", None)
        stats_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(loaders, "APP_DATA_DIR", empty)
        st.cache_data.clear()
        at = run_page("app.views.explain")
        assert not at.exception
        assert at.info[0].value.startswith("The explanatory-variables bundle has not been built")
        st.cache_data.clear()


class TestSeriesFigureConsistency:
    def test_fitted_line_matches_stored_slope_and_intercept(self, bundle_dir):
        from src.cleaning import to_decimal_decades
        from src.figures import render_city_anomaly_series

        trends = loaders.load_city_trends()
        row = trends.iloc[0]
        series = loaders.load_city_series(int(row["city_id"]))
        fig = render_city_anomaly_series(
            series,
            slope=float(row["slope_c_per_decade"]),
            intercept=float(row["intercept"]),
        )
        fit_y = np.asarray(fig.data[2].y, dtype=float)
        decades = to_decimal_decades(series["dt"])
        expected = row["slope_c_per_decade"] * decades + row["intercept"]
        assert fit_y == pytest.approx(expected.to_numpy())
        # And the line actually runs through the data it was fit on.
        residuals = series["anomaly"].to_numpy() - fit_y
        assert abs(float(np.median(residuals))) < 0.05
