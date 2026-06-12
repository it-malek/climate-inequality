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
        assert at.title[0].value == "Where has land warmed fastest?"


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
