"""Tests for src.figures — figure builders shared by pipeline and dashboard.

Synthetic frames only. Also guards the module's import-lightness: the
Streamlit app imports src.figures at runtime (including on Streamlit
Community Cloud's minimal environment), so it must never pull in the heavy
pipeline dependencies.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cleaning import to_decimal_decades
from src.figures import (
    build_trend_map,
    render_city_anomaly_series,
    render_inequality_scatter,
    render_trend_surface,
)


def make_trends_frame():
    """Minimal per-city-location trends frame for map figures."""
    return pd.DataFrame(
        {
            "City": ["A", "B", "C"],
            "Country": ["X", "X", "Y"],
            "Latitude": [10.0, 20.0, -30.0],
            "Longitude": [5.0, 15.0, 40.0],
            "slope_c_per_decade": [0.1, -0.3, 0.2],
            "ci_low": [0.05, -0.4, 0.1],
            "ci_high": [0.15, -0.2, 0.3],
        }
    )


def make_surface():
    """A 2x2 surface with one NaN (ocean) cell; max |value| = 0.4."""
    lons = np.array([0.0, 10.0])
    lats = np.array([0.0, 10.0])
    values = np.array([[0.2, -0.1], [np.nan, 0.4]])
    return lons, lats, values


def make_inequality_frame():
    """Six countries on two continents with monotone emissions/warming."""
    return pd.DataFrame(
        {
            "Country": [f"C{i}" for i in range(6)],
            "continent": ["Africa"] * 3 + ["Europe"] * 3,
            "n_cities": [2, 3, 4, 5, 6, 7],
            "trend_c_per_decade": [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
            "cum_co2_t_per_capita": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
        }
    )


class TestImportLightness:
    def test_module_avoids_heavy_pipeline_imports(self):
        # The dashboard's cloud environment installs only the app
        # requirements; importing src.figures there must not require the
        # pipeline stack. (px's OLS trendline imports statsmodels lazily at
        # call time, which the app environment does provide.)
        code = (
            "import sys; import src.figures; "
            "heavy = {'duckdb', 'scipy', 'geopandas', 'pykrige', 'shapely', "
            "'statsmodels', 'kagglehub'} "
            "& {m.split('.')[0] for m in sys.modules}; "
            "assert not heavy, f'heavy imports leaked: {heavy}'"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


class TestRenderTrendSurface:
    def test_returns_figure_with_diverging_colorscale(self):
        lons = np.array([0.0, 1.0])
        lats = np.array([0.0, 1.0])
        values = np.array([[-1.0, 2.0], [np.nan, 0.5]])
        fig = render_trend_surface(lons, lats, values)
        heatmap = fig.data[0]
        assert heatmap.colorscale is not None
        assert heatmap.zmid == 0.0
        assert heatmap.zmax == pytest.approx(2.0)
        assert heatmap.zmin == pytest.approx(-2.0)


class TestBuildTrendMap:
    def test_surface_mode(self):
        lons, lats, values = make_surface()
        fig = build_trend_map(lons, lats, values, make_trends_frame(), mode="surface")
        assert len(fig.data) == 1
        heatmap = fig.data[0]
        assert heatmap.type == "heatmap"
        assert heatmap.zmax == pytest.approx(0.4)
        assert heatmap.zmin == pytest.approx(-0.4)

    def test_cities_mode_symmetric_marker_scale(self):
        lons, lats, values = make_surface()
        trends = make_trends_frame()
        fig = build_trend_map(lons, lats, values, trends, mode="cities")
        assert len(fig.data) == 1
        scatter = fig.data[0]
        assert scatter.type == "scatter"
        # Colored by slope, symmetric around zero: max |slope| = 0.3.
        assert scatter.marker.cmid == 0.0
        assert scatter.marker.cmax == pytest.approx(0.3)
        assert scatter.marker.cmin == pytest.approx(-0.3)
        assert np.asarray(scatter.customdata).shape == (len(trends), 5)

    def test_both_mode_overlays_neutral_points_on_shared_scale(self):
        lons, lats, values = make_surface()
        fig = build_trend_map(lons, lats, values, make_trends_frame(), mode="both")
        assert [trace.type for trace in fig.data] == ["heatmap", "scatter"]
        # Shared scale spans both layers: max(0.4 surface, 0.3 cities).
        assert fig.data[0].zmax == pytest.approx(0.4)
        # Overlaid markers are neutral (the surface already encodes value).
        assert isinstance(fig.data[1].marker.color, str)

    def test_unknown_mode_raises(self):
        lons, lats, values = make_surface()
        with pytest.raises(ValueError, match="mode"):
            build_trend_map(lons, lats, values, make_trends_frame(), mode="globe")


class TestRenderCityAnomalySeries:
    SLOPE = 0.5
    INTERCEPT = -97.0

    def make_series(self):
        months = pd.date_range("1950-01-01", "1959-12-01", freq="MS")
        decades = to_decimal_decades(pd.Series(months))
        return pd.DataFrame(
            {"dt": months, "anomaly": self.SLOPE * decades + self.INTERCEPT}
        )

    def test_three_traces_with_fit_from_slope_and_intercept(self):
        series = self.make_series()
        fig = render_city_anomaly_series(
            series, slope=self.SLOPE, intercept=self.INTERCEPT
        )
        assert len(fig.data) == 3
        fit = fig.data[2]
        # The anomalies were constructed exactly on the fitted line.
        assert np.asarray(fit.y) == pytest.approx(series["anomaly"].to_numpy())
        assert f"{self.SLOPE:+.3f}" in fit.name

    def test_sorts_unordered_input(self):
        shuffled = self.make_series().sample(frac=1.0, random_state=0)
        fig = render_city_anomaly_series(
            shuffled, slope=self.SLOPE, intercept=self.INTERCEPT
        )
        x = pd.to_datetime(np.asarray(fig.data[0].x))
        assert (x.sort_values() == x).all()

    def test_rolling_trace_smooths(self):
        fig = render_city_anomaly_series(
            self.make_series(), slope=self.SLOPE, intercept=self.INTERCEPT
        )
        rolling = np.asarray(fig.data[1].y, dtype=float)
        # Centered 12-month window: NaN at the edges, values in the middle.
        assert np.isnan(rolling[0])
        assert np.isfinite(rolling[len(rolling) // 2])


class TestRenderInequalityScatter:
    def test_figure_log_axis_and_trendline(self):
        fig = render_inequality_scatter(make_inequality_frame())
        assert fig.layout.xaxis.type == "log"
        modes = [trace.mode for trace in fig.data]
        assert any(mode == "lines" for mode in modes)  # pooled trendline
        # One marker trace per continent plus the trendline.
        assert len(fig.data) == 3
