"""Unit tests for the dashboard chart builders and shared theme."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from app import charts, theme

SHARES = {"emissions": 0.08, "geography": 0.46, "socioeconomic": 0.06, "population": 0.04}
RESIDUAL = 0.36
SUMMARY = {
    "total_r2": 0.64,
    "residual_share": RESIDUAL,
    "shares": SHARES,
    "univariate_r2": {"emissions": 0.13, "geography": 0.58, "socioeconomic": 0.08, "population": 0.05},
    "group_features": {"emissions": ["cum_co2_per_capita"]},
}


class TestTheme:
    def test_every_group_has_a_color_and_order_is_complete(self):
        assert set(theme.GROUP_COLORS) == set(theme.GROUP_ORDER)
        assert len(theme.GROUP_ORDER) == 5
        assert theme.GROUP_ORDER[-1] == "residual"

    def test_group_color_fallback(self):
        assert theme.group_color("emissions") == theme.GROUP_COLORS["emissions"]
        assert theme.group_color("nonexistent") == "#9A9A9A"

    def test_apply_base_layout_sets_template(self):
        fig = theme.apply_base_layout(go.Figure())
        assert fig.layout.template is not None


class TestSharesBar:
    def test_includes_residual_and_uses_group_colors(self):
        fig = charts.shares_bar(SHARES, RESIDUAL)
        bar = fig.data[0]
        assert len(bar.x) == 5  # 4 named groups + residual
        assert pytest.approx(sum(bar.x), abs=1e-9) == sum(SHARES.values()) + RESIDUAL
        # Residual bar is the grey from the theme (consistent mapping).
        assert theme.GROUP_COLORS["residual"] in tuple(bar.marker.color)
        assert theme.GROUP_COLORS["geography"] in tuple(bar.marker.color)


class TestExplorerBar:
    def test_full_view_stacks_groups_plus_residual(self):
        fig = charts.explorer_bar(SUMMARY, "full")
        assert fig.layout.barmode == "stack"
        assert len(fig.data) == 5

    def test_single_axis_view_is_two_segments(self):
        fig = charts.explorer_bar(SUMMARY, "emissions")
        assert len(fig.data) == 2
        widths = [trace.x[0] for trace in fig.data]
        assert pytest.approx(sum(widths), abs=1e-9) == 1.0  # explained + unexplained


COUPLING_TABLE = pd.DataFrame(
    {
        "Country": ["A", "B", "C", "D"],
        "responsibility_index_v1": [1.0, 5.0, 20.0, 50.0],
        "impact_index_v1": [0.10, 0.20, 0.12, 0.25],
        "responsibility_rank": [4, 3, 2, 1],
        "impact_rank": [4, 2, 3, 1],
        "rank_gap": [0, -1, 1, 0],
        "z_gap": [0.1, -0.3, 0.4, -0.2],
    }
)


class TestCouplingCharts:
    def test_lorenz_chart_has_diagonal_and_curve(self):
        fig = charts.lorenz_chart(COUPLING_TABLE)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # equality diagonal + Lorenz step curve

    def test_mismatch_scatter_colors_by_z_gap(self):
        fig = charts.mismatch_scatter(COUPLING_TABLE)
        assert isinstance(fig, go.Figure)
        assert tuple(fig.data[0].marker.color) == tuple(COUPLING_TABLE["z_gap"])
        assert fig.data[0].marker.cmid == 0.0


class TestChoropleth:
    def test_builds_choropleth_with_hover_data(self):
        df = pd.DataFrame(
            {
                "location": ["Norway", "Kenya"],
                "warming_trend": [0.30, 0.10],
                "cum_co2_t_per_capita": [80.0, 1.0],
                "mean_latitude": [62.0, 1.5],
            }
        )
        fig = charts.warming_choropleth(df)
        trace = fig.data[0]
        assert isinstance(trace, go.Choropleth)
        assert list(trace.locations) == ["Norway", "Kenya"]
        assert trace.colorscale is not None


VULN_SUMMARY = {
    "income_order": [
        "Low-income countries", "Lower-middle-income countries",
        "Upper-middle-income countries", "High-income countries",
    ],
    "responsibility": {
        "by_tier": {
            "Low-income countries": {"mean": 0.8, "median": 0.8},
            "Lower-middle-income countries": {"mean": 5.5, "median": 5.5},
            "Upper-middle-income countries": {"mean": 32.0, "median": 32.0},
            "High-income countries": {"mean": 85.0, "median": 85.0},
        }
    },
    "exposure": {
        "area": {
            "by_tier": {
                "Low-income countries": {"pop_weighted_mean": 0.205, "median": 0.205},
                "Lower-middle-income countries": {"pop_weighted_mean": 0.198, "median": 0.198},
                "Upper-middle-income countries": {"pop_weighted_mean": 0.168, "median": 0.168},
                "High-income countries": {"pop_weighted_mean": 0.182, "median": 0.182},
            }
        }
    },
}

VULN_STRATA = pd.DataFrame({
    "owid_country": ["A", "B", "C", "D"],
    "income_group": [
        "Low-income countries", "Low-income countries",
        "High-income countries", "High-income countries",
    ],
    "trend_c_per_decade_area_weighted": [0.21, 0.20, 0.18, 0.19],
})


class TestVulnerabilityCharts:
    def test_gradient_chart_has_bars_and_line_on_two_axes(self):
        fig = charts.income_gradient_chart(VULN_SUMMARY, lens="area")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # responsibility bars + warming line
        bar, line = fig.data
        # Responsibility climbs steeply (left axis); warming is on the secondary axis.
        assert list(bar.y) == [0.8, 5.5, 32.0, 85.0]
        assert line.yaxis == "y2"
        assert list(line.x) == ["Low", "Lower-mid", "Upper-mid", "High"]

    def test_strata_box_orders_tiers_low_to_high(self):
        fig = charts.income_strata_box(
            VULN_STRATA, "trend_c_per_decade_area_weighted",
            VULN_SUMMARY["income_order"], title="Area warming by income group",
        )
        # Only the two populated tiers appear, in low -> high order.
        assert [b.name for b in fig.data] == ["Low", "High"]

    def test_strata_box_accepts_custom_group_col(self):
        strata = VULN_STRATA.assign(vuln_quartile=["Q1 (least)", "Q1 (least)",
                                                   "Q4 (most)", "Q4 (most)"])
        fig = charts.income_strata_box(
            strata, "trend_c_per_decade_area_weighted",
            ["Q1 (least)", "Q4 (most)"], title="by quartile",
            group_col="vuln_quartile", xaxis_title="vulnerability quartile",
        )
        assert [b.name for b in fig.data] == ["Q1 (least)", "Q4 (most)"]

    def test_ndgain_scatter_drops_missing_and_colors_by_warming(self):
        strata = pd.DataFrame({
            "owid_country": ["A", "B", "C"],
            "vulnerability": [0.7, 0.3, None],  # C dropped (no score)
            "cum_co2_t_per_capita": [1.0, 50.0, 80.0],
            "trend_c_per_decade_area_weighted": [0.21, 0.19, 0.18],
        })
        fig = charts.ndgain_scatter(strata)
        assert isinstance(fig, go.Figure)
        assert list(fig.data[0].x) == [0.7, 0.3]  # only finite-vulnerability rows
        assert tuple(fig.data[0].marker.color) == (0.21, 0.19)
        assert fig.layout.yaxis.type == "log"


class TestResidualCharts:
    def test_city_latitude_scatter_has_points_and_band_medians(self):
        features = pd.DataFrame({
            "City": [f"c{i}" for i in range(30)],
            "Country": ["X"] * 30,
            "Latitude": [5.0 + i for i in range(30)],
            "slope_c_per_decade": [0.10 + 0.002 * i for i in range(30)],
        })
        fig = charts.city_latitude_scatter(features)
        assert len(fig.data) == 2
        assert len(fig.data[0].x) == 30
        # Only the 10-20 and 20-30 degree bands hold ten locations, so two medians.
        assert len(fig.data[1].x) == 2

    def test_responsibility_panels_one_trace_per_definition(self):
        national = pd.DataFrame({
            "Country": ["A", "B", "C"],
            "cum_co2_t_per_capita": [1.0, 10.0, 100.0],
            "trend_station": [0.1, 0.2, 0.3],
            "trend_population": [0.1, 0.2, 0.3],
            "trend_area": [0.3, 0.2, None],
            "trend_era5_aligned": [0.2, 0.2, 0.2],
        })
        comparison = [
            {"column": c, "label": c, "n": 3, "spearman_rho": 0.5, "spearman_p": 0.5}
            for c in ["trend_station", "trend_population", "trend_area", "trend_era5_aligned"]
        ]
        fig = charts.responsibility_panels(national, comparison)
        assert len(fig.data) == 4
        assert len(fig.data[2].x) == 2  # the missing area value is dropped

    def test_delta_rmse_forest_colors_by_interval_sign(self):
        rows = [
            {"label": "worse", "delta_rmse": 0.002, "interval": [0.001, 0.003]},
            {"label": "unclear", "delta_rmse": -0.001, "interval": [-0.003, 0.001]},
            {"label": "better", "delta_rmse": -0.003, "interval": [-0.005, -0.001]},
        ]
        fig = charts.delta_rmse_forest(rows, static_rmse=0.043)
        colors = tuple(fig.data[0].marker.color)
        assert colors[0] == theme.GROUP_COLORS["emissions"]
        assert colors[2] == theme.GROUP_COLORS["geography"]
        assert colors[1] == theme.NEUTRAL_STRONG

    def test_robustness_forest_two_families_and_product_divider(self):
        settings = [
            {"label": "a", "product": "Berkeley Earth",
             "spatial_error": {"delta_rmse": -0.003, "interval": [-0.005, -0.001]},
             "spatial_lag": {"delta_rmse": -0.004, "interval": [-0.006, -0.002]}},
            {"label": "b", "product": "ERA5",
             "spatial_error": {"delta_rmse": 0.001, "interval": [-0.001, 0.003]},
             "spatial_lag": {"delta_rmse": 0.0, "interval": [-0.002, 0.002]}},
        ]
        fig = charts.robustness_forest(settings)
        assert [t.name for t in fig.data] == ["Spatial error model", "Spatial lag model"]
        assert any("product" in (a.text or "") for a in fig.layout.annotations)

    def test_accounting_bar_stacks_to_one(self):
        spatial = {
            "spatial_error": {"station": {"accounting": {"a_static": 0.6, "a_dependence": 0.15, "a_innovation": 0.25}}},
            "spatial_lag": {"station": {"accounting": {"a_static": 0.6, "a_dependence": 0.16, "a_innovation": 0.24}}},
        }
        fig = charts.accounting_bar(spatial)
        assert fig.layout.barmode == "stack"
        totals = [sum(trace.x[i] for trace in fig.data) for i in range(2)]
        assert totals == pytest.approx([1.0, 1.0])

    def test_choropleth_accepts_a_value_column(self):
        df = pd.DataFrame({
            "location": ["Norway", "Kenya"], "trend_area": [0.30, 0.10],
            "cum_co2_t_per_capita": [80.0, 1.0], "mean_latitude": [62.0, 1.5],
        })
        fig = charts.warming_choropleth(df, value_col="trend_area", value_label="Area-weighted")
        assert list(fig.data[0].z) == [0.30, 0.10]


def test_chart_presentation_preserves_values_and_wraps_labels(monkeypatch):
    """Long labels must fit narrow plots without changing plotted values."""
    shown = []
    monkeypatch.setattr(theme.st, "markdown", lambda value: None)
    monkeypatch.setattr(theme.st, "plotly_chart", lambda fig, **kwargs: shown.append(fig))
    fig = go.Figure(go.Scatter(x=[0.1, 0.2], y=["Short", "A long scientific category label"]))
    fig.update_layout(title="An analytical chart title", xaxis_title="Share of total cross-country variance")
    theme.plotly_chart(fig, width="stretch")
    assert list(shown[0].data[0].x) == [0.1, 0.2]
    assert list(shown[0].data[0].y) == ["Short", "A long scientific category label"]
    assert shown[0].layout.title.text == ""
    assert "<br>" in shown[0].layout.yaxis.ticktext[1]
    assert fig.layout.title.text == "An analytical chart title"
