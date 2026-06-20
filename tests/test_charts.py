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


class TestSensitivityCharts:
    def test_coef_ci_chart_has_error_bars(self):
        rows = [
            {"df": 4, "coef": 0.03, "ci_low": 0.01, "ci_high": 0.05},
            {"df": 6, "coef": 0.02, "ci_low": 0.00, "ci_high": 0.04},
        ]
        fig = charts.coef_ci_chart(rows, label_key="df", title="t")
        assert fig.data[0].error_x.array is not None
        assert list(fig.data[0].y) == ["4", "6"]

    def test_dfbeta_bar(self):
        fig = charts.dfbeta_bar([("Russia", 0.012), ("Canada", -0.008)])
        assert list(fig.data[0].y) == ["Russia", "Canada"]
