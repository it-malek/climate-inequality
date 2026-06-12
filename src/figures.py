"""Plotly figure builders shared by the analysis pipeline and the dashboard.

Import-light by design: the Streamlit app (``app/``) imports this module at
runtime, including on Streamlit Community Cloud, so it may depend only on
numpy/pandas/plotly plus :mod:`src.cleaning` -- never duckdb, scipy,
geopandas, or pykrige. Heavy computation happens upstream (``src.trends``,
``src.interpolate``, ``src.app_assets``); functions here only draw.

Conventions (README): trend maps use the colorblind-safe diverging
RdBu_r scale centered at 0; categorical colors use plotly's Safe palette.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.cleaning import to_decimal_decades

TREND_COLORSCALE = "RdBu_r"
MAP_MODES = ("surface", "cities", "both")

# Okabe-Ito colorblind-safe colors for the anomaly time-series traces.
_MONTHLY_COLOR = "rgba(135, 135, 135, 0.45)"
_ROLLING_COLOR = "#0072B2"
_FIT_COLOR = "#D55E00"


def render_trend_surface(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    values: np.ndarray,
    title: str = "Warming trend (°C/decade)",
) -> go.Figure:
    """Plotly heatmap of an interpolated trend surface.

    Uses a diverging RdBu_r colorscale centered at zero (warming positive
    in red, cooling negative in blue), per README plotting conventions.

    Args:
        grid_lon, grid_lat: 1D axes of grid-cell centers, in degrees.
        values: 2D array of shape (len(grid_lat), len(grid_lon)); NaN
            cells (e.g. ocean, masked by ``mask_to_land``) render blank.
        title: Figure title.

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    vmax = float(np.nanmax(np.abs(values)))
    fig = go.Figure(
        data=go.Heatmap(
            x=grid_lon,
            y=grid_lat,
            z=values,
            colorscale=TREND_COLORSCALE,
            zmid=0.0,
            zmin=-vmax,
            zmax=vmax,
            colorbar={"title": "°C/decade"},
        )
    )
    fig.update_layout(title=title, xaxis_title="Longitude", yaxis_title="Latitude")
    return fig


def build_trend_map(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    surface: np.ndarray,
    trends: pd.DataFrame,
    mode: str = "surface",
    value_col: str = "slope_c_per_decade",
    title: str = "Land warming trend (°C/decade)",
) -> go.Figure:
    """Trend map composed of an interpolated surface and/or city points.

    All layers share one symmetric RdBu_r scale centered at zero. In
    ``"both"`` mode the city markers are small neutral dots (the surface
    already encodes the value; the overlay shows where the surface is
    actually supported by stations), with the per-city hover retained.

    Args:
        grid_lon, grid_lat: 1D axes of grid-cell centers, in degrees.
        surface: 2D interpolated values, NaN over ocean.
        trends: Per-city-location frame with Latitude, Longitude,
            `value_col`, ci_low, ci_high, City, Country.
        mode: "surface", "cities", or "both".
        value_col: Trend column to color city markers by.
        title: Figure title.

    Returns:
        A `plotly.graph_objects.Figure`.

    Raises:
        ValueError: for an unknown `mode`.
    """
    if mode not in MAP_MODES:
        raise ValueError(f"unknown mode: {mode!r}; expected one of {MAP_MODES}")
    show_surface = mode in ("surface", "both")
    show_cities = mode in ("cities", "both")

    vmax_candidates = []
    if show_surface:
        vmax_candidates.append(float(np.nanmax(np.abs(surface))))
    if show_cities:
        vmax_candidates.append(float(trends[value_col].abs().max()))
    vmax = max(vmax_candidates)

    fig = go.Figure()
    if show_surface:
        fig.add_trace(
            go.Heatmap(
                x=grid_lon,
                y=grid_lat,
                z=surface,
                colorscale=TREND_COLORSCALE,
                zmid=0.0,
                zmin=-vmax,
                zmax=vmax,
                colorbar={"title": "°C/decade"},
                name="surface",
                hovertemplate=(
                    "(%{y:.0f}°, %{x:.0f}°): %{z:.3f} °C/decade<extra></extra>"
                ),
            )
        )
    if show_cities:
        if show_surface:
            marker = {"color": "rgba(30, 30, 30, 0.55)", "size": 4}
        else:
            marker = {
                "color": trends[value_col],
                "colorscale": TREND_COLORSCALE,
                "cmid": 0.0,
                "cmin": -vmax,
                "cmax": vmax,
                "size": 5,
                "colorbar": {"title": "°C/decade"},
                "line": {"width": 0.4, "color": "rgba(60, 60, 60, 0.6)"},
            }
        customdata = np.column_stack(
            [
                trends["City"],
                trends["Country"],
                trends[value_col],
                trends["ci_low"],
                trends["ci_high"],
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=trends["Longitude"],
                y=trends["Latitude"],
                mode="markers",
                marker=marker,
                name="cities",
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}, %{customdata[1]}</b><br>"
                    "%{customdata[2]:.3f} °C/decade "
                    "[%{customdata[3]:.3f}, %{customdata[4]:.3f}]"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        showlegend=False,
        height=520,
    )
    return fig


def render_city_anomaly_series(
    series: pd.DataFrame,
    slope: float,
    intercept: float,
    title: str = "Monthly temperature anomaly",
    date_col: str = "dt",
    value_col: str = "anomaly",
    rolling_months: int = 12,
) -> go.Figure:
    """One city's anomaly time series with its fitted Theil-Sen trend line.

    Three traces: the monthly anomalies (faint), a centered rolling mean
    for readability, and the fitted trend evaluated from `slope` and
    `intercept` on the :func:`src.cleaning.to_decimal_decades` axis -- the
    same axis the slope was fit on, so the line is exactly the Phase 2 fit.

    Args:
        series: One city's rows with `date_col` and `value_col`.
        slope: Fitted trend in °C/decade.
        intercept: Fitted intercept on the decimal-decades axis.
        title: Figure title.
        date_col: Date column name.
        value_col: Anomaly column name.
        rolling_months: Window for the centered rolling mean.

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    work = series.sort_values(date_col)
    dates = pd.to_datetime(work[date_col])
    fit = slope * to_decimal_decades(work[date_col]) + intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=work[value_col],
            mode="lines",
            line={"color": _MONTHLY_COLOR, "width": 1},
            name="Monthly anomaly",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=work[value_col].rolling(rolling_months, center=True).mean(),
            mode="lines",
            line={"color": _ROLLING_COLOR, "width": 2},
            name=f"{rolling_months}-month mean",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=fit,
            mode="lines",
            line={"color": _FIT_COLOR, "width": 2, "dash": "dash"},
            name=f"Theil–Sen fit ({slope:+.3f} °C/decade)",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Anomaly vs monthly climatology (°C)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        height=440,
    )
    return fig


def render_inequality_scatter(
    df: pd.DataFrame,
    x_col: str = "cum_co2_t_per_capita",
    y_col: str = "trend_c_per_decade",
    title: str = "Warming rate vs cumulative emissions responsibility",
) -> go.Figure:
    """Scatter of country warming trends against per-capita emissions.

    Log x-axis, colorblind-safe qualitative palette per continent (the
    categorical counterpart of the README diverging-map convention),
    marker area by the number of city-locations behind each country mean,
    and one pooled OLS trendline fit on log10(x).

    Args:
        df: One row per country (see :func:`src.emissions.join_country_data`).
        x_col, y_col: Axis columns.
        title: Figure title.

    Returns:
        A plotly Figure.
    """
    return px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="continent",
        size="n_cities",
        size_max=18,
        hover_name="Country",
        log_x=True,
        trendline="ols",
        trendline_options={"log_x": True},
        trendline_scope="overall",
        trendline_color_override="gray",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={
            x_col: "Cumulative CO₂ per capita (t/person, log scale)",
            y_col: "Warming trend (°C/decade)",
            "continent": "Continent",
            "n_cities": "city-locations",
        },
        title=title,
    )
