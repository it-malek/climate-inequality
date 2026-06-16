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
_GATED_COLOR = "rgba(150, 150, 150, 0.4)"


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


def render_residual_map(
    df: pd.DataFrame,
    value_col: str = "mean_residual",
    gate_col: str = "gate_pass",
    title: str = "Out-of-sample forecast residuals, 2014-present (°C)",
) -> go.Figure:
    """Map of per-city mean forecast residuals (observed − predicted).

    Gate-passing cities are colored on the symmetric RdBu_r scale
    centered at zero (red = the stored trend underpredicted, i.e.
    observed warming ran ahead of the fit); gate-failing cities are
    faint grey context markers with their overlap r in the hover.

    Args:
        df: One row per city-location with Latitude, Longitude, City,
            Country, `value_col`, `gate_col`, overlap_r.
        value_col: Residual column to color by (°C).
        gate_col: Boolean agreement-gate column.
        title: Figure title.

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    passed = df.loc[df[gate_col]]
    failed = df.loc[~df[gate_col]]
    vmax = float(passed[value_col].abs().max())

    fig = go.Figure()
    if len(failed):
        fig.add_trace(
            go.Scatter(
                x=failed["Longitude"],
                y=failed["Latitude"],
                mode="markers",
                marker={"color": _GATED_COLOR, "size": 4, "symbol": "x"},
                name="gated out",
                customdata=np.column_stack(
                    [failed["City"], failed["Country"], failed["overlap_r"]]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}, %{customdata[1]}</b><br>"
                    "gated out (overlap r = %{customdata[2]:.2f})"
                    "<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=passed["Longitude"],
            y=passed["Latitude"],
            mode="markers",
            marker={
                "color": passed[value_col],
                "colorscale": TREND_COLORSCALE,
                "cmid": 0.0,
                "cmin": -vmax,
                "cmax": vmax,
                "size": 5,
                "colorbar": {"title": "°C"},
                "line": {"width": 0.4, "color": "rgba(60, 60, 60, 0.6)"},
            },
            name="cities",
            customdata=np.column_stack(
                [passed["City"], passed["Country"], passed[value_col]]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}, %{customdata[1]}</b><br>"
                "mean residual %{customdata[2]:.3f} °C<extra></extra>"
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


def render_validation_series(
    df: pd.DataFrame,
    forecast_start: str,
    date_col: str = "dt",
    observed_col: str = "observed",
    predicted_col: str = "predicted",
    rolling_months: int = 12,
    title: str = "Observed land anomaly vs the extrapolated stored trend",
) -> go.Figure:
    """Global mean observed anomalies against the extrapolated stored fit.

    Same three-trace style as :func:`render_city_anomaly_series`
    (monthly faint, rolling mean, fitted line); right of the
    `forecast_start` marker the prediction is genuinely out of sample.

    Args:
        df: One row per month with `date_col`, `observed_col`,
            `predicted_col`.
        forecast_start: First out-of-sample month, drawn as a vertical
            cutoff line.
        date_col: Date column name.
        observed_col: Observed global-mean anomaly column (°C).
        predicted_col: Extrapolated-prediction column (°C).
        rolling_months: Window for the centered rolling mean.
        title: Figure title.

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    work = df.sort_values(date_col)
    dates = pd.to_datetime(work[date_col])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=work[observed_col],
            mode="lines",
            line={"color": _MONTHLY_COLOR, "width": 1},
            name="Observed monthly mean",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=work[observed_col].rolling(rolling_months, center=True).mean(),
            mode="lines",
            line={"color": _ROLLING_COLOR, "width": 2},
            name=f"{rolling_months}-month mean",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=work[predicted_col],
            mode="lines",
            line={"color": _FIT_COLOR, "width": 2, "dash": "dash"},
            name="Stored fit (extrapolated)",
        )
    )
    cutoff = pd.Timestamp(forecast_start)
    fig.add_shape(
        type="line",
        x0=cutoff,
        x1=cutoff,
        y0=0,
        y1=1,
        yref="paper",
        line={"color": "rgba(60, 60, 60, 0.7)", "dash": "dot", "width": 1.5},
    )
    fig.add_annotation(
        x=cutoff,
        y=1.0,
        yref="paper",
        text="out of sample →",
        showarrow=False,
        xanchor="left",
        font={"size": 11, "color": "rgba(60, 60, 60, 0.9)"},
    )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Anomaly vs 1951–1980 (°C)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        height=440,
    )
    return fig


def render_coefficient_stability(
    specs: list[dict],
    emissions_key: str = "emissions",
    title: str = "Emissions coefficient across model specifications",
) -> go.Figure:
    """Horizontal forest plot of the log10_emissions coefficient per spec.

    Each row is one country-level model spec; the x-axis is the
    log10_emissions coefficient with its 95% CI as error bars. A dotted
    vertical line at x=0 and significance coloring (CI excludes zero vs
    not) make the attenuation story readable at a glance.

    Args:
        specs: List of country-model spec dicts, each with ``spec_name``
            and an ``emissions`` sub-dict (``coef``, ``ci_low``,
            ``ci_high``, ``p_value``). From
            ``stats["explain"]["country_model"]["specs"]``.
        emissions_key: Key in each spec dict for the emissions effect.
        title: Figure title.

    Returns:
        A `plotly.graph_objects.Figure`.
    """
    names, coefs, err_plus, err_minus, colors, hovers = [], [], [], [], [], []
    for spec in specs:
        em = spec.get(emissions_key)
        if em is None:
            continue
        coef = em["coef"]
        ci_low, ci_high = em["ci_low"], em["ci_high"]
        sig = ci_low > 0 or ci_high < 0
        names.append(spec["spec_name"])
        coefs.append(coef)
        err_plus.append(ci_high - coef)
        err_minus.append(coef - ci_low)
        colors.append(_ROLLING_COLOR if sig else _GATED_COLOR)
        hovers.append(
            f"{spec['spec_name']}: {coef:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]"
            f"  p={em['p_value']:.3g}"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=coefs,
            y=names,
            mode="markers",
            marker={
                "size": 10,
                "color": colors,
                "line": {"width": 1.2, "color": "rgba(40,40,40,0.6)"},
            },
            error_x={
                "type": "data",
                "array": err_plus,
                "arrayminus": err_minus,
                "thickness": 1.8,
                "width": 6,
                "color": "rgba(80,80,80,0.7)",
            },
            text=hovers,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.add_shape(
        type="line",
        x0=0, x1=0, y0=-0.5, y1=len(names) - 0.5,
        line={"color": "rgba(60,60,60,0.6)", "dash": "dot", "width": 1.5},
    )
    fig.update_layout(
        title=title,
        xaxis_title="°C/decade per 10× cumulative per-capita CO₂",
        yaxis={"autorange": "reversed"},
        height=360,
    )
    return fig


def render_partial_effect_scatter(
    features: pd.DataFrame,
    x_col: str = "abs_latitude",
    y_col: str = "slope_c_per_decade",
    color_col: str = "koppen",
    title: str = "Warming rate vs |latitude|, by climate class",
) -> go.Figure:
    """Scatter of warming trend vs absolute latitude, colored by Köppen class.

    Mirrors the :func:`render_inequality_scatter` pattern: colorblind-safe
    qualitative palette, overall OLS trendline (statsmodels backend via
    plotly express), hover on city names.

    Args:
        features: Slim city-features frame with ``City``, ``Country``,
            ``abs_latitude``, ``slope_c_per_decade``, ``koppen``.
        x_col, y_col, color_col: Axis and color columns.
        title: Figure title.

    Returns:
        A plotly Figure.
    """
    return px.scatter(
        features,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_name="City",
        hover_data={"Country": True},
        trendline="ols",
        trendline_scope="overall",
        trendline_color_override="gray",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={
            x_col: "|Latitude| (degrees)",
            y_col: "Warming trend (°C/decade)",
            color_col: "Köppen class",
        },
        title=title,
    )


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
