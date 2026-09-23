"""Plotly figure builders for the dashboard (no Streamlit, unit-testable).

Each function takes plain data (dicts or DataFrames loaded by
:mod:`app.loaders`) and returns a ``go.Figure`` styled through
:func:`app.theme.apply_base_layout`, so color and layout stay consistent
across the interface. Streamlit calls live only in the view modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app import theme


def shares_bar(shares: dict[str, float], residual_share: float) -> go.Figure:
    """Horizontal bar of Shapley variance shares, residual included.

    Bars follow :data:`app.theme.GROUP_ORDER` and use the fixed group colors;
    labels are the share of total variance.
    """
    full = {**shares, "residual": residual_share}
    keys = [k for k in theme.GROUP_ORDER if k in full]
    values = [full[k] for k in keys]
    labels = [theme.GROUP_LABELS[k] for k in keys]
    colors = [theme.group_color(k) for k in keys]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors},
            text=[f"{v:.0%}" for v in values],
            textposition="outside",
            customdata=[theme.GROUP_GLOSS[k] for k in keys],
            hovertemplate="<b>%{y}</b><br>%{x:.1%} of total variance<br>"
            "<span style='font-size:0.8em'>%{customdata}</span><extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"title": "Share of total cross-country variance", "tickformat": ".0%"},
        yaxis={"autorange": "reversed"},
        height=320,
        showlegend=False,
    )
    return theme.apply_base_layout(fig)


def explorer_bar(summary: dict, view: str) -> go.Figure:
    """Stacked 'variance explained' bar for one explorer view.

    ``view`` is a group key (that group's standalone R squared against the
    unexplained remainder) or ``"full"`` (the group shares stacked, then the
    residual). One row, segments to 100%.
    """
    total_r2 = float(summary["total_r2"])
    if view == "full":
        segments = [
            (theme.GROUP_LABELS[k], summary["shares"][k], theme.group_color(k))
            for k in theme.GROUP_ORDER
            if k in summary["shares"]
        ]
        segments.append(("Residual", summary["residual_share"], theme.group_color("residual")))
        subtitle = f"Full model: {total_r2:.0%} of variance explained"
    else:
        r2 = float(summary["univariate_r2"][view])
        segments = [
            (f"{theme.GROUP_LABELS[view]} alone", r2, theme.group_color(view)),
            ("Unexplained", 1.0 - r2, theme.group_color("residual")),
        ]
        subtitle = f"{theme.GROUP_LABELS[view]} alone: {r2:.0%} of variance explained"

    fig = go.Figure()
    for label, value, color in segments:
        fig.add_bar(
            x=[value],
            y=["variance"],
            orientation="h",
            name=label,
            marker={"color": color},
            text=[f"{label} {value:.0%}" if value >= 0.06 else ""],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"<b>{label}</b><br>%{{x:.1%}} of total variance<extra></extra>",
        )
    fig.update_layout(
        barmode="stack",
        xaxis={"range": [0, 1], "tickformat": ".0%", "title": subtitle},
        yaxis={"visible": False},
        height=170,
        bargap=0.4,
    )
    return theme.apply_base_layout(fig)


def warming_choropleth(
    df: pd.DataFrame,
    value_col: str = "warming_trend",
    value_label: str = "Warming trend",
) -> go.Figure:
    """World choropleth of a national warming rate, with responsibility in the hover.

    ``df`` has one row per country with ``location`` (a country name for
    Plotly's ``country names`` mode), ``value_col``, ``cum_co2_t_per_capita``
    and ``mean_latitude``.
    """
    customdata = np.stack(
        [
            df["cum_co2_t_per_capita"].to_numpy(),
            df["mean_latitude"].to_numpy(),
        ],
        axis=-1,
    )
    fig = go.Figure(
        go.Choropleth(
            locations=df["location"],
            locationmode="country names",
            z=df[value_col],
            customdata=customdata,
            colorscale=theme.TREND_COLORSCALE,
            colorbar={"title": theme.TREND_UNIT},
            marker_line_color=theme.NEUTRAL_MID,
            marker_line_width=0.3,
            hovertemplate=(
                "<b>%{location}</b><br>"
                + value_label + ": %{z:.3f} " + theme.TREND_UNIT + "<br>"
                "Cumulative CO₂: %{customdata[0]:,.0f} t/person<br>"
                "Mean latitude: %{customdata[1]:.0f}°<extra></extra>"
            ),
        )
    )
    fig.update_geos(
        showframe=False,
        showcoastlines=False,
        projection_type="natural earth",
        bgcolor="rgba(0,0,0,0)",
    )
    return theme.apply_base_layout(fig, height=520, margin={"l": 0, "r": 0, "t": 10, "b": 0})


def city_latitude_scatter(features: pd.DataFrame, band_width: float = 10.0) -> go.Figure:
    """City-location warming trends against signed latitude, with band medians.

    Every location is a faint point; the line joins the median trend within
    each ``band_width``-degree latitude band that holds at least ten
    locations. The line is a summary of the sample, not a fitted model.
    """
    lat = features["Latitude"].to_numpy(dtype=float)
    slope = features["slope_c_per_decade"].to_numpy(dtype=float)
    edges = np.arange(-60.0, 80.0 + 1e-9, band_width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    medians, counts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (lat >= lo) & (lat < hi)
        counts.append(int(mask.sum()))
        medians.append(float(np.median(slope[mask])) if mask.any() else np.nan)
    keep = np.array(counts) >= 10

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=lat, y=slope, mode="markers", name="city location",
            marker={"color": theme.NEUTRAL_MID, "size": 4, "opacity": 0.5},
            customdata=np.stack([features["City"].to_numpy(), features["Country"].to_numpy()], axis=-1),
            hovertemplate="<b>%{customdata[0]}, %{customdata[1]}</b><br>"
            "latitude %{x:.1f}°<br>trend %{y:.3f} " + theme.TREND_UNIT + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=centres[keep], y=np.array(medians)[keep], mode="lines+markers",
            name=f"median within {band_width:g}° bands (10 or more locations)",
            line={"color": theme.group_color("emissions"), "width": 2.5},
            marker={"size": 8, "line": {"color": "#fff", "width": 1}},
            customdata=np.array(counts)[keep],
            hovertemplate="band centre %{x:.0f}°<br>median %{y:.3f} " + theme.TREND_UNIT
            + "<br>%{customdata} locations<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line={"color": theme.NEUTRAL_FAINT, "width": 1})
    fig.update_layout(
        title="City-location warming trends by latitude",
        xaxis={"title": "Latitude (degrees; negative is south)", "range": [-62, 82]},
        yaxis={"title": f"Theil\u2013Sen trend, 1950\u20132013 ({theme.TREND_UNIT})"},
        height=440,
        showlegend=True,
    )
    return theme.apply_base_layout(fig)


def responsibility_panels(national: pd.DataFrame, comparison: list[dict]) -> go.Figure:
    """Four panels of national warming against cumulative per-capita CO₂.

    One panel per definition of national warming (``comparison`` rows carry
    the column, label, sample size and Spearman statistics computed at bundle
    build time). Responsibility is on a log axis; every panel shares the same
    warming axis so the collapse of the rank correlation under area
    weighting is visible directly.
    """
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.06, vertical_spacing=0.14,
        subplot_titles=[row["label"] for row in comparison],
    )
    for i, row in enumerate(comparison):
        r, c = divmod(i, 2)
        frame = national.dropna(subset=["cum_co2_t_per_capita", row["column"]])
        fig.add_trace(
            go.Scatter(
                x=frame["cum_co2_t_per_capita"], y=frame[row["column"]], mode="markers",
                name=row["label"], showlegend=False,
                marker={"color": theme.group_color("geography"), "size": 7, "opacity": 0.75,
                        "line": {"color": "#fff", "width": 0.5}},
                customdata=frame["Country"],
                hovertemplate="<b>%{customdata}</b><br>%{x:,.1f} t CO₂/person<br>"
                "%{y:.3f} " + theme.TREND_UNIT + "<extra></extra>",
            ),
            row=r + 1, col=c + 1,
        )
        p = row["spearman_p"]
        p_text = f"p = {p:.2f}" if p >= 0.001 else "p < 0.001"
        fig.add_annotation(
            text=f"n = {row['n']}<br>Spearman ρ = {row['spearman_rho']:+.2f} ({p_text})",
            xref=f"x{i + 1 if i else ''} domain", yref=f"y{i + 1 if i else ''} domain",
            x=0.98, y=0.04, xanchor="right", yanchor="bottom", showarrow=False,
            align="right", font={"size": 11},
            bgcolor="rgba(128,128,128,0.10)", borderpad=4,
        )
    fig.update_xaxes(type="log", title_text="Cumulative CO₂ per 2013 resident (t/person)", row=2)
    fig.update_yaxes(title_text=theme.TREND_UNIT, col=1)
    fig.update_layout(height=640, showlegend=False)
    fig.update_annotations(font_size=12)
    return theme.apply_base_layout(fig, margin={"l": 10, "r": 10, "t": 40, "b": 10})


def _interval_color(low: float, high: float) -> str:
    """Blue when the whole interval is an improvement, vermillion when a deterioration, grey otherwise."""
    if high < 0:
        return theme.group_color("geography")
    if low > 0:
        return theme.group_color("emissions")
    return theme.NEUTRAL_STRONG


def delta_rmse_forest(rows: list[dict], static_rmse: float) -> go.Figure:
    """Paired change in geographically separated RMSE for each tested extension.

    ``rows`` carry ``label``, ``delta_rmse`` and ``interval``. Negative is
    better transfer; the interval is the paired country bootstrap.
    """
    labels = [r["label"] for r in rows]
    deltas = [float(r["delta_rmse"]) for r in rows]
    lows = [float(r["interval"][0]) for r in rows]
    highs = [float(r["interval"][1]) for r in rows]
    colors = [_interval_color(lo, hi) for lo, hi in zip(lows, highs)]
    fig = go.Figure(
        go.Scatter(
            x=deltas, y=labels, mode="markers",
            marker={"color": colors, "size": 11, "line": {"color": "#fff", "width": 1}},
            error_x={
                "type": "data", "symmetric": False,
                "array": [h - d for h, d in zip(highs, deltas)],
                "arrayminus": [d - lo for d, lo in zip(deltas, lows)],
                "color": theme.NEUTRAL_MID, "thickness": 2.5, "width": 4,
            },
            customdata=np.stack([lows, highs], axis=-1),
            hovertemplate="<b>%{y}</b><br>change %{x:+.4f}<br>"
            "95% interval [%{customdata[0]:+.4f}, %{customdata[1]:+.4f}]<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line={"color": theme.NEUTRAL_STRONG, "width": 1.2})
    fig.update_layout(
        title="Change in geographically separated prediction error",
        xaxis={
            "title": f"RMSE change ({theme.TREND_UNIT})",
            "tickformat": "+.3f", "zeroline": False,
        },
        yaxis={"autorange": "reversed"},
        height=90 + 52 * max(len(rows), 1),
        showlegend=False,
    )
    return theme.apply_base_layout(fig)


def robustness_forest(settings: list[dict]) -> go.Figure:
    """Spatial-model gains across graphs, exclusion distances and products.

    ``settings`` rows carry ``label``, ``product`` and, per family key in
    :data:`app.theme.FAMILY_COLORS`, ``delta_rmse`` and ``interval``. Rows of
    a different product than the first are separated by a dashed line.
    """
    n = len(settings)
    positions = np.arange(n)[::-1].astype(float)
    fig = go.Figure()
    for family, offset in (("spatial_error", 0.17), ("spatial_lag", -0.17)):
        deltas = [float(s[family]["delta_rmse"]) for s in settings]
        lows = [float(s[family]["interval"][0]) for s in settings]
        highs = [float(s[family]["interval"][1]) for s in settings]
        fig.add_trace(
            go.Scatter(
                x=deltas, y=positions + offset, mode="markers",
                name=theme.FAMILY_LABELS[family],
                marker={"color": theme.FAMILY_COLORS[family], "size": 10,
                        "symbol": "circle" if family == "spatial_error" else "square",
                        "line": {"color": "#fff", "width": 1}},
                error_x={
                    "type": "data", "symmetric": False,
                    "array": [h - d for h, d in zip(highs, deltas)],
                    "arrayminus": [d - lo for d, lo in zip(deltas, lows)],
                    "color": theme.FAMILY_COLORS[family], "thickness": 2, "width": 3,
                },
                customdata=np.stack([[s["label"] for s in settings], lows, highs], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>" + theme.FAMILY_LABELS[family]
                + ": %{x:+.4f} [%{customdata[1]:+.4f}, %{customdata[2]:+.4f}]<extra></extra>",
            )
        )
    fig.add_vline(x=0.0, line={"color": theme.NEUTRAL_STRONG, "width": 1.2})
    first_product = settings[0]["product"] if settings else None
    for i, s in enumerate(settings):
        if s["product"] != first_product:
            fig.add_hline(y=positions[i] + 0.5, line={"color": theme.NEUTRAL_FAINT, "dash": "dash"})
            fig.add_annotation(
                x=1.0, xref="paper", y=positions[i] + 0.5, text="temperature product changes",
                showarrow=False, xanchor="right", yanchor="bottom", font={"size": 11, "color": theme.NEUTRAL_STRONG},
            )
            break
    fig.update_layout(
        title="Spatial-model gain by graph, exclusion distance and product",
        xaxis={"title": f"RMSE change ({theme.TREND_UNIT})", "tickformat": "+.3f", "zeroline": False},
        yaxis={"tickmode": "array", "tickvals": positions, "ticktext": [s["label"] for s in settings],
               "range": [-0.7, n - 0.3]},
        height=120 + 56 * max(n, 1),
        showlegend=True,
    )
    return theme.apply_base_layout(fig)


def accounting_bar(spatial: dict) -> go.Figure:
    """In-sample variance accounting of the two spatial models.

    Each row splits the outcome variance into the static part (the static
    model's R squared), the neighbour-conditional dependence part and the
    innovation part. The dependence part represents covariance, not a
    mechanism, and is not additive with any group share.
    """
    rows = [(theme.FAMILY_LABELS[k], spatial[k]["station"]["accounting"]) for k in ("spatial_error", "spatial_lag")]
    parts = [
        ("Static model", "a_static", theme.group_color("geography")),
        ("Spatial dependence", "a_dependence", "#CC79A7"),
        ("Innovation", "a_innovation", theme.group_color("residual")),
    ]
    fig = go.Figure()
    for label, key, color in parts:
        fig.add_bar(
            x=[float(acc[key]) for _, acc in rows], y=[name for name, _ in rows],
            orientation="h", name=label, marker={"color": color},
            text=[f"{float(acc[key]):.0%}" for _, acc in rows], textposition="inside",
            hovertemplate="<b>%{y}</b><br>" + label + " %{x:.1%} of outcome variance<extra></extra>",
        )
    fig.update_layout(
        barmode="stack",
        title="In-sample variance accounting of the spatial models",
        xaxis={"range": [0, 1], "tickformat": ".0%", "title": "Share of outcome variance"},
        yaxis={"autorange": "reversed"},
        height=230,
        showlegend=True,
    )
    return theme.apply_base_layout(fig)


def share_ci_chart(
    groups: dict[str, dict],
    block_groups: dict[str, dict] | None = None,
) -> go.Figure:
    """Bootstrap 95% interval of each variance share (one row per group).

    ``groups`` maps a group key to ``point``, ``ci_low`` and ``ci_high`` (the
    country-bootstrap interval). When ``block_groups`` is given, the wider
    continent-block-bootstrap interval is drawn as a faint band behind each
    point. These are shares of total variance; the chart shows the stability
    of a descriptive decomposition, never significance or causation.
    """
    keys = [k for k in theme.GROUP_ORDER if k in groups]
    labels = [theme.GROUP_LABELS[k] for k in keys]
    points = [float(groups[k]["point"]) for k in keys]
    lo = [float(groups[k]["ci_low"]) for k in keys]
    hi = [float(groups[k]["ci_high"]) for k in keys]
    colors = [theme.group_color(k) for k in keys]

    fig = go.Figure()
    if block_groups:
        bl = [float(block_groups.get(k, groups[k])["ci_low"]) for k in keys]
        bh = [float(block_groups.get(k, groups[k])["ci_high"]) for k in keys]
        fig.add_trace(
            go.Scatter(
                x=points, y=labels, mode="markers", name="continent block bootstrap",
                marker={"color": "rgba(0,0,0,0)", "size": 1},
                error_x={
                    "type": "data", "symmetric": False,
                    "array": [h - p for h, p in zip(bh, points)],
                    "arrayminus": [p - low for p, low in zip(points, bl)],
                    "color": theme.NEUTRAL_MID, "thickness": 11, "width": 0,
                },
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=points, y=labels, mode="markers", name="country bootstrap",
            marker={"color": colors, "size": 11, "line": {"color": "#fff", "width": 1}},
            error_x={
                "type": "data", "symmetric": False,
                "array": [h - p for h, p in zip(hi, points)],
                "arrayminus": [p - low for p, low in zip(points, lo)],
                "color": theme.NEUTRAL_MID, "thickness": 2,
            },
            customdata=np.stack([lo, hi], axis=-1),
            hovertemplate="<b>%{y}</b><br>share %{x:.1%}"
            "<br>95% interval [%{customdata[0]:.1%}, %{customdata[1]:.1%}]<extra></extra>",
        )
    )
    fig.update_layout(
        title="Variance shares with bootstrap 95% intervals",
        xaxis={"title": "Share of total cross-country variance", "tickformat": ".0%"},
        yaxis={"autorange": "reversed"},
        height=90 + 46 * max(len(keys), 1),
        showlegend=bool(block_groups),
    )
    return theme.apply_base_layout(fig)


def influence_bar(
    items: list[tuple[str, float]],
    title: str,
    color: str | None = None,
    value_label: str = "Change in share (leave-one-out)",
) -> go.Figure:
    """Horizontal bar of the most influential countries by absolute change in a share."""
    names = [str(n) for n, _ in items]
    vals = [float(v) for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=vals, y=names, orientation="h",
            marker={"color": color or theme.group_color("emissions")},
            hovertemplate="<b>%{y}</b><br>" + value_label + " %{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line={"dash": "dot", "color": theme.NEUTRAL_FAINT})
    fig.update_layout(
        title=title,
        xaxis={"title": value_label},
        yaxis={"autorange": "reversed"},
        height=70 + 30 * max(len(names), 1),
        showlegend=False,
    )
    return theme.apply_base_layout(fig)


def lorenz_chart(
    table: pd.DataFrame,
    responsibility_col: str = "responsibility_index_v1",
    impact_col: str = "impact_index_v1",
    title: str = "Cumulative warming exposure vs cumulative responsibility",
) -> go.Figure:
    """Lorenz-style curve: cumulative warming share against cumulative responsibility share.

    Countries are ordered by responsibility only; the empirical cumulative
    shares are drawn as a step curve against the equality diagonal. The area
    between the two is what the inequality coefficient summarizes.
    """
    work = table.sort_values(responsibility_col, kind="stable")
    r = work[responsibility_col].to_numpy(dtype=float)
    im = work[impact_col].to_numpy(dtype=float)
    cum_r = np.concatenate([[0.0], np.cumsum(r) / r.sum()])
    cum_i = np.concatenate([[0.0], np.cumsum(im) / im.sum()])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0], y=[0.0, 1.0], mode="lines",
            line={"dash": "dot", "color": theme.NEUTRAL_FAINT}, name="equality",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cum_r, y=cum_i, mode="lines", line_shape="hv",
            line={"color": theme.group_color("emissions")},
            name="warming vs responsibility",
            hovertemplate="cumulative responsibility %{x:.0%}<br>"
            "cumulative warming %{y:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis={"title": "Cumulative responsibility share", "tickformat": ".0%",
               "range": [0, 1]},
        yaxis={"title": "Cumulative warming share", "tickformat": ".0%",
               "range": [0, 1]},
        height=380,
        showlegend=True,
    )
    return theme.apply_base_layout(fig)


def exposure_shift_scatter(
    table: pd.DataFrame,
    *,
    impact_col: str = "impact_index_population_weighted",
    z_col: str = "station_to_people_z_gap",
    basis_label: str = "population-weighted",
) -> go.Figure:
    """Station-weighted versus re-weighted national warming, per country.

    Each country is plotted at (station-weighted rate, re-weighted rate)
    against the no-shift diagonal; distance from it is how much the
    alternative weighting changes the country's value. Points are colored by
    the standardized shift (positive means more warming than the station
    mean suggests).
    """
    station = table["impact_index_v1"].to_numpy(dtype=float)
    other = table[impact_col].to_numpy(dtype=float)
    lo = float(min(station.min(), other.min()))
    hi = float(max(station.max(), other.max()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line={"dash": "dot", "color": theme.NEUTRAL_FAINT}, name="no shift",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=station, y=other, mode="markers", name="country",
            marker={
                "color": table[z_col],
                "colorscale": "RdBu_r",
                "cmid": 0.0,
                "size": 9,
                "line": {"color": theme.NEUTRAL_MID, "width": 0.5},
                "colorbar": {"title": "z-shift"},
            },
            customdata=table["Country"],
            hovertemplate="<b>%{customdata}</b><br>station %{x:.3f}<br>"
            f"{basis_label} %{{y:.3f}}<br>z-shift %{{marker.color:+.2f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Station-weighted vs {basis_label} national warming",
        xaxis={"title": f"Station-weighted warming ({theme.TREND_UNIT})"},
        yaxis={"title": f"{basis_label.capitalize()} warming ({theme.TREND_UNIT})"},
        height=420,
        showlegend=False,
    )
    return theme.apply_base_layout(fig)


def mismatch_scatter(table: pd.DataFrame) -> go.Figure:
    """Responsibility against station-weighted warming per country, colored by the standardized gap."""
    fig = go.Figure(
        go.Scatter(
            x=table["responsibility_index_v1"],
            y=table["impact_index_v1"],
            mode="markers",
            marker={
                "color": table["z_gap"],
                "colorscale": "RdBu_r",
                "cmid": 0.0,
                "size": 9,
                "line": {"color": theme.NEUTRAL_MID, "width": 0.5},
                "colorbar": {"title": "z-gap"},
            },
            customdata=table["Country"],
            hovertemplate="<b>%{customdata}</b><br>responsibility %{x:,.1f}<br>"
            "warming %{y:.3f}<br>z-gap %{marker.color:+.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Responsibility vs station-weighted warming (per country)",
        xaxis={"title": "Cumulative CO₂ per 2013 resident (t/person)", "type": "log"},
        yaxis={"title": f"Station-weighted warming ({theme.TREND_UNIT})"},
        height=420,
        showlegend=False,
    )
    return theme.apply_base_layout(fig)


# The global forcing page reuses the geography blue so it is visually distinct
# from the responsibility vermillion of the cross-country pages.
_PHYSICAL_COLOR = "#0072B2"
_PHYSICAL_BAND = "rgba(0,114,178,0.15)"
_VOLCANIC_COLOR = "#D55E00"


def physical_trajectory_chart(
    trajectory: pd.DataFrame,
    train_end: int,
    eruptions: list[tuple[int, str]] | None = None,
) -> go.Figure:
    """Observed vs predicted global temperature with the 95% predictive band.

    The model mean inside its band, observed annual anomalies as markers, the
    out-of-sample region (``year > train_end``) shaded, and the major volcanic
    eruptions in ``eruptions`` marked.
    """
    df = trajectory.sort_values("year")
    years = df["year"].to_numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=years, y=df["lower95"], mode="lines",
            line={"width": 0}, hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years, y=df["upper95"], mode="lines", name="95% predictive band",
            line={"width": 0}, fill="tonexty", fillcolor=_PHYSICAL_BAND,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years, y=df["predicted_mean"], mode="lines", name="model",
            line={"color": _PHYSICAL_COLOR, "width": 2},
            hovertemplate="%{x}<br>predicted %{y:.2f} °C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years, y=df["observed"], mode="markers", name="observed",
            marker={"color": theme.NEUTRAL_STRONG, "size": 5},
            hovertemplate="%{x}<br>observed %{y:.2f} °C<extra></extra>",
        )
    )
    fig.add_vrect(
        x0=train_end + 0.5, x1=float(years.max()) + 0.5,
        fillcolor="rgba(150,150,150,0.10)", line_width=0,
        annotation_text="out-of-sample", annotation_position="top left",
    )
    for year, label in eruptions or []:
        if years.min() <= year <= years.max():
            fig.add_vline(x=year, line={"color": _VOLCANIC_COLOR, "width": 1, "dash": "dot"})
            fig.add_annotation(
                x=year, y=1.0, yref="paper", text=label, showarrow=False,
                textangle=-90, xshift=-7, font={"size": 10, "color": _VOLCANIC_COLOR},
            )
    fig.update_layout(
        title="Global temperature: observed vs the forcing regression",
        xaxis={"title": "year"},
        yaxis={"title": "anomaly (°C, 1951\u20131980 baseline)"},
        height=460,
    )
    return theme.apply_base_layout(fig)


def sensitivity_forest(sensitivity: dict[str, dict]) -> go.Figure:
    """Per-driver sensitivity (°C per W/m²) with 95% intervals; CO₂ emphasized.

    ``sensitivity`` maps each driver to ``mean``, ``sd``, ``ci_low`` and
    ``ci_high``. ONI is a dimensionless ENSO regressor, not a forcing, and is
    labeled as such. These are regression coefficients, never an attribution.
    """
    labels = {
        "co2": "CO₂", "ch4": "CH₄", "n2o": "N₂O", "aerosol": "Aerosol",
        "volcanic": "Volcanic", "solar": "Solar", "oni": "ONI (ENSO)",
    }
    keys = [k for k in labels if k in sensitivity]
    names = [labels[k] for k in keys]
    means = [float(sensitivity[k]["mean"]) for k in keys]
    lo = [float(sensitivity[k]["ci_low"]) for k in keys]
    hi = [float(sensitivity[k]["ci_high"]) for k in keys]
    colors = [_VOLCANIC_COLOR if k == "co2" else _PHYSICAL_COLOR for k in keys]

    fig = go.Figure(
        go.Scatter(
            x=means, y=names, mode="markers",
            marker={"color": colors, "size": 10},
            error_x={
                "type": "data", "symmetric": False,
                "array": [h - m for h, m in zip(hi, means)],
                "arrayminus": [m - low for m, low in zip(means, lo)],
                "color": theme.NEUTRAL_MID,
            },
            customdata=np.stack([lo, hi], axis=-1),
            hovertemplate="<b>%{y}</b><br>%{x:+.3f} "
            "[%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line={"dash": "dot", "color": theme.NEUTRAL_FAINT})
    fig.update_layout(
        title="Driver sensitivities (°C per W/m²; ONI dimensionless)",
        xaxis={"title": "°C per W/m²"},
        yaxis={"autorange": "reversed"},
        height=80 + 42 * max(len(keys), 1),
        showlegend=False,
    )
    return theme.apply_base_layout(fig)


# Short tier labels for the income axis (low to high); the summary carries the
# canonical long names, the chart shows the compact ones.
INCOME_SHORT_LABELS: dict[str, str] = {
    "Low-income countries": "Low",
    "Lower-middle-income countries": "Lower-mid",
    "Upper-middle-income countries": "Upper-mid",
    "High-income countries": "High",
}


def income_gradient_chart(summary: dict, lens: str = "area") -> go.Figure:
    """Responsibility and per-person warming by income tier on one chart.

    Per tier (low to high), responsibility (mean cumulative per-capita CO₂) is
    drawn as bars on the left axis and the per-person warming of the chosen
    ``lens`` as a line on the right axis. Reads only the pre-computed summary.
    """
    order = summary["income_order"]
    resp_tiers = summary["responsibility"]["by_tier"]
    warm_tiers = summary["exposure"][lens]["by_tier"]
    tiers = [t for t in order if t in resp_tiers and t in warm_tiers]
    labels = [INCOME_SHORT_LABELS.get(t, t) for t in tiers]
    responsibility = [resp_tiers[t]["mean"] for t in tiers]
    warming = [warm_tiers[t]["pop_weighted_mean"] for t in tiers]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels, y=responsibility, name="Responsibility (mean t CO₂/capita)",
            marker={"color": theme.group_color("emissions")},
            hovertemplate="<b>%{x}</b><br>%{y:.1f} t CO₂/capita<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels, y=warming, name="Warming (per-person °C/decade)", yaxis="y2",
            mode="lines+markers", line={"color": "#D55E00"},
            marker={"color": theme.NEUTRAL_STRONG, "size": 9},
            hovertemplate="<b>%{x}</b><br>%{y:+.3f} °C/decade<extra></extra>",
        )
    )
    fig.update_layout(
        title="Responsibility rises with income; per-person warming barely moves",
        xaxis={"title": "World Bank income group (low to high)"},
        yaxis={"title": "Mean cumulative t CO₂/capita"},
        yaxis2={
            "title": "Per-person warming (°C/decade)",
            "overlaying": "y", "side": "right", "showgrid": False,
        },
        height=420,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return theme.apply_base_layout(fig)


def income_strata_box(
    strata: pd.DataFrame,
    value_col: str,
    income_order: list[str],
    title: str,
    group_col: str = "income_group",
    xaxis_title: str = "World Bank income group (low to high)",
) -> go.Figure:
    """Per-group distribution (box plus points) of one country-level column.

    Boxes follow ``income_order``; points are the individual countries. Rows
    with a missing ``value_col`` are dropped. ``group_col`` selects the
    stratifier: income tier by default, or the vulnerability quartile.
    """
    work = strata.dropna(subset=[value_col])
    fig = go.Figure()
    for tier in income_order:
        group = work.loc[work[group_col] == tier]
        if group.empty:
            continue
        fig.add_trace(
            go.Box(
                y=group[value_col].to_numpy(dtype=float),
                name=INCOME_SHORT_LABELS.get(tier, tier),
                boxpoints="all", jitter=0.4, pointpos=0,
                marker={"color": theme.NEUTRAL_STRONG, "size": 5},
                line={"color": theme.group_color("emissions")},
                customdata=group["owid_country"].to_numpy()[:, None],
                hovertemplate="<b>%{customdata[0]}</b><br>%{y:+.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis={"title": xaxis_title},
        yaxis={"title": theme.TREND_UNIT},
        height=420,
        showlegend=False,
    )
    return theme.apply_base_layout(fig)


def ndgain_scatter(strata: pd.DataFrame) -> go.Figure:
    """Responsibility against ND-GAIN vulnerability, colored by area-weighted warming.

    Per country, x is the ND-GAIN vulnerability score (right is more
    vulnerable), y is cumulative per-capita CO₂ on a log axis, and the marker
    color is area-weighted warming. Rows missing the score or responsibility
    are dropped.
    """
    work = strata.dropna(subset=["vulnerability", "cum_co2_t_per_capita"])
    fig = go.Figure(
        go.Scatter(
            x=work["vulnerability"].to_numpy(dtype=float),
            y=work["cum_co2_t_per_capita"].to_numpy(dtype=float),
            mode="markers",
            marker={
                "color": work["trend_c_per_decade_area_weighted"].to_numpy(dtype=float),
                "colorscale": theme.TREND_COLORSCALE,
                "colorbar": {"title": theme.TREND_UNIT},
                "size": 8,
                "line": {"color": theme.NEUTRAL_MID, "width": 0.5},
            },
            customdata=work["owid_country"].to_numpy()[:, None],
            hovertemplate="<b>%{customdata[0]}</b><br>vulnerability %{x:.2f}<br>"
            "%{y:.1f} t CO₂/capita<extra></extra>",
        )
    )
    fig.update_layout(
        title="Responsibility vs vulnerability (color: area-weighted warming)",
        xaxis={"title": "ND-GAIN vulnerability (least to most)"},
        yaxis={"title": "Cumulative t CO₂/capita", "type": "log"},
        height=420,
        showlegend=False,
    )
    return theme.apply_base_layout(fig)
