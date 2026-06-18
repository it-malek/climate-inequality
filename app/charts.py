"""Plotly figure builders for the dashboard (no Streamlit, unit-testable).

Each function takes plain data (dicts / DataFrames already loaded by
``app.loaders``) and returns a ``go.Figure`` styled through
:func:`app.theme.apply_base_layout`, so color and layout stay consistent with
the rest of the interface. Streamlit lives only in the view modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app import theme


def shares_bar(shares: dict[str, float], residual_share: float) -> go.Figure:
    """Horizontal bar of Shapley variance shares, residual included.

    Bars are in :data:`app.theme.GROUP_ORDER` and colored by
    :data:`app.theme.GROUP_COLORS`; labels are the share of *total* variance.
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

    ``view`` is ``"emissions"`` or ``"geography"`` (that axis's *standalone*
    R² vs the unexplained remainder) or ``"full"`` (the four group shares
    stacked, then residual). One row, segments to 100%.
    """
    total_r2 = float(summary["total_r2"])
    if view == "full":
        segments = [
            (theme.GROUP_LABELS[k], summary["shares"][k], theme.group_color(k))
            for k in theme.GROUP_ORDER
            if k in summary["shares"]
        ]
        segments.append(("Residual", summary["residual_share"], theme.group_color("residual")))
        subtitle = f"Full model — {total_r2:.0%} of variance explained"
    else:
        r2 = float(summary["univariate_r2"][view])
        segments = [
            (f"{theme.GROUP_LABELS[view]} alone", r2, theme.group_color(view)),
            ("Unexplained", 1.0 - r2, theme.group_color("residual")),
        ]
        subtitle = f"{theme.GROUP_LABELS[view]} alone — {r2:.0%} of variance explained"

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


def warming_choropleth(df: pd.DataFrame) -> go.Figure:
    """World choropleth of country warming trend, with rich hover.

    Args:
        df: one row per country with ``location`` (country name for Plotly's
            ``country names`` mode), ``warming_trend``, ``cum_co2_t_per_capita``
            and ``mean_latitude``.

    Returns:
        A ``go.Choropleth`` figure colored on the sequential trend scale.
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
            z=df["warming_trend"],
            customdata=customdata,
            colorscale=theme.TREND_COLORSCALE,
            colorbar={"title": theme.TREND_UNIT},
            marker_line_color="rgba(120,120,120,0.5)",
            marker_line_width=0.3,
            hovertemplate=(
                "<b>%{location}</b><br>"
                "Warming trend: %{z:.3f} " + theme.TREND_UNIT + "<br>"
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
    fig.update_layout(height=520, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    return theme.apply_base_layout(fig, margin={"l": 0, "r": 0, "t": 10, "b": 0})


def coef_ci_chart(
    rows: list[dict],
    label_key: str,
    title: str,
    color: str | None = None,
) -> go.Figure:
    """Horizontal coefficient ± 95% CI chart (one row per spec).

    Shared by the GAM df-sensitivity and the HC1-vs-Conley comparison. Each
    ``row`` needs ``coef``, ``ci_low``, ``ci_high`` and the field named by
    `label_key`.
    """
    labels = [str(r[label_key]) for r in rows]
    coefs = [float(r["coef"]) for r in rows]
    lo = [float(r["ci_low"]) for r in rows]
    hi = [float(r["ci_high"]) for r in rows]
    color = color or theme.group_color("emissions")

    fig = go.Figure(
        go.Scatter(
            x=coefs,
            y=labels,
            mode="markers",
            marker={"color": color, "size": 10},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [h - c for h, c in zip(hi, coefs)],
                "arrayminus": [c - low for c, low in zip(coefs, lo)],
                "color": color,
            },
            hovertemplate="<b>%{y}</b><br>coef %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line={"dash": "dot", "color": "#aaa"})
    fig.update_layout(
        title=title,
        xaxis={"title": "coefficient (°C/decade per 10× emissions)"},
        yaxis={"autorange": "reversed"},
        height=80 + 46 * len(rows),
        showlegend=False,
    )
    return theme.apply_base_layout(fig)


def dfbeta_bar(top_dfbeta: list[tuple[str, float]]) -> go.Figure:
    """Horizontal bar of the most influential countries by |DFBETA|."""
    names = [str(n) for n, _ in top_dfbeta]
    vals = [float(v) for _, v in top_dfbeta]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker={"color": theme.group_color("emissions")},
            hovertemplate="<b>%{y}</b><br>DFBETA %{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line={"dash": "dot", "color": "#aaa"})
    fig.update_layout(
        title="Most influential countries (DFBETA on the emissions term)",
        xaxis={"title": "DFBETA"},
        yaxis={"autorange": "reversed"},
        height=80 + 34 * max(len(names), 1),
        showlegend=False,
    )
    return theme.apply_base_layout(fig)
