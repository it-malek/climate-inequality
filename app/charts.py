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
            marker_line_color=theme.NEUTRAL_MID,
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
    fig.add_vline(x=0.0, line={"dash": "dot", "color": theme.NEUTRAL_FAINT})
    fig.update_layout(
        title=title,
        xaxis={"title": "coefficient (°C/decade per 10× emissions)"},
        yaxis={"autorange": "reversed"},
        height=80 + 46 * len(rows),
        showlegend=False,
    )
    return theme.apply_base_layout(fig)


def share_ci_chart(
    groups: dict[str, dict],
    block_groups: dict[str, dict] | None = None,
) -> go.Figure:
    """Bootstrap 95% CI of each variance *share* (one row per group).

    ``groups`` maps a group key to a dict with ``point``, ``ci_low`` and
    ``ci_high`` (the country-bootstrap interval). When ``block_groups`` is given,
    the wider continent-block-bootstrap interval is drawn as a faint band behind
    each point, so the gap between the two reads as the spatial-dependence
    correction. Rows follow :data:`app.theme.GROUP_ORDER` and use the fixed
    per-group palette. These are *shares*, not coefficients — the axis is a
    percentage of total variance, and the chart shows the stability of the
    decomposition, never a significance or causal claim.
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
            "<br>95% CI [%{customdata[0]:.1%}, %{customdata[1]:.1%}]<extra></extra>",
        )
    )
    fig.update_layout(
        title="Bootstrap 95% CI of each variance share",
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
    value_label: str = "Δ share (leave-one-out)",
) -> go.Figure:
    """Horizontal bar of the most influential countries by |Δ share|.

    The share-level analogue of :func:`dfbeta_bar`: one bar per country, signed
    by how the named share moves when that country is dropped.
    """
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
    """Lorenz-style curve: cumulative impact share vs cumulative responsibility share.

    Countries are ordered by responsibility only (the PCS comparator's
    construction); the empirical cumulative shares are drawn as a step curve (no
    smoothing) against the 45° equality diagonal. The area between the two is what
    the inequality coefficient summarizes. The columns are parametrized so the
    same curve serves the station-based (``impact_index_v1``) and people-weighted
    (``impact_index_population_weighted``) exposure lenses.
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
            name="impact vs responsibility",
            hovertemplate="cumulative responsibility %{x:.0%}<br>"
            "cumulative impact %{y:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis={"title": "Cumulative responsibility share", "tickformat": ".0%",
               "range": [0, 1]},
        yaxis={"title": "Cumulative impact share", "tickformat": ".0%",
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
    basis_label: str = "people-weighted",
) -> go.Figure:
    """Station-based vs re-weighted warming exposure, per country.

    Each country is plotted at (station-weighted rate, re-weighted rate) against
    the 45° no-shift line; distance from it is how much the alternative weighting
    changes the country's exposure. Points are colored by the standardized z-shift
    (`z_col`; positive = *more* exposed than the station mean suggests). Defaults
    render the people-weighted lens; pass the area columns for the area lens.
    """
    station = table["impact_index_v1"].to_numpy(dtype=float)
    people = table[impact_col].to_numpy(dtype=float)
    lo = float(min(station.min(), people.min()))
    hi = float(max(station.max(), people.max()))

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
            x=station, y=people, mode="markers", name="country",
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
        title=f"Station-based vs {basis_label} warming exposure",
        xaxis={"title": "Station-weighted warming (°C/decade)"},
        yaxis={"title": f"{basis_label.capitalize()} warming (°C/decade)"},
        height=420,
        showlegend=False,
    )
    return theme.apply_base_layout(fig)


def mismatch_scatter(table: pd.DataFrame) -> go.Figure:
    """Responsibility vs impact per country, colored by the standardized z_gap.

    A diverging RdBu_r scale centered at 0: positive z_gap (warming exposure above
    emissions responsibility) reads one way, negative (the inverse) the other.
    """
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
            "impact %{y:.3f}<br>z-gap %{marker.color:+.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Responsibility vs impact (per country)",
        xaxis={"title": "responsibility_index_v1 (t CO₂ per capita)"},
        yaxis={"title": "impact_index_v1 (°C/decade)"},
        height=420,
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
    fig.add_vline(x=0.0, line={"dash": "dot", "color": theme.NEUTRAL_FAINT})
    fig.update_layout(
        title="Most influential countries (DFBETA on the emissions term)",
        xaxis={"title": "DFBETA"},
        yaxis={"autorange": "reversed"},
        height=80 + 34 * max(len(names), 1),
        showlegend=False,
    )
    return theme.apply_base_layout(fig)


# Physical band reuses the geography blue so the L1 page is visually distinct
# from the emissions-vermillion decomposition pages.
_PHYSICAL_COLOR = "#0072B2"
_PHYSICAL_BAND = "rgba(0,114,178,0.15)"
_VOLCANIC_COLOR = "#D55E00"


def physical_trajectory_chart(
    trajectory: pd.DataFrame,
    train_end: int,
    eruptions: list[tuple[int, str]] | None = None,
) -> go.Figure:
    """Observed vs predicted global temperature with the 95% predictive band.

    The hero of the L1 page: the model mean (``predicted_mean``) inside its
    ``lower95``/``upper95`` band, observed annual anomalies as markers, the
    out-of-sample region (``year > train_end``) shaded, and the major volcanic
    eruptions in `eruptions` marked so the transient cooling dips the model
    reproduces stand out against the secular trend.
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
        title="Global temperature: observed vs the physical model",
        xaxis={"title": "year"},
        yaxis={"title": "anomaly (°C, 1951–1980 baseline)"},
        height=460,
    )
    return theme.apply_base_layout(fig)


def sensitivity_forest(sensitivity: dict[str, dict]) -> go.Figure:
    """Per-driver sensitivity (°C per W/m²) with 95% CIs; CO₂ emphasized.

    `sensitivity` maps each driver to ``{mean, sd, ci_low, ci_high}``. ONI is a
    dimensionless ENSO regressor, not a W/m² forcing — it is labeled as such. These
    are model coefficients (predictive association), never a causal attribution.
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
