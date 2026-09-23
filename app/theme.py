"""Shared visual language for the dashboard.

Centralizes what must stay consistent across every page: the fixed color per
decomposition group, the warming-trend colorscale, a minimal Plotly layout,
and the interpretation banner. View modules and chart builders import from
here so a color or copy change happens in exactly one place.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Fixed color per feature group (Okabe-Ito colorblind-safe palette). The
# residual is a neutral grey: it is the unexplained remainder, not an axis.
# Every plot that shows a group uses these; Plotly never auto-assigns them.
GROUP_COLORS: dict[str, str] = {
    "emissions": "#D55E00",      # vermillion: historical responsibility
    "geography": "#0072B2",      # blue: physical geography
    "socioeconomic": "#E69F00",  # orange: development
    "population": "#009E73",     # bluish green: population
    "residual": "#9A9A9A",       # grey: residual
}

# Canonical render order (named groups first, residual last).
GROUP_ORDER: tuple[str, ...] = (
    "geography", "socioeconomic", "population", "emissions", "residual",
)

GROUP_LABELS: dict[str, str] = {
    "emissions": "Historical responsibility",
    "geography": "Geography",
    "socioeconomic": "Socioeconomic development",
    "population": "Population",
    "residual": "Residual",
}
GROUP_GLOSS: dict[str, str] = {
    "emissions": "Cumulative production-based CO2 through 2013",
    "geography": "Latitude, elevation, continentality, climate zone, hemisphere, continent",
    "socioeconomic": "World Bank income group",
    "population": "Population and station density",
    "residual": "Variance no named group explains (1 minus R squared)",
}

# How a country's warming was constructed (the ``outcome_definition`` field
# of the decomposition and stability summaries).
OUTCOME_LABELS: dict[str, str] = {
    "area_weighted": "area-weighted",
    "station_weighted": "station-weighted",
}


def outcome_label(key: object) -> str:
    """Human label for an outcome definition key; the key itself if unknown."""
    return OUTCOME_LABELS.get(str(key), str(key))


# A warming trend is a strictly positive magnitude across countries, so it is
# encoded on a sequential scale (not the diverging scale used for anomalies).
TREND_COLORSCALE = "OrRd"
TREND_UNIT = "°C/decade"

# Neutral greys for non-semantic chart furniture. st.plotly_chart's Streamlit
# theme adapts the background, gridlines and font but not explicit trace
# colors, so these must read on both light and dark backgrounds: mid-grey
# (128) is the perceptual midpoint and opacity alone sets prominence.
NEUTRAL_STRONG = "rgba(128, 128, 128, 0.95)"
NEUTRAL_MID = "rgba(128, 128, 128, 0.65)"
NEUTRAL_FAINT = "rgba(128, 128, 128, 0.40)"

# Colors for the spatial-model families on the residual and robustness pages.
FAMILY_COLORS: dict[str, str] = {"spatial_error": "#0072B2", "spatial_lag": "#CC79A7"}
FAMILY_LABELS: dict[str, str] = {"spatial_error": "Spatial error model", "spatial_lag": "Spatial lag model"}

# Shared typography. No color and no Plotly template here: charts render
# through st.plotly_chart's Streamlit theme, which supplies background,
# gridlines and font color from the active theme, so charts invert cleanly
# between light and dark mode.
BASE_FONT = {"family": "Inter, system-ui, sans-serif", "size": 13}

# Interpretation banner shown at the top of every page.
BANNER_TEXT = (
    "Every result on this site is descriptive. It measures how observed "
    "warming aligns with country attributes; it does not estimate causal "
    "effects or attribute warming to its physical drivers."
)


def apply_base_layout(fig: go.Figure, **overrides) -> go.Figure:
    """Stamp the shared layout onto ``fig`` in place.

    Tight margins, consistent typography and a legend below the plot so it
    never collides with the chart title. ``overrides`` are forwarded to
    ``update_layout`` last, so a caller can still adjust a single figure.
    """
    layout = {
        "font": BASE_FONT,
        "margin": {"l": 10, "r": 10, "t": 36, "b": 10},
        "legend": {"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0},
    }
    # Plotly renders the literal text "undefined" when a title font is set on
    # a figure that has no title, so the font is applied only to titled figures.
    if fig.layout.title.text:
        layout["title_font"] = {"size": 16}
    fig.update_layout(**layout)
    if overrides:
        fig.update_layout(**overrides)
    return fig


def group_color(key: str) -> str:
    """Color for a group key, with a grey fallback for unknown keys."""
    return GROUP_COLORS.get(key, "#9A9A9A")


def interpretation_banner() -> None:
    """Render the interpretation banner (called once, from the entry point)."""
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid #0072B2;
            background: rgba(0, 114, 178, 0.08);
            padding: 0.55rem 0.9rem;
            margin: 0 0 0.9rem 0;
            border-radius: 4px;
            font-size: 0.86rem;
            line-height: 1.35;">
            <strong>Interpretation.</strong> {BANNER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_chart(fig: go.Figure, **kwargs):
    """Render titles as wrapping page text and keep plot labels readable on narrow screens."""
    import textwrap

    shown = go.Figure(fig)
    if shown.layout.title.text:
        st.markdown(f"**{shown.layout.title.text}**")
    shown.layout.title = {"text": ""}
    for axis in (shown.layout.xaxis, shown.layout.yaxis):
        if axis.title.text:
            axis.title.text = "<br>".join(textwrap.wrap(axis.title.text, width=24))
    labels = shown.layout.yaxis.ticktext
    if labels is not None:
        shown.layout.yaxis.ticktext = [
            "<br>".join(textwrap.wrap(str(label), width=24)) for label in labels
        ]
    elif shown.data and getattr(shown.data[0], "y", None) is not None:
        labels = list(shown.data[0].y)
        if labels and all(isinstance(label, str) for label in labels):
            labels = list(dict.fromkeys(labels))
            shown.update_yaxes(tickmode="array", tickvals=labels, ticktext=[
                "<br>".join(textwrap.wrap(label, width=24)) for label in labels
            ])
    if shown.layout.showlegend is not False:
        shown.update_layout(legend={"orientation": "h", "y": -0.45, "x": 0})
        shown.layout.height = (shown.layout.height or 450) + 80
    return st.plotly_chart(shown, **kwargs)
