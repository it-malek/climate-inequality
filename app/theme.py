"""Shared visual language for the dashboard (single source of truth).

Centralizes the things that must stay consistent across every page: the
fixed color per decomposition group, the warming-trend colorscale, a minimal
research-grade Plotly layout, and the always-on interpretation banner. View
modules and chart builders import from here so a color or copy change happens
in exactly one place.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Fixed color per feature-schema group. Okabe-Ito colorblind-safe palette;
# residual is a neutral grey (it is the *un*explained remainder, not an axis).
# Every plot that shows a group MUST use these — never let Plotly auto-assign.
GROUP_COLORS: dict[str, str] = {
    "emissions": "#D55E00",      # vermillion — responsibility axis
    "geography": "#0072B2",      # blue — physical structure axis
    "socioeconomic": "#E69F00",  # orange — development structure
    "population": "#009E73",     # bluish green — demographic scale
    "residual": "#9A9A9A",       # grey — unexplained structure
}

# Canonical render order (named axes first, residual last).
GROUP_ORDER: tuple[str, ...] = (
    "emissions", "geography", "socioeconomic", "population", "residual",
)

# Human-facing labels and one-line glosses for each group.
GROUP_LABELS: dict[str, str] = {
    "emissions": "Emissions",
    "geography": "Geography",
    "socioeconomic": "Socioeconomic",
    "population": "Population",
    "residual": "Residual",
}
GROUP_GLOSS: dict[str, str] = {
    "emissions": "Cumulative CO₂ responsibility (per-capita & total)",
    "geography": "Latitude, elevation, continentality, climate zone, hemisphere",
    "socioeconomic": "Development stage (income group)",
    "population": "Demographic scale & station density",
    "residual": "Variance no named axis explains (1 − R²)",
}

# Warming trend is a strictly positive magnitude across countries, so it is
# encoded on a sequential scale (not the diverging RdBu_r used for the
# anomaly *surface*, which straddles zero). One constant => one mapping.
TREND_COLORSCALE = "OrRd"
TREND_UNIT = "°C/decade"

# Neutral greys for non-semantic chart furniture (data markers, error bars,
# reference/zero lines). st.plotly_chart(theme="streamlit") adapts the background,
# gridlines and font but NOT explicit trace colors, so these must read on both a
# light and a dark plot background by construction. Mid-grey (128) is the
# light/dark perceptual midpoint; opacity alone sets prominence, so a single value
# stays legible in either mode -- never a near-black (#333, lost on dark) or a pale
# grey (#aaa, lost on light).
NEUTRAL_STRONG = "rgba(128, 128, 128, 0.95)"   # data-point markers
NEUTRAL_MID = "rgba(128, 128, 128, 0.65)"      # error bars, marker outlines
NEUTRAL_FAINT = "rgba(128, 128, 128, 0.40)"    # zero / reference lines

# Shared typography. No `color` and no Plotly template here: charts render through
# st.plotly_chart's default `theme="streamlit"`, which supplies the background,
# gridlines and font color from the *active* Streamlit theme -- so the charts invert
# cleanly between light and dark. Pinning a template or a font color would override
# that adaptation (and a dark `#222` would vanish on a dark background).
BASE_FONT = {"family": "Inter, system-ui, sans-serif", "size": 13}

# The exact, non-negotiable interpretation banner (visible on every page).
BANNER_TEXT = (
    "This is a structural variance decomposition of observed warming trends. "
    "It does not estimate causal effects or climate physics."
)


def apply_base_layout(fig: go.Figure, **overrides) -> go.Figure:
    """Stamp the shared research-grade layout onto `fig` (in place).

    Keeps margins tight, the background clean, and typography consistent so
    every chart reads as one coherent interface. `overrides` are forwarded to
    ``update_layout`` last, so a caller can still tweak a single figure.
    """
    fig.update_layout(
        font=BASE_FONT,
        margin={"l": 10, "r": 10, "t": 36, "b": 10},
        title_font={"size": 16},
        # Legend sits *below* the plot (autoexpand grows the bottom margin to fit)
        # so it never collides with the chart title; tighter top margin closes the
        # gap the old above-plot legend left behind.
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0},
    )
    if overrides:
        fig.update_layout(**overrides)
    return fig


def group_color(key: str) -> str:
    """Color for a group key, with a grey fallback for unknown keys."""
    return GROUP_COLORS.get(key, "#9A9A9A")


def interpretation_banner() -> None:
    """Render the always-on, non-flashy interpretation banner.

    Called once per script run (from the entry point), so it appears at the
    top of every page. A subtle left-accent strip, not an alert color block —
    present and legible, never shouty.
    """
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid #0072B2;
            background: #f4f7fa;
            padding: 0.55rem 0.9rem;
            margin: 0 0 0.9rem 0;
            border-radius: 4px;
            font-size: 0.86rem;
            color: #33414d;
            line-height: 1.35;">
            <strong>Interpretation —</strong> {BANNER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )
