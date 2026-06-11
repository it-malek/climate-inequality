"""Cleaning utilities for the Berkeley Earth city temperature dataset.

Ported from the original 2022 fellowship notebook (cleaning_l.ipynb) and
hardened: vectorized coordinate parsing, validation, and windowing helpers.
"""

from __future__ import annotations

import pandas as pd

# Default analysis window — see README "Known dataset quirks"
DEFAULT_START = "1950-01-01"
DEFAULT_END = "2013-09-01"


def parse_coordinate(value: str) -> float:
    """Convert a Berkeley Earth coordinate string to a signed float.

    Examples: "32.95N" -> 32.95, "100.53W" -> -100.53.
    N/E are positive; S/W are negative.

    Raises:
        ValueError: if the suffix is not one of N/S/E/W or the numeric
            part is not parseable.
    """
    value = value.strip()
    if not value:
        raise ValueError("empty coordinate string")
    suffix = value[-1].upper()
    if suffix not in "NSEW":
        raise ValueError(f"unexpected hemisphere suffix in {value!r}")
    magnitude = float(value[:-1])
    return magnitude if suffix in "NE" else -magnitude


def parse_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized version of the original notebook's `proper_l` logic.

    Returns a copy of `df` with `Latitude` and `Longitude` as signed floats.
    Validates that results are within physical bounds.
    """
    out = df.copy()
    for col, bound in (("Latitude", 90.0), ("Longitude", 180.0)):
        s = out[col].astype(str).str.strip()
        sign = s.str[-1].str.upper().map({"N": 1, "E": 1, "S": -1, "W": -1})
        if sign.isna().any():
            bad = s[sign.isna()].unique()[:5]
            raise ValueError(f"unexpected {col} suffixes, e.g. {list(bad)}")
        out[col] = s.str[:-1].astype(float) * sign
        if (out[col].abs() > bound).any():
            raise ValueError(f"{col} values exceed ±{bound}")
    return out


def add_date_parts(df: pd.DataFrame, date_col: str = "dt") -> pd.DataFrame:
    """Parse the date column and add Year / Month integer columns."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["Year"] = out[date_col].dt.year
    out["Month"] = out[date_col].dt.month
    return out


def filter_window(
    df: pd.DataFrame,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    date_col: str = "dt",
) -> pd.DataFrame:
    """Restrict to the analysis window (inclusive)."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    mask = (out[date_col] >= pd.Timestamp(start)) & (
        out[date_col] <= pd.Timestamp(end)
    )
    return out.loc[mask]


def coverage_by_city(
    df: pd.DataFrame,
    temp_col: str = "AverageTemperature",
    min_fraction: float = 0.9,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    group_keys: tuple[str, ...] = ("City", "Country"),
) -> pd.DataFrame:
    """Per-city observation coverage within the window.

    Returns one row per `group_keys` with `n_obs`, `n_possible`,
    `coverage`, and a boolean `keep` flag for groups meeting
    `min_fraction` non-null monthly coverage. Use this to decide which
    cities are eligible for trend fitting (Phase 2).

    Note: (City, Country) alone is not a unique city identifier — 18
    same-named pairs sit at 2-3 grid coordinates each, which pools and
    inflates their apparent coverage. Pass
    ``group_keys=("City", "Country", "Latitude", "Longitude")`` for true
    per-location coverage (what trend fitting uses).
    """
    n_possible = len(pd.period_range(start, end, freq="M"))
    grp = df.groupby(list(group_keys), as_index=False, observed=True).agg(
        n_obs=(temp_col, "count")
    )
    grp["n_possible"] = n_possible
    grp["coverage"] = grp["n_obs"] / n_possible
    grp["keep"] = grp["coverage"] >= min_fraction
    return grp
