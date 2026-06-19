"""Inequality of the warming burden across countries (descriptive only).

Quantifies *how unequal* observed 1950-2013 warming is across countries, with
the standard distributional summaries: variance, coefficient of variation, the
Gini coefficient, the Theil-T index (with an exact between-/within-group
decomposition), and the Lorenz curve. See ``docs/decomposition_design_memo.md``
for the framing.

Every quantity here is **descriptive and non-causal** -- a structural summary of
*observed* warming, not climate attribution. A Gini of country warming trends
says how unevenly warming is distributed; it says nothing about *why*. The
companion :mod:`src.decomposition` attributes the cross-country *variance* to the
structural axes of the frozen feature contract
(:data:`src.feature_schema.SCHEMA_V1`); this module measures the inequality that
decomposition then explains. The canonical scope disclaimer
(:data:`src.feature_schema.INTERPRETATION_NOTE`) is emitted into the summary
output.

The outcome is the country-mean Theil-Sen warming trend (``trend_c_per_decade``
in ``country_inequality.parquet``). Warming trends are positive across the
sample, so Gini/Theil are well defined; non-positive inputs are rejected
(Theil) or guarded (Gini), not silently transformed.

Weights are accepted everywhere and default to equal (one country, one unit).
Population/area weighting is a documented degree of freedom the stability layer
will perturb, not a default -- the headline inequality is *between countries as
units*.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.data_io import PROCESSED_DIR, round_floats
from src.emissions import DEFAULT_INEQUALITY_PATH
from src.feature_schema import INTERPRETATION_NOTE

logger = logging.getLogger(__name__)

DEFAULT_OUTCOME_COL = "trend_c_per_decade"
DEFAULT_GROUP_COL = "continent"
DEFAULT_SUMMARY_PATH = PROCESSED_DIR / "inequality_summary.json"


def _as_arrays(
    values: np.ndarray | pd.Series, weights: np.ndarray | pd.Series | None
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce values/weights to float arrays, defaulting to equal weights.

    Raises:
        ValueError: if `values` is empty or any weight is negative.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("no values to summarize")
    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != x.shape:
            raise ValueError(f"weights shape {w.shape} != values shape {x.shape}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
    return x, w


def weighted_mean(values: np.ndarray | pd.Series, weights=None) -> float:
    """Weighted arithmetic mean (equal weights by default)."""
    x, w = _as_arrays(values, weights)
    return float(np.sum(w * x) / np.sum(w))


def variance(values: np.ndarray | pd.Series, weights=None) -> float:
    """Population (ddof=0) variance of the warming distribution.

    The rawest inequality measure: the spread of country warming trends in
    squared °C/decade. Population variance (not sample) because we are
    describing the observed set of countries, not inferring to a superpopulation.
    """
    x, w = _as_arrays(values, weights)
    mu = np.sum(w * x) / np.sum(w)
    return float(np.sum(w * (x - mu) ** 2) / np.sum(w))


def coefficient_of_variation(values: np.ndarray | pd.Series, weights=None) -> float:
    """Std / mean -- scale-free dispersion.

    Uses the population standard deviation. Defined only for a non-zero mean.
    """
    mu = weighted_mean(values, weights)
    if mu == 0:
        raise ValueError("coefficient of variation undefined for zero mean")
    return float(np.sqrt(variance(values, weights)) / mu)


def gini(values: np.ndarray | pd.Series, weights=None) -> float:
    """Gini coefficient of a non-negative distribution.

    Computed from the (weighted) mean absolute difference,
    ``G = MAD / (2 * mean)`` where
    ``MAD = sum_ij w_i w_j |x_i - x_j| / (sum w)^2``. 0 = perfect equality, 1 =
    maximal concentration. O(n^2) in the number of units -- fine at ~157
    countries and exact for the weighted case.

    Args:
        values: non-negative magnitudes (e.g. country warming trends).
        weights: optional non-negative weights (equal by default).

    Returns:
        The Gini coefficient.

    Raises:
        ValueError: if any value is negative or the mean is non-positive.
    """
    x, w = _as_arrays(values, weights)
    if np.any(x < 0):
        raise ValueError("Gini requires non-negative values")
    total_w = np.sum(w)
    mu = np.sum(w * x) / total_w
    if mu <= 0:
        raise ValueError("Gini undefined for a non-positive mean")
    abs_diff = np.abs(x[:, None] - x[None, :])
    mad = np.sum((w[:, None] * w[None, :]) * abs_diff) / (total_w**2)
    return float(mad / (2.0 * mu))


def theil_t(values: np.ndarray | pd.Series, weights=None) -> float:
    """Theil-T index (GE(1)) of a strictly positive distribution.

    ``T = sum_i p_i (x_i/mu) ln(x_i/mu)`` with ``p_i = w_i / sum(w)``. 0 =
    perfect equality; rises with concentration. Additively decomposable into
    between- and within-group parts (:func:`theil_decomposition`).

    Raises:
        ValueError: if any value is non-positive (the log is undefined).
    """
    x, w = _as_arrays(values, weights)
    if np.any(x <= 0):
        raise ValueError("Theil-T requires strictly positive values")
    p = w / np.sum(w)
    mu = np.sum(p * x)
    r = x / mu
    return float(np.sum(p * r * np.log(r)))


@dataclass(frozen=True)
class TheilDecomposition:
    """Between/within split of the Theil-T index (sums to ``total``)."""

    total: float
    between: float
    within: float
    between_share: float
    within_by_group: dict[str, float] = field(default_factory=dict)


def theil_decomposition(
    values: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    weights=None,
) -> TheilDecomposition:
    """Exact between-/within-group decomposition of Theil-T.

    ``T = T_between + T_within`` with
    ``T_within = sum_g s_g T_g`` and ``T_between = sum_g s_g ln(mu_g/mu)``,
    where ``s_g`` is group g's share of the total magnitude and ``T_g`` is the
    within-group Theil-T. This is the model-free cross-check on the regression
    decomposition: how much warming inequality sits *between* continents vs
    *within* them.

    Args:
        values: strictly positive magnitudes.
        groups: group label per value (e.g. continent).
        weights: optional non-negative weights.

    Returns:
        A :class:`TheilDecomposition`; ``within_by_group`` holds each group's
        ``s_g * T_g`` contribution.

    Raises:
        ValueError: if any value is non-positive or lengths differ.
    """
    x, w = _as_arrays(values, weights)
    g = np.asarray(groups)
    if g.shape != x.shape:
        raise ValueError(f"groups shape {g.shape} != values shape {x.shape}")
    if np.any(x <= 0):
        raise ValueError("Theil decomposition requires strictly positive values")

    total = theil_t(x, w)
    mu = np.sum(w * x) / np.sum(w)
    total_wx = np.sum(w * x)

    between = 0.0
    within = 0.0
    within_by_group: dict[str, float] = {}
    for label in pd.unique(g):
        mask = g == label
        s_g = float(np.sum(w[mask] * x[mask]) / total_wx)  # magnitude share
        mu_g = float(np.sum(w[mask] * x[mask]) / np.sum(w[mask]))
        t_g = theil_t(x[mask], w[mask]) if mask.sum() > 1 else 0.0
        between += s_g * np.log(mu_g / mu)
        contribution = s_g * t_g
        within += contribution
        within_by_group[str(label)] = contribution

    return TheilDecomposition(
        total=float(total),
        between=float(between),
        within=float(within),
        between_share=float(between / total) if total else 0.0,
        within_by_group=within_by_group,
    )


def lorenz_points(values: np.ndarray | pd.Series, weights=None) -> pd.DataFrame:
    """Lorenz-curve coordinates (cumulative unit share vs magnitude share).

    Sorted ascending, prefixed with the origin (0, 0). The gap between this
    curve and the 45-degree line is what the Gini coefficient summarizes.

    Returns:
        DataFrame with ``cum_unit_share`` and ``cum_warming_share`` columns,
        ``n+1`` rows.
    """
    x, w = _as_arrays(values, weights)
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cum_units = np.cumsum(ws) / np.sum(ws)
    cum_value = np.cumsum(ws * xs) / np.sum(ws * xs)
    return pd.DataFrame(
        {
            "cum_unit_share": np.concatenate([[0.0], cum_units]),
            "cum_warming_share": np.concatenate([[0.0], cum_value]),
        }
    )


@dataclass(frozen=True)
class InequalitySummary:
    """Descriptive inequality of a warming distribution across units."""

    n: int
    mean: float
    variance: float
    coefficient_of_variation: float
    gini: float
    theil_t: float
    theil_between: float | None = None
    theil_within: float | None = None
    theil_between_share: float | None = None
    theil_within_by_group: dict[str, float] = field(default_factory=dict)


def summarize_inequality(
    values: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series | None = None,
    weights=None,
) -> InequalitySummary:
    """Bundle every inequality metric for a warming distribution.

    Args:
        values: country-level warming magnitudes (strictly positive).
        groups: optional group labels (continent) for the Theil split.
        weights: optional non-negative weights.

    Returns:
        An :class:`InequalitySummary`. The Theil decomposition fields are
        populated only when `groups` is given.
    """
    x, w = _as_arrays(values, weights)
    summary = InequalitySummary(
        n=int(x.size),
        mean=weighted_mean(x, w),
        variance=variance(x, w),
        coefficient_of_variation=coefficient_of_variation(x, w),
        gini=gini(x, w),
        theil_t=theil_t(x, w),
    )
    if groups is None:
        return summary
    decomp = theil_decomposition(x, groups, w)
    return InequalitySummary(
        n=summary.n,
        mean=summary.mean,
        variance=summary.variance,
        coefficient_of_variation=summary.coefficient_of_variation,
        gini=summary.gini,
        theil_t=summary.theil_t,
        theil_between=decomp.between,
        theil_within=decomp.within,
        theil_between_share=decomp.between_share,
        theil_within_by_group=decomp.within_by_group,
    )


def country_warming_inequality(
    table: pd.DataFrame,
    value_col: str = DEFAULT_OUTCOME_COL,
    group_col: str | None = DEFAULT_GROUP_COL,
    weights=None,
) -> InequalitySummary:
    """Inequality summary for a country table's warming trends.

    Args:
        table: one row per country (e.g. ``country_inequality.parquet``).
        value_col: warming-magnitude column.
        group_col: grouping column for the Theil split, or ``None`` to skip it.
        weights: optional weight column name or array (equal by default).

    Returns:
        An :class:`InequalitySummary` over the non-null rows of `value_col`.
    """
    work = table.dropna(subset=[value_col]).copy()
    groups = work[group_col].to_numpy() if group_col is not None else None
    if isinstance(weights, str):
        weights = work[weights].to_numpy()
    return summarize_inequality(work[value_col].to_numpy(), groups, weights)


def summary_to_dict(summary: InequalitySummary) -> dict:
    """JSON-safe dict for :class:`InequalitySummary`."""
    return asdict(summary)


def summary_payload(summary: InequalitySummary) -> dict:
    """Serialized summary with the scope disclaimer attached.

    The ``interpretation`` key carries
    :data:`src.feature_schema.INTERPRETATION_NOTE` so the descriptive,
    non-attribution boundary travels with the numbers into any consumer. Floats
    are rounded (:func:`src.data_io.round_floats`) so the committed JSON is
    byte-stable across platforms.
    """
    return round_floats({"interpretation": INTERPRETATION_NOTE, **summary_to_dict(summary)})


def main() -> None:
    """Print country warming-inequality metrics and write a JSON summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    table = pd.read_parquet(DEFAULT_INEQUALITY_PATH)
    summary = country_warming_inequality(table)

    print(f"country warming inequality (n={summary.n}, descriptive, non-causal)")
    print(f"  mean trend      : {summary.mean:.4f} °C/decade")
    print(f"  variance        : {summary.variance:.5f}")
    print(f"  CV              : {summary.coefficient_of_variation:.3f}")
    print(f"  Gini            : {summary.gini:.3f}")
    print(f"  Theil-T         : {summary.theil_t:.4f}")
    if summary.theil_between is not None:
        print(
            f"  Theil between   : {summary.theil_between:.4f} "
            f"({summary.theil_between_share:.0%} of total) -- between {DEFAULT_GROUP_COL}s"
        )
        print(f"  Theil within    : {summary.theil_within:.4f}")

    DEFAULT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SUMMARY_PATH.write_text(
        json.dumps(summary_payload(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(f"  note            : {INTERPRETATION_NOTE}")
    print(f"wrote {DEFAULT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
