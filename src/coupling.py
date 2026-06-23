"""Layer 3: deterministic projection comparator over the two PCS v1 projections.

L3 is a pure deterministic functional ``(R_c, I_c) -> metrics`` where
``R_c = responsibility_index_v1`` and ``I_c = impact_index_v1`` are the only inputs --
read through the artifact boundary (``projections_v1.parquet``), never by re-deriving
how they were computed (semantic closure). It performs **only** the closed operator set:

  A. ranking      -- sort descending, ``rank(method="min")``;
  B. z-score      -- fixed operator ``z(x) = (x - mean(x)) / std(x, ddof=0)``;
  C. differences  -- ``rank_gap = impact_rank - responsibility_rank``,
                     ``z_gap = z(I) - z(R)``;
  D. Lorenz       -- sort by responsibility only, cumulative sums normalized to [0,1],
                     no interpolation / smoothing / curve fitting;
  E. one scalar   -- ``inequality_coefficient`` = ``2 * |Lorenz area|``, bounded [0,1].

It additionally reports the rank correlation ``spearman_rho`` and the z_gap-ordered
mismatch lists. No feature engineering, no regression, no optimization, no latent
structure -- any operation not listed above is forbidden. It writes exactly two
artifacts, ``coupling.parquet`` and ``coupling_summary.json``; the summary carries only
the permitted fields and **no** interpretation / narrative / commentary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.data_io import PROCESSED_DIR, round_floats, write_typed_parquet
from src.pcs import (
    IMPACT_AREA_WEIGHTED_INDEX,
    IMPACT_INDEX,
    IMPACT_POPULATION_WEIGHTED_INDEX,
    RESPONSIBILITY_CONSUMPTION_INDEX,
    RESPONSIBILITY_INDEX,
    RESPONSIBILITY_PRODUCTION_MATCHED_INDEX,
)
from src.projections import (
    DEFAULT_INEQUALITY_PATH,
    DEFAULT_PROJECTIONS_AREA_PATH,
    DEFAULT_PROJECTIONS_CONSUMPTION_PATH,
    DEFAULT_PROJECTIONS_EXPOSURE_PATH,
    DEFAULT_PROJECTIONS_PATH,
    ID_COL,
    PROJECTIONS_AREA_COLUMNS,
    PROJECTIONS_CONSUMPTION_COLUMNS,
    PROJECTIONS_EXPOSURE_COLUMNS,
    area_coverage,
    consumption_window,
    population_coverage,
)

logger = logging.getLogger(__name__)

TOP_N = 10

DEFAULT_SUMMARY_PATH = PROCESSED_DIR / "coupling_summary.json"
DEFAULT_COUPLING_PATH = PROCESSED_DIR / "coupling.parquet"

DEFAULT_CONSUMPTION_SUMMARY_PATH = PROCESSED_DIR / "coupling_consumption_summary.json"
DEFAULT_CONSUMPTION_COUPLING_PATH = PROCESSED_DIR / "coupling_consumption.parquet"

DEFAULT_EXPOSURE_SUMMARY_PATH = PROCESSED_DIR / "coupling_exposure_summary.json"
DEFAULT_EXPOSURE_COUPLING_PATH = PROCESSED_DIR / "coupling_exposure.parquet"

DEFAULT_AREA_SUMMARY_PATH = PROCESSED_DIR / "coupling_area_summary.json"
DEFAULT_AREA_COUPLING_PATH = PROCESSED_DIR / "coupling_area.parquet"

# The default admissible set is the v1 closure (Country + the two v1 projections);
# v2 consumers pass the registered subset for their artifact instead.
V1_ADMISSIBLE: frozenset[str] = frozenset({ID_COL, RESPONSIBILITY_INDEX, IMPACT_INDEX})
WIDE_ADMISSIBLE: frozenset[str] = frozenset(PROJECTIONS_CONSUMPTION_COLUMNS)
EXPOSURE_ADMISSIBLE: frozenset[str] = frozenset(PROJECTIONS_EXPOSURE_COLUMNS)
AREA_ADMISSIBLE: frozenset[str] = frozenset(PROJECTIONS_AREA_COLUMNS)

# On-disk schema of coupling.parquet (DuckDB types), in order.
COUPLING_SCHEMA: dict[str, str] = {
    ID_COL: "VARCHAR",
    RESPONSIBILITY_INDEX: "DOUBLE",
    IMPACT_INDEX: "DOUBLE",
    "responsibility_rank": "BIGINT",
    "impact_rank": "BIGINT",
    "rank_gap": "BIGINT",
    "z_gap": "DOUBLE",
}
COUPLING_COLUMNS = tuple(COUPLING_SCHEMA)

# On-disk schema of coupling_consumption.parquet (the wide two-pass diagnostics):
# the registered v2 projections, plus the production->consumption rank-shift pass
# and the consumption-vs-impact pass diagnostics.
COUPLING_CONSUMPTION_SCHEMA: dict[str, str] = {
    ID_COL: "VARCHAR",
    IMPACT_INDEX: "DOUBLE",
    RESPONSIBILITY_CONSUMPTION_INDEX: "DOUBLE",
    RESPONSIBILITY_PRODUCTION_MATCHED_INDEX: "DOUBLE",
    "production_matched_rank": "BIGINT",
    "consumption_rank": "BIGINT",
    "prod_to_cons_rank_gap": "BIGINT",
    "prod_to_cons_z_gap": "DOUBLE",
    "impact_rank": "BIGINT",
    "consumption_impact_z_gap": "DOUBLE",
}
COUPLING_CONSUMPTION_COLUMNS = tuple(COUPLING_CONSUMPTION_SCHEMA)

# On-disk schema of coupling_exposure.parquet (the wide two-pass diagnostics):
# the registered projections, plus the station->people exposure rank-shift pass
# and the people-weighted-exposure-vs-responsibility pass diagnostics.
COUPLING_EXPOSURE_SCHEMA: dict[str, str] = {
    ID_COL: "VARCHAR",
    RESPONSIBILITY_INDEX: "DOUBLE",
    IMPACT_INDEX: "DOUBLE",
    IMPACT_POPULATION_WEIGHTED_INDEX: "DOUBLE",
    "station_rank": "BIGINT",
    "people_rank": "BIGINT",
    "station_to_people_rank_gap": "BIGINT",
    "station_to_people_z_gap": "DOUBLE",
    "responsibility_rank": "BIGINT",
    "people_responsibility_z_gap": "DOUBLE",
}
COUPLING_EXPOSURE_COLUMNS = tuple(COUPLING_EXPOSURE_SCHEMA)

# On-disk schema of coupling_area.parquet (the wide two-pass diagnostics): the
# registered projections, plus the station->area exposure rank-shift pass and the
# area-weighted-exposure-vs-responsibility pass diagnostics.
COUPLING_AREA_SCHEMA: dict[str, str] = {
    ID_COL: "VARCHAR",
    RESPONSIBILITY_INDEX: "DOUBLE",
    IMPACT_INDEX: "DOUBLE",
    IMPACT_AREA_WEIGHTED_INDEX: "DOUBLE",
    "station_rank": "BIGINT",
    "area_rank": "BIGINT",
    "station_to_area_rank_gap": "BIGINT",
    "station_to_area_z_gap": "DOUBLE",
    "responsibility_rank": "BIGINT",
    "area_responsibility_z_gap": "DOUBLE",
}
COUPLING_AREA_COLUMNS = tuple(COUPLING_AREA_SCHEMA)


def validate_projection_frame(
    columns, admissible: frozenset[str] = V1_ADMISSIBLE
) -> None:
    """Semantic-closure guard: only ``Country`` + registered projections may appear.

    Implements Invariant 1: the comparator operates *only* on the PCS-resolved
    scalars; a stray column (e.g. continent, population) would be forbidden
    feature leakage and must raise rather than enter a computation. `admissible`
    defaults to the v1 closure (exactly the two v1 projections); v2 callers pass
    the wider registered set, so closure widens from "exactly the two" to "exactly
    the registered set" without being dropped.

    Raises:
        ValueError: if any column is outside `admissible`, or a registered
            column is missing.
    """
    cols = list(columns)
    unknown = [c for c in cols if c not in admissible]
    if unknown:
        raise ValueError(
            f"projection frame carries non-PCS column(s) {sorted(unknown)}; the "
            f"comparator operates only on {sorted(admissible)} (semantic closure)"
        )
    missing = sorted(admissible - set(cols))
    if missing:
        raise ValueError(f"projection frame missing PCS column(s) {missing}")


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """Strict descending rank, ties via ``method="min"`` (operator A)."""
    return pd.Series(values).rank(method="min", ascending=False).astype("int64").to_numpy()


def _zscore(values: np.ndarray) -> np.ndarray:
    """Fixed z-score operator ``z(x) = (x - mean(x)) / std(x, ddof=0)`` (operator B)."""
    return (values - values.mean()) / values.std(ddof=0)


def _inequality_coefficient(responsibility: np.ndarray, impact: np.ndarray) -> float:
    """Gini-style scalar: ``2 * |area between the Lorenz curve and the diagonal|``.

    Operator D/E: order countries by responsibility only; take the cumulative
    responsibility share (x) and cumulative impact share (y), each normalized to
    [0, 1]; the coefficient is twice the absolute area between that empirical curve
    and the 45-degree line (the discrete Lorenz area -- no interpolation, smoothing,
    or curve fitting), clamped to [0, 1]. Both projections must be positive for the
    share construction; any non-positive row is dropped (logged) first.
    """
    positive = (responsibility > 0.0) & (impact > 0.0)
    n_drop = int((~positive).sum())
    if n_drop:
        logger.info(
            "inequality_coefficient: dropping %d non-positive row(s) before Lorenz",
            n_drop,
        )
    r = responsibility[positive]
    im = impact[positive]
    order = np.argsort(r, kind="stable")  # sort by responsibility only
    cum_r = np.concatenate([[0.0], np.cumsum(r[order]) / r.sum()])
    cum_i = np.concatenate([[0.0], np.cumsum(im[order]) / im.sum()])
    area_under = float(
        np.sum((cum_r[1:] - cum_r[:-1]) * (cum_i[1:] + cum_i[:-1]) / 2.0)
    )
    coefficient = abs(1.0 - 2.0 * area_under)
    return float(min(1.0, max(0.0, coefficient)))


@dataclass(frozen=True)
class CouplingResult:
    """Deterministic responsibility-impact comparison metrics (the L3 contract)."""

    spearman_rho: float
    n_high_impact_low_responsibility: int
    inequality_coefficient: float
    top_suffer_least_cause: list[tuple[str, float]] = field(default_factory=list)
    top_cause_least_suffer: list[tuple[str, float]] = field(default_factory=list)

    def check(self, atol: float = 1e-9) -> None:
        """Assert the inequality coefficient is bounded [0, 1] (§5.1.E)."""
        if not (0.0 - atol <= self.inequality_coefficient <= 1.0 + atol):
            raise AssertionError(
                f"inequality_coefficient {self.inequality_coefficient!r} outside [0, 1]"
            )


def compute_coupling(
    projections: pd.DataFrame,
    x_col: str = RESPONSIBILITY_INDEX,
    y_col: str = IMPACT_INDEX,
    admissible: frozenset[str] = V1_ADMISSIBLE,
) -> tuple[pd.DataFrame, CouplingResult]:
    """Run the closed operator set over a chosen pair of registered projections.

    The operators are unchanged; only the two columns they run on are
    parametrized. ``x_col`` plays the **responsibility** role (the Lorenz ordering
    axis and the z/rank baseline); ``y_col`` plays the **impact** role (the
    compared quantity). Defaults reproduce the v1 responsibility-vs-impact
    comparison exactly; v2 callers pass a wider `admissible` set and any registered
    pair (e.g. production-matched vs consumption).

    Args:
        projections: one row per country, closed over `admissible`.
        x_col: responsibility-role column (rank/z baseline, Lorenz ordering).
        y_col: impact-role column (compared against the baseline).
        admissible: the closure set the frame must equal.

    Returns:
        ``(table, result)`` -- the per-country comparison table (value columns
        named ``x_col``/``y_col``; diagnostic columns ``responsibility_rank``,
        ``impact_rank``, ``rank_gap``, ``z_gap``) and the summary
        :class:`CouplingResult`.

    Raises:
        ValueError: if the input is not closed over `admissible`, or ``x_col`` /
            ``y_col`` are not in it.
    """
    validate_projection_frame(projections.columns, admissible)
    for role, col in (("x_col", x_col), ("y_col", y_col)):
        if col not in admissible:
            raise ValueError(f"{role} {col!r} is not a registered projection")

    work = projections.reset_index(drop=True)
    countries = work[ID_COL].astype(str).to_numpy()
    responsibility = work[x_col].to_numpy(dtype=float)  # responsibility role
    impact = work[y_col].to_numpy(dtype=float)  # impact role

    responsibility_rank = _rank_desc(responsibility)
    impact_rank = _rank_desc(impact)
    rank_gap = impact_rank - responsibility_rank
    z_gap = _zscore(impact) - _zscore(responsibility)

    spearman_rho = float(stats.spearmanr(responsibility, impact)[0])
    inequality_coefficient = _inequality_coefficient(responsibility, impact)
    n_high_impact_low_responsibility = int((z_gap > 0.0).sum())

    table = pd.DataFrame(
        {
            ID_COL: countries,
            x_col: responsibility,
            y_col: impact,
            "responsibility_rank": responsibility_rank,
            "impact_rank": impact_rank,
            "rank_gap": rank_gap,
            "z_gap": z_gap,
        }
    )

    order = np.argsort(z_gap, kind="stable")  # ascending by z_gap
    top_cause_least_suffer = [(countries[i], float(z_gap[i])) for i in order[:TOP_N]]
    top_suffer_least_cause = [
        (countries[i], float(z_gap[i])) for i in order[::-1][:TOP_N]
    ]

    result = CouplingResult(
        spearman_rho=spearman_rho,
        n_high_impact_low_responsibility=n_high_impact_low_responsibility,
        inequality_coefficient=inequality_coefficient,
        top_suffer_least_cause=top_suffer_least_cause,
        top_cause_least_suffer=top_cause_least_suffer,
    )
    result.check()
    return table, result


def summary_payload(result: CouplingResult) -> dict:
    """JSON payload of the result, float-rounded for byte-stability.

    Carries only the permitted fields and **no** interpretation / narrative
    (§8). Floats are rounded (:func:`src.data_io.round_floats`) so the committed
    JSON is byte-stable across platforms.
    """
    return round_floats(asdict(result))


def build_coupling(
    projections_path=DEFAULT_PROJECTIONS_PATH,
    summary_path=DEFAULT_SUMMARY_PATH,
    table_path=DEFAULT_COUPLING_PATH,
) -> dict:
    """Read the projection table, compare, and write the two L3 artifacts.

    Args:
        projections_path: the L1 ``projections_v1.parquet``.
        summary_path: destination ``coupling_summary.json``.
        table_path: destination ``coupling.parquet``.

    Returns:
        Dict with ``table`` (DataFrame), ``result`` (:class:`CouplingResult`),
        ``summary_path`` and ``table_path``.
    """
    projections = pd.read_parquet(projections_path)
    table, result = compute_coupling(projections)
    write_typed_parquet(table, table_path, COUPLING_SCHEMA, order_by=(ID_COL,))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary_payload(result), indent=2) + "\n", encoding="utf-8"
    )
    logger.info("wrote %s and %s", table_path, summary_path)
    return {
        "table": table,
        "result": result,
        "summary_path": summary_path,
        "table_path": table_path,
    }


# ---------------------------------------------------------------------
# Consumption lens (PCS v2 wide registry) -- the two-pass comparator
# ---------------------------------------------------------------------


def compute_consumption_coupling(
    projections: pd.DataFrame,
) -> tuple[pd.DataFrame, CouplingResult, CouplingResult]:
    """Run the comparator twice over the wide consumption projection frame.

    Same closed operators, two registered pairs:

    - **Pass 1** ``(responsibility_index_consumption, impact_index_v1)`` -- the
      consumption-based climate-inequality coefficient (consumption responsibility
      vs warming impact, mirroring the v1 question's shape).
    - **Pass 2** ``(responsibility_index_production_matched,
      responsibility_index_consumption)`` -- the production->consumption rank-shift
      diagnostic, valid because both cumulatives share each country's window. A
      positive ``prod_to_cons_z_gap`` flags a net importer (more responsible under
      consumption accounting); negative flags a net exporter.

    Args:
        projections: the wide ``projections_consumption.parquet`` frame, closed
            over :data:`WIDE_ADMISSIBLE`.

    Returns:
        ``(wide_table, consumption_vs_impact, production_to_consumption_shift)``.
    """
    t_ci, consumption_vs_impact = compute_coupling(
        projections,
        x_col=RESPONSIBILITY_CONSUMPTION_INDEX,
        y_col=IMPACT_INDEX,
        admissible=WIDE_ADMISSIBLE,
    )
    t_pc, production_to_consumption_shift = compute_coupling(
        projections,
        x_col=RESPONSIBILITY_PRODUCTION_MATCHED_INDEX,
        y_col=RESPONSIBILITY_CONSUMPTION_INDEX,
        admissible=WIDE_ADMISSIBLE,
    )

    # Both passes ran on the same row order (reset_index inside compute_coupling),
    # so positional assembly into the wide diagnostic table is aligned.
    work = projections.reset_index(drop=True)
    wide = pd.DataFrame(
        {
            ID_COL: work[ID_COL].astype(str).to_numpy(),
            IMPACT_INDEX: work[IMPACT_INDEX].to_numpy(dtype=float),
            RESPONSIBILITY_CONSUMPTION_INDEX: work[
                RESPONSIBILITY_CONSUMPTION_INDEX
            ].to_numpy(dtype=float),
            RESPONSIBILITY_PRODUCTION_MATCHED_INDEX: work[
                RESPONSIBILITY_PRODUCTION_MATCHED_INDEX
            ].to_numpy(dtype=float),
            "production_matched_rank": t_pc["responsibility_rank"].to_numpy(),
            "consumption_rank": t_pc["impact_rank"].to_numpy(),
            "prod_to_cons_rank_gap": t_pc["rank_gap"].to_numpy(),
            "prod_to_cons_z_gap": t_pc["z_gap"].to_numpy(),
            "impact_rank": t_ci["impact_rank"].to_numpy(),
            "consumption_impact_z_gap": t_ci["z_gap"].to_numpy(),
        }
    )
    return (
        wide[list(COUPLING_CONSUMPTION_COLUMNS)],
        consumption_vs_impact,
        production_to_consumption_shift,
    )


def consumption_summary_payload(
    consumption_vs_impact: CouplingResult,
    production_to_consumption_shift: CouplingResult,
    window: dict,
) -> dict:
    """Unified, float-rounded JSON payload for the two-pass consumption summary.

    Carries the per-country consumption ``window`` provenance plus both passes'
    metrics under namespaced keys. Floats are rounded
    (:func:`src.data_io.round_floats`) so the committed JSON is byte-stable.
    """
    payload = {
        "window": window,
        "consumption_vs_impact": asdict(consumption_vs_impact),
        "production_to_consumption_shift": asdict(production_to_consumption_shift),
    }
    return round_floats(payload)


def build_coupling_consumption(
    projections_path=DEFAULT_PROJECTIONS_CONSUMPTION_PATH,
    inequality_path=DEFAULT_INEQUALITY_PATH,
    summary_path=DEFAULT_CONSUMPTION_SUMMARY_PATH,
    table_path=DEFAULT_CONSUMPTION_COUPLING_PATH,
) -> dict:
    """Read the wide projection table, run both passes, and write the L3 v2 artifacts.

    Args:
        projections_path: the wide ``projections_consumption.parquet``.
        inequality_path: ``country_inequality.parquet`` -- read only for the
            consumption-window provenance label (not for the comparison itself).
        summary_path: destination ``coupling_consumption_summary.json``.
        table_path: destination ``coupling_consumption.parquet``.

    Returns:
        Dict with ``table``, ``consumption_vs_impact``,
        ``production_to_consumption_shift``, ``summary_path`` and ``table_path``.
    """
    projections = pd.read_parquet(projections_path)
    window = consumption_window(pd.read_parquet(inequality_path))
    table, consumption_vs_impact, production_to_consumption_shift = (
        compute_consumption_coupling(projections)
    )
    consumption_vs_impact.check()
    production_to_consumption_shift.check()

    write_typed_parquet(
        table, table_path, COUPLING_CONSUMPTION_SCHEMA, order_by=(ID_COL,)
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            consumption_summary_payload(
                consumption_vs_impact, production_to_consumption_shift, window
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s and %s", table_path, summary_path)
    return {
        "table": table,
        "consumption_vs_impact": consumption_vs_impact,
        "production_to_consumption_shift": production_to_consumption_shift,
        "summary_path": summary_path,
        "table_path": table_path,
    }


# ---------------------------------------------------------------------
# Exposure lens (PCS v2 wide registry) -- people-weighted warming exposure
# ---------------------------------------------------------------------


def compute_exposure_coupling(
    projections: pd.DataFrame,
) -> tuple[pd.DataFrame, CouplingResult, CouplingResult]:
    """Run the comparator twice over the exposure projection frame.

    Same closed operators, two registered pairs:

    - **Pass A** ``(impact_index_v1, impact_index_population_weighted)`` -- the
      station-weighted vs people-weighted exposure rank-shift. A positive
      ``station_to_people_z_gap`` flags a country that looks *more* exposed once
      warming is weighted by where people actually live.
    - **Pass B** ``(responsibility_index_v1, impact_index_population_weighted)`` --
      the people-weighted climate-inequality Lorenz/Gini (cumulative
      people-weighted warming exposure vs cumulative responsibility).

    Returns:
        ``(wide_table, station_vs_people, people_weighted_inequality)``.
    """
    t_sp, station_vs_people = compute_coupling(
        projections,
        x_col=IMPACT_INDEX,
        y_col=IMPACT_POPULATION_WEIGHTED_INDEX,
        admissible=EXPOSURE_ADMISSIBLE,
    )
    t_pi, people_weighted_inequality = compute_coupling(
        projections,
        x_col=RESPONSIBILITY_INDEX,
        y_col=IMPACT_POPULATION_WEIGHTED_INDEX,
        admissible=EXPOSURE_ADMISSIBLE,
    )

    work = projections.reset_index(drop=True)
    wide = pd.DataFrame(
        {
            ID_COL: work[ID_COL].astype(str).to_numpy(),
            RESPONSIBILITY_INDEX: work[RESPONSIBILITY_INDEX].to_numpy(dtype=float),
            IMPACT_INDEX: work[IMPACT_INDEX].to_numpy(dtype=float),
            IMPACT_POPULATION_WEIGHTED_INDEX: work[
                IMPACT_POPULATION_WEIGHTED_INDEX
            ].to_numpy(dtype=float),
            "station_rank": t_sp["responsibility_rank"].to_numpy(),
            "people_rank": t_sp["impact_rank"].to_numpy(),
            "station_to_people_rank_gap": t_sp["rank_gap"].to_numpy(),
            "station_to_people_z_gap": t_sp["z_gap"].to_numpy(),
            "responsibility_rank": t_pi["responsibility_rank"].to_numpy(),
            "people_responsibility_z_gap": t_pi["z_gap"].to_numpy(),
        }
    )
    return (
        wide[list(COUPLING_EXPOSURE_COLUMNS)],
        station_vs_people,
        people_weighted_inequality,
    )


def exposure_summary_payload(
    station_vs_people: CouplingResult,
    people_weighted_inequality: CouplingResult,
    coverage: dict,
) -> dict:
    """Unified, float-rounded JSON payload for the two-pass exposure summary."""
    payload = {
        "coverage": coverage,
        "station_vs_people": asdict(station_vs_people),
        "people_weighted_inequality": asdict(people_weighted_inequality),
    }
    return round_floats(payload)


def build_coupling_exposure(
    projections_path=DEFAULT_PROJECTIONS_EXPOSURE_PATH,
    inequality_path=DEFAULT_INEQUALITY_PATH,
    summary_path=DEFAULT_EXPOSURE_SUMMARY_PATH,
    table_path=DEFAULT_EXPOSURE_COUPLING_PATH,
) -> dict:
    """Read the exposure projection table, run both passes, write the L3 artifacts.

    Args:
        projections_path: the ``projections_exposure.parquet``.
        inequality_path: ``country_inequality.parquet`` -- read only for the
            people-weighting coverage label (not for the comparison itself).
        summary_path: destination ``coupling_exposure_summary.json``.
        table_path: destination ``coupling_exposure.parquet``.

    Returns:
        Dict with ``table``, ``station_vs_people``,
        ``people_weighted_inequality``, ``summary_path`` and ``table_path``.
    """
    projections = pd.read_parquet(projections_path)
    coverage = population_coverage(pd.read_parquet(inequality_path))
    table, station_vs_people, people_weighted_inequality = compute_exposure_coupling(
        projections
    )
    station_vs_people.check()
    people_weighted_inequality.check()

    write_typed_parquet(
        table, table_path, COUPLING_EXPOSURE_SCHEMA, order_by=(ID_COL,)
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            exposure_summary_payload(
                station_vs_people, people_weighted_inequality, coverage
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s and %s", table_path, summary_path)
    return {
        "table": table,
        "station_vs_people": station_vs_people,
        "people_weighted_inequality": people_weighted_inequality,
        "summary_path": summary_path,
        "table_path": table_path,
    }


# ---------------------------------------------------------------------
# Area-weighted lens (PCS v2 wide registry) -- gridded cos-latitude exposure
# ---------------------------------------------------------------------


def compute_area_coupling(
    projections: pd.DataFrame,
) -> tuple[pd.DataFrame, CouplingResult, CouplingResult]:
    """Run the comparator twice over the area-weighted projection frame.

    Same closed operators, two registered pairs:

    - **Pass A** ``(impact_index_v1, impact_index_area_weighted)`` -- the
      station-weighted vs area-weighted exposure rank-shift. A positive
      ``station_to_area_z_gap`` flags a country that looks *more* exposed once
      warming is weighted by land area instead of by station density.
    - **Pass B** ``(responsibility_index_v1, impact_index_area_weighted)`` -- the
      area-weighted climate-inequality Lorenz/Gini (cumulative area-weighted
      warming exposure vs cumulative responsibility).

    Returns:
        ``(wide_table, station_vs_area, area_weighted_inequality)``.
    """
    t_sa, station_vs_area = compute_coupling(
        projections,
        x_col=IMPACT_INDEX,
        y_col=IMPACT_AREA_WEIGHTED_INDEX,
        admissible=AREA_ADMISSIBLE,
    )
    t_ai, area_weighted_inequality = compute_coupling(
        projections,
        x_col=RESPONSIBILITY_INDEX,
        y_col=IMPACT_AREA_WEIGHTED_INDEX,
        admissible=AREA_ADMISSIBLE,
    )

    work = projections.reset_index(drop=True)
    wide = pd.DataFrame(
        {
            ID_COL: work[ID_COL].astype(str).to_numpy(),
            RESPONSIBILITY_INDEX: work[RESPONSIBILITY_INDEX].to_numpy(dtype=float),
            IMPACT_INDEX: work[IMPACT_INDEX].to_numpy(dtype=float),
            IMPACT_AREA_WEIGHTED_INDEX: work[
                IMPACT_AREA_WEIGHTED_INDEX
            ].to_numpy(dtype=float),
            "station_rank": t_sa["responsibility_rank"].to_numpy(),
            "area_rank": t_sa["impact_rank"].to_numpy(),
            "station_to_area_rank_gap": t_sa["rank_gap"].to_numpy(),
            "station_to_area_z_gap": t_sa["z_gap"].to_numpy(),
            "responsibility_rank": t_ai["responsibility_rank"].to_numpy(),
            "area_responsibility_z_gap": t_ai["z_gap"].to_numpy(),
        }
    )
    return (
        wide[list(COUPLING_AREA_COLUMNS)],
        station_vs_area,
        area_weighted_inequality,
    )


def area_summary_payload(
    station_vs_area: CouplingResult,
    area_weighted_inequality: CouplingResult,
    coverage: dict,
) -> dict:
    """Unified, float-rounded JSON payload for the two-pass area summary."""
    payload = {
        "coverage": coverage,
        "station_vs_area": asdict(station_vs_area),
        "area_weighted_inequality": asdict(area_weighted_inequality),
    }
    return round_floats(payload)


def build_coupling_area(
    projections_path=DEFAULT_PROJECTIONS_AREA_PATH,
    inequality_path=DEFAULT_INEQUALITY_PATH,
    summary_path=DEFAULT_AREA_SUMMARY_PATH,
    table_path=DEFAULT_AREA_COUPLING_PATH,
) -> dict:
    """Read the area projection table, run both passes, write the L3 artifacts.

    Args:
        projections_path: the ``projections_area.parquet``.
        inequality_path: ``country_inequality.parquet`` -- read only for the
            area-weighting coverage label (not for the comparison itself).
        summary_path: destination ``coupling_area_summary.json``.
        table_path: destination ``coupling_area.parquet``.

    Returns:
        Dict with ``table``, ``station_vs_area``, ``area_weighted_inequality``,
        ``summary_path`` and ``table_path``.
    """
    projections = pd.read_parquet(projections_path)
    coverage = area_coverage(pd.read_parquet(inequality_path))
    table, station_vs_area, area_weighted_inequality = compute_area_coupling(
        projections
    )
    station_vs_area.check()
    area_weighted_inequality.check()

    write_typed_parquet(table, table_path, COUPLING_AREA_SCHEMA, order_by=(ID_COL,))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            area_summary_payload(
                station_vs_area, area_weighted_inequality, coverage
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s and %s", table_path, summary_path)
    return {
        "table": table,
        "station_vs_area": station_vs_area,
        "area_weighted_inequality": area_weighted_inequality,
        "summary_path": summary_path,
        "table_path": table_path,
    }


def main() -> None:
    """Run the comparator on the L1 projection table and print the headline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = build_coupling()
    r = out["result"]
    print(f"responsibility-impact coupling (n={len(out['table'])}, deterministic)")
    print(f"  Spearman rho            : {r.spearman_rho:+.3f}")
    print(f"  inequality coefficient  : {r.inequality_coefficient:.3f}")
    print(f"  high-impact/low-resp.   : {r.n_high_impact_low_responsibility}")
    print("  suffer most / caused least (top z_gap):")
    for country, z in r.top_suffer_least_cause[:5]:
        print(f"    {country:<22s} z_gap {z:+.3f}")
    print("  caused most / suffer least (low z_gap):")
    for country, z in r.top_cause_least_suffer[:5]:
        print(f"    {country:<22s} z_gap {z:+.3f}")
    print(f"  table:   {out['table_path']}")
    print(f"  summary: {out['summary_path']}")


if __name__ == "__main__":
    main()
