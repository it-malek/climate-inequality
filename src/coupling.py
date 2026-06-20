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
from src.pcs import IMPACT_INDEX, RESPONSIBILITY_INDEX
from src.projections import DEFAULT_PROJECTIONS_PATH, ID_COL

logger = logging.getLogger(__name__)

TOP_N = 10

DEFAULT_SUMMARY_PATH = PROCESSED_DIR / "coupling_summary.json"
DEFAULT_COUPLING_PATH = PROCESSED_DIR / "coupling.parquet"

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


def validate_projection_frame(columns) -> None:
    """Semantic-closure guard: only ``Country`` + the two PCS projections may appear.

    Implements Invariant 1: the comparator operates *only* on the PCS-resolved
    scalars; a stray column (e.g. continent, population) would be forbidden
    feature leakage and must raise rather than enter a computation.

    Raises:
        ValueError: if any column is outside the admissible set, or a required
            PCS column is missing.
    """
    admissible = {ID_COL, RESPONSIBILITY_INDEX, IMPACT_INDEX}
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


def compute_coupling(projections: pd.DataFrame) -> tuple[pd.DataFrame, CouplingResult]:
    """Run the closed operator set over the two PCS projections.

    Args:
        projections: one row per country with exactly ``Country``,
            ``responsibility_index_v1``, ``impact_index_v1`` (the L1 output).

    Returns:
        ``(table, result)`` -- the per-country comparison table (the
        ``coupling.parquet`` rows) and the summary :class:`CouplingResult`.

    Raises:
        ValueError: if the input is not closed over the PCS projections.
    """
    validate_projection_frame(projections.columns)
    work = projections.reset_index(drop=True)
    countries = work[ID_COL].astype(str).to_numpy()
    responsibility = work[RESPONSIBILITY_INDEX].to_numpy(dtype=float)
    impact = work[IMPACT_INDEX].to_numpy(dtype=float)

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
            RESPONSIBILITY_INDEX: responsibility,
            IMPACT_INDEX: impact,
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
