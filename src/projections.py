"""PCS v1 instantiation: resolve the two projections into the L1 output table.

This module **instantiates** PCS v1; it does **not** design, define, or redefine
Layer 1, and introduces no modeling, transformation, or reinterpretation. The frozen
upstream pipeline already realizes the two projection values; here they are merely
*bound* (by identity) to their PCS names and materialized as the constrained Layer 1
output -- exactly ``{Country, responsibility_index_v1, impact_index_v1}`` -- behind the
artifact boundary the comparator (:mod:`src.coupling`) reads.

The binding (:data:`PCS_V1_BINDING`) is a frozen, declared, **identity** mapping from
each PCS projection name to the existing source column. No log, no rescale, no
derivation. Emitting any other column would be forbidden intermediate leakage
(the minimal-observable-space / no-leakage invariant), so the resolver emits only the
two projections plus the ``Country`` key.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import pcs
from src.data_io import PROCESSED_DIR, write_typed_parquet
from src.emissions import DEFAULT_INEQUALITY_PATH

logger = logging.getLogger(__name__)

ID_COL = "Country"

# The frozen PCS v1 instantiation: each projection name bound by identity to the
# existing source column that realizes it. Identity only -- the resolver applies
# this mapping verbatim and makes no decision of its own.
PCS_V1_BINDING: dict[str, str] = {
    pcs.RESPONSIBILITY_INDEX: "cum_co2_t_per_capita",
    pcs.IMPACT_INDEX: "trend_c_per_decade",
}

# On-disk schema of projections_v1.parquet (DuckDB types), in order. Exactly the
# Country key plus the two PCS projections -- no other column may appear.
PROJECTIONS_SCHEMA: dict[str, str] = {
    ID_COL: "VARCHAR",
    pcs.RESPONSIBILITY_INDEX: "DOUBLE",
    pcs.IMPACT_INDEX: "DOUBLE",
}
PROJECTIONS_COLUMNS = tuple(PROJECTIONS_SCHEMA)

DEFAULT_PROJECTIONS_PATH = PROCESSED_DIR / "projections_v1.parquet"

# The binding must name exactly the registry's projections -- caught at import.
if set(PCS_V1_BINDING) != set(pcs.PROJECTION_NAMES):
    raise ValueError(
        f"PCS_V1_BINDING keys {sorted(PCS_V1_BINDING)} must equal the registry "
        f"projections {sorted(pcs.PROJECTION_NAMES)}"
    )


def resolve_projections(inequality: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen identity binding to emit the Layer 1 projection table.

    Selects the bound source columns and renames them to the PCS projection names
    -- identity, no transformation. Emits only ``Country`` and the two projections
    (no continent/population/feature leakage).

    Args:
        inequality: the upstream country table (``country_inequality.parquet``),
            which must carry the bound source columns and ``Country``.

    Returns:
        One row per country with columns :data:`PROJECTIONS_COLUMNS`.

    Raises:
        ValueError: if a bound source column or the ``Country`` key is absent.
    """
    required = [ID_COL, *PCS_V1_BINDING.values()]
    missing = [c for c in required if c not in inequality.columns]
    if missing:
        raise ValueError(
            f"cannot resolve PCS projections: source column(s) {missing} absent "
            f"from the upstream table"
        )
    out = pd.DataFrame({ID_COL: inequality[ID_COL].to_numpy()})
    for projection, source in PCS_V1_BINDING.items():
        out[projection] = inequality[source].to_numpy()  # identity binding
    return out[list(PROJECTIONS_COLUMNS)]


def build_projections(
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_path: Path = DEFAULT_PROJECTIONS_PATH,
) -> Path:
    """Materialize ``projections_v1.parquet`` from the upstream country table.

    Args:
        inequality_path: Phase 4 ``country_inequality.parquet``.
        out_path: Destination parquet (parents created).

    Returns:
        `out_path`.
    """
    inequality = pd.read_parquet(inequality_path)
    projections = resolve_projections(inequality)
    write_typed_parquet(projections, out_path, PROJECTIONS_SCHEMA, order_by=(ID_COL,))
    logger.info("wrote %s (%d countries)", out_path, len(projections))
    return out_path


def main() -> None:
    """Resolve the PCS v1 projections and write the Layer 1 output table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = build_projections()
    projections = pd.read_parquet(out)
    print(f"PCS {pcs.PCS_VERSION} instantiation (identity binding, no transformation)")
    for projection, source in PCS_V1_BINDING.items():
        print(f"  {projection} := {source}")
    print(f"  columns: {list(projections.columns)}  rows: {len(projections)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
