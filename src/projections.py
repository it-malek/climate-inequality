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

# The frozen PCS v2 (wide-registry) instantiation: the v1 impact projection plus
# the two window-matched responsibility lenses, each bound by identity to its
# source column in the (additively-extended) country table.
PCS_V2_BINDING: dict[str, str] = {
    pcs.IMPACT_INDEX: "trend_c_per_decade",
    pcs.RESPONSIBILITY_CONSUMPTION_INDEX: "cum_consumption_t_per_capita",
    pcs.RESPONSIBILITY_PRODUCTION_MATCHED_INDEX: "cum_co2_window_t_per_capita",
}

# On-disk schema of projections_consumption.parquet (DuckDB types), in order:
# the Country key plus the three registered v2 projections (the wide registry).
PROJECTIONS_CONSUMPTION_SCHEMA: dict[str, str] = {
    ID_COL: "VARCHAR",
    pcs.IMPACT_INDEX: "DOUBLE",
    pcs.RESPONSIBILITY_CONSUMPTION_INDEX: "DOUBLE",
    pcs.RESPONSIBILITY_PRODUCTION_MATCHED_INDEX: "DOUBLE",
}
PROJECTIONS_CONSUMPTION_COLUMNS = tuple(PROJECTIONS_CONSUMPTION_SCHEMA)

DEFAULT_PROJECTIONS_CONSUMPTION_PATH = (
    PROCESSED_DIR / "projections_consumption.parquet"
)

# The PCS v2 exposure binding: the v1 responsibility + impact projections plus
# the people-weighted impact lens, each bound by identity to its source column.
PCS_V2_EXPOSURE_BINDING: dict[str, str] = {
    pcs.RESPONSIBILITY_INDEX: "cum_co2_t_per_capita",
    pcs.IMPACT_INDEX: "trend_c_per_decade",
    pcs.IMPACT_POPULATION_WEIGHTED_INDEX: "trend_c_per_decade_pop_weighted",
}

# On-disk schema of projections_exposure.parquet (DuckDB types), in order: the
# Country key plus the three registered projections this lens compares.
PROJECTIONS_EXPOSURE_SCHEMA: dict[str, str] = {
    ID_COL: "VARCHAR",
    pcs.RESPONSIBILITY_INDEX: "DOUBLE",
    pcs.IMPACT_INDEX: "DOUBLE",
    pcs.IMPACT_POPULATION_WEIGHTED_INDEX: "DOUBLE",
}
PROJECTIONS_EXPOSURE_COLUMNS = tuple(PROJECTIONS_EXPOSURE_SCHEMA)

DEFAULT_PROJECTIONS_EXPOSURE_PATH = (
    PROCESSED_DIR / "projections_exposure.parquet"
)

# Each v2 artifact binds a registered subset of the wide registry (the registry
# grows; an artifact need not name every projection). Caught at import.
for _binding_name, _binding in (
    ("PCS_V2_BINDING", PCS_V2_BINDING),
    ("PCS_V2_EXPOSURE_BINDING", PCS_V2_EXPOSURE_BINDING),
):
    if not set(_binding) <= set(pcs.PROJECTION_NAMES_V2):
        raise ValueError(
            f"{_binding_name} keys {sorted(_binding)} must be registered v2 "
            f"projections {sorted(pcs.PROJECTION_NAMES_V2)}"
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


def resolve_consumption_projections(inequality: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen v2 identity binding to emit the wide projection table.

    Mirrors :func:`resolve_projections` for the PCS v2 wide registry: selects the
    bound source columns (identity, no transformation) and emits only ``Country``
    plus the three registered v2 projections. Countries with no OWID consumption
    series carry NULLs in the consumption sources; those rows are dropped here so
    the consumption lens operates only on countries with a real window.

    Args:
        inequality: the upstream country table (``country_inequality.parquet``),
            which must carry the bound source columns and ``Country``.

    Returns:
        One row per consumption-covered country with columns
        :data:`PROJECTIONS_CONSUMPTION_COLUMNS`.

    Raises:
        ValueError: if a bound source column or the ``Country`` key is absent.
    """
    required = [ID_COL, *PCS_V2_BINDING.values()]
    missing = [c for c in required if c not in inequality.columns]
    if missing:
        raise ValueError(
            f"cannot resolve PCS v2 projections: source column(s) {missing} absent "
            f"from the upstream table"
        )
    out = pd.DataFrame({ID_COL: inequality[ID_COL].to_numpy()})
    for projection, source in PCS_V2_BINDING.items():
        out[projection] = inequality[source].to_numpy()  # identity binding
    out = out[list(PROJECTIONS_CONSUMPTION_COLUMNS)]
    # Drop countries lacking a consumption window (NULL responsibility lenses).
    covered = out[
        [pcs.RESPONSIBILITY_CONSUMPTION_INDEX, pcs.RESPONSIBILITY_PRODUCTION_MATCHED_INDEX]
    ].notna().all(axis=1)
    n_drop = int((~covered).sum())
    if n_drop:
        logger.info("consumption projections: dropping %d countries without a window", n_drop)
    return out.loc[covered].reset_index(drop=True)


def consumption_window(inequality: pd.DataFrame) -> dict:
    """Window-provenance label for the consumption lens (summary metadata).

    The per-country consumption-available window is opaque to the closed
    projection frame (``consumption_start_year`` is not a projection), so this
    label is derived upstream from the country table and injected into the
    consumption summary -- mirroring how the forcings hash is stamped at the
    bundle layer rather than by the pure estimator. The window end is the shared
    analysis cutoff; only the per-country start years vary.

    Args:
        inequality: the country table carrying ``consumption_start_year``.

    Returns:
        ``{n_countries, consumption_start_year_min/median/max}`` over the
        consumption-covered countries (empty-safe).
    """
    years = inequality["consumption_start_year"].dropna()
    if years.empty:
        return {
            "n_countries": 0,
            "consumption_start_year_min": None,
            "consumption_start_year_median": None,
            "consumption_start_year_max": None,
        }
    return {
        "n_countries": int(len(years)),
        "consumption_start_year_min": int(years.min()),
        "consumption_start_year_median": float(years.median()),
        "consumption_start_year_max": int(years.max()),
    }


def build_consumption_projections(
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_path: Path = DEFAULT_PROJECTIONS_CONSUMPTION_PATH,
) -> Path:
    """Materialize ``projections_consumption.parquet`` from the country table.

    Args:
        inequality_path: Phase 4 ``country_inequality.parquet``.
        out_path: Destination parquet (parents created).

    Returns:
        `out_path`.
    """
    inequality = pd.read_parquet(inequality_path)
    projections = resolve_consumption_projections(inequality)
    write_typed_parquet(
        projections, out_path, PROJECTIONS_CONSUMPTION_SCHEMA, order_by=(ID_COL,)
    )
    logger.info("wrote %s (%d countries)", out_path, len(projections))
    return out_path


def resolve_exposure_projections(inequality: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen v2 exposure binding to emit the exposure projection table.

    Mirrors :func:`resolve_consumption_projections` for the people-weighted
    exposure lens: emits ``Country`` plus ``responsibility_index_v1``,
    ``impact_index_v1`` and ``impact_index_population_weighted`` (identity, no
    transformation). Countries with no people-weighted exposure (NULL, e.g. the
    population grid was absent) are dropped here.

    Raises:
        ValueError: if a bound source column or the ``Country`` key is absent.
    """
    required = [ID_COL, *PCS_V2_EXPOSURE_BINDING.values()]
    missing = [c for c in required if c not in inequality.columns]
    if missing:
        raise ValueError(
            f"cannot resolve PCS v2 exposure projections: source column(s) "
            f"{missing} absent from the upstream table"
        )
    out = pd.DataFrame({ID_COL: inequality[ID_COL].to_numpy()})
    for projection, source in PCS_V2_EXPOSURE_BINDING.items():
        out[projection] = inequality[source].to_numpy()  # identity binding
    out = out[list(PROJECTIONS_EXPOSURE_COLUMNS)]
    covered = out[pcs.IMPACT_POPULATION_WEIGHTED_INDEX].notna()
    n_drop = int((~covered).sum())
    if n_drop:
        logger.info(
            "exposure projections: dropping %d countries without people-weighting",
            n_drop,
        )
    return out.loc[covered].reset_index(drop=True)


def population_coverage(inequality: pd.DataFrame) -> dict:
    """Coverage-provenance label for the exposure lens (summary metadata).

    Like :func:`consumption_window`, this is derived upstream (``pop_weight_coverage``
    is not a projection) and injected into the exposure summary. Reports how many
    countries have a people-weighting and how complete that weighting is.

    Returns:
        ``{n_countries, mean_pop_weight_coverage}`` over the people-weighted
        countries (empty-safe).
    """
    weighted = inequality.loc[
        inequality["trend_c_per_decade_pop_weighted"].notna()
    ]
    if weighted.empty:
        return {"n_countries": 0, "mean_pop_weight_coverage": None}
    return {
        "n_countries": int(len(weighted)),
        "mean_pop_weight_coverage": float(weighted["pop_weight_coverage"].mean()),
    }


def build_exposure_projections(
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_path: Path = DEFAULT_PROJECTIONS_EXPOSURE_PATH,
) -> Path:
    """Materialize ``projections_exposure.parquet`` from the country table."""
    inequality = pd.read_parquet(inequality_path)
    projections = resolve_exposure_projections(inequality)
    write_typed_parquet(
        projections, out_path, PROJECTIONS_EXPOSURE_SCHEMA, order_by=(ID_COL,)
    )
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
