"""Projection tables: bind registered projection names to country-table columns.

Each table written here holds ``Country`` plus a subset of the projections
registered in :mod:`src.pcs`, each copied verbatim (no transform) from the
column of ``country_inequality.parquet`` that realizes it. The bindings are
declared once as dictionaries and checked against the registry at import. The
coupling stage (:mod:`src.coupling`) reads only these tables, so the columns a
comparison can see are exactly the registered ones.

Four tables: the v1 pair (``projections_v1.parquet``), the consumption lens,
the people-weighted exposure lens and the area-weighted lens. Each lens drops
the countries for which its extra column is null (no consumption series, no
population weighting, no grid cells).
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

# Projection name -> source column in country_inequality.parquet.
PCS_V1_BINDING: dict[str, str] = {
    pcs.RESPONSIBILITY_INDEX: "cum_co2_t_per_capita",
    pcs.IMPACT_INDEX: "trend_c_per_decade",
}

# On-disk schema of projections_v1.parquet (DuckDB types), in order.
PROJECTIONS_SCHEMA: dict[str, str] = {
    ID_COL: "VARCHAR",
    pcs.RESPONSIBILITY_INDEX: "DOUBLE",
    pcs.IMPACT_INDEX: "DOUBLE",
}
PROJECTIONS_COLUMNS = tuple(PROJECTIONS_SCHEMA)

DEFAULT_PROJECTIONS_PATH = PROCESSED_DIR / "projections_v1.parquet"

# The v1 binding must name exactly the v1 registry; caught at import.
if set(PCS_V1_BINDING) != set(pcs.PROJECTION_NAMES):
    raise ValueError(
        f"PCS_V1_BINDING keys {sorted(PCS_V1_BINDING)} must equal the registry "
        f"projections {sorted(pcs.PROJECTION_NAMES)}"
    )

# Consumption lens: the v1 impact projection plus the two window-matched
# responsibility projections.
PCS_V2_BINDING: dict[str, str] = {
    pcs.IMPACT_INDEX: "trend_c_per_decade",
    pcs.RESPONSIBILITY_CONSUMPTION_INDEX: "cum_consumption_t_per_capita",
    pcs.RESPONSIBILITY_PRODUCTION_MATCHED_INDEX: "cum_co2_window_t_per_capita",
}

# On-disk schema of projections_consumption.parquet (DuckDB types), in order.
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

# People-weighted exposure lens.
PCS_V2_EXPOSURE_BINDING: dict[str, str] = {
    pcs.RESPONSIBILITY_INDEX: "cum_co2_t_per_capita",
    pcs.IMPACT_INDEX: "trend_c_per_decade",
    pcs.IMPACT_POPULATION_WEIGHTED_INDEX: "trend_c_per_decade_pop_weighted",
}

# On-disk schema of projections_exposure.parquet (DuckDB types), in order.
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

# Area-weighted exposure lens.
PCS_V2_AREA_BINDING: dict[str, str] = {
    pcs.RESPONSIBILITY_INDEX: "cum_co2_t_per_capita",
    pcs.IMPACT_INDEX: "trend_c_per_decade",
    pcs.IMPACT_AREA_WEIGHTED_INDEX: "trend_c_per_decade_area_weighted",
}

# On-disk schema of projections_area.parquet (DuckDB types), in order.
PROJECTIONS_AREA_SCHEMA: dict[str, str] = {
    ID_COL: "VARCHAR",
    pcs.RESPONSIBILITY_INDEX: "DOUBLE",
    pcs.IMPACT_INDEX: "DOUBLE",
    pcs.IMPACT_AREA_WEIGHTED_INDEX: "DOUBLE",
}
PROJECTIONS_AREA_COLUMNS = tuple(PROJECTIONS_AREA_SCHEMA)

DEFAULT_PROJECTIONS_AREA_PATH = PROCESSED_DIR / "projections_area.parquet"

# Each lens binds a registered subset of the v2 registry; caught at import.
for _binding_name, _binding in (
    ("PCS_V2_BINDING", PCS_V2_BINDING),
    ("PCS_V2_EXPOSURE_BINDING", PCS_V2_EXPOSURE_BINDING),
    ("PCS_V2_AREA_BINDING", PCS_V2_AREA_BINDING),
):
    if not set(_binding) <= set(pcs.PROJECTION_NAMES_V2):
        raise ValueError(
            f"{_binding_name} keys {sorted(_binding)} must be registered v2 "
            f"projections {sorted(pcs.PROJECTION_NAMES_V2)}"
        )


def resolve_projections(inequality: pd.DataFrame) -> pd.DataFrame:
    """Emit the v1 projection table: ``Country`` plus the two v1 projections.

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
        out[projection] = inequality[source].to_numpy()
    return out[list(PROJECTIONS_COLUMNS)]


def build_projections(
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_path: Path = DEFAULT_PROJECTIONS_PATH,
) -> Path:
    """Materialize ``projections_v1.parquet`` from the upstream country table.

    Args:
        inequality_path: ``country_inequality.parquet``.
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
    """Emit the consumption-lens projection table.

    Countries with no OWID consumption series carry NULLs in the consumption
    sources and are dropped, so the lens covers only countries with a window.

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
        out[projection] = inequality[source].to_numpy()
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

    ``consumption_start_year`` is not a projection, so the window is summarised
    from the country table and injected into the consumption summary. The window
    end is the shared analysis cutoff; only the per-country start years vary.

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
        inequality_path: ``country_inequality.parquet``.
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
    """Emit the people-weighted exposure projection table.

    Countries with no people-weighted exposure (NULL, e.g. the population grid
    was absent at build time) are dropped.

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
        out[projection] = inequality[source].to_numpy()
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
    """Coverage label for the exposure lens: how many countries are people-weighted
    and how complete the weighting is (summary metadata).

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


def resolve_area_projections(inequality: pd.DataFrame) -> pd.DataFrame:
    """Emit the area-weighted exposure projection table.

    Countries with no area-weighted exposure (NULL: the Berkeley grid was absent
    at build time, or no grid cell resolved to the country) are dropped.

    Raises:
        ValueError: if a bound source column or the ``Country`` key is absent.
    """
    required = [ID_COL, *PCS_V2_AREA_BINDING.values()]
    missing = [c for c in required if c not in inequality.columns]
    if missing:
        raise ValueError(
            f"cannot resolve PCS v2 area projections: source column(s) "
            f"{missing} absent from the upstream table"
        )
    out = pd.DataFrame({ID_COL: inequality[ID_COL].to_numpy()})
    for projection, source in PCS_V2_AREA_BINDING.items():
        out[projection] = inequality[source].to_numpy()
    out = out[list(PROJECTIONS_AREA_COLUMNS)]
    covered = out[pcs.IMPACT_AREA_WEIGHTED_INDEX].notna()
    n_drop = int((~covered).sum())
    if n_drop:
        logger.info(
            "area projections: dropping %d countries without area-weighting", n_drop
        )
    return out.loc[covered].reset_index(drop=True)


def area_coverage(inequality: pd.DataFrame) -> dict:
    """Coverage label for the area lens: how many countries are area-weighted and
    how completely their assigned land cells were fit (summary metadata).

    Returns:
        ``{n_countries, mean_area_cell_coverage}`` over the area-weighted
        countries (empty-safe).
    """
    weighted = inequality.loc[
        inequality["trend_c_per_decade_area_weighted"].notna()
    ]
    if weighted.empty:
        return {"n_countries": 0, "mean_area_cell_coverage": None}
    return {
        "n_countries": int(len(weighted)),
        "mean_area_cell_coverage": float(weighted["area_cell_coverage"].mean()),
    }


def build_area_projections(
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_path: Path = DEFAULT_PROJECTIONS_AREA_PATH,
) -> Path:
    """Materialize ``projections_area.parquet`` from the country table."""
    inequality = pd.read_parquet(inequality_path)
    projections = resolve_area_projections(inequality)
    write_typed_parquet(
        projections, out_path, PROJECTIONS_AREA_SCHEMA, order_by=(ID_COL,)
    )
    logger.info("wrote %s (%d countries)", out_path, len(projections))
    return out_path


def main() -> None:
    """Write the v1 projection table and print the binding."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = build_projections()
    projections = pd.read_parquet(out)
    print(f"PCS {pcs.PCS_VERSION} projection table")
    for projection, source in PCS_V1_BINDING.items():
        print(f"  {projection} := {source}")
    print(f"  columns: {list(projections.columns)}  rows: {len(projections)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
