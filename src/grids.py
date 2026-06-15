"""Shared nearest-cell lookup for sampling regular lat/lon grids.

Both :mod:`src.validation` (Phase 6, the Berkeley Earth gridded
temperature NetCDF) and :mod:`src.explain` (Phase 7, the ETOPO elevation
and Koeppen climate-class NetCDFs) need to map a set of city-location
coordinates onto the nearest cell of a static ``(lat, lon)`` grid. This
module factors out that one idiom so both call sites -- and their
determinism/boundary-case tests -- share a single implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def nearest_cell_indices(
    lat_index: pd.Index,
    lon_index: pd.Index,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-cell positions in `lat_index`/`lon_index` for each point.

    Args:
        lat_index: Grid latitude coordinate values (any order).
        lon_index: Grid longitude coordinate values (any order).
        lats: Query latitudes.
        lons: Query longitudes.

    Returns:
        `(lat_pos, lon_pos)` integer position arrays into `lat_index` /
        `lon_index`. Points outside the grid's range clamp to the nearest
        edge cell (``pandas.Index.get_indexer`` with ``method="nearest"``).
    """
    lat_pos = lat_index.get_indexer(lats, method="nearest")
    lon_pos = lon_index.get_indexer(lons, method="nearest")
    return lat_pos, lon_pos
