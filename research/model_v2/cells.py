"""What the country outcome discards: within-country structure of the 1° trend field.

The V1 primary outcome collapses every 1° Berkeley Earth cell trend inside a country
to one cos-latitude-weighted mean (:mod:`src.area_weighting`). This module reads the
per-cell field (computed once with the identical operator and window and cached
under ``data/processed/model_v2/``) and reports, per country, the spread of the cell
trends around that mean, plus the area-weighted between/within-country variance
split of the whole land field. Diagnostics only; nothing enters the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.area_weighting import AREA_WEIGHTED_COL
from src.data_io import PROCESSED_DIR
from src.population import latitude_area_weights

CELL_TRENDS_PATH = PROCESSED_DIR / "model_v2" / "berkeley_cell_trends_1950_2013.npz"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CELL_STATS_PATH = OUTPUT_DIR / "country_cell_trend_stats.csv"


def load_cell_field(path: Path = CELL_TRENDS_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(lats, lons, slopes, iso3)`` of the cached 1° field (``iso3 == ""`` off-country)."""
    with np.load(path, allow_pickle=False) as npz:
        return npz["lats"], npz["lons"], npz["slopes"], npz["iso3"]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cw, q * cw[-1])])


def country_cell_stats(lats: np.ndarray, slopes: np.ndarray, iso3: np.ndarray) -> pd.DataFrame:
    """Per-country spread of the cell trends (cos-latitude weights, as in V1)."""
    lat_grid = np.broadcast_to(lats[:, None], slopes.shape)
    rows = []
    for code in sorted(set(iso3[iso3 != ""].tolist())):
        assigned = iso3 == code
        valid = assigned & np.isfinite(slopes)
        if not valid.any():
            continue
        w = latitude_area_weights(lat_grid[valid])
        v = slopes[valid]
        mean = float(np.dot(w, v))
        sd = float(np.sqrt(np.dot(w, (v - mean) ** 2)))
        rows.append(
            {
                "iso3": code,
                "n_cells_1deg": int(assigned.sum()),
                "n_cells_fit": int(valid.sum()),
                "cell_mean": mean,
                "cell_sd": sd,
                "cell_p10": _weighted_quantile(v, w, 0.10),
                "cell_p90": _weighted_quantile(v, w, 0.90),
                "cell_min": float(v.min()),
                "cell_max": float(v.max()),
                "cell_range": float(v.max() - v.min()),
            }
        )
    return pd.DataFrame(rows)


def variance_split(lats: np.ndarray, slopes: np.ndarray, iso3: np.ndarray, subset: set[str] | None = None) -> dict:
    """Area-weighted total, between-country and within-country variance of the cell trends."""
    valid = np.isfinite(slopes) & (iso3 != "")
    if subset is not None:
        valid &= np.isin(iso3, list(subset))
    lat_grid = np.broadcast_to(lats[:, None], slopes.shape)
    w = latitude_area_weights(lat_grid[valid])
    v = slopes[valid]
    codes = iso3[valid]
    grand = float(np.dot(w, v))
    total = float(np.dot(w, (v - grand) ** 2))
    means = pd.Series(v * w).groupby(codes).sum() / pd.Series(w).groupby(codes).sum()
    fitted = means.reindex(codes).to_numpy()
    between = float(np.dot(w, (fitted - grand) ** 2))
    within = float(np.dot(w, (v - fitted) ** 2))
    return {
        "n_cells": int(valid.sum()),
        "n_countries": int(len(means)),
        "grand_mean": grand,
        "total_variance": total,
        "between_country_variance": between,
        "within_country_variance": within,
        "between_share": between / total,
        "within_share": within / total,
    }


def build(inequality_path: Path | None = None) -> pd.DataFrame:
    """Write ``country_cell_trend_stats.csv`` after checking the means reproduce V1's column."""
    lats, lons, slopes, iso3 = load_cell_field()
    stats = country_cell_stats(lats, slopes, iso3)
    if inequality_path is not None:
        table = pd.read_parquet(inequality_path)
        owid = (
            pd.read_csv(PROCESSED_DIR.parent / "raw" / "owid" / "owid-co2-data.csv", usecols=["country", "iso_code"])
            .dropna().drop_duplicates("country").set_index("country")["iso_code"]
        )
        table["iso3"] = table["owid_country"].map(owid)
        check = table.dropna(subset=[AREA_WEIGHTED_COL]).merge(stats, on="iso3")
        gap = float(np.max(np.abs(check[AREA_WEIGHTED_COL] - check["cell_mean"])))
        if gap > 1e-9:
            raise AssertionError(f"cell means do not reproduce the V1 area-weighted column (max gap {gap:g})")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(CELL_STATS_PATH, index=False, float_format="%.8g")
    return stats


if __name__ == "__main__":
    from src.emissions import DEFAULT_INEQUALITY_PATH

    stats = build(DEFAULT_INEQUALITY_PATH)
    lats, lons, slopes, iso3 = load_cell_field()
    print(f"{len(stats)} countries with fitted cells; V1 area-weighted column reproduced")
    print("variance split, all countries:", variance_split(lats, slopes, iso3))
    print(stats.describe().T.to_string())
