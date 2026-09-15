"""Evaluate the candidate spatial cross-validation designs on the frozen M0.

The model is never changed: every candidate is scored with the V1 estimator
(:func:`research.model_v2.cv.fit_predict_v1`) so the comparison isolates the
fold design. Writes ``outputs/cv_design_candidates.json`` and
``outputs/cv_design_candidates.csv``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.model_v2 import borders, cv
from research.model_v2.diagnostics import COUNTRY_TABLE_PATH
from research.model_v2.m0 import OUTPUT_DIR, load_inputs, m0_complete_design
from research.model_v2.spatial import haversine_matrix

CANDIDATES_JSON = OUTPUT_DIR / "cv_design_candidates.json"
CANDIDATES_CSV = OUTPUT_DIR / "cv_design_candidates.csv"
N_PERMUTATIONS = 199


def candidate_designs(table: pd.DataFrame) -> dict[str, dict]:
    """Name -> {fold_id, buffer_km, family, description}."""
    lon, lat = table["centroid_lon"].to_numpy(), table["centroid_lat"].to_numpy()
    n = len(table)
    designs = {
        "random_10fold": {"fold_id": cv.folds_random(n, 10, seed=0), "family": "random", "description": "random 10-fold (non-spatial reference)"},
        "loo_unbuffered": {"fold_id": np.arange(n), "family": "loo", "description": "leave-one-country-out, no buffer"},
        "continent_lo": {"fold_id": cv.folds_from_labels(table["spatial_block"]), "family": "region", "description": "leave-one-continent-out (V1 spatial_block, 6 folds)"},
        "m49_subregion_lo": {"fold_id": cv.folds_from_labels(table["m49_subregion"]), "family": "region", "description": "leave-one-M49-sub-region-out (20 folds)"},
        "latitude_bands": {"fold_id": cv.folds_latitude_bands(table["abs_centroid_lat"].to_numpy()), "family": "gradient", "description": "absolute-latitude bands (adversarial)"},
    }
    for k in (6, 8, 10, 12, 15):
        designs[f"kmeans_{k}"] = {"fold_id": cv.folds_kmeans(lon, lat, k, seed=0), "family": "cluster", "description": f"k-means on land-area centroids, k = {k}"}
    for buffer in (500, 1000, 1500, 2000):
        designs[f"centroid_buffered_loo_{buffer}km"] = {"fold_id": np.arange(n), "buffer_km": float(buffer), "family": "buffered", "metric": "centroid", "description": f"leave-one-country-out, {buffer} km buffer on centroid distance"}
        designs[f"border_buffered_loo_{buffer}km"] = {"fold_id": np.arange(n), "buffer_km": float(buffer), "family": "buffered", "metric": "border", "description": f"leave-one-country-out, {buffer} km buffer on minimum inter-territory distance"}
    for buffer in (500, 1000):
        designs[f"m49_subregion_lo_buffer{buffer}km"] = {"fold_id": cv.folds_from_labels(table["m49_subregion"]), "buffer_km": float(buffer), "family": "region", "description": f"leave-one-M49-sub-region-out with a {buffer} km buffer"}
        designs[f"kmeans_10_buffer{buffer}km"] = {"fold_id": cv.folds_kmeans(lon, lat, 10, seed=0), "buffer_km": float(buffer), "family": "cluster", "description": f"k-means k = 10 with a {buffer} km buffer"}
    for d in designs.values():
        d.setdefault("buffer_km", None)
        d.setdefault("metric", "centroid")
    return designs


def evaluate(table: pd.DataFrame, design: pd.DataFrame, designs: dict[str, dict]) -> tuple[dict, pd.DataFrame]:
    distances = {
        "centroid": haversine_matrix(table["centroid_lon"].to_numpy(), table["centroid_lat"].to_numpy()),
        "border": borders.load_or_build(table),
    }
    w_v1 = cv.v1_weights(table["station_lon"].to_numpy(), table["station_lat"].to_numpy())
    fitted = cv.V1Design().fit(design).predict(design)
    results, rows = {}, []
    for name, spec in designs.items():
        dist = distances[spec["metric"]]
        descriptors = cv.design_descriptors(design, spec["fold_id"], dist, spec["buffer_km"])
        scores = {}
        for rule in ("mean_effect", "reference"):
            res = cv.cross_validate(design, spec["fold_id"], lambda tr, te, r=rule: cv.fit_predict_v1(tr, te, unseen=r), dist=dist, buffer_km=spec["buffer_km"])
            scores[rule] = cv.scorecard(design, fitted, res, w_v1, n_permutations=N_PERMUTATIONS, shares=False)
        results[name] = {"family": spec["family"], "description": spec["description"], "buffer_km": spec["buffer_km"], "distance_metric": spec["metric"],
                         "descriptors": descriptors, "m0_scores": scores}
        s = scores["mean_effect"]
        rows.append({
            "design": name, "family": spec["family"], "metric": spec["metric"], "buffer_km": spec["buffer_km"], "n_folds": descriptors["n_folds"],
            "fold_size_min": descriptors["fold_size_min"], "fold_size_max": descriptors["fold_size_max"],
            "fold_mean_outcome_sd": descriptors["fold_mean_outcome_sd"],
            "unseen_levels": sum(descriptors["test_rows_with_unseen_level_by_categorical"].values()),
            "nearest_train_km_median": descriptors["nearest_train_km_median"],
            "share_train_within_500km": descriptors["share_test_rows_with_train_within_500km"],
            "mean_n_train": s["mean_n_train"],
            "cv_r2": s["cv_r2"], "cv_rmse": s["cv_rmse"], "cv_mae": s["cv_mae"],
            "fold_rmse_median": s["fold_rmse_median"], "fold_rmse_max": s["fold_rmse_max"],
            "calibration_slope": s["calibration_slope"], "moran_cv": s["residual_morans_i_cv"],
            "cv_r2_reference_rule": scores["reference"]["cv_r2"],
        })
    summary = pd.DataFrame(rows)
    return results, summary


def fold_membership(table: pd.DataFrame, designs: dict[str, dict]) -> pd.DataFrame:
    """Country -> fold id for every candidate (for inspection and for the frozen protocol)."""
    out = table[["Country", "iso3", "m49_subregion", "spatial_block"]].copy()
    for name, spec in designs.items():
        if spec["family"] in ("region", "cluster", "gradient", "random"):
            out[name] = spec["fold_id"]
    return out


def main() -> None:
    table = pd.read_csv(COUNTRY_TABLE_PATH)
    inequality, city_features, income = load_inputs()
    design = m0_complete_design(inequality, city_features, income)
    assert (design["Country"].to_numpy() == table["Country"].to_numpy()).all()
    designs = candidate_designs(table)
    results, summary = evaluate(table, design, designs)
    CANDIDATES_JSON.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    summary.to_csv(CANDIDATES_CSV, index=False, float_format="%.5g")
    fold_membership(table, designs).to_csv(OUTPUT_DIR / "cv_fold_membership.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
