"""M0 scorecard entry: corrected territorial implementation; legacy logic retained.

Primary protocol: leave-one-country-out with a 500 km exclusion buffer on the
minimum inter-territory distance (every country any part of whose land lies within
500 km of the held-out country's land is removed from training). Secondary stress
test: leave-one-UN-M49-sub-region-out. Reference (not a test): random 10-fold,
seed 0. Sensitivities (reported, never criteria): a 1000 km border buffer and a
1500 km buffer on land-area-centroid distance. Unseen categorical levels receive
the mean training level effect. The current entry writes ``outputs/m0_scorecard_territory_corrected.json``.
``legacy_main`` retains the original 1 deg cell-centre implementation for provenance; do not run
it to replace the committed legacy scorecard.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.model_v2 import borders, cv
from research.model_v2.diagnostics import COUNTRY_TABLE_PATH
from research.model_v2.m0 import M0_REFERENCE, OUTPUT_DIR, load_inputs, m0_complete_design
from research.model_v2.spatial import haversine_matrix

SCORECARD_PATH = OUTPUT_DIR / "m0_scorecard.json"
PRIMARY_BUFFER_KM = 500.0  # on minimum inter-territory distance
SENSITIVITY_BORDER_BUFFER_KM = 1000.0
SENSITIVITY_CENTROID_BUFFER_KM = 1500.0
MIN_REGION_N = 3  # the worst-region headline ignores sub-regions with fewer countries
N_PERMUTATIONS = 999


def legacy_main() -> dict:
    table = pd.read_csv(COUNTRY_TABLE_PATH)
    inequality, city_features, income = load_inputs()
    design = m0_complete_design(inequality, city_features, income)
    assert (design["Country"].to_numpy() == table["Country"].to_numpy()).all()
    y = design["warming_trend"].to_numpy(dtype=float)
    dist_centroid = haversine_matrix(table["centroid_lon"].to_numpy(), table["centroid_lat"].to_numpy())
    dist_border = borders.load_or_build(table)
    w_v1 = cv.v1_weights(table["station_lon"].to_numpy(), table["station_lat"].to_numpy())
    fitted = cv.V1Design().fit(design).predict(design)
    region = table["m49_subregion"]
    n = len(design)

    protocols = {
        "primary_border_buffered_loo_500km": dict(fold_id=np.arange(n), buffer_km=PRIMARY_BUFFER_KM, dist=dist_border),
        "secondary_m49_subregion_lo": dict(fold_id=cv.folds_from_labels(region), buffer_km=None, dist=dist_border),
        "reference_random_10fold": dict(fold_id=cv.folds_random(n, 10, seed=0), buffer_km=None, dist=dist_border),
        "sensitivity_border_buffered_loo_1000km": dict(fold_id=np.arange(n), buffer_km=SENSITIVITY_BORDER_BUFFER_KM, dist=dist_border),
        "sensitivity_centroid_buffered_loo_1500km": dict(fold_id=np.arange(n), buffer_km=SENSITIVITY_CENTROID_BUFFER_KM, dist=dist_centroid),
    }
    out = {"m0_reference": M0_REFERENCE, "unseen_level_rule": "mean_effect", "weights_for_morans_i": M0_REFERENCE["morans_weights"],
           "primary": "leave-one-country-out, 500 km buffer on minimum inter-territory distance", "protocols": {}}
    for name, spec in protocols.items():
        res = cv.cross_validate(design, spec["fold_id"], dist=spec["dist"], buffer_km=spec["buffer_km"])
        score = cv.scorecard(design, fitted, res, w_v1, n_permutations=N_PERMUTATIONS, shares=name.startswith("primary"))
        regions = cv.region_errors(res, y, region)
        eligible = regions[regions["n"] >= MIN_REGION_N]
        score["worst_region"] = {"region": str(eligible.index[0]), "rmse": float(eligible.iloc[0]["rmse"]), "n": int(eligible.iloc[0]["n"]), "min_region_n": MIN_REGION_N}
        score["region_rmse"] = {str(k): float(v) for k, v in regions["rmse"].items()}
        score["region_bias"] = {str(k): float(v) for k, v in regions["bias"].items()}
        out["protocols"][name] = score
    # Per-country out-of-fold errors under the primary protocol, for the record.
    res = cv.cross_validate(design, np.arange(n), dist=dist_border, buffer_km=PRIMARY_BUFFER_KM)
    errors = pd.DataFrame({"Country": table["Country"], "iso3": table["iso3"], "m49_subregion": region, "observed": y, "cv_prediction": res.yhat,
                           "cv_error": y - res.yhat, "n_train": res.n_train, "nearest_train_km": res.nearest_train_km, "unseen_levels": res.unseen_levels})
    errors.to_csv(OUTPUT_DIR / "m0_cv_errors_primary.csv", index=False, float_format="%.6g")
    SCORECARD_PATH.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def main() -> dict:
    """Current M0 scorer; legacy files are never overwritten by this entry point."""
    from research.model_v2.run_territory_correction import main as corrected_main

    return corrected_main()


if __name__ == "__main__":
    out = main()
    print(out["corrected_primary"])
