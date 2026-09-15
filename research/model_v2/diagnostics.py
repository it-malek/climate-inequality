"""M0 residual diagnostics: everything the V2 plan needs to know about the V1 residual.

Descriptive only. The model is never refit with a different design; every quantity
here is computed from the tagged model's residuals (:mod:`research.model_v2.m0`)
joined to descriptive country geometry (:mod:`research.model_v2.geometry`), the
within-country cell-trend statistics (:mod:`research.model_v2.cells`), the ERA5
cross-check column already in the bundle, and the M49 sub-regions
(:mod:`research.model_v2.regions`).

Outputs (``research/model_v2/outputs/``):

* ``m0_countries.csv`` -- one row per country: outcome, fit, residual, influence
  diagnostics, V1 features, descriptors and the local Moran's I label;
* ``m0_residual_diagnostics.json`` -- the numeric diagnostics the report quotes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.model_v2 import cells, geometry, regions
from research.model_v2.m0 import OUTPUT_DIR, M0Fit, m0_country_table
from research.model_v2.spatial import (
    contiguity_weights,
    distance_band_weights,
    gearys_c,
    haversine_matrix,
    inverse_distance_weights,
    knn_weights,
    local_morans,
    morans_i,
)

COUNTRY_TABLE_PATH = OUTPUT_DIR / "m0_countries.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "m0_residual_diagnostics.json"

N_PERMUTATIONS = 999
SEED = 0
K_VALUES = (4, 6, 8, 10, 12, 15)
BAND_CUTOFFS_KM = (1000, 1500, 2000, 3000)
CORRELOGRAM_EDGES_KM = (0, 500, 1000, 1500, 2000, 3000, 4000, 6000, 9000)


def assemble_table() -> tuple[pd.DataFrame, M0Fit]:
    """M0 country table joined to geometry, cell statistics and M49 sub-regions."""
    table, fit = m0_country_table()
    geo = pd.read_csv(geometry.GEOMETRY_PATH)
    cell = pd.read_csv(cells.CELL_STATS_PATH)
    table = table.merge(geo, on="iso3", how="left").merge(cell, on="iso3", how="left")
    table = table.merge(regions.m49_table(), on="iso3", how="left")
    if table["m49_subregion"].isna().any():
        missing = table.loc[table["m49_subregion"].isna(), "Country"].tolist()
        raise ValueError(f"no M49 sub-region for {missing}")
    table["abs_residual"] = table["residual"].abs()
    table["sq_share"] = table["residual"] ** 2 / float(np.sum(table["residual"] ** 2))
    table["era5_minus_berkeley"] = table["era5_trend"] - table["warming_trend"]
    table["station_minus_area"] = table["station_trend"] - table["warming_trend"]
    table["station_lat_minus_centroid_lat"] = table["abs_latitude"] - table["abs_centroid_lat"]
    table["log10_land_area"] = np.log10(table["land_area_km2"])
    table["koppen_modal_vs_area_mismatch"] = table["climate_zone"] != table["koppen_area_dominant"]
    return table, fit


def distribution(resid: np.ndarray) -> dict:
    """Moments, quantiles and normality tests of the residual."""
    q = np.percentile(resid, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    shapiro = stats.shapiro(resid)
    jb = stats.jarque_bera(resid)
    return {
        "n": int(len(resid)),
        "mean": float(resid.mean()),
        "sd": float(resid.std(ddof=1)),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "skew": float(stats.skew(resid)),
        "excess_kurtosis": float(stats.kurtosis(resid)),
        "quantiles": dict(zip(["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"], map(float, q), strict=True)),
        "shapiro_w": float(shapiro.statistic),
        "shapiro_p": float(shapiro.pvalue),
        "jarque_bera_p": float(jb.pvalue),
    }


def concentration(table: pd.DataFrame) -> dict:
    """How much of the residual sum of squares a few countries carry."""
    shares = table.sort_values("sq_share", ascending=False)
    cum = shares["sq_share"].cumsum().to_numpy()
    top = shares[["Country", "residual", "sq_share"]].head(20)
    return {
        "top5_share_of_ss": float(cum[4]),
        "top10_share_of_ss": float(cum[9]),
        "top20_share_of_ss": float(cum[19]),
        "countries_for_half_of_ss": int(np.searchsorted(cum, 0.5) + 1),
        "expected_top10_share_if_uniform": 10 / len(table),
        "top20": [
            {"country": c, "residual": float(r), "share_of_ss": float(s)}
            for c, r, s in zip(top["Country"], top["residual"], top["sq_share"], strict=True)
        ],
    }


def heteroscedasticity(table: pd.DataFrame) -> dict:
    """Does the residual spread change with the fitted value or the outcome level?"""
    out = {}
    for col in ("fitted", "warming_trend", "abs_latitude", "log10_land_area"):
        rho, p = stats.spearmanr(table[col], table["abs_residual"])
        out[f"spearman_abs_residual_vs_{col}"] = {"rho": float(rho), "p": float(p)}
    # Breusch-Pagan style: regress e^2 on fitted, report R^2 and LM p.
    x = np.column_stack([np.ones(len(table)), table["fitted"]])
    e2 = table["residual"].to_numpy() ** 2
    coef, *_ = np.linalg.lstsq(x, e2, rcond=None)
    r2 = 1 - np.sum((e2 - x @ coef) ** 2) / np.sum((e2 - e2.mean()) ** 2)
    lm = len(table) * r2
    out["breusch_pagan_on_fitted"] = {"lm": float(lm), "p": float(1 - stats.chi2.cdf(lm, 1))}
    return out


def _spearman(table: pd.DataFrame, x: str, y: str = "residual") -> dict:
    sub = table[[x, y]].dropna()
    rho, p = stats.spearmanr(sub[x], sub[y])
    r, pr = stats.pearsonr(sub[x], sub[y])
    return {"n": int(len(sub)), "spearman_rho": float(rho), "spearman_p": float(p), "pearson_r": float(r), "pearson_p": float(pr)}


def descriptor_associations(table: pd.DataFrame) -> dict:
    """Rank correlations of the residual and |residual| with descriptive quantities.

    The residual is orthogonal to the V1 numeric columns in-sample (by OLS), so
    the interesting rows are the descriptors V1 does not contain: land area,
    coastal fraction, topographic spread, the within-country cell spread, the
    ERA5-Berkeley disagreement and the station-area disagreement.
    """
    descriptors = [
        "log10_land_area", "coastal_cell_fraction", "elev_sd_m", "elev_mean_m", "cell_sd",
        "lat_sd_area_weighted", "abs_centroid_lat", "station_lat_minus_centroid_lat",
        "era5_minus_berkeley", "station_minus_area", "koppen_share_B", "koppen_share_A",
        "koppen_share_D", "n_cities", "area_cell_coverage", "leverage",
        # V1 numeric columns, to demonstrate the orthogonality (Pearson r = 0 by construction)
        "abs_latitude", "elevation", "continentality", "station_density",
    ]
    return {
        d: {"residual": _spearman(table, d, "residual"), "abs_residual": _spearman(table, d, "abs_residual")}
        for d in descriptors if d in table.columns
    }


def group_structure(table: pd.DataFrame) -> dict:
    """Residual means by grouping, with a one-way ANOVA F test and eta squared."""
    out = {}
    groupings = {
        "m49_subregion": "m49_subregion",
        "koppen_area_dominant": "koppen_area_dominant",
        "koppen_modal_vs_area_mismatch": "koppen_modal_vs_area_mismatch",
        # V1 categoricals: residual means are zero by construction (recorded as evidence)
        "spatial_block_v1": "spatial_block",
        "climate_zone_v1": "climate_zone",
        "hemisphere_v1": "hemisphere",
        "income_group_v1": "income_group",
    }
    for key, col in groupings.items():
        groups = [g["residual"].to_numpy() for _, g in table.groupby(col) if len(g) > 0]
        f, p = stats.f_oneway(*groups) if len(groups) > 1 and all(len(g) > 1 for g in groups) else (np.nan, np.nan)
        grand = table["residual"].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = float(np.sum((table["residual"] - grand) ** 2))
        summary = (
            table.groupby(col)["residual"]
            .agg(n="size", mean="mean", sd="std", min="min", max="max")
            .sort_values("mean")
        )
        out[key] = {
            "anova_f": float(f), "anova_p": float(p), "eta_squared": float(ss_between / ss_total),
            "groups": {str(k): {kk: (float(v) if kk != "n" else int(v)) for kk, v in row.items()} for k, row in summary.iterrows()},
        }
    return out


def latitude_structure(table: pd.DataFrame) -> dict:
    """Binned residual means along absolute latitude, overall and by hemisphere."""
    edges = [0, 10, 20, 30, 40, 50, 70]
    table = table.assign(lat_bin=pd.cut(table["abs_centroid_lat"], edges, right=False))
    out = {"bins_deg": edges, "overall": {}, "by_hemisphere": {}}
    for b, g in table.groupby("lat_bin", observed=True):
        out["overall"][str(b)] = {"n": int(len(g)), "mean": float(g["residual"].mean()), "se": float(g["residual"].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else None}
    for h, gh in table.groupby("hemisphere"):
        out["by_hemisphere"][h] = {}
        for b, g in gh.groupby("lat_bin", observed=True):
            out["by_hemisphere"][h][str(b)] = {"n": int(len(g)), "mean": float(g["residual"].mean())}
        rho, p = stats.spearmanr(gh["abs_centroid_lat"], gh["residual"])
        out["by_hemisphere"][h]["spearman_residual_vs_abs_lat"] = {"rho": float(rho), "p": float(p), "n": int(len(gh))}
    # Within-hemisphere slope difference as a pre-registered interaction diagnostic.
    n = table[table["hemisphere"] == "N"]
    s = table[table["hemisphere"] == "S"]
    out["slope_residual_on_abs_lat"] = {
        "N": float(np.polyfit(n["abs_centroid_lat"], n["residual"], 1)[0]),
        "S": float(np.polyfit(s["abs_centroid_lat"], s["residual"], 1)[0]),
        "units": "degC/decade per degree of latitude",
    }
    return out


def weights_sensitivity(table: pd.DataFrame, resid: np.ndarray) -> dict:
    """Global Moran's I under the alternative weight definitions."""
    out = {}
    adjacency = pd.read_csv(geometry.ADJACENCY_PATH)
    for centroid in ("station", "area"):
        lon = table[f"{centroid}_lon" if centroid == "station" else "centroid_lon"].to_numpy()
        lat = table[f"{centroid}_lat" if centroid == "station" else "centroid_lat"].to_numpy()
        dist = haversine_matrix(lon, lat)
        for k in K_VALUES:
            r = morans_i(resid, knn_weights(dist, k), N_PERMUTATIONS, SEED)
            out[f"knn_k{k}_{centroid}"] = {"morans_i": r.statistic, "p": r.p_value, "z": r.z_score}
        for cutoff in BAND_CUTOFFS_KM:
            w, iso = distance_band_weights(dist, cutoff)
            r = morans_i(resid, w, N_PERMUTATIONS, SEED)
            out[f"band_{cutoff}km_{centroid}"] = {"morans_i": r.statistic, "p": r.p_value, "z": r.z_score, "n_isolates_given_nearest": iso}
        for power in (1.0, 2.0):
            r = morans_i(resid, inverse_distance_weights(dist, power, cutoff_km=3000), N_PERMUTATIONS, SEED)
            out[f"idw_p{int(power)}_3000km_{centroid}"] = {"morans_i": r.statistic, "p": r.p_value, "z": r.z_score}
        if centroid == "area":
            w, n_fallback = contiguity_weights(table["iso3"].to_numpy(), adjacency, dist, fallback_k=1)
            r = morans_i(resid, w, N_PERMUTATIONS, SEED)
            out["queen_contiguity_15min_raster"] = {"morans_i": r.statistic, "p": r.p_value, "z": r.z_score, "n_islands_given_nearest": n_fallback}
            g = gearys_c(resid, knn_weights(dist, 8), N_PERMUTATIONS, SEED)
            out["gearys_c_knn_k8_area"] = {"gearys_c": g.statistic, "p": g.p_value, "z": g.z_score}
    values = [v["morans_i"] for v in out.values() if "morans_i" in v]
    out["range_of_morans_i"] = {"min": float(min(values)), "max": float(max(values))}
    return out


def correlogram(table: pd.DataFrame, resid: np.ndarray) -> dict:
    """Moran's I by distance ring and a residual semivariogram (area centroids)."""
    dist = haversine_matrix(table["centroid_lon"].to_numpy(), table["centroid_lat"].to_numpy())
    z = resid - resid.mean()
    n = len(z)
    rings, semivariogram = [], []
    edges = CORRELOGRAM_EDGES_KM
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        w = ((dist >= lo) & (dist < hi)).astype(float)
        np.fill_diagonal(w, 0.0)
        n_pairs = int(w.sum() / 2)
        if n_pairs == 0:
            continue
        # Binary weights, general Moran's I formula (n / S0) * z'Wz / z'z.
        r = morans_i(z, w, N_PERMUTATIONS, SEED)
        rings.append({"lo_km": lo, "hi_km": hi, "n_pairs": n_pairs, "morans_i": r.statistic, "p": r.p_value, "z": r.z_score,
                      "expected": -1.0 / (n - 1)})
        iu = np.triu_indices(n, 1)
        sel = w[iu] > 0
        gamma = float(0.5 * np.mean((z[iu[0]][sel] - z[iu[1]][sel]) ** 2))
        semivariogram.append({"lo_km": lo, "hi_km": hi, "n_pairs": n_pairs, "semivariance": gamma})
    return {"rings": rings, "semivariogram": semivariogram, "residual_variance": float(z.var(ddof=1)),
            "practical_range_note": "the first ring whose Moran's I is not significantly positive bounds the residual correlation length"}


def local_structure(table: pd.DataFrame, resid: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Local Moran's I (LISA) on the V1 weights (station centroids, k = 8) and on area kNN k = 8."""
    dist_station = haversine_matrix(table["station_lon"].to_numpy(), table["station_lat"].to_numpy())
    dist_area = haversine_matrix(table["centroid_lon"].to_numpy(), table["centroid_lat"].to_numpy())
    lisa_v1 = local_morans(resid, knn_weights(dist_station, 8), N_PERMUTATIONS, SEED)
    lisa_area = local_morans(resid, knn_weights(dist_area, 8), N_PERMUTATIONS, SEED)
    table = table.assign(
        lisa_v1_local_i=lisa_v1["local_i"].to_numpy(), lisa_v1_p=lisa_v1["p_value"].to_numpy(),
        lisa_v1_quadrant=lisa_v1["quadrant"].to_numpy(), lisa_v1_label=lisa_v1["label"].to_numpy(),
        lisa_area_label=lisa_area["label"].to_numpy(),
    )
    clusters = {}
    for label in ("HH", "LL", "HL", "LH"):
        members = table[table["lisa_v1_label"] == label].sort_values("residual")
        clusters[label] = [
            {"country": c, "residual": float(r), "subregion": s, "p": float(p)}
            for c, r, s, p in zip(members["Country"], members["residual"], members["m49_subregion"], members["lisa_v1_p"], strict=True)
        ]
    agreement = float(np.mean(table["lisa_v1_label"] == table["lisa_area_label"]))
    return table, {"weights": "station-centroid kNN k=8 (V1) with conditional permutation p < 0.05", "clusters": clusters,
                   "n_significant": int((table["lisa_v1_label"] != "ns").sum()), "label_agreement_station_vs_area_weights": agreement}


def moran_influence(table: pd.DataFrame, resid: np.ndarray) -> dict:
    """Does a handful of countries carry the spatial pattern? Drop-k and jackknife Moran's I."""
    lon, lat = table["station_lon"].to_numpy(), table["station_lat"].to_numpy()

    def moran_subset(mask: np.ndarray) -> float:
        d = haversine_matrix(lon[mask], lat[mask])
        return _moran_no_perm(resid[mask], knn_weights(d, 8))

    order = np.argsort(-np.abs(resid))
    drop_top = {}
    for k in (5, 10, 15, 20, 30):
        mask = np.ones(len(resid), dtype=bool)
        mask[order[:k]] = False
        drop_top[f"drop_top_{k}_abs_residual"] = float(moran_subset(mask))
    full = float(moran_subset(np.ones(len(resid), dtype=bool)))
    jack = np.empty(len(resid))
    for i in range(len(resid)):
        mask = np.ones(len(resid), dtype=bool)
        mask[i] = False
        jack[i] = moran_subset(mask)
    delta = full - jack
    ranked = np.argsort(-np.abs(delta))[:10]
    lisa_hh = (table["lisa_v1_label"] == "HH").to_numpy()
    lisa_ll = (table["lisa_v1_label"] == "LL").to_numpy()
    return {
        "full_sample_knn8_station": full,
        **drop_top,
        "drop_lisa_HH_cluster": float(moran_subset(~lisa_hh)),
        "drop_lisa_LL_cluster": float(moran_subset(~lisa_ll)),
        "drop_both_lisa_clusters": float(moran_subset(~(lisa_hh | lisa_ll))),
        "jackknife_max_abs_delta": float(np.max(np.abs(delta))),
        "jackknife_top10": [{"country": str(table["Country"].iloc[i]), "delta_i": float(delta[i])} for i in ranked],
    }


def _moran_no_perm(values: np.ndarray, w: np.ndarray) -> float:
    z = values - values.mean()
    return float((len(z) / w.sum()) * (z @ w @ z) / np.sum(z**2))


def product_and_construction(table: pd.DataFrame) -> dict:
    """Residual against the observational-product and construction disagreements."""
    out = {
        "era5_minus_berkeley": _spearman(table, "era5_minus_berkeley"),
        "station_minus_area": _spearman(table, "station_minus_area"),
        "era5_vs_berkeley_outcome_spearman": float(stats.spearmanr(table["era5_trend"], table["warming_trend"]).statistic),
    }
    # Share of residual variance linearly aligned with the product disagreement (descriptive R^2).
    sub = table[["residual", "era5_minus_berkeley"]].dropna()
    x = np.column_stack([np.ones(len(sub)), sub["era5_minus_berkeley"]])
    coef, *_ = np.linalg.lstsq(x, sub["residual"], rcond=None)
    fitted = x @ coef
    out["residual_variance_aligned_with_product_gap_r2"] = float(1 - np.sum((sub["residual"] - fitted) ** 2) / np.sum((sub["residual"] - sub["residual"].mean()) ** 2))
    out["slope_residual_per_unit_product_gap"] = float(coef[1])
    big = table.reindex(table["era5_minus_berkeley"].abs().sort_values(ascending=False).index).head(10)
    out["largest_product_gaps"] = [
        {"country": c, "berkeley": float(b), "era5": float(e), "residual": float(r)}
        for c, b, e, r in zip(big["Country"], big["warming_trend"], big["era5_trend"], big["residual"], strict=True)
    ]
    return out


def influence_summary(table: pd.DataFrame, fit: M0Fit) -> dict:
    """Leverage, studentised residuals and Cook's distance, with the notable cases."""
    p = fit.rank
    n = fit.n
    high = table[table["leverage"] > 2 * p / n].sort_values("leverage", ascending=False)
    cooks = table.reindex(table["cooks_d"].sort_values(ascending=False).index).head(10)
    return {
        "rank": p, "n": n, "mean_leverage": float(p / n), "high_leverage_threshold_2p_over_n": float(2 * p / n),
        "n_high_leverage": int(len(high)),
        "high_leverage": [{"country": c, "leverage": float(h), "note": _leverage_note(row)} for c, h, (_, row) in zip(high["Country"], high["leverage"], high.iterrows(), strict=True)],
        "exactly_fitted": table.loc[table["leverage"] > 1 - 1e-8, "Country"].tolist(),
        "n_abs_studentized_gt_2": int((table["studentized"].abs() > 2).sum()),
        "abs_studentized_gt_2": table.loc[table["studentized"].abs() > 2, ["Country", "studentized"]].sort_values("studentized").to_dict("records"),
        "cooks_top10": [{"country": c, "cooks_d": float(d)} for c, d in zip(cooks["Country"], cooks["cooks_d"], strict=True)],
        "cooks_threshold_4_over_n": float(4 / n),
    }


def _leverage_note(row: pd.Series) -> str:
    notes = []
    if row["spatial_block"] == "Oceania":
        notes.append("Oceania block (3 countries)")
    if row["climate_zone"] == "E":
        notes.append("only Köppen E country")
    if row["station_density"] > 20:
        notes.append(f"station density {row['station_density']:.0f}")
    if row["elevation"] > 3000:
        notes.append(f"station elevation {row['elevation']:.0f} m")
    if row["elevation"] < 0:
        notes.append(f"negative station elevation {row['elevation']:.0f} m (offshore grid snap)")
    return "; ".join(notes)


def run() -> dict:
    """Compute every diagnostic block, write the country table and the JSON."""
    table, fit = assemble_table()
    resid = table["residual"].to_numpy()
    table, lisa = local_structure(table, resid)
    lats, _, slopes, iso3 = cells.load_cell_field()
    report = {
        "m0": {"n": fit.n, "r2": fit.r2, "rank": fit.rank, "columns": fit.p, "sigma": fit.sigma,
               "outcome_sd": float(table["warming_trend"].std(ddof=1))},
        "distribution": distribution(resid),
        "concentration": concentration(table),
        "heteroscedasticity": heteroscedasticity(table),
        "influence": influence_summary(table, fit),
        "descriptor_associations": descriptor_associations(table),
        "group_structure": group_structure(table),
        "latitude_structure": latitude_structure(table),
        "weights_sensitivity": weights_sensitivity(table, resid),
        "correlogram": correlogram(table, resid),
        "local_moran": lisa,
        "moran_influence": moran_influence(table, resid),
        "product_and_construction": product_and_construction(table),
        "collapse_information_loss": {
            "all_fitted_countries": cells.variance_split(lats, slopes, iso3),
            "m0_countries": cells.variance_split(lats, slopes, iso3, subset=set(table["iso3"])),
            "within_country_cell_sd": {"median": float(table["cell_sd"].median()), "p90": float(table["cell_sd"].quantile(0.9)), "max": float(table["cell_sd"].max()),
                                       "max_country": str(table.loc[table["cell_sd"].idxmax(), "Country"])},
            "residual_sd_for_comparison": float(resid.std(ddof=1)),
            "koppen_modal_vs_area_mismatch_count": int(table["koppen_modal_vs_area_mismatch"].sum()),
            "station_vs_centroid_abs_lat_gap": {"median_abs": float(table["station_lat_minus_centroid_lat"].abs().median()), "max_abs": float(table["station_lat_minus_centroid_lat"].abs().max())},
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(COUNTRY_TABLE_PATH, index=False, float_format="%.8g")
    DIAGNOSTICS_PATH.write_text(json.dumps(report, indent=1, default=_json_default) + "\n", encoding="utf-8")
    return report


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not serialisable: {type(value)}")


if __name__ == "__main__":
    report = run()
    print(json.dumps({k: report[k] for k in ("m0", "distribution", "concentration")}, indent=1)[:3000])
    print("wrote", COUNTRY_TABLE_PATH, "and", DIAGNOSTICS_PATH)
