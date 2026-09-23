"""Build the dashboard's summary of the residual-structure investigation.

The investigation under ``research/model_v2/`` writes its results as stage
records (scorecards, consolidated tables, bootstrap summaries, product checks).
The dashboard reads only the committed ``app/data/`` bundle, so this module
copies the values the site presents into two small assets:

* ``residual_structure_summary.json``: the full-rank static decomposition with
  its bootstrap intervals, the geographically separated scores of every tested
  extension, the spatial-model accounting, the exclusion-distance and
  temperature-product robustness rows, and the Spearman statistics of the
  responsibility comparison. Every number is copied from a research record
  (or, for the Spearman rows, computed from the committed country tables);
  nothing is re-estimated. The SHA-256 of each source file is recorded.
* ``national_warming.parquet``: one row per country with the station-,
  population- and area-weighted Berkeley Earth trends, the anomaly-aligned
  ERA5 area-weighted trend, and cumulative per-capita CO2, for the
  responsibility comparison and the national warming map.

    uv run python -m src.residual_assets
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from scipy import stats

from src.bundle import (
    APP_DATA_DIR,
    INEQUALITY_ASSET,
    NATIONAL_WARMING_ASSET,
    PROJECT_ROOT,
    RESIDUAL_SUMMARY_ASSET,
)
from src.data_io import round_floats

logger = logging.getLogger(__name__)

RESEARCH_OUT = PROJECT_ROOT / "research" / "model_v2" / "outputs"

SOURCES: dict[str, Path] = {
    "consolidated_table": RESEARCH_OUT / "m4_final" / "m4_consolidated_table.json",
    "bootstrap_summary": RESEARCH_OUT / "m4_final" / "m4_bootstrap_summary.json",
    "sensitivities": RESEARCH_OUT / "m4_final" / "m4_sensitivities.json",
    "products": RESEARCH_OUT / "m4_final" / "m4_products.json",
    "final_record": RESEARCH_OUT / "v2_final" / "v2_final_record.json",
    "static_scorecard": RESEARCH_OUT / "m0_scorecard_territory_corrected.json",
    "static_countries": RESEARCH_OUT / "m0_countries.csv",
    "era5_aligned_countries": RESEARCH_OUT / "product_stability_aligned_countries.csv",
}

# Consolidated-table stage label -> (dashboard key, label, one-line verdict).
INVESTIGATIONS: tuple[tuple[str, str, str, str], ...] = (
    ("M1a (area geography)", "geography_measurement",
     "Area-consistent geography measurement", "not promoted; kept as a measurement sensitivity"),
    ("M1b (M0* + C2 dryness)", "baseline_dryness",
     "Baseline hydroclimatic dryness", "not supported"),
    ("M2 (M0* + latitude functional form)", "latitude_form",
     "Flexible latitude functional form", "not supported"),
    ("M3-SEM(M0*)", "spatial_error",
     "Spatial error model", "qualifies for prediction"),
    ("M3-SAR(M0*)", "spatial_lag",
     "Spatial lag model", "qualifies for prediction"),
)
STATIC_STAGE = "M0* primary total CO2"
LAND_STAGES = {
    "spatial_error": "M3-SEM(M0*) [land-centroid kNN8 sensitivity]",
    "spatial_lag": "M3-SAR(M0*) [land-centroid kNN8 sensitivity]",
}
FAMILIES = {"spatial_error": "M3-SEM(M0*)", "spatial_lag": "M3-SAR(M0*)"}
GROUPS = ("geography", "socioeconomic", "population", "emissions", "residual")

# National warming definitions shown side by side (column, label, n differs).
WARMING_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("trend_station", "Station-weighted (Berkeley Earth city locations)"),
    ("trend_population", "Population-weighted (Berkeley Earth city locations)"),
    ("trend_area", "Area-weighted (Berkeley Earth 1 degree grid)"),
    ("trend_era5_aligned", "Area-weighted (ERA5, anomaly-aligned)"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict:
    return json.loads(SOURCES[name].read_text(encoding="utf-8"))


def _row(table: dict, stage: str) -> dict:
    for row in table["rows"]:
        if row["stage"] == stage:
            return row
    raise KeyError(f"stage {stage!r} not in the consolidated table")


def _shares(boot_arm: dict) -> dict:
    return {k: boot_arm["groups"][k] for k in GROUPS}


def build_national_warming(inequality_path: Path | None = None) -> pd.DataFrame:
    """Country table of the four national warming definitions and responsibility.

    Station, population and area trends come from the committed country table
    (157 countries; the area trend is absent for three). The anomaly-aligned
    ERA5 trend exists for the 151 decomposition countries and is joined by
    ISO3 through the research country table.
    """
    inequality_path = inequality_path or APP_DATA_DIR / INEQUALITY_ASSET
    country = pd.read_parquet(inequality_path)
    iso = pd.read_csv(SOURCES["static_countries"], usecols=["Country", "iso3"])
    era5 = pd.read_csv(SOURCES["era5_aligned_countries"], usecols=["iso3", "era5"])
    out = country[[
        "Country", "owid_country", "continent", "cum_co2_t_per_capita",
        "trend_c_per_decade", "trend_c_per_decade_pop_weighted",
        "trend_c_per_decade_area_weighted",
    ]].rename(columns={
        "trend_c_per_decade": "trend_station",
        "trend_c_per_decade_pop_weighted": "trend_population",
        "trend_c_per_decade_area_weighted": "trend_area",
    })
    out = out.merge(iso, on="Country", how="left")
    out = out.merge(era5.rename(columns={"era5": "trend_era5_aligned"}), on="iso3", how="left")
    return out.sort_values("Country", kind="stable").reset_index(drop=True)


def responsibility_comparison(national: pd.DataFrame) -> list[dict]:
    """Spearman rank correlation of each warming definition with responsibility."""
    rows = []
    for column, label in WARMING_DEFINITIONS:
        frame = national.dropna(subset=["cum_co2_t_per_capita", column])
        rho, p = stats.spearmanr(frame["cum_co2_t_per_capita"], frame[column])
        rows.append({
            "column": column, "label": label, "n": int(len(frame)),
            "spearman_rho": float(rho), "spearman_p": float(p),
        })
    return rows


def build_summary(national: pd.DataFrame) -> dict:
    """Assemble the summary from the research records (values copied verbatim)."""
    table = _load("consolidated_table")
    boot = _load("bootstrap_summary")["primary_total_co2"]
    boot_pc = _load("bootstrap_summary")["per_capita"]
    sens = _load("sensitivities")
    products = _load("products")
    final = _load("final_record")
    scorecard = _load("static_scorecard")

    static_row = _row(table, STATIC_STAGE)
    card = scorecard["corrected_primary"]
    static = {
        "n": int(card["n"]),
        "in_sample_r2": card["in_sample_r2"],
        "residual_share": static_row["allocation"]["residual"],
        "primary_cv_r2": static_row["primary_cv_r2"],
        "primary_cv_rmse": static_row["primary_cv_rmse"],
        "primary_cv_mae": static_row["primary_cv_mae"],
        "m49_cv_rmse": static_row["m49_cv_rmse"],
        "m49_cv_r2": scorecard["secondary_m49_unchanged"]["cv_r2"],
        "random_cv_rmse": static_row["random_cv_rmse"],
        "random_cv_r2": scorecard["reference_random_10fold"]["cv_r2"],
        "worst_region": static_row["worst_region"],
        "residual_morans_i_in_sample": card["residual_morans_i_in_sample"],
        "residual_morans_i_out_of_fold": static_row["oof_morans_i"],
        "calibration_slope": card["calibration_slope"],
        "shares": _shares(boot["country"]),
        "block_shares": _shares(boot["block"]),
        "p_geography_largest": {
            "country": boot["country"]["p_geography_largest"],
            "block": boot["block"]["p_geography_largest"],
        },
        "n_bootstrap": int(boot["country"]["n_draws"]),
        "per_capita_representation_shares": {
            k: boot_pc["country"]["groups"][k]["point"] for k in GROUPS
        },
    }

    investigations = []
    for stage, key, label, verdict in INVESTIGATIONS:
        row = _row(table, stage)
        investigations.append({
            "key": key, "label": label, "verdict": verdict,
            "primary_cv_r2": row["primary_cv_r2"],
            "primary_cv_rmse": row["primary_cv_rmse"],
            "m49_cv_rmse": row["m49_cv_rmse"],
            "oof_morans_i": row["oof_morans_i"],
            "delta_rmse": row["paired_delta_rmse"],
            "interval": row["paired_interval"],
            "in_sample_r2": (
                1.0 - row["allocation"]["residual"]
                if "residual" in row["allocation"] else None
            ),
            "allocation": (
                {k: row["allocation"][k] for k in row["allocation"] if k != "accounting"}
                if "residual" in row["allocation"] else None
            ),
        })

    spatial = {}
    for key, family in FAMILIES.items():
        station = final["stages"]["M3_station"]["SEM" if key == "spatial_error" else "SAR"]
        land = final["stages"]["M3_land_centroid_sensitivity"][
            "SEM" if key == "spatial_error" else "SAR"]
        land_row = _row(table, LAND_STAGES[key])
        spatial[key] = {
            "family": family,
            "station": {
                "delta_rmse": station["delta_rmse_vs_static"],
                "interval": station["interval"],
                "m49_rmse_change": station["m49_rmse_change"],
                "worst_region_rmse_change": station["worst_region_rmse_change"],
                "theta": station["theta_full_sample"],
                "qualifies": station["qualifies"],
                "accounting": station["accounting"],
                "oof_morans_i": _row(table, family)["oof_morans_i"],
                "primary_cv_r2": _row(table, family)["primary_cv_r2"],
            },
            "land_centroid": {
                "delta_rmse": land["delta_rmse_vs_static"],
                "interval": land["interval"],
                "m49_rmse_change": land["m49_rmse_change"],
                "worst_region_rmse_change": land["worst_region_rmse_change"],
                "theta": land["theta_full_sample"],
                "conditions_hold": land["qualifies"],
                "accounting": land["accounting"],
                "oof_morans_i": land_row["oof_morans_i"],
            },
        }

    def paired(block: dict, family: str) -> dict:
        p = block[f"paired_{family}_minus_S"]
        return {"delta_rmse": p["delta_rmse"], "interval": p["country_bootstrap_95_interval"]}

    def product_paired(block: dict, family: str) -> dict:
        p = block[family]["paired"]["vs_static"]
        return {"delta_rmse": p["delta_rmse"], "interval": p["country_bootstrap_95_interval"],
                "theta": block[family]["theta"]}

    robustness = [
        {"key": "berkeley_station", "product": "Berkeley Earth",
         "label": "Berkeley Earth, station-centroid graph, 500 km (primary)",
         "spatial_error": spatial["spatial_error"]["station"],
         "spatial_lag": spatial["spatial_lag"]["station"]},
        {"key": "berkeley_land", "product": "Berkeley Earth",
         "label": "Berkeley Earth, land-centroid graph, 500 km",
         "spatial_error": spatial["spatial_error"]["land_centroid"],
         "spatial_lag": spatial["spatial_lag"]["land_centroid"]},
        {"key": "berkeley_1000km", "product": "Berkeley Earth",
         "label": "Berkeley Earth, 1000 km territorial exclusion",
         "spatial_error": paired(sens["buffer_territory_1000km/primary_total_co2"], "M3-SEM(M0*)"),
         "spatial_lag": paired(sens["buffer_territory_1000km/primary_total_co2"], "M3-SAR(M0*)")},
        {"key": "berkeley_1500km", "product": "Berkeley Earth",
         "label": "Berkeley Earth, 1500 km centroid exclusion",
         "spatial_error": paired(sens["buffer_centroid_1500km/primary_total_co2"], "M3-SEM(M0*)"),
         "spatial_lag": paired(sens["buffer_centroid_1500km/primary_total_co2"], "M3-SAR(M0*)")},
        {"key": "era5_aligned", "product": "ERA5",
         "label": "ERA5, anomaly-aligned construction",
         "spatial_error": product_paired(products["aligned_era5/primary_total_co2"], "M3-SEM(M0*)"),
         "spatial_lag": product_paired(products["aligned_era5/primary_total_co2"], "M3-SAR(M0*)")},
        {"key": "era5_absolute", "product": "ERA5",
         "label": "ERA5, absolute-temperature construction",
         "spatial_error": product_paired(products["legacy_era5/primary_total_co2"], "M3-SEM(M0*)"),
         "spatial_lag": product_paired(products["legacy_era5/primary_total_co2"], "M3-SAR(M0*)")},
    ]
    # Strip nested accounting from the station/land rows copied into robustness.
    for row in robustness[:2]:
        for fam in ("spatial_error", "spatial_lag"):
            row[fam] = {k: row[fam][k] for k in ("delta_rmse", "interval", "theta")}

    product_static = {}
    for key, label in (("era5_aligned", "aligned_era5/primary_total_co2"),
                       ("era5_absolute", "legacy_era5/primary_total_co2")):
        s = products[label]["S"]
        product_static[key] = {
            "in_sample_r2": s["card"]["in_sample_r2"],
            "primary_cv_r2": s["card"]["cv_r2"],
            "primary_cv_rmse": s["card"]["cv_rmse"],
            "share_geography": s["card"]["share_geography"],
            "share_emissions": s["card"]["share_emissions"],
            "share_socioeconomic": s["card"]["share_socioeconomic"],
            "share_population": s["card"]["share_population"],
            "residual_agreement_with_berkeley": s["residual_agreement_with_berkeley"],
            "material_change": s["material_change_point"],
        }

    stopping = final["static_stopping_indicator"]
    return {
        "note": (
            "Values copied from the committed research records under "
            "research/model_v2/outputs; shares are descriptive allocations of "
            "explained cross-country variance, not causal contributions, and "
            "spatial accounting represents covariance, not mechanism."
        ),
        "static": static,
        "investigations": investigations,
        "spatial": spatial,
        "robustness": robustness,
        "product_static": product_static,
        "final": {
            "retained_static_specification": final["retained_static_specification"],
            "qualifying_spatial_extensions": final["qualifying_spatial_extensions"],
            "final_primary_predictive_model": final["final_primary_predictive_model"],
            "static_stopping_indicator_fires": stopping["fires"],
            "static_stopping_best_delta": stopping["best"],
            "static_stopping_threshold": stopping["threshold"],
            "material_change_to_conclusion": final["material_change_to_v1_conclusion"][
                "primary_total_co2"]["material_change_to_v1_conclusion"],
        },
        "responsibility_comparison": responsibility_comparison(national),
        "sources": {
            name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha256(path)}
            for name, path in SOURCES.items()
        },
    }


def build_residual_assets(out_dir: Path | None = None) -> dict[str, Path]:
    """Write both assets into ``out_dir`` (default ``app/data``); skip if records are absent."""
    out_dir = out_dir or APP_DATA_DIR
    missing = [str(p) for p in SOURCES.values() if not p.exists()]
    if missing:
        logger.warning("residual-structure records absent, skipping: %s", missing)
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    national = build_national_warming()
    national_dest = out_dir / NATIONAL_WARMING_ASSET
    national.to_parquet(national_dest, index=False, compression="zstd")
    summary_dest = out_dir / RESIDUAL_SUMMARY_ASSET
    summary_dest.write_text(
        json.dumps(round_floats(build_summary(national)), indent=2) + "\n", encoding="utf-8"
    )
    return {NATIONAL_WARMING_ASSET: national_dest, RESIDUAL_SUMMARY_ASSET: summary_dest}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    written = build_residual_assets()
    for name, path in written.items():
        print(f"{name}: {path.stat().st_size / 1024:,.1f} KB")
    if RESIDUAL_SUMMARY_ASSET in written:
        summary = json.loads(written[RESIDUAL_SUMMARY_ASSET].read_text(encoding="utf-8"))
        s = summary["static"]
        print(
            f"static: n={s['n']}, R^2 {s['in_sample_r2']:.3f}, geography share "
            f"{s['shares']['geography']['point']:.3f}, primary CV RMSE {s['primary_cv_rmse']:.4f}"
        )
        for row in summary["investigations"]:
            print(f"  {row['label']}: delta RMSE {row['delta_rmse']:+.5f} ({row['verdict']})")


if __name__ == "__main__":
    main()
