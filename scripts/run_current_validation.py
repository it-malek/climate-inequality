"""Portable fixed-spec current validation stages; no implicit source acquisition.

Use ``python -m scripts.run_current_validation --help``. Named stages are
exclusive; retries need a new --attempt or --stage-name. Historical comparisons
and verified bootstrap reuse are optional, explicitly selected inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import duckdb
import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely import contains_xy

from scripts.init_current_validation import initialize
from src import (
    data_io,
    forcings,
    physical_model,
    emissions,
    trends,
    explain,
    vulnerability,
)
from src import era5_validation as ev, era5_weighting as ew
from src import app_assets as aa, interpolate as ip, validation as va
from src import stability, vulnerability as vu, temporal_validation as tv
from src import bootstrap_population as bp
from src.area_weighting import GPW_NATID_LOOKUP_PATH
from src.cleaning import parse_window
from src.population import GPW_PATH
from src.prospective import prospective_execution
from src.trends import compute_anomalies
from research.model_v2.product_stability import monthly_anomalies

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "init",
    "inputs",
    "inputs-koppen",
    "inputs-ndgain",
    "preprocessing",
    "forcings",
    "physical",
    "era5",
    "core",
    "explain",
    "city-validation",
    "vulnerability",
    "rolling",
    "uncertainty",
    "comparisons",
    "stability",
)


class StagePaths:
    """Resolve explicitly accepted checkpoint stage names, preserving run identity."""

    def __init__(self, root, aliases=None):
        self.root = Path(root)
        self.aliases = dict(aliases or {})
        for source, destination in self.aliases.items():
            if (
                not source
                or Path(source).name != source
                or not destination
                or Path(destination).name != destination
                or destination in (".", "..")
            ):
                raise ValueError(
                    "dependency aliases must be simple confined stage names"
                )

    @property
    def name(self):
        return self.root.name

    def __truediv__(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("input checkpoint path escapes run")
        parts = relative.parts
        return self.root / self.aliases.get(parts[0], parts[0]) / Path(*parts[1:])


def raw_relative(path):
    """Route fixed raw filenames under the explicitly selected read-only raw root."""
    return RAW / Path(path).relative_to(ROOT / "data/raw")


def reference_inputs(names):
    return (
        [REFERENCE_ROOT / "data/processed" / name for name in names]
        if REFERENCE_ROOT is not None
        else []
    )


def app_reference_inputs(names):
    return (
        [APP_REFERENCE_ROOT / "app/data" / name for name in names]
        if APP_REFERENCE_ROOT is not None
        else []
    )


def do_forcings(d):
    frame, result = forcings.compute_forcings(
        forcings._read_gistemp(RAW / "forcings/gistemp_glb_ts_dsst.csv"),
        pd.read_csv(
            RAW / "climate-change-earth-surface-temperature-data/GlobalTemperatures.csv"
        ),
        pd.read_csv(RAW / "forcings/erf_best_aggregates.csv"),
        forcings._read_oni(RAW / "forcings/oni.ascii.txt"),
    )
    assert frame.year.is_unique and frame.year.max() == 2024
    data_io.write_typed_parquet(
        frame, d / "forcings.parquet", forcings.FORCINGS_SCHEMA, ("year",)
    )
    if REFERENCE_ROOT is not None:
        old = pd.read_parquet(REFERENCE_ROOT / "data/processed/forcings.parquet")
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            old.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    return {
        "integrity": asdict(result),
        "historical_preprocessing_equal": True if REFERENCE_ROOT is not None else None,
        "units": "temperature deg C, ERF W/m2, ONI dimensionless",
        "row_key": "year",
        "source_urls": [forcings.GISTEMP_URL, forcings.ERF_URL, forcings.ONI_URL],
    }


def do_physical(d):
    out = physical_model.build_physical_model(
        forcings_path=RUN / "forcings/forcings.parquet",
        summary_path=d / "physical_summary.json",
        trajectory_path=d / "physical_trajectory.parquet",
    )
    new = out["trajectory"]
    if APP_REFERENCE_ROOT is None:
        return {
            "new": physical_model.summary_payload(out["result"]),
            "historical_comparison": "not_requested",
        }
    old = pd.read_parquet(APP_REFERENCE_ROOT / "app/data/physical_trajectory.parquet")
    np.testing.assert_allclose(
        new.predicted_mean, old.predicted_mean, rtol=1e-8, atol=1e-8
    )
    train = new.year <= physical_model.TRAIN_END
    for c in ["lower95", "upper95"]:
        np.testing.assert_allclose(
            new.loc[train, c], old.loc[train, c], rtol=1e-8, atol=1e-8
        )
    comp = new.loc[
        ~train, ["year", "observed", "predicted_mean", "lower95", "upper95"]
    ].copy()
    for c in ["lower95", "upper95"]:
        comp["old_" + c] = old.loc[~train, c].to_numpy()
    comp["new_width"] = comp.upper95 - comp.lower95
    comp["old_width"] = comp.old_upper95 - comp.old_lower95
    comp.to_csv(d / "interval-comparison.csv", index=False)
    return {
        "new": physical_model.summary_payload(out["result"]),
        "old": json.loads(
            (APP_REFERENCE_ROOT / "app/data/physical_summary.json").read_text()
        ),
        "width_change_min": float((comp.new_width - comp.old_width).min()),
        "width_change_max": float((comp.new_width - comp.old_width).max()),
        "mean_unchanged": True,
        "in_sample_bands_unchanged": True,
    }


def do_preprocess(d):
    ing = data_io.load_city_temperatures(
        RAW
        / "climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv",
        d / "climate.duckdb",
    )
    city = trends.build_city_trends(d / "climate.duckdb", d / "city_trends.parquet")
    assert not city.duplicated(trends.CITY_KEYS).any()
    country = emissions.build_inequality_analysis(
        trends_path=d / "city_trends.parquet",
        co2_path=RAW / "owid/owid-co2-data.csv",
        continents_path=RAW / "owid/continents.csv",
        out_dir=d / "figures",
        table_path=d / "country_inequality.parquet",
        pop_grid_path=raw_relative(GPW_PATH),
        berkeley_grid_path=RAW / "berkeley_gridded/Complete_TAVG_LatLong1.nc",
        natid_lookup_path=raw_relative(GPW_NATID_LOOKUP_PATH),
    )["table"]
    import geopandas as gpd

    land = gpd.read_file(
        "zip://" + str(RAW / "natural_earth/ne_110m_land.zip")
    ).geometry.union_all()
    features = explain.build_city_features(
        trends_path=d / "city_trends.parquet",
        out_path=d / "city_features.parquet",
        land=land,
        etopo_path=RAW / "etopo/ETOPO_2022_v1_60s_N90W180_surface.nc",
        koppen_path=RUN / "inputs-koppen/koppen_present_0p5.nc",
    )
    compare = {}
    if REFERENCE_ROOT is not None:
        for name, keys in [
            ("city_trends", ["Country", "City", "Latitude", "Longitude"]),
            ("country_inequality", ["Country"]),
            ("city_features", ["Country", "City", "Latitude", "Longitude"]),
        ]:
            new = (
                pd.read_parquet(d / (name + ".parquet"))
                .sort_values(keys)
                .reset_index(drop=True)
            )
            old = (
                pd.read_parquet(REFERENCE_ROOT / "data/processed" / (name + ".parquet"))
                .sort_values(keys)
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(
                new, old, check_dtype=False, rtol=1e-10, atol=1e-10
            )
            compare[name] = {
                "equal_within": 1e-10,
                "rows": len(new),
                "keys": keys,
                "historical_sha256": sha(
                    REFERENCE_ROOT / "data/processed" / (name + ".parquet")
                ),
            }
    return {
        "ingestion": asdict(ing),
        "comparison": compare,
        "units": "temperature trends deg C/decade; emissions as schema; same period/coverage defaults",
        "country_rows": len(country),
        "feature_rows": len(features),
    }


def do_vulnerability(d):
    out = vulnerability.build_vulnerability(
        inequality_path=RUN / "preprocessing/country_inequality.parquet",
        income_path=RAW / "worldbank/world-bank-income-groups.csv",
        ndgain_path=NDGAIN_INPUT,
        co2_path=RAW / "owid/owid-co2-data.csv",
        summary_path=d / "vulnerability_summary.json",
        table_path=d / "vulnerability_strata.parquet",
    )
    return {
        "summary": out["summary"],
        "permutations": vulnerability.DEFAULT_N_PERMUTATIONS,
        "seed": vulnerability.DEFAULT_SEED,
    }


def do_explain(d):
    features = pd.read_parquet(RUN / "preprocessing/city_features.parquet")
    city = explain.compare_city_specs(features)
    country = explain.fit_country_model(
        explain.build_country_table(
            features,
            pd.read_parquet(RUN / "preprocessing/country_inequality.parquet"),
            explain.load_income_groups(RAW / "worldbank/world-bank-income-groups.csv"),
        )
    )
    explain.write_explain_summary(
        city,
        country,
        features,
        summary_path=d / "explain_summary.json",
        bundle_path=d / "explain_features.parquet",
    )
    return {"city_diagnostics": [explain.city_result_to_dict(r) for r in city]}


def era5(d):
    gpw = raw_relative(GPW_PATH)
    lookup = raw_relative(GPW_NATID_LOOKUP_PATH)
    result = ev.build_era5_validation(
        era5_grid_path=RAW / "era5/era5_t2m_monthly_1950_2013_1deg.nc",
        gpw_path=gpw,
        inequality_path=RUN / "preprocessing/country_inequality.parquet",
        co2_path=RAW / "owid/owid-co2-data.csv",
        berkeley_grid_path=RAW / "berkeley_gridded/Complete_TAVG_LatLong1.nc",
        lookup_path=lookup,
        trends_path=d / "era5_area_trends.parquet",
        summary_path=d / "era5_validation_summary.json",
    )
    with xr.open_dataset(RAW / "era5/era5_t2m_monthly_1950_2013_1deg.nc") as ds:
        anomalies = monthly_anomalies(ds.t2m).load()
    temp = d / "aligned-temporary.nc"
    anomalies.to_dataset(name="t2m").to_netcdf(temp)
    mask, lats, slopes = ew.era5_cell_slopes(temp, gpw, lookup_path=lookup)
    ew.reduce_era5_slopes(mask, lats, slopes).to_csv(
        d / "aligned_era5_trends.csv", index=False
    )
    np.savez_compressed(
        d / "aligned-cell-slopes.npz",
        iso3=np.where(mask == None, "", mask).astype("U3"),  # noqa: E711 - elementwise object-array comparison
        latitude=lats,
        slope=slopes,
    )
    temp.unlink()
    return {
        "absolute": result["summary"],
        "aligned": "own1951-1980calendar-monthbaseline; unchanged monthly Theil-Sen and GPW area reduction",
    }


def core(d):
    trends = pd.read_parquet(RUN / "preprocessing/city_trends.parquet")
    bs, be = parse_window(trends.baseline_window.iloc[0])
    start, end = parse_window(trends.analysis_window.iloc[0])
    con = duckdb.connect(str(RUN / "preprocessing/climate.duckdb"), read_only=True)
    try:
        anomalies = compute_anomalies(
            con,
            baseline_start=bs,
            baseline_end=be,
            start=start,
            end=end,
            min_coverage=0.9,
        )
    finally:
        con.close()
    tr, an = aa.attach_city_ids(trends, anomalies)
    tr["label"] = aa.disambiguate_labels(tr)
    tr["intercept"] = tr.city_id.map(aa.theil_sen_intercepts(tr, an))
    tr.to_parquet(d / "city_trends.parquet", index=False, compression="zstd")
    an[["city_id", "dt", "anomaly"]].sort_values(["city_id", "dt"]).assign(
        anomaly=lambda x: x.anomaly.astype("float32")
    ).to_parquet(d / "city_anomalies.parquet", index=False, compression="zstd")
    lon, lat = ip.build_grid(resolution=2.0)
    xx, yy = np.meshgrid(lon, lat)
    land = ip.load_land_geometry(RAW / "natural_earth/ne_110m_land.zip")
    keep = contains_xy(land, xx, yy)
    surface = np.full(xx.shape, np.nan)
    surface[keep] = ip.idw_interpolate(
        trends.Longitude.to_numpy(),
        trends.Latitude.to_numpy(),
        trends.slope_c_per_decade.to_numpy(),
        xx[keep],
        yy[keep],
        power=2,
        k=30,
    )
    aa._surface_to_long_form(lon, lat, surface).to_parquet(
        d / "trend_surface.parquet", index=False, compression="zstd"
    )
    ip.render_trend_surface(
        lon,
        lat,
        surface,
        title="Land warming trend, fixed IDW interpolation (degrees C/decade)",
    ).write_html(d / "trend_surface.html")
    inequality = pd.read_parquet(RUN / "preprocessing/country_inequality.parquet")
    ineq = aa.quantify_inequality(inequality)
    inequality.to_parquet(
        d / "country_inequality.parquet", index=False, compression="zstd"
    )
    stats = {
        "provenance": aa._owid_provenance(RAW / "owid/owid-co2-data.csv"),
        "trends": aa._sanity_stats(tr),
        "interpolation": {
            "winner": "idw",
            "surface_method_basis": "fixed for continuity before new validation scores",
            "k": 30,
            "resolution_deg": 2.0,
            "evaluation_status": "awaiting separately staged training-only evaluation",
        },
        "inequality": {
            "n_countries": ineq.n_countries,
            "n_continents": ineq.n_continents,
            "spearman_rho": ineq.spearman_rho,
            "spearman_p": ineq.spearman_p,
            "ols_pooled": asdict(ineq.ols_pooled),
            "ols_fe": asdict(ineq.ols_fe),
            "cutoff_year": int(end[:4]),
        },
    }
    dump(d / "stats.json", stats)
    return {
        "locations": len(tr),
        "anomalies": len(an),
        "surface_cells": int(keep.sum()),
        "surface": "fixed IDW k30/power2; no CV method selection",
    }


def comparisons(d):
    for func in [
        aa.build_decomposition_summaries,
        aa.build_coupling_summary_asset,
        aa.build_coupling_consumption_asset,
        aa.build_coupling_exposure_asset,
        aa.build_coupling_area_asset,
    ]:
        args = {
            "inequality_path": RUN / "preprocessing/country_inequality.parquet",
            "out_dir": d,
        }
        if func == aa.build_decomposition_summaries:
            args.update(
                city_features_path=RUN / "preprocessing/city_features.parquet",
                income_path=RAW / "worldbank/world-bank-income-groups.csv",
            )
        func(**args)
    return {
        "fresh_assets": sorted(
            p.name
            for p in d.glob("*")
            if p.suffix in [".json", ".parquet"] and p.name != "started.json"
        )
    }


def validation(d):
    result = va.run_validation(
        nc_path=RAW / "berkeley_gridded/Complete_TAVG_LatLong1.nc",
        bundle_dir=RUN / "core",
        out_path=d / "validation-detail.parquet",
        figures_dir=d / "figures",
        summary_path=d / "validation_summary.json",
        bundle_path=d / "validation.parquet",
        global_path=d / "validation_global.parquet",
    )
    return result["stats"]


def rolling(d):
    predictions, fits, summary = tv.rolling_origin(
        pd.read_parquet(RUN / "forcings/forcings.parquet")
    )
    assert (
        len(predictions) == 264 and predictions.groupby("horizon").size().eq(24).all()
    )
    predictions.to_csv(d / "predictions.csv", index=False)
    dump(d / "fits.json", fits)
    dump(
        d / "summary.json",
        {
            "evaluation": "conditional retrospective rolling-origin hindcast with observed future drivers",
            "origins": 24,
            "n_predictions": 264,
            "by_horizon": summary,
            "nonconverged_origins": [f["origin"] for f in fits if not f["converged"]],
        },
    )
    return {"n": 264, "nonconverged": [f["origin"] for f in fits if not f["converged"]]}


def uncertainty(d):
    frame = pd.read_parquet(RUN / "vulnerability/vulnerability_strata.parquet")
    frame["Country"] = frame.owid_country
    cols = [vu.RESPONSIBILITY_COL, *vu.EXPOSURE_LENSES.values()]
    axes = [vu.INCOME_RANK_COL, vu.VULN_SCORE_COL]
    common = frame.loc[np.isfinite(frame[cols + axes]).all(axis=1)]
    summaries = []
    draws = []
    for sample, data in [("pair_specific", frame), ("common_complete", common)]:
        for axis in axes:
            for col in cols:
                result, draw = tv.rank_uncertainty(data, axis, col, n=2000)
                result["sample"] = sample
                summaries.append(result)
                draws.append(draw.assign(sample=sample, x=axis, y=col))
    pd.concat(draws, ignore_index=True).to_parquet(
        d / "all-bootstrap-draws.parquet", index=False
    )
    dump(
        d / "summary.json",
        {
            "evaluations": summaries,
            "common_n": len(common),
            "interpretation": "Descriptive rank gradients; no equivalence margin or causal interpretation. Continent resampling has few heterogeneous blocks. ND-GAIN combines exposure, sensitivity and adaptive capacity. ND-GAIN uses the explicitly supplied prepared input; source provenance is separately recorded. No source version is silently substituted.",
        },
    )
    return {
        "pairs": len(summaries),
        "draws": len(pd.concat(draws)),
        "common_n": len(common),
    }


def convert(d):
    archive = RAW / "koppen/Beck_KG_V1.zip"
    with zipfile.ZipFile(archive) as z:
        raw = z.read(explain.KOPPEN_TIF_MEMBER)
    explain._geotiff_to_netcdf(raw, d / "koppen_present_0p5.nc", explain.KOPPEN_VAR)
    if KOPPEN_REFERENCE is not None:
        with (
            xr.open_dataset(d / "koppen_present_0p5.nc") as a,
            xr.open_dataset(KOPPEN_REFERENCE) as b,
        ):
            xr.testing.assert_identical(a, b)
    return {
        "archive_sha256": sha(archive),
        "member": explain.KOPPEN_TIF_MEMBER,
        "member_sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "grid_coordinates_categories_exact": True
        if KOPPEN_REFERENCE is not None
        else None,
    }


def retrieve(d):
    url = "https://gain.nd.edu/assets/647440/ndgain_countryindex_2026.zip"
    try:
        response = requests.get(url, timeout=60)
    except requests.RequestException as error:
        dump(
            d / "retrieval.json",
            {
                "status": "blocked",
                "url": url,
                "reason": str(error),
                "required_file": "ndgain_countryindex_2026.zip",
                "official_page": "https://gain.nd.edu/our-work/country-index/download-data/",
                "destination": str(d / "ndgain_countryindex_2026.zip"),
            },
        )
        return {"status": "blocked"}
    dump(
        d / "http.json",
        {
            "requested_url": url,
            "final_url": response.url,
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
        },
    )
    if response.status_code != 200 or not zipfile.is_zipfile(
        io.BytesIO(response.content)
    ):
        dump(
            d / "retrieval.json",
            {
                "status": "blocked",
                "required_file": "ndgain_countryindex_2026.zip",
                "required_release": "2026 source archive previously used for latest2023extract",
                "official_page": "https://gain.nd.edu/our-work/country-index/download-data/",
                "direct_url": url,
                "destination": str(d / "ndgain_countryindex_2026.zip"),
                "http_status": response.status_code,
            },
        )
        return {"status": "blocked", "http_status": response.status_code}
    (d / "ndgain_countryindex_2026.zip").write_bytes(response.content)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        frames = {
            metric: pd.read_csv(z.open("resources/" + metric + "/" + metric + ".csv"))
            for metric in ["vulnerability", "readiness", "gain"]
        }
    slim = vulnerability.derive_ndgain_latest(frames)
    slim.to_csv(d / "ndgain_latest.csv", index=False)
    equivalent = None
    if NDGAIN_INPUT.is_file():
        old = pd.read_csv(NDGAIN_INPUT)
        equivalent = False
        try:
            pd.testing.assert_frame_equal(slim, old, check_dtype=False, rtol=0, atol=0)
            equivalent = True
        except AssertionError:
            pass
    dump(
        d / "retrieval.json",
        {
            "status": "reconstructed",
            "sha256": sha(d / "ndgain_countryindex_2026.zip"),
            "old_extract_exact_numeric_equal": equivalent,
            "latest_year": int(slim.ndgain_year.max()),
            "note": "Do not silently adopt changed release if unequal.",
        },
    )
    return {
        "status": "reconstructed",
        "old_extract_exact_numeric_equal": equivalent,
        "rows": len(slim),
    }


def previous_stability_inputs():
    """Enumerate only selected prior evidence; absent prior means a fresh run."""
    if OLD is None:
        return []
    files = list((OLD / "stability-full").glob("*"))
    files.extend(
        OLD / "execution_sources/final-source" / rel
        for rel in ["src/stability.py", "src/decomposition.py", "src/feature_schema.py"]
    )
    return sorted(path for path in files if path.is_file())


def verified_reuse_sources():
    if OLD is None:
        return False, {"reason": "no_previous_run_selected"}, {}
    completed = OLD / "stability-full/completed.json"
    if not completed.is_file():
        return False, {"reason": "no_completed_prior_stability_receipt"}, {}
    record = json.loads(completed.read_text())
    details = record.get("details", {})
    if (
        record.get("status") != "completed"
        or details.get("n_boot") != 2000
        or details.get("seeds") != {"country": 0, "block": 1}
    ):
        return False, {"reason": "prior_completion_or_design_metadata_incompatible"}, {}
    manifest = record.get("outputs_sha256")
    if not manifest:
        return False, {"reason": "prior_receipt_missing_output_hashes"}, {}
    for relative, digest in manifest.items():
        path = OLD / relative
        if (
            not path.resolve().is_relative_to(OLD)
            or path.is_symlink()
            or sha(path) != digest
        ):
            raise ValueError("selected prior stability checkpoint drift: " + str(path))
    checked = {}
    for relative in [
        "src/stability.py",
        "src/decomposition.py",
        "src/feature_schema.py",
    ]:
        old = OLD / "execution_sources/final-source" / relative
        if not old.is_file():
            return False, {"reason": "prior_captured_source_missing"}, checked
        checked[relative] = sha(old)
        if sha(ROOT / relative) != checked[relative]:
            return False, {"reason": "prior_estimator_source_not_exact"}, checked
    for name in [stability.PRIMARY_OUTCOME, stability.STATION_WEIGHTED]:
        for suffix in [
            "input-design.parquet",
            "country-draws.jsonl",
            "block-draws.jsonl",
        ]:
            if not (OLD / "stability-full" / f"{name}-{suffix}").is_file():
                return False, {"reason": "prior_design_or_draws_missing"}, checked
    return True, {"reason": "completed_hashes_and_estimator_sources_exact"}, checked


def stability_check(d):
    features = pd.read_parquet(RUN / "preprocessing/city_features.parquet")
    inequality = pd.read_parquet(RUN / "preprocessing/country_inequality.parquet")
    income = explain.load_income_groups(RAW / "worldbank/world-bank-income-groups.csv")
    designs = stability.outcome_designs(inequality, features, income)
    centroids = stability.country_centroids(features)
    populations = {
        name: bp.population_audit(design) for name, design in designs.items()
    }
    if BOOTSTRAP_POPULATION == "complete-case":
        designs = {name: bp.complete_design(design) for name, design in designs.items()}
    can_reuse, prior_status, checked = verified_reuse_sources()
    original = stability._bootstrap_shares
    current = None
    records = []

    def reuse(design, schema, status, *, n_boot, rng, block):
        assert n_boot == 2000
        kind = "block" if block else "country"
        path = OLD / "stability-full" / f"{current}-{kind}-draws.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if (
            len(rows) != 2000
            or [row["draw"] for row in rows] != list(range(2000))
            or any(row["status"] not in ("success", "failed") for row in rows)
        ):
            raise ValueError("invalid selected prior draw inventory")
        records.append(
            {
                "outcome": current,
                "kind": kind,
                "sha256": sha(path),
                "draws": len(rows),
                "source": str(path),
                "action": "reuse_verified_completed_draws",
            }
        )
        return [row["shares"] for row in rows if row["status"] == "success"], sum(
            row["status"] != "success" for row in rows
        )

    def fresh(design, schema, status, *, n_boot, rng, block):
        assert n_boot == 2000
        kind = "block" if block else "country"
        shares, failures = original(
            design, schema, status, n_boot=n_boot, rng=rng, block=block
        )
        dump(
            d / f"{current}-{kind}-successful-shares.json",
            {"n_attempted": n_boot, "n_failed": failures, "shares": shares},
        )
        records.append(
            {
                "outcome": current,
                "kind": kind,
                "draws": n_boot,
                "n_failed": failures,
                "action": "fresh_2000_draws",
            }
        )
        return shares, failures

    results = {}
    equivalence = {}
    try:
        for name, design in designs.items():
            exact = False
            reference = (
                OLD / "stability-full" / f"{name}-input-design.parquet"
                if can_reuse
                else None
            )
            if reference is not None:
                old = pd.read_parquet(reference)
                try:
                    pd.testing.assert_frame_equal(design, old, check_exact=True)
                    exact = True
                except AssertionError:
                    pass
            equivalence[name] = {
                "rows": len(design),
                "exact_values_missingness_order_schema": exact,
                "reference_sha256": sha(reference) if reference is not None else None,
            }
            design.to_parquet(d / f"{name}-input-design.parquet", index=False)
            current = name
            stability._bootstrap_shares = reuse if exact else fresh
            results[name] = stability.build_stability_summary(
                design,
                centroids,
                n_boot=2000,
                seed=0,
                outcome_definition=name,
                include_influence=name == stability.PRIMARY_OUTCOME,
            )
            equivalence[name]["point_complete_n"] = results[name].n_countries
            equivalence[name]["bootstrap_action"] = (
                "reuse_verified_completed_draws" if exact else "fresh_2000_per_arm"
            )
    finally:
        stability._bootstrap_shares = original
    summary = stability.summary_payload(
        results[stability.PRIMARY_OUTCOME],
        sensitivity={stability.STATION_WEIGHTED: results[stability.STATION_WEIGHTED]},
    )
    for name, population in populations.items():
        target = (
            summary
            if name == stability.PRIMARY_OUTCOME
            else summary["sensitivity"][name]
        )
        target["bootstrap_population"] = {
            "policy_id": bp.COMPLETE_CASE_POLICY
            if BOOTSTRAP_POPULATION == "complete-case"
            else bp.LEGACY_POLICY,
            "conditional_observed_complete_countries": BOOTSTRAP_POPULATION
            == "complete-case",
            "sampling_n": len(designs[name]),
            "legacy_sampling_n": population["sampling_rows_legacy"],
            "country_seed": 0,
            "block_seed": 1,
            "n_attempted_each": 2000,
            "population_audit": population,
            "limitations": "Descriptive perturbation stability; no missingness-mechanism correction, causal claim or extension to excluded countries. Six heterogeneous continents limit block-bootstrap interpretation.",
        }
    dump(d / "stability_summary.json", summary)
    dump(
        d / "reuse-verification.json",
        {
            "designs": equivalence,
            "bootstrap_population_policy_id": bp.COMPLETE_CASE_POLICY
            if BOOTSTRAP_POPULATION == "complete-case"
            else bp.LEGACY_POLICY,
            "input_population_audits": populations,
            "prior_status": prior_status,
            "code_exact": checked,
            "draw_records": records,
            "estimator": "resample full design, then drop incomplete within fit; fixed schema, n_boot2000 per outcome/arm, country seed0 and block seed1",
            "fresh": "point shares, influence, Moran residuals and summaries recalculated from fresh inputs",
        },
    )
    return equivalence


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=STAGES)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--raw-root", type=Path, required=True)
    result.add_argument("--previous-run", type=Path)
    result.add_argument(
        "--bootstrap-population",
        choices=["legacy", "complete-case"],
        default="legacy",
        help="Explicit V1 stability sampling population; complete-case conditions on the fixed observed complete countries before resampling",
    )
    result.add_argument(
        "--reference-root",
        type=Path,
        help="Optional project root containing historical data/processed comparisons",
    )
    result.add_argument(
        "--app-reference-root",
        type=Path,
        help="Optional project root containing historical app/data physical comparisons",
    )
    result.add_argument(
        "--koppen-reference",
        type=Path,
        help="Optional prepared NetCDF for exact conversion comparison",
    )
    result.add_argument(
        "--ndgain-input",
        type=Path,
        help="Explicit prepared extract; default is RAW/ndgain/ndgain_latest.csv",
    )
    result.add_argument(
        "--retrieve-ndgain",
        action="store_true",
        help="Explicit opt-in to the exact recorded 2026 archive URL; never automatically adopts its extract",
    )
    result.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="1 uses canonical stage name; larger values create _attemptN",
    )
    result.add_argument("--stage-name", help="Explicit new immutable output stage name")
    result.add_argument(
        "--dependency",
        action="append",
        default=[],
        metavar="CANONICAL=ACCEPTED_STAGE",
        help="Select a verified retry checkpoint for downstream reads/dependencies",
    )
    return result


def configure(args):
    global \
        RUN, \
        RAW, \
        OLD, \
        REFERENCE_ROOT, \
        APP_REFERENCE_ROOT, \
        KOPPEN_REFERENCE, \
        NDGAIN_INPUT, \
        sha, \
        dump, \
        stage, \
        ALIASES, \
        BOOTSTRAP_POPULATION
    actual_run = args.run_root.resolve(strict=True)
    RAW = args.raw_root.resolve(strict=True)
    spec = importlib.util.spec_from_file_location(
        "current_validation_runlib", actual_run / "control/runlib.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load selected private run adapter")
    runlib = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runlib
    spec.loader.exec_module(runlib)
    if (
        Path(runlib.ROOT).resolve() != ROOT
        or Path(runlib.RUN).resolve() != actual_run
        or Path(runlib.RAW).resolve() != RAW
    ):
        raise ValueError(
            "explicit repository/run/raw roots disagree with initialized adapter"
        )
    from src.validation_receipts import ValidationReceipts

    ValidationReceipts(ROOT, actual_run, RAW, runlib.SPEC_HASH)
    ALIASES = {}
    BOOTSTRAP_POPULATION = args.bootstrap_population
    for value in args.dependency:
        if value.count("=") != 1:
            raise ValueError("dependency must have CANONICAL=ACCEPTED_STAGE form")
        source, destination = value.split("=")
        if source in ALIASES:
            raise ValueError("duplicate dependency alias")
        ALIASES[source] = destination
    RUN = StagePaths(actual_run, ALIASES)
    previous = (
        args.previous_run
        if args.previous_run is not None
        else getattr(runlib, "OLD", None)
    )
    OLD = Path(previous).resolve(strict=True) if previous is not None else None
    REFERENCE_ROOT = (
        args.reference_root.resolve(strict=True) if args.reference_root else None
    )
    APP_REFERENCE_ROOT = (
        args.app_reference_root.resolve(strict=True)
        if args.app_reference_root
        else None
    )
    KOPPEN_REFERENCE = (
        args.koppen_reference.resolve(strict=True) if args.koppen_reference else None
    )
    NDGAIN_INPUT = (
        args.ndgain_input.resolve(strict=True)
        if args.ndgain_input
        else RAW / "ndgain/ndgain_latest.csv"
    )
    sha, dump, stage = runlib.sha, runlib.dump, runlib.stage
    return runlib


def dispatch(args):
    if args.stage == "init":
        if args.attempt != 1 or args.stage_name or args.dependency:
            raise ValueError(
                "init creates an exclusively named run; stage retry options do not apply"
            )
        return initialize(args.run_root, args.raw_root, previous_run=args.previous_run)
    if args.attempt < 1:
        raise ValueError("attempt must be positive")
    if args.stage == "inputs-ndgain" and not args.retrieve_ndgain:
        raise ValueError("ND-GAIN retrieval requires explicit --retrieve-ndgain")
    configure(args)
    name = {"uncertainty": "vulnerability-uncertainty", "inputs": "inputs-koppen"}.get(
        args.stage, args.stage
    )
    output_name = args.stage_name or (
        name if args.attempt == 1 else f"{name}_attempt{args.attempt}"
    )
    if Path(output_name).name != output_name or output_name in (".", ".."):
        raise ValueError("stage output must be a simple confined name")
    pre = RUN / "preprocessing"

    def execute(inputs, action, dependencies=()):
        return stage(
            output_name,
            inputs,
            action,
            runner=__file__,
            dependencies=[ALIASES.get(dep, dep) for dep in dependencies],
        )

    with prospective_execution(run_id=RUN.name):
        if args.stage in ("inputs", "inputs-koppen"):
            result = execute(
                [RAW / "koppen/Beck_KG_V1.zip"]
                + ([KOPPEN_REFERENCE] if KOPPEN_REFERENCE is not None else []),
                convert,
            )
            if args.stage == "inputs" and args.retrieve_ndgain:
                stage(
                    "inputs-ndgain"
                    if args.attempt == 1
                    else f"inputs-ndgain_attempt{args.attempt}",
                    [NDGAIN_INPUT],
                    retrieve,
                    runner=__file__,
                )
            return result
        if args.stage == "inputs-ndgain":
            return execute(([NDGAIN_INPUT] if NDGAIN_INPUT.is_file() else []), retrieve)
        if args.stage == "forcings":
            return execute(
                [
                    RAW / "forcings" / f
                    for f in [
                        "gistemp_glb_ts_dsst.csv",
                        "erf_best_aggregates.csv",
                        "oni.ascii.txt",
                    ]
                ]
                + [
                    RAW
                    / "climate-change-earth-surface-temperature-data/GlobalTemperatures.csv"
                ]
                + reference_inputs(["forcings.parquet"]),
                do_forcings,
            )
        if args.stage == "preprocessing":
            return execute(
                [
                    RAW
                    / "climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv",
                    RAW / "owid/owid-co2-data.csv",
                    RAW / "owid/continents.csv",
                    RAW / "berkeley_gridded/Complete_TAVG_LatLong1.nc",
                    RAW / "etopo/ETOPO_2022_v1_60s_N90W180_surface.nc",
                    RUN / "inputs-koppen/koppen_present_0p5.nc",
                    RAW / "natural_earth/ne_110m_land.zip",
                    raw_relative(GPW_PATH),
                    raw_relative(GPW_NATID_LOOKUP_PATH),
                ]
                + reference_inputs(
                    [
                        name + ".parquet"
                        for name in [
                            "city_trends",
                            "country_inequality",
                            "city_features",
                        ]
                    ]
                ),
                do_preprocess,
                ["inputs-koppen"],
            )
        if args.stage == "physical":
            return execute(
                [RUN / "forcings/forcings.parquet"]
                + app_reference_inputs(
                    ["physical_summary.json", "physical_trajectory.parquet"]
                ),
                do_physical,
                ["forcings"],
            )
        if args.stage == "era5":
            return execute(
                [
                    RAW / "era5/era5_t2m_monthly_1950_2013_1deg.nc",
                    RAW / "berkeley_gridded/Complete_TAVG_LatLong1.nc",
                    raw_relative(GPW_PATH),
                    raw_relative(GPW_NATID_LOOKUP_PATH),
                    RAW / "owid/owid-co2-data.csv",
                    pre / "country_inequality.parquet",
                ],
                era5,
                ["preprocessing"],
            )
        if args.stage == "core":
            return execute(
                [
                    pre / p
                    for p in [
                        "climate.duckdb",
                        "city_trends.parquet",
                        "country_inequality.parquet",
                    ]
                ]
                + [
                    RAW / "natural_earth/ne_110m_land.zip",
                    RAW / "owid/owid-co2-data.csv",
                ],
                core,
                ["preprocessing"],
            )
        if args.stage == "city-validation":
            return execute(
                [
                    RUN / "core/city_trends.parquet",
                    RUN / "core/city_anomalies.parquet",
                    RAW / "berkeley_gridded/Complete_TAVG_LatLong1.nc",
                ],
                validation,
                ["core"],
            )
        if args.stage in ("comparisons", "explain", "stability"):
            action = {
                "comparisons": comparisons,
                "explain": do_explain,
                "stability": stability_check,
            }[args.stage]
            return execute(
                [
                    pre / "city_features.parquet",
                    pre / "country_inequality.parquet",
                    RAW / "worldbank/world-bank-income-groups.csv",
                ]
                + (previous_stability_inputs() if args.stage == "stability" else []),
                action,
                ["preprocessing"],
            )
        if args.stage == "vulnerability":
            return execute(
                [
                    pre / "country_inequality.parquet",
                    RAW / "worldbank/world-bank-income-groups.csv",
                    NDGAIN_INPUT,
                    RAW / "owid/owid-co2-data.csv",
                ],
                do_vulnerability,
                ["preprocessing"],
            )
        if args.stage == "rolling":
            return execute([RUN / "forcings/forcings.parquet"], rolling, ["forcings"])
        if args.stage == "uncertainty":
            return execute(
                [RUN / "vulnerability/vulnerability_strata.parquet"],
                uncertainty,
                ["vulnerability"],
            )
    raise ValueError(args.stage)


if __name__ == "__main__":
    dispatch(parser().parse_args())
