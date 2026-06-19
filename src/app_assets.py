"""Build the committed dashboard asset bundle in ``app/data/`` (Phase 5).

The Streamlit app must run on Streamlit Community Cloud, which sees only
the git repository: the 112 MB DuckDB database and the ~500 MB raw Kaggle
download cannot ship there. This module distills the phase 1-4 outputs
into a small bundle (parquet + JSON, ~10 MB, dominated by the per-city
anomaly series) that the app reads identically on a laptop and in the
cloud. Committing ``app/data/`` is a deliberate, documented exception to
the project's "never commit data files" rule, which continues to apply to
``data/`` and ``outputs/``.

Bundle contents:

- ``city_trends.parquet``  -- phase 2 trends + ``city_id``, a per-country
  unique ``label``, and the Theil-Sen ``intercept`` (so the dashboard can
  draw the exact fitted line without scipy at runtime).
- ``city_anomalies.parquet`` -- monthly anomalies per city-location
  (float32, zstd), keyed by ``city_id``.
- ``trend_surface.parquet`` -- the phase 3 winning interpolation on the
  land-masked grid, long form (lat, lon, value; NaN over ocean).
- ``country_inequality.parquet`` -- the phase 4 country table, as is.
- ``stats.json`` -- headline numbers the dashboard displays: phase 2
  sanity stats, phase 3 cross-validation, phase 4 inequality fits.

Rebuilding the anomalies re-runs the exact phase 2 computation, so the
builder cross-checks the refit Theil-Sen slopes against the stored ones
and refuses to publish a bundle from a stale or inconsistent pipeline.

Run after any pipeline change that alters published numbers:

    uv run python -m src.app_assets
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from shapely import Geometry

from src.cleaning import parse_window, to_decimal_decades
from src.data_io import DEFAULT_DB_PATH, OUTPUTS_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.emissions import DEFAULT_INEQUALITY_PATH, quantify_inequality
from src.interpolate import (
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_K_NEIGHBORS,
    build_interpolated_surface,
)
from src.trends import (
    CITY_KEYS,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_TRENDS_PATH,
    compute_anomalies,
)

logger = logging.getLogger(__name__)

# Must equal app.loaders.APP_DATA_DIR; the app cannot import this module
# (heavy pipeline dependencies), so the path is defined on both sides and
# tests assert they agree.
APP_DATA_DIR = PROJECT_ROOT / "app" / "data"

ARCTIC_LATITUDE = 60.0
# Refit slopes must match the stored parquet to float-noise levels; real
# drift (stale parquet vs database) shows up orders of magnitude larger.
SLOPE_CONSISTENCY_ATOL = 1e-8

TRENDS_ASSET = "city_trends.parquet"
ANOMALIES_ASSET = "city_anomalies.parquet"
SURFACE_ASSET = "trend_surface.parquet"
INEQUALITY_ASSET = "country_inequality.parquet"
STATS_ASSET = "stats.json"

# Headline decomposition-page summaries. Built by build_decomposition_summaries
# (not build_app_assets) because the LMG/Shapley decomposition needs the Phase-7
# city features and the income table; main() runs both so a single
# `python -m src.app_assets` regenerates the entire committed bundle.
INEQUALITY_SUMMARY_ASSET = "inequality_summary.json"
DECOMPOSITION_SUMMARY_ASSET = "decomposition_summary.json"

# Phase 6/7 optional bundle assets (present only when the heavy pipeline has run).
VALIDATION_ASSET = "validation.parquet"
VALIDATION_GLOBAL_ASSET = "validation_global.parquet"
EXPLAIN_FEATURES_ASSET = "explain_features.parquet"

# Source paths for the slim summary artifacts written by src.validation /
# src.explain. Defined here (not imported from those modules) to avoid a
# circular import: validation.py imports app_assets, so app_assets must not
# import validation/explain at module scope. Tests that assert agreement
# between these constants and the source-of-truth constants in validation.py /
# explain.py live in tests/test_app_assets.py.
_VALIDATION_SUMMARY_PATH = PROCESSED_DIR / "validation_summary.json"
_VALIDATION_BUNDLE_PATH = PROCESSED_DIR / "validation_bundle.parquet"
_VALIDATION_GLOBAL_PATH = PROCESSED_DIR / "validation_global.parquet"
_EXPLAIN_SUMMARY_PATH = PROCESSED_DIR / "explain_summary.json"
_EXPLAIN_BUNDLE_PATH = PROCESSED_DIR / "explain_features.parquet"


def disambiguate_labels(trends: pd.DataFrame) -> pd.Series:
    """City display labels, unique within each country.

    18 (City, Country) pairs sit at 2-3 grid coordinates each (README
    quirks); those get their coordinates appended so the dashboard's city
    picker can tell them apart.

    Args:
        trends: One row per city-location with City/Country/Latitude/
            Longitude columns.

    Returns:
        Label series aligned to `trends`.
    """
    duplicated = trends.duplicated(subset=["Country", "City"], keep=False)
    coords = (
        " ("
        + trends["Latitude"].map("{:.2f}".format)
        + "°, "
        + trends["Longitude"].map("{:.2f}".format)
        + "°)"
    )
    return trends["City"].where(~duplicated, trends["City"] + coords)


def attach_city_ids(
    trends: pd.DataFrame, anomalies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign a stable integer ``city_id`` and key the anomalies by it.

    The id is the row number after sorting by (Country, City, Latitude,
    Longitude) -- the deterministic order ``city_trends.parquet`` is
    written in.

    Args:
        trends: Phase 2 output, one row per city-location.
        anomalies: Output of :func:`src.trends.compute_anomalies`.

    Returns:
        (trends with city_id, anomalies with city_id), both copies.

    Raises:
        RuntimeError: if the anomaly locations and the trends rows do not
            match one-to-one with identical observation counts -- the
            parquet and the database disagree (stale pipeline output).
    """
    trends_out = (
        trends.sort_values(["Country", "City", "Latitude", "Longitude"])
        .reset_index(drop=True)
        .assign(city_id=lambda df: df.index.astype("int32"))
    )
    merged = anomalies.merge(
        trends_out[[*CITY_KEYS, "city_id"]], on=CITY_KEYS, how="left"
    )
    unmatched = merged["city_id"].isna()
    if unmatched.any():
        locations = (
            merged.loc[unmatched, CITY_KEYS].drop_duplicates().head(5).to_dict("records")
        )
        raise RuntimeError(
            f"{int(unmatched.sum())} anomaly rows have no trends row, e.g. "
            f"{locations}; rebuild city_trends.parquet (python -m src.trends)"
        )
    merged["city_id"] = merged["city_id"].astype("int32")
    counts = merged.groupby("city_id").size()
    expected = trends_out.set_index("city_id")["n_obs"]
    mismatched = counts.reindex(expected.index).fillna(0).astype(int) != expected
    if mismatched.any():
        raise RuntimeError(
            f"observation counts differ from trends n_obs for "
            f"{int(mismatched.sum())} locations; rebuild city_trends.parquet"
        )
    return trends_out, merged


def theil_sen_intercepts(
    trends: pd.DataFrame,
    anomalies: pd.DataFrame,
    atol: float = SLOPE_CONSISTENCY_ATOL,
) -> pd.Series:
    """Refit Theil-Sen per city to recover intercepts, verifying slopes.

    Phase 2 stored only slopes; the dashboard needs intercepts to draw the
    fitted line. Refitting on the identical anomalies must reproduce the
    stored slopes to within float noise -- a build-time integrity check
    that the bundle is being built from a consistent pipeline state.

    Args:
        trends: Trends frame with ``city_id`` and ``slope_c_per_decade``.
        anomalies: Anomalies frame with ``city_id``, ``dt``, ``anomaly``.
        atol: Allowed absolute slope difference.

    Returns:
        Intercepts (on the :func:`src.cleaning.to_decimal_decades` axis),
        indexed by ``city_id``.

    Raises:
        RuntimeError: if any refit slope deviates beyond `atol`.
    """
    work = anomalies.assign(_decades=to_decimal_decades(anomalies["dt"]))
    intercepts: dict[int, float] = {}
    slopes: dict[int, float] = {}
    for city_id, grp in work.groupby("city_id", sort=True):
        fit = stats.theilslopes(grp["anomaly"].to_numpy(), grp["_decades"].to_numpy())
        intercepts[city_id] = float(fit.intercept)
        slopes[city_id] = float(fit.slope)

    refit = pd.Series(slopes).reindex(trends["city_id"]).to_numpy()
    stored = trends["slope_c_per_decade"].to_numpy()
    deviation = np.abs(refit - stored)
    if np.any(deviation > atol):
        worst = int(np.argmax(deviation))
        raise RuntimeError(
            f"refit Theil-Sen slope deviates from stored slope by "
            f"{deviation.max():.2e} (> {atol:.0e}) at city_id "
            f"{trends['city_id'].iloc[worst]} "
            f"({trends['City'].iloc[worst]!r}); city_trends.parquet is "
            "stale -- rebuild it (python -m src.trends)"
        )
    return pd.Series(intercepts, name="intercept")


def _surface_to_long_form(
    grid_lon: np.ndarray, grid_lat: np.ndarray, surface: np.ndarray
) -> pd.DataFrame:
    """Flatten a (lat, lon) surface grid to tidy float32 rows (NaN kept)."""
    lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)
    return pd.DataFrame(
        {
            "lat": lat_grid.ravel().astype("float32"),
            "lon": lon_grid.ravel().astype("float32"),
            "value": np.asarray(surface, dtype=float).ravel().astype("float32"),
        }
    )


def _sanity_stats(trends: pd.DataFrame) -> dict:
    """Phase 2 headline numbers (the README validation checkpoints)."""
    global_mean = float(trends["slope_c_per_decade"].mean())
    arctic = trends.loc[trends["Latitude"] > ARCTIC_LATITUDE, "slope_c_per_decade"]
    return {
        "n_locations": int(len(trends)),
        "global_mean_c_per_decade": global_mean,
        "n_arctic": int(len(arctic)),
        "arctic_mean_c_per_decade": float(arctic.mean()) if len(arctic) else None,
        "arctic_ratio": float(arctic.mean() / global_mean) if len(arctic) else None,
        "baseline_window": str(trends["baseline_window"].iloc[0]),
        "analysis_window": str(trends["analysis_window"].iloc[0]),
    }


def _copy_findings_parquet(src: Path, dest: Path, required: tuple[str, ...]) -> None:
    """Copy a slim findings parquet into the bundle, failing loud on schema drift.

    The bundle build is the integrity checkpoint (matching this module's
    fail-loud stance elsewhere): a source parquet whose columns have drifted
    from its schema must raise here, not get silently copied and break the
    app's loader at runtime.
    """
    frame = pd.read_parquet(src)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{src} is missing required column(s) {missing}; "
            "re-run the phase that writes it to rebuild the bundle"
        )
    frame.to_parquet(dest, index=False, compression="zstd")


def _merge_optional_findings(
    stats_payload: dict,
    out_dir: Path,
    paths: dict,
    validation_summary_path: Path,
    validation_bundle_path: Path,
    validation_global_path: Path,
    explain_summary_path: Path,
    explain_bundle_path: Path,
) -> None:
    """Merge Phase 6/7 summary artifacts into the bundle, if present.

    Reads the small JSON + parquet artifacts written by
    ``python -m src.validation`` and ``python -m src.explain``, and:
    - adds ``stats_payload["validation"]`` / ``["explain"]`` keys;
    - copies the slim parquets into `out_dir`.

    If both artifacts for a phase are absent, logs a warning and skips
    that phase (the resulting bundle is a graceful 3-page bundle). If only
    one of the pair exists, raises RuntimeError to flag a stale pipeline.
    """
    # Schemas are the source of truth for the slim parquet columns; lazy-import
    # so the (circular) validation/explain modules stay out of app_assets's
    # module scope, matching the re-declared path constants above.
    from src.explain import EXPLAIN_BUNDLE_SCHEMA
    from src.validation import VALIDATION_BUNDLE_SCHEMA, VALIDATION_GLOBAL_SCHEMA

    for (
        phase, summary_path, bundle_path, global_path,
        asset, global_asset, key, bundle_cols, global_cols,
    ) in (
        (
            "validation",
            validation_summary_path,
            validation_bundle_path,
            validation_global_path,
            VALIDATION_ASSET,
            VALIDATION_GLOBAL_ASSET,
            "validation",
            tuple(VALIDATION_BUNDLE_SCHEMA),
            tuple(VALIDATION_GLOBAL_SCHEMA),
        ),
        (
            "explain",
            explain_summary_path,
            explain_bundle_path,
            None,
            EXPLAIN_FEATURES_ASSET,
            None,
            "explain",
            tuple(EXPLAIN_BUNDLE_SCHEMA),
            None,
        ),
    ):
        present = [p for p in (summary_path, bundle_path) if p is not None and p.exists()]
        expected = [p for p in (summary_path, bundle_path) if p is not None]
        if len(present) == 0:
            logger.warning(
                "%s summaries not found; bundle will omit the Phase %s pages -- "
                "run python -m src.%s first",
                phase, "6" if phase == "validation" else "7", phase,
            )
            continue
        if len(present) < len(expected):
            missing = [str(p) for p in expected if not p.exists()]
            raise RuntimeError(
                f"partial {phase} findings: some summary artifacts exist but "
                f"{missing} do not; re-run python -m src.{phase} to rebuild all"
            )

        stats_payload[key] = json.loads(summary_path.read_text(encoding="utf-8"))
        bundle_dest = out_dir / asset
        _copy_findings_parquet(bundle_path, bundle_dest, bundle_cols)
        paths[asset] = bundle_dest

        if global_path is not None and global_asset is not None:
            if not global_path.exists():
                raise RuntimeError(
                    f"partial validation findings: {global_path} missing; "
                    "re-run python -m src.validation to rebuild all"
                )
            global_dest = out_dir / global_asset
            _copy_findings_parquet(global_path, global_dest, global_cols)
            paths[global_asset] = global_dest

        logger.info("merged %s findings into bundle", phase)


def build_app_assets(
    db_path: Path = DEFAULT_DB_PATH,
    trends_path: Path = DEFAULT_TRENDS_PATH,
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_dir: Path = APP_DATA_DIR,
    surface_out_dir: Path = OUTPUTS_DIR,
    k: int = DEFAULT_K_NEIGHBORS,
    resolution: float = DEFAULT_GRID_RESOLUTION,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    land: Geometry | None = None,
    validation_summary_path: Path = _VALIDATION_SUMMARY_PATH,
    validation_bundle_path: Path = _VALIDATION_BUNDLE_PATH,
    validation_global_path: Path = _VALIDATION_GLOBAL_PATH,
    explain_summary_path: Path = _EXPLAIN_SUMMARY_PATH,
    explain_bundle_path: Path = _EXPLAIN_BUNDLE_PATH,
) -> dict:
    """Build the dashboard asset bundle from the phase 1-4 outputs.

    Recomputes the phase 2 anomalies (windows read from the trends parquet
    itself), verifies them against the stored slopes, re-runs the phase 3
    cross-validation and surface (also refreshing
    ``outputs/trend_surface.html``), and re-quantifies the phase 4
    inequality fits, then writes the five bundle files to `out_dir`.

    Args:
        db_path: DuckDB database with the ingested city temperatures.
        trends_path: Phase 2 ``city_trends.parquet``.
        inequality_path: Phase 4 ``country_inequality.parquet``.
        out_dir: Bundle destination, normally the committed ``app/data/``.
        surface_out_dir: Where the phase 3 HTML figure is (re)written.
        k: Interpolation neighborhood size.
        resolution: Surface grid spacing in degrees.
        min_coverage: Phase 2 coverage gate (must match the trends build).
        land: Land geometry override for tests; None downloads Natural
            Earth polygons.

    Returns:
        Dict with `paths` (name -> Path), `stats` (the stats.json dict),
        `trends`, `anomalies`, and `surface` (the build_interpolated_surface
        result) for inspection.
    """
    trends = pd.read_parquet(trends_path)
    baseline_start, baseline_end = parse_window(trends["baseline_window"].iloc[0])
    start, end = parse_window(trends["analysis_window"].iloc[0])
    cutoff_year = int(end[:4])

    logger.info("recomputing anomalies for %d locations ...", len(trends))
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        anomalies = compute_anomalies(
            con,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            start=start,
            end=end,
            min_coverage=min_coverage,
        )
    finally:
        con.close()

    trends_out, anomalies_out = attach_city_ids(trends, anomalies)
    trends_out["label"] = disambiguate_labels(trends_out)
    logger.info("refitting Theil-Sen intercepts (consistency check) ...")
    intercepts = theil_sen_intercepts(trends_out, anomalies_out)
    trends_out["intercept"] = trends_out["city_id"].map(intercepts)

    logger.info("interpolating surface and re-running LOO-CV ...")
    surface_result = build_interpolated_surface(
        trends_path=trends_path,
        out_dir=surface_out_dir,
        k=k,
        resolution=resolution,
        land=land,
    )

    inequality = pd.read_parquet(inequality_path)
    inequality_result = quantify_inequality(inequality)

    stats_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trends": _sanity_stats(trends_out),
        "interpolation": {
            "winner": str(surface_result["winner"]),
            "k": k,
            "resolution_deg": resolution,
            "cv_leave_location_out": surface_result["cv"].to_dict(orient="records"),
            "cv_leave_row_out": surface_result["cv_leave_row_out"].to_dict(
                orient="records"
            ),
        },
        "inequality": {
            "n_countries": inequality_result.n_countries,
            "n_continents": inequality_result.n_continents,
            "spearman_rho": inequality_result.spearman_rho,
            "spearman_p": inequality_result.spearman_p,
            "ols_pooled": asdict(inequality_result.ols_pooled),
            "ols_fe": asdict(inequality_result.ols_fe),
            "cutoff_year": cutoff_year,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / name for name in (
        TRENDS_ASSET, ANOMALIES_ASSET, SURFACE_ASSET, INEQUALITY_ASSET, STATS_ASSET
    )}
    trends_out.to_parquet(paths[TRENDS_ASSET], index=False, compression="zstd")
    (
        anomalies_out[["city_id", "dt", "anomaly"]]
        .sort_values(["city_id", "dt"])
        .assign(anomaly=lambda df: df["anomaly"].astype("float32"))
        .to_parquet(paths[ANOMALIES_ASSET], index=False, compression="zstd")
    )
    _surface_to_long_form(
        surface_result["grid_lon"], surface_result["grid_lat"],
        surface_result["surface"],
    ).to_parquet(paths[SURFACE_ASSET], index=False, compression="zstd")
    inequality.to_parquet(paths[INEQUALITY_ASSET], index=False, compression="zstd")

    _merge_optional_findings(
        stats_payload, out_dir, paths,
        validation_summary_path=validation_summary_path,
        validation_bundle_path=validation_bundle_path,
        validation_global_path=validation_global_path,
        explain_summary_path=explain_summary_path,
        explain_bundle_path=explain_bundle_path,
    )

    paths[STATS_ASSET].write_text(
        json.dumps(stats_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for name, path in paths.items():
        logger.info("wrote %s (%.1f KB)", path, path.stat().st_size / 1024)

    return {
        "paths": paths,
        "stats": stats_payload,
        "trends": trends_out,
        "anomalies": anomalies_out,
        "surface": surface_result,
    }


def build_decomposition_summaries(
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_dir: Path = APP_DATA_DIR,
    *,
    city_features_path: Path | None = None,
    income_path: Path | None = None,
) -> dict[str, Path]:
    """Write the headline inequality + LMG/Shapley summaries into the bundle.

    These two JSON artifacts power the dashboard's headline *decomposition* page
    but are deliberately not produced by :func:`build_app_assets`: the
    decomposition consumes the Phase-7 city-feature table and the World Bank
    income groups, neither of which ships to the cloud. Wiring them here means a
    single ``python -m src.app_assets`` regenerates the *entire* committed
    bundle deterministically, replacing the old manual copy from
    ``data/processed/``.

    ``inequality_summary.json`` needs only the country table, so it is always
    written. ``decomposition_summary.json`` additionally needs the city features
    and income groups; when either input is absent it is skipped with a warning
    and the dashboard renders its explicit "not built yet" state. Both summaries
    are float-rounded at serialization (:func:`src.data_io.round_floats`) so the
    committed JSON is byte-stable across platforms.

    Args:
        inequality_path: Phase 4 ``country_inequality.parquet``.
        out_dir: Bundle destination, normally the committed ``app/data/``.
        city_features_path: Phase 7 ``city_features.parquet`` override (tests);
            ``None`` uses ``src.explain.DEFAULT_FEATURES_PATH``.
        income_path: World Bank income-groups CSV override (tests); ``None``
            uses ``src.explain.INCOME_PATH``.

    Returns:
        Dict of asset-name -> written Path (omits the decomposition summary when
        its inputs are missing).
    """
    # Lazy imports: keep the decomposition/inequality/explain stack out of this
    # module's import graph (build_app_assets is the heavy entry point) and
    # mirror the existing _merge_optional_findings discipline.
    from src.decomposition import build_country_design, group_lmg_shares
    from src.decomposition import summary_payload as decomp_payload
    from src.explain import DEFAULT_FEATURES_PATH, INCOME_PATH, load_income_groups
    from src.inequality import country_warming_inequality
    from src.inequality import summary_payload as ineq_payload

    city_features_path = city_features_path or DEFAULT_FEATURES_PATH
    income_path = income_path or INCOME_PATH

    inequality = pd.read_parquet(inequality_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    ineq_summary = country_warming_inequality(inequality)
    ineq_dest = out_dir / INEQUALITY_SUMMARY_ASSET
    ineq_dest.write_text(
        json.dumps(ineq_payload(ineq_summary), indent=2) + "\n", encoding="utf-8"
    )
    written[INEQUALITY_SUMMARY_ASSET] = ineq_dest
    logger.info("wrote %s (n=%d)", ineq_dest, ineq_summary.n)

    if not city_features_path.exists() or not income_path.exists():
        logger.warning(
            "decomposition inputs missing (city_features present=%s, income "
            "present=%s); bundle will omit the decomposition summary -- run "
            "python -m src.explain to build the city features",
            city_features_path.exists(), income_path.exists(),
        )
        return written

    city_features = pd.read_parquet(city_features_path)
    income = load_income_groups(income_path)
    design = build_country_design(inequality, city_features, income)
    result = group_lmg_shares(design)
    decomp_dest = out_dir / DECOMPOSITION_SUMMARY_ASSET
    decomp_dest.write_text(
        json.dumps(decomp_payload(result), indent=2) + "\n", encoding="utf-8"
    )
    written[DECOMPOSITION_SUMMARY_ASSET] = decomp_dest
    logger.info(
        "wrote %s (n=%d, R^2=%.3f, residual=%.3f)",
        decomp_dest, result.n, result.total_r2, result.residual_share,
    )
    return written


def main() -> None:
    """Build the default bundle and print the README sanity checks."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    out = build_app_assets()
    summaries = build_decomposition_summaries()
    trends = out["stats"]["trends"]
    interp = out["stats"]["interpolation"]
    ineq = out["stats"]["inequality"]
    print(f"city-locations: {trends['n_locations']}")
    print(
        f"global mean slope: {trends['global_mean_c_per_decade']:.3f} °C/decade; "
        f">60°N: {trends['arctic_mean_c_per_decade']:.3f} "
        f"({trends['n_arctic']} locations), ratio {trends['arctic_ratio']:.2f}x"
    )
    loo = {row["method"]: row["rmse"] for row in interp["cv_leave_location_out"]}
    print(
        f"interpolation winner: {interp['winner']} "
        f"(leave-location-out RMSE idw {loo['idw']:.4f} "
        f"vs kriging {loo['kriging']:.4f})"
    )
    print(
        f"countries: {ineq['n_countries']}; "
        f"Spearman rho {ineq['spearman_rho']:+.3f} (p={ineq['spearman_p']:.2g})"
    )
    fe = ineq["ols_fe"]
    print(
        f"continent-FE OLS: {fe['coef']:+.4f} °C/decade per 10x emissions "
        f"[95% CI {fe['ci_low']:+.4f}, {fe['ci_high']:+.4f}], "
        f"p={fe['p_value']:.2g}, R²={fe['r2']:.3f}"
    )
    if "validation" in out["stats"]:
        v = out["stats"]["validation"]
        print(
            f"validation: mean residual {v['mean_residual']:+.3f} °C "
            f"(excl 2023+: {v['mean_residual_pre2023']:+.3f}); "
            f"full-record slope {v['mean_slope_full']:.3f} vs "
            f"stored {v['mean_slope_stored']:.3f} °C/decade"
        )
    if "explain" in out["stats"]:
        country_specs = out["stats"]["explain"]["country_model"]["specs"]
        lat_cont = next((s for s in country_specs if s["spec_name"] == "lat_continent"), None)
        if lat_cont and lat_cont["emissions"]:
            em = lat_cont["emissions"]
            print(
                f"explain (lat_continent): log10_emissions {em['coef']:+.4f} "
                f"[{em['ci_low']:+.4f}, {em['ci_high']:+.4f}] p={em['p_value']:.3g}"
            )

    if DECOMPOSITION_SUMMARY_ASSET in summaries:
        decomp = json.loads(
            summaries[DECOMPOSITION_SUMMARY_ASSET].read_text(encoding="utf-8")
        )
        shares = decomp["shares"]
        top = max(shares, key=shares.get)
        print(
            f"decomposition: total R^2 {decomp['total_r2']:.3f}, residual "
            f"{decomp['residual_share']:.3f}; largest axis {top} "
            f"{shares[top]:.3f} (descriptive, non-causal)"
        )
    else:
        print("decomposition: skipped (city features / income not built)")

    all_paths = {**out["paths"], **summaries}
    total_kb = sum(p.stat().st_size for p in all_paths.values()) / 1024
    print(f"bundle: {len(all_paths)} files, {total_kb:,.0f} KB total")
    for name, path in all_paths.items():
        print(f"  {name}: {path.stat().st_size / 1024:,.1f} KB")


if __name__ == "__main__":
    main()
