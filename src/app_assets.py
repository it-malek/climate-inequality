"""Build the committed dashboard bundle in ``app/data/``.

The Streamlit app runs on Streamlit Community Cloud, which sees only the git
repository: the DuckDB database and the raw downloads cannot ship there. This
module distills the pipeline outputs into a small bundle (parquet + JSON, ~5 MB,
dominated by the per-city anomaly series) that the app reads identically on a
laptop and in the cloud. Committing ``app/data/`` is the one deliberate
exception to the rule that ``data/`` and ``outputs/`` are never committed.

:func:`build_app_assets` writes the core assets (trends with ``city_id``,
``label`` and the Theil-Sen ``intercept``; monthly anomalies; the interpolated
surface; the country table; ``stats.json``) and folds in the validation and
explanatory summaries when they exist. The remaining ``build_*`` functions
each write one optional group of artifacts (decomposition, stability, the
coupling lenses, the ERA5 cross-check, the vulnerability strata, the physical
model) and skip with a warning when their inputs are absent, so a partial
pipeline still yields a working bundle and the dashboard shows a pending
state for the missing pages. :func:`main` runs all of them.

Rebuilding the anomalies re-runs the trend computation, so the builder
cross-checks the refit Theil-Sen slopes against the stored ones and refuses to
publish from a stale or inconsistent pipeline state.

    uv run python -m src.app_assets
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from shapely import Geometry

from src.bundle import (
    ANOMALIES_ASSET,
    APP_DATA_DIR,
    COUPLING_AREA_ASSET,
    COUPLING_AREA_SUMMARY_ASSET,
    COUPLING_ASSET,
    COUPLING_CONSUMPTION_ASSET,
    COUPLING_CONSUMPTION_SUMMARY_ASSET,
    COUPLING_EXPOSURE_ASSET,
    COUPLING_EXPOSURE_SUMMARY_ASSET,
    COUPLING_SUMMARY_ASSET,
    DECOMPOSITION_SUMMARY_ASSET,
    ERA5_AREA_TRENDS_ASSET,
    ERA5_VALIDATION_SUMMARY_ASSET,
    EXPLAIN_FEATURES_ASSET,
    INEQUALITY_ASSET,
    INEQUALITY_SUMMARY_ASSET,
    PHYSICAL_SUMMARY_ASSET,
    PHYSICAL_TRAJECTORY_ASSET,
    STABILITY_SUMMARY_ASSET,
    STATS_ASSET,
    SURFACE_ASSET,
    TRENDS_ASSET,
    VALIDATION_ASSET,
    VALIDATION_GLOBAL_ASSET,
    VULNERABILITY_STRATA_ASSET,
    VULNERABILITY_SUMMARY_ASSET,
)
from src.cleaning import parse_window, to_decimal_decades
from src.coupling import (
    COUPLING_AREA_SCHEMA,
    COUPLING_CONSUMPTION_SCHEMA,
    COUPLING_EXPOSURE_SCHEMA,
    COUPLING_SCHEMA,
    area_summary_payload,
    compute_area_coupling,
    compute_consumption_coupling,
    compute_coupling,
    compute_exposure_coupling,
    consumption_summary_payload,
    exposure_summary_payload,
)
from src.coupling import summary_payload as coupling_payload
from src.data_io import DEFAULT_DB_PATH, OUTPUTS_DIR, PROCESSED_DIR, write_typed_parquet
from src.decomposition import COUNTRY_COL, build_country_design, group_lmg_shares
from src.decomposition import summary_payload as decomp_payload
from src.emissions import (
    CONSUMPTION_COLUMNS,
    DEFAULT_INEQUALITY_PATH,
    OWID_CO2_COMMIT,
    OWID_CO2_PATH,
    OWID_CO2_RETRIEVED,
    OWID_CO2_SHA256,
    quantify_inequality,
    sha256_file,
)
from src.era5_validation import ERA5_GRID_PATH, build_era5_validation
from src.explain import (
    DEFAULT_EXPLAIN_BUNDLE_PATH,
    DEFAULT_EXPLAIN_SUMMARY_PATH,
    DEFAULT_FEATURES_PATH,
    EXPLAIN_BUNDLE_SCHEMA,
    INCOME_PATH,
    load_income_groups,
)
from src.forcings import DEFAULT_FORCINGS_PATH
from src.inequality import country_warming_inequality
from src.inequality import summary_payload as ineq_payload
from src.interpolate import (
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_K_NEIGHBORS,
    build_interpolated_surface,
)
from src.physical_model import TRAJECTORY_SCHEMA, compute_physical_model
from src.physical_model import summary_payload as physical_payload
from src.projections import (
    ID_COL,
    area_coverage,
    consumption_window,
    population_coverage,
    resolve_area_projections,
    resolve_consumption_projections,
    resolve_exposure_projections,
    resolve_projections,
)
from src.stability import DEFAULT_N_BOOT, build_stability_summary
from src.stability import summary_payload as stability_payload
from src.trends import (
    CITY_KEYS,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_TRENDS_PATH,
    compute_anomalies,
)
from src.validation import (
    DEFAULT_VALIDATION_BUNDLE_PATH,
    DEFAULT_VALIDATION_GLOBAL_PATH,
    DEFAULT_VALIDATION_SUMMARY_PATH,
    VALIDATION_BUNDLE_SCHEMA,
    VALIDATION_GLOBAL_SCHEMA,
)
from src.vulnerability import NDGAIN_PATH, build_vulnerability

logger = logging.getLogger(__name__)

ARCTIC_LATITUDE = 60.0
# Refit slopes must match the stored parquet to float-noise levels; real
# drift (stale parquet vs database) shows up orders of magnitude larger.
SLOPE_CONSISTENCY_ATOL = 1e-8

_STABILITY_SUMMARY_PATH = PROCESSED_DIR / "stability_summary.json"


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
        trends: one row per city-location (``city_trends.parquet``).
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

    The trends table stores only slopes; the dashboard needs intercepts to draw the
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


def _owid_provenance(co2_path: Path = OWID_CO2_PATH) -> dict:
    """The OWID CO2 vintage behind the country table, for stats.json.

    Records the pinned commit, retrieval date and digest, plus the digest of the
    local file when it is present, so a bundle built from a different vintage is
    visible in the committed JSON.
    """
    return {
        "owid_co2_commit": OWID_CO2_COMMIT,
        "owid_co2_retrieved": OWID_CO2_RETRIEVED,
        "owid_co2_sha256_pinned": OWID_CO2_SHA256,
        "owid_co2_sha256_local": sha256_file(co2_path) if co2_path.exists() else None,
    }


def _sanity_stats(trends: pd.DataFrame) -> dict:
    """Headline trend numbers (the sanity checkpoints)."""
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
    """Merge the validation and explanatory summaries into the bundle, if present.

    Reads the small JSON + parquet artifacts written by
    ``python -m src.validation`` and ``python -m src.explain``, and:
    - adds ``stats_payload["validation"]`` / ``["explain"]`` keys;
    - copies the slim parquets into `out_dir`.

    If both artifacts for a stage are absent, logs a warning and skips that
    stage. If only one of the pair exists, raises RuntimeError to flag a
    stale pipeline.
    """
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
                "%s summaries not found; bundle will omit that page -- "
                "run python -m src.%s first",
                phase, phase,
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
    validation_summary_path: Path = DEFAULT_VALIDATION_SUMMARY_PATH,
    validation_bundle_path: Path = DEFAULT_VALIDATION_BUNDLE_PATH,
    validation_global_path: Path = DEFAULT_VALIDATION_GLOBAL_PATH,
    explain_summary_path: Path = DEFAULT_EXPLAIN_SUMMARY_PATH,
    explain_bundle_path: Path = DEFAULT_EXPLAIN_BUNDLE_PATH,
) -> dict:
    """Build the core bundle assets.

    Recomputes the anomalies (windows read from the trends parquet itself),
    verifies them against the stored slopes, re-runs the interpolation
    cross-validation and surface (also refreshing
    ``outputs/trend_surface.html``), and re-quantifies the emissions fits,
    then writes the five core files to `out_dir`.

    Args:
        db_path: DuckDB database with the ingested city temperatures.
        trends_path: ``city_trends.parquet``.
        inequality_path: ``country_inequality.parquet``.
        out_dir: Bundle destination, normally the committed ``app/data/``.
        surface_out_dir: Where the surface HTML figure is (re)written.
        k: Interpolation neighborhood size.
        resolution: Surface grid spacing in degrees.
        min_coverage: Coverage gate (must match the trends build).
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
        "provenance": _owid_provenance(),
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

    ``inequality_summary.json`` needs only the country table, so it is always
    written. ``decomposition_summary.json`` additionally needs the city features
    and income groups; when either input is absent it is skipped with a warning
    and the dashboard renders its "not built yet" state. Both summaries are
    float-rounded at serialization (:func:`src.data_io.round_floats`).

    Args:
        inequality_path: ``country_inequality.parquet``.
        out_dir: Bundle destination, normally the committed ``app/data/``.
        city_features_path: ``city_features.parquet`` override (tests);
            ``None`` uses ``src.explain.DEFAULT_FEATURES_PATH``.
        income_path: World Bank income-groups CSV override (tests); ``None``
            uses ``src.explain.INCOME_PATH``.

    Returns:
        Dict of asset-name -> written Path (omits the decomposition summary when
        its inputs are missing).
    """
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


def build_stability_summary_asset(
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_dir: Path = APP_DATA_DIR,
    *,
    city_features_path: Path | None = None,
    income_path: Path | None = None,
    n_boot: int | None = None,
    seed: int = 0,
) -> dict[str, Path]:
    """Write the decomposition's perturbation-stability summary into the bundle.

    Needs the city features (for the design and for the country centroids the
    residual Moran's I uses) and the income groups; when either is absent it is
    skipped with a warning and the page keeps its pending state. The bootstraps
    run here, offline; only the small JSON ships.

    Args:
        inequality_path: ``country_inequality.parquet``.
        out_dir: Bundle destination, normally the committed ``app/data/``.
        city_features_path: ``city_features.parquet`` override (tests);
            ``None`` uses ``src.explain.DEFAULT_FEATURES_PATH``.
        income_path: World Bank income-groups CSV override (tests); ``None`` uses
            ``src.explain.INCOME_PATH``.
        n_boot: bootstrap resamples; ``None`` uses the stability-module default.
        seed: base RNG seed for reproducible resamples.

    Returns:
        ``{STABILITY_SUMMARY_ASSET: path}`` when written, else ``{}``.
    """
    city_features_path = city_features_path or DEFAULT_FEATURES_PATH
    income_path = income_path or INCOME_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if not city_features_path.exists() or not income_path.exists():
        logger.warning(
            "stability inputs missing (city_features present=%s, income present=%s); "
            "bundle will omit the stability summary -- run python -m src.explain to "
            "build the city features",
            city_features_path.exists(), income_path.exists(),
        )
        return written

    inequality = pd.read_parquet(inequality_path)
    city_features = pd.read_parquet(city_features_path)
    income = load_income_groups(income_path)
    design = build_country_design(inequality, city_features, income)
    centroids = (
        city_features.groupby(COUNTRY_COL, as_index=False)[["Longitude", "Latitude"]]
        .mean()
    )

    result = build_stability_summary(
        design, centroids, n_boot=n_boot or DEFAULT_N_BOOT, seed=seed
    )
    dest = out_dir / STABILITY_SUMMARY_ASSET
    dest.write_text(
        json.dumps(stability_payload(result), indent=2) + "\n", encoding="utf-8"
    )
    written[STABILITY_SUMMARY_ASSET] = dest
    logger.info(
        "wrote %s (n=%d, B=%d, P(geo largest)=%s)",
        dest, result.n_countries, result.n_boot,
        result.share_stability.get("p_geography_largest"),
    )
    return written


def build_coupling_summary_asset(
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_dir: Path = APP_DATA_DIR,
) -> dict[str, Path]:
    """Write the station-based coupling artifacts into the bundle.

    The comparator's only input is the country table that always ships, so this
    is built unconditionally: the v1 projections are resolved in memory and the
    comparator writes ``coupling.parquet`` and ``coupling_summary.json``.

    Args:
        inequality_path: ``country_inequality.parquet``.
        out_dir: Bundle destination, normally the committed ``app/data/``.

    Returns:
        Dict of asset-name -> written Path.
    """
    inequality = pd.read_parquet(inequality_path)
    projections = resolve_projections(inequality)
    table, result = compute_coupling(projections)
    result.check()

    out_dir.mkdir(parents=True, exist_ok=True)
    table_dest = out_dir / COUPLING_ASSET
    write_typed_parquet(table, table_dest, COUPLING_SCHEMA, order_by=(ID_COL,))
    summary_dest = out_dir / COUPLING_SUMMARY_ASSET
    summary_dest.write_text(
        json.dumps(coupling_payload(result), indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "wrote %s and %s (n=%d, inequality_coefficient=%.3f)",
        table_dest, summary_dest, len(table), result.inequality_coefficient,
    )
    return {COUPLING_ASSET: table_dest, COUPLING_SUMMARY_ASSET: summary_dest}


def build_coupling_consumption_asset(
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_dir: Path = APP_DATA_DIR,
) -> dict[str, Path]:
    """Write the consumption-lens coupling artifacts into the bundle (best-effort).

    Runs the two-pass comparator (consumption vs impact, and the window-matched
    production-to-consumption rank shift) and writes
    ``coupling_consumption.parquet`` + ``coupling_consumption_summary.json``.
    When the country table lacks the consumption columns, or no country has a
    consumption window, it logs and returns ``{}``.

    Args:
        inequality_path: ``country_inequality.parquet``.
        out_dir: Bundle destination, normally the committed ``app/data/``.

    Returns:
        Dict of asset-name -> written Path, or ``{}`` when the consumption lens
        cannot be built.
    """
    inequality = pd.read_parquet(inequality_path)
    missing = [c for c in CONSUMPTION_COLUMNS if c not in inequality.columns]
    if missing or inequality["cum_consumption_t_per_capita"].notna().sum() == 0:
        logger.warning(
            "consumption columns absent/empty (missing=%s); bundle will omit the "
            "consumption lens -- rebuild country_inequality.parquet (python -m "
            "src.emissions)",
            missing,
        )
        return {}

    projections = resolve_consumption_projections(inequality)
    table, consumption_vs_impact, production_to_consumption_shift = (
        compute_consumption_coupling(projections)
    )
    consumption_vs_impact.check()
    production_to_consumption_shift.check()
    window = consumption_window(inequality)

    out_dir.mkdir(parents=True, exist_ok=True)
    table_dest = out_dir / COUPLING_CONSUMPTION_ASSET
    write_typed_parquet(
        table, table_dest, COUPLING_CONSUMPTION_SCHEMA, order_by=(ID_COL,)
    )
    summary_dest = out_dir / COUPLING_CONSUMPTION_SUMMARY_ASSET
    summary_dest.write_text(
        json.dumps(
            consumption_summary_payload(
                consumption_vs_impact, production_to_consumption_shift, window
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "wrote %s and %s (n=%d, consumption inequality_coefficient=%.3f, "
        "prod->cons rho=%.3f)",
        table_dest, summary_dest, len(table),
        consumption_vs_impact.inequality_coefficient,
        production_to_consumption_shift.spearman_rho,
    )
    return {
        COUPLING_CONSUMPTION_ASSET: table_dest,
        COUPLING_CONSUMPTION_SUMMARY_ASSET: summary_dest,
    }


def build_coupling_exposure_asset(
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_dir: Path = APP_DATA_DIR,
) -> dict[str, Path]:
    """Write the people-weighted exposure coupling artifacts (best-effort).

    Runs the two passes (station-vs-people rank shift and the people-weighted
    inequality) and writes ``coupling_exposure.parquet`` +
    ``coupling_exposure_summary.json``. Returns ``{}`` when the country table
    carries no people-weighting (the population grid was absent at build time).

    Args:
        inequality_path: ``country_inequality.parquet``.
        out_dir: Bundle destination, normally the committed ``app/data/``.

    Returns:
        Dict of asset-name -> written Path, or ``{}`` when the exposure lens
        cannot be built.
    """
    inequality = pd.read_parquet(inequality_path)
    if (
        "trend_c_per_decade_pop_weighted" not in inequality.columns
        or inequality["trend_c_per_decade_pop_weighted"].notna().sum() == 0
    ):
        logger.warning(
            "people-weighted column absent/empty; bundle will omit the exposure "
            "lens -- rebuild country_inequality.parquet with the population grid "
            "present (python -m src.emissions)"
        )
        return {}

    projections = resolve_exposure_projections(inequality)
    table, station_vs_people, people_weighted_inequality = compute_exposure_coupling(
        projections
    )
    station_vs_people.check()
    people_weighted_inequality.check()
    coverage = population_coverage(inequality)

    out_dir.mkdir(parents=True, exist_ok=True)
    table_dest = out_dir / COUPLING_EXPOSURE_ASSET
    write_typed_parquet(
        table, table_dest, COUPLING_EXPOSURE_SCHEMA, order_by=(ID_COL,)
    )
    summary_dest = out_dir / COUPLING_EXPOSURE_SUMMARY_ASSET
    summary_dest.write_text(
        json.dumps(
            exposure_summary_payload(
                station_vs_people, people_weighted_inequality, coverage
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "wrote %s and %s (n=%d, people-weighted inequality_coefficient=%.3f, "
        "station->people rho=%.3f)",
        table_dest, summary_dest, len(table),
        people_weighted_inequality.inequality_coefficient,
        station_vs_people.spearman_rho,
    )
    return {
        COUPLING_EXPOSURE_ASSET: table_dest,
        COUPLING_EXPOSURE_SUMMARY_ASSET: summary_dest,
    }


def build_coupling_area_asset(
    inequality_path: Path = DEFAULT_INEQUALITY_PATH,
    out_dir: Path = APP_DATA_DIR,
) -> dict[str, Path]:
    """Write the area-weighted exposure coupling artifacts (best-effort).

    Runs the two passes (station-vs-area rank shift and the area-weighted
    inequality) and writes ``coupling_area.parquet`` +
    ``coupling_area_summary.json``. Returns ``{}`` when the country table
    carries no area-weighting (the Berkeley grid was absent at build time).

    Args:
        inequality_path: ``country_inequality.parquet``.
        out_dir: Bundle destination, normally the committed ``app/data/``.

    Returns:
        Dict of asset-name -> written Path, or ``{}`` when the area lens cannot
        be built.
    """
    inequality = pd.read_parquet(inequality_path)
    if (
        "trend_c_per_decade_area_weighted" not in inequality.columns
        or inequality["trend_c_per_decade_area_weighted"].notna().sum() == 0
    ):
        logger.warning(
            "area-weighted column absent/empty; bundle will omit the area lens -- "
            "rebuild country_inequality.parquet with the Berkeley grid present "
            "(python -m src.emissions)"
        )
        return {}

    projections = resolve_area_projections(inequality)
    table, station_vs_area, area_weighted_inequality = compute_area_coupling(
        projections
    )
    station_vs_area.check()
    area_weighted_inequality.check()
    coverage = area_coverage(inequality)

    out_dir.mkdir(parents=True, exist_ok=True)
    table_dest = out_dir / COUPLING_AREA_ASSET
    write_typed_parquet(table, table_dest, COUPLING_AREA_SCHEMA, order_by=(ID_COL,))
    summary_dest = out_dir / COUPLING_AREA_SUMMARY_ASSET
    summary_dest.write_text(
        json.dumps(
            area_summary_payload(
                station_vs_area, area_weighted_inequality, coverage
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "wrote %s and %s (n=%d, area-weighted inequality_coefficient=%.3f, "
        "station->area rho=%.3f)",
        table_dest, summary_dest, len(table),
        area_weighted_inequality.inequality_coefficient,
        station_vs_area.spearman_rho,
    )
    return {
        COUPLING_AREA_ASSET: table_dest,
        COUPLING_AREA_SUMMARY_ASSET: summary_dest,
    }


def build_era5_validation_asset(
    era5_grid_path: Path | None = None,
    out_dir: Path = APP_DATA_DIR,
) -> dict[str, Path]:
    """Write the ERA5 cross-check artifacts into the bundle (best-effort).

    Recomputes the area-weighted warming on the ERA5 grid and the
    station/Berkeley/ERA5 coupling comparison (:mod:`src.era5_validation`),
    writing ``era5_area_trends.parquet`` + ``era5_validation_summary.json``.
    Returns ``{}`` when the ~200 MB ERA5 grid is absent (fetch it with
    ``scripts/fetch_era5.py``); the validation page then omits the panel.

    Returns:
        Dict of asset-name -> written Path, or ``{}`` when the grid is absent.
    """
    grid_path = ERA5_GRID_PATH if era5_grid_path is None else era5_grid_path
    if not grid_path.exists():
        logger.warning(
            "ERA5 grid absent (%s); bundle will omit the ERA5 cross-check -- run "
            "scripts/fetch_era5.py to enable it",
            grid_path,
        )
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    trends_dest = out_dir / ERA5_AREA_TRENDS_ASSET
    summary_dest = out_dir / ERA5_VALIDATION_SUMMARY_ASSET
    build_era5_validation(
        era5_grid_path=grid_path,
        trends_path=trends_dest,
        summary_path=summary_dest,
    )
    logger.info("wrote %s and %s", trends_dest, summary_dest)
    return {
        ERA5_AREA_TRENDS_ASSET: trends_dest,
        ERA5_VALIDATION_SUMMARY_ASSET: summary_dest,
    }


def build_vulnerability_asset(
    income_path: Path | None = None,
    inequality_path: Path | None = None,
    ndgain_path: Path | None = None,
    co2_path: Path | None = None,
    out_dir: Path = APP_DATA_DIR,
) -> dict[str, Path]:
    """Write the income x vulnerability lens artifacts into the bundle (best-effort).

    Runs :func:`src.vulnerability.build_vulnerability` against the country table
    and the vendored World Bank income CSV, writing
    ``vulnerability_strata.parquet`` + ``vulnerability_summary.json``. The ND-GAIN
    block is added when the vendored ND-GAIN CSV and the OWID CO2 table (the
    ISO3 bridge) are both present. Returns ``{}`` only when the income CSV is
    absent.

    Args:
        income_path: World Bank income CSV; ``None`` uses
            :data:`src.vulnerability.INCOME_PATH`.
        inequality_path: ``country_inequality.parquet``; ``None`` uses
            :data:`src.emissions.DEFAULT_INEQUALITY_PATH` (parametrised for tests).
        ndgain_path: vendored ND-GAIN slim CSV; ``None`` uses
            :data:`src.vulnerability.NDGAIN_PATH`.
        co2_path: OWID CO2 table (ISO3 bridge); ``None`` uses
            :data:`src.emissions.OWID_CO2_PATH`.
        out_dir: Bundle destination, normally the committed ``app/data/``.

    Returns:
        Dict of asset-name -> written Path, or ``{}`` when the income CSV is absent.
    """
    income = INCOME_PATH if income_path is None else income_path
    inequality = DEFAULT_INEQUALITY_PATH if inequality_path is None else inequality_path
    ndgain = NDGAIN_PATH if ndgain_path is None else ndgain_path
    co2 = OWID_CO2_PATH if co2_path is None else co2_path
    if not income.exists():
        logger.warning(
            "income CSV absent (%s); bundle will omit the vulnerability lens -- "
            "download it via src.explain.INCOME_URL to enable it",
            income,
        )
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    table_dest = out_dir / VULNERABILITY_STRATA_ASSET
    summary_dest = out_dir / VULNERABILITY_SUMMARY_ASSET
    build_vulnerability(
        inequality_path=inequality,
        income_path=income,
        ndgain_path=ndgain,
        co2_path=co2,
        table_path=table_dest,
        summary_path=summary_dest,
    )
    logger.info("wrote %s and %s", table_dest, summary_dest)
    return {
        VULNERABILITY_STRATA_ASSET: table_dest,
        VULNERABILITY_SUMMARY_ASSET: summary_dest,
    }


def build_physical_summary_asset(
    forcings_path: Path | None = None,
    out_dir: Path = APP_DATA_DIR,
) -> dict[str, Path]:
    """Write the physical-model artifacts into the bundle (best-effort).

    Reads ``forcings.parquet``, refits the model and writes
    ``physical_trajectory.parquet`` + ``physical_summary.json``. The forcings
    table is network-derived and not committed; when it is absent the builder
    logs and returns ``{}``.

    Args:
        forcings_path: the assembled ``forcings.parquet`` (defaults to
            :data:`src.forcings.DEFAULT_FORCINGS_PATH`).
        out_dir: Bundle destination, normally the committed ``app/data/``.

    Returns:
        Dict of asset-name -> written Path, or ``{}`` when forcings are absent.
    """
    forcings_path = forcings_path or DEFAULT_FORCINGS_PATH
    if not forcings_path.exists():
        logger.info("physical: skipped (no %s; run python -m src.forcings)", forcings_path)
        return {}

    forcings = pd.read_parquet(forcings_path)
    # The forcings table is gitignored, so stamp its hash into the summary: the
    # committed artifact then names the exact input vintage it was fit on.
    forcings_hash = sha256_file(forcings_path)
    trajectory, result = compute_physical_model(forcings)
    result.check()

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dest = out_dir / PHYSICAL_TRAJECTORY_ASSET
    write_typed_parquet(trajectory, traj_dest, TRAJECTORY_SCHEMA, order_by=("year",))
    payload = {**physical_payload(result), "forcings_hash": forcings_hash}
    summary_dest = out_dir / PHYSICAL_SUMMARY_ASSET
    summary_dest.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "wrote %s and %s (n=%d, train<=%d, test coverage %.0f%%)",
        traj_dest, summary_dest, result.n_years, result.train_end,
        100 * result.hindcast["test_band_coverage"],
    )
    return {PHYSICAL_TRAJECTORY_ASSET: traj_dest, PHYSICAL_SUMMARY_ASSET: summary_dest}


def main() -> None:
    """Build the default bundle and print the README sanity checks."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    out = build_app_assets()
    summaries = build_decomposition_summaries()
    stability = build_stability_summary_asset()
    coupling = build_coupling_summary_asset()
    coupling_consumption = build_coupling_consumption_asset()
    coupling_exposure = build_coupling_exposure_asset()
    coupling_area = build_coupling_area_asset()
    era5_validation = build_era5_validation_asset()
    vulnerability = build_vulnerability_asset()
    physical = build_physical_summary_asset()
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

    if STABILITY_SUMMARY_ASSET in stability:
        stab = json.loads(
            stability[STABILITY_SUMMARY_ASSET].read_text(encoding="utf-8")
        )
        ss = stab["share_stability"]
        rs = stab["residual_spatial"]
        print(
            f"stability: P(geography largest) {ss['p_geography_largest']}, "
            f"P(emissions>0) {ss['p_emissions_positive']}, "
            f"residual Moran's I {rs['morans_i']} (p={rs['p_value']}) "
            "(descriptive, non-causal)"
        )
    else:
        print("stability: skipped (city features / income not built)")

    coupling_summary = json.loads(
        coupling[COUPLING_SUMMARY_ASSET].read_text(encoding="utf-8")
    )
    print(
        f"coupling: Spearman rho {coupling_summary['spearman_rho']:+.3f}, "
        f"inequality coefficient {coupling_summary['inequality_coefficient']:.3f}, "
        f"high-impact/low-responsibility "
        f"{coupling_summary['n_high_impact_low_responsibility']} "
        "(deterministic comparator)"
    )

    if COUPLING_CONSUMPTION_SUMMARY_ASSET in coupling_consumption:
        cc = json.loads(
            coupling_consumption[COUPLING_CONSUMPTION_SUMMARY_ASSET].read_text("utf-8")
        )
        win = cc["window"]
        shift = cc["production_to_consumption_shift"]
        print(
            f"coupling (consumption lens): n={win['n_countries']} over window "
            f"[{win['consumption_start_year_min']}..cutoff], prod->cons rank "
            f"rho {shift['spearman_rho']:+.3f}, consumption inequality coefficient "
            f"{cc['consumption_vs_impact']['inequality_coefficient']:.3f} "
            "(window-matched, descriptive)"
        )
    else:
        print("coupling (consumption lens): skipped (consumption columns not built)")

    if COUPLING_EXPOSURE_SUMMARY_ASSET in coupling_exposure:
        ce = json.loads(
            coupling_exposure[COUPLING_EXPOSURE_SUMMARY_ASSET].read_text("utf-8")
        )
        cov = ce["coverage"]
        sp = ce["station_vs_people"]
        print(
            f"coupling (exposure lens): n={cov['n_countries']} people-weighted, "
            f"station->people rank rho {sp['spearman_rho']:+.3f}, people-weighted "
            f"inequality coefficient "
            f"{ce['people_weighted_inequality']['inequality_coefficient']:.3f} "
            "(descriptive)"
        )
    else:
        print("coupling (exposure lens): skipped (population weighting not built)")

    if COUPLING_AREA_SUMMARY_ASSET in coupling_area:
        ca = json.loads(
            coupling_area[COUPLING_AREA_SUMMARY_ASSET].read_text("utf-8")
        )
        cov = ca["coverage"]
        sa = ca["station_vs_area"]
        print(
            f"coupling (area lens): n={cov['n_countries']} area-weighted, "
            f"station->area rank rho {sa['spearman_rho']:+.3f}, area-weighted "
            f"inequality coefficient "
            f"{ca['area_weighted_inequality']['inequality_coefficient']:.3f} "
            "(descriptive)"
        )
    else:
        print("coupling (area lens): skipped (area weighting not built)")

    if ERA5_VALIDATION_SUMMARY_ASSET in era5_validation:
        ev = json.loads(
            era5_validation[ERA5_VALIDATION_SUMMARY_ASSET].read_text("utf-8")
        )
        cc = ev["coupling_common"]
        wl = ev["world_land_mean"]
        print(
            f"ERA5 cross-check: world-land mean {wl['era5_area']:.3f} "
            f"(Berkeley {wl['berkeley_area']}, ref ~{wl['berkeley_reference']}); "
            f"common n={cc['n']} area-vs-responsibility rho "
            f"station {cc['station']['spearman_vs_responsibility']['rho']:+.3f} -> "
            f"Berkeley {cc['berkeley_area']['spearman_vs_responsibility']['rho']:+.3f} "
            f"-> ERA5 {cc['era5_area']['spearman_vs_responsibility']['rho']:+.3f} "
            "(independent reanalysis)"
        )
    else:
        print("ERA5 cross-check: skipped (ERA5 grid not fetched)")

    if VULNERABILITY_SUMMARY_ASSET in vulnerability:
        vuln = json.loads(
            vulnerability[VULNERABILITY_SUMMARY_ASSET].read_text("utf-8")
        )
        rg = vuln["responsibility"]["gradient_vs_income"]
        ag = vuln["exposure"]["area"]["gradient_vs_income"]
        ti = vuln["triple_inequality"]
        print(
            f"vulnerability (income lens): n={vuln['coverage']['n_countries']}, "
            f"responsibility vs income rho {rg['rho']:+.3f} (perm p={rg['p_permutation']:.3g}) "
            f"vs area-warming rho {ag['rho']:+.3f} (perm p={ag['p_permutation']:.3g}); "
            f"triple inequality holds={ti['holds']} (descriptive, non-causal)"
        )
    else:
        print("vulnerability (income lens): skipped (income CSV not present)")

    if PHYSICAL_SUMMARY_ASSET in physical:
        physical_summary = json.loads(
            physical[PHYSICAL_SUMMARY_ASSET].read_text(encoding="utf-8")
        )
        h = physical_summary["hindcast"]
        print(
            f"physical: n={physical_summary['n_years']}, "
            f"train<={physical_summary['train_end']}, AR(1) rho "
            f"{physical_summary['ar1_rho']:+.3f}, train R^2 {h['train_r2']:.3f}, "
            f"test band coverage {h['test_band_coverage']:.0%} "
            "(predictive association)"
        )
    else:
        print("physical: skipped (forcings.parquet not built)")

    all_paths = {
        **out["paths"], **summaries, **stability, **coupling,
        **coupling_consumption, **coupling_exposure, **coupling_area,
        **era5_validation, **vulnerability, **physical,
    }
    total_kb = sum(p.stat().st_size for p in all_paths.values()) / 1024
    print(f"bundle: {len(all_paths)} files, {total_kb:,.0f} KB total")
    for name, path in all_paths.items():
        print(f"  {name}: {path.stat().st_size / 1024:,.1f} KB")


if __name__ == "__main__":
    main()
