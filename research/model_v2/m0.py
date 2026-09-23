"""M0: the V1 decomposition model, refit exactly as ``src.decomposition`` fits it.

Everything here reuses the V1 code path (:func:`src.decomposition.build_country_design`,
:func:`src.decomposition.feature_block`, the same complete-case rule and the same
least-squares solve) so that the residuals inspected in the Model V2 research line
are the residuals of the tagged model, not of a re-implementation. The module adds
only *diagnostic* quantities (leverage, studentised residuals, Cook's distance) and
descriptive columns carried for inspection; it never changes the design.

The reference row :data:`M0_REFERENCE` is transcribed from the ``v1.3.0`` bundle
(``app/data/decomposition_summary.json`` and ``stability_summary.json`` at that
tag) and :func:`verify_against_bundle` checks a refit against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import feature_schema as fs
from src.decomposition import (
    COUNTRY_COL,
    OUTCOME_COL,
    PRIMARY_OUTCOME,
    build_country_design,
    decompose_country_warming,
    feature_block,
)
from src.emissions import DEFAULT_INEQUALITY_PATH, OWID_CO2_PATH
from src.explain import DEFAULT_FEATURES_PATH, INCOME_PATH, load_income_groups, morans_i
from src.stability import _complete_case, _used_features, country_centroids

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = ROOT / "app" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

V1_TAG = "v1.3.0"
V1_COMMIT = "a6733cb78292b9b27466e7ebfe7c4fdc6b59592f"

# The immutable M0 comparison row (bundle values at v1.3.0; 10 significant figures).
M0_REFERENCE: dict[str, float | int | str] = {
    "tag": V1_TAG,
    "commit": V1_COMMIT,
    "outcome_definition": "area_weighted",
    "outcome_source_column": "trend_c_per_decade_area_weighted",
    "n": 151,
    "in_sample_r2": 0.6364460028,
    "residual_share": 0.3635539972,
    "share_geography": 0.5130328657,
    "share_emissions": 0.02139283879,
    "share_socioeconomic": 0.06932302426,
    "share_population": 0.032697274,
    "residual_morans_i": 0.2686890161,
    "residual_morans_i_p": 0.005,
    "morans_weights": "station-centroid kNN, k=8, row-standardised, 199 permutations, seed 0",
}

# Numeric-feature order used by the design matrix (schema order within groups).
V1_GROUP_ORDER = ("emissions", "geography", "socioeconomic", "population")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three V1 artifacts the decomposition consumes (country table, city features, income)."""
    inequality = pd.read_parquet(DEFAULT_INEQUALITY_PATH)
    city_features = pd.read_parquet(DEFAULT_FEATURES_PATH)
    income = load_income_groups(INCOME_PATH)
    return inequality, city_features, income


def iso3_bridge(inequality: pd.DataFrame, owid_csv: Path = OWID_CO2_PATH) -> pd.Series:
    """``Country`` -> ISO3 through the OWID name, the bridge the area weighting used."""
    owid = (
        pd.read_csv(owid_csv, usecols=["country", "iso_code"])
        .dropna()
        .drop_duplicates("country")
        .set_index("country")["iso_code"]
    )
    return pd.Series(
        inequality["owid_country"].map(owid).to_numpy(),
        index=inequality[COUNTRY_COL].to_numpy(),
        name="iso3",
    )


def m0_complete_design(
    inequality: pd.DataFrame, city_features: pd.DataFrame, income: pd.DataFrame
) -> pd.DataFrame:
    """The complete-case V1 design (151 rows) for the primary, area-weighted outcome."""
    design = build_country_design(
        inequality, city_features, income, outcome_definition=PRIMARY_OUTCOME
    )
    return _complete_case(design, fs.SCHEMA_V1, fs.STATUS_AVAILABLE)


def design_matrix(complete: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    """Intercept + the V1 feature blocks, in the order ``group_lmg_shares`` builds them.

    Returns ``(X, column_names, columns_by_group)``.
    """
    used = _used_features(complete.columns, fs.SCHEMA_V1, fs.STATUS_AVAILABLE)
    blocks, names, by_group = [], ["intercept"], {}
    for group, features in used.items():
        by_group[group] = []
        for feature in features:
            block, cols = feature_block(complete, feature)
            blocks.append(block)
            names.extend(cols)
            by_group[group].extend(cols)
    X = np.hstack([np.ones((len(complete), 1)), *blocks])
    return X, names, by_group


@dataclass(frozen=True)
class M0Fit:
    """The full-coalition OLS behind M0's ``total_r2``, with influence diagnostics."""

    countries: np.ndarray
    y: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    leverage: np.ndarray
    studentized: np.ndarray
    cooks_d: np.ndarray
    coef: pd.Series
    r2: float
    sigma: float
    n: int
    p: int  # columns including the intercept
    rank: int


def fit_m0(complete: pd.DataFrame) -> M0Fit:
    """Refit the full V1 model with the same least-norm solve and add influence diagnostics."""
    X, names, _ = design_matrix(complete)
    y = complete[OUTCOME_COL].to_numpy(dtype=float)
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    resid = y - fitted
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot

    # Leverage = diagonal of the projection onto the column space of X. The V1
    # design is rank-deficient (see :func:`collinearity_report`), so the projection
    # is built from the rank-truncated SVD, never from a QR that would count the
    # null direction.
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    keep = s > s[0] * max(X.shape) * np.finfo(float).eps
    h = np.sum(u[:, keep] ** 2, axis=1)
    n, p = X.shape
    rank = int(keep.sum())
    dof = n - rank
    sigma2 = float(np.sum(resid**2)) / dof
    one_minus_h = np.clip(1.0 - h, 1e-12, None)
    # Externally studentised residual: e_i / (s_(i) * sqrt(1 - h_i)).
    s2_i = (sigma2 * dof - resid**2 / one_minus_h) / (dof - 1)
    s2_i = np.clip(s2_i, 1e-18, None)
    student = resid / np.sqrt(s2_i * one_minus_h)
    cooks = (resid**2 / (rank * sigma2)) * (h / one_minus_h**2)
    exact = h > 1 - 1e-8  # a unit fitted by its own dummy has zero residual
    student[exact] = np.nan
    cooks[exact] = np.nan
    return M0Fit(
        countries=complete[COUNTRY_COL].to_numpy(),
        y=y,
        fitted=fitted,
        residual=resid,
        leverage=h,
        studentized=student,
        cooks_d=cooks,
        coef=pd.Series(coef, index=names),
        r2=r2,
        sigma=float(np.sqrt(sigma2)),
        n=n,
        p=p,
        rank=int(rank),
    )


def collinearity_report(complete: pd.DataFrame) -> dict:
    """Singular values of the V1 design and the exact linear dependency it carries.

    ``cum_co2_per_capita`` is cumulative CO2 (Mt) times 1e6 over the 2013 population,
    so after the log10 transforms ``log10(per_capita) - log10(total) + log10(population)``
    is the constant 6 for every country and the column is a linear combination of the
    other two and the intercept. The report states the residual of that identity and
    the smallest singular value so the dependency is on record.
    """
    X, names, _ = design_matrix(complete)
    s = np.linalg.svd(X, compute_uv=False)
    ipc = names.index("cum_co2_per_capita")
    itot = names.index("cum_co2_total")
    ipop = names.index("population")
    identity = X[:, ipc] - X[:, itot] + X[:, ipop]
    return {
        "n_columns": X.shape[1],
        "rank": int(np.linalg.matrix_rank(X)),
        "smallest_singular_values": [float(v) for v in s[-3:]],
        "identity": "log10(cum_co2_per_capita) - log10(cum_co2_total) + log10(population)",
        "identity_mean": float(identity.mean()),
        "identity_max_abs_deviation": float(np.max(np.abs(identity - identity.mean()))),
    }


def m0_country_table() -> tuple[pd.DataFrame, M0Fit]:
    """One row per M0 country: outcome, fit, residual, diagnostics, features, descriptors.

    Descriptor columns (station-weighted trend, coverage, number of cities, the ERA5
    area trend, the station centroid) are carried for inspection only; none enters
    the model.
    """
    inequality, city_features, income = load_inputs()
    complete = m0_complete_design(inequality, city_features, income)
    fit = fit_m0(complete)
    table = complete.copy()
    table["fitted"] = fit.fitted
    table["residual"] = fit.residual
    table["leverage"] = fit.leverage
    table["studentized"] = fit.studentized
    table["cooks_d"] = fit.cooks_d
    table["iso3"] = table[COUNTRY_COL].map(iso3_bridge(inequality))

    extra = inequality.set_index(COUNTRY_COL)[
        ["trend_c_per_decade", "trend_c_per_decade_pop_weighted", "area_cell_coverage", "n_cities"]
    ].rename(columns={"trend_c_per_decade": "station_trend", "trend_c_per_decade_pop_weighted": "people_trend"})
    table = table.join(extra, on=COUNTRY_COL)
    cen = country_centroids(city_features).set_index(COUNTRY_COL)
    table["station_lon"] = table[COUNTRY_COL].map(cen["Longitude"])
    table["station_lat"] = table[COUNTRY_COL].map(cen["Latitude"])
    era5_path = BUNDLE_DIR / "era5_area_trends.parquet"
    if era5_path.exists():
        era5 = pd.read_parquet(era5_path).set_index("iso3")["trend_c_per_decade_era5_area"]
        table["era5_trend"] = table["iso3"].map(era5)
    return table, fit


def verify_against_bundle(atol: float = 1e-9) -> dict:
    """Refit M0 from local artifacts and compare with the tagged bundle summaries."""
    inequality, city_features, income = load_inputs()
    primary, _ = decompose_country_warming(inequality, city_features, income)
    decomposition = json.loads((BUNDLE_DIR / "decomposition_summary.json").read_text())
    stability = json.loads((BUNDLE_DIR / "stability_summary.json").read_text())

    complete = m0_complete_design(inequality, city_features, income)
    fit = fit_m0(complete)
    cen = country_centroids(city_features).set_index(COUNTRY_COL)
    moran, p_value = morans_i(
        fit.residual,
        cen.loc[fit.countries, "Longitude"].to_numpy(),
        cen.loc[fit.countries, "Latitude"].to_numpy(),
        k=8, n_permutations=199, seed=0,
    )
    rows = {
        "n": (primary.n, decomposition["n"], M0_REFERENCE["n"]),
        "in_sample_r2": (primary.total_r2, decomposition["total_r2"], M0_REFERENCE["in_sample_r2"]),
        "residual_share": (primary.residual_share, decomposition["residual_share"], M0_REFERENCE["residual_share"]),
        "residual_morans_i": (moran, stability["residual_spatial"]["morans_i"], M0_REFERENCE["residual_morans_i"]),
        "residual_morans_i_p": (p_value, stability["residual_spatial"]["p_value"], M0_REFERENCE["residual_morans_i_p"]),
    }
    for group in V1_GROUP_ORDER:
        rows[f"share_{group}"] = (
            primary.shares[group], decomposition["shares"][group], M0_REFERENCE[f"share_{group}"]
        )
    report = {}
    for key, (refit, bundle, reference) in rows.items():
        report[key] = {
            "refit": float(refit),
            "bundle": float(bundle),
            "reference": float(reference),
            "refit_minus_bundle": float(refit - bundle),
            "bundle_equals_reference": abs(float(bundle) - float(reference)) <= atol,
        }
    report["full_model_r2_equals_total_r2"] = abs(fit.r2 - primary.total_r2) < 1e-12
    report["design_rank"] = fit.rank
    report["design_columns"] = fit.p
    report["all_bundle_values_match_reference"] = all(
        v["bundle_equals_reference"] for v in report.values() if isinstance(v, dict)
    )
    return report
