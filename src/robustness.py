"""Phase 9: robustness hardening of the country-level inequality coefficient.

Phase 7 (:mod:`src.explain`) reports that the headline
+0.029 C/decade-per-10x-emissions result is *specification-fragile*: adding
a linear ``mean_abs_lat`` control inside continent fixed effects roughly
halves it to an insignificant +0.012. Three assumptions make that swing hard
to interpret, and this module probes each one without re-running phases 1-6
and without adding any third-party dependency (GAM and robust regression both
ship inside ``statsmodels>=0.14``):

A. **Nonlinear latitude control.** Warming-vs-latitude amplification is
   strongly nonlinear, so a linear ``mean_abs_lat`` term can under-control the
   dominant confounder. :func:`compare_latitude_controls` re-estimates the
   emissions coefficient under three explicitly-defined specs -- A0 linear
   baseline, A1 GAM smoother on latitude, A2 GAM smoother + continent fixed
   effects -- and reports how much the estimate *moves*. This is a robustness
   check, not a causal verdict: a stable coefficient means the Phase 7 swing is
   more consistent with linear-control misspecification; a coefficient that
   stays attenuated and imprecise means the instability persists under a more
   flexible control.

B. **Independence.** City residuals are strongly spatially autocorrelated, yet
   the country model never checks its own residuals and trusts HC1 SEs.
   :func:`country_residual_morans_i` runs the existing
   :func:`src.explain.morans_i` on country-centroid residuals to *measure* the
   dependence. :func:`conley_hac_se` then produces the spatial-dependence-robust
   CI: it is a Conley spatial HAC estimator with a Bartlett kernel on
   great-circle distance between country centroids, so it models the
   distance-decay covariance the Moran's I detects. :func:`continent_cluster_bootstrap_se`
   is retained only as a coarse between-continent sensitivity, **not** as a
   valid CI: with ~6 continents the cluster count is far below where the
   cluster bootstrap has correct coverage, and it absorbs only continent-block
   (not sub-continental) dependence. Its result carries a
   ``clusters_sufficient`` flag that is ``False`` at this cluster count; prefer
   the Conley CI for inference.

C. **Estimator / weighting monoculture.** Everything upstream is unweighted
   OLS. :func:`fit_country_robust` (Huber RLM) checks robustness to *vertical*
   (response) outliers -- it is an M-estimator that bounds the influence of
   large residuals, **not** of high-leverage points, which it leaves essentially
   untouched. The design-matrix leverage question is handled separately by
   :func:`influence_diagnostics` (DFBETA on the emissions coefficient and Cook's
   distance from the fitted spec, plus a refit dropping the most influential
   countries). :func:`jackknife_emissions_coef` adds leave-one-country-out
   plus a named high-latitude drop-set, and :func:`weighted_inequality_fit`
   re-weights the country means. Together they report whether the headline
   rides on a handful of outlying / high-influence emitters.

:func:`run_robustness_suite` bundles A+B+C into a serializable
:class:`RobustnessSummary`, parallel to :func:`src.explain.build_explain_summary`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.gam.api import BSplines, GLMGam

from src.explain import (
    COUNTRY_MODEL_SPECS,
    TermFit,
    fit_country_model,
    morans_i,
    term_to_dict,
)
from src.interpolate import haversine_km

logger = logging.getLogger(__name__)

# The coefficient every spec in this module is built to compare.
EMISSIONS_TERM = "log10_emissions"

# A0/A1/A2 latitude-control comparison (section A). A0 reuses the existing
# linear spec so the three estimates sit on the same 157-country table; A1/A2
# replace ``mean_abs_lat`` with a B-spline smoother fit by ``GLMGam``.
LINEAR_LATITUDE_SPEC = COUNTRY_MODEL_SPECS["lat"]
GAM_LATITUDE_FORMULA = "trend_c_per_decade ~ log10_emissions"
GAM_LATITUDE_CONTINENT_FORMULA = "trend_c_per_decade ~ log10_emissions + C(continent)"
SMOOTH_COL = "mean_abs_lat"

# High-latitude, high-emission countries that sit at the leverage points where
# the latitude confound bites hardest (section C). Names follow the Berkeley /
# inequality-table convention; absent names are silently skipped.
HIGH_LATITUDE_LEVERAGE: tuple[str, ...] = (
    "Russia",
    "Canada",
    "Norway",
    "Sweden",
    "Finland",
    "Iceland",
    "Mongolia",
)


# ---------------------------------------------------------------------
# Shared extractor (RLM / GLMGam results don't go through
# ``_extract_all_terms``, which is shaped for the full OLS term list).
# ---------------------------------------------------------------------


def term_from_fit(fit, name: str = EMISSIONS_TERM) -> TermFit:
    """Extract one named term as a :class:`~src.explain.TermFit`.

    Works for OLS, WLS, RLM and GLMGam results: ``conf_int()`` is a
    DataFrame indexed by term name for formula-built models, but this also
    falls back to positional indexing if it comes back as a bare array.

    Args:
        fit: any fitted statsmodels results object with named ``params``.
        name: the term to extract (defaults to ``log10_emissions``).

    Returns:
        The term's coefficient, SE, 95% CI and p-value.
    """
    ci = fit.conf_int()
    if isinstance(ci, pd.DataFrame):
        lo = float(ci.loc[name].iloc[0])
        hi = float(ci.loc[name].iloc[1])
    else:
        pos = list(fit.params.index).index(name)
        lo = float(np.asarray(ci)[pos, 0])
        hi = float(np.asarray(ci)[pos, 1])
    return TermFit(
        term=name,
        coef=float(fit.params[name]),
        se=float(fit.bse[name]),
        ci_low=lo,
        ci_high=hi,
        p_value=float(fit.pvalues[name]),
    )


# ---------------------------------------------------------------------
# A. Nonlinear latitude control (GAM smoother)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ControlFit:
    """One latitude-control spec's emissions coefficient.

    ``kind`` is ``"ols"`` for the linear baseline (A0) or ``"gam"`` for the
    B-spline-smoothed specs (A1/A2). ``smooth_df`` is the spline basis size
    for GAM specs, ``None`` for OLS.
    """

    spec_name: str
    kind: str
    formula: str
    n: int
    emissions: TermFit
    smooth_df: int | None = None


def fit_country_gam(
    country_table: pd.DataFrame,
    spec_name: str,
    formula: str,
    *,
    smooth_col: str = SMOOTH_COL,
    df: int = 4,
    degree: int = 3,
) -> ControlFit:
    """Fit ``formula`` with a B-spline smoother on ``smooth_col`` via GLMGam.

    The parametric part (``formula``) carries ``log10_emissions`` and any
    fixed effects; ``smooth_col`` enters as a penalized B-spline instead of a
    linear term, so the latitude adjustment is free to be nonlinear.
    ``GLMGam.from_formula`` has no ``missing="drop"``, so complete cases are
    selected here over the parametric columns and the smooth column.

    Args:
        country_table: from :func:`src.explain.build_country_table`.
        spec_name: label for the returned :class:`ControlFit` (e.g. ``"A1_gam_lat"``).
        formula: parametric (linear) part, excluding the smooth term.
        smooth_col: column smoothed with a B-spline (default ``mean_abs_lat``).
        df: B-spline basis dimension.
        degree: B-spline degree.

    Returns:
        A :class:`ControlFit` with ``kind="gam"`` holding the
        ``log10_emissions`` coefficient under the smooth latitude control.
    """
    needed = [c for c in ("trend_c_per_decade", EMISSIONS_TERM, smooth_col) if c in country_table]
    if "continent" in formula:
        needed.append("continent")
    if "income_group" in formula:
        needed.append("income_group")
    data = country_table.dropna(subset=needed)

    x = data[[smooth_col]].to_numpy(dtype=float)
    smoother = BSplines(x, df=[df], degree=[degree])
    fit = GLMGam.from_formula(formula, data=data, smoother=smoother).fit()
    return ControlFit(
        spec_name=spec_name,
        kind="gam",
        formula=f"{formula} + s({smooth_col}, df={df}, degree={degree})",
        n=int(fit.nobs),
        emissions=term_from_fit(fit),
        smooth_df=df,
    )


def _latitude_control_sample(
    country_table: pd.DataFrame, *, smooth_col: str = SMOOTH_COL
) -> pd.DataFrame:
    """Complete-case rows over every column any of A0/A1/A2 touches.

    A2 carries ``C(continent)`` while A0/A1 do not, so fitting each spec on its
    own ``missing="drop"`` sample would silently put A2 on fewer rows than
    A0/A1 whenever a continent label is missing -- and the A0 -> A2 movement
    would then mix a control change with a sample change. Restricting all three
    to the same complete-case sample keeps the comparison clean.
    """
    needed = [
        c
        for c in ("trend_c_per_decade", EMISSIONS_TERM, smooth_col, "continent")
        if c in country_table.columns
    ]
    return country_table.dropna(subset=needed)


def compare_latitude_controls(
    country_table: pd.DataFrame, *, df: int = 4, degree: int = 3
) -> list[ControlFit]:
    """A0 vs A1 vs A2: how much does the emissions coefficient move?

    - **A0_linear_lat**: OLS, ``log10_emissions + mean_abs_lat`` (the Phase 7
      linear-control reference point), via :func:`src.explain.fit_country_model`.
    - **A1_gam_lat**: GAM, ``log10_emissions + s(mean_abs_lat)``.
    - **A2_gam_lat_continent**: GAM, ``log10_emissions + C(continent) + s(mean_abs_lat)``.

    All three are fit on the *same* complete-case sample (see
    :func:`_latitude_control_sample`), so the three ``emissions`` terms are
    directly comparable -- the A0 -> A2 movement reflects the control change,
    not a changing sample. A coefficient that is stable across A0 -> A1 -> A2
    is evidence the Phase 7 swing was linear-control misspecification; one that
    stays attenuated means the instability persists under a more flexible
    control. This returns the estimates, not a verdict. Note the GAM CIs are
    model-based GLM SEs (not spatial-dependence-robust); read precision
    alongside :func:`conley_hac_se`, not from these intervals alone.

    Args:
        country_table: from :func:`src.explain.build_country_table`.
        df, degree: B-spline basis dimension and degree for A1/A2.

    Returns:
        ``[A0, A1, A2]`` as :class:`ControlFit` objects, in that order.
    """
    data = _latitude_control_sample(country_table)
    a0_res = fit_country_model(data, {"A0_linear_lat": LINEAR_LATITUDE_SPEC})[0]
    a0 = ControlFit(
        spec_name="A0_linear_lat",
        kind="ols",
        formula=a0_res.formula,
        n=a0_res.n,
        emissions=next(t for t in a0_res.terms if t.term == EMISSIONS_TERM),
    )
    a1 = fit_country_gam(data, "A1_gam_lat", GAM_LATITUDE_FORMULA, df=df, degree=degree)
    a2 = fit_country_gam(
        data,
        "A2_gam_lat_continent",
        GAM_LATITUDE_CONTINENT_FORMULA,
        df=df,
        degree=degree,
    )
    return [a0, a1, a2]


def gam_latitude_df_sensitivity(
    country_table: pd.DataFrame,
    *,
    dfs: tuple[int, ...] = (4, 6, 8),
    degree: int = 3,
) -> list[ControlFit]:
    """Re-fit A1/A2 across spline basis sizes to expose df-dependence.

    The GAM uses an unpenalized B-spline (``GLMGam`` default ``alpha=0``), so
    the emissions estimate is conditional on the basis dimension ``df``. A
    conclusion that holds at ``df=4`` but not ``df=8`` is not a robustness
    result. This refits both ``s(mean_abs_lat)`` (A1) and
    ``C(continent) + s(mean_abs_lat)`` (A2) at each value in ``dfs`` on the
    shared complete-case sample, labelling each fit ``..._df{df}``.

    Args:
        country_table: from :func:`src.explain.build_country_table`.
        dfs: B-spline basis dimensions to sweep (default ``(4, 6, 8)``).
        degree: B-spline degree, held fixed across the sweep.

    Returns:
        ``ControlFit`` objects ``[A1_df4, A2_df4, A1_df6, A2_df6, ...]``.
    """
    data = _latitude_control_sample(country_table)
    out: list[ControlFit] = []
    for df in dfs:
        out.append(
            fit_country_gam(
                data, f"A1_gam_lat_df{df}", GAM_LATITUDE_FORMULA, df=df, degree=degree
            )
        )
        out.append(
            fit_country_gam(
                data,
                f"A2_gam_lat_continent_df{df}",
                GAM_LATITUDE_CONTINENT_FORMULA,
                df=df,
                degree=degree,
            )
        )
    return out


# ---------------------------------------------------------------------
# B. Country-level spatial inference
# ---------------------------------------------------------------------


def country_centroids(features: pd.DataFrame) -> pd.DataFrame:
    """Spherical (unit-vector) mean (lon, lat) per country.

    A plain arithmetic mean of longitude is wrong across the antimeridian: a
    country with cities near +179 and -179 averages to ~0 (the opposite side
    of the globe), which then corrupts the nearest-neighbor graph the spatial
    diagnostics build. This averages the cities' unit vectors on the sphere
    and projects back to (lon, lat), which is antimeridian-safe and the
    correct mean direction at any latitude. For a tightly clustered country it
    coincides with the arithmetic mean to rounding.

    Args:
        features: from :func:`src.explain.build_city_features`; needs
            ``Country``, ``Longitude``, ``Latitude``.

    Returns:
        One row per ``Country`` with ``cen_lon`` and ``cen_lat`` (degrees).
    """
    df = features[["Country", "Longitude", "Latitude"]].dropna(
        subset=["Longitude", "Latitude"]
    )
    lon = np.radians(df["Longitude"].to_numpy(dtype=float))
    lat = np.radians(df["Latitude"].to_numpy(dtype=float))
    xyz = pd.DataFrame(
        {
            "Country": df["Country"].to_numpy(),
            "x": np.cos(lat) * np.cos(lon),
            "y": np.cos(lat) * np.sin(lon),
            "z": np.sin(lat),
        }
    )
    means = xyz.groupby("Country", observed=True)[["x", "y", "z"]].mean().reset_index()
    cen_lon = np.degrees(np.arctan2(means["y"].to_numpy(), means["x"].to_numpy()))
    cen_lat = np.degrees(
        np.arctan2(means["z"].to_numpy(), np.hypot(means["x"].to_numpy(), means["y"].to_numpy()))
    )
    return pd.DataFrame(
        {"Country": means["Country"].to_numpy(), "cen_lon": cen_lon, "cen_lat": cen_lat}
    )


@dataclass(frozen=True)
class MoranResult:
    """Moran's I of one country spec's residuals over country centroids."""

    spec_name: str
    n: int
    k: int
    moran_i: float
    moran_p: float


def country_residual_morans_i(
    country_table: pd.DataFrame,
    centroids: pd.DataFrame,
    spec_name: str = "lat_continent",
    *,
    k: int = 8,
    n_permutations: int = 199,
    seed: int = 0,
) -> MoranResult:
    """Permutation Moran's I on a country spec's residuals.

    Fits ``COUNTRY_MODEL_SPECS[spec_name]`` (OLS), attaches country centroids,
    and feeds residuals + coordinates to the existing
    :func:`src.explain.morans_i`. A large positive I with a small p-value means
    the spec leaves spatially clustered residuals -- the diagnostic that
    justifies preferring the cluster-bootstrap CI over HC1.

    Args:
        country_table: from :func:`src.explain.build_country_table`.
        centroids: from :func:`country_centroids`.
        spec_name: key into :data:`src.explain.COUNTRY_MODEL_SPECS`.
        k: nearest-neighbor count for the spatial weights.
        n_permutations, seed: passed through to :func:`src.explain.morans_i`.

    Returns:
        The spec's residual Moran's I and permutation p-value.
    """
    formula = COUNTRY_MODEL_SPECS[spec_name]
    merged = country_table.merge(centroids, on="Country", how="left")
    fit = smf.ols(formula, data=merged, missing="drop").fit()

    used = merged.loc[fit.resid.index]
    has_xy = used[["cen_lon", "cen_lat"]].notna().all(axis=1)
    used = used.loc[has_xy]
    resid = fit.resid.loc[used.index].to_numpy()

    n = len(used)
    moran_i, moran_p = morans_i(
        resid,
        used["cen_lon"].to_numpy(),
        used["cen_lat"].to_numpy(),
        k=min(k, n - 1),
        n_permutations=n_permutations,
        seed=seed,
    )
    return MoranResult(
        spec_name=spec_name, n=n, k=min(k, n - 1), moran_i=moran_i, moran_p=moran_p
    )


# Cluster resampling needs many clusters for correct coverage; below this many
# the percentile cluster bootstrap is unreliable and its interval is reported
# as a sensitivity, not a CI. ~6 continents sit far below it.
MIN_CLUSTERS_FOR_VALID_BOOTSTRAP = 30


@dataclass(frozen=True)
class ClusterBootstrapResult:
    """Continent-cluster bootstrap interval vs HC1 for the emissions coefficient.

    A coarse between-continent *sensitivity*, not a valid CI: it absorbs only
    continent-block dependence, and with ~6 continents the cluster count is far
    below where the cluster bootstrap has correct coverage
    (:data:`MIN_CLUSTERS_FOR_VALID_BOOTSTRAP`). ``clusters_sufficient`` records
    whether that threshold is met. For spatial-dependence-robust inference use
    :func:`conley_hac_se` instead.
    """

    spec_name: str
    cluster_col: str
    n_clusters: int
    clusters_sufficient: bool
    n_boot: int
    n_effective: int
    point: float
    boot_se: float
    boot_ci_low: float
    boot_ci_high: float
    hc1: TermFit


def continent_cluster_bootstrap_se(
    country_table: pd.DataFrame,
    spec_name: str = "lat_continent",
    *,
    cluster_col: str = "continent",
    n_boot: int = 2000,
    seed: int = 0,
) -> ClusterBootstrapResult:
    """Cluster bootstrap over whole continents -- a coarse sensitivity, not a CI.

    Resamples ``cluster_col`` groups with replacement, refits the spec on each
    resample, and reports the percentile interval and bootstrap SE of the
    emissions coefficient alongside the HC1 CI from the full-sample fit.

    Inference caveat: with only ~6 continents the cluster count is far below
    where the percentile cluster bootstrap attains correct coverage
    (:data:`MIN_CLUSTERS_FOR_VALID_BOOTSTRAP`), and resampling whole continents
    only absorbs dependence that is constant *within* a continent, not the
    sub-continental, distance-decaying autocorrelation
    :func:`country_residual_morans_i` measures. The returned interval is a
    between-continent sensitivity check; ``clusters_sufficient`` is ``False``
    here. Use :func:`conley_hac_se` for the spatial-dependence-robust CI.

    Args:
        country_table: from :func:`src.explain.build_country_table`.
        spec_name: key into :data:`src.explain.COUNTRY_MODEL_SPECS`.
        cluster_col: the resampling block (default ``continent``).
        n_boot: number of bootstrap resamples.
        seed: RNG seed (deterministic).

    Returns:
        Point estimate, bootstrap SE / percentile CI, and the HC1 term.
    """
    formula = COUNTRY_MODEL_SPECS[spec_name]
    base = smf.ols(formula, data=country_table, missing="drop").fit(cov_type="HC1")
    hc1 = term_from_fit(base)

    groups = {
        key: sub for key, sub in country_table.groupby(cluster_col, observed=True)
    }
    clusters = np.array(list(groups))
    rng = np.random.default_rng(seed)

    coefs: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([groups[c] for c in pick], ignore_index=True)
        try:
            fit = smf.ols(formula, data=sample, missing="drop").fit()
        except Exception:  # pragma: no cover - degenerate resample
            continue
        if EMISSIONS_TERM in fit.params.index:
            coefs.append(float(fit.params[EMISSIONS_TERM]))

    arr = np.asarray(coefs, dtype=float)
    lo, hi = (float("nan"), float("nan"))
    if arr.size:
        lo, hi = (float(x) for x in np.percentile(arr, [2.5, 97.5]))
    return ClusterBootstrapResult(
        spec_name=spec_name,
        cluster_col=cluster_col,
        n_clusters=len(clusters),
        clusters_sufficient=len(clusters) >= MIN_CLUSTERS_FOR_VALID_BOOTSTRAP,
        n_boot=n_boot,
        n_effective=int(arr.size),
        point=float(base.params[EMISSIONS_TERM]),
        boot_se=float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        boot_ci_low=lo,
        boot_ci_high=hi,
        hc1=hc1,
    )


@dataclass(frozen=True)
class ConleyHACResult:
    """Conley spatial-HAC CI for the emissions coefficient.

    A distance-decay (Bartlett-kernel) sandwich SE on country centroids -- the
    spatial-dependence-robust interval the Moran's I diagnostic motivates,
    reported next to the naive HC1 interval.
    """

    spec_name: str
    n: int
    cutoff_km: float
    emissions: TermFit
    hc1: TermFit


def conley_hac_se(
    country_table: pd.DataFrame,
    centroids: pd.DataFrame,
    spec_name: str = "lat_continent",
    *,
    cutoff_km: float = 1000.0,
) -> ConleyHACResult:
    """Conley spatial-HAC standard error for ``log10_emissions``.

    Builds the OLS design matrix for ``spec_name``, attaches country centroids,
    and forms the spatial-HAC sandwich ``(X'X)^-1 (X' W X_weighted) (X'X)^-1``
    where the meat is ``sum_ij K(d_ij) e_i e_j x_i x_j'`` with ``K`` a Bartlett
    kernel ``max(0, 1 - d/cutoff)`` on great-circle distance ``d`` between
    centroids. Unlike :func:`continent_cluster_bootstrap_se` this models the
    distance-decaying covariance directly and does not depend on a coarse
    cluster partition, so it is the CI to read for spatial-dependence-robust
    inference. The normal-approximation CI (``coef +/- 1.96 se``) is returned
    alongside the HC1 interval for contrast.

    Args:
        country_table: from :func:`src.explain.build_country_table`.
        centroids: from :func:`country_centroids`.
        spec_name: key into :data:`src.explain.COUNTRY_MODEL_SPECS`.
        cutoff_km: Bartlett-kernel bandwidth in km; pairs farther apart get
            zero weight. 1000 km is a continental-neighborhood default.

    Returns:
        The emissions term under the Conley HAC SE and under HC1.
    """
    formula = COUNTRY_MODEL_SPECS[spec_name]
    merged = country_table.merge(centroids, on="Country", how="left")
    probe = smf.ols(formula, data=merged, missing="drop").fit()
    used = merged.loc[probe.model.data.row_labels]
    has_xy = used[["cen_lon", "cen_lat"]].notna().all(axis=1)
    used = used.loc[has_xy]

    # HC1 cov leaves params/resid/exog unchanged, so one fit serves both the
    # HC1 contrast term and the Conley sandwich inputs.
    fit = smf.ols(formula, data=used).fit(cov_type="HC1")
    hc1 = term_from_fit(fit)

    x = np.asarray(fit.model.exog, dtype=float)
    resid = np.asarray(fit.resid, dtype=float)
    names = list(fit.model.exog_names)
    j = names.index(EMISSIONS_TERM)

    lon = used["cen_lon"].to_numpy(dtype=float)
    lat = used["cen_lat"].to_numpy(dtype=float)
    dist = haversine_km(lon[:, None], lat[:, None], lon[None, :], lat[None, :])
    kernel = np.maximum(0.0, 1.0 - dist / cutoff_km)  # Bartlett, self-weight 1

    u = x * resid[:, None]  # score contributions, (n, p)
    meat = u.T @ kernel @ u  # (p, p)
    bread = np.linalg.inv(x.T @ x)
    cov = bread @ meat @ bread
    se = float(np.sqrt(cov[j, j]))

    coef = float(fit.params.iloc[j])
    z = 1.959963984540054
    conley = TermFit(
        term=EMISSIONS_TERM,
        coef=coef,
        se=se,
        ci_low=coef - z * se,
        ci_high=coef + z * se,
        p_value=float(2.0 * (1.0 - _std_normal_cdf(abs(coef) / se))) if se > 0 else float("nan"),
    )
    return ConleyHACResult(
        spec_name=spec_name,
        n=int(fit.nobs),
        cutoff_km=cutoff_km,
        emissions=conley,
        hc1=hc1,
    )


def _std_normal_cdf(value: float) -> float:
    """Standard-normal CDF via ``erf`` (no scipy import for one p-value)."""
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


# ---------------------------------------------------------------------
# C. Estimator plurality + leverage / weighting sensitivity
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class EstimatorComparison:
    """OLS vs Huber-RLM emissions coefficient for one spec."""

    spec_name: str
    n: int
    ols: TermFit
    rlm: TermFit


def fit_country_robust(
    country_table: pd.DataFrame, spec_name: str = "lat_continent"
) -> EstimatorComparison:
    """Compare HC1-OLS and Huber-T RLM on the same spec.

    This is a *vertical-outlier* (response-direction) robustness check: Huber-T
    M-estimation down-weights observations with large residuals, so a large
    OLS-vs-RLM gap in the emissions coefficient means a few off-trend countries
    are pulling the OLS estimate. It is **not** a leverage check -- M-estimators
    have unbounded influence in the design (x) direction and barely move a
    high-leverage point whose residual stays small. Use
    :func:`influence_diagnostics` for the leverage / design-matrix question.

    Args:
        country_table: from :func:`src.explain.build_country_table`.
        spec_name: key into :data:`src.explain.COUNTRY_MODEL_SPECS`.

    Returns:
        The emissions term under each estimator.
    """
    formula = COUNTRY_MODEL_SPECS[spec_name]
    ols = smf.ols(formula, data=country_table, missing="drop").fit(cov_type="HC1")
    rlm = smf.rlm(
        formula, data=country_table, missing="drop", M=sm.robust.norms.HuberT()
    ).fit()
    return EstimatorComparison(
        spec_name=spec_name,
        n=int(ols.nobs),
        ols=term_from_fit(ols),
        rlm=term_from_fit(rlm),
    )


@dataclass(frozen=True)
class InfluenceResult:
    """Design-matrix influence on the emissions coefficient.

    DFBETA on ``log10_emissions`` and Cook's distance from the fitted spec
    answer the leverage question Huber RLM cannot. ``drop_top_dfbeta_coef`` is
    the spec refit after dropping the ``k`` countries with the largest
    ``|DFBETA|`` -- the empirically-most-influential points, not a hand-picked
    list.
    """

    spec_name: str
    n: int
    k: int
    full_coef: float
    top_dfbeta: tuple[tuple[str, float], ...]
    top_cooks: tuple[tuple[str, float], ...]
    drop_top_dfbeta: tuple[str, ...]
    drop_top_dfbeta_coef: float | None


def influence_diagnostics(
    country_table: pd.DataFrame,
    spec_name: str = "lat_continent",
    *,
    k: int = 5,
    country_col: str = "Country",
) -> InfluenceResult:
    """Rank countries by influence on the emissions coefficient.

    Computes DFBETA for ``log10_emissions`` (the per-observation shift in that
    coefficient when the observation is deleted) and Cook's distance from the
    fitted ``spec_name`` OLS, both straight off ``OLSInfluence``. Reports the
    ``k`` countries with the largest ``|DFBETA|`` and the largest Cook's D, then
    refits dropping the top-``k`` by ``|DFBETA|`` so the headline can be read
    against its empirically-most-influential points -- the design-matrix
    analogue of, and complement to, the named drop-set in
    :func:`jackknife_emissions_coef`.

    Args:
        country_table: from :func:`src.explain.build_country_table`.
        spec_name: key into :data:`src.explain.COUNTRY_MODEL_SPECS`.
        k: how many top-influence countries to report and drop.
        country_col: country identifier column.

    Returns:
        Top DFBETA / Cook's-D countries and the top-``k``-dropped refit.
    """
    formula = COUNTRY_MODEL_SPECS[spec_name]
    fit = smf.ols(formula, data=country_table, missing="drop").fit()
    used = country_table.loc[fit.model.data.row_labels]
    full_coef = float(fit.params[EMISSIONS_TERM])

    names = list(fit.model.exog_names)
    j = names.index(EMISSIONS_TERM)
    infl = fit.get_influence()
    dfbeta = np.asarray(infl.dfbetas)[:, j]
    cooks = np.asarray(infl.cooks_distance[0])
    countries = used[country_col].astype(str).to_numpy()

    kk = int(min(k, len(countries)))
    df_order = np.argsort(np.abs(dfbeta))[::-1][:kk]
    ck_order = np.argsort(cooks)[::-1][:kk]
    top_dfbeta = tuple((str(countries[i]), float(dfbeta[i])) for i in df_order)
    top_cooks = tuple((str(countries[i]), float(cooks[i])) for i in ck_order)

    drop = tuple(str(countries[i]) for i in df_order)
    sub = used[~used[country_col].astype(str).isin(set(drop))]
    refit = smf.ols(formula, data=sub, missing="drop").fit()
    drop_coef = (
        float(refit.params[EMISSIONS_TERM])
        if EMISSIONS_TERM in refit.params.index
        else None
    )

    return InfluenceResult(
        spec_name=spec_name,
        n=int(fit.nobs),
        k=kk,
        full_coef=full_coef,
        top_dfbeta=top_dfbeta,
        top_cooks=top_cooks,
        drop_top_dfbeta=drop,
        drop_top_dfbeta_coef=drop_coef,
    )


@dataclass(frozen=True)
class JackknifeResult:
    """Leave-one-country-out range and a named-leverage drop-set refit."""

    spec_name: str
    full_coef: float
    loo_min: float
    loo_max: float
    loo_range: float
    most_influential_country: str
    most_influential_coef: float
    drop_set: tuple[str, ...]
    drop_set_present: tuple[str, ...]
    drop_set_coef: float | None


def jackknife_emissions_coef(
    country_table: pd.DataFrame,
    spec_name: str = "lat_continent",
    *,
    drop_set: tuple[str, ...] = HIGH_LATITUDE_LEVERAGE,
    country_col: str = "Country",
) -> JackknifeResult:
    """Leave-one-country-out jackknife plus a high-latitude drop-set refit.

    Refits the spec dropping each country in turn (reporting the min/max
    emissions coefficient and the single most influential country), then once
    more dropping every present member of ``drop_set`` together. A wide LOO
    range or a large drop-set shift means the headline rides on a few points.

    Args:
        country_table: from :func:`src.explain.build_country_table`.
        spec_name: key into :data:`src.explain.COUNTRY_MODEL_SPECS`.
        drop_set: named high-latitude leverage countries to drop together;
            members absent from the table are skipped.
        country_col: country identifier column.

    Returns:
        Full-sample coefficient, LOO range, most influential country, and the
        drop-set refit coefficient (``None`` if no member is present).
    """
    formula = COUNTRY_MODEL_SPECS[spec_name]
    full = smf.ols(formula, data=country_table, missing="drop").fit()
    full_coef = float(full.params[EMISSIONS_TERM])

    loo: dict[str, float] = {}
    for country in country_table[country_col].dropna().unique():
        sub = country_table[country_table[country_col] != country]
        fit = smf.ols(formula, data=sub, missing="drop").fit()
        if EMISSIONS_TERM in fit.params.index:
            loo[str(country)] = float(fit.params[EMISSIONS_TERM])

    coefs = np.array(list(loo.values()))
    most = max(loo, key=lambda c: abs(loo[c] - full_coef))

    present = tuple(
        c for c in drop_set if c in set(country_table[country_col].astype(str))
    )
    drop_set_coef: float | None = None
    if present:
        sub = country_table[~country_table[country_col].astype(str).isin(present)]
        fit = smf.ols(formula, data=sub, missing="drop").fit()
        if EMISSIONS_TERM in fit.params.index:
            drop_set_coef = float(fit.params[EMISSIONS_TERM])

    return JackknifeResult(
        spec_name=spec_name,
        full_coef=full_coef,
        loo_min=float(coefs.min()),
        loo_max=float(coefs.max()),
        loo_range=float(coefs.max() - coefs.min()),
        most_influential_country=most,
        most_influential_coef=loo[most],
        drop_set=tuple(drop_set),
        drop_set_present=present,
        drop_set_coef=drop_set_coef,
    )


@dataclass(frozen=True)
class WeightedFitResult:
    """Emissions coefficient under a re-weighted country aggregation."""

    spec_name: str
    weight_col: str
    n: int
    emissions: TermFit


def weighted_inequality_fit(
    country_table: pd.DataFrame,
    weight_col: str = "n_cities",
    spec_name: str = "lat_continent",
) -> WeightedFitResult:
    """WLS sensitivity for the documented unweighted-aggregation limitation.

    Country means are unweighted across city locations upstream. This refits
    the spec weighting each country by ``weight_col`` (e.g. ``n_cities``, or
    any precomputed area weight passed in as a column) with HC1 SEs. Complete
    cases are taken from an OLS probe so the explicit weight vector stays
    aligned with the fitted rows.

    Args:
        country_table: from :func:`src.explain.build_country_table`.
        weight_col: column of non-negative weights.
        spec_name: key into :data:`src.explain.COUNTRY_MODEL_SPECS`.

    Returns:
        The emissions term under the weighted fit.
    """
    formula = COUNTRY_MODEL_SPECS[spec_name]
    candidates = country_table.dropna(subset=[weight_col])
    probe = smf.ols(formula, data=candidates, missing="drop").fit()
    used = candidates.loc[probe.model.data.row_labels]
    fit = smf.wls(
        formula, data=used, weights=used[weight_col].astype(float)
    ).fit(cov_type="HC1")
    return WeightedFitResult(
        spec_name=spec_name,
        weight_col=weight_col,
        n=int(fit.nobs),
        emissions=term_from_fit(fit),
    )


# ---------------------------------------------------------------------
# Orchestrator + serialization
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessSummary:
    """Bundle of the A/B/C robustness checks for one country table."""

    spec_name: str
    latitude_controls: list[ControlFit]
    gam_df_sensitivity: list[ControlFit]
    residual_moran: list[MoranResult]
    conley: ConleyHACResult
    cluster_bootstrap: ClusterBootstrapResult
    estimator: EstimatorComparison
    influence: InfluenceResult
    jackknife: JackknifeResult
    weighted: list[WeightedFitResult]


def run_robustness_suite(
    country_table: pd.DataFrame,
    features: pd.DataFrame,
    *,
    spec_name: str = "lat_continent",
    moran_specs: tuple[str, ...] = ("lat", "lat_continent"),
    weight_cols: tuple[str, ...] = ("n_cities",),
    n_boot: int = 2000,
    seed: int = 0,
) -> RobustnessSummary:
    """Run upgrades A, B and C and bundle the results.

    Args:
        country_table: from :func:`src.explain.build_country_table`.
        features: from :func:`src.explain.build_city_features` (for centroids).
        spec_name: the headline spec the B/C checks center on.
        moran_specs: specs to run residual Moran's I on.
        weight_cols: weight columns to run :func:`weighted_inequality_fit` on;
            columns absent from ``country_table`` are skipped.
        n_boot, seed: passed to :func:`continent_cluster_bootstrap_se`.

    Returns:
        A :class:`RobustnessSummary`.
    """
    centroids = country_centroids(features)
    return RobustnessSummary(
        spec_name=spec_name,
        latitude_controls=compare_latitude_controls(country_table),
        gam_df_sensitivity=gam_latitude_df_sensitivity(country_table),
        residual_moran=[
            country_residual_morans_i(country_table, centroids, s) for s in moran_specs
        ],
        conley=conley_hac_se(country_table, centroids, spec_name),
        cluster_bootstrap=continent_cluster_bootstrap_se(
            country_table, spec_name, n_boot=n_boot, seed=seed
        ),
        estimator=fit_country_robust(country_table, spec_name),
        influence=influence_diagnostics(country_table, spec_name),
        jackknife=jackknife_emissions_coef(country_table, spec_name),
        weighted=[
            weighted_inequality_fit(country_table, w, spec_name)
            for w in weight_cols
            if w in country_table.columns
        ],
    )


def _control_to_dict(c: ControlFit) -> dict:
    return {
        "spec_name": c.spec_name,
        "kind": c.kind,
        "formula": c.formula,
        "n": c.n,
        "smooth_df": c.smooth_df,
        "emissions": term_to_dict(c.emissions),
    }


def robustness_summary_to_dict(summary: RobustnessSummary) -> dict:
    """Serialize a :class:`RobustnessSummary` to a JSON-safe dict.

    Mirrors :func:`src.explain.country_result_to_dict`: every emissions term is
    pre-shaped via :func:`src.explain.term_to_dict` so a render layer never has
    to hunt through nested objects.
    """
    cb = summary.cluster_bootstrap
    jk = summary.jackknife
    cn = summary.conley
    inf = summary.influence
    return {
        "spec_name": summary.spec_name,
        "latitude_controls": [_control_to_dict(c) for c in summary.latitude_controls],
        "gam_df_sensitivity": [_control_to_dict(c) for c in summary.gam_df_sensitivity],
        "residual_moran": [
            {
                "spec_name": m.spec_name,
                "n": m.n,
                "k": m.k,
                "moran_i": m.moran_i,
                "moran_p": m.moran_p,
            }
            for m in summary.residual_moran
        ],
        "conley": {
            "spec_name": cn.spec_name,
            "n": cn.n,
            "cutoff_km": cn.cutoff_km,
            "emissions": term_to_dict(cn.emissions),
            "hc1": term_to_dict(cn.hc1),
        },
        "cluster_bootstrap": {
            "spec_name": cb.spec_name,
            "cluster_col": cb.cluster_col,
            "n_clusters": cb.n_clusters,
            "clusters_sufficient": cb.clusters_sufficient,
            "n_boot": cb.n_boot,
            "n_effective": cb.n_effective,
            "point": cb.point,
            "boot_se": cb.boot_se,
            "boot_ci_low": cb.boot_ci_low,
            "boot_ci_high": cb.boot_ci_high,
            "hc1": term_to_dict(cb.hc1),
        },
        "estimator": {
            "spec_name": summary.estimator.spec_name,
            "n": summary.estimator.n,
            "ols": term_to_dict(summary.estimator.ols),
            "rlm": term_to_dict(summary.estimator.rlm),
        },
        "influence": {
            "spec_name": inf.spec_name,
            "n": inf.n,
            "k": inf.k,
            "full_coef": inf.full_coef,
            "top_dfbeta": [[c, v] for c, v in inf.top_dfbeta],
            "top_cooks": [[c, v] for c, v in inf.top_cooks],
            "drop_top_dfbeta": list(inf.drop_top_dfbeta),
            "drop_top_dfbeta_coef": inf.drop_top_dfbeta_coef,
        },
        "jackknife": {
            "spec_name": jk.spec_name,
            "full_coef": jk.full_coef,
            "loo_min": jk.loo_min,
            "loo_max": jk.loo_max,
            "loo_range": jk.loo_range,
            "most_influential_country": jk.most_influential_country,
            "most_influential_coef": jk.most_influential_coef,
            "drop_set": list(jk.drop_set),
            "drop_set_present": list(jk.drop_set_present),
            "drop_set_coef": jk.drop_set_coef,
        },
        "weighted": [
            {
                "spec_name": w.spec_name,
                "weight_col": w.weight_col,
                "n": w.n,
                "emissions": term_to_dict(w.emissions),
            }
            for w in summary.weighted
        ],
    }
