"""Perturbation stability of the group-level Shapley shares (descriptive).

Stress-tests the *existing* LMG/Shapley decomposition
(:func:`src.decomposition.group_lmg_shares`) under country resampling and
single-country deletion, and probes the spatial structure of the model residual.
It adds **no new estimator**: every number here is the same decomposition
recomputed on a perturbed sample, so the layer can only ever speak to the
**stability of the variance shares**, never to significance or causation. Like the
decomposition itself, this is a *descriptive variance attribution of observed
warming trends*, not climate attribution; the canonical disclaimer
(:data:`src.feature_schema.INTERPRETATION_NOTE`) travels on the summary output.

Three robustness dimensions, mapped to the ``stability_summary.json`` blocks the
sensitivity page (``app/views/sensitivity.py``) renders:

* ``share_stability`` -- a nonparametric **country bootstrap** (resample the
  countries with replacement, recompute the full decomposition) gives a 95%
  percentile interval, mean and std for every group share plus the residual, the
  probability geography stays the largest named axis, and the probability the
  emissions share stays positive. A **continent block bootstrap** (resample whole
  ``spatial_block`` groups) gives a spatially-honest interval alongside it; the
  gap between the two *is* the spatial-dependence correction for the shares.
* ``influence`` -- **leave-one-country-out**: how far each group share moves when
  a single country is dropped, with the ten most influential countries per share.
* ``residual_spatial`` -- **Moran's I** on the full-model country residuals
  (centroid k-nearest-neighbor weights), reusing :func:`src.explain.morans_i`.

Contract: the input is the schema-named design from
:func:`src.decomposition.build_country_design`, so the layer never sees a feature
outside :data:`SCHEMA_V1`; the schema is perturbed in *sampling*, never in
membership. See ``docs/stability_roadmap.md`` for the full blueprint.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src import feature_schema as fs
from src.data_io import PROCESSED_DIR, round_floats
from src.decomposition import (
    COUNTRY_COL,
    OUTCOME_COL,
    DecompositionResult,
    _feature_block,
    build_country_design,
    group_lmg_shares,
)
from src.explain import morans_i
from src.feature_schema import INTERPRETATION_NOTE, SCHEMA_V1, FeatureSchema

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_PATH = PROCESSED_DIR / "stability_summary.json"
SPATIAL_BLOCK_COL = "spatial_block"
RESIDUAL_KEY = "residual"

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 0
DEFAULT_K_NEIGHBORS = 8
DEFAULT_N_PERMUTATIONS = 199
TOP_INFLUENCE = 10

# Percentile bounds for the 95% bootstrap interval.
_CI_LOW_PCT = 2.5
_CI_HIGH_PCT = 97.5


# ---------------------------------------------------------------------
# Shared selection helpers (mirror group_lmg_shares so residuals/refits
# see exactly the features and complete-case rows the decomposition uses)
# ---------------------------------------------------------------------


def _used_features(
    columns: pd.Index, schema: FeatureSchema, status: str
) -> dict[str, list[str]]:
    """Available schema features present per named group (residual excluded).

    Same selection :func:`group_lmg_shares` performs, so a refit here consumes
    exactly the decomposition's design.
    """
    used: dict[str, list[str]] = {}
    for group in schema.groups:
        if group.key == RESIDUAL_KEY:
            continue
        present = [
            f.name for f in group.features if f.status == status and f.name in columns
        ]
        if present:
            used[group.key] = present
    return used


def _complete_case(
    design: pd.DataFrame, schema: FeatureSchema, status: str
) -> pd.DataFrame:
    """Restrict to complete cases over the used features and the outcome."""
    used = _used_features(design.columns, schema, status)
    all_features = [name for names in used.values() for name in names]
    return design.dropna(subset=[OUTCOME_COL, *all_features]).reset_index(drop=True)


def _full_model_residuals(
    design: pd.DataFrame, schema: FeatureSchema, status: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-country residuals of the full (all-available-features) OLS.

    Builds the identical design blocks :func:`group_lmg_shares` uses for the full
    coalition (so the residual is consistent with ``total_r2``) and returns
    ``(countries, residuals)`` over the complete-case rows.
    """
    used = _used_features(design.columns, schema, status)
    all_features = [name for names in used.values() for name in names]
    complete = _complete_case(design, schema, status)
    y = complete[OUTCOME_COL].to_numpy(dtype=float)
    blocks = [_feature_block(complete, name)[0] for name in all_features]
    intercept = np.ones((len(y), 1))
    matrix = np.hstack([intercept, *blocks]) if blocks else intercept
    coef, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    residuals = y - matrix @ coef
    return complete[COUNTRY_COL].to_numpy(), residuals


# ---------------------------------------------------------------------
# Bootstrap resampling of the country units
# ---------------------------------------------------------------------


def _resample_countries(design: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample the country rows with replacement (i.i.d.-country bootstrap)."""
    idx = rng.integers(0, len(design), len(design))
    return design.iloc[idx].reset_index(drop=True)


def _resample_blocks(design: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample whole ``spatial_block`` groups with replacement (block bootstrap).

    Keeps each continent's countries together so within-block spatial correlation
    is respected; the resulting intervals are wider (spatially honest).
    """
    if SPATIAL_BLOCK_COL not in design.columns:
        return _resample_countries(design, rng)
    blocks = pd.unique(design[SPATIAL_BLOCK_COL].dropna())
    if len(blocks) == 0:
        return _resample_countries(design, rng)
    chosen = rng.choice(blocks, size=len(blocks), replace=True)
    parts = [design[design[SPATIAL_BLOCK_COL] == b] for b in chosen]
    return pd.concat(parts, ignore_index=True)


def _bootstrap_shares(
    design: pd.DataFrame,
    schema: FeatureSchema,
    status: str,
    *,
    n_boot: int,
    rng: np.random.Generator,
    block: bool,
) -> tuple[list[dict[str, float]], int]:
    """Recompute the decomposition on `n_boot` resamples; collect share vectors.

    A resample that degenerates (e.g. drops a categorical level so the model is
    unusable) is skipped and counted, never silently substituted.
    """
    resample = _resample_blocks if block else _resample_countries
    rows: list[dict[str, float]] = []
    n_failed = 0
    for _ in range(n_boot):
        sample = resample(design, rng)
        try:
            result = group_lmg_shares(sample, schema, status=status)
        except (ValueError, np.linalg.LinAlgError) as exc:
            n_failed += 1
            logger.debug("skipped %s bootstrap draw: %s", "block" if block else "country", exc)
            continue
        rows.append({**result.shares, RESIDUAL_KEY: result.residual_share})
    return rows, n_failed


def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    """95% percentile interval ``(2.5, 97.5)`` as plain floats."""
    lo, hi = np.percentile(values, [_CI_LOW_PCT, _CI_HIGH_PCT])
    return float(lo), float(hi)


def _share_stability(
    point: DecompositionResult,
    boot: tuple[list[dict[str, float]], int],
    block: tuple[list[dict[str, float]], int],
    *,
    n_boot: int,
) -> dict:
    """Assemble the ``share_stability`` block from the two bootstrap runs."""
    boot_rows, n_failed = boot
    block_rows, block_failed = block
    named = list(point.shares)  # used named axes, in schema order
    all_keys = [*named, RESIDUAL_KEY]
    point_map = {**point.shares, RESIDUAL_KEY: point.residual_share}

    def column(rows: list[dict[str, float]], key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=float)

    groups: dict[str, dict] = {}
    for key in all_keys:
        vals = column(boot_rows, key)
        lo, hi = _percentile_ci(vals)
        groups[key] = {
            "point": float(point_map[key]),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "ci_low": lo,
            "ci_high": hi,
        }

    # P(geography is the largest *named axis*) -- residual excluded, since it is
    # the unexplained remainder, not an axis. Directly tests acceptance C1.
    n_ok = len(boot_rows)
    p_geo_largest = None
    if "geography" in named and n_ok:
        wins = sum(max(named, key=lambda k, r=r: r[k]) == "geography" for r in boot_rows)
        p_geo_largest = wins / n_ok
        groups["geography"]["p_largest"] = float(p_geo_largest)

    # P(emissions share > 0) -- tests acceptance C2 (small but positive).
    p_em_positive = None
    if "emissions" in named and n_ok:
        p_em_positive = float((column(boot_rows, "emissions") > 0.0).mean())

    block_groups: dict[str, dict] = {}
    for key in all_keys:
        vals = column(block_rows, key)
        if len(vals):
            lo, hi = _percentile_ci(vals)
            block_groups[key] = {"ci_low": lo, "ci_high": hi}

    return {
        "method": "country_bootstrap",
        "n_boot": n_boot,
        "n_failed": n_failed,
        "groups": groups,
        "p_geography_largest": (
            float(p_geo_largest) if p_geo_largest is not None else None
        ),
        "p_emissions_positive": p_em_positive,
        "block_bootstrap": {
            "by": SPATIAL_BLOCK_COL,
            "n_boot": n_boot,
            "n_failed": block_failed,
            "groups": block_groups,
        },
    }


def _leave_one_out_influence(
    design: pd.DataFrame,
    schema: FeatureSchema,
    status: str,
    point: DecompositionResult,
) -> dict:
    """Top-`TOP_INFLUENCE` countries by leave-one-out movement of each share.

    Drops each country in turn, recomputes the shares, and records
    ``delta = share_full - share_without_country`` per named group; the largest
    ``|delta|`` countries are the most influential. A share that rides on a few
    countries is fragile.
    """
    complete = _complete_case(design, schema, status)
    named = list(point.shares)
    deltas: dict[str, list[tuple[str, float]]] = {g: [] for g in named}
    for i in range(len(complete)):
        country = str(complete.at[i, COUNTRY_COL])
        loo = complete.drop(index=i)
        try:
            result = group_lmg_shares(loo, schema, status=status)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("skipped leave-one-out for %s: %s", country, exc)
            continue
        for group in named:
            delta = point.shares[group] - result.shares.get(group, 0.0)
            deltas[group].append((country, float(delta)))

    by_group: dict[str, list[list]] = {}
    for group in named:
        ranked = sorted(deltas[group], key=lambda t: abs(t[1]), reverse=True)
        by_group[group] = [[c, d] for c, d in ranked[:TOP_INFLUENCE]]
    return {"method": "leave_one_country_out", "by_group": by_group}


def _residual_spatial(
    design: pd.DataFrame,
    centroids: pd.DataFrame,
    schema: FeatureSchema,
    status: str,
    *,
    k: int,
    n_permutations: int,
    seed: int,
) -> dict:
    """Moran's I of the full-model residuals on a country-centroid kNN weight."""
    countries, residuals = _full_model_residuals(design, schema, status)
    cen = centroids.dropna(subset=["Longitude", "Latitude"]).drop_duplicates(COUNTRY_COL)
    cen = cen.set_index(COUNTRY_COL)

    lon, lat, resid = [], [], []
    for country, value in zip(countries, residuals):
        if country in cen.index:
            lon.append(float(cen.at[country, "Longitude"]))
            lat.append(float(cen.at[country, "Latitude"]))
            resid.append(float(value))

    base = {
        "n_permutations": n_permutations,
        "k_neighbors": k,
        "method": "centroid kNN",
        "n": len(resid),
    }
    if len(resid) < k + 1:
        logger.warning(
            "only %d countries have centroids; need > k=%d for Moran's I",
            len(resid), k,
        )
        return {"morans_i": None, "p_value": None, **base}
    morans, p_value = morans_i(
        np.array(resid), np.array(lon), np.array(lat),
        k=k, n_permutations=n_permutations, seed=seed,
    )
    return {"morans_i": float(morans), "p_value": float(p_value), **base}


# ---------------------------------------------------------------------
# Result container + public entry points
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityResult:
    """Perturbation-stability summary of the group-level Shapley shares."""

    schema_version: str
    n_countries: int
    n_boot: int
    seed: int
    share_stability: dict = field(default_factory=dict)
    influence: dict = field(default_factory=dict)
    residual_spatial: dict = field(default_factory=dict)

    def check(self, atol: float = 1e-6) -> None:
        """Assert the bootstrap-mean shares + residual sum to 1 and CIs order."""
        groups = self.share_stability.get("groups", {})
        mean_total = sum(g["mean"] for g in groups.values())
        if abs(mean_total - 1.0) > atol:
            raise AssertionError(
                f"bootstrap-mean shares sum to {mean_total!r}, expected 1.0"
            )
        for key, g in groups.items():
            if g["ci_low"] > g["ci_high"] + atol:
                raise AssertionError(
                    f"{key}: ci_low {g['ci_low']!r} > ci_high {g['ci_high']!r}"
                )


def build_stability_summary(
    design: pd.DataFrame,
    centroids: pd.DataFrame,
    schema: FeatureSchema = SCHEMA_V1,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    k: int = DEFAULT_K_NEIGHBORS,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    status: str = fs.STATUS_AVAILABLE,
) -> StabilityResult:
    """Perturbation stability of the decomposition's group shares.

    Args:
        design: schema-named country design from
            :func:`src.decomposition.build_country_design` (one row per country).
        centroids: ``Country`` -> ``Longitude``/``Latitude`` table (country
            centroids; aggregate the city features to get them) used only for the
            residual Moran's I weight.
        schema: the feature contract (defaults to :data:`SCHEMA_V1`); membership
            is never altered, only the sample is perturbed.
        n_boot: bootstrap resamples for each of the country and block runs.
        seed: base RNG seed; the block bootstrap uses ``seed + 1`` and Moran's I
            uses `seed`, so the whole summary is reproducible.
        k: nearest neighbors for the residual spatial weight.
        n_permutations: label permutations for the Moran's I p-value.
        status: feature status admitted into the design (``"available"``).

    Returns:
        A :class:`StabilityResult` whose ``share_stability`` / ``influence`` /
        ``residual_spatial`` blocks the sensitivity page renders.
    """
    point = group_lmg_shares(design, schema, status=status)

    boot = _bootstrap_shares(
        design, schema, status,
        n_boot=n_boot, rng=np.random.default_rng(seed), block=False,
    )
    block = _bootstrap_shares(
        design, schema, status,
        n_boot=n_boot, rng=np.random.default_rng(seed + 1), block=True,
    )

    result = StabilityResult(
        schema_version=schema.version,
        n_countries=point.n,
        n_boot=n_boot,
        seed=seed,
        share_stability=_share_stability(point, boot, block, n_boot=n_boot),
        influence=_leave_one_out_influence(design, schema, status, point),
        residual_spatial=_residual_spatial(
            design, centroids, schema, status,
            k=k, n_permutations=n_permutations, seed=seed,
        ),
    )
    result.check()
    return result


def result_to_dict(result: StabilityResult) -> dict:
    """JSON-safe dict for :class:`StabilityResult`."""
    return asdict(result)


def summary_payload(result: StabilityResult) -> dict:
    """Serialized result with the scope disclaimer attached.

    The ``interpretation`` key carries
    :data:`src.feature_schema.INTERPRETATION_NOTE` so the stability numbers cannot
    be lifted without the variance-attribution-only boundary; floats are rounded
    (:func:`src.data_io.round_floats`) for a byte-stable committed summary.
    """
    return round_floats({"interpretation": INTERPRETATION_NOTE, **result_to_dict(result)})


def main() -> None:
    """Build the design + centroids from real artifacts and write the summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from src.emissions import DEFAULT_INEQUALITY_PATH
    from src.explain import DEFAULT_FEATURES_PATH, INCOME_PATH, load_income_groups

    inequality = pd.read_parquet(DEFAULT_INEQUALITY_PATH)
    city_features = pd.read_parquet(DEFAULT_FEATURES_PATH)
    income = load_income_groups(INCOME_PATH)

    design = build_country_design(inequality, city_features, income)
    centroids = (
        city_features.groupby(COUNTRY_COL, as_index=False)[["Longitude", "Latitude"]]
        .mean()
    )

    result = build_stability_summary(design, centroids)

    share = result.share_stability
    print(
        f"Share stability (country bootstrap B={result.n_boot}, n={result.n_countries}, "
        "descriptive/non-causal)"
    )
    print(f"  {'group':<14s} {'point':>7s} {'2.5%':>7s} {'97.5%':>7s}")
    for key, g in share["groups"].items():
        print(f"  {key:<14s} {g['point']:7.3f} {g['ci_low']:7.3f} {g['ci_high']:7.3f}")
    print(f"  P(geography largest axis): {share['p_geography_largest']}")
    print(f"  P(emissions share > 0):    {share['p_emissions_positive']}")
    rs = result.residual_spatial
    print(f"  residual Moran's I: {rs['morans_i']} (p={rs['p_value']}, n={rs['n']})")
    print(f"  note: {INTERPRETATION_NOTE}")

    DEFAULT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SUMMARY_PATH.write_text(
        json.dumps(summary_payload(result), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {DEFAULT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
