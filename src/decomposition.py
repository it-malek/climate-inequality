"""Group-level LMG/Shapley variance decomposition of country warming (descriptive).

Attributes the cross-country variance in mean warming trend to the *structural
axes* of the frozen feature contract (:data:`src.feature_schema.SCHEMA_V1`):
emissions, geography, socioeconomic, population -- plus an explicit residual.
See ``docs/decomposition_design_memo.md``.

The method is LMG (a.k.a. Shapley-Owen R^2 attribution) at the **group** level:
each axis's share is its incremental R^2 averaged over every ordering in which
the axes could enter the model. Group shares plus the residual (``1 - R^2``) sum
to 1. Because correlated axes (emissions and geography are strongly correlated)
have their *shared* explained variance split evenly across orderings rather than
handed to whichever enters first, a Shapley share is stable where a single
regression coefficient is fragile.

**These shares are descriptive, not causal** -- a structural decomposition of
*observed* warming trends, not climate attribution (the project has no
physical/driver layer, by an explicit architectural decision; see the memo's
scope boundary). A large emissions share means the warming ranking aligns with
the emissions ranking in a way the other axes do not absorb -- not that
emissions caused warming (CO2 is well-mixed). The canonical disclaimer
(:data:`src.feature_schema.INTERPRETATION_NOTE`) is emitted into the summary
output.

Contract enforcement: every model here consumes features **only** through
:data:`SCHEMA_V1`. :func:`group_lmg_shares` calls
:func:`src.feature_schema.validate_design_matrix`, which raises if the input
carries any column that is not a schema feature (ids and the outcome must be
whitelisted explicitly). ``status="proposed"`` features are excluded until their
data source is wired.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from src import feature_schema as fs
from src.data_io import PROCESSED_DIR, round_floats
from src.emissions import DEFAULT_INEQUALITY_PATH
from src.feature_schema import INTERPRETATION_NOTE, SCHEMA_V1, FeatureSchema

logger = logging.getLogger(__name__)

OUTCOME_COL = "warming_trend"
COUNTRY_COL = "Country"
DEFAULT_SUMMARY_PATH = PROCESSED_DIR / "decomposition_summary.json"

# Modelling transforms of schema features (a monotone transform of a feature is
# the *same* feature, not a new one -- the contract is about identity, not
# scale). Heavy-tailed positive magnitudes enter as log10; everything else is
# identity. Membership here never adds or removes a schema feature.
LOG10_FEATURES = frozenset({"cum_co2_per_capita", "cum_co2_total", "population"})

# Schema features that enter as a categorical fixed-effect block (dummy-coded,
# one reference level dropped) rather than a numeric column.
CATEGORICAL_FEATURES = frozenset(
    {"climate_zone", "hemisphere", "spatial_block", "income_group"}
)


def _feature_block(data: pd.DataFrame, feature: str) -> tuple[np.ndarray, list[str]]:
    """Numeric design block (n, k) for one schema feature.

    Numeric features are a single (optionally log10) column; categoricals are
    dummy-coded with the first sorted level dropped. Columns are ordered
    deterministically so the decomposition is reproducible.
    """
    if feature in CATEGORICAL_FEATURES:
        cats = pd.Categorical(data[feature].astype("string"))
        dummies = pd.get_dummies(cats, drop_first=True, dtype=float)
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        return dummies.to_numpy(dtype=float), [f"{feature}={c}" for c in dummies.columns]

    col = data[feature].to_numpy(dtype=float)
    if feature in LOG10_FEATURES:
        col = np.log10(col)
    return col.reshape(-1, 1), [feature]


def _r2(blocks: list[np.ndarray], y: np.ndarray) -> float:
    """R^2 of OLS (with intercept) of `y` on the concatenated `blocks`.

    Empty design (no predictors) gives R^2 = 0. Uses a least-norm least squares
    solve, so a rank-deficient design still yields the correct projection R^2.
    """
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    intercept = np.ones((len(y), 1))
    design = np.hstack([intercept, *blocks]) if blocks else intercept
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(1.0 - np.sum(resid**2) / ss_tot)


def _shapley_weight(subset_size: int, n_groups: int) -> float:
    """LMG/Shapley averaging weight for a coalition of `subset_size` groups."""
    return factorial(subset_size) * factorial(n_groups - subset_size - 1) / factorial(n_groups)


@dataclass(frozen=True)
class DecompositionResult:
    """Group-level LMG/Shapley attribution of warming-trend variance."""

    outcome: str
    schema_version: str
    n: int
    total_r2: float
    residual_share: float
    shares: dict[str, float] = field(default_factory=dict)
    univariate_r2: dict[str, float] = field(default_factory=dict)
    group_features: dict[str, list[str]] = field(default_factory=dict)

    def check_sums_to_one(self, atol: float = 1e-9) -> None:
        """Assert group shares + residual sum to 1 (the LMG identity)."""
        total = sum(self.shares.values()) + self.residual_share
        if abs(total - 1.0) > atol:
            raise AssertionError(f"shares + residual = {total!r}, expected 1.0")


def group_lmg_shares(
    data: pd.DataFrame,
    schema: FeatureSchema = SCHEMA_V1,
    *,
    outcome: str = OUTCOME_COL,
    id_cols: tuple[str, ...] = (COUNTRY_COL,),
    status: str = fs.STATUS_AVAILABLE,
) -> DecompositionResult:
    """LMG/Shapley R^2 shares for the named feature groups in `schema`.

    Steps: (1) enforce the contract -- every non-id, non-outcome column of
    `data` must be a schema feature; (2) select the available features actually
    present per named group (the ``residual`` group has none by construction);
    (3) restrict to complete cases over the used features and outcome so all
    coalition R^2 values share one sample; (4) for each subset of groups compute
    the model R^2 and Shapley-average each group's incremental R^2.

    Args:
        data: one row per unit (country), with schema-named feature columns,
            the `outcome` column, and `id_cols`.
        schema: the feature contract (defaults to :data:`SCHEMA_V1`).
        outcome: the response column.
        id_cols: identifier columns carried alongside the features.
        status: which feature status to admit into the live design
            (``"available"`` by default; ``"proposed"`` features are excluded).

    Returns:
        A :class:`DecompositionResult`; ``shares`` are per named group,
        ``residual_share`` is ``1 - total_r2``, and they sum to 1.

    Raises:
        ValueError: if `data` carries a column outside the schema, or if no
            usable feature is present.
    """
    fs.validate_design_matrix(data.columns, schema, allow=(outcome, *id_cols))

    named_groups = [g for g in schema.groups if g.key != "residual"]
    used: dict[str, list[str]] = {}
    for group in named_groups:
        present = [
            f.name
            for f in group.features
            if f.status == status and f.name in data.columns
        ]
        if present:
            used[group.key] = present
    if not used:
        raise ValueError("no usable schema features present in data")

    all_features = [name for names in used.values() for name in names]
    complete = data.dropna(subset=[outcome, *all_features]).reset_index(drop=True)
    if complete.empty:
        raise ValueError("no complete-case rows over the used features and outcome")
    y = complete[outcome].to_numpy(dtype=float)

    # Pre-build each group's design block and its standalone (univariate) R^2.
    group_blocks: dict[str, list[np.ndarray]] = {}
    univariate: dict[str, float] = {}
    for key, names in used.items():
        blocks = [_feature_block(complete, name)[0] for name in names]
        group_blocks[key] = blocks
        univariate[key] = _r2(blocks, y)

    keys = list(used)
    n_groups = len(keys)

    # R^2 of every coalition, memoized by frozenset of group keys.
    r2_cache: dict[frozenset[str], float] = {}

    def coalition_r2(subset: frozenset[str]) -> float:
        if subset not in r2_cache:
            blocks = [b for k in subset for b in group_blocks[k]]
            r2_cache[subset] = _r2(blocks, y)
        return r2_cache[subset]

    shares = {k: 0.0 for k in keys}
    for key in keys:
        others = [k for k in keys if k != key]
        for size in range(len(others) + 1):
            weight = _shapley_weight(size, n_groups)
            for combo in combinations(others, size):
                s = frozenset(combo)
                marginal = coalition_r2(s | {key}) - coalition_r2(s)
                shares[key] += weight * marginal

    total_r2 = coalition_r2(frozenset(keys))
    result = DecompositionResult(
        outcome=outcome,
        schema_version=schema.version,
        n=int(len(complete)),
        total_r2=float(total_r2),
        residual_share=float(1.0 - total_r2),
        shares={k: float(v) for k, v in shares.items()},
        univariate_r2={k: float(v) for k, v in univariate.items()},
        group_features=used,
    )
    result.check_sums_to_one()
    return result


def result_to_dict(result: DecompositionResult) -> dict:
    """JSON-safe dict for :class:`DecompositionResult`."""
    return asdict(result)


def summary_payload(result: DecompositionResult) -> dict:
    """Serialized result with the scope disclaimer attached.

    The ``interpretation`` key carries
    :data:`src.feature_schema.INTERPRETATION_NOTE` so the shares cannot be lifted
    into a consumer without the variance-attribution-only boundary. Floats are
    rounded (:func:`src.data_io.round_floats`) so the committed JSON is
    byte-stable across the BLAS/platform noise in the LMG ``lstsq`` solves.
    """
    return round_floats({"interpretation": INTERPRETATION_NOTE, **result_to_dict(result)})


# ---------------------------------------------------------------------
# Country design assembly (real-data path; tests build frames directly)
# ---------------------------------------------------------------------


def aggregate_city_features_to_country(city_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-city-location features to one row per country.

    Numeric geography/population features become country means; categorical
    ones (Koeppen, hemisphere) become the country's modal (most common) class.

    Args:
        city_features: ``city_features.parquet`` rows (see
            :data:`src.explain.FEATURES_COLUMNS`).

    Returns:
        One row per ``Country`` with schema-named columns: ``abs_latitude``,
        ``elevation``, ``continentality``, ``climate_zone``, ``hemisphere``,
        ``station_density``.
    """

    def _mode(s: pd.Series):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else np.nan

    grouped = city_features.groupby("Country", observed=True)
    out = grouped.agg(
        abs_latitude=("abs_latitude", "mean"),
        elevation=("elevation_m", "mean"),
        continentality=("coast_km", "mean"),
        station_density=("station_density", "mean"),
        climate_zone=("koppen", _mode),
        hemisphere=("hemisphere", _mode),
    ).reset_index()
    return out


def build_country_design(
    inequality: pd.DataFrame,
    city_features: pd.DataFrame,
    income: pd.DataFrame,
    schema: FeatureSchema = SCHEMA_V1,
) -> pd.DataFrame:
    """Assemble the schema-named country design matrix from project artifacts.

    Maps existing columns onto :data:`SCHEMA_V1` feature names (e.g.
    ``cum_co2_t_per_capita`` -> ``cum_co2_per_capita``, ``continent`` ->
    ``spatial_block``) and aggregates city features to country level. Only
    ``status="available"`` features are materialized; proposed features
    (``gdp_per_capita``, ``urbanization_rate``, ``co2_intensity_gdp``) are
    intentionally absent.

    Args:
        inequality: ``country_inequality.parquet``.
        city_features: ``city_features.parquet`` (Phase 7).
        income: ``owid_country`` -> ``income_group`` table
            (:func:`src.explain.load_income_groups`).
        schema: the contract whose names the output uses.

    Returns:
        One row per country: ``Country``, ``warming_trend`` (outcome), and the
        available schema feature columns.
    """
    geo = aggregate_city_features_to_country(city_features)
    table = inequality.merge(geo, on="Country", how="left")

    income_map = income.set_index("owid_country")["income_group"]
    design = pd.DataFrame(
        {
            COUNTRY_COL: table["Country"],
            OUTCOME_COL: table["trend_c_per_decade"],
            # emissions
            "cum_co2_per_capita": table["cum_co2_t_per_capita"],
            "cum_co2_total": table["cumulative_co2_mt"],
            # geography
            "abs_latitude": table["abs_latitude"],
            "elevation": table["elevation"],
            "continentality": table["continentality"],
            "climate_zone": table["climate_zone"],
            "hemisphere": table["hemisphere"],
            "spatial_block": table["continent"],
            # socioeconomic
            "income_group": table["owid_country"].map(income_map),
            # population
            "population": table["population"],
            "station_density": table["station_density"],
        }
    )
    # Validate up front so contamination is caught before any modelling.
    fs.validate_design_matrix(design.columns, schema, allow=(OUTCOME_COL, COUNTRY_COL))
    return design


def main() -> None:
    """Build the country design from real artifacts, decompose, write summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from src.explain import DEFAULT_FEATURES_PATH, INCOME_PATH, load_income_groups

    inequality = pd.read_parquet(DEFAULT_INEQUALITY_PATH)
    city_features = pd.read_parquet(DEFAULT_FEATURES_PATH)
    income = load_income_groups(INCOME_PATH)

    design = build_country_design(inequality, city_features, income)
    result = group_lmg_shares(design)

    print(
        f"LMG/Shapley decomposition of {result.outcome} variance "
        f"(n={result.n}, schema {result.schema_version}, descriptive/non-causal)"
    )
    print(f"  total R^2 explained: {result.total_r2:.3f}")
    print(f"  {'group':<14s} {'shapley':>8s} {'of R^2':>8s} {'univ R^2':>9s}")
    for key in result.shares:
        share = result.shares[key]
        of_r2 = share / result.total_r2 if result.total_r2 else float("nan")
        print(f"  {key:<14s} {share:8.3f} {of_r2:8.0%} {result.univariate_r2[key]:9.3f}")
    print(f"  {'residual':<14s} {result.residual_share:8.3f} (unexplained)")
    print(f"  note: {INTERPRETATION_NOTE}")

    DEFAULT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SUMMARY_PATH.write_text(
        json.dumps(summary_payload(result), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {DEFAULT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
