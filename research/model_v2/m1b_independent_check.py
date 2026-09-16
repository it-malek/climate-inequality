"""Independent re-derivation of the frozen M1b scorecard from its own saved evidence.

This module shares no code with the evaluator it checks. It re-implements, from the
contract and the evaluator specification alone, every reported quantity that the saved
per-country evidence determines, and compares each one with the frozen scorecard:

1. in-sample and per-protocol R2, RMSE and MAE, from ``m1b_country_predictions.csv``;
2. primary-protocol fold summaries (RMSE median/min/max, worst fold, |error| IQR) and
   region summaries (per-M49-subregion RMSE and bias, worst eligible region, n >= 3);
3. the contract §4.3 paired resampling interval, re-implemented here: 2,000 resamples,
   ``default_rng(0)``, ``rng.integers(0, n, size=(2000, n))``, the same sampled indices
   for candidate and comparator, fixed out-of-fold errors, statistic
   RMSE(M1b) - RMSE(M0*), quantiles 0.025 and 0.975;
4. the §4.4 / spec §5 sign condition and verdict label, rebuilt from the saved
   coefficients and the recomputed deltas;
5. representation equivalence between ``primary_total_co2`` and ``per_capita``;
6. LMG/Shapley group shares, recomputed by the Shapley formula over the saved coalition
   R2 values, and the two share-accounting identities;
7. residual Moran's I, in-sample and out-of-fold, on row-standardised k = 8 nearest
   neighbour weights rebuilt from the ``m0_countries.csv`` station centroids;
8. the result manifest's SHA-256 digests against the files on disk;
9. the recorded area-centroid Moran statistics, on weights rebuilt from the
   ``country_geometry.csv`` area centroids the evaluator uses, after checking those
   centroids against the ones ``m0_countries.csv`` carries (required to agree exactly, so
   that neither source is selected for matching);
10. internal consistency of the saved fold evidence: ``m1b_cv_folds.csv`` against the fold
    ids and training sizes in the predictions file, the pipe-joined training memberships
    (self-exclusion and symmetry, the latter implied by a distance-threshold rule), the
    M49 labelling, the random-reference construction, and the held-out country set behind
    the training-fit coefficients.

Every quantity here is recomputed from the same saved float64 values, so the tolerance is
1e-10 absolute throughout; labels, counts and identifiers are compared exactly. The CSVs are
parsed with ``float_precision="round_trip"``: pandas' default parser is accurate only to
about 1e-14 relative, which would silently compare slightly different inputs. The k = 8
neighbour weights are the one re-derived input: the selection depends only on the ordering
of great-circle distances, which a haversine and any other proper great-circle formula
share, and the row-standardised weights are then 1/8 each.

What the saved evidence cannot establish: the predictions, coefficients and coalition R2
values are taken as given. This check confirms that the reported statistics follow from
them, not that the underlying regressions, encoders, folds or coalition fits are correct.

Run: ``uv run python research/model_v2/m1b_independent_check.py``
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "research" / "model_v2" / "outputs" / "m1b_primary"
OUTPUT_PATH = (
    ROOT / "research" / "model_v2" / "outputs" / "m1b_primary_verification" / "m1b_independent_check.json"
)
COUNTRIES_PATH = ROOT / "research" / "model_v2" / "outputs" / "m0_countries.csv"
GEOMETRY_PATH = ROOT / "research" / "model_v2" / "outputs" / "country_geometry.csv"

TOLERANCE = 1e-10
REPRESENTATIONS = ("primary_total_co2", "per_capita")
MODELS = ("M0star", "M1b")
PROTOCOLS = ("in_sample", "primary_loco", "m49_subregion_lo", "random10")
CARD_OF_PROTOCOL = {
    "primary_loco": (),
    "m49_subregion_lo": ("secondary",),
    "random10": ("random_reference_only",),
}
GROUPS = {"M0star": ("emissions", "geography", "socioeconomic", "population")}
GROUPS["M1b"] = GROUPS["M0star"] + ("hydroclimate",)
K_NEIGHBOURS = 8
RESAMPLES = 2000
SEED = 0
EARTH_RADIUS_KM = 6371.0088


def _plain(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def numeric(name: str, recomputed: Any, reported: Any, tol: float = TOLERANCE) -> dict[str, Any]:
    """One numeric comparison record; a missing or non-finite value never passes."""
    try:
        diff = abs(float(recomputed) - float(reported))
        passes = math.isfinite(diff) and diff <= tol
    except (TypeError, ValueError):
        diff, passes = float("nan"), False
    return {
        "check": name,
        "recomputed": _plain(recomputed),
        "reported": _plain(reported),
        "abs_diff": _plain(diff),
        "tolerance": tol,
        "passes": passes,
    }


def exact(name: str, recomputed: Any, reported: Any) -> dict[str, Any]:
    """One exact comparison record, for labels, identifiers and counts."""
    return {
        "check": name,
        "recomputed": _plain(recomputed),
        "reported": _plain(reported),
        "abs_diff": 0.0 if recomputed == reported else None,
        "tolerance": "exact",
        "passes": bool(recomputed == reported),
    }


def digest_of(values: Iterable[Any]) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()[:16]


def sequence(name: str, recomputed: list[Any], reported: list[Any]) -> dict[str, Any]:
    """Exact comparison of a long sequence, recorded compactly as lengths, mismatch count and digests."""
    pairs = itertools.zip_longest(recomputed, reported, fillvalue=object())
    return {
        "check": name,
        "recomputed": {"n": len(recomputed), "sha256_16": digest_of(recomputed)},
        "reported": {"n": len(reported), "sha256_16": digest_of(reported)},
        "n_differing": sum(1 for a, b in pairs if a != b),
        "abs_diff": None,
        "tolerance": "exact",
        "passes": list(recomputed) == list(reported),
    }


def read_csv(path: Path) -> pd.DataFrame:
    """Exact float64 round-trip; pandas' default parser is only ~1e-14 relative."""
    return pd.read_csv(path, float_precision="round_trip")


def card(scorecard: dict[str, Any], representation: str, model: str, path: Iterable[str] = ()) -> dict[str, Any]:
    node = scorecard["representations"][representation][model]
    for key in path:
        node = node[key]
    return node


def slice_predictions(predictions: pd.DataFrame, representation: str, model: str, protocol: str) -> pd.DataFrame:
    frame = predictions[
        (predictions["representation"] == representation)
        & (predictions["model"] == model)
        & (predictions["protocol"] == protocol)
    ]
    if len(frame) != 151:
        raise ValueError(f"expected 151 rows for {representation}/{model}/{protocol}, found {len(frame)}")
    return frame


def r2_rmse_mae(observed: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    error = observed - prediction
    sse = float((error**2).sum())
    sst = float(((observed - observed.mean()) ** 2).sum())
    return 1.0 - sse / sst, float(np.sqrt((error**2).mean())), float(np.abs(error).mean())


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt((error**2).mean()))


def check_scores(scorecard: dict[str, Any], predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """In-sample R2/RMSE and per-protocol R2/RMSE/MAE for every representation and model."""
    records: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for model in MODELS:
            frame = slice_predictions(predictions, representation, model, "in_sample")
            r2, root_mse, _ = r2_rmse_mae(frame["observed"].to_numpy(), frame["prediction"].to_numpy())
            reported = card(scorecard, representation, model)
            stem = f"{representation}/{model}"
            records.append(numeric(f"{stem}/in_sample_r2", r2, reported["in_sample_r2"]))
            records.append(numeric(f"{stem}/in_sample_rmse", root_mse, reported["in_sample_rmse"]))
            records.append(exact(f"{stem}/n", len(frame), reported["n"]))
            for protocol, path in CARD_OF_PROTOCOL.items():
                frame = slice_predictions(predictions, representation, model, protocol)
                r2, root_mse, mae = r2_rmse_mae(frame["observed"].to_numpy(), frame["prediction"].to_numpy())
                block = card(scorecard, representation, model, path)
                label = f"{stem}/{protocol}"
                records.append(numeric(f"{label}/cv_r2", r2, block["cv_r2"]))
                records.append(numeric(f"{label}/cv_rmse", root_mse, block["cv_rmse"]))
                records.append(numeric(f"{label}/cv_mae", mae, block["cv_mae"]))
    return records


def fold_rmse(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("fold_id")["error"].apply(lambda e: rmse(e.to_numpy())).sort_index()


def region_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("m49_subregion")
    return pd.DataFrame(
        {
            "rmse": grouped["error"].apply(lambda e: rmse(e.to_numpy())),
            "bias": grouped["error"].mean(),
            "n": grouped.size(),
        }
    )


def worst_eligible_region(summary: pd.DataFrame, min_n: int = 3) -> tuple[str, float, int]:
    eligible = summary[summary["n"] >= min_n]
    region = eligible["rmse"].idxmax()
    return str(region), float(eligible.loc[region, "rmse"]), int(eligible.loc[region, "n"])


def check_fold_and_region(scorecard: dict[str, Any], predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Primary-protocol fold and region summaries, for every representation and model."""
    records: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for model in MODELS:
            frame = slice_predictions(predictions, representation, model, "primary_loco")
            reported = card(scorecard, representation, model)
            stem = f"{representation}/{model}/primary_loco"

            per_fold = fold_rmse(frame)
            absolute = frame["abs_error"].to_numpy()
            records.append(exact(f"{stem}/n_folds", len(per_fold), reported["n_folds"]))
            records.append(numeric(f"{stem}/fold_rmse_median", float(np.median(per_fold.to_numpy())), reported["fold_rmse_median"]))
            records.append(numeric(f"{stem}/fold_rmse_min", float(per_fold.min()), reported["fold_rmse_min"]))
            records.append(numeric(f"{stem}/fold_rmse_max", float(per_fold.max()), reported["fold_rmse_max"]))
            records.append(exact(f"{stem}/worst_fold", int(per_fold.idxmax()), reported["worst_fold"]))
            worst_rows = frame[frame["fold_id"] == per_fold.idxmax()]
            records.append(exact(f"{stem}/worst_fold_units", sorted(worst_rows["iso3"]), sorted(reported["worst_fold_units"])))
            records.append(
                numeric(
                    f"{stem}/fold_error_iqr",
                    float(np.quantile(absolute, 0.75) - np.quantile(absolute, 0.25)),
                    reported["fold_error_iqr"],
                )
            )

            summary = region_summary(frame)
            for region, row in summary.iterrows():
                records.append(numeric(f"{stem}/region_rmse[{region}]", row["rmse"], reported["region_rmse"][region]))
                records.append(numeric(f"{stem}/region_bias[{region}]", row["bias"], reported["region_bias"][region]))
            records.append(exact(f"{stem}/region_count", len(summary), len(reported["region_rmse"])))
            name, value, size = worst_eligible_region(summary)
            records.append(exact(f"{stem}/worst_region.region", name, reported["worst_region"]["region"]))
            records.append(numeric(f"{stem}/worst_region.rmse", value, reported["worst_region"]["rmse"]))
            records.append(exact(f"{stem}/worst_region.n", size, reported["worst_region"]["n"]))
    return records


def paired_interval(comparator_error: np.ndarray, candidate_error: np.ndarray) -> dict[str, float]:
    """Contract §4.3: paired country resampling of fixed out-of-fold errors, no refitting."""
    n = comparator_error.size
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, n, size=(RESAMPLES, n))
    statistic = np.sqrt((candidate_error[index] ** 2).mean(axis=1)) - np.sqrt(
        (comparator_error[index] ** 2).mean(axis=1)
    )
    low, high = np.quantile(statistic, [0.025, 0.975])
    return {
        "delta_rmse": rmse(candidate_error) - rmse(comparator_error),
        "low": float(low),
        "high": float(high),
    }


def primary_errors(predictions: pd.DataFrame, representation: str, order: list[str]) -> dict[str, np.ndarray]:
    errors = {}
    for model in MODELS:
        frame = slice_predictions(predictions, representation, model, "primary_loco").set_index("iso3")
        errors[model] = frame.loc[order, "error"].to_numpy()
    return errors


def check_paired_interval(
    scorecard: dict[str, Any], predictions: pd.DataFrame, order: list[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    records: list[dict[str, Any]] = []
    computed: dict[str, dict[str, float]] = {}
    for representation in REPRESENTATIONS:
        errors = primary_errors(predictions, representation, order)
        interval = paired_interval(errors["M0star"], errors["M1b"])
        computed[representation] = interval
        reported = scorecard["representations"][representation]["paired_primary_comparison"]
        stem = f"{representation}/paired"
        records.append(numeric(f"{stem}/delta_rmse", interval["delta_rmse"], reported["delta_rmse"]))
        records.append(numeric(f"{stem}/interval_low", interval["low"], reported["country_bootstrap_95_interval"][0]))
        records.append(numeric(f"{stem}/interval_high", interval["high"], reported["country_bootstrap_95_interval"][1]))
        records.append(exact(f"{stem}/resamples", RESAMPLES, reported["resamples"]))
        records.append(exact(f"{stem}/seed", SEED, reported["seed"]))
    return records, computed


def dryness_coefficients(coefficients: pd.DataFrame, representation: str) -> tuple[float, np.ndarray]:
    rows = coefficients[
        (coefficients["representation"] == representation)
        & (coefficients["model"] == "M1b")
        & (coefficients["column"] == "baseline_dryness")
    ]
    full = rows[rows["fit"] == "full_sample"]["coefficient"]
    training = rows[rows["fit"] == "primary_training"]["coefficient"].to_numpy()
    if len(full) != 1:
        raise ValueError(f"expected one full-sample baseline_dryness coefficient, found {len(full)}")
    return float(full.iloc[0]), training


def check_verdict(
    scorecard: dict[str, Any],
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    intervals: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Sign condition, S1/S2, A1-A5 and the verdict label, rebuilt from the frozen rules."""
    records: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        full, training = dryness_coefficients(coefficients, representation)
        finite = np.isfinite(training)
        negative_n = int((training[finite] < 0).sum())
        zero_n = int((training[finite] == 0).sum())
        n_fits = int(finite.sum())

        interval = intervals[representation]
        delta = interval["delta_rmse"]
        m49_change = rmse(
            slice_predictions(predictions, representation, "M1b", "m49_subregion_lo")["error"].to_numpy()
        ) - rmse(slice_predictions(predictions, representation, "M0star", "m49_subregion_lo")["error"].to_numpy())
        worst = {
            model: worst_eligible_region(
                region_summary(slice_predictions(predictions, representation, model, "primary_loco"))
            )[1]
            for model in MODELS
        }
        worst_change = worst["M1b"] - worst["M0star"]

        s1 = bool(delta < 0 and interval["high"] < 0)
        s2 = bool(full < 0 and negative_n * 5 >= n_fits * 4)
        a1, a5 = s1, s2
        a2 = bool(delta <= -0.002)
        a3 = bool(m49_change <= 0.001)
        a4 = bool(worst_change <= 0.001)
        if a1 and a2 and a3 and a4 and a5:
            label = "supported and promoted"
        elif s1 and s2:
            label = "supported but sub-material / not promoted"
        else:
            label = "not supported"

        reported = (
            scorecard["representations"][representation]["verdict"]
            if representation == scorecard["primary_representation"]
            else scorecard["representations"][representation]["representation_conditions"]
        )
        label_key = "verdict" if representation == scorecard["primary_representation"] else "label_descriptive_only"
        stem = f"{representation}/verdict"
        records.append(exact(f"{stem}/coefficient_n_fits", n_fits, reported["coefficient_n_fits"]))
        records.append(exact(f"{stem}/coefficient_n_fits_is_151", n_fits, 151))
        records.append(exact(f"{stem}/coefficient_negative_n", negative_n, reported["coefficient_negative_n"]))
        records.append(exact(f"{stem}/coefficient_zero_n", zero_n, reported["coefficient_zero_n"]))
        records.append(
            numeric(f"{stem}/coefficient_negative_fraction", negative_n / n_fits, reported["coefficient_negative_fraction"])
        )
        records.append(numeric(f"{stem}/coefficient_full", full, reported["coefficient_full"]))
        records.append(numeric(f"{stem}/m49_rmse_change", m49_change, reported["m49_rmse_change"]))
        records.append(numeric(f"{stem}/worst_region_rmse_change", worst_change, reported["worst_region_rmse_change"]))
        records.append(exact(f"{stem}/S1", s1, reported["S1_interval_below_zero"]))
        records.append(exact(f"{stem}/S2", s2, reported["S2_sign_negative_and_stable"]))
        records.append(exact(f"{stem}/A1", a1, reported["A1"]))
        records.append(exact(f"{stem}/A2", a2, reported["A2_practical_le_minus_0.002"]))
        records.append(exact(f"{stem}/A3", a3, reported["A3_m49_veto_passed"]))
        records.append(exact(f"{stem}/A4", a4, reported["A4_worst_region_veto_passed"]))
        records.append(exact(f"{stem}/A5", a5, reported["A5"]))
        records.append(exact(f"{stem}/label", label, reported[label_key]))
        records.append(exact(f"{stem}/association_supported", bool(s1 and s2), reported["association_supported"]))
        records.append(
            exact(
                f"{stem}/improvement_without_prestated_sign",
                bool(s1 and not s2),
                reported["improvement_without_prestated_sign"],
            )
        )
        records.append(
            exact(
                f"{stem}/worsened_generalization",
                bool(delta > 0 and interval["low"] > 0),
                reported["worsened_generalization"],
            )
        )
    records.append(exact("scorecard/verdict_label", scorecard["verdict"]["verdict"], scorecard["representations"][scorecard["primary_representation"]]["verdict"]["verdict"]))
    return records


def check_equivalence(
    scorecard: dict[str, Any], predictions: pd.DataFrame, coefficients: pd.DataFrame, order: list[str]
) -> list[dict[str, Any]]:
    """Max absolute primary-vs-per-capita difference in predictions and dryness coefficients."""
    records: list[dict[str, Any]] = []
    reported = scorecard["representation_equivalence"]
    for model in MODELS:
        for protocol in PROTOCOLS:
            frames = {
                representation: slice_predictions(predictions, representation, model, protocol)
                .set_index("iso3")
                .loc[order, "prediction"]
                .to_numpy()
                for representation in REPRESENTATIONS
            }
            gap = float(np.max(np.abs(frames["primary_total_co2"] - frames["per_capita"])))
            records.append(numeric(f"equivalence/predictions/{model}/{protocol}", gap, reported["predictions"][model][protocol]))

    full = {rep: dryness_coefficients(coefficients, rep)[0] for rep in REPRESENTATIONS}
    records.append(
        numeric(
            "equivalence/coefficients/baseline_dryness_full",
            abs(full["primary_total_co2"] - full["per_capita"]),
            reported["coefficients"]["baseline_dryness_full"],
        )
    )
    training = {}
    for representation in REPRESENTATIONS:
        rows = coefficients[
            (coefficients["representation"] == representation)
            & (coefficients["model"] == "M1b")
            & (coefficients["column"] == "baseline_dryness")
            & (coefficients["fit"] == "primary_training")
        ]
        training[representation] = rows.set_index("held_out_iso3").loc[order, "coefficient"].to_numpy()
    gap = float(np.max(np.abs(training["primary_total_co2"] - training["per_capita"])))
    records.append(
        numeric(
            "equivalence/coefficients/baseline_dryness_training_max",
            gap,
            reported["coefficients"]["baseline_dryness_training_max"],
        )
    )
    records.append(
        exact(
            "equivalence/coefficients/training_fits_compared",
            len(training["primary_total_co2"]),
            reported["coefficients"]["training_fits_compared"],
        )
    )
    return records


def shapley_shares(coalitions: pd.DataFrame, groups: tuple[str, ...]) -> dict[str, float]:
    """LMG/Shapley share of each group from the saved coalition R2 values."""
    lookup = {
        frozenset() if name == "intercept_only" else frozenset(name.split("+")): float(value)
        for name, value in zip(coalitions["coalition"], coalitions["r2"])
    }
    total = len(groups)
    shares = {}
    for group in groups:
        others = [g for g in groups if g != group]
        accumulated = 0.0
        for size in range(total):
            weight = math.factorial(size) * math.factorial(total - size - 1) / math.factorial(total)
            for subset in itertools.combinations(others, size):
                base = frozenset(subset)
                accumulated += weight * (lookup[base | {group}] - lookup[base])
        shares[group] = accumulated
    return shares


def check_shares(scorecard: dict[str, Any], coalitions: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for model in MODELS:
            groups = GROUPS[model]
            subset = coalitions[(coalitions["representation"] == representation) & (coalitions["model"] == model)]
            expected_coalitions = 2 ** len(groups)
            records.append(exact(f"{representation}/{model}/shapley/n_coalitions", len(subset), expected_coalitions))
            shares = shapley_shares(subset, groups)
            reported = card(scorecard, representation, model)
            stem = f"{representation}/{model}/shapley"
            for group, value in shares.items():
                records.append(numeric(f"{stem}/share_{group}", value, reported[f"share_{group}"]))
            named_sum = sum(shares.values())
            records.append(numeric(f"{stem}/named_share_sum_minus_in_sample_r2", named_sum, reported["in_sample_r2"]))
            records.append(
                numeric(f"{stem}/named_plus_residual", named_sum + reported["residual_share"], 1.0)
            )
    return records


def knn_weights(latitude: np.ndarray, longitude: np.ndarray, k: int = K_NEIGHBOURS) -> np.ndarray:
    """Row-standardised k-nearest-neighbour weights; self excluded, ties by distance then index."""
    phi = np.radians(latitude)[:, None]
    lam = np.radians(longitude)[:, None]
    dphi = phi - phi.T
    dlam = lam - lam.T
    haversine = np.sin(dphi / 2) ** 2 + np.cos(phi) * np.cos(phi.T) * np.sin(dlam / 2) ** 2
    distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    n = distance.shape[0]
    weights = np.zeros((n, n))
    for i in range(n):
        candidates = sorted((j for j in range(n) if j != i), key=lambda j: (distance[i, j], j))
        weights[i, candidates[:k]] = 1.0 / k
    return weights


def morans_i(residual: np.ndarray, weights: np.ndarray) -> float:
    z = residual - residual.mean()
    return float((len(z) / weights.sum()) * (z @ weights @ z) / (z @ z))


def check_morans_i(
    scorecard: dict[str, Any], predictions: pd.DataFrame, weights: np.ndarray, order: list[str]
) -> list[dict[str, Any]]:
    """Residual Moran's I for both models of the primary representation."""
    records: list[dict[str, Any]] = []
    representation = scorecard["primary_representation"]
    for model in MODELS:
        reported = card(scorecard, representation, model)
        for protocol, key in (("in_sample", "residual_morans_i_in_sample"), ("primary_loco", "residual_morans_i_cv")):
            residual = (
                slice_predictions(predictions, representation, model, protocol)
                .set_index("iso3")
                .loc[order, "error"]
                .to_numpy()
            )
            records.append(numeric(f"{representation}/{model}/morans_i/{protocol}", morans_i(residual, weights), reported[key]))
    return records


def check_area_centroid_morans_i(
    scorecard: dict[str, Any],
    predictions: pd.DataFrame,
    geometry: pd.DataFrame,
    countries: pd.DataFrame,
    order: list[str],
) -> list[dict[str, Any]]:
    """The recorded area-centroid Moran statistics, from the geometry table's centroids.

    ``m0_countries.csv`` carries the same two columns. They are required to agree exactly,
    so that a disagreement is reported rather than resolved by picking the matching source.
    """
    records: list[dict[str, Any]] = []
    centroids = geometry.set_index("iso3").loc[order]
    records.append(exact("area_centroid/geometry_covers_frozen_order", len(centroids), len(order)))
    for column in ("centroid_lon", "centroid_lat"):
        gap = float(np.max(np.abs(centroids[column].to_numpy() - countries[column].to_numpy())))
        records.append(
            numeric(f"area_centroid/source_agreement/{column}", gap, 0.0, tol=0.0)
            | {"note": "country_geometry.csv vs m0_countries.csv, frozen order; required to be bit-identical"}
        )

    weights = knn_weights(centroids["centroid_lat"].to_numpy(), centroids["centroid_lon"].to_numpy())
    representation = scorecard["primary_representation"]
    for model in MODELS:
        reported = card(scorecard, representation, model)
        for protocol, key in (
            ("in_sample", "recorded_area_centroid_moran_in_sample"),
            ("primary_loco", "recorded_area_centroid_moran_cv"),
        ):
            residual = (
                slice_predictions(predictions, representation, model, protocol)
                .set_index("iso3")
                .loc[order, "error"]
                .to_numpy()
            )
            records.append(
                numeric(f"{representation}/{model}/area_centroid_morans_i/{protocol}", morans_i(residual, weights), reported[key])
            )
    return records


def check_fold_evidence(
    folds: pd.DataFrame, predictions: pd.DataFrame, coefficients: pd.DataFrame, order: list[str]
) -> list[dict[str, Any]]:
    """Internal consistency of m1b_cv_folds.csv against the predictions and coefficients."""
    records: list[dict[str, Any]] = []
    records.append(sequence("folds/iso3_order", list(folds["iso3"]), order))
    folds = folds.set_index("iso3").loc[order]
    records.append(sequence("folds/primary_fold_is_leave_one_out", sorted(folds["primary_fold"]), list(range(len(order)))))

    membership = {iso: row.split("|") for iso, row in folds["primary_training_iso3"].items()}
    records.append(
        exact(
            "folds/training_list_length_equals_primary_n_train",
            int((np.array([len(membership[iso]) for iso in order]) != folds["primary_n_train"].to_numpy()).sum()),
            0,
        )
    )
    records.append(exact("folds/held_out_never_in_own_training", sum(1 for iso, m in membership.items() if iso in m), 0))
    sets = {iso: set(m) for iso, m in membership.items()}
    records.append(
        exact("folds/training_members_are_frozen_countries", len(set().union(*sets.values()) - set(order)), 0)
    )
    records.append(
        exact(
            "folds/training_membership_symmetric",
            sum(1 for a, m in sets.items() for b in m if a not in sets.get(b, frozenset())),
            0,
        )
    )

    for representation in REPRESENTATIONS:
        for model in MODELS:
            for protocol, column in (
                ("primary_loco", "primary_fold"),
                ("m49_subregion_lo", "m49_fold"),
                ("random10", "random10_fold"),
            ):
                frame = slice_predictions(predictions, representation, model, protocol).set_index("iso3").loc[order]
                stem = f"folds/{representation}/{model}/{protocol}"
                records.append(
                    exact(f"{stem}/fold_id_matches", int((frame["fold_id"].to_numpy() != folds[column].to_numpy()).sum()), 0)
                )
                if protocol == "primary_loco":
                    records.append(
                        exact(
                            f"{stem}/n_train_matches",
                            int((frame["n_train"].to_numpy() != folds["primary_n_train"].to_numpy()).sum()),
                            0,
                        )
                    )

    by_label = folds.groupby("m49_subregion")["m49_fold"]
    records.append(exact("folds/m49_fold_constant_within_label", int((by_label.nunique() != 1).sum()), 0))
    records.append(exact("folds/m49_fold_distinct_across_labels", by_label.first().nunique(), by_label.ngroups))
    records.append(
        sequence("folds/m49_label_matches_predictions", sorted(set(folds["m49_subregion"])), sorted(set(predictions["m49_subregion"])))
    )

    expected_random = np.random.default_rng(SEED).permutation(len(order)) % 10
    records.append(
        exact(
            "folds/random10_equals_default_rng0_permutation_mod_10",
            int((folds["random10_fold"].to_numpy() != expected_random).sum()),
            0,
        )
    )

    for representation in REPRESENTATIONS:
        for model in MODELS:
            held_out = set(
                coefficients[
                    (coefficients["representation"] == representation)
                    & (coefficients["model"] == model)
                    & (coefficients["fit"] == "primary_training")
                ]["held_out_iso3"]
            )
            records.append(sequence(f"folds/{representation}/{model}/held_out_iso3_set", sorted(held_out), sorted(order)))
    return records


def check_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, digest in manifest["artifact_sha256"].items():
        path = RESULT_DIR / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        records.append(exact(f"manifest/{name}", actual, digest))
    return records


def main() -> int:
    scorecard = json.loads((RESULT_DIR / "m1b_scorecard.json").read_text())
    manifest = json.loads((RESULT_DIR / "m1b_result_manifest.json").read_text())
    predictions = read_csv(RESULT_DIR / "m1b_country_predictions.csv")
    coefficients = read_csv(RESULT_DIR / "m1b_coefficients.csv")
    coalitions = read_csv(RESULT_DIR / "m1b_shapley_coalitions.csv")
    folds = read_csv(RESULT_DIR / "m1b_cv_folds.csv")
    countries = read_csv(COUNTRIES_PATH)
    geometry = read_csv(GEOMETRY_PATH)

    order = list(countries["iso3"])
    weights = knn_weights(countries["station_lat"].to_numpy(), countries["station_lon"].to_numpy())
    interval_records, intervals = check_paired_interval(scorecard, predictions, order)

    checks = {
        "1_scores": check_scores(scorecard, predictions),
        "2_fold_and_region_summaries": check_fold_and_region(scorecard, predictions),
        "3_paired_interval": interval_records,
        "4_sign_condition_and_verdict": check_verdict(scorecard, predictions, coefficients, intervals),
        "5_representation_equivalence": check_equivalence(scorecard, predictions, coefficients, order),
        "6_share_accounting": check_shares(scorecard, coalitions),
        "7_residual_morans_i": check_morans_i(scorecard, predictions, weights, order),
        "8_manifest_digests": check_manifest(manifest),
        "9_area_centroid_morans_i": check_area_centroid_morans_i(scorecard, predictions, geometry, countries, order),
        "10_fold_evidence_consistency": check_fold_evidence(folds, predictions, coefficients, order),
    }

    summary = {
        name: {
            "n_comparisons": len(records),
            "n_failed": sum(1 for r in records if not r["passes"]),
            "passes": all(r["passes"] for r in records),
            "comparisons": records,
        }
        for name, records in checks.items()
    }
    failures = [r["check"] for records in checks.values() for r in records if not r["passes"]]
    report = {
        "what_this_is": "independent re-derivation of research/model_v2/outputs/m1b_primary from its saved evidence",
        "implementation": "research/model_v2/m1b_independent_check.py; shares no code with the evaluator",
        "default_tolerance_absolute": TOLERANCE,
        "tolerance_rationale": (
            "every quantity is recomputed from the same saved float64 values, parsed with "
            "float_precision='round_trip', so only float summation order differs; labels, counts and "
            "identifiers are compared exactly"
        ),
        "not_verifiable_from_saved_evidence": [
            "the regressions behind the predictions, coefficients and coalition R2 values are taken as given",
            "the Shapley check validates the aggregation of the saved coalition R2 values, not those fits",
            "encoders and the outcome vector are not re-derived",
            "check 10 establishes that the saved fold evidence is mutually consistent and that the LOCO "
            "memberships obey a symmetric exclusion rule; it does not re-derive the 500 km territorial "
            "bounds, so it cannot confirm that the excluded set is the correct one",
            "the 999-permutation Moran p-values and z-scores are not recomputed",
        ],
        "all_checks_pass": not failures,
        "failed_checks": failures,
        "checks": summary,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    total = sum(len(records) for records in checks.values())
    print(f"{total - len(failures)}/{total} comparisons pass; all_checks_pass={not failures}")
    for name, block in summary.items():
        print(f"  {name}: {block['n_comparisons'] - block['n_failed']}/{block['n_comparisons']}")
    for failure in failures:
        print(f"  FAIL {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
