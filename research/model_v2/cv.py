"""Spatial cross-validation candidates and the common scoring harness for Model V2.

Two things live here, both fixed before any predictor is evaluated:

1. **Fold constructions** (deterministic): leave-one-group-out over a label,
   k-means clusters of the land-area centroids, latitude bands, random K-fold (the
   non-spatial reference), and an optional exclusion buffer that removes training
   countries within ``buffer_km`` of any test country.
2. **The scoring harness**: :func:`cross_validate` produces out-of-fold predictions
   for any ``fit_predict(train, test) -> yhat`` callable, and :func:`scorecard`
   turns in-sample and out-of-fold predictions into the fixed set of metrics.

The M0 estimator is provided through :class:`V1Design` + :func:`fit_predict_v1`,
which reproduces the tagged model's design (same transforms, same drop-first
dummy coding) and adds one rule the in-sample fit never needed: a test country
whose categorical level does not occur in the training fold receives the mean of
the training level effects (``unseen="mean_effect"``) or the reference level
(``unseen="reference"``). The rule is part of the protocol, not of the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from src import feature_schema as fs
from src.decomposition import CATEGORICAL_FEATURES, LOG10_FEATURES, OUTCOME_COL, group_lmg_shares
from src.stability import _used_features

from research.model_v2.spatial import haversine_matrix, knn_weights, morans_i

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


# ---------------------------------------------------------------------
# Fold constructions
# ---------------------------------------------------------------------


def folds_from_labels(labels: pd.Series | np.ndarray) -> np.ndarray:
    """Integer fold id per row from a categorical label (sorted label order)."""
    codes, _ = pd.factorize(pd.Series(labels).astype(str), sort=True)
    return codes.astype(int)


def _unit_vectors(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    phi, lam = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])


def folds_kmeans(lon: np.ndarray, lat: np.ndarray, k: int, seed: int = 0, n_restarts: int = 20) -> np.ndarray:
    """k-means clusters of centroids on the unit sphere; best of ``n_restarts`` seeded runs.

    Deterministic for a given ``seed``; clusters are relabelled by mean longitude so
    the fold ids are stable across runs.
    """
    xyz = _unit_vectors(lon, lat)
    best, best_inertia = None, np.inf
    for r in range(n_restarts):
        centroids, labels = kmeans2(xyz, k, minit="++", seed=seed + r, iter=100)
        inertia = float(np.sum((xyz - centroids[labels]) ** 2))
        if inertia < best_inertia - 1e-12 and len(np.unique(labels)) == k:
            best, best_inertia = labels, inertia
    order = np.argsort([np.mean(lon[best == c]) for c in range(k)])
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[c] for c in best], dtype=int)


def folds_latitude_bands(abs_lat: np.ndarray, edges: tuple[float, ...] = (0, 12, 24, 36, 48, 90)) -> np.ndarray:
    """Absolute-latitude bands (an adversarial, gradient-wise hold-out)."""
    return (np.digitize(abs_lat, edges[1:-1])).astype(int)


def folds_random(n: int, k: int, seed: int = 0) -> np.ndarray:
    """Random K-fold assignment (non-spatial reference)."""
    rng = np.random.default_rng(seed)
    return (rng.permutation(n) % k).astype(int)


# ---------------------------------------------------------------------
# The V1 estimator as a fit/predict pair
# ---------------------------------------------------------------------


@dataclass
class V1Design:
    """The V1 design matrix as a fitted encoder, with the unseen-level rule."""

    unseen: str = "mean_effect"
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    levels: dict[str, list[str]] = field(default_factory=dict)
    coef: np.ndarray | None = None
    columns: list[str] = field(default_factory=list)

    def _features(self, columns: pd.Index) -> list[str]:
        used = _used_features(columns, fs.SCHEMA_V1, fs.STATUS_AVAILABLE)
        return [name for names in used.values() for name in names]

    def fit(self, train: pd.DataFrame) -> V1Design:
        features = self._features(train.columns)
        self.numeric = [f for f in features if f not in CATEGORICAL_FEATURES]
        self.categorical = [f for f in features if f in CATEGORICAL_FEATURES]
        self.levels = {c: sorted(train[c].astype(str).unique().tolist()) for c in self.categorical}
        x, self.columns = self._matrix(train)
        y = train[OUTCOME_COL].to_numpy(dtype=float)
        self.coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        return self

    def _matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        blocks, names = [np.ones((len(frame), 1))], ["intercept"]
        for f in self.numeric:
            col = frame[f].to_numpy(dtype=float)
            blocks.append((np.log10(col) if f in LOG10_FEATURES else col).reshape(-1, 1))
            names.append(f)
        for c in self.categorical:
            levels = self.levels[c]
            values = frame[c].astype(str).to_numpy()
            for level in levels[1:]:  # drop-first coding on the training levels
                blocks.append((values == level).astype(float).reshape(-1, 1))
                names.append(f"{c}={level}")
        return np.hstack(blocks), names

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        x, _ = self._matrix(test)
        yhat = x @ self.coef
        if self.unseen == "mean_effect":
            for c in self.categorical:
                levels = self.levels[c]
                effects = np.array([0.0] + [self.coef[self.columns.index(f"{c}={lv}")] for lv in levels[1:]])
                unseen = ~test[c].astype(str).isin(levels).to_numpy()
                yhat = yhat + unseen * effects.mean()
        elif self.unseen != "reference":
            raise ValueError(f"unknown unseen-level rule {self.unseen!r}")
        return yhat


def fit_predict_v1(train: pd.DataFrame, test: pd.DataFrame, unseen: str = "mean_effect") -> np.ndarray:
    """M0's estimator on ``train``, predictions for ``test``."""
    return V1Design(unseen=unseen).fit(train).predict(test)


# ---------------------------------------------------------------------
# Cross-validation and the scorecard
# ---------------------------------------------------------------------


def iter_folds(fold_id: np.ndarray, dist: np.ndarray | None = None, buffer_km: float | None = None) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(fold, test_mask, train_mask)``; a buffer removes training rows near any test row."""
    if buffer_km is not None:
        if not np.isfinite(buffer_km) or buffer_km < 0:
            raise ValueError("buffer_km must be finite and nonnegative")
        if dist is None:
            raise ValueError("a distance matrix is required for buffered validation")
    if dist is not None:
        dist = np.asarray(dist, dtype=float)
        if (dist.shape != (len(fold_id), len(fold_id))
                or not np.isfinite(dist).all() or (dist < 0).any()
                or not np.array_equal(dist, dist.T) or np.any(np.diag(dist) != 0)):
            raise ValueError("distance matrix must be finite, symmetric, nonnegative and aligned")
    for f in np.unique(fold_id):
        test = fold_id == f
        train = ~test
        if buffer_km is not None and dist is not None:
            near = (dist[test][:, :] <= buffer_km).any(axis=0)
            train &= ~near
        yield int(f), test, train


@dataclass(frozen=True)
class CVResult:
    yhat: np.ndarray
    fold_id: np.ndarray
    n_train: np.ndarray  # per row: training size used for its prediction
    nearest_train_km: np.ndarray  # per row: distance to the closest training country
    unseen_levels: np.ndarray  # per row: number of categorical levels unseen in its training fold


def cross_validate(
    design: pd.DataFrame,
    fold_id: np.ndarray,
    fit_predict: FitPredict = fit_predict_v1,
    *,
    dist: np.ndarray | None = None,
    buffer_km: float | None = None,
    categoricals: tuple[str, ...] = tuple(sorted(CATEGORICAL_FEATURES)),
) -> CVResult:
    """Out-of-fold predictions for every row under the fold assignment (and buffer)."""
    n = len(design)
    yhat = np.full(n, np.nan)
    n_train = np.zeros(n, dtype=int)
    nearest = np.full(n, np.nan)
    unseen = np.zeros(n, dtype=int)
    for _, test, train in iter_folds(fold_id, dist, buffer_km):
        if train.sum() < 2:
            raise ValueError("a fold has fewer than two training rows")
        tr, te = design[train], design[test]
        yhat[test] = fit_predict(tr, te)
        n_train[test] = int(train.sum())
        if dist is not None:
            nearest[test] = dist[test][:, train].min(axis=1)
        for c in categoricals:
            if c in design.columns:
                unseen[test] += (~te[c].astype(str).isin(tr[c].astype(str).unique())).to_numpy().astype(int)
    return CVResult(yhat, fold_id.copy(), n_train, nearest, unseen)


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def scorecard(
    design: pd.DataFrame,
    fitted_in_sample: np.ndarray,
    cv: CVResult,
    w: np.ndarray,
    *,
    n_permutations: int = 999,
    seed: int = 0,
    shares: bool = True,
) -> dict:
    """The fixed metric set every model stage reports.

    Args:
        design: the schema-named design (one row per country) the model was fit on.
        fitted_in_sample: in-sample fitted values (same row order).
        cv: out-of-fold predictions from :func:`cross_validate`.
        w: row-standardised spatial weights for the residual Moran's I.
        shares: also compute the V1 group LMG/Shapley shares (in-sample) with
            :func:`src.decomposition.group_lmg_shares`; ``False`` for estimators
            whose group contribution is defined elsewhere.
    """
    y = design[OUTCOME_COL].to_numpy(dtype=float)
    e_in = y - fitted_in_sample
    e_cv = y - cv.yhat
    folds = np.unique(cv.fold_id)
    fold_rmse = np.array([np.sqrt(np.mean(e_cv[cv.fold_id == f] ** 2)) for f in folds])
    fold_n = np.array([(cv.fold_id == f).sum() for f in folds])
    slope, intercept = np.polyfit(cv.yhat, y, 1)
    out = {
        "n": int(len(y)),
        "in_sample_r2": _r2(y, fitted_in_sample),
        "cv_r2": _r2(y, cv.yhat),
        "cv_rmse": float(np.sqrt(np.mean(e_cv**2))),
        "cv_mae": float(np.mean(np.abs(e_cv))),
        "in_sample_rmse": float(np.sqrt(np.mean(e_in**2))),
        "outcome_sd": float(y.std(ddof=1)),
        "n_folds": int(len(folds)),
        "fold_rmse_median": float(np.median(fold_rmse)),
        "fold_rmse_min": float(fold_rmse.min()),
        "fold_rmse_max": float(fold_rmse.max()),
        "worst_fold": int(folds[np.argmax(fold_rmse)]),
        "worst_fold_n": int(fold_n[np.argmax(fold_rmse)]),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "residual_morans_i_in_sample": morans_i(e_in, w, n_permutations, seed).statistic,
        "residual_morans_i_cv": morans_i(e_cv, w, n_permutations, seed).statistic,
        "mean_n_train": float(cv.n_train.mean()),
        "rows_with_unseen_level": int((cv.unseen_levels > 0).sum()),
        "nearest_train_km_median": float(np.nanmedian(cv.nearest_train_km)) if np.isfinite(cv.nearest_train_km).any() else None,
        "nearest_train_km_min": float(np.nanmin(cv.nearest_train_km)) if np.isfinite(cv.nearest_train_km).any() else None,
    }
    if shares:
        result = group_lmg_shares(design)
        out.update({f"share_{k}": float(v) for k, v in result.shares.items()})
        out["residual_share"] = float(result.residual_share)
    return out


def design_descriptors(design: pd.DataFrame, fold_id: np.ndarray, dist: np.ndarray, buffer_km: float | None = None) -> dict:
    """Fold sizes, outcome balance, regime coverage and leakage proxies for a fold design."""
    y = design[OUTCOME_COL].to_numpy(dtype=float)
    folds = np.unique(fold_id)
    sizes = np.array([(fold_id == f).sum() for f in folds])
    fold_means = np.array([y[fold_id == f].mean() for f in folds])
    unseen_by_cat, nearest = {}, []
    for _, test, train in iter_folds(fold_id, dist, buffer_km):
        for c in sorted(CATEGORICAL_FEATURES):
            if c in design.columns:
                te, tr = design.loc[test, c].astype(str), design.loc[train, c].astype(str)
                unseen_by_cat[c] = unseen_by_cat.get(c, 0) + int((~te.isin(tr.unique())).sum())
        nearest.append(dist[test][:, train].min(axis=1))
    nearest = np.concatenate(nearest)
    return {
        "n_folds": int(len(folds)),
        "fold_size_min": int(sizes.min()), "fold_size_median": float(np.median(sizes)), "fold_size_max": int(sizes.max()),
        "fold_mean_outcome_sd": float(fold_means.std(ddof=1)) if len(folds) > 1 else 0.0,
        "fold_mean_outcome_range": [float(fold_means.min()), float(fold_means.max())],
        "test_rows_with_unseen_level_by_categorical": unseen_by_cat,
        "nearest_train_km_median": float(np.median(nearest)),
        "nearest_train_km_p10": float(np.percentile(nearest, 10)),
        "nearest_train_km_min": float(nearest.min()),
        "share_test_rows_with_train_within_500km": float(np.mean(nearest < 500)),
    }


def v1_weights(lon: np.ndarray, lat: np.ndarray, k: int = 8) -> np.ndarray:
    """The V1 Moran's I weights (station-centroid kNN, k = 8)."""
    return knn_weights(haversine_matrix(lon, lat), k)


def region_errors(cv: CVResult, y: np.ndarray, region: pd.Series | np.ndarray) -> pd.DataFrame:
    """Out-of-fold RMSE, MAE and mean error by region (the worst-region metric of the scorecard)."""
    e = y - cv.yhat
    frame = pd.DataFrame({"region": np.asarray(region), "e": e})
    out = frame.groupby("region")["e"].agg(n="size", rmse=lambda v: float(np.sqrt(np.mean(v**2))), mae=lambda v: float(np.mean(np.abs(v))), bias="mean")
    return out.sort_values("rmse", ascending=False)
