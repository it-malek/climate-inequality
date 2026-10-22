"""Fixed prospective interpolation and row-order diagnostics (no artifact writes)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from research.model_v2.cv import folds_kmeans
from src.interpolate import (
    _to_unit_sphere, fit_variogram_parameters,
    idw_interpolate, knn_indices, ordinary_kriging_interpolate,
)

IDENTITY = ["Country", "City", "Latitude", "Longitude"]
COORDINATES = ["Latitude", "Longitude"]


def canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Stable, unique persistent source identities, with signed coordinates."""
    if frame.duplicated(IDENTITY).any():
        raise ValueError("duplicate persistent identity")
    if not np.isfinite(frame[COORDINATES].to_numpy(float)).all():
        raise ValueError("nonfinite coordinates")
    return frame.sort_values(IDENTITY, kind="stable").reset_index(drop=True)


def interpolation_folds(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Coordinate-only assignments; all colocated records remain together."""
    frame = canonical_frame(frame)
    groups = frame[COORDINATES].drop_duplicates().sort_values(COORDINATES).reset_index(drop=True)
    random = np.empty(len(groups), dtype=int)
    random[np.random.default_rng(0).permutation(len(groups))] = np.arange(len(groups)) % 5
    groups["coordinate_group_5fold"] = random
    groups["spatial_region_10fold_250km"] = folds_kmeans(
        groups.Longitude.to_numpy(), groups.Latitude.to_numpy(), 10, seed=0, n_restarts=20,
    )
    merged = frame.merge(groups, on=COORDINATES, validate="many_to_one", sort=False)
    return frame, {name: merged[name].to_numpy() for name in groups.columns[2:]}


def interpolation_masks(frame: pd.DataFrame, labels: np.ndarray, fold: int, buffer_km: float):
    """Apply spatial buffer to training rows only, using nearest test distance."""
    test = labels == fold
    train = ~test
    if buffer_km:
        tree = cKDTree(_to_unit_sphere(frame.loc[test, "Longitude"].to_numpy(), frame.loc[test, "Latitude"].to_numpy()))
        chord, _ = tree.query(_to_unit_sphere(frame.Longitude.to_numpy(), frame.Latitude.to_numpy()))
        distance = 2 * 6371.0088 * np.arcsin(np.clip(chord / 2, 0, 1))
        train &= distance > buffer_km
    return train, test


def predict_interpolation_fold(frame: pd.DataFrame, train: np.ndarray, test: np.ndarray):
    """Fit the variogram exclusively on this fold's unchanged training rows."""
    if train.sum() < 30 or not test.any():
        raise ValueError("unusable fixed fold")
    tr, te = frame.loc[train], frame.loc[test]
    args = [tr.Longitude.to_numpy(), tr.Latitude.to_numpy(), tr.slope_c_per_decade.to_numpy(), te.Longitude.to_numpy(), te.Latitude.to_numpy()]
    parameters = fit_variogram_parameters(*args[:3], variogram_model="spherical")
    idw = idw_interpolate(*args, power=2, k=30)
    # PyKrige returns partial sill internally, while list input means full sill.
    fitted = dict(zip(("psill", "range", "nugget"), parameters))
    kriging, variance = ordinary_kriging_interpolate(*args, variogram_model="spherical", variogram_parameters=fitted, k=30)
    # Record predictions and uncertainty; failed/nonfinite values remain visible to the caller.
    return {"idw": idw, "kriging": kriging}, parameters, variance


def interpolation_scores(predictions: pd.DataFrame) -> dict:
    """Equal weight per coordinate group, plus ordinary city-row scores."""
    out = {}
    for method in ("idw", "kriging"):
        errors = predictions[method] - predictions.slope_c_per_decade
        finite = np.isfinite(errors)
        groups = predictions.loc[finite, COORDINATES].copy()
        groups["squared"] = errors[finite] ** 2
        groups["absolute"] = errors[finite].abs()
        groups["bias"] = errors[finite]
        grouped = groups.groupby(COORDINATES)[["squared", "absolute", "bias"]].mean()
        out[method] = {"n_rows": int(finite.sum()), "n_groups": len(grouped), "coverage": float(finite.mean())}
        for name, values in (("coordinate_group", grouped.mean()), ("city_row", groups[["squared", "absolute", "bias"]].mean())):
            out[method][name] = {"rmse": float(np.sqrt(values.squared)), "mae": float(values.absolute), "bias": float(values.bias)}
    return out


def neighbor_graph(frame: pd.DataFrame, policy: str, k: int = 8, tolerance: float = 1e-12) -> np.ndarray:
    """Directed kNN with explicit identity exclusion and fixed boundary ties."""
    if frame.duplicated(IDENTITY).any():
        raise ValueError("duplicate persistent identity")
    k = min(k, len(frame) - 1)
    lon, lat = frame.Longitude.to_numpy(), frame.Latitude.to_numpy()
    if policy == "historical":
        idx = knn_indices(lon, lat, lon, lat, k=k + 1)
        return np.array([row[row != i][:k] for i, row in enumerate(idx)])
    if policy != "identity-distance-ties-v1":
        raise ValueError("unknown neighbor policy")
    xyz = _to_unit_sphere(lon, lat)
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=k + 1)
    rank = np.empty(len(frame), dtype=int)
    rank[frame.reset_index(drop=True).sort_values(IDENTITY, kind="stable").index] = np.arange(len(frame))
    rows = []
    for i, point in enumerate(xyz):
        boundary = distances[i, -1]
        candidates = np.array(tree.query_ball_point(point, boundary + tolerance), dtype=int)
        candidates = candidates[candidates != i]
        distance = np.linalg.norm(xyz[candidates] - point, axis=1)
        closer = candidates[distance < boundary - tolerance]
        tied = candidates[distance >= boundary - tolerance]
        chosen = np.concatenate([closer, tied[np.argsort(rank[tied])][:k - len(closer)]])
        rows.append(chosen[np.argsort(rank[chosen])])
    return np.array(rows)


def row_orders(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The fourteen declared serialized, sorted, reversed and seeded orders."""
    frame = frame.reset_index(drop=True)
    canonical = frame.sort_values(IDENTITY, kind="stable").index.to_numpy()
    return {"serialized": np.arange(len(frame)), "country_first": frame.sort_values("Country", kind="stable").index.to_numpy(), "identity_sorted": canonical, "reverse_identity": canonical[::-1], **{f"seed_{seed}": np.random.default_rng(seed).permutation(len(frame)) for seed in range(10)}}


def moran_permutations(residuals: np.ndarray, graph: np.ndarray, permutations: np.ndarray):
    """Canonical summation and exact finite two-sided permutation p-value."""
    z = residuals - residuals.mean()
    def compute(values):
        return float(np.sum(values * values[graph].mean(axis=1)) / np.sum(values ** 2))
    observed = compute(z)
    permuted = np.array([compute(z[p]) for p in permutations])
    return observed, float((1 + np.count_nonzero(abs(permuted) >= abs(observed))) / (len(permutations) + 1)), permuted


def neighbor_order_validation(frame: pd.DataFrame, residuals: np.ndarray):
    """Identity-align every graph; isolate graph and randomization order effects."""
    canonical = frame.reset_index(drop=True).sort_values(IDENTITY, kind="stable").index.to_numpy()
    rank = np.empty(len(frame), dtype=int)
    rank[canonical] = np.arange(len(frame))
    residuals = np.asarray(residuals)[canonical]
    rng = np.random.default_rng(0)
    permutations = np.array([rng.permutation(len(frame)) for _ in range(199)])
    records, graphs = [], {}
    for policy in ("historical", "identity-distance-ties-v1"):
        baseline = None
        for order_name, order in row_orders(frame).items():
            graph = neighbor_graph(frame.iloc[order].reset_index(drop=True), policy)
            aligned = np.empty_like(graph)
            aligned[rank[order]] = np.sort(rank[order[graph]], axis=1)
            if baseline is None:
                baseline = aligned.copy()
            overlap = np.array([len(set(a) & set(b)) / len(a) for a, b in zip(aligned, baseline)])
            positional = np.empty_like(permutations)
            positional[:, rank[order]] = rank[order[permutations]]
            for randomization, perms in (("identity", permutations), ("positional", positional)):
                observed, p, _ = moran_permutations(residuals, aligned, perms)
                records.append({"policy": policy, "order": order_name, "randomization": randomization, "I": observed, "p": p, "reject_alpha_005": p < .05, "changed_rows": int((overlap < 1).sum()), "mean_neighbor_overlap": float(overlap.mean())})
            graphs[f"{policy}:{order_name}"] = aligned
    result = pd.DataFrame(records)
    fixed = result[(result.policy == "identity-distance-ties-v1") & (result.randomization == "identity")]
    acceptance = {"graph_identical": all(np.array_equal(g, graphs["identity-distance-ties-v1:serialized"]) for key, g in graphs.items() if key.startswith("identity-distance")), "I_range": float(fixed.I.max() - fixed.I.min()), "p_exact": fixed.p.nunique() == 1}
    acceptance["accepted"] = acceptance["graph_identical"] and acceptance["I_range"] <= 1e-12 and acceptance["p_exact"]
    return result, graphs, acceptance
