"""Versioned spatial replay from immutable fitted predictions.

This adapter does not fit or select models. Source result manifests stay frozen;
its output has a separate diagnostic schema and records every reused input hash.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.model_v2 import spatial
from src.correction_identity import correction_metadata
from src.prospective import prospective_outputs

REPLAY_SCHEMA = "spatial-diagnostic-replay-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(directory: Path, manifest_name: str) -> dict[str, str]:
    """Verify every deterministic result before reading its prediction table."""
    manifest = json.loads((directory / manifest_name).read_text())
    identities = {}
    for name, expected in manifest["artifact_sha256"].items():
        path = directory / name
        if path.resolve().parent != directory.resolve():
            raise ValueError(f"Artifact escapes its result directory: {name}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Historical artifact checksum mismatch: {path}")
        identities[name] = actual
    identities[manifest_name] = sha256(directory / manifest_name)
    return identities


def ordered_group(group: pd.DataFrame, country_order: list[str]) -> pd.DataFrame:
    """Require the exact country set and preserve the original weight order."""
    if group.iso3.duplicated().any() or len(set(country_order)) != len(country_order):
        raise ValueError("Duplicate country keys")
    if set(group.iso3) != set(country_order):
        raise ValueError("Prediction countries differ from spatial weight countries")
    return group.set_index("iso3").loc[country_order].reset_index()


def diagnostic(values, weights, *, permutations=999, seed=0) -> dict:
    result = spatial.morans_i(values, weights, permutations, seed)
    return {**asdict(result), "status": result.status, "seed": seed}


def prediction_diagnostics(
    predictions: pd.DataFrame,
    country_order: list[str],
    weights: dict[str, np.ndarray],
    groups: list[str],
    *,
    permutations: int = 999,
    seed: int = 0,
) -> list[dict]:
    """Recompute diagnostics from saved predictions, including M3 structural residuals.

    Missing or nonfinite evidence remains unavailable. A replay never turns an
    unavailable diagnostic into a pass, a negative finding, or a model decision.
    """
    rows = []
    for key, group in predictions.groupby(groups, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(groups, key, strict=True))
        aligned = ordered_group(group, country_order)
        residuals = {"prediction_error": aligned.observed.to_numpy(float) - aligned.prediction.to_numpy(float)}
        if labels.get("protocol") == "in_sample" and "trend" in aligned:
            residuals["structural_residual"] = aligned.observed.to_numpy(float) - aligned.trend.to_numpy(float)
        for residual_kind, residual in residuals.items():
            for weight_name, matrix in weights.items():
                rows.append({**labels, "residual_kind": residual_kind, "diagnostic_weights": weight_name,
                             "n": len(aligned), **diagnostic(residual, matrix, permutations=permutations, seed=seed)})
    return rows


@prospective_outputs(out_path="diagnostics/spatial_replay.json")
def write_replay(records: list[dict], inputs: dict, *, out_path: Path, limitations: list[str]) -> dict:
    """Write an explicitly labelled supplement, never a historical result manifest."""
    report = {**correction_metadata(), "schema": REPLAY_SCHEMA,
              "input_reuse": inputs, "records": records, "limitations": limitations,
              "selection_performed": False,
              "availability": {status: sum(row["status"] == status for row in records)
                               for status in ("available", "unavailable")}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report
