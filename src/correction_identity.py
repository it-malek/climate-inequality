"""Prospective implementation identity, independent of frozen Model V2 research."""
from __future__ import annotations

CORRECTION_SET_ID = "prospective-corrections-2026-10-22"
EVIDENCE_SCHEMA_ID = "finite-evidence-v1"
FORECAST_METHOD_ID = "fixed-rho-conditional-t-v1"


def correction_metadata(*, forecast: bool = False) -> dict:
    metadata = {
        "correction_set_id": CORRECTION_SET_ID,
        "evidence_schema_id": EVIDENCE_SCHEMA_ID,
        "computation_status": "prospective-development",
    }
    if forecast:
        metadata["forecast_method_id"] = FORECAST_METHOD_ID
    return metadata


def summary_origin(summary: dict) -> str:
    """Label old records without rewriting or reinterpreting their saved numbers."""
    identity = summary.get("correction_set_id")
    if identity:
        return f"Prospective computation: {identity}. Not a validated scientific release."
    return "Historical / unversioned saved summary; not recomputed with the prospective corrections."
