"""Read-only verification and a generated index of the frozen research results.

Run ``python -m src.research_audit`` without raw data or third-party packages.
``--write`` updates only reports/canonical_results.json, never scientific outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.snapshot import fingerprint
from src.evidence import MANIFEST, evidence_path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = "research/model_v2/outputs"
INVENTORY = "reports/canonical_results.json"


def read_json(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def verify_digest(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Digest mismatch: {path}")


def verify_evidence(root: Path = ROOT) -> dict:
    """Verify result and recovered-document bytes, not the truth of their claims."""
    identity = fingerprint(root / "app/data", require_manifest=True)
    recovered = read_json(root, MANIFEST)
    for entry in recovered["documents"].values():
        verify_digest(root / entry["path"], entry["sha256"])
    final = read_json(root, f"{OUTPUTS}/v2_final/v2_final_record.json")
    for collection in ("sources", "verification"):
        for entry in final[collection].values():
            verify_digest(evidence_path(entry["path"], root), entry["sha256"])
    manifests = sorted((root / OUTPUTS).rglob("*_result_manifest.json"))
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for name, digest in manifest["artifact_sha256"].items():
            verify_digest(path.parent / name, digest)
    return {"bundle_sha256": identity, "result_manifests_checked": len(manifests),
            "historical_documents_checked": len(recovered["documents"])}


def resolve_pointer(value, pointer: str):
    """Resolve a JSON Pointer in a source record without copying its values."""
    for key in pointer.strip("/").split("/"):
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def canonical_results(root: Path = ROOT) -> dict:
    """Compact source index: estimates live only in the authoritative artifacts."""
    sources, rows = {}, []

    def add(source, pointer, stage, model, dataset, regime, status, interpretation):
        resolve_pointer(read_json(root, source), pointer)  # fail on a stale citation
        sources[source] = hashlib.sha256((root / source).read_bytes()).hexdigest()
        rows.append(dict(stage=stage, model=model, dataset=dataset,
                         validation_regime=regime, source_artifact=source,
                         json_pointer=pointer, status=status, interpretation=interpretation))

    source = f"{OUTPUTS}/m4_final/m4_consolidated_table.json"
    table = read_json(root, source)
    for i, row in enumerate(table["rows"]):
        add(source, f"/rows/{i}", "M4 consolidation", row["stage"], "Berkeley Earth",
            "Corrected 500 km LOCO; M49/random metrics separately named",
            "HISTORICAL" if i == 0 else "FINAL", row["status"])
    for name in table["legacy_block"]:
        escaped = name.replace("~", "~0").replace("/", "~1")
        add(source, f"/legacy_block/{escaped}", "M0", "Legacy 1-degree centres",
            "Berkeley Earth", "Approximate territorial buffer", "SUPERSEDED",
            "Does not certify 500 km footprint clearance; not a current headline result")
    source = "app/data/residual_structure_summary.json"
    summary = read_json(root, source)
    add(source, "/static", "M4 static assessment", "M0*", "Berkeley Earth",
        "In-sample, corrected LOCO, M49, random and bootstrap, as named", "FINAL",
        "Full-rank descriptive allocation; country and continent-block intervals are distinct")
    for i, row in enumerate(summary["robustness"]):
        add(source, f"/robustness/{i}", "M4 robustness", "SEM and SAR", row["product"],
            row["label"], "FINAL", "Paired country-resampling intervals; not spatially corrected")
    for key in summary["product_static"]:
        add(source, f"/product_static/{key}", "M4 product check", "M0*", key,
            "Corrected 500 km LOCO and in-sample allocation", "FINAL",
            "Static result survives; spatial predictive gain does not replicate")
    add(source, "/responsibility_comparison", "Responsibility comparison", "Spearman",
        "Berkeley Earth and ERA5, as named", "Cross-sectional association", "FINAL",
        "Different representations and country sets; not causal")
    for name, keys, dataset, regime in (
        ("stats", ("trends", "validation", "interpolation"), "Berkeley Earth",
         "City trends / post-2013 check / leave-location-out as named"),
        ("physical_summary", ("hindcast", "sensitivity"), "GISTEMP and forcings",
         "Train through 2013; 2014-2024 hindcast"),
    ):
        for key in keys:
            add(f"app/data/{name}.json", f"/{key}", "Main pipeline", name, dataset,
                regime, "FINAL", "Separate estimand from country decomposition")
    return {"schema_version": 2, "rebuild_command": "python -m src.research_audit --write",
            "analysis_window": "1950-01..2013-09 unless the source explicitly names a hindcast",
            "units": "RMSE and trends: degC/decade; temperature residuals: degC; R2/shares: unitless",
            "authority": "Index only. Values remain in the cited artifacts; no estimates are duplicated here.",
            "source_sha256": sources,
            "status_definitions": {"FINAL": "Retained result, including negative findings and sensitivities",
                                   "SUPERSEDED": "Invalid for current headline use",
                                   "HISTORICAL": "Earlier representation, not the headline allocation"},
            "results": rows}


def inventory_text(root: Path = ROOT) -> str:
    return json.dumps(canonical_results(root), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the result index only")
    args = parser.parse_args()
    verified = verify_evidence()
    expected = inventory_text()
    path = ROOT / INVENTORY
    if args.write:
        path.write_text(expected, encoding="utf-8")
    elif path.read_text(encoding="utf-8") != expected:
        raise ValueError("Canonical index is stale; inspect sources before using --write")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
