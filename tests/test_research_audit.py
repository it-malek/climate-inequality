"""Integrity of the public result index and recovered historical evidence."""
import json

import pytest

from src import research_audit as audit


def test_committed_index_matches_sources_and_evidence():
    result = audit.verify_evidence()
    assert result["historical_documents_checked"] == 5
    assert result["result_manifests_checked"] >= 9
    assert (audit.ROOT / audit.INVENTORY).read_text() == audit.inventory_text()


def test_index_keeps_superseded_scores_out_of_final_results():
    rows = audit.canonical_results()["results"]
    legacy = [r for r in rows if r["status"] == "SUPERSEDED"]
    assert len(legacy) == 1
    old = audit.resolve_pointer(audit.read_json(audit.ROOT, legacy[0]["source_artifact"]),
                                legacy[0]["json_pointer"])
    assert old["primary_cv_r2"] == pytest.approx(0.212, abs=0.001)
    corrected = [r for r in rows if r["model"] == "M0* primary total CO2"
]
    assert corrected[0]["status"] == "FINAL"
    current = audit.resolve_pointer(audit.read_json(audit.ROOT, corrected[0]["source_artifact"]),
                                    corrected[0]["json_pointer"])
    assert current["primary_cv_r2"] == pytest.approx(0.15307991296259638)


def test_changed_historical_evidence_is_rejected(tmp_path):
    manifest = audit.read_json(audit.ROOT, audit.MANIFEST)
    entry = next(iter(manifest["documents"].values()))
    name, digest = entry["path"], entry["sha256"]
    path = tmp_path / "evidence.md"
    path.write_bytes((audit.ROOT / name).read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="Digest mismatch"):
        audit.verify_digest(path, digest)


def test_dashboard_pins_match_lock():
    import tomllib

    lock = tomllib.loads((audit.ROOT / "uv.lock").read_text())
    versions = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])
    for line in (audit.ROOT / "app/requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            version = version.split(";", 1)[0].strip()
            assert version in versions[name], json.dumps({name: version})
