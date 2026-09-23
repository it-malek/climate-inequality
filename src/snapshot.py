"""Content identity and integrity checks for the fixed dashboard snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = "release.json"


def fingerprint(directory: Path, *, require_manifest: bool = False) -> str:
    """Hash actual bundle bytes, rejecting incomplete or mismatched releases.

    Unmanifested directories support building and inspecting partial bundles.
    The public entrypoint requires a manifest before rendering any results.
    """
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.suffix in {".json", ".parquet"} and p.name != MANIFEST
    }
    identity = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = directory / MANIFEST
    if path.exists():
        release = json.loads(path.read_text())
        if release.get("schema_version") != 1:
            raise ValueError("Unsupported dashboard release schema")
        if release.get("files") != files or release.get("bundle_sha256") != identity:
            raise ValueError("Dashboard bundle is missing or mismatched with its release manifest")
    elif require_manifest:
        raise FileNotFoundError("Dashboard release manifest is missing")
    return identity


def write_manifest(directory: Path, version: str = "2.0.0") -> dict:
    """Record an explicitly built bundle; this is never run by the dashboard."""
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.suffix in {".json", ".parquet"} and p.name != MANIFEST
    }
    release = {
        "schema_version": 1, "version": version,
        "analysis_period": "1950-01-01..2013-09-01",
        "bundle_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "files": files,
    }
    (directory / MANIFEST).write_text(json.dumps(release, indent=2) + "\n")
    return release
