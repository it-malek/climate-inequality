"""Resolve immutable scientific citations whose documents were relocated."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "research/model_v2/evidence/manifest.json"


def evidence_path(relative: str, root: Path = ROOT) -> Path:
    """Return an original artifact or its explicitly recorded document location."""
    original = root / relative
    if original.is_file():
        return original
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    entry = manifest["documents"].get(relative)
    return root / entry["path"] if entry else original
