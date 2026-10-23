"""Resolve historical citations and verify explicitly labeled public derivatives."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "research/model_v2/evidence/manifest.json"
PUBLIC_DERIVATIVES = "reports/public_evidence_derivatives.json"
PUBLIC_DOCUMENTS = "reports/public_document_derivatives.json"
DOCUMENT_STATUS = "Labeled public derivatives of historical documents; original records remain in private recovery"
REDACTED = "[publication-redaction]"


def _path(relative: str, root: Path) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Invalid evidence path")
    parts = PurePosixPath(relative)
    if (parts.is_absolute() or ":" in parts.parts[0]
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError("Evidence path escapes repository")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes repository")
    return path


def _json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def evidence_path(relative: str, root: Path = ROOT) -> Path:
    """Return an original artifact or its explicitly recorded document location."""
    original = _path(relative, root)
    if original.is_file():
        return original
    manifest = _json(_path(MANIFEST, root))
    entry = manifest["documents"].get(relative)
    return _path(entry["path"], root) if entry else original


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _pointer(pointer: str) -> list[str]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Invalid derivative pointer")
    keys = pointer[1:].split("/")
    if any(re.search(r"~(?![01])", key) for key in keys):
        raise ValueError("Malformed derivative pointer escape")
    return [key.replace("~1", "/").replace("~0", "~") for key in keys]


def _parent(value, keys: list[str]):
    def key_for(container, key):
        if isinstance(container, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", key):
                raise ValueError("Invalid derivative array index")
            return int(key)
        if not isinstance(container, dict):
            raise ValueError("Derivative pointer crosses a scalar")
        return key
    try:
        for key in keys[:-1]:
            value = value[key_for(value, key)]
        key = key_for(value, keys[-1])
        leaf = value[key]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Derivative pointer does not exist") from error
    if not isinstance(leaf, str):
        raise ValueError("Derivative pointer must identify a string")
    return value, key, leaf


def _public_value(value) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if value == "[private workstation identifier omitted from public derivative]":
        return True
    path = value
    for prefix in ("${VIRTUAL_ENV}/", "${RUN_ROOT}/"):
        path = path.removeprefix(prefix)
    return ("/" in value and not PurePosixPath(path).is_absolute()
            and not re.match(r"[A-Za-z]:", path) and "\\" not in path
            and all(part not in ("", ".", "..") for part in path.split("/"))
            and "$" not in path)


def _manifest(root: Path) -> dict:
    path = _path(PUBLIC_DERIVATIVES, root)
    if not path.exists():
        return {}
    manifest = _json(path)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("prepared_on"), str)
            or manifest.get("status") != "Labeled public derivatives of historical execution metadata; original records remain in private recovery"
            or not isinstance(manifest.get("files"), dict)):
        raise ValueError("Malformed public derivative manifest")
    for relative, entry in manifest["files"].items():
        _path(relative, root)
        if (not isinstance(entry, dict)
                or not all(_digest(entry.get(field)) for field in
                           ("original_sha256", "public_sha256", "protected_sha256"))
                or not isinstance(entry.get("edits"), list) or not entry["edits"]):
            raise ValueError("Malformed public derivative entry")
        pointers = []
        for edit in entry["edits"]:
            if (not isinstance(edit, dict) or not _digest(edit.get("original_value_sha256"))
                    or not isinstance(edit.get("reason"), str) or not edit["reason"].strip()
                    or not _public_value(edit.get("public_value"))):
                raise ValueError("Malformed public derivative edit")
            keys = _pointer(edit.get("pointer"))
            if any(keys[:len(old)] == old or old[:len(keys)] == keys for old in pointers):
                raise ValueError("Duplicate or overlapping derivative pointer")
            pointers.append(keys)
    return manifest["files"]


def _verify_public(path: Path, entry: dict) -> None:
    if _sha(path.read_bytes()) != entry["public_sha256"]:
        raise ValueError(f"Public derivative digest mismatch: {path}")
    projection = _json(path)
    for edit in entry["edits"]:
        parent, key, value = _parent(projection, _pointer(edit["pointer"]))
        if value != edit["public_value"]:
            raise ValueError(f"Public derivative value mismatch: {path}")
        parent[key] = REDACTED
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    if _sha(canonical) != entry["protected_sha256"]:
        raise ValueError(f"Public derivative protected projection mismatch: {path}")


def _document_manifest(root: Path) -> dict:
    path = _path(PUBLIC_DOCUMENTS, root)
    if not path.exists():
        return {}
    manifest = _json(path)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("status") != DOCUMENT_STATUS
            or not isinstance(manifest.get("prepared_on"), str)
            or not isinstance(manifest.get("files"), dict)):
        raise ValueError("Malformed public document manifest")
    for relative, entry in manifest["files"].items():
        _path(relative, root)
        if (not isinstance(entry, dict)
                or entry.get("format") not in ("utf8-markdown", "utf8-json", "utf8-python")
                or not isinstance(entry.get("notice"), str) or not entry["notice"].strip()
                or not all(_digest(entry.get(field)) for field in
                           ("original_sha256", "public_sha256", "protected_sha256"))
                or not isinstance(entry.get("edits"), list) or not entry["edits"]):
            raise ValueError("Malformed public document entry")
    return manifest["files"]


def _verify_document(path: Path, entry: dict) -> None:
    raw = path.read_bytes()
    if _sha(raw) != entry["public_sha256"]:
        raise ValueError(f"Public document digest mismatch: {path}")
    text = raw.decode("utf-8")
    if entry["format"] == "utf8-markdown":
        if not text.startswith(entry["notice"] + "\n\n"):
            raise ValueError("Missing public document disclosure")
    elif entry["format"] == "utf8-python":
        if not text.startswith("# " + entry["notice"] + "\n\n"):
            raise ValueError("Missing public Python document disclosure")
        ast.parse(text)
    elif _json(path).get("public_derivative_note") != entry["notice"]:
        raise ValueError("Missing public JSON document disclosure")
    pieces, end, original_end, seen = [], 0, 0, set()
    for edit in entry["edits"]:
        values = [edit.get(field) for field in
                  ("public_start", "public_end", "original_start", "original_end")]
        if (any(type(value) is not int for value in values)
                or not isinstance(edit.get("public_value"), str)
                or not isinstance(edit.get("reason"), str) or not edit["reason"].strip()
                or not _digest(edit.get("original_value_sha256"))):
            raise ValueError("Malformed public document span")
        start, stop, old_start, old_stop = values
        if (not end <= start <= stop <= len(raw) or not original_end <= old_start <= old_stop
                or (start, stop) in seen):
            raise ValueError("Invalid or overlapping public document spans")
        seen.add((start, stop))
        # Each boundary must be a complete UTF-8 boundary, never half a symbol.
        raw[:start].decode("utf-8")
        raw[start:stop].decode("utf-8")
        if raw[start:stop] != edit["public_value"].encode("utf-8"):
            raise ValueError("Public document span value mismatch")
        pieces.extend((raw[end:start], REDACTED.encode("utf-8")))
        end, original_end = stop, old_stop
    pieces.append(raw[end:])
    if _sha(b"".join(pieces)) != entry["protected_sha256"]:
        raise ValueError("Public document protected projection mismatch")


def verify_public_documents(root: Path = ROOT) -> int:
    """Verify exact public bytes and all untouched document byte ranges."""
    entries = _document_manifest(root)
    for relative, entry in entries.items():
        _verify_document(_path(relative, root), entry)
    return len(entries)


def verify_evidence_digest(relative: str, expected: str, root: Path = ROOT) -> str:
    """Verify original bytes or an exact, reviewed derivative of that identity."""
    path = evidence_path(relative, root)
    if _sha(path.read_bytes()) == expected:
        return "original"
    public_relative = path.relative_to(root.resolve()).as_posix()
    metadata = _manifest(root).get(public_relative)
    document = _document_manifest(root).get(public_relative)
    if metadata is not None and document is not None:
        raise ValueError("Ambiguous public derivative identity")
    entry = metadata or document
    if entry is None or expected != entry["original_sha256"]:
        raise ValueError(f"Digest mismatch: {path}")
    if document is not None:
        _verify_document(path, entry)
    else:
        _verify_public(path, entry)
    return "public derivative"


def verify_public_derivatives(root: Path = ROOT) -> int:
    """Verify all listed public copies, including uncited execution metadata."""
    entries = _manifest(root)
    for relative, entry in entries.items():
        _verify_public(_path(relative, root), entry)
    return len(entries)


def verify_public_document_digest(relative: str, expected: str, root: Path = ROOT,
                                  required_format: str | None = None) -> str:
    """Verify a named original citation through its explicitly typed public copy."""
    entry = _document_manifest(root).get(relative)
    if (entry is None or entry["original_sha256"] != expected
            or (required_format is not None and entry["format"] != required_format)):
        raise ValueError(f"Document derivative identity or format mismatch: {relative}")
    _verify_document(_path(relative, root), entry)
    return "public derivative"
