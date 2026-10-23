"""Integrity checks for exact, labeled historical document derivatives."""
import copy
import hashlib
import json

import pytest

from src import evidence


NAME = "records/design.md"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def document(tmp_path):
    original = b"# Design\n\nInternal note.\nScore 0.25.\n"
    notice = "Public historical derivative prepared 2026-10-23; original preserved privately."
    changes = [(0, 0, notice + "\n\n"),
               (original.index(b"Internal note."), original.index(b"Internal note.") + 14,
                "Recorded decision.")]
    public, projection, edits, end = b"", b"", [], 0
    for start, stop, value in changes:
        public += original[end:start]
        public_start = len(public)
        public += value.encode()
        projection += original[end:start] + evidence.REDACTED.encode()
        edits.append({"original_start": start, "original_end": stop,
                      "original_value_sha256": digest(original[start:stop]),
                      "public_start": public_start, "public_end": len(public),
                      "public_value": value, "reason": "Neutral historical description."})
        end = stop
    public += original[end:]
    projection += original[end:]
    entry = {"format": "utf8-markdown", "notice": notice,
             "original_sha256": digest(original), "public_sha256": digest(public),
             "protected_sha256": digest(projection), "edits": edits}
    manifest = {"schema_version": 1, "prepared_on": "2026-10-23",
                "status": evidence.DOCUMENT_STATUS, "files": {NAME: entry}}
    (tmp_path / NAME).parent.mkdir()
    (tmp_path / evidence.PUBLIC_DOCUMENTS).parent.mkdir()
    save(tmp_path, public, manifest)
    return tmp_path, original, public, manifest


def save(root, public, manifest):
    (root / NAME).write_bytes(public)
    (root / evidence.PUBLIC_DOCUMENTS).write_text(json.dumps(manifest))


def test_named_document_retains_original_citation(document):
    root, original, _, _ = document
    assert evidence.verify_evidence_digest(NAME, digest(original), root) == "public derivative"
    assert evidence.verify_public_documents(root) == 1
    with pytest.raises(ValueError, match="Digest mismatch"):
        evidence.verify_evidence_digest(NAME, "0" * 64, root)


def test_document_bytes_and_unlisted_numeric_text_remain_strict(document):
    root, _, public, manifest = document
    public = public.replace(b"0.25", b"0.50")
    save(root, public, manifest)
    with pytest.raises(ValueError, match="digest mismatch"):
        evidence.verify_public_documents(root)
    manifest["files"][NAME]["public_sha256"] = digest(public)
    save(root, public, manifest)
    with pytest.raises(ValueError, match="protected projection mismatch"):
        evidence.verify_public_documents(root)


@pytest.mark.parametrize("field,value", [("public_start", -1), ("public_end", 100000),
                                         ("original_start", -1), ("public_start", True),
                                         ("public_value", "Wrong value"),
                                         ("original_value_sha256", "bad")])
def test_invalid_exact_spans_are_rejected(document, field, value):
    root, _, public, manifest = document
    manifest["files"][NAME]["edits"][1][field] = value
    save(root, public, manifest)
    with pytest.raises(ValueError):
        evidence.verify_public_documents(root)


def test_overlapping_document_spans_are_rejected(document):
    root, _, public, manifest = document
    edits = manifest["files"][NAME]["edits"]
    edits.insert(1, copy.deepcopy(edits[0]))
    save(root, public, manifest)
    with pytest.raises(ValueError, match="overlapping"):
        evidence.verify_public_documents(root)


def test_missing_or_altered_disclosure_is_rejected(document):
    root, _, public, manifest = document
    manifest["files"][NAME]["notice"] = "Different disclosure."
    save(root, public, manifest)
    with pytest.raises(ValueError, match="disclosure"):
        evidence.verify_public_documents(root)


def test_unlisted_document_cannot_claim_original_identity(document):
    root, original, public, manifest = document
    manifest["files"] = {}
    save(root, public, manifest)
    with pytest.raises(ValueError, match="Digest mismatch"):
        evidence.verify_evidence_digest(NAME, digest(original), root)


def test_document_manifest_path_escape_is_rejected(document):
    root, _, public, manifest = document
    manifest["files"]["../outside.md"] = manifest["files"].pop(NAME)
    save(root, public, manifest)
    with pytest.raises(ValueError, match="escapes repository"):
        evidence.verify_public_documents(root)


@pytest.mark.parametrize("format,payload", [
    ("utf8-python", "raise RuntimeError('must not execute')\n"),
    ("utf8-json", '{"score": 0.25}\n'),
])
def test_structured_documents_verify_without_execution(document, format, payload):
    root, _, _, manifest = document
    notice = manifest["files"][NAME]["notice"]
    if format == "utf8-python":
        prefix = "# " + notice + "\n\n"
        public = (prefix + payload).encode()
    else:
        prefix = '{"public_derivative_note": ' + json.dumps(notice) + ', '
        public = (prefix + payload[1:]).encode()
    # This fixture records the entire generic disclosure insertion exactly.
    size = len(prefix.encode())
    old_start = 0 if format == "utf8-python" else 1
    start = 0 if format == "utf8-python" else 1
    edit = {"original_start": old_start, "original_end": old_start,
            "original_value_sha256": digest(b""), "public_start": start,
            "public_end": size, "public_value": public[start:size].decode(),
            "reason": "Labeled generic disclosure."}
    projection = public[:start] + evidence.REDACTED.encode() + public[size:]
    manifest["files"][NAME].update(format=format, public_sha256=digest(public),
                                  protected_sha256=digest(projection), edits=[edit])
    save(root, public, manifest)
    assert evidence.verify_public_documents(root) == 1



def test_descriptive_inventory_is_separate_from_numerical_artifacts(document):
    from src import research_audit
    root, original, public, manifest = document
    allowed = next(iter(research_audit.PUBLIC_INVENTORIES))
    target = root / allowed
    target.parent.mkdir(parents=True)
    target.write_bytes(public)
    manifest["files"][allowed] = manifest["files"].pop(NAME)
    (root / evidence.PUBLIC_DOCUMENTS).write_text(json.dumps(manifest))
    # A Markdown entry on a JSON path cannot use this archival exception.
    with pytest.raises(ValueError, match="format mismatch"):
        research_audit.verify_result_artifact(root, target, digest(original))
    notice = manifest["files"][allowed]["notice"]
    original = b'{"score": 0.25}\n'
    prefix = ('"public_derivative_note": ' + json.dumps(notice) + ', ').encode()
    public = original[:1] + prefix + original[1:]
    entry = manifest["files"][allowed]
    entry.update(format="utf8-json", original_sha256=digest(original),
                 public_sha256=digest(public),
                 protected_sha256=digest(original[:1] + evidence.REDACTED.encode() + original[1:]),
                 edits=[{"original_start": 1, "original_end": 1,
                         "original_value_sha256": digest(b""), "public_start": 1,
                         "public_end": 1 + len(prefix), "public_value": prefix.decode(),
                         "reason": "Labeled generic disclosure."}])
    target.write_bytes(public)
    (root / evidence.PUBLIC_DOCUMENTS).write_text(json.dumps(manifest))
    assert research_audit.verify_result_artifact(root, target, digest(original))
    with pytest.raises(ValueError, match="identity or format mismatch"):
        research_audit.verify_result_artifact(root, target, "0" * 64)
    numerical = target.with_name("numeric_result.json")
    numerical.write_bytes(public)
    with pytest.raises(ValueError, match="Digest mismatch"):
        research_audit.verify_result_artifact(root, numerical, digest(original))
    # Metadata-only admission is forbidden, even at one of the named paths.
    manifest["files"] = {}
    (root / evidence.PUBLIC_DOCUMENTS).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity or format mismatch"):
        research_audit.verify_result_artifact(root, target, digest(original))
