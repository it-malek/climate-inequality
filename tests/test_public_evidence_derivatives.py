"""Narrow integrity checks for labeled historical metadata derivatives."""
import copy
import hashlib
import json

import pytest

from src import evidence


NAME = "records/execution.json"
STATUS = "Labeled public derivatives of historical execution metadata; original records remain in private recovery"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


@pytest.fixture
def derivative(tmp_path):
    original = {"location": "original-location", "other_location": "retained-location", "score": 0.25}
    public = dict(original, location="data/input.csv")
    projection = dict(original, location=evidence.REDACTED)
    entry = {"original_sha256": digest(encoded(original)), "public_sha256": digest(encoded(public)),
             "protected_sha256": digest(encoded(projection)), "edits": [
                 {"pointer": "/location", "public_value": public["location"],
                  "original_value_sha256": digest(original["location"].encode()),
                  "reason": "Omit incidental execution location."}]}
    manifest = {"schema_version": 1, "prepared_on": "2026-10-23", "status": STATUS,
                "files": {NAME: entry}}
    (tmp_path / NAME).parent.mkdir(parents=True)
    (tmp_path / NAME).write_bytes(encoded(public))
    (tmp_path / evidence.PUBLIC_DERIVATIVES).parent.mkdir(parents=True)
    (tmp_path / evidence.PUBLIC_DERIVATIVES).write_bytes(encoded(manifest))
    return tmp_path, public, manifest


def save(root, public, manifest):
    (root / NAME).write_bytes(encoded(public))
    (root / evidence.PUBLIC_DERIVATIVES).write_bytes(encoded(manifest))


def test_original_bytes_need_no_derivative_waiver(tmp_path):
    path = tmp_path / NAME
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original evidence\n")
    assert evidence.verify_evidence_digest(NAME, digest(path.read_bytes()), tmp_path) == "original"
    assert evidence.verify_public_derivatives(tmp_path) == 0


def test_approved_derivative_checks_exact_original_identity(derivative):
    root, _, manifest = derivative
    expected = manifest["files"][NAME]["original_sha256"]
    assert evidence.verify_evidence_digest(NAME, expected, root) == "public derivative"
    assert evidence.verify_public_derivatives(root) == 1
    with pytest.raises(ValueError, match="Digest mismatch"):
        evidence.verify_evidence_digest(NAME, "0" * 64, root)


def test_wrong_public_bytes_are_rejected(derivative):
    root, public, manifest = derivative
    public["score"] = 0.5
    save(root, public, manifest)
    with pytest.raises(ValueError, match="digest mismatch"):
        evidence.verify_public_derivatives(root)


@pytest.mark.parametrize("key,value", [("score", 0.5), ("other_location", "data/other.csv")])
def test_unlisted_changes_fail_even_with_updated_public_digest(derivative, key, value):
    root, public, manifest = derivative
    public[key] = value
    manifest["files"][NAME]["public_sha256"] = digest(encoded(public))
    save(root, public, manifest)
    with pytest.raises(ValueError, match="protected projection mismatch"):
        evidence.verify_public_derivatives(root)


@pytest.mark.parametrize("pointer", ["location", "/bad~2escape", "/missing", "/score"])
def test_malformed_missing_or_numeric_pointer_is_rejected(derivative, pointer):
    root, public, manifest = derivative
    manifest["files"][NAME]["edits"][0]["pointer"] = pointer
    save(root, public, manifest)
    with pytest.raises(ValueError):
        evidence.verify_public_derivatives(root)


def test_duplicate_pointer_is_rejected(derivative):
    root, public, manifest = derivative
    edits = manifest["files"][NAME]["edits"]
    edits.append(copy.deepcopy(edits[0]))
    save(root, public, manifest)
    with pytest.raises(ValueError, match="Duplicate or overlapping"):
        evidence.verify_public_derivatives(root)


@pytest.mark.parametrize("name", ["../outside.json", "/outside.json", "records/../outside.json"])
def test_manifest_path_escape_is_rejected(derivative, name):
    root, public, manifest = derivative
    manifest["files"][name] = manifest["files"].pop(NAME)
    save(root, public, manifest)
    with pytest.raises(ValueError, match="escapes repository"):
        evidence.verify_public_derivatives(root)


def test_symlink_escape_is_rejected(derivative, tmp_path_factory):
    root, _, _ = derivative
    outside = tmp_path_factory.mktemp("outside") / "record.json"
    outside.write_text("{}")
    (root / NAME).unlink()
    (root / NAME).symlink_to(outside)
    with pytest.raises(ValueError, match="escapes repository"):
        evidence.verify_public_derivatives(root)


def test_public_value_must_match_the_exact_reviewed_string(derivative):
    root, public, manifest = derivative
    public["location"] = "data/changed.csv"
    manifest["files"][NAME]["public_sha256"] = digest(encoded(public))
    save(root, public, manifest)
    with pytest.raises(ValueError, match="value mismatch"):
        evidence.verify_public_derivatives(root)


def test_duplicate_manifest_json_keys_are_rejected(derivative):
    root, _, _ = derivative
    (root / evidence.PUBLIC_DERIVATIVES).write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        evidence.verify_public_derivatives(root)


def test_strict_digest_verifier_does_not_accept_a_derivative(derivative):
    from src.research_audit import verify_digest

    root, _, manifest = derivative
    with pytest.raises(ValueError, match="Digest mismatch"):
        verify_digest(root / NAME, manifest["files"][NAME]["original_sha256"])


def test_escaped_pointer_and_array_element(derivative):
    root, _, manifest = derivative
    public = {"a/b": {"~field": ["data/input.csv"]}, "score": 0.25}
    projection = {"a/b": {"~field": [evidence.REDACTED]}, "score": 0.25}
    entry = manifest["files"][NAME]
    entry["edits"][0]["pointer"] = "/a~1b/~0field/0"
    entry["public_sha256"] = digest(encoded(public))
    entry["protected_sha256"] = digest(encoded(projection))
    save(root, public, manifest)
    assert evidence.verify_public_derivatives(root) == 1
