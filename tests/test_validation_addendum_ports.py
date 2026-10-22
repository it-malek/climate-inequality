"""Synthetic portability tests; no real sources, network, or bootstrap fits."""

import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from scripts import run_bootstrap_population_audit as bootstrap
from scripts import run_ndgain_update as ndgain
from scripts import build_validation_candidate as candidate
from scripts.init_current_validation import initialize
from src.validation_receipts import ValidationReceipts, dump, sha
from src.correction_identity import CORRECTION_SET_ID
from tests.test_decomposition import make_country_design

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture
def portable(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    (root / "docs").mkdir(parents=True)
    (root / "src").mkdir()
    for name in ["uv.lock", "pyproject.toml", "driver.py"]:
        (root / name).write_text("# synthetic")
    for name in ["temporal_validation.py", "vulnerability.py"]:
        (root / "src" / name).write_text("# synthetic source " + name)
    dump(
        root / "docs/current_validation_spec.json",
        {
            "inputs": {"raw_root": "${RAW_ROOT}", "previous_run": "${PREVIOUS_RUN}"},
            "provenance": {"original_frozen_spec_sha256": "synthetic"},
        },
    )
    raw = tmp_path / "separate-raw"
    (raw / "ndgain").mkdir(parents=True)
    run = root / "outputs/prospective" / CORRECTION_SET_ID / "synthetic-ports"
    initialize(run, raw, repository_root=root)
    receipts = ValidationReceipts(root, run, raw, sha(run / "VALIDATION_SPEC.json"))
    design = make_country_design(n=24)
    design.loc[0, "warming_trend"] = float("nan")

    def population_stage(directory):
        for name in ["area_weighted", "station_weighted"]:
            design.to_parquet(directory / f"{name}-input-design.parquet", index=False)

    receipts.stage("stability", [], population_stage, runner=root / "driver.py")
    receipts.stage(
        "vulnerability-uncertainty",
        [],
        lambda d: None,
        runner=root / "driver.py",
        sources=[root / "src/temporal_validation.py", root / "src/vulnerability.py"],
    )
    template = json.loads(
        (REPOSITORY / "docs/bootstrap_population_spec.json").read_text()
    )
    template["settings"]["new_country_fits"]["sample_size"] = 23
    template["settings"]["expected_input_rows"] = {
        "area_weighted": 24,
        "station_weighted": 24,
    }
    dump(root / "docs/bootstrap_population_spec.json", template)
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setattr(
        bootstrap, "TEMPLATE", root / "docs/bootstrap_population_spec.json"
    )
    monkeypatch.setattr(ndgain, "ROOT", root)
    monkeypatch.setattr(
        ndgain, "TEMPLATE", root / "docs/ndgain_updated_source_spec.json"
    )
    archive = tmp_path / "local-official-archive.zip"
    values = {
        "vulnerability": (0.4, 0.8),
        "readiness": (0.6, 0.2),
        "gain": (60.0, 20.0),
    }
    with zipfile.ZipFile(archive, "w") as zipped:
        for metric, (selected, newer) in values.items():
            frame = pd.DataFrame(
                {
                    "ISO3": ["AAA"],
                    "Name": ["Synthetic"],
                    "2022": [selected - 0.01],
                    "2023": [selected],
                    "2024": [newer],
                }
            )
            zipped.writestr(
                f"resources 2/{metric}/{metric}.csv", frame.to_csv(index=False)
            )
        zipped.writestr(
            "resources 2/indicators/2026 Codebook_ ND-GAIN Country Index.pdf",
            b"synthetic-codebook",
        )
    digest = sha(archive)
    template = json.loads(
        (REPOSITORY / "docs/ndgain_updated_source_spec.json").read_text()
    )
    template["source"]["sha256"] = digest
    dump(root / "docs/ndgain_updated_source_spec.json", template)
    monkeypatch.setattr(ndgain, "PINNED_SHA256", digest)
    old = ndgain.forensic_latest_through_2023(archive)
    old.to_csv(raw / "ndgain/ndgain_latest.csv", index=False)
    return root, run, raw, archive, digest


def bootstrap_args(mode, run, raw, *extra):
    return bootstrap.parser().parse_args(
        [mode, "--run-root", str(run), "--raw-root", str(raw), *extra]
    )


def ndgain_args(mode, run, raw, archive, digest, *extra):
    return ndgain.parser().parse_args(
        [
            mode,
            "--run-root",
            str(run),
            "--raw-root",
            str(raw),
            "--archive",
            str(archive),
            "--expected-sha256",
            digest,
            *extra,
        ]
    )


def test_bootstrap_freeze_is_explicit_input_pinned_and_exclusive(portable):
    root, run, raw, archive, digest = portable
    bootstrap.configure(bootstrap_args("freeze", run, raw))
    frozen = bootstrap.freeze()
    before = frozen.read_bytes()
    record = json.loads(before)
    assert record["input_population_audit"]["area_weighted"]["source_sha256"] == sha(
        run / "stability/area_weighted-input-design.parquet"
    )
    assert record["runtime_inputs"] == {
        "raw_root": str(raw),
        "stability_stage": "stability",
    }
    assert not (run / "bootstrap-population").exists()
    with pytest.raises(FileExistsError):
        bootstrap.freeze()
    assert frozen.read_bytes() == before
    bootstrap.configure(bootstrap_args("run", run, raw))
    frozen.write_text("{}")
    with pytest.raises(ValueError, match="addendum drift"):
        bootstrap.configure(bootstrap_args("run", run, raw))


def test_ndgain_local_copy_and_forensic_do_not_require_private_checkpoint(portable):
    root, run, raw, archive, digest = portable
    ndgain.configure(ndgain_args("freeze", run, raw, archive, digest))
    ndgain.freeze()
    with pytest.raises(FileExistsError):
        ndgain.freeze()
    ndgain.configure(ndgain_args("extract", run, raw, archive, digest))
    old_hash = sha(raw / "ndgain/ndgain_latest.csv")
    destination = run / "synthetic-local-extract"
    destination.mkdir()
    record = ndgain.extract(destination)
    copied = destination / record["release"]["archive"]
    assert sha(copied) == digest and sha(archive) == digest
    assert sha(raw / "ndgain/ndgain_latest.csv") == old_hash
    selected = pd.read_csv(destination / "ndgain_2026v2_observation2023.csv")
    assert selected.loc[0, "vulnerability"] == 0.4
    assert selected.loc[0, "ndgain_year"] == 2023
    assert not (run / "inputs-ndgain-browser").exists()
    ndgain.verify_income_reuse_sources()
    (root / "src/temporal_validation.py").write_text("# changed")
    with pytest.raises(ValueError, match="source changed"):
        ndgain.verify_income_reuse_sources()


def test_wrong_archive_or_escaping_selector_refused_before_output(portable):
    root, run, raw, archive, digest = portable
    with pytest.raises(ValueError, match="frozen updated-source release pin"):
        ndgain.configure(ndgain_args("freeze", run, raw, archive, "wrong"))
    with pytest.raises(ValueError, match="simple confined"):
        ndgain.configure(
            ndgain_args(
                "freeze", run, raw, archive, digest, "--input-stage", "../escape"
            )
        )
    with pytest.raises(ValueError, match="simple confined"):
        bootstrap.configure(
            bootstrap_args("freeze", run, raw, "--stage-name", "../escape")
        )
    assert not (run / "NDGAIN_UPDATED_SOURCE_SPEC.json").exists()
    assert not (run / "BOOTSTRAP_POPULATION_SPEC.json").exists()


def test_candidate_selectors_preserve_defaults_and_change_only_named_stages():
    assert candidate.accepted_stages() == candidate.STAGES
    selected = candidate.accepted_stages("physical", "bootstrap-comparison-retry")
    assert "physical" in selected and "physical_attempt2" not in selected
    assert (
        "bootstrap-comparison-retry" in selected
        and "bootstrap-population" not in selected
    )
    assert len(selected) == len(candidate.STAGES)
    with pytest.raises(ValueError, match="simple confined"):
        candidate.accepted_stages("../escape")


def test_public_templates_are_method_only_and_machine_path_free():
    for filename in [
        "bootstrap_population_spec.json",
        "ndgain_updated_source_spec.json",
    ]:
        text = (REPOSITORY / "docs" / filename).read_text()
        assert "/Users/" not in text and "/private/" not in text
        spec = json.loads(text)
        assert len(spec["provenance"]["original_frozen_spec_sha256"]) == 64
        assert any(
            str(value).startswith("${") for value in spec["runtime_inputs"].values()
        )
    assert (
        json.loads((REPOSITORY / "docs/bootstrap_population_spec.json").read_text())[
            "input_population_audit"
        ]
        == "${POPULATION_INPUT_AUDIT}"
    )


def test_bootstrap_freeze_rejects_already_filtered_checkpoint(portable):
    root, run, raw, archive, digest = portable
    receipts = ValidationReceipts(root, run, raw, sha(run / "VALIDATION_SPEC.json"))

    def filtered(directory):
        for name in ["area_weighted", "station_weighted"]:
            frame = pd.read_parquet(run / "stability" / f"{name}-input-design.parquet")
            frame.dropna().reset_index(drop=True).to_parquet(
                directory / f"{name}-input-design.parquet", index=False
            )

    receipts.stage("already-filtered", [], filtered, runner=root / "driver.py")
    bootstrap.configure(
        bootstrap_args("freeze", run, raw, "--stability-stage", "already-filtered")
    )
    with pytest.raises(ValueError, match="legacy pre-filter population"):
        bootstrap.freeze()
    assert not (run / "BOOTSTRAP_POPULATION_SPEC.json").exists()
