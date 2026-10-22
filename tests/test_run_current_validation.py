"""Synthetic portability routing; never executes scientific stage actions."""

from contextlib import nullcontext
import importlib
import json
from pathlib import Path

import pytest

from scripts import run_current_validation as driver
from scripts.init_current_validation import initialize
from src.correction_identity import CORRECTION_SET_ID
from src.validation_receipts import dump


@pytest.fixture
def portable(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    (root / "docs").mkdir(parents=True)
    for name in ["uv.lock", "pyproject.toml"]:
        (root / name).write_text("# synthetic")
    dump(
        root / "docs/current_validation_spec.json",
        {
            "inputs": {"raw_root": "${RAW_ROOT}", "previous_run": "${PREVIOUS_RUN}"},
            "provenance": {"original_frozen_spec_sha256": "test-original"},
        },
    )
    raw = tmp_path / "separate-raw-location"
    raw.mkdir()
    run = root / "outputs/prospective" / CORRECTION_SET_ID / "synthetic-portable"
    initialize(run, raw, repository_root=root)
    monkeypatch.setattr(driver, "ROOT", root)
    monkeypatch.setattr(driver, "GPW_PATH", root / "data/raw/population/pop.nc")
    monkeypatch.setattr(
        driver, "GPW_NATID_LOOKUP_PATH", root / "data/raw/population/lookup.txt"
    )
    monkeypatch.setattr(driver, "prospective_execution", lambda **kwargs: nullcontext())
    records = []
    real_configure = driver.configure

    def configure(args):
        recorder = real_configure(args)

        def capture(name, inputs, action, **kwargs):
            records.append(
                {"name": name, "inputs": inputs, "action": action.__name__, **kwargs}
            )
            return run / name

        monkeypatch.setattr(driver, "stage", capture)
        return recorder

    monkeypatch.setattr(driver, "configure", configure)
    return root, run, raw, records


def arguments(stage, run, raw, *extra):
    return driver.parser().parse_args(
        [stage, "--run-root", str(run), "--raw-root", str(raw), *extra]
    )


def test_import_and_path_routing_with_optional_references_absent(portable):
    root, run, raw, records = portable
    driver.dispatch(
        arguments(
            "preprocessing",
            run,
            raw,
            "--attempt",
            "2",
            "--dependency",
            "inputs-koppen=inputs-koppen_attempt2",
        )
    )
    record = records[0]
    assert record["name"] == "preprocessing_attempt2"
    assert record["dependencies"] == ["inputs-koppen_attempt2"]
    assert raw / "population/pop.nc" in record["inputs"]
    assert raw / "population/lookup.txt" in record["inputs"]
    assert run / "inputs-koppen_attempt2/koppen_present_0p5.nc" in record["inputs"]
    assert not any(
        path.is_relative_to(root / "data/processed") for path in record["inputs"]
    )
    assert driver.OLD is None
    assert driver.verified_reuse_sources()[0] is False
    assert driver.previous_stability_inputs() == []


def test_physical_comparison_is_explicit_optional_input(portable):
    root, run, raw, records = portable
    driver.dispatch(arguments("physical", run, raw))
    assert records[-1]["inputs"] == [run / "forcings/forcings.parquet"]
    driver.dispatch(
        arguments(
            "physical",
            run,
            raw,
            "--app-reference-root",
            str(root),
            "--stage-name",
            "physical_retry",
        )
    )
    assert records[-1]["name"] == "physical_retry"
    assert root / "app/data/physical_trajectory.parquet" in records[-1]["inputs"]


def test_ndgain_has_no_implicit_network_stage(portable):
    root, run, raw, records = portable
    driver.dispatch(arguments("inputs", run, raw))
    assert [record["action"] for record in records] == ["convert"]
    with pytest.raises(ValueError, match="explicit --retrieve-ndgain"):
        driver.dispatch(arguments("inputs-ndgain", run, raw))
    driver.dispatch(arguments("inputs-ndgain", run, raw, "--retrieve-ndgain"))
    assert records[-1]["action"] == "retrieve"
    assert (
        records[-1]["inputs"] == []
    )  # acquisition does not require an existing extract


def test_retry_and_dependency_paths_cannot_escape(portable):
    root, run, raw, records = portable
    for extra in [
        ("--stage-name", "../escape"),
        ("--dependency", "core=../escape"),
        ("--attempt", "0"),
    ]:
        with pytest.raises(ValueError):
            driver.dispatch(arguments("core", run, raw, *extra))
    assert not records
    with pytest.raises(ValueError, match="escapes"):
        driver.StagePaths(run) / "../escape"


def test_public_modules_import_without_loading_a_run():
    module = importlib.import_module("scripts.run_current_validation")
    assert callable(module.dispatch)
    assert not Path("outputs/prospective/implicit-run").exists()


@pytest.mark.parametrize("population", ["legacy", "complete-case"])
def test_absent_previous_run_uses_four_fresh_fixed_bootstrap_arms(
    portable, monkeypatch, population
):
    import numpy as np
    import pandas as pd

    root, run, raw, records = portable
    driver.configure(
        arguments("stability", run, raw, "--bootstrap-population", population)
    )
    sample = pd.DataFrame(
        {
            "Country": ["complete", "missing"],
            "warming_trend": [0.1, np.nan],
            "spatial_block": ["synthetic-block", "synthetic-block"],
        }
    )
    monkeypatch.setattr(driver.pd, "read_parquet", lambda path: sample.copy())
    monkeypatch.setattr(
        driver.explain, "load_income_groups", lambda path: sample.copy()
    )
    outcomes = [driver.stability.PRIMARY_OUTCOME, driver.stability.STATION_WEIGHTED]
    monkeypatch.setattr(
        driver.stability,
        "outcome_designs",
        lambda *args: {name: sample.copy() for name in outcomes},
    )
    monkeypatch.setattr(
        driver.stability, "country_centroids", lambda frame: sample.copy()
    )
    calls = []

    def original(design, schema, status, *, n_boot, rng, block):
        calls.append((n_boot, block, rng.random(), len(design)))
        return [{"synthetic": 1.0}], 0

    monkeypatch.setattr(driver.stability, "_bootstrap_shares", original)

    class Result:
        n_countries = 1

    def summary(
        design, centroids, *, n_boot, seed, outcome_definition, include_influence
    ):
        for block in [False, True]:
            driver.stability._bootstrap_shares(
                design,
                None,
                None,
                n_boot=n_boot,
                rng=np.random.default_rng(seed + int(block)),
                block=block,
            )
        return Result()

    monkeypatch.setattr(driver.stability, "build_stability_summary", summary)
    monkeypatch.setattr(
        driver.stability,
        "summary_payload",
        lambda *args, **kwargs: {
            "synthetic": True,
            "sensitivity": {driver.stability.STATION_WEIGHTED: {}},
        },
    )
    destination = run / "synthetic-bootstrap-orchestration"
    destination.mkdir()
    result = driver.stability_check(destination)
    assert len(calls) == 4
    assert sum(call[0] for call in calls) == 8000
    for n_boot, block, first_random, sampled_n in calls:
        assert n_boot == 2000
        assert first_random == np.random.default_rng(int(block)).random()
        assert sampled_n == (1 if population == "complete-case" else 2)
    assert all(
        value["bootstrap_action"] == "fresh_2000_per_arm" for value in result.values()
    )
    assert driver.stability._bootstrap_shares is original
    summary = json.loads((destination / "stability_summary.json").read_text())
    metadata = summary["bootstrap_population"]
    assert metadata["legacy_sampling_n"] == 2
    assert metadata["sampling_n"] == (1 if population == "complete-case" else 2)
    assert metadata["conditional_observed_complete_countries"] == (
        population == "complete-case"
    )
