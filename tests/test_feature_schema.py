"""Tests for the X-Schema v1 feature representation contract."""

from __future__ import annotations

import dataclasses

import pytest

from src import feature_schema as fs
from src.feature_schema import (
    SCHEMA_V1,
    SCHEMAS,
    FeatureSchema,
    FeatureSpec,
    feature_names,
    group_of,
    to_yaml,
    validate_design_matrix,
)

EXPECTED_GROUP_KEYS = (
    "emissions",
    "geography",
    "socioeconomic",
    "population",
    "residual",
)


def test_v1_has_the_five_declared_groups_in_order():
    assert tuple(g.key for g in SCHEMA_V1.groups) == EXPECTED_GROUP_KEYS
    assert SCHEMAS["v1"] is SCHEMA_V1
    # Pre-final contract, but enforced strictly.
    assert SCHEMA_V1.maturity == fs.MATURITY_CANDIDATE
    assert SCHEMA_V1.unit_of_analysis == "country"


def test_groups_partition_the_design_matrix():
    # assert_groups_disjoint runs in __post_init__; an explicit overlap raises.
    names = [n for g in SCHEMA_V1.groups for n in g.feature_names]
    assert len(names) == len(set(names)), "feature names must be unique across groups"

    dup = FeatureSpec("abs_latitude", "dup", "degrees")
    bad_geo = dataclasses.replace(
        SCHEMA_V1.group("population"),
        features=SCHEMA_V1.group("population").features + (dup,),
    )
    with pytest.raises(ValueError, match="partition"):
        dataclasses.replace(
            SCHEMA_V1,
            groups=tuple(
                bad_geo if g.key == "population" else g for g in SCHEMA_V1.groups
            ),
        )


def test_every_feature_has_description_and_unit():
    for group in SCHEMA_V1.groups:
        for spec in group.features:
            assert spec.description.strip()
            assert spec.unit.strip()
            assert spec.status in fs.STATUSES


def test_emissions_is_a_group_not_the_axis():
    # The narrative correction: emissions is one group among several, and it
    # is not the outcome.
    assert "emissions" in {g.key for g in SCHEMA_V1.groups}
    assert SCHEMA_V1.outcome.name == "warming_trend"
    assert group_of("cum_co2_per_capita") == "emissions"


def test_feature_names_status_filtering():
    available = feature_names(SCHEMA_V1, status=fs.STATUS_AVAILABLE)
    proposed = feature_names(SCHEMA_V1, status=fs.STATUS_PROPOSED)
    every = feature_names(SCHEMA_V1, status=None)

    assert "co2_intensity_gdp" in proposed
    assert "co2_intensity_gdp" not in available
    assert set(available).isdisjoint(proposed)
    assert set(every) == set(available) | set(proposed)


def test_feature_names_group_filtering():
    geo = feature_names(SCHEMA_V1, status=None, groups=["geography"])
    assert "abs_latitude" in geo
    assert "cum_co2_per_capita" not in geo


def test_residual_block_is_declared_but_empty():
    assert SCHEMA_V1.group("residual").features == ()


def test_validate_design_matrix_rejects_out_of_schema_column():
    good = list(feature_names(SCHEMA_V1, status=fs.STATUS_AVAILABLE))
    # A clean available design matrix passes.
    validate_design_matrix(good)

    with pytest.raises(ValueError, match="not in schema"):
        validate_design_matrix(good + ["gni_coefficient"])


def test_validate_design_matrix_allow_whitelist():
    cols = ["Country", "warming_trend", "abs_latitude", "cum_co2_per_capita"]
    # Without the whitelist the id/outcome columns are rejected.
    with pytest.raises(ValueError):
        validate_design_matrix(cols)
    # With them whitelisted it passes.
    validate_design_matrix(cols, allow=("Country", "warming_trend"))


def test_validate_design_matrix_group_scope():
    # A geography-only model may not carry an emissions feature.
    with pytest.raises(ValueError, match="not in schema"):
        validate_design_matrix(["abs_latitude", "cum_co2_per_capita"], groups=["geography"])


def test_validate_warns_on_missing_available_feature(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="src.feature_schema"):
        validate_design_matrix(["abs_latitude"], groups=["geography"])
    assert any("missing available schema features" in r.message for r in caplog.records)


def test_schema_objects_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SCHEMA_V1.version = "v2"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        SCHEMA_V1.groups[0].key = "x"  # type: ignore[misc]


def test_bad_status_and_empty_fields_rejected():
    with pytest.raises(ValueError):
        FeatureSpec("x", "desc", "unit", status="speculative")
    with pytest.raises(ValueError):
        FeatureSpec("x", "  ", "unit")
    with pytest.raises(ValueError):
        FeatureSpec("x", "desc", "")


def test_unknown_maturity_rejected():
    with pytest.raises(ValueError, match="maturity"):
        FeatureSchema(
            version="vX",
            maturity="eternal",
            unit_of_analysis="country",
            outcome=SCHEMA_V1.outcome,
            groups=(),
        )


def test_yaml_mirror_round_trips_names():
    text = to_yaml(SCHEMA_V1)
    # Optional dependency: only assert structure if PyYAML is importable.
    yaml = pytest.importorskip("yaml")
    parsed = yaml.safe_load(text)

    assert parsed["version"] == "v1"
    assert parsed["maturity"] == "candidate"
    assert parsed["outcome"]["name"] == "warming_trend"
    assert [g["key"] for g in parsed["groups"]] == list(EXPECTED_GROUP_KEYS)

    emissions = next(g for g in parsed["groups"] if g["key"] == "emissions")
    assert {f["name"] for f in emissions["features"]} == set(
        feature_names(SCHEMA_V1, status=None, groups=["emissions"])
    )
    residual = next(g for g in parsed["groups"] if g["key"] == "residual")
    assert residual["features"] == []


def test_group_of_unknown_raises():
    with pytest.raises(KeyError):
        group_of("not_a_feature")


def test_interpretation_note_disclaims_causal_attribution():
    note = fs.INTERPRETATION_NOTE
    assert note.strip()
    assert "NOT" in note
    assert "attribution" in note.lower()
    # The boundary must name the descriptive/variance framing explicitly.
    assert "variance attribution" in note.lower()
