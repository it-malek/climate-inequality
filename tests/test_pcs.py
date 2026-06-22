"""Tests for src.pcs -- the PCS v1 semantic registry (minimality, immutability)."""

from __future__ import annotations

import pytest

from src import pcs


class TestRegistry:
    def test_exactly_two_projections(self):
        assert set(pcs.PCS_V1) == {"responsibility_index_v1", "impact_index_v1"}
        assert len(pcs.PCS_V1) == 2

    def test_projection_names_match_keys(self):
        for key, contract in pcs.PCS_V1.items():
            assert key == contract.name

    def test_validate_registry_rejects_wrong_size(self):
        extra = dict(pcs.PCS_V1)
        extra["impact_index_v1_extra"] = pcs.IMPACT_CONTRACT
        with pytest.raises(ValueError, match="exactly"):
            pcs.validate_registry(extra)

    def test_no_forbidden_variant_suffix(self):
        with pytest.raises(ValueError, match="forbidden variant"):
            pcs.ProjectionContract(
                name="impact_index_v1_enhanced",
                version="v1",
                units="x",
                aggregation_rule="y",
                computation_definition="z",
            )

    def test_empty_field_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            pcs.ProjectionContract(
                name="ok_index", version="v1", units="x",
                aggregation_rule="y", computation_definition="   ",
            )


class TestDefinitionHash:
    def test_hash_is_deterministic_and_sha256(self):
        c = pcs.RESPONSIBILITY_CONTRACT
        assert c.definition_hash == c.definition_hash
        assert len(c.definition_hash) == 64

    def test_hash_changes_with_semantics(self):
        base = pcs.IMPACT_CONTRACT
        altered = pcs.ProjectionContract(
            name=base.name,
            version=base.version,
            units=base.units,
            aggregation_rule=base.aggregation_rule,
            computation_definition=base.computation_definition + " (edited)",
        )
        assert altered.definition_hash != base.definition_hash


class TestYamlMirror:
    def test_committed_yaml_matches_registry(self):
        # Anti-drift: the committed mirror must equal the regenerated one.
        committed = pcs.DEFAULT_PCS_YAML_PATH.read_text(encoding="utf-8")
        assert committed == pcs.to_yaml()

    def test_yaml_records_each_hash(self):
        text = pcs.to_yaml()
        for contract in pcs.PCS_V1.values():
            assert contract.definition_hash in text


class TestRegistryV2:
    def test_wide_registry_holds_three_projections(self):
        assert set(pcs.PCS_V2) == {
            "impact_index_v1",
            "responsibility_index_consumption",
            "responsibility_index_production_matched",
        }
        # Wide: more than the v1 cap of two.
        assert len(pcs.PCS_V2) > 2

    def test_reuses_v1_impact_projection_by_reference(self):
        assert pcs.PCS_V2["impact_index_v1"] is pcs.IMPACT_CONTRACT

    def test_projection_names_match_keys(self):
        for key, contract in pcs.PCS_V2.items():
            assert key == contract.name

    def test_validate_registry_v2_rejects_drift(self):
        drifted = dict(pcs.PCS_V2)
        del drifted["responsibility_index_consumption"]
        with pytest.raises(ValueError, match="must register exactly"):
            pcs.validate_registry_v2(drifted)

    def test_v1_registry_untouched(self):
        # v2 is additive: the frozen v1 registry still passes its own governance.
        pcs.validate_registry()
        assert len(pcs.PCS_V1) == 2

    def test_new_contracts_have_distinct_hashes(self):
        hashes = {c.definition_hash for c in pcs.PCS_V2.values()}
        assert len(hashes) == 3


class TestYamlMirrorV2:
    def test_committed_v2_yaml_matches_registry(self):
        committed = pcs.DEFAULT_PCS_V2_YAML_PATH.read_text(encoding="utf-8")
        assert committed == pcs.to_yaml_v2()

    def test_v2_yaml_records_each_hash(self):
        text = pcs.to_yaml_v2()
        for contract in pcs.PCS_V2.values():
            assert contract.definition_hash in text
