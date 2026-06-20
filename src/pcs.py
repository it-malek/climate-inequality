"""PCS v1 -- the Projection Contract System: the project's frozen semantic registry.

PCS is a **pure semantic registry**, not a computation. It is the only authority for
what the two cross-country projections *mean*; no layer defines, modifies, infers, or
extends that meaning. PCS contains no formulas and no algorithms -- every field is
human-readable specification plus units. The ``computation_definition`` is prose only
(non-executable); the actual estimation lives in the (frozen, upstream) pipeline, and
the binding of a projection to its realization is an *instantiation* step
(``src.projections``), not part of this contract.

**Minimality (v1).** PCS v1 contains EXACTLY two projections:
``responsibility_index_v1`` and ``impact_index_v1``. No other projection exists in v1.

**Immutability / anti-drift.** Each contract carries a :pyattr:`ProjectionContract.definition_hash`
(sha256 over its semantic fields). The committed ``docs/pcs_v1.yaml`` mirror records the
hashes; any edit to a contract changes its hash, so silent drift is caught. A revision is
a new ``v2`` registry, never an in-place edit of a v1 contract.

**No semantic variants.** Projection names may not carry the forbidden variant suffixes
(``_enhanced``/``_variant``/``_adjusted``/``_hybrid``/``_corrected``/``_composite``);
:func:`validate_registry` enforces this and the exact-two-projections rule at import.

This module mirrors the established ``src.feature_schema`` pattern (frozen dataclasses are
the in-repo source of truth; the YAML is a deterministic, diff-able mirror), so the
registry needs no new dependency and stays byte-stable across machines.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PCS_VERSION = "v1"

# Projection identities (the only two semantic objects in v1).
RESPONSIBILITY_INDEX = "responsibility_index_v1"
IMPACT_INDEX = "impact_index_v1"
PROJECTION_NAMES: tuple[str, ...] = (RESPONSIBILITY_INDEX, IMPACT_INDEX)

# Forbidden variant suffixes -- a projection name carrying any of these is a
# semantic variant, which PCS governance prohibits (a revision is a new version).
FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    "_enhanced", "_variant", "_adjusted", "_hybrid", "_corrected", "_composite",
)

DEFAULT_PCS_YAML_PATH = Path(__file__).resolve().parents[1] / "docs" / "pcs_v1.yaml"


@dataclass(frozen=True)
class ProjectionContract:
    """One PCS projection, described purely semantically (non-executable).

    Fields are specification + units only; ``computation_definition`` is a
    human-readable description, never a formula or algorithm. ``definition_hash``
    is derived from the semantic fields for immutability checking.
    """

    name: str
    version: str
    units: str
    aggregation_rule: str
    computation_definition: str

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "units", "aggregation_rule",
                            "computation_definition"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"ProjectionContract {self.name!r}: empty {field_name}")
        bad = [s for s in FORBIDDEN_SUFFIXES if s in self.name]
        if bad:
            raise ValueError(
                f"projection name {self.name!r} carries forbidden variant "
                f"suffix(es) {bad}; PCS forbids semantic variants -- bump the version"
            )

    @property
    def definition_hash(self) -> str:
        """sha256 over the semantic fields (the immutability fingerprint)."""
        payload = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "units": self.units,
                "aggregation_rule": self.aggregation_rule,
                "computation_definition": self.computation_definition,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# PCS v1 (exactly two projections -- the minimal observable space)
# ---------------------------------------------------------------------

RESPONSIBILITY_CONTRACT = ProjectionContract(
    name=RESPONSIBILITY_INDEX,
    version=PCS_VERSION,
    units="t CO2 per capita",
    aggregation_rule="sum-then-divide",
    computation_definition=(
        "Cumulative production-based CO2 emissions summed through the analysis "
        "cutoff year, divided by cutoff-year population (tonnes per capita); a "
        "historical-responsibility projection."
    ),
)

IMPACT_CONTRACT = ProjectionContract(
    name=IMPACT_INDEX,
    version=PCS_VERSION,
    units="degC per decade",
    aggregation_rule="unweighted mean across city-locations",
    computation_definition=(
        "Country mean of the per-city-location Theil-Sen warming slopes over the "
        "analysis window (degrees Celsius per decade); a warming-exposure projection."
    ),
)

PCS_V1: dict[str, ProjectionContract] = {
    RESPONSIBILITY_CONTRACT.name: RESPONSIBILITY_CONTRACT,
    IMPACT_CONTRACT.name: IMPACT_CONTRACT,
}


def validate_registry(registry: dict[str, ProjectionContract] = PCS_V1) -> None:
    """Enforce PCS v1 governance: exactly two named projections, no variants.

    Raises:
        ValueError: if the registry does not contain exactly the two v1
            projection names (minimality + no semantic duplication), or a key
            disagrees with its contract name.
    """
    if set(registry) != set(PROJECTION_NAMES):
        raise ValueError(
            f"PCS {PCS_VERSION} must contain exactly {sorted(PROJECTION_NAMES)}, "
            f"got {sorted(registry)}"
        )
    for key, contract in registry.items():
        if key != contract.name:
            raise ValueError(f"registry key {key!r} != contract name {contract.name!r}")


# Enforce governance at import so any drift fails loudly (mirrors the
# FeatureSchema.__post_init__ partition check in src.feature_schema).
validate_registry()


# ---------------------------------------------------------------------
# YAML mirror (deterministic, dependency-free -- mirrors src.feature_schema)
# ---------------------------------------------------------------------


def _yaml_scalar(value: str) -> str:
    """Double-quote a string scalar with minimal escaping (deterministic)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def to_yaml(registry: dict[str, ProjectionContract] = PCS_V1) -> str:
    """Serialize the registry to a deterministic YAML string (the doc mirror).

    The Python dataclasses are the in-repo source of truth; this is a review /
    diff artifact (including each contract's ``definition_hash``), regenerated by
    :func:`main`.
    """
    lines = [
        "# PCS -- generated mirror of src.pcs.PCS_V1 (the semantic registry).",
        "# Canonical source is the Python dataclasses; regenerate via:",
        "#   uv run python -m src.pcs",
        f"version: {_yaml_scalar(PCS_VERSION)}",
        "projections:",
    ]
    for name in PROJECTION_NAMES:
        c = registry[name]
        lines.extend(
            [
                f"  - name: {_yaml_scalar(c.name)}",
                f"    version: {_yaml_scalar(c.version)}",
                f"    units: {_yaml_scalar(c.units)}",
                f"    aggregation_rule: {_yaml_scalar(c.aggregation_rule)}",
                f"    computation_definition: {_yaml_scalar(c.computation_definition)}",
                f"    definition_hash: {_yaml_scalar(c.definition_hash)}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_yaml(
    registry: dict[str, ProjectionContract] = PCS_V1,
    path: Path = DEFAULT_PCS_YAML_PATH,
) -> Path:
    """Write the YAML mirror of `registry` to `path` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(registry), encoding="utf-8")
    return path


def main() -> None:
    """Regenerate the YAML mirror and print a one-line summary per projection."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    validate_registry()
    out = write_yaml()
    print(f"PCS {PCS_VERSION}: {len(PCS_V1)} projections (minimal, non-executable)")
    for name in PROJECTION_NAMES:
        c = PCS_V1[name]
        print(f"  {c.name} [{c.units}] hash={c.definition_hash[:12]}…")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
