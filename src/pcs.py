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

# PCS v2 -- the "wide registry". v1's exactly-two rule is a v1 invariant; v2
# governance lifts it (a registered frame may hold N >= 2 explicitly-registered
# projections), keeping semantic closure (only registered columns may appear) but
# widening it from "exactly the two" to "exactly the registered set". This is a
# new version, never an in-place edit of v1 -- PCS_V1 and validate_registry stay
# byte-identical and frozen.
PCS_VERSION_2 = "v2"
RESPONSIBILITY_CONSUMPTION_INDEX = "responsibility_index_consumption"
RESPONSIBILITY_PRODUCTION_MATCHED_INDEX = "responsibility_index_production_matched"
IMPACT_POPULATION_WEIGHTED_INDEX = "impact_index_population_weighted"
IMPACT_AREA_WEIGHTED_INDEX = "impact_index_area_weighted"
# v2 registers the v1 projections (by reference) plus the new lenses. The wide
# registry grows additively: the consumption lens (window-matched responsibility),
# the exposure lens (people-weighted impact) and the area-weighted lens (gridded
# cos-latitude impact) all live here; each artifact is a registered subset. Order
# is the canonical registry/yaml order.
PROJECTION_NAMES_V2: tuple[str, ...] = (
    IMPACT_INDEX,
    RESPONSIBILITY_CONSUMPTION_INDEX,
    RESPONSIBILITY_PRODUCTION_MATCHED_INDEX,
    RESPONSIBILITY_INDEX,
    IMPACT_POPULATION_WEIGHTED_INDEX,
    IMPACT_AREA_WEIGHTED_INDEX,
)
DEFAULT_PCS_V2_YAML_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "pcs_v2.yaml"
)


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


# ---------------------------------------------------------------------
# PCS v2 -- the wide registry (the consumption-based responsibility lens)
# ---------------------------------------------------------------------

RESPONSIBILITY_CONSUMPTION_CONTRACT = ProjectionContract(
    name=RESPONSIBILITY_CONSUMPTION_INDEX,
    version=PCS_VERSION_2,
    units="t CO2 per capita",
    aggregation_rule="sum-then-divide over the consumption-available window",
    computation_definition=(
        "Cumulative consumption-based CO2 emissions (emissions counted where goods "
        "are consumed) summed over each country's consumption-available window "
        "[first year consumption data exists .. analysis cutoff], divided by "
        "cutoff-year population (tonnes per capita); a consumption-based "
        "historical-responsibility projection."
    ),
)

RESPONSIBILITY_PRODUCTION_MATCHED_CONTRACT = ProjectionContract(
    name=RESPONSIBILITY_PRODUCTION_MATCHED_INDEX,
    version=PCS_VERSION_2,
    units="t CO2 per capita",
    aggregation_rule="sum-then-divide over the consumption-available window",
    computation_definition=(
        "Cumulative production-based CO2 emissions summed over the SAME "
        "consumption-available window as responsibility_index_consumption, divided "
        "by cutoff-year population (tonnes per capita); the window-matched "
        "production baseline for an apples-to-apples production-vs-consumption "
        "responsibility comparison."
    ),
)

IMPACT_POPULATION_WEIGHTED_CONTRACT = ProjectionContract(
    name=IMPACT_POPULATION_WEIGHTED_INDEX,
    version=PCS_VERSION_2,
    units="degC per decade",
    aggregation_rule="population-weighted mean across city-locations",
    computation_definition=(
        "Country mean of the per-city-location Theil-Sen warming slopes weighted "
        "by each location's population (sampled from a static population grid at "
        "its coordinates), degrees Celsius per decade; a people-weighted "
        "warming-exposure projection -- the warming the average resident "
        "experiences, rather than the average station."
    ),
)

IMPACT_AREA_WEIGHTED_CONTRACT = ProjectionContract(
    name=IMPACT_AREA_WEIGHTED_INDEX,
    version=PCS_VERSION_2,
    units="degC per decade",
    aggregation_rule="cos(latitude) area-weighted mean across gridded land cells",
    computation_definition=(
        "Country mean of per-grid-cell Theil-Sen warming slopes from the Berkeley "
        "Earth 1-degree gridded field over the analysis window, each cell assigned "
        "to a country via the GPW national-identifier grid and weighted by "
        "cos(latitude) so every unit of land area counts equally (degrees Celsius "
        "per decade); an area-weighted warming-exposure projection -- the warming "
        "of the average square kilometre, rather than the average station. "
        "cos(latitude) weighting is required because a temperature trend is an "
        "intensive field, the mirror of the population-count rule where it is "
        "forbidden."
    ),
)

# The wide registry: the v1 impact and responsibility projections (by reference)
# plus the consumption (window-matched responsibility), exposure (people-weighted
# impact) and area-weighted (gridded cos-latitude impact) lenses.
PCS_V2: dict[str, ProjectionContract] = {
    IMPACT_CONTRACT.name: IMPACT_CONTRACT,
    RESPONSIBILITY_CONSUMPTION_CONTRACT.name: RESPONSIBILITY_CONSUMPTION_CONTRACT,
    RESPONSIBILITY_PRODUCTION_MATCHED_CONTRACT.name: (
        RESPONSIBILITY_PRODUCTION_MATCHED_CONTRACT
    ),
    RESPONSIBILITY_CONTRACT.name: RESPONSIBILITY_CONTRACT,
    IMPACT_POPULATION_WEIGHTED_CONTRACT.name: IMPACT_POPULATION_WEIGHTED_CONTRACT,
    IMPACT_AREA_WEIGHTED_CONTRACT.name: IMPACT_AREA_WEIGHTED_CONTRACT,
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


def validate_registry_v2(registry: dict[str, ProjectionContract] = PCS_V2) -> None:
    """Enforce PCS v2 (wide-registry) governance: the registered set, no variants.

    Unlike :func:`validate_registry`, v2 does **not** cap the count at two -- a
    wide registry may hold N >= 2 projections. It still enforces the stable
    registered name set (anti-drift) and key==name; the no-variant-suffix rule is
    enforced per contract at construction (:meth:`ProjectionContract.__post_init__`).

    Raises:
        ValueError: if the registry's names disagree with :data:`PROJECTION_NAMES_V2`,
            or a key disagrees with its contract name.
    """
    if set(registry) != set(PROJECTION_NAMES_V2):
        raise ValueError(
            f"PCS {PCS_VERSION_2} must register exactly "
            f"{sorted(PROJECTION_NAMES_V2)}, got {sorted(registry)}"
        )
    for key, contract in registry.items():
        if key != contract.name:
            raise ValueError(f"registry key {key!r} != contract name {contract.name!r}")


# Enforce governance at import so any drift fails loudly (mirrors the
# FeatureSchema.__post_init__ partition check in src.feature_schema).
validate_registry()
validate_registry_v2()


# ---------------------------------------------------------------------
# YAML mirror (deterministic, dependency-free -- mirrors src.feature_schema)
# ---------------------------------------------------------------------


def _yaml_scalar(value: str) -> str:
    """Double-quote a string scalar with minimal escaping (deterministic)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def to_yaml(
    registry: dict[str, ProjectionContract] = PCS_V1,
    version: str = PCS_VERSION,
    names: tuple[str, ...] = PROJECTION_NAMES,
    header: str = "# PCS -- generated mirror of src.pcs.PCS_V1 (the semantic registry).",
) -> str:
    """Serialize a registry to a deterministic YAML string (the doc mirror).

    The Python dataclasses are the in-repo source of truth; this is a review /
    diff artifact (including each contract's ``definition_hash``), regenerated by
    :func:`main`. Defaults serialize the v1 registry byte-for-byte; pass the v2
    arguments (and v2 `header`) to mirror the wide registry.
    """
    lines = [
        header,
        "# Canonical source is the Python dataclasses; regenerate via:",
        "#   uv run python -m src.pcs",
        f"version: {_yaml_scalar(version)}",
        "projections:",
    ]
    for name in names:
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
    version: str = PCS_VERSION,
    names: tuple[str, ...] = PROJECTION_NAMES,
    header: str = "# PCS -- generated mirror of src.pcs.PCS_V1 (the semantic registry).",
) -> Path:
    """Write the YAML mirror of `registry` to `path` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(registry, version, names, header), encoding="utf-8")
    return path


_PCS_V2_HEADER = "# PCS -- generated mirror of src.pcs.PCS_V2 (the wide semantic registry)."


def to_yaml_v2() -> str:
    """Deterministic YAML mirror of the v2 wide registry."""
    return to_yaml(PCS_V2, PCS_VERSION_2, PROJECTION_NAMES_V2, _PCS_V2_HEADER)


def main() -> None:
    """Regenerate the YAML mirrors and print a one-line summary per projection."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    validate_registry()
    validate_registry_v2()
    out = write_yaml()
    out_v2 = write_yaml(
        PCS_V2, DEFAULT_PCS_V2_YAML_PATH, PCS_VERSION_2, PROJECTION_NAMES_V2,
        _PCS_V2_HEADER,
    )
    print(f"PCS {PCS_VERSION}: {len(PCS_V1)} projections (minimal, non-executable)")
    for name in PROJECTION_NAMES:
        c = PCS_V1[name]
        print(f"  {c.name} [{c.units}] hash={c.definition_hash[:12]}…")
    print(f"PCS {PCS_VERSION_2}: {len(PCS_V2)} projections (wide registry)")
    for name in PROJECTION_NAMES_V2:
        c = PCS_V2[name]
        print(f"  {c.name} [{c.units}] hash={c.definition_hash[:12]}…")
    print(f"wrote {out} and {out_v2}")


if __name__ == "__main__":
    main()
