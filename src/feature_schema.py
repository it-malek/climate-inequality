"""X-Schema v1: the project's frozen feature representation contract.

The project is a **multivariate variance-attribution system for global spatial
warming inequality**, not a CO2 causal-effect study. The outcome is
country-level warming-trend inequality; emissions is one *responsibility axis*
among several structuring axes, **not** the primary parameter.

**Scope boundary (architectural decision):** this is a single-layer, strictly
cross-sectional, *descriptive* decomposition of **observed** warming trends.
There is no physical/driver (time-series climate-attribution) layer, by design:
its output would be spatially constant across countries and so explain none of
the cross-country variance, and the project lacks the aerosol/ENSO/volcanic data
such a model needs. :data:`INTERPRETATION_NOTE` is the canonical disclaimer that
travels with every model output, and every module here imports it rather than
restating it. See ``docs/decomposition_design_memo.md`` (scope boundary).

Before any model (OLS / GAM / LMG-Shapley / robustness suite) is written, this
module fixes *which features exist and how they are grouped*, so that:

- feature groups are explicit and stable -- LMG/Shapley shares are only
  interpretable against a fixed partition of the design matrix;
- emissions is demoted to *one group of proxies*, never privileged;
- **no feature outside this schema may enter any model or decomposition**
  (enforced by :func:`validate_design_matrix`).

Maturity
--------
:data:`SCHEMA_V1` is a **candidate** contract (``maturity="candidate"``), not a
permanently frozen ontology: one revision cycle after the first decomposition
run may still adjust grouping or feature membership. The *enforcement* is
nonetheless strict today -- nothing outside the declared schema reaches a model
-- but downstream code should not assume v1's group boundaries are final. A
future revision is a **new** :data:`FeatureSchema` (``"v2"``) added to
:data:`SCHEMAS`; an immutable (frozen) ``SCHEMA_V*`` object is never mutated in
place.

Rationale for the grouping
---------------------------
- **Four structuring axes + an explicit residual** keep the Shapley shares
  interpretable: each share answers "how much warming inequality aligns with
  *this kind* of structure." Emissions, geography, socioeconomic development and
  population/urbanization are conceptually distinct mechanisms, so merging or
  re-splitting them would change the meaning of every share.
- **Emissions is one group of proxies**, deliberately not the axis -- the core
  narrative correction.
- **Geography is a first-class axis**, not a nuisance control: latitude,
  elevation, continentality, Koeppen class, hemisphere and the spatial block
  are the physical structure the decomposition is *about*.
- **socioeconomic vs population** are separated because development stage
  (adaptive capacity) and demographic/urban scale are different mechanisms the
  decomposition should be able to tell apart.
- **residual is named**, so per-group shares plus residual always sum to 1 and
  the unexplained part is a testable object (the stability layer later probes
  whether it is *spatially structured* rather than noise).

How this schema interacts with LMG/Shapley decomposition
--------------------------------------------------------
- LMG/Shapley attributes R^2 to **groups**, not individual features. The
  ``group.key`` values are the atomic units averaged over orderings, so group
  membership is part of the contract, not a model-time choice.
- Groups must **partition** the design matrix (:func:`assert_groups_disjoint`),
  so the per-group shares + residual sum to exactly 1 with no double counting.
- ``status="proposed"`` features are excluded from the live design matrix
  (:func:`feature_names` with ``status="available"``) until their data source
  is wired -- adding a source later is a reviewable schema event, not a silent
  change to what a model sees.
- Categorical features (``income_group``, ``climate_zone``, ``hemisphere``,
  ``spatial_block``) enter as a single grouped fixed-effect block and are
  attributed to their group as a unit, never per dummy level.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Feature availability states. "available" features have a wired (or trivially
# derivable) data source and form the live design matrix; "proposed" features
# are part of the contract but excluded from models until their source lands.
STATUS_AVAILABLE = "available"
STATUS_PROPOSED = "proposed"
STATUSES = (STATUS_AVAILABLE, STATUS_PROPOSED)

# Schema maturity states. "candidate" => enforced strictly but group boundaries
# may still shift in one revision cycle; "frozen" => locked for downstream use.
MATURITY_CANDIDATE = "candidate"
MATURITY_FROZEN = "frozen"

# Canonical interpretation disclaimer. The single source of truth for the
# project's scope boundary; the inequality and decomposition modules import this
# and emit it into their summary outputs so the boundary travels with the
# numbers (a dashboard or reader cannot pick up the shares without it).
INTERPRETATION_NOTE = (
    "Descriptive structural decomposition of observed warming trends across "
    "countries. Shapley/LMG group shares are a variance attribution only -- "
    "they are NOT physical or causal climate attribution, and a country's "
    "emissions share does not mean its emissions caused its warming."
)


@dataclass(frozen=True)
class FeatureSpec:
    """One feature in the contract, described at the definition level.

    ``description`` says *what the feature measures*, deliberately not how it
    is computed or sourced -- this is a representation contract, not an ETL
    spec.
    """

    name: str
    description: str
    unit: str
    status: str = STATUS_AVAILABLE

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"FeatureSpec {self.name!r}: status {self.status!r} not in {STATUSES}"
            )
        if not self.description.strip():
            raise ValueError(f"FeatureSpec {self.name!r}: empty description")
        if not self.unit.strip():
            raise ValueError(f"FeatureSpec {self.name!r}: empty unit")


@dataclass(frozen=True)
class FeatureGroup:
    """A stable, named block of features -- the atomic unit of LMG/Shapley.

    ``key`` is the identifier the decomposition reports shares against; it must
    be stable across a schema version. ``features`` may be empty only for a
    declared structural block (e.g. ``residual``).
    """

    key: str
    title: str
    rationale: str
    features: tuple[FeatureSpec, ...] = ()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)


@dataclass(frozen=True)
class FeatureSchema:
    """A versioned set of feature groups that explains one outcome."""

    version: str
    maturity: str
    unit_of_analysis: str
    outcome: FeatureSpec
    groups: tuple[FeatureGroup, ...]

    def __post_init__(self) -> None:
        if self.maturity not in (MATURITY_CANDIDATE, MATURITY_FROZEN):
            raise ValueError(f"unknown maturity {self.maturity!r}")
        assert_groups_disjoint(self)

    def group(self, key: str) -> FeatureGroup:
        for g in self.groups:
            if g.key == key:
                return g
        raise KeyError(f"no group {key!r} in schema {self.version!r}")


def assert_groups_disjoint(schema: FeatureSchema) -> None:
    """Assert no feature name appears in more than one group (LMG partition).

    Raises:
        ValueError: if any feature name is shared across groups -- Shapley
            requires the groups to partition the design matrix.
    """
    seen: dict[str, str] = {}
    for group in schema.groups:
        for name in group.feature_names:
            if name in seen:
                raise ValueError(
                    f"feature {name!r} is in both {seen[name]!r} and "
                    f"{group.key!r}; groups must partition the design matrix"
                )
            seen[name] = group.key


# ---------------------------------------------------------------------
# SCHEMA v1 (candidate contract)
# ---------------------------------------------------------------------

OUTCOME_V1 = FeatureSpec(
    name="warming_trend",
    description=(
        "Country mean 1950-2013 Theil-Sen warming slope -- the quantity whose "
        "cross-country inequality (variance) is attributed to the feature groups."
    ),
    unit="degC/decade",
)

EMISSIONS_GROUP_V1 = FeatureGroup(
    key="emissions",
    title="Emissions / industrial-activity responsibility",
    rationale=(
        "Proxies for historical industrial responsibility. One group of "
        "several structuring axes -- a responsibility axis, never a causal "
        "driver and never the privileged parameter."
    ),
    features=(
        FeatureSpec(
            "cum_co2_per_capita",
            "Cumulative production-based CO2 through the cutoff year per "
            "capita; historical emissions responsibility relative to "
            "population.",
            "t/person",
        ),
        FeatureSpec(
            "cum_co2_total",
            "Absolute cumulative production-based CO2 through the cutoff year; "
            "total scale of a country's industrial footprint.",
            "Mt",
        ),
        FeatureSpec(
            "co2_intensity_gdp",
            "CO2 emitted per unit of economic output; carbon intensity of the "
            "economy.",
            "t/USD",
            status=STATUS_PROPOSED,
        ),
    ),
)

GEOGRAPHY_GROUP_V1 = FeatureGroup(
    key="geography",
    title="Geography / physical structuring constraints",
    rationale=(
        "Where a country sits on the planet. A first-class physical axis the "
        "decomposition is about, not a nuisance control to be partialled out."
    ),
    features=(
        FeatureSpec(
            "abs_latitude",
            "Mean absolute latitude of the country's stations; distance from "
            "the equator and proxy for the polar-amplification gradient.",
            "degrees",
        ),
        FeatureSpec(
            "elevation",
            "Mean station elevation above sea level; altitude regime.",
            "m",
        ),
        FeatureSpec(
            "continentality",
            "Mean distance from the country's stations to the coastline; "
            "maritime vs continental thermal regime.",
            "km",
        ),
        FeatureSpec(
            "climate_zone",
            "Dominant Koeppen-Geiger major group (A-E); physical climate "
            "regime.",
            "category",
        ),
        FeatureSpec(
            "hemisphere",
            "Northern or Southern hemisphere; hemispheric land/ocean asymmetry "
            "in warming.",
            "category",
        ),
        FeatureSpec(
            "spatial_block",
            "Continent or spatial-cluster identifier; coarse spatial-"
            "autocorrelation block capturing regional co-structure.",
            "category",
        ),
    ),
)

SOCIOECONOMIC_GROUP_V1 = FeatureGroup(
    key="socioeconomic",
    title="Socioeconomic development structure",
    rationale=(
        "Development stage and adaptive-capacity structure -- distinct from "
        "demographic scale, so the decomposition can separate the two."
    ),
    features=(
        FeatureSpec(
            "income_group",
            "World Bank income classification (low / lower-middle / "
            "upper-middle / high); development stage.",
            "category",
        ),
        FeatureSpec(
            "gdp_per_capita",
            "Gross domestic product per capita; economic development level.",
            "USD/person",
            status=STATUS_PROPOSED,
        ),
    ),
)

POPULATION_GROUP_V1 = FeatureGroup(
    key="population",
    title="Population / urbanization structure",
    rationale=(
        "Demographic scale and how concentrated / urban a country is -- a "
        "mechanism separate from development stage."
    ),
    features=(
        FeatureSpec(
            "population",
            "Country population at the cutoff year; demographic scale.",
            "people",
        ),
        FeatureSpec(
            "urbanization_rate",
            "Share of population living in urban areas; urbanization level.",
            "fraction",
            status=STATUS_PROPOSED,
        ),
        FeatureSpec(
            "station_density",
            "Local density of city stations around a country's locations; "
            "urban-concentration and sampling-density proxy.",
            "count",
        ),
    ),
)

RESIDUAL_GROUP_V1 = FeatureGroup(
    key="residual",
    title="Residual / unobserved structured block",
    rationale=(
        "The variance not attributed to the four named groups (1 - R^2). "
        "Declared with no input features so Shapley output always reports a "
        "residual share and the stability layer can later test whether it is "
        "spatially structured (Moran's I / Conley HAC) rather than noise. No "
        "feature may be materialized here without bumping the schema version."
    ),
    features=(),
)

SCHEMA_V1 = FeatureSchema(
    version="v1",
    maturity=MATURITY_CANDIDATE,
    unit_of_analysis="country",
    outcome=OUTCOME_V1,
    groups=(
        EMISSIONS_GROUP_V1,
        GEOGRAPHY_GROUP_V1,
        SOCIOECONOMIC_GROUP_V1,
        POPULATION_GROUP_V1,
        RESIDUAL_GROUP_V1,
    ),
)

# Registry: future revisions add new keys; existing entries are never mutated.
SCHEMAS: dict[str, FeatureSchema] = {SCHEMA_V1.version: SCHEMA_V1}

# Where main() writes the human-readable mirror.
DEFAULT_YAML_PATH = Path(__file__).resolve().parents[1] / "docs" / "feature_schema_v1.yaml"


# ---------------------------------------------------------------------
# Lookups and contract enforcement
# ---------------------------------------------------------------------


def feature_names(
    schema: FeatureSchema = SCHEMA_V1,
    *,
    status: str | None = STATUS_AVAILABLE,
    groups: Collection[str] | None = None,
) -> tuple[str, ...]:
    """Feature names in the schema, optionally filtered by status and group.

    Args:
        schema: the schema to read (defaults to :data:`SCHEMA_V1`).
        status: keep only features with this status; ``None`` keeps all
            statuses. Defaults to ``"available"`` -- the live design matrix.
        groups: restrict to these group keys; ``None`` uses every group.

    Returns:
        Feature names in declaration order.
    """
    keys = set(groups) if groups is not None else None
    out: list[str] = []
    for group in schema.groups:
        if keys is not None and group.key not in keys:
            continue
        for feat in group.features:
            if status is None or feat.status == status:
                out.append(feat.name)
    return tuple(out)


def group_of(name: str, schema: FeatureSchema = SCHEMA_V1) -> str:
    """Return the group key owning feature `name`.

    Raises:
        KeyError: if `name` is not a feature in `schema`.
    """
    for group in schema.groups:
        if name in group.feature_names:
            return group.key
    raise KeyError(f"feature {name!r} is not in schema {schema.version!r}")


def validate_design_matrix(
    columns: Iterable[str],
    schema: FeatureSchema = SCHEMA_V1,
    *,
    groups: Collection[str] | None = None,
    allow: Collection[str] = (),
) -> None:
    """Enforce the contract: every design-matrix column must be a schema feature.

    The hard rule of the representation layer -- nothing outside the schema
    reaches a model. Identifier / outcome columns that are intentionally
    carried alongside the features (e.g. ``Country``, ``warming_trend``) must
    be whitelisted explicitly via `allow`, so an *accidental* stray column
    still raises.

    Args:
        columns: the design matrix's column names.
        schema: contract to check against (defaults to :data:`SCHEMA_V1`).
        groups: if given, only features from these groups are admissible (a
            model that uses a subset of axes).
        allow: non-feature columns intentionally present (ids, the outcome).

    Raises:
        ValueError: if any column is neither a schema feature (any status,
            within `groups`) nor in `allow`.
    """
    admissible = set(feature_names(schema, status=None, groups=groups)) | set(allow)
    cols = list(columns)
    unknown = [c for c in cols if c not in admissible]
    if unknown:
        raise ValueError(
            f"columns {sorted(unknown)} are not in schema {schema.version!r} "
            f"(groups={groups or 'all'}); add them to the schema or drop them "
            "-- no feature outside the contract may enter a model"
        )

    live = set(feature_names(schema, status=STATUS_AVAILABLE, groups=groups))
    missing = sorted(live - set(cols))
    if missing:
        logger.warning(
            "design matrix is missing available schema features: %s", missing
        )


# ---------------------------------------------------------------------
# YAML mirror (deterministic, dependency-free)
# ---------------------------------------------------------------------


def _yaml_scalar(value: str) -> str:
    """Double-quote a string scalar with minimal escaping (deterministic)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _spec_lines(spec: FeatureSpec, indent: str) -> list[str]:
    return [
        f"{indent}- name: {_yaml_scalar(spec.name)}",
        f"{indent}  description: {_yaml_scalar(spec.description)}",
        f"{indent}  unit: {_yaml_scalar(spec.unit)}",
        f"{indent}  status: {_yaml_scalar(spec.status)}",
    ]


def to_yaml(schema: FeatureSchema = SCHEMA_V1) -> str:
    """Serialize `schema` to a deterministic YAML string (the doc mirror).

    The Python dataclasses are the single source of truth; this is a
    review/diff artifact, regenerated by :func:`main`.
    """
    lines = [
        "# X-Schema -- generated mirror of src.feature_schema.SCHEMA_V1.",
        "# Canonical source is the Python dataclasses; regenerate via:",
        "#   uv run python -m src.feature_schema",
        f"version: {_yaml_scalar(schema.version)}",
        f"maturity: {_yaml_scalar(schema.maturity)}",
        f"unit_of_analysis: {_yaml_scalar(schema.unit_of_analysis)}",
        "outcome:",
        f"  name: {_yaml_scalar(schema.outcome.name)}",
        f"  description: {_yaml_scalar(schema.outcome.description)}",
        f"  unit: {_yaml_scalar(schema.outcome.unit)}",
        f"  status: {_yaml_scalar(schema.outcome.status)}",
        "groups:",
    ]
    for group in schema.groups:
        lines.append(f"  - key: {_yaml_scalar(group.key)}")
        lines.append(f"    title: {_yaml_scalar(group.title)}")
        lines.append(f"    rationale: {_yaml_scalar(group.rationale)}")
        if group.features:
            lines.append("    features:")
            for spec in group.features:
                lines.extend(_spec_lines(spec, indent="      "))
        else:
            lines.append("    features: []")
    return "\n".join(lines) + "\n"


def write_yaml(
    schema: FeatureSchema = SCHEMA_V1, path: Path = DEFAULT_YAML_PATH
) -> Path:
    """Write the YAML mirror of `schema` to `path` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(schema), encoding="utf-8")
    return path


def main() -> None:
    """Regenerate the YAML mirror and print a one-line summary per group."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = write_yaml()
    print(f"X-Schema {SCHEMA_V1.version} ({SCHEMA_V1.maturity}), "
          f"unit = {SCHEMA_V1.unit_of_analysis}")
    print(f"outcome: {SCHEMA_V1.outcome.name} [{SCHEMA_V1.outcome.unit}]")
    for group in SCHEMA_V1.groups:
        avail = feature_names(SCHEMA_V1, status=STATUS_AVAILABLE, groups=[group.key])
        proposed = feature_names(SCHEMA_V1, status=STATUS_PROPOSED, groups=[group.key])
        print(
            f"  {group.key:<14s} available={list(avail)} proposed={list(proposed)}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
