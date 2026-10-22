# Public historical document derivative, prepared 2026-10-23. Original bytes are preserved privately. Decisions and timing remain those of this revision; unchanged passages reproduce the historical source. Exact edits and both identities are recorded in reports/public_document_derivatives.json.

"""Explicit freeze/run CLI for the fixed V1 bootstrap-population comparison.

A verified legacy-population stability stage must already exist. ``freeze``
records its descriptive input audit before ``run`` can compute new results.
No source acquisition or private machine paths are embedded in this script.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

from src import bootstrap_population as bp, stability
from src.data_io import round_floats
from src.decomposition import group_lmg_shares
from src.feature_schema import SCHEMA_V1, STATUS_AVAILABLE
from src.validation_receipts import ValidationReceipts

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs/bootstrap_population_spec.json"
METRICS = [
    "sampled_n",
    "retained_n",
    "unique_retained_countries",
    "kish_multiplicity_ess",
    "retained_block_count",
]


def write_row(stream, row):
    stream.write(json.dumps(row, allow_nan=False, default=str) + "\n")


def action(d):
    if sha(SPEC) != SPEC_SHA:
        raise ValueError("separate bootstrap population specification drift")
    spec = json.loads(SPEC.read_text())
    source = verified_stage("stability")
    current = json.loads((source / "stability_summary.json").read_text())
    started = json.loads((source / "started.json").read_text())
    code = {}
    for relative in [
        "src/stability.py",
        "src/decomposition.py",
        "src/feature_schema.py",
    ]:
        captured = source / "execution-source" / relative
        if sha(captured) != sha(ROOT / relative):
            raise ValueError("current completed estimator code differs: " + relative)
        code[relative] = sha(captured)
    for package, version in started["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("current completed runtime package differs: " + package)
    proposed = copy.deepcopy(current)
    result = {
        "spec_sha256": SPEC_SHA,
        "proposed_primary_policy_id": bp.COMPLETE_CASE_POLICY,
        "legacy_policy_id": bp.LEGACY_POLICY,
        "estimator_source_exact": code,
        "outcomes": {},
    }
    paired = None
    for outcome in spec["settings"]["outcomes"]:
        design_path = source / f"{outcome}-input-design.parquet"
        if sha(design_path) != spec["input_population_audit"][outcome]["source_sha256"]:
            raise ValueError("frozen population input changed")
        design = pd.read_parquet(design_path)
        audit = bp.population_audit(design)
        complete = bp.complete_design(design)
        complete.to_parquet(d / f"{outcome}-complete-design.parquet", index=False)
        if (
            audit["complete_country_order"]
            != spec["input_population_audit"][outcome]["complete_country_order"]
        ):
            raise ValueError("frozen complete population order changed")
        if paired is None:
            paired = audit["complete_country_order"]
        elif paired != audit["complete_country_order"]:
            raise ValueError("complete country populations are not paired")
        legacy_summary = (
            current
            if outcome == stability.PRIMARY_OUTCOME
            else current["sensitivity"][outcome]
        )
        if (
            legacy_summary["n_boot"] != 2000
            or legacy_summary["seed"] != 0
            or legacy_summary["n_countries"] != 151
            or legacy_summary["share_stability"]["block_bootstrap"]["n_boot"] != 2000
        ):
            raise ValueError("current completed fixed bootstrap settings changed")
        country_metrics = {"legacy": [], "complete_case": []}
        with (d / f"{outcome}-legacy-country-identities.jsonl").open("x") as stream:
            for draw, sample, diagnostics in bp.reconstruct_draws(
                design, n=2000, seed=0
            ):
                write_row(
                    stream,
                    {
                        "draw": draw,
                        "fit_action": "not_refitted; compare sealed current country summary",
                        **diagnostics,
                    },
                )
                country_metrics["legacy"].append(
                    {key: diagnostics[key] for key in METRICS}
                )
        block_metrics = {"legacy": [], "complete_case": []}
        with (d / f"{outcome}-block-equivalence.jsonl").open("x") as stream:

            def record_block(row):
                write_row(stream, row)
                for policy in block_metrics:
                    block_metrics[policy].append(
                        {key: row[policy][key] for key in METRICS}
                    )

            proof = bp.block_equivalence(design, n=2000, seed=1, record=record_block)
        with (d / f"{outcome}-complete-case-country-draws.jsonl").open("x") as stream:

            def record(row):
                write_row(stream, row)
                country_metrics["complete_case"].append(
                    {key: row[key] for key in METRICS}
                )

            country, failed = bp.evaluate_draws(complete, n=2000, seed=0, record=record)
        block_rows, block_failed = (
            [],
            legacy_summary["share_stability"]["block_bootstrap"]["n_failed"],
        )
        block_action = "reuse_current_completed_block_summary_exact_retained_inputs"
        if not proof["exact_all_retained_inputs_identities_weights"]:
            with (d / f"{outcome}-complete-case-block-draws.jsonl").open("x") as stream:
                block_rows, block_failed = bp.evaluate_draws(
                    complete,
                    n=2000,
                    seed=1,
                    block=True,
                    record=lambda row: write_row(stream, row),
                )
            block_action = "fresh_2000_complete_case_block_fits"
        point = group_lmg_shares(
            complete, SCHEMA_V1, status=STATUS_AVAILABLE, outcome_definition=outcome
        )
        shares = stability._share_stability(
            point, (country, failed), (block_rows, block_failed), n_boot=2000
        )
        if proof["exact_all_retained_inputs_identities_weights"]:
            shares["block_bootstrap"] = copy.deepcopy(
                legacy_summary["share_stability"]["block_bootstrap"]
            )
        target = (
            proposed
            if outcome == stability.PRIMARY_OUTCOME
            else proposed["sensitivity"][outcome]
        )
        target["share_stability"] = shares
        target["bootstrap_population"] = {
            "policy_id": bp.COMPLETE_CASE_POLICY,
            "conditional_observed_complete_countries": True,
            "sampling_n": len(complete),
            "country_seed": 0,
            "block_seed": 1,
            "n_attempted_each": 2000,
            "point_and_influence_population_unchanged": True,
            "block_action": block_action,
            "block_equivalence_proof": proof,
            "legacy_sampling_n": len(design),
            "reused_point_influence_residual_source": str(source),
            "limitations": "Descriptive conditional stability on observed complete countries; no missingness-mechanism correction or population extrapolation. Six heterogeneous continents limit block-bootstrap interpretation.",
        }
        comparison = {}
        for key, values in shares["groups"].items():
            old = legacy_summary["share_stability"]["groups"][key]
            comparison[key] = {
                "point": values["point"],
                "complete_case": {
                    name: values[name] for name in ["mean", "std", "ci_low", "ci_high"]
                },
                "legacy_saved_precision": {
                    name: old[name] for name in ["mean", "std", "ci_low", "ci_high"]
                },
                "complete_minus_legacy": {
                    name: values[name] - old[name]
                    for name in ["mean", "std", "ci_low", "ci_high"]
                },
            }
        result["outcomes"][outcome] = {
            "population_audit": audit,
            "country_sampling_diagnostics": {
                policy: bp.diagnostics_summary(rows)
                for policy, rows in country_metrics.items()
            },
            "block_sampling_diagnostics": {
                policy: bp.diagnostics_summary(rows)
                for policy, rows in block_metrics.items()
            },
            "new_country_attempts": 2000,
            "new_country_failures": failed,
            "legacy_country_failures": legacy_summary["share_stability"]["n_failed"],
            "block_equivalence": proof,
            "block_action": block_action,
            "block_failures": block_failed,
            "groups": comparison,
            "complete_case_p_geography_largest": shares["p_geography_largest"],
            "legacy_p_geography_largest": legacy_summary["share_stability"][
                "p_geography_largest"
            ],
        }
        print(
            "POPULATION",
            outcome,
            "new_country_failures",
            failed,
            "block",
            block_action,
            flush=True,
        )
    dump(d / "stability_summary_complete_case.json", round_floats(proposed))
    dump(d / "population_comparison.json", result)
    report = [
        "# Bootstrap population audit and predeclared comparison",
        "",
        "Agent-generated technical audit notes. The conditional complete-case policy was selected on method grounds before new comparison results; see the separately frozen BOOTSTRAP_POPULATION_SPEC.json.",
        "",
        "The documented country bootstrap samples151complete countries. The legacy code instead samples157area-outcome rows or154station-outcome rows and drops incomplete rows inside every all-coalition fit. Point estimates, leave-one-country-out influence and residual diagnostics already use151complete cases. There is no imputation, survey weighting, inverse-probability weighting or missingness-mechanism model.",
        "",
        "Area exclusions: Gambia, Mauritius and Guyana lack climate_zone; Bahrain, Hong Kong and Singapore lack warming_trend. Station input construction already removes the three missing-primary-outcome countries; its remaining exclusions are the same three climate_zone countries. Both complete-case designs have exactly the same151countries in the same order.",
        "",
        "Legacy country sampling is a bootstrap of the unconditional empirical input rows followed by within-fit complete-case selection. It has random retained N (expected151), even though each point estimate uses151. The proposed bootstrap is conditional on the observed complete-case population and draws exactly151rows with replacement. Duplicate countries carry ordinary frequency weights. Kish multiplicity ESS summarizes duplicate frequency concentration; it is not a spatial effective sample size or a claim of151independent countries.",
        "",
        "All2000whole-continent resamples under each outcome were checked for exact ordered retained identities, all estimator input values and country multiplicities. Identical retained inputs permit reuse of the current completed block summaries at their saved precision; a discrepancy would trigger2000new fits for the affected block arm. No old fit was rerun merely to recover unsaved historical draw arrays.",
        "",
        "All new fits use unchanged SCHEMA_V1 OLS coalitions and Shapley definitions. Failed ValueError/linear-algebra draws are logged and excluded from percentile summaries without replacement. New complete-case country draws use paired seed0 and the same151country order across outcomes. Block seed1 retains the original six-continent order.",
        "",
        "The tiny previously observed floating difference affects exact-reuse eligibility only. It does not determine which bootstrap population is scientifically appropriate. This audit changes neither retained V2 models nor the separate151-country full-rank M4 bootstrap.",
        "",
        "The primary conditional bootstrap describes perturbation stability for these observed complete countries. It does not repair missingness bias or extend inference to excluded countries. Six heterogeneous continent blocks do not establish calibrated interval coverage, and block intervals need not always be wider. Original prose calling this gap a spatial-dependence correction is stronger than the design supports.",
        "",
        "See population_comparison.json for exact exclusions, per-policy sample-size/ESS ranges, all share intervals and differences, and block reuse proofs. Per-draw JSONL files retain ordered identities, multiplicity weights, block counts, shares and any new failure reason.",
    ]
    (d / "REPORT.md").write_text("\n".join(report) + "\n")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=["freeze", "run"])
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--raw-root", type=Path, required=True)
    result.add_argument("--stability-stage", default="stability")
    result.add_argument("--stage-name", default="bootstrap-population")
    return result


def configure(args):
    global RUN, RAW, SPEC, SPEC_SHA, dump, sha, stage, verified_stage
    RUN = args.run_root.resolve(strict=True)
    RAW = args.raw_root.resolve(strict=True)
    for name in [args.stability_stage, args.stage_name]:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("stage selectors must be simple confined names")
    loaded = importlib.util.spec_from_file_location(
        "bootstrap_audit_runlib", RUN / "control/runlib.py"
    )
    if loaded is None or loaded.loader is None:
        raise ValueError("selected run recorder unavailable")
    recorder = importlib.util.module_from_spec(loaded)
    sys.modules[loaded.name] = recorder
    loaded.loader.exec_module(recorder)
    if (
        Path(recorder.ROOT).resolve() != ROOT
        or Path(recorder.RAW).resolve() != RAW
        or Path(recorder.RUN).resolve() != RUN
    ):
        raise ValueError("explicit roots disagree with initialized adapter")
    ValidationReceipts(ROOT, RUN, RAW, recorder.SPEC_HASH)
    dump, sha, stage = recorder.dump, recorder.sha, recorder.stage

    def selected_stage(name):
        return recorder.verified_stage(
            args.stability_stage if name == "stability" else name
        )

    verified_stage = selected_stage
    SPEC = RUN / "BOOTSTRAP_POPULATION_SPEC.json"
    if args.mode == "run":
        frozen = json.loads((RUN / "BOOTSTRAP_POPULATION_FREEZE.json").read_text())
        SPEC_SHA = frozen["sha256"]
        if sha(SPEC) != SPEC_SHA:
            raise ValueError("bootstrap population addendum drift")
    return recorder


def freeze():
    """Explicit command invocation freezes methods plus descriptive inputs only."""
    paths = [SPEC, RUN / "BOOTSTRAP_POPULATION_FREEZE.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError(
            "bootstrap addendum already exists; never overwrite a freeze"
        )
    template = json.loads(TEMPLATE.read_text())
    if template["input_population_audit"] != "${POPULATION_INPUT_AUDIT}":
        raise ValueError("public bootstrap method template lost its input placeholder")
    source = verified_stage("stability")
    audits = {}
    previous = None
    for outcome in template["settings"]["outcomes"]:
        path = source / f"{outcome}-input-design.parquet"
        audit = bp.population_audit(pd.read_parquet(path))
        if (
            audit["sampling_rows_legacy"]
            != template["settings"]["expected_input_rows"][outcome]
        ):
            raise ValueError(
                "audit requires the fixed legacy pre-filter population checkpoint"
            )
        if (
            audit["sampling_rows_complete_case"]
            != template["settings"]["new_country_fits"]["sample_size"]
        ):
            raise ValueError("complete-case population differs from the fixed method")
        if previous is not None and previous != audit["complete_country_order"]:
            raise ValueError("complete country populations are not paired")
        previous = audit["complete_country_order"]
        audits[outcome] = {
            "source_sha256": sha(path),
            "complete_country_order": audit["complete_country_order"],
            "descriptive_audit": audit,
        }
    template["input_population_audit"] = audits
    template["runtime_inputs"] = {"raw_root": str(RAW), "stability_stage": source.name}
    template["frozen_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with SPEC.open("x") as stream:
        json.dump(template, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with paths[1].open("x") as stream:
        json.dump(
            {
                "sha256": sha(SPEC),
                "public_template_sha256": sha(TEMPLATE),
                "original_frozen_spec_sha256": template["provenance"][
                    "original_frozen_spec_sha256"
                ],
                "status": "frozen by explicit invocation before any new comparison outcome",
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return SPEC


def main():
    args = parser().parse_args()
    configure(args)
    if args.mode == "freeze":
        print(freeze())
        return
    frozen = json.loads(SPEC.read_text())
    if frozen["runtime_inputs"] != {
        "raw_root": str(RAW),
        "stability_stage": args.stability_stage,
    }:
        raise ValueError("selected runtime inputs differ from the frozen addendum")
    source = verified_stage("stability")
    inputs = [
        SPEC,
        RUN / "BOOTSTRAP_POPULATION_FREEZE.json",
        source / "completed.json",
        source / "started.json",
        source / "stability_summary.json",
        source / "reuse-verification.json",
        *[
            source / f"{name}-input-design.parquet"
            for name in ["area_weighted", "station_weighted"]
        ],
    ]
    stage(
        args.stage_name,
        inputs,
        action,
        runner=__file__,
        dependencies=[args.stability_stage],
        sources=[
            ROOT / "src/bootstrap_population.py",
            ROOT / "src/stability.py",
            ROOT / "src/decomposition.py",
            ROOT / "src/feature_schema.py",
            ROOT / "docs/stability.md",
            TEMPLATE,
        ],
    )


if __name__ == "__main__":
    main()
