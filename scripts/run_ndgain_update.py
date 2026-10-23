"""Run the pinned updated-source ND-GAIN analysis from a local official archive.

No network acquisition occurs. Explicitly freeze the public method template
before extracting or calculating; preserve the historical prepared extract.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import vulnerability as vu, temporal_validation as tv
from src.ndgain_update import archive_snapshot, revision_table
from src.prospective import prospective_execution
from src.validation_receipts import ValidationReceipts

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs/ndgain_updated_source_spec.json"
PINNED_SHA256 = "e854c7fe6ccef8c5d24a5ef7f370b7f61d1b8c6816da19fda4cd892ca6cdf0de"


def forensic_latest_through_2023(archive):
    """Replay the existing latest-nonmissing rule on source columns through2023."""
    frames = {}
    with zipfile.ZipFile(archive) as source:
        for metric in ["vulnerability", "readiness", "gain"]:
            frame = pd.read_csv(source.open(f"resources 2/{metric}/{metric}.csv"))
            keep = [
                column
                for column in frame
                if not str(column).isdigit() or int(column) <= 2023
            ]
            frames[metric] = frame[keep]
    return vu.derive_ndgain_latest(frames)


def extract(d):
    spec = json.loads(SPEC.read_text())
    checksum = spec["source"]["sha256"]
    local_archive = d / spec["source"]["archive"]
    shutil.copy2(ARCHIVE, local_archive)
    if sha(local_archive) != checksum:
        raise ValueError("copied archive checksum changed")
    current, meta = archive_snapshot(
        local_archive, expected_sha256=checksum, year=2023, prefix="resources 2"
    )
    next_year, meta2024 = archive_snapshot(
        local_archive, expected_sha256=checksum, year=2024, prefix="resources 2"
    )
    current.to_csv(d / "ndgain_2026v2_observation2023.csv", index=False)
    old = pd.read_csv(RAW / "ndgain/ndgain_latest.csv")
    revisions = revision_table(old, current)
    revisions.to_csv(d / "same-year-revisions.csv", index=False)
    additions = revision_table(current, next_year)
    additions.to_csv(d / "new2024-observations-vs2023.csv", index=False)
    forensic = forensic_latest_through_2023(local_archive)
    forensic.to_csv(d / "forensic-latest-through-2023.csv", index=False)
    pd.testing.assert_frame_equal(
        current, forensic, check_dtype=False, check_exact=True
    )
    with zipfile.ZipFile(local_archive) as z:
        (d / "ndgain-2026-codebook.pdf").write_bytes(
            z.read("resources 2/indicators/2026 Codebook_ ND-GAIN Country Index.pdf")
        )
    record = {
        "analysis_id": spec["analysis_id"],
        "release": spec["source"],
        "selected_observation_year": 2023,
        "extraction": meta,
        "latest_year_not_selected": meta2024,
        "historical_latest_rule_same_as_exact2023_for_this_archive": True,
        "comparison": {
            metric: {
                "existing_values_revised": int(
                    (
                        (revisions[metric + "_status"] == "overlapping_values")
                        & (abs(revisions[metric + "_change"]) > 0)
                    ).sum()
                ),
                "newly_available": revisions.loc[
                    revisions[metric + "_status"] == "newly_available", "iso3"
                ].tolist(),
                "now_missing": revisions.loc[
                    revisions[metric + "_status"] == "now_missing", "iso3"
                ].tolist(),
                "max_abs_revision": float(revisions[metric + "_change"].abs().max()),
            }
            for metric in ["vulnerability", "readiness", "gain"]
        },
        "methodology": spec["definition_check"],
        "official_sources": [
            "https://gain.nd.edu/news/nd-gain-country-index-2026-world-bank-business-ready-integration/",
            "https://gain.nd.edu/our-work/country-index/methodology/",
            "https://gain.nd.edu/assets/581554/nd_gain_countryindex_technicalreport_2024.pdf",
        ],
        "historical_recovery": "Original June-documented archive not found in knownlocaldownloads/projects/code/documents/recovery snapshots; officialcurrentlink delivers revisedv2; officialdata portal certificate mismatch was not bypassed. Historical filename andretrievalmonth are documented butbyteidentity notverified. No historicalcomponentrevisionlog explainingallvulnerabilitychanges located.",
    }
    dump(d / "source-record.json", record)
    return record


def calculate(d):
    result = vu.build_vulnerability(
        inequality_path=RUN / "preprocessing/country_inequality.parquet",
        income_path=RAW / "worldbank/world-bank-income-groups.csv",
        ndgain_path=INPUT_STAGE / "ndgain_2026v2_observation2023.csv",
        co2_path=RAW / "owid/owid-co2-data.csv",
        summary_path=d / "vulnerability_summary.json",
        table_path=d / "vulnerability_strata.parquet",
    )
    frame = (
        result["table"]
        if "table" in result
        else pd.read_parquet(d / "vulnerability_strata.parquet")
    )
    frame["Country"] = frame.owid_country
    old = pd.read_parquet(RUN / "vulnerability/vulnerability_strata.parquet")
    old["Country"] = old.owid_country
    cols = [vu.RESPONSIBILITY_COL, *vu.EXPOSURE_LENSES.values()]
    axes = [vu.INCOME_RANK_COL, vu.VULN_SCORE_COL]
    common = frame.loc[np.isfinite(frame[cols + axes]).all(axis=1)]
    oldcommon = old.loc[np.isfinite(old[cols + axes]).all(axis=1)]
    oldsummary = json.loads((INCOME_STAGE / "summary.json").read_text())
    olddraw = pd.read_parquet(INCOME_STAGE / "all-bootstrap-draws.parquet")
    summaries = []
    draws = []
    reuse = []
    for sample, data, prior in [
        ("pair_specific", frame, old),
        ("common_complete", common, oldcommon),
    ]:
        for axis in axes:
            for col in cols:
                pair = data.loc[
                    np.isfinite(data[axis]) & np.isfinite(data[col]),
                    ["Country", "continent", axis, col],
                ].reset_index(drop=True)
                previous = prior.loc[
                    np.isfinite(prior[axis]) & np.isfinite(prior[col]),
                    ["Country", "continent", axis, col],
                ].reset_index(drop=True)
                exact = pair.equals(previous)
                if exact:
                    record = next(
                        x.copy()
                        for x in oldsummary["evaluations"]
                        if x["sample"] == sample and x["x"] == axis and x["y"] == col
                    )
                    draw = olddraw.loc[
                        (olddraw["sample"] == sample)
                        & (olddraw.x == axis)
                        & (olddraw.y == col)
                    ].copy()
                    assert len(draw) == 4000
                    reuse.append(
                        {
                            "sample": sample,
                            "x": axis,
                            "y": col,
                            "exact_ordered_id_values_missingness_blocks": True,
                            "draws": 4000,
                            "source_stage": INCOME_STAGE.name,
                        }
                    )
                else:
                    record, draw = tv.rank_uncertainty(data, axis, col, n=2000)
                    record["sample"] = sample
                    draw = draw.assign(sample=sample, x=axis, y=col)
                summaries.append(record)
                draws.append(draw)
    pd.concat(draws, ignore_index=True).to_parquet(
        d / "all-bootstrap-draws.parquet", index=False
    )
    source = json.loads((INPUT_STAGE / "source-record.json").read_text())
    dump(
        d / "uncertainty.json",
        {
            "evaluations": summaries,
            "common_n": len(common),
            "reused_exact_inputs": reuse,
            "source_analysis_id": source["analysis_id"],
        },
    )
    matched_ids = sorted(set(common.Country) & set(oldcommon.Country))
    same_country = []
    for metric in cols:
        for name, data in [("historical", old), ("updated_source", frame)]:
            block = data.set_index("Country").loc[matched_ids]
            same_country.append(
                {
                    "vintage": name,
                    "quantity": metric,
                    "n": len(block),
                    "rho": float(
                        spearmanr(block[vu.VULN_SCORE_COL], block[metric]).statistic
                    ),
                }
            )
    dump(
        d / "vintage-comparison.json",
        {
            "matched_country_ids": matched_ids,
            "matched_complete_n": len(matched_ids),
            "same_country_correlations": same_country,
            "historical_summary": json.loads(
                (RUN / "vulnerability/vulnerability_summary.json").read_text()
            ),
            "updated_summary": result["summary"],
            "note": "Same2023observationcrosssection, differentreleasevintage. 2024notused. Incomecriteriaunchanged exceptanyexplicitcommonsamplecoverageeffect; noattributionofvulnerabilityrevisions toBREADY.",
        },
    )
    return {
        "common_n": len(common),
        "matched_n": len(matched_ids),
        "reuse_draws": sum(x["draws"] for x in reuse),
        "new_draws": 64000 - sum(x["draws"] for x in reuse),
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=["freeze", "extract", "calculate"])
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--raw-root", type=Path, required=True)
    result.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Already downloaded local official archive; never acquired by this command",
    )
    result.add_argument("--expected-sha256", default=PINNED_SHA256)
    result.add_argument(
        "--income-draw-stage",
        default="vulnerability-uncertainty",
        help="Verified reusable checkpoint name inside the selected run",
    )
    result.add_argument("--input-stage", default="ndgain-updated-input")
    result.add_argument("--output-stage", default="vulnerability-updated")
    return result


def configure(args):
    global \
        RUN, \
        RAW, \
        ARCHIVE, \
        SPEC, \
        EXPECTED, \
        INPUT_STAGE, \
        INCOME_STAGE, \
        dump, \
        sha, \
        stage, \
        verified_stage
    RUN, RAW = args.run_root.resolve(strict=True), args.raw_root.resolve(strict=True)
    ARCHIVE = args.archive.resolve(strict=True)
    for name in [args.income_draw_stage, args.input_stage, args.output_stage]:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("stage selectors must be simple confined names")
    if args.expected_sha256 != PINNED_SHA256:
        raise ValueError(
            "expected checksum must equal the frozen updated-source release pin"
        )
    loaded = importlib.util.spec_from_file_location(
        "ndgain_update_runlib", RUN / "control/runlib.py"
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
    dump, sha, stage, verified_stage = (
        recorder.dump,
        recorder.sha,
        recorder.stage,
        recorder.verified_stage,
    )
    if sha(ARCHIVE) != args.expected_sha256:
        raise ValueError("local archive checksum differs from the pinned release")
    SPEC = RUN / "NDGAIN_UPDATED_SOURCE_SPEC.json"
    INPUT_STAGE = RUN / args.input_stage
    INCOME_STAGE = RUN / args.income_draw_stage
    if args.mode != "freeze":
        EXPECTED = json.loads((RUN / "NDGAIN_UPDATE_FREEZE.json").read_text())["sha256"]
        if sha(SPEC) != EXPECTED:
            raise ValueError("updated-source frozen specification drift")
        selected = json.loads(SPEC.read_text())["runtime_inputs"]
        if (
            selected["raw_root"] != str(RAW)
            or selected["archive_path"] != str(ARCHIVE)
            or selected["income_draw_source_stage"] != args.income_draw_stage
        ):
            raise ValueError("selected runtime inputs differ from the frozen addendum")
        if selected["preserved_extract_sha256"] != sha(
            RAW / "ndgain/ndgain_latest.csv"
        ):
            raise ValueError(
                "preserved historical extract changed after addendum freeze"
            )
        if selected["income_draw_receipt_sha256"] != sha(
            INCOME_STAGE / "completed.json"
        ):
            raise ValueError("selected reusable income checkpoint changed after freeze")
    return recorder


def freeze():
    """Freeze the pinned local source and explicit reusable checkpoint before analysis."""
    paths = [SPEC, RUN / "NDGAIN_UPDATE_FREEZE.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError("updated-source addendum already exists; never overwrite")
    income = verified_stage(INCOME_STAGE.name)
    template = json.loads(TEMPLATE.read_text())
    if template["source"]["sha256"] != PINNED_SHA256:
        raise ValueError(
            "public updated-source method template has an unexpected archive pin"
        )
    template["runtime_inputs"] = {
        "raw_root": str(RAW),
        "archive_path": str(ARCHIVE),
        "income_draw_source_stage": income.name,
        "preserved_extract_sha256": sha(RAW / "ndgain/ndgain_latest.csv"),
        "income_draw_receipt_sha256": sha(income / "completed.json"),
    }
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
                "status": "frozen by explicit invocation before updated-source stages",
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return SPEC


def verify_income_reuse_sources():
    """No relaxed draw reuse when code, runtime, outputs or paired inputs differ."""
    source = verified_stage(INCOME_STAGE.name)
    started = json.loads((source / "started.json").read_text())
    for relative in ["src/temporal_validation.py", "src/vulnerability.py"]:
        if sha(source / "execution-source" / relative) != sha(ROOT / relative):
            raise ValueError(
                "selected reusable uncertainty estimator source changed: " + relative
            )
    for package, version in started["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(
                "selected reusable uncertainty runtime changed: " + package
            )
    return source


def main():
    args = parser().parse_args()
    configure(args)
    if args.mode == "freeze":
        print(freeze())
        return
    if args.mode == "extract":
        stage(
            args.input_stage,
            [
                SPEC,
                RUN / "NDGAIN_UPDATE_FREEZE.json",
                ARCHIVE,
                RAW / "ndgain/ndgain_latest.csv",
            ],
            extract,
            runner=__file__,
            sources=[
                TEMPLATE,
                ROOT / "src/ndgain_update.py",
                ROOT / "src/vulnerability.py",
            ],
        )
        return
    verified_stage(args.input_stage)
    verify_income_reuse_sources()
    with prospective_execution(run_id=RUN.name):
        stage(
            args.output_stage,
            [
                SPEC,
                RUN / "NDGAIN_UPDATE_FREEZE.json",
                INPUT_STAGE / "ndgain_2026v2_observation2023.csv",
                INPUT_STAGE / "source-record.json",
                RUN / "preprocessing/country_inequality.parquet",
                RAW / "worldbank/world-bank-income-groups.csv",
                RAW / "owid/owid-co2-data.csv",
                RUN / "vulnerability/vulnerability_strata.parquet",
                RUN / "vulnerability/vulnerability_summary.json",
                INCOME_STAGE / "completed.json",
                INCOME_STAGE / "summary.json",
                INCOME_STAGE / "all-bootstrap-draws.parquet",
            ],
            calculate,
            runner=__file__,
            dependencies=["preprocessing", args.input_stage, args.income_draw_stage],
            sources=[
                TEMPLATE,
                ROOT / "src/ndgain_update.py",
                ROOT / "src/temporal_validation.py",
                ROOT / "src/vulnerability.py",
            ],
        )


if __name__ == "__main__":
    main()
