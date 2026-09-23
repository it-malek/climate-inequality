"""Plot frozen validation results; no fitting or raw-data access.

Contract: compare fit/validation R², then paired RMSE changes for all five
primary candidates and both ERA5 constructions. Points and country-resampling
95% intervals show negative results as well as gains. Zero is always visible.
Output: reports/figures/validation_summary.png and a source-digest sidecar.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app/data/residual_structure_summary.json"


def main() -> None:
    summary = json.loads(SOURCE.read_text())
    static = summary["static"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2), gridspec_kw={"width_ratios": [1, 1.7]})
    blue, orange = "#245c87", "#ad6226"
    ax = axes[0]
    names = ["In sample", "Random\n10-fold", "500 km\nLOCO"]
    values = [static[k] for k in ("in_sample_r2", "random_cv_r2", "primary_cv_r2")]
    ax.bar(np.arange(3), values, color=["#b8bec4", blue, blue], width=0.65)
    for i, value in enumerate(values):
        ax.text(i, value + 0.015, f"{value:.3f}", ha="center")
    ax.set(xticks=np.arange(3), xticklabels=names, ylim=(0, 0.72), ylabel="R²",
           title="Static model: fit and validation")
    labels = {"geography_measurement": "Area-measured geography", "baseline_dryness": "Baseline dryness",
              "latitude_form": "Latitude form", "spatial_error": "Spatial error", "spatial_lag": "Spatial lag"}
    rows = [(labels[row["key"]] + " · Berkeley", row["delta_rmse"], row["interval"], blue)
            for row in summary["investigations"]]
    for key, label in (("era5_aligned", "ERA5 anomalies"), ("era5_absolute", "ERA5 absolute")):
        record = next(r for r in summary["robustness"] if r["key"] == key)
        for family, family_label in (("spatial_error", "Spatial error"), ("spatial_lag", "Spatial lag")):
            row = record[family]
            rows.append((f"{family_label} · {label}", row["delta_rmse"], row["interval"], orange))
    ax = axes[1]
    for i, (_, point, (low, high), color) in enumerate(rows):
        ax.errorbar(point, i, xerr=[[point-low], [high-point]], fmt="o" if i < 5 else "s",
                    color=color, capsize=3, markersize=5)
    ax.axvline(0, color="#424242", linewidth=1)
    ax.set(yticks=np.arange(len(rows)), yticklabels=[r[0] for r in rows],
           xlabel="Paired change in RMSE (°C/decade)\nNegative = improvement over the static comparator",
           title="Candidate models and product checks")
    ax.invert_yaxis()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x" if ax is axes[1] else "y", alpha=0.18)
    fig.suptitle("Cross-country land warming, 1950–2013 · 151 countries", fontsize=14)
    fig.text(0.04, 0.035,
             "Right: 95% paired country-resampling intervals; not corrected for spatial dependence. "
             "All comparisons use the corrected 500 km exclusion.\n"
             "Spatial predictions use observed training-country outcomes. ERA5 absolute is a preprocessing sensitivity. "
             "Source: committed residual_structure_summary.json.", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.12, 1, 0.95), w_pad=3)
    out = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "validation_summary.png", dpi=180)
    plt.close(fig)
    (out / "validation_summary.sources.json").write_text(json.dumps({
        "generator": "python -m scripts.research_summary_figure",
        "source": str(SOURCE.relative_to(ROOT)),
        "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "matplotlib": matplotlib.__version__,
        "note": "Presentation only; no new estimates. Intervals do not imply independent countries."
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
