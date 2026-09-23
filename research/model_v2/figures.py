"""Research figures for the M0 residual diagnostics (PNG, light theme).

Reads the outputs of :mod:`research.model_v2.diagnostics` and the cached national
grid; writes to ``research/model_v2/outputs/figures/``. Run with matplotlib on the
path, e.g. ``uv run --locked --extra figures python -m research.model_v2.figures``
(matplotlib is a locked optional dependency).
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from research.model_v2 import geometry  # noqa: E402
from research.model_v2.diagnostics import COUNTRY_TABLE_PATH, DIAGNOSTICS_PATH  # noqa: E402
from research.model_v2.m0 import OUTPUT_DIR  # noqa: E402

FIG_DIR = OUTPUT_DIR / "figures"
LAND_ZIP = Path("data/raw/natural_earth/ne_110m_land.zip")

# Palette (light surface); see the dataviz reference palette.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DIV_BLUE, DIV_MID, DIV_RED = "#256abf", "#f0efec", "#e34948"
NON_SAMPLE = "#e6e5df"
DIVERGING = LinearSegmentedColormap.from_list("resid", [DIV_BLUE, DIV_MID, DIV_RED])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 8,
    }
)


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _load() -> tuple[pd.DataFrame, dict]:
    table = pd.read_csv(COUNTRY_TABLE_PATH)
    report = json.loads(DIAGNOSTICS_PATH.read_text())
    return table, report


def fig_distribution(table: pd.DataFrame, report: dict) -> Path:
    resid = table["residual"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    ax = axes[0]
    ax.hist(resid, bins=21, color=BLUE, edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("M0 residual, observed − fitted (°C/decade)")
    ax.set_ylabel("countries")
    d = report["distribution"]
    ax.set_title(f"Residual distribution (sd {d['sd']:.3f}, skew {d['skew']:+.2f}, excess kurtosis {d['excess_kurtosis']:+.2f})")
    ax = axes[1]
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax.plot(osm, slope * osm + intercept, color=MUTED, linewidth=1)
    ax.scatter(osm, osr, s=14, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    ax.set_xlabel("normal quantile")
    ax.set_ylabel("residual quantile (°C/decade)")
    ax.set_title(f"Normal Q\u2013Q (Shapiro p = {d['shapiro_p']:.2f})")
    return _save(fig, "m0_residual_distribution.png")


def fig_residual_vs_fitted(table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.scatter(table["fitted"], table["residual"], s=16, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    bins = pd.qcut(table["fitted"], 8)
    means = table.groupby(bins, observed=True).agg(x=("fitted", "mean"), y=("residual", "mean"))
    ax.plot(means["x"], means["y"], color=ORANGE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
    top = table.reindex(table["residual"].abs().sort_values(ascending=False).index).head(7)
    for _, r in top.iterrows():
        ax.annotate(r["Country"], (r["fitted"], r["residual"]), xytext=(4, 2), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlabel("M0 fitted warming rate (°C/decade)")
    ax.set_ylabel("residual (°C/decade)")
    ax.set_title("Residual against fitted value; orange = octile means")
    return _save(fig, "m0_residual_vs_fitted.png")


def _raster(values_by_iso: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    iso, lats, lons = geometry.load_national_grid()
    field = np.full(iso.shape, np.nan)
    sample = np.zeros(iso.shape, dtype=bool)
    for code, value in values_by_iso.items():
        m = iso == code
        field[m] = value
        sample |= m
    land = iso != ""
    return field, land & ~sample


def _map_axes(ax: plt.Axes) -> None:
    land = gpd.read_file(f"zip://{LAND_ZIP}")
    land.boundary.plot(ax=ax, color=MUTED, linewidth=0.35)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 84)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def fig_residual_map(table: pd.DataFrame) -> Path:
    field, other_land = _raster(dict(zip(table["iso3"], table["residual"], strict=True)))
    vmax = float(np.nanmax(np.abs(field)))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.imshow(np.where(other_land, 1.0, np.nan), extent=(-180, 180, -90, 90), origin="upper",
              cmap=ListedColormap([NON_SAMPLE]), interpolation="nearest")
    im = ax.imshow(field, extent=(-180, 180, -90, 90), origin="upper", cmap=DIVERGING,
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax), interpolation="nearest")
    _map_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("M0 residual, observed − fitted (°C/decade)")
    cbar.outline.set_visible(False)
    ax.set_title("M0 residual by country (red: warmed faster than M0 predicts; grey land: outside the 151-country sample)", fontsize=9)
    return _save(fig, "m0_residual_map.png")


def fig_lisa_map(table: pd.DataFrame) -> Path:
    codes = {"ns": 0, "HH": 1, "LL": 2, "HL": 3, "LH": 4}
    field, other_land = _raster({i: codes[lab] for i, lab in zip(table["iso3"], table["lisa_v1_label"], strict=True)})
    colors = ["#d9d8d1", DIV_RED, DIV_BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.imshow(np.where(other_land, 1.0, np.nan), extent=(-180, 180, -90, 90), origin="upper",
              cmap=ListedColormap([NON_SAMPLE]), interpolation="nearest")
    ax.imshow(field, extent=(-180, 180, -90, 90), origin="upper", cmap=ListedColormap(colors), vmin=-0.5, vmax=4.5, interpolation="nearest")
    _map_axes(ax)
    sig = table[(table["lisa_v1_label"] != "ns") & (table["land_area_km2"] > 150_000)]
    for _, r in sig.iterrows():
        ax.annotate(r["iso3"], (r["centroid_lon"], r["centroid_lat"]), fontsize=6, color=INK, ha="center", va="center")
    counts = table["lisa_v1_label"].value_counts()
    handles = [Patch(color=colors[codes[k]], label=f"{name} ({counts.get(k, 0)})") for k, name in
               (("HH", "high residual, high neighbours"), ("LL", "low residual, low neighbours"), ("HL", "high among low"), ("LH", "low among high"), ("ns", "not significant"))]
    ax.legend(handles=handles, loc="lower left", fontsize=7, ncol=2)
    ax.set_title("Local Moran's I of the M0 residual (station-centroid kNN, k = 8; conditional permutation p < 0.05; labels: countries over 150,000 km²)", fontsize=9)
    return _save(fig, "m0_lisa_map.png")


def fig_latitude(table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(0, color=AXIS, linewidth=0.8)
    edges = np.array([0, 10, 20, 30, 40, 50, 70])
    for hemi, color, label in (("N", BLUE, "Northern hemisphere"), ("S", ORANGE, "Southern hemisphere")):
        sub = table[table["hemisphere"] == hemi]
        ax.scatter(sub["abs_centroid_lat"], sub["residual"], s=16, color=color, edgecolor=SURFACE, linewidth=0.6, label=label)
        bins = pd.cut(sub["abs_centroid_lat"], edges, right=False)
        means = sub.groupby(bins, observed=True).agg(x=("abs_centroid_lat", "mean"), y=("residual", "mean"), n=("residual", "size"))
        means = means[means["n"] >= 3]
        ax.plot(means["x"], means["y"], color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
    ax.set_xlabel("absolute latitude of the land-area centroid (degrees)")
    ax.set_ylabel("M0 residual (°C/decade)")
    ax.set_title("Residual against latitude by hemisphere; lines = 10° bin means (n ≥ 3)")
    ax.legend(loc="upper right")
    return _save(fig, "m0_residual_vs_latitude.png")


def fig_correlogram(report: dict) -> Path:
    rings = pd.DataFrame(report["correlogram"]["rings"])
    semi = pd.DataFrame(report["correlogram"]["semivariogram"])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    ax = axes[0]
    x = (rings["lo_km"] + rings["hi_km"]) / 2
    ax.axhline(rings["expected"].iloc[0], color=AXIS, linewidth=0.8)
    ax.plot(x, rings["morans_i"], color=BLUE, linewidth=2)
    sig = rings["p"] < 0.05
    ax.scatter(x[sig], rings["morans_i"][sig], s=36, color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.scatter(x[~sig], rings["morans_i"][~sig], s=36, facecolor=SURFACE, edgecolor=BLUE, linewidth=1.2, zorder=3)
    for xi, yi, n in zip(x, rings["morans_i"], rings["n_pairs"], strict=True):
        ax.annotate(f"{n}", (xi, yi), xytext=(0, 6), textcoords="offset points", fontsize=6, color=MUTED, ha="center")
    border_path = OUTPUT_DIR / "border_distance_correlogram.json"
    if border_path.exists():
        brings = pd.DataFrame(json.loads(border_path.read_text())["rings"])
        bx = (brings["lo_km"] + brings["hi_km"]) / 2
        ax.plot(bx, brings["morans_i"], color=ORANGE, linewidth=2)
        bsig = brings["p"] < 0.05
        ax.scatter(bx[bsig], brings["morans_i"][bsig], s=36, color=ORANGE, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.scatter(bx[~bsig], brings["morans_i"][~bsig], s=36, facecolor=SURFACE, edgecolor=ORANGE, linewidth=1.2, zorder=3)
        ax.plot([], [], color=BLUE, linewidth=2, label="distance between land-area centroids")
        ax.plot([], [], color=ORANGE, linewidth=2, label="minimum inter-territory distance")
        ax.legend(loc="upper right")
    ax.set_xlabel("distance (km, ring midpoint)")
    ax.set_ylabel("Moran's I of the residual")
    ax.set_title("Correlogram (filled: permutation p < 0.05; labels: centroid pairs per ring)")
    ax = axes[1]
    xs = (semi["lo_km"] + semi["hi_km"]) / 2
    ax.axhline(report["correlogram"]["residual_variance"], color=AXIS, linewidth=0.8)
    ax.plot(xs, semi["semivariance"], color=BLUE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
    ax.annotate("residual variance (sill)", (xs.iloc[-1], report["correlogram"]["residual_variance"]), xytext=(0, 4), textcoords="offset points", fontsize=7, color=MUTED, ha="right")
    ax.set_xlabel("distance between land-area centroids (km)")
    ax.set_ylabel("semivariance of the residual")
    ax.set_title("Residual semivariogram")
    return _save(fig, "m0_correlogram.png")


def fig_top_residuals(table: pd.DataFrame, report: dict) -> Path:
    top = table.reindex(table["residual"].abs().sort_values(ascending=False).index).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    colors = np.where(top["residual"] > 0, DIV_RED, DIV_BLUE)
    ax.barh(top["Country"], top["residual"], color=colors, height=0.62)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("M0 residual (°C/decade)")
    c = report["concentration"]
    ax.set_title(f"Twenty largest residuals; top 10 carry {c['top10_share_of_ss']:.0%} of the residual sum of squares (uniform: {c['expected_top10_share_if_uniform']:.0%})")
    ax.grid(axis="y", visible=False)
    return _save(fig, "m0_top_residuals.png")


def fig_descriptor_panels(table: pd.DataFrame, report: dict) -> Path:
    panels = [
        ("log10_land_area", "log10 land area (km²)"),
        ("coastal_cell_fraction", "coastal fraction of land cells"),
        ("elev_sd_m", "within-country elevation sd (m)"),
        ("cell_sd", "within-country sd of 1° cell trends (°C/decade)"),
        ("era5_minus_berkeley", "ERA5 − Berkeley country trend (°C/decade)"),
        ("station_minus_area", "station − area country trend (°C/decade)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    assoc = report["descriptor_associations"]
    for ax, (col, label) in zip(axes.ravel(), panels, strict=True):
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.scatter(table[col], table["residual"], s=14, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
        rho = assoc[col]["residual"]["spearman_rho"]
        p = assoc[col]["residual"]["spearman_p"]
        ax.set_title(f"Spearman ρ = {rho:+.2f} (p = {p:.2g})", fontsize=9)
        ax.set_xlabel(label)
        ax.set_ylabel("residual (°C/decade)")
    fig.suptitle("M0 residual against descriptors V1 does not contain", y=1.01)
    fig.tight_layout()
    return _save(fig, "m0_residual_vs_descriptors.png")


def fig_moran_sensitivity(report: dict) -> Path:
    rows = [(k, v["morans_i"], v["p"]) for k, v in report["weights_sensitivity"].items() if "morans_i" in v]
    df = pd.DataFrame(rows, columns=["weights", "I", "p"]).sort_values("I")
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    y = np.arange(len(df))
    ax.hlines(y, 0, df["I"], color=GRID, linewidth=1)
    sig = df["p"] < 0.05
    ax.scatter(df["I"][sig], y[sig], s=40, color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.scatter(df["I"][~sig], y[~sig], s=40, facecolor=SURFACE, edgecolor=BLUE, linewidth=1.2, zorder=3)
    v1 = report["weights_sensitivity"]["knn_k8_station"]["morans_i"]
    ax.axvline(v1, color=ORANGE, linewidth=1.2)
    ax.annotate("V1 definition (station kNN, k = 8)", (v1, len(df) - 0.5), xytext=(4, 0), textcoords="offset points", fontsize=7, color=ORANGE)
    ax.set_yticks(y)
    ax.set_yticklabels(df["weights"], fontsize=7)
    ax.set_xlabel("global Moran's I of the M0 residual")
    ax.set_title("Moran's I under alternative spatial-weight definitions (filled: p < 0.05)")
    ax.grid(axis="y", visible=False)
    return _save(fig, "m0_moran_sensitivity.png")


def fig_subregions(table: pd.DataFrame) -> Path:
    order = table.groupby("m49_subregion")["residual"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.axvline(0, color=AXIS, linewidth=0.8)
    rng = np.random.default_rng(0)
    for i, name in enumerate(order):
        sub = table[table["m49_subregion"] == name]
        ax.scatter(sub["residual"], i + rng.uniform(-0.18, 0.18, len(sub)), s=14, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
        ax.scatter([sub["residual"].mean()], [i], s=60, marker="|", color=ORANGE, linewidth=2.2, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{n} ({(table['m49_subregion'] == n).sum()})" for n in order], fontsize=8)
    ax.set_xlabel("M0 residual (°C/decade)")
    ax.set_title("Residual by UN M49 sub-region (orange tick = mean); continent means are zero by construction")
    ax.grid(axis="y", visible=False)
    return _save(fig, "m0_residual_by_subregion.png")


def fig_moran_scatter(table: pd.DataFrame, report: dict) -> Path:
    z = table["lisa_v1_local_i"]  # noqa: F841  (local I is available; the scatter uses z and lag)
    zz = table["residual"] - table["residual"].mean()
    lag = zz * 0  # placeholder replaced below
    from research.model_v2.spatial import haversine_matrix, knn_weights  # local import keeps module load light

    w = knn_weights(haversine_matrix(table["station_lon"].to_numpy(), table["station_lat"].to_numpy()), 8)
    lag = w @ zz.to_numpy()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.scatter(zz, lag, s=14, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    i_v1 = report["weights_sensitivity"]["knn_k8_station"]["morans_i"]
    xs = np.linspace(zz.min(), zz.max(), 2)
    ax.plot(xs, i_v1 * xs, color=ORANGE, linewidth=2)
    ax.set_xlabel("residual (centred)")
    ax.set_ylabel("mean residual of 8 nearest countries")
    ax.set_title(f"Moran scatterplot (slope = I = {i_v1:.3f})")
    return _save(fig, "m0_moran_scatter.png")


def main() -> list[Path]:
    table, report = _load()
    return [
        fig_distribution(table, report),
        fig_residual_vs_fitted(table),
        fig_residual_map(table),
        fig_lisa_map(table),
        fig_latitude(table),
        fig_correlogram(report),
        fig_top_residuals(table, report),
        fig_descriptor_panels(table, report),
        fig_moran_sensitivity(report),
        fig_subregions(table),
        fig_moran_scatter(table, report),
    ]


if __name__ == "__main__":
    for path in main():
        print("wrote", path)
