"""Layer 1 input assembly: build ``forcings.parquet`` from public climate data.

This module produces the annual driver table that :mod:`src.physical_model`
consumes -- one contiguous row per year carrying the global temperature anomaly
and the effective radiative forcings (ERF) that the L1 estimator regresses it on:

  * ``temp_anomaly``      -- NASA GISTEMP v4 global ``J-D`` mean (native 1951-1980
                             baseline), re-centred on the baseline window. GISTEMP
                             extends the outcome to the present so the post-2013
                             hindcast window is meaningful.
  * ``temp_uncertainty``  -- annual aggregate (RMS within year) of the in-repo
                             Berkeley Earth monthly uncertainty; the only per-year
                             uncertainty source. Berkeley is also the cross-check.
  * ``erf_*`` / ``erf_total`` -- IPCC AR6 / Forster "Indicators of Global Climate
                             Change" effective radiative forcing (W/m^2): CO2, CH4,
                             N2O, aerosol (radiation+cloud), volcanic, solar, total.
  * ``oni``               -- NOAA CPC Oceanic Nino Index, annual mean of the 3-month
                             ANOM seasons; the ENSO state controlling interannual noise.

Three remote sources are downloaded once into ``data/raw/forcings/`` (idempotent
cache, via :func:`src.data_io.download_file`); Berkeley is read from the committed
``data/raw/`` tree. The transforms are pure and the row order is fixed, so the parquet
is byte-stable on-platform (written through :func:`src.data_io.write_typed_parquet`).
``temp_uncertainty`` and ``erf_total`` are schema-contract columns the estimator does
not read; they round out :mod:`07-data-schemas` without affecting the fit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_io import PROCESSED_DIR, RAW_DIR, download_file, write_typed_parquet

logger = logging.getLogger(__name__)

# --- sources --------------------------------------------------------------
GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
ERF_URL = (
    "https://github.com/ClimateIndicator/forcing-timeseries/raw/main/"
    "output/ERF_best_aggregates_1750-2024.csv"
)
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

RAW_FORCINGS_DIR = RAW_DIR / "forcings"
BERKELEY_CSV = (
    RAW_DIR / "climate-change-earth-surface-temperature-data" / "GlobalTemperatures.csv"
)

# Fixed anomaly baseline (matches GISTEMP's native reference period).
BASELINE: tuple[int, int] = (1951, 1980)

# Sanity gates asserted before the table is written.
MIN_CROSS_CORR = 0.95   # GISTEMP vs Berkeley annual anomaly, over their overlap
MAX_ABS_ONI = 5.0       # ENSO index never approaches this magnitude

# Recent-warming magnitude guard. The cross-check correlation is affine-invariant and
# so blind to a temperature units error (e.g. a source switching degC -> 0.01 degC);
# pin the absolute level instead: the mean anomaly over MAGNITUDE_WINDOW must land in
# TEMP_MAGNITUDE_BAND (degC) for the table to be trusted.
MAGNITUDE_WINDOW: tuple[int, int] = (2010, 2020)
TEMP_MAGNITUDE_BAND: tuple[float, float] = (0.3, 2.0)

# Source-column -> schema-column map for the ERF aggregates file. The file's
# ``aerosol`` column already sums radiation + cloud interactions.
ERF_COLUMN_MAP: dict[str, str] = {
    "CO2": "erf_co2",
    "CH4": "erf_ch4",
    "N2O": "erf_n2o",
    "aerosol": "erf_aerosol",
    "volcanic": "erf_volcanic",
    "solar": "erf_solar",
    "total": "erf_total",
}

# On-disk schema of forcings.parquet (DuckDB types), in order (07-data-schemas.md).
FORCINGS_SCHEMA: dict[str, str] = {
    "year": "BIGINT",
    "temp_anomaly": "DOUBLE",
    "temp_uncertainty": "DOUBLE",
    "erf_co2": "DOUBLE",
    "erf_ch4": "DOUBLE",
    "erf_n2o": "DOUBLE",
    "erf_aerosol": "DOUBLE",
    "erf_volcanic": "DOUBLE",
    "erf_solar": "DOUBLE",
    "erf_total": "DOUBLE",
    "oni": "DOUBLE",
}
FORCINGS_COLUMNS = tuple(FORCINGS_SCHEMA)

# Columns the estimator reads; these must be NaN-free over the contiguous span.
ESTIMATOR_COLUMNS = (
    "temp_anomaly",
    "erf_co2",
    "erf_ch4",
    "erf_n2o",
    "erf_aerosol",
    "erf_volcanic",
    "erf_solar",
    "oni",
)

DEFAULT_FORCINGS_PATH = PROCESSED_DIR / "forcings.parquet"


# ---------------------------------------------------------------------
# Pure parsers (operate on already-read frames; no I/O, fully testable)
# ---------------------------------------------------------------------
def _baseline_mean(frame: pd.DataFrame, col: str, baseline: tuple[int, int]) -> float:
    """Mean of `col` over the inclusive `baseline` year window.

    Raises:
        ValueError: if no rows fall in the baseline window (a series too short
            to anchor the anomaly is a hard error, not a silent zero).
    """
    lo, hi = baseline
    window = frame.loc[(frame["year"] >= lo) & (frame["year"] <= hi), col]
    if window.empty:
        raise ValueError(f"baseline window {baseline} absent from the {col!r} series")
    return float(window.mean())


def parse_gistemp(raw: pd.DataFrame, baseline: tuple[int, int] = BASELINE) -> pd.DataFrame:
    """GISTEMP ``GLB.Ts+dSST`` -> ``year, temp_anomaly`` (re-centred on `baseline`).

    The annual column is ``J-D``; missing months render as ``***`` (coerced to NaN
    and dropped). Re-subtracting the baseline mean makes the anomaly exactly zero
    over the reference window regardless of the source's own centring.
    """
    df = pd.DataFrame(
        {
            "year": pd.to_numeric(raw["Year"], errors="coerce"),
            "temp_anomaly": pd.to_numeric(raw["J-D"], errors="coerce"),
        }
    )
    df = df.dropna(subset=["year", "temp_anomaly"]).astype({"year": int})
    out = df.drop_duplicates("year").sort_values("year").reset_index(drop=True)
    out["temp_anomaly"] = out["temp_anomaly"] - _baseline_mean(out, "temp_anomaly", baseline)
    return out


def parse_berkeley(raw: pd.DataFrame, baseline: tuple[int, int] = BASELINE) -> pd.DataFrame:
    """Berkeley monthly land+ocean -> ``year, berkeley_anomaly, temp_uncertainty``.

    Only complete years (12 months of both the land+ocean mean and its uncertainty)
    are kept. The annual uncertainty is the RMS of the monthly uncertainties; the
    annual mean is re-expressed as an anomaly on `baseline` for the GISTEMP cross-check.
    """
    df = pd.DataFrame(
        {
            "year": raw["dt"].astype(str).str[:4].astype(int),
            "temp": pd.to_numeric(raw["LandAndOceanAverageTemperature"], errors="coerce"),
            "unc": pd.to_numeric(
                raw["LandAndOceanAverageTemperatureUncertainty"], errors="coerce"
            ),
        }
    ).dropna(subset=["temp", "unc"])
    df["unc_sq"] = np.square(df["unc"])
    annual = df.groupby("year").agg(
        n=("temp", "size"),
        temp_mean=("temp", "mean"),
        mean_unc_sq=("unc_sq", "mean"),
    )
    annual = annual[annual["n"] == 12].reset_index()
    annual["temp_uncertainty"] = np.sqrt(annual["mean_unc_sq"])  # RMS of monthly unc
    annual["berkeley_anomaly"] = annual["temp_mean"] - _baseline_mean(
        annual, "temp_mean", baseline
    )
    return annual[["year", "berkeley_anomaly", "temp_uncertainty"]].sort_values(
        "year"
    ).reset_index(drop=True)


def parse_erf(raw: pd.DataFrame) -> pd.DataFrame:
    """ERF aggregates -> ``year, erf_co2 .. erf_solar, erf_total``.

    The first column holds the mid-year index (e.g. ``1750.5``); the integer year
    is its floor. Remaining columns are renamed via :data:`ERF_COLUMN_MAP`.
    """
    year_col = raw.columns[0]
    years = np.floor(pd.to_numeric(raw[year_col], errors="coerce"))
    out = pd.DataFrame({"year": years})
    for src_col, dst_col in ERF_COLUMN_MAP.items():
        out[dst_col] = pd.to_numeric(raw[src_col], errors="coerce")
    out = out.dropna(subset=["year"]).astype({"year": int})
    return out.drop_duplicates("year").sort_values("year").reset_index(drop=True)


def parse_oni(raw: pd.DataFrame) -> pd.DataFrame:
    """NOAA ONI seasonal ANOM -> ``year, oni`` (annual mean of the 3-month seasons).

    Only complete years (all 12 overlapping 3-month seasons, DJF..NDJ) are kept, so the
    latest still-accumulating year is dropped rather than contributing a biased
    partial-season mean. (The ERF inner-join in :func:`assemble_forcings` would also drop
    any year past the ERF series' end, but filtering here keeps ``oni`` self-consistent
    irrespective of the other sources' coverage.)
    """
    df = pd.DataFrame(
        {
            "year": pd.to_numeric(raw["YR"], errors="coerce"),
            "oni": pd.to_numeric(raw["ANOM"], errors="coerce"),
        }
    ).dropna(subset=["year", "oni"]).astype({"year": int})
    annual = df.groupby("year").agg(n=("oni", "size"), oni=("oni", "mean"))
    annual = annual[annual["n"] == 12].reset_index()
    return annual[["year", "oni"]].sort_values("year").reset_index(drop=True)


def assemble_forcings(
    gistemp: pd.DataFrame,
    berkeley: pd.DataFrame,
    erf: pd.DataFrame,
    oni: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Inner-join the parsed series on ``year`` into the schema frame.

    The outcome (GISTEMP), forcings (ERF) and ENSO (ONI) tables define the span via
    an inner join; Berkeley's annual uncertainty is left-joined on top. Years past
    Berkeley's coverage (the post-2015 tail) get a fallback uncertainty: the mean of
    Berkeley's final-decade annual uncertainty.

    Returns:
        ``(frame, n_filled)`` -- the schema-ordered frame and the count of tail years
        whose uncertainty was backfilled.
    """
    core = (
        gistemp.merge(erf, on="year", how="inner")
        .merge(oni, on="year", how="inner")
        .merge(berkeley[["year", "temp_uncertainty"]], on="year", how="left")
        .sort_values("year")
        .reset_index(drop=True)
    )
    n_filled = int(core["temp_uncertainty"].isna().sum())
    if n_filled:
        last_year = int(berkeley["year"].max())
        recent = berkeley.loc[berkeley["year"] >= last_year - 9, "temp_uncertainty"]
        core["temp_uncertainty"] = core["temp_uncertainty"].fillna(float(recent.mean()))
    return core[list(FORCINGS_COLUMNS)].copy(), n_filled


# ---------------------------------------------------------------------
# Result + compute
# ---------------------------------------------------------------------
class ForcingsError(RuntimeError):
    """Base class for fail-loud forcings-assembly invariant violations."""


class ForcingsCrossCheckError(ForcingsError):
    """The GISTEMP vs Berkeley annual-anomaly cross-check fell below tolerance.

    Two independently-sourced global temperature series must track each other very
    tightly (corr >= :data:`MIN_CROSS_CORR`) over their overlap. A low -- or NaN --
    correlation almost always means an upstream source silently changed its on-disk
    format, or the year alignment skewed during the join. Raising a distinct, named
    error here surfaces that *at assembly time* instead of letting it masquerade as an
    "odd" L1 model fit to be debugged much later.
    """


class ForcingsMagnitudeError(ForcingsError):
    """The recent temperature-anomaly magnitude is outside its plausible band.

    The correlation cross-check (:class:`ForcingsCrossCheckError`) is invariant to
    affine rescaling, so it is blind to a temperature *units* error -- a source moving
    from degC to 0.01 degC keeps corr ~1.0 while inflating the level 100x, and ridge
    R^2 / hindcast coverage are scale-invariant too. This guard pins the absolute level
    of recent warming, catching the silent scale drift the other gates cannot see.
    """


@dataclass(frozen=True)
class ForcingsResult:
    """Provenance + fail-loud invariants for one assembled forcings table."""

    year_min: int
    year_max: int
    n_years: int
    n_uncertainty_filled: int
    cross_check_corr: float
    recent_temp_mean: float
    max_abs_oni: float
    n_estimator_nan: int

    def check(self) -> None:
        """Assert the table is contiguous, complete, and physically plausible."""
        if self.n_years != self.year_max - self.year_min + 1:
            raise ForcingsError(
                f"years not contiguous: span {self.year_min}-{self.year_max} "
                f"but n={self.n_years}"
            )
        if self.n_estimator_nan:
            raise ForcingsError(
                f"{self.n_estimator_nan} NaN(s) in estimator columns {ESTIMATOR_COLUMNS}"
            )
        if not (self.cross_check_corr >= MIN_CROSS_CORR):  # also catches NaN
            raise ForcingsCrossCheckError(
                f"GISTEMP/Berkeley annual-anomaly correlation "
                f"{self.cross_check_corr!r} is below the {MIN_CROSS_CORR} tolerance "
                "over their overlap: a source format change or skewed year alignment "
                "is the likely cause -- inspect the raw sources before trusting any "
                "L1 fit derived from this table."
            )
        lo, hi = TEMP_MAGNITUDE_BAND
        if not (lo <= self.recent_temp_mean <= hi):  # also catches NaN
            raise ForcingsMagnitudeError(
                f"mean temp_anomaly over {MAGNITUDE_WINDOW} is "
                f"{self.recent_temp_mean!r} degC, outside the plausible [{lo}, {hi}] "
                "band: a temperature units/scale error is the likely cause (the "
                "correlation cross-check is scale-invariant and cannot detect it)."
            )
        if self.max_abs_oni >= MAX_ABS_ONI:
            raise ForcingsError(f"implausible |oni| max {self.max_abs_oni}")


def compute_forcings(
    gistemp_raw: pd.DataFrame,
    berkeley_raw: pd.DataFrame,
    erf_raw: pd.DataFrame,
    oni_raw: pd.DataFrame,
    baseline: tuple[int, int] = BASELINE,
) -> tuple[pd.DataFrame, ForcingsResult]:
    """Parse the four raw sources and assemble the validated forcings table.

    Network-free: callers pass already-read frames (so tests drive this with
    synthetic fixtures). Returns ``(frame, result)`` where ``frame`` matches
    :data:`FORCINGS_SCHEMA` and ``result`` has passed :meth:`ForcingsResult.check`.
    """
    gistemp = parse_gistemp(gistemp_raw, baseline)
    berkeley = parse_berkeley(berkeley_raw, baseline)
    erf = parse_erf(erf_raw)
    oni = parse_oni(oni_raw)

    frame, n_filled = assemble_forcings(gistemp, berkeley, erf, oni)

    overlap = frame.merge(berkeley[["year", "berkeley_anomaly"]], on="year", how="inner")
    if len(overlap) < 10:
        raise ValueError(
            f"insufficient GISTEMP/Berkeley overlap ({len(overlap)} years) for cross-check"
        )
    corr = float(np.corrcoef(overlap["temp_anomaly"], overlap["berkeley_anomaly"])[0, 1])

    years = frame["year"].to_numpy()
    result = ForcingsResult(
        year_min=int(years.min()),
        year_max=int(years.max()),
        n_years=len(frame),
        n_uncertainty_filled=n_filled,
        cross_check_corr=corr,
        recent_temp_mean=_baseline_mean(frame, "temp_anomaly", MAGNITUDE_WINDOW),
        max_abs_oni=float(frame["oni"].abs().max()),
        n_estimator_nan=int(frame[list(ESTIMATOR_COLUMNS)].isna().to_numpy().sum()),
    )
    result.check()
    return frame, result


# ---------------------------------------------------------------------
# I/O readers + build (side effects only here)
# ---------------------------------------------------------------------
def _read_gistemp(path: Path) -> pd.DataFrame:
    """Read the GISTEMP CSV, skipping its one-line ``Land-Ocean: Global Means`` title."""
    return pd.read_csv(path, skiprows=1)


def _read_oni(path: Path) -> pd.DataFrame:
    """Read the whitespace-delimited NOAA ONI ascii table (``SEAS YR TOTAL ANOM``)."""
    return pd.read_csv(path, sep=r"\s+")


def build_forcings(
    forcings_path: Path = DEFAULT_FORCINGS_PATH,
    raw_dir: Path = RAW_FORCINGS_DIR,
    berkeley_csv: Path = BERKELEY_CSV,
) -> dict:
    """Download the remote sources, assemble, validate, and write ``forcings.parquet``.

    The three remote files are cached idempotently under `raw_dir`; Berkeley is read
    from the committed `berkeley_csv`. Returns a dict with ``frame``, ``result`` and
    ``forcings_path``.
    """
    gistemp_raw = _read_gistemp(download_file(GISTEMP_URL, raw_dir / "gistemp_glb_ts_dsst.csv"))
    erf_raw = pd.read_csv(download_file(ERF_URL, raw_dir / "erf_best_aggregates.csv"))
    oni_raw = _read_oni(download_file(ONI_URL, raw_dir / "oni.ascii.txt"))
    if not berkeley_csv.exists():
        raise FileNotFoundError(
            f"Berkeley GlobalTemperatures.csv not found at {berkeley_csv}; "
            "run src.data_io.download_raw_data() first"
        )
    berkeley_raw = pd.read_csv(berkeley_csv)

    frame, result = compute_forcings(gistemp_raw, berkeley_raw, erf_raw, oni_raw)
    write_typed_parquet(frame, forcings_path, FORCINGS_SCHEMA, order_by=("year",))
    logger.info(
        "wrote %s (%d years %d-%d, %d uncertainty-filled, GISTEMP/Berkeley corr %.3f)",
        forcings_path,
        result.n_years,
        result.year_min,
        result.year_max,
        result.n_uncertainty_filled,
        result.cross_check_corr,
    )
    return {"frame": frame, "result": result, "forcings_path": forcings_path}


def main() -> None:
    """Assemble the real forcings table and print the coverage headline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = build_forcings()
    r = out["result"]
    print(
        f"L1 forcings table: {r.n_years} years {r.year_min}-{r.year_max} "
        f"({r.n_uncertainty_filled} uncertainty-filled tail years)"
    )
    print(f"  GISTEMP/Berkeley anomaly corr : {r.cross_check_corr:.3f}")
    print(f"  |ONI| max                     : {r.max_abs_oni:.2f}")
    print(f"  table: {out['forcings_path']}")


if __name__ == "__main__":
    main()
