"""Country-level climate inequality: warming rates vs emissions responsibility.

Phase 4 pipeline, answering the project's second question -- is warming
proportional to cumulative emissions responsibility?

1. Aggregate per-city-location Theil-Sen trends (Phase 2) to country level.
   The mean is deliberately **unweighted**: no city-population data exists
   in the project datasets, so population weighting would need a third
   dataset with fragile city-name matching (documented limitation; see
   :func:`aggregate_trends_by_country`).
2. Compute cumulative per-capita CO2 per country: annual production-based
   emissions summed through the analysis cutoff year, divided by
   cutoff-year population (tonnes per person). Emissions come fresh from
   github.com/owid/co2-data -- the project's Kaggle snapshot is annual
   emissions only, with no population column, so per-capita responsibility
   cannot be computed from it.
3. Join on country (explicit Berkeley Earth -> OWID name overrides) and
   attach OWID's continent classification.
4. Quantify: Spearman rank correlation, plus OLS of warming trend on log10
   cumulative per-capita CO2 with and without continent fixed effects (HC1
   robust SEs). Effect sizes read as °C/decade per 10x emissions.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

from src.cleaning import parse_window
from src.data_io import (
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    download_file,
    write_typed_parquet,
)
from src.figures import render_inequality_scatter
from src.trends import DEFAULT_TRENDS_PATH

logger = logging.getLogger(__name__)

OWID_CO2_URL = (
    "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
)
OWID_CO2_PATH = RAW_DIR / "owid" / "owid-co2-data.csv"
CONTINENTS_URL = (
    "https://ourworldindata.org/grapher/continents-according-to-our-world-in-data.csv"
)
CONTINENTS_PATH = RAW_DIR / "owid" / "continents.csv"
CONTINENT_COL = "World region according to OWID"  # column name in that CSV

DEFAULT_INEQUALITY_PATH = PROCESSED_DIR / "country_inequality.parquet"

# Berkeley Earth country names whose OWID entity name differs. Berkeley's
# Puerto Rico and Reunion have no OWID emissions series at all and are
# dropped (logged) at join time.
BERKELEY_TO_OWID = {
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Burma": "Myanmar",
    "Congo (Democratic Republic Of The)": "Democratic Republic of Congo",
    "Czech Republic": "Czechia",
    "Côte D'Ivoire": "Cote d'Ivoire",
    "Guinea Bissau": "Guinea-Bissau",
    "Macedonia": "North Macedonia",
    "Swaziland": "Eswatini",
}

# On-disk schema of country_inequality.parquet (DuckDB types), in order.
INEQUALITY_SCHEMA = {
    "Country": "VARCHAR",
    "owid_country": "VARCHAR",
    "continent": "VARCHAR",
    "n_cities": "BIGINT",
    "trend_c_per_decade": "DOUBLE",
    "cumulative_co2_mt": "DOUBLE",
    "population": "BIGINT",
    "cum_co2_t_per_capita": "DOUBLE",
}
INEQUALITY_COLUMNS = list(INEQUALITY_SCHEMA)


def load_owid_co2(csv_path: Path = OWID_CO2_PATH) -> pd.DataFrame:
    """Load the OWID CO2 columns needed for cumulative per-capita emissions.

    Args:
        csv_path: Path to owid-co2-data.csv.

    Returns:
        Frame with country, iso_code, year, co2 (Mt/yr), population.

    Raises:
        FileNotFoundError: if `csv_path` is missing -- the pipeline
            downloads it via :func:`src.data_io.download_file`.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"no such file: {csv_path}; download it first")
    return pd.read_csv(
        csv_path, usecols=["country", "iso_code", "year", "co2", "population"]
    )


def load_continents(csv_path: Path = CONTINENTS_PATH) -> pd.DataFrame:
    """Load OWID's country -> continent classification.

    Args:
        csv_path: Path to the "continents according to OWID" grapher CSV.

    Returns:
        Frame with columns `country` and `continent`, one row per entity.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"no such file: {csv_path}; download it first")
    raw = pd.read_csv(csv_path)
    renamed = raw.rename(columns={"Entity": "country", CONTINENT_COL: "continent"})
    return renamed[["country", "continent"]]


def aggregate_trends_by_country(
    trends: pd.DataFrame, value_col: str = "slope_c_per_decade"
) -> pd.DataFrame:
    """Mean warming trend per country across its city-locations.

    The mean is unweighted: the project datasets carry no city populations,
    so population weighting would require a third dataset and fragile
    city-name matching (documented limitation). The result is therefore
    station-weighted, not area-weighted -- dense urban station clusters
    count more than their land area would.

    Args:
        trends: Phase 2 output, one row per city-location.
        value_col: Trend column to average.

    Returns:
        One row per Country with `n_cities` and `trend_c_per_decade`.
    """
    return (
        trends.groupby("Country", observed=True)[value_col]
        .agg(n_cities="size", trend_c_per_decade="mean")
        .reset_index()
    )


def cumulative_emissions_per_capita(
    owid: pd.DataFrame, cutoff_year: int
) -> pd.DataFrame:
    """Cumulative CO2 through `cutoff_year`, per cutoff-year resident.

    Annual production-based emissions (Mt) are summed from the start of
    each country's record through `cutoff_year` and divided by that year's
    population: tonnes per person, the standard "historical responsibility"
    framing. OWID aggregate rows (World, continents, income groups) are
    excluded via their missing or OWID_* iso codes.

    Args:
        owid: Frame from :func:`load_owid_co2`.
        cutoff_year: Last year included (the analysis-window end).

    Returns:
        One row per country: country, cumulative_co2_mt, population,
        cum_co2_t_per_capita. Countries with no emissions record or no
        cutoff-year population are dropped and logged.
    """
    iso = owid["iso_code"].fillna("")
    countries = owid[(iso != "") & ~iso.str.startswith("OWID_")]

    cumulative = (
        countries.loc[countries["year"] <= cutoff_year]
        .groupby("country")["co2"]
        .sum(min_count=1)
        .rename("cumulative_co2_mt")
    )
    population = (
        countries.loc[countries["year"] == cutoff_year]
        .set_index("country")["population"]
    )
    out = pd.concat([cumulative, population], axis=1).reset_index(names="country")

    incomplete = out["cumulative_co2_mt"].isna() | out["population"].isna()
    if incomplete.any():
        logger.info(
            "dropping %d countries without emissions or %d population: %s",
            int(incomplete.sum()),
            cutoff_year,
            sorted(out.loc[incomplete, "country"]),
        )
        out = out.loc[~incomplete]

    out = out.assign(
        cum_co2_t_per_capita=out["cumulative_co2_mt"] * 1e6 / out["population"]
    )
    return out.reset_index(drop=True)


def join_country_data(
    country_trends: pd.DataFrame,
    emissions: pd.DataFrame,
    continents: pd.DataFrame,
    overrides: Mapping[str, str] = BERKELEY_TO_OWID,
) -> pd.DataFrame:
    """Join country trends with emissions and continents on OWID names.

    Berkeley Earth names map to OWID entity names by exact match plus the
    explicit `overrides`; countries that still have no OWID emissions match
    are dropped and logged (Puerto Rico and Reunion, in the real data).

    Args:
        country_trends: From :func:`aggregate_trends_by_country`.
        emissions: From :func:`cumulative_emissions_per_capita`.
        continents: From :func:`load_continents`.
        overrides: Berkeley Earth -> OWID name mapping.

    Returns:
        One row per matched country, columns `INEQUALITY_COLUMNS`.
    """
    mapped = country_trends.assign(
        owid_country=country_trends["Country"].map(lambda c: overrides.get(c, c))
    )
    unmatched = sorted(set(mapped["owid_country"]) - set(emissions["country"]))
    if unmatched:
        logger.warning(
            "dropping %d countries with no OWID emissions match: %s",
            len(unmatched),
            unmatched,
        )

    joined = mapped.merge(
        emissions, left_on="owid_country", right_on="country", how="inner"
    ).drop(columns="country")
    joined = joined.merge(
        continents, left_on="owid_country", right_on="country", how="left"
    ).drop(columns="country")

    missing_continent = joined["continent"].isna()
    if missing_continent.any():
        logger.warning(
            "no continent for: %s",
            sorted(joined.loc[missing_continent, "owid_country"]),
        )
    return joined[INEQUALITY_COLUMNS]


@dataclass(frozen=True)
class OLSFit:
    """Emissions effect from one OLS fit, with HC1-robust uncertainty.

    `coef` reads as °C/decade of additional warming per tenfold increase
    in cumulative per-capita CO2.
    """

    coef: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    r2: float


@dataclass(frozen=True)
class InequalityResult:
    """Headline statistics for the warming-vs-emissions relationship."""

    n_countries: int
    n_continents: int
    spearman_rho: float
    spearman_p: float
    ols_pooled: OLSFit
    ols_fe: OLSFit


def _extract_fit(
    fit: RegressionResultsWrapper, term: str = "log10_emissions"
) -> OLSFit:
    """Pull `term`'s effect size and uncertainty out of a statsmodels fit."""
    ci_low, ci_high = fit.conf_int().loc[term]
    return OLSFit(
        coef=float(fit.params[term]),
        se=float(fit.bse[term]),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(fit.pvalues[term]),
        r2=float(fit.rsquared),
    )


def quantify_inequality(
    df: pd.DataFrame,
    x_col: str = "cum_co2_t_per_capita",
    y_col: str = "trend_c_per_decade",
    continent_col: str = "continent",
) -> InequalityResult:
    """Spearman correlation and OLS effect of emissions on warming rate.

    The OLS predictor is log10 of cumulative per-capita emissions (the
    variable spans orders of magnitude), so coefficients read as °C/decade
    per tenfold increase in emissions responsibility. Fit twice -- pooled,
    and with continent fixed effects (a within-continent comparison that
    absorbs continent-level confounders like latitude) -- both with HC1
    robust standard errors. Spearman is rank-based and indifferent to the
    log transform.

    Args:
        df: One row per country (see :func:`join_country_data`).
        x_col: Cumulative per-capita emissions column (tonnes/person).
        y_col: Warming-trend column (°C/decade).
        continent_col: Continent label column for the fixed effects.

    Returns:
        InequalityResult with the Spearman test and both OLS fits.
    """
    work = df[[x_col, y_col, continent_col]].dropna()
    nonpositive = work[x_col] <= 0
    if nonpositive.any():
        logger.warning(
            "dropping %d countries with non-positive emissions before log10",
            int(nonpositive.sum()),
        )
        work = work.loc[~nonpositive]

    rho, rho_p = stats.spearmanr(work[x_col], work[y_col])
    work = work.assign(log10_emissions=np.log10(work[x_col]))

    pooled = smf.ols(f"{y_col} ~ log10_emissions", data=work).fit(cov_type="HC1")
    fe = smf.ols(f"{y_col} ~ log10_emissions + C({continent_col})", data=work).fit(
        cov_type="HC1"
    )

    return InequalityResult(
        n_countries=len(work),
        n_continents=int(work[continent_col].nunique()),
        spearman_rho=float(rho),
        spearman_p=float(rho_p),
        ols_pooled=_extract_fit(pooled),
        ols_fe=_extract_fit(fe),
    )


def build_inequality_analysis(
    trends_path: Path = DEFAULT_TRENDS_PATH,
    co2_path: Path = OWID_CO2_PATH,
    continents_path: Path = CONTINENTS_PATH,
    out_dir: Path = OUTPUTS_DIR,
    table_path: Path = DEFAULT_INEQUALITY_PATH,
    cutoff_year: int | None = None,
) -> dict:
    """Run the Phase 4 pipeline end to end.

    Downloads the OWID inputs if missing, aggregates Phase 2 trends to
    country level, joins against cumulative per-capita emissions and
    continents, quantifies the relationship, and writes the country table
    plus the scatter figure.

    Args:
        trends_path: Parquet from :func:`src.trends.build_city_trends`.
        co2_path: OWID CO2 csv (downloaded if absent).
        continents_path: OWID continents csv (downloaded if absent).
        out_dir: Destination directory for inequality_scatter.html.
        table_path: Destination parquet for the joined country table.
        cutoff_year: Last year of cumulative emissions; None derives the
            analysis-window end year from the trends file.

    Returns:
        Dict with keys `table` (DataFrame), `result` (InequalityResult),
        `figure_path` and `table_path` (Paths).
    """
    download_file(OWID_CO2_URL, co2_path)
    download_file(CONTINENTS_URL, continents_path)

    trends = pd.read_parquet(trends_path)
    if cutoff_year is None:
        _, window_end = parse_window(trends["analysis_window"].iloc[0])
        cutoff_year = int(window_end[:4])
        logger.info("cutoff year %d derived from analysis_window", cutoff_year)

    country_trends = aggregate_trends_by_country(trends)
    emissions = cumulative_emissions_per_capita(load_owid_co2(co2_path), cutoff_year)
    table = join_country_data(
        country_trends, emissions, load_continents(continents_path)
    )
    result = quantify_inequality(table)

    fig = render_inequality_scatter(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / "inequality_scatter.html"
    fig.write_html(figure_path)
    write_typed_parquet(
        table, table_path, INEQUALITY_SCHEMA, order_by=("continent", "Country")
    )
    logger.info("wrote %s and %s", table_path, figure_path)

    return {
        "table": table,
        "result": result,
        "figure_path": figure_path,
        "table_path": table_path,
    }


def main() -> None:
    """Run the Phase 4 pipeline and print effect sizes with uncertainty."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = build_inequality_analysis()
    r = out["result"]
    print(f"countries: {r.n_countries} across {r.n_continents} continents")
    print(f"Spearman rho: {r.spearman_rho:+.3f} (p={r.spearman_p:.2g})")
    print("OLS, °C/decade per 10x cumulative per-capita CO2 (HC1 robust):")
    for label, fit in (("pooled", r.ols_pooled), ("continent FE", r.ols_fe)):
        print(
            f"  {label:>12}: {fit.coef:+.4f} "
            f"[95% CI {fit.ci_low:+.4f}, {fit.ci_high:+.4f}], "
            f"p={fit.p_value:.2g}, R²={fit.r2:.3f}"
        )
    print(f"table:  {out['table_path']}")
    print(f"figure: {out['figure_path']}")


if __name__ == "__main__":
    main()
