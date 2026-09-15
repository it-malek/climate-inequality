"""UN M49 sub-regions for the M0 countries (hand-transcribed from the M49 standard).

Used only as an independent, sub-continental regionalisation for residual
inspection and for the leave-one-region-out cross-validation candidates. The V1
``spatial_block`` (OWID continent) is left untouched; Cyprus is a European
country there and a Western Asian one here, as in M49.
"""

from __future__ import annotations

import pandas as pd

M49_SUBREGION: dict[str, tuple[str, ...]] = {
    "Northern Africa": ("DZA", "EGY", "LBY", "MAR", "SDN", "TUN"),
    "Western Africa": ("BEN", "BFA", "CIV", "GHA", "GIN", "GNB", "LBR", "MLI", "MRT", "NER", "NGA", "SEN", "SLE", "TGO"),
    "Middle Africa": ("AGO", "CMR", "CAF", "TCD", "COG", "COD", "GNQ", "GAB"),
    "Eastern Africa": ("BDI", "DJI", "ERI", "ETH", "KEN", "MDG", "MWI", "MOZ", "RWA", "SOM", "TZA", "UGA", "ZMB", "ZWE"),
    "Southern Africa": ("BWA", "LSO", "NAM", "ZAF", "SWZ"),
    "Central Asia": ("KAZ", "TJK", "TKM", "UZB"),
    "Eastern Asia": ("CHN", "JPN", "MNG", "KOR", "TWN"),
    "South-eastern Asia": ("MMR", "KHM", "IDN", "LAO", "MYS", "PHL", "THA", "VNM"),
    "Southern Asia": ("AFG", "BGD", "IND", "IRN", "NPL", "PAK", "LKA"),
    "Western Asia": ("ARM", "AZE", "CYP", "GEO", "IRQ", "ISR", "JOR", "LBN", "OMN", "QAT", "SAU", "SYR", "TUR", "ARE", "YEM"),
    "Northern Europe": ("DNK", "EST", "FIN", "ISL", "IRL", "LVA", "LTU", "NOR", "SWE", "GBR"),
    "Western Europe": ("AUT", "BEL", "FRA", "DEU", "NLD", "CHE"),
    "Eastern Europe": ("BLR", "BGR", "CZE", "HUN", "MDA", "POL", "ROU", "RUS", "SVK", "UKR"),
    "Southern Europe": ("ALB", "BIH", "HRV", "GRC", "ITA", "MKD", "MNE", "PRT", "SRB", "SVN", "ESP"),
    "Northern America": ("CAN", "USA"),
    "Central America": ("CRI", "SLV", "GTM", "HND", "MEX", "NIC", "PAN"),
    "Caribbean": ("BHS", "CUB", "DOM", "HTI", "JAM"),
    "South America": ("ARG", "BOL", "BRA", "CHL", "COL", "ECU", "PRY", "PER", "SUR", "URY", "VEN"),
    "Australia and New Zealand": ("AUS", "NZL"),
    "Melanesia": ("PNG",),
}

M49_REGION_OF_SUBREGION: dict[str, str] = {
    **{k: "Africa" for k in ("Northern Africa", "Western Africa", "Middle Africa", "Eastern Africa", "Southern Africa")},
    **{k: "Asia" for k in ("Central Asia", "Eastern Asia", "South-eastern Asia", "Southern Asia", "Western Asia")},
    **{k: "Europe" for k in ("Northern Europe", "Western Europe", "Eastern Europe", "Southern Europe")},
    **{k: "Americas" for k in ("Northern America", "Central America", "Caribbean", "South America")},
    **{k: "Oceania" for k in ("Australia and New Zealand", "Melanesia")},
}


def m49_table() -> pd.DataFrame:
    """One row per ISO3: ``iso3``, ``m49_subregion``, ``m49_region``."""
    rows = [
        {"iso3": iso, "m49_subregion": sub, "m49_region": M49_REGION_OF_SUBREGION[sub]}
        for sub, members in M49_SUBREGION.items()
        for iso in members
    ]
    table = pd.DataFrame(rows)
    if table["iso3"].duplicated().any():
        raise ValueError("an ISO3 code is listed in two sub-regions")
    return table
