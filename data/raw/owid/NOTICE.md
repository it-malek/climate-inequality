# Our World in Data continent classification (vendored copy)

`continents.csv` is the "Continents according to Our World in Data" grapher
table (columns `Entity`, `Code`, `Year`, `World region according to OWID`).

- Source: <https://ourworldindata.org/grapher/continents-according-to-our-world-in-data>
- Retrieved: 2026-06
- Licence: CC BY 4.0

It supplies the continent labels used for the fixed-effects regression, the
Theil between/within split, the `spatial_block` feature of the decomposition
and the block bootstrap. The file is committed so those groupings are pinned;
the much larger `owid-co2-data.csv` in the same directory is downloaded by
`src.emissions` and is not committed.
