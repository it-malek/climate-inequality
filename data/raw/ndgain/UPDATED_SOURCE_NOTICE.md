# ND-GAIN updated-source extract

ND-GAIN Country Index 2026 release_data_v2, obtained from the official download
page on 2026-10-22: https://gain.nd.edu/our-work/country-index/download-data/

Archive SHA-256: `e854c7fe6ccef8c5d24a5ef7f370b7f61d1b8c6816da19fda4cd892ca6cdf0de`.
All metrics select exactly observation year 2023, without older-year fallback.
192 ISO3 rows; vulnerability/readiness 0–1, combined index 0–100.
Original prepared `ndgain_latest.csv` and its notice are historical and unchanged.
The original raw archive was not recovered. This is an explicitly updated source.
See the original notice for attribution/licensing and `docs/current_validation.md`
for the checksum-pinned local-archive extraction command.

## Source terms and attribution

The ND-GAIN data-download terms identify **Creative Commons Attribution 3.0
Unported (CC BY 3.0)**: <https://data-nd-gain.crc.nd.edu/about/download>.
License: <https://creativecommons.org/licenses/by/3.0/>.

Citation: Notre Dame Global Adaptation Initiative Country Index (ND-GAIN).
(2026). University of Notre Dame.

This extract selects the 2023 vulnerability, readiness and combined-index
columns, reshapes them to one row per ISO3 country and preserves missing values.
No score is re-estimated. It is derived from ND-GAIN and does not imply
endorsement by ND-GAIN or the University of Notre Dame. The source material
retains its CC BY 3.0 terms; no additional restriction is imposed by this repository.
