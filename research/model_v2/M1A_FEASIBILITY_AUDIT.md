# M1a feasibility audit under the original measurement gates

**Recorded 2026-09-15. This is a measurement-feasibility audit, not a model result.** No M1a model has been fitted or scored.

## What happened

The first complete M1a geography build applied `M1A_MEASUREMENT_SPEC.md` as frozen before implementation, with its original §3.3 gates:
* ≥98% valid area for elevation and continentality;
* for climate, ≥95% classified area plus a robust winner;
* 100% geometry.

Two independent full builds were byte-identical. Both used builder `m1a_geography.py` at sha256 `920c0577…`, the version committed with this audit.

* **147 of 151 countries passed all six gates.**
* **Five country × feature gates failed across four countries:**
  * elevation: Philippines, Denmark, Norway, Bahamas;
  * climate zone: Bahamas.
* The failures were **independently reproduced**. A brute-force recomputation (no quadtree or full-block shortcut) matched the build to every printed digit for the Bahamas and Denmark, and a per-pixel diagnostic confirmed all four countries (`feasibility/`).
* **Cause of the elevation failures:** GSHHG terrestrial area that lies in coastline-straddling 60″ ETOPO pixels whose centres are water. The original centre-on-land rule gives that land no elevation.
* **Cause of the Bahamas climate failure:** coarse Köppen 0.5° cells coded as ocean over Bahamian land. The winning class was already provably invariant: A exceeds B by more than all unclassified land.

These failures are source-resolution/support limitations under the original rules, not implementation defects. They show a mismatch between the gate heuristics and whether the intended geography quantity is measurable. The project owner therefore approved a **pre-result specification amendment** (see `M1A_MEASUREMENT_SPEC.md`, Amendment 1). No threshold was lowered and no country was dropped, imputed or given a station-derived value. The artifacts below are kept unchanged as the evidence for that amendment.

Artifacts: `outputs/m1a_feasibility_original_gates/` (features, QA, manifest, per-cell land area, 151 × 6 gate report and failure diagnostics). Audit scripts: `feasibility/`.

## Complete 151 × 6 coverage table

Coverage = valid terrestrial support / terrestrial support. Required: latitude and hemisphere 1.00 (geometry); elevation and continentality ≥ 0.98; climate ≥ 0.95 classified **and** winner margin > unclassified area; spatial block nonmissing. ✗ marks a failed gate.

| ISO3 | Country | Terrestrial km² | abs_latitude | hemisphere | elevation | continentality | climate_zone (classified) | Köppen margin/unclassified | spatial_block |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DZA | Algeria | 2,314,400 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 0.9999 | 15613.5 | 1.0000 |
| AGO | Angola | 1,255,423 | 1.0000 | 1.0000 | 0.9996 | 1.0000 | 0.9998 | 45.1 | 1.0000 |
| BEN | Benin | 114,531 | 1.0000 | 1.0000 | 0.9996 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| BWA | Botswana | 579,597 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| BFA | Burkina Faso | 276,301 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| BDI | Burundi | 24,951 | 1.0000 | 1.0000 | 0.9981 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| CMR | Cameroon | 461,708 | 1.0000 | 1.0000 | 0.9992 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| CAF | Central African Republic | 623,106 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| TCD | Chad | 1,272,600 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| COG | Congo | 340,982 | 1.0000 | 1.0000 | 0.9991 | 1.0000 | 1.0000 | 209862.8 | 1.0000 |
| COD | Congo (Democratic Republic Of The) | 2,301,557 | 1.0000 | 1.0000 | 0.9993 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| CIV | Côte D'Ivoire | 321,055 | 1.0000 | 1.0000 | 0.9988 | 1.0000 | 1.0000 | 46665.5 | 1.0000 |
| DJI | Djibouti | 22,047 | 1.0000 | 1.0000 | 0.9957 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| EGY | Egypt | 997,578 | 1.0000 | 1.0000 | 0.9985 | 1.0000 | 0.9996 | 2525.5 | 1.0000 |
| GNQ | Equatorial Guinea | 27,095 | 1.0000 | 1.0000 | 0.9953 | 1.0000 | 0.9994 | 1728.8 | 1.0000 |
| ERI | Eritrea | 120,700 | 1.0000 | 1.0000 | 0.9950 | 1.0000 | 0.9984 | 641.2 | 1.0000 |
| ETH | Ethiopia | 1,127,266 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| GAB | Gabon | 268,289 | 1.0000 | 1.0000 | 0.9982 | 1.0000 | 0.9995 | 1936.2 | 1.0000 |
| GHA | Ghana | 229,026 | 1.0000 | 1.0000 | 0.9958 | 1.0000 | 0.9994 | 1563.5 | 1.0000 |
| GIN | Guinea | 246,704 | 1.0000 | 1.0000 | 0.9987 | 1.0000 | 0.9995 | 1923.0 | 1.0000 |
| GNB | Guinea Bissau | 34,238 | 1.0000 | 1.0000 | 0.9840 | 1.0000 | 0.9999 | 19218.2 | 1.0000 |
| KEN | Kenya | 581,505 | 1.0000 | 1.0000 | 0.9989 | 1.0000 | 0.9999 | 3645.2 | 1.0000 |
| LSO | Lesotho | 30,915 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| LBR | Liberia | 96,519 | 1.0000 | 1.0000 | 0.9984 | 1.0000 | 0.9988 | 827.8 | 1.0000 |
| LBY | Libya | 1,618,837 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 0.9999 | 8205.6 | 1.0000 |
| MDG | Madagascar | 592,900 | 1.0000 | 1.0000 | 0.9972 | 1.0000 | 0.9983 | 163.4 | 1.0000 |
| MWI | Malawi | 98,042 | 1.0000 | 1.0000 | 0.9974 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| MLI | Mali | 1,251,659 | 1.0000 | 1.0000 | 0.9992 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| MRT | Mauritania | 1,041,473 | 1.0000 | 1.0000 | 0.9996 | 1.0000 | 0.9998 | 4874.5 | 1.0000 |
| MAR | Morocco | 409,415 | 1.0000 | 1.0000 | 0.9991 | 1.0000 | 0.9998 | 2144.7 | 1.0000 |
| MOZ | Mozambique | 779,437 | 1.0000 | 1.0000 | 0.9979 | 1.0000 | 0.9992 | 644.4 | 1.0000 |
| NAM | Namibia | 820,093 | 1.0000 | 1.0000 | 0.9996 | 1.0000 | 0.9997 | 3443.4 | 1.0000 |
| NER | Niger | 1,187,341 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| NGA | Nigeria | 894,660 | 1.0000 | 1.0000 | 0.9986 | 1.0000 | 0.9997 | 2020.7 | 1.0000 |
| RWA | Rwanda | 24,861 | 1.0000 | 1.0000 | 0.9960 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| SEN | Senegal | 192,441 | 1.0000 | 1.0000 | 0.9977 | 1.0000 | 0.9995 | 987.8 | 1.0000 |
| SLE | Sierra Leone | 74,510 | 1.0000 | 1.0000 | 0.9960 | 1.0000 | 0.9978 | 447.0 | 1.0000 |
| SOM | Somalia | 633,117 | 1.0000 | 1.0000 | 0.9988 | 1.0000 | 0.9997 | 3677.6 | 1.0000 |
| ZAF | South Africa | 1,220,022 | 1.0000 | 1.0000 | 0.9992 | 1.0000 | 0.9995 | 825.7 | 1.0000 |
| SDN | Sudan | 1,862,260 | 1.0000 | 1.0000 | 0.9998 | 1.0000 | 1.0000 | 20132.2 | 1.0000 |
| SWZ | Swaziland | 17,280 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| TZA | Tanzania | 890,810 | 1.0000 | 1.0000 | 0.9982 | 1.0000 | 0.9998 | 2949.7 | 1.0000 |
| TGO | Togo | 61,397 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| TUN | Tunisia | 155,503 | 1.0000 | 1.0000 | 0.9977 | 1.0000 | 0.9985 | 497.7 | 1.0000 |
| UGA | Uganda | 205,926 | 1.0000 | 1.0000 | 0.9960 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| ZMB | Zambia | 741,217 | 1.0000 | 1.0000 | 0.9994 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| ZWE | Zimbabwe | 388,796 | 1.0000 | 1.0000 | 0.9995 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| AFG | Afghanistan | 642,179 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| ARM | Armenia | 29,226 | 1.0000 | 1.0000 | 0.9986 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| AZE | Azerbaijan | 85,760 | 1.0000 | 1.0000 | 0.9978 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| BGD | Bangladesh | 135,738 | 1.0000 | 1.0000 | 0.9925 | 1.0000 | 0.9997 | 2515.8 | 1.0000 |
| MMR | Burma | 667,744 | 1.0000 | 1.0000 | 0.9963 | 1.0000 | 0.9985 | 58.5 | 1.0000 |
| KHM | Cambodia | 176,115 | 1.0000 | 1.0000 | 0.9959 | 1.0000 | 0.9997 | 3934.1 | 1.0000 |
| CHN | China | 9,314,818 | 1.0000 | 1.0000 | 0.9990 | 1.0000 | 0.9998 | 494.5 | 1.0000 |
| GEO | Georgia | 69,287 | 1.0000 | 1.0000 | 0.9988 | 1.0000 | 0.9999 | 3708.8 | 1.0000 |
| IND | India | 3,158,591 | 1.0000 | 1.0000 | 0.9989 | 1.0000 | 0.9995 | 127.8 | 1.0000 |
| IDN | Indonesia | 1,897,440 | 1.0000 | 1.0000 | 0.9907 | 1.0000 | 0.9947 | 179.2 | 1.0000 |
| IRN | Iran | 1,617,661 | 1.0000 | 1.0000 | 0.9993 | 1.0000 | 0.9997 | 3043.2 | 1.0000 |
| IRQ | Iraq | 440,935 | 1.0000 | 1.0000 | 0.9990 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| ISR | Israel | 21,871 | 1.0000 | 1.0000 | 0.9970 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| JPN | Japan | 371,511 | 1.0000 | 1.0000 | 0.9878 | 1.0000 | 0.9945 | 2.9 | 1.0000 |
| JOR | Jordan | 86,156 | 1.0000 | 1.0000 | 0.9998 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| KAZ | Kazakhstan | 2,641,278 | 1.0000 | 1.0000 | 0.9990 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| LAO | Laos | 230,513 | 1.0000 | 1.0000 | 0.9988 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| LBN | Lebanon | 10,045 | 1.0000 | 1.0000 | 0.9957 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| MYS | Malaysia | 330,714 | 1.0000 | 1.0000 | 0.9949 | 1.0000 | 0.9981 | 520.0 | 1.0000 |
| MNG | Mongolia | 1,549,502 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| NPL | Nepal | 143,584 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| OMN | Oman | 311,211 | 1.0000 | 1.0000 | 0.9982 | 1.0000 | 0.9987 | 763.0 | 1.0000 |
| PAK | Pakistan | 879,331 | 1.0000 | 1.0000 | 0.9990 | 1.0000 | 0.9999 | 15143.8 | 1.0000 |
| PHL | Philippines | 296,328 | 1.0000 | 1.0000 | 0.9798 ✗ | 1.0000 | 0.9907 | 106.7 | 1.0000 |
| QAT | Qatar | 11,392 | 1.0000 | 1.0000 | 0.9833 | 1.0000 | 0.9896 | 95.5 | 1.0000 |
| SAU | Saudi Arabia | 1,913,198 | 1.0000 | 1.0000 | 0.9993 | 1.0000 | 0.9996 | 2600.3 | 1.0000 |
| KOR | South Korea | 98,072 | 1.0000 | 1.0000 | 0.9816 | 1.0000 | 0.9924 | 85.7 | 1.0000 |
| LKA | Sri Lanka | 66,241 | 1.0000 | 1.0000 | 0.9919 | 1.0000 | 0.9957 | 232.1 | 1.0000 |
| SYR | Syria | 186,305 | 1.0000 | 1.0000 | 0.9996 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| TWN | Taiwan | 36,340 | 1.0000 | 1.0000 | 0.9899 | 1.0000 | 0.9894 | 77.4 | 1.0000 |
| TJK | Tajikistan | 139,994 | 1.0000 | 1.0000 | 0.9996 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| THA | Thailand | 516,325 | 1.0000 | 1.0000 | 0.9974 | 1.0000 | 0.9985 | 688.5 | 1.0000 |
| TUR | Turkey | 771,186 | 1.0000 | 1.0000 | 0.9974 | 1.0000 | 0.9995 | 28.2 | 1.0000 |
| TKM | Turkmenistan | 468,852 | 1.0000 | 1.0000 | 0.9991 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| ARE | United Arab Emirates | 79,472 | 1.0000 | 1.0000 | 0.9942 | 1.0000 | 0.9968 | 313.4 | 1.0000 |
| UZB | Uzbekistan | 417,484 | 1.0000 | 1.0000 | 0.9992 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| VNM | Vietnam | 329,368 | 1.0000 | 1.0000 | 0.9937 | 1.0000 | 0.9981 | 82.6 | 1.0000 |
| YEM | Yemen | 454,854 | 1.0000 | 1.0000 | 0.9985 | 1.0000 | 0.9984 | 632.5 | 1.0000 |
| ALB | Albania | 27,823 | 1.0000 | 1.0000 | 0.9949 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| AUT | Austria | 83,605 | 1.0000 | 1.0000 | 0.9992 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| BLR | Belarus | 205,773 | 1.0000 | 1.0000 | 0.9991 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| BEL | Belgium | 30,652 | 1.0000 | 1.0000 | 0.9996 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| BIH | Bosnia And Herzegovina | 51,473 | 1.0000 | 1.0000 | 0.9996 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| BGR | Bulgaria | 111,390 | 1.0000 | 1.0000 | 0.9992 | 1.0000 | 0.9996 | 831.2 | 1.0000 |
| HRV | Croatia | 56,417 | 1.0000 | 1.0000 | 0.9856 | 1.0000 | 0.9987 | 125.5 | 1.0000 |
| CYP | Cyprus | 9,300 | 1.0000 | 1.0000 | 0.9843 | 1.0000 | 0.9935 | 80.5 | 1.0000 |
| CZE | Czech Republic | 79,805 | 1.0000 | 1.0000 | 0.9995 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| DNK | Denmark | 43,337 | 1.0000 | 1.0000 | 0.9765 ✗ | 1.0000 | 0.9930 | 10.8 | 1.0000 |
| EST | Estonia | 43,810 | 1.0000 | 1.0000 | 0.9904 | 1.0000 | 0.9941 | 169.7 | 1.0000 |
| FIN | Finland | 303,262 | 1.0000 | 1.0000 | 0.9826 | 1.0000 | 0.9989 | 936.3 | 1.0000 |
| FRA | France | 547,027 | 1.0000 | 1.0000 | 0.9979 | 1.0000 | 0.9996 | 2267.4 | 1.0000 |
| DEU | Germany | 355,499 | 1.0000 | 1.0000 | 0.9976 | 1.0000 | 0.9997 | 949.6 | 1.0000 |
| GRC | Greece | 132,580 | 1.0000 | 1.0000 | 0.9812 | 1.0000 | 0.9922 | 69.2 | 1.0000 |
| HUN | Hungary | 90,770 | 1.0000 | 1.0000 | 0.9994 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| ISL | Iceland | 101,299 | 1.0000 | 1.0000 | 0.9883 | 1.0000 | 0.9972 | 327.1 | 1.0000 |
| IRL | Ireland | 69,129 | 1.0000 | 1.0000 | 0.9855 | 1.0000 | 0.9974 | 377.2 | 1.0000 |
| ITA | Italy | 298,918 | 1.0000 | 1.0000 | 0.9945 | 1.0000 | 0.9951 | 135.5 | 1.0000 |
| LVA | Latvia | 63,182 | 1.0000 | 1.0000 | 0.9971 | 1.0000 | 0.9993 | 1386.9 | 1.0000 |
| LTU | Lithuania | 63,429 | 1.0000 | 1.0000 | 0.9977 | 1.0000 | 0.9997 | 3585.6 | 1.0000 |
| MKD | Macedonia | 23,833 | 1.0000 | 1.0000 | 0.9989 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| MDA | Moldova | 34,579 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| MNE | Montenegro | 13,085 | 1.0000 | 1.0000 | 0.9953 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| NLD | Netherlands | 34,324 | 1.0000 | 1.0000 | 0.9887 | 1.0000 | 0.9989 | 927.7 | 1.0000 |
| NOR | Norway | 320,145 | 1.0000 | 1.0000 | 0.9792 ✗ | 1.0000 | 0.9976 | 239.9 | 1.0000 |
| POL | Poland | 307,429 | 1.0000 | 1.0000 | 0.9982 | 1.0000 | 0.9997 | 3740.8 | 1.0000 |
| PRT | Portugal | 92,423 | 1.0000 | 1.0000 | 0.9949 | 1.0000 | 0.9957 | 231.5 | 1.0000 |
| ROU | Romania | 235,923 | 1.0000 | 1.0000 | 0.9991 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| RUS | Russia | 16,653,693 | 1.0000 | 1.0000 | 0.9982 | 1.0000 | 0.9996 | 1724.7 | 1.0000 |
| SRB | Serbia | 79,153 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| SVK | Slovakia | 49,894 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| SVN | Slovenia | 22,029 | 1.0000 | 1.0000 | 0.9994 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| ESP | Spain | 503,318 | 1.0000 | 1.0000 | 0.9967 | 1.0000 | 0.9983 | 12.1 | 1.0000 |
| SWE | Sweden | 428,161 | 1.0000 | 1.0000 | 0.9893 | 1.0000 | 0.9990 | 916.2 | 1.0000 |
| CHE | Switzerland | 38,494 | 1.0000 | 1.0000 | 0.9960 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| UKR | Ukraine | 589,296 | 1.0000 | 1.0000 | 0.9976 | 1.0000 | 0.9996 | 1945.2 | 1.0000 |
| GBR | United Kingdom | 243,712 | 1.0000 | 1.0000 | 0.9880 | 1.0000 | 0.9976 | 422.6 | 1.0000 |
| BHS | Bahamas | 13,426 | 1.0000 | 1.0000 | 0.8888 ✗ | 1.0000 | 0.8433 ✗ | 3.8 | 1.0000 |
| CAN | Canada | 9,489,887 | 1.0000 | 1.0000 | 0.9937 | 1.0000 | 0.9991 | 555.4 | 1.0000 |
| CRI | Costa Rica | 51,491 | 1.0000 | 1.0000 | 0.9930 | 1.0000 | 0.9981 | 525.2 | 1.0000 |
| CUB | Cuba | 111,481 | 1.0000 | 1.0000 | 0.9818 | 1.0000 | 0.9932 | 145.9 | 1.0000 |
| DOM | Dominican Republic | 48,032 | 1.0000 | 1.0000 | 0.9930 | 1.0000 | 0.9967 | 262.8 | 1.0000 |
| SLV | El Salvador | 21,360 | 1.0000 | 1.0000 | 0.9924 | 1.0000 | 0.9999 | 9105.1 | 1.0000 |
| GTM | Guatemala | 104,898 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 0.9986 | 380.8 | 1.0000 |
| HTI | Haiti | 27,247 | 1.0000 | 1.0000 | 0.9849 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| HND | Honduras | 111,052 | 1.0000 | 1.0000 | 0.9966 | 1.0000 | 0.9970 | 328.6 | 1.0000 |
| JAM | Jamaica | 11,034 | 1.0000 | 1.0000 | 0.9838 | 1.0000 | 0.9960 | 248.6 | 1.0000 |
| MEX | Mexico | 1,952,540 | 1.0000 | 1.0000 | 0.9975 | 1.0000 | 0.9987 | 189.2 | 1.0000 |
| NIC | Nicaragua | 120,726 | 1.0000 | 1.0000 | 0.9952 | 1.0000 | 0.9975 | 391.7 | 1.0000 |
| PAN | Panama | 75,970 | 1.0000 | 1.0000 | 0.9869 | 1.0000 | 0.9957 | 229.7 | 1.0000 |
| USA | United States | 9,254,175 | 1.0000 | 1.0000 | 0.9973 | 1.0000 | 0.9994 | 206.2 | 1.0000 |
| AUS | Australia | 7,699,513 | 1.0000 | 1.0000 | 0.9985 | 1.0000 | 0.9993 | 949.8 | 1.0000 |
| NZL | New Zealand | 265,218 | 1.0000 | 1.0000 | 0.9880 | 1.0000 | 0.9958 | 229.8 | 1.0000 |
| PNG | Papua New Guinea | 464,900 | 1.0000 | 1.0000 | 0.9917 | 1.0000 | 0.9935 | 128.6 | 1.0000 |
| ARG | Argentina | 2,763,076 | 1.0000 | 1.0000 | 0.9991 | 1.0000 | 0.9998 | 759.2 | 1.0000 |
| BOL | Bolivia | 1,080,947 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| BRA | Brazil | 8,471,386 | 1.0000 | 1.0000 | 0.9988 | 1.0000 | 0.9999 | 5918.2 | 1.0000 |
| CHL | Chile | 753,162 | 1.0000 | 1.0000 | 0.9848 | 1.0000 | 0.9983 | 73.0 | 1.0000 |
| COL | Colombia | 1,140,967 | 1.0000 | 1.0000 | 0.9990 | 1.0000 | 0.9998 | 5282.7 | 1.0000 |
| ECU | Ecuador | 256,247 | 1.0000 | 1.0000 | 0.9970 | 1.0000 | 0.9974 | 217.0 | 1.0000 |
| PRY | Paraguay | 399,108 | 1.0000 | 1.0000 | 0.9996 | 1.0000 | 1.0000 | no unclassified | 1.0000 |
| PER | Peru | 1,288,540 | 1.0000 | 1.0000 | 0.9992 | 1.0000 | 0.9999 | 7208.4 | 1.0000 |
| SUR | Suriname | 144,637 | 1.0000 | 1.0000 | 0.9983 | 1.0000 | 1.0000 | 76058.8 | 1.0000 |
| URY | Uruguay | 176,132 | 1.0000 | 1.0000 | 0.9977 | 1.0000 | 0.9998 | 4296.0 | 1.0000 |
| VEN | Venezuela | 906,979 | 1.0000 | 1.0000 | 0.9976 | 1.0000 | 0.9994 | 1676.8 | 1.0000 |

## Near-threshold passes

Elevation coverage in [0.980, 0.985): GRC 0.9812, KOR 0.9816, CUB 0.9818, FIN 0.9826, QAT 0.9833, JAM 0.9838, GNB 0.9840, CYP 0.9843, CHL 0.9848, HTI 0.9849

Classified Köppen coverage in [0.95, 0.96): none

Köppen winner margin < 3 × unclassified area (passes, but least robust): JPN margin 5,948 km² vs unclassified 2,042 km²

## The five failed country × feature gates

Every value below was reproduced by an independent brute-force computation (no quadtree, no full-block shortcut) for Bahamas and Denmark, and by a per-pixel diagnostic for all four countries. Build 1 and build 2 are byte-identical.

| Country | Feature | Achieved | Required | Shortfall | Valid support | Missing support | Concentration | How missing support differs |
|---|---|---:|---:|---:|---|---|---|---|
| Philippines | elevation | 0.979797 | 0.98 | 0.000203 | 88,297 px / 290,341 km² | 9,956 px / 5,987 km² (mean land fraction 0.18) | 93/94 1° cells; top-5 cells 17% | valid mean 324.2 m; valid coastal pixels 38.9 m; bound if missing=coastal 318.4 m, if 0 m 317.6 m |
| Denmark | elevation | 0.976491 | 0.98 | 0.003509 | 22,491 px / 42,318 km² | 2,484 px / 1,019 km² (mean land fraction 0.21) | 22/22 1° cells; top-5 cells 53% | valid mean 31.3 m; valid coastal pixels 8.0 m; bound if missing=coastal 30.8 m, if 0 m 30.6 m |
| Norway | elevation | 0.979155 | 0.98 | 0.000845 | 217,028 px / 313,471 km² | 20,919 px / 6,673 km² (mean land fraction 0.22) | 105/117 1° cells; top-5 cells 19% | valid mean 561.7 m; valid coastal pixels 129.4 m; bound if missing=coastal 552.7 m, if 0 m 550.0 m |
| Bahamas | elevation | 0.888799 | 0.98 | 0.091201 | 4,288 px / 11,933 km² | 2,819 px / 1,493 km² (mean land fraction 0.17) | 32/32 1° cells; top-5 cells 50% | valid mean 0.8 m; valid coastal pixels -2.2 m; bound if missing=coastal 0.5 m, if 0 m 0.7 m |
| Bahamas | climate_zone | 0.843343 | 0.95 | 0.106657 | 23 of 79 Köppen 0.5° px / 11,323 km² (A 9,701, B 1,622) | 56 Köppen 0.5° px coded 0 / 2,103 km² | top-5 0.5° px 31% of unclassified; spread over the archipelago | neighbour-majority class of unclassified land: A 1,633, B 223, none 247 km²; margin A−B 8,079 km² > unclassified 2,103 km² |

**Origin of the elevation failures (PHL, DNK, NOR, BHS).** GSHHG full-resolution land intersected with the ETOPO 2022 60″ pixel-centre rule (spec §4 elevation). Land in a pixel whose centre GSHHG classifies as water is, by specification, not valid elevation support. The ETOPO value at such a centre is a water/bathymetry sample: median −12, −1.6, −26 and −5 m, and 70–78% below 0 m. The missing land consists of coastal slivers, skerries, islets and the edges of fjords/lagoons, with a mean land fraction of 0.17–0.22 per missing pixel. The cause is not GPW country assignment (the support area is identical under independent recomputation) or Köppen. It is the documented rule operating at the resolution of the two sources.

**Origin of the Bahamas climate failure.** The Beck et al. 0.5° Köppen raster codes 56 of the 79 half-degree pixels overlapping Bahamian GSHHG land as 0 (ocean/no data). This is a source-raster land/sea resolution limitation. The robust-winner margin rule passes (A−B margin exceeds all unclassified area), so the category could not change even if every unclassified km² were the runner-up. Only the ≥95% classified-coverage gate fails.

**Faithful implementation or defect?** Faithful. The implementation reproduces the frozen definition exactly, and the independent recomputation matches to all printed digits. No implementation using the same frozen sources and definitions can recover this support:
* recovering the elevation support would require changing the pixel-centre validity rule or using a finer elevation source;
* recovering the Köppen support would require nearest-land filling (forbidden by the spec) or a different climate raster.

These are intrinsic source-resolution/coverage limitations under the frozen definitions.

**Is the missing area systematically different?** For elevation, yes, in direction: the missing slivers are coastal, and valid coastal pixels are far lower than the national means. The printed bounds show the resulting national-mean shifts are small:
* PHL 324 → ~318 m;
* DNK 31.3 → ~30.8 m;
* NOR 562 → ~553 m;
* BHS 0.8 → ~0.5 m.

For the Bahamas' Köppen class, the unclassified land's neighbour-majority class is mostly A, the observed winner.

**Distance method check (not a gate failure).**
* Adaptive rule: 3′ block quadrature, 5′ comparison globally, exact 60″ pixels for 10 representative countries; the fallback was defined before any score.
* Accuracy: max |3′−5′| = 0.095 km; max |3′−60″| = 0.034 km. The refinement pass was not triggered; 151/151 countries use 3′.
* Seams: 5 Natural Earth seam/polar-cap edges were removed.
* Source roles as frozen: Natural Earth 110m for distance, GSHHG as mask.
* Audit-trail note: a mid-session instruction briefly named GSHHG as the coastline source. The frozen spec already resolves this, so it is not an open decision.
