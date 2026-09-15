# M0 residual diagnostics

What the V1 model leaves unexplained, characterised without changing the model.
Every number is from `outputs/m0_residual_diagnostics.json`, the country table
`outputs/m0_countries.csv` and the figures in `outputs/figures/`, produced by
`research/model_v2/diagnostics.py` and `figures.py`. Residual = observed − fitted,
so a positive residual is a country that warmed faster than M0 predicts.

Two facts frame everything below. First, OLS residuals are exactly orthogonal to
every design column, so the residual has zero mean within every continent,
Köppen class, hemisphere and income group and zero linear correlation with the
V1 numeric features; sub-continental and nonlinear structure is the only kind
that can show. Second, the residual sd is 0.028 °C/decade against an outcome sd
of 0.047: the unexplained part is a third of the variance but 60% of the spread.

## 1. Distribution and scale

| | |
|---|---|
| n | 151 |
| sd / RMSE / MAE | 0.0282 / 0.0281 / 0.0224 °C/decade |
| Skew / excess kurtosis | +0.03 / −0.23 |
| Shapiro–Wilk p / Jarque–Bera p | 0.98 / 0.83 |
| 5th, 50th, 95th percentile | −0.046, +0.001, +0.043 |

The residual is as Gaussian as 151 values can be
(`m0_residual_distribution.png`). There is no heteroscedasticity in the fitted
value (Breusch–Pagan p = 0.93; Spearman of \|e\| with the fitted value 0.06),
none with latitude (−0.00), and only a weak increase of \|e\| with land area
(ρ = +0.16, p = 0.05) and with the arid share of the land (ρ = +0.19, p = 0.02).
Nothing about the error distribution argues for a robust or transformed
estimator; the problem is where the errors sit, not their shape.

## 2. Magnitude by country

The twenty largest residuals (`m0_top_residuals.png`):

| Faster than predicted | e | Slower than predicted | e |
|---|---|---|---|
| Brazil | +0.074 | Egypt | −0.072 |
| Canada | +0.064 | Mexico | −0.061 |
| Iran | +0.063 | Israel | −0.061 |
| Turkmenistan | +0.062 | Botswana | −0.056 |
| Mauritania | +0.051 | Argentina | −0.052 |
| Angola | +0.048 | Syria | −0.051 |
| Estonia | +0.045 | South Korea | −0.047 |
| Dominican Republic | +0.043 | New Zealand | −0.046 |
| Indonesia | +0.043 | Libya | −0.046 |
| Tanzania, Kenya, Morocco, Latvia | +0.040 to +0.041 | India, Slovakia, Burma, Romania | −0.040 to −0.045 |

The top 10 carry 32% of the residual sum of squares (a uniform spread would give
7%) and 20 countries carry half. That is moderate concentration, not
domination: the largest Cook's distance is 0.10 (New Zealand), eight countries
have \|studentised residual\| > 2 (the expected count under normality is about
seven), and no single country moves Moran's I by more than 0.026 (jackknife;
Brazil).

Leverage: Iceland is the only Köppen E country, so its dummy fits it exactly
(leverage 1, residual 0, and no out-of-sample information about class E exists
in the sample). Japan has leverage 0.47 through a station density of 30 (the
median is 1). The three Oceania countries sit at 0.36–0.39 because the block
has three members. Bolivia (0.31) is the elevation extreme (3,353 m at its
stations). These are properties of M0's design that any CV design must tolerate.

## 3. Spatial structure

**Global.** Moran's I is 0.269 under the V1 definition and positive and
significant under every reasonable alternative (`m0_moran_sensitivity.png`):
0.11–0.39 for station-centroid kNN with k = 15 down to 4; 0.15–0.44 for
land-area-centroid kNN; 0.47 for queen contiguity derived from the 15' national
grid (islands given their nearest neighbour); 0.20–0.45 for distance bands of
1000–2000 km; Geary's C = 0.66 (p = 0.001). Only the 3000 km distance band is
indistinguishable from zero. The conclusion "the residual is regionally
clustered" does not depend on the weight definition; the size of I does.

**Scale.** The correlogram (`m0_correlogram.png`, land-area centroids) shows
positive dependence to about 1500 km (I = 0.39 below 500 km, 0.30 at 500–1000,
0.14 at 1000–1500, all p ≤ 0.005), nothing at 1500–2000 km (0.04, p = 0.38), and
**significantly negative dependence at 2000–4000 km** (−0.13 and −0.14,
p = 0.001). The semivariogram rises to its sill between 2000 and 3000 km. Measured
instead on the minimum distance between territories (`borders.py`,
`outputs/border_distance_correlogram.json`), the positive dependence is
confined to the 0–500 km ring (I = 0.33, p = 0.001; 0.03 at 500–1000 km) and
turns negative beyond 1000 km: what the centroid metric spreads to 1500 km is
mostly adjacency, inflated by the size of the countries involved. The residual
is therefore not a smooth global trend surface; it is a pattern of regional
dipoles with a characteristic scale of 1000–2000 km between centroids, driven
by adjacent and near-adjacent countries.

**Local clusters** (`m0_lisa_map.png`; local Moran's I on the V1 weights,
conditional-permutation p < 0.05; 88% of labels unchanged under area-centroid
weights):

| Cluster | Countries | Mean residual |
|---|---|---|
| High–high, Central Asia | Kazakhstan, Uzbekistan, Turkmenistan, Tajikistan | +0.028 (sub-region) |
| High–high, Atlantic Sahel / West Africa | Mauritania, Senegal, Guinea-Bissau, Guinea, Sierra Leone, Liberia | +0.018 (sub-region) |
| High–high, Gulf | United Arab Emirates, Qatar | |
| High–high, Guianas | Suriname | |
| Low–low, eastern Mediterranean / Levant | Egypt, Israel, Syria, Turkey, Lebanon, Cyprus, Jordan | |
| Low–low, mainland South and South-east Asia | Burma, Bangladesh, Thailand, Laos, Sri Lanka | |
| Low–low, south-eastern Europe | Romania, Moldova | −0.020 (Eastern Europe) |
| Low–low, southern Africa | Zimbabwe, Lesotho | −0.027 (Southern Africa) |

Removing the ten largest \|e\| leaves I at 0.23; removing the 29 countries in the
two cluster types brings it to 0.07. The spatial pattern is carried by regions,
not by a few outliers, and it survives the removal of any single country.

**Sub-continental means** (`m0_residual_by_subregion.png`): the UN M49
sub-regions explain 26% of the residual variance (continents explain 0% by
construction). Positive: Northern America +0.039 (n = 2), Central Asia +0.028,
Western Africa +0.018, Caribbean +0.018, Western Europe +0.011, Northern Europe
+0.009. Negative: Southern Africa −0.027, Central America −0.024, Eastern Europe
−0.020, Northern Africa −0.014, Middle Africa −0.013, South-eastern Asia −0.010.

## 4. Latitude

Overall the binned residual is flat in latitude, as the linear term guarantees.
Split by hemisphere it is not (`m0_residual_vs_latitude.png`): in the Southern
Hemisphere the residual falls steeply with latitude (Spearman ρ = −0.68,
p = 5×10⁻⁵, n = 29; slope −0.0017 °C/decade per degree): the SH tropics are
under-predicted (+0.015, n = 18) and the SH subtropics and mid-latitudes
over-predicted (−0.024, n = 11: Argentina, Chile, Uruguay, South Africa,
Botswana, Namibia, Lesotho, Swaziland, Zimbabwe, Australia, New Zealand). The
Northern Hemisphere is flat (slope +0.0002) except that the 50–70° band is
under-predicted (+0.013, n = 17: Canada, Estonia, Latvia, Finland, Russia,
Lithuania, Sweden, Norway…). A single hemisphere level shift plus a single
latitude slope cannot represent a latitudinal gradient that differs by
hemisphere, nor a gradient that steepens toward the NH high latitudes.

## 5. Country size, coast, topography, regime

* **Size.** \|e\| grows weakly with land area (ρ = +0.16). More telling, the
  countries whose station network is most displaced from their land area are
  the largest residuals: Canada (station-mean latitude 46.8° against a land
  centroid at 61.0°; e = +0.064), Brazil (19.3° vs 10.7°; +0.074), Russia (53.7°
  vs 66.3°; +0.025), the United States (36.6° vs 45.3°; +0.014), Algeria (35.2°
  vs 28.1°; +0.031). Seven large countries have a within-country spread of cell
  trends larger than the residual sd.
* **Coast / continentality.** No relation with the coastal fraction of land
  cells (ρ = +0.11, p = 0.20); the residual is orthogonal to V1 continentality by
  construction and shows no binned nonlinearity in the scatter.
* **Topography.** No relation with within-country elevation spread (ρ = −0.03)
  or land-mean elevation (ρ = −0.12, p = 0.14). The V1 elevation feature itself
  is bathymetric for 17 countries (§9 of `M0_BASELINE.md`); its coefficient is
  effectively zero (−6×10⁻⁷ per metre), so the artefact costs little fit but
  says the feature as measured is not the country's altitude.
* **Climatic regime.** Mean residuals do not differ by the area-dominant Köppen
  class (η² = 0.006) or by whether the station-modal class disagrees with it
  (26 countries; η² = 0.004). Arid (B) countries have the widest residual spread
  (sd 0.036 vs 0.025–0.027 for A, C, D): both the fastest-warming HH clusters
  (Central Asia, Sahel, Gulf) and the slowest LL cluster (Levant, Egypt, Libya)
  are arid. Aridity by itself does not sort the residual; something that differs
  between arid regions does.

## 6. Observational product and construction

* **ERA5 versus Berkeley.** The ERA5 area trend disagrees with the Berkeley
  outcome by a country sd of 0.063 °C/decade (mean +0.030; rank agreement
  ρ = 0.64), larger than the outcome's own cross-country sd. The disagreement is
  spatially clustered (I = 0.18, p = 0.001) and regionally systematic: ERA5 warms
  Northern Africa +0.11, Western Asia +0.07, Central America +0.07 and Northern
  Europe +0.06 faster than Berkeley, and Australia/New Zealand −0.07 and the
  Caribbean −0.04 slower. Linearly, the residual is only weakly aligned with the
  gap (ρ = −0.14, p = 0.08; R² 0.02), but the low–low clusters sit exactly where
  Berkeley warms much less than ERA5 (Egypt 0.15 vs 0.28; Mexico 0.04 vs 0.18;
  Guatemala 0.05 vs 0.23; Libya 0.19 vs 0.30; Saudi Arabia 0.19 vs 0.42; Iraq
  0.18 vs 0.37). Country-level product uncertainty is at least as large as the
  residual, so part of what M0 leaves may be specific to the Berkeley field in
  station-sparse regions; it cannot be resolved from Berkeley alone.
* **Station versus area construction.** The residual is not aligned with the
  station-minus-area difference (ρ = −0.11, p = 0.16).

## 7. What the residual suggests, and what it cannot yet tell

Broad classes of omitted structure consistent with §§3–6, in the order the
evidence supports them:

1. **Functional form of the latitude effect**: the hemispheric asymmetry (§4) is
   the single clearest pattern and needs no new data.
2. **Measurement inconsistency between the area-based outcome and the
   station-based geography predictors** (§5, and `M0_BASELINE.md` §9): the
   largest residuals are the countries whose stations misrepresent their land.
3. **Regional physical structure at 1000–2000 km scale** (§3): arid interiors
   and the Sahel warming faster than their geography implies, the eastern
   Mediterranean, mainland South-east Asia, south-eastern Europe, southern
   Africa and Central America slower. Candidate mechanisms are soil-moisture
   limitation and land–atmosphere coupling, regional aerosol histories,
   snow–albedo feedback at NH high latitudes and ocean moderation of SH
   mid-latitude land (`HYPOTHESIS_REGISTER.md`).
4. **Observational-product structure** (§6): the low–low clusters coincide
   with the largest Berkeley–ERA5 disagreements.
5. **Internal decadal variability aliased into a 64-year linear trend**: the
   Pacific-facing negatives (Mexico, Central America, Chile, Argentina, New
   Zealand, Korea) and Atlantic-facing positives (West Africa, Caribbean,
   Brazil, north-eastern Europe) resemble the imprint of Pacific and Atlantic
   decadal modes over 1950–2013. Not testable cross-sectionally with static
   covariates.

What cannot be inferred yet: the residual alone does not separate classes 3, 4
and 5. A regional cluster can be a physical regime, a product artefact or a
decadal-mode imprint, and the same cluster (the Levant) is plausible under all
three. That is why the plan tests functional form and measurement first (they
are cheap and unambiguous), then pre-specified physical covariates under a
frozen spatial validation, and treats explicit spatial terms as an accounting of
what remains rather than as an explanation. Whether any of it generalises is a
question for the validation protocol, not for the residual.
