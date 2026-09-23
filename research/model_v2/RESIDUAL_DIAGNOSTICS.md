# Residual diagnostics of the static model

What the public decomposition leaves unexplained, characterised without
changing the model. Every number is from `outputs/m0_residual_diagnostics.json`,
the country table `outputs/m0_countries.csv` and the figures in
`outputs/figures/`, produced by `diagnostics.py` and `figures.py`. Residual is
observed minus fitted, so a positive residual is a country that warmed faster
than the model predicts.

Two facts frame everything below. OLS residuals are exactly orthogonal to
every design column, so the residual has zero mean within every continent,
Köppen class, hemisphere and income group and zero linear correlation with the
numeric features; sub-continental and nonlinear structure is the only kind
that can show. And the residual standard deviation is 0.028 °C/decade against
an outcome standard deviation of 0.047: the unexplained part is a third of the
variance but 60% of the spread.

## 1. Distribution and scale

| | |
|---|---|
| n | 151 |
| sd / RMSE / MAE | 0.0282 / 0.0281 / 0.0224 °C/decade |
| Skew / excess kurtosis | +0.03 / −0.23 |
| Shapiro–Wilk p / Jarque–Bera p | 0.98 / 0.83 |
| 5th, 50th, 95th percentile | −0.046, +0.001, +0.043 |

The residual is as Gaussian as 151 values can be (`m0_residual_distribution.png`).
There is no heteroscedasticity in the fitted value (Breusch–Pagan p = 0.93),
none with latitude, and only a weak increase of the absolute residual with land
area (ρ = +0.16, p = 0.05) and with the arid share of the land (ρ = +0.19,
p = 0.02). Nothing about the error distribution argues for a robust or
transformed estimator; the question is where the errors sit, not their shape.

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

The top ten carry 32% of the residual sum of squares (a uniform spread would
give 7%) and twenty countries carry half. That is moderate concentration, not
domination: the largest Cook's distance is 0.10 (New Zealand), eight countries
have an absolute studentised residual above 2 (about seven are expected under
normality), and no single country moves Moran's I by more than 0.026
(jackknife; Brazil).

Leverage: Iceland is the only Köppen E country, so its dummy fits it exactly
(leverage 1, residual 0). Japan has leverage 0.47 through a station density of
30 (the median is 1). The three Oceania countries sit at 0.36–0.39 because the
continent block has three members. Bolivia (0.31) is the elevation extreme
(3,353 m at its stations).

## 3. Spatial structure

**Global.** Moran's I is 0.269 under the public definition (station-centroid
kNN, k = 8) and positive and significant under every reasonable alternative
(`m0_moran_sensitivity.png`): 0.11–0.39 for station-centroid kNN with k from 15
down to 4; 0.15–0.44 for land-centroid kNN; 0.47 for queen contiguity derived
from the 15 arc-minute national grid; 0.20–0.45 for distance bands of
1000–2000 km; Geary's C = 0.66 (p = 0.001). Only the 3000 km distance band is
indistinguishable from zero. The conclusion that the residual is regionally
clustered does not depend on the weight definition; the size of I does.

**Scale.** The land-centroid correlogram (`m0_correlogram.png`) shows positive
dependence to about 1500 km (I = 0.39 below 500 km, 0.30 at 500–1000 km, 0.14
at 1000–1500 km, all p <= 0.005), nothing at 1500–2000 km, and significantly
negative dependence at 2000–4000 km (−0.13 and −0.14, p = 0.001). Measured on
territorial distance instead, the positive dependence is confined to the
0–500 km ring (I = 0.33) and turns negative beyond 1000 km: what the centroid
metric spreads to 1500 km is mostly adjacency, inflated by the size of the
countries involved. The residual is not a smooth global trend surface; it is a
pattern of regional dipoles driven by adjacent and near-adjacent countries.

**Local clusters** (`m0_lisa_map.png`; local Moran's I, conditional-permutation
p < 0.05; 88% of labels unchanged under land-centroid weights):

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

Removing the ten largest residuals leaves I at 0.23; removing the 29 countries
in the two cluster types brings it to 0.07. The spatial pattern is carried by
regions, not by a few outliers, and it survives the removal of any single
country. UN M49 sub-regions explain 26% of the residual variance (continents
explain 0% by construction): Northern America +0.039 (n = 2), Central Asia
+0.028, Western Africa +0.018, Caribbean +0.018 on the positive side; Southern
Africa −0.027, Central America −0.024, Eastern Europe −0.020, Northern Africa
−0.014 on the negative side (`m0_residual_by_subregion.png`).

## 4. Latitude

Overall the binned residual is flat in latitude, as the linear term guarantees.
Split by hemisphere it is not (`m0_residual_vs_latitude.png`): in the Southern
Hemisphere the residual falls steeply with latitude (Spearman ρ = −0.68,
p = 5e-5, n = 29): the southern tropics are under-predicted (+0.015, n = 18)
and the southern subtropics and mid-latitudes over-predicted (−0.024, n = 11:
Argentina, Chile, Uruguay, South Africa, Botswana, Namibia, Lesotho, Eswatini,
Zimbabwe, Australia, New Zealand). The Northern Hemisphere is flat except that
the 50–70° band is under-predicted (+0.013, n = 17: Canada, Estonia, Latvia,
Finland, Russia, Lithuania, Sweden, Norway and others). A single hemisphere
level shift plus a single latitude slope cannot represent a gradient that
differs by hemisphere. This pattern motivated the latitude functional-form test
in [`RESULTS.md`](RESULTS.md) §5; because the hypothesis was generated from the
same outcome, that test could not count as independent confirmation.

## 5. Country size, coast, topography, regime

* **Size.** The absolute residual grows weakly with land area (ρ = +0.16). More
  telling, the countries whose station network is most displaced from their
  land area are among the largest residuals: Canada (station-mean latitude
  46.8° against a land centroid at 61.0°; e = +0.064), Brazil (19.3° vs 10.7°;
  +0.074), Russia (53.7° vs 66.3°; +0.025), the United States (36.6° vs
  45.3°; +0.014), Algeria (35.2° vs 28.1°; +0.031). Seven large countries have
  a within-country spread of cell trends larger than the residual sd. This
  motivated the area-consistent re-measurement of the geography features.
* **Coast.** No relation with the coastal fraction of land cells (ρ = +0.11,
  p = 0.20).
* **Topography.** No relation with within-country elevation spread (ρ = −0.03)
  or land-mean elevation (ρ = −0.12). The station-sampled elevation feature is
  bathymetric for 17 countries because grid-snapped station coordinates fall in
  ocean cells of the relief model; its coefficient is effectively zero, so the
  artefact costs little fit but the feature as measured is not the country's
  altitude.
* **Climatic regime.** Mean residuals do not differ by the area-dominant Köppen
  class (η² = 0.006). Arid (B) countries have the widest residual spread
  (sd 0.036 against 0.025–0.027 for A, C and D): both the fastest-warming
  clusters (Central Asia, Sahel, Gulf) and the slowest (Levant, Egypt, Libya)
  are arid. Aridity by itself does not sort the residual.

## 6. Observational product

The ERA5 area-weighted trend disagrees with the Berkeley Earth outcome by a
country standard deviation of 0.063 °C/decade (rank agreement ρ = 0.64),
larger than the outcome's own cross-country standard deviation of 0.047. The
disagreement is spatially clustered (I = 0.18) and regionally systematic: ERA5
warms Northern Africa, Western Asia, Central America and Northern Europe faster
than Berkeley Earth, and Australia, New Zealand and the Caribbean slower.
Linearly the residual is only weakly aligned with the gap (ρ = −0.14), but the
low–low clusters sit where Berkeley Earth warms much less than ERA5 (Egypt 0.15
vs 0.28; Mexico 0.04 vs 0.18; Guatemala 0.05 vs 0.23; Libya 0.19 vs 0.30;
Saudi Arabia 0.19 vs 0.42; Iraq 0.18 vs 0.37 °C/decade). Country-level product
uncertainty is at least as large as the residual, so part of what the model
leaves may be specific to the Berkeley field in station-sparse regions.

Re-computing ERA5 as anomalies against its own 1951–1980 monthly climatology
(matching Berkeley Earth's convention) halves the mean gap (from −0.030 to
−0.016 °C/decade, Berkeley minus ERA5) and reduces its spatial structure, but
does not improve residual agreement: the refitted residuals correlate at
Pearson 0.334 (Spearman 0.326) and agree in sign for 89 of 151 countries
(0.362, 0.355 and 90 under the absolute-temperature construction). Positive
residual autocorrelation is present in both products under both weightings,
but ERA5's Moran's I is 30–41% of Berkeley Earth's. Of the regional and local
residual patterns, only Southern Africa's negative regional mean and the single-country
extremes of Canada, Estonia, Indonesia and Kenya (positive) and Argentina, the
Central African Republic and South Korea (negative) reproduce in both products
under both constructions; the Levant low–low clusters, the Central Asian and
West African high–high patterns and most top-15 lists are product-sensitive
(`outputs/product_stability_pattern_classification.csv`).

## 7. What the residual suggests

Broad classes of omitted structure consistent with the sections above, in the
order the evidence supports them:

1. **Functional form of the latitude effect**: the hemispheric asymmetry (§4)
   is the clearest pattern and needs no new data.
2. **Measurement inconsistency** between the area-based outcome and the
   station-based geography predictors (§5): the largest residuals are the
   countries whose stations misrepresent their land.
3. **Regional physical structure at 1000–2000 km scale** (§3): arid interiors
   and the Sahel warming faster than their geography implies; the eastern
   Mediterranean, mainland South-east Asia, south-eastern Europe, southern
   Africa and Central America slower. Candidate mechanisms include
   soil-moisture limitation and land–atmosphere coupling, regional aerosol
   histories, snow–albedo feedback at northern high latitudes and ocean
   moderation of southern mid-latitude land.
4. **Observational-product structure** (§6).
5. **Internal decadal variability aliased into a 64-year linear trend**: the
   Pacific-facing negatives (Mexico, Central America, Chile, Argentina, New
   Zealand, Korea) and Atlantic-facing positives (West Africa, Caribbean,
   Brazil, north-eastern Europe) resemble the imprint of Pacific and Atlantic
   decadal modes over 1950–2013. This is not testable cross-sectionally with
   static covariates.

The residual alone does not separate classes 3, 4 and 5: a regional cluster
can be a physical regime, a product artefact or a decadal-mode imprint, and the
same cluster (the Levant) is plausible under all three. The investigation
therefore tested measurement and functional form first, then one pre-specified
physical covariate, under a fixed spatial validation, and treated explicit
spatial terms as an accounting of what remains rather than as an explanation.
