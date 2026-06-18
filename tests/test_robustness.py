"""Tests for src.robustness -- the Phase 9 A/B/C robustness checks.

Synthetic, seeded, no network -- matching tests/test_explain.py conventions.
Each synthetic frame plants a known structure (nonlinear latitude term,
spatial cluster, leverage point) so the check's behavior is verifiable rather
than merely "it runs".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.robustness import (
    EMISSIONS_TERM,
    HIGH_LATITUDE_LEVERAGE,
    MIN_CLUSTERS_FOR_VALID_BOOTSTRAP,
    compare_latitude_controls,
    conley_hac_se,
    continent_cluster_bootstrap_se,
    country_centroids,
    country_residual_morans_i,
    fit_country_gam,
    fit_country_robust,
    gam_latitude_df_sensitivity,
    influence_diagnostics,
    jackknife_emissions_coef,
    robustness_summary_to_dict,
    run_robustness_suite,
    term_from_fit,
    weighted_inequality_fit,
)
from tests.test_explain import make_synthetic_city_features


def make_nonlinear_latitude_table(
    seed=0, n_per_continent=40, beta_emissions=0.01, curv=0.0006, noise_sd=0.01
):
    """Country table where latitude enters as a U-shape, not a line.

    ``trend = 0.1 + beta_emissions*log10_em + curv*(lat-35)^2 + noise``. A
    linear ``mean_abs_lat`` control cannot absorb the quadratic, so its
    residual leaks into ``log10_emissions`` whenever the two are correlated;
    the GAM smoother can. Emissions and latitude are deliberately correlated
    so an under-fit latitude control biases the emissions coefficient.
    """
    rng = np.random.default_rng(seed)
    continents = ["Africa", "Europe", "Asia"]
    income_groups = ["Low income", "High income"]
    rows = []
    for ci, continent in enumerate(continents):
        for j in range(n_per_continent):
            mean_abs_lat = rng.uniform(0.0, 70.0)
            # emissions correlated with latitude -> confounding pathway.
            log10_em = -1.0 + 0.05 * mean_abs_lat + rng.normal(0.0, 0.5)
            trend = (
                0.1
                + beta_emissions * log10_em
                + curv * (mean_abs_lat - 35.0) ** 2
                + rng.normal(0.0, noise_sd)
            )
            rows.append(
                {
                    "Country": f"C{ci}_{j}",
                    "continent": continent,
                    "income_group": income_groups[j % 2],
                    "n_cities": int(rng.integers(1, 12)),
                    "log10_emissions": log10_em,
                    "mean_abs_lat": mean_abs_lat,
                    "trend_c_per_decade": trend,
                }
            )
    return pd.DataFrame(rows)


class TestTermFromFit:
    def test_matches_ols_extract(self):
        import statsmodels.formula.api as smf

        table = make_nonlinear_latitude_table()
        fit = smf.ols(
            "trend_c_per_decade ~ log10_emissions + mean_abs_lat", data=table
        ).fit()
        term = term_from_fit(fit)
        assert term.term == EMISSIONS_TERM
        assert term.coef == pytest.approx(float(fit.params[EMISSIONS_TERM]))
        assert term.ci_low < term.coef < term.ci_high


class TestLatitudeControls:
    """Upgrade A: GAM smoother vs linear latitude control."""

    def test_gam_recovers_emissions_under_nonlinear_confound(self):
        # True effect is beta_emissions; the quadratic latitude term plus the
        # emissions-latitude correlation biases the *linear* control. The GAM
        # estimate should land closer to the planted value than A0.
        table = make_nonlinear_latitude_table(beta_emissions=0.01)
        a0, a1, a2 = compare_latitude_controls(table)

        assert a0.kind == "ols" and a1.kind == "gam" and a2.kind == "gam"
        truth = 0.01
        err_linear = abs(a0.emissions.coef - truth)
        err_gam = abs(a1.emissions.coef - truth)
        # The smoother absorbs more of the nonlinear latitude term, moving the
        # emissions estimate toward truth.
        assert err_gam < err_linear

    def test_specs_share_sample_and_are_labeled(self):
        table = make_nonlinear_latitude_table()
        controls = compare_latitude_controls(table)
        names = [c.spec_name for c in controls]
        assert names == ["A0_linear_lat", "A1_gam_lat", "A2_gam_lat_continent"]
        assert {c.n for c in controls} == {len(table)}

    def test_fit_country_gam_df_propagates(self):
        table = make_nonlinear_latitude_table()
        fit = fit_country_gam(
            table, "probe", "trend_c_per_decade ~ log10_emissions", df=5
        )
        assert fit.smooth_df == 5
        assert "s(mean_abs_lat, df=5" in fit.formula

    def test_controls_share_sample_when_continent_missing(self):
        # A2 carries C(continent); a missing continent must not silently shrink
        # A2's sample below A0/A1. All three should sit on the same n.
        table = make_nonlinear_latitude_table()
        table.loc[table.index[:3], "continent"] = np.nan
        controls = compare_latitude_controls(table)
        assert {c.n for c in controls} == {len(table) - 3}

    def test_gam_df_sensitivity_sweeps_basis_sizes(self):
        table = make_nonlinear_latitude_table()
        fits = gam_latitude_df_sensitivity(table, dfs=(4, 6, 8))
        # A1 + A2 at each of the three df values.
        assert len(fits) == 6
        assert {f.smooth_df for f in fits} == {4, 6, 8}
        names = {f.spec_name for f in fits}
        assert "A1_gam_lat_df4" in names
        assert "A2_gam_lat_continent_df8" in names
        # All fit the same complete-case sample.
        assert {f.n for f in fits} == {len(table)}


class TestSpatialInference:
    """Upgrade B: residual Moran's I and the continent-cluster bootstrap."""

    def make_clustered_table(self, seed=0, n_clusters=6, per_cluster=15):
        """Country table + centroids with planted spatial residual clusters.

        Countries fall into ``n_clusters`` spatial blobs. Each blob has its own
        mean emissions *and* an unexplained trend offset, with little
        within-blob variation -- the textbook case where the effective sample
        size is the number of clusters, not the number of points. Pooled
        residuals then carry the offsets (so they cluster in space, lifting
        Moran's I), and the emissions slope is driven by a handful of
        between-cluster contrasts (so a continent-cluster bootstrap is wider
        than the iid-style HC1 CI).
        """
        rng = np.random.default_rng(seed)
        rows, cents = [], []
        cid = 0
        for g in range(n_clusters):
            cen_lat = rng.uniform(-70.0, 70.0)
            cen_lon = rng.uniform(-170.0, 170.0)
            em_mean = rng.uniform(-1.0, 4.0)
            offset = rng.normal(0.0, 0.12)  # cluster-level unexplained trend
            for _ in range(per_cluster):
                lat = cen_lat + rng.normal(0.0, 2.0)
                lon = cen_lon + rng.normal(0.0, 2.0)
                log10_em = em_mean + rng.normal(0.0, 0.1)
                trend = 0.1 + 0.01 * log10_em + offset + rng.normal(0.0, 0.01)
                name = f"C{cid}"
                rows.append(
                    {
                        "Country": name,
                        "continent": f"cl{g}",
                        "log10_emissions": log10_em,
                        "mean_abs_lat": abs(lat),
                        "trend_c_per_decade": trend,
                    }
                )
                cents.append({"Country": name, "cen_lon": lon, "cen_lat": lat})
                cid += 1
        return pd.DataFrame(rows), pd.DataFrame(cents)

    def test_country_centroids_from_features(self):
        features = make_synthetic_city_features(n_countries=4, n_per_country=5)
        cents = country_centroids(features)
        assert set(cents.columns) == {"Country", "cen_lon", "cen_lat"}
        assert len(cents) == 4
        # Centroid is the unit-vector (spherical) mean of the city coordinates.
        c0 = features[features["Country"] == "Country0"]
        lon = np.radians(c0["Longitude"].to_numpy(dtype=float))
        lat = np.radians(c0["Latitude"].to_numpy(dtype=float))
        x = (np.cos(lat) * np.cos(lon)).mean()
        y = (np.cos(lat) * np.sin(lon)).mean()
        z = np.sin(lat).mean()
        exp_lon = np.degrees(np.arctan2(y, x))
        exp_lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
        got = cents[cents["Country"] == "Country0"].iloc[0]
        assert got["cen_lat"] == pytest.approx(exp_lat)
        assert got["cen_lon"] == pytest.approx(exp_lon)

    def test_country_centroids_antimeridian_safe(self):
        # Cities straddling +/-179 deg longitude: the arithmetic mean collapses
        # to ~0 (wrong hemisphere); the spherical mean stays near +/-180.
        features = pd.DataFrame(
            {
                "Country": ["Fiji"] * 4,
                "Longitude": [179.0, 178.5, -179.0, -178.5],
                "Latitude": [-17.0, -18.0, -17.5, -18.5],
            }
        )
        cents = country_centroids(features)
        cen_lon = cents.iloc[0]["cen_lon"]
        # Near the antimeridian, not collapsed toward the prime meridian.
        assert abs(cen_lon) > 170.0
        assert cents.iloc[0]["cen_lat"] == pytest.approx(-17.75, abs=0.3)

    def test_morans_i_detects_clustering(self):
        table, centroids = self.make_clustered_table()
        # The "pooled" spec leaves the blob offset in the residuals.
        res = country_residual_morans_i(table, centroids, spec_name="pooled", k=5)
        assert res.moran_i > 0.3
        assert res.moran_p < 0.05

    def test_morans_i_near_zero_for_shuffled_residuals(self):
        # Random trend (no spatial structure) -> I near zero, not significant.
        rng = np.random.default_rng(1)
        n = 60
        table = pd.DataFrame(
            {
                "Country": [f"C{i}" for i in range(n)],
                "continent": rng.choice(["Africa", "Asia"], n),
                "log10_emissions": rng.uniform(-1, 4, n),
                "mean_abs_lat": rng.uniform(0, 70, n),
                "trend_c_per_decade": rng.normal(0, 0.1, n),
            }
        )
        centroids = pd.DataFrame(
            {
                "Country": table["Country"],
                "cen_lon": rng.uniform(-180, 180, n),
                "cen_lat": rng.uniform(-80, 80, n),
            }
        )
        res = country_residual_morans_i(table, centroids, spec_name="pooled", k=5)
        # No spatial structure: I sits near its null expectation, far below the
        # clustered case (>0.3). The permutation p is uniform under the null, so
        # it is not asserted on -- the magnitude of I is the meaningful signal.
        assert abs(res.moran_i) < 0.2

    def test_cluster_bootstrap_wider_than_hc1(self):
        table, _ = self.make_clustered_table()
        res = continent_cluster_bootstrap_se(
            table, spec_name="pooled", n_boot=300, seed=0
        )
        hc1_width = res.hc1.ci_high - res.hc1.ci_low
        boot_width = res.boot_ci_high - res.boot_ci_low
        assert res.n_effective > 0
        # Clustering inflates the honest CI relative to HC1.
        assert boot_width > hc1_width

    def test_cluster_bootstrap_is_deterministic(self):
        table, _ = self.make_clustered_table()
        a = continent_cluster_bootstrap_se(table, "pooled", n_boot=100, seed=7)
        b = continent_cluster_bootstrap_se(table, "pooled", n_boot=100, seed=7)
        assert a.boot_ci_low == b.boot_ci_low
        assert a.boot_ci_high == b.boot_ci_high

    def test_cluster_bootstrap_flags_insufficient_clusters(self):
        # 6 clusters is far below the validity threshold: the result must mark
        # itself as not a valid CI so the render layer cannot present it as one.
        table, _ = self.make_clustered_table(n_clusters=6)
        res = continent_cluster_bootstrap_se(table, "pooled", n_boot=100)
        assert res.n_clusters == 6
        assert res.n_clusters < MIN_CLUSTERS_FOR_VALID_BOOTSTRAP
        assert res.clusters_sufficient is False

    def test_conley_hac_inflates_se_under_spatial_dependence(self):
        # Spatially clustered residuals -> Conley HAC SE should exceed the
        # naive HC1 SE, since near pairs carry positive residual covariance.
        table, centroids = self.make_clustered_table()
        res = conley_hac_se(table, centroids, spec_name="pooled", cutoff_km=1000.0)
        assert res.emissions.term == EMISSIONS_TERM
        assert np.isfinite(res.emissions.se)
        assert res.emissions.ci_low < res.emissions.coef < res.emissions.ci_high
        assert res.emissions.se > res.hc1.se


class TestEstimatorPlurality:
    """Upgrade C: RLM, jackknife, weighting."""

    def make_leverage_table(self, seed=0):
        """Clean linear table plus one extreme high-latitude leverage country."""
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(40):
            log10_em = rng.uniform(-1.0, 4.0)
            mean_abs_lat = rng.uniform(0.0, 60.0)
            trend = 0.1 + 0.01 * log10_em + 0.001 * mean_abs_lat + rng.normal(0, 0.01)
            rows.append(
                {
                    "Country": f"C{i}",
                    "continent": rng.choice(["Africa", "Europe", "Asia"]),
                    "n_cities": int(rng.integers(1, 12)),
                    "log10_emissions": log10_em,
                    "mean_abs_lat": mean_abs_lat,
                    "trend_c_per_decade": trend,
                }
            )
        # One outlier: high emissions, high latitude, wildly off-trend warming.
        rows.append(
            {
                "Country": "Russia",
                "continent": "Europe",
                "n_cities": 9,
                "log10_emissions": 4.0,
                "mean_abs_lat": 60.0,
                "trend_c_per_decade": 1.5,
            }
        )
        return pd.DataFrame(rows)

    def test_rlm_downweights_leverage_point(self):
        table = self.make_leverage_table()
        cmp = fit_country_robust(table, spec_name="lat_continent")
        # The outlier inflates the OLS slope; Huber RLM pulls it back.
        assert abs(cmp.rlm.coef - 0.01) < abs(cmp.ols.coef - 0.01)

    def test_jackknife_flags_outlier_and_widens_range(self):
        table = self.make_leverage_table()
        jk = jackknife_emissions_coef(table, spec_name="lat_continent")
        assert jk.most_influential_country == "Russia"
        # Dropping the planted outlier moves the coefficient materially.
        assert jk.loo_range > 0.0
        assert jk.drop_set == HIGH_LATITUDE_LEVERAGE
        assert "Russia" in jk.drop_set_present
        assert jk.drop_set_coef is not None
        assert abs(jk.drop_set_coef - 0.01) < abs(jk.full_coef - 0.01)

    def test_influence_flags_leverage_point_by_dfbeta(self):
        # The planted "Russia" outlier should surface as a top-DFBETA country,
        # and dropping the top-k by |DFBETA| should pull the coefficient toward
        # the true 0.01 -- the design-matrix leverage tool RLM is not.
        table = self.make_leverage_table()
        inf = influence_diagnostics(table, spec_name="lat_continent", k=3)
        top_countries = [c for c, _ in inf.top_dfbeta]
        assert "Russia" in top_countries
        assert "Russia" in inf.drop_top_dfbeta
        assert inf.drop_top_dfbeta_coef is not None
        assert abs(inf.drop_top_dfbeta_coef - 0.01) < abs(inf.full_coef - 0.01)
        assert len(inf.top_cooks) == 3

    def test_jackknife_empty_drop_set_when_absent(self):
        rng = np.random.default_rng(0)
        n = 30
        table = pd.DataFrame(
            {
                "Country": [f"X{i}" for i in range(n)],
                "continent": rng.choice(["Africa", "Asia"], n),
                "log10_emissions": rng.uniform(-1, 4, n),
                "mean_abs_lat": rng.uniform(0, 70, n),
                "trend_c_per_decade": rng.normal(0.1, 0.05, n),
            }
        )
        jk = jackknife_emissions_coef(table, spec_name="lat_continent")
        assert jk.drop_set_present == ()
        assert jk.drop_set_coef is None

    def test_weighted_fit_runs_for_n_cities(self):
        table = self.make_leverage_table()
        res = weighted_inequality_fit(table, weight_col="n_cities", spec_name="lat_continent")
        assert res.weight_col == "n_cities"
        assert res.n == len(table)
        assert np.isfinite(res.emissions.coef)
        assert np.isfinite(res.emissions.se)

    def test_weighted_fit_custom_weight_column(self):
        # Any precomputed weight column works (e.g. an area weight).
        table = self.make_leverage_table()
        table = table.assign(area_weight=np.linspace(0.5, 2.0, len(table)))
        res = weighted_inequality_fit(table, weight_col="area_weight", spec_name="lat")
        assert res.weight_col == "area_weight"
        assert np.isfinite(res.emissions.coef)


def make_matching_features(table, seed=0, n_per_country=3):
    """City features whose ``Country`` names cover ``table`` (for centroids)."""
    rng = np.random.default_rng(seed)
    rows = []
    for country in table["Country"]:
        for _ in range(n_per_country):
            lat = rng.uniform(-80.0, 80.0)
            rows.append(
                {
                    "Country": country,
                    "Latitude": lat,
                    "Longitude": rng.uniform(-180.0, 180.0),
                    "abs_latitude": abs(lat),
                }
            )
    return pd.DataFrame(rows)


class TestRunSuite:
    def test_end_to_end_and_serializable(self):
        table = make_nonlinear_latitude_table()
        features = make_matching_features(table)
        summary = run_robustness_suite(
            table, features, spec_name="lat_continent", n_boot=120, seed=0
        )
        assert len(summary.latitude_controls) == 3
        assert len(summary.gam_df_sensitivity) == 6  # A1+A2 at df 4/6/8
        assert len(summary.residual_moran) == 2
        assert summary.conley.emissions.term == EMISSIONS_TERM
        assert summary.estimator.spec_name == "lat_continent"
        assert summary.influence.spec_name == "lat_continent"
        assert len(summary.weighted) == 1  # n_cities present

        blob = robustness_summary_to_dict(summary)
        # JSON-serializable end to end.
        import json

        json.loads(json.dumps(blob))
        assert blob["cluster_bootstrap"]["hc1"]["term"] == EMISSIONS_TERM
        assert blob["cluster_bootstrap"]["clusters_sufficient"] in (True, False)
        assert blob["conley"]["emissions"]["term"] == EMISSIONS_TERM
        assert blob["latitude_controls"][0]["spec_name"] == "A0_linear_lat"
        assert len(blob["gam_df_sensitivity"]) == 6
        assert "top_dfbeta" in blob["influence"]

    def test_suite_skips_absent_weight_columns(self):
        table = make_nonlinear_latitude_table().drop(columns=["n_cities"])
        features = make_matching_features(table)
        summary = run_robustness_suite(table, features, n_boot=60, weight_cols=("n_cities",))
        assert summary.weighted == []
