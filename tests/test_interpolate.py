"""Tests for src.interpolate — IDW, ordinary kriging, LOO-CV, land masking.

No real dataset and no network: synthetic point sets with known fields, and
a small synthetic land polygon for the masking tests.
"""

import numpy as np
import pytest
from shapely.geometry import box

from src.interpolate import (
    build_grid,
    haversine_km,
    idw_interpolate,
    leave_one_out_cv,
    mask_to_land,
    ordinary_kriging_interpolate,
    render_trend_surface,
)


def grid_points(n_per_side=3, lon_span=(0.0, 20.0), lat_span=(0.0, 20.0)):
    """An n x n grid of (lon, lat) points and a deterministic linear field."""
    lons = np.linspace(*lon_span, n_per_side)
    lats = np.linspace(*lat_span, n_per_side)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon = lon_grid.ravel()
    lat = lat_grid.ravel()
    value = 0.1 * lon + 0.2 * lat
    return lon, lat, value


class TestHaversine:
    def test_zero_distance_for_identical_points(self):
        assert haversine_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-9)

    def test_matches_known_distance(self):
        # London to Paris, ~344 km great-circle.
        d = haversine_km(-0.1276, 51.5074, 2.3522, 48.8566)
        assert d == pytest.approx(344.0, rel=0.02)

    def test_broadcasts_to_pairwise_matrix(self):
        lon = np.array([0.0, 10.0])
        lat = np.array([0.0, 10.0])
        d = haversine_km(lon[:, None], lat[:, None], lon[None, :], lat[None, :])
        assert d.shape == (2, 2)
        assert np.diag(d) == pytest.approx([0.0, 0.0], abs=1e-9)
        assert d[0, 1] == d[1, 0]


class TestIDW:
    def test_exact_recovery_at_known_points(self):
        lon, lat, value = grid_points()
        out = idw_interpolate(lon, lat, value, lon, lat)
        assert out == pytest.approx(value, abs=1e-9)

    def test_exact_recovery_with_knn(self):
        lon, lat, value = grid_points()
        out = idw_interpolate(lon, lat, value, lon, lat, k=4)
        assert out == pytest.approx(value, abs=1e-9)

    def test_interpolated_value_between_neighbors(self):
        lon = np.array([0.0, 10.0])
        lat = np.array([0.0, 0.0])
        value = np.array([0.0, 10.0])
        out = idw_interpolate(lon, lat, value, 5.0, 0.0)
        # Symmetric midpoint -> average of the two known values.
        assert out[0] == pytest.approx(5.0, abs=1e-6)

    def test_closer_point_dominates(self):
        lon = np.array([0.0, 10.0])
        lat = np.array([0.0, 0.0])
        value = np.array([0.0, 10.0])
        out = idw_interpolate(lon, lat, value, 1.0, 0.0)
        assert out[0] < 5.0

    def test_scalar_query_returns_length_one_array(self):
        lon, lat, value = grid_points()
        out = idw_interpolate(lon, lat, value, 5.0, 5.0)
        assert out.shape == (1,)


class TestKriging:
    def test_exact_recovery_global(self):
        lon, lat, value = grid_points()
        z, var = ordinary_kriging_interpolate(lon, lat, value, lon, lat)
        assert z == pytest.approx(value, abs=1e-6)
        assert var == pytest.approx(np.zeros_like(var), abs=1e-6)

    def test_exact_recovery_local_knn(self):
        lon, lat, value = grid_points()
        z, _ = ordinary_kriging_interpolate(lon, lat, value, lon, lat, k=5)
        assert z == pytest.approx(value, abs=1e-6)

    def test_exact_recovery_with_fixed_variogram_parameters(self):
        from src.interpolate import fit_variogram_parameters

        lon, lat, value = grid_points()
        params = fit_variogram_parameters(lon, lat, value)
        z, _ = ordinary_kriging_interpolate(
            lon, lat, value, lon, lat, variogram_parameters=params, k=5
        )
        assert z == pytest.approx(value, abs=1e-6)

    def test_interpolated_value_between_neighbors(self):
        lon, lat, value = grid_points()
        # Midpoint of the field's domain.
        z, _ = ordinary_kriging_interpolate(lon, lat, value, [10.0], [10.0])
        assert z[0] == pytest.approx(3.0, abs=0.5)


class TestLeaveOneOutCV:
    def test_returns_both_methods_with_nonnegative_rmse(self):
        lon, lat, value = grid_points(n_per_side=4)
        out = leave_one_out_cv(lon, lat, value, k=4)
        assert set(out["method"]) == {"idw", "kriging"}
        assert (out["rmse"] >= 0).all()
        assert (out["mae"] >= 0).all()
        assert (out["n"] == len(lon)).all()

    def test_low_rmse_on_smooth_linear_field(self):
        lon, lat, value = grid_points(n_per_side=5)
        out = leave_one_out_cv(lon, lat, value, k=4)
        # A smooth linear field should be easy to predict from neighbors.
        span = value.max() - value.min()
        assert (out["rmse"] < 0.25 * span).all()

    @staticmethod
    def duplicate_points():
        """Two coincident points with an outlier value amid a ring of zeros."""
        lon = np.array([10.0, 10.0, 0.0, 20.0, 0.0, 20.0, 10.0, 0.0])
        lat = np.array([10.0, 10.0, 0.0, 0.0, 20.0, 20.0, 0.0, 10.0])
        value = np.array([99.0, 99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return lon, lat, value

    def test_same_coordinate_twin_excluded_from_fold(self):
        lon, lat, value = self.duplicate_points()
        fair = leave_one_out_cv(lon, lat, value, k=3).set_index("method")
        naive = leave_one_out_cv(
            lon, lat, value, k=3, exclude_same_coordinate=False
        ).set_index("method")
        # Per row, the duplicate's twin leaks the held-out value, so both
        # methods look much better than they should; IDW becomes exact.
        assert (fair["rmse"] >= naive["rmse"] - 1e-12).all()
        # Folds of unique-coordinate points are identical in both modes, so
        # the gap is exactly the two duplicate folds: error 0 per row when
        # the twin leaks (IDW exact match), error 99 when excluded.
        n = len(lon)
        assert fair.loc["idw", "rmse"] ** 2 - naive.loc["idw", "rmse"] ** 2 == (
            pytest.approx(2 * 99.0**2 / n, rel=1e-9)
        )
        assert fair["rmse"].min() > 1.0

    def test_fair_equals_naive_without_duplicates(self):
        lon, lat, value = grid_points(n_per_side=4)
        fair = leave_one_out_cv(lon, lat, value, k=4)
        naive = leave_one_out_cv(lon, lat, value, k=4, exclude_same_coordinate=False)
        assert fair["rmse"].to_numpy() == pytest.approx(naive["rmse"].to_numpy())
        assert fair["mae"].to_numpy() == pytest.approx(naive["mae"].to_numpy())

    def test_k_clamps_to_points_outside_coordinate_group(self):
        lon, lat, value = self.duplicate_points()
        out = leave_one_out_cv(lon, lat, value, k=100)
        assert np.isfinite(out["rmse"]).all()
        assert (out["n"] == len(lon)).all()

    def test_all_points_coincident_raises(self):
        lon = np.zeros(3)
        lat = np.zeros(3)
        value = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="no neighbors"):
            leave_one_out_cv(lon, lat, value, k=2)


class TestBuildGrid:
    def test_axes_within_bounds_and_spaced(self):
        lons, lats = build_grid(
            lon_min=-10.0, lon_max=10.0, lat_min=-5.0, lat_max=5.0, resolution=2.0
        )
        assert lons.min() >= -10.0
        assert lons.max() < 10.0
        assert lats.min() >= -5.0
        assert lats.max() < 5.0
        assert np.diff(lons) == pytest.approx(2.0)
        assert np.diff(lats) == pytest.approx(2.0)


class TestMaskToLand:
    def test_masks_outside_polygon_to_nan(self):
        land = box(0.0, 0.0, 10.0, 10.0)  # a 10x10 square "continent"
        grid_lon, grid_lat = np.meshgrid([5.0, 15.0], [5.0, 15.0])
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = mask_to_land(grid_lon, grid_lat, values, land)
        assert out[0, 0] == pytest.approx(1.0)  # (5, 5) inside
        assert np.isnan(out[0, 1])  # (15, 5) outside
        assert np.isnan(out[1, 0])  # (5, 15) outside
        assert np.isnan(out[1, 1])  # (15, 15) outside

    def test_does_not_mutate_input(self):
        land = box(0.0, 0.0, 10.0, 10.0)
        grid_lon, grid_lat = np.meshgrid([5.0, 15.0], [5.0, 15.0])
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask_to_land(grid_lon, grid_lat, values, land)
        assert not np.isnan(values).any()


class TestRenderTrendSurface:
    def test_returns_figure_with_diverging_colorscale(self):
        lons = np.array([0.0, 1.0])
        lats = np.array([0.0, 1.0])
        values = np.array([[-1.0, 2.0], [np.nan, 0.5]])
        fig = render_trend_surface(lons, lats, values)
        heatmap = fig.data[0]
        assert heatmap.colorscale is not None
        assert heatmap.zmid == 0.0
        assert heatmap.zmax == pytest.approx(2.0)
        assert heatmap.zmin == pytest.approx(-2.0)
