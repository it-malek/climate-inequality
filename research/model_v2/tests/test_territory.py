"""Behavioral regressions for full-footprint territorial exclusion."""
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod

from research.model_v2 import cv


def bounds(a, b, width=0.25):
    from research.model_v2 import territory

    return territory.pair_bounds(np.asarray(a, float), np.asarray(b, float), width)


def test_shared_edge_and_corner_have_zero_distance():
    assert bounds([[0, 0]], [[0, 0.25]]) == (0.0, 0.0)
    assert bounds([[0, 0]], [[0.25, 0.25]]) == (0.0, 0.0)


def test_long_country_and_disconnected_island_are_all_used():
    a = [[0, 0], [0, 60], [0, 120]]
    b = [[0, 121]]
    lo, hi = bounds(a, b)
    assert 0 < lo < hi < 120
    # Using A's centroid would put this neighbour thousands of km away.
    assert bounds([[0, 0], [0, 1]], [[0, 90], [0, 1.25]]) == (0, 0)


def test_antimeridian_shared_cells_and_neighbours():
    assert bounds([[0, 179.875]], [[0, -179.875]]) == (0, 0)
    assert bounds([[0, 179]], [[0, -179]])[1] < 225


def test_inside_outside_threshold_with_small_footprints():
    geod = Geod(ellps='WGS84')
    lon_in, lat_in, _ = geod.fwd(0, 0, 90, 499_000)
    lon_out, lat_out, _ = geod.fwd(0, 0, 90, 502_000)
    assert bounds([[0, 0]], [[lat_in, lon_in]], 0.001)[1] < 500
    assert bounds([[0, 0]], [[lat_out, lon_out]], 0.001)[0] > 500


def test_cell_edges_can_be_inside_when_centres_are_outside():
    # Equatorial centres 4.75 degrees apart are 528.77 km apart, but
    # opposing edges of 0.5 degree footprints are only 473.11 km apart.
    lo, hi = bounds([[0, 0]], [[0, 4.75]], 0.5)
    assert lo < 474 < 500 < hi


def test_nicaragua_colombia_witness_cannot_return_to_training():
    # Actual GPW witness cell centres from the committed preflight evidence.
    lo, hi = bounds([[13.875, -81.875]], [[13.375, -81.375]])
    assert lo <= hi < 80


def test_conservative_bound_covers_interior_and_edge_points():
    geod = Geod(ellps='WGS84')
    a, b = [[60, 170]], [[61, -175]]
    lo, hi = bounds(a, b)
    for da in [-0.125, 0, 0.125]:
        for db in [-0.125, 0, 0.125]:
            _, _, meters = geod.inv(170 + da, 60 + da, -175 + db, 61 + db)
            assert lo <= meters / 1000
    assert lo <= hi


def test_fold_construction_deterministic_and_excludes_neighbours():
    lo, _ = bounds([[0, 0]], [[0, 4.75]], 0.5)
    distance = np.array([[0, lo, 2000], [lo, 0, 2000], [2000, 2000, 0]])
    first = list(cv.iter_folds(np.arange(3), distance, 500))
    second = list(cv.iter_folds(np.arange(3), distance, 500))
    assert first[0][2].tolist() == [False, False, True]
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(first, second))


def test_missing_geometry_fails_closed():
    with pytest.raises(ValueError):
        bounds([], [[0, 0]])


def test_cached_matrix_requires_matching_both_axes(tmp_path):
    from research.model_v2 import territory

    p = tmp_path / 'bounds.csv'
    pd.DataFrame([[0, 600], [600, 0]], index=['A', 'B'], columns=['B', 'A']).to_csv(p)
    with pytest.raises(ValueError):
        territory.load_bounds(['A', 'B'], p)


def test_matrix_geometry_deterministic_and_order_independent():
    from research.model_v2 import territory

    units = {'A': np.array([[0., 0.], [60., 80.]]),
             'B': np.array([[0., 4.75]]), 'C': np.array([[0., 30.]])}
    a, upper = territory.distance_bounds(units)
    b, upper2 = territory.distance_bounds(units)
    reversed_lower, _ = territory.distance_bounds(dict(reversed(list(units.items()))))
    assert np.array_equal(a, b) and np.array_equal(upper, upper2)
    assert np.array_equal(a, reversed_lower.loc[a.index, a.columns])


def test_all_recorded_witnesses_excluded_and_all_retained_certified():
    from research.model_v2 import territory
    from research.model_v2.m0 import OUTPUT_DIR

    frame = pd.read_csv(territory.LOWER_PATH, index_col=0)
    witnesses = pd.read_csv(OUTPUT_DIR / 'm0_5_buffer_witnesses.csv')
    assert all(frame.loc[r.a, r.b] <= 500 for r in witnesses.itertuples())
    membership = pd.read_csv(OUTPUT_DIR / 'territory_cv_membership_comparison.csv')
    for r in membership.itertuples():
        assert bool(r.corrected_train) == (frame.loc[r.held_out, r.candidate] > 500)


def test_saved_geometry_preserves_every_source_assigned_cell_and_iso_unit():
    from research.model_v2 import territory
    from research.model_v2.geometry import load_national_grid
    from src.population import GPW_PATH

    if not GPW_PATH.exists():
        pytest.skip('requires original local GPW data')
    iso, lats, lons = load_national_grid()
    with np.load(territory.SNAPSHOT_PATH) as saved:
        for code in saved.files:
            rows, cols = np.where(iso == code)
            expected = set(zip(lats[rows], lons[cols]))
            assert set(map(tuple, saved[code])) == expected
        assert np.all(saved['FRA'][:, 0] > 40)  # no separately coded GUF
        # Real island witness must survive materialization, not just synthetic tests.
        assert (13.375, -81.375) in set(map(tuple, saved['COL']))
        assert territory.pair_bounds(saved['NIC'], saved['COL'])[1] < 80
