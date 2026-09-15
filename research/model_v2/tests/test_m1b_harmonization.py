"""M1b Amendment 1: one-ring structural-mask harmonization of C2, on synthetic CRU grids."""
import io
import json
import os
import subprocess
import sys

from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest

from research.model_v2 import m1b_hydroclimate as h
from research.model_v2.territory import GEOD

N_LAT, N_LON = 360, 720
AREA_RTOL = 1e-10
TIE_KM = 1e-6  # 1 mm; distinct neighbour candidates differ by >= ~4e-5 km on the 0.5° grid


def empty_grids():
    return np.zeros((N_LAT, N_LON), bool), np.full((N_LAT, N_LON), np.nan), np.zeros((N_LAT, N_LON), bool)


def native_cell(native, x, i, k, value):
    native[i, k], x[i, k] = True, value


def donor_row(donors, i, k):
    row = donors[(donors.target_lat_index == i) & (donors.target_lon_index == k)]
    assert len(row) == 1
    return row.iloc[0]


def donor_of(donors, i, k):
    row = donor_row(donors, i, k)
    return int(row.donor_lat_index), int(row.donor_lon_index)


def cells_frame(rows):
    """(country index, CRU row, CRU col, km²) rows in the builder's sorted long format."""
    frame = pd.DataFrame(rows, columns=['idx', 'i', 'k', 'area'])
    return frame.sort_values(['idx', 'i', 'k'], kind='stable').reset_index(drop=True)


def run(cells, native, x, n, supported=None, mean_stn=None, pure=None):
    """Pure Amendment 1 pipeline on synthetic grids: every invalid support cell is a structural target."""
    structural = np.zeros_like(native)
    structural[cells.i, cells.k] = True
    structural &= ~native
    donors = h.one_ring_donors(structural, native, x)
    resolved = h.resolve_support(cells, x, native, structural, donors)
    ones = np.ones((N_LAT, N_LON))
    qa = h.aggregate(resolved, ones, ones, ones if supported is None else supported,
                     ones if mean_stn is None else mean_stn,
                     np.zeros((N_LAT, N_LON), bool) if pure is None else pure, n)
    return donors, resolved, qa


def masked_states():
    return np.full((N_LAT, N_LON), h.MASKED, dtype=object)


def geodesic_km(i1, k1, i2, k2):
    return GEOD.inv(-179.75 + 0.5 * k1, -89.75 + 0.5 * i1, -179.75 + 0.5 * k2, -89.75 + 0.5 * i2)[2] / 1000


def reference_donor(i, k, native):
    """Independent brute force: pyproj per candidate pair, 1 mm tie tolerance, then (lat index, lon index)."""
    candidates = []
    for di in (-1, 0, 1):
        for dk in (-1, 0, 1):
            ii, kk = i + di, (k + dk) % N_LON
            if (di, dk) != (0, 0) and 0 <= ii < N_LAT and native[ii, kk]:
                candidates.append((geodesic_km(i, k, ii, kk), ii, kk))
    if not candidates:
        return None
    nearest = min(c[0] for c in candidates)
    return min((c for c in candidates if c[0] <= nearest + TIE_KM), key=lambda c: (c[1], c[2]))


# --- distances and the donor search against an independent reference --------------------------

@pytest.mark.parametrize('i', [0, 1, 90, 179, 180, 181, 270, 358, 359])
def test_every_neighbour_distance_matches_the_wgs84_geodesic(i):
    for di in (-1, 0, 1):
        for dk in (-1, 0, 1):
            if (di, dk) == (0, 0):
                continue
            if not 0 <= i + di < N_LAT:
                assert h.neighbour_distance_km(i, di, dk) == np.inf
                continue
            for k in (0, 350, 719):
                expected = geodesic_km(i, k, i + di, (k + dk) % N_LON)
                assert h.neighbour_distance_km(i, di, dk) == pytest.approx(expected, rel=1e-9)


def test_donor_search_matches_an_independent_brute_force_everywhere():
    rng = np.random.default_rng(20260915)
    native = rng.random((N_LAT, N_LON)) < 0.35
    x = np.where(native, rng.normal(size=(N_LAT, N_LON)), np.nan)
    targets = np.zeros((N_LAT, N_LON), bool)
    rows = np.r_[0, 1, 179, 180, 358, 359, rng.integers(0, N_LAT, 600)]
    cols = np.r_[0, 719, 0, 719, 719, 0, rng.integers(0, N_LON, 600)]
    targets[rows, cols] = True
    native[targets] = False
    x[targets] = np.nan
    donors = h.one_ring_donors(targets, native, x)
    assert len(donors) == targets.sum()
    for r in donors.itertuples():
        ref = reference_donor(r.target_lat_index, r.target_lon_index, native)
        if ref is None:
            assert r.donor_lat_index == -1 and np.isnan(r.donor_c2)
        else:
            assert (r.donor_lat_index, r.donor_lon_index) == (ref[1], ref[2])
            assert r.donor_distance_km == pytest.approx(ref[0], rel=1e-9)
            assert r.donor_c2 == x[ref[1], ref[2]]


# --- donor search ------------------------------------------------------------------------------

def test_native_cells_and_values_are_never_altered():
    native, x, targets = empty_grids()
    native_cell(native, x, 200, 101, 0.3)
    native_cell(native, x, 200, 99, -0.7)
    targets[200, 100] = True
    before = x.copy()
    donors = h.one_ring_donors(targets, native, x)
    np.testing.assert_array_equal(x, before)
    cells = cells_frame([(0, 200, 99, 2.0), (0, 200, 100, 1.0), (0, 200, 101, 3.0)])
    resolved = h.resolve_support(cells, x, native, targets, donors)
    assert resolved[resolved.status == h.NATIVE].c2_value.tolist() == [-0.7, 0.3]
    with pytest.raises(ValueError):
        h.one_ring_donors(native, native, x)          # a native cell can never be a target


def test_only_the_first_ring_is_searched():
    native, x, targets = empty_grids()
    targets[200, 100] = True
    for i, k in [(200, 102), (202, 100), (198, 98), (202, 102), (200, 98)]:
        native_cell(native, x, i, k, 1.0)
    row = donor_row(h.one_ring_donors(targets, native, x), 200, 100)
    assert row.donor_lat_index == -1 and row.donor_lon_index == -1
    assert np.isnan(row.donor_distance_km) and np.isnan(row.donor_c2)


@pytest.mark.parametrize('donor, near, far', [
    ((200, 102), (200, 101), (200, 100)),   # donor east of the chain
    ((200, 98), (200, 99), (200, 100)),     # donor west: a row-major in-place fill would propagate here
    ((198, 100), (199, 100), (200, 100)),   # donor south: a column-major in-place fill would propagate here
    ((202, 100), (201, 100), (200, 100)),   # donor north
])
def test_harmonized_targets_never_become_donors(donor, near, far):
    native, x, targets = empty_grids()
    native_cell(native, x, *donor, 0.5)
    targets[near] = targets[far] = True
    donors = h.one_ring_donors(targets, native, x)
    assert donor_of(donors, *near) == donor
    outer = donor_row(donors, *far)
    assert outer.donor_lat_index == -1 and np.isnan(outer.donor_c2)


def test_cross_border_donor_is_used_and_flagged():
    native, x, _ = empty_grids()
    native_cell(native, x, 200, 101, 0.25)
    cells = cells_frame([(0, 200, 100, 5.0), (1, 200, 101, 50.0)])  # country 0's only cell is masked
    _, resolved, qa = run(cells, native, x, 2)
    target = resolved[resolved.idx == 0].iloc[0]
    assert target.status == h.HARMONIZED and target.c2_value == 0.25
    audit = h.cell_audit(resolved, ['AAA', 'BBB'], masked_states(), masked_states())
    assert audit.loc[0, 'iso3'] == 'AAA' and audit.loc[0, 'donor_countries'] == 'BBB' and bool(audit.loc[0, 'cross_border'])
    assert qa.loc[0, 'c2_harmonized'] == 0.25 and qa.loc[0, 'coverage'] == 1.0
    assert qa.loc[0, 'n_cross_border_cells'] == 1 and qa.loc[0, 'cross_border_area_km2'] == 5.0


def test_same_country_donor_is_not_cross_border_and_unsupported_donor_is():
    native, x, _ = empty_grids()
    native_cell(native, x, 200, 101, -0.1)                  # native, no M1a support row at all
    native_cell(native, x, 150, 11, 0.2)                    # native, supports country 0 itself
    cells = cells_frame([(0, 200, 100, 5.0), (0, 150, 10, 2.0), (0, 150, 11, 9.0)])
    _, resolved, qa = run(cells, native, x, 1)
    audit = h.cell_audit(resolved, ['AAA'], masked_states(), masked_states())
    audit = audit.set_index(['target_lat_index', 'target_lon_index'])
    assert audit.loc[(200, 100), 'donor_countries'] == 'none' and bool(audit.loc[(200, 100), 'cross_border'])
    assert audit.loc[(150, 10), 'donor_countries'] == 'AAA' and not bool(audit.loc[(150, 10), 'cross_border'])
    assert qa.loc[0, 'n_cross_border_cells'] == 1


def test_nearest_wgs84_donor_is_selected():
    native, x, targets = empty_grids()
    targets[260, 100] = targets[340, 300] = targets[260, 500] = targets[100, 600] = True
    native_cell(native, x, 259, 100, 1.0)                   # 40.25°N: south (~55 km) ...
    native_cell(native, x, 260, 101, 2.0)                   # ... loses to east (~43 km)
    native_cell(native, x, 339, 300, 3.0)                   # 80.25°N: south (~56 km) ...
    native_cell(native, x, 340, 299, 4.0)                   # ... loses to west (~9 km)
    native_cell(native, x, 261, 501, 5.0)                   # north-east diagonal ...
    native_cell(native, x, 261, 500, 6.0)                   # ... loses to orthogonal north
    native_cell(native, x, 99, 599, 7.0)                    # 39.75°S: the two western diagonals differ;
    native_cell(native, x, 101, 599, 8.0)                   # the geodesically nearer one must win
    donors = h.one_ring_donors(targets, native, x)
    assert donor_row(donors, 260, 100).donor_c2 == 2.0
    assert donor_row(donors, 340, 300).donor_c2 == 4.0
    assert donor_row(donors, 260, 500).donor_c2 == 6.0
    south_west, north_west = geodesic_km(100, 600, 99, 599), geodesic_km(100, 600, 101, 599)
    assert abs(south_west - north_west) > 1e-3
    assert donor_row(donors, 100, 600).donor_c2 == (7.0 if south_west < north_west else 8.0)


def test_north_and_south_neighbours_are_not_tied_on_the_ellipsoid():
    native, x, targets = empty_grids()
    targets[260, 100] = targets[100, 400] = True             # 40.25°N and 39.75°S
    for i, k in [(261, 100), (259, 100), (101, 400), (99, 400)]:
        native_cell(native, x, i, k, float(i))
    donors = h.one_ring_donors(targets, native, x)
    # WGS84 meridian arcs lengthen poleward, so the equatorward neighbour is strictly nearer in both
    # hemispheres; an index tie-break (as on a sphere) would pick row 99 in the south instead of row 101.
    assert donor_row(donors, 260, 100).donor_lat_index == 259
    assert donor_row(donors, 100, 400).donor_lat_index == 101


def test_exact_east_west_ties_resolve_to_smaller_wrapped_longitude_index():
    native, x, targets = empty_grids()
    targets[200, 100] = targets[50, 600] = targets[210, 0] = targets[211, 719] = targets[230, 0] = True
    native_cell(native, x, 200, 101, 1.0)                   # east
    native_cell(native, x, 200, 99, 2.0)                    # west: smaller longitude index
    native_cell(native, x, 51, 601, 5.0)                    # north-east
    native_cell(native, x, 51, 599, 6.0)                    # north-west: smaller longitude index
    native_cell(native, x, 210, 719, 7.0)                   # seam: west of column 0 is index 719 ...
    native_cell(native, x, 210, 1, 8.0)                     # ... so east (index 1) wins
    native_cell(native, x, 211, 718, 9.0)                   # seam: west of column 719 is 718 ...
    native_cell(native, x, 211, 0, 10.0)                    # ... and east wraps to index 0, which wins
    native_cell(native, x, 231, 719, 11.0)                  # seam diagonal: north-west wraps to 719 ...
    native_cell(native, x, 231, 1, 12.0)                    # ... so north-east (index 1) wins
    assert h.neighbour_distance_km(200, 0, 1) == h.neighbour_distance_km(200, 0, -1)
    assert h.neighbour_distance_km(50, 1, 1) == h.neighbour_distance_km(50, 1, -1)
    first = h.one_ring_donors(targets, native, x)
    expected = {(200, 100): (200, 99), (50, 600): (51, 599), (210, 0): (210, 1), (211, 719): (211, 0),
                (230, 0): (231, 1)}
    for target, donor in expected.items():
        assert donor_of(first, *target) == donor
    pd.testing.assert_frame_equal(first, h.one_ring_donors(targets.copy(), native.copy(), x.copy()))


def test_south_diagonals_are_candidates():
    native, x, targets = empty_grids()
    targets[200, 100] = targets[150, 300] = True
    native_cell(native, x, 199, 101, 1.0)
    native_cell(native, x, 149, 299, 2.0)
    donors = h.one_ring_donors(targets, native, x)
    assert donor_of(donors, 200, 100) == (199, 101) and donor_of(donors, 150, 300) == (149, 299)


def test_longitude_wraps_across_the_antimeridian():
    native, x, targets = empty_grids()
    targets[200, 0] = targets[150, 719] = True
    native_cell(native, x, 200, 719, 0.7)                   # 179.75°E next to 179.75°W
    native_cell(native, x, 150, 0, -0.3)
    donors = h.one_ring_donors(targets, native, x)
    west = donor_row(donors, 200, 0)
    assert (west.donor_lat_index, west.donor_lon_index, west.donor_c2) == (200, 719, 0.7)
    assert west.donor_distance_km == h.neighbour_distance_km(200, 0, 1)
    assert west.donor_distance_km == pytest.approx(GEOD.inv(-179.75, 10.25, 179.75, 10.25)[2] / 1000, rel=1e-9)
    assert donor_row(donors, 150, 719).donor_lon_index == 0
    assert west.donor_lon == pytest.approx(179.75) and west.target_lon == pytest.approx(-179.75)


def test_latitude_never_wraps_at_the_poles():
    native, x, targets = empty_grids()
    targets[359, 10] = targets[0, 20] = targets[359, 200] = True
    for k in (9, 10, 11):
        native_cell(native, x, 0, k, 1.0)                   # every modulo-wrapped candidate of (359, 10)
        native_cell(native, x, 359, k + 10, 1.0)            # every modulo-wrapped candidate of (0, 20)
    native_cell(native, x, 359, 370, 1.0)                   # the over-the-North-Pole mirror of (359, 10)
    native_cell(native, x, 0, 380, 1.0)                     # the over-the-South-Pole mirror of (0, 20)
    native_cell(native, x, 358, 200, 2.0)                   # a legitimate southern neighbour
    donors = h.one_ring_donors(targets, native, x)
    assert donor_row(donors, 359, 10).donor_lat_index == -1
    assert donor_row(donors, 0, 20).donor_lat_index == -1
    assert donor_row(donors, 359, 200).donor_c2 == 2.0


def test_pre_and_pet_travel_together_as_one_derived_value():
    days = np.asarray(pd.period_range('1920-01', '1949-12', freq='M').days_in_month, dtype=float)
    pre = np.full((360, 2, 2), np.nan)                      # rows 200-201, columns 100-101
    pet = np.full((360, 2, 2), np.nan)
    pre[:, 0, 1], pet[:, 0, 1] = 100.0, 1.0                 # donor A (200, 101), east: wet
    pre[:, 1, 0], pet[:, 1, 0] = 10.0, 8.0                  # donor B (201, 100), north and farther: dry
    pre[:, 0, 0] = 50.0                                     # target (200, 100): PRE on land, PET masked
    _, _, x_small, native_small, _ = h.cell_climatology(pre, pet, days)
    native, x, targets = empty_grids()
    native[200:202, 100:102], x[200:202, 100:102] = native_small, x_small
    targets[200, 100] = True
    row = donor_row(h.one_ring_donors(targets, native, x), 200, 100)
    p_a, e_a, e_b = 1200.0, 1.0 * days.sum() / 30, 8.0 * days.sum() / 30
    assert row.donor_c2 == x[200, 101] == pytest.approx(np.log10(p_a / e_a))
    assert row.donor_c2 != pytest.approx(np.log10(p_a / e_b))
    assert row.donor_c2 != pytest.approx(np.log10(50.0 * 12 / e_a))   # the target's own PRE is never mixed in


# --- structural eligibility --------------------------------------------------------------------

def test_variable_states_and_structural_eligibility():
    rec, win = 1500, 360
    #                        masked  complete  window-long gap  nonpositive  partial   malformed  NaN in window, fill outside
    record = np.array([[rec,     0,        360,             0,           12,       0,         5]])
    window = np.array([[win,     0,        360,             0,           12,       0,         0]])
    finite = np.array([[0,       win,      0,               win,         win - 12, win - 1,   win - 1]])
    mean = np.array([[0.0,       5.0,      0.0,             0.0,         5.0,      5.0,       5.0]])
    states = h.variable_state(record, window, finite, mean, rec, win)
    assert states.tolist() == [['masked', 'complete', 'partial_fill', 'nonpositive_mean', 'partial_fill', 'malformed',
                                'malformed']]
    m, c = np.array([['masked']], dtype=object), np.array([['complete']], dtype=object)
    assert h.structural_mask_cells(m, m).item() and h.structural_mask_cells(m, c).item() and h.structural_mask_cells(c, m).item()
    assert not h.structural_mask_cells(c, c).item()          # native, never a target
    for bad in ['partial_fill', 'malformed', 'nonpositive_mean']:
        b = np.array([[bad]], dtype=object)
        assert not h.structural_mask_cells(m, b).item()      # PRE masked, PET defective
        assert not h.structural_mask_cells(b, m).item()      # PET masked, PRE defective
        assert not h.structural_mask_cells(c, b).item() and not h.structural_mask_cells(b, b).item()


def test_non_structural_invalid_support_stays_unresolved_counts_against_coverage_and_does_not_stop():
    native, x, _ = empty_grids()
    native_cell(native, x, 200, 101, 0.4)
    searched = np.zeros((N_LAT, N_LON), bool)
    searched[200, 100] = True                                # even with a donor-search record ...
    structural = np.zeros((N_LAT, N_LON), bool)              # ... a non-structural cell is never harmonized
    cells = cells_frame([(0, 200, 100, 1.0), (0, 200, 101, 99.0)])
    resolved = h.resolve_support(cells, x, native, structural, h.one_ring_donors(searched, native, x))
    bad = resolved[(resolved.i == 200) & (resolved.k == 100)].iloc[0]
    assert bad.status == h.UNRESOLVED and bad.reason == h.NON_STRUCTURAL and np.isnan(bad.c2_value)
    assert pd.isna(bad.donor_lat_index) and pd.isna(bad.cross_border)
    ones = np.ones((N_LAT, N_LON))
    qa = h.aggregate(resolved, ones, ones, ones, ones, np.zeros((N_LAT, N_LON), bool), 1)
    assert qa.loc[0, 'coverage'] == pytest.approx(0.99) and qa.loc[0, 'non_structural_invalid_area_km2'] == 1.0
    assert qa.loc[0, 'unresolved_area_km2'] == 1.0 and qa.loc[0, 'n_non_structural_invalid_cells'] == 1
    assert h.hard_stops(qa, ['AAA']) == []                 # Amendment 3: presence alone no longer stops


def test_non_structural_cells_never_donate():
    native, x, targets = empty_grids()
    x[200, 101] = 0.9                                        # a finite value in a non-native cell is never a candidate
    targets[200, 100] = True
    donors = h.one_ring_donors(targets, native, x)
    assert donor_row(donors, 200, 100).donor_lat_index == -1


def test_resolved_coverage_of_exactly_0_98_passes_and_below_fails():
    native, x, _ = empty_grids()
    native_cell(native, x, 200, 100, 0.1)
    native_cell(native, x, 250, 100, 0.1)
    structural = np.zeros((N_LAT, N_LON), bool)             # the invalid cells are non-structural
    ones = np.ones((N_LAT, N_LON))
    for native_area, missing, expected in [(98.0, 2.0, []), (97.9, 2.1, ['coverage < 0.98'])]:
        cells = cells_frame([(0, 200, 100, native_area), (0, 300, 300, missing)])
        resolved = h.resolve_support(cells, x, native, structural, h.one_ring_donors(structural, native, x))
        qa = h.aggregate(resolved, ones, ones, ones, ones, np.zeros((N_LAT, N_LON), bool), 1)
        assert bool(qa.loc[0, 'coverage'] == 0.98) is (native_area == 98.0)
        assert [s['rule'] for s in h.hard_stops(qa, ['AAA'])] == expected


def test_unresolved_area_is_structural_plus_non_structural_and_the_table_lists_every_row():
    rng = np.random.default_rng(11)
    native, x, _ = empty_grids()
    rows = [(c, int(rng.integers(100, 140)), int(rng.integers(0, 40)), float(rng.uniform(1, 50))) for c in range(5)
            for _ in range(60)]
    cells = cells_frame(rows).groupby(['idx', 'i', 'k'], as_index=False, sort=True).area.sum()
    for i, k in sorted(set(zip(cells.i, cells.k)))[::3]:
        native_cell(native, x, i, k, float(rng.normal()))
    invalid = sorted(set(zip(cells.i, cells.k)) - set(zip(*np.nonzero(native))))
    structural = np.zeros((N_LAT, N_LON), bool)
    for i, k in invalid[::2]:
        structural[i, k] = True
    resolved = h.resolve_support(cells, x, native, structural, h.one_ring_donors(structural, native, x))
    ones = np.ones((N_LAT, N_LON))
    qa = h.aggregate(resolved, ones, ones, ones, ones, np.zeros((N_LAT, N_LON), bool), 5)
    assert (qa.non_structural_invalid_area_km2 > 0).all() and (qa.unresolved_structural_area_km2 >= 0).all()
    np.testing.assert_allclose(qa.unresolved_structural_area_km2 + qa.non_structural_invalid_area_km2,
                               qa.unresolved_area_km2, rtol=AREA_RTOL, atol=0)
    np.testing.assert_allclose(qa.native_valid_area_km2 + qa.harmonized_area_km2 + qa.unresolved_area_km2,
                               qa.total_land_area_km2, rtol=AREA_RTOL, atol=0)
    codes = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']
    states = np.full((N_LAT, N_LON), h.MASKED, dtype=object)
    table = h.unresolved_support(h.cell_audit(resolved, codes, states, states))
    assert len(table) == int((resolved.status == h.UNRESOLVED).sum())
    assert set(table.reason) <= {h.STRUCTURAL, h.NON_STRUCTURAL}
    by_country = table.groupby('iso3').target_land_area_km2.sum().reindex(codes, fill_value=0.0).to_numpy()
    np.testing.assert_allclose(by_country, qa.unresolved_area_km2, rtol=AREA_RTOL, atol=0)
    assert table.columns.tolist() == ['target_lat_index', 'target_lon_index', 'target_lat', 'target_lon', 'iso3',
                                      'target_land_area_km2', 'pre_state', 'pet_state', 'reason']


def test_zero_precipitation_with_masked_pet_stays_unresolved_without_a_donor():
    """Regression for CRU cell (164, 200) in Peru: PRE exactly 0 in every month, PET fill in every month."""
    days = np.asarray(pd.period_range('1920-01', '1949-12', freq='M').days_in_month, dtype=float)
    pre = np.full((360, 1, 2), 30.0)                         # (164, 200) target, (164, 201) native neighbour
    pet = np.full((360, 1, 2), 3.0)
    pre[:, 0, 0], pet[:, 0, 0] = 0.0, np.nan
    p_bar, e_bar, x_small, native_small, _ = h.cell_climatology(pre, pet, days)
    record_fill = {'pre': np.array([[0, 0]]), 'pet': np.array([[1500, 0]])}
    window_fill = {'pre': np.array([[0, 0]]), 'pet': np.array([[360, 0]])}
    finite = {'pre': np.isfinite(pre).sum(axis=0), 'pet': np.isfinite(pet).sum(axis=0)}
    pre_state = h.variable_state(record_fill['pre'], window_fill['pre'], finite['pre'], p_bar, 1500)
    pet_state = h.variable_state(record_fill['pet'], window_fill['pet'], finite['pet'], e_bar, 1500)
    assert (pre_state[0, 0], pet_state[0, 0]) == ('nonpositive_mean', 'masked')
    assert not h.structural_mask_cells(pre_state, pet_state)[0, 0]
    native, x, _ = empty_grids()
    native[164, 200:202], x[164, 200:202] = native_small[0], x_small[0]
    structural = np.zeros((N_LAT, N_LON), bool)
    structural[164, 200:202] = h.structural_mask_cells(pre_state, pet_state)[0]
    support = cells_frame([(0, 164, 200, 23.01253170656061), (0, 164, 201, 5000.0)])
    donors = h.one_ring_donors(structural, native, x)
    resolved = h.resolve_support(support, x, native, structural, donors)
    peru = resolved[(resolved.i == 164) & (resolved.k == 200)].iloc[0]
    assert (peru.status, peru.reason) == (h.UNRESOLVED, h.NON_STRUCTURAL)
    assert pd.isna(peru.donor_lat_index) and np.isnan(peru.donor_c2) and np.isnan(peru.c2_value)
    ones = np.ones((N_LAT, N_LON))
    qa = h.aggregate(resolved, ones, ones, ones, ones, np.zeros((N_LAT, N_LON), bool), 1)
    assert qa.loc[0, 'coverage'] == pytest.approx(5000.0 / (5000.0 + 23.01253170656061))
    assert qa.loc[0, 'c2_harmonized'] == pytest.approx(x[164, 201], rel=1e-15) and h.hard_stops(qa, ['PER']) == []


def test_frozen_input_pins_fail_closed(tmp_path):
    good = tmp_path / 'input.bin'
    good.write_bytes(b'frozen')
    import hashlib
    h.check_frozen_inputs({good: hashlib.sha256(b'frozen').hexdigest()})
    with pytest.raises(ValueError):
        h.check_frozen_inputs({good: '0' * 64})
    assert {Path(p).name for p in h.FROZEN_INPUT_SHA256} == {'m1a_cell_land_area_km2.npz', 'm1a_geography_qa.csv',
                                                          'gpw_v4_population_count_adjusted_rev11_15_min.nc',
                                                          'gpw_v4_national_identifier_grid_rev11_lookup.txt'}


def test_fill_count_distinguishes_fill_from_malformed_values(tmp_path):
    path = tmp_path / 'toy.nc'
    fill = np.float32(9.96921e36)
    with netCDF4.Dataset(path, 'w', format='NETCDF3_CLASSIC') as nc:
        nc.createDimension('time', 4)
        nc.createDimension('lat', 1)
        nc.createDimension('lon', 3)
        v = nc.createVariable('pet', 'f4', ('time', 'lat', 'lon'), fill_value=fill)
        v.missing_value = fill
        v.set_auto_maskandscale(False)
        data = np.ones((4, 1, 3), np.float32)
        data[:, 0, 0] = fill                                  # masked every month
        data[1:3, 0, 1] = fill                                # fill in some months only
        data[2, 0, 2] = np.nan                                # malformed, not the fill value
        v[:] = data
    assert h.fill_count(path, 'pet', np.arange(4)).tolist() == [[4, 2, 0]]
    assert h.fill_count(path, 'pet', np.array([1, 2])).tolist() == [[2, 2, 0]]
    with pytest.raises(ValueError):
        h.fill_count(path, 'pet', np.array([0, 2]))


# --- country aggregation -----------------------------------------------------------------------

def test_country_value_coverage_and_donor_distances():
    native, x, _ = empty_grids()
    native_cell(native, x, 200, 100, np.log10(0.5))
    native_cell(native, x, 200, 103, np.log10(2.0))          # donor for (200, 102); outside country 0
    cells = cells_frame([(0, 200, 100, 4.0), (0, 200, 102, 1.0), (0, 250, 400, 1.0)])  # (250, 400): no donor
    _, resolved, qa = run(cells, native, x, 1)
    r = qa.iloc[0]
    assert r.c2_harmonized == pytest.approx((4 * np.log10(0.5) + np.log10(2.0)) / 5)
    assert r.c2_native_support_only == pytest.approx(np.log10(0.5))
    assert r.delta_c2_harmonization == pytest.approx(r.c2_harmonized - r.c2_native_support_only)
    assert (r.native_fraction, r.harmonized_fraction, r.unresolved_fraction) == pytest.approx((4 / 6, 1 / 6, 1 / 6))
    assert r.coverage == pytest.approx(5 / 6)
    assert (r.n_native_cells, r.n_harmonized_cells, r.n_unresolved_cells) == (1, 1, 1)
    d = h.neighbour_distance_km(200, 0, 1)
    assert r.mean_donor_distance_km == r.area_weighted_mean_donor_distance_km == r.max_donor_distance_km == d
    orphan = resolved[(resolved.i == 250) & (resolved.k == 400)].iloc[0]
    assert (orphan.status, orphan.reason) == (h.UNRESOLVED, h.STRUCTURAL)
    assert r.n_non_structural_invalid_cells == 0 and r.unresolved_structural_area_km2 == 1.0
    assert [s['rule'] for s in h.hard_stops(qa, ['AAA'])] == ['coverage < 0.98']


def test_country_without_native_support_reports_na_and_stops():
    native, x, _ = empty_grids()
    native_cell(native, x, 200, 101, 0.1)
    cells = cells_frame([(0, 200, 100, 3.0)])
    _, _, qa = run(cells, native, x, 1)
    assert np.isnan(qa.loc[0, 'c2_native_support_only']) and np.isnan(qa.loc[0, 'delta_c2_harmonization'])
    assert qa.loc[0, 'coverage'] == 1.0
    assert any('no native-valid' in s['rule'] for s in h.hard_stops(qa, ['AAA']))


def test_station_audit_stays_on_native_cells_when_fragments_are_harmonized():
    native, x, _ = empty_grids()
    native_cell(native, x, 200, 100, 0.1)
    native_cell(native, x, 200, 101, 0.2)
    cells = cells_frame([(0, 200, 99, 50.0), (0, 200, 100, 3.0), (0, 200, 101, 1.0)])  # harmonized area dominates
    supported, mean_stn = np.zeros((N_LAT, N_LON)), np.zeros((N_LAT, N_LON))
    supported[200, 100], supported[200, 101] = 0.2, 0.6
    mean_stn[200, 100], mean_stn[200, 101] = 2.0, 6.0
    pure = np.zeros((N_LAT, N_LON), bool)
    pure[200, 100] = pure[200, 101] = True
    _, _, qa = run(cells, native, x, 1, supported, mean_stn, pure)
    assert qa.loc[0, 'harmonized_area_km2'] == 50.0
    assert qa.loc[0, 'pre_station_supported_share'] == pytest.approx((3 * 0.2 + 0.6) / 4)
    assert qa.loc[0, 'pre_mean_station_count'] == pytest.approx((3 * 2.0 + 6.0) / 4)
    assert qa.loc[0, 'pre_pure_climatology_share'] == 1.0
    assert any(s['rule'].startswith('pathological') for s in h.hard_stops(qa, ['AAA']))


def test_shared_target_cell_resolves_identically_for_every_country():
    native, x, _ = empty_grids()
    native_cell(native, x, 200, 99, -0.2)
    native_cell(native, x, 201, 100, 0.9)
    cells = cells_frame([(0, 200, 100, 2.0), (1, 200, 100, 3.0), (1, 201, 100, 7.0), (2, 200, 100, 0.5)])
    _, resolved, _ = run(cells, native, x, 3)
    shared = resolved[(resolved.i == 200) & (resolved.k == 100)]
    assert shared.status.tolist() == [h.HARMONIZED] * 3
    assert shared.donor_lat_index.nunique() == 1 and shared.donor_lon_index.nunique() == 1
    assert shared.c2_value.nunique() == 1


def test_area_is_conserved_for_every_country():
    rng = np.random.default_rng(7)
    native, x, _ = empty_grids()
    rows = []
    for c in range(12):
        for _ in range(40):
            i, k = int(rng.integers(100, 260)), int(rng.integers(0, N_LON))
            rows.append((c, i, k, float(rng.uniform(0.01, 3000))))
            if rng.random() < 0.7:
                native_cell(native, x, i, k, float(rng.normal()))
    cells = cells_frame(rows).groupby(['idx', 'i', 'k'], as_index=False, sort=True).area.sum()
    _, resolved, qa = run(cells, native, x, 12)
    total = cells.groupby('idx').area.sum().to_numpy()
    np.testing.assert_allclose(qa.total_land_area_km2, total, rtol=AREA_RTOL, atol=0)
    np.testing.assert_allclose(qa.native_valid_area_km2 + qa.harmonized_area_km2 + qa.unresolved_area_km2,
                               qa.total_land_area_km2, rtol=AREA_RTOL, atol=0)
    np.testing.assert_allclose(qa.native_fraction + qa.harmonized_fraction + qa.unresolved_fraction, 1,
                               rtol=AREA_RTOL, atol=0)
    assert set(resolved.status) <= {h.NATIVE, h.HARMONIZED, h.UNRESOLVED}


def test_without_targets_the_amended_aggregate_equals_the_original_rule():
    land = np.zeros((720, 1440))
    country = np.full((720, 1440), -1, dtype=np.int16)
    land[0:2, 0:2] = 1.0
    land[0, 2] = 3.0
    land[2, 0] = 1.0                                          # CRU cell (1, 0): invalid and not structural
    country[0:3, 0:4] = 0
    cells = h.country_cells(['AAA'], land, country)
    native, x, _ = empty_grids()
    native_cell(native, x, 0, 0, np.log10(0.5))
    native_cell(native, x, 0, 1, np.log10(2.0))
    structural = np.zeros((N_LAT, N_LON), bool)
    resolved = h.resolve_support(cells, x, native, structural, h.one_ring_donors(structural, native, x))
    ones = np.ones((N_LAT, N_LON))
    qa = h.aggregate(resolved, ones, ones, ones, ones * 2, np.zeros((N_LAT, N_LON), bool), 1)
    geometric = np.exp((4 * np.log(0.5) + 3 * np.log(2.0)) / 7)
    assert qa.c2_harmonized[0] == qa.c2_native_support_only[0] == pytest.approx(np.log10(geometric))
    assert qa.coverage[0] == qa.native_fraction[0] == pytest.approx(7 / 8)
    assert qa.pre_mean_station_count[0] == 2.0


def test_hard_stops_cover_the_coverage_gate_and_non_finite_values():
    qa = pd.DataFrame({'coverage': [0.97999, 0.98, 1.0, np.nan, 1.0, 1.0],
                       'c2_harmonized': [0.1, np.nan, 0.2, 0.3, 0.4, 0.5],
                       'total_land_area_km2': [1.0, 1.0, 1.0, 1.0, np.nan, 1.0],
                       'native_valid_area_km2': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       'harmonized_area_km2': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       'unresolved_area_km2': [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                       'n_non_structural_invalid_cells': [0, 0, 0, 0, 0, 7],
                       'pre_pure_climatology_share': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       'unresolved_structural_area_km2': [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                       'non_structural_invalid_area_km2': [0.0, 0.0, 0.0, 0.0, 0.0, 0.01]})
    stops = {(s['iso3'], s['rule'].split(':')[0]) for s in h.hard_stops(qa, ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF'])}
    assert stops == {('AAA', 'coverage < 0.98'), ('BBB', 'non-finite C2'), ('CCC', 'pathological'),
                     ('DDD', 'coverage < 0.98'), ('EEE', 'malformed area QA')}
    ok = qa.iloc[[5]].reset_index(drop=True)
    assert h.hard_stops(ok, ['FFF']) == []
    for column, bad in [('harmonized_area_km2', -1e-9), ('non_structural_invalid_area_km2', -0.5), ('total_land_area_km2', 0.0)]:
        assert [s['rule'] for s in h.hard_stops(ok.assign(**{column: bad}), ['FFF'])][:1] == ['malformed area QA']
    nan_share = qa.iloc[[0]].assign(coverage=1.0, pre_pure_climatology_share=np.nan, unresolved_area_km2=0.0,
                                     unresolved_structural_area_km2=0.0)
    assert [s['rule'].split(':')[0] for s in h.hard_stops(nan_share, ['EEE'])] == ['pathological']


def test_harmonization_summary_counts_distinct_cells_and_sums_rows():
    native, x, _ = empty_grids()
    native_cell(native, x, 200, 101, 1.0)                    # east donor for (200, 100), outside all support
    native_cell(native, x, 261, 300, 2.0)                    # north donor for (260, 300), supports country 1
    cells = cells_frame([(0, 200, 100, 2.0), (1, 200, 100, 3.0), (1, 260, 300, 4.0), (1, 261, 300, 9.0),
                         (0, 100, 600, 0.5)])                # (100, 600): no donor
    _, resolved, _ = run(cells, native, x, 2)
    support = np.zeros((N_LAT, N_LON), bool)
    support[cells.i, cells.k] = True
    s = h.harmonization_summary(resolved, support)
    d = sorted([h.neighbour_distance_km(200, 0, 1), h.neighbour_distance_km(260, 1, 0)])
    assert (s['n_structural_target_cells'], s['n_harmonized_cells'], s['n_unresolved_structural_cells']) == (3, 2, 1)
    assert s['harmonized_area_km2'] == 9.0 and s['unresolved_area_km2'] == s['unresolved_structural_area_km2'] == 0.5
    assert s['non_structural_invalid_area_km2'] == 0.0
    assert s['donor_distance_km']['max'] == d[1] and s['donor_distance_km']['median'] == pytest.approx(np.mean(d))
    assert s['n_cross_border_country_cells'] == 2 and s['n_donor_cells_outside_all_support'] == 1
    json.dumps(s, allow_nan=False)
    empty = h.harmonization_summary(resolved[resolved.status == h.NATIVE], support)
    assert empty['donor_distance_km'] is None and json.dumps(empty, allow_nan=False)


def test_audit_csv_writes_integer_donor_indices_and_explicit_na():
    native, x, _ = empty_grids()
    native_cell(native, x, 201, 100, 0.5)
    cells = cells_frame([(0, 200, 100, 1.0), (0, 300, 300, 1.0), (0, 201, 100, 5.0)])
    _, resolved, qa = run(cells, native, x, 1)
    text = h.csv_text(h.cell_audit(resolved, ['AAA'], masked_states(), masked_states()))
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    done = frame[frame.target_lat_index == '200'].iloc[0]
    assert (done.donor_lat_index, done.donor_lon_index, done.cross_border, done.donor_countries) == ('201', '100', 'False', 'AAA')
    orphan = frame[frame.target_lat_index == '300'].iloc[0]
    for col in ['donor_lat_index', 'donor_lon_index', 'donor_lat', 'donor_lon', 'donor_distance_km', 'donor_c2',
                'donor_countries', 'cross_border']:
        assert orphan[col] == 'NA', col
    assert (orphan.pre_state, orphan.status, orphan.reason) == ('masked', 'unresolved', 'structural_cru_mask')


def test_serialized_outputs_are_byte_identical_across_runs():
    rng = np.random.default_rng(3)
    native, x, _ = empty_grids()
    rows = [(c, int(rng.integers(150, 200)), int(rng.integers(0, 50)), float(rng.uniform(1, 9))) for c in range(4)
            for _ in range(30)]
    for i, k in sorted({(r[1], r[2]) for r in rows[::2]}):
        native_cell(native, x, i, k, float(rng.normal()))
    cells = cells_frame(rows).groupby(['idx', 'i', 'k'], as_index=False, sort=True).area.sum()
    blobs = []
    for _ in range(2):
        _, resolved, qa = run(cells, native, x, 4)
        blobs.append((h.csv_text(qa), h.csv_text(h.cell_audit(resolved, ['AAA', 'BBB', 'CCC', 'DDD'],
                                                              masked_states(), masked_states()))))
    assert blobs[0] == blobs[1]


@pytest.mark.skipif(os.environ.get('M1B_FULL_BUILD') != '1' or not h.CRU_DIR.exists(),
                    reason='opt-in full C2 build (set M1B_FULL_BUILD=1; needs the pinned CRU files)')
def test_two_full_builds_in_separate_processes_are_byte_identical(tmp_path):
    for name, seed in [('a', '1'), ('b', '2')]:
        subprocess.run([sys.executable, '-m', 'research.model_v2.m1b_hydroclimate', str(tmp_path / name)], check=True,
                       cwd=h.m0.ROOT, env={**os.environ, 'PYTHONHASHSEED': seed}, capture_output=True)
    for f in h.DETERMINISTIC_OUTPUTS:
        assert (tmp_path / 'a' / f).read_bytes() == (tmp_path / 'b' / f).read_bytes(), f
