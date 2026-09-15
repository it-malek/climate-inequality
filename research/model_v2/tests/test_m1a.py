"""M1a measurement and contract rules on synthetic fixtures (M1A_MEASUREMENT_SPEC.md §6)."""
import numpy as np
import pandas as pd
import pytest
import shapely

from research.model_v2 import m1a_geography as g


class NoCoast:
    def distance_km(self, lon, lat):
        return np.zeros(len(lon))


def run_cell(geom, r0, c0, z, koppen=None, n_countries=1):
    records = g.cell_records(geom, r0, c0)
    country = np.full((720, 1440), -1, dtype=np.int16)
    country[r0 // g.CELL_PIX, c0 // g.CELL_PIX] = 0
    kg = np.zeros((360, 720), dtype=np.int8) if koppen is None else koppen
    sums = g.Sums(n_countries)
    g.accumulate(records, sums, country, lambda r, c: z(r, c), kg, NoCoast(), g.pixel_area_rows(1))
    g.finalize_elevation(sums, lambda r, c: z(r, c))
    return sums


def test_pixel_lattice_area_conserves_the_sphere():
    total = g.pixel_area_rows(1).sum() * g.N_COLS
    assert total == pytest.approx(4 * np.pi * g.R_KM**2, rel=1e-12)
    assert g.pixel_area_rows(2).sum() * 2 * g.N_COLS == pytest.approx(total, rel=1e-12)


def test_negative_land_elevation_retained_and_negative_ocean_excluded():
    # One 0.25° cell at 10–10.25°N, 20–20.25°E; land ends at 20.09°E, inside the pixel
    # 20.0833–20.1°E whose centre (20.0917°E) is water.
    r0, c0 = (90 + 10) * g.PIX, (180 + 20) * g.PIX
    land = shapely.box(20, 10, 20.09, 10.25)

    def z(r, c):
        lon = -180 + (c + .5) / g.PIX
        return np.where(lon < 20.0834, -5.0, -50.0)

    sums = run_cell(land, r0, c0, z)
    qa = g.features_from_sums(sums, ['AAA'])
    assert sums.area[0] == pytest.approx(g.row_band_area_km2(10, 10.25, .09), rel=1e-9)
    assert qa.elevation_original_centre_only[0] == pytest.approx(-5.0)    # genuine negative land kept
    assert qa.elevation_original_coverage[0] < 1                           # original centre rule misses the sliver
    assert qa.elevation_coverage[0] == pytest.approx(1.0)                  # Amendment 1 completes it
    assert qa.elevation[0] == pytest.approx(-5.0)                          # from adjacent land, never the -50 water value
    assert qa.elevation_unresolved_area_km2[0] == 0
    assert qa.elevation_s1_completed_at_0m[0] > -5.0


def test_isolated_islet_stays_unresolved_and_ring2_is_only_a_sensitivity():
    r0, c0 = (90 + 10) * g.PIX, (180 + 20) * g.PIX
    main = shapely.box(20, 10, 20.05, 10.25)                     # land-centred pixels, z = 7
    islet = shapely.box(20.1 + .002, 10.1 + .002, 20.1 + .006, 10.1 + .006)  # inside one water-centred pixel
    far = shapely.box(20.2 + .002, 10.1 + .002, 20.2 + .006, 10.1 + .006)
    sums = run_cell(shapely.union_all([main, islet, far]), r0, c0, lambda r, c: np.full(len(r), 7.0))
    qa = g.features_from_sums(sums, ['ISL'])
    islets = sums.completion['unresolved_area'][0]
    assert islets == pytest.approx(shapely.area(islet) / shapely.area(main) * sums.elev_area[0] * 2, rel=.02)
    assert qa.elevation[0] == pytest.approx(7.0) and qa.elevation_coverage[0] < 1
    assert qa.elevation_s2_unresolved_at_0m[0] < 7.0
    assert np.isnan(qa.elevation_s4_unresolved_at_completed_min[0])  # no completed values: bound undefined
    assert qa.elevation_s3_coverage[0] == pytest.approx(qa.elevation_coverage[0])  # 3 pixels away: not ring 2 either


def test_completion_never_crosses_analytical_units():
    resolved = np.zeros((g.N_ROWS, g.N_COLS), dtype=np.int16)
    resolved[100, 100] = 2          # country index 1
    resolved[100, 102] = 1          # country index 0
    z = lambda r, c: np.where(c == 100, 50.0, 9.0)  # noqa: E731
    got = g.neighbour_mean(np.array([100, 100]), np.array([101, 101]), np.array([0, 1]), resolved, z, 1)
    assert got.tolist() == [9.0, 50.0]
    edge = np.zeros_like(resolved)
    edge[5, g.N_COLS - 1] = 1       # longitude wraps
    assert g.neighbour_mean(np.array([5]), np.array([0]), np.array([0]), edge, lambda r, c: np.full(len(r), 3.), 1)[0] == 3.


def test_mean_absolute_latitude_differs_from_centroid_and_hemisphere_majority():
    records = g.Records()
    rr, cc = np.meshgrid(np.arange(89 * g.PIX, 92 * g.PIX), np.arange(0, 30), indexing='ij')  # -1°..+2°
    records.add_full(rr, cc)
    country = np.zeros((720, 1440), dtype=np.int16)
    sums = g.Sums(1)
    g.accumulate(records, sums, country, lambda r, c: np.zeros(len(r)), np.zeros((360, 720), np.int8),
                 NoCoast(), g.pixel_area_rows(1))
    g.finalize_elevation(sums, lambda r, c: np.zeros(len(r)))
    qa = g.features_from_sums(sums, ['EQX'])
    assert qa.abs_latitude[0] == pytest.approx((0.5 + 1 * 2 * 1.0) / 3, abs=2e-3)  # mean |lat| ≈ 0.833
    assert qa.hemisphere[0] == 'N'
    assert qa.north_area_fraction[0] == pytest.approx(2 / 3, abs=1e-3)


def test_hemisphere_tie_uses_north():
    sums = g.Sums(1)
    sums.area[:] = 2.0
    sums.north_area[:] = 1.0
    for b in g.DISTANCE_BLOCKS:
        sums.dist[b][0][:] = 2.0
    sums.elev_area[:] = 2.0
    sums.koppen[0, 0] = 2.0
    g.finalize_elevation(sums, lambda r, c: np.zeros(len(r)))
    assert g.features_from_sums(sums, ['TIE']).hemisphere[0] == 'N'


def test_climate_identifiability_rule():
    assert g.dominant_climate(np.array([5, 3, 0, 0, 0, 0.])) == ('A', '')
    assert g.dominant_climate(np.array([5, 5, 0, 0, 0, 0.]))[0] is None            # exact tie is not a measurement
    assert g.dominant_climate(np.array([50, 48, 0, 0, 0, 3.]))[0] is None          # 50 <= 48 + 3
    assert g.dominant_climate(np.array([60, 30, 0, 0, 0, 29.]))[0] == 'A'          # 60 > 59, only 51% classified
    assert g.dominant_climate(np.array([0, 97, 96, 0, 0, 2.]))[0] is None           # 98% classified but not identifiable
    assert g.dominant_climate(np.array([9700.6, 1622.1, 0, 0, 0, 2103.3]))[0] == 'A'  # Bahamas-like margin


def test_lake_hierarchy_and_full_block_detection():
    levels = {1: np.array([shapely.box(0, 0, 1, 1)]),
              2: np.array([shapely.box(.25, .25, .5, .5)]),
              3: np.array([shapely.box(.3, .3, .35, .35)]), 4: np.array([], dtype=object)}
    land = g.land_geometry(levels)
    assert land.area == pytest.approx(1 - .0625 + .0025)
    country = np.full((720, 1440), -1, dtype=np.int16)
    country[360:364, 720:724] = 0  # cells covering 0–1°N, 0–1°E
    events = []
    g.walk(levels, country, lambda kind, r0, r1, c0, c1, geom: events.append((kind, r0, r1, c0, c1)),
           360, 364, 720, 724)
    kinds = [e[0] for e in events]
    assert 'cell' in kinds and 'full' in kinds
    covered = sum((r1 - r0) * (c1 - c0) for _, r0, r1, c0, c1 in events)
    assert covered == 16  # every target cell exactly once


def test_coast_seams_removed_and_distance_wraps_dateline():
    east = shapely.box(170, -10, 180, 10)
    west = shapely.box(-180, -10, -170, 10)
    land = shapely.MultiPolygon([east, west])
    segments, removed = g.coastline_segments(land)
    assert removed == 2
    coast = g.Coast.build(land, spacing_km=1.0)
    # Point on the seam, 10° from the east/west edges but 5° from the northern shore at 10°N.
    d = coast.distance_km(np.array([180.0]), np.array([5.0]))[0]
    assert d == pytest.approx(np.radians(5) * g.R_KM, abs=0.6)


def test_densified_arcs_have_bounded_spacing():
    seg = np.array([[0.0, 0.0, 10.0, 10.0]])
    xyz, parent, a, b = g.densify_arcs(seg, spacing_km=5.0)
    assert np.allclose(np.linalg.norm(xyz, axis=1), 1) and (parent == 0).all()
    gaps = g.chord_to_km(np.linalg.norm(np.diff(xyz[np.argsort(xyz @ np.cross(a[0], b[0]) + xyz @ a[0])], axis=0), axis=1))
    assert gaps.max() <= 5.0 + 1e-9 or len(xyz) > 300


def test_exact_arc_distance_matches_brute_force_over_all_segments():
    rng = np.random.default_rng(3)
    # A jagged closed polygon plus a far island, so candidate sets must be certified.
    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    radius = 8 + 2 * np.sin(7 * angles) + rng.uniform(-.5, .5, len(angles))
    main = shapely.Polygon(np.column_stack([20 + radius * np.cos(angles), 10 + radius * np.sin(angles)]))
    land = shapely.MultiPolygon([main, shapely.box(60, -40, 61, -39)])
    coast = g.Coast.build(land, spacing_km=200.0)
    lon, lat = rng.uniform(-60, 100, 3000), rng.uniform(-70, 70, 3000)
    fast = coast.distance_km(lon, lat)
    p = g.unit_xyz(lon, lat)
    brute = np.array([g.arc_distance_rad(p[i], coast.a, coast.b).min() for i in range(len(p))]) * g.R_KM
    np.testing.assert_allclose(fast, brute, rtol=0, atol=1e-6)
    # Point exactly on the arc interior and beyond an endpoint.
    a, b = g.unit_xyz(np.array([0.0]), np.array([0.0])), g.unit_xyz(np.array([10.0]), np.array([0.0]))
    assert g.arc_distance_rad(g.unit_xyz(np.array([5.0]), np.array([3.0])), a, b)[0] == pytest.approx(np.radians(3))
    assert g.arc_distance_rad(g.unit_xyz(np.array([13.0]), np.array([0.0])), a, b)[0] == pytest.approx(np.radians(3))


def test_block_centroid_distance_quadrature_matches_pixels_for_a_linear_field():
    class Linear:  # distance proportional to latitude: block centroid quadrature is exact to O(pixel^2)
        def distance_km(self, lon, lat):
            return 1000 + 10 * np.asarray(lat)
    records = g.Records()
    rr, cc = np.meshgrid(np.arange(110 * g.PIX, 110 * g.PIX + 15), np.arange(600, 615), indexing='ij')
    records.add_full(rr, cc)
    country = np.full((720, 1440), -1, dtype=np.int16)
    country[rr[0, 0] // g.CELL_PIX, 600 // g.CELL_PIX] = 0
    sums = g.Sums(1)
    g.accumulate(records, sums, country, lambda r, c: np.zeros(len(r)), np.zeros((360, 720), np.int8),
                 Linear(), g.pixel_area_rows(1), check_idx=[0])
    g.finalize_elevation(sums, lambda r, c: np.zeros(len(r)))
    qa = g.features_from_sums(sums, ['LIN'])
    assert qa.continentality[0] == pytest.approx(qa.continentality_60s_check[0], abs=1e-4)
    assert qa.continentality_5min[0] == pytest.approx(qa.continentality_60s_check[0], abs=1e-4)
    for b in g.DISTANCE_BLOCKS:
        assert sums.dist[b][0][0] == pytest.approx(sums.area[0])


def test_contract_classification_rules():
    from research.model_v2.m1a_evaluate import classify

    def card(rmse, lo, hi, i_in, i_cv, geo=.5, emis=.02, sec=.04, region=.06, r2=.6, cv_r2=.15, in_rmse=.028):
        return {'paired_delta_vs_m0': {'delta_rmse': rmse, 'country_bootstrap_95_interval': [lo, hi]},
                'secondary': {'cv_rmse': sec}, 'worst_region': {'rmse': region},
                'residual_morans_i_in_sample': i_in, 'residual_morans_i_cv': i_cv,
                'share_geography': geo, 'share_emissions': emis, 'share_socioeconomic': .07, 'share_population': .03,
                'in_sample_r2': r2, 'cv_r2': cv_r2, 'cv_rmse': .043 + rmse, 'in_sample_rmse': in_rmse}
    errors = pd.DataFrame({'C0_abs_error': [.1] * 4, 'M1a_abs_error': [.05] * 4}, index=list('ABCD'))
    errors.index = ['CAN', 'BRA', 'RUS', 'DZA']
    base = card(0, 0, 0, .27, .32)
    better = classify(base, card(-.003, -.005, -.001, .20, .25), errors)
    assert better['summary'] == 'improved generalization and spatial specification'
    vetoed = classify(base, card(-.003, -.005, -.001, .27, .32, sec=.0412), errors)
    assert vetoed['transfer'] == 'no detectable change in transfer'
    spatial_only = classify(base, card(-.001, -.004, .002, .20, .26), errors)
    assert spatial_only['summary'] == 'improved spatial specification only'
    assert spatial_only['gate_m1a_to_m1']['not_worse']
    worse = classify(base, card(.004, .001, .007, .27, .32, geo=.01, emis=.2), errors)
    assert worse['transfer'] == 'worsened generalization'
    assert worse['material_change_to_conclusion']
    assert classify(base, card(0, -.001, .001, .27, .32, r2=.7, cv_r2=.15), errors)['overfitting_signal']
