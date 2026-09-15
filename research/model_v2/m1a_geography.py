"""M1a: area-consistent remeasurement of the station-derived geography features.

Implements ``M1A_MEASUREMENT_SPEC.md`` (approved 2026-09-15) under the frozen
``M1A_EVALUATION_CONTRACT.md``. No model is fitted here and nothing outside
``research/model_v2/outputs/m1a_*`` is written. Run with
``python -m research.model_v2.m1a_geography``.

Support ``D_c`` is the union of the country's GPW v4 rev11 band-11 0.25° cells
intersected with GSHHG 2.3.7 full-resolution land, (L1 − L2) ∪ (L3 − L4). The
ETOPO 2022 60″ pixel lattice is the quadrature grid. Its pixel edges fall exactly
on the 0.25° GPW and 0.5° Köppen edges, so every pixel lies in exactly one GPW
cell and one Köppen pixel. Pixel land fractions come from exact polygon clipping
in longitude/latitude, scaled by the pixel's spherical area
``R² Δλ (sin φ₂ − sin φ₁)``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import scipy
import shapely
import xarray as xr
from scipy.spatial import cKDTree

from research.model_v2 import m0
from research.model_v2.geometry import load_national_grid
from src.area_weighting import GPW_NATID_LOOKUP_PATH
from src.explain import ETOPO_PATH, ETOPO_VAR, KOPPEN_GROUPS, KOPPEN_PATH, KOPPEN_VAR
from src.interpolate import LAND_ZIP_PATH
from src.population import GPW_PATH

OUT = m0.OUTPUT_DIR
OUTPUT_DIR_TABLE = OUT / 'm0_countries.csv'
SPEC_PATH = Path(__file__).resolve().parent / 'M1A_MEASUREMENT_SPEC.md'
GSHHG_DIR = m0.ROOT / 'data' / 'raw' / 'gshhg'
GSHHG_ZIP = GSHHG_DIR / 'gshhg-shp-2.3.7.zip'
GSHHG_URL = 'https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip'
GSHHG_ZIP_SHA256 = '8dbbe7e071e77e9e75f2d639239099ebca8d5c16d6a07df8169729d49f15cf41'
GSHHG_LEVELS = (1, 2, 3, 4)

R_KM = 6371.0088
PIX = 60                      # ETOPO pixels per degree
CELL_PIX = 15                 # ETOPO pixels per 0.25° GPW cell
KOPPEN_PIX = 30               # ETOPO pixels per 0.5° Köppen pixel
N_ROWS, N_COLS = 180 * PIX, 360 * PIX
BLOCK_CELLS = 64              # initial recursion block, 16°
COAST_VERTEX_SPACING_KM = 50.0  # coarse candidate tree; distances are exact arc distances
COAST_K = 16
FULL_TOL = 1e-9               # relative area tolerance for a fully covered box
CLASSES = ('A', 'B', 'C', 'D', 'E')
FEATURES = ('abs_latitude', 'elevation', 'continentality', 'climate_zone', 'hemisphere', 'spatial_block')
GATES = {'numeric_min_coverage': 0.98, 'climate_min_classified': 0.95, 'geometry_min_coverage': 1.0}
DISTANCE_TOL = {'absolute_km': 0.1, 'relative': 0.001}
DISTANCE_BLOCKS = (3, 5)       # pixels per side: 3′ primary quadrature, 5′ convergence check
DISTANCE_60S_CHECK = ('BHS', 'JAM', 'NOR', 'CHL', 'GRC', 'PHL', 'IDN', 'MNG', 'KAZ', 'CAN')


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------

def row_band_area_km2(lat0, lat1, dlon_deg):
    """Spherical area of latitude band [lat0, lat1] with longitude width dlon_deg."""
    return R_KM**2 * np.radians(dlon_deg) * (np.sin(np.radians(lat1)) - np.sin(np.radians(lat0)))


def pixel_area_rows(step=1):
    """Area of one pixel in each ascending row of a lattice with 1/(PIX*step) degree spacing."""
    n = N_ROWS * step
    edges = -90 + np.arange(n + 1) / (PIX * step)
    return row_band_area_km2(edges[:-1], edges[1:], 1 / (PIX * step))


def unit_xyz(lon, lat):
    lam, phi = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])


def chord_to_km(chord):
    return 2 * R_KM * np.arcsin(np.clip(chord / 2, 0, 1))


# ---------------------------------------------------------------------------
# Coastline distance (Natural Earth 110m, spherical arcs)
# ---------------------------------------------------------------------------

def coastline_segments(land):
    """Boundary segments of the frozen land geometry without artificial seam edges.

    Edges lying on the ±180° meridian or the south-pole cap are polygon cuts, not
    shoreline, and are excluded.
    """
    segments, removed = [], 0
    for part in shapely.get_parts(land.boundary):
        c = shapely.get_coordinates(part)
        a, b = c[:-1], c[1:]
        seam = ((np.abs(a[:, 0]) == 180) & (np.abs(b[:, 0]) == 180)) | ((a[:, 1] < -89.9) & (b[:, 1] < -89.9))
        removed += int(seam.sum())
        segments.append(np.concatenate([a[~seam], b[~seam]], axis=1))
    return np.concatenate(segments), removed


def densify_arcs(segments, spacing_km):
    """Coarse vertices along each segment's shorter great-circle arc, with parent segment ids."""
    a, b = unit_xyz(segments[:, 0], segments[:, 1]), unit_xyz(segments[:, 2], segments[:, 3])
    omega = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1))
    n = np.maximum(1, np.ceil(omega * R_KM / spacing_km).astype(int))
    points, parents = [], []
    for count in np.unique(n):
        rows = np.flatnonzero(n == count)
        w, s = omega[rows][:, None], np.sin(omega[rows])[:, None]
        for k in range(count + 1):
            t = k / count
            with np.errstate(invalid='ignore', divide='ignore'):
                p = (np.sin((1 - t) * w) * a[rows] + np.sin(t * w) * b[rows]) / s
            points.append(np.where(s > 1e-15, p, a[rows]))
            parents.append(rows)
    xyz = np.concatenate(points)
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True), np.concatenate(parents), a, b


def arc_distance_rad(p, a, b):
    """Exact angular distance from unit points p to the minor great-circle arcs a→b (broadcast)."""
    n = np.cross(a, b)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    nhat = n / np.where(norm > 0, norm, 1)
    s = np.sum(p * nhat, axis=-1)
    q = p - s[..., None] * nhat
    inside = (norm[..., 0] > 0) & (np.sum(np.cross(a, q) * nhat, axis=-1) >= 0) & (
        np.sum(np.cross(q, b) * nhat, axis=-1) >= 0)
    to_plane = np.arcsin(np.clip(np.abs(s), 0, 1))

    def angle(u, v):
        return np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), np.sum(u * v, axis=-1))
    return np.where(inside, to_plane, np.minimum(angle(p, a), angle(p, b)))


@dataclass
class Coast:
    """Exact minimum great-circle distance to the shoreline segments (no densification error).

    A coarse vertex tree (spacing S) bounds the answer: the true nearest point lies
    within S/2 of a vertex of its segment, so every segment that can be nearest has
    a vertex within d_min_vertex + S/2. Candidate completeness is certified per point;
    uncertified points are re-queried with more neighbours.
    """
    tree: cKDTree
    parent: np.ndarray
    a: np.ndarray
    b: np.ndarray
    spacing_km: float
    n_segments: int
    n_seam_removed: int
    n_vertices: int
    stats: dict = field(default_factory=lambda: {'points': 0, 'refined_points': 0, 'max_k': 0})

    @classmethod
    def build(cls, land, spacing_km=COAST_VERTEX_SPACING_KM):
        segments, removed = coastline_segments(land)
        xyz, parent, a, b = densify_arcs(segments, spacing_km)
        return cls(cKDTree(xyz), parent, a, b, spacing_km, len(segments), removed, len(xyz))

    def distance_km(self, lon, lat, chunk=100_000):
        out = np.empty(len(lon))
        for start in range(0, len(lon), chunk):
            sl = slice(start, start + chunk)
            out[sl] = self._distance(unit_xyz(lon[sl], lat[sl]))
        self.stats['points'] += len(lon)
        return out

    def _distance(self, p, k=COAST_K):
        result = np.full(len(p), np.nan)
        todo = np.arange(len(p))
        while len(todo):
            kk = min(k, len(self.parent))
            chord, idx = self.tree.query(p[todo], k=kk, workers=-1)
            chord, idx = chord.reshape(len(todo), kk), idx.reshape(len(todo), kk)
            bound = 2 * np.sin(np.minimum(np.arcsin(np.clip(chord[:, 0] / 2, 0, 1)) + self.spacing_km / (2 * R_KM),
                                          np.pi / 2))
            complete = (chord[:, -1] > bound) | (kk == len(self.parent))
            seg = self.parent[idx]
            d = arc_distance_rad(p[todo][:, None, :], self.a[seg], self.b[seg]).min(axis=1)
            result[todo[complete]] = d[complete] * R_KM
            if k > COAST_K:
                self.stats['refined_points'] += int(len(todo))
            self.stats['max_k'] = max(self.stats['max_k'], kk)
            todo = todo[~complete]
            k *= 4
        return result


# ---------------------------------------------------------------------------
# Land mask clipping
# ---------------------------------------------------------------------------

def load_gshhg(directory=GSHHG_DIR):
    """Polygons per GSHHG level, with any invalid polygon repaired by make_valid."""
    levels, repaired = {}, []
    for level in GSHHG_LEVELS:
        frame = pyogrio.read_dataframe(directory / 'GSHHS_shp' / 'f' / f'GSHHS_f_L{level}.shp')
        geoms = frame.geometry.to_numpy()
        bad = ~shapely.is_valid(geoms)
        for i in np.flatnonzero(bad):
            repaired.append({'level': level, 'id': str(frame.id.iloc[i]),
                             'reason': shapely.is_valid_reason(geoms[i])})
            geoms[i] = shapely.make_valid(geoms[i])
        levels[level] = polygons_only(geoms)
    return levels, repaired


def polygons_only(geoms):
    parts = shapely.get_parts(np.asarray(geoms, dtype=object))
    parts = parts[np.isin(shapely.get_type_id(parts), [3])]
    return parts[shapely.area(parts) > 0]


def clip_levels(levels, lon0, lat0, lon1, lat1):
    box = shapely.box(lon0, lat0, lon1, lat1)
    out = {}
    for level, geoms in levels.items():
        if len(geoms) == 0:
            out[level] = geoms
            continue
        b = shapely.bounds(geoms)
        keep = (b[:, 0] < lon1) & (b[:, 2] > lon0) & (b[:, 1] < lat1) & (b[:, 3] > lat0)
        out[level] = polygons_only(shapely.intersection(geoms[keep], box)) if keep.any() else geoms[:0]
    return out


def land_geometry(levels):
    """(L1 − L2) ∪ (L3 − L4) for already clipped level polygons."""
    def union(level):
        return shapely.union_all(levels[level]) if len(levels[level]) else shapely.Polygon()
    return shapely.union(shapely.difference(union(1), union(2)), shapely.difference(union(3), union(4)))


# ---------------------------------------------------------------------------
# Pixel records
# ---------------------------------------------------------------------------

@dataclass
class Records:
    """60″ pixel quadrature records (a GPW cell's pixels are always emitted together)."""
    r: list = field(default_factory=list)
    c: list = field(default_factory=list)
    frac: list = field(default_factory=list)
    lat: list = field(default_factory=list)
    lon: list = field(default_factory=list)
    centre_land: list = field(default_factory=list)
    size: int = 0

    def add_full(self, r, c):
        self.r.append(r.ravel())
        self.c.append(c.ravel())
        n = r.size
        self.frac.append(np.ones(n))
        self.lat.append(-90 + (r.ravel() + .5) / PIX)
        self.lon.append(-180 + (c.ravel() + .5) / PIX)
        self.centre_land.append(np.ones(n, bool))
        self.size += n


def rep_points(pieces):
    """Area centroid when inside the piece, else a deterministic interior point."""
    cent = shapely.centroid(pieces)
    x, y = shapely.get_x(cent), shapely.get_y(cent)
    inside = shapely.contains_xy(pieces, x, y)
    if not inside.all():
        pos = shapely.point_on_surface(pieces[~inside])
        x[~inside], y[~inside] = shapely.get_x(pos), shapely.get_y(pos)
    return x, y


def cell_records(geom, r0, c0, n=CELL_PIX, records=None):
    """Clip an n × n block of 60″ pixels (lower-left pixel r0, c0) against land geometry."""
    records = records if records is not None else Records()
    rr, cc = np.meshgrid(np.arange(r0, r0 + n), np.arange(c0, c0 + n), indexing='ij')
    rr, cc = rr.ravel(), cc.ravel()
    x0, y0 = -180 + cc / PIX, -90 + rr / PIX
    boxes = shapely.box(x0, y0, x0 + 1 / PIX, y0 + 1 / PIX)
    pieces = shapely.intersection(boxes, geom)
    frac = shapely.area(pieces) * PIX * PIX
    full = frac >= 1 - FULL_TOL
    part = (frac > 0) & ~full
    if full.any():
        records.add_full(rr[full], cc[full])
    if part.any():
        pp = pieces[part]
        px, py = rep_points(pp)
        records.r.append(rr[part])
        records.c.append(cc[part])
        records.frac.append(frac[part])
        records.lat.append(py)
        records.lon.append(px)
        records.centre_land.append(shapely.intersects_xy(geom, x0[part] + .5 / PIX, y0[part] + .5 / PIX))
        records.size += int(part.sum())
    return records


# ---------------------------------------------------------------------------
# Accumulation per country
# ---------------------------------------------------------------------------

@dataclass
class Sums:
    n: int

    def __post_init__(self):
        z = lambda: np.zeros(self.n)  # noqa: E731
        self.area, self.mixed_area, self.north_area, self.abs_lat = z(), z(), z(), z()
        self.elev_area, self.elev_sum, self.neg_land_area, self.water_centred_area = z(), z(), z(), z()
        self.dist = {b: [z(), z()] for b in DISTANCE_BLOCKS}  # block size -> [area, area*distance]
        self.dist60s = [z(), z()]  # exact 60″ pixel quadrature, DISTANCE_60S_CHECK countries only
        self.koppen = np.zeros((self.n, len(CLASSES) + 1))
        self.pixels = np.zeros(self.n, dtype=np.int64)
        self.cell_land_area = np.zeros((180 * 4, 360 * 4))


def koppen_group_index(codes):
    table = np.full(256, len(CLASSES), dtype=np.int8)
    for code, letter in KOPPEN_GROUPS.items():
        table[code] = CLASSES.index(letter)
    return table[np.asarray(codes, dtype=np.int64)]


def accumulate(records, sums, country_of_cell, elevation, koppen_group, coast, area_rows, check_idx=()):
    """Integrate one batch of records into per-country sums.

    ``country_of_cell``: ascending-row (720, 1440) country index, -1 outside targets.
    ``elevation(r, c)`` and ``koppen_group`` (ascending (360, 720) group index) are lookups.
    """
    if records.size == 0:
        return
    r, c = np.concatenate(records.r), np.concatenate(records.c)
    frac, lat, lon = np.concatenate(records.frac), np.concatenate(records.lat), np.concatenate(records.lon)
    centre = np.concatenate(records.centre_land)
    idx = country_of_cell[r // CELL_PIX, c // CELL_PIX]
    keep = idx >= 0
    r, c, frac, lat, lon, centre, idx = r[keep], c[keep], frac[keep], lat[keep], lon[keep], centre[keep], idx[keep]
    area = frac * area_rows[r]
    n = sums.n

    def add(values):
        return np.bincount(idx, weights=values, minlength=n)

    sums.area += add(area)
    sums.pixels += np.bincount(idx, minlength=n)
    sums.mixed_area += add(area * (frac < 1))
    north = (-90 + (r + .5) / PIX) >= 0
    sums.north_area += add(area * north)
    sums.abs_lat += add(area * np.abs(lat))
    z = elevation(r, c)
    valid = centre & np.isfinite(z)
    sums.elev_area += add(area * valid)
    sums.elev_sum += add(np.where(valid, area * z, 0))
    sums.neg_land_area += add(area * (valid & (z < 0)))
    sums.water_centred_area += add(area * ~centre)
    group = koppen_group[r // KOPPEN_PIX, c // KOPPEN_PIX]
    width = len(CLASSES) + 1
    sums.koppen += np.bincount(idx * width + group, weights=area, minlength=n * width).reshape(n, width)
    cells = sums.cell_land_area.shape
    sums.cell_land_area += np.bincount((r // CELL_PIX) * cells[1] + c // CELL_PIX, weights=area,
                                       minlength=cells[0] * cells[1]).reshape(cells)
    # Distance quadrature: one evaluation per land-area centroid of each b × b pixel block.
    # Blocks nest inside GPW cells, and a cell's pixels always arrive in one batch.
    xyz = unit_xyz(lon, lat) * area[:, None]
    for b in DISTANCE_BLOCKS:
        key = (r // b).astype(np.int64) * (N_COLS // b) + c // b
        uniq, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        block_area = np.bincount(inverse, weights=area)
        centroid = np.column_stack([np.bincount(inverse, weights=xyz[:, k]) for k in range(3)])
        centroid /= np.linalg.norm(centroid, axis=1, keepdims=True)
        blon = np.degrees(np.arctan2(centroid[:, 1], centroid[:, 0]))
        blat = np.degrees(np.arcsin(np.clip(centroid[:, 2], -1, 1)))
        bidx = idx[first]
        sums.dist[b][0] += np.bincount(bidx, weights=block_area, minlength=n)
        sums.dist[b][1] += np.bincount(bidx, weights=block_area * coast.distance_km(blon, blat), minlength=n)
    check = np.isin(idx, check_idx)
    if check.any():
        sums.dist60s[0] += add(np.where(check, area, 0))
        sums.dist60s[1] += np.bincount(idx[check], weights=area[check] * coast.distance_km(lon[check], lat[check]),
                                       minlength=n)


# ---------------------------------------------------------------------------
# Recursion over the GPW cell grid
# ---------------------------------------------------------------------------

def walk(levels, country_of_cell, emit, r0=0, r1=720, c0=0, c1=1440, stats=None):
    """Depth-first quadtree over target GPW cells; emits full blocks or clipped cells.

    ``emit(kind, r0, r1, c0, c1, geom)`` receives ``kind='full'`` for a block wholly
    covered by L1 land with no lake levels present, or ``kind='cell'`` with the
    clipped land geometry of one partially covered cell.
    """
    stats = stats if stats is not None else {'full_blocks': 0, 'cells': 0, 'water_blocks': 0}
    if not (country_of_cell[r0:r1, c0:c1] >= 0).any():
        return stats
    lon0, lon1, lat0, lat1 = -180 + c0 / 4, -180 + c1 / 4, -90 + r0 / 4, -90 + r1 / 4
    clipped = clip_levels(levels, lon0, lat0, lon1, lat1)
    if len(clipped[1]) == 0 and len(clipped[3]) == 0:
        stats['water_blocks'] += 1
        return stats
    box_area = (lon1 - lon0) * (lat1 - lat0)
    if (len(clipped[2]) == len(clipped[3]) == len(clipped[4]) == 0
            and shapely.area(clipped[1]).sum() >= box_area * (1 - FULL_TOL)):
        stats['full_blocks'] += 1
        emit('full', r0, r1, c0, c1, None)
        return stats
    if r1 - r0 == 1 and c1 - c0 == 1:
        stats['cells'] += 1
        emit('cell', r0, r1, c0, c1, land_geometry(clipped))
        return stats
    rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
    for a, b in ([(r0, rm), (rm, r1)] if r1 - r0 > 1 else [(r0, r1)]):
        for c, d in ([(c0, cm), (cm, c1)] if c1 - c0 > 1 else [(c0, c1)]):
            walk(clipped, country_of_cell, emit, a, b, c, d, stats)
    return stats


# ---------------------------------------------------------------------------
# Feature rules and gates
# ---------------------------------------------------------------------------

def dominant_climate(areas, min_classified=GATES['climate_min_classified']):
    """Area-dominant A–E with coverage and robust-winner gates (areas: A..E, unclassified)."""
    classified = areas[:len(CLASSES)]
    total = areas.sum()
    if total <= 0:
        return None, 'no terrestrial area'
    coverage = classified.sum() / total
    if coverage < min_classified:
        return None, f'classified coverage {coverage:.4f} < {min_classified}'
    order = sorted(range(len(CLASSES)), key=lambda i: (-classified[i], i))  # ties: alphabetical
    a1, a2 = classified[order[0]], classified[order[1]]
    unclassified = areas[len(CLASSES)]
    if unclassified > 0 and not a1 - a2 > unclassified:
        return None, 'winner margin does not exceed unclassified area'
    return CLASSES[order[0]], ''


def features_from_sums(sums, codes):
    """Feature and QA rows; failed gates give missing features with explicit reasons."""
    rows = []
    for i, code in enumerate(codes):
        area = sums.area[i]
        reasons = {}
        row = {'iso3': code, 'terrestrial_area_km2': area, 'n_pixels_60s': int(sums.pixels[i])}
        if area <= 0:
            for f in FEATURES[:5]:
                row[f] = np.nan
                reasons[f] = 'zero terrestrial area'
            row['missing_reasons'] = json.dumps(reasons)
            rows.append(row)
            continue
        row['abs_latitude'] = sums.abs_lat[i] / area
        row['hemisphere'] = 'N' if sums.north_area[i] >= area - sums.north_area[i] else 'S'
        row['north_area_fraction'] = sums.north_area[i] / area
        cov_e = sums.elev_area[i] / area
        row['elevation_coverage'] = cov_e
        row['elevation_if_computed'] = sums.elev_sum[i] / sums.elev_area[i] if sums.elev_area[i] > 0 else np.nan
        row['elevation'] = row['elevation_if_computed'] if cov_e >= GATES['numeric_min_coverage'] else np.nan
        if cov_e < GATES['numeric_min_coverage']:
            reasons['elevation'] = f'coverage {cov_e:.4f} < {GATES["numeric_min_coverage"]}'
        primary, coarse = sums.dist[DISTANCE_BLOCKS[0]], sums.dist[DISTANCE_BLOCKS[1]]
        cov_d = primary[0][i] / area
        row['continentality_coverage'] = cov_d
        row['continentality'] = primary[1][i] / primary[0][i]
        row['continentality_5min'] = coarse[1][i] / coarse[0][i]
        row['continentality_60s_check'] = (sums.dist60s[1][i] / sums.dist60s[0][i]
                                           if sums.dist60s[0][i] > 0 else np.nan)
        if cov_d < GATES['numeric_min_coverage']:
            reasons['continentality'] = f'coverage {cov_d:.4f}'
            row['continentality'] = np.nan
        climate, why = dominant_climate(sums.koppen[i])
        row['climate_zone'] = climate if climate is not None else np.nan
        if why:
            reasons['climate_zone'] = why
        k = sums.koppen[i]
        for j, letter in enumerate(CLASSES):
            row[f'koppen_area_share_{letter}'] = k[j] / area
        row['koppen_unclassified_share'] = k[len(CLASSES)] / area
        ordered = np.sort(k[:len(CLASSES)])[::-1]
        row['koppen_winner_margin_km2'] = ordered[0] - ordered[1]
        row['koppen_unclassified_km2'] = k[len(CLASSES)]
        row['mixed_pixel_area_km2'] = sums.mixed_area[i]
        row['elevation_water_centred_land_km2'] = sums.water_centred_area[i]
        row['negative_land_area_km2'] = sums.neg_land_area[i]
        row['elevation_missing_fraction'] = sums.water_centred_area[i] / area
        row['elevation_missing_cause'] = ('terrestrial support inside 60″ ETOPO pixels whose centre is water '
                                          '(spec §4 elevation rule; coastline/island resolution)'
                                          if sums.water_centred_area[i] > 0 else '')
        row['climate_missing_cause'] = ('terrestrial support inside Köppen 0.5° pixels coded 0/unmapped '
                                        '(source raster land/sea resolution)'
                                        if k[len(CLASSES)] > 0 else '')
        row['missing_reasons'] = json.dumps(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def emit_into(records, kind, r0, r1, c0, c1, geom):
    if kind == 'full':
        rr, cc = np.meshgrid(np.arange(r0 * CELL_PIX, r1 * CELL_PIX), np.arange(c0 * CELL_PIX, c1 * CELL_PIX),
                             indexing='ij')
        records.add_full(rr, cc)
    elif not shapely.is_empty(geom):
        cell_records(geom, r0 * CELL_PIX, c0 * CELL_PIX, records=records)


def walk_globe(levels, country_of_cell, emit, coast):
    stats = {'full_blocks': 0, 'cells': 0, 'water_blocks': 0}
    started = time.time()
    for r0 in range(0, 720, BLOCK_CELLS):
        print(f'[{time.time() - started:7.0f}s] rows {r0}/720 {stats} coast {coast.stats}', file=sys.stderr, flush=True)
        for c0 in range(0, 1440, BLOCK_CELLS):
            walk(levels, country_of_cell, emit, r0, min(r0 + BLOCK_CELLS, 720), c0, min(c0 + BLOCK_CELLS, 1440), stats)
    return stats


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def gpw_country_grid(codes):
    iso, lats, lons = load_national_grid()
    lookup = {code: i for i, code in enumerate(codes)}
    idx = np.vectorize(lambda v: lookup.get(v, -1), otypes=[np.int16])(iso)
    cell_area = row_band_area_km2(lats - .125, lats + .125, .25)
    footprint = np.bincount(idx[idx >= 0], weights=np.broadcast_to(cell_area[:, None], idx.shape)[idx >= 0],
                            minlength=len(codes))
    cells = np.bincount(idx[idx >= 0].astype(int), minlength=len(codes))
    return idx[::-1].copy(), footprint, cells  # ascending latitude rows


def station_continentality(coast, features):
    """Exact spherical distance at V1 station points, separating algorithm from support."""
    city = pd.read_parquet(m0.DEFAULT_FEATURES_PATH)
    city = city[city.Country.isin(features.Country)]
    d = coast.distance_km(city.Longitude.to_numpy(float), city.Latitude.to_numpy(float))
    return (city.assign(coast_km_exact=d).groupby('Country', observed=True)
            .agg(continentality_station_v1=('coast_km', 'mean'),
                 continentality_station_exact_arc=('coast_km_exact', 'mean')))


def build(out=OUT, batch_pixels=6_000_000):
    code_hash = sha256(Path(__file__))  # pinned before any work; never edit the file mid-run
    design = m0.m0_complete_design(*m0.load_inputs())
    table = pd.read_csv(OUTPUT_DIR_TABLE)
    assert design.Country.tolist() == table.Country.tolist()
    codes = table.iso3.tolist()
    country_of_cell, footprint_km2, n_cells = gpw_country_grid(codes)
    levels, repaired = load_gshhg()
    land_ne = shapely.union_all(pyogrio.read_dataframe(f'zip://{LAND_ZIP_PATH}').geometry.to_numpy())
    coast = Coast.build(land_ne)
    with xr.open_dataset(ETOPO_PATH) as ds:
        lat = ds.lat.to_numpy()
        assert np.allclose(lat[:2], [-90 + .5 / PIX, -90 + 1.5 / PIX]) and len(lat) == N_ROWS
        assert np.allclose(ds.lon.to_numpy()[:2], [-180 + .5 / PIX, -180 + 1.5 / PIX])
        z = ds[ETOPO_VAR].to_numpy()
        etopo_attrs = {k: str(v) for k, v in ds[ETOPO_VAR].attrs.items()}
    with xr.open_dataset(KOPPEN_PATH) as ds:
        klat = ds.lat.to_numpy()
        assert klat[0] > klat[-1] and np.isclose(klat[0], 89.75) and np.isclose(ds.lon.to_numpy()[0], -179.75)
        koppen_group = koppen_group_index(ds[KOPPEN_VAR].to_numpy())[::-1].copy()
    area_rows = pixel_area_rows(1)
    check_idx = [codes.index(c) for c in DISTANCE_60S_CHECK if c in codes]
    sums = Sums(len(codes))
    batch = [Records()]

    def elevation(r, c):
        return z[r, c].astype(float)

    def flush():
        accumulate(batch[0], sums, country_of_cell, elevation, koppen_group, coast, area_rows, check_idx)
        batch[0] = Records()

    def emit(kind, r0, r1, c0, c1, geom):
        emit_into(batch[0], kind, r0, r1, c0, c1, geom)
        if batch[0].size >= batch_pixels:
            flush()

    stats = walk_globe(levels, country_of_cell, emit, coast)
    flush()
    qa = features_from_sums(sums, codes)
    # Deterministic refinement: any country whose 3′ and 5′ distance quadratures disagree
    # beyond tolerance, and has no 60″ value yet, gets an exact 60″ pixel pass.
    tol = np.maximum(DISTANCE_TOL['absolute_km'], DISTANCE_TOL['relative'] * qa.continentality)
    refine = [i for i in np.flatnonzero((qa.continentality - qa.continentality_5min).abs().to_numpy() > tol)
              if i not in check_idx]
    if refine:
        subset = np.where(np.isin(country_of_cell, refine), country_of_cell, -1).astype(np.int16)
        extra, extra_batch = Sums(len(codes)), [Records()]

        def emit_refine(kind, r0, r1, c0, c1, geom):
            emit_into(extra_batch[0], kind, r0, r1, c0, c1, geom)

        walk_globe(levels, subset, emit_refine, coast)
        accumulate(extra_batch[0], extra, subset, elevation, koppen_group, coast, area_rows, refine)
        for i in refine:
            qa.loc[i, 'continentality_60s_check'] = extra.dist60s[1][i] / extra.dist60s[0][i]
    qa['continentality_60s_pass'] = ['representative' if i in check_idx else 'refinement' if i in refine else ''
                                     for i in range(len(codes))]
    qa.insert(1, 'Country', table.Country)
    qa['gpw_footprint_area_km2'] = footprint_km2
    qa['gpw_cells'] = n_cells
    qa['removed_water_area_km2'] = footprint_km2 - qa.terrestrial_area_km2
    qa['spatial_block'] = design.spatial_block.to_numpy()
    for f in FEATURES[:5]:
        qa[f'v1_{f}'] = design[f].to_numpy()
    stations = station_continentality(coast, design).reindex(table.Country)
    qa['continentality_station_v1'] = stations.continentality_station_v1.to_numpy()
    qa['continentality_station_exact_arc'] = stations.continentality_station_exact_arc.to_numpy()
    def within(a, ref):
        return (a - ref).abs() <= np.maximum(DISTANCE_TOL['absolute_km'], DISTANCE_TOL['relative'] * ref)
    qa['continentality_diff_3min_vs_5min_km'] = (qa.continentality - qa.continentality_5min).abs()
    qa['continentality_diff_3min_vs_60s_km'] = (qa.continentality - qa.continentality_60s_check).abs()
    three_ok_vs_60s = within(qa.continentality, qa.continentality_60s_check)
    qa['continentality_quadrature'] = np.where(qa.continentality_60s_check.notna() & ~three_ok_vs_60s, '60s', '3min')
    qa['continentality_3min'] = qa.continentality
    qa.loc[qa.continentality_quadrature == '60s', 'continentality'] = qa.continentality_60s_check
    qa['continentality_quadrature_ok'] = np.where(qa.continentality_60s_check.notna(), True,
                                                  within(qa.continentality_5min, qa.continentality))
    qa['climate_zone_changed'] = qa.climate_zone != qa.v1_climate_zone
    qa['hemisphere_changed'] = qa.hemisphere != qa.v1_hemisphere
    for f in ['abs_latitude', 'elevation', 'continentality']:
        qa[f'{f}_change'] = qa[f] - qa[f'v1_{f}']
    features = qa[['iso3', 'Country', *FEATURES]].copy()
    all_pass = bool(features[list(FEATURES)].notna().all().all() and qa.continentality_quadrature_ok.all())

    out.mkdir(exist_ok=True)
    features.to_csv(out / 'm1a_geography_features.csv', index=False)
    qa.to_csv(out / 'm1a_geography_qa.csv', index=False)
    np.savez_compressed(out / 'm1a_cell_land_area_km2.npz', cell_land_area_km2=sums.cell_land_area.astype(np.float64))
    manifest = manifest_payload(stats, repaired, coast, etopo_attrs, all_pass, qa, code_hash)
    (out / 'm1a_measurement_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return features, qa, manifest



def manifest_payload(stats, repaired, coast, etopo_attrs, all_pass, qa, code_hash):
    def git(*args):
        return subprocess.run(['git', *args], cwd=m0.ROOT, capture_output=True, text=True).stdout.strip()
    with zipfile.ZipFile(GSHHG_ZIP) as archive:
        members = {name: sha256_bytes(archive.read(name)) for name in sorted(archive.namelist())
                   if name.startswith('GSHHS_shp/f/GSHHS_f_L') and name[-5] in '1234'}
    extracted = {p.name: sha256(p) for p in sorted((GSHHG_DIR / 'GSHHS_shp' / 'f').glob('GSHHS_f_L[1-4].*'))}
    if {Path(k).name: v for k, v in members.items()} != extracted:
        raise ValueError('extracted GSHHG members differ from the pinned archive')
    zip_hash = sha256(GSHHG_ZIP)
    if zip_hash != GSHHG_ZIP_SHA256:
        raise ValueError('GSHHG archive hash changed')
    return {
        'spec': {'path': 'research/model_v2/M1A_MEASUREMENT_SPEC.md', 'sha256': sha256(SPEC_PATH),
                 'evaluation_contract_commit': git('log', '-1', '--format=%H', '--', 'research/model_v2/M1A_EVALUATION_CONTRACT.md')},
        'sources': {
            'gshhg': {'release': 'GSHHG 2.3.7 shapefile, full resolution, levels 1-4', 'url': GSHHG_URL,
                      'archive_sha256': zip_hash, 'archive_bytes': GSHHG_ZIP.stat().st_size,
                      'member_sha256': members, 'crs': 'EPSG:4326 (WGS84 geographic)',
                      'license': 'LGPL-3.0 (LICENSE.TXT in archive)',
                      'repaired_invalid_polygons': repaired},
            'etopo': {'path': str(ETOPO_PATH.relative_to(m0.ROOT)), 'sha256': sha256(ETOPO_PATH),
                      'variable': ETOPO_VAR, 'attrs': etopo_attrs,
                      'registration': 'pixel-is-area, 60 arc-second, centres at (k+0.5)/60 deg, ascending latitude'},
            'koppen': {'path': str(KOPPEN_PATH.relative_to(m0.ROOT)), 'sha256': sha256(KOPPEN_PATH),
                       'variable': KOPPEN_VAR, 'registration': '0.5 deg pixels, centres at +-0.25, descending latitude',
                       'groups': {str(k): v for k, v in KOPPEN_GROUPS.items()}, 'unclassified': 'code 0 or unmapped'},
            'natural_earth_land': {'path': str(LAND_ZIP_PATH.relative_to(m0.ROOT)), 'sha256': sha256(LAND_ZIP_PATH),
                                   'scale': '1:110 million'},
            'gpw_national_identifier': {'path': str(GPW_PATH.relative_to(m0.ROOT)), 'sha256': sha256(GPW_PATH),
                                        'band': 11, 'lookup_sha256': sha256(GPW_NATID_LOOKUP_PATH),
                                        'registration': '0.25 deg cells, centres at +-0.125, descending latitude'},
            'territory_snapshot_sha256': sha256(OUT / 'territory_gpw_footprints.npz'),
            'm0_countries_sha256': sha256(OUTPUT_DIR_TABLE),
        },
        'mask_hierarchy': 'terrestrial = (L1 - L2) U (L3 - L4); L2 includes river-lakes (water); Antarctica levels 5-6 not used (no sample country)',
        'support': 'union of GPW band-11 cells with the country ISO3, intersected with the terrestrial mask; unassigned GPW space excluded',
        'area_convention': {'radius_km': R_KM, 'pixel_area': 'R^2 dlon (sin lat1 - sin lat0)',
                            'partial_pixel': 'exact lon/lat polygon clip fraction times the pixel spherical area'},
        'quadrature': {'lattice': 'ETOPO 60 arc-second pixels for area, latitude, hemisphere, elevation and Koppen',
                       'distance': f'one evaluation per land-area centroid of {DISTANCE_BLOCKS[0]}x{DISTANCE_BLOCKS[0]}-pixel '
                                   f'(3 arc-minute) blocks; {DISTANCE_BLOCKS[1]}x{DISTANCE_BLOCKS[1]} (5 arc-minute) blocks '
                                   f'globally and exact 60 arc-second pixels for {list(DISTANCE_60S_CHECK)} as the resolution check',
                       'representative_point': 'clipped-piece area centroid if inside the piece, else shapely point_on_surface',
                       'full_pixel_tolerance': FULL_TOL},
        'features': {
            'abs_latitude': 'sum(area * |lat of representative point|) / area',
            'hemisphere': 'N if pixel-centre-north area >= south area (pixel edges lie on the equator)',
            'elevation': 'ETOPO surface z weighted by pixel land area; valid only if the pixel centre is on terrestrial support; negatives retained',
            'continentality': 'area mean of min great-circle distance to Natural Earth 110m land boundary; seam edges removed; densified great-circle arcs',
            'climate_zone': 'area-dominant Koppen major group over land area per 0.5 deg pixel; alphabetical tie order',
            'spatial_block': 'OWID continent copied unchanged',
        },
        'coast': {'n_segments': coast.n_segments, 'seam_segments_removed': coast.n_seam_removed,
                  'candidate_vertex_spacing_km': coast.spacing_km, 'n_candidate_vertices': coast.n_vertices,
                  'distance': 'exact minimum angular distance to minor great-circle arcs, certified candidate set',
                  'query_stats': coast.stats},
        'gates': GATES, 'distance_quadrature_tolerance': DISTANCE_TOL,
        'walk_stats': stats,
        'all_features_pass_gates': all_pass,
        'gate_failures': qa.loc[qa.missing_reasons != '{}', ['iso3', 'missing_reasons']].to_dict('records'),
        'quadrature_failures': qa.loc[~qa.continentality_quadrature_ok, 'iso3'].tolist(),
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'shapely': shapely.__version__, 'geos': shapely.geos_version_string,
                     'scipy': scipy.__version__, 'pyogrio': pyogrio.__version__, 'xarray': xr.__version__},
        'code_sha256': code_hash,
    }


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == '__main__':
    features, qa, manifest = build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    print(json.dumps({k: manifest[k] for k in ['walk_stats', 'all_features_pass_gates', 'gate_failures',
                                               'quadrature_failures']}, indent=2))
