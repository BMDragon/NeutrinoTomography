import numpy as np
import pytest

from nutomo.tesseroids import TesseroidGrid, DEFAULT_R_EDGES_KM, R_EARTH_KM
from nutomo.raytrace import (trace_from_site, chord_length, xyz_to_rlatlon, build_ray_table,
                             latlon_to_xyz, source_direction)


@pytest.fixture(scope="module")
def grid():
    return TesseroidGrid(DEFAULT_R_EDGES_KM, 10, 10)


def test_segment_lengths_sum_to_chord(grid):
    rng = np.random.default_rng(0)
    for _ in range(200):
        lat, lon, depth = rng.uniform(-89.9, 89.9), rng.uniform(-180, 180), rng.uniform(0, 3)
        cz, az = rng.uniform(-1, -0.01), rng.uniform(0, 360)
        ch = trace_from_site(grid, lat, lon, depth, cz, az)
        L = chord_length(R_EARTH_KM - depth, cz)
        assert abs(ch.widths_km.sum() - L) < 1e-9 * L
        assert np.all(ch.widths_km > 0)


def test_surface_detector_vertical_chord(grid):
    ch = trace_from_site(grid, 20.0, 30.0, 0.0, -1.0, 0.0)
    assert np.isclose(ch.length_km, 2 * R_EARTH_KM)
    # symmetric radial profile of midpoints: first and last shells are the top shell
    assert grid.shell_of(ch.cell_ids[0]) == grid.n_r - 1
    assert grid.shell_of(ch.cell_ids[-1]) == grid.n_r - 1
    assert ch.r_mid_km.min() < 600.0     # passes through the inner core


def test_cells_consistent_along_segments(grid):
    rng = np.random.default_rng(1)
    for _ in range(100):
        lat, lon = rng.uniform(-89.9, 89.9), rng.uniform(-180, 180)
        cz, az = rng.uniform(-1, -0.05), rng.uniform(0, 360)
        ch = trace_from_site(grid, lat, lon, 1.0, cz, az)
        t_edges = np.concatenate([[0.0], np.cumsum(ch.widths_km)])
        for frac in (0.1, 0.5, 0.9):
            t = t_edges[:-1] + frac * ch.widths_km
            r, la, lo = xyz_to_rlatlon(ch.positions(t))
            assert np.all(grid.locate(r, la, lo) == ch.cell_ids)


def test_horizontal_chord_from_pole(grid):
    ch = trace_from_site(grid, -89.99, 0.0, 2.0, -0.05, 45.0)
    assert ch.n_segments > 1
    assert np.isclose(ch.widths_km.sum(), chord_length(R_EARTH_KM - 2.0, -0.05))


def test_source_direction_is_unit_and_upgoing():
    up = latlon_to_xyz(30.0, 40.0, 1.0)
    s = source_direction(30.0, 40.0, -0.6, 120.0)
    assert np.isclose(np.linalg.norm(s), 1.0)
    assert np.isclose(np.dot(s, up), -0.6)


def test_ray_table_roundtrip(tmp_path, grid):
    sites = [{"name": "a", "lat_deg": 10.0, "lon_deg": 20.0, "depth_km": 1.0},
             {"name": "b", "lat_deg": -50.0, "lon_deg": 100.0, "depth_km": 0.0}]
    rt = build_ray_table(grid, sites, [-0.9, -0.5], [0.0, 90.0, 180.0])
    assert rt.n_rays == 12
    path = tmp_path / "rays.npz"
    rt.save(str(path))
    rt2 = rt.load(str(path))
    assert np.all(rt2.cell_ids == rt.cell_ids)
    assert rt2.meta["grid_hash"] == grid.config_hash()
    w, c, r = rt2.segments(rt2.ray_index(1, 0, 2))
    assert np.isclose(w.sum(), chord_length(R_EARTH_KM, -0.9))
