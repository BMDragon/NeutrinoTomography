import numpy as np

from nutomo.tesseroids import TesseroidGrid, DEFAULT_R_EDGES_KM, lateral_average, R_EARTH_KM


def test_index_roundtrip_and_locate():
    g = TesseroidGrid(DEFAULT_R_EDGES_KM, 15, 20)
    ids = np.arange(g.n_cells)
    ir, ilat, ilon = g.unravel(ids)
    assert np.all(g.cell_index(ir, ilat, ilon) == ids)
    r, lat, lon = g.centers(ids)
    assert np.all(g.locate(r, lat, lon) == ids)
    # longitude wrapping and surface point
    assert g.locate(R_EARTH_KM, 10.0, 190.0) == g.locate(R_EARTH_KM, 10.0, -170.0)
    assert g.shell_of(g.locate(R_EARTH_KM, 0.0, 0.0)) == g.n_r - 1


def test_volumes_sum_to_sphere():
    g = TesseroidGrid(DEFAULT_R_EDGES_KM, 10, 10)
    assert np.isclose(g.volumes().sum(), 4 / 3 * np.pi * R_EARTH_KM ** 3, rtol=1e-12)
    assert np.isclose(g.shell_volumes().sum(), g.volumes().sum())


def test_lateral_average_recovers_shell_field():
    g = TesseroidGrid(DEFAULT_R_EDGES_KM, 30, 30)
    shell_vals = np.arange(g.n_r, dtype=float)
    f = shell_vals[g.shell_of(np.arange(g.n_cells))]
    assert np.allclose(lateral_average(g, f), shell_vals)


def test_config_hash_stable():
    a = TesseroidGrid(DEFAULT_R_EDGES_KM, 10, 10)
    b = TesseroidGrid.from_dict(a.to_dict())
    assert a.config_hash() == b.config_hash()
