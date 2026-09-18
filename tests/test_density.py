import numpy as np

import earth

from nutomo.tesseroids import TesseroidGrid, DEFAULT_R_EDGES_KM
from nutomo.density import (DensityModel, density_prem, shell_mean_prem, blob_anomaly,
                            electron_fraction_profile, model_from_config)


def test_prem_matches_library():
    r = np.linspace(0.0, 6371.0, 20001)
    assert np.max(np.abs(density_prem(r) - earth.density_prem(r))) < 1e-12


def test_shell_means_bracketed():
    g = TesseroidGrid(DEFAULT_R_EDGES_KM, 30, 30)
    m = shell_mean_prem(g)
    for k in range(g.n_r):
        r = np.linspace(g.r_edges[k], g.r_edges[k + 1], 1000)
        d = density_prem(r)
        assert d.min() - 1e-9 <= m[k] <= d.max() + 1e-9


def test_nested_hypotheses_are_one_code_path():
    g = TesseroidGrid(DEFAULT_R_EDGES_KM, 30, 30)
    lat = blob_anomaly(g, 0.0, 0.0, 3480.0, 4500.0, 40.0, 0.02)
    m3 = DensityModel(g, radial=shell_mean_prem(g), lateral=lat)
    null = m3.with_lateral_scale(0.0)
    ids = np.arange(g.n_cells)
    r_c, _, _ = g.centers(ids)
    rho3 = m3.densities(ids, r_c)
    rho0 = null.densities(ids, r_c)
    inside = lat != 0
    assert np.allclose(rho3[~inside], rho0[~inside])
    assert np.allclose(rho3[inside] / rho0[inside], 1.02)
    # laterally averaged model has the same shell means as the 3D one
    la = m3.laterally_averaged()
    vol = g.volumes()
    for k in range(g.n_r):
        sel = g.shell_of(ids) == k
        assert np.isclose(np.sum(vol[sel] * rho3[sel]) / vol[sel].sum(),
                          la.densities(ids[sel], r_c[sel])[0])


def test_radial_scale_and_ye():
    g = TesseroidGrid(DEFAULT_R_EDGES_KM, 30, 30)
    rs = np.ones(g.n_r); rs[3] = 1.1
    m = DensityModel(g, radial="prem", radial_scale=rs)
    ids = np.arange(g.n_cells); r_c, _, _ = g.centers(ids)
    ratio = m.densities(ids, r_c) / density_prem(r_c)
    assert np.allclose(ratio[g.shell_of(ids) == 3], 1.1) and np.allclose(ratio[g.shell_of(ids) != 3], 1.0)
    ye = electron_fraction_profile(np.array([1000.0, 5000.0]))
    assert ye[0] < ye[1]
    assert np.all(electron_fraction_profile(np.array([1000.0, 5000.0]), "constant") == 0.5)


def test_model_from_config():
    g = TesseroidGrid(DEFAULT_R_EDGES_KM, 30, 30)
    m = model_from_config(g, {"radial": "shell_mean_prem",
                              "anomalies": [{"kind": "blob", "lat_deg": 0, "lon_deg": 0, "r_min_km": 3480,
                                             "r_max_km": 4000, "radius_deg": 30, "amplitude": 0.01}]})
    assert m.lateral is not None and m.lateral.max() == 0.01
