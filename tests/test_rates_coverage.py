import numpy as np
import pytest

from nutomo.tesseroids import TesseroidGrid, DEFAULT_R_EDGES_KM
from nutomo.raytrace import build_ray_table
from nutomo.density import DensityModel, shell_mean_prem, blob_anomaly
from nutomo.detectors import Detector, Binning, response_kernels
from nutomo.rates import RateCalculator, GaisserLikeFlux, CrossSection
from nutomo.probability import OscParams
from nutomo import coverage as cov, likelihood as lk


@pytest.fixture(scope="module")
def small():
    g = TesseroidGrid(DEFAULT_R_EDGES_KM, 30, 30)
    dets = [Detector("A", 42.8, 6.03, 2.45, mass_mton=7, n_e_bins=5, e_oversample=1, sigma_cosz=0.15, sigma_az_deg=20),
            Detector("B", -89.99, 0.0, 2.0, mass_mton=5, n_e_bins=5, e_oversample=1)]
    b = Binning(np.linspace(-1, 0, 4), np.linspace(0, 360, 5))
    rt = build_ray_table(g, [d.site() for d in dets], b.cosz_centers, b.az_centers)
    rc = RateCalculator(rt, dets, b, GaisserLikeFlux(), CrossSection())
    m = DensityModel(g, radial=shell_mean_prem(g), lateral=blob_anomaly(g, 0, -160, 3480, 4500, 40, 0.02))
    return g, dets, b, rt, rc, m, OscParams()


def test_response_kernels_conserve_or_lose(small):
    g, dets, b, rt, rc, m, pars = small
    K = response_kernels(dets[0], b)
    for k in ("E", "cosz", "az"):
        assert np.all(K[k].sum(0) <= 1 + 1e-9)
    assert np.allclose(K["az"].sum(0), 1)   # periodic: nothing lost


def test_rate_jacobian_vs_fd(small):
    g, dets, b, rt, rc, m, pars = small
    i = 5
    rr = rc.ray_rates(i, m, pars, with_jacobian=True)
    assert np.all(rr.counts >= 0) and rr.counts.sum() > 0
    c = rr.cells[len(rr.cells) // 2]
    d = np.zeros(g.n_cells); d[c] = 1e-3
    up = rc.ray_rates(i, DensityModel(g, radial=m.radial, lateral=m.lateral + d), pars).counts
    dn = rc.ray_rates(i, DensityModel(g, radial=m.radial, lateral=m.lateral - d), pars).counts
    fd = (up - dn) / 2e-3
    j = rr.dcounts_dfrac[:, :, list(rr.cells).index(c)]
    assert np.max(np.abs(fd - j)) < 1e-6 * max(1.0, np.max(np.abs(fd)))


def test_fisher_matches_brute_force(small):
    g, dets, b, rt, rc, m, pars = small
    acc, res = cov.assemble_fisher(rc, g, m, pars, {}, ["dm31"], ["flux_norm"], True, log=lambda *a: None)
    N = lk.concat_reco(res)
    w = np.where(N > 1e-3, 1 / np.maximum(N, 1e-3), 0)
    active = acc.active_cells
    test = active[[0, len(active) // 2, -1]]
    eps = 1e-3
    def reco(dl):
        return lk.concat_reco(rc.compute(DensityModel(g, radial=m.radial, lateral=m.lateral + dl), pars))
    gv = []
    for c in test:
        d = np.zeros(g.n_cells); d[c] = eps
        gv.append((reco(d) - reco(-d)) / (2 * eps))
    gv = np.array(gv)
    idx = [int(np.flatnonzero(active == c)[0]) for c in test]
    F_bf = gv @ (w[:, None] * gv.T)
    assert np.allclose(F_bf, acc.F_cc[np.ix_(idx, idx)], rtol=1e-4, atol=1e-8 * np.abs(F_bf).max())
    # radial-scale column from the Jacobian vs direct finite difference
    k = 6; h = 1e-3
    rs_u = np.ones(g.n_r); rs_u[k] += h; rs_d = np.ones(g.n_r); rs_d[k] -= h
    gr = (lk.concat_reco(rc.compute(m.with_radial_scale(rs_u), pars))
          - lk.concat_reco(rc.compute(m.with_radial_scale(rs_d), pars))) / (2 * h)
    col = list(acc.nuisance_names).index(f"radial_scale_{k}")
    assert np.isclose(gr @ (w * gr), acc.F_nn[col, col], rtol=1e-4)
    # derived quantities are finite and ordered sensibly
    t = m.lateral[active] / 0.02
    s0 = acc.template_sensitivity(t, marginalize_nuisance=False)
    s1 = acc.template_sensitivity(t, {"dm31": 3e-5, "flux_norm": 0.2}, True)
    assert 0 < s0 <= s1 < np.inf
    errs = acc.cell_errors(0.05, {"dm31": 3e-5, "flux_norm": 0.2})
    assert np.all(errs <= 0.05 + 1e-9)


def test_geometry_coverage_and_kernel(small):
    g, dets, b, rt, rc, m, pars = small
    cg = cov.geometry_coverage(rt, g)
    assert np.isclose(cg["path_km"].sum(), rt.length_km.sum())
    active = np.flatnonzero(cg["hits"] > 0)
    G = cov.path_length_kernel(rt, active)
    assert np.allclose(np.asarray(G.sum(1)).ravel(), rt.length_km)
    r = cov.kernel_resolution(G, 1e-3)
    assert np.all(r["diag"] >= -1e-9) and np.all(r["diag"] <= 1 + 1e-9)


def test_asimov_ts_properties():
    mu = np.array([10.0, 100.0, 1000.0])
    assert lk.asimov_delta_ts(mu, mu) == 0.0
    assert lk.asimov_delta_ts(mu, mu * 1.01) > 0
    assert np.isclose(lk.asimov_delta_ts(mu, mu * 1.01), np.sum(mu * 0.01 ** 2), rtol=0.02)


def test_subsampled_bins_aggregate(small):
    g, dets, b, rt, rc, m, pars = small
    b2 = Binning(b.cosz_edges, b.az_edges_deg, cosz_sub=2, az_sub=2)
    assert b2.cosz_sub_centers.size == 2 * b.n_cosz and np.allclose(b2.d_cosz_sub.sum(), b.d_cosz.sum())
    assert np.all(b2.parent_cosz(np.arange(2 * b.n_cosz)) == np.repeat(np.arange(b.n_cosz), 2))
    rt2 = build_ray_table(g, [d.site() for d in dets], b2.cosz_sub_centers, b2.az_sub_centers)
    rc2 = RateCalculator(rt2, dets, b2, GaisserLikeFlux(), CrossSection())
    r1 = rc.compute(m, pars)
    r2 = rc2.compute(m, pars)
    assert r2.truth[0].shape == r1.truth[0].shape
    # same bins, finer sampling: totals agree to a few percent
    assert np.isclose(r2.total(), r1.total(), rtol=0.05)
    # Fisher assembly with sub-rays matches brute force on one cell
    acc, res = cov.assemble_fisher(rc2, g, m, pars, {}, [], [], False, log=lambda *a: None)
    N = lk.concat_reco(res); w = np.where(N > 1e-3, 1 / np.maximum(N, 1e-3), 0)
    c = acc.active_cells[len(acc.active_cells) // 2]
    d = np.zeros(g.n_cells); d[c] = 1e-3
    gv = (lk.concat_reco(rc2.compute(DensityModel(g, radial=m.radial, lateral=m.lateral + d), pars))
          - lk.concat_reco(rc2.compute(DensityModel(g, radial=m.radial, lateral=m.lateral - d), pars))) / 2e-3
    j = int(np.flatnonzero(acc.active_cells == c)[0])
    assert np.isclose(gv @ (w * gv), acc.F_cc[j, j], rtol=1e-4)
