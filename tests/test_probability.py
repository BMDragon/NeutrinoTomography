import numpy as np
import pytest

import earth
import slabs
import globaldefs as gd

from nutomo import probability as prob
from nutomo.probability import OscParams
from nutomo.density import electron_fraction_profile


def random_hermitian(rng, n=3, scale=1e-12):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return scale * (a + a.conj().T)


def test_unitarity_random_hamiltonians():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = rng.integers(1, 12)
        h = np.stack([random_hermitian(rng) for _ in range(n)])
        w = rng.uniform(1e10, 1e13, size=n)
        u = slabs.evolution_operator_3nu_slabs(h, w)
        assert np.allclose(u @ u.conj().T, np.eye(3), atol=1e-10)
        p = np.array(slabs.probabilities_3nu_slabs(h, w)).reshape(3, 3)
        assert np.allclose(p.sum(0), 1) and np.allclose(p.sum(1), 1)


def test_wrapper_probabilities_closure():
    E = np.logspace(0, 2, 10)
    w = np.array([1000.0, 2500.0, 3000.0])
    rho = np.array([3.3, 5.0, 11.0])
    ye = np.array([0.5, 0.5, 0.47])
    for anti in (False, True):
        P = prob.probabilities(w, rho, ye, E, OscParams(), anti)
        assert P.shape == (10, 3, 3)
        assert np.allclose(P.sum(-1), 1) and np.allclose(P.sum(-2), 1)
        assert np.all(P >= -1e-12)


def test_matched_leg_reproduces_native_prem():
    """Feeding the library's own PREM slabs through our wrapper must reproduce
    earth.probabilities_3nu_earth exactly: catches unit and convention errors."""
    pars = OscParams()
    E = np.array([3.0, 8.0, 25.0])
    for cosz in (-1.0, -0.6, -0.2):
        widths_km, dens = earth.earth_slabs(cosz, 8)
        ye = np.full_like(dens, 0.5)
        P = prob.probabilities(widths_km, dens, ye, E, pars)
        Pn = prob.probabilities_prem_native(cosz, E, pars, 8, electron_fraction=0.5)
        assert np.max(np.abs(P - Pn)) < 1e-10
        # antineutrinos
        Pa = prob.probabilities(widths_km, dens, ye, E, pars, antineutrino=True)
        Pna = prob.probabilities_prem_native(cosz, E, pars, 8, antineutrino=True, electron_fraction=0.5)
        assert np.max(np.abs(Pa - Pna)) < 1e-10


def test_vacuum_limit_matches_analytic_two_flavor_like_formula():
    # rho = 0: P_mumu must equal the vacuum expression from the library's own helper
    import hamiltonians3nu
    pars = OscParams()
    E = np.array([2.0, 5.0, 20.0])
    L = 8000.0
    P = prob.probabilities(np.array([L]), np.array([0.0]), np.array([0.5]), E, pars)
    U = hamiltonians3nu.pmns_mixing_matrix(np.sqrt(pars.s12sq), np.sqrt(pars.s23sq),
                                           np.sqrt(pars.s13sq), pars.dcp)
    for i, e in enumerate(E):
        pv = hamiltonians3nu.probabilities_3nu_vacuum_std(
            U, pars.dm21, pars.dm31, e * 1e9, L * gd.CONV_KM_TO_INV_EV)
        assert np.allclose(P[i].ravel(), np.array(pv), atol=1e-10)


def test_antineutrino_differs_in_matter_only():
    pars = OscParams()
    E = np.array([5.0])
    P0 = prob.probabilities(np.array([5000.0]), np.array([0.0]), np.array([0.5]), E, pars)
    P0a = prob.probabilities(np.array([5000.0]), np.array([0.0]), np.array([0.5]), E, pars, True)
    # in vacuum, nubar probabilities equal nu probabilities with dcp -> -dcp; P_ee is CP-even
    assert np.isclose(P0[0, 0, 0], P0a[0, 0, 0])
    Pm = prob.probabilities(np.array([5000.0]), np.array([5.0]), np.array([0.5]), E, pars)
    Pma = prob.probabilities(np.array([5000.0]), np.array([5.0]), np.array([0.5]), E, pars, True)
    assert not np.isclose(Pm[0, 0, 0], Pma[0, 0, 0], atol=1e-4)


def test_three_legs_difference_is_small_and_signed():
    """3D-null differences are O(anomaly), null vs pointwise PREM are O(grid)."""
    from nutomo.tesseroids import TesseroidGrid, DEFAULT_R_EDGES_KM
    from nutomo.raytrace import trace_from_site
    from nutomo.density import DensityModel, shell_mean_prem, blob_anomaly
    g = TesseroidGrid(DEFAULT_R_EDGES_KM, 10, 10)
    ch = trace_from_site(g, 42.8, 6.03, 2.45, -0.85, 200.0)
    E = np.logspace(np.log10(2), 2, 30)
    lat = blob_anomaly(g, 0.0, -160.0, 3480.0, 4500.0, 180.0, 0.05)   # whole-shell band
    m3 = DensityModel(g, radial=shell_mean_prem(g), lateral=lat)
    legs = {"3d": m3, "null": m3.with_lateral_scale(0), "prem": DensityModel(g, radial="prem")}
    P = {k: prob.probabilities(ch.widths_km, m.densities(ch.cell_ids, ch.r_mid_km),
                               m.electron_fraction(ch.r_mid_km), E, OscParams()) for k, m in legs.items()}
    d3 = np.abs(P["3d"] - P["null"]).max()
    dp = np.abs(P["null"] - P["prem"]).max()
    assert 0 < d3 < 0.2 and 0 < dp < 0.2
