import numpy as np
import pytest

from nutomo.tesseroids import TesseroidGrid, DEFAULT_R_EDGES_KM
from nutomo.raytrace import trace_from_site
from nutomo.density import DensityModel, shell_mean_prem, blob_anomaly
from nutomo import probability as prob, jacobian as jac
from nutomo.probability import OscParams


@pytest.fixture(scope="module")
def setup():
    g = TesseroidGrid(DEFAULT_R_EDGES_KM, 10, 10)
    ch = trace_from_site(g, 42.8, 6.03, 2.45, -0.8, 200.0)
    m = DensityModel(g, radial=shell_mean_prem(g),
                     lateral=blob_anomaly(g, 0, -160, 3480, 3900, 30, 0.02))
    rho = m.densities(ch.cell_ids, ch.r_mid_km)
    ye = m.electron_fraction(ch.r_mid_km)
    E = np.logspace(np.log10(2), 2, 8)
    return g, ch, rho, ye, E, OscParams()


def test_single_slab_derivative(setup):
    g, ch, rho, ye, E, pars = setup
    u, h = prob.slab_operators(ch.widths_km, rho, ye, E, pars)
    w = prob.widths_inv_ev(ch.widths_km)
    for anti in (False, True):
        if anti:
            u, h = prob.slab_operators(ch.widths_km, rho, ye, E, pars, True)
        dva = jac.slab_operator_derivatives(h, w, prob.dpotential_drho(ye), anti)
        dvn = jac.dslab_drho_numeric(h, w, prob.dpotential_drho(ye), 1e-3, anti)
        assert np.max(np.abs(dva - dvn)) < 1e-7 * max(1.0, np.max(np.abs(dvn)) / 0.1)


def test_chord_assembly_ordering(setup):
    g, ch, rho, ye, E, pars = setup
    u, h = prob.slab_operators(ch.widths_km, rho, ye, E, pars)
    w = prob.widths_inv_ev(ch.widths_km)
    du = jac.slab_operator_derivatives(h, w, prob.dpotential_drho(ye))
    dU, U = jac.chord_operator_jacobian(u, du)
    assert np.allclose(U, prob.compose(u))
    for k in (0, 3, len(rho) - 1):
        d = np.zeros_like(rho); d[k] = 1e-3
        fd = (prob.evolution_operator(ch.widths_km, rho + d, ye, E, pars)
              - prob.evolution_operator(ch.widths_km, rho - d, ye, E, pars)) / 2e-3
        assert np.max(np.abs(dU[:, k] - fd)) < 1e-8


def test_cell_jacobian_vs_finite_difference(setup):
    g, ch, rho, ye, E, pars = setup
    for anti in (False, True):
        P, dP = jac.probability_jacobian(ch.widths_km, rho, ye, E, pars, anti)
        assert np.allclose(P, prob.probabilities(ch.widths_km, rho, ye, E, pars, anti))
        cells, dPc = jac.aggregate_slabs_to_cells(dP, ch.cell_ids)
        fd = jac.finite_difference_cell_jacobian(ch.widths_km, rho, ye, E, pars, ch.cell_ids,
                                                 cells, 1e-3, anti)
        assert np.max(np.abs(dPc - fd)) < 1e-8
        # probability closure of the derivative: rows and columns sum to zero
        assert np.allclose(dPc.sum(-1), 0, atol=1e-12) and np.allclose(dPc.sum(-2), 0, atol=1e-12)


def test_degenerate_eigenvalue_branch():
    # zero Hamiltonian: all eigenvalues equal, derivative must be -i L dH
    h = np.zeros((1, 1, 3, 3), complex)
    w = np.array([1e12])
    dv = np.array([1e-13])
    du = jac.slab_operator_derivatives(h, w, dv)
    expected = np.zeros((3, 3), complex); expected[0, 0] = -1j * w[0] * dv[0]
    expected -= np.eye(3) * (-1j * w[0] * dv[0] / 3)     # traceless convention
    assert np.allclose(du[0, 0], expected)
