"""dP/drho_j for every slab j along a chord, from cached partial products.

With U = U_n ... U_1,

    dU/drho_j = (U_n ... U_{j+1}) . (dU_j/drho_j) . (U_{j-1} ... U_1)
              =        S_j        .   dU_j        .      P_{j-1}

so with prefix products P and suffix products S cached, all n derivatives
cost ~2n operator multiplications instead of n full re-solves.

The single-slab derivative is exact (Daleckii-Krein / Loewner form):
for H = V diag(lam) V^+ and U_j = exp(-i H L),

    dU_j = V ( Phi o (V^+ dH V) ) V^+ ,
    Phi_ab = (e^{-i lam_a L} - e^{-i lam_b L}) / (lam_a - lam_b),   a != b
    Phi_aa = -i L e^{-i lam_a L}

with dH/drho = (dV_CC/drho) e_ee (a rank-one, real perturbation of H_ee).
NuOscProbExact propagates the traceless part of H, so the derivative is
taken of exp(-i H_tl L) to stay in the library's phase convention.
The operator ordering is the easiest place in the project to slip; the
test suite finite-differences every piece.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import probability as prob
from .probability import OscParams


# ---------------------------------------------------------------------------
# single-slab derivative
# ---------------------------------------------------------------------------
def _loewner_phase(lam: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Phi_ab for eigenvalues lam (..., 3) and widths L (...,)."""
    ph = np.exp(-1j * lam * L[..., None])                       # (..., 3)
    dl = lam[..., :, None] - lam[..., None, :]                  # (..., 3, 3)
    x = dl * L[..., None, None]                                 # dimensionless
    # exact expression where safe
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (ph[..., :, None] - ph[..., None, :]) / (-1j * x)
    # series (e^{-ix}-1)/(-ix) = 1 - ix/2 - x^2/6 + i x^3/24 ... near x -> 0
    small = np.abs(x) < 1e-6
    ser = ph[..., None, :] * (1.0 - 0.5j * x - x ** 2 / 6.0 + 1j * x ** 3 / 24.0)
    phi = np.where(small, ser, phi)
    idx = np.arange(lam.shape[-1])
    phi[..., idx, idx] = ph
    # Daleckii-Krein: Phi_ab = (f(lam_a) - f(lam_b)) / (lam_a - lam_b) with
    # f = exp(-i lam L); the ratio above is that divided by (-i L), and the
    # diagonal is f'(lam_a) = -i L f(lam_a).  Restore the common factor.
    return phi * (-1j * L[..., None, None])


def slab_operator_derivatives(h: np.ndarray, widths_inv_ev: np.ndarray,
                              dV_drho: np.ndarray,
                              antineutrino: bool = False) -> np.ndarray:
    """dU_k/drho_k for every slab, shape (nE, n, 3, 3).

    Parameters
    ----------
    h : (nE, n, 3, 3) Hamiltonians (eV).
    widths_inv_ev : (n,) widths in eV^-1.
    dV_drho : (n,) dV_CC/drho per slab (eV per g/cm^3).  For antineutrinos the
        potential enters with the opposite sign; pass ``antineutrino=True``.
    """
    # NuOscProbExact's SU(3) expansion propagates the *traceless* part of H
    # (the dropped phase cancels in every probability).  The derivative has
    # to follow the same convention or 2 Re(conj(U) dU) mixes two phases:
    # differentiate exp(-i H_tl L) with H_tl = H - tr(H)/3, so that
    # dH_tl = dV/drho (e_ee - I/3).
    n_f = h.shape[-1]
    tr = np.trace(h, axis1=-2, axis2=-1)[..., None, None] / n_f
    h_tl = h - tr * np.eye(n_f)
    lam, V = np.linalg.eigh(h_tl)                                # (nE,n,3), (nE,n,3,3)
    L = np.broadcast_to(widths_inv_ev, lam.shape[:-1])
    phi = _loewner_phase(lam, L)                                 # (nE,n,3,3)
    sgn = -1.0 if antineutrino else 1.0
    v0 = V[..., 0, :]                                            # (nE,n,3)
    inner = (np.conj(v0)[..., :, None] * v0[..., None, :]) - np.eye(n_f) / n_f
    core = phi * ((sgn * dV_drho)[None, :, None, None] * inner)
    return V @ core @ np.conj(np.swapaxes(V, -1, -2))


def dslab_drho_numeric(h: np.ndarray, widths_inv_ev: np.ndarray, dV_drho: np.ndarray,
                       eps: float = 1e-3, antineutrino: bool = False) -> np.ndarray:
    """Central finite-difference of the *library* single-slab operator (tests)."""
    import oscprob3nu
    sgn = -1.0 if antineutrino else 1.0
    dh = np.zeros_like(h)
    dh[..., 0, 0] = sgn * dV_drho[None, :] * eps
    w_b = np.broadcast_to(widths_inv_ev, h.shape[:-2])
    up = np.asarray(oscprob3nu.evolution_operator_3nu(h + dh, w_b))
    dn = np.asarray(oscprob3nu.evolution_operator_3nu(h - dh, w_b))
    return (up - dn) / (2.0 * eps)


# ---------------------------------------------------------------------------
# chord assembly
# ---------------------------------------------------------------------------
def partial_products(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prefix P_j = U_j...U_1 (j=0..n, P_0 = I) and suffix S_j = U_n...U_{j+1}
    (j=0..n, S_n = I).  Shapes (nE, n+1, 3, 3)."""
    nE, n = u.shape[:2]
    eye = np.broadcast_to(np.eye(3, dtype=complex), (nE, 3, 3))
    P = np.empty((nE, n + 1, 3, 3), dtype=complex)
    S = np.empty((nE, n + 1, 3, 3), dtype=complex)
    P[:, 0] = eye
    for j in range(n):
        P[:, j + 1] = u[:, j] @ P[:, j]
    S[:, n] = eye
    for j in range(n - 1, -1, -1):
        S[:, j] = S[:, j + 1] @ u[:, j]
    return P, S


def chord_operator_jacobian(u: np.ndarray, du: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """dU_total/drho_k for all k, shape (nE, n, 3, 3), plus U_total (nE, 3, 3).

    dU/drho_k = S_k . dU_k . P_{k-1}  with 0-based slab index k -> S[k+1], P[k].
    """
    P, S = partial_products(u)
    n = u.shape[1]
    dU = S[:, 1:n + 1] @ du @ P[:, 0:n]
    return dU, P[:, n]


def probability_jacobian(widths_km, rho, ye, energies_gev, params: OscParams,
                         antineutrino: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """P (nE, 3, 3) [initial, final] and dP/drho_slab (nE, n, 3, 3) [.., initial, final].

    dP_ab / drho_k = 2 Re( conj(U_ba) dU_k[b, a] ).
    """
    u_slabs, h = prob.slab_operators(widths_km, rho, ye, energies_gev, params, antineutrino)
    w = prob.widths_inv_ev(widths_km)
    du = slab_operator_derivatives(h, w, prob.dpotential_drho(ye), antineutrino)
    dU, U = chord_operator_jacobian(u_slabs, du)
    P = np.abs(np.swapaxes(U, -1, -2)) ** 2
    dP = 2.0 * np.real(np.conj(U)[:, None, :, :] * dU)            # (nE, n, final, initial)
    return P, np.swapaxes(dP, -1, -2)


def aggregate_slabs_to_cells(dP_slab: np.ndarray, cell_ids: np.ndarray,
                             weights: Optional[np.ndarray] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Sum slab derivatives into the unique cells they belong to.

    Returns (unique_cells (m,), dP_cell (nE, m, 3, 3)).  With ``weights``
    (n,) the slab derivative is multiplied first, e.g. by the radial
    density to convert d/drho into d/d(lateral fraction).
    """
    cells, inv = np.unique(np.asarray(cell_ids), return_inverse=True)
    src = dP_slab if weights is None else dP_slab * np.asarray(weights)[None, :, None, None]
    out = np.zeros((dP_slab.shape[0], cells.size) + dP_slab.shape[2:], dtype=dP_slab.dtype)
    np.add.at(out, (slice(None), inv), src)
    return cells, out


def finite_difference_cell_jacobian(widths_km, rho, ye, energies_gev, params,
                                    cell_ids, cells, eps=1e-3, antineutrino=False,
                                    weights=None) -> np.ndarray:
    """Brute-force dP/d(rho or fraction) per cell, for tests. (nE, m, 3, 3)."""
    out = np.empty((np.size(energies_gev), len(cells), 3, 3))
    w = np.ones_like(rho) if weights is None else np.asarray(weights)
    for j, c in enumerate(cells):
        mask = (np.asarray(cell_ids) == c)
        d = np.where(mask, w * eps, 0.0)
        pu = prob.probabilities(widths_km, rho + d, ye, energies_gev, params, antineutrino)
        pd = prob.probabilities(widths_km, rho - d, ye, energies_gev, params, antineutrino)
        out[:, j] = (pu - pd) / (2.0 * eps)
    return out
