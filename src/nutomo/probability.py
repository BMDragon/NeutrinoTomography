"""Thin wrapper over NuOscProbExact.  The *only* module that imports it.

Everything upstream hands in ``(widths_km, rho, ye)`` per segment and a set
of oscillation parameters; everything downstream receives probabilities
indexed ``[energy, initial flavor, final flavor]`` or evolution operators
indexed ``[energy, final, initial]`` (the solver's own convention).

Batching: the batch axis of the slab kernels requires *shared* widths, so
we batch over energy at fixed geometry.  Directions are looped outside.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np

import globaldefs as gd
import hamiltonians3nu
import oscprob3nu
import slabs as _slabs
import earth as _earth

CONV_KM_TO_INV_EV = gd.CONV_KM_TO_INV_EV
GEV = 1.0e9

#: dV_CC / d(rho Y_e)  in eV per (g/cm^3): V = sqrt(2) G_F n_e
_VCC_PER_RHO_YE = (np.sqrt(2.0) * gd.GF * gd.CONV_G_TO_EV
                   / ((gd.MASS_PROTON + gd.MASS_NEUTRON) / 2.0)
                   / gd.CONV_CM_TO_INV_EV ** 3)


@dataclass
class OscParams:
    """Standard three-flavor parameters.  Angles as sin^2, masses in eV^2."""
    s12sq: float = 0.310
    s23sq: float = 0.582
    s13sq: float = 0.02240
    dcp: float = 217.0 / 180.0 * np.pi
    dm21: float = 7.39e-5
    dm31: float = 2.525e-3

    @classmethod
    def normal_ordering(cls) -> "OscParams":
        return cls()

    @classmethod
    def inverted_ordering(cls) -> "OscParams":
        return cls(s13sq=0.02263, dcp=280.0 / 180.0 * np.pi,
                   dm31=-2.512e-3 + 7.39e-5)

    def flipped_ordering(self) -> "OscParams":
        """Same |dm32|, other ordering."""
        dm32 = self.dm31 - self.dm21
        return OscParams(self.s12sq, self.s23sq, self.s13sq, self.dcp,
                         self.dm21, -dm32 + self.dm21)

    def to_dict(self) -> Dict:
        return asdict(self)

    def replace(self, **kw) -> "OscParams":
        d = self.to_dict()
        d.update(kw)
        return OscParams(**d)

    def h_vacuum(self, antineutrino: bool = False) -> np.ndarray:
        """Energy-independent vacuum Hamiltonian (eV^2), to be divided by 2E."""
        h = np.asarray(hamiltonians3nu.hamiltonian_3nu_vacuum_energy_independent(
            np.sqrt(self.s12sq), np.sqrt(self.s23sq), np.sqrt(self.s13sq),
            self.dcp, self.dm21, self.dm31), dtype=complex)
        return h.conj() if antineutrino else h


def matter_potential(rho, ye) -> np.ndarray:
    """V_CC (eV) for density rho (g/cm^3) and electron fraction ye."""
    return _VCC_PER_RHO_YE * np.asarray(rho, float) * np.asarray(ye, float)


def dpotential_drho(ye) -> np.ndarray:
    """dV_CC/drho (eV per g/cm^3)."""
    return _VCC_PER_RHO_YE * np.asarray(ye, float)


def hamiltonian_stack(params: OscParams, energies_gev, rho, ye,
                      antineutrino: bool = False) -> np.ndarray:
    """Per-slab Hamiltonians, shape (nE, n_slabs, 3, 3), in eV."""
    E = np.atleast_1d(np.asarray(energies_gev, float)) * GEV
    V = matter_potential(rho, ye)
    if antineutrino:
        V = -V
    h = hamiltonians3nu.hamiltonian_3nu_matter(params.h_vacuum(antineutrino),
                                               E[:, None], V)
    return np.asarray(h, dtype=complex)


def widths_inv_ev(widths_km) -> np.ndarray:
    return np.asarray(widths_km, float) * CONV_KM_TO_INV_EV


def evolution_operator(widths_km, rho, ye, energies_gev, params: OscParams,
                       antineutrino: bool = False) -> np.ndarray:
    """Total evolution operator per energy, shape (nE, 3, 3), indexed (final, initial)."""
    h = hamiltonian_stack(params, energies_gev, rho, ye, antineutrino)
    w = widths_inv_ev(widths_km)
    batch = getattr(_slabs, "_evolution_operator_slabs_batch", None)
    if batch is not None:
        return np.asarray(batch(h, w, 3, "nutomo.evolution_operator"))
    return np.stack([_slabs.evolution_operator_3nu_slabs(h[i], w) for i in range(h.shape[0])])


def probabilities(widths_km, rho, ye, energies_gev, params: OscParams,
                  antineutrino: bool = False) -> np.ndarray:
    """Oscillation probabilities, shape (nE, 3, 3) indexed [initial, final].

    Flavor order (e, mu, tau).  P[:, a, b] = P(nu_a -> nu_b).
    """
    u = evolution_operator(widths_km, rho, ye, energies_gev, params, antineutrino)
    return np.abs(np.swapaxes(u, -1, -2)) ** 2


def slab_operators(widths_km, rho, ye, energies_gev, params: OscParams,
                   antineutrino: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Per-slab exact evolution operators U_k, shape (nE, n_slabs, 3, 3),
    together with the Hamiltonian stack that produced them."""
    h = hamiltonian_stack(params, energies_gev, rho, ye, antineutrino)
    w = widths_inv_ev(widths_km)
    w_b = np.broadcast_to(w, h.shape[:-2])
    u = np.asarray(oscprob3nu.evolution_operator_3nu(h, w_b), dtype=complex)
    return u, h


def compose(u_slabs: np.ndarray) -> np.ndarray:
    """U = U_n ... U_1 along axis -3."""
    u_total = u_slabs[..., 0, :, :]
    for k in range(1, u_slabs.shape[-3]):
        u_total = u_slabs[..., k, :, :] @ u_total
    return u_total


def probabilities_prem_native(cosz: float, energies_gev, params: OscParams,
                              n_slabs_per_segment: int = 8, antineutrino: bool = False,
                              electron_fraction: float = 0.5) -> np.ndarray:
    """PREM probabilities through NuOscProbExact's *own* chord decomposition.

    Cross-check on absolute normalisation only.  Never subtract this from a
    slab-API result: the discretisations differ (see README, matched legs).
    Returns (nE, 3, 3) indexed [initial, final].
    """
    E = np.atleast_1d(np.asarray(energies_gev, float)) * GEV
    hvac = params.h_vacuum(antineutrino)
    if antineutrino:
        # native routine has no antineutrino switch; flip the sign of the
        # potential by handing it a negative electron fraction
        electron_fraction = -electron_fraction
    p = np.asarray(_earth.probabilities_3nu_earth(hvac, E, cosz, n_slabs_per_segment,
                                                  electron_fraction=electron_fraction), float)
    return p.reshape(-1, 3, 3)


def prem_chord_native(cosz: float, n_slabs_per_segment: int = 8):
    """NuOscProbExact's own (widths_km, densities) for a PREM chord."""
    return _earth.earth_slabs(cosz, n_slabs_per_segment)


def solver_info() -> Dict:
    import fastkernels
    ver = None
    try:
        from importlib.metadata import version
        ver = version("nuoscprobexact")
    except Exception:
        pass
    return {"nuoscprobexact": ver, "numba_kernels": bool(fastkernels.available()),
            "use_numba": bool(getattr(fastkernels, "USE_NUMBA", None))}
