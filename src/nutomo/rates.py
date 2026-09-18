"""Event-rate expectations: probability x flux x cross section x exposure x response.

The observable is a Poisson count per (detector, cos zenith, azimuth,
energy, channel) bin.  Everything here is linear in the probabilities, so
the Jacobian of the counts with respect to per-cell density perturbations
follows from ``jacobian.probability_jacobian`` with no extra solves.

Channels
--------
* ``track``   : nu_mu + nubar_mu charged current (final flavor mu)
* ``cascade`` : nu_e  + nubar_e  charged current (final flavor e)

Neutral-current and nu_tau charged-current contributions are neglected.

Flux
----
The default ``GaisserLikeFlux`` is an analytic placeholder (Gaisser-type
power law with the pion/kaon zenith dependence, energy-dependent nu_e
fraction and constant nubar/nu ratios).  It is within a factor ~1.5 of
Honda tables at 1-100 GeV, which is adequate for sensitivity *scaling*
studies but not for an absolute rate claim.  Use ``TableFlux`` with a
tabulated flux for that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import jacobian as jac
from . import probability as prob
from .density import DensityModel
from .detectors import Binning, Detector, CHANNELS, response_kernels
from .probability import OscParams
from .raytrace import RayTable

FLAVORS = ("nue", "numu")            # initial flavors with atmospheric flux
FLAVOR_INDEX = {"nue": 0, "numu": 1, "nutau": 2}
CHANNEL_FINAL = {"track": 1, "cascade": 0}


# ---------------------------------------------------------------------------
# flux models
# ---------------------------------------------------------------------------
class FluxModel:
    """Interface: flux(flavor, antineutrino, E_gev, cosz) -> /(GeV cm^2 s sr)."""
    nuisance_names: Tuple[str, ...] = ()

    def flux(self, flavor: str, antineutrino: bool, e_gev: np.ndarray,
             cosz: float, nuis: Optional[Dict[str, float]] = None) -> np.ndarray:
        raise NotImplementedError


@dataclass
class GaisserLikeFlux(FluxModel):
    """Analytic conventional atmospheric flux.

    Phi_numu(E) = norm * A * E^-gamma * [1/(1 + 3.7 E c/115) + 0.38/(1 + 1.7 E c/850)]
    with c = max(|cosz|, c_min) an effective cosine, in /(GeV cm^2 s sr).
    nu_e / nu_mu = r_e * (E/GeV)^-0.25 ; nubar/nu = r_bar (flavor dependent).

    Nuisances: flux_norm (multiplicative), delta_gamma (spectral tilt about
    E_pivot), nue_numu_ratio (multiplicative on nu_e), nubar_nu_ratio
    (multiplicative on all antineutrinos).
    """
    A: float = 0.0096
    gamma: float = 2.7
    nue_ratio_1gev: float = 0.5
    nubar_ratio_mu: float = 0.78
    nubar_ratio_e: float = 0.72
    c_min: float = 0.05
    e_pivot_gev: float = 10.0
    nuisance_names: Tuple[str, ...] = ("flux_norm", "delta_gamma", "nue_numu_ratio", "nubar_nu_ratio")

    def flux(self, flavor, antineutrino, e_gev, cosz, nuis=None):
        nuis = nuis or {}
        E = np.asarray(e_gev, float)
        c = max(abs(float(cosz)), self.c_min)
        gamma = self.gamma + nuis.get("delta_gamma", 0.0)
        base = (self.A * E ** (-gamma) * self.e_pivot_gev ** nuis.get("delta_gamma", 0.0)
                * (1.0 / (1.0 + 3.7 * E * c / 115.0) + 0.38 / (1.0 + 1.7 * E * c / 850.0)))
        base = base * nuis.get("flux_norm", 1.0)
        if flavor == "nue":
            base = base * self.nue_ratio_1gev * E ** (-0.25) * nuis.get("nue_numu_ratio", 1.0)
            if antineutrino:
                base = base * self.nubar_ratio_e * nuis.get("nubar_nu_ratio", 1.0)
        elif flavor == "numu":
            if antineutrino:
                base = base * self.nubar_ratio_mu * nuis.get("nubar_nu_ratio", 1.0)
        else:
            return np.zeros_like(E)
        return base


@dataclass
class TableFlux(FluxModel):
    """Flux tabulated on (log10 E, cosz); bilinear interpolation in log flux.

    ``table[key]`` for key in ('nue','numu','nuebar','numubar') has shape
    (nE, ncosz).  Same nuisances as GaisserLikeFlux, applied multiplicatively.
    """
    log10e: np.ndarray
    cosz: np.ndarray
    table: Dict[str, np.ndarray]
    e_pivot_gev: float = 10.0
    nuisance_names: Tuple[str, ...] = ("flux_norm", "delta_gamma", "nue_numu_ratio", "nubar_nu_ratio")

    @classmethod
    def from_npz(cls, path: str) -> "TableFlux":
        with np.load(path) as z:
            return cls(np.asarray(z["log10e"]), np.asarray(z["cosz"]),
                       {k: np.asarray(z[k]) for k in ("nue", "numu", "nuebar", "numubar")})

    def flux(self, flavor, antineutrino, e_gev, cosz, nuis=None):
        from scipy.interpolate import RegularGridInterpolator
        nuis = nuis or {}
        key = flavor + ("bar" if antineutrino else "")
        if key not in self.table:
            return np.zeros_like(np.asarray(e_gev, float))
        f = RegularGridInterpolator((self.log10e, self.cosz), np.log(np.maximum(self.table[key], 1e-300)),
                                    bounds_error=False, fill_value=None)
        E = np.asarray(e_gev, float)
        pts = np.column_stack([np.log10(E), np.full(E.shape, float(cosz))])
        out = np.exp(f(pts))
        out *= nuis.get("flux_norm", 1.0) * (E / self.e_pivot_gev) ** (-nuis.get("delta_gamma", 0.0))
        if flavor == "nue":
            out *= nuis.get("nue_numu_ratio", 1.0)
        if antineutrino:
            out *= nuis.get("nubar_nu_ratio", 1.0)
        return out


# ---------------------------------------------------------------------------
# cross sections
# ---------------------------------------------------------------------------
@dataclass
class CrossSection:
    """Charged-current cross section per nucleon, linear in E (cm^2)."""
    sigma_per_gev_nu: float = 0.677e-38
    sigma_per_gev_nubar: float = 0.334e-38

    def sigma_cc(self, antineutrino: bool, e_gev) -> np.ndarray:
        c = self.sigma_per_gev_nubar if antineutrino else self.sigma_per_gev_nu
        return c * np.asarray(e_gev, float)


# ---------------------------------------------------------------------------
# rate calculator
# ---------------------------------------------------------------------------
@dataclass
class RayRates:
    """Truth-level result for one ray: counts and optional cell Jacobian."""
    counts: np.ndarray                      # (nE_bins, n_ch)
    cells: Optional[np.ndarray] = None      # (m,)
    dcounts_dfrac: Optional[np.ndarray] = None   # (nE_bins, n_ch, m) d/d(fractional density) per cell


@dataclass
class RateResult:
    truth: List[np.ndarray]                 # per detector (ncz, naz, nE, nch)
    reco: List[np.ndarray]                  # per detector (ncz, naz, nE, nch)
    ray_jacobians: Optional[List[RayRates]] = None   # per ray, or None

    def total(self) -> float:
        return float(sum(r.sum() for r in self.reco))


class RateCalculator:
    """Turns a ray table + density model + parameters into binned expectations."""

    def __init__(self, ray_table: RayTable, detectors: Sequence[Detector], binning: Binning,
                 flux: FluxModel, xsec: Optional[CrossSection] = None):
        if ray_table.n_det != len(detectors):
            raise ValueError("ray table and detector list disagree on the number of sites")
        if (ray_table.n_cosz != binning.n_cosz * binning.cosz_sub
                or ray_table.n_az != binning.n_az * binning.az_sub):
            raise ValueError("ray table binning does not match the requested binning "
                             "(including cosz_sub / az_sub)")
        self.rays = ray_table
        self.detectors = list(detectors)
        self.binning = binning
        self.flux = flux
        self.xsec = xsec or CrossSection()
        self.kernels = [response_kernels(d, binning) for d in self.detectors]
        self._esamp = [d.energy_samples() for d in self.detectors]
        self._d_cosz_sub = binning.d_cosz_sub
        self._d_az_sub_rad = binning.d_az_sub_rad
        self._parent_cz = binning.parent_cosz(np.arange(ray_table.n_cosz))
        self._parent_az = binning.parent_az(np.arange(ray_table.n_az))

    # -- per-ray truth rates ------------------------------------------------
    def _bin_weights(self, idet: int, icz_sub: int, iaz_sub: int) -> np.ndarray:
        """N_targets * T * dOmega(sub-bin) * dE, shape (n_samples,)."""
        d = self.detectors[idet]
        _, de, _ = self._esamp[idet]
        domega = self._d_cosz_sub[icz_sub] * self._d_az_sub_rad[iaz_sub]
        return d.n_targets * d.exposure_s * domega * de

    def ray_bin(self, iray: int) -> Tuple[int, int, int]:
        """(detector, analysis cosz bin, analysis az bin) of a ray."""
        rt = self.rays
        icz_sub = (iray // rt.n_az) % rt.n_cosz
        iaz_sub = iray % rt.n_az
        return int(rt.det_index[iray]), int(self._parent_cz[icz_sub]), int(self._parent_az[iaz_sub])

    def ray_rates(self, iray: int, model: DensityModel, params: OscParams,
                  flux_nuis: Optional[Dict[str, float]] = None,
                  with_jacobian: bool = False) -> RayRates:
        rt = self.rays
        idet = int(rt.det_index[iray])
        icz_sub = (iray // rt.n_az) % rt.n_cosz
        iaz_sub = iray % rt.n_az
        det = self.detectors[idet]
        e_s, de, parent = self._esamp[idet]
        widths, cells, r_mid = rt.segments(iray)
        rho = model.densities(cells, r_mid)
        ye = model.electron_fraction(r_mid)
        rho_rad = model.radial_density(cells, r_mid)
        w_bin = self._bin_weights(idet, icz_sub, iaz_sub)      # (n_samples,)
        cosz = float(rt.cosz[iray])
        nE = det.n_e_bins
        nch = len(CHANNELS)
        counts = np.zeros((nE, nch))
        dcounts = None
        ucells = None
        for anti in (False, True):
            if with_jacobian:
                P, dP = jac.probability_jacobian(widths, rho, ye, e_s, params, anti)
                ucells, dPc = jac.aggregate_slabs_to_cells(dP, cells, weights=rho_rad)  # (nS, m, 3, 3)
            else:
                P = prob.probabilities(widths, rho, ye, e_s, params, anti)
                dPc = None
            sig = self.xsec.sigma_cc(anti, e_s)                 # (nS,)
            for fl in FLAVORS:
                phi = self.flux.flux(fl, anti, e_s, cosz, flux_nuis)
                pref = phi * sig * w_bin                       # (nS,)
                a = FLAVOR_INDEX[fl]
                for ich, ch in enumerate(CHANNELS):
                    b = CHANNEL_FINAL[ch]
                    eff = det.efficiency.get(ch, 1.0)
                    contrib = pref * P[:, a, b] * eff          # (nS,)
                    np.add.at(counts[:, ich], parent, contrib)
                    if with_jacobian:
                        if dcounts is None:
                            dcounts = np.zeros((nE, nch, ucells.size))
                        c2 = (pref * eff)[:, None] * dPc[:, :, a, b]   # (nS, m)
                        np.add.at(dcounts, (parent, ich), c2)
        return RayRates(counts, ucells, dcounts)

    # -- full tables ----------------------------------------------------------
    def compute(self, model: DensityModel, params: OscParams,
                flux_nuis: Optional[Dict[str, float]] = None,
                with_jacobian: bool = False, progress=None) -> RateResult:
        rt = self.rays
        b = self.binning
        truth = [np.zeros((b.n_cosz, b.n_az, d.n_e_bins, len(CHANNELS))) for d in self.detectors]
        jacs = [] if with_jacobian else None
        for i in range(rt.n_rays):
            rr = self.ray_rates(i, model, params, flux_nuis, with_jacobian)
            idet, icz, iaz = self.ray_bin(i)
            truth[idet][icz, iaz] += rr.counts
            if with_jacobian:
                jacs.append(rr)
            if progress is not None and (i + 1) % 200 == 0:
                progress(i + 1, rt.n_rays)
        reco = [self.apply_response(idet, truth[idet]) for idet in range(len(self.detectors))]
        return RateResult(truth, reco, jacs)

    def apply_response(self, idet: int, truth: np.ndarray) -> np.ndarray:
        K = self.kernels[idet]
        return np.einsum("Zz,Aa,Ee,zaec->ZAEc", K["cosz"], K["az"], K["E"], truth, optimize=True)

    def bin_index(self, idet: int, icz, iaz, ie, ich) -> int:
        d = self.detectors[idet]
        return ((icz * self.binning.n_az + iaz) * d.n_e_bins + ie) * len(CHANNELS) + ich

    def n_bins(self, idet: int) -> int:
        return self.binning.n_cosz * self.binning.n_az * self.detectors[idet].n_e_bins * len(CHANNELS)
