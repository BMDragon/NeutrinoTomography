"""Detector sites, exposure, binning and detector response kernels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

N_AVOGADRO = 6.02214076e23
SECONDS_PER_YEAR = 365.25 * 86400.0
CHANNELS = ("track", "cascade")


@dataclass
class Detector:
    name: str
    lat_deg: float
    lon_deg: float
    depth_km: float = 0.0
    mass_mton: float = 1.0
    livetime_yr: float = 10.0
    e_min_gev: float = 2.0
    e_max_gev: float = 100.0
    n_e_bins: int = 16
    e_oversample: int = 2
    sigma_log10e: float = 0.15
    sigma_cosz: float = 0.10
    sigma_az_deg: float = 15.0
    efficiency: Dict[str, float] = field(default_factory=lambda: {"track": 1.0, "cascade": 1.0})

    # -- exposure ----------------------------------------------------------
    @property
    def n_targets(self) -> float:
        """Nucleons in the fiducial mass (1 Mton = 1e12 g)."""
        return self.mass_mton * 1.0e12 * N_AVOGADRO

    @property
    def exposure_s(self) -> float:
        return self.livetime_yr * SECONDS_PER_YEAR

    # -- energy binning ----------------------------------------------------
    def energy_edges(self) -> np.ndarray:
        return np.logspace(np.log10(self.e_min_gev), np.log10(self.e_max_gev), self.n_e_bins + 1)

    def energy_centers(self) -> np.ndarray:
        e = self.energy_edges()
        return np.sqrt(e[:-1] * e[1:])

    def energy_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sub-bin sample energies (nE*k,), their widths (GeV) and parent bin."""
        e = self.energy_edges()
        k = max(1, int(self.e_oversample))
        sub = np.concatenate([np.logspace(np.log10(e[i]), np.log10(e[i + 1]), k + 1)
                              for i in range(self.n_e_bins)]).reshape(self.n_e_bins, k + 1)
        centers = np.sqrt(sub[:, :-1] * sub[:, 1:]).ravel()
        widths = np.diff(sub, axis=1).ravel()
        parent = np.repeat(np.arange(self.n_e_bins), k)
        return centers, widths, parent

    def site(self) -> Dict:
        return {"name": self.name, "lat_deg": self.lat_deg, "lon_deg": self.lon_deg,
                "depth_km": self.depth_km}

    @classmethod
    def from_dict(cls, d: Dict) -> "Detector":
        d = dict(d)
        eff = d.pop("efficiency", None)
        det = cls(**d)
        if eff:
            det.efficiency = {k: float(v) for k, v in eff.items()}
        return det


@dataclass
class Binning:
    """Analysis bins in (cos zenith, azimuth), each sampled by ``cosz_sub x az_sub``
    sub-rays at the sub-bin centres.  Sub-sampling matters most near
    cosz = -1, where a single bin-centre ray misses the inner core."""
    cosz_edges: np.ndarray
    az_edges_deg: np.ndarray
    cosz_sub: int = 1
    az_sub: int = 1

    @property
    def cosz_centers(self) -> np.ndarray:
        return 0.5 * (self.cosz_edges[:-1] + self.cosz_edges[1:])

    @property
    def az_centers(self) -> np.ndarray:
        return 0.5 * (self.az_edges_deg[:-1] + self.az_edges_deg[1:])

    # -- sub-bin sampling (what the ray table is built on) ---------------
    @staticmethod
    def _sub_edges(edges: np.ndarray, k: int) -> np.ndarray:
        return np.concatenate([np.linspace(edges[i], edges[i + 1], k + 1)[:-1] for i in range(edges.size - 1)]
                              + [edges[-1:]])

    @property
    def cosz_sub_edges(self) -> np.ndarray:
        return self._sub_edges(self.cosz_edges, self.cosz_sub)

    @property
    def az_sub_edges(self) -> np.ndarray:
        return self._sub_edges(self.az_edges_deg, self.az_sub)

    @property
    def cosz_sub_centers(self) -> np.ndarray:
        e = self.cosz_sub_edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def az_sub_centers(self) -> np.ndarray:
        e = self.az_sub_edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def d_cosz_sub(self) -> np.ndarray:
        return np.diff(self.cosz_sub_edges)

    @property
    def d_az_sub_rad(self) -> np.ndarray:
        return np.deg2rad(np.diff(self.az_sub_edges))

    def parent_cosz(self, icz_sub) -> np.ndarray:
        return np.asarray(icz_sub) // self.cosz_sub

    def parent_az(self, iaz_sub) -> np.ndarray:
        return np.asarray(iaz_sub) // self.az_sub

    @property
    def d_cosz(self) -> np.ndarray:
        return np.diff(self.cosz_edges)

    @property
    def d_az_rad(self) -> np.ndarray:
        return np.deg2rad(np.diff(self.az_edges_deg))

    @property
    def n_cosz(self) -> int:
        return self.cosz_edges.size - 1

    @property
    def n_az(self) -> int:
        return self.az_edges_deg.size - 1

    @classmethod
    def from_dict(cls, d: Dict) -> "Binning":
        c = d["cosz_edges"]
        a = d["az_edges"]
        cz = np.linspace(c["min"], c["max"], int(c["n"]) + 1) if isinstance(c, dict) else np.asarray(c, float)
        az = np.linspace(a["min"], a["max"], int(a["n"]) + 1) if isinstance(a, dict) else np.asarray(a, float)
        return cls(cz, az, int(d.get("cosz_sub", 1)), int(d.get("az_sub", 1)))


def load_detectors(path: str) -> Tuple[List[Detector], Binning]:
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    dets = [Detector.from_dict(d) for d in cfg["detectors"]]
    return dets, Binning.from_dict(cfg["binning"])


# ---------------------------------------------------------------------------
# response kernels  K[reco, truth], columns sum to <= 1
# ---------------------------------------------------------------------------
def _gaussian_bin_kernel(edges: np.ndarray, sigma: float, truth_centers: np.ndarray,
                         periodic: float = None, n_sigma: float = 4.0) -> np.ndarray:
    """Probability that a truth value at each centre is reconstructed in each bin."""
    from scipy.special import erf
    lo, hi = edges[:-1], edges[1:]
    if sigma <= 0:
        K = np.zeros((lo.size, truth_centers.size))
        idx = np.clip(np.searchsorted(edges, truth_centers, side="right") - 1, 0, lo.size - 1)
        K[idx, np.arange(truth_centers.size)] = 1.0
        return K
    def cdf(x):
        return 0.5 * (1.0 + erf(x / (np.sqrt(2.0) * sigma)))
    t = truth_centers[None, :]
    K = cdf(hi[:, None] - t) - cdf(lo[:, None] - t)
    if periodic is not None:
        for shift in (-periodic, periodic):
            K += cdf(hi[:, None] - t + shift) - cdf(lo[:, None] - t + shift)
    return K


def response_kernels(det: Detector, binning: Binning) -> Dict[str, np.ndarray]:
    """Separable Gaussian smearing in log10(E), cos(zenith) and azimuth.

    Returns {'E': (nE, nE), 'cosz': (ncz, ncz), 'az': (naz, naz)} acting on
    truth-bin centres.  Energy events smeared outside the analysis range
    are lost; cos(zenith) is truncated at the physical range [-1, 1] with
    events smeared into the down-going hemisphere lost as well.  Azimuth
    is periodic.  (Approximation: angular smearing is not a true space
    angle; choose reco bins no finer than the resolution.)
    """
    e_edges = np.log10(det.energy_edges())
    e_c = 0.5 * (e_edges[:-1] + e_edges[1:])
    K_E = _gaussian_bin_kernel(e_edges, det.sigma_log10e, e_c)
    K_cz = _gaussian_bin_kernel(binning.cosz_edges, det.sigma_cosz, binning.cosz_centers)
    K_az = _gaussian_bin_kernel(binning.az_edges_deg, det.sigma_az_deg, binning.az_centers,
                                periodic=360.0)
    return {"E": K_E, "cosz": K_cz, "az": K_az}
