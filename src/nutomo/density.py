"""Density and electron-fraction models on the tesseroid grid.

The model evaluated along a chord is

    rho(segment) = radial(r_mid or shell) * radial_scale[shell]
                   * (1 + lateral_scale * lateral[cell])

so the three analysis legs and the nested hypotheses are all *parameter
choices* of one object, never separate code paths:

* 3D model            : radial = shell means, lateral = anomaly field, lateral_scale = 1
* lateral-zeroed null : same, lateral_scale = 0 (radial_scale left free in fits)
* PREM on same grid   : radial = 'prem' (pointwise analytic PREM at r_mid), lateral_scale = 0
* laterally averaged  : radial = lateral_average(grid, rho_3d), lateral_scale = 0

This module imports nothing from NuOscProbExact.  PREM is re-implemented
here (identical polynomials) so the geometry/density layer has no solver
dependency; ``tests/`` checks it against ``earth.density_prem``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional, Union

import numpy as np

from .tesseroids import TesseroidGrid, lateral_average, R_EARTH_KM

# --------------------------------------------------------------------------
# PREM (Dziewonski & Anderson 1981), density in g/cm^3 as a polynomial in
# x = r / R_EARTH.  Right-closed shells, same table as NuOscProbExact.
# --------------------------------------------------------------------------
_PREM_BOUNDARIES = np.array([1221.5, 3480.0, 5701.0, 5771.0, 5971.0, 6151.0,
                             6346.6, 6356.0, 6368.0, R_EARTH_KM])
_PREM_COEFFS = np.array([
    [13.0885, 0.0, -8.8381, 0.0],
    [12.5815, -1.2638, -3.6426, -5.5281],
    [7.9565, -6.4761, 5.5283, -3.0807],
    [5.3197, -1.4836, 0.0, 0.0],
    [11.2494, -8.0298, 0.0, 0.0],
    [7.1089, -3.8045, 0.0, 0.0],
    [2.6910, 0.6924, 0.0, 0.0],
    [2.9, 0.0, 0.0, 0.0],
    [2.6, 0.0, 0.0, 0.0],
    [1.02, 0.0, 0.0, 0.0],
])

R_CMB_KM = 3480.0
R_ICB_KM = 1221.5

#: Electron fractions commonly used in oscillation work (e.g. nuCraft/OscProb):
YE_CORE = 0.4656
YE_MANTLE = 0.4957


def density_prem(r_km) -> np.ndarray:
    """PREM density (g/cm^3) at radius r (km), vectorised."""
    r = np.asarray(r_km, dtype=float)
    r = np.minimum(r, R_EARTH_KM)
    x = r / R_EARTH_KM
    c = _PREM_COEFFS[np.searchsorted(_PREM_BOUNDARIES, r, side="left")]
    return c[..., 0] + x * (c[..., 1] + x * (c[..., 2] + x * c[..., 3]))


def electron_fraction_profile(r_km, kind: str = "core_mantle") -> np.ndarray:
    """Electron fraction Y_e at radius r.

    kind = 'core_mantle' : YE_CORE inside the CMB, YE_MANTLE outside.
    kind = 'constant'    : 0.5 everywhere (NuOscProbExact's default).
    """
    r = np.asarray(r_km, dtype=float)
    if kind == "constant":
        return np.full_like(r, 0.5)
    if kind == "core_mantle":
        return np.where(r < R_CMB_KM, YE_CORE, YE_MANTLE)
    raise ValueError(f"unknown Ye profile kind {kind!r}")


def shell_mean_prem(grid: TesseroidGrid, n_sub: int = 400) -> np.ndarray:
    """Volume-weighted mean PREM density in each radial shell (n_r,)."""
    out = np.empty(grid.n_r)
    r_e = grid.r_edges
    for k in range(grid.n_r):
        r = np.linspace(r_e[k], r_e[k + 1], n_sub + 1)
        rm = 0.5 * (r[:-1] + r[1:])
        w = rm ** 2 * np.diff(r)
        out[k] = np.sum(w * density_prem(rm)) / np.sum(w)
    return out


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
RadialSpec = Union[str, np.ndarray, Callable[[np.ndarray], np.ndarray]]


@dataclass
class DensityModel:
    """rho(cell, r) = radial * radial_scale[shell] * (1 + lateral_scale*lateral[cell]).

    Parameters
    ----------
    grid : the tesseroid grid.
    radial : 'prem' (pointwise analytic PREM at the segment midpoint radius),
             an array of per-shell values (n_r,), or a callable of r (km).
    lateral : fractional perturbation per cell (n_cells,), or None for zero.
    lateral_scale : multiplies the lateral field (0 -> null hypothesis).
    radial_scale : per-shell multiplicative factors (n_r,), the free radial
             nuisance parameters of the fits.
    ye_kind : electron-fraction profile, see ``electron_fraction_profile``.
    """
    grid: TesseroidGrid
    radial: RadialSpec = "prem"
    lateral: Optional[np.ndarray] = None
    lateral_scale: float = 1.0
    radial_scale: Optional[np.ndarray] = None
    ye_kind: str = "core_mantle"
    name: str = "model"

    def __post_init__(self):
        if isinstance(self.radial, np.ndarray) and self.radial.shape != (self.grid.n_r,):
            raise ValueError("per-shell radial profile must have shape (n_r,)")
        if self.lateral is not None:
            self.lateral = np.asarray(self.lateral, float)
            if self.lateral.shape != (self.grid.n_cells,):
                raise ValueError("lateral must have shape (n_cells,)")
        if self.radial_scale is None:
            self.radial_scale = np.ones(self.grid.n_r)
        else:
            self.radial_scale = np.asarray(self.radial_scale, float)

    # -- variants (the nested hypotheses) ---------------------------------
    def with_lateral_scale(self, s: float) -> "DensityModel":
        return replace(self, lateral_scale=float(s))

    def with_radial_scale(self, rs) -> "DensityModel":
        return replace(self, radial_scale=np.asarray(rs, float))

    def laterally_averaged(self) -> "DensityModel":
        """Same grid, lateral structure replaced by its per-shell mean."""
        rho_cells = self.cell_densities()
        return replace(self, radial=lateral_average(self.grid, rho_cells),
                       lateral=None, lateral_scale=0.0,
                       radial_scale=np.ones(self.grid.n_r),
                       name=self.name + "_latavg")

    # -- evaluation ---------------------------------------------------------
    def radial_density(self, cell_ids, r_mid_km) -> np.ndarray:
        """Radial part only (including radial_scale), per segment."""
        cell_ids = np.asarray(cell_ids)
        shell = self.grid.shell_of(cell_ids)
        if isinstance(self.radial, str):
            if self.radial != "prem":
                raise ValueError(f"unknown radial spec {self.radial!r}")
            base = density_prem(r_mid_km)
        elif callable(self.radial):
            base = np.asarray(self.radial(np.asarray(r_mid_km, float)), float)
        else:
            base = np.asarray(self.radial)[shell]
        return base * self.radial_scale[shell]

    def lateral_factor(self, cell_ids) -> np.ndarray:
        if self.lateral is None or self.lateral_scale == 0.0:
            return np.ones(np.shape(cell_ids))
        return 1.0 + self.lateral_scale * self.lateral[np.asarray(cell_ids)]

    def densities(self, cell_ids, r_mid_km) -> np.ndarray:
        """Density (g/cm^3) per segment."""
        return self.radial_density(cell_ids, r_mid_km) * self.lateral_factor(cell_ids)

    def electron_fraction(self, r_mid_km) -> np.ndarray:
        return electron_fraction_profile(r_mid_km, self.ye_kind)

    def cell_densities(self) -> np.ndarray:
        """Density at every cell centre (n_cells,)."""
        ids = np.arange(self.grid.n_cells)
        r_c, _, _ = self.grid.centers(ids)
        return self.densities(ids, r_c)

    def describe(self) -> Dict:
        return {"name": self.name,
                "radial": self.radial if isinstance(self.radial, str)
                else ("callable" if callable(self.radial) else "shell_array"),
                "lateral_scale": float(self.lateral_scale),
                "has_lateral": self.lateral is not None,
                "ye_kind": self.ye_kind,
                "radial_scale": [float(x) for x in self.radial_scale]}


# --------------------------------------------------------------------------
# anomaly templates (fractional density perturbations per cell)
# --------------------------------------------------------------------------
def angular_distance_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    la1, lo1, la2, lo2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    c = (np.sin(la1) * np.sin(la2)
         + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2))
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


def blob_anomaly(grid: TesseroidGrid, lat_deg: float, lon_deg: float,
                 r_min_km: float, r_max_km: float, radius_deg: float,
                 amplitude: float, profile: str = "top_hat") -> np.ndarray:
    """A compact anomaly: cells whose centres lie within ``radius_deg`` of
    (lat, lon) and whose shell centre radius lies in [r_min, r_max].

    ``amplitude`` is the fractional density perturbation (0.01 = +1%).
    ``profile`` = 'top_hat' or 'gaussian' (radius_deg is then 1 sigma).
    """
    r_c, lat_c, lon_c = grid.centers()
    dist = angular_distance_deg(lat_c, lon_c, lat_deg, lon_deg)
    in_r = (r_c >= r_min_km) & (r_c <= r_max_km)
    out = np.zeros(grid.n_cells)
    if profile == "top_hat":
        out[in_r & (dist <= radius_deg)] = amplitude
    elif profile == "gaussian":
        out[in_r] = amplitude * np.exp(-0.5 * (dist[in_r] / radius_deg) ** 2)
    else:
        raise ValueError(profile)
    return out


def shell_anomaly(grid: TesseroidGrid, shell_index: int, amplitude: float) -> np.ndarray:
    """A whole-shell fractional perturbation (a *radial* signal, for reference)."""
    out = np.zeros(grid.n_cells)
    out[grid.shell_of(np.arange(grid.n_cells)) == shell_index] = amplitude
    return out


def anomaly_from_config(grid: TesseroidGrid, cfg: Dict) -> np.ndarray:
    """Sum of anomaly templates described in a config list."""
    total = np.zeros(grid.n_cells)
    for a in cfg:
        kind = a.get("kind", "blob")
        if kind == "blob":
            total += blob_anomaly(grid, a["lat_deg"], a["lon_deg"], a["r_min_km"],
                                  a["r_max_km"], a["radius_deg"], a["amplitude"],
                                  a.get("profile", "top_hat"))
        elif kind == "shell":
            total += shell_anomaly(grid, int(a["shell_index"]), a["amplitude"])
        elif kind == "npz":
            with np.load(a["path"]) as z:
                total += np.asarray(z[a.get("key", "lateral")], float)
        else:
            raise ValueError(f"unknown anomaly kind {kind!r}")
    return total


def model_from_config(grid: TesseroidGrid, cfg: Dict) -> DensityModel:
    """Build a DensityModel from a config dict.

    cfg keys: radial ('prem' | 'shell_mean_prem'), anomalies (list),
    lateral_scale, ye_kind, name.
    """
    radial = cfg.get("radial", "shell_mean_prem")
    if radial == "shell_mean_prem":
        radial = shell_mean_prem(grid)
    elif radial != "prem":
        raise ValueError(f"unknown radial spec {radial!r}")
    lateral = anomaly_from_config(grid, cfg.get("anomalies", [])) if cfg.get("anomalies") else None
    return DensityModel(grid, radial=radial, lateral=lateral,
                        lateral_scale=float(cfg.get("lateral_scale", 1.0)),
                        ye_kind=cfg.get("ye_kind", "core_mantle"),
                        name=cfg.get("name", "model"))
