"""Tesseroid grid: radial shells x latitude bands x longitude sectors.

A tesseroid is the volume bounded by two spheres, two cones of constant
latitude and two half-planes of constant longitude.  Cells are indexed
``(ir, ilat, ilon)`` and flattened with ``ilon`` varying fastest.

This module imports nothing from NuOscProbExact.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

R_EARTH_KM = 6371.0

#: PREM discontinuities (km) - a sensible default set of shell edges.
PREM_BOUNDARIES_KM = (1221.5, 3480.0, 5701.0, 5771.0, 5971.0, 6151.0,
                      6346.6, 6356.0, 6368.0)

#: A default radial split that keeps PREM's major boundaries and subdivides
#: the thick shells so that depth resolution is roughly uniform (~500 km).
DEFAULT_R_EDGES_KM = (0.0, 600.0, 1221.5, 1800.0, 2400.0, 2900.0, 3480.0,
                      4000.0, 4500.0, 5000.0, 5400.0, 5701.0, 5971.0,
                      6151.0, 6371.0)


@dataclass(frozen=True)
class TesseroidGrid:
    """A regular tesseroid grid.

    Parameters
    ----------
    r_edges_km : ascending radii of shell edges, from 0 to the surface.
    dlat_deg, dlon_deg : angular cell sizes; must divide 180 and 360.
    name : optional label carried into cache provenance.
    """

    r_edges_km: Tuple[float, ...]
    dlat_deg: float
    dlon_deg: float
    name: str = "grid"
    lat_edges_deg: Tuple[float, ...] = field(init=False, repr=False)
    lon_edges_deg: Tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self):
        r = np.asarray(self.r_edges_km, dtype=float)
        if r.ndim != 1 or r.size < 2 or np.any(np.diff(r) <= 0):
            raise ValueError("r_edges_km must be ascending with >= 2 entries")
        if abs(r[0]) > 1e-9 or abs(r[-1] - R_EARTH_KM) > 1e-6:
            raise ValueError("r_edges_km must run from 0 to R_EARTH_KM")
        nlat = 180.0 / self.dlat_deg
        nlon = 360.0 / self.dlon_deg
        if abs(nlat - round(nlat)) > 1e-9 or abs(nlon - round(nlon)) > 1e-9:
            raise ValueError("dlat_deg must divide 180 and dlon_deg must divide 360")
        object.__setattr__(self, "r_edges_km", tuple(float(x) for x in r))
        object.__setattr__(self, "lat_edges_deg",
                           tuple(np.linspace(-90.0, 90.0, int(round(nlat)) + 1)))
        object.__setattr__(self, "lon_edges_deg",
                           tuple(np.linspace(-180.0, 180.0, int(round(nlon)) + 1)))

    # ---- sizes ---------------------------------------------------------
    @property
    def n_r(self) -> int:
        return len(self.r_edges_km) - 1

    @property
    def n_lat(self) -> int:
        return len(self.lat_edges_deg) - 1

    @property
    def n_lon(self) -> int:
        return len(self.lon_edges_deg) - 1

    @property
    def n_cells(self) -> int:
        return self.n_r * self.n_lat * self.n_lon

    @property
    def r_edges(self) -> np.ndarray:
        return np.asarray(self.r_edges_km)

    @property
    def lat_edges(self) -> np.ndarray:
        return np.asarray(self.lat_edges_deg)

    @property
    def lon_edges(self) -> np.ndarray:
        return np.asarray(self.lon_edges_deg)

    # ---- indexing ------------------------------------------------------
    def cell_index(self, ir, ilat, ilon) -> np.ndarray:
        ir = np.asarray(ir)
        ilat = np.asarray(ilat)
        ilon = np.asarray(ilon)
        return (ir * self.n_lat + ilat) * self.n_lon + ilon

    def unravel(self, cell_ids) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cell_ids = np.asarray(cell_ids)
        ilon = cell_ids % self.n_lon
        rest = cell_ids // self.n_lon
        ilat = rest % self.n_lat
        ir = rest // self.n_lat
        return ir, ilat, ilon

    def shell_of(self, cell_ids) -> np.ndarray:
        return np.asarray(cell_ids) // (self.n_lat * self.n_lon)

    def locate(self, r_km, lat_deg, lon_deg) -> np.ndarray:
        """Flat cell ids for points given in spherical coordinates.

        Points at exactly r = R_EARTH land in the top shell; longitudes are
        wrapped into [-180, 180).
        """
        r = np.asarray(r_km, dtype=float)
        lat = np.asarray(lat_deg, dtype=float)
        lon = (np.asarray(lon_deg, dtype=float) + 180.0) % 360.0 - 180.0
        ir = np.clip(np.searchsorted(self.r_edges, r, side="right") - 1,
                     0, self.n_r - 1)
        ilat = np.clip(np.floor((lat + 90.0) / self.dlat_deg).astype(int),
                       0, self.n_lat - 1)
        ilon = np.clip(np.floor((lon + 180.0) / self.dlon_deg).astype(int),
                       0, self.n_lon - 1)
        return self.cell_index(ir, ilat, ilon)

    # ---- geometry ------------------------------------------------------
    def centers(self, cell_ids=None):
        """(r_km, lat_deg, lon_deg) of cell centres (volume-naive midpoints)."""
        if cell_ids is None:
            cell_ids = np.arange(self.n_cells)
        ir, ilat, ilon = self.unravel(cell_ids)
        r_e, la_e, lo_e = self.r_edges, self.lat_edges, self.lon_edges
        return (0.5 * (r_e[ir] + r_e[ir + 1]),
                0.5 * (la_e[ilat] + la_e[ilat + 1]),
                0.5 * (lo_e[ilon] + lo_e[ilon + 1]))

    def volumes(self, cell_ids=None) -> np.ndarray:
        """Cell volumes in km^3."""
        if cell_ids is None:
            cell_ids = np.arange(self.n_cells)
        ir, ilat, ilon = self.unravel(cell_ids)
        r_e = self.r_edges
        la = np.deg2rad(self.lat_edges)
        dlon = np.deg2rad(self.dlon_deg)
        return ((r_e[ir + 1] ** 3 - r_e[ir] ** 3) / 3.0
                * (np.sin(la[ilat + 1]) - np.sin(la[ilat])) * dlon)

    def shell_volumes(self) -> np.ndarray:
        r_e = self.r_edges
        return 4.0 * np.pi / 3.0 * (r_e[1:] ** 3 - r_e[:-1] ** 3)

    # ---- provenance ----------------------------------------------------
    def to_dict(self) -> Dict:
        return {"name": self.name,
                "r_edges_km": list(self.r_edges_km),
                "dlat_deg": float(self.dlat_deg),
                "dlon_deg": float(self.dlon_deg)}

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: Dict) -> "TesseroidGrid":
        r_edges = d.get("r_edges_km", DEFAULT_R_EDGES_KM)
        return cls(tuple(float(x) for x in r_edges),
                   float(d["dlat_deg"]), float(d["dlon_deg"]),
                   name=str(d.get("name", "grid")))

    @classmethod
    def from_yaml(cls, path: str) -> "TesseroidGrid":
        import yaml
        with open(path) as fh:
            d = yaml.safe_load(fh)
        return cls.from_dict(d.get("grid", d))


def lateral_average(grid: TesseroidGrid, values: np.ndarray,
                    weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Volume-weighted lateral average per shell of a per-cell field.

    Returns an array of shape (n_r,).  This is the "laterally averaged"
    third leg of the analysis: the same grid, lateral structure removed.
    """
    values = np.asarray(values, dtype=float)
    w = grid.volumes() if weights is None else np.asarray(weights, float)
    ir = grid.shell_of(np.arange(grid.n_cells))
    num = np.bincount(ir, weights=w * values, minlength=grid.n_r)
    den = np.bincount(ir, weights=w, minlength=grid.n_r)
    return num / den
