"""Straight-line chords through a tesseroid grid.

Given a detector position and an arrival direction, the chord is cut at
every crossing of a shell sphere, a latitude cone or a longitude
half-plane.  The output is the list of segment widths (km), the flat cell
id of each segment and the radius at each segment midpoint, ordered along
the neutrino's direction of travel (entry point first, detector last).

Conventions
-----------
* ``cosz`` is the cosine of the zenith angle of the *arrival* direction:
  ``cosz = -1`` is a neutrino coming straight up through the centre of
  the Earth, ``cosz = 0`` is horizontal.  Only ``cosz < 0`` chords cross
  a meaningful amount of Earth.
* ``az_deg`` is the azimuth of the arrival direction (where the neutrino
  comes *from*), measured clockwise from geographic north through east.
* The Earth is a sphere of radius ``R_EARTH_KM``; geodetic latitudes are
  treated as geocentric.

This module imports nothing from NuOscProbExact.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .tesseroids import TesseroidGrid, R_EARTH_KM

#: Segments shorter than this (km) are merged with their neighbour.
MIN_SEGMENT_KM = 1e-6


# ---------------------------------------------------------------------------
# basic geometry
# ---------------------------------------------------------------------------
def latlon_to_xyz(lat_deg, lon_deg, r_km=R_EARTH_KM) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return np.array([r_km * np.cos(lat) * np.cos(lon),
                     r_km * np.cos(lat) * np.sin(lon),
                     r_km * np.sin(lat)])


def xyz_to_rlatlon(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=float)
    r = np.sqrt(np.sum(xyz ** 2, axis=-1))
    lat = np.rad2deg(np.arcsin(np.clip(xyz[..., 2] / np.where(r > 0, r, 1.0), -1, 1)))
    lon = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return r, lat, lon


def local_frame(lat_deg, lon_deg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors (up, north, east) at a site."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    return up, north, east


def source_direction(lat_deg, lon_deg, cosz, az_deg) -> np.ndarray:
    """Unit vector pointing from the detector towards the neutrino source."""
    up, north, east = local_frame(lat_deg, lon_deg)
    sinz = np.sqrt(max(0.0, 1.0 - cosz ** 2))
    az = np.deg2rad(az_deg)
    return cosz * up + sinz * (np.cos(az) * north + np.sin(az) * east)


def chord_length(r_det_km: float, cosz: float, R=R_EARTH_KM) -> float:
    """Analytic path length inside the sphere from entry point to detector."""
    b = r_det_km * cosz
    disc = b * b + R * R - r_det_km * r_det_km
    return float(-b + np.sqrt(disc))


def chord_endpoints(det_xyz: np.ndarray, s_hat: np.ndarray, R=R_EARTH_KM):
    """Entry point, travel direction and length of the chord ending at det_xyz."""
    b = float(np.dot(det_xyz, s_hat))
    c = float(np.dot(det_xyz, det_xyz) - R * R)
    L = -b + np.sqrt(b * b - c)
    x_entry = det_xyz + L * s_hat
    d_hat = -s_hat
    return x_entry, d_hat, float(L)


# ---------------------------------------------------------------------------
# the tracer
# ---------------------------------------------------------------------------
@dataclass
class Chord:
    """One traced chord."""
    widths_km: np.ndarray      # (n,)
    cell_ids: np.ndarray       # (n,) flat tesseroid ids
    r_mid_km: np.ndarray       # (n,) radius at segment midpoints
    length_km: float
    x_entry: np.ndarray
    d_hat: np.ndarray

    @property
    def n_segments(self) -> int:
        return int(self.widths_km.size)

    def positions(self, t_km) -> np.ndarray:
        return self.x_entry[None, :] + np.asarray(t_km)[:, None] * self.d_hat[None, :]


def _quadratic_roots(A, B, C):
    """Real roots of A t^2 + B t + C = 0, vectorised over the last axis.

    Returns an array of shape (..., 2) with NaN where no real root exists.
    Linear cases (A ~ 0) return one root and a NaN.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    out = np.full(A.shape + (2,), np.nan)
    lin = np.abs(A) < 1e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        # linear
        out[lin, 0] = np.where(np.abs(B[lin]) > 1e-14, -C[lin] / B[lin], np.nan)
        # quadratic
        q = ~lin
        disc = B[q] ** 2 - 4.0 * A[q] * C[q]
        ok = disc >= 0.0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        # numerically stable form
        sgnB = np.where(B[q] >= 0, 1.0, -1.0)
        qq = -0.5 * (B[q] + sgnB * sq)
        t1 = np.where(ok, qq / A[q], np.nan)
        t2 = np.where(ok & (np.abs(qq) > 1e-300), C[q] / qq, np.nan)
        # when qq == 0 (B == 0 and disc == 0) both roots are 0
        t2 = np.where(ok & (np.abs(qq) <= 1e-300), 0.0, t2)
        out[q, 0] = t1
        out[q, 1] = t2
    return out


def trace_chord(grid: TesseroidGrid, det_xyz: np.ndarray, s_hat: np.ndarray,
                R=R_EARTH_KM) -> Chord:
    """Trace the chord ending at ``det_xyz`` arriving from direction ``s_hat``."""
    x0, d, L = chord_endpoints(np.asarray(det_xyz, float), np.asarray(s_hat, float), R)
    ts: List[np.ndarray] = []

    # --- spheres r = R_k (skip r=0 and the surface) ----------------------
    r_k = grid.r_edges[1:-1]
    if r_k.size:
        b = float(np.dot(x0, d))
        c = float(np.dot(x0, x0)) - r_k ** 2
        roots = _quadratic_roots(np.ones_like(r_k), 2.0 * b * np.ones_like(r_k), c)
        ts.append(roots.ravel())

    # --- latitude cones (skip the poles) --------------------------------
    lat_k = grid.lat_edges[1:-1]
    if lat_k.size:
        x0x, x0y, x0z = x0
        dx, dy, dz = d
        eq = np.abs(lat_k) < 1e-12
        tan2 = np.tan(np.deg2rad(lat_k)) ** 2
        A = dz ** 2 - tan2 * (dx ** 2 + dy ** 2)
        B = 2.0 * (x0z * dz - tan2 * (x0x * dx + x0y * dy))
        C = x0z ** 2 - tan2 * (x0x ** 2 + x0y ** 2)
        # the equator is a plane z = 0
        A = np.where(eq, 0.0, A)
        B = np.where(eq, dz, B)
        C = np.where(eq, x0z, C)
        roots = _quadratic_roots(A, B, C)              # (n_lat-1, 2)
        z_at = x0z + roots * dz
        # keep the nappe on the same side as the cone's latitude
        sgn = np.sign(lat_k)[:, None]
        good = eq[:, None] | (np.sign(z_at) == sgn)
        ts.append(np.where(good, roots, np.nan).ravel())

    # --- longitude half-planes ------------------------------------------
    lon_k = grid.lon_edges[:-1]                        # -180 and 180 coincide
    if lon_k.size:
        lam = np.deg2rad(lon_k)
        s, cth = np.sin(lam), np.cos(lam)
        num = -(x0[0] * s - x0[1] * cth)
        den = d[0] * s - d[1] * cth
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(den) > 1e-14, num / den, np.nan)
        xt = x0[0] + t * d[0]
        yt = x0[1] + t * d[1]
        good = (xt * cth + yt * s) > 0.0
        ts.append(np.where(good, t, np.nan))

    t_all = np.concatenate(ts) if ts else np.array([])
    t_all = t_all[np.isfinite(t_all)]
    t_all = t_all[(t_all > 0.0) & (t_all < L)]
    t_all = np.concatenate([[0.0], np.sort(t_all), [L]])
    widths = np.diff(t_all)
    keep = widths > MIN_SEGMENT_KM
    # merge tiny segments by dropping their upper edge
    t_edges = np.concatenate([[0.0], t_all[1:][keep]])
    if t_edges[-1] != L:
        t_edges[-1] = L
    widths = np.diff(t_edges)
    t_mid = 0.5 * (t_edges[:-1] + t_edges[1:])
    pos = x0[None, :] + t_mid[:, None] * d[None, :]
    r_mid, lat_mid, lon_mid = xyz_to_rlatlon(pos)
    cell_ids = grid.locate(r_mid, lat_mid, lon_mid)
    return Chord(widths, cell_ids.astype(np.int64), r_mid, L, x0, d)


def trace_from_site(grid: TesseroidGrid, lat_deg: float, lon_deg: float,
                    depth_km: float, cosz: float, az_deg: float) -> Chord:
    det_xyz = latlon_to_xyz(lat_deg, lon_deg, R_EARTH_KM - depth_km)
    s_hat = source_direction(lat_deg, lon_deg, cosz, az_deg)
    return trace_chord(grid, det_xyz, s_hat)


# ---------------------------------------------------------------------------
# ray tables: many chords, cached
# ---------------------------------------------------------------------------
@dataclass
class RayTable:
    """Ragged storage for a set of traced chords.

    Rays are ordered detector-major, then cosz, then azimuth.  Segment data
    for ray ``i`` lives in ``[offsets[i], offsets[i+1])``.
    """
    det_index: np.ndarray       # (n_rays,)
    cosz: np.ndarray            # (n_rays,)
    az_deg: np.ndarray          # (n_rays,)
    length_km: np.ndarray       # (n_rays,)
    offsets: np.ndarray         # (n_rays+1,)
    widths_km: np.ndarray       # (n_seg_total,)
    cell_ids: np.ndarray        # (n_seg_total,)
    r_mid_km: np.ndarray        # (n_seg_total,)
    cosz_centers: np.ndarray    # (n_cosz,)
    az_centers: np.ndarray      # (n_az,)
    meta: Dict

    @property
    def n_rays(self) -> int:
        return int(self.det_index.size)

    @property
    def n_det(self) -> int:
        return int(self.meta.get("n_det", self.det_index.max() + 1))

    @property
    def n_cosz(self) -> int:
        return int(self.cosz_centers.size)

    @property
    def n_az(self) -> int:
        return int(self.az_centers.size)

    def ray_index(self, idet: int, icz: int, iaz: int) -> int:
        return (idet * self.n_cosz + icz) * self.n_az + iaz

    def segments(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.widths_km[a:b], self.cell_ids[a:b], self.r_mid_km[a:b]

    def __iter__(self):
        for i in range(self.n_rays):
            yield i, self.segments(i)

    def unique_cells(self) -> np.ndarray:
        return np.unique(self.cell_ids)

    # ---- I/O -----------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez_compressed(
            path, det_index=self.det_index, cosz=self.cosz, az_deg=self.az_deg,
            length_km=self.length_km, offsets=self.offsets, widths_km=self.widths_km,
            cell_ids=self.cell_ids, r_mid_km=self.r_mid_km,
            cosz_centers=self.cosz_centers, az_centers=self.az_centers,
            meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path: str) -> "RayTable":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return cls(z["det_index"], z["cosz"], z["az_deg"], z["length_km"],
                       z["offsets"], z["widths_km"], z["cell_ids"], z["r_mid_km"],
                       z["cosz_centers"], z["az_centers"], meta)


def ray_table_hash(grid: TesseroidGrid, sites: Sequence[Dict],
                   cosz_centers, az_centers) -> str:
    payload = {"grid": grid.to_dict(),
               "sites": [{k: s[k] for k in ("lat_deg", "lon_deg", "depth_km")}
                         for s in sites],
               "cosz": [float(x) for x in cosz_centers],
               "az": [float(x) for x in az_centers]}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_ray_table(grid: TesseroidGrid, sites: Sequence[Dict], cosz_centers,
                    az_centers, code_version: str = "unknown",
                    progress: Optional[callable] = None) -> RayTable:
    """Trace every (site, cosz, az) chord.

    ``sites`` is a sequence of dicts with ``lat_deg``, ``lon_deg``,
    ``depth_km`` (and any other keys, which are ignored here).
    """
    cosz_centers = np.asarray(cosz_centers, float)
    az_centers = np.asarray(az_centers, float)
    n_rays = len(sites) * cosz_centers.size * az_centers.size
    det_index = np.empty(n_rays, np.int32)
    cz = np.empty(n_rays)
    az = np.empty(n_rays)
    length = np.empty(n_rays)
    offsets = np.zeros(n_rays + 1, np.int64)
    W, C, RM = [], [], []
    i = 0
    for idet, s in enumerate(sites):
        det_xyz = latlon_to_xyz(s["lat_deg"], s["lon_deg"], R_EARTH_KM - s.get("depth_km", 0.0))
        for c in cosz_centers:
            for a in az_centers:
                s_hat = source_direction(s["lat_deg"], s["lon_deg"], c, a)
                ch = trace_chord(grid, det_xyz, s_hat)
                det_index[i] = idet
                cz[i] = c
                az[i] = a
                length[i] = ch.length_km
                offsets[i + 1] = offsets[i] + ch.n_segments
                W.append(ch.widths_km)
                C.append(ch.cell_ids)
                RM.append(ch.r_mid_km)
                i += 1
                if progress is not None and i % 500 == 0:
                    progress(i, n_rays)
    meta = {"grid": grid.to_dict(), "grid_hash": grid.config_hash(),
            "n_det": len(sites),
            "sites": [{k: s.get(k) for k in ("name", "lat_deg", "lon_deg", "depth_km")}
                      for s in sites],
            "table_hash": ray_table_hash(grid, sites, cosz_centers, az_centers),
            "code_version": code_version}
    return RayTable(det_index, cz, az, length, offsets,
                    np.concatenate(W), np.concatenate(C), np.concatenate(RM),
                    cosz_centers, az_centers, meta)
