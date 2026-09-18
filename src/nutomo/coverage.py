"""Coverage and resolution: what four detectors can and cannot see.

Two levels, both pure linear algebra once the rays and Jacobians exist:

1. **Geometry only** - ray counts and path length per cell, number of
   detectors seeing each cell, and the damped least-squares resolution
   matrix of the path-length kernel G (rays x cells).  Costs nothing and
   already shows the null space.

2. **Physics-weighted** - the Fisher information over per-cell fractional
   density perturbations, F = sum_bins g g^T / N with g = dN/d(delta),
   including cross terms with oscillation, flux and radial-profile
   nuisances.  From it: marginalised per-cell errors, the resolution
   matrix under a Gaussian prior, and template amplitude sensitivities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .density import DensityModel
from .detectors import CHANNELS
from .probability import OscParams
from .rates import RateCalculator, RateResult
from .raytrace import RayTable
from .tesseroids import TesseroidGrid


# ---------------------------------------------------------------------------
# 1. geometry
# ---------------------------------------------------------------------------
def geometry_coverage(rt: RayTable, grid: TesseroidGrid) -> Dict[str, np.ndarray]:
    """Per-cell ray counts, path length, and detector multiplicity."""
    n = grid.n_cells
    hits = np.zeros(n, np.int64)
    path = np.zeros(n)
    det_mask = np.zeros(n, np.int64)
    for i, (w, c, _) in rt:
        u = np.unique(c)
        hits[u] += 1
        np.add.at(path, c, w)
        det_mask[u] |= (1 << int(rt.det_index[i]))
    n_det = np.array([bin(int(m)).count("1") for m in det_mask])
    ir = grid.shell_of(np.arange(n))
    shell = {
        "n_cells": np.bincount(ir, minlength=grid.n_r),
        "n_hit": np.bincount(ir, weights=(hits > 0), minlength=grid.n_r),
        "n_hit_2det": np.bincount(ir, weights=(n_det >= 2), minlength=grid.n_r),
        "rays_per_hit_cell": np.divide(np.bincount(ir, weights=hits, minlength=grid.n_r),
                                       np.maximum(np.bincount(ir, weights=(hits > 0), minlength=grid.n_r), 1)),
    }
    return {"hits": hits, "path_km": path, "n_det": n_det, "det_mask": det_mask, "shell": shell}


def path_length_kernel(rt: RayTable, active_cells: np.ndarray) -> sp.csr_matrix:
    """Sparse G (n_rays x n_active) of path length (km) in each cell."""
    cell_map = -np.ones(int(rt.cell_ids.max()) + 1, np.int64)
    cell_map[active_cells] = np.arange(active_cells.size)
    rows = np.repeat(np.arange(rt.n_rays), np.diff(rt.offsets))
    cols = cell_map[rt.cell_ids]
    ok = cols >= 0
    G = sp.coo_matrix((rt.widths_km[ok], (rows[ok], cols[ok])),
                      shape=(rt.n_rays, active_cells.size))
    return G.tocsr()


def kernel_resolution(G: sp.spmatrix, damping_rel: float = 1e-3,
                      null_tol: float = 1e-10) -> Dict[str, np.ndarray]:
    """Damped least-squares resolution of a linear kernel.

    R = (G^T G + lam I)^-1 G^T G with lam = damping_rel * max eigenvalue.
    Returns diag(R), the eigenvalues of G^T G and the null-space
    dimension (eigenvalues below null_tol * max).
    """
    A = (G.T @ G).toarray() if sp.issparse(G) else G.T @ G
    A = 0.5 * (A + A.T)
    evals, evecs = np.linalg.eigh(A)
    lam = damping_rel * evals.max()
    R = evecs @ ((evals / (evals + lam))[:, None] * evecs.T)
    n_null = int(np.sum(evals < null_tol * evals.max()))
    return {"diag": np.diag(R).copy(), "eigenvalues": evals[::-1], "n_null": n_null,
            "damping": lam}


# ---------------------------------------------------------------------------
# 2. physics-weighted Fisher information
# ---------------------------------------------------------------------------
def active_cell_map(rt: RayTable, grid: TesseroidGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Cells crossed by at least one ray, and a global->local index map."""
    active = rt.unique_cells()
    cmap = -np.ones(grid.n_cells, np.int64)
    cmap[active] = np.arange(active.size)
    return active, cmap


def truth_jacobian_sparse(rc: RateCalculator, res: RateResult, idet: int,
                          cell_map: np.ndarray) -> sp.csr_matrix:
    """d(truth counts)/d(delta_cell) for one detector, (n_bins x n_active)."""
    rt = rc.rays
    n_active = int(cell_map.max()) + 1
    rows, cols, data = [], [], []
    nE = rc.detectors[idet].n_e_bins
    nch = len(CHANNELS)
    for icz_s in range(rt.n_cosz):
        for iaz_s in range(rt.n_az):
            i = rt.ray_index(idet, icz_s, iaz_s)
            rr = res.ray_jacobians[i]
            if rr.cells is None:
                continue
            loc = cell_map[rr.cells]
            _, icz, iaz = rc.ray_bin(i)
            base = ((icz * rc.binning.n_az + iaz) * nE) * nch
            # dcounts (nE, nch, m) -> rows base + ie*nch + ich
            r = base + np.arange(nE * nch)[:, None]
            rows.append(np.broadcast_to(r, (nE * nch, loc.size)).ravel())
            cols.append(np.broadcast_to(loc[None, :], (nE * nch, loc.size)).ravel())
            data.append(rr.dcounts_dfrac.reshape(nE * nch, loc.size).ravel())
    J = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(rc.n_bins(idet), n_active))
    return J.tocsr()


def response_matrix_sparse(rc: RateCalculator, idet: int, prune: float = 1e-8) -> sp.csr_matrix:
    """Sparse Kronecker response (n_bins x n_bins) = K_cz x K_az x K_E x I_ch."""
    K = rc.kernels[idet]
    def sparsify(M):
        M = np.where(np.abs(M) < prune, 0.0, M)
        return sp.csr_matrix(M)
    R = sp.kron(sp.kron(sp.kron(sparsify(K["cosz"]), sparsify(K["az"])), sparsify(K["E"])),
                sp.identity(len(CHANNELS), format="csr"), format="csr")
    return R


@dataclass
class FisherAccumulator:
    """Fisher information over (active cells | nuisances)."""
    active_cells: np.ndarray
    nuisance_names: Tuple[str, ...]

    def __post_init__(self):
        nc, nn = self.active_cells.size, len(self.nuisance_names)
        self.F_cc = np.zeros((nc, nc))
        self.F_cn = np.zeros((nc, nn))
        self.F_nn = np.zeros((nn, nn))
        self.n_bins_used = 0
        self.n_events = 0.0

    def add(self, G: sp.spmatrix, g_nuis: Optional[np.ndarray], N: np.ndarray,
            min_expected: float = 1e-3) -> None:
        """Accumulate bins: G (n_bins x n_cells) sparse, g_nuis (n_bins x n_nuis), N (n_bins,)."""
        N = np.asarray(N, float)
        w = np.where(N > min_expected, 1.0 / np.maximum(N, min_expected), 0.0)
        W = sp.diags(w)
        Gw = W @ G
        self.F_cc += (G.T @ Gw).toarray()
        if g_nuis is not None and g_nuis.size:
            self.F_cn += np.asarray(G.T @ (w[:, None] * g_nuis))
            self.F_nn += g_nuis.T @ (w[:, None] * g_nuis)
        self.n_bins_used += int(np.sum(w > 0))
        self.n_events += float(N[w > 0].sum())

    # -- derived quantities --------------------------------------------------
    def nuisance_prior_matrix(self, prior_sigma: Dict[str, float]) -> np.ndarray:
        P = np.zeros_like(self.F_nn)
        for k, name in enumerate(self.nuisance_names):
            s = prior_sigma.get(name)
            if s is not None and s > 0:
                P[k, k] = 1.0 / s ** 2
        return P

    def marginalized_cell_fisher(self, prior_sigma: Optional[Dict[str, float]] = None) -> np.ndarray:
        """F_cc - F_cn (F_nn + P_n)^-1 F_nc  (Schur complement)."""
        if self.F_nn.size == 0:
            return self.F_cc.copy()
        A = self.F_nn + self.nuisance_prior_matrix(prior_sigma or {})
        A = 0.5 * (A + A.T) + 1e-300 * np.eye(A.shape[0])
        X = np.linalg.solve(A, self.F_cn.T)
        return self.F_cc - self.F_cn @ X

    def cell_errors(self, prior_sigma_cell: float,
                    nuisance_prior: Optional[Dict[str, float]] = None,
                    marginalize_nuisance: bool = True) -> np.ndarray:
        """1-sigma error on each cell's fractional density with a Gaussian
        prior of width prior_sigma_cell on every cell (a damping)."""
        F = self.marginalized_cell_fisher(nuisance_prior) if marginalize_nuisance else self.F_cc
        P = F + np.eye(F.shape[0]) / prior_sigma_cell ** 2
        cov = np.linalg.inv(0.5 * (P + P.T))
        return np.sqrt(np.maximum(np.diag(cov), 0.0))

    def resolution(self, prior_sigma_cell: float,
                   nuisance_prior: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Posterior resolution matrix R = (F + P)^-1 F under the cell prior."""
        F = self.marginalized_cell_fisher(nuisance_prior)
        F = 0.5 * (F + F.T)
        evals, evecs = np.linalg.eigh(F)
        lam = 1.0 / prior_sigma_cell ** 2
        R = evecs @ ((np.maximum(evals, 0) / (np.maximum(evals, 0) + lam))[:, None] * evecs.T)
        n_null = int(np.sum(evals < 1e-10 * max(evals.max(), 1e-300)))
        return {"diag": np.diag(R).copy(), "eigenvalues": evals[::-1], "n_null": n_null,
                "n_resolved": int(np.sum(evals > lam))}

    def template_sensitivity(self, template: np.ndarray,
                             nuisance_prior: Optional[Dict[str, float]] = None,
                             marginalize_nuisance: bool = True) -> float:
        """1-sigma error on the amplitude A of delta_cell = A * template_cell,
        with every other cell fixed and the nuisances profiled."""
        t = np.asarray(template, float)
        F = self.marginalized_cell_fisher(nuisance_prior) if marginalize_nuisance else self.F_cc
        faa = float(t @ F @ t)
        return np.inf if faa <= 0 else 1.0 / np.sqrt(faa)

    def nuisance_errors(self, nuisance_prior: Optional[Dict[str, float]] = None,
                        cells_free: bool = False, prior_sigma_cell: float = 0.05) -> np.ndarray:
        """Errors on the nuisances (cells fixed, or free under the cell prior)."""
        A = self.F_nn + self.nuisance_prior_matrix(nuisance_prior or {})
        if cells_free:
            Fc = self.F_cc + np.eye(self.F_cc.shape[0]) / prior_sigma_cell ** 2
            A = A - self.F_cn.T @ np.linalg.solve(Fc, self.F_cn)
        return np.sqrt(np.maximum(np.diag(np.linalg.inv(A)), 0.0))


# ---------------------------------------------------------------------------
# driver: nuisance gradients and full assembly
# ---------------------------------------------------------------------------
OSC_STEPS = {"dm31": 2.0e-5, "s23sq": 0.01, "s13sq": 0.0005, "dcp": 0.2, "dm21": 1e-6, "s12sq": 0.005}
FLUX_STEPS = {"flux_norm": 0.02, "delta_gamma": 0.02, "nue_numu_ratio": 0.02, "nubar_nu_ratio": 0.02}


def flatten_reco(res: RateResult, idet: int) -> np.ndarray:
    return res.reco[idet].ravel()


def nuisance_gradients(rc: RateCalculator, model: DensityModel, params: OscParams,
                       flux_nuis: Dict[str, float], osc_names: Sequence[str],
                       flux_names: Sequence[str], progress=None) -> List[np.ndarray]:
    """Central finite differences of reco counts w.r.t. oscillation and flux
    nuisances.  Returns per detector an array (n_bins, n_osc + n_flux)."""
    n_det = len(rc.detectors)
    cols = [[] for _ in range(n_det)]
    for name in osc_names:
        h = OSC_STEPS[name]
        up = rc.compute(model, params.replace(**{name: getattr(params, name) + h}), flux_nuis)
        dn = rc.compute(model, params.replace(**{name: getattr(params, name) - h}), flux_nuis)
        for d in range(n_det):
            cols[d].append((flatten_reco(up, d) - flatten_reco(dn, d)) / (2 * h))
        if progress:
            progress(name)
    for name in flux_names:
        h = FLUX_STEPS[name]
        base = dict(flux_nuis)
        v0 = base.get(name, 1.0 if name != "delta_gamma" else 0.0)
        up = rc.compute(model, params, {**base, name: v0 + h})
        dn = rc.compute(model, params, {**base, name: v0 - h})
        for d in range(n_det):
            cols[d].append((flatten_reco(up, d) - flatten_reco(dn, d)) / (2 * h))
        if progress:
            progress(name)
    return [np.column_stack(c) if c else np.zeros((rc.n_bins(d), 0)) for d, c in enumerate(cols)]


def radial_scale_gradients(G_reco: sp.spmatrix, active_cells: np.ndarray, grid: TesseroidGrid,
                           model: DensityModel) -> np.ndarray:
    """d(reco counts)/d(radial_scale[shell]) from the cell Jacobian.

    rho = radial*scale*(1+delta) so dN/dscale_s = sum_{cells in s} dN/ddelta * (1+delta)/scale_s.
    Returns a dense (n_bins, n_r) array.
    """
    shell = grid.shell_of(active_cells)
    fac = np.ones(active_cells.size)
    if model.lateral is not None:
        fac = 1.0 + model.lateral_scale * model.lateral[active_cells]
    fac = fac / model.radial_scale[shell]
    M = sp.coo_matrix((fac, (np.arange(active_cells.size), shell)),
                      shape=(active_cells.size, grid.n_r)).tocsr()
    return np.asarray((G_reco @ M).todense())


def assemble_fisher(rc: RateCalculator, grid: TesseroidGrid, model: DensityModel,
                    params: OscParams, flux_nuis: Dict[str, float],
                    osc_names: Sequence[str], flux_names: Sequence[str],
                    include_radial_scale: bool = True, min_expected: float = 1e-3,
                    res: Optional[RateResult] = None, log=print) -> Tuple[FisherAccumulator, RateResult]:
    """Full Fisher assembly for the model: cells + nuisances, smearing applied."""
    active, cmap = active_cell_map(rc.rays, grid)
    if res is None:
        log(f"  computing rates + Jacobians for {rc.rays.n_rays} rays ...")
        res = rc.compute(model, params, flux_nuis, with_jacobian=True)
    radial_names = tuple(f"radial_scale_{k}" for k in range(grid.n_r)) if include_radial_scale else ()
    names = tuple(osc_names) + tuple(flux_names) + radial_names
    acc = FisherAccumulator(active, names)
    log(f"  nuisance gradients ({len(osc_names)} osc + {len(flux_names)} flux) ...")
    g_fd = nuisance_gradients(rc, model, params, flux_nuis, osc_names, flux_names)
    for idet in range(len(rc.detectors)):
        J = truth_jacobian_sparse(rc, res, idet, cmap)
        R = response_matrix_sparse(rc, idet)
        G = (R @ J).tocsr()
        N = flatten_reco(res, idet)
        g_n = g_fd[idet]
        if include_radial_scale:
            g_n = np.column_stack([g_n, radial_scale_gradients(G, active, grid, model)])
        acc.add(G, g_n, N, min_expected)
        log(f"  detector {rc.detectors[idet].name}: {G.shape[0]} bins, {G.nnz} nonzeros, "
            f"{N.sum():.3g} events")
    return acc, res
