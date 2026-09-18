#!/usr/bin/env python3
"""Geometry-only coverage and resolution analysis (no oscillation physics).

Counts how many chords cross each tesseroid, from how many detectors, and
computes the damped least-squares resolution matrix of the path-length
kernel.  This is the honest statement of what four sites can see before
any neutrino physics enters.

Usage: python scripts/run_coverage.py --grid configs/grid_10deg.yaml --out results/coverage_10deg
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nutomo import config as C
from nutomo import coverage as cov


def plot_maps(grid, fields, titles, out_png, cmap="viridis", log=False, vmin=None, vmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(fields)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), squeeze=False)
    lon_e, lat_e = grid.lon_edges, grid.lat_edges
    for ax, f, t in zip(axes[:, 0], fields, titles):
        v = np.log10(np.maximum(f, 1e-300)) if log else f
        if log:
            v = np.where(f > 0, v, np.nan)
        im = ax.pcolormesh(lon_e, lat_e, v.reshape(grid.n_lat, grid.n_lon), cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="flat")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="configs/grid_10deg.yaml")
    ap.add_argument("--detectors", default="configs/detectors.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--damping", type=float, default=1e-3, help="relative damping for the kernel resolution")
    ap.add_argument("--max-active", type=int, default=12000, help="skip the dense resolution above this many cells")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    grid = C.load_grid(a.grid)
    dets, binning = C.load_sites(a.detectors)
    out = C.resolve(a.out or f"results/coverage_{grid.name}")
    os.makedirs(out, exist_ok=True)
    rt = C.get_ray_table(grid, dets, binning, a.cache_dir)

    t0 = time.time()
    g = cov.geometry_coverage(rt, grid)
    hits, path, n_det = g["hits"], g["path_km"], g["n_det"]
    active = np.flatnonzero(hits > 0)
    print(f"geometry coverage in {time.time() - t0:.1f} s")
    print(f"cells: {grid.n_cells}   crossed by >=1 ray: {active.size} "
          f"({100 * active.size / grid.n_cells:.1f}%)   by >=2 detectors: {np.sum(n_det >= 2)}   "
          f"by all {len(dets)}: {np.sum(n_det == len(dets))}")
    print("\nper shell (r_lo-r_hi km): cells  hit  frac  2det_frac  rays/hit_cell")
    sh = g["shell"]
    for k in range(grid.n_r):
        print(f"  {grid.r_edges[k]:7.1f}-{grid.r_edges[k + 1]:7.1f}: {int(sh['n_cells'][k]):6d} "
              f"{int(sh['n_hit'][k]):6d} {sh['n_hit'][k] / sh['n_cells'][k]:6.2f}   "
              f"{sh['n_hit_2det'][k] / sh['n_cells'][k]:6.2f}   {sh['rays_per_hit_cell'][k]:8.1f}")

    result = {"hits": hits, "path_km": path, "n_det": n_det, "det_mask": g["det_mask"],
              "active": active}
    summary = {"grid": grid.to_dict(), "n_cells": grid.n_cells, "n_rays": rt.n_rays,
               "n_active": int(active.size), "n_2det": int(np.sum(n_det >= 2)),
               "n_alldet": int(np.sum(n_det == len(dets))),
               "shell": {k: [float(x) for x in v] for k, v in sh.items()},
               "sites": rt.meta["sites"], "provenance": C.provenance(ray_table=rt.meta)}

    if active.size <= a.max_active:
        t0 = time.time()
        G = cov.path_length_kernel(rt, active)
        r = cov.kernel_resolution(G, a.damping)
        diag = np.zeros(grid.n_cells)
        diag[active] = r["diag"]
        result["resolution_diag"] = diag
        result["kernel_eigenvalues"] = r["eigenvalues"]
        ev = r["eigenvalues"]
        n_eff = int(np.sum(ev > 1e-6 * ev.max()))
        print(f"\npath-length kernel: {G.shape[0]} rays x {G.shape[1]} cells, "
              f"resolution in {time.time() - t0:.1f} s")
        print(f"  eigenvalue spectrum: {n_eff} of {active.size} directions above 1e-6 of max; "
              f"null space (<1e-10): {r['n_null']}")
        print(f"  cells with resolution diag > 0.5: {int(np.sum(r['diag'] > 0.5))}; > 0.9: {int(np.sum(r['diag'] > 0.9))}")
        ir = grid.shell_of(active)
        print("  mean resolution diag per shell:",
              " ".join(f"{np.mean(r['diag'][ir == k]) if np.any(ir == k) else 0:.2f}" for k in range(grid.n_r)))
        summary.update({"n_eff_directions": n_eff, "n_null": int(r["n_null"]),
                        "n_res_gt_0p5": int(np.sum(r["diag"] > 0.5)),
                        "mean_res_per_shell": [float(np.mean(r["diag"][ir == k])) if np.any(ir == k) else 0.0
                                               for k in range(grid.n_r)]})
    else:
        print(f"\nskipping dense resolution: {active.size} active cells > --max-active {a.max_active}")

    np.savez_compressed(os.path.join(out, "coverage.npz"), **result)
    C.dump_json(summary, os.path.join(out, "summary.json"))

    if not a.no_plots:
        shells = {"outer core (2400-2900)": 4, "lowermost mantle (3480-4000)": 6,
                  "mid mantle (5000-5400)": 9, "upper mantle (5971-6151)": 12}
        per_shell = lambda f, k: f.reshape(grid.n_r, -1)[k]
        plot_maps(grid, [per_shell(hits.astype(float), k) for k in shells.values()],
                  [f"rays crossing each cell, {n}" for n in shells], os.path.join(out, "hits_maps.png"), log=True)
        plot_maps(grid, [per_shell(n_det.astype(float), k) for k in shells.values()],
                  [f"number of detectors seeing each cell, {n}" for n in shells],
                  os.path.join(out, "ndet_maps.png"), cmap="plasma", vmin=0, vmax=len(dets))
        if "resolution_diag" in result:
            plot_maps(grid, [per_shell(result["resolution_diag"], k) for k in shells.values()],
                      [f"path-length kernel resolution diag, {n}" for n in shells],
                      os.path.join(out, "resolution_maps.png"), cmap="magma", vmin=0, vmax=1)
        print(f"\nwrote {out}/(coverage.npz, summary.json, *.png)")


if __name__ == "__main__":
    main()
