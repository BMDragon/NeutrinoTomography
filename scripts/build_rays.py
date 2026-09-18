#!/usr/bin/env python3
"""Trace and cache the chords for a (grid, detectors, binning) combination.

Usage: python scripts/build_rays.py --grid configs/grid_10deg.yaml --detectors configs/detectors.yaml
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nutomo import config as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="configs/grid_10deg.yaml")
    ap.add_argument("--detectors", default="configs/detectors.yaml")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    grid = C.load_grid(a.grid)
    dets, binning = C.load_sites(a.detectors)
    t0 = time.time()
    rt = C.get_ray_table(grid, dets, binning, a.cache_dir, a.rebuild)
    print(f"grid {grid.name}: {grid.n_cells} cells; {rt.n_rays} rays; "
          f"{rt.widths_km.size / rt.n_rays:.1f} segments/ray; "
          f"{rt.unique_cells().size} cells touched; {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
