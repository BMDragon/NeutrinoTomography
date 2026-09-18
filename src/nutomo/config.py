"""Config loading shared by the scripts: grid, detectors, model, parameters, cache."""
from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import yaml

from . import __version__
from .density import DensityModel, model_from_config
from .detectors import Binning, Detector, load_detectors
from .probability import OscParams
from .rates import CrossSection, FluxModel, GaisserLikeFlux, TableFlux
from .raytrace import RayTable, build_ray_table, ray_table_hash
from .tesseroids import TesseroidGrid


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def code_version() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root(),
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=repo_root(),
                                stderr=subprocess.DEVNULL) != 0
        return f"{__version__}+{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return __version__


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_root(), path)


def load_yaml(path: str) -> Dict:
    with open(resolve(path)) as fh:
        return yaml.safe_load(fh)


def load_grid(path: str) -> TesseroidGrid:
    return TesseroidGrid.from_yaml(resolve(path))


def load_sites(path: str) -> Tuple[List[Detector], Binning]:
    return load_detectors(resolve(path))


def osc_params_from_config(cfg: Dict) -> OscParams:
    base = OscParams.inverted_ordering() if cfg.get("ordering", "normal") == "inverted" \
        else OscParams.normal_ordering()
    kw = {k: float(cfg[k]) for k in ("s12sq", "s23sq", "s13sq", "dm21", "dm31") if k in cfg}
    if "dcp_deg" in cfg:
        kw["dcp"] = np.deg2rad(float(cfg["dcp_deg"]))
    return base.replace(**kw)


def flux_from_config(cfg: Dict) -> FluxModel:
    kind = cfg.get("kind", "gaisser_like")
    if kind == "gaisser_like":
        allowed = {"A", "gamma", "nue_ratio_1gev", "nubar_ratio_mu", "nubar_ratio_e", "c_min", "e_pivot_gev"}
        return GaisserLikeFlux(**{k: v for k, v in cfg.items() if k in allowed})
    if kind == "table":
        return TableFlux.from_npz(resolve(cfg["path"]))
    raise ValueError(f"unknown flux kind {kind!r}")


def get_ray_table(grid: TesseroidGrid, detectors: List[Detector], binning: Binning,
                  cache_dir: str = "cache", rebuild: bool = False, log=print) -> RayTable:
    """Load the cached ray table for (grid, sites, binning) or build and cache it."""
    sites = [d.site() for d in detectors]
    h = ray_table_hash(grid, sites, binning.cosz_sub_centers, binning.az_sub_centers)
    cache_dir = resolve(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"rays_{grid.name}_{h}.npz")
    if os.path.exists(path) and not rebuild:
        rt = RayTable.load(path)
        log(f"loaded ray table {path} ({rt.n_rays} rays, {rt.widths_km.size} segments)")
        return rt
    log(f"tracing {len(sites) * binning.cosz_sub_centers.size * binning.az_sub_centers.size} rays on {grid.name} ...")
    rt = build_ray_table(grid, sites, binning.cosz_sub_centers, binning.az_sub_centers,
                         code_version=code_version(),
                         progress=lambda i, n: log(f"  {i}/{n}") if i % 2000 == 0 else None)
    rt.save(path)
    log(f"saved ray table {path} ({rt.n_rays} rays, {rt.widths_km.size} segments)")
    return rt


def provenance(**extra) -> Dict:
    from .probability import solver_info
    return {"code_version": code_version(), "solver": solver_info(), **extra}


def dump_json(obj, path: str) -> None:
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=conv)
