#!/usr/bin/env python3
"""Sensitivity study: three legs, difference maps, Fisher resolution and
template amplitude sensitivity; optional full profile-likelihood fit.

Legs (all through one code path, see density.py):
  3d      : shell-mean PREM radial profile + lateral anomalies
  null    : same radial profile, lateral_scale = 0
  latavg  : 3d model laterally averaged per shell
  prem    : pointwise analytic PREM on the identical segments

Usage: python scripts/run_sensitivity.py --config configs/scan_baseline.yaml [--profile]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nutomo import config as C
from nutomo import coverage as cov
from nutomo import likelihood as lk
from nutomo.density import DensityModel, model_from_config
from nutomo.rates import CrossSection, RateCalculator


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/scan_baseline.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--profile", action="store_true",
                    help="also run the (slow) profile-likelihood fit of the template amplitude")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    cfg = C.load_yaml(a.config)
    name = cfg.get("name", "scan")
    out = C.resolve(a.out or f"results/sensitivity_{name}")
    os.makedirs(out, exist_ok=True)
    grid = C.load_grid(cfg["grid"])
    dets, binning = C.load_sites(cfg["detectors"])
    params = C.osc_params_from_config(cfg.get("oscillation", {}))
    flux = C.flux_from_config(cfg.get("flux", {}))
    flux_nuis = dict(cfg.get("flux", {}).get("nuisances", {}) or {})
    rt = C.get_ray_table(grid, dets, binning, a.cache_dir, log=log)
    rc = RateCalculator(rt, dets, binning, flux, CrossSection())

    # ---- the legs --------------------------------------------------------
    m3d = model_from_config(grid, cfg["model"])
    legs = {"3d": m3d,
            "null": m3d.with_lateral_scale(0.0),
            "latavg": m3d.laterally_averaged(),
            "prem": DensityModel(grid, radial="prem", ye_kind=m3d.ye_kind, name="prem_pointwise")}
    rates = {}
    for k, m in legs.items():
        t0 = time.time()
        rates[k] = rc.compute(m, params, flux_nuis)
        log(f"leg {k:7s}: {rates[k].total():12.1f} reco events  ({time.time() - t0:.1f} s)")
    mu = {k: lk.concat_reco(r) for k, r in rates.items()}
    ts = {"3d_vs_null_unprofiled": lk.asimov_delta_ts(mu["3d"], mu["null"]),
          "3d_vs_latavg_unprofiled": lk.asimov_delta_ts(mu["3d"], mu["latavg"]),
          "null_vs_prem_unprofiled": lk.asimov_delta_ts(mu["null"], mu["prem"])}
    log("Asimov -2dlnL (no nuisances, upper bounds on separability):")
    for k, v in ts.items():
        log(f"  {k:28s} {v:10.3f}   (~{np.sqrt(max(v, 0)):.2f} sigma if 1 dof)")

    # ---- Fisher: cells + nuisances -------------------------------------
    nz = cfg.get("nuisances", {})
    osc_names = list((nz.get("oscillation") or {}).keys())
    flux_names = list((nz.get("flux") or {}).keys())
    prior = {k: float(v["sigma"]) for k, v in (nz.get("oscillation") or {}).items()}
    prior.update({k: float(v["sigma"]) for k, v in (nz.get("flux") or {}).items()})
    rs_sigma = float((nz.get("radial_scale") or {}).get("sigma", 0.0)) or None
    fz = cfg.get("fisher", {})
    active, _ = cov.active_cell_map(rt, grid)
    result = {"active": active, "ts": ts}
    if active.size > int(fz.get("max_active_cells", 8000)):
        log(f"skipping Fisher: {active.size} active cells > max_active_cells")
        acc = None
    else:
        t0 = time.time()
        acc, res3d = cov.assemble_fisher(rc, grid, m3d, params, flux_nuis, osc_names, flux_names,
                                         include_radial_scale=rs_sigma is not None,
                                         min_expected=float(fz.get("min_expected", 1e-3)), log=log)
        log(f"Fisher assembled in {time.time() - t0:.1f} s: {acc.active_cells.size} cells, "
            f"{len(acc.nuisance_names)} nuisances, {acc.n_bins_used} bins, {acc.n_events:.3g} events")
        if rs_sigma is not None:
            prior.update({f"radial_scale_{k}": rs_sigma for k in range(grid.n_r)})
        psc = float(fz.get("prior_sigma_cell", 0.05))

        # template amplitude: delta = A * template (template = the configured anomaly / its amplitude)
        amp = float(cfg["model"]["anomalies"][0]["amplitude"]) if cfg["model"].get("anomalies") else 1.0
        template = (m3d.lateral[active] / amp) if m3d.lateral is not None else np.zeros(active.size)
        sA_stat = acc.template_sensitivity(template, marginalize_nuisance=False)
        sA_marg = acc.template_sensitivity(template, prior, marginalize_nuisance=True)
        log("\nTemplate amplitude (fractional density of the configured anomaly):")
        log(f"  sigma_A statistics only   : {sA_stat:.4f}   -> 3 sigma at {3 * sA_stat * 100:.2f} %")
        log(f"  sigma_A nuisances profiled: {sA_marg:.4f}   -> 3 sigma at {3 * sA_marg * 100:.2f} %")
        log(f"  configured amplitude {amp * 100:.2f} % -> linearised TS {lk.linearized_ts(amp, sA_marg):.2f}"
            f"  ({amp / sA_marg:.2f} sigma)")

        errs = acc.cell_errors(psc, prior)
        r = acc.resolution(psc, prior)
        nuis_err = acc.nuisance_errors(prior, cells_free=True, prior_sigma_cell=psc)
        log(f"\nPer-cell errors with a {psc * 100:.0f}% prior (nuisances profiled):")
        ir = grid.shell_of(active)
        for k in range(grid.n_r):
            sel = ir == k
            if np.any(sel):
                log(f"  shell {k:2d} {grid.r_edges[k]:7.1f}-{grid.r_edges[k + 1]:7.1f} km: "
                    f"{sel.sum():5d} cells  best {100 * errs[sel].min():6.2f}%  median {100 * np.median(errs[sel]):6.2f}%  "
                    f"cells < 2%: {int(np.sum(errs[sel] < 0.02)):4d}   mean R_ii {np.mean(r['diag'][sel]):.3f}")
        log(f"  resolved directions (eigenvalue above prior): {r['n_resolved']} of {active.size}; null: {r['n_null']}")
        log("\nNuisance errors (cells free under prior): " + ", ".join(
            f"{n}={e:.3g}" for n, e in zip(acc.nuisance_names, nuis_err) if not n.startswith("radial_scale")))
        rs = [(n, e) for n, e in zip(acc.nuisance_names, nuis_err) if n.startswith("radial_scale")]
        if rs:
            log("  radial scale errors per shell: " + " ".join(f"{100 * e:.2f}%" for _, e in rs))
        cell_err = np.full(grid.n_cells, np.nan)
        cell_err[active] = errs
        res_diag = np.full(grid.n_cells, np.nan)
        res_diag[active] = r["diag"]
        result.update({"cell_error": cell_err, "resolution_diag": res_diag,
                       "fisher_eigenvalues": r["eigenvalues"], "template": template,
                       "sigma_A_stat": sA_stat, "sigma_A_marg": sA_marg,
                       "nuisance_names": np.array(acc.nuisance_names), "nuisance_errors": nuis_err})
        np.savez_compressed(os.path.join(out, "fisher.npz"), F_cc=acc.F_cc, F_cn=acc.F_cn, F_nn=acc.F_nn,
                            active=active, nuisance_names=np.array(acc.nuisance_names))

    # ---- difference maps: where the sensitivity lives ------------------
    diff = {}
    for idet, d in enumerate(dets):
        n3, n0 = rates["3d"].reco[idet], rates["null"].reco[idet]
        z = (n3 - n0) / np.sqrt(np.maximum(n0, 1e-3))          # per-bin significance
        diff[d.name] = {"z_cosz_az": np.sqrt(np.sum(z ** 2, axis=(2, 3))),
                        "z_cosz_E": np.sqrt(np.sum(z ** 2, axis=(1, 3))),
                        "frac_cosz_az": np.sum(n3 - n0, axis=(2, 3)) / np.maximum(np.sum(n0, axis=(2, 3)), 1e-3)}
        log(f"  {d.name:16s}: total z^2 (3d-null) = {np.sum(z ** 2):8.3f}")
        result[f"diff_z_cosz_az_{idet}"] = diff[d.name]["z_cosz_az"]
        result[f"diff_z_cosz_E_{idet}"] = diff[d.name]["z_cosz_E"]

    # ---- optional full profile likelihood ------------------------------
    if a.profile:
        log("\nprofile-likelihood fit (slow) ...")
        pf = lk.ProfileFit(rc, m3d, params, flux_nuis, log=log)
        data = mu["3d"]
        nuis = [lk.FitParam("dm31", params.dm31, prior.get("dm31"), 2e-5),
                lk.FitParam("s23sq", params.s23sq, prior.get("s23sq"), 0.01, (0.3, 0.7)),
                lk.FitParam("s13sq", params.s13sq, prior.get("s13sq"), 0.0005, (0.01, 0.04)),
                lk.FitParam("flux_norm", 1.0, prior.get("flux_norm"), 0.02, (0.2, 5.0))]
        amp_par = lk.FitParam("lateral_scale", 1.0, None, 0.05, (-20.0, 20.0))
        t0 = time.time()
        pr = pf.profiled_ts(data, nuis, amp_par, 0.0, maxiter=60)
        log(f"  profiled TS = {pr['ts']:.3f} ({pr['significance']:.2f} sigma), "
            f"{pr['n_eval']} evaluations, {time.time() - t0:.0f} s")
        result["profile"] = pr

    np.savez_compressed(os.path.join(out, "sensitivity.npz"),
                        **{k: v for k, v in result.items() if isinstance(v, (np.ndarray, float, int))})
    C.dump_json({"config": cfg, "ts": ts,
                 "sigma_A_stat": result.get("sigma_A_stat"), "sigma_A_marg": result.get("sigma_A_marg"),
                 "nuisance_errors": dict(zip(map(str, result.get("nuisance_names", [])),
                                             result.get("nuisance_errors", []))),
                 "profile": result.get("profile"),
                 "provenance": C.provenance(ray_table=rt.meta)},
                os.path.join(out, "summary.json"))

    if not a.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(dets), 2, figsize=(11, 3 * len(dets)), squeeze=False)
        for idet, d in enumerate(dets):
            ax = axes[idet, 0]
            im = ax.pcolormesh(binning.az_edges_deg, binning.cosz_edges, diff[d.name]["z_cosz_az"], cmap="viridis")
            ax.set_title(f"{d.name}: |3D-null|/sqrt(N) summed over E, channel", fontsize=9)
            ax.set_xlabel("azimuth [deg]"); ax.set_ylabel("cos zenith"); fig.colorbar(im, ax=ax)
            ax = axes[idet, 1]
            im = ax.pcolormesh(d.energy_edges(), binning.cosz_edges, diff[d.name]["z_cosz_E"], cmap="viridis")
            ax.set_xscale("log"); ax.set_title(f"{d.name}: |3D-null|/sqrt(N) summed over azimuth, channel", fontsize=9)
            ax.set_xlabel("E [GeV]"); ax.set_ylabel("cos zenith"); fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(out, "difference_maps.png"), dpi=130); plt.close(fig)
        if acc is not None:
            shells = [6, 7, 9, 12]
            fig, axes = plt.subplots(len(shells), 2, figsize=(12, 3 * len(shells)), squeeze=False)
            for row, k in enumerate(shells):
                for col, (fld, ttl, cm, vmax) in enumerate([(result["cell_error"] * 100, "1 sigma density error [%]", "magma_r", psc * 100),
                                                            (result["resolution_diag"], "resolution diag", "viridis", 1.0)]):
                    ax = axes[row, col]
                    v = fld.reshape(grid.n_r, grid.n_lat, grid.n_lon)[k]
                    im = ax.pcolormesh(grid.lon_edges, grid.lat_edges, v, cmap=cm, vmin=0, vmax=vmax)
                    ax.set_title(f"{ttl}, shell {grid.r_edges[k]:.0f}-{grid.r_edges[k + 1]:.0f} km", fontsize=9)
                    fig.colorbar(im, ax=ax)
            fig.tight_layout(); fig.savefig(os.path.join(out, "cell_error_maps.png"), dpi=130); plt.close(fig)
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
