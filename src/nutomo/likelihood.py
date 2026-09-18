"""Poisson likelihood, Asimov test statistics, profiling and Fisher helpers.

Hypotheses are *configs* of one DensityModel (see density.py): the null is
the same model with ``lateral_scale = 0`` and the radial profile free, so
the likelihood ratio between 3D and null is a nested test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .density import DensityModel
from .probability import OscParams
from .rates import RateCalculator, RateResult


# ---------------------------------------------------------------------------
# Poisson statistics
# ---------------------------------------------------------------------------
def neg2_log_likelihood(n_obs: np.ndarray, mu: np.ndarray, floor: float = 1e-12) -> float:
    """-2 ln L (Poisson, up to constants) summed over bins."""
    mu = np.maximum(np.asarray(mu, float), floor)
    n = np.asarray(n_obs, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(n > 0, n * np.log(n / mu), 0.0)
    return float(2.0 * np.sum(mu - n + term))


def asimov_delta_ts(mu_truth: np.ndarray, mu_hyp: np.ndarray, floor: float = 1e-12) -> float:
    """Asimov -2 Delta ln L of hypothesis ``mu_hyp`` against data = ``mu_truth``.

    Equals the Poisson -2 ln(L_hyp / L_truth) with observed counts set to
    their expectation under the truth.  Not profiled: use ``ProfileFit``
    (or the Fisher machinery in coverage.py) for nuisance marginalisation.
    """
    return neg2_log_likelihood(mu_truth, mu_hyp, floor) - neg2_log_likelihood(mu_truth, mu_truth, floor)


def concat_reco(res: RateResult) -> np.ndarray:
    return np.concatenate([r.ravel() for r in res.reco])


def significance_from_ts(ts: float, dof: int = 1) -> float:
    """Gaussian-equivalent significance of a chi^2 test statistic."""
    from scipy.stats import chi2, norm
    p = chi2.sf(max(ts, 0.0), dof)
    return float(norm.isf(max(p, 1e-300) / 2.0)) if dof == 1 else float(norm.isf(max(p, 1e-300)))


# ---------------------------------------------------------------------------
# full profiling (expensive: every evaluation recomputes all rays)
# ---------------------------------------------------------------------------
@dataclass
class FitParam:
    name: str
    value: float
    prior_sigma: Optional[float] = None     # Gaussian prior, None = flat
    step: float = 1e-3
    bounds: Tuple[Optional[float], Optional[float]] = (None, None)


OSC_NAMES = ("dm31", "s23sq", "s13sq", "dcp", "dm21", "s12sq")
FLUX_NAMES = ("flux_norm", "delta_gamma", "nue_numu_ratio", "nubar_nu_ratio")


class ProfileFit:
    """Profile the Poisson likelihood over oscillation, flux and radial
    nuisances plus (optionally) the lateral amplitude.

    Parameters are identified by name:
      * OSC_NAMES              -> OscParams fields
      * FLUX_NAMES             -> flux nuisances
      * 'radial_scale_<k>'     -> DensityModel.radial_scale[k]
      * 'lateral_scale'        -> DensityModel.lateral_scale
    """

    def __init__(self, rc: RateCalculator, base_model: DensityModel, base_params: OscParams,
                 base_flux: Optional[Dict[str, float]] = None, log: Callable = print):
        self.rc = rc
        self.model = base_model
        self.params = base_params
        self.flux = dict(base_flux or {})
        self.log = log
        self.n_eval = 0

    def expectation(self, values: Dict[str, float]) -> np.ndarray:
        params = self.params.replace(**{k: v for k, v in values.items() if k in OSC_NAMES})
        flux = dict(self.flux)
        flux.update({k: v for k, v in values.items() if k in FLUX_NAMES})
        rs = np.array(self.model.radial_scale, float)
        for k, v in values.items():
            if k.startswith("radial_scale_"):
                rs[int(k.split("_")[-1])] = v
        model = self.model.with_radial_scale(rs)
        if "lateral_scale" in values:
            model = model.with_lateral_scale(values["lateral_scale"])
        self.n_eval += 1
        return concat_reco(self.rc.compute(model, params, flux))

    def objective(self, data: np.ndarray, free: Sequence[FitParam]) -> Callable:
        def f(x):
            vals = {p.name: xi for p, xi in zip(free, x)}
            mu = self.expectation(vals)
            n2ll = neg2_log_likelihood(data, mu)
            for p, xi in zip(free, x):
                if p.prior_sigma:
                    n2ll += ((xi - p.value) / p.prior_sigma) ** 2
            return n2ll
        return f

    def fit(self, data: np.ndarray, free: Sequence[FitParam], method: str = "L-BFGS-B",
            maxiter: int = 200) -> Dict:
        from scipy.optimize import minimize
        f = self.objective(data, free)
        x0 = np.array([p.value for p in free], float)
        bounds = [p.bounds for p in free] if method in ("L-BFGS-B", "Powell", "TNC") else None
        eps = np.array([p.step for p in free], float)
        opts = {"maxiter": maxiter}
        if method == "L-BFGS-B":
            opts["eps"] = eps
        r = minimize(f, x0, method=method, bounds=bounds, options=opts)
        return {"x": {p.name: float(v) for p, v in zip(free, r.x)}, "fun": float(r.fun),
                "success": bool(r.success), "nfev": int(r.nfev), "message": str(r.message)}

    def profiled_ts(self, data: np.ndarray, nuisances: Sequence[FitParam],
                    amplitude_param: FitParam, amplitude_null: float = 0.0,
                    method: str = "L-BFGS-B", maxiter: int = 200) -> Dict:
        """TS = min_null(-2lnL) - min_alt(-2lnL) with the amplitude free in the
        alternative and fixed at ``amplitude_null`` in the null."""
        null_free = list(nuisances)
        null_fixed = FitParam(amplitude_param.name, amplitude_null)
        f_null = self.objective(data, null_free + [null_fixed])
        from scipy.optimize import minimize
        x0 = np.array([p.value for p in null_free] + [amplitude_null])
        def fn(x):
            return f_null(np.concatenate([x, [amplitude_null]]))
        opts = {"maxiter": maxiter}
        if method == "L-BFGS-B":
            opts["eps"] = np.array([p.step for p in null_free])
        r0 = minimize(fn, x0[:-1], method=method,
                      bounds=[p.bounds for p in null_free] if method == "L-BFGS-B" else None,
                      options=opts)
        r1 = self.fit(data, list(nuisances) + [amplitude_param], method, maxiter)
        ts = float(r0.fun - r1["fun"])
        return {"ts": ts, "significance": significance_from_ts(ts, 1),
                "null": {"fun": float(r0.fun), "x": {p.name: float(v) for p, v in zip(null_free, r0.x)}},
                "alt": r1, "n_eval": self.n_eval}


# ---------------------------------------------------------------------------
# Fisher-based sensitivity (cheap; uses coverage.FisherAccumulator)
# ---------------------------------------------------------------------------
def amplitude_for_significance(sigma_amplitude: float, n_sigma: float = 3.0) -> float:
    """Smallest template amplitude detectable at n_sigma (linearised)."""
    return n_sigma * sigma_amplitude


def linearized_ts(amplitude: float, sigma_amplitude: float) -> float:
    return (amplitude / sigma_amplitude) ** 2 if np.isfinite(sigma_amplitude) and sigma_amplitude > 0 else 0.0
