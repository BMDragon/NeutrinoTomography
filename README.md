# NeutrinoTomography

Can four neutrino detectors on Earth's surface resolve interior density
structure?  This package does atmospheric-neutrino oscillation tomography
on a **laterally heterogeneous Earth discretised on a tesseroid grid**,
with [NuOscProbExact](https://github.com/mbustama/NuOscProbExact) as the
oscillation backend.

Status: geometry, density models, probabilities, analytic Jacobian, event
rates, Fisher/resolution analysis and Poisson likelihood are implemented
and tested.  Fluxes and detector responses are analytic placeholders (see
"Caveats").  Atmospheric neutrinos are assumed throughout.

## Install

```bash
python -m pip install -e ".[dev]"        # pulls nuoscprobexact>=1.13,<2
python -m pytest -q                      # ~1 min
```

## Run

```bash
python scripts/build_rays.py     --grid configs/grid_10deg.yaml --detectors configs/detectors.yaml
python scripts/run_coverage.py   --grid configs/grid_10deg.yaml      # geometry only, seconds
python scripts/run_coverage.py   --grid configs/grid_2deg.yaml       # finer counting, minutes
python scripts/run_sensitivity.py --config configs/scan_baseline.yaml  # Fisher study, minutes
python scripts/run_sensitivity.py --config configs/scan_baseline.yaml --profile   # + slow profile fit
```

Outputs land in `results/` (gitignored); cached ray tables in `cache/`,
keyed by a hash of (grid, sites, binning) and stamped with the code
version.

## Layout

```
src/nutomo/
  tesseroids.py   grid definition, indexing, volumes, lateral averaging   (no solver imports)
  raytrace.py     chord -> (widths_km, cell_ids, r_mid); cached RayTable  (no solver imports)
  density.py      DensityModel: radial x radial_scale x (1 + lateral); PREM; Ye; anomaly templates
  probability.py  the ONLY module importing NuOscProbExact (energy-batched slab API)
  jacobian.py     dP/drho_j for every slab from cached prefix/suffix products (exact, tested vs FD)
  rates.py        flux x xsec x exposure x response -> Poisson expectations (+ cell Jacobians)
  detectors.py    sites, exposure, energy binning, separable Gaussian response kernels
  coverage.py     ray counts, path-length kernel resolution, Fisher accumulator (cells + nuisances)
  likelihood.py   Poisson LLH, Asimov TS, ProfileFit (slow, exact), Fisher helpers
  config.py       YAML loading, cache lookup, provenance
configs/          detectors.yaml, grid_10deg.yaml, grid_2deg.yaml, scan_baseline.yaml
scripts/          build_rays.py, run_coverage.py, run_sensitivity.py
tests/            unitarity, matched-leg PREM, ray invariants, Jacobian FD, Fisher brute force
```

## Design rules (why the code is shaped like this)

1. **Ray tracing is a separate cached stage.** Geometry depends on
   detector positions and the grid only.  `raytrace.py` never sees a
   density or an oscillation parameter.
2. **`probability.py` is the backend seam.** Upstream hands it
   `(widths_km, rho, ye)`; nothing else imports the solver.  Swapping the
   backend is a one-file job.
3. **Matched legs.** The 3D and reference (PREM / null / laterally
   averaged) probabilities are always computed on the *same segments*
   through the *same call*.  `probabilities_prem_native` (the library's own
   PREM chord) is a cross-check on absolute normalisation, never a
   subtrahend.  Discretisation error then cancels in every difference.
4. **Hypotheses are configs, not code paths.**  `DensityModel` evaluates
   `radial * radial_scale[shell] * (1 + lateral_scale * lateral[cell])`;
   the null is `lateral_scale = 0` with `radial_scale` free.  That makes the
   3D-vs-null test nested with a countable number of extra parameters.
5. **Event rates, not probabilities, are the observable.**  The Fisher
   matrix and likelihood work on Poisson expectations per (detector,
   cos zenith, azimuth, energy, channel) bin, after smearing, with
   oscillation, flux and radial-profile nuisances.

## Conventions

* `cosz` is the cosine of the arrival zenith angle: -1 is straight up
  through the centre.  Azimuth is where the neutrino comes *from*,
  clockwise from north.  Chords are ordered entry point -> detector.
* Probabilities are `P[E, initial, final]`, flavors (e, mu, tau).
  Evolution operators keep the library's `(final, initial)` indexing.
* Densities in g/cm^3, widths in km (converted to eV^-1 only inside
  `probability.py`), energies in GeV at the API boundary.
* Per-cell inversion parameter is the fractional density perturbation
  `delta_cell`; `radial_scale_k` is the per-shell multiplicative nuisance.

## Jacobian

`jacobian.py` computes `dU/drho_j = S_j . dU_j . P_{j-1}` from cached
prefix (`P`) and suffix (`S`) products of the per-slab exact operators.
The single-slab derivative is the exact Daleckii-Krein (Loewner) formula.
NuOscProbExact propagates the *traceless* Hamiltonian, so the derivative
is taken of `exp(-i H_tl L)` to stay in the same phase convention; mixing
conventions silently breaks `2 Re(conj(U) dU)`.  All of this is
finite-difference tested per slab, per chord and per cell.

## Performance notes

* One full rate pass (8640 rays, 4 sites, 2 channels, 32 energy samples)
  takes ~4 s on 4 cores; a pass with cell Jacobians ~2 min; the dense
  Fisher on ~8400 active cells needs ~0.6 GB per matrix.
* **Do not run two nutomo jobs at once on the same machine** unless you
  pin threads (`NUMBA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`).  numba's
  parallel kernels and BLAS spin-wait against each other and slow both
  jobs down by 10-40x rather than 2x.

## Caveats (read before quoting a number)

* **Flux** is an analytic Gaisser-type parameterisation, within a factor
  ~1.5 of tabulated fluxes at 1-100 GeV.  Use `TableFlux.from_npz` with
  real tables for absolute rates.
* **Cross sections** are linear in E (DIS asymptotics); low-energy
  quasi-elastic/resonance structure is not modelled.
* **Response** is separable Gaussian smearing in log10 E, cos zenith and
  azimuth.  Angular smearing is not a proper space angle.  Choose reco bins
  no finer than the resolution.
* **Channels**: nu_mu CC (tracks) and nu_e CC (cascades) only; NC and
  nu_tau CC are neglected.
* **Detector sites, masses, resolutions** in `configs/detectors.yaml` are
  round numbers for a feasibility study, not collaboration figures.
* Y_e is 0.4656 in the core and 0.4957 in the mantle (NuOscProbExact
  1.13 ships no Y_e profile of its own).
* The installed NuOscProbExact is 1.13.0; the energy-batched slab call is
  the private `slabs._probabilities_slabs_batch`.  `probability.py` falls
  back to a per-energy loop if it disappears.

## Method reference

Tesseroid-grid Earth tomography follows the approach of the IOP CPC
article `10.1088/1674-1137/ae5c85`.  Frame results as "what anomaly
amplitude, at what depth and scale, is detectable at 3 sigma for this
exposure", which is what `run_sensitivity.py` reports.
