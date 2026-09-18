"""nutomo: Earth tomography with atmospheric neutrinos on a tesseroid grid.

Module map (see README for the design rules behind the split):

- tesseroids  grid definition and indexing              (no solver imports)
- raytrace    chord -> (widths, cell ids, midpoint radii) (no solver imports)
- density     cell ids + model -> densities, Ye           (no solver imports)
- probability the only module that touches NuOscProbExact
- jacobian    dP/drho_j via cached partial products of per-slab operators
- rates       flux x xsec x mass x smearing -> Poisson expectations
- coverage    ray counts per cell, path-length kernel, Fisher/resolution
- likelihood  Poisson LLH, profiling, test statistic, Fisher sensitivity
- detectors   sites, exposure, binning, response kernels
"""

__version__ = "0.1.0"

R_EARTH_KM = 6371.0
