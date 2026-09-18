# Baseline results (2026-09-18)

Configs: `configs/grid_10deg.yaml`, `configs/detectors.yaml`,
`configs/scan_baseline.yaml`.  Four sites (IceCube-Upgrade 5 Mton, ORCA 7 Mton,
Hyper-K 0.19 Mton, P-ONE 5 Mton), 10 years each, upgoing only, 20 cos-zenith
x 36 azimuth bins with 3 sub-rays per cos-zenith bin (8640 chords), 2 channels.
Total expectation about 1.0e6 events.  Flux, cross sections and responses
are the analytic placeholders described in the README, so treat every number
as a scaling estimate, not a prediction.

## Geometry (no physics)

10 deg grid, 9072 cells:

| quantity | value |
|---|---|
| cells crossed by >= 1 chord | 8399 (92.6%) |
| cells seen by >= 2 sites | 8240 |
| cells seen by all 4 sites | 5469 |
| innermost shell (r < 600 km) | 0 chords (requires cos zenith < -0.9956; 0.4% of the hemisphere) |
| path-length kernel: independent directions (> 1e-6 of max eigenvalue) | 5333 of 8399 |
| path-length kernel: null space (< 1e-10) | 2554 |
| cells with resolution diag > 0.5 / > 0.9 (damping 1e-3) | 2550 / 92 |
| mean resolution diag per shell, inner to outer | 0.00 0.03 0.07 0.15 0.21 0.32 0.38 0.43 0.49 0.50 0.48 0.48 0.43 0.55 |

Upper-mantle cells are resolved in rings around the three northern sites plus
a latitude band from the polar site; the lower mantle is best resolved at low
latitudes where chords from several sites cross.  Maps in
`results/coverage_grid_10deg/*.png`.  On the 2 deg grid the coverage maps
mainly show the 10 deg azimuth ray sampling; raise `az_sub` before reading
them as geometry.

## Physics-weighted (Fisher, oscillation + flux + radial nuisances)

3 sigma detectable amplitude of a +1 % template (Asimov, no nuisances unless stated):

| template | 3 sigma amplitude |
|---|---|
| whole core, radial | 6.8 % (stat) |
| outer core, radial | 7.4 % |
| lower mantle 3480-5701 km, radial | 3.5 % |
| D'' shell 3480-4000 km, radial | 26 % |
| blob 25 deg, 3480-4000 km, under the Pacific (baseline) | 224 % stat, 232 % with nuisances profiled |
| same blob, 60 deg radius | 65 % |
| blob 25 deg, 3480-4000 km, under Europe | 192 % |
| blob 25 deg, 5701-6151 km, under Europe | 33 % |
| outer-core blob 30 deg / 60 deg | 35 % / 17 % |

Radial-shell scale factors from the Fisher matrix (lateral cells fixed, oscillation
and flux nuisances profiled), 1 sigma per ~500 km shell: 16-34 % in the outer
core and mantle, 180 % in the two inner-core shells.  Combining shells:
core common scale 2.3 % (stat only) / 11 % (other shells free); lower mantle
1.2 % / 4.8 %.

Per-cell 10 deg errors never beat a 5 % prior; the Fisher matrix over 8399
cells has no eigenvalue above the 5 % prior and a 2060-dimensional null space.

## Reading

With this exposure the reachable quantities are coarse radial integrals
(core and lower-mantle mass at the few-percent level).  Lateral structure at
the scale of an LLSVP needs order-unity density contrasts, two orders of
magnitude above the geophysical ~1 %.  The limiting factor is the size of the
matter effect at 2-100 GeV over a 500-2700 km path, not the geometry: the
geometry alone would resolve about 5000 independent directions.

## Cost

Rate pass 4 s; Jacobian pass ~2 min; Fisher assembly ~50 min for four
detectors (sparse smeared Jacobians with 6e7 nonzeros each), dominated by the
sparse products.  Do not run two jobs concurrently (numba thread contention).
