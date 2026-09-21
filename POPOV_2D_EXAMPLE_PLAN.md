# Plan: simplest 2D Popov example

Status: staged; material-point, cap-to-momentum pressure, cohesion-law, and
residual/Jacobian history gates are implemented, and the spatial driver now
carries the authors' localization seed. Full paper reproduction is not yet
confirmed against reference data.

The benchmark specification in §1 is no longer unverified. The archived
GeoTech2D release was read directly, and its `MESH/tensile.py` and
`CODE/tensile.py` supply the seed, the boundary conditions, the softening law,
and the timestep policy. The driver now matches them; see the table below for
the values and the remaining differences.

Three driver defects were found and fixed while measuring this. `τ_old` was
never refreshed between steps, freezing the deviatoric response. The reported
"corrected pressure" was `dr.P .+ dr.Pnum`, a solver relaxation term rather
than the cap's pressure; it is now recorded at integration points through a
four-matrix `τ_store`, which shows the cap holding pressure at `-0.147` against
`pT = -0.1` while the trial field ramps past `-0.768`, the expected split under
the trial-pressure scheme. And `vy = 0` was applied on the whole boundary
rather than on the top and bottom only, which pinned the side walls the band
must open against.

Measuring this driver also exposed a shared-solver defect: the inner velocity
loop scored progress only against the residual at its first check, so a step
starting at its own solution could never satisfy the exit test and spent the
whole inner budget. Fixed.

Before the seed the driver deformed homogeneously, with volumetric plastic
strain spread at machine zero, `4e-18`. Meshing with Triangulate instead of a
structured grid changed nothing, because T7 elements represent a linear
velocity field exactly. With the seed the spread is `5.3e-3` after 25 steps and
the peak sits directly above the inclusion.

Remaining before reproduction: run to 100 kyr and compare the profile against
Figure 6b, and settle the `P0` handoff question in §2.

## Target

Add the regularized 2D restrained-extension test in Section 4.2, Figure 6b of
[Popov, Berlie and Kaus (2025)](https://gmd.copernicus.org/articles/18/7035/2025/),
"A dilatant visco-elasto-viscoplasticity model with globally continuous tensile
cap: stable two-field mixed formulation."

This is the paper's first spatial benchmark and exercises tensile localization.
The preceding Section 4.1 example is a material-point test.

The first milestone is a converged coarse extension run with correct pressure
coupling and softening. Full Figure 6 reproduction follows after that gate.

## Current implementation and prerequisites

The repository contains the tensile-cap return map and integration-point
plastic-history storage and accumulation. The cap momentum path now uses locally
corrected pressure, while the global pressure field, pressure residual, and
density equation of state retain trial pressure. Cohesion softening is
available as validated `C_min`/`H_C` parameters and a bounded local law,
threaded through residual and augmented Jacobian assembly. The spatial driver
remains before treating the example as a reproduction of the paper.

Relevant implementation paths:

- `src/stokes/assemblers/rheology.jl`
- `src/stokes/assemblers/momentum_residuals.jl`
- `src/stokes/assemblers/pressure_residual.jl`
- `src/stokes/types/stokes_types.jl`
- `src/stokes/solvers/DR.jl`

The existing [cap implementation plan](DRUCKER_PRAGER_CAP_PLAN.md) and subsystem
notes contain outdated status descriptions; reconcile them with the code during
implementation.

## 1. Fix the benchmark specification

Use the extension case in the "Regularization" column of Table 1:

| Parameter | Target value |
|---|---|
| Domain | 1 x 0.7 m |
| Shear modulus G | 40 GPa |
| Bulk modulus K | 64 GPa |
| Friction angle | 30 degrees |
| Dilation angle | 0 degrees |
| Tensile strength | -1 MPa, compression-positive pressure convention |
| Viscoplastic regularization viscosity | 1e19 Pa s |
| Cohesion | Softens from 20 MPa to 5 MPa |
| Comparison time | 100 kyr, Figure 6b |

Table 1 also specifies the regularized extension loading as
`ε̇xx = 6.338e-15 s⁻¹`, `ε̇yy = 0`, with 50-year physical steps and
`ηvp = 1e19 Pa s`. The driver uses only these paper values, expressed with
`L₀=1 m`, `S₀=10 MPa`, and `t₀=50 yr` so the solver receives dimensionless
geometry, stresses, viscosities, and time steps.

Before coding, boundary conditions, the localization seed, softening details,
and timestep handling were to be verified against the authors' setup in the
[archived GeoTech2D release](https://doi.org/10.5281/zenodo.15496843). That is
now done. The release is a Python code; `MESH/tensile.py` builds this
benchmark's mesh and `CODE/tensile.py` its physics. What they specify:

| Item | Release value | In the driver |
|---|---|---|
| Domain | `L = 1.0`, `H = 0.7` | same |
| Localization seed | half-disc on the bottom boundary, centre `(0.5, 0)`, radius `0.025`, 9 arc points over `0..π`, own Triangle region | same, via `segments`/`regions` |
| Seed contrast | `G = 4e9` against bulk `4e10`; every other property equal | same, as phase 2 |
| Mesh | Triangle, `area = 3e-4`, `angle = 30` | same |
| Horizontal loading | `vx = ∓1e-4 mm/yr` on left/right, `vy` free there | same |
| Vertical restraint | `vy = 0` on top and bottom, `vx` free there | same |
| Cohesion softening | `c = 2e7 + (-1e8)·aps`, floored at `5e6` | same |
| Softening measure | `aps`, accumulated *deviatoric* plastic strain | same |
| Tensile strength | `ps = -1e6` Pa | same |
| Regularization viscosity | `eta = 1e19` Pa s | same |
| Friction / dilation | `phi = 30°`, `psi` defaulted to `0°` | same |
| Timestep | `dt = 100 yr`, adaptive down to `dtmin = 0.05 yr` | fixed `dt = 50 yr` |
| Total time | `tmax = 100 kyr` | same, 2000 steps |
| Gravity | `g = 9.81`, `rhob = 3000` | `g = 0`, `ρ0 = 1` |

`1e-4 mm/yr` on each side over `L = 1 m` is `6.3376e-15 s⁻¹`, which reproduces
Table 1's `ε̇xx = 6.338e-15 s⁻¹` exactly and confirms the domain is in metres.

Two deliberate differences remain. The driver takes fixed 50-year steps rather
than the release's adaptive 100-year steps, matching the paper's stated
increment. And it runs without gravity: on a 0.7 m column the lithostatic
stress is about 21 kPa against a 1 MPa tensile strength and 20 MPa cohesion, so
it is negligible here, but it is a difference to record rather than assume away.
Record units and any nondimensionalization explicitly.

## 2. Complete the required constitutive coupling

Implement the changes in the shared Stokes layer:

- Return corrected pressure alongside deviatoric stress and use it in momentum
  balance.
- Define how corrected pressure is carried into the next physical timestep.
- Feed integration-point plastic strain into the paper's cohesion-softening law.
- Commit stress and plastic history only after convergence. Residual evaluations
  must leave accepted history unchanged.
- Retain finite bulk modulus and check consistency between material `K` and
  plastic-model `Kb`.

Add focused regressions for pressure coupling, softening, and history updates.
Preserve the existing non-cap paths and backend-neutral assembly. Any changed
residual must remain consistent with its Jacobian and adjoint paths.

## 3. Add one small CPU example

Proposed directory: `examples/miniapps/stokes/popov_extension_2D/`.

Use a function-only setup file plus a driver that calls `main()` unconditionally.
Reuse T7/P1-discontinuous elements, existing mesh helpers, and
`solve_stokes_dyrel!`.

Expose mesh resolution, timestep, final time, plotting, and output controls.
Provide one paper configuration with a headless option and rely on
`--project=examples` rather than activating an environment inside the script.

## 4. Make the result inspectable

Return a `NamedTuple` containing velocity, trial and corrected pressure,
accumulated plastic strains, time history, and convergence statistics.

Provide optional VTK output and a plot of accumulated volumetric plastic strain
with the cross-section used for comparison. Keep trial pressure and corrected
physical pressure clearly distinguished in returned fields and output labels.
Fail explicitly on non-convergence.

## 5. Validate in stages

1. Compare homogeneous deformation with an independent material-point
   calculation.
2. Run a small headless localization case and verify convergence and history
   behavior.
3. Compare the localization profile across progressively
   finer meshes and smaller timesteps.
4. Set quantitative tolerances from reference data. A similar-looking image
   alone is insufficient.

Keep expensive reproduction and refinement runs outside the default unit suite.
Record differences from GeoTech2D, including the global solution algorithm.

## 6. Document and integrate

- Add the driver and helper to `test/test_example_paths.jl`.
- Document the run command, nondimensional scales, outputs, and numerical differences
  from GeoTech2D.
- Reconcile outdated cap-status notes and the cap implementation plan.
- Review all nine `.agents/` guides and update affected ones, including the
  miniapp inventory once the example exists.
- Run focused checks, the full package tests for shared solver changes, and the
  documentation build.
- Check the final diff and keep generated output out of source control.

Validation commands from the repository root:

```sh
julia --startup-file=no test/test_example_paths.jl
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl
git diff --check
```

The example execution command will use `julia --project=examples` followed by
the driver path chosen during implementation.
