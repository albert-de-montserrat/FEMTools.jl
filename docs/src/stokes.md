# Stokes

FEMTools.jl includes an incompressible Stokes-flow solver for mixed
velocity–pressure elements (e.g. a T6/P1-disc Taylor–Hood-like pair). The
rheology is viscoelastic, with optional Drucker–Prager elasto-viscoplasticity,
and a temperature-dependent buoyancy body force. Like the other solvers it runs
on both CPU and GPU via KernelAbstractions.jl.

## Physical model

The solver finds a velocity `v` and pressure `P` satisfying the momentum and
continuity balances

```math
\nabla \cdot \boldsymbol{\tau} - \nabla P + \rho \mathbf{g} = 0, \qquad
\nabla \cdot v + \frac{1}{\eta_b}\frac{\partial P}{\partial t}
- \alpha\frac{\partial T}{\partial t} = 0,
```

with a Maxwell viscoelastic deviatoric stress that carries stress history
`τ_old` across time steps. Density uses the linearised equation of state
`ρ = ρ0 (1 − α(T − Tref) + P/K)`. Per-phase properties (`η`, `ηb`, `G`, `α`,
`ρ0`, `K`) and the body-force parameters are grouped in a typed
`StokesMaterial`.

The saddle-point system is exposed through `solve_stokes_dyrel!`. For the 2-D
mixed-mesh state, an outer Arrow–Hurwicz pressure update wraps an inner
Chebyshev-accelerated dynamic-relaxation sweep on the momentum residual. The
3-D T11/P1-discontinuous and Hex27/Q2--P1 discretisations are available through
a `StokesDR`/`MixedMesh` state for coupled visco-elasto-plastic problems. The
original Hex27 viscous interface with three caller-owned velocity arrays and a
`4 × nels` pressure array remains available.

The discrete adjoint uses the transpose of the same assembled element
operators and the same mixed spaces. It therefore computes gradients of the
*discrete* objective rather than a separately discretised continuous adjoint.

## Solver state

```@docs
StokesMaterial
StokesDR
DruckerPrager
pressure_mass
```

The velocity and pressure fields live on separate node sets described by a
[`MixedMesh`](mesh.md); precompute per-field geometry once with a
[`MixedMeshCache`](mesh.md). Fill `dr.T` (and `dr.T0`) before solving. The lumped
pressure mass `dr.M_P` must be assembled — see
[`FEMTools.assemble_viscosity_weighted_pressure_scaling!`](@ref) — before the
first call.

### Compact setup

```julia
material = StokesMaterial(; η, ηb, G, α, ρ0, K, g, Tref)
dr = StokesDR(backend, mesh.nnodes, mesh.nnodesP, material;
              stress_size=(nq, mesh.nels))

cache = MixedMeshCache(backend, workgroup, mesh, element_v, element_P)
bc_vx = DirichletBoundaryCondition(nothing, vx_nodes, vx_vals)
bc_vy = DirichletBoundaryCondition(nothing, vy_nodes, vy_vals)

assemble_viscosity_weighted_pressure_scaling!(
    γP, dr, mesh, cache, γfact, Δt; workgroup,
)
solve_stokes_dyrel!(dr, mesh, cache, bc_vx, bc_vy, Δt, γP; workgroup)
```

`MixedMeshCache` retains the reference elements alongside both geometry arrays,
so the high-level assembly and solver calls infer elements and backend. The
expanded positional methods remain available for custom and adjoint workflows.
The pressure kernel interpolates nodal pressure and temperature increments
directly, avoiding temporary per-node rate calculations.

### Coupled thermal--Stokes relaxation

`solve_coupled_dyrel!` advances one thermal DR iteration during every inner
Stokes velocity iteration. The thermal mesh must share the Stokes velocity-node
numbering; after each thermal update, its continuous nodal temperature is
gathered onto the discontinuous pressure DoFs used by the Stokes residuals.

```julia
stats = solve_coupled_dyrel!(
    thermal, stokes, thermal_mesh, stokes_mesh, cache,
    bc_T, (bc_vx, bc_vy, bc_vz), Δt, γP; workgroup,
)
stats.converged || error("coupled solve did not converge")
```

The returned Stokes statistics additionally contain `err_T` and
`thermal_iterations`. The caller still owns physical-time history: set
`thermal.T0`, `stokes.P0`, and the old Stokes stresses before each coupled
solve. The thermal `ncheck` cadence follows the Stokes `ncheck` keyword.

In 2-D, pass `bc_vx` and `bc_vy` positionally as before. In 3-D, pass the tuple
shown above. The 3-D signed pressure basis uses a positive Jacobi modal mass
instead of direct lumping.

### Three-dimensional viscous array layout

The original viscous Hex27/Q2--P1 method keeps caller-owned arrays:

```julia
velocity = ntuple(_ -> zeros(mesh.nnodes), 3)
pressure = zeros(4, mesh.nels)

stats = solve_stokes_dyrel!(
    velocity, pressure, mesh, cell_phase, η, ρ, g, fixed_nodes;
    ϵ_tol = 1e-6,
)
```

`fixed_nodes` is an `NTuple{3}` containing the constrained nodes for each
velocity component. The matching `solve_stokes_adjoint_dyrel!` method accepts
the same storage plus a three-component objective load. `solve_stokes_3d!` and
`solve_stokes_adjoint_3d!` remain compatibility wrappers.

### Two-dimensional spectral estimate and frozen Jacobian

The forward velocity sweep uses a diagonal Jacobi preconditioner ``P`` and the
augmented momentum Jacobian ``A``. By default, `solve_stokes_dyrel!` obtains a
conservative upper estimate from the assembled absolute row sums. Set
`measure_λmax = true` to instead apply power iteration to

```math
\widehat A = P^{-1/2} A P^{-1/2}.
```

``P^{-1}A`` and ``\widehat A`` are similar and therefore have the same
eigenvalues, while the symmetric scaling avoids the artificial non-normality
introduced by left Jacobi scaling when ``A`` is symmetric. The DYREL step is

```math
\Delta\tau = \frac{2\,\mathrm{CFL}_v}{\sqrt{\lambda_{\max}}},
```

so an unnecessarily large Gershgorin estimate shortens every pseudo-time step.
Power iteration is opt-in because it stores one dense augmented velocity block
per element. Its dominant vector is warm-started across Jacobian refreshes.

For linear viscous and viscoelastic rheologies (`plastic === nothing`), the
augmented Jacobian is independent of the iterated velocity and pressure.
`freeze_jacobian` therefore defaults to `true` and the blocks, row sums,
preconditioner, and ``\lambda_{\max}`` are assembled only once. Convergence
checks continue updating ``\lambda_{\min}`` and the Chebyshev coefficients.
Set `freeze_jacobian = false` to force refreshes.

Drucker–Prager plasticity is state-dependent, so its default remains
`freeze_jacobian = false`. A non-associated plastic tangent is also non-normal:
the solver consequently applies at least a 1.5 safety factor to the measured
value, compared with the configurable `λmax_safety = 1.1` on the linear path.
The power controls are `λmax_power_iterations` and `λmax_power_rtol`.

The returned statistics include the `λmax` actually used, the
`λmax_gershgorin` reference, cumulative `λmax_iterations`, and
`jacobian_assemblies`. The inner loop also fuses both velocity-component rate
and field updates into one kernel and copies residual history only at
convergence checks.

## Example

The scripts under `examples/stokes/` set up complete problems, including a
viscoelasto-plastic pure-shear test and a sinking-block buoyancy test:

```sh
julia --project=examples examples/stokes/vevp/stokes_2D_pure_shear.jl
julia --project=examples examples/stokes/sinking_block/sinking_block.jl
julia --project=examples examples/stokes/sinking_block/sinking_block_adj.jl
julia --project=examples examples/stokes/sinking_block/sinking_block_3D.jl
julia --project=examples examples/stokes/sinking_block/sinking_block_3D_adj.jl
julia --project=examples examples/stokes/volcano/volcano_thermal_stokes.jl
julia --project=examples examples/stokes/volcano/volcano_thermal_stokes_3D.jl
```

The 3-D volcano driver uses a locally refined unstructured T11 mesh. Its Gmsh
geometry is fragmented by the chamber ellipsoid, so the crust and chamber share
the curved interface instead of assigning phases across cut cells.

See the [Sinking block](sinking_block.md) page for the 2-D and 3-D
discretisations, physical setup, output, figure, and material-gradient checks.

The 2-D adjoint sinking-block example accepts an explicit backend. It builds the
Gmsh mesh on the host, then uploads mesh arrays, mixed connectivity,
geometry caches, phase indices, and boundary data before launching kernels:

```julia
using CUDA
include("examples/stokes/sinking_block/sinking_block_adj.jl")

result = main(backend = CUDABackend(), show_plot = false)
```

Loading CUDA activates the FEMTools CUDA extension, for which
`TA(CUDABackend()) === CuArray`. A functional NVIDIA driver is required for
allocation and execution. Plotting is host-side; keep `show_plot = false` for
headless accelerator runs.

## Drivers

```@docs
solve_coupled_dyrel!
solve_stokes_dyrel!
solve_stokes_adjoint_dyrel!
solve_stokes_3d!
solve_stokes_adjoint_3d!
stokes_material_gradient_3d
FEMTools.FrozenAdjointOperator
update_stokes_current_stress!
```

## Discrete adjoint and material sensitivities

For an objective `J(u)` and forward residual `R(u, m) = 0`, the adjoint solves

```math
\left(\frac{\partial R}{\partial u}\right)^T \lambda
= -\frac{\partial J}{\partial u}.
```

The reduced material derivative is then contracted elementwise as

```math
\frac{\mathrm d J}{\mathrm d m_e}
= \frac{\partial J}{\partial m_e}
+\lambda_e^T \frac{\partial R_e}{\partial m_e},
```

for the 2-D convention above. The current examples have no explicit material
term in the objective, so only the contraction remains. The 2-D example forms a
finite-element objective load for
`J(v_y) = -∫_{Ωobs} v_y dΩ`, solves the transpose system with
`solve_stokes_adjoint_dyrel!`, and uses Enzyme reverse mode on the element
momentum residual contraction to obtain density and viscosity sensitivities.
The returned sensitivity arrays contain raw element integrals. Their sums give
phase gradients; division by element area is used only to visualise a spatial
sensitivity density.

The 3-D viscous operator is symmetric, so its adjoint DYREL method reuses the
3-D forward residual and preconditioner with the objective derivative as the
momentum load. This wrapper instead solves `A^Tλ = J_u`, so its material
derivative uses the opposite contraction `-λ^T R_m`.
`stokes_material_gradient_3d` contracts the forward and adjoint velocity fields
analytically. `test/test_stokes_3d_reference.jl`
validates those contractions against a sparse finite-difference oracle.

### Why the discrete transpose matters

The adjoint uses the same velocity and pressure spaces, quadrature points,
constitutive update, and element residuals as the forward solve. Transposing
those discrete operators exactly makes the resulting gradient the derivative of
the objective that the code actually evaluates. Changing the adjoint space,
quadrature, or rheology independently would instead produce a gradient of a
different discretisation. When adding an objective or material parameter,
validate that contract with a central finite difference as demonstrated in
`test/test_stokes_adjoint_api.jl`.

### Two-dimensional frozen operator and solver controls

The forward state must be converged before the adjoint solve. At that fixed
state the transpose Jacobian is constant, so the default
`frozen_operator = true` path assembles three dense blocks per element once and
reuses them throughout the Powell–Hestenes / DYREL solve. Each subsequent
operator application is only an element gather, dense products, and scatter;
it does not reevaluate the rheology or invoke automatic differentiation.

The inner velocity iteration stops after reducing its residual by `rel_drop`;
the outer pressure iteration continues until `adjoint_tol` or
`max_ph_iterations`. `iterMax` limits one inner solve and `total_iterMax` limits
the complete adjoint. With `measure_λmax = true`, power iteration measures the
largest eigenvalue of the Jacobi-preconditioned velocity block instead of using
its looser Gershgorin bound. The returned statistics report both values and the
number of power iterations. Set `measure_λmax = false` to use the Gershgorin
estimate directly; the Enzyme fallback also uses that estimate because it has no
cheap frozen operator application for power iteration.

`λvx`, `λvy`, and `λP` are initial guesses as well as output arrays. Zero them
for a cold solve; in an optimization loop, leave the previous design's adjoint
in place to warm-start the next solve.

If the velocity residual stalls, inspect the measured-to-Gershgorin ratio and
increase the iteration budgets before changing tolerances. If the velocity
residual drops but the pressure residual does not, the outer PH iteration is
the bottleneck; lowering `rel_drop` only spends more work on the already-solved
subproblem.

The cached T7/P1-disc operator stores 280 floating-point values per element,
about 2.2 kB per element in `Float64`. Set `frozen_operator = false` when that
memory footprint is unsuitable, notably for larger three-dimensional elements.
The fallback reconstructs the same transpose products with Enzyme on every
iteration and is therefore slower but avoids the block storage.

See `examples/stokes/sinking_block/sinking_block_adj.jl` for a complete solve
and `examples/benchmarks/adjoint_perf.jl` for a headless mesh/contrast sweep.
Forward comparisons are available in
`examples/benchmarks/forward_lambda_perf.jl` and
`examples/benchmarks/forward_lambda_shear_band_perf.jl`.

## Assembly

```@docs
FEMTools.assemble_momentum_residual_matrices_atomix!
FEMTools.assemble_momentum_jacobian_matrices_atomix!
FEMTools.assemble_augmented_momentum_jacobian_matrices_atomix!
FEMTools.assemble_pressure_residual_matrices_atomix!
FEMTools.assemble_viscosity_weighted_pressure_scaling!
FEMTools.momentum_element_residual
FEMTools.element_momentum_jacobians
FEMTools.element_augmented_momentum_jacobians
FEMTools.pressure_element_residual
FEMTools.integrate_momentum_residual
FEMTools.integrate_PH_pressure_residual
FEMTools.compute_velocity_divergence
FEMTools.second_invariant
FEMTools.deviatoric_stress
FEMTools.viscoelastic_coefficients_phase
```

## Post-processing

```@docs
compute_strain_rate_stress_postprocess
rotate_stress!
update_old_stress_from_cells!
write_stokes_vtk
```
