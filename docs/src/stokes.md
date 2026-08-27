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
3-D Hex27/Q2--P1 method uses three caller-owned velocity arrays and a `4 × nels`
cell-local pressure array with diagonally preconditioned residual updates.

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

Velocity-node quantities are grouped as [field containers](field_containers.md).
The velocity is `dr.v`, its pseudo-transient rate `dr.∂v∂τ`, the momentum
residual `dr.Rv` with snapshot `dr.Rv0`, the row-sum Jacobian estimate
`dr.∂Rv∂v`, and the diagonal preconditioner `dr.PC_v` — each a vector field
whose `.x` and `.y` are the arrays for the x- and y-momentum equation. The
current and previous deviatoric stress are `dr.τ` and `dr.τ_old`, with
components `.xx`, `.yy`, `.xy`. Pressure quantities remain plain arrays.
`velocity(dr)` and `stress(dr)` return the component arrays as tuples for code
that wants them positionally.

### Spatial dimension

`StokesDR` carries its dimension as the type parameter `ndim`, taken from the
length of the gravity vector. The default `g` has two components and gives the
two-dimensional state above. A three-component `g` — including
`(0.0, 0.0, 0.0)` for a gravity-free problem — gives `VectorField3D` velocity
fields and a `SymmetricTensor3D` stress history with the six independent
components, so `stress(dr)` returns six arrays instead of three:

```julia
material = StokesMaterial(; η, ηb, G, α, ρ0, K, g = (0.0, 0.0, -9.81), Tref)
dr = StokesDR(backend, nnodes_v, nnodes_P, material)
dr.v.z          # the third velocity component
dr.τ.yz         # a stress component that has no two-dimensional counterpart
```

`nnodes_v` and `nnodes_P` accept a dimension tuple as well as a node count, so
a cell-local pressure layout is expressed as `StokesDR(backend, nnodes_v,
(4, nels), material)`.

The mixed-mesh solvers on this page are two-dimensional and accept only
`StokesDR{<:Any, 2}`; passing a three-dimensional state is a `MethodError`
rather than a silent solve that ignores the third component. The existing
matrix-free 3-D method takes its arrays positionally and does not consume a
`StokesDR` — see [Sinking block (3-D)](sinking_block_3d.md).

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

### Three-dimensional array layout

The Hex27/Q2--P1 method uses multiple dispatch rather than `StokesDR`, because
its four pressure modes are cell-local rather than stored on a pressure-node
mesh:

```julia
velocity = ntuple(_ -> zeros(mesh.nnodes), 3)
pressure = zeros(4, mesh.nels)

stats = solve_stokes_dyrel!(
    velocity, pressure, mesh, cell_phase, η, ρ, g, fixed_nodes;
    ϵ_tol = 1e-6,
)
```

`fixed_nodes` is an `NTuple{3}` containing the constrained nodes for each
velocity component, held at zero unless `bc_values` supplies one velocity per
entry of `fixed_nodes`, which is how a far-field flow is imposed on the
boundary. The matching `solve_stokes_adjoint_dyrel!` method accepts the same
storage plus a three-component objective load. `solve_stokes_3d!` and
`solve_stokes_adjoint_3d!` remain compatibility wrappers.

`mesh.geometry` is shared by every kernel above and indexed
`geometry[q, cell]`; see [Geometry Precomputation](mesh.md#Geometry-Precomputation).

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

The scripts under `examples/miniapps/stokes/` set up complete problems, including a
viscoelasto-plastic pure-shear test and a sinking-block buoyancy test:

```sh
julia --project=examples examples/miniapps/stokes/stokes_2D_pure_shear/stokes_2D_pure_shear.jl
julia --project=examples examples/miniapps/stokes/sinking_block/sinking_block.jl
julia --project=examples examples/miniapps/stokes/sinking_block_adj/sinking_block_adj.jl
julia --project=examples examples/miniapps/stokes/sinking_block_3D/sinking_block_3D.jl
julia --project=examples examples/miniapps/stokes/sinking_block_3D_adj/sinking_block_3D_adj.jl
julia --project=examples examples/miniapps/stokes/ice_bridge_2D/ice_bridge_2D.jl
```

The ice-bridge miniapp generates a 20 km by 6 km arch-shaped body with a
4 km-radius semicircular opening cut into its bottom, then applies gravity and
linear visco-elastic ice rheology. Mesh advection is enabled by default and
rebuilds the geometric cache after each Lagrangian update.

See the [Sinking block](sinking_block.md) page for the 2-D and 3-D
discretisations, physical setup, output, figure, and material-gradient checks.

The 2-D adjoint sinking-block example accepts an explicit backend. It builds the
Triangle mesh on the host, then uploads mesh arrays, mixed connectivity,
geometry caches, phase indices, and boundary data before launching kernels:

```julia
using CUDA
include("examples/miniapps/stokes/sinking_block_adj/sinking_block_adj.jl")

result = main(backend = CUDABackend(), show_plot = false)
```

Loading CUDA activates the FEMTools CUDA extension, for which
`TA(CUDABackend()) === CuArray`. A functional NVIDIA driver is required for
allocation and execution. Plotting is host-side; keep `show_plot = false` for
headless accelerator runs.

## Drivers

```@docs
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
= -\lambda_e^T \frac{\partial R_e}{\partial m_e},
```

using the sign convention of the sinking-block reference. The 2-D example forms a
finite-element objective load for
`J(v_y) = -∫_{Ωobs} v_y dΩ`, solves the transpose system with
`solve_stokes_adjoint_dyrel!`, and uses Enzyme reverse mode on the element
momentum residual contraction to obtain density and viscosity sensitivities.
The returned sensitivity arrays contain raw element integrals. Their sums give
phase gradients; division by element area is used only to visualise a spatial
sensitivity density.

The 3-D viscous operator is symmetric, so its adjoint DYREL method reuses the
3-D forward residual and preconditioner with the objective derivative as the
momentum load. `stokes_material_gradient_3d` then contracts the forward and
adjoint velocity fields analytically. `test/test_stokes_3d_reference.jl`
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

See `examples/miniapps/stokes/sinking_block_adj/sinking_block_adj.jl` for a complete solve
and `examples/benchmarks/stokes/adjoint_perf/adjoint_perf.jl` for a headless mesh/contrast sweep.
Forward comparisons are available in
`examples/benchmarks/stokes/forward_lambda_perf/forward_lambda_perf.jl` and
`examples/benchmarks/stokes/forward_lambda_shear_band_perf/forward_lambda_shear_band_perf.jl`.

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
