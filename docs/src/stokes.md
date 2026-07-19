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

The saddle-point system is solved with a Powell–Hestenes / DYREL iteration: an
outer Arrow–Hurwicz pressure update wraps an inner Chebyshev-accelerated
dynamic-relaxation sweep on the momentum residual.

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

## Example

The scripts under `examples/stokes/` set up complete problems, including a
viscoelasto-plastic pure-shear test and a sinking-block buoyancy test:

```sh
julia --project=examples examples/stokes/vevp/stokes_2D_pure_shear.jl
julia --project=examples examples/stokes/sinking_block/sinking_block.jl
julia --project=examples examples/stokes/sinking_block/sinking_block_adj.jl
```

The adjoint sinking-block example accepts an explicit backend. It builds the
Triangle mesh on the host, then uploads mesh arrays, mixed connectivity,
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

## Driver

```@docs
solve_stokes_dyrel!
solve_stokes_adjoint_dyrel!
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

using the sign convention of the sinking-block reference. The example forms a
finite-element objective load for
`J(v_y) = -∫_{Ωobs} v_y dΩ`, solves the transpose system with
`solve_stokes_adjoint_dyrel!`, and uses Enzyme reverse mode on the element
momentum residual contraction to obtain density and viscosity sensitivities.
The returned sensitivity arrays contain raw element integrals. Their sums give
phase gradients; division by element area is used only to visualise a spatial
sensitivity density.

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
