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
\nabla \cdot v + \frac{1}{\eta_b}\frac{\partial P}{\partial t} = 0,
```

with a Maxwell viscoelastic deviatoric stress that carries stress history
`τ_old` across time steps. Density uses the linearised equation of state
`ρ = ρ0 (1 − α(T − Tref) + P/K)`. Per-phase properties (`η`, `ηb`, `α`, `ρ0`,
`K`) are stored as `NTuple{nphases, FP}` scalars encoded in the type parameters;
`g` and `Tref` are stored on the solver state.

The saddle-point system is solved with a Powell–Hestenes / DYREL iteration: an
outer Arrow–Hurwicz pressure update wraps an inner Chebyshev-accelerated
dynamic-relaxation sweep on the momentum residual.

## Solver state

```@docs
StokesDR
DruckerPrager
```

The velocity and pressure fields live on separate node sets described by a
[`MixedMesh`](mesh.md); precompute per-field geometry once with a
[`MixedMeshCache`](mesh.md). Fill `dr.T` (and `dr.T0`) before solving. The lumped
pressure mass `dr.M_P` must be assembled — see
[`FEMTools.assemble_viscosity_weighted_pressure_scaling!`](@ref) — before the
first call.

## Example

The scripts under `examples/stokes/` set up complete problems, including a
viscoelasto-plastic pure-shear test and a sinking-block buoyancy test:

```sh
julia --project=examples examples/stokes/vevp/stokes_2D_pure_shear.jl
julia --project=examples examples/stokes/sinking_block/sinking_block.jl
```

## Driver

```@docs
solve_stokes_dyrel!
update_stokes_current_stress!
```

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
