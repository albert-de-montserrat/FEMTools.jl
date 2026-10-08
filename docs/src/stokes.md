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
- \alpha\frac{\partial T}{\partial t} = Q,
```

`Q` is the backend-resident volumetric source/sink array on the Stokes
pressure nodes. Positive values produce volume and negative values remove it;
the default is zero.

with a Maxwell viscoelastic deviatoric stress that carries stress history
`τ_old` across time steps. Density uses the linearised equation of state
`ρ = ρ0 (1 − α(T − Tref) + P/K)`. Per-phase properties (`η`, `ηb`, `G`, `α`,
`ρ0`, `K`) and the body-force parameters are grouped in a typed
`StokesMaterial`.

The pressure storage coefficient is written as `ηp` below. In the legacy
formulation `ηp = ηb`; with finite elastic compressibility it is `ηp = K`:

```math
R^P_i = \int_{\Omega_e} N_i\left[
    -\nabla\cdot v
    -\frac{P-P^n}{\eta_p\,\Delta t}
    +\alpha\frac{T-T^n}{\Delta t}
    +Q
\right]d\Omega.
```

This distinction matters: `ηb` is the bulk-viscosity parameter used by the
historical pressure formulation, while `K` is the elastic bulk modulus in the
equation of state and in the finite-compressibility pressure storage term.
`finite_K` selects the latter explicitly; plastic-model dispatch does not.

For the 2-D mixed-mesh state the saddle-point system is solved by `solve!`, which
wraps `solve_stokes_dyrel!`. An
outer Arrow–Hurwicz pressure update wraps an inner
Chebyshev-accelerated dynamic-relaxation sweep on the momentum residual. The
3-D T11/P1-discontinuous and Hex27/Q2--P1 discretisations are available through
a `StokesDR`/`MixedMesh` state for coupled visco-elasto-plastic problems. The
purely viscous Hex27 discretisation with four cell-local pressure modes has its
own state, [`CellPressureStokesDR`](@ref), solved by the same `solve!`.

The discrete adjoint uses the transpose of the same assembled element
operators and the same mixed spaces. It therefore computes gradients of the
*discrete* objective rather than a separately discretised continuous adjoint.

## Prescribed viscosity and assembled loads

For purely viscous 2-D forward flow, `solve_stokes_dyrel!` accepts
`viscosity=ηq`, an `nq × nels` matrix of positive finite viscosity values at
velocity quadrature points. It must match the state backend and precision.
Pass the same matrix as `η=ηq` to `assemble_viscosity_weighted_pressure_scaling!`.
The physical residual uses sampled viscosity; the conservative preconditioner
uses its element maximum. This override excludes plasticity, viscoelasticity,
coupled solves, and measured spectral estimates, and does not alter the material
stored in `dr` for subsequent stress-history or adjoint operations.

`body_force=(fx, fy)` adds assembled velocity-node loads,
``f_i = \int N_i b\,d\Omega``, to the existing EOS/gravity forcing. Supply load
vectors, not pointwise force densities. The solver subtracts them from both
outer and inner momentum residuals before applying Dirichlet constraints.
Loads must be finite and match the velocity size, backend, and precision.
The [SolKz and SolCx benchmarks](benchmarks.md) exercise these inputs.

## Solver state

```@docs
StokesMaterial
StokesDR
CellPressureStokesDR
StokesAdjointWorkspace
DruckerPrager
DruckerPragerCap
pressure_mass
FEMTools.velocity
FEMTools.stress
FEMTools.pressure
FEMTools.temperature
```

`DruckerPragerCap` takes angles in radians and finite numeric material
parameters. Its bulk modulus `Kb` must be positive, and both the initial
cohesion `C` and softening floor `C_min` must satisfy
`C*cos(ϕ) + sin(ϕ)*pT > 0` (substitute `C_min` for `C` at the floor).
This keeps the tensile-cap radius positive throughout cohesion softening.

The velocity and pressure fields live on separate node sets described by a
[`MixedMesh`](mesh.md), which also holds the precomputed per-field geometry in
`mesh.geometry`. Fill `dr.T` (and `dr.T0`) before solving. `solve!` assembles
the lumped pressure mass `dr.M_P` and the pressure scale `dr.γP` itself; only
the low-level [`solve_stokes_dyrel!`](@ref) expects them prepared with
[`FEMTools.assemble_viscosity_weighted_pressure_scaling!`](@ref).

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
dr = StokesDR(mesh, material)   # `mesh` is a 3-D MixedMesh
dr.v.z          # the third velocity component
dr.τ.yz         # a stress component that has no two-dimensional counterpart
```

The mixed-mesh solvers on this page accept a `StokesDR` and `MixedMesh` of the
same spatial dimension, 2 or 3, with one Dirichlet boundary condition per
velocity component; a dimension mismatch is a `MethodError`. The viscous Hex27
cell-pressure discretisation uses a [`CellPressureStokesDR`](@ref) and a plain
`Mesh` instead — see [Sinking block (3-D)](sinking_block_3d.md).

### Compact setup

Material properties accept scalars or phase tuples. For example,
`StokesMaterial(; η = (1f0, 10f0), ηb = Inf32)` creates two Float32 phases;
omitted shear modulus and bulk modulus are infinite, expansivity is zero,
density is one, and gravity is zero. Bulk viscosity still defaults to one;
set `ηb = Inf32` explicitly for the incompressible limit. Scalars apply to
every phase. Supplied properties, gravity, and `Tref` must share precision.

`StokesDR(mesh, material)` infers the backend, velocity and pressure node
counts, and integration-point stress layout from the mixed mesh. Material
precision and gravity dimension must match the coordinates. Use `stress_size`
to override the history layout, including `:none` for no stress history.

For cell-wise materials, pass the same backend-resident `1 × nels` phase array
as `phases_v` and `phases_P`. Each column holds one phase index; repeating it
for every local node is unnecessary. Omit both keywords for a uniform phase.

For cell-wise materials, pass the same backend-resident `1 × nels` phase array
as `phases_v` and `phases_P`. Each column holds one phase index; repeating it
for every local node is unnecessary. Omit both keywords for a uniform phase.

```julia
material = StokesMaterial(; η, ηb, G, α, ρ0, K, g, Tref)
dr = StokesDR(mesh, material)

bc_vx = DirichletBoundaryCondition(vx_nodes, vx_vals)
bc_vy = DirichletBoundaryCondition(vy_nodes, vy_vals)

stats = solve!(dr, mesh, bc_vx, bc_vy; dt = Δt, pressure_factor = γfact, workgroup)
```

The solve assembles the pressure scale into `dr.γP` itself and returns statistics
with `converged`, `iterations`, `residual`, and `history`; it throws when the
iteration does not converge unless `throw_on_failure = false`. `tolerance`,
`max_iterations`, and `check_interval` replace `ϵ_tol`, `total_iterMax`, and
`ncheck` of the positional form. Pass
`scaling_viscosity` to scale pressure with a different viscosity than the
momentum solve. Adjoint workflows still prepare `γP` with
`assemble_viscosity_weighted_pressure_scaling!` and use the `γP` positional form.

`mesh.geometry` retains the reference elements alongside both geometry arrays,
so the high-level assembly and solver calls infer elements and backend. The
expanded positional methods remain available for custom and adjoint workflows.
For the experimental Drucker--Prager cap, add
`plastic_history_size=(nq, mesh.nels)` to allocate per-integration-point `γ`
and `θ` history arrays; they are zeroed and are not allocated by default.
Call `commit_stokes_plastic_history!` once per converged step, before
overwriting `τ_old`, to accumulate that step into those arrays. Residual
iterations and `update_stokes_current_stress!` never update history.
`dr.P` is the trial pressure under the cap, so do not copy it into `dr.P0`:
record the corrected pressure with a store that appends one matrix to the
stress components — `(τxx, τyy, τxy, P_corrected)` in plane strain,
`(stress(dr)..., P_corrected)` in 3-D — in `update_stokes_current_stress!`, copy it into
an `nq × nels` array after the commit, and pass that array as
`solve_stokes_dyrel!(...; P_old)` on the next step.
For plain Drucker--Prager, pass an `IntegrationPointPlasticHistory(λ, ε̇pl, εpl)`
of `nq × nels` arrays as `update_stokes_current_stress!(...; plastic_history)`.
That call overwrites the multiplier `λ` and the plastic strain-rate invariant
`ε̇pl`, in plane strain and in 3-D, so repeating it is harmless. The accumulated
plastic strain `εpl` changes only in `FEMTools.update_plastic_history!`, which
adds `Δt·ε̇pl` and must be called once per accepted step.
The pressure kernel interpolates nodal pressure and temperature increments
directly, avoiding temporary per-node rate calculations.

### Finite compressibility, pressure history, and sources

These options control the pressure part of the mixed-mesh solver:

| Keyword | Meaning | Default |
|:--|:--|:--|
| `finite_K` | Use the material `dr.K` in `(P - P_old)/(K*Δt)` | `plastic isa DruckerPragerCap` |
| `P_old` | Accepted pressure from the previous physical step | `dr.P0` |
| `Qq` | Direct source field, or nonnegative spatial weights with `Q2D` | `nothing` |
| `Q2D` | Total signed source rate: area per time in plane strain, volume per time in 3-D | `nothing` |

`finite_K` is an independent physical choice. It can be set to `true` for a
viscous or Drucker--Prager solve, and to `false` for a cap solve when the
legacy `ηb` storage is wanted. When it is `true`, every phase's `K` must be
finite and strictly positive. When it is `false`, the pressure residual uses
`ηb`, preserving the pre-existing behavior. The default enables finite `K`
only for `DruckerPragerCap` because that model's trial pressure is corrected
by its cap return map.

`P_old` is the pressure accepted at the end of the previous physical step,
not merely the last nonlinear iterate:

- a pressure-node vector is interpolated to each pressure quadrature point;
- an `nq × nels` matrix is read directly at the quadrature points;
- the matrix form is the required form when the accepted pressure includes
  the Drucker--Prager cap correction.

For a cap solve, `dr.P` is the current trial pressure. Do not use it as the
next `P_old` without first applying the constitutive correction. The accepted
pressure workflow is:

```julia
nq = size(dr.τ.xx, 1)
P_old = similar(dr.P, nq, mesh.nels)
fill!(P_old, zero(eltype(dr.P)))
P_corrected = similar(P_old)
τ_and_P = (dr.τ.xx, dr.τ.yy, dr.τ.xy, P_corrected)

for step in 1:nsteps
    stats = solve_stokes_dyrel!(
        dr, mesh, bc_vx, bc_vy, Δt, γP;
        plastic = cap, finite_K = true, P_old,
        verbose = false,
    )
    stats.converged || error("Stokes solve did not converge")

    # The fourth entry receives the accepted/corrected pressure at each IP.
    update_stokes_current_stress!(dr, mesh, τ_and_P, Δt;
                                  plastic = cap)
    commit_stokes_plastic_history!(dr, mesh, Δt; plastic = cap)
    copyto!(P_old, P_corrected)
end
```

For a non-cap model, a nodal `P_old` is usually sufficient. The caller owns
this physical-time update; the solver does not overwrite `P_old` or guess
when a step has been accepted.

`Q` is the volumetric source in the weak continuity residual. Positive `Q`
creates material and negative `Q` removes it. It is stored in `dr.Q`, and the
solver can fill it from the `Qq` and `Q2D` keywords:

```julia
# A prescribed total injection in a plane-strain model.
Qq = similar(dr.P)
fill!(Qq, one(eltype(dr.P)))
stats = solve_stokes_dyrel!(
    dr, mesh, bc_vx, bc_vy, Δt, γP;
    Qq, Q2D = 1.0e-6,
)
```

There are two deliberately different `Qq` modes:

1. `Qq` without `Q2D` is copied directly into `dr.Q`. Its entries are local
   rates with units of inverse time and may be signed.
2. `Qq` with `Q2D` is a nonnegative distribution weight. The solver rescales
   it using the pressure quadrature so that
   `∑ₑ ∫_{Ωₑ} Q dΩ = Q2D`. Thus `Q2D` has units of area per time in 2-D,
   positive means injection, and negative means extraction. The shape of
   `Qq` must match the pressure field and its backend must match the Stokes
   state.

`Q2D = 0` clears the stored source. A nonzero `Q2D` with no positive weight
is rejected rather than silently producing no injection. These source options
belong to the two-dimensional mixed-mesh solver; the specialized three-
dimensional cell-local pressure path does not currently carry `Q`.

### Fluid pressure in the yield model

`dr.Pf` is a fluid (pore or magma) pressure on the pressure DoFs, zero by
default. The yield models see the effective pressure `P − Pf`: Drucker--Prager
yields at `τII = C cosϕ + (P − Pf) sinϕ`, and the tensile cap of
`DruckerPragerCap` fails when `P − Pf` reaches `pT`. The momentum balance and the
equation of state keep the total pressure `P`, and a pressure corrected by the
cap is returned as a total pressure. Fill `Pf` before the solve, for example
with a lithostatic pressure for a fluid-saturated crust:

```julia
copyto!(dr.Pf, P_litho)
```

The plastic adjoint solver does not support a nonzero `Pf` and rejects it.

### Coupled thermal--Stokes relaxation

`solve_coupled!` advances one thermal DR iteration during every inner
Stokes velocity iteration. The thermal mesh must share the Stokes velocity-node
numbering; after each thermal update, its continuous nodal temperature is
gathered onto the discontinuous pressure DoFs used by the Stokes residuals.

```julia
stats = solve_coupled!(
    thermal, stokes, thermal_mesh, stokes_mesh, bc_T, (bc_vx, bc_vy, bc_vz);
    dt = Δt, pressure_factor = γfact, workgroup,
)
```

The returned Stokes statistics additionally contain `err_T` and
`thermal_iterations`. On the 2-D path `converged` is true only when the outer
test `min(err_abs, err_rel) < ϵ_tol` passed (and the thermal state converged), so
a run that ends on `total_iterMax` reports `false` and `reached_total_iter` says
why; `err` is then the inner velocity residual, and `err_abs` and `err_rel` carry
the outer error. The caller still owns physical-time history: set
`thermal.T0`, `stokes.P0`, and the old Stokes stresses before each coupled
solve. The thermal `ncheck` cadence follows the Stokes `ncheck` keyword.

#### Shear heating

`shear_heating = true` makes the coupling two-way, in plane strain and in 3-D:
the dissipation

```math
\Phi = \boldsymbol{\tau} : \left(\dot{\boldsymbol{\varepsilon}} - \dot{\boldsymbol{\varepsilon}}^{el}\right),
\qquad
\dot{\boldsymbol{\varepsilon}}^{el} = \frac{\boldsymbol{\tau} - \boldsymbol{\tau}^{old}}{2G\Delta t},
```

is added to the thermal heat source. `τ` is the deviatoric stress of the
momentum residual, plastic correction included, so `Φ` is the viscous plus
plastic shear dissipation without the elastically stored power; volumetric
plastic work is not included. `Φ` is recomputed from the current velocity at
every thermal convergence check and integrated at the velocity quadrature
points, so the thermal element must share the velocity element's quadrature.
`thermal.source` keeps the caller's own source and is not modified.

```julia
stats = solve_coupled!(
    thermal, stokes, thermal_mesh, stokes_mesh, bc_T, bc_vx, bc_vy;
    dt = Δt, workgroup, shear_heating = true,
)
```

In 2-D, pass `bc_vx` and `bc_vy` positionally as before. In 3-D, pass the tuple
shown above. The 3-D signed pressure basis uses a positive Jacobi modal mass
instead of direct lumping.

### Three-dimensional viscous cell-pressure state

The viscous Hex27/Q2--P1 discretisation stores four pressure modes per cell
and is solved through a [`CellPressureStokesDR`](@ref) built from a Hex27 `Mesh`:

```julia
material = StokesMaterial(; η = (1.0, 100.0), ρ0 = (1.0, 2.0), g = (0.0, 0.0, -1.0))
dr = CellPressureStokesDR(mesh, material; phases = cell_phase)
bc_v = (bc_vx, bc_vy, bc_vz)
stats = solve!(dr, mesh, bc_v; tolerance = 1e-6, max_iterations = 50_000)
```

`bc_v` holds one `DirichletBoundaryCondition` per velocity component, so each
component may be constrained on its own node set; nonzero values impose a
far-field flow. The solver applies them before the first residual assembly and
after every velocity update, so the initial guess in `dr.v` need not satisfy
them. The state owns the residuals, preconditioner, pressure mass, and reference
tables, so repeated solves on the same state allocate no mesh-sized storage. The
preconditioner is refilled from the material at the start of every solve.

For an objective `J(u)` with velocity derivative `objective_v`, the adjoint
solve reuses the state's material and scratch and writes caller-owned outputs:

```julia
λv = ntuple(_ -> zeros(mesh.nnodes), 3)
λP = zeros(4, mesh.nels)
solve_adjoint!(dr, mesh, bc_v; objective_v, λv, λP, tolerance = 1e-6)
gradients = stokes_material_gradient_3d(dr, mesh, λv)
```

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
julia --project=examples examples/miniapps/stokes/popov_extension_2D/popov_extension_2D.jl
```

The Popov extension driver is a small unstructured T7/P1-discontinuous
tensile-cap case using only the Table 1 / Figure 6b setup. It meshes the domain
with `FEMTools.triangulate_t7_mesh` from a target triangle area, `max_area`, in
the paper's range of `5e-6` to `3e-4`, so the script needs `using Triangulate`
to load that extension. Inputs are nondimensionalised
with `L0=1 m`, `S0=10 MPa`, and `t0=50 yr`; therefore `nsteps=2000` represents
100 kyr and one solver step is one paper time increment. The returned `scales`
named tuple converts dimensionless fields back to SI units.
The returned fields distinguish trial pressure, corrected pressure, and
integration-point accumulated volumetric plastic strain `χ` (the Figure 6b
quantity), plus deviatoric plastic strain. Trial pressure is the nodal field the
solver carries, formed each step from the previous step's corrected pressure,
which the driver carries at integration points; corrected pressure is the tensile-cap return map's pressure at
integration points, returned as an `nq × nels` array rather than a nodal field.
Set `write_output=true` to write legacy ASCII VTK files with velocity and
projected trial pressure as point data, plus cell fields for both pressure
states, both plastic-strain measures, and the stress/strain-rate second
invariants.
The returned `cross_section` contains the piecewise-constant cell profile
`(; x, values, y)` through `χ`; `section_y=0.25` matches Figure 6b's `A–A′`
line and can be changed for diagnostics.

The driver carries the weak seed of the authors' GeoTech2D release: a
semicircular inclusion on the middle of the bottom boundary, centre `(Lx/2, 0)`
and radius `0.025`, meshed as its own Triangle region so the mesh conforms to
it, and given a ten times smaller shear modulus than the bulk. Every other
property is shared. Without it the domain deforms homogeneously and cannot
localise at any step count, because a single phase plus fully prescribed
boundary velocities makes the uniform field an exact solution; an unstructured
mesh does not change that on its own. `vy` is prescribed on the top and bottom
only, so the side walls stay free to move vertically. The returned `phases`
vector gives each element's region, and `phase` is written as a VTK cell field.

The ice-bridge miniapp generates a 20 km by 6 km arch-shaped body with a
4 km-radius semicircular opening cut into its bottom, then applies gravity and
linear visco-elastic ice rheology. Mesh advection is enabled by default and
recomputes the mesh geometry in place after each Lagrangian update.

See [Sinking block](sinking_block.md) and [Sinking block (3-D)](sinking_block_3d.md)
for the discretisations, physical setup, output, figure, and material-gradient
checks of each.

The 2-D adjoint sinking-block example accepts an explicit backend. It builds the
Gmsh mesh on the host, then uploads mesh arrays, mixed connectivity,
geometry, phase indices, and boundary data before launching kernels:

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
solve_coupled!
solve!(::StokesDR{<:Any, D}, ::MixedMesh{D}, ::NTuple{D, DirichletBoundaryCondition}) where {D}
solve_stokes_dyrel!
solve_adjoint!(::StokesDR{<:Any, D}, ::MixedMesh{D}, ::NTuple{D, DirichletBoundaryCondition}) where {D}
solve_stokes_adjoint_dyrel!
solve!(::CellPressureStokesDR, ::Mesh, ::NTuple{3, DirichletBoundaryCondition})
solve_adjoint!(::CellPressureStokesDR, ::Mesh, ::NTuple{3, DirichletBoundaryCondition})
stokes_material_gradient_3d
FEMTools.FrozenAdjointOperator
FEMTools.MatrixFreeAdjointOperator
update_stokes_current_stress!
commit_stokes_plastic_history!
FEMTools.IntegrationPointPlasticHistory
FEMTools.update_plastic_history!
```

## Discrete adjoint and material sensitivities

After a converged mesh-based `solve!`, the 2-D adjoint is one call that reuses the
forward pressure scale left in the state:

```julia
stats = solve!(dr, mesh, bc_vx, bc_vy; dt, pressure_factor)
λ = (zeros(mesh.nnodes), zeros(mesh.nnodes), zeros(mesh.nnodesP))
solve_adjoint!(dr, mesh, bc_vx, bc_vy; dt, objective_vx, objective_vy,
               λvx = λ[1], λvy = λ[2], λP = λ[3])
```

The inputs `λ` are the initial iterate, so pass the previous design iteration's
fields to warm-start. `dt`, `plastic`, and the phase layouts must match the forward
solve. `solve_adjoint!` throws if the state's `γP` was never assembled, and, like
`solve!`, on non-convergence unless `throw_on_failure = false`.

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
`solve_adjoint!`, and uses Enzyme reverse mode on the element
momentum residual contraction to obtain density and viscosity sensitivities.
The returned sensitivity arrays contain raw element integrals. Their sums give
phase gradients; division by element area is used only to visualise a spatial
sensitivity density.

The 3-D viscous operator is symmetric, so the cell-pressure `solve_adjoint!`
reuses the forward residual and preconditioner with the objective derivative as
the momentum load. It solves `A^Tλ = J_u`, so its material derivative uses the
opposite contraction `-λ^T R_m`.
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

### Frozen operator and solver controls

The mixed-mesh adjoint runs in two and three dimensions. In three dimensions,
pass the objective load and the adjoint velocity as one array per direction,
`(objective_x, objective_y, objective_z)` and `(λvx, λvy, λvz)`, with the
constrained nodes as `v_nodes = (vx_nodes, vy_nodes, vz_nodes)`; the
plane-strain spelling with separate `vx`/`vy` arguments remains available.
Only `operator = :blocks` is implemented in three dimensions. It is also the only
operator that supports a plastic model together with a nonzero fluid pressure
`dr.Pf`, because only the stored blocks differentiate the yield model at the
effective pressure.

The solver transposes the Powell–Hestenes augmented system, whose momentum rows
carry `W·RP` with the element-local `W = ∂Rv/∂Pnum · γP/M_P`. A parameter
contraction against the plain residual `[Rv; RP]` therefore uses the pressure
multiplier `λP + Wᵀλv`.

The forward state must be converged before the adjoint solve. At that fixed
state the transpose Jacobian is constant, and `operator` chooses how it is
applied throughout the Powell–Hestenes / DYREL solve. The default
`operator = :blocks` assembles the element blocks once; each subsequent
application is an element gather, dense products, and scatter, reevaluating no
rheology and invoking no automatic differentiation.

The inner velocity iteration stops after reducing its residual by `rel_drop`;
the outer pressure iteration continues until `adjoint_tol` or
`max_ph_iterations`. `iterMax` limits one inner solve and `total_iterMax` limits
the complete adjoint. With `measure_λmax = true`, power iteration measures the
largest eigenvalue of the Jacobi-preconditioned velocity block instead of using
its looser Gershgorin bound. The returned statistics report both values and the
number of power iterations. Set `measure_λmax = false` to use the Gershgorin
estimate directly; `operator = :enzyme` always uses that estimate, because
power iteration needs an operator application and the reverse-mode path has
none to offer cheaply.

The adjoint velocity and `λP` are initial guesses as well as output arrays. Zero them
for a cold solve; in an optimization loop, leave the previous design's adjoint
in place to warm-start the next solve.

An optimization loop can also construct
`StokesAdjointWorkspace(dr, vx_nodes, vy_nodes)` (with `vz_nodes` appended in
three dimensions) once and pass it as
`workspace` on every adjoint solve. This reuses the residual, rate, pullback,
and boundary buffers instead of allocating them for every design. Construct it
with `enzyme=true` only when using `operator = :enzyme`; the default block and
matrix-free workspaces omit the nine reverse-mode-only arrays.

If the velocity residual stalls, inspect the measured-to-Gershgorin ratio and
increase the iteration budgets before changing tolerances. If the velocity
residual drops but the pressure residual does not, the outer PH iteration is
the bottleneck; lowering `rel_drop` only spends more work on the already-solved
subproblem.

For T7/P1-disc the cached blocks hold 147 floating-point values per element,
about 1.2 kB per element in `Float64`, once the symmetry of a viscous tangent is
exploited; a plastic model raises that to 280 values, about 2.2 kB. For the
three-dimensional T11/P1-disc pair a non-symmetric tangent, plastic or with a
finite bulk modulus, holds 1 501 values per element, about 12 kB. In two
dimensions, two alternatives store nothing per element when that footprint is
unsuitable.

`operator = :matrix_free` rebuilds the transpose products by forward-mode
directional differentiation of the element residuals, at about three residual
evaluations per element per application. It needs `plastic === nothing`: for a
symmetric element tangent a directional derivative *is* the transposed product,
which is what removes the need to store anything. The symmetry is verified once
at construction rather than assumed.

`operator = :enzyme` rebuilds the same products by reverse-mode differentiation,
three sweeps per application. It is the slowest of the three and the only one
that avoids block storage for a plastic tangent.

See `examples/miniapps/stokes/sinking_block_adj/sinking_block_adj.jl` for a complete solve.

## Assembly

```@docs
FEMTools.assemble_momentum_residual_matrices_atomix!
FEMTools.assemble_momentum_jacobian_matrices_atomix!
FEMTools.assemble_augmented_momentum_jacobian_matrices_atomix!
FEMTools.assemble_pressure_residual_matrices_atomix!
FEMTools.assemble_viscosity_weighted_pressure_scaling!
FEMTools.assemble_shear_heating!
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

### [Principal stresses](@id principal-stresses)

For a symmetric stress tensor ``\boldsymbol{\sigma}``, principal stresses
``\lambda_k`` and directions ``\mathbf{n}_k`` satisfy

```math
\boldsymbol{\sigma}\,\mathbf{n}_k = \lambda_k\mathbf{n}_k,
\qquad \mathbf{n}_j^\mathsf{T}\mathbf{n}_k = \delta_{jk},
\qquad \lambda_1 \geq \lambda_2 \geq \lambda_3.
```

In 2D there are two eigenpairs; in 3D there are three. Values have the same
units as the supplied stress, and directions are dimensionless unit vectors.

`compute_principal_stresses` computes eigenpairs at each supplied stress
sample and returns a `PrincipalStresses` container, retaining its array shape,
floating-point type, and backend. It accepts
`dr.τ` directly or component tuples in `FEMTools.stress(dr)` order. It does not
refresh solver stress, interpolate pressure, or average samples.

```julia
τ = ([2.0, -2.0], [-1.0, -1.0], [0.5, 0.0]) # xx, yy, xy
P = [7.0, 7.0]                               # collocated physical pressure
principal = compute_principal_stresses(τ; pressure=P)
σ1, σ2 = principal.values                    # descending algebraic order
n1x, n1y = principal.directions[1]            # direction paired with σ1
compute_principal_stresses!(principal, τ; pressure=P) # reuse the buffers
```

`principal.values` and `principal.directions` hold the arrays. The in-place
method fills those same arrays and returns `principal`. To supply your own
buffers, wrap them with `PrincipalStresses(value_arrays, direction_arrays)`;
this constructor does not copy them, and the compute method validates them.

Without pressure these are eigenpairs of the supplied tensor. Supplying
positive-compressive pressure gives principal **total Cauchy stresses** of
`σ = τ - P I`, with positive values denoting tension. Pressure shifts values
without changing directions. For the tensile-cap model, use corrected physical
pressure from the integration-point stress/pressure store; trial `dr.P` and
the solver relaxation field `dr.Pnum` are not substitutes.

The 2D method returns two in-plane eigenpairs. Full plane-strain principal
stresses include the out-of-plane direction and require a 3D tensor:

```julia
τxx, τyy, τxy = τ
τzz = .-(τxx .+ τyy)  # deviatoric plane-strain constraint
zero_shear = zero.(τxx)
full = compute_principal_stresses(
    (τxx, τyy, τzz, τxy, zero_shear, zero_shear); pressure=P,
)
```

The out-of-plane value can be the largest, middle, or smallest. In general 3D,
tuple input uses `(xx, yy, zz, xy, xz, yz)`; tensor containers are read by named
fields, because their `Tuple` conversion uses a different shear ordering.

For example, a diagonal 3D tensor gives three principal values directly:

```jldoctest
julia> using FEMTools

julia> τ = ([3.0], [1.0], [-4.0], [0.0], [0.0], [0.0]);

julia> principal = compute_principal_stresses(τ; pressure=[2.0]);

julia> map(only, principal.values)
(1.0, -1.0, -6.0)

julia> compute_principal_stresses!(principal, τ; pressure=[2.0]) === principal
true
```

Directions are unoriented axes: `n` and `-n` mean the same direction. The
largest-magnitude component is made nonnegative. Repeated eigenvalues define
an eigenspace rather than unique axes; the returned basis is orthonormal but
need not vary continuously between samples or time steps.

For cell output, compute eigenpairs from the existing cell diagnostics:

```julia
# post holds the element-averaged stress diagnostics and P_cell is collocated.
cells = compute_principal_stresses((post.τxx, post.τyy, post.τxy); pressure=P_cell)
write_vtk(path, mesh; cell_data=(
    sigma1=cells.values[1], principal_direction1=cells.directions[1],
    sigma2=cells.values[2], principal_direction2=cells.directions[2],
))
```

These are eigenpairs of the averaged tensor, which differ from averages of
integration-point principal values. Do not average directions. Integration-point
input remains `nq × nels` output and requires a separate, explicit sampling or
projection step before writing cell fields.

Buffers passed to the mutating method must not alias input or output arrays.
Both methods synchronize the backend before returning and throw on nonfinite
input or failed local eigenpair checks. CPU Float32/Float64 are checked; CUDA execution requires separate hardware
validation.

```@docs
PrincipalStresses
compute_strain_rate_stress_postprocess
compute_principal_stresses
compute_principal_stresses!
rotate_stress!
update_old_stress_from_cells!
write_stokes_vtk
```

## Q2/P1 quadrilateral meshes

Use `ReferenceElement(QuadraticElement{2, 9, Float64})` for velocity and
`ReferenceElement(LinearElement{2, 3, Float64})` for pressure in `MixedMesh`.
For Q9 velocity cells, the three pressure values are cell-local at the center
and positive x/y edge midpoints. The pressure basis `(1-ξ-η, ξ, η)` is evaluated
on the reference square, so this is P1 pressure, not bilinear Q1 pressure.
Pressure integration uses velocity-cell geometry; pressure scaling uses positive
`∫Nᵢ² dΩ` weights. The [quadrilateral benchmarks](benchmarks.md) show complete runs.

## Recording component convergence histories

With `collect_history=true`, `solve_stokes_dyrel!` returns check records in
`stats.history` with `iter`, `err_v`, `err_v_components`, and `err_P`.
`err_v_components` contains a scaled residual norm for each velocity component;
its maximum equals `err_v`. Both outer and inner checks record the pressure
residual assembled at that check. The final outer record corresponds to the
returned convergence errors. The [Stokes benchmarks](benchmarks.md) overlay
`vx`, `vy`, and `p` in one separate convergence figure.
