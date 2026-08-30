# Solver

## Current reality

The package uses matrix-free finite-element residual assembly and
pseudo-transient dynamic relaxation (DR).

Shared DR behavior lives in `src/dynamic_relaxation/`. A problem subtype
provides arrays through `dr_fields`, a label through `dr_name`, and scalar
controls such as `CFL`, `c_fact`, and `ϵ`. `solve_dynamic_relaxation!` assembles
residuals, periodically refreshes Jacobian row-sum/diagonal information,
enforces Dirichlet values, updates Chebyshev parameters, and errors on
non-convergence.

Physics layers:

- Heat diffusion: `ThermalMaterial`, `ThermalDiffusionDR`, atomic/colored
  residual assembly, and `solver!` for one physical time step.
- Lithostatic pressure: `LithostaticPressureDR`, pressure-dependent density,
  atomic/colored assembly, and `solver!`.
- 2-D Stokes: `StokesDR` on a `MixedMesh`, viscoelastic rheology with optional
  Drucker-Prager plasticity, pressure scaling, Powell-Hestenes outer iteration,
  DYREL velocity iteration, stress history, and a discrete adjoint.
- Coupled 2-D/3-D thermal--Stokes: `solve_coupled_dyrel!` advances one thermal DR
  step per inner Stokes velocity step and gathers the continuous thermal field
  onto the discontinuous pressure-temperature DoFs.
- 3-D Stokes: T11 or Hex27/Q2 velocity with cell-local P1 pressure in a
  `StokesDR`/`MixedMesh` state for coupled visco-elasto-plastic solves. The
  original caller-owned viscous forward/adjoint and material-gradient path is
  Hex27-specific.
- The frozen forward/adjoint operator paths intentionally assemble dense
  element blocks for repeated application. Sparse matrices in tests serve as
  reference oracles, not the production solve path.

Automatic differentiation is part of the numerical implementation:
ForwardDiff builds selected element Jacobians and Enzyme supports transpose and
material-gradient paths. `ADJOINT_PERF_PLAN.md` contains detailed historical
measurements and decisions for the 2-D adjoint; consult it before revisiting
those optimizations.

Primary checks are the residual, assembler, convergence, Stokes, adjoint,
3-D reference, inference, and allocation tests under `test/`.

## Mathematical formulation

### Finite-element notation

On an element `e`, FEMTools maps reference coordinates `ξ` to physical
coordinates with

```math
x(ξ) = \sum_a N_a(ξ)x_a,
\qquad
\nabla_x N = \nabla_ξ N\,J^{-1},
\qquad
dΩ_q = |\det J_q|w_q.
```

`precompute_geometry` stores `(∇N, dΩ)` for every element and quadrature point.
Element residuals are evaluated in statically sized vectors, then atomically or
color-wise scattered into global vectors. The atomic and colored paths change
only the scatter strategy, not the element mathematics.

For multi-phase fields, material properties are selected from nodal/cell phase
indices and interpolated at quadrature points. The shared linearized equation
of state is

```math
ρ_q = ρ_{0,q}\left[1-α_q(T_q-T_\mathrm{ref})+β_qP_q\right],
\qquad β_q=(1/K)_q.
```

Compliance `1/K` is interpolated instead of `K`, so `K=∞` produces the finite
incompressible limit `β=0`, including where quadratic shape functions are
negative.

### Heat diffusion

The target physical equation is

```math
ρC_p\frac{\partial T}{\partial t}
= \nabla\!\cdot(k\nabla T)+s.
```

With backward Euler in physical time, the implemented element residual for
node `i` is exactly

```math
R^T_{e,i}=\sum_q\left[
N_i\left(-T_i^{n+1}+T_i^n+
\frac{Δt}{ρ_q C_{p,q}}s_i\right)
-\frac{Δt\,k_q}{ρ_q C_{p,q}}
\nabla N_i\!\cdot\nabla T^{n+1}
\right]dΩ_q.
```

Thus the transient and source terms use the element nodal values `T_i` and
`s_i`, while the diffusion term uses the quadrature-point gradient. The solver
holds `T0=T^n` fixed and iterates `T=T^{n+1}` until `R^T=0`. ForwardDiff
differentiates this same discrete residual with respect to the local current
temperature to obtain row sums and diagonal preconditioning data.

### Lithostatic pressure

The strong balance and its implemented weak residual are

```math
-\nabla P+ρ(P,T)\,g=0,
```

```math
R^P_{e,i}=\int_{Ω_e}
\nabla N_i\!\cdot\left(ρg-\nabla P\right)dΩ.
```

At quadrature points this becomes

```math
R^P_e=\sum_q
\left[B_q(ρ_qg)-B_q(B_q^TP_e)\right]dΩ_q,
```

where rows of `B_q` are physical shape-function gradients. Pressure is the
unknown; temperature is a frozen coefficient. Natural boundaries impose zero
normal flux of `∇P-ρg`, while supplied Dirichlet nodes pin pressure.

### Shared dynamic relaxation

Heat and lithostatic pressure solve `R(u)=0` with a diagonally preconditioned,
Chebyshev-accelerated pseudo-time iteration. If `D` is the assembled absolute
Jacobian diagonal and `s` its absolute row sum, the conservative upper spectral
estimate is

```math
λ_\max=\max_i\frac{s_i}{D_i}.
```

The rate and field updates are

```math
r^{k+1}=D^{-1}R(u^k)+β_kr^k,
\qquad
u^{k+1}=u^k+α_kr^{k+1}.
```

At each check, a Rayleigh-like estimate is formed from the accepted pseudo-step
`δu` and residual change:

```math
λ_\min\approx
\frac{|δu^T D^{-1}ΔR|}{δu^Tδu}.
```

The Chebyshev parameters used in code are

```math
Δτ=\frac{2\,\mathrm{CFL}}{\sqrt{λ_\max}},
\quad c=2c_\mathrm{fact}\sqrt{λ_\min},
\quad α=\frac{2Δτ^2}{2+cΔτ},
\quad β=\frac{2-cΔτ}{2+cΔτ}.
```

Dirichlet residuals and rates are zeroed before spectral estimation and update;
the unknown is repinned afterward. The scalar DR solve stops when
`‖R^k‖₂/max(‖R^0‖₂, eps) < ϵ` and throws if it exhausts its budget.

### Stokes equations and mixed weak form

The Stokes paths solve momentum and pressure/mass balance

```math
\nabla\!\cdot(τ-PI)+ρg=0,
```

```math
\nabla\!\cdot v
+\frac{1}{η_b}\frac{P-P^n}{Δt}
-α\frac{T-T^n}{Δt}=0.
```

The code uses the sign-equivalent element residuals

```math
R^v_{e,i}=\int_{Ω_e}
\left[\nabla N_i\!\cdot(τ-PI)-N_iρg\right]dΩ,
```

```math
R^p_{e,i}=\int_{Ω_e}N_i\left[
-\nabla\!\cdot v
-\frac{P-P^n}{η_bΔt}
+α\frac{T-T^n}{Δt}
\right]dΩ.
```

In 2-D, velocity and pressure live on the two fields of `MixedMesh` (commonly
T7/P1-discontinuous). In the 3-D state path, velocity is continuous T11 or
Hex27/Q2 and pressure has four cell-local linear modes.

### Viscoelastic and plastic constitutive update

For 2-D plane strain, with

```math
\dot ε_{xy}=\tfrac12(v_{x,y}+v_{y,x}),
\qquad
\operatorname{tr}\dot ε=v_{x,x}+v_{y,y},
```

the deviatoric in-plane normal rates subtract one third of the in-plane trace;
the implied out-of-plane stress is `τzz=-(τxx+τyy)`. The Maxwell effective
viscosity and trial stress are

```math
η_\mathrm{ve}=\left(\frac1η+\frac1{GΔt}\right)^{-1},
```

```math
τ^\mathrm{trial}=2η_\mathrm{ve}
\left(\dot ε^\prime+\frac{τ^n}{2GΔt}\right).
```

The purely viscous limit is `G=∞`, so `ηve=η`. The original caller-owned
3-D path remains purely viscous and computes

```math
τ=2η\left[\operatorname{sym}(\nabla v)
-\tfrac13(\nabla\!\cdot v)I\right].
```

Optional 2-D Drucker-Prager plasticity evaluates

```math
τ_{II}=\sqrt{\tfrac12(τ_{xx}^2+τ_{yy}^2+τ_{zz}^2)+τ_{xy}^2},
```

```math
F=τ_{II}-C\cos φ-P\sin φ,
\qquad
Q=τ_{II}-P\sin ψ.
```

For `F>0`, the regularized multiplier and stress correction are

```math
λ_p=\frac{F}{η_\mathrm{ve}+η_\mathrm{reg}
+K_bΔt\,(\partial Q/\partial P)(\partial F/\partial P)},
```

```math
τ\leftarrow τ^\mathrm{trial}
-2η_\mathrm{ve}λ_p\frac{\partial Q}{\partial τ}.
```

Here `∂Q/∂P=-sinψ` and `∂F/∂P=-sinφ`, making their product non-negative for the
supported angles.

### Two-dimensional Powell-Hestenes/DYREL iteration

The 2-D saddle-point solver forms a lumped pressure scale `M_P`. During each
inner velocity solve it augments momentum with

```math
P_\mathrm{num}=γ_P M_P^{-1}R^p,
```

and evaluates `R^v(v,P+Pnum)`. Velocity uses the DR recurrence with a negative
field step because of the momentum-residual sign:

```math
r_v^{k+1}=D_v^{-1}R^v+βr_v^k,
\qquad
v^{k+1}=v^k-αr_v^{k+1}.
```

After the inner residual has dropped by the required factor, the outer
Arrow-Hurwicz/Powell-Hestenes update is

```math
P\leftarrow P+γ_P M_P^{-1}R^p.
```

The coupled entry point uses this same loop. At each inner velocity iteration
it first performs one standard thermal Chebyshev DR update, applies the thermal
Dirichlet values, and gathers `T[mesh.el2nP]` into the Stokes temperature at
`mesh.DoFsP`. The solve converges only when both the Stokes criterion and the
thermal state's relative tolerance are satisfied. Thermal and Stokes
previous-time fields remain fixed and caller-owned during the solve.

For linear viscous/viscoelastic rheology, the augmented velocity Jacobian is
constant and is frozen by default. Plastic tangents are state-dependent and are
normally rebuilt. `λmax` is either the row-sum Gershgorin bound or a power
estimate of the symmetrically Jacobi-scaled operator
`D_v^{-1/2}AD_v^{-1/2}`.

### Three-dimensional iteration

The caller-owned 3-D Hex27/Q2--P1 path uses a simpler diagonally preconditioned
fixed-point scheme rather than the Chebyshev recurrence:

```math
P\leftarrow P+γ_P M_P^{-1}R^p,
\qquad
v\leftarrow v-ωD_v^{-1}(R^v-f).
```

The constant pressure mode is mass-weighted to zero after each pressure update.
The four cell-local pressure residuals test `-div(v)` against `(1,ξ,η,ζ)`.

The `StokesDR`/`MixedMesh` 3-D path instead extends the Powell-Hestenes/DYREL
iteration to three velocity components and supports viscoelastic stress history,
Drucker-Prager plasticity, and coupled thermal relaxation. Its signed linear
pressure basis uses the positive Jacobi mass `∫Nᵢ²dΩ`; direct lumping by
`∫NᵢdΩ` is invalid because three modes integrate to zero.

### Discrete adjoints

Let the assembled forward residual be

```math
R(u,m)=0,
```

where `u` contains all velocity and pressure unknowns, `m` is a material or
design parameter, and `J(u,m)` is a scalar objective. The adjoint must transpose
the exact discrete residual—same mesh, spaces, quadrature, constitutive update,
boundary projection, and augmentation.

The implemented **2-D convention** is

```math
R_u^Tλ+J_u=0
\quad\Longleftrightarrow\quad
R_u^Tλ=-J_u.
```

Accordingly, `objective_vx` and `objective_vy` are the assembled components of
`J_u`, and the total derivative is

```math
\frac{dJ}{dm}=J_m+λ^T R_m.
```

`test/test_stokes_adjoint_api.jl` validates this sign with a central finite
difference. The 2-D sinking-block and pure-shear adjoint miniapps use the same
positive contraction for their element and phase sensitivities.

At the frozen 2-D forward state, the augmented linearization is stored per
element as

```math
A=\frac{\partial R^v_\mathrm{aug}}{\partial v},
\qquad
B=\frac{\partial R^v}{\partial P},
\qquad
C=\frac{\partial R^p}{\partial v}.
```

The adjoint residual applied in each iteration is

```math
R^{λ_v}=J_v+A^Tλ_v+C^Tλ_P,
\qquad
R^{λ_P}=J_P+B^Tλ_v,
```

with `J_P=0` in current drivers. The default frozen-operator path assembles
`A`, `B`, and `C` once; the memory-light fallback obtains the same transpose
products with Enzyme on every iteration. The velocity adjoint uses DYREL and
homogeneous primal Dirichlet conditions; the pressure adjoint uses

```math
λ_P\leftarrow λ_P+γ_PM_P^{-1}R^{λ_P}.
```

The forward state must be converged before freezing these blocks.

The linear viscous **3-D convention** differs: for `Au=b` and `J=c^Tu`, the
wrapper solves

```math
A^Tλ=c,
```

by reusing the symmetric forward operator. Its material derivative is

```math
\frac{dJ}{dm}=λ^T\left(b_m-A_mu\right)=-λ^TR_m.
```

`stokes_material_gradient_3d` implements this contraction for density and
viscosity and is checked against a sparse finite-difference oracle. Never move a
sensitivity formula between the 2-D and 3-D paths without also translating the
adjoint sign convention.

### Roles of automatic differentiation

- ForwardDiff differentiates element residuals to obtain Jacobian blocks,
  absolute row sums, and diagonal preconditioners.
- Enzyme supplies the memory-light 2-D transpose fallback and selected material
  contractions.
- Frozen operators are still derivatives of the same discrete residual; they
  cache blocks rather than changing the mathematics.
- Finite differences and sparse assembled systems remain independent validation
  oracles, not production solvers.

## Invariants

- Element residual functions are the mathematical source of truth. Atomic,
  colored, forward, adjoint, and diagnostic paths must agree with them rather
  than reimplementing slightly different physics.
- Material tuples encode phase count and numeric type. Preserve phase indexing,
  `Float32`/`Float64`, and finite incompressible limits.
- DR residuals and rates are constrained before spectral estimates and updates;
  solution values are repinned after updates.
- Invalid preconditioners, NaNs, explosive residuals, and exhausted iteration
  budgets must fail or return explicit non-convergence status according to the
  documented API. They must never masquerade as converged solutions.
- A discrete adjoint uses the exact discrete forward residual, spaces,
  quadrature, constitutive update, and sign convention. Gradient changes require
  a finite-difference or equivalent independent oracle.
- Frozen Jacobians/operators are valid only while the forward state and all
  operator-defining inputs remain fixed.
- Solver kernels remain backend-neutral and avoid host scalar access to device
  arrays. Synchronization is added only where data dependency or host
  inspection requires it.
- Convergence tolerances, norms, iteration counters, and returned statistics are
  part of observable behavior. Change them deliberately and document the
  semantics.

## Direction

Keep one shared solver mechanism where the mathematics truly matches, while
allowing the mixed 2-D and cell-pressure 3-D Stokes layouts to remain distinct.
Do not create a generalized solver framework merely to make their signatures
look alike.

Priorities:

1. Make high-level entry points own routine inference of geometry, backend, and
   boundary data; retain low-level methods for specialized and adjoint work.
2. Keep residual, Jacobian, transpose, and material-gradient agreement
   executable through tiny independent checks.
3. Improve convergence using representative heterogeneous cases and measured
   spectra, not only homogeneous toy problems.
4. Track allocations, element sweeps, synchronization, and memory per element
   before accepting performance complexity.
5. Prepare global reduction points and owned/ghost update boundaries before
   claiming distributed solver support.

## Acceptance checks

Every non-trivial solver change needs one focused regression and, when
applicable:

- an analytic zero/body-force/constant-field residual case;
- atomic and colored assembler agreement;
- element Jacobian versus finite differences or AD;
- adjoint/operator transpose identity and objective gradient finite difference;
- convergence and deliberate non-convergence behavior;
- `Float32` and `Float64` result/type preservation;
- 2-D and 3-D checks when shared mechanics change;
- CPU plus the affected accelerator backend when kernel behavior changes;
- warmed timing and allocation data for performance claims.

Do not loosen tolerances just to make a changed algorithm pass. Explain the
scale and conditioning that justify any tolerance revision.

## Update this guide when

- a solver, state field, residual, rheology, convergence rule, or public return
  value changes;
- an assembled/frozen operator becomes part of or leaves the production path;
- new benchmark evidence changes a performance decision;
- distributed reductions or halo semantics enter a solve;
- a solver limitation or required precondition changes.
