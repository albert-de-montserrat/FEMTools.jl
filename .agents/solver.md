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
- Coupled 2-D thermal--Stokes: `solve_coupled_dyrel!` advances one thermal DR
  step per inner Stokes velocity step and gathers the continuous thermal field
  onto the discontinuous pressure-temperature DoFs.
- 3-D Stokes: Hex27/Q2 velocity with cell-local four-mode pressure, caller-owned
  arrays, forward/adjoint wrappers, and material gradients.
- Repeated 2-D adjoint solves may reuse a caller-owned
  `StokesAdjointWorkspace`. Its common scratch serves every operator mode, while
  the nine Enzyme-only arrays are allocated only with `enzyme=true`; workspace
  dimensions and boundary counts are checked before use.
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

`precompute_geometry` stores `(J⁻¹, dΩ)` for every element and quadrature point.
`∇_ξN` is a property of the reference element, supplied by
`shape_function_gradients`, so storing it per element would cost `N·nDim` numbers
per point where `J⁻¹` costs `nDim²` — a factor of 8 for Hex27. `element_geometry`
pairs the two and forms `∇_xN = ∇_ξN·J⁻¹` on access, so quadrature loops still
read `∂N∂x, dΩ = geo_el[q]`. The output array's element type selects what is
kept, so a field whose gradients no consumer needs — the mixed-mesh pressure
geometry `geo_P` — stores `dΩ` alone and is indexed as `dΩ = geo_P_el[q]`.
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
T7/P1-discontinuous). In the 3-D path, velocity is continuous Hex27/Q2 and
pressure has four cell-local modes `(1, ξ, η, ζ)`.

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

The purely viscous limit is `G=∞`, so `ηve=η`. The 3-D production path is
currently purely viscous and computes

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

### Drucker-Prager tensile cap (experimental 2-D path)

`cap_geometry`, `cap_yield_function`, `cap_invariants`, `cap_return_map`, and
`cap_local_update` in `src/stokes/assemblers/rheology.jl` implement the
globally continuous tensile cap of Popov, Berlie and Kaus (2025), GMD 18,
7035-7058, following JustRelax's coupled return without GeoParams. They are
internal functions used by the experimental `DruckerPragerCap` 2-D momentum
path.

`cap_invariants` returns `(F,Aτ,Ap)` with `Aτ=(∂Q/∂τII)/2` and `Ap=-∂Q/∂p`,
selecting yield and potential branches independently. `cap_return_map` solves
`(s-s_trial+2ηve*λ*Aτ, p-p_trial-KΔt*λ*Ap, F-η_reg*λ)=0` by Newton with a
ForwardDiff 3×3 Jacobian and Armijo backtracking for every yielded state,
including the shear segment; there is no separate DP shortcut or scalar solver.
A converged solve ends with one full Newton step so dual-number derivatives
equal the implicit-function derivative. `cap_local_update` reconstructs the
radial tensor return, corrected pressure, and rates `γdot=λAτ`, `θdot=λAp`;
momentum, pressure output, and history rates all call it through
`_cap_ip_update`, with cohesion softened by the accepted IP `γ`. A failed local
solve returns NaN, which the DR solver's non-finite residual checks stop on.
`update_stokes_current_stress!` is read-only for plastic history;
`commit_stokes_plastic_history!` is its only writer, called once per converged
step before `τ_old` is refreshed.
Material parameters are still interpolated to the IP before evaluating one cap,
not blended per phase as JustRelax's ratio adapter does.

The cap closes the shear envelope on the tensile side with a circle of radius
`R_y` centred at `(p_y,0)`, tangent to `τII=kP+c` and meeting the pressure axis
at the tensile strength `pT≤0`, with `k=sinϕ` and `c=C cosϕ` as above. Tangency
fixes `p_y=(a pT+c)/(a-k)` with `a=√(1+k²)`, and requires `c+k·pT>0` for a
positive radius. The cap branch carries the factor `a` so that `‖∇F‖=a` on both
branches: Perzyna viscoplasticity reads the overstress `⟨F⟩` *away* from the
surface, so agreement only on the surface is not enough. Each branch pair
agrees exactly along its whole switching ray `P+k·τII=p_y`, which the tests use
as an exact identity rather than a tolerance.

Two consequences are already fixed and should not be rediscovered:

- A nonlinear local solve is needed on the cap. The coupled `(s,p,λ)` Newton
  solve uses bounded iterations and Armijo backtracking; do not restore the
  superseded scalar solve or legacy shear shortcut.
- **Every dilatant plasticity model needs a finite elastic bulk modulus.** Both
  of the paper's pressure schemes fail as `K→∞`. FEMTools uses `K=Inf` in the
  incompressible gauge that several miniapps and the adjoint tests rely on, so
  the cap and that gauge are mutually exclusive.

**Decided 2026-09-20: the cap uses the trial pressure scheme.** The global
pressure is the trial visco-elastic pressure, the local stress update applies
`p=p̄+K ε̇_vol^vp Δt`, and the viscoplastic volumetric term therefore *cancels*
from the global continuity residual, which keeps the residual the solver
assembles unchanged in form. The alternative — global pressure as the true
spherical Cauchy stress, with `ε̇_vol^vp` added to the continuity residual — is
the same model but converges less robustly in the source's own testing, so it is
not implemented. These are different models numerically, not two spellings of
one; do not mix them. Consequences to hold onto:

- Global and local pressure do not converge to each other under this scheme.
  The global value converges to the trial pressure and the local value is the
  true spherical stress. A diagnostic that compares them is measuring the
  dilation, not an error.
- The cap constitutive dispatch now returns the locally corrected pressure to
  the 2-D momentum residual. The global pressure field, pressure residual, and
  density equation of state still use trial pressure; this split is deliberate.
- The corrected pressure is now observable. Passing a four-matrix `τ_store`
  to `update_stokes_current_stress!` — `(τxx, τyy, τxy, P)`, each `nq × nels` —
  records the return map's pressure at integration points alongside stress. A
  three-matrix store keeps the previous stress-only behavior, so every existing
  caller is unchanged. Without this there was no way to read the physical
  pressure back out: `dr.Pnum` is the Arrow-Hurwicz update `γP·RP/M_P`, not a
  plastic correction, and using it as one is a mistake that reads plausible.
- Next-step pressure memory is the accepted *corrected* pressure at
  integration points, matching JustRelax's `P += ΔPψ` handoff.
  `solve_stokes_dyrel!` takes `P_old` (default `dr.P0`): a nodal vector is
  interpolated as before, an `nq × nels` matrix is read directly at the
  velocity quadrature points, which are also the continuity quadrature points
  (`IntegrationPointPressure`, `pressure_increment_at_ip`). Only the pressure
  residual (and hence `Pnum`) reads it; the augmented momentum Jacobian keeps
  `dr.P0` because `∂Rv/∂v` is independent of the old pressure. Seeding `P0`
  from trial `dr.P` instead makes the global field unbounded under sustained
  extension (it falls `K·Δt·∇·v` per step: trial `-0.768` vs corrected
  `-0.147` after 12 Popov steps with `pT = -0.1`). No P1 projection of the
  corrected pressure exists; add one only with a constant-preservation test.
  The trial/physical split lives inside a solve, not in accepted history.
- Consistent with that decision, the authors' GeoTech2D
  release carries the *corrected* pressure into the next step. In
  `CODE/src/update.py` the local update stores `svar['svp'] = pcor` and the
  next step forms its trial pressure as `pstar = pn - dp` from that stored
  value, so nothing accumulates an uncorrected trial field across steps. That
  code solves the two-field system with Newton rather than Powell-Hestenes and
  DYREL, while the inspected JustRelax DYREL path also commits corrected
  pressure. FEMTools stores and reuses the accepted corrected IP pressure.
- `DruckerPragerCap` accepts optional per-phase `C_min` and `H_C` tuples for
  bounded linear cohesion softening. Residual and augmented Jacobian assembly
  accept per-IP γ and apply the same local law.
- The 2-D pressure residual now accepts an optional per-phase `K`; the solver
  selects it by plastic-model dispatch only for the experimental
  `DruckerPragerCap` path. Existing
  non-cap Stokes and adjoint paths retain the historical bulk-viscosity form,
  including the `K = Inf` incompressible gauge. Direct low-level callers that
  omit `K` retain that behavior. The cap return map is dispatched by the
  experimental 2-D constitutive path. Per-IP history, stress, corrected
  pressure, and rates share the same softened local result.
- The forward 2-D solver now exposes `finite_K` explicitly. When `true`, the
  pressure storage coefficient is the finite positive material `K`, regardless
  of whether plasticity is enabled; when `false`, the legacy `ηb` coefficient
  remains in use. `Qq` without `Q2D` is copied as a signed local source, while
  nonnegative `Qq` weights with signed `Q2D` are normalized by the discrete
  pressure quadrature so `Σ Q dΩ = Q2D`. `P_old` may be nodal or accepted
  `nq × nels` integration-point pressure. The finite-`K` mean is not removed,
  and the specialized 3-D cell-local pressure path does not carry `Q`.
- The elastic bulk modulus must be finite, so the cap cannot be used in the
  `K=Inf` incompressible gauge. Any miniapp or adjoint test that wants the cap
  needs a finite `K` first.
- The cap-state 2-D adjoint path is covered by a ForwardDiff frozen-block versus
  Enzyme transpose regression (`test/test_adjoint_operator.jl`); this validates
  the local return-map derivative at a tensile-cap state, but not a full cap
  benchmark or history/softening evolution.
- `StokesDR(...; plastic_history_size=(nq, nels))` provides zeroed,
  caller-visible per-integration-point `γ` and `θ` arrays. Only
  `commit_stokes_plastic_history!` advances them, one physical-time increment
  per call through a backend kernel; residual iterations and
  `update_stokes_current_stress!` never do. The default remains `nothing` to
  preserve non-cap memory behavior.

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

**The inner loop scores progress relatively, and that needs an absolute floor.**
Its error is `max(err_v_inner / err_v00, err_v_inner)`, where `err_v00` is the
velocity residual at the *first check of the solve*. When the velocity already
satisfies the momentum balance at that first check, the ratio is ~1 and pins the
error there however small the true residual is, so the exit test
`err ≤ err_outer · rel_drop` can never be met and the solve spends its whole
inner budget. The absolute term alone now decides once `err_v_inner ≤ ϵ`.

Two things are easy to get wrong here:

- The outer loop uses `min(err_abs, err_rel)` — converged when *either*
  criterion holds. The inner `max` is the opposite, and needs the absolute
  escape to stay satisfiable. Keep the two consistent in intent if either is
  touched.
- It wastes work rather than corrupting the answer. A block pulled at a constant
  rate from its exact velocity solution reaches the right pressure either way,
  but spends its entire inner budget on the first outer pass: 501 iterations
  against 25. The Popov extension driver showed the same 501 per physical step.
  Any step that starts near its own solution is exposed — small time steps,
  restarts, steady continuation. The regression lives in
  `test/test_solver_convergence_api.jl` and asserts the iteration count, since
  the converged values alone do not distinguish the two.

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

The 3-D Hex27/Q2--P1 path uses a simpler diagonally preconditioned fixed-point
scheme rather than the 2-D Chebyshev recurrence:

```math
P\leftarrow P+γ_P M_P^{-1}R^p,
\qquad
v\leftarrow v-ωD_v^{-1}(R^v-f).
```

Prescribed 3-D velocities are applied before the first residual assembly and
repinned after each update. Omitting `bc_values` prescribes zero; component
lengths must match `fixed_nodes`. The constant pressure mode is mass-weighted
to zero after each pressure update.
The four cell-local pressure residuals test `-div(v)` against `(1,ξ,η,ζ)`.

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

3-D plasticity and the tensile cap are planned on the mixed path (Tet15/P1-disc
on `MixedMesh{3}`), not on the Hex27 cell-pressure path.

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
