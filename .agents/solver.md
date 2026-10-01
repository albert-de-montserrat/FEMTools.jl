# Solver

## Current reality

The package uses matrix-free finite-element residual assembly and
pseudo-transient dynamic relaxation (DR).

Stokes velocity work fields use vector containers. Convert them with `Tuple`,
and use `stress(dr)`/`stress_old(dr)` for assembler stress order; 3-D tensors
store shear components in Voigt order, while assemblers expect `(xy, xz, yz)`.

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
  Hex27-specific. Repeated 2-D adjoint solves may also reuse a caller-owned
  `StokesAdjointWorkspace`; its common scratch serves every operator mode, while
  the nine Enzyme-only arrays are allocated only with `enzyme=true`; workspace
  dimensions and boundary counts are checked before use.
- The frozen forward/adjoint operator paths intentionally assemble dense
  element blocks for repeated application. Sparse matrices in tests serve as
  reference oracles, not the production solve path.

Automatic differentiation is part of the numerical implementation:
ForwardDiff builds selected element Jacobians and Enzyme supports transpose and
material-gradient paths. The removed `ADJOINT_PERF_PLAN.md` contains historical
measurements in git (`git show b5fd5b5^:ADJOINT_PERF_PLAN.md`). Current Reykjanes
planning and the compressible-adjoint audit are in `REYKJANES_PLAN.md`; historical
timings do not establish correctness or cost for that model.

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
+Q\right]dΩ.
```

`StokesDR.Q` is a pressure-node array for a prescribed volumetric
source/sink. Positive values denote volume production and negative values
denote removal. It is initialized to zero and populated by the caller on the
selected backend. The 2-D pressure residual and adjoint treat it as an input
coefficient; the specialized 3-D cell-local pressure residual does not yet
carry `Q`.

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
- The 2-D adjoint pressure residual is `ResλP = (∂Rv/∂P)ᵀλv` alone in every
  operator path (`:blocks`, `:matrix_free`, `:enzyme`); the storage self-coupling
  `(∂RP/∂P)ᵀλP = -M/(ηₚΔt)·λP` is omitted. The adjoint is therefore exact only in
  the `ηₚ = Inf` incompressible gauge, and the choice of `ηb` versus finite `K`
  never reaches it. Miniapps that run the adjoint with `ηb = K` rely on this
  approximation.
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

#### Measured convergence in the elastic-dominated regime

`examples/reykjanes/elliptical_cavity.jl` solves one step of a soft compressible inclusion
injected through `Q` in a Maxwell host (`G = 3e10`, `K = 5e10`, `η = 1e20`, Maxwell time 106 yr),
against Muskhelishvili's closed form for the pressurised elliptical hole. The solution is scaled
to order one. Values below are inner iterations to `ϵ_tol = 1e-6` on Julia 1.13 with 16 threads,
`rel_drop0 = 1e-2`, `ncheck = 100`, CFL 0.99, `c_fact = 0.9`, and change one setting at a time from
2 563 elements, `Δt` = 1 day and an inclusion shear modulus of `1e-3` of the host's.

| Varied setting | Iterations |
|---|---|
| Elements | 376: 4 600; 1 527: 5 500; 2 563: 8 200; 4 209: 9 900; 7 330: 8 700; 13 316: 11 100 |
| Inclusion / host shear modulus | 0.1: 5 300; 1e-2: 7 200; 1e-3: 8 200; 1e-4: 9 200; 1e-5: about 21 000 |
| `Δt` | 1 h to 1 day: 8 200; 30 days and 1 yr: 8 000; 10 yr: 9 600; 100 yr: 24 100; 1 kyr: 47 800 |
| `ϵ_tol` | 1e-2: 1 300; 1e-3: 2 800; 1e-4: 4 600; 1e-5: 6 400; 1e-6: 8 200; 1e-7: 9 700; 1e-8: 11 500 |
| `rel_drop0` | 0.1: 5 800; 0.01: 8 200; 0.001: 13 400 |
| `ncheck` | 50: 7 200; 100: 8 200; 200: 9 400; 400: 14 000 |

- **Accuracy.** Cavity pressure, area change and opening agree with the closed form to 0.3 %
  at 2 563 elements. The remainder is the physical stiffness of the inclusion: the pressure
  excess falls with its shear modulus (`P/P_ref − 1` is 0.20, 0.025, 2.7e-3, 4.5e-4, 2e-4 for
  ratios 0.1 to 1e-5) until discretisation takes over. At 13 316 elements and ratio 1e-5 the
  pressure, area change and opening are within 0.02 % and the displacement error is 6e-4. The
  answer does not depend on `Δt` when the reference uses `G/(1 + Δt/t_M)`.
- **Cost.** It is flat while `Δt ≪ t_M`, grows once `G/(1 + Δt/t_M)` falls (the material turns
  nearly incompressible), and grows mildly with the shear-modulus contrast up to `1e4` and more
  than twofold at `1e5`. Each
  decade of `ϵ_tol` costs about 1 750 iterations, an outer Powell-Hestenes contraction of about
  0.62 per step. `ncheck` sets the granularity of every count, since the stopping test runs only
  at multiples of it.
- **Tolerance.** The pressure error from stopping early is 8e-4 at `ϵ_tol = 1e-2`, 1.5e-4 at
  1e-3 and about 1e-5 from 1e-4.
- **Scale.** The stopping test is `min(err_abs, err_rel) < ϵ_tol` with an absolute branch, so the
  same problem stops after very different residual reductions in different units. The relative
  reduction reached was 6e-6 at `σ_c = 1e6`, 6e-5 at 1e7, 7e-4 at 1e8, 6e-3 at 1e9, 7e-2 at 1e10
  and 0.58 at 1e11 (pressure `p/σ_c` from 10 to 1e-4). The pressure moved by 1e-3 here only because
  the far-field seed starts close; scale so the increment is of order one.
- **Reproducibility.** Five identical runs gave identical iteration counts and a pressure spread of
  7e-13; the atomic momentum assembly is not a noise source at this size. At a contrast of `1e5`
  two identical runs differed by 200 iterations.
- **Pressure scaling.** `assemble_viscosity_weighted_pressure_scaling!` uses the viscous `η`, not
  `ηve`, so `γ_eff` equals `K Δt` whenever `γfact η ≫ K Δt`, which holds throughout this regime.
- **`converged` after the iteration cap (fixed).** `solve_stokes_dyrel!` used to return
  `converged = err < ϵ`, but its inner loop overwrites `err` with the inner velocity residual, so a
  run that ended on `total_iterMax` inside the inner loop could report `converged = true` while the
  outer error was above `ϵ` (three of the twelve runs of the regression sweep on a 3 × 3 buoyancy
  case did, and one 13 316-element cavity run). `converged` is now set where the outer test passes, so a capped run reports `false`;
  `err` is still the inner residual in that case, so read `err_abs` and `err_rel`. The counts above
  are unaffected. `test/test_solver_convergence_api.jl` sweeps the cap to guard it.
- **Reproduce.** `examples/benchmarks/stokes/elliptical_cavity/elliptical_cavity.jl`
  (`run_cavity_benchmark()`) runs every sweep above.

#### Eigenstrain dike against the crack solution

`examples/reykjanes/dike_crack.jl` opens a flat elliptical band of *ordinary host material* by the
eigenstrain of the Reykjanes plan — the stress history of its integration points shifted by
`-2G Δε*` and the same volumetric increment in the continuity source — and compares one elastic step
with the same closed form at `b/a ≪ 1`, where it is Sneddon's crack. Nothing is soft and no hole is
meshed. A band of semi-axes `(a, b)` is `2b √(1 − (x/a)²)` thick, the shape of the opening of a
uniformly pressurised crack, so one uniform eigenstrain `f = w/(2b)` opens it into that profile.
Host as above, `a = 2.5 km`, `b = 125 m`, `Δt` = 1 day, `ϵ_tol = 1e-6`, one fixed-point pass on the
band's elastic storage; `profile` is the largest relative error of the opening on the upper face
inside `|x| ≤ 0.8a`, and `σn` is the band's closure traction `P − τ_yy` over the crack pressure.

| Varied setting | profile | `σn` | Iterations |
|---|---|---|---|
| Elements | 686: 1.6e-2, 1.027; 1 164: 4.4e-3, 1.005; 1 651: 1.6e-3, 1.002; 1 917: 1.1e-3, 1.001; 6 034: 6.4e-4, 1.0000 | | 3 300 to 5 700 |
| Band `b/a` | 0.10: 3.4e-3, 0.999; 0.05: 1.1e-3, 1.001; 0.025: 7.8e-4, 1.002; 0.0125: 8.6e-4, 1.002 | | 2 800 to 5 300 |
| Storage passes | 0: 9.1e-3, 1.009; 1: 1.1e-3, 1.001; 2: 8.9e-4, 1.001 | | 5 300 per pass |
| `ϵ_tol` | fixed from 1e-3 (1.0e-3, 1.001) to 1e-7 (1.1e-3, 1.001) | | 2 100; 3 100; 4 100; 5 300; 6 300 |
| Outer radius | 4a to 20a: profile 1.1e-3 to 1.2e-3, `σn` 1.001, identical to four digits | | 4 600 to 5 300 |

- **The mechanism is validated.** The opening profile converges to the closed form (1.6 % to 0.06 %
  under refinement) and the closure traction to the crack pressure, at every band aspect from 0.1 to
  0.0125 and for `Δt` from an hour to a century. The numerical band width is therefore a free
  parameter of the method, as the plan requires, once the injected volume is normalised with the
  pressure quadrature.
- **The band's mean pressure is not the dike pressure.** `P/p` is near 0.8 and moves with the band
  aspect (0.76 at `b/a = 0.1`, 0.82 at 0.0125), because the band is also held in compression along
  its length. Only `P − τ_yy`, the traction that would close it, is the crack pressure. A protocol
  that reads a dike or reservoir pressure off `P` in the band is wrong by about 20 %.
- **The band stores part of the injection.** It has the host's `ηb = K`, so `P/K` of the eigenstrain
  never reaches the opening. The setup removes this with one fixed-point pass on the solved band
  pressure; without it the error is 0.9 %, still inside the plan's 2 % gate. The band's continuity
  identity `ΔA/A + P/K` over the injected eigenstrain closes to 1e-6 in every run.
- **Truncation.** With the analytic displacement on the outer circle the answer does not depend on
  the disc radius from 4a to 20a, as in the cavity case.
- **Cost.** About 1 050 iterations per decade of `ϵ_tol`, and the measured quantities are fixed from
  `ϵ_tol = 1e-3`, so the tolerance here buys residual, not accuracy.
- **Reproduce.** `examples/benchmarks/stokes/dike_crack/dike_crack.jl`
  (`run_dike_crack_benchmark()`) runs every sweep above.

#### First failure threshold under mesh refinement (Gate G1: 77.7% on the mesh graph, 45.6% with the corridor detector, 11.1% once the reservoir geometry is resolved, 1.9% from an equilibrated baseline — and that last one is not a pass)

`examples/benchmarks/stokes/first_threshold/first_threshold.jl` runs the event protocol of
`REYKJANES_PLAN.md` to its first event and reports `ΔP_crit`, the rise of the reservoir pressure over
its spun-up baseline. Gate G1 asks for under 5% under refinement. Measured with a recharge of
2e-7 m² s⁻¹, 1 kyr steps, one spin-up step and the default protocol, whose detector is now
`:corridor_column`:

| `max_area`, sill refinement | Elements | Baseline | `P` at event | `ΔP_crit` | Event | Corridor | Band | `s_n` | Opening | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.6e7, 3 | 228 | 121.76 MPa | 126.04 MPa | 4.283 MPa | 3.359 kyr | 0.79, 4/4 bins | 8 els | 74.37 MPa | 0.108 m | 56 s |
| 8.0e6, 3 | 284 | 121.76 MPa | 126.04 MPa | 4.282 MPa | 3.359 kyr | 0.79, 4/4 bins | 8 els | 74.37 MPa | 0.108 m | 57 s |
| 4.0e6, 4 | 492 | 120.21 MPa | 126.59 MPa | 6.385 MPa | 3.219 kyr | 0.79, 4/4 bins | 16 els | 78.71 MPa | 0.166 m | 90 s |
| 2.0e6, 4 | 928 | 119.88 MPa | 126.64 MPa | 6.758 MPa | 2.781 kyr | 0.82, 4/4 bins | 30 els | 78.45 MPa | 0.168 m | 186 s |

Every bracket is ±0.016 kyr and every search resolved, as before.

- **The gate still fails, at 45.6%**, down from the 77.7% the adjacency detector gave. Nothing
  downstream of the threshold can be read yet.
- **Read this table with the two that follow.** It varies element size and sill refinement together,
  and the `sill` sweep below shows the reservoir geometry carries the jump: restricted to meshes
  whose sill is resolved, the same quantity spreads 11.1%, not 45.6%. The 45.6% is what this
  sweep's design measures, not the model's convergence.
- **The detector is no longer the term that dominates.** The two meshes at sill refinement 3 now
  agree to **0.015%** (4.283 against 4.282 MPa), where the graph rule gave 0.8%, and they find the
  same corridor state and the same band. At 2 684 elements the corridor and graph rules agree to
  **0.3%** (6.036 against 6.055 MPa): the two detectors converge to the same threshold, and their
  disagreement on coarse meshes was a resolution effect, not a difference of rule.
- **What is left is the baseline, and it is an amplification problem.** The spread tracks the *sill*
  refinement, 3 to 4, and both of the pressures it is built from are far better converged than their
  difference: `P` at the event moves 0.48% across the four meshes (126.04 to 126.64 MPa) and the
  spun-up baseline moves 1.56% (121.76 to 119.88 MPa), while `ΔP_crit`, their difference, moves
  45.6%. The amplification is `P/ΔP ≈ 25` to `30`, so **a 5% gate on `ΔP_crit` is a demand for about
  0.2% convergence of two ~126 MPa pressures**. Read that way the model is not badly converged; the
  observable is badly conditioned.
- **Therefore the next lever is the spun-up state, not the detector.** This sweep runs
  `spinup_steps = 1`; the `solver` sweep varies it over 1, 2 and 4 precisely because it changes the
  state the threshold is measured from. Run that, and the `time_step` sweep, before touching the
  detector again. If the baseline cannot be converged to the fraction of a percent the gate implies,
  the gate belongs on a better-conditioned observable — an absolute event pressure, or a threshold
  measured from a fixed reference rather than from a spun-up one — and that is a change to the
  science plan, not to the code.
- **The rules answer different questions, by design.** In the `detector` sweep at 2 684 elements,
  `:corridor_fraction` trips at 0.531 kyr with 2 of 4 bins filled (`ΔP_crit` 2.878 MPa) because it
  never asks the failure to reach the shallow end. The 63.7% spread of that sweep is protocol
  sensitivity to be quoted, *not* a convergence failure, and must not be added to the G1 budget.
- **The band differs with the rule even where the threshold does not.** At 2 684 elements the graph
  rule opens a 10-element chain with `s_n` 60.08 MPa and needs 0.307 m, while the corridor rule opens
  a 74-element band with `s_n` 76.75 MPa and needs 0.200 m. The thresholds agree to 0.3%; what gets
  intruded does not, so M2 cannot treat the two rules as interchangeable downstream of the event.
- **Cost.** 56 to 186 s per run on these meshes, and 448 to 546 s at the 2 684-element default,
  each being a spin-up step, two to four recharge steps, a crossing search and an amplitude search.
- **Reproduce.** `run_first_threshold_benchmark(; sweeps = (:mesh,))`, `(; sweeps = (:detector,))`
  and `(; sweeps = (:baseline,))`, the last crossing the spin-up with the sill refinement the spread
  belongs to. Its printed spread line mixes both axes and means nothing on its own; read the rows.

*Historical, the same sweep under the adjacency detector (`:graph_path`), 2026-09-19.* Kept because
it is the measurement the 77.7% figure and the decision to replace the detector came from, and
because the rule is still selectable.

| `max_area`, sill refinement | Elements | `ΔP_crit` | Event | Bracket | Path | `s_n` | Opening |
|---|---|---|---|---|---|---|---|
| 1.6e7, 3 | 228 | 2.899 MPa | 2.016 kyr | ±0.016 kyr | 5 els, 1 544 m | 71.09 MPa | 0.155 m |
| 8.0e6, 3 | 284 | 2.877 MPa | 2.000 kyr | ±0.016 kyr | 5 els, 1 544 m | 71.09 MPa | 0.154 m |
| 4.0e6, 4 | 492 | 5.200 MPa | 2.141 kyr | ±0.016 kyr | 6 els, 1 839 m | 71.62 MPa | 0.128 m |
| 2.0e6, 4 | 928 | 6.218 MPa | 2.203 kyr | ±0.016 kyr | 8 els, 1 105 m | 70.61 MPa | 0.419 m |

The threshold was the moment a face-connected chain of failed elements first spanned from the
reservoir to the target depth. That is a property of the mesh graph: refining changed which elements
existed, so it changed when a chain closed even where the stress field had converged, and the path
moved non-monotonically, 5 to 6 to 8 elements and 1 544 to 1 839 to 1 105 m. The coarse row of this
table is reproduced exactly by the current code under `detector = :graph_path` (2.899 MPa at
2.016 kyr, 5 elements, 1 544 m), which is how the detector change was isolated from everything else.


#### The reservoir geometry is the dominant error (`sill` sweep)

`run_first_threshold_benchmark(; sweeps = (:sill,))` moves the sill refinement alone, at fixed
`max_area = 8.0e6`, which the `mesh` family cannot do because it moves both axes together.

| Sill refinement | Elements | Baseline | `P` at event | `ΔP_crit` | Event |
|---|---|---|---|---|---|
| 3 | 284 | 121.7600 MPa | 126.0424 MPa | 4.2825 MPa | 3.359 kyr |
| 4 | 284 | 121.7600 MPa | 126.0424 MPa | 4.2825 MPa | 3.359 kyr |
| 6 | 374 | 119.9564 MPa | 126.5441 MPa | 6.5877 MPa | 3.031 kyr |
| 8 | 502 | 119.8745 MPa | 126.6351 MPa | 6.7607 MPa | 2.781 kyr |

- **The event pressure converges in the reservoir geometry.** Successive differences are 0.0000,
  0.5017 and 0.0910 MPa, so the error left at refinement 8 is of order 0.02 to 0.09 MPa. The
  ±0.5 MPa read off the `mesh` family was not a discretisation floor; it was refinement 3 failing to
  resolve the polygonal ellipse the reservoir pressure is read on.
- **Refinement 3 and 4 are the same run here**, to every digit and the same 284 elements: the
  parameter saturates against `max_area`, so a refinement number alone does not say how well the
  reservoir is resolved. Any row of a mesh study has to report both, and a convergence claim needs
  the refinement varied at fixed element size, as here, or the element size varied at a refinement
  known to be converged.
- **Use sill refinement 6 or more.** Below it the reservoir geometry, not the physics, sets the
  threshold. The default model (`max_area = 1e6`, `refinement = 8`) is already in that regime.
- **What is left after the geometry.** Pooling every run whose sill is resolved — (8.0e6, 6),
  (8.0e6, 8), (4.0e6, 4), (2.0e6, 4) and the 2 684-element default — the event pressure spans
  126.544 to 126.900 MPa (0.356 MPa, 0.28%) and the baseline 119.875 to 120.864 MPa (0.990 MPa,
  0.82%), so `ΔP_crit` spans 6.036 to 6.761 MPa, an 11.1% spread. Down from 45.6%, still failing the
  5% gate, and the residual is again the baseline rather than the event: the finest mesh has both the
  highest baseline and the highest event pressure, which is the signature of a transient that has not
  equilibrated rather than of a converging discretisation.
- **Element refinement at a resolved sill does not close it either.** At fixed `refinement = 8`,
  varying `max_area` over 8.0e6, 4.0e6, 2.0e6 and 1.0e6:

| Elements | Baseline | `P` at event | `ΔP_crit` | Event | Cost |
|---|---|---|---|---|---|
| 502 | 119.874 MPa | 126.635 MPa | 6.761 MPa | 2.781 kyr | 118 s |
| 847 | 120.244 MPa | 127.081 MPa | 6.837 MPa | 3.062 kyr | 142 s |
| 1 460 | 120.612 MPa | 126.890 MPa | 6.278 MPa | 2.766 kyr | 336 s |
| 2 684 | 120.862 MPa | 126.898 MPa | 6.036 MPa | 2.891 kyr | 560 s |

  `ΔP_crit` spreads 12.4%, the event pressure 0.35% (0.446 MPa) and the baseline 0.82%
  (0.987 MPa). The shapes differ and that is the diagnosis: the event pressure *wobbles* by about
  0.2 MPa with no trend, while the baseline rises *monotonically* with every refinement. A
  one-sided drift under refinement is what an unequilibrated transient looks like, not a converging
  discretisation — and the spin-up section below shows it is exactly that.
- **Reproduce.** `run_first_threshold_benchmark(; sweeps = (:sill,))`.


#### The baseline's mesh dependence is a transient (`spinup_history`)

`run_cycles` now records the reservoir pressure after every spin-up step and returns it as
`spinup_history`. Tectonic loading alone, no recharge, twelve steps of 1 kyr, at two resolutions:

| Spin-up step | 228 elements | step change | 502 elements | step change |
|---|---|---|---|---|
| 1 | 121.7590 MPa | | 119.8745 MPa | |
| 2 | 121.9431 MPa | +0.184 | 122.2516 MPa | +2.377 |
| 4 | 124.0598 MPa | +1.012 | 125.1422 MPa | +0.975 |
| 8 | 126.0289 MPa | +0.272 | 126.1969 MPa | +0.127 |
| 12 | 126.5467 MPa | +0.079 | 126.5123 MPa | +0.061 |

- **The two meshes start 1.885 MPa apart and agree to 0.034 MPa by step 12**, with step changes
  shrinking geometrically at both resolutions. The baseline's mesh dependence, which is what Gate G1
  has been measuring, is the *initial-condition error of the warm start*, not a property of the
  discretisation: the two meshes begin with different errors and relax toward the same state.
- **0.034 MPa is the accuracy the science needs.** It is the 0.04 MPa a 1% memory signal requires,
  and it is 30 times better than the 0.99 MPa baseline range that a one-step spin-up leaves.
  `spinup_steps` should be set by a tolerance on the step change, not by a count, and every run
  record should carry the last step change so a reader can see the state was equilibrated.
- **Unfinished at step 12.** The trajectory is still rising by about 0.06 MPa per step with a ratio
  near 0.86, so the asymptote is near 126.9 MPa with an extrapolation uncertainty of a few tenths.
  That range overlaps the measured event pressures (126.635 to 127.081 MPa), which raised a question
  the numbers here could not answer: whether a section spun up to equilibrium is already at its own
  failure criterion, in which case `ΔP_crit` measured from a one-step spin-up is largely the distance
  the transient still had to travel rather than a physical threshold. It was run to 40 steps, and the
  answer is yes — see the next section.
- **Reproduce.** `run_cycles(; spinup_steps = 12, max_steps = 0, verbose = true)` prints the
  trajectory; the returned `spinup_history` carries it.

#### The spun-up baseline, and what is actually converged (`baseline` sweep)

`run_first_threshold_benchmark(; sweeps = (:baseline,))` crosses `spinup_steps` with the sill
refinement the G1 spread belongs to, at the same recharge and step as the mesh sweep. Pressures in
MPa:

| Sill refinement | Elements | Spin-up | Baseline | `P` at event | `ΔP_crit` | Event | Cost |
|---|---|---|---|---|---|---|---|
| 3 | 228 | 1 | 121.759 | 126.0421 | 4.283 | 3.359 kyr | 54 s |
| 3 | 228 | 2 | 121.943 | 126.0336 | 4.091 | 2.375 kyr | 47 s |
| 3 | 228 | 4 | 124.060 | 125.9356 | 1.876 | 0.766 kyr | 54 s |
| 4 | 492 | 1 | 120.206 | 126.5907 | 6.385 | 3.219 kyr | 90 s |
| 4 | 492 | 2 | 121.704 | 126.5913 | 4.888 | 2.250 kyr | 91 s |
| 4 | 492 | 4 | 124.733 | 126.3752 | 1.642 | 0.594 kyr | 114 s |

- **The baseline is not an equilibrated state.** It rises monotonically with the number of spin-up
  steps, by 2.3 MPa at refinement 3 and 4.5 MPa at refinement 4 over 1 to 4 steps, so `ΔP_crit`
  measured from it collapses from 4.283 to 1.876 MPa and from 6.385 to 1.642 MPa. The spread across
  spin-up at fixed mesh (−56%, −74%) is *larger* than the spread across mesh at fixed spin-up. A
  threshold measured from a moving reference is not a property of the model, and `spinup_steps = 1`
  is not a converged state to measure one from.
- **The absolute event pressure is invariant, and that is the physical statement.** Changing the
  spin-up from 1 to 4 moves `P` at the event by 0.107 MPa at refinement 3 and 0.215 MPa at
  refinement 4, that is 0.08% and 0.17%. Failure is a stress criterion, so the reservoir pressure at
  which the host fails is a property of the model and not of how long it was loaded beforehand. The
  whole set of six runs spans 125.94 to 126.59 MPa, 0.52%.
- **The model's accuracy, stated honestly, is absolute: ±0.5 MPa across the sill refinement and
  ±0.1 MPa across the spin-up.** At fixed spin-up the two refinements differ by 0.549, 0.558 and
  0.440 MPa. Cross-mesh baseline agreement improves with spin-up (1.29%, 0.20%, 0.54%) and
  cross-mesh `ΔP_crit` improves with it too (49.1%, 19.5%, 14.3%), but never approaches 5%.
- **A lithostatic reference does not rescue the gate.** With `ρ0[1] = 2900` and the sill centre at
  4.5 km, lithostatic there is 128.02 MPa, *above* every measured event pressure: the section is
  under-pressured relative to lithostatic because it is extending. `P_event − P_lith` is about
  −1.5 MPa, so that difference is worse conditioned than the spun-up one, not better.
- **What the absolute error implies for the science.** With thresholds of about 4 MPa, a 0.5 MPa
  uncertainty on the event pressure is roughly ±12% on `M_n`. Reading a 1% memory signal would need
  about 0.04 MPa, an order of magnitude better than the present discretisation gives. Either the
  smallest claimed signal is of order 10%, or convergence improves by that order; that is a science
  decision, and it is the reason the gate's wording matters rather than its threshold value.
- **Where the error lives.** Within sill refinement 4, going from 492 to 928 elements moves the
  event pressure by 0.05 MPa; going from refinement 3 to 4 moves it by 0.55. The reservoir is a
  polygonal ellipse and its pressure is read on its own elements, so this is a geometry error rather
  than a discretisation error. The `mesh` family varies element size and sill refinement together
  and cannot separate them, which is why the `sill` family exists: refinement 3, 4, 6, 8 at fixed
  `max_area`.
- **Reproduce.** `run_first_threshold_benchmark(; sweeps = (:baseline,))`. Its printed spread line
  mixes both axes and is meaningless; read the rows.

#### An equilibrated baseline passes Gate G1, and it is not a pass

The trajectory above was carried to 40 spin-up steps of 1 kyr at two resolutions, and the protocol
was then run from there. Pressures in MPa:

| Spin-up step | 228 elements | step change | 502 elements | step change |
|---|---|---|---|---|
| 1 | 121.7590 | | 119.8745 | |
| 8 | 126.0289 | +0.272 | 126.1969 | +0.127 |
| 16 | 126.7322 | +0.034 | 126.6910 | +0.037 |
| 24 | 126.9033 | +0.016 | 126.8902 | +0.018 |
| 32 | 127.0047 | +0.011 | 126.9999 | +0.011 |
| 40 | 127.0758 | +0.008 | 127.0705 | +0.007 |

| Elements | Baseline | `P` at event | `ΔP_crit` | Event | Band | Cost |
|---|---|---|---|---|---|---|
| 228 | 127.0758 | 127.1762 | 0.1004 | 0.016 kyr | 8 els | 127 s |
| 502 | 127.0705 | 127.1690 | 0.0985 | 0.016 kyr | 30 els | 274 s |

- **Every quantity converges.** The two baselines agree to 0.0053 MPa (0.004%), the event pressures
  to 0.0072 MPa, and `ΔP_crit` spreads 1.9% — against a gate of 5%, and against 45.6% from a
  one-step spin-up. Both runs bracket their crossing and their amplitude. Read as a convergence
  study, this is the first configuration that meets G1 as written, and it confirms the diagnosis: the
  mesh dependence of the threshold was the mesh dependence of the *warm start*, and refining the time
  axis of the spin-up removes it where refining the element size did not.
- **It must not be reported as a pass.** Both runs fire at the *first* recharge step, 0.016 kyr, with
  `ΔP_crit ≈ 0.10 MPa` — which is the resolution floor of the crossing search itself, not a measured
  threshold. The section this model relaxes to is already at its own failure criterion, so there is
  nothing left to load and the "threshold" is a measurement of the first time step. Agreement between
  two runs that both trip immediately is agreement about `Δt`, not about the mechanics.
- **The lever is the configuration, not the discretisation.** `T₀`, the corridor's reach depth and
  the extension rate together decide whether the equilibrated section is marginally stable or has a
  real margin to load through. Nothing in the solver changes that. Gate G1 needs both at once: an
  equilibrated baseline *and* a configuration whose first event is many steps away from it.
- **What to quote in the meantime.** The absolute event pressure with its element-refinement error,
  0.45 MPa over 502 to 2 684 elements at fixed sill refinement 8, and `ΔP_crit` only alongside the
  spin-up state its reference came from.
- **Reproduce.** `run_cycles(; spinup_steps = 40, max_steps = 30, verbose = true)` at
  `max_area = 1.6e7, refinement = 3` and `max_area = 8.0e6, refinement = 4`. It costs about 7 minutes
  for the pair.

### Three-dimensional iteration

The caller-owned 3-D Hex27/Q2--P1 path uses a simpler diagonally preconditioned
fixed-point scheme rather than the Chebyshev recurrence:

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

**Coverage limitation:** the displayed residual is what the current code
applies, not the full finite-compressibility transpose. For finite `ηb`, the
physical pressure residual has `D_P = ∂R^p/∂P = −∫N_P N_Pᵀ/(ηb Δt)dΩ`.
`FrozenAdjointOperator` stores no `D_P` block, and the Enzyme solver path discards
the pressure pullback in `dP_scratch` after copying the momentum pullback to
`ResλP`. The end-to-end test in `test_stokes_adjoint_api.jl` uses
`ηb = K = G = Inf`; passing it does not verify compressible VEP gradients.
`REYKJANES_PLAN.md` GAP-25 requires an independent full-system transpose/gradient
oracle and a consistent augmentation/multiplier mapping before such gradients
are used. Operator-mode agreement alone can preserve the same omission. Also
verify symmetry assumptions with pressure-dependent EOS before reusing a
symmetric forward operator as its adjoint.

For Reykjanes failure diagnostics, `τ−PI` is tension-positive stress;
compression-positive closure uses `PI−τ` and includes the out-of-plane stress in
plane strain. The science yield coefficients require an explicit mapping to
the code's `C cosϕ + P sinϕ`. Nonzero plastic dilation is not yet validated
against a corresponding plastic-volume term in the continuity residual; use
zero dilation for the initial study. These are planning constraints, not new
implemented constitutive capabilities.

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
