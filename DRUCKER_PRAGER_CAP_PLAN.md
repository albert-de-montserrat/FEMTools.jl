# `DruckerPragerCap`: a globally continuous tensile cap for FEMTools.jl

Status: implementation plan, not implemented · 2026-09-20 · branch `adm/volcanos`

Source: Popov, Berlie and Kaus (2025), *A dilatant visco-elasto-viscoplasticity model with globally
continuous tensile cap: stable two-field mixed formulation*, Geosci. Model Dev. 18, 7035–7058,
[doi:10.5194/gmd-18-7035-2025](https://doi.org/10.5194/gmd-18-7035-2025), local copy `popov2025.pdf`.
Reference implementation: [GeoTech2D](https://github.com/UniMainzGeo/GeoTech2D) (MIT). The script
that draws the yield surface and flow potential maps of their Fig. 2 is archived at
[doi:10.5281/zenodo.16877978](https://doi.org/10.5281/zenodo.16877978); the model setups are at
[doi:10.5281/zenodo.15496843](https://doi.org/10.5281/zenodo.15496843).

**Equation status: resolved 2026-09-20 against `popov2025.pdf` itself.** The earlier text
extraction dropped scaled `\left(...\right)` delimiters — those come from a math-extension font that
`pdftotext` cannot map — which is why Eqs. 15 and 18 read as if a factor sat outside the bracket it
actually multiplies. Both were re-read from the paper and, more usefully, each was pinned
independently by the geometric condition the paper states in words, so the reading no longer rests on
the extraction at all. The Zenodo map script is therefore no longer a prerequisite for CAP-3; V3
remains worth doing as a picture, not as the thing that decides the formulae. Details in §1.

Why this matters here: the Reykjanes plan's GAP-5 option 2 is "a constitutive tensile cutoff", and
its detector (GAP-6) currently decides mode-I failure with a *diagnostic* hydraulic margin
`P_f − s₃ − T₀ ≥ 0` that the constitutive law knows nothing about. This model is GAP-5 option 2. It
also produces the accumulated volumetric viscoplastic strain `θ`, which is the field that actually
marks tensile failure zones, and which the event detector would rather use than a stress-space
proxy. Implementing it lets the criterion that declares an event and the law that produces the
failure be the same statement.

## 1. The model

Four independent material parameters: friction angle `φ`, dilation angle `ψ`, Mohr–Coulomb cohesion
`c_MC`, tensile strength `p_T`. Everything else is derived. Pressure is compression-positive, which
is the convention `deviatoric_stress` already uses (`F = τII − C cosφ − P sinφ`), so `p_T` is the
pressure at which the rock fails in pure tension and is negative in that convention — a sign the
constructor must pin with a test, because "tensile strength 5 MPa" in the event protocol is a
positive number (`EventProtocol.tensile_strength`).

**Derived coefficients** (their Eq. 13, 14):

```
k = sin φ        k_q = sin ψ        c = c_MC cos φ
a = √(1 + k²)    b = √(1 + k_q²)
```

**Cap geometry** (their Eq. 15) — a circle in the meridional `(p, τII)` plane, centred on the
pressure axis at `p_y` with radius `R_y`, tangent to the Drucker–Prager line `τII = k p + c`:

```
R_y = (k p_y + c) / a                 tangency: distance from (p_y, 0) to the line
R_y = p_y − p_T                       the cap meets the pressure axis at p_T
⟹  p_y = (a p_T + c) / (a − k) = p_T + (c + k p_T) / (a − k)      ✅ confirmed
```

**Resolved.** The paper writes Eq. 15 as `p_y = (p_T + c/a)(1 − k/a)^{−1}`, which is the same number.
The extraction had lost the scaled parentheses, making it look like `p_T + (c/a)(1 − k/a)^{−1}`; that
reading gives `p_T + c/(a−k)`, which fails tangency outright — with `φ=30°, c_MC=10, p_T=−5` it puts
the centre at 9.01 with radius 14.01 while the tangent distance is 11.78. The form above satisfies
both conditions simultaneously and is what `cap_geometry` implements.

Both expressions for `R_y` must agree to round-off; that is the unit test for this block, and it is
now `test/test_drucker_prager_cap.jl`.

**Delimiter point** (their Eq. 16), the tangency point where the two segments meet:

```
p_d = p_y − R_y k / a            τ_d = k p_d + c = R_y / a
```

**Flow-potential centre** (their Eq. 17), placed so the shear/tensile transition of the potential
passes through the same delimiter:

```
p_q = p_d + k_q τ_d
```

**Yield function** (their Eq. 18), with `R̂_y = √(τII² + (p − p_y)²)`:

```
F = τII − k p − c                  if  τII (p_y − p_d) ≥ τ_d (p_y − p)      shear domain
F = a (R̂_y − R_y)                  otherwise                               tensile cap
```

**Resolved: `F = a (R̂_y − R_y)`.** The extraction read `a R̂_y − R_y`, which does not vanish on the
cap (`R̂_y = R_y` there leaves `(a−1) R_y`), so it is not a yield surface at all. Three things settle
it. Fig. 2's caption describes panel (a) as the same *yield function map* as (b) with `a = 1` — a
scaling that changes the map outside the surface while leaving the surface itself alone, which only
`a(R̂_y − R_y)` does. It makes `‖∇F‖ = a` on both branches. And at the delimiter the cap gradient is
`a(τ_d, p_d − p_y)/R_y = (1, −k)`, exactly the shear gradient, so the composite is C¹ and not merely
C⁰. That is the "scaling to eliminate the discontinuity outside the yield surface" the paper
describes, and the reason Perzyna viscoplasticity needs it: `λ̇ = ⟨F⟩/η_vp` reads `F` *away* from the
surface, so a merely continuous-on-the-surface construction is not enough.

**Domain test, simplified.** Substituting `p_y − p_d = R_y k/a` and `τ_d = R_y/a` into the shear
condition and cancelling the positive factor `R_y/a` collapses it to a half-plane test:

```
shear domain ⟺ p + k τII ≥ p_y            (yield surface)
shear domain ⟺ p + k_q τII ≥ p_q          (flow potential, same reduction on Eq. 19)
```

Two fused multiply-adds, no division, and the delimiter lies exactly on both switching rays. Better
still, each branch pair agrees *identically along its whole ray*, not just at the delimiter: on
`p = p_y − k τII` the cap radius is `R̂_y = a τII`, so both branches equal `a(a τII − R_y)`. The
verification tests exploit this — they are exact-agreement checks, not tolerance-based continuity
checks.

**Flow potential** (their Eq. 19), with `R̂_q = √(τII² + (p − p_q)²)`:

```
Q = τII − k_q p − const            if  τII (p_q − p_d) ≥ τ_d (p_q − p)
Q = b R̂_q − const                  otherwise
```

**Flow direction** (their Eq. 21, 22), `∂Q/∂σ_ij = B_τ τ_ij + B_p δ_ij`:

| domain | `B_τ` | `B_p` |
|---|---|---|
| shear | `1 / (2 τII)` | `k_q / 3` |
| cap | `b / (2 R̂_q)` | `−b (p − p_q) / (3 R̂_q)` |

The domain test is a sign check on one line through the delimiter, so no active-surface search is
needed and the return direction is single-valued everywhere above the surface. That is the property
that makes this model cheap.

**Perzyna regularisation** (their Eq. 11, 12): `λ̇ = ⟨F⟩ / η_vp`, Macaulay brackets, rate-independent
plasticity recovered as `η_vp → 0` and viscoelasticity enforced as `η_vp → ∞`. FEMTools' `η_reg`
plays this role already, in the same place.

**Softening** (their Eq. 23–25): linear in the accumulated deviatoric viscoplastic strain
`γ = ∫ ε̇II^vp dt`, with `φ = max(φ_init + H_φ γ, φ_min)` and `c_MC = max(c_init + H_c γ, c_min)`,
hardening moduli negative for softening. The accumulated *volumetric* viscoplastic strain
`θ = ∫ ε̇_vol^vp dt` is carried as a second history variable and is what identifies tensile zones.

## 2. What FEMTools already has

**Corrected 2026-09-20 against `main` at 6adeeef.** The table below previously listed a `damage`
field, a `DamageLaw`, per-IP plastic history, and a family of dispatch wrappers. None of them exist —
not on `main`, not on `adm/volcanos`, not on any branch (`git grep DamageLaw` over every remote head
returns nothing). Its line numbers pointed into a `rheology.jl` of 600+ lines; the real file is 208
lines on `main` and 322 on `adm/volcanos`. Only the adjoint row was accurate. The consequences for
sizing are recorded under the table.

| Piece | Where | State |
|---|---|---|
| `DruckerPrager{nphases,FP}` with `cosϕ, sinϕ, sinΨ, C, η_reg, Kb` | `src/stokes/types/stokes_types.jl:313` | matches the paper's `k = sinφ`, `c = c_MC cosφ` exactly; **no** `damage` field |
| 2-D return map, closed-form multiplier | `src/stokes/assemblers/rheology.jl:148` | `λ = F / (ηve + η_reg + Kb Δt ∂Q/∂P ∂F/∂P)` |
| `Nothing` dispatch for the no-plasticity path | `src/stokes/assemblers/rheology.jl:133` | the only dispatch wrapper that exists |
| 3-D return map, radial flow | **absent on `main`**; `rheology.jl:291` on `origin/adm/volcanos` | documented there to differ from plane strain |
| Adjoint blocks differentiate the residual that calls the return map | `src/stokes/assemblers/adjoint_operator.jl:256` | ForwardDiff; Enzyme on the other path |
| `_assert_frozen_symmetry` | `src/stokes/assemblers/adjoint_operator.jl:116`, fired from `:400` and `:402` | message to fix per §7 |
| Cap geometry, yield function, flow direction | `src/stokes/assemblers/rheology.jl:211–324` | **added**; pure functions, not yet wired into any solver path |

What this changes in the gap register:

- **CAP-6 is not "reuse `DamageLaw` or add linear moduli"** — there is no softening machinery of any
  kind to reuse, so the paper's Eq. 24 moduli are the only candidate and CAP-6 grows from M to L.
- **CAP-4 has a hidden prerequisite.** Both `γ` (Eq. 23) and `θ` (Eq. 25) are accumulated per
  integration point, and there is no per-IP history array to accumulate into. That storage, and its
  update at the physical-time boundary, is infrastructure this plan never registered. It is the same
  storage CAP-6 needs, so build it once, under CAP-4.
- **CAP-7 depends on which branch this lands on.** A 3-D method to extend exists only on
  `adm/volcanos`.

The structural fact that shapes this whole plan: **FEMTools' multiplier is closed-form because its
yield surface is linear**. On the DP line `∂F/∂τII = 1` and `∂Q/∂τII = 1`, so consistency is one
linear equation in `λ`. On the cap both derivatives are state-dependent (`a τII/R̂_y`,
`b τII/(2R̂_q)`), and the closed form stops existing. A local solve is not optional.

## 3. Gap register

| ID | Gap | Where | Size |
|---|---|---|---|
| CAP-1 | The type, its derived cap geometry, and the constructor's validity conditions | `types/stokes_types.jl` | S |
| CAP-2 | ~~Cap geometry as an inline function, because softening moves it~~ **done** | `assemblers/rheology.jl:211` | S |
| CAP-3 | Local scalar solve for `λ` replacing the closed form | `assemblers/rheology.jl` | **L** |
| CAP-4 | Per-IP history storage (prerequisite, does not exist) plus `γ` and volumetric `θ` | `assemblers/rheology.jl`, Stokes state | **L** |
| CAP-5 | ~~Pressure-scheme decision~~ **decided: trial pressure**; remaining work is an elastic `K` in the continuity residual | `assemblers/pressure_residual.jl` | **L** |
| CAP-6 | Softening from scratch: the paper's linear moduli (Eq. 24); no `DamageLaw` exists to reuse | `types/`, `rheology.jl` | **L** |
| CAP-7 | 3-D method | `rheology.jl:567` | M |
| CAP-8 | Adjoint: differentiating through a local iteration | `adjoint_operator.jl`, `DR_adjoint.jl` | **L, risky** |
| CAP-9 | Verification ladder | `test/`, `examples/benchmarks/` | M |
| CAP-10 | Detector and `θ` reconciliation in the Reykjanes workflow | `examples/reykjanes/` | M |

## 4. CAP-1, CAP-2: the type

```julia
struct DruckerPragerCap{nphases, FP}
    cosϕ::NTuple{nphases, FP}
    sinϕ::NTuple{nphases, FP}      # k
    sinΨ::NTuple{nphases, FP}      # k_q
    C::NTuple{nphases, FP}         # c_MC; the yield uses c = C cosϕ, as DruckerPrager does
    pT::NTuple{nphases, FP}        # tensile strength, compression-positive (so ≤ 0)
    η_reg::NTuple{nphases, FP}     # η_vp
    Kb::NTuple{nphases, FP}
    damage::Union{Nothing, DamageLaw{nphases, FP}}
end
```

Keep the field names and order of `DruckerPrager` for everything they share, so the two read the
same at the call sites and a reviewer can diff them.

Do **not** store `p_y, R_y, p_d, τ_d, p_q` per phase. Softening and `DamageLaw` both change `φ` and
`c` per integration point, and the cap geometry is a function of those; a precomputed per-phase
geometry would silently freeze the cap while the shear branch softened. Compute them in one inline
call from the already-interpolated `(k, k_q, c, p_T)`:

```julia
@inline function cap_geometry(k, k_q, c, pT)
    a = sqrt(one(k) + k^2)
    b = sqrt(one(k) + k_q^2)
    p_y = (a * pT + c) / (a - k)
    R_y = p_y - pT
    p_d = p_y - R_y * k / a
    τ_d = R_y / a
    p_q = p_d + k_q * τ_d
    return (; a, b, p_y, R_y, p_d, τ_d, p_q)
end
```

Constructor conditions, each one a test: `a > k` (always true for `k = sinφ ∈ [0,1)`, but assert it
so a future `k` from a different friction law cannot break it); `R_y > 0`; apex ordering; `η_reg > 0`;
and the tangency identity `R_y ≈ (k p_y + c)/a`.

## 5. CAP-3: the integration-point algorithm

The paper solves a 3-unknown system `(τII, p, λ̇)` (their Eq. 42, Jacobian Eq. 47) because it carries
diffusion and dislocation creep, so even the deviatoric relation is nonlinear. FEMTools' kernel takes
`η` as given and its deviatoric relation is linear, so the system collapses. **Recommended: a scalar
Newton in `λ`**, with the state expressed through it:

```
τII(λ) = τII_trial − 2 ηve λ B_τ(λ) τII(λ)     radial in the deviatoric plane, both branches
p(λ)   = p_trial  − 3 Kb Δt λ B_p(λ)           volumetric return, sign per the CAP-5 convention
r(λ)   = F(τII(λ), p(λ)) − λ η_reg = 0         Perzyna consistency
```

with a bounded iteration count, an Armijo back-tracking line search (`ρ = 0.9`, `α_min = 0.1`, as the
paper uses) and the trial state as the initial guess. The paper is explicit that the line search is
not optional: without it the local iterations form closed loops in stress space and never converge,
even though the surface is smooth (their Fig. 3a versus 3c). Budget that finding; do not rediscover
it.

Three constraints from this codebase rather than from the paper:

1. **The loop runs inside a KernelAbstractions kernel on GPU.** Fixed `max_iter`, no allocation, no
   early `return` that diverges the warp, `Float32`-safe tolerances. Prefer a fixed iteration count
   with a convergence flag carried out, over a `while` on a residual test.
2. **The deviatoric flow direction must match the existing 2-D convention.** The current 2-D code
   differentiates the invariant with `τzz = −τxx − τyy` slaved, which is *not* the radial direction
   the 3-D method uses — `rheology.jl:548` documents that the two deliberately differ. The cap must
   use one convention consistently on both sides of the delimiter, or the flow direction will jump
   exactly where the model promises smoothness.
3. **`λ` must reduce exactly.** With the cap pushed out of reach the scalar Newton must return the
   closed-form `λ` of `DruckerPrager` to round-off, in one iteration. That is CAP-9's first test and
   the cheapest guard against a regression in the shear branch.

## 6. CAP-5: the pressure scheme, which is the blocking decision

The paper offers two readings of the global pressure and is emphatic that they are different models,
not different implementations (their Sect. 3.2–3.4):

- **True pressure**: the global `p` is the spherical Cauchy stress; the viscoplastic volumetric
  strain rate is added to the continuity residual (their Eq. 29) so the global pressure feels the
  dilation.
- **Trial pressure**: the global `p` is the trial visco-elastic pressure; the local update is
  `p = p̄ + K ε̇_vol^vp Δt` (their Eq. 31) and the volumetric term then *cancels* from the global
  continuity residual (their Eq. 34). They report this scheme converges more robustly and use it
  throughout.

FEMTools does neither yet. `integrate_PH_pressure_residual` carries
`−∇·v − (p − p_n)/(η_b Δt) + α ΔT/Δt + Q`: a bulk-viscosity term, not an elastic `K`, and no
viscoplastic volumetric term at all. The dilation currently enters only through
`Kb Δt ∂Q/∂P ∂F/∂P` in the multiplier's denominator — a local stabilisation, not a mass balance.
Adopting the cap without settling this means the mode-I opening it predicts has no path into the
global mass balance, which is the entire point of the model.

Note also the paper's warning: **every dilatant plasticity model needs a finite elastic bulk
modulus**; both schemes fail for `K → ∞`. FEMTools' `K` is `Inf` in the incompressible gauge that the
adjoint tests and several miniapps use, so this interacts with GAP-25 in `REYKJANES_PLAN.md` and with
the oracle finding that the frozen adjoint operator already refuses to assemble at finite `K`.

**Decided 2026-09-20: trial pressure scheme.** Recorded in `.agents/solver.md`. The local update is
`p = p̄ + K ε̇_vol^vp Δt` and the volumetric term cancels from the global continuity residual, so the
residual the solver assembles keeps its current form and only gains an elastic `K` in place of the
bulk-viscosity term. Two things follow that CAP-3 must respect: `p(λ)` in the local solve is the
*local* pressure recovered from the trial value, not the global unknown; and global and local
pressure are not expected to agree at convergence under this scheme, so no diagnostic should treat
their difference as an error — it is the dilation.

The `K → ∞` consequence stands and is now a hard constraint, not a caveat: the cap cannot be used in
the incompressible gauge at all.

## 7. CAP-8: the adjoint, and why it is the real risk

`_deviatoric_stress_with_multiplier` is called from `integrate_momentum_residual`, which
`element_adjoint_operator_blocks` differentiates with ForwardDiff and `DR_adjoint.jl` differentiates
with Enzyme. Putting a Newton loop with a line search inside it has three consequences:

1. ForwardDiff through a converged fixed-point iteration gives the right derivative only once the
   iteration has converged in the dual components too; an iteration cutoff that is fine for the
   primal can be wrong for the derivative. The clean answer is the implicit function theorem: solve
   the primal with the loop, then obtain `∂λ/∂(state)` from `∂r/∂λ` analytically and wrap the result
   so AD sees a closed-form expression. Budget this inside CAP-3, not as a later fix.
2. Enzyme through a line-search branch is fragile. The non-smooth switch list in `.agents/solver.md`
   has to gain the cap's domain test and the Macaulay bracket, with their treatment stated.
3. The existing symmetry assertions will fire. A non-associated cap (`ψ ≠ φ`) makes `A ≠ Aᵀ` and
   `C ≠ Bᵀ`, which `_assert_frozen_symmetry` already reports for plastic DP (0.125 defect at 97 % of
   points at yield). The GAP-25 oracle also measured `‖C − Bᵀ‖/‖C‖ = 7.8e-3` at finite `K` with **no**
   plasticity at all, so that assertion's message — "Without a plastic model the tangent is symmetric
   and this cannot happen" — is already wrong and will be doubly wrong here. Fix it as part of this
   work, since the cap is exactly the configuration that needs finite `K`.

## 8. CAP-9: verification ladder

| # | Test | Passes when | Cost |
|---|---|---|---|
| V1 ✅ | Cap geometry unit tests (`test/test_drucker_prager_cap.jl`, 140 checks) | tangency identity holds; `F = 0` on the cap circle; `‖∇F‖ = a` on both sides of the delimiter; branches agree *exactly* along the whole switching ray; same for `Q` with `b`; `Float32`/`Float64` and inference | ms |
| V2 | Reduction to `DruckerPrager` | with the cap out of reach, `τij`, `λ` and `∂Q∂τ` match the existing return map to round-off, in one local iteration | ms |
| V3 | Yield-surface map | reproduces Fig. 2a–c against the Zenodo script; **this is what resolves the reconstructed formulae** | minutes |
| V4 | 0-D stress integration (their §4.1) | stress paths for uniaxial restrained extension and pure shear, with their Table 1 parameters | seconds |
| V5 | Perzyna regularisation (their §4.2) | mode-I band width independent of resolution at fixed `η_vp`, and mesh-dependent without it | hours |
| V6 | Brittle crust localisation (their §4.3) | the Kaus (2010) benchmark, which FEMTools should be run against anyway | hours |
| V7 | Tensile failure zone propagation (their §4.4) | the dike-relevant case; the one that feeds the Reykjanes work | hours |
| V8 | Adjoint gradient | ForwardDiff through the return map against central differences at a point in each domain and, separately, near the delimiter | seconds |

V1, V2 and V8 must exist before the kernel is used anywhere. V3 is what turns the reconstructed
equations into checked ones.

## 9. Build order

1. ~~**CAP-2 geometry + V1 + V3.**~~ **Done 2026-09-20** except V3. `cap_geometry`,
   `cap_yield_function` and `cap_flow_direction` are in `src/stokes/assemblers/rheology.jl`, pure and
   uncalled; `test/test_drucker_prager_cap.jl` is V1 at 140 checks. Every reconstructed formula was
   resolved against the paper and pinned by an independent geometric identity (§1). V3 is still open
   but is now a picture, not a decision.
2. ~~**CAP-5 decision**, written into `.agents/solver.md` with its consequence for `K = Inf` runs.~~
   **Done 2026-09-20: trial pressure scheme.** See §6.
3. **CAP-1 type + CAP-3 scalar Newton + V2.** The reduction test is the acceptance gate.
4. **CAP-8 adjoint derivative + V8**, and the symmetry-assertion message fix.
5. **CAP-4 volumetric history + CAP-6 softening**, then V4.
6. **V5, V6, V7** as benchmarks under `examples/benchmarks/stokes/`, following the `first_threshold`
   layout: a function-only setup plus a sweep driver.
7. **CAP-7 3-D**, last, and only if a 3-D consumer exists.
8. **CAP-10**: point the Reykjanes detector at `θ` instead of the hydraulic-margin proxy, and record
   in `REYKJANES_PLAN.md` whether that changes the threshold. GAP-5 closes here.

## 10. What this plan does not settle

- Whether `p_T` is a per-phase constant or itself softens. The paper softens `φ` and `c_MC` only.
- Whether the Reykjanes protocol's `T₀` and the constitutive `p_T` are the same number. They are the
  same *concept*; making them the same *parameter* is the point of CAP-10, but it changes the
  detector's calibration and is therefore a Phase-0 decision, not a code decision.
- The value of `η_vp` for any Reykjanes run. The paper is explicit that regularisation viscosity is a
  numerical parameter with a localisation-sharpness trade-off, to be chosen per process, and that it
  introduces a length scale. It belongs in the run record, not in a default.
