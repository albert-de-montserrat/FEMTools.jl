# Adjoint solver: performance and convergence plan

Working document for branch `adm/adjoint-perf`. It records the analysis of the
current adjoint solver, the planned changes, the gate each change must pass, and
the measurements each phase produced. Sections marked *(to fill)* are completed
as the work lands, so this file doubles as the record of what was tried and what
it bought.

Scope:

- `src/stokes/solvers/DR_adjoint.jl` — `solve_stokes_adjoint_dyrel!`
- `src/stokes/assemblers/momentum_residuals_adj.jl` — momentum transpose
- `src/stokes/assemblers/pressure_residual_adj.jl` — pressure transpose
- `src/stokes/assemblers/momentum_residuals.jl` — Jacobian blocks reused below
- `examples/stokes/sinking_block/sinking_block_adj.jl` — driver and benchmark case
- `test/test_stokes_adjoint_api.jl` — finite-difference gradient gate

---

## 1. How the solver currently works

`solve_stokes_adjoint_dyrel!` solves the discrete adjoint

```
(∂R/∂u)ᵀ λ = -∂J/∂u
```

with the same Powell–Hestenes (PH) outer / dynamic-relaxation (DYREL) inner
iteration as the forward solve. The forward state in `dr` is converged before
entry, and the docstring states the invariant explicitly: *the transpose
Jacobian, its diagonal preconditioner, and λmax are frozen at that state.*

Structure of one solve:

- Once, at setup: `assemble_augmented_momentum_jacobian_matrices_atomix!`
  produces the diagonal preconditioners `PC_vx`, `PC_vy` and the absolute row
  sums `∂Rv_x∂vx`, `∂Rv_y∂vy`; `λmax` and the Chebyshev pair `(α, β)` follow.
- Outer loop (`max_ph_iterations`, default 100): evaluate the adjoint residual,
  test convergence, run the inner loop until the velocity residual drops by
  `rel_drop`, then take one Arrow–Hurwicz step on `λP`.
- Inner loop: Chebyshev-damped DYREL updates of `λvx`, `λvy`, homogeneous
  Dirichlet projection, and a **full re-evaluation of the adjoint residual every
  iteration**. Every `ncheck` iterations, `λmin` is re-estimated and the
  Chebyshev pair refreshed.

The residual evaluation, `assemble_adjoint_residual!`, is the entire per-iteration
cost. It runs three Enzyme reverse-mode assemblies over all elements:

1. `assemble_momentum_residual_matrices_atomix_adj!` — seeds `λv` on the momentum
   residual, producing `(∂Rv/∂v)ᵀλv → dvx, dvy`, `(∂Rv/∂P)ᵀλv → dP`, and
   `(∂Rv/∂Pnum)ᵀλv → dPnum`.
2. `assemble_pressure_residual_matrices_atomix_adj!` seeded with
   `γP·dPnum/M_P` — the PH augmented grad-div self-coupling.
3. The same assembler seeded with `λP` — the saddle-point coupling.

## 2. The two findings that drive this plan

### 2.1 A constant operator is being rebuilt by AD on every iteration

Because the forward state is frozen, the adjoint residual is an **affine function
of `λ` with a constant operator**:

```
ResλV = objective_v + Kᵀλv + Cᵀ(γP·Bnumᵀλv / M_P) + CᵀλP
ResλP = Bᵀλv
```

with `K = ∂Rv/∂v`, `B = ∂Rv/∂P`, `Bnum = ∂Rv/∂Pnum`, `C = ∂RP/∂v`, none of which
change during the solve.

Each of the three Enzyme passes above nevertheless re-runs the *nonlinear primal*
residual — rheology evaluation, stress update, quadrature over all integration
points — before its reverse sweep, and then discards the primal
(`Rv_x_buf`, `Rv_y_buf` are write-only scratch). So every iteration pays for
automatic differentiation of a nonlinear operator purely to evaluate a
matrix-vector product with a matrix that never changes.

This is the dominant cost and the target of Phase 2.

### 2.2 λmax is a Gershgorin bound, not an eigenvalue

`_checked_λmax` (`src/utils.jl`) returns

```julia
maximum(jacobian ./ PC)
```

where `jacobian` holds the assembled **absolute row sums** from
`element_augmented_momentum_jacobians` and `PC` the assembled diagonal. That is a
Gershgorin bound on the Jacobi-preconditioned spectral radius: correct as an
upper bound, but typically 2–4× above the true λmax for an elliptic operator.

The consequence is direct. The pseudo-time step is `Δτ = 2/sqrt(λmax)·CFL_v`, and
the Chebyshev-accelerated iteration count scales as `sqrt(λmax/λmin)`. An
overestimate of λmax by a factor `f` costs roughly `sqrt(f)` in iterations while
also shrinking every step. Phase 3a replaces it with a measured λmax, which
becomes almost free once Phase 2 provides a cheap operator apply.

### 2.3 Secondary observations

- **The two pressure transposes are fusable.** A pullback is linear in its seed,
  and both calls accumulate into the same `dvx`, `dvy` (the `dP_scratch` output is
  zeroed and discarded both times). One call seeded with `λP + γP·dPnum/M_P` is
  algebraically identical and removes one of three element sweeps.
- **Synchronization granularity.** `stokes_update_rate!`,
  `stokes_update_variable!`, and `apply_dirichlet!` each call
  `KernelAbstractions.synchronize` internally. The inner loop issues eight such
  calls per iteration on top of the assembly barriers — negligible on CPU, pure
  latency on an accelerator.
- **Per-iteration allocations.** `ResλVx .- ResλVx0` allocates a temporary at
  every `ncheck` for the λmin estimate.
- **Redundant tail assembly.** After the loop exits, `assemble_adjoint_residual!`
  runs once more for reporting even when the loop broke on the convergence test
  with that residual already in hand.
- **The benchmark case has no viscosity contrast.** The example sets
  `η = (1.0, 1e0)` — both phases are 1.0, so only density differs
  (`ρ0 = (1.0, 2e0)`). The operator is therefore trivially well-conditioned and
  any convergence-rate measurement on it is uninformative about the regime the
  solver actually needs to handle. Phase 0 fixes this before any convergence
  claim is made.

## 3. Phases

Each phase lands separately with its own benchmark delta. Phases 1 and 2 reduce
the cost per iteration; Phase 3 reduces the number of iterations and is only
honestly measurable once iterations are cheap and the operator is explicit.

### Phase 0 — baseline, benchmark harness, and guard rails

Nothing else lands until this is in place.

1. Add `bench/adjoint_bench.jl` driving `main(; show_plot = false)` over
   `max_area ∈ {1/32², 1/64², 1/128²}`. Record forward iterations and wall time,
   adjoint outer/inner iterations and wall time, time per adjoint iteration, and
   the phase-summed gradients.
2. Add viscosity-contrast cases `η₂/η₁ ∈ {1, 10, 10², 10³}` to both the benchmark
   and the example. Convergence claims are judged on the contrast cases.
3. Profile one adjoint solve headlessly and attribute time across the momentum
   transpose, the two pressure transposes, the DYREL vector kernels, and
   synchronization.
4. Correctness gates for every later phase:
   - `test/test_stokes_adjoint_api.jl` — adjoint gradient against finite
     differences at `rtol = 1e-4` (already exists).
   - New: adjoint residual vectors from the old and new operator paths agree to
     ~1e-12 on a small mesh.

**Gate:** baseline recorded, full test suite green.

**Result** *(to fill)*

### Phase 1 — cost reductions with no change to the iterates

Every item here is algebraically identical to the current iteration; iteration
counts must not move.

1. Fuse the two pressure-transpose passes into one seeded with
   `λP + γP·dPnum/M_P`. Expect 20–30% off each residual evaluation.
2. Fuse the four velocity updates into one kernel launch and the four Dirichlet
   projections into another, so the inner loop synchronizes once rather than
   eight times.
3. Give `_stokes_λmin` a preallocated difference buffer; audit the `fill!` block
   at the head of `assemble_adjoint_residual!` (primal buffers Enzyme overwrites
   need no pre-zeroing — only the shadows do).
4. Reuse the in-hand residual instead of the redundant tail assembly when the
   loop breaks on convergence.

**Gate:** FD gradient test unchanged; iteration counts bit-comparable to
baseline; measured per-iteration time down.

**Result** *(to fill)*

### Phase 2 — freeze the operator (main performance work)

Assemble the transpose operator once, then apply it matrix-free.

- Cache per element: `Kᵉ` (2NV×2NV = 14×14 for T7), `Bᵉ` (14×3), `Bnumᵉ` (14×3),
  `Cᵉ` (3×14), stored as `SMatrix`.
- `element_augmented_momentum_jacobians` already builds the ForwardDiff blocks
  and then reduces them to row sums and diagonals; extend that path to retain the
  full blocks rather than adding a second assembly route.
- New apply kernel: gather element `λ`, dense GEMV, scatter with the existing
  Atomix pattern. No rheology, no AD, no primal recompute.
- Memory: ≈322 Float64 ≈ 2.6 kB per element in 2D T7/P1, so ≈26 MB at 10⁴
  elements. 3D elements make this materially heavier, so the Enzyme path stays
  selectable as a fallback rather than being deleted.

**Gate:** new-operator residual matches the Enzyme residual to ~1e-12 on a small
mesh; FD gradient test unchanged; benchmark shows the per-iteration speedup.

**Result** *(to fill)*

### Phase 3 — convergence rate

**3a. Measured λmax.** With a cheap apply available, run power iterations on the
Jacobi-preconditioned operator until λmax settles (20–50 typically). The cost is
a fraction of a percent of the solve and should recover most of the Gershgorin
overestimate.

**3b. Preconditioned conjugate gradients for the inner solve.** The augmented
velocity block is symmetric positive definite whenever the forward Jacobian is
symmetric — true for viscous and power-law Stokes — and the PH augmentation
`Cᵀ(γP/M_P)Bnum` is symmetric positive semi-definite. CG needs no spectral
estimates at all, adapts to the actual spectrum, and reuses `PC_vx`/`PC_vy` as
its Jacobi preconditioner.

Guard: probe symmetry at setup by comparing `⟨Ax, y⟩` against `⟨x, Ay⟩` on random
vectors. On mismatch, error out or fall back to the DYREL path explicitly — never
degrade silently, since a non-symmetric operator makes CG converge to the wrong
answer rather than fail visibly.

**3c. Outer PH loop.** The adjoint inherits the forward `γP` (built with
`γfact = 40`). With a strong inner solver the adjoint augmentation can be raised
independently, trading inner conditioning for fewer outer iterations. Sweep γ and
`rel_drop` (currently 0.1, one decade of velocity-residual drop per outer
iteration) on the contrast cases.

**Gate:** FD gradient unchanged; total iteration count down across the mesh *and*
viscosity-contrast sweep, not only the trivial case.

**Result** *(to fill)*

### Phase 4 — opportunistic

Pursue only if Phases 1–3 leave something material on the table.

- **Warm start.** `λ` starts from zero. In an optimization loop the previous
  design's `λ` is an excellent initial guess — no benefit to a single solve, large
  benefit to the workflow the adjoint exists to serve.
- **Direct 2D CPU path.** The operator is frozen and linear and
  `generate_sparsity_pattern` already exists, so the full transposed saddle-point
  matrix can be assembled and factorized once, giving the exact adjoint in one
  solve for moderate 2D meshes. Keep it behind an explicit solver-choice
  argument; the iterative path remains the portable, GPU-capable, 3D default.
- **Feed the frozen-operator machinery back into the forward Newton steps** if the
  profile justifies it.

**Result** *(to fill)*

### Phase 5 — documentation

Once the solver's behavior is settled, extend `docs/src/stokes.md`. The existing
*Discrete adjoint and material sensitivities* section states the mathematics but
says nothing about how the system is actually solved or how to drive it. Add:

- Why the adjoint is discretely consistent (same spaces, quadrature, and element
  operators, transposed exactly) and why that matters for gradient accuracy.
- The solver structure that results from the frozen forward state: constant
  linear operator, assembled once, applied many times.
- The inner/outer split, what each knob does (`adjoint_tol`, `rel_drop`,
  `max_ph_iterations`, `γP`), and how to diagnose a stalling solve.
- Cost and memory characteristics, including the cached-operator footprint and
  when to prefer the fallback path.
- A worked pointer to `examples/stokes/sinking_block/sinking_block_adj.jl` and to
  the finite-difference gradient check as the way to validate a new objective.

Update the docstring of `solve_stokes_adjoint_dyrel!` in step, and keep both
statements about what the code *is*, not about this plan.

**Result** *(to fill)*

---

## 4. Measurement notes

- Benchmark from **one warm session**, after a discarded warm-up run. The
  quantity of interest is steady-state solver throughput, so compilation must be
  amortized rather than folded into the measurement. Fresh one-shot processes are
  the right tool for package-load and time-to-first-solve questions, which are
  not what this work is about.
- Run with `show_plot = false` and `verbose = false`.
- Report iteration counts alongside wall time. A change that cuts wall time while
  raising iteration counts (or vice versa) is telling you something, and the plan
  distinguishes the two deliberately.
- Every convergence measurement is taken across the viscosity-contrast sweep, not
  at unit contrast alone.

## 5. Harness

- `examples/benchmarks/adjoint_perf.jl` — `run_adjoint_benchmark` sweeps mesh
  refinement against viscosity contrast and prints one row per case;
  `compare_adjoint_benchmarks(baseline, candidate)` reports iteration, wall-time,
  and per-iteration ratios between two sweeps.
- `examples/stokes/sinking_block/sinking_block_adj.jl` gained `η_incl` (inclusion
  viscosity against a unit matrix viscosity, hence the contrast), a `verbose`
  switch for the forward trace, and `t_forward`/`t_adjoint` in its return value.
