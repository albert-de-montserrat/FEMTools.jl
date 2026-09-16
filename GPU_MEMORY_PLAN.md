# GPU memory plan: forward DYREL solvers and adjoints

Working document. It records the measured memory map of the Stokes solvers, the
planned changes in the order they are to be made, the gate each change must
pass, and the measurement each one produced. Sections marked *(to fill)* are
completed as the work lands, so this file doubles as the record of what was
tried and what it bought.

Memory is the binding constraint on GPU problem size: a case that does not fit
cannot be run at all, whereas a case that runs slowly still produces an answer.
The ordering below therefore follows bytes saved per unit of disruption, not
arithmetic saved.

Scope:

- `src/mesh/mesh.jl` — `precompute_geometry` and the geometry kernel
- `src/mesh/mixed_mesh.jl` — `MixedMeshCache`, `MixedMesh.normals`
- `src/stokes/types/stokes_types.jl` — `StokesDR` state
- `src/stokes/solvers/DR.jl` — forward Powell–Hestenes / DYREL iteration
- `src/stokes/assemblers/adjoint_operator.jl` — `FrozenAdjointOperator`
- `src/stokes/solvers/DR_adjoint.jl` — adjoint solve and its scratch
- `examples/benchmarks/memory_footprint.jl` — harness and baseline record

---

## 1. Tracking against the baseline

Every change in this plan is judged against a recorded baseline, not against an
expectation. `examples/benchmarks/memory_footprint.jl` produces the record:

```julia
julia --project=.
include("examples/benchmarks/memory_footprint.jl")
results = run_memory_benchmark()
write_memory_baseline(results, "examples/benchmarks/memory_baseline.jl")
```

and, from a session running the changed code,

```julia
base = read_memory_baseline("examples/benchmarks/memory_baseline.jl")
compare_memory_footprints(base, run_memory_benchmark())
```

The comparison spans a code change and therefore cannot happen inside one
session, which is why the baseline goes to disk. Measure the baseline from a
worktree at the pre-change commit, so both halves run on the same machine under
the same conditions:

```
git worktree add /tmp/femtools-base <pre-change commit> --detach
cp examples/benchmarks/memory_footprint.jl /tmp/femtools-base/examples/benchmarks/
julia --project=/tmp/femtools-base -e 'using Pkg; Pkg.instantiate()'
julia --project=/tmp/femtools-base -t 6 -e '...run_memory_benchmark()...'
```

Pass absolute project paths: a `cd` into the worktree silently makes
`--project=.` the wrong tree, and the run then measures the baseline twice.

Four quantities are tracked together, because a change that moves one at the
expense of another has to be visible:

| Quantity | Why it is tracked |
|:---|:---|
| Bytes per element, by component | The constraint on problem size. Reported per component so a saving in one place is not hidden by growth in another. |
| Inner iteration count | A trend indicator for solver behavior — see the caveat below; it is *not* the equivalence check. |
| Wall time per solve | Where a byte is traded for arithmetic, this is the price. |
| Host allocation volume per solve | Transient churn. It scales with iteration count, so a small per-call allocation is a large sustained cost. |

**Equivalence is checked with `residual_fingerprint`, not with iteration counts.**
The assemblers scatter element contributions with atomics, so the order in which
they are summed depends on thread scheduling. Two runs of *identical* code
converge at different iteration counts: the same commit measured on this machine
gave 9 550 iterations in a one-shot six-thread process and 9 450 in a warm REPL.
An iteration count is therefore evidence about solver behavior, not about
arithmetic. To show that a reformulation changed nothing, write the momentum
residual at a fixed state under **one thread** before and after and compare the
files byte for byte:

```
julia --project=. -t 1 -e 'include("examples/benchmarks/memory_footprint.jl"); residual_fingerprint("before.bin")'
# apply the change
julia --project=. -t 1 -e 'include("examples/benchmarks/memory_footprint.jl"); residual_fingerprint("after.bin")'
cmp before.bin after.bin
```

Rules for reading a comparison:

- Ratios are `baseline / candidate`; above 1 is an improvement.
- Baseline and candidate must run under the same Julia version, thread count,
  and backend. The harness records all three and refuses a comparison across
  them: on this machine the same code is roughly three times faster on six
  threads than on one, which swamps every effect this plan is chasing.
- Measure the baseline from a worktree at the pre-change commit rather than from
  a record taken under other conditions.
- Wall-time regressions beyond a few percent must be explained before the change
  lands. Recomputing a small quantity per quadrature point is expected to be
  free or better on a GPU, where the loads it replaces are the scarce resource;
  a measurable slowdown means that reasoning did not hold and needs checking.
- Report the footprint from the same case that produced the timing. A footprint
  from one mesh and a time from another cannot be compared.

**Baseline.** Recorded in `examples/benchmarks/memory_baseline.jl` from a warm
CPU session at `f2c13a6`, Float64 throughout:

| Row | Case | Result |
|:---|:---|:---|
| footprint | 32×32, 2 048 elements | 4.47 MiB, 2 288.8 B/element |
| footprint | 64×64, 8 192 elements | 17.85 MiB, 2 284.4 B/element |
| converged | 16×16, 512 elements | 9 450 inner iterations, 2 PH steps, 1.163 s, 190.1 MiB allocated (20.6 KiB/iteration) |
| budgeted | 32×32, 20 000 inner iterations | 6.265 s, 724.6 MiB allocated (37.1 KiB/iteration) |

## 2. The measured memory map

### 2-D, T7/P1-disc, 64×64 (8 192 elements, 24 833 velocity nodes, 24 576 pressure DoFs)

| Component | MiB | Share | B/element |
|:---|---:|---:|---:|
| `cache.geo_v` | 6.563 | 36.8 % | 840.0 |
| `cache.geo_P` | 3.063 | 17.2 % | 392.0 |
| stress history (6 arrays) | 2.625 | 14.7 % | 336.0 |
| nodal solver state (21 arrays) | 3.961 | 22.2 % | 507.0 |
| topology (coords, connectivities, DoFs) | 0.880 | 4.9 % | 112.6 |
| `mesh.normals` | 0.379 | 2.1 % | 48.5 |
| phase tags (`Int`) | 0.377 | 2.1 % | 48.3 |
| **total** | **17.85** | | **2 284.4** |

Element-indexed data — geometry plus stress history — is 68.6 % of the total.
The nodal block is 21 arrays over two node sets and is close to irreducible: it
is the solution, its rates, its residuals, the residual snapshots the spectral
estimates need, and the diagonal preconditioners.

### Single-field geometry, per element

| Element | Elements measured | Geometry | B/element | Share of mesh |
|:---|---:|---:|---:|---:|
| T7 (`QuadraticElement{2,7}`) | 8 192 | 6.56 MiB | 840 | 90.4 % |
| Hex8 (`LinearElement{3,8}`) | 32 768 | 50.0 MiB | 1 600 | 96.2 % |
| Hex27 (`QuadraticElement{3,27}`) | 4 096 | 69.2 MiB | 17 712 | 98.0 % |

Geometry dominates because its per-element cost grows with the product of element
arity and quadrature-point count: `NQ · (N · nDim + 1)` numbers. Hex27 carries 27
quadrature points and 27 nodes, hence 17 712 bytes for a single element. At 64³
that is 4.32 GiB of geometry before any solver state exists.

## 3. Phase F — forward solvers

Items appear in the order the work is done, not in numerical order: F8 was
identified while measuring F3 and belongs next to it.

### F1 — store the inverse Jacobian, not the physical gradients

`precompute_geometry` stores, per element and per quadrature point, the physical
shape-function gradients `∂N∂ξ_q · J_q⁻¹`: an `N × nDim` matrix. Of the two
factors only `J_q⁻¹` varies between elements, and it is `nDim × nDim` whatever
the element arity. `∂N∂ξ_q` is a property of the reference element, already
available to every kernel that consumes the geometry.

Storing `(J_q⁻¹, dΩ_q)` and re-forming the product at the point of use is
arithmetically identical and turns a cost that grows with element arity into one
that does not:

| Element | Now (B/element) | After (B/element) | Ratio |
|:---|---:|---:|---:|
| T3 | 168 | 120 | 1.4× |
| Q4 | 288 | 160 | 1.8× |
| T6 | 624 | 240 | 2.6× |
| T7 | 840 | 280 | 3.0× |
| Hex8 | 1 600 | 640 | 2.5× |
| Hex27 | 17 712 | 2 160 | 8.2× |

The saving grows with arity, which is exactly where it is needed: Hex27 geometry
is the overwhelming majority of a 3-D footprint.

The reference gradients are threaded into the assemblers the way
`shape_function_values` already is, and combined with the stored inverse
Jacobian by a lightweight per-element view, so quadrature loops keep the form
`∂N∂x, dΩ = geo_el[q]`. The stored per-point entry is a named type rather than a
plain tuple, so any consumer that was not updated fails immediately instead of
destructuring an inverse Jacobian as if it were a gradient matrix.

**Gate:** momentum-residual fingerprint byte-identical under one thread; solver
tests green; footprint ratios at least those tabulated above.

**Result**

Landed. The fingerprint is byte-identical: `assemble_momentum_residual_matrices_atomix!`
at a fixed state on a 16×16 T7/P1-disc mesh under one thread produces the same
3 202 `Float64` before and after, so the element mathematics did not move. The
full test suite passes.

Footprint, six threads, Julia 1.12.7, baseline from a worktree at `f2c13a6`:

| Quantity | Baseline | After F1 | Ratio |
|:---|---:|---:|---:|
| `geo_v`, 8 192 elements | 6.562 MiB | 2.188 MiB | 3.00× |
| `geo_P`, 8 192 elements | 3.062 MiB | 2.188 MiB | 1.40× |
| 2-D total | 17.85 MiB | 12.60 MiB | 1.42× |
| 2-D bytes per element | 2 284.4 | 1 612.4 | 1.42× |
| T7 geometry | 840 B/el | 280 B/el | 3.00× |
| Hex8 geometry | 1 600 B/el | 640 B/el | 2.50× |
| Hex27 geometry | 17 712 B/el | 2 160 B/el | 8.20× |

`geo_P` gains only 1.40× because it still stores an inverse Jacobian its two
consumers never read; F2 removes the rest.

**The Hex27 case does not merely shrink — it becomes runnable.** On an RTX 3080
the baseline `precompute_geometry_kernel!` for Hex27 fails at `cuLaunchKernelEx`
with `Out of GPU memory` while only 36.7 % of the device is in use and the memory
pool holds 82 MiB. The failure is a launch reservation, not an allocation: each
thread's stored `NTuple{27, Tuple{SMatrix{27,3}, Float64}}` is a 17 712-byte
local frame, so a 128-thread block needs 2.27 MiB of local memory and the driver
cannot reserve enough for the resident blocks. After F1 the frame is 2 160 bytes
and the same kernel launches and completes.

Speed. On the CPU backend the change costs 6–9 % wall time — the price of
re-forming `∂N∂ξ·J⁻¹` per quadrature point instead of loading it. On CUDA, where
the loads it replaces are the scarce resource, it is a small *gain*:

| Case | Backend | Baseline | After F1 |
|:---|:---|---:|---:|
| budgeted, 20 000 iterations | CPU, 6 threads | 6.068 s | 6.425 s |
| converged | CPU, 6 threads | 3.197 s | 3.513 s |
| budgeted, 20 000 iterations | CUDA | 6.025 s | 5.872 s |
| converged | CUDA | 8.513 s | 7.718 s |

Host allocation regressed, by a fixed ~12 KB per kernel launch on both backends
(31.9 → 53.0 KiB per inner iteration on CUDA, 37.1 → 61.3 on CPU). The cause is
the reference-gradient tuple joining the kernel argument list: the amount is
independent of mesh size and scales with launches, not elements, because
KernelAbstractions materialises the argument pack per launch. It is the same
mechanism that already carries `Nq`. F3 targets the allocation column and should
be measured against these numbers rather than against the pre-F1 ones.

### F2 — `geo_P` stores only the weighted volume

The pressure-field geometry is read in exactly two places, and both destructure
it as `_, dΩ = geo_P_el[q]`. The gradient block has no consumer. Storing
`NTuple{NQ, FP}` takes `geo_P` from 392 to 56 bytes per element for T7/P1-disc.

**Gate:** as F1.

**Result**

Landed. The element type of the output array now selects what
`precompute_geometry_kernel!` stores: an `NTuple` of `QuadraturePointGeometry`
keeps the inverse Jacobian and the weighted volume, an `NTuple` of floats keeps
the volume alone. `MixedMeshCache` allocates `geo_P` as the latter, and its two
consumers read `dΩ = geo_P_el[q]`. One kernel still fills both fields.

`geo_P` goes from 392 to 56 bytes per element, **7.00×**. The fingerprint is
byte-identical and the solve is unchanged in both time and allocation, which is
what a pure storage change should look like.

Cumulative for Phase F so far, against `f2c13a6`, six threads:

| Component, 8 192 elements | `f2c13a6` | F1+F3+F2 | Ratio |
|:---|---:|---:|---:|
| `geo_v` | 6.562 MiB | 2.188 MiB | 3.00× |
| `geo_P` | 3.062 MiB | 0.438 MiB | 7.00× |
| stress history | 2.625 MiB | 2.625 MiB | — |
| nodal state | 3.961 MiB | 3.961 MiB | — |
| **total** | **17.847 MiB** | **10.847 MiB** | **1.65×** |
| bytes per element | 2 284.4 | 1 388.4 | 1.65× |

Element-indexed data is now 48 % of the 2-D footprint, down from 69 %.

### F3 — allocation-free spectral estimates

`_stokes_λmin` builds `dV = step .* rate` and is called on a freshly allocated
`Rv .- Rv0`: three full nodal arrays per call, twice per convergence check.
Rewriting it to take `(R, R0)` and fuse the arithmetic into `mapreduce` removes
the temporaries. The same pattern appears in the coupled thermal estimate and in
the 3-D pressure mean.

This is the only item whose target is the allocation column rather than the
footprint column.

**Gate:** host allocation per inner iteration falls; solver tests green.

**Result**

Landed. `fused_sum` in `src/utils.jl` sums a lazy `Broadcasted` instead of
materialising the elementwise expression. On a 24 833-node array the reduction
the estimate performs goes from 596 KB allocated and 213 µs to **zero allocated
and 10 µs**, returning bit-identical values — summing a `Broadcasted` uses the
same pairwise reduction `sum` applies to an array, so this is a pure removal of
temporaries, not a change of arithmetic.

Applied at three sites: `_stokes_λmin`, which now takes `(R, R0)` and forms the
difference inside the reduction; the coupled thermal estimate; and the 3-D
pressure mean, which ran *every* iteration rather than every check. The adjoint
solver's call site loses the in-place `ResλVx0 = ResλVx - ResλVx0` trick along
with the aliasing contract it needed.

Measured against the post-F1 tree, six threads:

| Case | Post-F1 | After F3 | Ratio |
|:---|---:|---:|---:|
| budgeted, allocation | 1 196.4 MiB (61.25 KiB/iter) | 1 042.9 MiB (53.39 KiB/iter) | 1.15× |
| converged, allocation | 1 428.6 MiB | 1 047.6 MiB | 1.05× |
| converged, wall time | 3.513 s | 3.382 s | 1.13× |

The saving is 7.9 KiB per inner iteration, which is what the estimate costs when
amortised over `ncheck = 50` iterations. It does not recover all of F1's
allocation regression because that regression is not in the estimates — see F8.

On CUDA the effect is much larger than the harness reports, because `@allocated`
counts host allocations and the temporaries removed here were *device* arrays.
Measured directly on 24 833-element `CuArray`s, one estimate:

| | Device bytes | Host bytes | Time |
|:---|---:|---:|---:|
| before | 795 200 | 23 328 | 0.62 ms |
| after | 544 | 12 480 | 0.61 ms |

Four full nodal arrays per call become none. Over a 20 000-iteration solve
checking every 50 iterations that is roughly 640 MB of device allocator traffic
removed at no cost in time.

GPU wall-time comparisons on this machine proved unusable: the card drives a
display, and the same code measured 7.7 s and 41.5 s for the converged case in
two runs. Report GPU timings only from a quiet device, or not at all.

### F8 — keep the reference tables out of the kernel argument pack

Per-launch host allocation scales with the byte size of the tuple arguments a
kernel carries. Measured on the momentum residual kernel, mesh-independent and
unchanged from 512 to 8 192 elements:

| Kernel | Tuple arguments (T7/P1-disc) | Allocation per launch, before F1 | after F1 |
|:---|:---|---:|---:|
| momentum residual | `Nq` 392 B, `NqP` 168 B, `∂N∂ξ_v` 784 B | 10 720 B | 23 360 B |
| pressure residual | `NqP` 168 B, `∂N∂ξ_v` 784 B | 13 376 B | 25 952 B |

The momentum-minus-pressure difference is ≈ 5.3 kB in both trees for the 392-byte
`Nq`, and F1's 784-byte gradient table costs ≈ 12.6 kB. A kernel taking a single
tuple argument allocates `3.2 kB + 10 × payload` per launch, so a differenced
figure of fifteen overstates the slope; the Result below measures the tables'
share of an inner iteration from a matched solve instead of from these
differences.

This predates F1 — `Nq` and `NqP` have always been passed this way — but F1
doubled the payload, so the item belongs here. Two candidate fixes:

- Hold the tables in small device arrays, allocated once alongside the geometry.
  The kernel argument becomes a pointer, the data is read from global memory and
  stays cache-resident, and the treatment is identical on every backend. This
  also removes a looming limit: Hex27's reference gradients are a 17.5 kB tuple,
  against a 32 kB CUDA kernel-parameter budget that `Nq` already draws on.
- Make them compile-time constants of the element type through a `@generated`
  accessor that splices the literals, so they carry no argument payload at all.
  Cheaper at run time, heavier machinery, and it must not defeat precompilation.

The first is simpler and backend-uniform, and the second is unsound: a reference
element's type does not determine its tables. `IntegrationPoints` holds its
coordinates and weights as values, so two elements of the same type can carry
different quadrature rules, and a `@generated` accessor keyed on the type would
splice the wrong numbers for the second one without saying so.

**Gate:** momentum-residual fingerprint byte-identical under one thread; host
allocation per inner iteration falls to a small constant; no GPU wall-time
regression.

**Result**

Landed. `quadrature_table(backend, table)` copies a shape-function or
reference-gradient table onto a backend as an array, indexed by quadrature point
exactly as the tuple is. The forward solver builds the three it needs once,
before the Powell–Hestenes loop, and launches the kernel-level entry points that
already took the tables explicitly for Enzyme. The 3-D path holds its bundle in
`Stokes3DWorkspace`; the per-call assembler wrappers still take reference
elements and still build tuples, which is free for a caller that assembles once.

Measured on this machine the effect is far larger than the fifteen-bytes-per-byte
rule predicted, because the rule does not hold. Per-launch allocation is roughly
`3.2 kB + 10 × payload` on the CPU backend, independent of `ndrange` and of block
count, and isolated per-launch figures are **not additive**: they overstate the
CPU cost inside a loop and understate the CUDA one. Only a matched full solve
settles the size of the change.

| Case | Backend | Tuple tables | Array tables | Ratio |
|:---|:---|---:|---:|---:|
| budgeted, 20 000 iterations | CPU, 6 threads | 53.39 KiB/iter | 21.92 | **2.44×** |
| converged, 512 elements | CPU, 6 threads | 38.35 KiB/iter | 12.33 | **3.11×** |
| budgeted, 20 000 iterations | CUDA | 61.99 KiB/iter | 53.78 | 1.15× |

The CUDA gain is much smaller because the argument pack is only part of what a
device launch allocates on the host; the other five kernels of an inner iteration
cost roughly as much again.

**The Hex27 3-D momentum kernel could not launch at all with the tables in the
argument pack.** On an eight-cell mesh, at every workgroup size from 1 to 64,
`cuLaunchKernelEx` returns `Out of GPU memory` with the memory pool holding
52 KiB. Block-size independence separates this from the per-thread local frame F1
removed. The kernel's argument payload is 24 192 bytes — `N` 5 832, `∂N∂ξ` 17 496,
`modes` 864 — and with the tables held as arrays it is 344 bytes and the kernel
launches.

Equivalence. Under one thread the momentum-residual fingerprint and a
500-iteration solve are both byte-identical to the same code with tuple tables.
Under six threads the momentum comparison is not exact and cannot be: the scatter
is atomic. On CUDA the momentum kernel is not reproducible against *itself* —
two runs of one table form differ by 2.0 × 10⁻¹⁶ relative — and the tuple-versus-array
difference, 1.6 × 10⁻¹⁶, is smaller than that. The pressure scatter is not atomic
and is exactly equal on both backends.

Cost. On the CPU backend the momentum kernel is **3.2 % slower** with array
tables: 525.7 µs against 542.5 µs at 8 192 elements, minima of six interleaved
rounds, with no overlap between the two spreads. Reading a table from memory
replaces an immediate the compiler could fold into the unrolled quadrature loop.
The harness time column cannot confirm this — the budgeted case spans 6.562 s to
7.201 s across runs of one tree — so the kernel comparison is the measurement.
This is the same trade as F1 and for the same reason: the loads it adds are cheap
where the argument pack is expensive.

A table must not be the source of a compile-time size. `_gather_old_stress` sized
its stress-history `SVector` with `Val(length(Nq))`, which is a constant for a
tuple and a runtime value for an array; the array form therefore inferred
`Tuple{SVector{7,Int64}, Any, Any}` and allocated 3.7 MB per launch instead of
15 kB. The count now comes from `quadrature_points_val(geo_el)`, which the
geometry carries in its type. `test_type_stability.jl` pins the element residual's
return type with both table forms, and reverting the size to the table
reintroduces the `Any` and fails that test.


### F4 — optional stress history

The six stress arrays are integration-point sized and are dead when the shear
modulus is infinite, which is the viscous case.

**Gate:** viscous cases unchanged; visco-elastic cases still allocate and still
reproduce their reference results.

**Result**

Landed as an opt-in, not as a change of meaning. The plan originally proposed
making `stress_size = nothing` mean *no history*; that is what it already means
for existing callers to get *nodal* storage, so redefining it would silently
change every current caller. `stress_size = :none` is the new third mode: all six
fields become `nothing`, `stress(dr)` returns `nothing`, and the assemblers must
be called with `τ_old = nothing`. An unrecognised `Symbol` raises rather than
falling through to the tuple branch.

On the purely viscous benchmark case the solve converges unchanged with no stress
history at all, confirming that nothing reads it: 2.625 MiB at 8 192 elements
becomes zero, and the 2-D footprint drops **1.35×** on top of F1+F3+F2+F5.

**Decision.** `:none` stays opt-in. The condition under which dropping the
history is safe — every `G` infinite and no plastic model — is one the
constructor could test for itself, and selecting it automatically would collect
the 2.625 MiB without anyone asking. It would also make the field *types* of the
solver state depend on material *values*, so `τ = (dr.τxx, ...)` would return
`nothing`s for a viscous material and break existing drivers with no diagnostic.
A caller that wants the saving asks for it.

### F5 — narrow the phase index

Phase tags are small integers stored as `Int`. `Int32` halves that in all four
solver-state constructors.

**Gate:** phase-dependent results bit-identical.

**Result**

Landed. `StokesDR`, `ThermalDiffusionDR` and `LithostaticPressureDR` allocate
their default phase arrays as `Int32`. `_phase_at` already converts to `Int`
before indexing the material tuples, so no consumer changes and results are
unaffected; a caller supplying its own phase array is likewise unaffected, since
the assemblers never constrained its element type.

Phase storage halves, **2.00×**, which is 1.02× of the 2-D total — the smallest
item in Phase F, included because it is nearly free.

### Phase F so far

Cumulative against `f2c13a6`, T7/P1-disc, 8 192 elements, six threads. The last
column opts into `stress_size = :none`, which the benchmark case is entitled to
do because it is purely viscous.

| Component | `f2c13a6` | F1+F3+F2+F5 | +F4 (viscous) |
|:---|---:|---:|---:|
| `geo_v` | 6.562 MiB | 2.188 | 2.188 |
| `geo_P` | 3.062 | 0.438 | 0.438 |
| stress history | 2.625 | 2.625 | 0.000 |
| nodal state | 3.961 | 3.961 | 3.961 |
| phase tags | 0.377 | 0.188 | 0.188 |
| topology | 0.880 | 0.880 | 0.880 |
| `normals` | 0.379 | 0.379 | 0.379 |
| **total** | **17.847** | **10.658** | **8.033** |
| bytes per element | 2 284.4 | 1 364.3 | 1 028.3 |
| ratio | — | 1.67× | **2.22×** |

Footprint is unchanged by F8, which targets the allocation column. What remains
in Phase F is the nodal block and `normals` (F7).

Wall time on CPU, against `f2c13a6`: F1 costs 6–9 % and F8 a further 3 %, with F3
returning part of F1's. Each is the same trade — arithmetic or a load in place of
a stored or packed byte — and each is free or a gain on CUDA. Read these from the
isolated kernel comparisons, not from the harness time column: the budgeted case
measured 6.562 s and 7.201 s on the same tree in consecutive runs, a 10 % spread
that cannot resolve a 3 % effect.

Host allocation per inner iteration, budgeted case, CPU, six threads: 37.1 KiB at
`f2c13a6`, 61.3 after F1, 53.4 after F3, **21.9 after F8**, or 1.69× below where
the phase started.

### F6 — caller-owned scratch in the 3-D path

`solve_stokes_dyrel!` for 3-D allocates its work arrays on every call. Hoisting
them into caller-owned state removes a per-call allocation whose size scales with
the mesh.

**Gate:** results unchanged; allocation per call independent of mesh size.

**Result**

Landed. `Stokes3DWorkspace` holds the momentum and pressure residuals, the Jacobi
preconditioner, the lumped pressure mass, and the per-component zero boundary
values. `solve_stokes_dyrel!` and `solve_stokes_adjoint_dyrel!` take it as a
`workspace` keyword that defaults to a freshly built one, so nothing breaks and a
loop that wants the reuse asks for it.

Two things keep reuse from becoming a hazard. The preconditioner and pressure
mass depend on the material, so the solver *refills* them through the new
`stokes_preconditioner_3d!` at the start of every solve rather than trusting what
the workspace holds. And the zero boundary values are sized by `fixed_nodes`, so
a workspace belongs to one boundary-condition layout; handing over a mismatched
one raises a `DimensionMismatch` naming both lengths instead of silently
projecting against the wrong nodes.

Measured on a 6³ Hex27 mesh (2 197 nodes, 216 elements), 20 iterations per call:

| | Allocated per call |
|:---|---:|
| fresh scratch | 11 811 280 B |
| caller-owned workspace | 11 670 440 B |
| removed | 140 840 B |

The removed amount is exactly the scratch — six nodal arrays, two element-sized
pressure arrays, three boundary vectors — and it scales linearly: 48 B per node
plus 64 B per element plus the boundary vectors. At 64³ Hex27 that is **≈ 122 MB
allocated and freed on every call**, which on a device is pool traffic the solver
has no reason to generate. Velocity and pressure come out bit-identical either
way.

The 11.7 MB that remains is per-call kernel-launch overhead over 20 iterations,
which is F8's territory.

### F7 — make `MixedMesh.normals` optional

A full `SVector{nDim, FP}` per node is stored whether or not the model uses
normals. Making the field optional is a deliberate public-API change and is
listed last for that reason.

**Gate:** to be decided with the API change.

**Result**

Not taken. At 8 192 elements the field is 24 833 nodes × `SVector{2, Float64}` =
0.38 MiB, against the 9.19 MiB of adjoint blocks A4 removes outright. Buying that
would make the field type of a public struct vary, so every consumer reading
`mesh.normals` would have to handle both forms. The ratio does not justify the
break. Phase F closes at F1–F6 and F8.

## 4. Phase A — adjoints

`FrozenAdjointOperator` stores three dense element blocks: `A` (2NV × 2NV),
`B` (2NV × NP), and `C` (NP × 2NV). For T7/P1-disc that is 2 240 bytes per
element — as much as the entire forward state.

Two structural identities make the packings below possible: `A` is symmetric and
`C` is exactly `+Bᵀ` — note the sign; `C` is *not* `−Bᵀ`.

Establishing this needs a case that genuinely yields, and that takes time
stepping: with `G = 1` under background pure shear `ε̇ = 1`, deviatoric stress
accumulates elastically and only crosses `C·cos ϕ ≈ 1.386` after several steps. A
single solve from zero stress measures the viscoelastic tangent no matter how low
the cohesion is set — sweeping cohesion over four decades on such a case moves
neither figure by a digit, which is a trap worth knowing about.

Measured on a structured 8×8 T7/P1-disc shear-band case after 20 steps of
`Δt = 0.1`, with a weak circular inclusion, `ϕ = 30°`, `ψ = 3°` (non-associated),
and 97.3 % of quadrature points at yield (`τII` peaks at 1.596 against a yield
stress of 1.386):

| max over elements | plastic tangent | same state, `plastic = nothing` |
|:---|---:|---:|
| `‖A − Aᵀ‖ / ‖A‖` | 2.4 × 10⁻⁴ | 7.0 × 10⁻¹⁵ |
| `‖C − Bᵀ‖ / ‖C‖` | **0.125** | 0 |

The second column is the control: the same converged state, differing only in
whether the plastic tangent is taken. So it is the non-associated flow rule, not
the state, that breaks the identities — and `C` differs from `Bᵀ` by 12.5 %,
which packed would be a badly wrong gradient rather than a slightly wrong one.

Every packing below is therefore conditional on `plastic === nothing`, keeps the
full three-block path otherwise, and **asserts its identity at assembly**, per
element, rather than trusting either the theory or this measurement.

### A1 — drop `C` in the viscous case

With `C = Bᵀ` the apply becomes `resv = Aᵀ·λv + B·λP`, `resp = Bᵀ·λv`.
−336 B/element.

**Result**

Landed. `FrozenAdjointOperator.C` is `nothing` when `plastic === nothing`; the
apply reads `B` in its place. Storage falls from `4NV² + 4NV·NP` floating-point
numbers per element to `4NV² + 2NV·NP` — **2 240 to 1 904 bytes** for T7/P1-disc,
1.18×, or 2.6 MiB at 8 192 elements.

The assembly kernel still forms `C` — it is a ForwardDiff jacobian either way, so
assembly cost is unchanged — and records `‖C − Bᵀ‖/‖C‖` per element into a
transient array. The host takes the maximum and errors above `sqrt(eps)` rather
than storing a layout the operator does not satisfy. A2 reuses the same mechanism
for `A`.

Note what that assertion is and is not. It cannot fire through user error: passing
`plastic = nothing` is itself what makes the tangent symmetric, so the two always
agree. It is a regression guard on an assumption that storage now depends on — if
a future rheology or a change to the momentum residual makes the viscous element
operator non-symmetric, the adjoint fails loudly instead of returning a wrong
gradient.

The existing residual comparison against the Enzyme transpose is what proves the
apply still reproduces the full operator, and it passes unchanged. Two new
regressions pin the layout itself: `C === nothing` for a viscous model, and
`C !== nothing` with the right shape for a plastic one.

### A2 — pack `A` as a symmetric triangle

196 stored floats become 105. −728 B/element. With A1 this takes the operator
from 2 240 to 1 176 bytes per element, 1.9×.

**Result**

Landed, and the projection is exactly what the operator now measures: 2 240 →
**1 176 bytes per element, 1.905×**, confirmed on a live 2 048-element operator
(2.297 MiB where the dense layout took 4.375 MiB).

`A` is stored as `SVector{NV(2NV+1)}` holding the upper triangle column by
column — 105 entries where the dense block held 196. `_symmetric_matvec` applies
it without rebuilding it, reading each stored entry twice rather than storing it
twice, so the apply performs the same `N²` multiply-adds as the dense product.
`_unpack_symmetric` exists for diagnostics and tests, not for the solve.

Storage dispatches on the array's element type, so the plastic path keeps whole
blocks and transposes on apply with no branch in the kernel. `‖A − Aᵀ‖/‖A‖` joins
`‖C − Bᵀ‖/‖C‖` in the assembly-time check, both through
`_assert_frozen_symmetry`, whose message names which identity failed.

Regressions: the packed block round-trips to a symmetric matrix, applying the
packed form matches applying the unpacked one, every element agrees with the
dense block the forward velocity operator builds independently, and — the real
gate — the adjoint residual still matches the Enzyme transpose to 10⁻¹² and the
finite-difference gradient check stays green.

### A3 — stop allocating the Enzyme-only scratch on the frozen path

Nine of the eighteen nodal scratch arrays in `solve_stokes_adjoint_dyrel!` serve
only the Enzyme fallback and are dead when `frozen_operator = true`, the default.

**Result**

Landed. The nine are grouped into an `enzyme_scratch` named tuple built only when
no frozen operator was assembled, and `assemble_adjoint_residual_enzyme!` takes
it as an argument rather than closing over nine separate bindings. Four are
velocity-node sized and five pressure-node sized, so on the default path a
64×64 T7/P1-disc adjoint solve stops allocating **1.78 MiB** it never reads.

The fallback path pays one dynamic dispatch per residual evaluation, since the
scratch is `Nothing` or a named tuple. That path already runs three Enzyme
reverse sweeps per evaluation, so the dispatch is not measurable against it.

### A4 — matrix-free symmetric applies

When `A` is symmetric, `Aᵀλ` is a forward-mode directional derivative, which
ForwardDiff supplies at zero per-element storage. This is a far cheaper fallback
than the current Enzyme reverse path and makes a no-cached-operator mode
practical for the largest cases.

**Result**

Landed as `MatrixFreeAdjointOperator`, selected by `operator = :matrix_free`.
The operator holds the forward state plus three reference tables — **1 344 bytes
total, independent of the element count** — where the frozen blocks cost 1 176
bytes *per element*: 5.17 MiB at 4 608 elements, gone.

The apply is three seeded residual evaluations per element, fused so that two
chains produce all three products:

1. Seed `λv` into `(vx, vy)` and evaluate the pressure residual. Its partial is
   `C·λv`, which is `resp` outright.
2. Scale that dual through `Pnum = γ_eff·RP/M_P` and hand it to the momentum
   residual, still seeded in velocity. Its partial is `A·λv`, augmentation
   included — the augmented residual forms `Pnum` inline from the same chain, so
   feeding the seeded `Pnum` forward reproduces exactly what the block assembler
   differentiates.
3. Seed `λP` into `P` with `Pnum` held at zero. Its partial is `B·λP`.

Inverting `pressure_scale` to recover `C·λv` from the `Pnum` partial was
rejected: it divides by `γ_eff`, which may be zero.

A single fixed `ForwardDiff.Dual` tag is safe here because no nesting can occur —
the plastic return map is the only inner differentiation in the momentum
residual, and a plastic model is rejected at construction. Both `η` conventions
are replicated verbatim from the block assembler: element-max viscosity for `A`,
the full per-phase tuple for `B`.

Measured on a 48×48 structured T7/P1-disc mesh (4 608 elements), single-threaded
CPU:

| | Storage | One apply | Whole adjoint solve | Iterations |
|:---|---:|---:|---:|---:|
| `:blocks` | 1 176 B/element | 0.43 ms | 3.1 s | 5 800 |
| `:matrix_free` | 0 B/element | 5.03 ms | 31.1 s | 5 800 |
| `:enzyme` | 0 B/element | — | 131.7 s | 6 800 |

So the trade is about 12× the arithmetic per apply against the whole per-element
footprint. That is the right trade only when the blocks do not fit; `:blocks`
stays the default.

Against the other storage-free option the result is one-sided: **4.2× faster than
`:enzyme` end to end**, and in the same iteration count as `:blocks` rather than
`:enzyme`'s 6 800, because power iteration needs only an apply and so measures
λmax instead of falling back on the Gershgorin bound. `:enzyme` is now worth
selecting only for a plastic tangent with no room for blocks.

The tables go through `quadrature_table`, not as tuples. This kernel launches on
every adjoint iteration, so a tuple table would be rebuilt into the argument pack
each launch — the F8 lesson, and the reason the block assembler (launched once)
can still pass tuples.

Correctness rests on `A = Aᵀ` and `C = Bᵀ`, the same identities A1 and A2 pack
against, since the apply substitutes forward-mode products for transposed ones.
With no blocks to inspect per element, `_assert_matrix_free_symmetry` checks the
assembled operator instead: two applies on deterministic probes, comparing
`⟨u, Mw⟩` against `⟨Mu, w⟩` for `M = [A B; C 0]`, relative to `sqrt(eps)`. Like
the frozen assertions it cannot fire through user error and exists to catch the
tangent itself changing.

Regressions: the matrix-free apply matches the Enzyme transpose to 10⁻¹² on the
existing 3×3 fixture; power iteration on it measures the same λmax as on the
frozen blocks; the state carries only forward-state aliases and quadrature-sized
tables; the tables are backend arrays; a plastic model is rejected; and a full
adjoint solve driven by it reaches the same λ field as `:blocks`.

**API change.** The `frozen_operator::Bool` keyword is replaced by
`operator = :blocks | :matrix_free | :enzyme`. Three modes do not fit a Bool, and
an unrecognized symbol raises an `ArgumentError` naming the valid ones, so a
caller still passing the old keyword fails loudly rather than silently taking the
default path.

### A5 — reuse frozen forward buffers

Lowest priority: it trades storage for a lifetime coupling between the forward
and adjoint states that would have to be documented and enforced.

**Result**

Not taken. Measured at 8 192 elements (64×64 T7/P1-disc, 24 833 velocity nodes,
24 576 pressure nodes):

| | MiB |
|:---|---:|
| forward buffers dead during the adjoint (`∂vx∂τ`, `∂vy∂τ`, `Rv_x`, `Rv_x0`, `Rv_y`, `Rv_y0`) | 1.14 |
| adjoint velocity-node scratch, 8 arrays | 1.52 |
| adjoint pressure-node scratch, 1 array | 0.19 |
| frozen blocks, packed `A` + `B` | 9.19 |

Six dead velocity buffers cover eight of the adjoint's, so the ceiling is 1.33
MiB — 14 % of the block storage A1 and A2 already halved, and A4 removes that
storage outright rather than trimming it.

What the saving costs is worse than its size. After an adjoint solve `dr.Rv_x`
would hold the adjoint velocity residual and `dr.RP` the adjoint pressure
residual, under names that say otherwise. Nothing raises; forward diagnostics
read afterwards are simply wrong. That is the failure mode the assertions
throughout Phase A exist to prevent, bought here for a fraction of what A4
already delivers.

The disjointness itself is fine — the six are untouched by both operators, and
`vx`, `vy`, `P`, `P0`, `T`, `T0`, `M_P`, `PC_vx`, `PC_vy`, `∂Rv_x∂vx` and
`∂Rv_y∂vy` are all live and excluded. The objection is to the aliasing being
invisible at the point of use, not to its safety.

An explicit adjoint workspace argument, in the manner of `Stokes3DWorkspace`,
would remove the same repeated allocation across an optimization loop with no
hidden aliasing at all. That is an API addition rather than a packing, and is not
part of this plan.

### Phase A so far

| | Bytes per element | Ratio |
|:---|---:|---:|
| dense `A` + `B` + `C` | 2 240 | — |
| packed `A` + `B` (viscous) | **1 176** | **1.905×** |
| matrix-free (viscous) | **0** | — |

At 8 192 elements the frozen operator goes from 17.5 to 9.2 MiB, and A3 removes a
further 1.78 MiB of scratch the frozen path never reads. The matrix-free operator
removes the remaining 9.2 MiB for about 10× the solve time. The plastic path is
unchanged in both storage and arithmetic.

**Gate for all of Phase A:** the finite-difference gradient check stays green on
both the viscous and the plastic case; adjoint iteration counts unchanged; the
plastic path still takes the full three-block route; and each packing carries an
assembly-time assertion of the identity it relies on, so a case that violates it
fails loudly instead of returning a wrong gradient.

**Result**

Gate met. A1–A4 are landed; A5 is not taken. The suite is green at 2 340 tests,
including the finite-difference gradient check on both the viscous and the
plastic case.

The default `:blocks` path halves the per-element operator, 2 240 → 1 176 bytes,
and stops allocating 1.78 MiB of Enzyme-only scratch it never reads. The plastic
path keeps whole blocks and unchanged arithmetic, and every identity the packings
rely on is asserted where it is used rather than assumed.

`:matrix_free` gives the largest cases a third option that stores nothing per
element and still converges in the block path's iteration count, at about 10×
its wall time and a quarter of `:enzyme`'s. The keyword is now
`operator = :blocks | :matrix_free | :enzyme`, replacing `frozen_operator::Bool`.

## 5. Measurement notes

- Benchmark from **one warm session**, after a discarded warm-up run.
- Footprints are payload bytes (`sizeof(eltype) * length`), not
  `Base.summarysize`, so host and device runs are directly comparable and array
  headers are not counted as solver state.
- Host allocation volume is measured with `@allocated` around a complete solve.
  On a device backend this counts host allocations **only**, and therefore misses
  device temporaries entirely — exactly the quantity F3 removes. Read the harness
  allocation column as a CPU-backend measurement, and reach for `CUDA.@allocated`
  around a specific call when the question is device pool churn.
- The 3-D probe measures mesh geometry only. A Hex27 forward solve is too
  expensive to run on every comparison, and geometry is the term that dominates
  a 3-D footprint.
- Repeated solves inside one case must start from the same iterate. The harness
  restores velocity, pressure, rates and residuals between them; without that,
  the timed and the allocation-measured runs take different Powell-Hestenes paths
  and rebuild the Jacobian a different number of times.
- A CUDA `Out of GPU memory` raised from `cuLaunchKernelEx` while the device is
  mostly free is a local-memory reservation, not an allocation. Read it as: the
  per-thread frame is too large. Divide the reported reservation by the block
  size to get bytes per thread and compare against what the kernel stores per
  element.

## 6. Follow-on — reusable 2-D adjoint scratch

The explicit workspace suggested after A5 is now implemented as
`StokesAdjointWorkspace`. It removes the repeated allocation of the nine common
mesh-sized adjoint buffers and two boundary buffers without aliasing forward
state. The nine additional reverse-mode buffers remain opt-in with
`enzyme=true`, so block and matrix-free solves preserve A3's saving.

The solver validates velocity/pressure axes and boundary-node counts before
using a workspace. Residual and pullback scratch is overwritten by the residual
paths; the DYREL rates carry momentum between iterations, so the solver zeroes
them at the start of every solve. A workspace can therefore survive across
design iterations: two cold solves through the same workspace take the same
iteration count and agree to roundoff (tested with `rtol = 1e-12`). They are
bitwise identical only under one thread; with more, threaded assembly sums
element contributions in scheduling order and the last digits move.

## 7. Re-measurement after the container and mesh refactors

Two refactors landed after the baseline: stress and velocity state moved into
`VectorField2D`/`SymmetricTensor2D` containers, and the precomputed geometry
moved into `MixedMesh.geometry`. The harness now counts the containers
(`STRESS_FIELDS = (:τ, :τ_old)`) and reads geometry from the mesh. Measured
against the recorded baseline, Julia 1.12.7, six threads, CPU:

| | Stress | Total | B/element |
|:---|---:|---:|---:|
| baseline | 2.625 MiB | 10.658 MiB | 1 364.3 |
| containers, full `II` slots | 3.500 | 11.533 | 1 476.3 |
| `II` slots zero-length | 2.625 | 10.658 | **1 364.3** |

`SymmetricTensor2D` carries an invariant slot `II` beside the three components,
and `StokesDR` allocated it at full stress size in both `τ` and `τ_old`: two
arrays, 112 B/element in 2-D, that no solver path reads. `StokesDR` now
allocates `II` as a zero-length array of the component type, so the container
type is unchanged. In 3-D the saving is one of seven arrays per
tensor.

Everything else is unchanged: geometry, nodal state, phase tags, topology,
normals, and host allocation per iteration all measure 1.00× the baseline, so
moving the cache into `MixedMesh` cost nothing.

The 3-D geometry probe also settles how far F1 moved the 3-D limit. Hex27
geometry is 2 160 B/element against the 17 712 of §2, so a 64³ Hex27 mesh holds
0.53 GiB of geometry rather than 4.32 GiB.

