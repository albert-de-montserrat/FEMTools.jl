# Surface Traction and Pressure BC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prescribed 2-D surface tractions and pressures for the Stokes DR solvers, used on the volcano topography.

**Architecture:** `boundary_facets` builds a topology-only edge list (element id + local edge) on the backend. `TractionBoundaryCondition` / `PressureBoundaryCondition` pair facets with a load callable. A KernelAbstractions kernel integrates `F = ∫ N t ds` on curved edges. `solve_stokes_dyrel!` subtracts `F` from `Rv` after every momentum assembly, before Dirichlet zeroing.

**Tech Stack:** Julia, KernelAbstractions, Atomix, StaticArrays.

**Spec:** `docs/superpowers/specs/2026-10-19-surface-traction-bc-design.md`

## Global Constraints

- 2-D only; 3-D entry points given `traction` throw `ArgumentError("traction boundary conditions are implemented only in 2-D")`.
- Residual sign: solver drives `R = ∫∇N·σ − ∫Nρg − F → 0`, `F = ∫_Γ N t ds`, pressure `t = −p n`, `n` outward unit normal.
- Constant loads stored as `Returns(value)`; kernel always calls `load(x)`.
- `p` has the sign of solver `P` (compressive positive). No unit conversion in the package.
- Empty selection throws `ArgumentError("no boundary edges have all their nodes in the given set")`.
- Edge quadrature: 3-point Gauss–Legendre on s ∈ [−1, 1] (nodes `0, ±√(3/5)`, weights `8/9, 5/9`).
- Quadratic edge path order is `(corner_a, midpoint, corner_b)` (from `_boundary_edge_paths_2d`); basis `s(s−1)/2, 1−s², s(s+1)/2`. Linear edge `(a, b)`: `(1−s)/2, (1+s)/2`.
- No host scalar indexing in the relaxation loop; `F` allocated once per solve.
- Commits end with `Assisted-by: Claude Opus 5.5 <noreply@anthropic.com>`; subject ≤ 50 chars.
- Targeted test runs: `julia --project=. -e 'using TestEnv; TestEnv.activate(); include("test/test_traction_bc.jl")'`.

## Review Focus

- Dirichlet-shared corner nodes (surface corner with fixed `vx`): Dirichlet must win in that component; load still applies in the free one. Test in Task 3.
- Curved quadratic edges with off-center midpoint: total load must equal `−p ∮ n ds` of the true curve, not the chord. Test in Task 2.
- Normal orientation: must point outward on every side, independent of edge traversal direction. Task 2 tests top and bottom of a box (opposite signs).
- `Float32` meshes: `F` must stay `Float32`. Test in Task 2.
- Node set touching edges only at a corner (side wall sharing a surface corner): side-wall edge excluded. Test in Task 1.

---

### Task 1: Boundary facets

**Files:**
- Create: `src/mesh/facets.jl` (include after `mesh/mixed_mesh.jl` in `src/FEMTools.jl`)
- Modify: `src/mesh/mesh.jl:216` (add `Val` methods of `_boundary_edge_paths_2d` returning the same tuples, for compile-time use in kernels)
- Test: `test/test_traction_bc.jl` (new)

**Interfaces:**
- Produces: `struct BoundaryFacets{V}; elements::V; local_edges::V; end` (both `Int32` vectors on mesh backend); `boundary_facets(mesh::Union{Mesh{2}, MixedMesh{2}}, nodes::AbstractVector{<:Integer}) -> BoundaryFacets`; `boundary_facets(mesh, predicate) -> BoundaryFacets` (predicate on `SVector` coordinates, all edge nodes must satisfy); `Base.length(f::BoundaryFacets)`; `_boundary_edge_paths_2d(::Val{N})`.

- [ ] **Step 1: Write failing tests** in `@testset "boundary facets"` on `Mesh(CPU(), (0.0 .. 2.0) × (0.0 .. 1.0), ReferenceElement(QuadraticElement{2, 6, Float64}), (4, 2))`:
  - `top = boundary_facets(mesh, x -> x[2] ≈ 1.0)`; `@test length(top) == 4` (structured T6 splits each quad into 2 triangles; adjust expected count to the number of top edges = nx).
  - node-set form on the same top nodes returns the same `(elements, local_edges)` pairs (as sets).
  - top node set plus the top-right corner only: no right-wall edge selected (`length == 4`).
  - `@test_throws "no boundary edges have all their nodes" boundary_facets(mesh, x -> false)`.
  - interior-only predicate (`x -> 0.2 < x[2] < 0.8`) throws the same message.
  - `MixedMesh(mesh, ReferenceElement(LinearElement{2, 3, Float64}))` gives the same facets.
- [ ] **Step 2: Run, expect FAIL** (`UndefVarError: boundary_facets`).
- [ ] **Step 3: Implement** on host: count corner-pair keys as in `_unstructured_boundary_nodes`, keep edges with count 1 and all path nodes selected, copy to backend via `KA.allocate` + `copyto!`. Export `BoundaryFacets`, `boundary_facets`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `Add boundary_facets edge selection`.

### Task 2: Traction/pressure conditions and load assembly

**Files:**
- Create: `src/boundary_conditions/traction.jl` (include after `boundary_conditions/apply.jl`)
- Modify: `src/boundary_conditions/boundary_conditions.jl:52-66` (replace placeholder `TractionBoundaryCondition`)
- Modify: `src/FEMTools.jl` (export `TractionBoundaryCondition, PressureBoundaryCondition`; `public assemble_boundary_load, assemble_boundary_load!`), `test/test_exports.jl`
- Test: `test/test_traction_bc.jl`

**Interfaces:**
- Consumes: `BoundaryFacets`, `_boundary_edge_paths_2d(::Val{N})`.
- Produces: `TractionBoundaryCondition(facets::BoundaryFacets, traction)` (fields `facets`, `traction`; `SVector{2}`/`NTuple{2}` → `Returns(SVector(...))`, callable kept, else `ArgumentError`); `PressureBoundaryCondition(facets, pressure)` (fields `facets`, `pressure`; `Number` → `Returns`); `assemble_boundary_load!(F::NTuple{2, <:AbstractVector}, bcs, mesh; workgroup = 256) -> nothing` accumulates; `assemble_boundary_load(bcs, mesh; workgroup = 256) -> NTuple{2}` zero-initialised `similar` of coordinate eltype; `bcs` is one condition or a `Tuple` of them.

- [ ] **Step 1: Write failing tests** in `@testset "boundary load"`, unit box `L = 2`, T6 `(4, 2)`:
  - constant `p = 3.0` on top: `sum(F[2]) ≈ -3.0 * 2` , `abs(sum(F[1])) < 1e-12`.
  - `p = x -> x[1]` on top: `sum(F[2]) ≈ -2.0` (−∫₀² x dx).
  - same `p` on bottom (`x[2] ≈ 0`): `sum(F[2]) ≈ +3.0 * 2` (outward normal −y).
  - traction `SVector(1.0, 0.0)` on top: `sum(F[1]) ≈ 2.0`; `NTuple` form equal.
  - nodal check: constant `p` on one quadratic straight edge of length `h` gives corner/mid/corner loads `−p h .* (1/6, 2/3, 1/6)` in y.
  - curved: copy `coords`, move the midpoint node of one top edge up by `δ = 0.1`, rebuild with `Mesh(CPU(), coords, el2n, element)`. Because `−p ∫ n ds = −p (Δy, −Δx)` rotated depends only on endpoints, assert `sum(F[2]) ≈ -p * 2` and `abs(sum(F[1])) < 1e-12`, and that the bumped edge's midpoint load differs from the flat case (`F[1]` at the two corners of that edge have opposite signs).
  - tilted: move the top-right corner up by `0.2` (and its edge midpoint to the chord midpoint): `sum(F[1]) ≈ p * 0.2`.
  - `Float32` mesh: `eltype(F[1]) == Float32`.
  - `@test_throws "traction must be" TractionBoundaryCondition(top, 1.0)`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** kernel `boundary_load_kernel!(F, coords, el2n, elements, local_edges, load_traction, ::Val{NV})` with one work-item per facet: gather edge path nodes, loop over 3 Gauss points, `x(s)`, `x'(s)`, `ds = |x'|`, `n = (x'_y, −x'_x)/|x'|` flipped if `dot(n, centroid − x) > 0` (centroid of corner nodes `first.(paths)`), `t = _traction(bc, x, n)` (`TractionBoundaryCondition`: `bc.traction(x)`; `PressureBoundaryCondition`: `-bc.pressure(x) * n`), `Atomix.@atomic F[c][node] += N_a * t[c] * w * ds`. Launch via existing `launch!`.
- [ ] **Step 4: Run, expect PASS.** Also run `test/test_exports.jl`.
- [ ] **Step 5: Commit** `Add traction and pressure boundary loads`.

### Task 3: Solver integration

**Files:**
- Modify: `src/stokes/solvers/DR.jl` (low-level `solve_stokes_dyrel!` near line 278; both `assemble_momentum_residual_kernel!` call sites; docstring keyword list; 3-D `solve_stokes_3d!`/`solve_stokes_adjoint_3d!` guards)
- Test: `test/test_traction_bc.jl`

**Interfaces:**
- Consumes: `assemble_boundary_load!`, `assemble_boundary_load`.
- Produces: keyword `traction = nothing` on `solve_stokes_dyrel!` (all forms, via `kwargs...`) and `solve_coupled_dyrel!`.

- [ ] **Step 1: Write failing tests** in `@testset "stokes under surface pressure"`: T6 `(4, 4)` unit box, `StokesMaterial(; η = (1.0,), ηb = (1.0,), G = (Inf,), α = (0.0,))`, `g = (0, 0)`; free slip: `vx = 0` on left/right, `vy = 0` on bottom (follow `test/test_coupled_solver.jl` setup); `bc = PressureBoundaryCondition(boundary_facets(mesh, x -> x[2] ≈ 1), 2.0)`; solve with `traction = bc`, `ϵ_tol = 1e-8`:
  - `@test stats.converged`; `@test all(≈(2.0; rtol = 1e-5), stokes.P)`; `@test maximum(abs, stokes.v.y) < 1e-6`.
  - Dirichlet-shared corner: corner nodes on top with `vx` fixed keep `vx == 0` exactly.
  - Gravity variant `ρ0 = (1.0,)`, `g = (0.0, -1.0)`: P at pressure DoFs ≈ `2 + (1 − y)` (`rtol = 1e-3`, P1-disc is exact for linear P).
  - `traction = (bc_half1, bc_half2)` splitting the top in two equals the single-bc result.
  - `@test_throws "implemented only in 2-D"` on `solve_stokes_3d!(...; traction = bc)` using the smallest existing 3-D test setup (see `test/test_stokes_3d_reference.jl`).
- [ ] **Step 2: Run, expect FAIL** (`P ≈ 0`, or unsupported keyword).
- [ ] **Step 3: Implement**: if `traction !== nothing`, `F = assemble_boundary_load(traction, mesh_stokes; workgroup)` once before the loop; after each momentum assembly `foreach((R, f) -> R .-= f, Rv, F)` before `_apply_dirichlet_all!`. 3-D entry points: `traction = nothing` keyword, throw if not `nothing`.
- [ ] **Step 4: Run, expect PASS.** Then run `test/test_stokes.jl`, `test/test_coupled_solver.jl`, `test/test_allocations.jl`, `test/test_type_stability.jl`.
- [ ] **Step 5: Commit** `Apply traction loads in Stokes DR solver`.

### Task 4: Docs and agent guides

**Files:**
- Modify: `docs/src/boundary_conditions.md` (section "Traction and pressure" with `@docs` blocks for `BoundaryFacets`, `boundary_facets`, `TractionBoundaryCondition`, `PressureBoundaryCondition`, `FEMTools.assemble_boundary_load`, `FEMTools.assemble_boundary_load!`, and the time-varying rebuild example from the spec), `docs/src/api.md` if it lists exports, `.agents/API.md` (exported surface), `.agents/solver.md` (residual sign + `traction` keyword).
- [ ] **Step 1:** Write docstrings (signature line, one-sentence summary, GPU-callable note, units note) per `.claude/rules/docstring-rules.md`.
- [ ] **Step 2:** `julia --project=docs docs/make.jl` builds with no missing-docstring warnings for the new names.
- [ ] **Step 3: Commit** `Document traction boundary conditions`.

### Task 5: Volcano ice-cap load

**Files:**
- Modify: `examples/stokes/volcano/volcano_thermal_stokes.jl`

- [ ] **Step 1:** Keywords `ice_thickness = 1.0e3` [m], `ice_radius = 8.0e3` [m], `ice_melt_time = 50kyr` (linear thinning to zero; `Inf` keeps it constant), `ρ_ice = 917.0`. SI load `p(x, t) = ρ_ice * 9.81 * h0 * max(0, 1 − t/ice_melt_time) * max(0, 1 − (x/ice_radius)^2)`; scaled callable `x -> p(x[1] * L_c, t) / σ_c`. `ice_thickness = 0` skips the condition (`traction = nothing`).
- [ ] **Step 2:** `surface = boundary_facets(mesh_stokes, groups.surface)` once; per step build `PressureBoundaryCondition(surface, ...)` at the step's `t`, pass `traction` to `solve_coupled_dyrel!`. Pressure warm start adds the load at `(x, topography(x))`.
- [ ] **Step 3:** Update docstring (cone carries an ice load, thinning over `ice_melt_time`; rest of topography traction-free) and BC comment block.
- [ ] **Step 4 (manual):** run `main(nsteps = 2, show_plot = false, write_output = false)` with and without ice; record convergence and the mean P change in the band `|x| < 1 km`, `topography − 1 km < y < topography`; expect within ~10% of `ρ_ice g h` at step 1.
- [ ] **Step 5: Commit** `Load volcano cone with a thinning ice cap`.
