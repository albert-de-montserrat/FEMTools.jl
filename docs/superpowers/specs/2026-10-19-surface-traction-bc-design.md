# Surface traction and pressure boundary conditions (2-D)

## Intent

Stokes models in FEMTools currently support only Dirichlet velocity
conditions; every other boundary is traction-free. Geodynamic models need a
prescribed load on a free surface: an ice cap, an ash or sediment pile, or a
lake/sea column on a volcano's topography. The load varies along the
topography, may be constant, and may change between time steps (e.g. an ice
cap thinning over kyr). The first consumer is
`examples/stokes/volcano/volcano_thermal_stokes.jl`.

## Goals

- A generic, public, 2-D API for natural (Neumann) momentum conditions on a
  set of boundary edges:
  - `TractionBoundaryCondition`: prescribed traction vector `t(x)`.
  - `PressureBoundaryCondition`: prescribed normal pressure `p(x)`, i.e.
    `t = −p n` with `n` the outward unit normal.
- Constant and position-dependent loads through one constructor: a `Number`
  (or `SVector` for traction) means constant, a callable means `x -> value`.
- Loads that change between time steps without re-searching the mesh.
- Correct on curved (quadratic, T6/T7 and Q8/Q9) edges.
- Backend-neutral: the load assembly runs as a KernelAbstractions kernel.
- Wire into `solve_stokes_dyrel!` and `solve_coupled_dyrel!` (2-D).
- Use it in the volcano script with a topography-dependent load.

## Non-goals

- 3-D surfaces (passing a traction condition to a 3-D solve throws
  `ArgumentError("traction boundary conditions are implemented only in 2-D")`).
- Follower loads (traction direction updated with the velocity during a
  solve), traction-dependent Jacobians, surface-tension or stabilization terms
  (e.g. free-surface stabilization / FSSA).
- Adjoint sensitivities with respect to the load.
- Thermal (heat-flux) Neumann conditions, although the facet machinery should
  not preclude them.
- Explicit `time` argument plumbed through solvers.

## Anti-goals

- A silent no-op: a condition whose node set selects zero edges must throw,
  not quietly apply nothing.
- Wrong-sign or inward-normal loads. Normal orientation must be derived and
  tested, not assumed.
- Per-solver-iteration host↔device transfers or host scalar indexing inside the
  relaxation loop.

## Mathematics

The momentum residual in `src/stokes/assemblers/momentum_residuals.jl` is

    Rᵢ = ∫ ∇Nᵢ · σ dΩ − ∫ Nᵢ ρ g dΩ,    σ = τ − P I,

and the solver drives `R → 0`. With traction `t` on Γ_N the weak form gains
`−∫_Γ Nᵢ t ds`, so the solver uses `R − F` with the nodal load

    Fᵢ = ∫_Γ Nᵢ t ds,             (traction)
    Fᵢ = −∫_Γ Nᵢ p n ds.          (pressure, t = −p n)

The load is a dead load during one solve: it does not depend on velocity or
pressure, so the momentum Jacobian, Gershgorin bounds, preconditioner, and
pressure scaling are unchanged.

Edge integration: each boundary edge of a quadratic element has three nodes
(two corners, one midpoint; ordering from `_boundary_edge_paths_2d`). Use the
1-D quadratic Lagrange basis on s ∈ [−1, 1], `x(s) = Σ Nₐ(s) xₐ`,
`ds = |x'(s)| ds`, tangent `x'(s)`, normal from the tangent rotated outward.
Linear edges (T3/Q4) use the 2-node basis. Gauss–Legendre with 3 points per
edge (exact for quadratic `p` on straight quadratic edges); the rule is a
compile-time constant.

Outward orientation: rotate the tangent by −90° and flip if it points toward
the element centroid (same rule as `_compute_node_normals` in
`src/mesh/mixed_mesh.jl`). Evaluated per edge inside the kernel from the
element's corner coordinates.

Shared corners with Dirichlet nodes: the load is added before Dirichlet
zeroing of `Rv`, so Dirichlet wins at any node constrained in that component.

## API

```julia
# Geometry: a set of boundary edges, built once on the host, stored on backend.
surface = boundary_facets(mesh, nodes)          # nodes: AbstractVector of node ids
surface = boundary_facets(mesh, predicate)      # predicate(x::SVector) -> Bool, true on all edge nodes

# Conditions: cheap to rebuild; hold `facets` plus the load callable.
bc = PressureBoundaryCondition(surface, 1.0e5)                    # constant
bc = PressureBoundaryCondition(surface, x -> ρ_ice * g * h(x[1])) # varies along topography
bc = TractionBoundaryCondition(surface, x -> SVector(τ_wind, 0.0))

# Solver: one condition or a tuple.
solve_stokes_dyrel!(dr, mesh, bc_vx, bc_vy, Δt, γP; traction = bc, ...)
solve_coupled_dyrel!(thermal, dr, mesh_T, mesh, bc_T, bc_vx, bc_vy, Δt, γP;
                     traction = (bc_ice, bc_lake), ...)

# Low level (public, for custom loops and tests):
F = assemble_boundary_load(bc_or_tuple, mesh)        # NTuple{2} of nodal vectors
assemble_boundary_load!(F, bc_or_tuple, mesh)        # accumulates into F
```

Types:

- `BoundaryFacets{I, V}` — fields `elements::V` (element id per edge,
  `Int32`) and `local_edges::V` (local edge index into
  `_boundary_edge_paths_2d`), both on the mesh backend. Selection: only true
  boundary edges (corner pair belongs to exactly one element) are candidates;
  an edge is selected when *all* its nodes are in the node set (or satisfy the
  predicate). A side-wall edge touching a surface corner node is therefore
  excluded. An empty selection throws
  `ArgumentError("no boundary edges have all their nodes in the given set")`.
  Works for `Mesh` and `MixedMesh` (uses the velocity connectivity
  `mesh.el2n`). Meshes have one element type, so one `local_edges` numbering
  applies. *(Builder's calls, settled during review.)*
- Facets store topology only. Coordinate changes (deformed mesh,
  `update_geometry!`) keep them valid, since `F` is assembled from current
  coordinates each solve; a remesh requires rebuilding them.
- Accepted load forms: `PressureBoundaryCondition` takes a `Number` or a
  callable returning a `Number`; `TractionBoundaryCondition` takes an
  `SVector{2}` (an `NTuple{2}` is converted) or a callable returning one.
  Other inputs throw `ArgumentError`.
- Units: the callable receives coordinates in the mesh's units and returns a
  value in the solver's stress units. No conversion happens inside the
  package; a scaled model (like the volcano script) scales inside its callable.
- Sign: `p` has the sign of the solver pressure `P` (compressive positive,
  `σ = τ − P I`), so a load `p` on a static column raises `P` by `p`.
- `TractionBoundaryCondition{F, L} <: AbstractBoundaryCondition` — fields
  `facets::F`, `traction::L`. Replaces the current unused placeholder
  `TractionBoundaryCondition(Γ, DoFs, vals)`, which no code, test, or doc uses.
- `PressureBoundaryCondition{F, L} <: AbstractBoundaryCondition` — fields
  `facets::F`, `pressure::L`.
- Constant loads are stored as `Returns(value)`, so the kernel always calls
  `load(x)`. User callables must be GPU-compatible (isbits closures) to run on
  a GPU backend; CPU accepts any callable. Document this.

Exports: `boundary_facets`, `BoundaryFacets`, `TractionBoundaryCondition`,
`PressureBoundaryCondition`. `assemble_boundary_load[!]` public, unexported.

Time variation: rebuild the condition each step from the cached facets,

```julia
surface = boundary_facets(mesh_stokes, groups.surface)
for istep in 1:nsteps
    bc_ice = PressureBoundaryCondition(surface, x -> ice_load(x, istep * Δt))
    solve_coupled_dyrel!(...; traction = bc_ice)
end
```

## Solver integration

In the low-level 2-D `solve_stokes_dyrel!` (`src/stokes/solvers/DR.jl`), add
keyword `traction = nothing`. If non-`nothing`: allocate `F` (D backend
vectors, `similar(first(Rv))`) once per call, `assemble_boundary_load!` from
the current `mesh_stokes.coords`, and after every
`assemble_momentum_residual_kernel!` (outer and inner loop) subtract `F`
componentwise before `_apply_dirichlet_all!`. The outer-loop residual
norms therefore include the load. The high-level wrappers forward the keyword
through `kwargs...`; `solve_coupled_dyrel!` already forwards. 3-D entry points
throw if `traction` is given.

`update_stokes_current_stress!` and post-processing are unaffected (they do
not form the residual).

## Volcano script

- Build `surface = boundary_facets(mesh_stokes, groups.surface)` once.
- Load: ice cap on the cone, `p(x) = ρ_ice g h_ice(x)`, `h_ice` maximal at the
  summit and tapering to zero at a chosen radius (default ~1 km thick over the
  cone, scaled to characteristic units). Keyword `ice_thickness` (0 disables)
  and `ice_radius`. Optionally a thinning rate to show time variation
  (rebuild per step). Specific numbers are the builder's call; document them.
- Add the load to the pressure warm start: at a pressure node with abscissa
  `x`, `P_init += p((x, topography(x)))`, i.e. the surface load directly
  above, extended down the whole column (matches the static solution
  `P = p + ρ g (h − y)`). The relaxation starts from the overburden.
  *(Builder's call.)*
- Update the docstring: topography is no longer traction-free where loaded.

## Verification

1. Unit: `boundary_facets` on a structured T7/Q9 box picks exactly the top
   edges; empty selection and interior edges throw (`@test_throws` with
   message).
2. Unit: `assemble_boundary_load` with constant `p` on the top of a unit box
   gives `ΣF_y = −p·L`, `ΣF_x = 0`; with `p(x) = x` gives `ΣF_y = −L²/2`;
   on a curved (circular arc) quadratic boundary, `Σ F ≈ −p ∮ n ds` and
   normals point outward. Sign test on a box: top normal = +y.
3. Physics: box, no gravity, free-slip sides and base, uniform `p` on top:
   converged pressure ≈ `p` everywhere, velocity ≈ 0. With gravity:
   `P = p + ρ g (H − y)`.
4. Allocation/inference tests if the existing solver tests check them.
5. Manual (not CI): run the volcano script with and without ice load; check
   the mean pressure change in a shallow band under the summit is within ~10%
   of `ρ_ice g h` and the solve still converges. CI coverage comes from items
   1–3. *(Builder's call.)*
6. Docs: `docs/src/boundary_conditions.md` section + `api.md` entries;
   update `.agents/API.md` and `.agents/solver.md`.

## Left to the builder

- Internal file layout (suggest `src/boundary_conditions/traction.jl` and
  facet extraction in `src/mesh/`).
- Kernel scatter details (Atomix atomics, as in the momentum kernel).
- Exact ice-cap profile and parameter defaults in the script.
- Whether `boundary_facets` predicate form tests corners only or all edge
  nodes (all nodes preferred).
