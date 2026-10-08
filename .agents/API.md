# API

## Purpose

This guide defines the intended public surface of FEMTools.jl and how to evolve
it. It is an engineering contract, not a substitute for public docstrings or
the Documenter manual. The module declaration in `src/FEMTools.jl`, executable
API tests, and method implementations remain the source of truth.

Read [solver.md](solver.md) for the mathematical contract behind solver entry
points and [meshing.md](meshing.md) for topology/layout invariants.

## API tiers

FEMTools uses three tiers:

1. **Exported API** — names brought into scope by `using FEMTools`. This is the
   normal user surface and receives the strongest compatibility treatment.
2. **Public, unexported API** — names declared with `public` and called as
   `FEMTools.name`. These support advanced assembly, extensions, miniapps, and
   testing without crowding the default namespace.
3. **Internal implementation** — all other names, especially underscore-prefixed
   helpers and raw element routines not declared public. These may change with
   their callers and tests.

Moving a name between tiers is an API change. Do not export a helper merely to
avoid qualifying it in one example.

## Exported surface

### Elements and quadrature

- Abstract hierarchy: `AbstractElement`, `AbstractLinearElement`,
  `AbstractQuadraticElement`, `AbstractShapeFunction`,
  `AbstractIntegrationPoints`.
- Element tags: `LinearElement`, `QuadraticElement`, `CubicElement`.
- Containers: `ReferenceElement`, `ShapeFunctions`, `IntegrationPoints`.
- Queries/evaluation: `order`, `eval_shape_function`,
  `eval_shape_function_gradient`, `eval_shape_function_jacobian`,
  `shape_function_values`, `shape_function_gradients`, `quadrature_table`,
  `gauss_legendre_triangle`.

An element tag identifies dimension, local-node count, and numeric type. A
`ReferenceElement` is the user-facing bundle of tag, shape functions, and
quadrature. Not every syntactically constructible tag has a complete
implementation; unsupported combinations must fail explicitly.

### Meshes and boundary conditions

- Types: `AbstractMesh`, `Mesh`, `MixedMesh`, `MixedMeshCache`,
  `AbstractBoundaryCondition`, `DirichletBoundaryCondition`.
- Mesh producers: `generate_element2node`, `generate_node2element`,
  `generate_boundary_elements`, `generate_coordinates`, `generate_dofs`,
  `generate_discontinuous_linear_mesh`.
- Derived data: `precompute_geometry`, `update_geometry!`, `QuadraturePointGeometry`,
  `ElementGeometry`, `element_geometry`, `generate_sparsity_pattern`,
  `color_mesh`, `generate_element_groups`.
- Boundary application: `apply_bc!`.

High-level `Mesh` constructors either build a structured mesh from a domain or
accept validated coordinates/connectivity. Passing a `ReferenceElement`
precomputes solver geometry; omitting it produces a topology-only mesh.
`MixedMesh` owns distinct velocity and pressure layouts and, when built from an
element-aware velocity mesh, a `MixedMeshCache` in `mesh.geometry` holding their
precomputed geometry and reference elements. `update_geometry!(mesh)` recomputes
it in place after the coordinates move.

Precomputed geometry holds one `QuadraturePointGeometry` per element and
quadrature point: the inverse isoparametric Jacobian and the weighted volume.
`MixedMeshCache.geo_P` instead holds the weighted volume alone, since no consumer
takes pressure-field gradients.
Physical shape-function gradients are formed on access by pairing it with the
reference-element gradients through `element_geometry`, so an assembler that
consumes geometry also takes `shape_function_gradients(element)` alongside
`shape_function_values(element)`.

Both tables reach kernels either as the `NTuple`s those accessors return or as
the backend arrays `quadrature_table` builds from them. Tuples travel in the
kernel argument pack, which is rebuilt on every launch, so a solver builds the
arrays once and passes those; the per-call assembler wrappers still take
reference elements and rebuild tuples.

### Field containers

Import `FEMTools.VectorField2D/3D` and `FEMTools.SymmetricTensor2D/3D`
explicitly. Vector components use `.x`, `.y`, and `.z`; stress components
include the invariant `II`. `Tuple` unpacks independent components for
assemblers. `StokesDR` owns `v`, `Rv`, `PC_v`, `τ`, and `τ_old`
containers; its spatial dimension follows the length of `g`. Passing
`plastic_history_size=(nq, nels)` additionally allocates zeroed integration-
point `dr.plastic_history.γ` and `.θ` arrays for the experimental cap path;
the default remains `nothing`.
The revised cap plan targets a once-per-accepted-step commit of corrected
pressure and plastic history, matching JustRelax without GeoParams. This is
pending API/lifecycle work: `update_stokes_current_stress!` currently advances
history on each call, so it is not yet a read-only diagnostic operation.
The `DruckerPragerCap` constructor takes radians, rejects nonfinite material
parameters and nonpositive `Kb`, and requires positive cap radius at both
`C` and `C_min`. The new scalar `cap_invariants` evaluator is internal API.

### Physics states and solvers

- Materials/states: `ThermalMaterial`, `ThermalDiffusionDR`,
  `LithostaticPressureDR`, `StokesMaterial`, `StokesDR`, `CellPressureStokesDR`,
  `StokesAdjointWorkspace`, `DruckerPrager`, `DruckerPragerCap`.
- Scalar entry point: `solve!(dr, mesh, bc; dt, ...)` for thermal diffusion and
  `solve!(dr, mesh, bc; ...)` for lithostatic pressure (`dt` is thermal only).
- Stokes entry points: `solve_stokes_dyrel!` (prepared scale),
  `solve_coupled!`; `solve!`/`solve_adjoint!` for mixed `StokesDR`
  and Hex27 `CellPressureStokesDR`.
- Stokes operations: `assemble_viscosity_weighted_pressure_scaling!`,
  `pressure_mass`, `rotate_stress!`, `update_stokes_current_stress!`,
  `stokes_material_gradient_3d`.

`solve!` mutates a scalar problem state and returns
`(; converged, iterations, residual, history)`; non-convergence throws unless
`throw_on_failure=false`. Stokes solvers mutate caller/state arrays and return a
statistics `NamedTuple` with explicit convergence status. The scalar controls
are `tolerance` (default `dr.ϵ`), `max_iterations`, `check_interval`, `verbose`,
and `collect_history`; the Stokes solvers still use `ϵ_tol`, `iterMax`,
`ncheck` until their entry points migrate.

### Results and output

`compute_principal_stresses` and `compute_principal_stresses!` compute pointwise
2D/3D eigenpairs in an exported `PrincipalStresses` container, preserving
sampling shape, Float32/Float64, and backend. Its `values` and `directions`
fields wrap tuples of arrays; the two-argument constructor borrows buffers
without copying. The in-place method accepts this container, updates its
existing arrays, and returns the same object. Only component tuples are passed
to the kernel, so the result type needs no GPU adaptation dependency.
They accept tensor containers by named fields or tuples in `stress(dr)` order
(not the 3D tensor's `Tuple` order). Results contain descending `values` arrays
and paired `directions[k][c]` arrays. Optional collocated physical pressure
shifts values for `σ = τ - P I`. The 2D operation is in-plane; full plane-strain
eigenpairs use a 3D tensor including `τzz`. Repeated values permit any
orthonormal eigenspace basis. Mutating buffers may not alias inputs or each
other. Both methods synchronize and raise `DomainError` on numerical failure;
failed samples are NaN and output is partially updated. The implementation
uses analytic 2D and scaled StaticArrays 3D eigenpairs with local residual and
orthogonality checks. Its native KA launch supplies the workgroup size at runtime
to keep the kernel type concrete; fixed-size tuples preserve buffer-validation
inference. See [`PRINCIPAL_STRESS_PLAN.md`](../PRINCIPAL_STRESS_PLAN.md).

GPU alias-validation requirement: device wrappers over shared storage need
backend-specific `dataids` so `Base.mightalias` detects overlapping views.
Principal-stress in-place validation relies on this to enforce its no-alias
contract; overlapping outputs/inputs can race or corrupt input fields. CUDA
view-alias validation requires a hardware check.

- Derived fields: `compute_strain_rate_stress_postprocess`,
  `compute_principal_stresses`, `compute_principal_stresses!`,
  `update_old_stress_from_cells!`.
- Writers: `write_vtk`, `write_stokes_vtk`.

Writers are visualization output, not restart/checkpoint APIs. High-order VTK
cells are currently linearized to their corners.

## Public unexported surface

The module currently marks the following categories `public`:

- backend and interpolation: `TA`, `interp2ip`, `interp2ip_phase`;
- low-level Dirichlet application: `apply_dirichlet!`;
- thermal, lithostatic, momentum, pressure, Jacobian, and adjoint assemblers;
- raw KA update/geometry kernels and their Stokes launch wrappers;
- coloring/pressure utilities: `color_mesh_greedy`, `remove_pressure_mean!`;
- external-mesh ingestion helpers in `src/mesh/utils.jl`:
  `renumber_connectivity`, `orient_triangle_elements!`, `add_t7_bubbles!`,
  `straighten_t7_geometry!`, `rectangle_boundary_nodes`,
  `circle_boundary_nodes`, and the `triangulate_t7_mesh` stub whose method the
  Triangulate extension supplies;
- result accessors: `velocity`, `stress`, `pressure`, `temperature`.

These names are callable as `FEMTools.name` but are deliberately not imported
by `using FEMTools`. Before adding another public low-level method, confirm that
an extension, maintained miniapp, or downstream package genuinely needs it.

The 2-D pressure residual assembler uses the solver state's finite elastic bulk
modulus `K` for the experimental `DruckerPragerCap` path. Its element-level integration
helper retains the legacy bulk-viscosity behavior when `K` is omitted, and uses
`K` when supplied for the trial-pressure formulation.

The forward solver makes that choice explicit with `finite_K`: `true` selects
finite positive `K` independently of plastic-model dispatch, while `false`
keeps the legacy `ηb` storage coefficient. `Qq` can be supplied directly as a
signed local source, or paired with nonnegative weights and signed `Q2D` to
enforce the discrete area-rate identity `Σ Q dΩ = Q2D`. `P_old` accepts either
a nodal pressure vector or accepted `nq × nels` integration-point pressure.

The forward Stokes solver accepts `viscosity` as phase values or an
`nq × nels` matrix for purely viscous 2-D flow. Matrix values are validated for
shape, precision, backend, and finite positivity; matrix overrides exclude
plasticity, coupling, and measured spectral estimates. Pressure scaling takes
the same matrix through its existing `η` keyword. The override does not change
the state material used by stress-history or adjoint APIs.
`body_force` adds one finite assembled nodal load vector per velocity component
before residual constraints in both forward loops. It is additive to EOS/gravity
forcing; pointwise force densities must be integrated by the caller.

## Canonical user flows

### Single-field problem

```julia
element = ReferenceElement(LinearElement{2, 3, Float64})
mesh = Mesh(backend, coords, el2n, element)
material = ThermalMaterial(; k, Cp, ρ0, α, K)
dr = ThermalDiffusionDR(mesh, material)
bc = DirichletBoundaryCondition(nodes, values)
stats = solve!(dr, mesh, bc; dt = Δt, workgroup)
T = FEMTools.temperature(dr)
```

Lithostatic pressure follows the same mesh/material/BC shape with
`LithostaticPressureDR` and `solve!(dr, mesh, bc; ...)`.

### Two-dimensional mixed Stokes problem

```julia
mesh_v = Mesh(backend, coords, el2n_v, element_v)
mesh = MixedMesh(mesh_v, element_P; workgroup)
material = StokesMaterial(; η, ηb, G, α, ρ0, K, g, Tref)
dr = StokesDR(backend, mesh.nnodes, mesh.nnodesP, material;
              stress_size=(nq, mesh.nels))
stats = solve!(dr, mesh, bc_vx, bc_vy; dt = Δt, pressure_factor = γfact)
```

`solve_coupled!(thermal, stokes, thermal_mesh, stokes_mesh, bc_T, bc_v; dt, ...)`
accepts thermal and mixed Stokes states plus their meshes and boundary
conditions, and forwards to the Stokes `solve!` (owned pressure scaling, common
controls, `throw_on_failure`). It advances one thermal DR step per inner
velocity iteration and transfers continuous thermal-node values to the
discontinuous Stokes pressure DoFs. Its statistics add `err_T` and
`thermal_iterations`; `converged` requires both criteria.

The expanded positional methods remain available for adjoints and specialized
workflows, but new normal-user examples should start from the high-level forms.
Repeated 2-D adjoint solves can pass a `StokesAdjointWorkspace`; its
reverse-mode scratch is opt-in through `enzyme=true`, matching
`operator = :enzyme` without charging the default block path for unused arrays.

## Mutation and ownership contract

- A trailing `!` means one or more supplied arrays/states are mutated.
- Constructors allocate state on the selected backend; accessors return the
  live arrays, not defensive copies.
- Convert once with `Array(field)` for host-only plotting/output logic.
- Caller-owned previous-time fields (`T0`, `P0`, old stress) must be updated at
  the physical-time boundary documented by the solver; a solver must not guess
  that lifecycle. Under `DruckerPragerCap` the pressure memory is the accepted
  corrected IP pressure, passed as `solve_stokes_dyrel!(...; P_old)`.
- Returned convergence statistics describe the final in-place state. Never use
  a state as converged without checking the documented return/exception path.
- Connectivity, geometry, phases, fields, BC indices/values, and scratch arrays
  consumed by one kernel path must reside on compatible backends.

## Error and validation contract

Use `ArgumentError` for unsupported values/configuration,
`DimensionMismatch` for incompatible shapes, and bounds/domain errors when
their standard Julia meaning applies. Numerical failure must report the
problem, iteration/budget, and last meaningful residual where available.

Validate at the highest shared trust boundary:

- external coordinates/connectivity and element arity;
- material tuple lengths and phase indices;
- mesh/state/backend compatibility;
- required geometry/cache/preconditioner state;
- writer field lengths and supported topology.

Do not silently copy mixed-backend inputs or silently replace invalid values;
those behaviors hide costly scientific errors.

## Backend and extension contract

`TA(::CPU)` maps to `Array`. The optional CUDA extension in `ext/` adds the
`CUDABackend` mapping when CUDA is loaded. CUDA is the only supported GPU backend. Core code
must remain loadable without any GPU dependency.

Extensions also carry optional non-backend capability. `FEMToolsTriangulateExt`
adds the only method of `triangulate_t7_mesh`, so 2-D mesh generation is
available without making Triangulate a core dependency. A capability added this
way needs three pieces: a stub in `src/` that owns the docstring and raises an
error naming the package to load, the method in `ext/`, and the weak dependency
listed in both `[weakdeps]` and `[extras]` so the test target can exercise it.

Backend-generic API additions should:

- infer the backend from owned arrays when a high-level object already carries
  it;
- accept an explicit backend only at construction/allocation boundaries;
- use existing KA launch wrappers and synchronize only at real dependencies;
- keep host topology preparation separate from device compute arrays;
- test CPU behavior and every accelerator explicitly claimed as supported.

## Evolving the API

Before adding or changing a public name:

1. Search existing constructors, accessors, wrappers, and public low-level
   methods for the same operation.
2. Choose the narrowest tier that serves a real caller.
3. Define accepted types/layouts, mutation, backend placement, return value,
   failure behavior, and numerical meaning.
4. Add one focused API/regression check and inference coverage for hot generic
   paths.
5. Update `src/FEMTools.jl`, source docstrings, `docs/`, examples, and all
   affected `.agents/` guides together.
6. Prefer a compatibility wrapper and deprecation path when changing an
   established exported signature.

Avoid keyword aliases, abstract interfaces with one implementation, and
convenience overloads without a demonstrated call site. A small coherent API
is easier to document, optimize, and support across CPU/GPU/MPI backends.

## Current API pressure points

[`API_SIMPLIFICATION_PLAN.md`](../API_SIMPLIFICATION_PLAN.md) proposes a
breaking simplification driven by maintained examples and root benchmarks:
mesh-aware state construction, solver-owned pressure preparation, canonical
dispatch-based solver entry points, and removal of redundant call variants.
Work on `refactor/api-simplification` includes caller inventory, baselines, and
mesh-based constructors described below. Other proposed signatures remain
planned changes rather than the current contract.
All changed code must remain GPU-friendly, with explicit host-only boundaries
and CPU/CUDA checks with scalar indexing disabled. Preserve numerical layouts
and explicit accepted-history ownership.

- `solve!` covers the two scalar physics states, while Stokes uses named
  entry points and distinct 2-D/3-D layouts.
- Advanced Stokes/adjoint workflows still require long positional signatures;
  high-level mesh-owned methods cover only the common paths.
- Several boundary-condition types exist internally but only Dirichlet behavior
  is exported and applied as supported API.
- Example-local mesh import, plotting, and configuration are not core API.
- There is no distributed mesh/solver, checkpoint, or frontend API yet.

Resolve these only from real workflows. Do not create a universal problem or
solver abstraction in anticipation of future backends.

Q9 velocity mixed meshes now support discontinuous three-value P1 pressure
using `LinearElement{2,3}` evaluated on the reference square. Pressure topology
selects velocity nodes `(9,6,7)` (center, +x, +y), and pressure geometry uses
velocity-cell quadrature. `assemble_viscosity_weighted_pressure_scaling!` uses
positive Jacobi weights on Q9 cells, matching the signed pressure basis.

## Update this guide when

- an exported/public name, signature, default, return value, mutation, or error
  contract changes;
- a high-level workflow replaces an expanded low-level call;
- an extension/backend or distributed API is added;
- an internal type becomes supported API or a public name is deprecated;
- tests/docs reveal a compatibility promise not recorded here.

## Solver convergence histories

`collect_history=true` enriches Stokes `stats.history` entries with a per-velocity
`err_v_components` tuple while retaining `iter`, `err_v`, and `err_P`. Records
now include outer and inner checks; pressure norms use the residual assembled
at that check, and the final outer record matches returned convergence errors.
Scalar DR `solve!` with `collect_history=true` returns `(iter,residual,relative)`
records at checks, including convergence, in `stats.history`; otherwise it is empty.

## Mesh-aware state construction

Material keyword constructors accept scalars and tuples. The first property
tuple fixes phase count and precision; otherwise the first supplied scalar
fixes precision. Scalars repeat across phases and omitted properties preserve
their existing physical defaults in the inferred layout. Supplied floating
values are not promoted across precision. With no phase properties, Stokes
uses Tref/gravity precision, or Float64 when nothing is supplied. Gravity
controls dimension separately from phase count. Empty tuples are rejected.
Parameterized generated keyword constructors are removed; use the ordinary
constructor and infer its type. Repository callers used the ordinary form.

Ordinary setup uses `ThermalDiffusionDR(mesh, material)`,
`LithostaticPressureDR(mesh, material)`, and `StokesDR(mixed_mesh, material)`.
These dispatch on mesh type, infer backend and field sizes, and reject material
precision mismatches. Stokes also checks gravity dimension and requires cached
geometry, defaulting stress history to velocity quadrature points × elements.
Explicit stress layouts (including `:none`) and count-based constructors remain
available for custom layouts; maintained examples and benchmarks use the mesh forms.
Every state constructor takes a typed material (`ThermalMaterial` or
`StokesMaterial`); property-positional tuple forms do not exist. Dirichlet construction accepts
`(nodes, values)`, borrows both arrays, and rejects unequal lengths in both forms.

## Owned pressure scaling

`solve!(dr, mesh, bc_v; dt, pressure_factor=50, scaling_viscosity)` (and the
`bc_vx, bc_vy` form) for 2-D/3-D mixed Stokes assembles `dr.M_P` and the scale `dr.γP` into
state-owned storage before solving; `dr.γP` replaces the unused Stokes `∂P∂τ`
field. `scaling_viscosity` defaults to the momentum `viscosity`, so a separate
scaling viscosity (sinking block) stays explicit. The scale is recomputed per
call and uses the continuity residual's storage modulus (`K` with `finite_K`,
else `ηb`); a scale built from `K = Inf` while the residual stores
`(P−P_old)/(ηb Δt)` with finite `ηb` diverges. It maps `tolerance`, `max_iterations`, `check_interval` to `ϵ_tol`,
`total_iterMax`, `ncheck`, returns the `solve_stokes_dyrel!` statistics (now with
`iterations` and `residual` aliases), and throws on non-convergence unless
`throw_on_failure=false`. `solve_stokes_dyrel!(dr, mesh, bc_v, Δt, γP; ...)` remains the
low-level form for adjoints, prepared scales, and tests that need failed statistics.
Its array-positional core is internal (`_solve_stokes_dyrel!`); the `MixedMeshCache`
and split `bc_vx_vals, bc_vy_vals` forwarding methods are removed.

`solve_adjoint!(dr, mesh, bc_v; dt, objective_v, λv, λP, ...)` (and the
`bc_vx, bc_vy` form with `objective_vx/vy`, `λvx/vy`) is the mesh-owned 2-D/mixed
adjoint: geometry and elements come from the mesh and the forward scale `dr.γP`
is reused, with an `ArgumentError` when it was never assembled. Names map
`tolerance`→`adjoint_tol`, `max_iterations`→`total_iterMax`,
`check_interval`→`ncheck`; `λ` inputs are the warm start; non-convergence throws
unless `throw_on_failure=false`. `solve_adjoint!` is the only public mixed-mesh
adjoint; its array-positional core is internal (`_solve_stokes_adjoint_dyrel!`).
A caller that prepares its own scale writes it into `dr.γP` before both solves.

## Hex27 cell-pressure state

`CellPressureStokesDR(mesh::Mesh, material::StokesMaterial; phases)` is the state
of the purely viscous Hex27/Q2--P1 solver with four pressure modes per cell. It
owns `v` (`VectorField3D`), `P` (`4 × nels`), borrowed cell `phases` (default
all ones, validated for length, backend, and range), `η`, `ρ = material.ρ0`,
`g`, residuals, preconditioner, pressure mass, and reference tables. Finite `G`
or `K` and nonzero `α` are rejected; `ηb` and `Tref` are ignored.

`solve!(dr, mesh, bc_v::NTuple{3})` keeps the solver's former defaults
(`tolerance=1e-5`, `max_iterations=3000`, `check_interval=100`,
`velocity_step=0.6`, `pressure_step=0.2`) and returns `converged`, `iterations`,
`residual`, `history`, `err_v`, `err_P`, `iter`, `err`, `reached_total_iter`;
it throws on non-convergence unless `throw_on_failure=false`.
`solve_adjoint!(dr, mesh, bc_v; objective_v, λv, λP)` writes caller-owned
outputs and leaves `dr.v`/`dr.P` untouched; `stokes_material_gradient_3d(dr,
mesh, λv)` reads the forward fields and material from the state. The former
array-positional `solve_stokes_dyrel!`/`solve_stokes_adjoint_dyrel!` 3-D
methods, `solve_stokes_3d!`, `solve_stokes_adjoint_3d!`, and
`Stokes3DWorkspace` no longer exist.
