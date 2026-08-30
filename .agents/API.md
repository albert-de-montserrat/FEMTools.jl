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
  `shape_function_values`, `gauss_legendre_triangle`.

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
- Derived data: `precompute_geometry`, `generate_sparsity_pattern`,
  `color_mesh`, `generate_element_groups`.
- Boundary application: `apply_bc!`.

High-level `Mesh` constructors either build a structured mesh from a domain or
accept validated coordinates/connectivity. Passing a `ReferenceElement`
precomputes solver geometry; omitting it produces a topology-only mesh.
`MixedMesh` owns distinct velocity and pressure layouts, while
`MixedMeshCache` owns their precomputed geometry/reference elements.

### Physics states and solvers

- Materials/states: `ThermalMaterial`, `ThermalDiffusionDR`,
  `LithostaticPressureDR`, `StokesMaterial`, `StokesDR`, `DruckerPrager`.
- Scalar entry point: `solver!` for thermal diffusion and lithostatic pressure.
- Stokes entry points: `solve_stokes_dyrel!`,
  `solve_coupled_dyrel!`,
  `solve_stokes_adjoint_dyrel!`, `solve_stokes_3d!`,
  `solve_stokes_adjoint_3d!`.
- Stokes operations: `assemble_viscosity_weighted_pressure_scaling!`,
  `pressure_mass`, `rotate_stress!`, `update_stokes_current_stress!`,
  `stokes_material_gradient_3d`.

`solver!` mutates a scalar problem state and returns `nothing` on convergence;
non-convergence throws. Stokes solvers mutate caller/state arrays and return a
statistics `NamedTuple` with explicit convergence status. Preserve this
difference until a concrete migration justifies changing user code.

### Results and output

- Derived fields: `compute_strain_rate_stress_postprocess`,
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
  `circle_boundary_nodes`;
- result accessors: `velocity`, `stress`, `pressure`, `temperature`.

These names are callable as `FEMTools.name` but are deliberately not imported
by `using FEMTools`. Before adding another public low-level method, confirm that
an extension, maintained miniapp, or downstream package genuinely needs it.

## Canonical user flows

### Single-field problem

```julia
element = ReferenceElement(LinearElement{2, 3, Float64})
mesh = Mesh(backend, coords, el2n, element)
material = ThermalMaterial(; k, Cp, ρ0, α, K)
dr = ThermalDiffusionDR(backend, mesh.nnodes, material)
bc = DirichletBoundaryCondition(nothing, nodes, values)
solver!(dr, Δt, mesh, bc; workgroup)
T = FEMTools.temperature(dr)
```

Lithostatic pressure follows the same mesh/material/BC shape with
`LithostaticPressureDR` and `solver!(dr, mesh, bc; ...)`.

### Two-dimensional mixed Stokes problem

```julia
mesh_v = Mesh(backend, coords, el2n_v, element_v)
mesh = MixedMesh(mesh_v, element_P)
cache = MixedMeshCache(backend, workgroup, mesh, element_v, element_P)
material = StokesMaterial(; η, ηb, G, α, ρ0, K, g, Tref)
dr = StokesDR(backend, mesh.nnodes, mesh.nnodesP, material;
              stress_size=(nq, mesh.nels))
assemble_viscosity_weighted_pressure_scaling!(γP, dr, mesh, cache, γfact, Δt)
stats = solve_stokes_dyrel!(dr, mesh, cache, bc_vx, bc_vy, Δt, γP)
stats.converged || error("Stokes solve did not converge")
```

`solve_coupled_dyrel!` accepts thermal and 2-D Stokes states plus their meshes
and boundary conditions. It advances one thermal DR step per inner velocity
iteration and transfers continuous thermal-node values to the discontinuous
Stokes pressure DoFs. Its statistics add `err_T` and `thermal_iterations`.

The expanded positional methods remain available for adjoints and specialized
workflows, but new normal-user examples should start from the high-level forms.

## Mutation and ownership contract

- A trailing `!` means one or more supplied arrays/states are mutated.
- Constructors allocate state on the selected backend; accessors return the
  live arrays, not defensive copies.
- Convert once with `Array(field)` for host-only plotting/output logic.
- Caller-owned previous-time fields (`T0`, `P0`, old stress) must be updated at
  the physical-time boundary documented by the solver; a solver must not guess
  that lifecycle.
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

`TA(::CPU)` maps to `Array`. Optional package extensions in `ext/` add mappings
for CUDA, AMDGPU, and Metal backends when those packages are loaded. Core code
must remain loadable without any GPU dependency.

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

- `solver!` is shared by two scalar physics states, while Stokes uses named
  entry points and distinct 2-D/3-D layouts.
- Advanced Stokes/adjoint workflows still require long positional signatures;
  high-level cache-owned methods cover only the common paths.
- Several boundary-condition types exist internally but only Dirichlet behavior
  is exported and applied as supported API.
- Example-local mesh import, plotting, and configuration are not core API.
- There is no distributed mesh/solver, checkpoint, or frontend API yet.

Resolve these only from real workflows. Do not create a universal problem or
solver abstraction in anticipation of future backends.

## Update this guide when

- an exported/public name, signature, default, return value, mutation, or error
  contract changes;
- a high-level workflow replaces an expanded low-level call;
- an extension/backend or distributed API is added;
- an internal type becomes supported API or a public name is deprecated;
- tests/docs reveal a compatibility promise not recorded here.
