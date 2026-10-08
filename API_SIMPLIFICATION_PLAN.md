# API and source simplification plan

Status: stages 1–5 implemented on `refactor/api-simplification`, with CPU/CUDA
agreement and warmed timings recorded in the baseline. `advance!` is deferred:
time-stepping miniapps differ in which history they accept between steps.
Breaking API changes are allowed. The goal
is fewer concepts, fewer caller responsibilities, and clearer numerical code.
Implementation should proceed in small independently verified changes.

## GPU compatibility is mandatory

All code introduced or changed by this plan must be GPU-friendly. Numerical
paths must run through backend-neutral KernelAbstractions kernels, broadcasts,
and supported reductions, with no host scalar indexing of device arrays or
implicit device-to-host copies. Infer the backend from owned arrays and allocate
fields, phases, indices, geometry, tables, history, and scratch on that backend.
Keep precision and statically sized element-local work intact. Dispatch and
kernel arguments must remain concrete and device-compilable.

Host-only mesh preparation, file I/O, analytical sampling, and plotting must
have explicit boundaries and bounded transfers. They must not enter solver
iterations. CUDA remains optional; core loads without CUDA or a GPU driver.
This applies to constructors, validation, solvers, adjoints, history updates,
post-processing, and example orchestration, not only assembly kernels.

Every numerical implementation stage requires CPU/CUDA agreement with
`CUDA.allowscalar(false)`, Float32/Float64 checks, and relevant inference and
allocation checks. Report unavailable hardware and unsupported paths explicitly;
a CPU pass, source review, or skipped CUDA test is not GPU validation. Do not
introduce a CPU-only shortcut merely to reduce source length.

## Evidence and scope

- `benchmarks/stokes/solkz2D/SolKz2D_triangle.jl:78-112` repeats mesh sizes,
  stress dimensions, phase expansion, backend conversion, pressure-scale
  allocation, and a separate scaling assembly before the solve.
- `benchmarks/thermal/thermal_diffusion2D/ThermalDiffusion2D.jl:84-110`
  repeats backend/node counts and explicitly manages previous temperature,
  pseudo-time rate, boundary values, and separate convergence histories.
- `examples/miniapps/stokes/sinking_block/sinking_block.jl` intentionally uses a
  different viscosity for pressure scaling. Simplification must preserve that
  scientific choice rather than silently use the solve viscosity everywhere.
- `src/stokes/solvers/DR.jl` already has a dimension-generic mixed-state solver,
  but exposes mesh-owned, cache, raw-geometry, split-component, tuple-component,
  and specialized cell-pressure call forms under the same long name.
- `src/stokes/solvers/DR_adjoint.jl` still requires geometry, elements, phases,
  history, backend, and pressure scaling already available to the forward path.
- `src/stokes/types/stokes_types.jl` and `src/heat_diffusion/types/` expose
  material constructors alongside expanded property constructors.
- Existing useful dispatch includes `dr_fields`, `dr_name`, `_phase_at`,
  viscosity evaluation, constitutive models, and topology-dependent mass weights.
  Build on these rather than introducing a replacement framework.

The six existing deletions under `examples/benchmarks/` are user changes.
Do not restore, relocate, or otherwise resolve them as part of this work.
Use maintained root `benchmarks/` and current miniapps as acceptance workflows.

## Proposed user surface

Keep existing element and mesh types initially. Make mesh-aware state
construction and one high-level solver family the primary interface:

```julia
thermal = ThermalDiffusionDR(mesh, material)
bc = DirichletBoundaryCondition(nodes, values)
stats = solve!(thermal, mesh, bc; dt, tolerance, max_iterations)

stokes = StokesDR(mixed_mesh, material; phases, plastic)
stats = solve!(stokes, mixed_mesh, (bc_x, bc_y);
               dt, tolerance, max_iterations, pressure_factor,
               viscosity, body_force)
```

These are proposed signatures. `solve!` mutates current fields and does not
advance accepted physical-time history. `dt` is required for transient/elastic
physics; steady lithostatic solving has no timestep. Repeated solves should
reuse state-owned scratch. Optional advanced workspaces remain available where
their memory cost or lifetime warrants separate ownership.

Keep `solve_adjoint!` and `solve_coupled!` distinct operations. Dispatch each
on actual state and mesh types. Do not encode physics in symbols or a universal
problem configuration. Initially retain state type names; renaming them adds
migration work without removing setup responsibilities.

Use `tolerance`, `max_iterations`, `check_interval`, `verbose`, and
`collect_history` consistently at the user boundary. Keep algorithm-specific
outer/inner controls explicit for advanced use. Translate common names once;
do not pretend different residual definitions are numerically interchangeable.

All high-level solves should return statistics with common fields `converged`,
`iterations`, `residual`, and `history`, plus physics-specific residuals and
diagnostics. Default non-convergence throws; an explicit opt-out returns failed
statistics for diagnostic workflows. Preserve each solver's stopping mathematics
while changing the reporting contract. Decide concrete statistics storage from
the actual schemas; a new result hierarchy is not required.

## Implementation order

### 1. Establish the baseline and API inventory

Initial inventory tooling and CPU/CUDA correctness smokes are implemented.
See [API_SIMPLIFICATION_BASELINE.md](API_SIMPLIFICATION_BASELINE.md) for measured
API counts, executed checks, and remaining gates. This stage is not complete.

Inventory exported/public names, method variants, and every caller in source,
tests, docs, examples, benchmarks, and extensions. Classify each as ordinary
workflow, supported advanced capability, internal implementation, or obsolete
compatibility. Record the canonical replacement before deleting a signature.

Capture small CPU reference runs for thermal, lithostatic, T7/P1, Q9/P1,
mixed 3-D, specialized Hex27 cell-pressure, coupled, and adjoint workflows.
Include heterogeneous viscosity, finite bulk modulus, and cap history where
applicable. Record fields, convergence errors, histories, warmed allocations,
timings, and Julia/backend versions. Keep generated data outside source control.

Acceptance: a caller inventory and reproducible correctness gates exist before
the implementation changes. No performance improvement is claimed yet.

### 2. Remove routine setup from callers

Implemented first slice: mesh-based thermal, lithostatic, and mixed Stokes
constructors; inferred backend/counts/stress layout; material compatibility
checks; two-argument Dirichlet data with length validation; migration of the
six maintained exact-field drivers and solver topic pages. CPU/CUDA Float32
and Float64 constructor and solve agreement checks pass. Stage 2 remains in
progress. Typed material defaults and scalar properties are implemented;
SolCx shares one cell-phase row, while SolKz uses default phases. Owned pressure
scaling is implemented for the mixed 2-D forward solve (`dr.γP`,
`pressure_factor`, `scaling_viscosity`); SolKz/SolCx, Solvi, sinking-block,
shear-band, ice-bridge, Popov, pure-shear, and 2-D adjoint drivers use it.
Hand-written DR loops (`stokes_2D_pure_shear`, elastic build-up) keep the
low-level scaling call because they own their pressure update. Count-based entry points remain during caller
migration. Generated parameterized material keyword forms had no repository
callers and are removed in favor of inferred ordinary construction.

Add mesh-aware constructors for thermal, lithostatic, and mixed Stokes states.
Infer sizes, dimension, quadrature extents, precision, and backend from the mesh;
validate material compatibility. Default phase assignment to phase one.
Accept cell phases without expanding identical rows for every local node.
Retain nodal and element-local phases when their meaning is needed.

Use material-dependent typed defaults so a one-phase material accepts scalar
properties and omitted properties preserve precision and phase count. Normalize
once into the existing tuple representation. Do not silently convert supplied
incompatible materials or mixed-backend arrays.

Add the two-argument Dirichlet constructor; make boundary labels optional
metadata. Validate lengths at construction. Preserve independent node sets per
velocity component. Add mesh-based value sampling only if repeated callers
justify it, with explicit host sampling and one backend transfer.

Move pressure-scale allocation and assembly into the high-level Stokes solve.
Own buffers with the reusable solve state/workspace. Initially recompute values
per solve; add cache invalidation machinery only if measurements justify it.
Use the same viscosity override as momentum by default, but allow a separate
scaling viscosity and an explicit prepared scaling override for specialized
experiments and adjoints. Preserve the current scaling formula and mass weights.

Acceptance: SolKz setup needs no manual pressure-scale buffer, mass preparation,
stress dimensions, or all-one phase matrices. Existing reference fields agree.

### 3. Consolidate solver entry points and ownership

Implemented first slice: scalar `solve!(thermal, mesh, bc; dt, ...)` and
`solve!(lithostatic, mesh, bc; ...)` with `tolerance`, `max_iterations`,
`check_interval`, `collect_history`, `throw_on_failure`, and returned statistics
`(; converged, iterations, residual, history)`. `solver!` and its expanded
callers are removed; the low-level forms are internal (`_solve_thermal!`,
`_solve_lithostatic!`). Mixed 2-D `solve!(stokes, mesh, bc_v; dt, ...)` is also
implemented (owned pressure scaling, common controls, `throw_on_failure`); the
SolKz/SolCx, Solvi, sinking-block, shear-band, ice-bridge, Popov, and
unstructured pure-shear drivers use it. The positional `γP` form is the
low-level form. Mixed 2-D `solve_adjoint!` (forward scale reused from the state)
is implemented and used by both 2-D adjoint miniapps. The Hex27 cell-pressure
path has its own state, `CellPressureStokesDR(mesh, material; phases)`, owning
fields, phases, material, and scratch, with `solve!`, `solve_adjoint!`, and
`stokes_material_gradient_3d(dr, mesh, λv)`; its array-positional methods,
`solve_stokes_3d!`, `solve_stokes_adjoint_3d!`, and `Stokes3DWorkspace` are
removed after migrating all callers. CPU/CUDA Float32/Float64 agreement and
old/new field equality (≤4e-14, equal iteration counts) were checked.
`solve_coupled!(thermal, stokes, thermal_mesh, stokes_mesh, bc_T, bc_v; dt, ...)`
replaces `solve_coupled_dyrel!`; it forwards to the mixed Stokes `solve!`, so it
owns pressure scaling and shares the common controls and failure contract.

Introduce the canonical `solve!` methods and consistent controls/statistics.
Remove old scalar/Stokes solver names, expanded constructors, split-component
wrappers, and keyword aliases after migrating their repository callers.
Keep low-level implementation entry points separate from the normal user API.
Keep a low-level function public only when a concrete extension or advanced
caller needs its contract; tests alone do not justify public status.

Wrap specialized Hex27 cell-pressure arrays in a small concrete state only if
needed to support the common call shape. Preserve its four-mode pressure layout
and algorithm. Do not route it through the mixed-mesh solver just to share a name.

Add a mesh-owned adjoint entry point that takes forward state, boundary data,
objective loads, and explicit adjoint output storage. Reuse forward scaling and
configuration; preserve warm starts and optional adjoint workspace reuse.
Validate supported adjoint physics explicitly, including the existing finite-
storage approximation. Do not imply that unifying names fixes that limitation.

Keep history acceptance explicit. If repeated workflows justify `advance!`, add
it as a separate physical-step operation: snapshot accepted fields, solve,
update stress, and commit plastic history exactly once on success. A failed
solve must not commit history. Do not introduce rollback copies without a
documented need. Corrected IP pressure must remain the cap's accepted pressure.

Acceptance: one documented high-level call per operation, with convergence and
failure tests. Multi-step cap recurrence and adjoint transpose/gradient gates pass.

### 4. Simplify implementation after callers migrate

Implemented first slice: maintained examples and benchmarks construct thermal,
lithostatic, and mixed Stokes states from meshes. `solve_stokes_dyrel!` keeps
only its mesh-owned forms; the array-positional core is internal
(`_solve_stokes_dyrel!`, per-direction node sets positional) and its
`MixedMeshCache` and split-component forwarding methods are removed.
Atomic and colored scalar assembly already share element mathematics through
`assemble_dr_matrices_{atomix,colored}!`; only the scatter differs. Warmed
Stokes and thermal timings match `main` (see the baseline record).
State constructors take a typed material: the property-positional
`ThermalDiffusionDR`, `LithostaticPressureDR`, and `StokesDR` forms are removed,
and `StokesMaterial` accepts any two- or three-element gravity collection.

Remove forwarding layers made unnecessary by the new API. Derive backend and
sizes once at the owning boundary. Avoid rebuilding or passing reference
tables through layers that already own them. Reuse existing workspace/table
storage before adding another cache type.

Use dispatch for meaningful alternatives: absent/present loading, nodal/IP
pressure history, tuple/IP viscosity, nodal/cell phases, constitutive models,
and actual mesh layouts. Normalize any user symbols once before hot loops.
Keep numerical iteration checks and algorithm controls as ordinary branches.
Do not replace every Boolean with a policy type or create one-method abstractions.

Share element mathematics between atomic and colored assembly; keep scatter
mechanics separate. Extract repeated setup, constraints, or residual evaluation
only when doing so removes substantial repetition and leaves the solver readable.
Keep physics-specific convergence loops separate where their mathematics differs.

Replace decorative section banners and comments that restate code with short
numerical explanations where needed. Keep descriptive names, conventional Julia
formatting, short functions with clear ownership, and visible governing equations.
Do not shorten scientific names, flatten useful types, or add macros for style.

Acceptance: fewer forwarding methods, repeated argument bundles, and duplicate
blocks; no new hot-path allocation or loss of inference. Representative warmed
timings stay within measured baseline variability, or regressions are investigated.

### 5. Migrate workflows and reduce repeated benchmark support

Implemented first slice: the manual home page starts with executed thermal and
Stokes workflows; maintained callers use the two-argument Dirichlet constructor.
`benchmarks/support.jl` holds the quadrature sampling, nodal-load, weighted-error,
gauge, archive, and figure helpers of the six exact-field scripts; their fields,
errors, histories, and metadata are bitwise unchanged, and
`benchmarks/check_exact_fields.jl` passes.

Migrate all maintained workflows, README, docstrings, export tests, and manual
pages in the same change that removes their old API. Put a short complete
thermal and Stokes workflow first in the manual. Keep advanced assembly separate.

Consolidate repeated quadrature sampling, interpolation, weighted errors, and
history archive handling in a small benchmark support file. Promote a helper
to core only when it is a reusable library operation with a supported contract.
Keep exact solutions, plotting, JLD2, and experiment metadata outside core.
Retain explicit scientific parameters and unconditional script entry points.

Acceptance: all maintained scripts parse; selected small headless cases execute;
exact-field refinement, high-contrast checks, and archive round trips pass.

## Completion gates and stopping rule

Each implementation change gets one focused behavioral regression plus affected
inference, precision, and ownership checks. Shared API/solver/mesh changes require
the full Julia 1.12 package suite and docs build. Run exact-field checks through
the examples environment and CUDA agreement when hardware is available; report
skips as missing coverage. Preserve independent sparse and finite-difference
oracles rather than comparing two wrappers around the same implementation.

Compare canonical method counts, required setup statements, duplicated blocks,
and warmed allocations/timings before and after. Do not set arbitrary percentage
targets or trade clarity for line count. Stop when representative users specify
mesh, material, phases, boundaries, physical controls, and forcing without
handling routine solver scratch. No new UI, MPI layer, mesher, solver algorithm,
checkpoint format, or rheology model belongs in this effort.

Review all nine subsystem guides after each implementation stage and update
those whose contracts change. No solver contract has changed yet.
