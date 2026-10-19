# FEMTools.jl Agent Guide

This is the shared repository contract for Codex and Claude Code. Codex reads
`AGENTS.md`; the root `CLAUDE.md` imports it. Maintain project rules here and
subsystem knowledge under `.agents/`, so both tools follow the same guidance.
Keep personal settings, absolute machine paths, and tool permissions in each
tool's user configuration.

Source and executable tests establish current behavior. Check them when a guide,
docstring, or plan disagrees; update affected guidance rather than implementing
an obsolete plan. A roadmap or experimental benchmark is not package support.

## Mandatory reading

Before modifying the repository, read this file and all nine guides:

- [API](.agents/API.md)
- [Meshing](.agents/meshing.md)
- [Solver](.agents/solver.md)
- [Frontend](.agents/frontend.md)
- [Miniapps](.agents/miniapps.md)
- [I/O](.agents/io.md)
- [Distributed parallelization](.agents/distributed.md)
- [Testing](.agents/testing.md)
- [Documentation](.agents/documentation.md)

For a narrowly scoped follow-up in the same session, reread the affected guide,
plus the testing and documentation guides. For architecture, planning, or a
cross-cutting change, reread all nine.

These files are living project memory. At the end of every task, review all
nine and update every guide whose current facts, invariants, decisions, open
questions, or next priorities changed. Do not edit unaffected guides merely to
change a date. Record durable knowledge, not a diary of individual commits.

## Project in one paragraph

FEMTools.jl is a Julia finite-element package for structured and unstructured
meshes. It provides reference elements, quadrature, connectivity, geometry
precomputation, mesh coloring, boundary conditions, matrix-free assembly, and
pseudo-transient dynamic-relaxation solvers for heat diffusion, lithostatic
pressure, and incompressible Stokes flow. KernelAbstractions makes compute paths
backend-agnostic; CUDA array selection is supplied through a
package extension. Triangulate meshing is another optional extension. The package
also contains discrete Stokes adjoints, post-processing, legacy ASCII VTK output,
examples, and Documenter docs.

The current package version is `0.2.0`. CPU is the default; CUDA is the only
supported GPU backend. Mixed `StokesDR` and Hex27 cell-pressure
`CellPressureStokesDR` have distinct layouts and solvers. Distributed meshes,
MPI solvers, parallel output, checkpoint/restart, and a package frontend are
not implemented. Consult the distributed guide for the proposed MPI work.

## Repository map

| Path | Responsibility |
|---|---|
| `src/FEMTools.jl` | Module load order and exported/public API |
| `src/elements/` | Element tags, shape functions, and quadrature |
| `src/mesh/` | Mesh containers, connectivity, geometry, sparsity, and coloring |
| `src/boundary_conditions/` | Boundary-condition data and application |
| `src/dynamic_relaxation/` | Shared DR iteration and update kernels |
| `src/heat_diffusion/` | Thermal state, residual assembly, and solver wrapper |
| `src/lithostatic_pressure/` | Pressure state, residual assembly, and solver wrapper |
| `src/stokes/` | 2-D/3-D Stokes states, rheology, operators, forward and adjoint solvers |
| `src/postprocess/` | Derived fields and VTK writers |
| `ext/` | Optional CUDA backend mapping and Triangulate meshing |
| `test/` | Unit, API, inference, allocation, convergence, and reference tests |
| `examples/` | Runnable scientific drivers, plotting, meshing, and benchmarks |
| `benchmarks/` | Exact-field verification, refinement, CUDA checks, and experiments |
| `docs/` | Documenter configuration and user documentation |
| `.agents/` | Shared subsystem contracts and living engineering notes |
| `CLAUDE.md` | Claude Code entry point importing this shared contract |
| `.github/workflows/CI.yml` | Package and documentation CI |

## Working rules

1. Inspect before editing. Read the complete code path, search every caller of
   functions being changed, and check nearby tests and docs.
2. Fix the shared cause once. Avoid caller-by-caller guards when all paths pass
   through a common implementation.
3. Preserve existing user changes. Check `git status --short`; do not overwrite,
   remove, format, or commit unrelated work.
4. Prefer the smallest complete change. Reuse an existing helper, Julia's
   standard library, or an installed dependency before adding machinery.
5. Keep public API deliberate. Exports are user-facing; `public` names are the
   extension/API surface; other names are implementation details. Update
   `src/FEMTools.jl`, export tests, docstrings, and docs together when the API
   changes.
6. Keep compute code backend-neutral. Derive the backend with
   `KernelAbstractions.get_backend`, allocate through existing backend-aware
   paths, and avoid host scalar indexing in device kernels or solver loops.
7. Preserve numerical type and layout. Hot element routines use static arrays
   and compile-time dimensions. New code must not silently promote `Float32`,
   change element node ordering, or mix host and device arrays.
8. Make failures explicit. Validate external mesh/data input and impossible
   solver states; never convert non-convergence, invalid geometry, or malformed
   connectivity into plausible output.
9. Add one focused regression check for non-trivial behavior. Expand only when
   distinct risks require distinct cases.
10. Update user docs for user-visible behavior and the relevant `.agents/`
    notes for durable engineering knowledge.

For normal user workflows, prefer typed materials, mesh-based state
constructors, `DirichletBoundaryCondition(nodes, values)`, and
`solve!`/`solve_adjoint!`. High-level mixed Stokes `solve!` prepares its own
pressure mass/scaling; the adjoint reuses them. Retain the prepared-scale
`solve_stokes_dyrel!` path for its specialized callers. Do not restore removed
positional APIs or introduce aliases without a demonstrated compatibility need.

## Numerical and performance invariants

- Connectivity is Julia 1-based and stored with one element per column.
- Mesh coordinates and solution fields must live on compatible backends before
  launching kernels.
- Element-local work should remain statically sized where element arity and
  dimension are known.
- Assembly code is matrix-free unless a named diagnostic/operator path
  explicitly assembles element blocks or a sparse oracle.
- Dirichlet residual/rate constraints are applied before DR spectral estimates
  and field updates.
- Physical time and pseudo time have different ownership. Hold previous-time
  fields fixed during a solve; commit stress and plastic history once per
  accepted physical step. Stress refresh must not advance accumulated history.
- Pressure scaling must use the storage modulus used by the continuity
  residual. Preserve the distinction between trial pressure, corrected physical
  pressure, and the numerical augmentation `Pnum`, especially for the tensile cap.
- Adjoint changes must transpose the actual discrete residual, including
  storage, augmentation, and boundary projection. Preserve the sign convention
  of the selected mixed or cell-pressure path and check gradients independently.
- Solver convergence must be assessed from mathematically meaningful residuals,
  including global reductions once MPI exists.
- Performance claims require a representative, warmed benchmark and a
  correctness gate. Do not optimize only a zero-contrast or tiny toy case.
- Avoid new allocations in established hot paths. The current suite checks
  zero-allocation element construction and shape-function evaluation.
- CUDA verification uses `CUDA.allowscalar(false)`. A CPU pass, a backend
  dispatch check, or a skipped hardware test does not prove device correctness.

## Change workflow

### Before coding

- Run `git status --short` and identify user-owned changes.
- Read the mandatory guides and the source, tests, examples, and docs relevant
  to the change.
- State the current behavior and the smallest acceptance check that proves the
  requested outcome.
- Identify any nested agent instructions before editing their subtree. Keep
  shared requirements in this file; tool-specific entry points must not copy
  the rules into a second independently maintained contract.
- Search for existing implementations before creating a new helper or
  dependency.

### While coding

- Keep the diff focused and follow the source file's established Julia style.
- Put generic mechanics in the shared layer and physics-specific behavior in
  its physics module.
- Keep optional capabilities out of the core dependency path when practical.
- Comment only non-obvious numerical intent, layout constraints, or deliberate
  simplifications with a known ceiling.
- Example scripts run their top-level entry point unconditionally. Do not wrap
  it in `abspath(PROGRAM_FILE) == abspath(@__FILE__)`. An example is a script:
  running it and including it do the same thing. Code that a test or another
  script needs without that side effect belongs in a shared helper file or in
  the package, not behind a guard.

### Before handing off

- Run the narrowest relevant check, then the full package tests when risk or
  scope warrants it. Shared element, geometry, DR, API, or module-load changes
  require the full suite; use the testing guide's change-to-check map.
- Build docs for changes to exports, docstrings, examples referenced by docs, or
  `docs/` content.
- Review all nine subsystem guides and update affected ones.
- Recheck `git diff --check`, `git status --short`, and the final diff for
  accidental files, generated output, or unrelated edits.
- Report what changed, what was verified, and any remaining limitation.
- Report exact commands, Julia version/backend, and passed, failed, skipped,
  or unrun checks. Distinguish pre-existing working-tree or environment failures
  from regressions; do not repair unrelated work to obtain a green run.

For changes confined to agent instructions/internal notes, validate imports,
local links, referenced paths, and `git diff --check`. A package test or
Documenter build is unnecessary unless executable code, public docs, or their
contracts also change.

## Standard commands

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl
julia --project=examples path/to/example.jl
git diff --check
```

`Project.toml` declares Julia 1.11, 1.12, and 1.13 compatibility. Use Julia 1.12
for the current full-suite gate: CI tests it on Ubuntu and macOS (with a
duplicate macOS matrix entry) and builds docs on Julia `1`. Declared
compatibility is not proof that every version/backend was tested. See
[testing](.agents/testing.md) before changing this matrix or claiming coverage
for accelerators or MPI.

### Persistent Julia session (Kaimon MCP)

Compilation latency dominates iterative work here. When available, use one
persistent Kaimon Julia session for development checks, with the repository's
project active. Start Kaimon from the repository root before the agent session:

```sh
kaimon -r              # Revise loaded, MCP endpoint on http://localhost:2828/mcp
kaimon -r --headless   # same, without the terminal dashboard
```

Configure the MCP connection in each client's user settings; it is not a
package dependency. The Julia tools are exposed here as `mcp__julia__julia_eval`,
`mcp__julia__julia_list_sessions`, and `mcp__julia__julia_restart`; discover the
actual names available in the client. Check the active environment/session
before using persistent state. If MCP is unavailable, use shell Julia when
permitted and report the fallback.

`julia_eval` runs `Revise.revise()` before every evaluation, so edits under
`src/` are already applied; do not call `Revise.revise()` again. Use
`julia_restart` after changes Revise cannot track: `Project.toml`, `struct`
definitions, and the `include` order in `src/FEMTools.jl`.

Run `julia` from the shell instead whenever a cold process is the measurement —
package load timing, invalidation analysis, benchmark baselines — and for the
final `Pkg.test()` and docs build before a pull request.

## Definition of done

A change is done when:

- the requested behavior works at the correct shared layer;
- relevant serial and backend invariants remain intact;
- a focused regression check covers non-trivial logic;
- affected public docs and subsystem notes agree with the implementation;
- validation was run, or the exact unrun checks and reason are reported;
- unrelated user work and generated artifacts are untouched.

## Scope of these instructions

This `AGENTS.md` applies to the entire repository. A future nested `AGENTS.md`
may add stricter local rules; it must not silently weaken the numerical,
testing, documentation, or living-notes obligations above.
