# Documentation

## Current reality

Documentation has four layers:

1. `README.md` is the concise project overview, installation entry point, and
   high-level API sampler.
2. Source docstrings describe public/exported APIs and important internal
   extension points.
3. `docs/src/` contains the Documenter manual for elements, meshes, boundary
   conditions, heat diffusion, lithostatic pressure, Stokes, examples, and API
   reference.
4. `AGENTS.md` and `.agents/*.md` are internal living engineering guidance.

`docs/make.jl` uses Documenter with `checkdocs=:exports`, a recursive
`DocTestSetup`, a fixed page list, and deployment from `main`. Adding a Markdown
page does not publish it automatically; add it to the `pages` tree when it is
meant for users.

The docs environment is separate under `docs/Project.toml`. Examples referenced
by the manual usually run under `examples/Project.toml`, not the docs
environment.

## Source of truth

- Code and executable tests define behavior.
- Public docstrings define call signatures, mutation, defaults, return values,
  failure behavior, units/layout, and backend constraints.
- The manual explains concepts and complete workflows.
- README advertises only stable, supported capabilities.
- `.agents/` records implementation boundaries, invariants, current gaps, and
  next engineering decisions; it must not contradict user docs.

When these disagree, fix all affected layers in the same change. Do not copy
large blocks between layers when a link or `@docs` block keeps one source of
truth.

## Documentation rules

- Document current behavior as fact and future work as a plan or open decision.
- Use exact type/function names and runnable Julia syntax.
- State array layout, indexing, units, mutation, backend transfers, and return
  values where users could otherwise make a scientifically plausible mistake.
- Keep examples headless-capable and identify the correct project environment.
- Add public names deliberately to `src/FEMTools.jl`; exported names must have
  docstrings that satisfy Documenter checks.
- Explain numerical methods enough to state assumptions and interpretation,
  but put derivations only where they help users choose or validate a method.
- Store generated images under `docs/src/assets/` only when the source workflow
  and scientific setup are documented.
- Avoid claims of GPU, MPI, format, element, or solver support that lack an
  executable check.
- Preserve terminology: element-local nodes, element-per-column connectivity,
  velocity/pressure mixed fields, physical versus pseudo time, and
  point-versus-cell data.

## Change-to-doc map

| Change | Documentation to review |
|---|---|
| Installation/dependency/support | `README.md`, docs home, relevant guide |
| Exported API/signature/default | docstring, `docs/src/api.md` or topic page, README if showcased |
| API tier or extension contract | `API.md`, `src/FEMTools.jl`, export/API tests |
| Element/mesh support | element and mesh pages, meshing guide, I/O guide if ordering changes |
| Solver model/control/result | physics page, docstring, solver guide, maintained examples |
| Frontend/example workflow | relevant topic/tutorial, frontend guide, command/environment |
| 2-D/3-D miniapp | miniapps guide, referenced topic page, headless command and outputs |
| I/O format/field | mesh or postprocess docs, I/O guide, round-trip limitations |
| MPI behavior | distributed guide and user manual page once actually supported |
| Test/CI command | `AGENTS.md`, testing guide, contributor-facing docs |

## Verification

Build locally from the repository root:

```sh
julia --project=docs docs/make.jl
```

For documentation changes, check:

- Documenter completes without missing exports or unresolved references;
- every page intended for publication appears in `docs/make.jl`;
- code snippets use the right environment and public API;
- local relative links resolve, including every link from `AGENTS.md`;
- formulas, field names, and defaults match source;
- referenced example output is not accidentally committed unless it is a
  deliberate docs asset.

## Living-guide maintenance

After each task, review all `.agents/` guides. Edit a guide when durable facts,
decisions, invariants, accepted limitations, priorities, or validation commands
changed. Remove resolved gaps instead of appending a chronological log. Keep
deep investigation records in a dedicated plan such as
`ADJOINT_PERF_PLAN.md` only when their measurements and history remain useful.

## Update this guide when

- documentation structure, build, deployment, or environments change;
- a new public manual section or internal subsystem guide is added;
- source-of-truth or maintenance policy changes;
- terminology or supported user workflows change.
