---
name: scrutinise
description: Scrutinise newly added or changed code on the current branch against main. Checks new types, methods, changed signatures, and helpers for necessity, correctness, clarity, consistency, robustness, and minimality. Reviews new tests for gap coverage, overlap with existing tests, and minimality. Invoke with /scrutinise.
tools: Bash, Glob, Grep, Read, Edit, Write
---

# Scrutinise

Review newly added or changed code on the current branch against `main`.

## 1. Gather the diff

```bash
git diff main...HEAD --stat
git diff main...HEAD -- src/ test/ examples/
```

Read every changed file in full, not just the hunks.

## 2. Source review (`src/`)

The package layout is:

- `src/elements/` — element types, shape functions, integration points
- `src/mesh/` — mesh construction, connectivity, sparsity, coloring
- `src/boundary_conditions/` — BC types and `apply_bc!`

For each new/changed type, method, or helper check:

- **Necessity** — does existing infrastructure already provide this?
  (e.g. connectivity helpers in `src/mesh/connectivity.jl`, element traits
  like `order(element)`).
- **Correctness** — type parameters consistent (`AbstractElement{nDim, N}`),
  `Int32` connectivity preserved, works for all element orders (linear,
  quadratic, cubic) and dimensions (1D/2D/3D), not just the case that
  motivated the change.
- **GPU compatibility** — anything stored in `Mesh` or passed to kernels must
  be isbits-element arrays (no `Vector{Vector}`, no closures over host data);
  backend-generic code goes through `TA(backend)` / KernelAbstractions, never
  bare `CuArray`.
- **Clarity & consistency** — naming and idiom match the surrounding file;
  docstrings follow the existing `"""\n    signature\n\ndescription\n"""` style.
- **Exports** — new public names added to the export list in `src/FEMTools.jl`
  and covered by `test/test_exports.jl`; internal helpers not exported.
- **Minimality** — no dead code, no speculative generality.

## 3. Test review (`test/`)

- Does each new test cover a genuine gap, or does an existing
  `test/test_*.jl` file already exercise it?
- Are new tests in the right file (`test_mesh.jl`, `test_elements.jl`,
  `test_shape_function_evaluations.jl`, `test_boundary_conditions.jl`,
  `test_allocations.jl`, `test_exports.jl`) rather than a new file?
- Do they follow established patterns (parametrised over `FP64`/`FP32`,
  element types, dimensions)?
- Are they minimal — smallest mesh / element count that exhibits the
  behaviour?

## 4. Output

Report findings grouped by file, each tagged with the failed criterion
(necessity / correctness / GPU compatibility / clarity / consistency /
exports / minimality). Propose concrete simplifications. Apply fixes only
when asked.
