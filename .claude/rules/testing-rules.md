---
paths:
  - test/**/*.jl
---

# Testing Rules

## Running tests

Use the smallest relevant test command first, then widen only when needed.

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

For a focused check, run the file most likely to cover the change, for example:

```sh
julia --project=. -e 'include("test/test_mesh.jl")'
```

If the change is in a solver or kernel path, exercise the closest numerical regression first before broad suite coverage.

## Writing tests

- Put tests into the existing `test/test_*.jl` files instead of creating a separate ad hoc file unless the change is truly domain-specific.
- Prefer the smallest mesh and element count that still exercises the bug.
- Cover both normal and edge-case behavior: valid geometry, invalid constraints, Dirichlet application, and backend-neutral code paths.
- When numerical accuracy matters, compare against an analytic solution or a trusted reference result.
- Keep tests deterministic and backend-agnostic unless the backend is the point under test.

## Debugging strategy

- Reproduce the bug on the CPU path before diagnosing GPU differences.
- Fail early with explicit conditions instead of converting invalid states into plausible output.
- Check shared root cause before adding guards at individual call sites.
- Validate the fix against the relevant package-level invariant: residual convergence, connectivity sanity, or allocation bounds.

## Quality bar

A regression test should tell a reader which failure mode it protects against. If the test only confirms a mock path, it is not a useful package test.
