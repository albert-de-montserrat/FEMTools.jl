---
name: verify-example
description: Run a Poisson example headlessly on the CPU backend to verify that library or kernel changes still converge to the right solution. Use after changing src/ internals or example kernels.
tools: Bash, Glob, Grep, Read, Edit
---

# Verify example

Run one of the `examples/Poisson/*.jl` scripts as a smoke/convergence test
without a GPU or a display.

## 1. Pick the example

- `1D/2D/3D_Poisson.jl` — direct sparse solves, fastest sanity check
- `*_Poisson_AD.jl` — CPU pseudo-transient (PT) solvers with ForwardDiff
- `*_Poisson_AD_KA.jl` — KernelAbstractions PT solvers (CPU/GPU backends)

Prefer the example closest to the changed code; for `src/` changes the plain
`3D_Poisson.jl` plus one KA script is usually enough.

## 2. Make it headless and cheap

Run a *modified copy* (or `include` with overrides) — do not commit these
edits:

- backend: `const backend = CPU()` (KA scripts hardcode `CUDABackend()`)
- grid: shrink `n` (e.g. 8–16 per dimension) so the run takes seconds
- plotting: skip the GLMakie block (everything after the iteration loop and
  `display(to)`), or run with `xvfb-run` if the figure itself is under test
- iterations: PT loops break on `nr/nr0 < epsi`; leave the tolerance alone so
  convergence is actually exercised

```bash
julia --project -t auto /tmp/example_modified.jl
```

## 3. What "passing" means

- The convergence trace (`Iter. xxxxx: r/r0`) must decrease monotonically
  over checkpoints and reach the tolerance — not stall, not NaN
  (the scripts already `error("NaNs")`).
- For PT scripts, both assembly variants (atomic and colored) must agree:
  same trace when swapping which one is enabled.
- For solution correctness, compare against the direct-solve example on the
  same mesh/BCs, or against an analytic case: uniform source with
  homogeneous Dirichlet BCs on the unit cube has a known symmetric bump
  (and matches `examples/Poisson/KA_sandbox/poisson_2step.jl`).
- Beware trivial setups: zero source with constant-matching BCs converges
  instantly to a constant and verifies nothing.

## 4. Report

Quote the final `nr/nr0`, the iteration count, and which variants were run.
If the trace diverges or stalls, bisect: backend (CPU vs GPU), assembly
variant (atomic vs colored), element order (Q1 vs Q2) — in that order.
