---
paths:
  - src/**/*.jl
  - examples/**/*.jl
---

# Kernel Rules

## Backend neutrality

- Keep code backend-neutral by deriving the backend through `KernelAbstractions.get_backend` and using existing backend-aware conversion helpers.
- Do not hard-code CUDA, AMDGPU, or Metal behavior into data structures or numerical kernels.
- Prefer shared code paths over backend-specific conditionals unless the backend really requires a different algorithm.

## Data layout

- Only pass isbits-compatible arrays and scalar values into kernels.
- Avoid host-owned nested containers such as `Vector{Vector{...}}` in kernel inputs or stored mesh state.
- Do not capture mutable host state in closures used by kernels.
- Preserve static-array conventions and element-local memory layout; do not silently reshape or reorder local data in a way that changes the discrete operator.

## Safety

- Validate that mesh coordinates and solution fields live on compatible backends before launching kernels.
- Keep kernels free of host-side scalar indexing and loop behavior that depends on CPU-only assumptions.
- Favor batched or static computations over dynamically growing temporary arrays in hot paths.

## Verification

- Check the kernel logic on the CPU path before relying on a GPU result.
- For any kernel change, validate against a small reference case that exercises the same assembly or update pattern.
- When a kernel fails or becomes unexpectedly memory-heavy, inspect the live per-thread footprint before changing the algorithm.
