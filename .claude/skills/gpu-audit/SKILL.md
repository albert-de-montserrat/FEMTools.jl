---
name: gpu-audit
description: Audit KernelAbstractions kernels and data structures for GPU compatibility and per-thread memory footprint. Use when a kernel crashes on CUDABackend, runs out of GPU memory, or performs poorly relative to CPU.
tools: Bash, Glob, Grep, Read
---

# GPU audit

Diagnose GPU failures and slowdowns in KernelAbstractions code (library code in
`src/`, solver kernels in `examples/`). Work through the checks in order; each
maps to a failure mode this repo has actually hit.

## 1. Host-only data reaching the device

A `CuArray` can only hold isbits elements. Grep everything that is converted
with `TA(backend)` / `TDev` / `to_backend` or stored in `Mesh` and flag:

- ragged arrays (`Vector{Vector{...}}`, e.g. node-to-element adjacency) —
  needs a CSR-style flat `data` + `offsets` pair or a padded dense matrix;
- arrays of mutable structs, strings, or anything with pointers;
- `BitVector` masks passed to kernels — convert via `TDev(Vector{Bool}(mask))`;
- CPU arrays captured by kernel arguments or closures (e.g. a host `Nq` tuple
  is fine — it is passed by value — but a host `Vector` is not).

Typical error: `CuArray only supports element types that are allocated inline`.

## 2. Per-thread local-memory footprint (launch-time OOM)

CUDA backs spilled registers with *local memory*, allocated as
`bytes_per_thread × max resident threads on the device` (~100k threads on a
consumer card) — independent of `ndrange` and workgroup size. Tens of KB per
thread therefore reserves GBs at launch. Estimate the live state of each
kernel thread:

- whole-element copies of precomputed data (`geo[iel]` returning an
  `NTuple{NQ, ...}`) — restructure to an `NQ × nels` matrix and stream
  `geo[q, iel]` inside the quadrature loop instead;
- ForwardDiff Duals: a Dual with `C` partials is `(C+1) × sizeof(T)` per
  scalar; a full N-partial elemental Jacobian over `SVector{N}` costs
  O(N²) bytes — use the chunked accumulation pattern
  (`jacobian_rowsums_diags` in `examples/Poisson/3D_Poisson_AD_KA.jl`,
  chunk constant `jacobian_chunk`);
- large `SMatrix`/`SVector` temporaries: anything beyond ~255 registers
  (≈ 2 KB) per thread spills.

Rule of thumb: footprint scales with element node count N and quadrature
count NQ; Q2 hexes (N = NQ = 27) are ~30× heavier than Q1 (N = NQ = 8).

## 3. Verification on CPU

KA kernels run unchanged on `CPU()`. Before touching GPU, validate any kernel
restructuring numerically on a tiny mesh (e.g. `(2, 2, 2)` elements) against
the straightforward loop implementation, for both Q1 and Q2 elements.
Tolerance, not bitwise: compiler fma/reassociation differs between code
shapes; expect agreement to ~1e-13 relative.

## 4. Performance checks

- `CUDA.registers(kernel.fun)` and local-memory stats confirm or rule out
  spilling; `CUDA.@profile` for a launch-level view.
- Workgroup size: too small (≤16) starves SMs once register pressure is gone;
  64–128 is the usual sweet spot. It does *not* fix local-memory OOM.
- Full loop unrolling (`@generated` over quadrature points) has been tried
  and **made things worse** (register pressure); prefer the rolled loop.
- Don't run atomic *and* colored assembly per iteration except when
  benchmarking; atomics have measured faster in this repo.
- Count kernel launches per iteration; fuse pointwise nodal kernels (rate,
  variable update, Dirichlet) and remove `KA.synchronize` between same-queue
  launches — synchronize only before host-side reductions.
- FP64 runs at 1/64 rate on consumer NVIDIA GPUs; Float32 is the single
  biggest throughput lever if accuracy allows.

## 5. Report

State the failure mode found, the numbers behind it (bytes per thread,
reserved local memory, array sizes), and the minimal restructuring that fixes
it. Apply changes only when asked.
