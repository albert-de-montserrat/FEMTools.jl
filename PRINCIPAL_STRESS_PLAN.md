# Principal stress postprocessing plan

Status: implemented and validated on CPU and Metal.
Branch: `adm/principal_stress`.

## Implementation and validation record

- Added exported allocating/mutating APIs and dimension-specialized KA kernels
  in `src/postprocess/principal_stresses.jl`, returning an exported
  `PrincipalStresses` container. The in-place API fills the container's existing
  value/direction arrays and returns it. They retain pointwise sampling and
  reject incompatible/aliased arrays before launching. Kernels receive component
  tuples, so the container needs no GPU adaptation dependency.
- Reused the installed StaticArrays 1.9.22 symmetric 3×3 eigensolver, which
  passed a Metal device probe, instead of introducing a custom Jacobi solver.
  Its bounded iteration has no convergence flag, so scaled residual and
  orthogonality checks guard every computed 3D eigenpair set.
- Settled numerical failure reporting: write NaN for every output of a failed
  sample, synchronize, and reduce the first value field to detect failure;
  raise `DomainError`. Mutating output is partially updated on failure.
- CPU Float32/Float64 regression checks and API/export checks passed on
  Julia 1.13.1. A deterministic additional sweep of 2,000 3D tensors per
  precision passed; warmed local eigenpair routines allocated zero bytes.
- Metal Float32 API checks passed with scalar indexing disabled on Julia
  1.13.1. CUDA/AMDGPU hardware checks are unrun, and Metal does not support
  Float64. The optional test helper documents the reusable hardware gate.
- Mathematical local names and fixed-size tuple validation keep numerical
  code concise. Both APIs pass inference and JET checks in Float32/Float64,
  2D/3D, including pressure and custom workgroups; mixed vector/view input
  signatures also pass JET. Native runtime workgroup sizing avoids launch dispatch.
- Final validation passed: 4,373 package checks on Julia 1.12.7; 568 focused
  CPU checks, 174 export checks, and 229 Metal Float32 checks on Julia 1.13.1;
  the Documenter build and documented plane-strain recipe on Julia 1.13.1.
  `git diff --check` passed. CUDA/AMDGPU hardware checks remain unrun.

The sections below retain the original design and acceptance criteria; the
implementation above resolves the eigensolver and failure-reporting decisions.

## Goal and scope

Add backend-neutral 2D and 3D postprocessing kernels that compute principal
stress values and directions from supplied symmetric stress components.
Keep this operation separate from constitutive updates, solver iteration,
tensor averaging, and mesh interpolation. No new dependency is planned.

Existing `src/postprocess/postprocess.jl` computes volume-averaged cell stress
diagnostics through CPU loops. The new operation accepts nodal, cell, or
integration-point component arrays directly, preserving their shape and backend.

## Numerical contract

- Return principal values in descending algebraic order, with corresponding
  orthonormal direction vectors. Positive values denote tension when the input
  follows the Stokes Cauchy stress convention.
- Without pressure, diagonalize the supplied tensor. With positive-compressive
  physical pressure `P`, compute total stress `σ = τ - P I`; diagonalize `τ`
  first and subtract `P` from the values to preserve directions and avoid
  unnecessary loss of deviatoric precision.
- The 2D path returns two in-plane eigenpairs. Full plane-strain principal
  stresses use the 3D path with `τzz = -(τxx + τyy)` for deviatoric Stokes stress
  and zero `τxz`, `τyz`. The out-of-plane eigenvalue participates in sorting.
- Directions are axes: `v` and `-v` are equivalent. Choose a deterministic sign
  by making the largest-magnitude component nonnegative, with fixed tie order.
- Repeated eigenvalues admit any orthonormal basis of their eigenspace. No
  unique direction or spatial/temporal continuity is promised at degeneracy.
- Preserve Float32/Float64; scale local calculations to avoid avoidable
  overflow/underflow. Handle zero and isotropic stress explicitly.
- Invalid numerical input or failed local convergence must be observable,
  never replaced with plausible eigenpairs. Settle the device-compatible
  failure reporting contract before publishing the API.

## Proposed API and layouts

```julia
compute_principal_stresses(τ; pressure=nothing, workgroup=256)
compute_principal_stresses!(out, τ; pressure=nothing, workgroup=256)
```

Accept existing `SymmetricTensor2D`/`SymmetricTensor3D` containers and tuples in
the order used by `FEMTools.stress(dr)`:

- 2D: `(xx, yy, xy)`;
- 3D: `(xx, yy, zz, xy, xz, yz)`.

Read tensor containers by named fields: their 3D `Tuple` order differs from the
stress accessor order. Their stored invariant `II` is not needed.

Return a `PrincipalStresses` struct with `values` and `directions` fields:
`values` is a tuple of two or three
component arrays, and `directions` is a tuple of matching vectors, each stored
as a tuple of two or three component arrays. Every array has the input shape.
The allocating method creates these arrays on the input backend; the mutating
method accepts a `PrincipalStresses` container, reuses caller buffers, and
returns the same `out`. `PrincipalStresses(values, directions)` wraps supplied
buffers without copying; structural validation occurs before computation.

Validate component/output shapes, supported floating types, compatible array
indexing, backend agreement, and pressure shape/type at the shared wrapper.
Pressure is initially either `nothing` or a matching array. Output buffers
must not alias inputs or one another. Support empty arrays without launching.
Choose the API tier consistently with existing exported postprocessing functions;
keep raw kernels and local eigenpair helpers internal.

## Implementation sequence

1. **Prove the 3D eigenpair primitive.** Inspect the installed StaticArrays
   symmetric eigensolver and test whether it compiles on an available
   accelerator without dynamic allocation or host LAPACK. Check degenerate
   cases. Reuse it if it passes; otherwise use a scaled, statically sized
   symmetric Jacobi method with a bounded iteration budget and convergence
   check. Record the selected method and any untested backend explicitly.
2. **Implement local 2D and 3D routines.** Use an analytic symmetric 2×2
   calculation with `hypot` and stable rotation in 2D. Sort values and vectors
   together and apply the sign convention in both dimensions.
3. **Add wrappers and kernels.** Put the implementation in
   `src/postprocess/principal_stresses.jl`, include it from `src/FEMTools.jl`,
   and launch one KA work item per stress sample. Use native KA runtime
   workgroup sizing to keep the launch fully inferred.
   Infer placement with `KernelAbstractions.get_backend`; avoid host scalar
   indexing, host copies, atomics, and dynamic per-sample allocations.
4. **Document integration.** Show calls on stored integration-point stress
   and existing cell diagnostics. Principal stresses of an averaged tensor
   differ from averaged principal stresses: do not average directions or
   silently change sampling locations. Pressure must already be collocated.
   For tensile-cap plasticity use corrected physical IP pressure from the
   stress/pressure store, not trial `dr.P` or relaxation `dr.Pnum`.
   Existing VTK `cell_data` scalar/vector support can write cell eigenpairs;
   no writer redesign or automatic integration-point projection is needed.
5. **Complete validation and API documentation.** Update API declarations,
   relevant export/public tests, docstrings, and the Stokes/API manual.
   Review all nine subsystem guides and update affected API, testing, and I/O
   notes to describe actual implemented behavior and supported backends.

## Acceptance checks

Add `test/test_principal_stresses.jl`, discovered by the existing test runner.
Use deterministic inputs and independent CPU `eigen(Symmetric(...))` oracles.

- Both dimensions and Float32/Float64: diagonal and rotated tensors, pure
  shear, zero/isotropic stress, double/triple and near-repeated eigenvalues,
  and representative small/large finite scales.
- Sorted eigenvalues, eigenpair residuals `σ*v ≈ λ*v`, orthonormality, and
  reconstruction `σ ≈ V*Diagonal(λ)*V'`, with scale-aware tolerances.
  Compare eigenspaces rather than individual vectors for repeated roots.
- A plane-strain case where the out-of-plane value changes the 3D ordering.
- Pressure shifts values without changing directions; allocating/mutating
  agreement; vector/matrix shape, precision, empty-input, and validation cases.
- CPU kernel agreement with the local reference. Run an accelerator smoke
  check with scalar indexing disabled on available hardware; report unrun
  CUDA/AMDGPU/Metal checks rather than claiming coverage.
- Check inference and warmed local allocation behavior. Benchmark only if
  making a performance claim, using representative non-isotropic tensors.

Final gates after implementation:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl
git diff --check
```

Use the persistent Kaimon session for iterative checks when available, and
Julia 1.12 for the current full-suite gate as described in `.agents/testing.md`.

## Stop condition

Both dimensions return correct paired eigenvalues/directions, preserve shape,
precision and backend, and satisfy the documented degeneracy/failure contract.
Tests and docs pass, supported accelerator evidence is recorded, and affected
living guides agree with the implementation. Solver state/history, existing
diagnostic output contracts, and VTK writers need no behavioral changes.
