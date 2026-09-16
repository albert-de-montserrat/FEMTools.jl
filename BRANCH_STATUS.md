# Branch status: `adm/mixedmesh3d`

Goal: extend the mixed-mesh Stokes solver (T7/P1-disc velocity–pressure
pairing already used in 2-D) to 3-D, pairing a new Tet10+centroid-bubble
("T11") velocity element with P1-discontinuous tetrahedral pressure.

## Done

- **New element**: `QuadraticElement{3, 11}` (Tet10 + centroid bubble).
  Shape functions/gradients and integration rule already existed; this
  branch added connectivity (`generate_element2node`) and coordinates
  (`generate_coordinates`) for a structured Kuhn/Freudenthal-split
  tetrahedral mesh — [connectivity.jl](src/mesh/connectivity.jl),
  [mesh.jl](src/mesh/mesh.jl).
- **3-D `MixedMesh`**: `MixedMeshCache` 3-D method, and
  `generate_discontinuous_linear_mesh` generalized from triangle-only to any
  simplex (`nDim + 1` corners) so it now covers T3/T6/T7 and
  Tet4/Tet10/Tet11 — [mixed_mesh.jl](src/mesh/mixed_mesh.jl). 3-D free-slip
  BCs use axis-aligned fixed-node sets rather than the 2-D node-normal
  rotation, so `_compute_node_normals` is a documented zero-vector stub for
  `SVector{3}` coordinates.
- **Assembly**: `stokes_preconditioner_mixedmesh_3d` (diagonal Jacobi
  preconditioner + lumped pressure mass) and the matching momentum/pressure
  residual assembly for the T11/P1-disc pair —
  [momentum_residuals.jl](src/stokes/assemblers/momentum_residuals.jl),
  [pressure_residual.jl](src/stokes/assemblers/pressure_residual.jl).
- **Solver**: new `solve_stokes_dyrel!(dr::StokesDR{<:Any,3}, mesh_stokes::MixedMesh, ...)`
  dispatch — damped-Jacobi velocity update + explicit Arrow–Hurwicz pressure
  update, purely viscous only (no thermal/compressibility coupling yet) —
  [DR.jl](src/stokes/solvers/DR.jl).
- **Test**: `test/test_stokes_3d_mixedmesh_reference.jl` — DYREL solve vs. a
  hand-assembled sparse oracle, plus a matrix-free-vs-sparse residual check
  and a direct-solve comparison. Was failing on 2 of 4 assertions because the
  free-slip fixed-node sets didn't match between the DYREL call and the
  oracle (same `Γnodes` set was reused for all three velocity components,
  i.e. accidental no-slip); fixed by building matching per-component
  free-slip node sets. All 4 assertions pass now.
- **Docs**: added a "Supported Elements" catalog page
  ([elements_catalog.md](docs/src/elements_catalog.md)) with SVG sketches of
  every 2-D/3-D element, including the new T11. Along the way, fixed a
  broken `docs/make.jl` build: `stokes_preconditioner_mixedmesh_3d`'s
  docstring referenced `stokes_preconditioner_3d` via `@ref`, but that
  (pre-existing, main-branch) function had no docstring at all and wasn't
  listed in any `@docs` block — added the missing docstring and both
  `@docs` entries in [stokes.md](docs/src/stokes.md).

## Verified

- Full test suite: 2847 passed / 7 failed / 1 errored, identical failure set
  to a clean `main` worktree (Gmsh native-library init flake in the
  sinking-block Hex27 test, JET type-stability checks sensitive to package
  versions) — none of it caused by this branch.
- `julia --project=docs docs/make.jl` builds clean (only the expected
  "could not auto-detect deployment environment" warning for a local build).

## Left to do

- **No example/miniapp**: every 3-D directory under
  `examples/miniapps/stokes/` (`sinking_block_3D`, `sinking_sphere_block_3D`,
  ...) still uses the raw-array Hex27/P1 solver. There is no runnable
  end-to-end example for the new Tet11/`MixedMesh` 3-D path, only the
  reference test.
- **Structured meshes only**: the new element only has a structured
  Cartesian generator (`Mesh(CPU(), domain, element_v, (nx,ny,nz))`). Unlike
  Hex27 (via `build_gmsh_hex27_mesh`), there's no gmsh/unstructured mesh
  path for T11 tetrahedra yet, so it's currently limited to box domains.
- **No unit-level `MixedMesh` 3-D coverage**: `test/test_mixed_mesh.jl` (the
  generic constructor/cache/connectivity unit tests) has zero 3-D cases —
  everything for 3-D currently lives in the one big DYREL/oracle integration
  test. Things like the 3-D "rejects inconsistent pressure connectivity"
  path, or `_compute_node_normals`'s zero-vector stub, are untested.
- **GPU untested**: all new kernels (`precompute_geometry_kernel!` 3-D
  dispatch, `stokes_preconditioner_mixedmesh_3d_kernel!`, the momentum/
  pressure assembly) are only exercised on `CPU()` in tests. Worth an
  explicit GPU-audit pass before relying on this off CPU.
- **No adjoint**: `solve_stokes_adjoint_dyrel!` / `solve_stokes_adjoint_3d!`
  have no `MixedMesh` 3-D counterpart.
- **Narrative docs still 2-D-centric**: `docs/src/stokes.md`'s "3-D
  Hex27/Q2–P1 method" paragraph and `docs/src/mesh.md`'s `MixedMesh` example
  don't mention the new T11/P1-disc option at all — only the `@docs`
  API listings cover it.
- Everything above is currently **uncommitted** in the working tree.
