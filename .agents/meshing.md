# Meshing

## Current reality

Meshing spans `src/elements/` and `src/mesh/`:

- `LinearElement`, `QuadraticElement`, and `CubicElement` are lightweight tags;
  implemented reference elements are assembled by `ReferenceElement` from
  shape functions and integration points.
- Structured connectivity and coordinates exist for selected 1-D, 2-D, and
  3-D linear/quadratic elements.
- `Mesh` supports structured construction from a `DomainSets` domain and
  unstructured construction from caller-provided coordinates/connectivity.
- Element-aware mesh constructors precompute geometry for single-field
  solvers. Constructors without an element retain `nothing` for element and
  geometry.
- Unstructured boundary-node detection supports T3, Q4, T6/T7, Q8/Q9,
  Tet4/T10/T11, Hex8, and Hex27 topology. It identifies exterior entities by occurrence
  count; it does not preserve named boundary groups.
- `MixedMesh` stores distinct velocity and pressure connectivity. Its common
  construction path derives discontinuous linear pressure connectivity and
  nodal normals from a velocity mesh.
- `MixedMeshCache` stores velocity/pressure geometry and the corresponding
  reference elements.
- In three dimensions, `MixedMesh` supports T10/T11 tetrahedral or Hex27
  velocity connectivity with four discontinuous linear pressure DoFs per cell;
  3-D nodal normals are currently
  zero because no solver path consumes them.
- Sparsity, node-to-element adjacency, greedy coloring, color groups, and
  discontinuous linear mesh generation are available.
- `src/mesh/utils.jl` holds the host-side external-mesh helpers:
  `renumber_connectivity`, `orient_triangle_elements!`, `add_t7_bubbles!`,
  `straighten_t7_geometry!`, `rectangle_boundary_nodes`, and
  `circle_boundary_nodes`. They are `public`, not exported, and cover tag
  renumbering, triangle orientation, T6-to-T7 promotion, and coordinate-based
  boundary selection.
- Gmsh is the sole external mesher in the examples environment. The shared
  `examples/gmsh_meshing.jl` helper drives the utilities above to convert its
  triangle tags/order to FEMTools connectivity. Reading mesh files is still
  outside the package API; only the post-import conversion is package code.

Primary code:

- `src/elements/elements.jl`
- `src/elements/shape_functions.jl`
- `src/elements/integration_points.jl`
- `src/mesh/connectivity.jl`
- `src/mesh/mesh.jl`
- `src/mesh/mixed_mesh.jl`
- `src/mesh/sparsity.jl`
- `src/mesh/coloring.jl`
- `src/mesh/utils.jl`

Primary checks: `test/test_elements.jl`,
`test/test_shape_function_evaluations.jl`, `test/test_mesh.jl`,
`test/test_mixed_mesh.jl`, `test/test_mesh_producer_api.jl`, and
`test/test_mesh_utils.jl`.
`examples/gmsh_meshing.jl` defines mesh builders only; it is exercised by the
example scripts that include it, not by the package test suite.

## Invariants

- `el2n` has local nodes in rows and elements in columns.
- All connectivity exposed to Julia code is 1-based. Preserve the integer type
  where practical; generated connectivity commonly uses `Int32`.
- Coordinate ordering, connectivity ordering, shape functions, quadrature, and
  VTK corner selection form one contract. A node-order change must update and
  test every part together.
- Boundary detection must include high-order edge/face nodes while using a
  canonical corner/entity key to count shared entities.
- `mesh.nnodes == length(mesh.coords)` and `mesh.nels == size(mesh.el2n, 2)`.
- Mixed velocity and pressure connectivities must describe the same number of
  elements.
- Geometry stores physical gradients and integration weights for each element
  and quadrature point. Reject singular or inverted mappings where the
  mathematical path cannot support them.
- Kernel-consumed mesh arrays must be moved to the selected backend together.
  Host-side topology preparation is acceptable; mixed host/device solver input
  is not.
- Preserve `Float32` and `Float64` behavior and type inference in element-local
  operations.

## Direction

Polish the existing mesh model before broadening it. The next useful increments
should be driven by a solver or I/O requirement, not a universal mesh
abstraction.

Priorities:

1. Keep mesh-file reading in the examples helper; extend `src/mesh/utils.jl`
   only for a concrete topology a maintained miniapp actually ingests.
2. Represent boundary entities and user labels when boundary conditions or
   distributed ownership need more than the current flat `Γnodes` list.
3. Define stable global entity IDs before distributed partitioning; do not infer
   them later from rank-local ordering.
4. Add element families only with a complete shape-function, quadrature,
   geometry, boundary, output, and test story.
5. Measure coloring and adjacency construction before replacing the current
   simple host algorithms.

Do not add a mandatory mesher dependency to the core package merely to remove
a few lines from examples. Prefer a small import boundary or optional extension
once repeated real use justifies it.

## Acceptance checks

For a mesh change, cover the smallest applicable set:

- exact coordinate/connectivity ordering on a tiny mesh;
- partition of unity and gradient consistency for an element change;
- boundary nodes/entities for interior-sharing and high-order cases;
- invalid arity, dimension, ordering, or geometry rejection;
- `Float32` and `Float64` preservation;
- structured and unstructured constructors where both share the changed path;
- backend construction when device placement changes;
- mixed-mesh velocity/pressure consistency;
- valid coloring: elements sharing a node never share a color.

Run the full package suite after changing ordering, element definitions,
geometry, or public mesh constructors because those contracts feed every
solver and writer.

## Update this guide when

- an element family, ordering, mesh field, importer, tag model, or boundary
  representation changes;
- partitioning adds global/local/ghost topology;
- geometry storage or backend placement changes;
- a previously example-only meshing path becomes supported package API;
- a limitation above is removed or a new one is discovered.
