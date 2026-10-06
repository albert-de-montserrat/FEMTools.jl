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
- Unstructured boundary-node detection supports T3, Q4, T6/T7, Q8/Q9, Tet4,
  Hex8, and Hex27 topology. It identifies exterior entities by occurrence
  count; it does not preserve named boundary groups.
- `MixedMesh` stores distinct velocity and pressure connectivity. Its common
  construction path derives discontinuous linear pressure connectivity and
  nodal normals from a velocity mesh.
- `MixedMeshCache` stores velocity/pressure geometry and the corresponding
  reference elements. A `MixedMesh` built from an element-aware velocity mesh
  keeps one in `mesh.geometry`; `update_geometry!` refills it in place.
- Sparsity, node-to-element adjacency, greedy coloring, color groups, and
  discontinuous linear mesh generation are available.
- `generate_element_adjacency(...; shared=:face)` throws for 4- and 8-node
  connectivity without `dimension`: those counts are ambiguous (Quad4/Tet4,
  Quad8/Hex8).
- `src/mesh/utils.jl` holds the host-side external-mesh helpers:
  `renumber_connectivity`, `orient_triangle_elements!`, `add_t7_bubbles!`,
  `straighten_t7_geometry!`, `rectangle_boundary_nodes`, and
  `circle_boundary_nodes`, plus the `triangulate_t7_mesh` stub whose method the
  Triangulate extension supplies. They are `public`, not exported, and cover tag
  renumbering, triangle orientation, T6-to-T7 promotion, coordinate-based
  boundary selection, and 2-D mesh generation.
- Two external meshers are in use. Gmsh covers both 2-D triangles and 3-D
  hexahedra from the examples environment; the shared `examples/gmsh_meshing.jl`
  helper drives the utilities above to convert its triangle tags and ordering to
  FEMTools connectivity. Reading mesh files is still outside the package API;
  only the post-import conversion is package code.
- Triangulate covers 2-D triangles only and is now package code behind an
  extension. `Triangulate` is a weak dependency, `ext/FEMToolsTriangulateExt.jl`
  supplies the single method `triangulate_t7_mesh(points; max_area, min_angle,
  segments, regions)`, and the stub in `src/mesh/utils.jl` carries the
  docstring and raises a message naming the missing `using Triangulate`. The
  core package therefore still installs no mesher and no `Triangle_jll`.
  `min_angle` is rejected at 34 degrees and above, because beyond that Triangle
  can refine forever instead of failing.
- `segments` and `regions` cover the conforming-interface case. Without them
  `points` is a simple polygon and the closing segment is added automatically;
  with them the caller describes any planar straight-line graph and names the
  subdomains it encloses, and the returned per-element attributes say which
  subdomain each element is in. Two Triangle details matter. A numeric `a` flag
  overrides per-region area constraints, so with regions the flag carries no
  number and every region is defaulted to `max_area` instead. And a per-region
  constraint does not leave its neighbour untouched, because quality refinement
  propagates across the shared interface.
- Either mesher feeds the same conversion path. Triangle's `o2` output lists
  the midside node opposite each corner, so rows 4-6 need the remap
  `(6, 4, 5)` before `add_t7_bubbles!`. The extension does that remap, then
  `add_t7_bubbles!` and `orient_triangle_elements!`, and returns `Float64`
  coordinates with `Int32` T7 connectivity. Boundary selection stays with the
  caller, which knows the domain shape. Do not hand-roll centroid insertion,
  the midside remap, or boundary selection in a new script; `ice_bridge_2D` and
  `popov_extension_2D` both call the extension.

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
- `ext/FEMToolsTriangulateExt.jl`

Primary checks: `test/test_elements.jl`,
`test/test_shape_function_evaluations.jl`, `test/test_mesh.jl`,
`test/test_mixed_mesh.jl`, `test/test_mesh_producer_api.jl`, and
`test/test_mesh_utils.jl`, and `test/test_triangulate_mesh_ext.jl`.
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
- Geometry stores one `QuadraturePointGeometry` (inverse Jacobian and weighted
  volume) per element and quadrature point; `element_geometry(geo, iel, ∂N∂ξ)`
  forms physical gradients from it on access.
  Reject singular or inverted mappings where the
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
once repeated real use justifies it. Triangulate reached that bar with two
miniapps duplicating the same PSLG-to-T7 conversion, so it entered as a weak
dependency and an extension, not as a core dependency. Gmsh has not: it is
still only a test and examples dependency.

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

## Q2/P1 quadrilateral mixed meshes

`MixedMesh` with Q9 velocity and `LinearElement{2,3}` pressure stores three
discontinuous pressure values per cell at local velocity nodes `(9,6,7)`.
The basis `(1-ξ-η,ξ,η)` spans linear polynomials on the reference square.
Pressure volume weights come from Q9 velocity geometry, including deformed
cells; pressure coordinates are not a separate triangular integration domain.
Triangle topology and geometry retain their existing paths.

## Update this guide when

- an element family, ordering, mesh field, importer, tag model, or boundary
  representation changes;
- partitioning adds global/local/ghost topology;
- geometry storage or backend placement changes;
- a previously example-only meshing path becomes supported package API;
- a limitation above is removed or a new one is discovered.
