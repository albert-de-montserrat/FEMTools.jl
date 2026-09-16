# Mesh

FEMTools.jl includes a structured mesh container for 1-D, 2-D, and 3-D
domains and helper functions for connectivity, geometry precomputation, and
graph coloring.

## Mesh Container

```@docs
FEMTools.AbstractMesh
Mesh
MixedMesh
MixedMeshCache
update_geometry!
generate_discontinuous_linear_mesh
```

Backend-aware constructors move every kernel-consumed mesh array to the target
device:

```julia
mesh_v = Mesh(backend, coords_cpu, el2n_cpu, velocity_element)
mesh   = MixedMesh(mesh_v, pressure_element; workgroup)
```

An element-aware `Mesh` stores its reference element and precomputed geometry;
single-field thermal and lithostatic solvers therefore need only the mesh and a
boundary-condition object. `Mesh(backend, coords, el2n; order)` remains
available when geometry is not needed.

`MixedMesh(mesh_v, pressure_element)` constructs discontinuous pressure
connectivity and nodal normals on the CPU, then returns them on the same array
backend as `mesh_v`. Because `mesh_v` stores its reference element, the mixed
mesh also computes both fields' geometry on that backend and keeps it, with both
reference elements, as a [`MixedMeshCache`](@ref) in `mesh.geometry`. High-level
Stokes calls take geometry, elements, and backend from the mesh alone. After
moving `mesh.coords`, call `update_geometry!(mesh)` to recompute the geometry in
place. Avoid mixing host connectivity with device solution arrays.

## Coordinates and Degrees of Freedom

```@docs
generate_coordinates
generate_dofs
```

## Connectivity

```@docs
generate_element2node
generate_node2element
generate_boundary_elements
FEMTools.element_coordinate_matrix
FEMTools._unstructured_boundary_nodes
```

## External Mesh Ingestion

Meshes produced by an external generator arrive with arbitrary node tags,
generator-specific node ordering, and no bubble nodes. These helpers convert
such input to the layout the solvers expect: one-based `Int32` connectivity with
one element per column, counter-clockwise triangles, and straight-sided T7
geometry. They run on the host, before the backend-aware `Mesh` constructor
uploads the result.

```@docs
FEMTools.renumber_connectivity
FEMTools.orient_triangle_elements!
FEMTools.add_t7_bubbles!
FEMTools.straighten_t7_geometry!
```

`add_t7_bubbles!` appends one centroid node per element to `coords`, so pass a
resizable host vector. `straighten_t7_geometry!` recomputes the edge and bubble
coordinates from the element corners, which is what keeps a T7 mesh consistent
after the corner nodes are moved — by a generator that curved the high-order
nodes, or by Lagrangian mesh advection.

Boundary nodes for the common analytic geometries are selected by coordinate
tolerance rather than by generator tags, which the package does not preserve:

```@docs
FEMTools.rectangle_boundary_nodes
FEMTools.circle_boundary_nodes
```

Both default `atol` to `sqrt(eps(Float64))` scaled by the largest supplied
coordinate. `examples/gmsh_meshing.jl` chains these helpers into a complete
Gmsh-to-FEMTools triangle pipeline.

## Sparsity and Coloring

```@docs
generate_sparsity_pattern
color_mesh
FEMTools.color_mesh_greedy
generate_element_groups
```

## Geometry Precomputation

Element geometry holds one [`QuadraturePointGeometry`](@ref) per element and
quadrature point: the inverse isoparametric Jacobian and the weighted measure
`|det J| ωq`. `element_geometry(geo, iel, ∂N∂ξ)` pairs an element's entries with
the reference-element gradients `∂N∂ξ = shape_function_gradients(element)` and
forms the physical shape-function gradients on access.

```@docs
precompute_geometry
QuadraturePointGeometry
ElementGeometry
element_geometry
FEMTools.precompute_geometry_kernel!
FEMTools.element_geometry
```

## VTK Output

```@docs
write_vtk
```
