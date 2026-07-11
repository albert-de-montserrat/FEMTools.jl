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
generate_discontinuous_linear_mesh
```

Backend-aware constructors move every kernel-consumed mesh array to the target
device:

```julia
mesh_v = Mesh(backend, coords_cpu, el2n_cpu; order = 2)
mesh   = MixedMesh(mesh_v, pressure_element)
cache  = MixedMeshCache(backend, workgroup, mesh, velocity_element, pressure_element)
```

`MixedMesh(mesh_v, pressure_element)` constructs discontinuous pressure
connectivity and nodal normals on the CPU, then returns them on the same array
backend as `mesh_v`. `MixedMeshCache` allocates both geometry arrays on
`backend`; its `geo_v` and `geo_P` must be passed together with that mesh and
backend to the Stokes assemblers. Avoid mixing host connectivity with device
solution arrays.

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

## Sparsity and Coloring

```@docs
generate_sparsity_pattern
color_mesh
FEMTools.color_mesh_greedy
generate_element_groups
```

## Geometry Precomputation

```@docs
FEMTools.precompute_geometry_kernel!
```

## VTK Output

```@docs
write_vtk
```
