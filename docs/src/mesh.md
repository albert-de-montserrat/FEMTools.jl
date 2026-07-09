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
