# Mesh

FEMTools.jl includes a small one-dimensional mesh container and helper functions
for connectivity.

## Mesh Container

```@docs
FEMTools.AbstractMesh
Mesh
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
```

## Sparsity and Coloring

```@docs
generate_sparsity_pattern
color_mesh
color_mesh_greedy
```
