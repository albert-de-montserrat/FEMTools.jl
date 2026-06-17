# API

```@index
Modules = [FEMTools]
```

## Heat Diffusion

```@docs
solver!
apply_dirichlet!
assemble_diffusion_matrices_atomix!
assemble_diffusion_matrices_colored!
FEMTools.element_residual
FEMTools.element_jacobian
FEMTools.integrate_residual
```

## Lithostatic Pressure

```@docs
LithostaticPressureDR
assemble_lithostatic_pressure_matrices_atomix!
assemble_lithostatic_pressure_matrices_colored!
FEMTools.lp_element_residual
FEMTools.lp_element_jacobian
FEMTools.lp_integrate_residual
```

## Mesh Helpers

```@docs
build_element_groups
FEMTools.element_coordinate_matrix
FEMTools._unstructured_boundary_nodes
```

## Shape Function Helpers

```@docs
shape_function_values
FEMTools._eval_shape_function
```
