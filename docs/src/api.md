# API

```@index
Modules = [FEMTools]
```

## Heat Diffusion

```@docs
ThermalDiffusionDR
solver!
apply_dirichlet!
update_rate_kernel!
update_variable_kernel!
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
MixedMesh
build_discontinuous_linear_mesh
build_element_groups
FEMTools.element_coordinate_matrix
FEMTools._unstructured_boundary_nodes
```

## Stokes

```@docs
StokesDR
DruckerPrager
assemble_momentum_residual_matrices_atomix!
assemble_momentum_jacobian_matrices_atomix!
assemble_augmented_momentum_jacobian_matrices_atomix!
assemble_pressure_residual_matrices_atomix!
FEMTools.momentum_element_residual
FEMTools.element_momentum_jacobians
FEMTools.element_augmented_momentum_jacobians
FEMTools.pressure_element_residual
FEMTools.integrate_momentum_residual
FEMTools.integrate_PH_pressure_residual
FEMTools.compute_velocity_divergence
FEMTools.second_invariant
FEMTools.deviatoric_stress
FEMTools.viscoelastic_coefficients_phase
```

## Shape Function Helpers

```@docs
shape_function_values
FEMTools.interp2ip
FEMTools.interp2ip_phase
FEMTools._eval_shape_function
```
