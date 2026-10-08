# Boundary Conditions

FEMTools.jl provides small containers and helpers for boundary conditions.

## Types

```@docs
FEMTools.AbstractBoundaryCondition
DirichletBoundaryCondition
```

## Application

```@docs
apply_bc!
```

## Construction

Use `DirichletBoundaryCondition(nodes, values)` when no boundary label is
needed. The optional three-argument form retains a label for reference.
Indices and values are retained without copying and must have equal lengths.
Keep both arrays on the solver backend; homogeneous residual values are
allocated on the same backend as the prescribed values.
