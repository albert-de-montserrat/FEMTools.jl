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

## Traction and pressure

A natural (Neumann) momentum condition is a load on a set of boundary edges of
a 2-D mesh. [`boundary_facets`](@ref) selects the edges once; a
[`TractionBoundaryCondition`](@ref) or [`PressureBoundaryCondition`](@ref)
pairs them with a constant or position-dependent load and is passed to the
Stokes solver through its `traction` keyword:

```julia
surface = boundary_facets(mesh_stokes, surface_nodes)

# Constant pressure, or one that varies along the boundary.
bc = PressureBoundaryCondition(surface, 1.0e5)
bc = PressureBoundaryCondition(surface, x -> ρ_ice * g * ice_thickness(x[1]))

solve_stokes_dyrel!(dr, mesh_stokes, bc_vx, bc_vy, Δt, γP; traction = bc)
```

The load is fixed during one solve. To change it between time steps, build a
new condition from the same facets:

```julia
for istep in 1:nsteps
    bc = PressureBoundaryCondition(surface, x -> load(x, istep * Δt))
    solve_stokes_dyrel!(dr, mesh_stokes, bc_vx, bc_vy, Δt, γP; traction = bc)
end
```

```@docs
BoundaryFacets
boundary_facets
TractionBoundaryCondition
PressureBoundaryCondition
FEMTools.assemble_boundary_load
FEMTools.assemble_boundary_load!
```
