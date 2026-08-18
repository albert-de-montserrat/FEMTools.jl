# Sinking block

The sinking-block examples place a dense, viscous inclusion in a lighter
matrix under gravity. Free-slip walls suppress normal flow while allowing
tangential motion. They provide compact forward and discrete-adjoint checks in
two and three dimensions.

## Two-dimensional example

The 2-D problem uses an unstructured, interface-conforming triangular mesh.
Quadratic T7 velocity functions are paired with three cell-local P1 pressure
unknowns. The package's Powell--Hestenes/DYREL Stokes solver advances the
velocity and pressure residuals, and the companion adjoint example applies the
transpose of that same discrete operator.

```sh
julia --project=examples examples/stokes/sinking_block/sinking_block.jl
julia --project=examples examples/stokes/sinking_block/sinking_block_adj.jl
```

The forward script writes `stokes_2D_sinking_block.vtk`. Its main parameters
include the maximum triangle area, phase density and viscosity, solver
tolerance, and free-slip boundary conditions. The adjoint script defines an
observation-region velocity objective and reports density and viscosity
sensitivities.

## Three-dimensional example

The 3-D pair has its own page: [Three-dimensional sinking block](@ref).

## Choosing an example

Use the 2-D pair for rheology. The 3-D pair is linear viscous and provides
matrix-free forward and adjoint solvers.
