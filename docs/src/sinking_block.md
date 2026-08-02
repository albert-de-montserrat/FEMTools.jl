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

The 3-D domain is
``[0,1]\times[-1,0]\times[0,1]``. Gmsh creates a recombined quadrilateral
surface mesh, extrudes it vertically, and raises the resulting hexahedra to
Hex27. A fixed permutation converts Gmsh's type-12 node ordering to the
[`QuadraticElement{3,27}`](@ref) ordering used by FEMTools.

Velocity uses continuous Q2 functions with 27 nodes per cell. Pressure uses
the four discontinuous P1 modes

```math
(1,\xi,\eta,\zeta),
```

so every cell owns four pressure unknowns. The example assembles the symmetric
saddle-point matrix directly with `SparseArrays`, fixes one pressure gauge
unknown, applies free slip on all six walls, and solves the free degrees of
freedom on the CPU.

```sh
julia --project=examples examples/stokes/sinking_block/sinking_block_3D.jl
```

The default problem contains 225 Hex27 cells and 2,255 velocity nodes. Its
7,665-by-7,665 system reaches a free-degree-of-freedom residual of about
``2.5\times10^{-15}``. The script writes `stokes_3D_sinking_block.vtk` with
the three velocity components, cell-centre pressure, and material phase.

[`solve_stokes_3d!`](@ref) provides the matrix-free package path for the same
viscous Hex27/Q2--P1 discretization. It updates caller-owned velocity and
pressure arrays with a diagonally preconditioned Uzawa iteration; the sparse
solve remains the regression oracle.

![Orthogonal volume slices through the 3-D vertical-velocity field at the block centre. The orange wireframe marks the dense, viscous block.](assets/stokes_sinking_block_3d.png)

The blue region around the block has negative vertical velocity, while the
wall-normal velocity is exactly zero on every boundary face. For this figure,
the unstructured nodal field is sampled to a regular ``51^3`` visualization
grid and sliced through the block centre.

### Discrete adjoint

For the linear system

```math
A(m)u=b(m), \qquad J=c^T u,
```

the 3-D adjoint reuses the exact forward matrix and solves

```math
A^T\lambda=c.
```

The material gradient is

```math
\frac{\mathrm dJ}{\mathrm dm}
=\lambda^T\left(\frac{\partial b}{\partial m}
-\frac{\partial A}{\partial m}u\right).
```

Density changes the gravity load, whereas viscosity changes the phase-2
viscous matrix contribution. The executable checks both contractions against
centred finite differences:

```sh
julia --project=examples examples/stokes/sinking_block/sinking_block_3D_adj.jl
```

With the default mesh, the relative differences are approximately
``4.4\times10^{-11}`` for block density and ``6.9\times10^{-7}`` for block
viscosity. The forward command checks its free-degree-of-freedom residual; the
adjoint command raises an error if either gradient check exceeds its tolerance.

## Choosing an example

Use the 2-D pair for rheology and the iterative adjoint. Use the 3-D pair for
the viscous matrix-free forward solver and exact sparse forward/transpose
references.
