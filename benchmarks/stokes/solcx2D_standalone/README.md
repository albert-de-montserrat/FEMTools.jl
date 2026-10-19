# SolCx triangle with a local DYREL solver

Run from the repository root:

```sh
julia --project=examples benchmarks/stokes/solcx2D_standalone/SolCx2D_triangle.jl
```

Set `FEMTOOLS_BENCHMARK_PLOTS=false` for headless execution and
`FEMTOOLS_BENCHMARK_HISTORY=false` to suppress the JLD2 archive. Running or
including the script executes its default case; `main` then accepts the same
resolution, contrast, tolerance, backend, and output controls as before.

`dyrel.jl` contains the complete numerical solve in the `SolCxDYREL` module:
pressure mass/scaling, physical and augmented momentum residuals, continuity
residual, viscous stress, the ForwardDiff augmented-Jacobian preconditioner,
atomic scatter, boundary constraints, spectral estimates, velocity updates,
Powell–Hestenes pressure updates, convergence checks, and history recording.
It imports no FEMTools code and includes no files from `src/`.

`dyrel_setup.jl` is the setup boundary. FEMTools still constructs reference
elements, meshes, geometry, material/state buffers, and boundary data. This
helper borrows those arrays and prepares quadrature tables, then passes ordinary
NamedTuples to the solver. Analytical comparisons, plotting, archives, and VTK
output retain the existing benchmark helpers and FEMTools output routines.

The extraction specializes the package's algorithm to the triangle miniapp:
purely viscous incompressible 2-D T7 velocity, discontinuous P1 pressure,
one phase per cell, assembled body force, and a frozen augmented Jacobian with
Gershgorin spectral bounds. Plasticity, elasticity, thermal coupling, finite
bulk storage, power-iteration estimates, and the quadrilateral driver are
outside this extraction. The kernels remain backend-neutral; accelerator
coverage requires a separate hardware run.

The local solver is called explicitly as `SolCxDYREL.solve_stokes_dyrel!`;
editing the package solver does not change it. Changes to the local method
must be maintained independently. `test/test_standalone_solcx_dyrel.jl` compares
its fields, pressure scaling, and preconditioner with the package on a small
heterogeneous mesh, including 1e6 viscosity contrast, Float32, prescribed
nonzero boundaries, and deliberate non-convergence.
