module FEMTools

@doc """
    FEMTools

Finite-element utilities for structured and unstructured meshes, reference
elements, shape functions, integration points, sparsity construction, mesh
coloring, and simple boundary-condition handling.

The package is organized around lightweight reference-element tags such as
`LinearElement{2, 4, Float64}` and `QuadraticElement{3, 27, Float64}`. A
`ReferenceElement` bundles the tag-specific shape functions and quadrature rule;
`Mesh` uses that reference element to generate structured connectivity and
coordinates. GPU backend dispatch is handled via `TA(backend)`, which returns
the array constructor for the target compute backend (e.g. `CuArray` for CUDA).
""" FEMTools

using Printf
using ForwardDiff
using Enzyme
using Atomix
using StaticArrays
using DomainSets
using SparseArrays
using KernelAbstractions
import KernelAbstractions as KA
using LinearAlgebra

"""
    TA(backend) -> Array type

Return the array constructor associated with `backend`.

| Backend        | Array type |
|:-------------- |:---------- |
| `CPU()`        | `Array`    |
| `CUDABackend()`| `CuArray`  |
| `ROCBackend()` | `ROCArray` |
| `MetalBackend()`| `MtlArray`|

GPU backends are registered by the corresponding package extension and are only
available when that package is loaded. For CUDA, load `CUDA` and pass
`CUDA.CUDABackend()`; the extension maps it to `CUDA.CuArray`. This dispatch can
be queried without a GPU, but allocating a `CuArray` requires a functional CUDA
driver.
"""
TA(::CPU) = Array

include("utils.jl")

# Field containers are used by the physics modules below.
include("tensor_fields/symmetric_tensor.jl")
include("vector_fields/vector_fields.jl")

# Reference-element definitions come first because the remaining source files
# extend constructors and methods specialized on these tags.
include("elements/elements.jl")
include("elements/shape_functions.jl")
include("elements/integration_points.jl")

# Mesh helpers are kept separate by responsibility: connectivity, coordinates,
# sparsity patterns, and conflict-free element coloring.
include("mesh/connectivity.jl")
include("mesh/mesh.jl")
include("mesh/mixed_mesh.jl")
include("mesh/sparsity.jl")
include("mesh/coloring.jl")

# Boundary-condition containers and in-place application routines.
include("boundary_conditions/boundary_conditions.jl")
include("boundary_conditions/apply.jl")

# Dynamic relaxation: the scatter kernels, Dirichlet enforcement, update
# kernels, and pseudo-transient loop shared by every DR problem.
include("dynamic_relaxation/solver.jl")
include("dynamic_relaxation/assembly.jl")

# Heat-diffusion solver: types, assembly, and PT solver.
include("heat_diffusion/types/heat_diffusion_types.jl")
include("heat_diffusion/assembly/residual.jl")
include("heat_diffusion/assembly/residual_atomics.jl")
include("heat_diffusion/assembly/residual_colored.jl")
include("heat_diffusion/solvers/DR.jl")

# Lithostatic-pressure solver: types, assembly, and PT solver.
include("lithostatic_pressure/types/lithostatic_pressure_types.jl")
include("lithostatic_pressure/assembly/residual.jl")
include("lithostatic_pressure/assembly/residual_atomics.jl")
include("lithostatic_pressure/assembly/residual_colored.jl")
include("lithostatic_pressure/solvers/DR.jl")

# Stokes solver: types and assembly.
include("stokes/types/stokes_types.jl")
include("stokes/assemblers/pressure_residual.jl")
include("stokes/assemblers/pressure_residual_adj.jl")
include("stokes/assemblers/pressure_scaling.jl")
include("stokes/assemblers/rheology.jl")
include("stokes/assemblers/momentum_residuals.jl")
include("stokes/assemblers/momentum_residuals_adj.jl")
include("stokes/assemblers/adjoint_operator.jl")
include("stokes/helpers.jl")
include("stokes/stress_rotation.jl")
include("stokes/solvers/DR.jl")
include("stokes/solvers/DR_adjoint.jl")

# Post-processing: strain-rate/stress diagnostics and VTK output of solver results.
include("postprocess/postprocess.jl")

# Public type hierarchy and constructors.
export AbstractElement
export AbstractLinearElement,
    AbstractQuadraticElement,
    AbstractShapeFunction,
    AbstractIntegrationPoints,
    AbstractMesh,
    AbstractBoundaryCondition
export LinearElement, QuadraticElement, CubicElement
export ReferenceElement, ShapeFunctions, IntegrationPoints
export order
export Mesh, MixedMesh, MixedMeshCache
export DirichletBoundaryCondition, apply_bc!

# Mesh construction and graph utilities.
export generate_element2node,
    generate_node2element,
    generate_boundary_elements,
    generate_coordinates,
    generate_dofs,
    precompute_geometry,
    generate_sparsity_pattern,
    color_mesh,
    generate_element_groups,
    generate_discontinuous_linear_mesh

# Reference-element evaluation helpers.
export eval_shape_function,
    eval_shape_function_gradient,
    eval_shape_function_jacobian,
    shape_function_values,
    gauss_legendre_triangle

# Solver types and user-facing entry points.
export ThermalMaterial, ThermalDiffusionDR, solver!
export LithostaticPressureDR
export StokesMaterial, StokesDR, DruckerPrager,
    assemble_viscosity_weighted_pressure_scaling!,
    pressure_mass,
    rotate_stress!,
    solve_stokes_dyrel!,
    solve_stokes_3d!,
    solve_stokes_adjoint_3d!,
    stokes_material_gradient_3d,
    solve_stokes_adjoint_dyrel!,
    update_stokes_current_stress!

# Post-processing: strain-rate/stress diagnostics and VTK output.
export compute_strain_rate_stress_postprocess,
    update_old_stress_from_cells!,
    write_vtk,
    write_stokes_vtk

# Extension-API surface: documented and callable as `FEMTools.foo`, but not
# brought into scope by `using FEMTools`. Assembly kernels, boundary-condition
# application, the raw KernelAbstractions kernels, the DR-loop step wrappers, and
# solver-loop utilities live here rather than in the exported user API.
public TA, interp2ip, interp2ip_phase
public apply_dirichlet!
public assemble_diffusion_matrices_atomix!,
    assemble_diffusion_matrices_colored!,
    assemble_lithostatic_pressure_matrices_atomix!,
    assemble_lithostatic_pressure_matrices_colored!,
    assemble_momentum_residual_matrices_atomix!,
    assemble_momentum_residual_kernel!,
    assemble_momentum_jacobian_matrices_atomix!,
    assemble_augmented_momentum_jacobian_matrices_atomix!,
    assemble_pressure_residual_matrices_atomix!,
    assemble_pressure_residual_kernel!,
    assemble_pressure_residual_matrices_atomix_adj!,
    assemble_momentum_residual_matrices_atomix_adj!
public update_rate_kernel!, update_variable_kernel!, precompute_geometry_kernel!
public element_geometry
public stokes_update_rate!, stokes_update_variable!, precompute_stokes_geometry!
public color_mesh_greedy, remove_pressure_mean!
public velocity, stress, pressure, temperature

end # module FEMTools
