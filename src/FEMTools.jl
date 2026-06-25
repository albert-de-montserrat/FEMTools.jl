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
available when that package is loaded.
"""
TA(::CPU) = Array

include("utils.jl")

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

# Heat-diffusion solver: types, assembly, and PT solver.
include("heat_diffusion/types/heat_diffusion_types.jl")
include("heat_diffusion/assembly/residual.jl")
include("heat_diffusion/solvers/DR.jl")

# Lithostatic-pressure solver: types, assembly, and PT solver.
include("lithostatic_pressure/types/lithostatic_pressure_types.jl")
include("lithostatic_pressure/assembly/residual.jl")
include("lithostatic_pressure/solvers/DR.jl")

# Stokes solver: types and assembly.
include("stokes/types/stokes_types.jl")
include("stokes/assemblers/pressure_residual.jl")
include("stokes/assemblers/pressure_scaling.jl")
include("stokes/assemblers/momentum_residuals.jl")

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
export Mesh, MixedMesh
export DirichletBoundaryCondition, apply_bc!

# Mesh construction and graph utilities.
export generate_element2node,
    generate_node2element,
    generate_boundary_elements,
    generate_coordinates,
    generate_dofs,
    generate_sparsity_pattern,
    color_mesh,
    color_mesh_greedy,
    build_element_groups,
    build_discontinuous_linear_mesh

# Reference-element evaluation helpers.
export eval_shape_function,
    eval_shape_function_gradient,
    eval_shape_function_jacobian,
    shape_function_values,
    gauss_legendre_triangle

# Heat-diffusion solver.
export ThermalDiffusionDR,
    solver!,
    assemble_diffusion_matrices_atomix!,
    assemble_diffusion_matrices_colored!,
    apply_dirichlet!,
    update_rate_kernel!,
    update_variable_kernel!,
    precompute_geometry_kernel!

# Lithostatic-pressure solver.
export LithostaticPressureDR,
    assemble_lithostatic_pressure_matrices_atomix!,
    assemble_lithostatic_pressure_matrices_colored!

# Stokes solver.
export StokesDR, DruckerPrager,
    assemble_momentum_residual_matrices_atomix!,
    assemble_momentum_jacobian_matrices_atomix!,
    assemble_augmented_momentum_jacobian_matrices_atomix!,
    assemble_viscosity_weighted_pressure_scaling!,
    assemble_pressure_residual_matrices_atomix!

end # module FEMTools
