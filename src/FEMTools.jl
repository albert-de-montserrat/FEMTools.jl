module FEMTools

@doc """
    FEMTools

Finite-element utilities for structured meshes, reference elements, shape
functions, integration points, sparsity construction, mesh coloring, and simple
boundary-condition handling.

The package is organized around lightweight reference-element tags such as
`LinearElement{2, 4, Float64}` and `QuadraticElement{3, 27, Float64}`. A
`ReferenceElement` bundles the tag-specific shape functions and quadrature rule;
`Mesh` uses that reference element to generate structured connectivity and
coordinates.
""" FEMTools

using ForwardDiff
using StaticArrays
using DomainSets
using SparseArrays
using KernelAbstractions

TA(::CPU) = Array

# Reference-element definitions come first because the remaining source files
# extend constructors and methods specialized on these tags.
include("elements/elements.jl")
include("elements/shape_functions.jl")
include("elements/integration_points.jl")

# Mesh helpers are kept separate by responsibility: connectivity, coordinates,
# sparsity patterns, and conflict-free element coloring.
include("mesh/connectivity.jl")
include("mesh/mesh.jl")
include("mesh/sparsity.jl")
include("mesh/coloring.jl")

# Boundary-condition containers and in-place application routines.
include("boundary_conditions/boundary_conditions.jl")
include("boundary_conditions/apply.jl")

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
export Mesh
export DirichletBoundaryCondition, apply_bc!

# Mesh construction and graph utilities.
export generate_element2node,
    generate_node2element,
    generate_boundary_elements,
    generate_coordinates,
    generate_dofs,
    generate_sparsity_pattern,
    color_mesh,
    color_mesh_greedy

# Reference-element evaluation helpers.
export eval_shape_function,
    eval_shape_function_gradient,
    eval_shape_function_jacobian

end # module FEMTools
