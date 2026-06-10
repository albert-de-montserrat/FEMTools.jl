module FEMTools

@doc """
    FEMTools

Finite-element utilities for reference elements, shape functions, and
integration points.
""" FEMTools

using ForwardDiff
using StaticArrays
using DomainSets
using SparseArrays

include("elements/elements.jl")
include("elements/shape_functions.jl")
include("elements/integration_points.jl")
include("mesh/connectivity.jl")
include("mesh/mesh.jl")
include("mesh/sparsity.jl")
include("mesh/coloring.jl")
include("boundary_conditions/boundary_conditions.jl")
include("boundary_conditions/apply.jl")

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
export generate_element2node,
    generate_node2element,
    generate_boundary_elements,
    generate_coordinates,
    generate_dofs,
    generate_sparsity_pattern,
    color_mesh,
    color_mesh_greedy
export eval_shape_function,
    eval_shape_function_gradient,
    eval_shape_function_jacobian

end # module FEMTools
