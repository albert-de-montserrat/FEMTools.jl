"""
    AbstractBoundaryCondition

Abstract supertype for boundary-condition containers.
"""
abstract type AbstractBoundaryCondition end

###
# DIRICHLET
###

"""
    DirichletBoundaryCondition(Γ, DoFs, vals)

Container for Dirichlet boundary data.

Fields:
- `Γ`: boundary domain or boundary marker.
- `DoFs`: degrees of freedom constrained on `Γ`.
- `vals`: prescribed values for each constrained degree of freedom.
"""
struct DirichletBoundaryCondition{T, D, V} <: AbstractBoundaryCondition
    Γ::T
    DoFs::D
    vals::V
    function DirichletBoundaryCondition(Γ::T, DoFs::D, vals::V) where {T, D, V}
        new{T, D, V}(Γ, DoFs, vals)
    end
end
