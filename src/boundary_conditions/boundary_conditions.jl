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

`DoFs` and `vals` must have the same length: `vals[i]` is the prescribed value
for degree of freedom `DoFs[i]`.

Fields:
- `Γ`    : boundary domain or boundary marker (stored for reference).
- `DoFs` : constrained degree-of-freedom indices.
- `vals` : prescribed values aligned with `DoFs`.
"""
struct DirichletBoundaryCondition{D, T, V} <: AbstractBoundaryCondition
    Γ::D
    DoFs::T
    vals::V
    function DirichletBoundaryCondition(Γ::D, DoFs::T, vals::V) where {D, T, V}
        new{D, T, V}(Γ, DoFs, vals)
    end
end

"""
    TangentialFreeSlipBoundaryCondition(Γ, DoFs, vals)

Container for a tangential free-slip boundary condition.

Structure mirrors `DirichletBoundaryCondition`: `Γ` is the boundary marker,
`DoFs` are the constrained degrees of freedom, and `vals` are the prescribed
tangential values.
"""
struct TangentialFreeSlipBoundaryCondition{D, T, V} <: AbstractBoundaryCondition
    Γ::D
    DoFs::T
    vals::V
    function TangentialFreeSlipBoundaryCondition(Γ::D, DoFs::T, vals::V) where {D, T, V}
        new{D, T, V}(Γ, DoFs, vals)
    end
end

"""
    TractionBoundaryCondition(Γ, DoFs, vals)

Container for a traction (Neumann) boundary condition.

Structure mirrors `DirichletBoundaryCondition`: `Γ` is the boundary marker,
`DoFs` are the traction degrees of freedom, and `vals` are the prescribed
traction values.
"""
struct TractionBoundaryCondition{D, T, V} <: AbstractBoundaryCondition
    Γ::D
    DoFs::T
    vals::V
    function TractionBoundaryCondition(Γ::D, DoFs::T, vals::V) where {D, T, V}
        new{D, T, V}(Γ, DoFs, vals)
    end
end
