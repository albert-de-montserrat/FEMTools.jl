abstract type AbstractElement end

abstract type AbstractShapeFunction end

abstract type AbstractIntegrationPoints end

struct LinearElement{nDim, nVert} <: AbstractElement end

struct QuadraticElement{nDim, nVert} <: AbstractElement end

struct CubicElement{nDim, nVert} <: AbstractElement end

struct ReferenceElement{Element, SF, IP} 
    shape_functions::SF
    integration_points::IP

    function ReferenceElement(::T) where {T<:AbstractElement}
        shape_functions = ShapeFunctions(T)
        integration_points = IntegrationPoints(T)

        new{T, typeof(shape_functions), typeof(integration_points)}(shape_functions, integration_points)
    end
end
ReferenceElement(::Type{T}) where {T<:AbstractElement} = ReferenceElement(T())

struct ShapeFunctions{SF, G} <: AbstractShapeFunction
    N::SF
    ∇N::G
    
    ShapeFunctions(N::SF, ∇N::G) where {SF, G} = new{SF, G}(N, ∇N)
end

ShapeFunctions(::Type{T}) where {T<:AbstractElement} = ShapeFunctions(T())

# 1D ELEMENTS ====

function ShapeFunctions(::LinearElement{1, 2})
    N1 = (ξ) -> (1 - ξ) * 0.5
    N2 = (ξ) -> (1 + ξ) * 0.5
    ∇N1 = (ξ) -> -0.5
    ∇N2 = (ξ) -> +0.5
    return ShapeFunctions((N1, N2), (∇N1, ∇N2))
end

function ShapeFunctions(::QuadraticElement{1, 3})
    N1 = (ξ) -> (1 - ξ) * (1 - ξ) * 0.5
    N2 = (ξ) -> (1 + ξ) * (1 + ξ) * 0.5
    N3 = (ξ) -> (1 - ξ) * (1 + ξ) * 0.5
    ∇N1 = (ξ) -> ξ - 1
    ∇N2 = (ξ) -> ξ + 1
    ∇N3 = (ξ) -> -ξ
    return ShapeFunctions((N1, N2, N3), (∇N1, ∇N2, ∇N3))
end

# 2D ELEMENTS ====

# Triangular elements

function ShapeFunctions(::LinearElement{2, 3})
    N1 = (ξ, η) -> 1 - ξ - η
    N2 = (ξ, η) -> ξ
    N3 = (ξ, η) -> η
    ∇N1 = (ξ, η) -> (-1.0, -1.0)
    ∇N2 = (ξ, η) -> (+1.0, 0.0)
    ∇N3 = (ξ, η) -> (0.0, +1.0)
    return ShapeFunctions((N1, N2, N3), (∇N1, ∇N2, ∇N3))
end

# Rectangular elements

function ShapeFunctions(::LinearElement{2, 4})
    N1 = (ξ, η) -> (1 - ξ) * (1 - η) * 0.25
    N2 = (ξ, η) -> (1 + ξ) * (1 - η) * 0.25
    N3 = (ξ, η) -> (1 + ξ) * (1 + η) * 0.25
    N4 = (ξ, η) -> (1 - ξ) * (1 + η) * 0.25
    ∇N1 = (ξ, η) -> (-(1 - η) * 0.25, -(1 - ξ) * 0.25)
    ∇N2 = (ξ, η) -> (+(1 - η) * 0.25, -(1 + ξ) * 0.25)
    ∇N3 = (ξ, η) -> (+(1 + η) * 0.25, +(1 + ξ) * 0.25)
    ∇N4 = (ξ, η) -> (-(1 + η) * 0.25, +(1 - ξ) * 0.25)
    return ShapeFunctions((N1, N2, N3, N4), (∇N1, ∇N2, ∇N3, ∇N4))
end

###
# INTEGRATION POINTS ====
###

struct IntegrationPoints{Dim, Nip, T} <: AbstractIntegrationPoints
    ξ::NTuple{Nip, T}
    η::Union{NTuple{Nip, T}, Nothing}
    ζ::Union{NTuple{Nip, T}, Nothing}
end

IntegrationPoints(::Type{T}) where {T<:AbstractElement} = IntegrationPoints(T())

function IntegrationPoints(::LinearElement{1, 2})
    ξ = -√(1 / 3), +√(1 / 3)
    η = nothing
    ζ = nothing
    return IntegrationPoints{1, 2, Float64}(ξ, η, ζ)
end

function IntegrationPoints(::QuadraticElement{1, 3})
    ξ = -√(3 / 5), 0.0, +√(3 / 5)
    η = nothing
    ζ = nothing
    return IntegrationPoints{1, 3, Float64}(ξ, η, ζ)
end

function IntegrationPoints(::LinearElement{2, 3})
    ξ = (1 / 3,)
    η = (1 / 3,)
    ζ = nothing
    return IntegrationPoints{2, 1, Float64}(ξ, η, ζ)
end

function IntegrationPoints(::LinearElement{2, 4})
    ξ = -√(1 / 3), +√(1 / 3), -√(1 / 3), +√(1 / 3)
    η = -√(1 / 3), -√(1 / 3), +√(1 / 3), +√(1 / 3)
    ζ = nothing
    return IntegrationPoints{2, 4, Float64}(ξ, η, ζ)
end
