"""
    IntegrationPoints{Dim, Nip, T}

Reference-coordinate integration points for an element.

Fields:
- `ξ`: first reference coordinate for each integration point.
- `η`: second reference coordinate, or `nothing` for 1D rules.
- `ζ`: third reference coordinate, or `nothing` for lower-dimensional rules.
- `ω`: integration weight for each integration point.
"""
struct IntegrationPoints{Dim, Nip, T} <: AbstractIntegrationPoints
    ξ::SVector{Nip, T}
    η::Union{SVector{Nip, T}, Nothing}
    ζ::Union{SVector{Nip, T}, Nothing}
    ω::SVector{Nip, T}
end

"""
    IntegrationPoints(::Type{T}) where {T<:AbstractElement{nDim, nVert}}

Construct integration points for element type `T`.
"""
IntegrationPoints(::Type{T}) where {nDim, nVert, T<:AbstractElement{nDim, nVert}} = IntegrationPoints(T())

"""
    IntegrationPoints(::LinearElement{1, 2})

Return the two-point Gauss rule on the reference line.
"""
function IntegrationPoints(::LinearElement{1, 2, T}) where T
    ξ = SVector(-√T(1 / 3), +√T(1 / 3))
    η = nothing
    ζ = nothing
    ω = SVector(one(T), one(T))
    return IntegrationPoints{1, 2, T}(ξ, η, ζ, ω)
end

"""
    IntegrationPoints(::QuadraticElement{1, 3})

Return the three-point Gauss rule on the reference line.
"""
function IntegrationPoints(::QuadraticElement{1, 3, T}) where T
    ξ = SVector(-√T(3 / 5), zero(T), +√T(3 / 5))
    η = nothing
    ζ = nothing
    ω = SVector(T.((5 / 9, 8 / 9, 5 / 9))...)
    return IntegrationPoints{1, 3, T}(ξ, η, ζ, ω)
end

"""
    IntegrationPoints(::LinearElement{2, 3})

Return the one-point centroid rule on the reference triangle.
"""
function IntegrationPoints(::LinearElement{2, 3, T}) where T
    ξ = SVector(T.(1 / 3))
    η = SVector(T.(1 / 3))
    ζ = nothing
    ω = SVector(T.(1 / 2))
    return IntegrationPoints{2, 1, T}(ξ, η, ζ, ω)
end

"""
    IntegrationPoints(::QuadraticElement{2, 6})

Return a three-point degree-two rule on the reference triangle.
"""
function IntegrationPoints(::QuadraticElement{2, 6, T}) where T
    ξ = SVector(T.((1 / 6, 2 / 3, 1 / 6))...)
    η = SVector(T.((1 / 6, 1 / 6, 2 / 3))...)
    ζ = nothing
    ω = SVector(T.((1 / 6, 1 / 6, 1 / 6))...)
    return IntegrationPoints{2, 3, T}(ξ, η, ζ, ω)
end

"""
    IntegrationPoints(::LinearElement{2, 4})

Return the tensor-product two-by-two Gauss rule on the reference quadrilateral.
"""
function IntegrationPoints(::LinearElement{2, 4, T}) where T
    ξ = SVector(-√T(1 / 3), +√T(1 / 3), -√T(1 / 3), +√T(1 / 3))
    η = SVector(-√T(1 / 3), -√T(1 / 3), +√T(1 / 3), +√T(1 / 3))
    ζ = nothing
    ω = SVector(one(T), one(T), one(T), one(T))
    return IntegrationPoints{2, 4, T}(ξ, η, ζ, ω)
end

"""
    IntegrationPoints(::QuadraticElement{2, 9})

Return the tensor-product three-by-three Gauss rule on the reference
quadrilateral.
"""
function IntegrationPoints(::QuadraticElement{2, 9, T}) where T
    # Tensor-product ordering: ξ varies fastest, then η.
    a = √(3 / 5)
    ξ = SVector(-a, 0.0, +a, -a, 0.0, +a, -a, 0.0, +a)
    η = SVector(-a, -a, -a, 0.0, 0.0, 0.0, +a, +a, +a)
    ζ = nothing
    ω = SVector(
        25 / 81,
        40 / 81,
        25 / 81,
        40 / 81,
        64 / 81,
        40 / 81,
        25 / 81,
        40 / 81,
        25 / 81,
    )
    return IntegrationPoints{2, 9, T}(ξ, η, ζ, ω)
end

"""
    IntegrationPoints(::LinearElement{3, 8})

Return the tensor-product two-by-two-by-two Gauss rule on the reference
hexahedron.
"""
function IntegrationPoints(::LinearElement{3, 8, T}) where T
    a = √T(1 / 3)
    ξ = SVector(-a, +a, -a, +a, -a, +a, -a, +a)
    η = SVector(-a, -a, +a, +a, -a, -a, +a, +a)
    ζ = SVector(-a, -a, -a, -a, +a, +a, +a, +a)
    ω = SVector(one(T), one(T), one(T), one(T), one(T), one(T), one(T), one(T))
    return IntegrationPoints{3, 8, T}(ξ, η, ζ, ω)
end

"""
    IntegrationPoints(::QuadraticElement{3, 27})

Return the tensor-product three-by-three-by-three Gauss rule on the reference
hexahedron.
"""
function IntegrationPoints(::QuadraticElement{3, 27, T}) where T
    a = √T(3 / 5)
    points = (-a, zero(T), +a)
    weights = T.((5 / 9, 8 / 9, 5 / 9))

    # Tensor-product ordering: ξ varies fastest, followed by η and then ζ.
    ξ = SVector{27, T}(ntuple(i -> points[mod1(i, 3)], Val(27)))
    η = SVector{27, T}(ntuple(i -> points[mod1(cld(i, 3), 3)], Val(27)))
    ζ = SVector{27, T}(ntuple(i -> points[cld(i, 9)], Val(27)))
    ω = SVector{27, T}(ntuple(i -> weights[mod1(i, 3)] * weights[mod1(cld(i, 3), 3)] * weights[cld(i, 9)], Val(27)))

    return IntegrationPoints{3, 27, T}(ξ, η, ζ, ω)
end
