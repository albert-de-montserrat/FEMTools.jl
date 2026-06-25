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

# ---------------------------------------------------------------------------
# Triangular Gauss-Legendre grid generator (Duffy transform)
# ---------------------------------------------------------------------------

"""
    gauss_legendre_triangle([T=Float64,] n) -> IntegrationPoints{2, n², T}

Generate `n²` Gauss-Legendre integration points on the reference triangle
`{ξ≥0, η≥0, ξ+η≤1}` via the Duffy transform.

The `n`-point Gauss-Legendre rule on `[-1,1]` is collapsed onto the triangle
using the substitution

    ξ = s,    η = t·(1−s),    dΩ = (1−s)·ds dt

where `s,t ∈ [0,1]`.  Because the Duffy Jacobian `(1−s)` costs one polynomial
degree in the `s`-direction, the rule is exact for bivariate polynomials of
total degree ≤ **2n−2**.  Use `n = ceil((d+4)/2)` to integrate degree `d`
exactly.  For straight-sided T6/T7 elements (affine Jacobian, degree-4
integrands) `n=3` (9 points) suffices.

The GL abscissas are computed via the Golub–Welsch algorithm (eigendecomposition
of the symmetric tridiagonal Jacobi matrix); weights are exact to machine
precision for any `n`.
"""
gauss_legendre_triangle(n::Int) = gauss_legendre_triangle(Float64, n)

function gauss_legendre_triangle(::Type{T}, n::Int) where {T <: AbstractFloat}
    s1d, w1d = _gauss_legendre_01(T, n)

    nq  = n^2
    ξv  = Vector{T}(undef, nq)
    ηv  = Vector{T}(undef, nq)
    ωv  = Vector{T}(undef, nq)

    q = 1
    for i in 1:n
        si, wi = s1d[i], w1d[i]
        for j in 1:n
            sj, wj = s1d[j], w1d[j]
            ξv[q] = si
            ηv[q] = sj * (1 - si)
            ωv[q] = wi * wj * (1 - si)
            q += 1
        end
    end

    return IntegrationPoints{2, nq, T}(
        SVector{nq, T}(ξv),
        SVector{nq, T}(ηv),
        nothing,
        SVector{nq, T}(ωv),
    )
end

# Compute n Gauss-Legendre abscissas and weights on [0,1] via Golub-Welsch.
function _gauss_legendre_01(::Type{T}, n::Int) where {T <: AbstractFloat}
    n == 1 && return (SVector{1,T}(T(1)/2),  SVector{1,T}(one(T)))

    # Off-diagonal entries of the symmetric Jacobi matrix for Legendre polys
    β = T[i / sqrt(T(4i^2 - 1)) for i in 1:(n - 1)]
    # Eigendecomposition of the symmetric tridiagonal matrix (diagonal = 0)
    vals, vecs = eigen(SymTridiagonal(zeros(T, n), β))
    # GL points on [-1,1] → shift to [0,1]; weights = 2*(v₁ᵢ)² → divide by 2
    pts = SVector{n,T}((vals .+ 1) ./ 2)
    wts = SVector{n,T}(vecs[1, :] .^ 2)   # already normalised: sum = 1
    return pts, wts
end

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

Return a three-point degree-two rule on the reference triangle.
"""
function IntegrationPoints(::LinearElement{2, 3, T}) where T
    ξ = SVector(T.((1 / 6, 2 / 3, 1 / 6))...)
    η = SVector(T.((1 / 6, 1 / 6, 2 / 3))...)
    ζ = nothing
    ω = SVector(T.((1 / 6, 1 / 6, 1 / 6))...)
    return IntegrationPoints{2, 3, T}(ξ, η, ζ, ω)
end

"""
    IntegrationPoints(::QuadraticElement{2, 6})

Return a six-point degree-four rule on the reference triangle.
"""
function IntegrationPoints(::QuadraticElement{2, 6, T}) where T
    a = T(0.816847572980459)
    b = T(0.091576213509771)
    c = T(0.108103018168070)
    d = T(0.445948490915965)
    w1 = T(0.054975871827661)
    w2 = T(0.111690794839005)

    ξ = SVector(b, a, b, d, c, d)
    η = SVector(b, b, a, d, d, c)
    ζ = nothing
    ω = SVector(w1, w1, w1, w2, w2, w2)
    return IntegrationPoints{2, 6, T}(ξ, η, ζ, ω)
end

"""
    IntegrationPoints(::QuadraticElement{2, 7})

Return the 7-point degree-5 Dunavant rule on the reference triangle.

The rule integrates polynomials of degree ≤ 5 exactly — sufficient for the
cubic bubble of the T7 element. Points come from two symmetric orbits around
the centroid plus the centroid itself; weights sum to 1/2 (area of the
reference triangle).
"""
function IntegrationPoints(::QuadraticElement{2, 7, T}) where T
    a1 = T(0.101286507323456)
    a2 = T(0.470142064105115)
    b1 = 1 - 2a1              # 0.797426985353088
    b2 = 1 - 2a2              # 0.059715871789770
    w1 = T(0.062969590272414)
    w2 = T(0.066197076394253)
    wc = T(0.112500000000000)  # 9/80

    ξ = SVector(a1, b1, a1, a2, b2, a2, T(1/3))
    η = SVector(a1, a1, b1, a2, a2, b2, T(1/3))
    ζ = nothing
    ω = SVector(w1, w1, w1, w2, w2, w2, wc)
    return IntegrationPoints{2, 7, T}(ξ, η, ζ, ω)
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
    IntegrationPoints(::LinearElement{3, 4})

Return the one-point centroid rule on the reference tetrahedron.
"""
function IntegrationPoints(::LinearElement{3, 4, T}) where T
    ξ = SVector(T(1/4))
    η = SVector(T(1/4))
    ζ = SVector(T(1/4))
    ω = SVector(T(1/6))
    return IntegrationPoints{3, 1, T}(ξ, η, ζ, ω)
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
