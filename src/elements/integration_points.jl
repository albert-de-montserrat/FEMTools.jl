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
IntegrationPoints(::Type{LinearElement{nDim, nVert}}) where {nDim, nVert} =
    IntegrationPoints(LinearElement{nDim, nVert, Float64})
IntegrationPoints(::Type{QuadraticElement{nDim, nVert}}) where {nDim, nVert} =
    IntegrationPoints(QuadraticElement{nDim, nVert, Float64})

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
total degree ≤ **2n−2**.  Use `n = ceil((d+2)/2)` to integrate degree `d`
exactly.  For straight-sided T6/T7 elements (affine Jacobian, degree-4
integrands) `n=3` (9 points) suffices.

The GL abscissas are computed via the Golub–Welsch algorithm (eigendecomposition
of the symmetric tridiagonal Jacobi matrix); weights are exact to machine
precision for any `n`.

# Examples
```jldoctest
julia> ip = gauss_legendre_triangle(2);

julia> length(ip.ω)
4

julia> sum(ip.ω) ≈ 0.5
true
```
"""
gauss_legendre_triangle(n::Int) = gauss_legendre_triangle(Float64, n)

function gauss_legendre_triangle(::Type{T}, n::Int) where {T <: AbstractFloat}
    n > 0 || throw(ArgumentError("quadrature order must be positive, got $n"))
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

"""
    _gauss_legendre_01(T, n) -> (pts, wts)

Compute `n` Gauss-Legendre abscissas and weights on `[0,1]` using the
Golub-Welsch algorithm: eigendecompose the symmetric tridiagonal Jacobi matrix
whose off-diagonal entries are `β_i = i/√(4i²−1)`, then shift from `[-1,1]`
and normalize. Weights sum to 1.
"""
function _gauss_legendre_01(::Type{T}, n::Int) where {T <: AbstractFloat}
    n > 0 || throw(ArgumentError("quadrature order must be positive, got $n"))
    n == 1 && return (SVector{1, T}(T(1) / 2), SVector{1, T}(one(T)))

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
    a = √T(3 / 5)
    z = zero(T)
    ξ = SVector(-a, z, +a, -a, z, +a, -a, z, +a)
    η = SVector(-a, -a, -a, z, z, z, +a, +a, +a)
    ζ = nothing
    ω = SVector{9, T}(
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
    IntegrationPoints(::QuadraticElement{3, 10})

Return the symmetric four-point degree-two rule on the reference tetrahedron.
"""
function IntegrationPoints(::QuadraticElement{3, 10, T}) where T
    a, b = T((5 + 3sqrt(5)) / 20), T((5 - sqrt(5)) / 20)
    ξ = SVector(a, b, b, b)
    η = SVector(b, a, b, b)
    ζ = SVector(b, b, a, b)
    ω = SVector{4, T}(ntuple(_ -> T(1/24), Val(4)))
    return IntegrationPoints{3, 4, T}(ξ, η, ζ, ω)
end

"""
    IntegrationPoints(::QuadraticElement{3, 11})

Return the symmetric fifteen-point degree-five rule for the T11 element.
"""
function IntegrationPoints(::QuadraticElement{3, 11, T}) where T
    pairs = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    points = (
        ntuple(_ -> T(1 / 4), 4),
        (ntuple(i -> ntuple(j -> j == i ? zero(T) : T(1 / 3), 4), 4))...,
        (ntuple(i -> ntuple(j -> j == i ? T(8 / 11) : T(1 / 11), 4), 4))...,
        (ntuple(6) do i
            a, b = pairs[i]
            ntuple(j -> (j == a || j == b) ? T(0.4334498464263357) : T(0.0665501535736643), 4)
        end)...,
    )
    weights = (T(0.030283678097089), ntuple(_ -> T(0.006026785714286), 4)...,
               ntuple(_ -> T(0.011645249086029), 4)...,
               ntuple(_ -> T(0.010949141561386), 6)...)
    ξ = SVector{15, T}(p[2] for p in points)
    η = SVector{15, T}(p[3] for p in points)
    ζ = SVector{15, T}(p[4] for p in points)
    return IntegrationPoints{3, 15, T}(ξ, η, ζ, SVector{15, T}(weights))
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
    ω = SVector{27, T}(ntuple(Val(27)) do i
        weights[mod1(i, 3)] * weights[mod1(cld(i, 3), 3)] * weights[cld(i, 9)]
    end)

    return IntegrationPoints{3, 27, T}(ξ, η, ζ, ω)
end
