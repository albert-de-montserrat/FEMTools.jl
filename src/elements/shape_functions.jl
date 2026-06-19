"""
    ShapeFunctions(N, ∇N)
    ShapeFunctions(ElementType)

Container for shape functions `N` and their reference-coordinate gradients
`∇N`.

For an element constructor such as `ShapeFunctions(LinearElement{2, 3})`, `N`
and `∇N` are tuples of callable functions ordered by local node number.
"""
struct ShapeFunctions{SF, G} <: AbstractShapeFunction
    N::SF
    ∇N::G

    # Keep the tuple types concrete so callers can specialize on element order
    # and dimension without dynamic dispatch inside quadrature loops.
    ShapeFunctions(N::SF, ∇N::G) where {SF, G} = new{SF, G}(N, ∇N)
end

"""
    ShapeFunctions(::Type{T}) where {T<:AbstractElement{nDim, nVert}}

Construct shape functions for element type `T`.
"""
ShapeFunctions(::Type{T}) where {nDim, nVert, T<:AbstractElement{nDim, nVert}} = ShapeFunctions(T())

"""
    ShapeFunctions(::LinearElement{1, 2})

Return the two linear shape functions and constant gradients on the reference
line `-1 <= ξ <= 1`.
"""
function ShapeFunctions(::LinearElement{1, 2, T}) where T
    N1 = ξ -> (1 - ξ) * T(1/2)
    N2 = ξ -> (1 + ξ) * T(1/2)
    ∇N1 = ξ -> -T(1/2)
    ∇N2 = ξ -> +T(1/2)
    return ShapeFunctions((N1, N2), (∇N1, ∇N2))
end

"""
    ShapeFunctions(::QuadraticElement{1, 3})

Return the three quadratic shape functions and gradients on the reference line
`-1 <= ξ <= 1`.
"""
function ShapeFunctions(::QuadraticElement{1, 3, T}) where T
    N1 = ξ -> ξ * (ξ - 1) * T(1/2)
    N2 = ξ -> 1 - T(ξ^2)
    N3 = ξ -> ξ * (ξ + 1) * T(1/2)
    ∇N1 = ξ -> ξ - T(1/2)
    ∇N2 = ξ -> -2ξ
    ∇N3 = ξ -> ξ + T(1/2)
    return ShapeFunctions((N1, N2, N3), (∇N1, ∇N2, ∇N3))
end

"""
    ShapeFunctions(::QuadraticElement{2, 6})

Return the six quadratic shape functions and gradients on the reference
triangle `ξ >= 0`, `η >= 0`, and `ξ + η <= 1`.
"""
function ShapeFunctions(::QuadraticElement{2, 6, T}) where T
    N1 = (ξ, η) -> begin
        L1 = 1 - ξ - η
        L1 * (2L1 - 1)
    end
    N2 = (ξ, η) -> ξ * (2ξ - 1)
    N3 = (ξ, η) -> η * (2η - 1)
    N4 = (ξ, η) -> 4 * (1 - ξ - η) * ξ
    N5 = (ξ, η) -> 4 * ξ * η
    N6 = (ξ, η) -> 4 * η * (1 - ξ - η)

    ∇N1 = (ξ, η) -> begin
        L1 = 1 - ξ - η
        (1 - 4L1, 1 - 4L1)
    end
    ∇N2 = (ξ, η) -> (4ξ - 1, zero(T))
    ∇N3 = (ξ, η) -> (zero(T), 4η - 1)
    ∇N4 = (ξ, η) -> (4 * (1 - 2ξ - η), -4ξ)
    ∇N5 = (ξ, η) -> (4η, 4ξ)
    ∇N6 = (ξ, η) -> (-4η, 4 * (1 - ξ - 2η))

    return ShapeFunctions((N1, N2, N3, N4, N5, N6), (∇N1, ∇N2, ∇N3, ∇N4, ∇N5, ∇N6))
end

_quadratic_line_N1(ξ::T) where T = ξ * (ξ - 1) * T(1/2)
_quadratic_line_N2(ξ::T) where T = 1 - ξ^2
_quadratic_line_N3(ξ::T) where T = ξ * (ξ + 1) * T(1/2)
_quadratic_line_∇N1(ξ::T) where T = ξ - T(1/2)
_quadratic_line_∇N2(ξ::T) where T = -2ξ
_quadratic_line_∇N3(ξ::T) where T = ξ + T(1/2)

# Tensor-product quadrilateral and hexahedral elements reuse the 1D Lagrange
# basis. The `side` or `node` argument is the reference coordinate of the local
# node in that coordinate direction.
_linear_line_N(ξ::T, side) where T = side == -1 ? (1 - ξ) * T(1/2) : (1 + ξ) * T(1/2)
_linear_line_∇N(side::T) where T= side == -1 ? -T(1/2) : +T(1/2)

function _quadratic_line_N(ξ, node)
    node == -1 && return _quadratic_line_N1(ξ)
    node == 0 && return _quadratic_line_N2(ξ)
    return _quadratic_line_N3(ξ)
end

function _quadratic_line_∇N(ξ, node)
    node == -1 && return _quadratic_line_∇N1(ξ)
    node == 0 && return _quadratic_line_∇N2(ξ)
    return _quadratic_line_∇N3(ξ)
end

"""
    ShapeFunctions(::LinearElement{2, 3})

Return the three linear shape functions and constant gradients on the reference
triangle `ξ >= 0`, `η >= 0`, and `ξ + η <= 1`.
"""
function ShapeFunctions(::LinearElement{2, 3, T}) where T
    N1 = (ξ, η) -> 1 - ξ - η
    N2 = (ξ, η) -> ξ
    N3 = (ξ, η) -> η
    ∇N1 = (ξ, η) -> (-one(T), -one(T))
    ∇N2 = (ξ, η) -> (+one(T), zero(T))
    ∇N3 = (ξ, η) -> (zero(T), +one(T))
    return ShapeFunctions((N1, N2, N3), (∇N1, ∇N2, ∇N3))
end

"""
    ShapeFunctions(::LinearElement{2, 4})

Return the four bilinear shape functions and gradients on the reference
quadrilateral `-1 <= ξ <= 1`, `-1 <= η <= 1`.
"""
function ShapeFunctions(::LinearElement{2, 4, T}) where T
    N1 = (ξ, η) -> (1 - ξ) * (1 - η) * T(1/4)
    N2 = (ξ, η) -> (1 + ξ) * (1 - η) * T(1/4)
    N3 = (ξ, η) -> (1 + ξ) * (1 + η) * T(1/4)
    N4 = (ξ, η) -> (1 - ξ) * (1 + η) * T(1/4)
    ∇N1 = (ξ, η) -> (-(1 - η) * T(1/4), -(1 - ξ) * T(1/4))
    ∇N2 = (ξ, η) -> (+(1 - η) * T(1/4), -(1 + ξ) * T(1/4))
    ∇N3 = (ξ, η) -> (+(1 + η) * T(1/4), +(1 + ξ) * T(1/4))
    ∇N4 = (ξ, η) -> (-(1 + η) * T(1/4), +(1 - ξ) * T(1/4))
    return ShapeFunctions((N1, N2, N3, N4), (∇N1, ∇N2, ∇N3, ∇N4))
end

"""
    ShapeFunctions(::QuadraticElement{2, 9})

Return the nine tensor-product quadratic shape functions and gradients on the
reference quadrilateral `-1 <= ξ <= 1`, `-1 <= η <= 1`.
"""
function ShapeFunctions(::QuadraticElement{2, 9, T}) where T
    nodes = (
        (-one(T), -one(T)),
        (+one(T), -one(T)),
        (+one(T), +one(T)),
        (-one(T), +one(T)),
        (zero(T), -one(T)),
        (+one(T), zero(T)),
        (zero(T), +one(T)),
        (-one(T), zero(T)),
        (zero(T), zero(T)),
    )

    N = ntuple(Val(9)) do i
        node = nodes[i]
        (ξ, η) -> _quadratic_line_N(ξ, node[1]) * _quadratic_line_N(η, node[2])
    end
    ∇N = ntuple(Val(9)) do i
        node = nodes[i]
        (ξ, η) -> (
            _quadratic_line_∇N(ξ, node[1]) * _quadratic_line_N(η, node[2]),
            _quadratic_line_N(ξ, node[1]) * _quadratic_line_∇N(η, node[2]),
        )
    end

    return ShapeFunctions(N, ∇N)
end

"""
    ShapeFunctions(::LinearElement{3, 4})

Return the four linear shape functions and constant gradients on the reference
tetrahedron `ξ >= 0`, `η >= 0`, `ζ >= 0`, `ξ + η + ζ <= 1`.
"""
function ShapeFunctions(::LinearElement{3, 4, T}) where T
    N1 = (ξ, η, ζ) -> 1 - ξ - η - ζ
    N2 = (ξ, η, ζ) -> ξ
    N3 = (ξ, η, ζ) -> η
    N4 = (ξ, η, ζ) -> ζ
    ∇N1 = (ξ, η, ζ) -> (-one(T), -one(T), -one(T))
    ∇N2 = (ξ, η, ζ) -> (+one(T),  zero(T),  zero(T))
    ∇N3 = (ξ, η, ζ) -> (zero(T), +one(T),  zero(T))
    ∇N4 = (ξ, η, ζ) -> (zero(T),  zero(T), +one(T))
    return ShapeFunctions((N1, N2, N3, N4), (∇N1, ∇N2, ∇N3, ∇N4))
end

"""
    ShapeFunctions(::LinearElement{3, 8})

Return the eight trilinear shape functions and gradients on the reference
hexahedron `-1 <= ξ, η, ζ <= 1`.
"""
function ShapeFunctions(::LinearElement{3, 8, T}) where T
    # Local node coordinates follow the ordering documented in
    # `LinearElement{3, 8}`.
    nodes = (
        (-one(T), -one(T), -one(T)),
        (+one(T), -one(T), -one(T)),
        (+one(T), +one(T), -one(T)),
        (-one(T), +one(T), -one(T)),
        (-one(T), -one(T), +one(T)),
        (+one(T), -one(T), +one(T)),
        (+one(T), +one(T), +one(T)),
        (-one(T), +one(T), +one(T)),
    )

    N = ntuple(i -> begin
        node = nodes[i]
        (ξ, η, ζ) -> _linear_line_N(ξ, node[1]) * _linear_line_N(η, node[2]) * _linear_line_N(ζ, node[3])
    end, Val(8))
    ∇N = ntuple(i -> begin
        node = nodes[i]
        (ξ, η, ζ) -> (
            _linear_line_∇N(node[1]) * _linear_line_N(η, node[2]) * _linear_line_N(ζ, node[3]),
            _linear_line_N(ξ, node[1]) * _linear_line_∇N(node[2]) * _linear_line_N(ζ, node[3]),
            _linear_line_N(ξ, node[1]) * _linear_line_N(η, node[2]) * _linear_line_∇N(node[3]),
        )
    end, Val(8))

    return ShapeFunctions(N, ∇N)
end

"""
    ShapeFunctions(::QuadraticElement{3, 27})

Return the twenty-seven tensor-product quadratic shape functions and gradients
on the reference hexahedron `-1 <= ξ, η, ζ <= 1`.
"""
function ShapeFunctions(::QuadraticElement{3, 27, T}) where T
    # Tensor-product node coordinates in the ordering documented in
    # `QuadraticElement{3, 27}`.
    nodes = (
        (-one(T), -one(T), -one(T)),
        (+one(T), -one(T), -one(T)),
        (+one(T), +one(T), -one(T)),
        (-one(T), +one(T), -one(T)),
        (-one(T), -one(T), +one(T)),
        (+one(T), -one(T), +one(T)),
        (+one(T), +one(T), +one(T)),
        (-one(T), +one(T), +one(T)),
        (zero(T), -one(T), -one(T)),
        (+one(T), zero(T), -one(T)),
        (zero(T), +one(T), -one(T)),
        (-one(T), zero(T), -one(T)),
        (zero(T), -one(T), +one(T)),
        (+one(T), zero(T), +one(T)),
        (zero(T), +one(T), +one(T)),
        (-one(T), zero(T), +one(T)),
        (-one(T), -one(T), zero(T)),
        (+one(T), -one(T), zero(T)),
        (+one(T), +one(T), zero(T)),
        (-one(T), +one(T), zero(T)),
        (zero(T), zero(T), -one(T)),
        (zero(T), -one(T), zero(T)),
        (+one(T), zero(T), zero(T)),
        (zero(T), +one(T), zero(T)),
        (-one(T), zero(T), zero(T)),
        (zero(T), zero(T), +one(T)),
        (zero(T), zero(T), zero(T)),
    )


    N = ntuple(Val(27)) do i 
        @inline
        node = nodes[i]
        (ξ, η, ζ) -> _quadratic_line_N(ξ, node[1]) * _quadratic_line_N(η, node[2]) * _quadratic_line_N(ζ, node[3])
    end
    ∇N = ntuple(Val(27)) do i 
        @inline
        node = nodes[i]
        (ξ, η, ζ) -> (
            _quadratic_line_∇N(ξ, node[1]) * _quadratic_line_N(η, node[2]) * _quadratic_line_N(ζ, node[3]),
            _quadratic_line_N(ξ, node[1]) * _quadratic_line_∇N(η, node[2]) * _quadratic_line_N(ζ, node[3]),
            _quadratic_line_N(ξ, node[1]) * _quadratic_line_N(η, node[2]) * _quadratic_line_∇N(ζ, node[3]),
        )
    end

    return ShapeFunctions(N, ∇N)
end

"""
    eval_shape_function(element, coords)

Evaluate all shape functions of a `ReferenceElement` at reference coordinates
`coords`.

`coords` is an `NTuple`, for example `(ξ,)` on a line or `(ξ, η)` on a
two-dimensional reference element. The return value is an `SVector` ordered by
local node number.
"""
@inline function eval_shape_function(element, coords::NTuple{M, T}) where {M, T}
    return _eval_shape_function(element.shape_functions.N, coords)
end

@inline eval_shape_function(element, coords::SVector) = eval_shape_function(element, tuple(coords...))

"""
    eval_shape_function_gradient(element, coords)

Evaluate the stored reference-coordinate gradients of all shape functions of a
`ReferenceElement` at `coords`.
"""
@inline function eval_shape_function_gradient(element, coords::NTuple{M, T}) where {M, T}
    return _eval_shape_function(element.shape_functions.∇N, coords)
end

"""
    eval_shape_function_jacobian(element, coords)

Compute the Jacobian of the shape-function vector with respect to reference
coordinates at `coords`.

For one-dimensional elements this returns the stored analytical gradients. For
higher-dimensional elements the Jacobian is computed with `ForwardDiff`.
"""
@inline function eval_shape_function_jacobian(element, coords::NTuple{M, T}) where {M, T}
    return eval_shape_function_jacobian(element, SVector{M, T}(coords...))
end

@inline function eval_shape_function_jacobian(element, coords::SVector{M, T}) where {M, T}
    return ForwardDiff.jacobian(
        coords -> _eval_shape_function(element.shape_functions.N, coords),
        coords,
    )
end

@inline function eval_shape_function_jacobian(element, coords::SVector{1, T}) where {T}
    return _eval_shape_function(element.shape_functions.∇N, coords)
end


"""
    shape_function_values(element)

Return shape-function values evaluated at every quadrature point of `element`.

The result is an `NTuple` of length `Nq` (number of quadrature points). Each
entry is an `SVector` of length `N` (number of local nodes) holding `Nᵢ(ξ_q)`
for node `i` at quadrature point `q`.  Used inside assembly kernels to form the
consistent mass matrix and source-term integrals.
"""
@inline function shape_function_values(element::ReferenceElement{T}, ip) where T<:AbstractElement{1}
    NQ = length(ip.ω)
    coords = ntuple(NQ) do q
        SVector(ip.ξ[q])
    end
    return ntuple(q -> eval_shape_function(element, coords[q]), NQ)
end


@inline function shape_function_values(element::ReferenceElement{T}, ip) where T<:AbstractElement{2}
    NQ = length(ip.ω)
    coords = ntuple(NQ) do q
        SVector(ip.ξ[q], ip.η[q])
    end
    return ntuple(q -> eval_shape_function(element, coords[q]), NQ)
end

@inline function shape_function_values(element::ReferenceElement{T}, ip) where T<:AbstractElement{3}
    NQ = length(ip.ω)
    coords = ntuple(NQ) do q
        SVector(ip.ξ[q], ip.η[q], ip.ζ[q])
    end
    return ntuple(q -> eval_shape_function(element, coords[q]), NQ)
end

@inline shape_function_values(element::ReferenceElement) = shape_function_values(element, element.integration_points)

"""
    _eval_shape_function(N, coords)

Evaluate every callable in the `NTuple` `N` at `coords` and return an
`SVector` of length `M`.

Uses `@generated` to unroll the evaluation loop at compile time, so the return
type is concretely known and the result can be used inside element assembly
kernels without dynamic dispatch.
"""
@generated function _eval_shape_function(N::T, coords) where {T<:Tuple}
    M = fieldcount(T)
    quote
        @inline
        Base.@nexprs $M i -> x_i = N[i](coords...)
        Base.@ncall $M SVector x
    end
end
