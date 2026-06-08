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
function ShapeFunctions(::LinearElement{1, 2})
    N1 = ξ -> (1 - ξ) * 0.5
    N2 = ξ -> (1 + ξ) * 0.5
    ∇N1 = ξ -> -0.5
    ∇N2 = ξ -> +0.5
    return ShapeFunctions((N1, N2), (∇N1, ∇N2))
end

"""
    ShapeFunctions(::QuadraticElement{1, 3})

Return the three quadratic shape functions and gradients on the reference line
`-1 <= ξ <= 1`.
"""
function ShapeFunctions(::QuadraticElement{1, 3})
    N1 = ξ -> ξ * (ξ - 1) * 0.5
    N2 = ξ -> 1 - ξ^2
    N3 = ξ -> ξ * (ξ + 1) * 0.5
    ∇N1 = ξ -> ξ - 0.5
    ∇N2 = ξ -> -2ξ
    ∇N3 = ξ -> ξ + 0.5
    return ShapeFunctions((N1, N2, N3), (∇N1, ∇N2, ∇N3))
end

"""
    ShapeFunctions(::QuadraticElement{2, 6})

Return the six quadratic shape functions and gradients on the reference
triangle `ξ >= 0`, `η >= 0`, and `ξ + η <= 1`.
"""
function ShapeFunctions(::QuadraticElement{2, 6})
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
    ∇N2 = (ξ, η) -> (4ξ - 1, 0.0)
    ∇N3 = (ξ, η) -> (0.0, 4η - 1)
    ∇N4 = (ξ, η) -> (4 * (1 - 2ξ - η), -4ξ)
    ∇N5 = (ξ, η) -> (4η, 4ξ)
    ∇N6 = (ξ, η) -> (-4η, 4 * (1 - ξ - 2η))

    return ShapeFunctions((N1, N2, N3, N4, N5, N6), (∇N1, ∇N2, ∇N3, ∇N4, ∇N5, ∇N6))
end

_quadratic_line_N1(ξ) = ξ * (ξ - 1) * 0.5
_quadratic_line_N2(ξ) = 1 - ξ^2
_quadratic_line_N3(ξ) = ξ * (ξ + 1) * 0.5
_quadratic_line_∇N1(ξ) = ξ - 0.5
_quadratic_line_∇N2(ξ) = -2ξ
_quadratic_line_∇N3(ξ) = ξ + 0.5

_linear_line_N(ξ, side) = side == -1 ? (1 - ξ) * 0.5 : (1 + ξ) * 0.5
_linear_line_∇N(side) = side == -1 ? -0.5 : +0.5

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
function ShapeFunctions(::LinearElement{2, 3})
    N1 = (ξ, η) -> 1 - ξ - η
    N2 = (ξ, η) -> ξ
    N3 = (ξ, η) -> η
    ∇N1 = (ξ, η) -> (-1.0, -1.0)
    ∇N2 = (ξ, η) -> (+1.0, 0.0)
    ∇N3 = (ξ, η) -> (0.0, +1.0)
    return ShapeFunctions((N1, N2, N3), (∇N1, ∇N2, ∇N3))
end

"""
    ShapeFunctions(::LinearElement{2, 4})

Return the four bilinear shape functions and gradients on the reference
quadrilateral `-1 <= ξ <= 1`, `-1 <= η <= 1`.
"""
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

"""
    ShapeFunctions(::QuadraticElement{2, 9})

Return the nine tensor-product quadratic shape functions and gradients on the
reference quadrilateral `-1 <= ξ <= 1`, `-1 <= η <= 1`.
"""
function ShapeFunctions(::QuadraticElement{2, 9})
    N1 = (ξ, η) -> _quadratic_line_N1(ξ) * _quadratic_line_N1(η)
    N2 = (ξ, η) -> _quadratic_line_N3(ξ) * _quadratic_line_N1(η)
    N3 = (ξ, η) -> _quadratic_line_N3(ξ) * _quadratic_line_N3(η)
    N4 = (ξ, η) -> _quadratic_line_N1(ξ) * _quadratic_line_N3(η)
    N5 = (ξ, η) -> _quadratic_line_N2(ξ) * _quadratic_line_N1(η)
    N6 = (ξ, η) -> _quadratic_line_N3(ξ) * _quadratic_line_N2(η)
    N7 = (ξ, η) -> _quadratic_line_N2(ξ) * _quadratic_line_N3(η)
    N8 = (ξ, η) -> _quadratic_line_N1(ξ) * _quadratic_line_N2(η)
    N9 = (ξ, η) -> _quadratic_line_N2(ξ) * _quadratic_line_N2(η)

    ∇N1 = (ξ, η) -> (
        _quadratic_line_∇N1(ξ) * _quadratic_line_N1(η),
        _quadratic_line_N1(ξ) * _quadratic_line_∇N1(η),
    )
    ∇N2 = (ξ, η) -> (
        _quadratic_line_∇N3(ξ) * _quadratic_line_N1(η),
        _quadratic_line_N3(ξ) * _quadratic_line_∇N1(η),
    )
    ∇N3 = (ξ, η) -> (
        _quadratic_line_∇N3(ξ) * _quadratic_line_N3(η),
        _quadratic_line_N3(ξ) * _quadratic_line_∇N3(η),
    )
    ∇N4 = (ξ, η) -> (
        _quadratic_line_∇N1(ξ) * _quadratic_line_N3(η),
        _quadratic_line_N1(ξ) * _quadratic_line_∇N3(η),
    )
    ∇N5 = (ξ, η) -> (
        _quadratic_line_∇N2(ξ) * _quadratic_line_N1(η),
        _quadratic_line_N2(ξ) * _quadratic_line_∇N1(η),
    )
    ∇N6 = (ξ, η) -> (
        _quadratic_line_∇N3(ξ) * _quadratic_line_N2(η),
        _quadratic_line_N3(ξ) * _quadratic_line_∇N2(η),
    )
    ∇N7 = (ξ, η) -> (
        _quadratic_line_∇N2(ξ) * _quadratic_line_N3(η),
        _quadratic_line_N2(ξ) * _quadratic_line_∇N3(η),
    )
    ∇N8 = (ξ, η) -> (
        _quadratic_line_∇N1(ξ) * _quadratic_line_N2(η),
        _quadratic_line_N1(ξ) * _quadratic_line_∇N2(η),
    )
    ∇N9 = (ξ, η) -> (
        _quadratic_line_∇N2(ξ) * _quadratic_line_N2(η),
        _quadratic_line_N2(ξ) * _quadratic_line_∇N2(η),
    )

    return ShapeFunctions(
        (N1, N2, N3, N4, N5, N6, N7, N8, N9),
        (∇N1, ∇N2, ∇N3, ∇N4, ∇N5, ∇N6, ∇N7, ∇N8, ∇N9),
    )
end

"""
    ShapeFunctions(::LinearElement{3, 8})

Return the eight trilinear shape functions and gradients on the reference
hexahedron `-1 <= ξ, η, ζ <= 1`.
"""
function ShapeFunctions(::LinearElement{3, 8})
    nodes = (
        (-1, -1, -1),
        (+1, -1, -1),
        (+1, +1, -1),
        (-1, +1, -1),
        (-1, -1, +1),
        (+1, -1, +1),
        (+1, +1, +1),
        (-1, +1, +1),
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
function ShapeFunctions(::QuadraticElement{3, 27})
    nodes = (
        (-1, -1, -1),
        (+1, -1, -1),
        (+1, +1, -1),
        (-1, +1, -1),
        (-1, -1, +1),
        (+1, -1, +1),
        (+1, +1, +1),
        (-1, +1, +1),
        (0, -1, -1),
        (+1, 0, -1),
        (0, +1, -1),
        (-1, 0, -1),
        (0, -1, +1),
        (+1, 0, +1),
        (0, +1, +1),
        (-1, 0, +1),
        (-1, -1, 0),
        (+1, -1, 0),
        (+1, +1, 0),
        (-1, +1, 0),
        (0, 0, -1),
        (0, -1, 0),
        (+1, 0, 0),
        (0, +1, 0),
        (-1, 0, 0),
        (0, 0, +1),
        (0, 0, 0),
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

@generated function _eval_shape_function(N::NTuple{M, Any}, coords) where {M}
    quote
        @inline
        Base.@nexprs $M i -> x_i = N[i](coords...)
        Base.@ncall $M SVector x
    end
end
