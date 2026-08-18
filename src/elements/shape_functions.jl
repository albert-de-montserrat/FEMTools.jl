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
    ShapeFunctions(N, ∇N) = new{typeof(N), typeof(∇N)}(N, ∇N)
end

"""
    ShapeFunctions(::Type{T}) where {T<:AbstractElement{nDim, nVert}}

Construct shape functions for element type `T`.
"""
ShapeFunctions(::Type{T}) where {nDim, nVert, T<:AbstractElement{nDim, nVert}} = ShapeFunctions(T())
ShapeFunctions(::Type{LinearElement{nDim, nVert}}) where {nDim, nVert} =
    ShapeFunctions(LinearElement{nDim, nVert, Float64})
ShapeFunctions(::Type{QuadraticElement{nDim, nVert}}) where {nDim, nVert} =
    ShapeFunctions(QuadraticElement{nDim, nVert, Float64})

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

"""
    ShapeFunctions(::QuadraticElement{2, 7})

Return the seven shape functions and gradients for the T7 element: the
standard T6 quadratic triangle enriched with a cubic bubble at the centroid.

Node layout (same as T6 for nodes 1–6; node 7 at the centroid):

```
    3
    |\\
    6  5
    |   \\
    1--4--2
       7 (centroid, at ξ=η=1/3)
```

The bubble function is `N7 = 27·L1·L2·L3` (value 1 at centroid, 0 on all
edges). Nodes 1–6 are corrected to preserve the Kronecker-delta and
partition-of-unity properties:

    Nᵢ = Nᵢ_T6 + N7/9    (corner nodes 1–3)
    Nᵢ = Nᵢ_T6 − 4·N7/9  (edge-midpoint nodes 4–6)
"""
function ShapeFunctions(::QuadraticElement{2, 7, T}) where T
    # Bubble and its partial derivatives
    bubble(ξ, η)   = T(27) * (1 - ξ - η) * ξ * η
    ∂b∂ξ(ξ, η)    = T(27) * η * (1 - 2ξ - η)
    ∂b∂η(ξ, η)    = T(27) * ξ * (1 - ξ - 2η)

    # Corner nodes: T6 function + bubble/9
    N1 = (ξ, η) -> begin
        L1 = 1 - ξ - η
        L1 * (2L1 - 1) + bubble(ξ, η) / 9
    end
    N2 = (ξ, η) -> ξ * (2ξ - 1) + bubble(ξ, η) / 9
    N3 = (ξ, η) -> η * (2η - 1) + bubble(ξ, η) / 9

    # Edge-midpoint nodes: T6 function − 4·bubble/9
    N4 = (ξ, η) -> 4 * (1 - ξ - η) * ξ - 4 * bubble(ξ, η) / 9
    N5 = (ξ, η) -> 4 * ξ * η            - 4 * bubble(ξ, η) / 9
    N6 = (ξ, η) -> 4 * η * (1 - ξ - η) - 4 * bubble(ξ, η) / 9

    # Bubble node
    N7 = (ξ, η) -> bubble(ξ, η)

    # Gradients — (∂N/∂ξ, ∂N/∂η)
    ∇N1 = (ξ, η) -> begin
        L1 = 1 - ξ - η
        (1 - 4L1 + ∂b∂ξ(ξ, η) / 9,
         1 - 4L1 + ∂b∂η(ξ, η) / 9)
    end
    ∇N2 = (ξ, η) -> (4ξ - 1 + ∂b∂ξ(ξ, η) / 9,
                           ∂b∂η(ξ, η) / 9)
    ∇N3 = (ξ, η) -> (      ∂b∂ξ(ξ, η) / 9,
                      4η - 1 + ∂b∂η(ξ, η) / 9)
    ∇N4 = (ξ, η) -> (4 * (1 - 2ξ - η) - 4 * ∂b∂ξ(ξ, η) / 9,
                      -4ξ               - 4 * ∂b∂η(ξ, η) / 9)
    ∇N5 = (ξ, η) -> (4η - 4 * ∂b∂ξ(ξ, η) / 9,
                      4ξ - 4 * ∂b∂η(ξ, η) / 9)
    ∇N6 = (ξ, η) -> (-4η               - 4 * ∂b∂ξ(ξ, η) / 9,
                      4 * (1 - ξ - 2η) - 4 * ∂b∂η(ξ, η) / 9)
    ∇N7 = (ξ, η) -> (∂b∂ξ(ξ, η), ∂b∂η(ξ, η))

    return ShapeFunctions(
        (N1, N2, N3, N4, N5, N6, N7),
        (∇N1, ∇N2, ∇N3, ∇N4, ∇N5, ∇N6, ∇N7),
    )
end

_quadratic_line_N1(ξ::T) where T = ξ * (ξ - 1) * T(1/2)
_quadratic_line_N2(ξ::T) where T = 1 - ξ^2
_quadratic_line_N3(ξ::T) where T = ξ * (ξ + 1) * T(1/2)
_quadratic_line_∇N1(ξ::T) where T = ξ - T(1/2)
_quadratic_line_∇N2(ξ::T) where T = -2ξ
_quadratic_line_∇N3(ξ::T) where T = ξ + T(1/2)

# Tensor-product quadrilateral and hexahedral elements reuse the 1D Lagrange
# basis. The `node` argument is the reference coordinate of the local node in
# that coordinate direction.
_linear_line_N(ξ::T, node) where T = node == -1 ? (1 - ξ) * T(1/2) : (1 + ξ) * T(1/2)
_linear_line_∇N(ξ::T, node) where T = node == -1 ? -T(1 / 2) : +T(1 / 2)

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
    _tensor_product_shape_functions(line_N, line_∇N, nodes) -> ShapeFunctions

Build the shape functions of a tensor-product element from a one-dimensional
Lagrange basis.

`nodes[a]` gives the reference coordinate of local node `a` in each direction,
so `Nₐ(x) = ∏_d line_N(x_d, nodes[a][d])`. Each gradient component replaces one
factor with `line_∇N` in the matching direction.
"""
function _tensor_product_shape_functions(
        line_N, line_∇N,
        nodes::Tuple{NTuple{D, Any}, Vararg{NTuple{D, Any}, M}},
    ) where {D, M}
    N = ntuple(Val(M + 1)) do a
        node = nodes[a]
        (x::Vararg{Any, D}) -> prod(ntuple(d -> line_N(x[d], node[d]), Val(D)))
    end
    ∇N = ntuple(Val(M + 1)) do a
        node = nodes[a]
        (x::Vararg{Any, D}) -> ntuple(Val(D)) do i
            prod(ntuple(d -> (d == i ? line_∇N : line_N)(x[d], node[d]), Val(D)))
        end
    end
    return ShapeFunctions(N, ∇N)
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
    # Local node coordinates follow the ordering documented in
    # `LinearElement{2, 4}`.
    nodes = (
        (-one(T), -one(T)),
        (+one(T), -one(T)),
        (+one(T), +one(T)),
        (-one(T), +one(T)),
    )
    return _tensor_product_shape_functions(_linear_line_N, _linear_line_∇N, nodes)
end

"""
    ShapeFunctions(::QuadraticElement{2, 9})

Return the nine tensor-product quadratic shape functions and gradients on the
reference quadrilateral `-1 <= ξ <= 1`, `-1 <= η <= 1`.
"""
function ShapeFunctions(::QuadraticElement{2, 9, T}) where T
    # Local node coordinates follow the ordering documented in
    # `QuadraticElement{2, 9}`.
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
    return _tensor_product_shape_functions(_quadratic_line_N, _quadratic_line_∇N, nodes)
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
    ShapeFunctions(::QuadraticElement{3, 10})

Return the ten quadratic Lagrange shape functions and gradients on the
reference tetrahedron. Nodes 1–4 are the vertices; nodes 5–10 are the edge
midpoints (12, 23, 13, 14, 24, 34).
"""
function ShapeFunctions(::QuadraticElement{3, 10, T}) where T
    gradients = ((-one(T), -one(T), -one(T)), (one(T), zero(T), zero(T)),
                 (zero(T), one(T), zero(T)), (zero(T), zero(T), one(T)))
    barycentric = ((ξ, η, ζ) -> 1 - ξ - η - ζ, (ξ, η, ζ) -> ξ,
                   (ξ, η, ζ) -> η, (ξ, η, ζ) -> ζ)
    edges = ((1, 2), (2, 3), (1, 3), (1, 4), (2, 4), (3, 4))

    N = (
        ntuple(i -> (ξ, η, ζ) -> begin
            L = barycentric[i](ξ, η, ζ)
            L * (2L - 1)
        end, Val(4))...,
        ntuple(i -> (ξ, η, ζ) -> begin
            a, b = edges[i]
            4 * barycentric[a](ξ, η, ζ) * barycentric[b](ξ, η, ζ)
        end, Val(6))...,
    )
    ∇N = (
        ntuple(i -> (ξ, η, ζ) -> begin
            L = barycentric[i](ξ, η, ζ)
            ntuple(j -> (4L - 1) * gradients[i][j], Val(3))
        end, Val(4))...,
        ntuple(i -> (ξ, η, ζ) -> begin
            a, b = edges[i]
            La, Lb = barycentric[a](ξ, η, ζ), barycentric[b](ξ, η, ζ)
            ntuple(j -> 4 * (gradients[a][j] * Lb + La * gradients[b][j]), Val(3))
        end, Val(6))...,
    )
    return ShapeFunctions(N, ∇N)
end

"""
    ShapeFunctions(::QuadraticElement{3, 11})

Return the T10 basis enriched by a centroid bubble. The tetrahedral bubble is
the quartic polynomial `256L₁L₂L₃L₄`; the T10 functions are corrected
so all eleven functions retain the Kronecker-delta and partition-of-unity
properties.
"""
function ShapeFunctions(::QuadraticElement{3, 11, T}) where T
    t10 = ShapeFunctions(QuadraticElement{3, 10, T}())
    bubble(ξ, η, ζ) = T(256) * (1 - ξ - η - ζ) * ξ * η * ζ
    ∇bubble(ξ, η, ζ) = (
        T(256) * η * ζ * (1 - 2ξ - η - ζ),
        T(256) * ξ * ζ * (1 - ξ - 2η - ζ),
        T(256) * ξ * η * (1 - ξ - η - 2ζ),
    )
    coefficients = (ntuple(_ -> T(1/8), Val(4))..., ntuple(_ -> -T(1/4), Val(6))...)
    N = (ntuple(i -> (ξ, η, ζ) -> t10.N[i](ξ, η, ζ) + coefficients[i] * bubble(ξ, η, ζ), Val(10))...,
         bubble)
    ∇N = (ntuple(i -> (ξ, η, ζ) -> begin
              g, gb = t10.∇N[i](ξ, η, ζ), ∇bubble(ξ, η, ζ)
              ntuple(j -> g[j] + coefficients[i] * gb[j], Val(3))
          end, Val(10))..., ∇bubble)
    return ShapeFunctions(N, ∇N)
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

    return _tensor_product_shape_functions(_linear_line_N, _linear_line_∇N, nodes)
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


    return _tensor_product_shape_functions(_quadratic_line_N, _quadratic_line_∇N, nodes)
end

"""
    eval_shape_function(element, coords)

Evaluate all shape functions of a `ReferenceElement` at reference coordinates
`coords`.

`coords` is an `NTuple`, for example `(ξ,)` on a line or `(ξ, η)` on a
two-dimensional reference element. The return value is an `SVector` ordered by
local node number.

# Examples
```jldoctest
julia> using StaticArrays

julia> el = ReferenceElement(LinearElement{1, 2});

julia> eval_shape_function(el, (0.0,))
2-element SVector{2, Float64} with indices SOneTo(2):
 0.5
 0.5
```
"""
@inline function eval_shape_function(element, coords::Tuple{T, Vararg{T}}) where {T}
    return _eval_shape_function(element.shape_functions.N, coords)
end

@inline eval_shape_function(element, coords::SVector) = eval_shape_function(element, tuple(coords...))

"""
    eval_shape_function_gradient(element, coords)

Evaluate the stored reference-coordinate gradients of all shape functions of a
`ReferenceElement` at `coords`.
"""
@inline function eval_shape_function_gradient(element, coords::Tuple{T, Vararg{T}}) where {T}
    return _eval_shape_function(element.shape_functions.∇N, coords)
end

@inline eval_shape_function_gradient(element, coords::SVector) =
    eval_shape_function_gradient(element, tuple(coords...))

"""
    eval_shape_function_jacobian(element, coords)

Compute the Jacobian of the shape-function vector with respect to reference
coordinates at `coords`.

The implementation uses the stored analytical gradients for element families
where that is faster, and `ForwardDiff` otherwise.
"""
@inline function eval_shape_function_jacobian(element, coords::Tuple{T, Vararg{T}}) where {T}
    return eval_shape_function_jacobian(element, SVector(coords))
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

for (Element, Dim) in (
    (QuadraticElement{2, 7}, 2),
    (QuadraticElement{2, 9}, 2),
    (QuadraticElement{3, 11}, 3),
    (LinearElement{3, 8}, 3),
)
    @eval @inline function eval_shape_function_jacobian(
        element::ReferenceElement{<:$Element},
        coords::SVector{$Dim},
    )
        return _gradient_matrix(eval_shape_function_gradient(element, coords))
    end
end

@generated function _gradient_matrix(gradients::SVector{N, G}) where {N, G<:Tuple}
    M = fieldcount(G)
    entries = [:(gradients[$i][$j]) for j in 1:M for i in 1:N]
    return :(SMatrix{$N, $M}($(entries...)))
end


"""
    shape_function_values(element)
    shape_function_values(element, ip)

Return shape-function values of `element` evaluated at every quadrature point.

With one argument, the points are `element.integration_points`. Pass an
`IntegrationPoints` object `ip` to evaluate at a different quadrature rule
(e.g. a pressure element's shape functions at the velocity element's points).

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
        @nexprs $M i -> x_i = N[i](coords...)
        @ncall $M SVector x
    end
end
