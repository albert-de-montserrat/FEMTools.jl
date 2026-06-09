"""
    AbstractElement

Abstract supertype for finite element reference-element tags.
"""
abstract type AbstractElement{nDim, nVert, T} end

"""
    AbstractLinearElement{nDim, nVert}

Abstract supertype for linear finite element reference-element tags.
"""
abstract type AbstractLinearElement{nDim, nVert, T} <: AbstractElement{nDim, nVert, T} end

"""
    AbstractQuadraticElement{nDim, nVert}

Abstract supertype for quadratic finite element reference-element tags.
"""
abstract type AbstractQuadraticElement{nDim, nVert, T} <: AbstractElement{nDim, nVert, T} end

"""
    AbstractShapeFunction

Abstract supertype for containers that store element shape functions.
"""
abstract type AbstractShapeFunction end

"""
    AbstractIntegrationPoints

Abstract supertype for containers that store reference-element integration
points.
"""
abstract type AbstractIntegrationPoints end

"""
    LinearElement{nDim, nVert}

Reference element tag for a linear element with `nDim` reference dimensions
and `nVert` vertices.

Implemented linear node orderings:

```text
LinearElement{1, 2}

    1 -------- 2
   ξ=-1      ξ=+1


LinearElement{2, 3}

    η
    ^
    |
    3
    |\
    | \
    |  \
    1---2 --> ξ


LinearElement{2, 4}

    η
    ^
    |
    4---3
    |   |
    |   |
    1---2 --> ξ


LinearElement{3, 8}

        ζ
        ^
        |
        5--------6
       /|       /|
      8--------7 |
      | |      | |
      | 1------|-2 --> ξ
      |/       |/
      4--------3
     /
    η
```
"""
struct LinearElement{nDim, nVert, T} <: AbstractLinearElement{nDim, nVert, T} end

"""
    QuadraticElement{nDim, nVert}

Reference element tag for a quadratic element with `nDim` reference dimensions
and `nVert` vertices.

Implemented quadratic node orderings:

```text
QuadraticElement{1, 3}

    1 ---- 2 ---- 3
   ξ=-1  ξ=0    ξ=+1


QuadraticElement{2, 6}

    η
    ^
    |
    3
    |\
    | \
    6  5
    |   \
    |    \
    1--4--2 --> ξ


QuadraticElement{2, 9}

    η
    ^
    |
    4--7--3
    |  |  |
    8--9--6
    |  |  |
    1--5--2 --> ξ


QuadraticElement{3, 27}

    ζ = -1 layer

        η
        ^
        |
        4--11--3
        |  |   |
       12--21-10
        |  |   |
        1---9--2 --> ξ

    ζ = 0 layer

       20--24-19
        |  |   |
       25--27-23
        |  |   |
       17--22-18

    ζ = +1 layer

        8--15--7
        |  |   |
       16--26-14
        |  |   |
        5--13--6
```
"""
struct QuadraticElement{nDim, nVert, T} <: AbstractQuadraticElement{nDim, nVert, T} end

"""
    CubicElement{nDim, nVert}

Reference element tag for a cubic element with `nDim` reference dimensions
and `nVert` vertices.

No cubic node ordering is implemented yet.
"""
struct CubicElement{nDim, nVert, T} <: AbstractElement{nDim, nVert, T} end

"""
    ReferenceElement{Element, SF, IP}

Bundle shape functions and integration points for one reference element.

Fields:
- `shape_functions`: shape-function data.
- `integration_points`: integration-point data.
"""
struct ReferenceElement{Element<:AbstractElement, SF, IP}
    shape_functions::SF
    integration_points::IP

    function ReferenceElement(element::T) where {T<:AbstractElement}
        shape_functions = ShapeFunctions(element)
        integration_points = IntegrationPoints(element)

        new{T, typeof(shape_functions), typeof(integration_points)}(
            shape_functions,
            integration_points,
        )
    end
end

ReferenceElement(::Type{T}) where {T<:AbstractElement} = ReferenceElement(T())

###
# OTHER FUNCTIONS
###

Base.length(::ReferenceElement{T}) where {T<:AbstractElement} = length(T())
Base.length(::AbstractElement{nDim, nVert}) where {nDim, nVert} = nVert

"""
    order(element)
    order(ElementType)

Return the polynomial order associated with an element tag, element type, or
`ReferenceElement`.
"""
@inline order(::ReferenceElement{T}) where {T} = order(T)
@inline order(::Type{T}) where T<:AbstractElement = order(T())
@inline order(::AbstractLinearElement) = 1
@inline order(::AbstractQuadraticElement) = 2
