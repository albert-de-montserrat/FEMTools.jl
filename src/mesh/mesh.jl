"""
    AbstractMesh

Abstract supertype for mesh containers.
"""
abstract type AbstractMesh end

"""
    Mesh(Ω, nels)

Construct a one-dimensional quadratic mesh over the interval domain `Ω` with
`nels` elements.

The constructor derives the boundary `Γ`, coordinates, degrees of freedom,
element-to-node connectivity, node-to-element adjacency, boundary nodes,
boundary elements, and stores the node/element counts.
"""
struct Mesh{nDim, O, D, B, T1, T2, T3, T4, T5, T6} <: AbstractMesh
    Ω::D        # model domain
    Γ::B        # model boundary
    coords::T1  # vertex coordinates
    DoFs::T2    # degrees of freedom
    el2n::T3    # element-to-node connectivity
    n2el::T4    # node-to-elements connectivity
    Γnodes::T5  # boundary nodes
    Γels::T6    # boundary elements
    nnodes::Int # number of nodes
    nels::Int   # number of elements

    function Mesh(Ω, element::ReferenceElement{nDim}, nels) where {nDim}
        Γ = boundary(Ω)
        coords = generate_coordinates(element, Ω, nels)
        DoFs = generate_dofs(element, length(coords))
        el2n = generate_element2node(element, nels)
        n2el = generate_node2element(el2n, length(coords))
        Γnodes = DoFs[in.(coords, Γ)]
        Γels = generate_boundary_elements(Γnodes, n2el)
        nnodes = length(coords)

        return new{
            nDim,
            order(element),
            typeof(Ω),
            typeof(Γ),
            typeof(coords),
            typeof(DoFs),
            typeof(el2n),
            typeof(n2el),
            typeof(Γnodes),
            typeof(Γels),
        }(Ω, Γ, coords, DoFs, el2n, n2el, Γnodes, Γels, nnodes, length(n2el) == 0 ? 0 : size(el2n, 2))
    end
end

"""
    generate_coordinates(element::ReferenceElement{1, 2}, Ω, nels)

Generate coordinates for a linear one-dimensional mesh over interval `Ω`.
"""
function generate_coordinates(::ReferenceElement{1, 2}, Ω::ClosedInterval, nels)
    domain = LinRange(leftendpoint(Ω), rightendpoint(Ω), nels + 1)
    return collect(domain)
end

"""
    generate_coordinates(element::ReferenceElement{1, 3}, Ω, nels)

Generate coordinates for a quadratic one-dimensional mesh over interval `Ω`.

Vertex nodes occupy odd indices and midpoint nodes occupy even indices.
"""
function generate_coordinates(::ReferenceElement{1, 3}, Ω::ClosedInterval, nels)
    domain = LinRange(leftendpoint(Ω), rightendpoint(Ω), nels + 1)
    npoints = 2 * length(domain) - 1
    coords = zeros(npoints)
    coords[1:2:npoints] .= domain
    for i in 2:2:length(coords)
        coords[i] = (coords[i - 1] + coords[i + 1]) / 2
    end
    return coords
end

"""
    generate_coordinates(element::ReferenceElement{2, 4}, Ω, nels)

Generate coordinates for a linear quadrilateral mesh over a rectangular domain.

The rectangular domain is passed as `(xmin..xmax) × (ymin..ymax)` and the
number of elements as `(nx, ny)`. Coordinates are returned as
`SVector{2, Float64}` values with the x-coordinate varying fastest.
"""
function generate_coordinates(
    ::ReferenceElement{2, 4},
    Ω,
    nels::NTuple{2, <:Integer},
)
    nx, ny = nels
    left = leftendpoint(Ω)
    right = rightendpoint(Ω)
    xs = LinRange(left[1], right[1], nx + 1)
    ys = LinRange(left[2], right[2], ny + 1)

    coords = Vector{SVector{2, Float64}}(undef, length(xs) * length(ys))
    inode = 1
    for y in ys, x in xs
        coords[inode] = SVector{2, Float64}(x, y)
        inode += 1
    end

    return coords
end

"""
    generate_coordinates(element::ReferenceElement{2, 9}, Ω, nels)

Generate coordinates for a quadratic quadrilateral mesh over a rectangular
domain.

The rectangular domain is passed as `(xmin..xmax) × (ymin..ymax)` and the
number of elements as `(nx, ny)`. Vertex, edge-midpoint, and cell-center nodes
lie on the refined `(2nx + 1) × (2ny + 1)` tensor-product grid, with the
x-coordinate varying fastest.
"""
function generate_coordinates(
    ::ReferenceElement{2, 9},
    Ω,
    nels::NTuple{2, <:Integer},
)
    nx, ny = nels
    left = leftendpoint(Ω)
    right = rightendpoint(Ω)
    xs = LinRange(left[1], right[1], 2nx + 1)
    ys = LinRange(left[2], right[2], 2ny + 1)

    coords = Vector{SVector{2, Float64}}(undef, length(xs) * length(ys))
    inode = 1
    for y in ys, x in xs
        coords[inode] = SVector{2, Float64}(x, y)
        inode += 1
    end

    return coords
end

"""
    generate_coordinates(element::ReferenceElement{3, 8}, Ω, nels)

Generate coordinates for a linear hexahedral mesh over a rectangular box.

The box domain is passed as `(xmin..xmax) × (ymin..ymax) × (zmin..zmax)` and the
number of elements as `(nx, ny, nz)`. Coordinates are returned as
`SVector{3, Float64}` values with the x-coordinate varying fastest.
"""
function generate_coordinates(
    ::ReferenceElement{3, 8},
    Ω,
    nels::NTuple{3, <:Integer},
)
    nx, ny, nz = nels
    left = leftendpoint(Ω)
    right = rightendpoint(Ω)
    xs = LinRange(left[1], right[1], nx + 1)
    ys = LinRange(left[2], right[2], ny + 1)
    zs = LinRange(left[3], right[3], nz + 1)

    coords = Vector{SVector{3, Float64}}(undef, length(xs) * length(ys) * length(zs))
    inode = 1
    for z in zs, y in ys, x in xs
        coords[inode] = SVector{3, Float64}(x, y, z)
        inode += 1
    end

    return coords
end

"""
    generate_coordinates(element::ReferenceElement{3, 27}, Ω, nels)

Generate coordinates for a quadratic hexahedral mesh over a rectangular box.

The box domain is passed as `(xmin..xmax) × (ymin..ymax) × (zmin..zmax)` and the
number of elements as `(nx, ny, nz)`. Nodes lie on the refined
`(2nx + 1) × (2ny + 1) × (2nz + 1)` tensor-product grid, with the x-coordinate
varying fastest.
"""
function generate_coordinates(
    ::ReferenceElement{3, 27},
    Ω,
    nels::NTuple{3, <:Integer},
)
    nx, ny, nz = nels
    left = leftendpoint(Ω)
    right = rightendpoint(Ω)
    xs = LinRange(left[1], right[1], 2nx + 1)
    ys = LinRange(left[2], right[2], 2ny + 1)
    zs = LinRange(left[3], right[3], 2nz + 1)

    coords = Vector{SVector{3, Float64}}(undef, length(xs) * length(ys) * length(zs))
    inode = 1
    for z in zs, y in ys, x in xs
        coords[inode] = SVector{3, Float64}(x, y, z)
        inode += 1
    end

    return coords
end

"""
    generate_dofs(element::ReferenceElement{1}, npoints)

Generate one degree of freedom per mesh point for a one-dimensional element.
"""
generate_dofs(::ReferenceElement, npoints) = [Int32(i) for i in 1:npoints]

generete_coordinates(args...) = generate_coordinates(args...)
generete_DoFs(args...) = generate_dofs(args...)
