"""
    AbstractMesh

Abstract supertype for mesh containers.
"""
abstract type AbstractMesh end

"""
    Mesh(Ω, nels)
    Mesh(Ω, element, nels)

Construct a structured mesh over domain `Ω` using `element` and `nels`.

The constructor derives the boundary `Γ`, coordinates, degrees of freedom,
element-to-node connectivity, node-to-element adjacency, boundary nodes,
boundary elements, and stores the node and element counts. For tensor-product
domains, `nels` is a tuple such as `(nx, ny)` or `(nx, ny, nz)`.
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

    function Mesh(backend, Ω, element::ReferenceElement{T}, nels) where T<:AbstractElement{nDim} where nDim
        
        TDev       = TA(backend) 
        Γ          = boundary(Ω)
        coords_cpu = generate_coordinates(element, Ω, nels)
        nnodes     = length(coords_cpu)
        DoFs_cpu   = generate_dofs(element, nnodes)
        el2n_cpu   = generate_element2node(element, nels)
        n2el_cpu   = generate_node2element(el2n_cpu, nnodes)
        Γmask       = Bool[p ∈ Γ for p in coords_cpu]
        Γnodes_cpu  = Vector{Int32}(DoFs_cpu[Γmask])
        Γels_cpu    = generate_boundary_elements(Γnodes_cpu, n2el_cpu)

        coords = TDev(coords_cpu)
        DoFs   = TDev(DoFs_cpu)
        el2n   = TDev(el2n_cpu)
        n2el   = n2el_cpu
        Γnodes = TDev(Γnodes_cpu)
        Γels   = TDev(Γels_cpu)

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

Mesh(Ω, element, nels) = Mesh(CPU(), Ω, element, nels)


"""
    generate_coordinates(element::ReferenceElement{<:LinearElement{1, 2}}, Ω, nels)

Generate coordinates for a linear one-dimensional mesh over interval `Ω`.
"""
function generate_coordinates(::ReferenceElement{LinearElement{1, 2, T}}, Ω::ClosedInterval, nels) where {T}
    domain = LinRange(leftendpoint(Ω), rightendpoint(Ω), nels + 1)
    return collect(domain)
end

"""
    generate_coordinates(element::ReferenceElement{<:QuadraticElement{1, 3}}, Ω, nels)

Generate coordinates for a quadratic one-dimensional mesh over interval `Ω`.

Vertex nodes occupy odd indices and midpoint nodes occupy even indices.
"""
function generate_coordinates(::ReferenceElement{QuadraticElement{1, 3, T}}, Ω::ClosedInterval, nels) where {T}
    domain = LinRange(leftendpoint(Ω), rightendpoint(Ω), nels + 1)
    npoints = 2 * length(domain) - 1
    coords = zeros(T, npoints)
    coords[1:2:npoints] .= domain
    for i in 2:2:length(coords)
        coords[i] = (coords[i - 1] + coords[i + 1]) / 2
    end
    return coords
end

"""
    generate_coordinates(element::ReferenceElement{<:LinearElement{2, 4}}, Ω, nels)

Generate coordinates for a linear quadrilateral mesh over a rectangular domain.

The rectangular domain is passed as `(xmin..xmax) × (ymin..ymax)` and the
number of elements as `(nx, ny)`. Coordinates are returned as
`SVector{2, Float64}` values with the x-coordinate varying fastest.
"""
function generate_coordinates(
    ::ReferenceElement{LinearElement{2, 4, T}},
    Ω,
    nels::NTuple{2, <:Integer},
) where {T}
    nx, ny = nels
    left = leftendpoint(Ω)
    right = rightendpoint(Ω)
    xs = LinRange(left[1], right[1], nx + 1)
    ys = LinRange(left[2], right[2], ny + 1)

    coords = Vector{SVector{2, T}}(undef, length(xs) * length(ys))
    inode = 1
    for y in ys, x in xs
        coords[inode] = SVector{2, T}(x, y)
        inode += 1
    end

    return coords
end

"""
    generate_coordinates(element::ReferenceElement{<:QuadraticElement{2, 9}}, Ω, nels)

Generate coordinates for a quadratic quadrilateral mesh over a rectangular
domain.

The rectangular domain is passed as `(xmin..xmax) × (ymin..ymax)` and the
number of elements as `(nx, ny)`. Vertex, edge-midpoint, and cell-center nodes
lie on the refined `(2nx + 1) × (2ny + 1)` tensor-product grid, with the
x-coordinate varying fastest.
"""
function generate_coordinates(
    ::ReferenceElement{QuadraticElement{2, 9, T}},
    Ω,
    nels::NTuple{2, <:Integer},
) where {T}
    nx, ny = nels
    left = leftendpoint(Ω)
    right = rightendpoint(Ω)
    xs = LinRange(left[1], right[1], 2nx + 1)
    ys = LinRange(left[2], right[2], 2ny + 1)

    coords = Vector{SVector{2, T}}(undef, length(xs) * length(ys))
    inode = 1
    for y in ys, x in xs
        coords[inode] = SVector{2, T}(x, y)
        inode += 1
    end

    return coords
end

"""
    generate_coordinates(element::ReferenceElement{<:LinearElement{3, 8}}, Ω, nels)

Generate coordinates for a linear hexahedral mesh over a rectangular box.

The box domain is passed as `(xmin..xmax) × (ymin..ymax) × (zmin..zmax)` and the
number of elements as `(nx, ny, nz)`. Coordinates are returned as
`SVector{3, Float64}` values with the x-coordinate varying fastest.
"""
function generate_coordinates(
    ::ReferenceElement{LinearElement{3, 8, T}},
    Ω,
    nels::NTuple{3, <:Integer},
) where {T}
    nx, ny, nz = nels
    left = leftendpoint(Ω)
    right = rightendpoint(Ω)
    xs = LinRange(left[1], right[1], nx + 1)
    ys = LinRange(left[2], right[2], ny + 1)
    zs = LinRange(left[3], right[3], nz + 1)

    coords = Vector{SVector{3, T}}(undef, length(xs) * length(ys) * length(zs))
    inode = 1
    for z in zs, y in ys, x in xs
        coords[inode] = SVector{3, T}(x, y, z)
        inode += 1
    end

    return coords
end

"""
    generate_coordinates(element::ReferenceElement{<:QuadraticElement{3, 27}}, Ω, nels)

Generate coordinates for a quadratic hexahedral mesh over a rectangular box.

The box domain is passed as `(xmin..xmax) × (ymin..ymax) × (zmin..zmax)` and the
number of elements as `(nx, ny, nz)`. Nodes lie on the refined
`(2nx + 1) × (2ny + 1) × (2nz + 1)` tensor-product grid, with the x-coordinate
varying fastest.
"""
function generate_coordinates(
    ::ReferenceElement{QuadraticElement{3, 27, T}},
    Ω,
    nels::NTuple{3, <:Integer},
) where {T}
    nx, ny, nz = nels
    left = leftendpoint(Ω)
    right = rightendpoint(Ω)
    xs = LinRange(left[1], right[1], 2nx + 1)
    ys = LinRange(left[2], right[2], 2ny + 1)
    zs = LinRange(left[3], right[3], 2nz + 1)

    coords = Vector{SVector{3, T}}(undef, length(xs) * length(ys) * length(zs))
    inode = 1
    for z in zs, y in ys, x in xs
        coords[inode] = SVector{3, T}(x, y, z)
        inode += 1
    end

    return coords
end

"""
    generate_dofs(element, npoints)

Generate one degree of freedom per mesh point.
"""
generate_dofs(::ReferenceElement, npoints) = [Int32(i) for i in 1:npoints]

# Backwards-compatible misspelled aliases retained for older examples.
generete_coordinates(args...) = generate_coordinates(args...)
generete_DoFs(args...) = generate_dofs(args...)

# ---------------------------------------------------------------------------
# KA kernels
# ---------------------------------------------------------------------------

# Precompute `(∂N∂x_q, dΩ_q)` for every element and quadrature point. This
# depends only on the mesh geometry, so time-stepping or pseudo-transient
# iterations can reuse it.
@kernel function precompute_geometry_kernel!(geo, @Const(coords), @Const(el2n), ∂N∂ξq, ω, ::Val{N}) where N
    iel = @index(Global)
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    c = element_coordinate_matrix(coords, local_nodes)
    geo[iel] = ntuple(Val(length(ω))) do q
        J = ∂N∂ξq[q]' * c
        (∂N∂ξq[q] * inv(J), abs(det(J)) * ω[q])
    end
end

"""
    element_coordinate_matrix(coords, local_nodes)

Return the `N × 2` coordinate matrix for a two-dimensional element.

Rows follow `local_nodes`; columns are physical `x` and `y`. The static matrix
layout is intended for small element-local geometry calculations.
"""
@inline function element_coordinate_matrix(coords, local_nodes::SVector{N, Int}) where {N}
    data = ntuple(Val(2N)) do k
        col = cld(k, N)
        row = k - (col - 1) * N
        coords[local_nodes[row]][col]
    end
    return SMatrix{N, 2, Float64, 2N}(data)
end

"""
    local_nodes_of(el2n, iel, Val(N))

Gather the `N` local-to-global node ids for element `iel` as an `SVector`.
"""
@inline local_nodes_of(el2n, iel, ::Val{N}) where N =
    SVector{N, Int}(ntuple(i -> Int(el2n[i, iel]), Val(N)))
