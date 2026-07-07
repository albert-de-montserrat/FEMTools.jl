"""
    AbstractMesh

Abstract supertype for mesh containers.
"""
abstract type AbstractMesh end

"""
    Mesh(Ω, nels)
    Mesh(Ω, element, nels)
    Mesh(backend, Ω, element, nels)

Construct a structured mesh over domain `Ω` using `element` and `nels`.

The constructor derives the boundary `Γ`, coordinates, degrees of freedom,
element-to-node connectivity, boundary nodes, and stores the node and element
counts. For tensor-product domains, `nels` is a tuple such as `(nx, ny)` or
`(nx, ny, nz)`.

    Mesh(coords, el2n)
    Mesh(backend, coords, el2n)

Construct an unstructured mesh from pre-built arrays.

`coords` is a `Vector{SVector{nDim, T}}` of node coordinates and `el2n` is an
`N × nels` `Matrix{Int32}` of element-to-node connectivity (one column per
element). Boundary nodes are detected automatically as nodes on mesh edges
shared by exactly one element. `Ω` and `Γ` are set to `nothing`.
"""
struct Mesh{nDim, O, D, B, T1, T2, T3, T4} <: AbstractMesh
    Ω::D        # model domain
    Γ::B        # model boundary
    coords::T1  # vertex coordinates
    DoFs::T2    # degrees of freedom
    el2n::T3    # element-to-node connectivity
    Γnodes::T4  # boundary nodes
    nnodes::Int # number of nodes
    nels::Int   # number of elements

    function Mesh{nDim, O, D, B, T1, T2, T3, T4}(
        Ω::D,
        Γ::B,
        coords::T1,
        DoFs::T2,
        el2n::T3,
        Γnodes::T4,
        nnodes::Int,
        nels::Int,
    ) where {nDim, O, D, B, T1, T2, T3, T4}
        return new{nDim, O, D, B, T1, T2, T3, T4}(Ω, Γ, coords, DoFs, el2n, Γnodes, nnodes, nels)
    end

    function Mesh(backend, Ω, element::ReferenceElement{T}, nels) where T<:AbstractElement{nDim} where nDim

        TDev       = TA(backend)
        Γ          = boundary(Ω)
        coords_cpu = generate_coordinates(element, Ω, nels)
        nnodes     = length(coords_cpu)
        DoFs_cpu   = generate_dofs(element, nnodes)
        el2n_cpu   = generate_element2node(element, nels)
        Γmask       = Bool[p ∈ Γ for p in coords_cpu]
        Γnodes_cpu  = Vector{Int32}(DoFs_cpu[Γmask])

        coords = TDev(coords_cpu)
        DoFs   = TDev(DoFs_cpu)
        el2n   = TDev(el2n_cpu)
        Γnodes = TDev(Γnodes_cpu)

        return new{
            nDim,
            order(element),
            typeof(Ω),
            typeof(Γ),
            typeof(coords),
            typeof(DoFs),
            typeof(el2n),
            typeof(Γnodes),
        }(Ω, Γ, coords, DoFs, el2n, Γnodes, nnodes, size(el2n_cpu, 2))
    end

    function Mesh(backend, coords_cpu::Vector{SVector{nDim, T}}, el2n_cpu::Matrix{Int32}; order::Int = 1) where {nDim, T}
        TDev       = TA(backend)
        nnodes     = length(coords_cpu)
        DoFs_cpu   = Int32.(1:nnodes)
        Γnodes_cpu = _unstructured_boundary_nodes(el2n_cpu)

        coords = TDev(coords_cpu)
        DoFs   = TDev(DoFs_cpu)
        el2n   = TDev(el2n_cpu)
        Γnodes = TDev(Γnodes_cpu)

        return new{nDim, order, Nothing, Nothing, typeof(coords), typeof(DoFs), typeof(el2n), typeof(Γnodes)}(
            nothing, nothing, coords, DoFs, el2n, Γnodes, nnodes, size(el2n_cpu, 2)
        )
    end
end

Mesh(Ω, element, nels) = Mesh(CPU(), Ω, element, nels)
Mesh(coords_cpu::Vector{<:SVector}, el2n_cpu::Matrix{Int32}; kwargs...) =
    Mesh(CPU(), coords_cpu, el2n_cpu; kwargs...)

"""
    Mesh(element, Ω, Γ, coords, DoFs, el2n, Γnodes)

Construct a `Mesh` from pre-assembled arrays, bypassing the structured generator.

Use this when the coordinates, connectivity, and boundary nodes have already been
built externally (e.g. from an imported mesh or after manually postprocessing a
structured mesh). `element` supplies the polynomial order stored in the type
parameter `O`. The remaining arguments are stored verbatim: no generation, no
boundary detection.
"""
function Mesh(
    element::ReferenceElement{T},
    Ω::D,        # model domain
    Γ::B,        # model boundary
    coords::Vector{SVector{nDim, FP}},  # vertex coordinates
    DoFs::T2,    # degrees of freedom
    el2n::T3,    # element-to-node connectivity
    Γnodes::T4,  # boundary nodes
) where {D, B, nDim, FP, T2, T3, T4, T<:AbstractElement{nDim}}

    return Mesh{
        nDim,
        order(element),
        D,
        B,
        typeof(coords),
        T2,
        T3,
        T4,
    }(Ω, Γ, coords, DoFs, el2n, Γnodes, length(coords), size(el2n, 2))

end

"""
    _unstructured_boundary_nodes(el2n) -> Vector{Int32}

Return sorted unique node indices that lie on the mesh boundary.

A mesh edge (consecutive node pair within an element) is a boundary edge when
it appears in exactly one element. All nodes incident to such edges are
collected and returned. Assumes elements are ordered so that consecutive rows
of `el2n` form edges (i.e. the last node wraps to the first).
"""
function _unstructured_boundary_nodes(el2n::Matrix{Int32})
    N = size(el2n, 1)
    edge_count = Dict{Tuple{Int32, Int32}, Int}()
    for iel in axes(el2n, 2)
        for i in 1:N
            j = mod1(i + 1, N)
            a, b = minmax(el2n[i, iel], el2n[j, iel])
            edge_count[(a, b)] = get(edge_count, (a, b), 0) + 1
        end
    end
    bnd = Int32[]
    for ((a, b), count) in edge_count
        count == 1 && push!(bnd, a, b)
    end
    return sort!(unique!(bnd))
end


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
    generate_coordinates(element::ReferenceElement{<:LinearElement{2, 3}}, Ω, nels)

Generate coordinates for a structured linear triangular mesh over a rectangular
domain. Each quadrilateral cell is split into two triangles, so the node grid is
the same `(nx+1) × (ny+1)` layout as the quadrilateral case.
"""
function generate_coordinates(
    ::ReferenceElement{LinearElement{2, 3, T}},
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
    generate_coordinates(element::ReferenceElement{<:QuadraticElement{2, 7}}, Ω, nels)

Generate coordinates for a structured T7 mesh (T6 + centroid bubble node).

The first `(2nx+1)×(2ny+1)` entries are the T6 tensor-grid nodes (identical
to `QuadraticElement{2, 6}`). The remaining `2·nx·ny` entries are the element
centroids, appended one per element in the same loop order used by
`generate_element2node(QuadraticElement{2, 7})`: `ey` outer, `ex` inner,
first triangle then second triangle of each quad cell.
"""
function generate_coordinates(
    ::ReferenceElement{QuadraticElement{2, 7, T}},
    Ω,
    nels::NTuple{2, <:Integer},
) where {T}
    nx, ny = nels
    left   = leftendpoint(Ω)
    right  = rightendpoint(Ω)
    x0, y0 = left[1], left[2]
    dx = (right[1] - x0) / nx
    dy = (right[2] - y0) / ny

    xs = LinRange(x0, right[1], 2nx + 1)
    ys = LinRange(y0, right[2], 2ny + 1)

    coords = Vector{SVector{2, T}}(undef, length(xs) * length(ys) + 2 * nx * ny)

    inode = 1
    for y in ys, x in xs
        coords[inode] = SVector{2, T}(x, y)
        inode += 1
    end

    # Centroid nodes — one per element, matching generate_element2node order.
    for ey in 0:(ny - 1), ex in 0:(nx - 1)
        if iseven(ex + ey)
            # Triangle A (lower-right): BL + BR + TR centroid
            coords[inode] = SVector{2, T}(x0 + (3ex + 2) * dx / 3, y0 + (3ey + 1) * dy / 3)
            inode += 1
            # Triangle B (upper-left): BL + TR + TL centroid
            coords[inode] = SVector{2, T}(x0 + (3ex + 1) * dx / 3, y0 + (3ey + 2) * dy / 3)
            inode += 1
        else
            # Triangle C (lower-left): BL + BR + TL centroid
            coords[inode] = SVector{2, T}(x0 + (3ex + 1) * dx / 3, y0 + (3ey + 1) * dy / 3)
            inode += 1
            # Triangle D (upper-right): BR + TR + TL centroid
            coords[inode] = SVector{2, T}(x0 + (3ex + 2) * dx / 3, y0 + (3ey + 2) * dy / 3)
            inode += 1
        end
    end

    return coords
end

"""
    generate_coordinates(element::ReferenceElement{<:QuadraticElement{2, 6}}, Ω, nels)

Generate coordinates for a structured quadratic triangular (T6) mesh.

Nodes lie on the same refined `(2nx + 1) × (2ny + 1)` tensor-product grid as
`QuadraticElement{2, 9}`, with the x-coordinate varying fastest. Corner nodes
occupy even grid positions, edge-midpoint nodes occupy positions where exactly
one index is odd, and cell-center nodes (diagonal midpoints) occupy positions
where both indices are odd.
"""
function generate_coordinates(
    ::ReferenceElement{QuadraticElement{2, 6, T}},
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

# ---------------------------------------------------------------------------
# KA kernels
# ---------------------------------------------------------------------------

"""
    precompute_geometry_kernel!(geo, coords, el2n, ∂N∂ξq, ω, Val(N))

KernelAbstractions kernel that fills `geo` with per-element geometry data.

For each element `iel`, computes `(∂N∂x_q, dΩ_q)` at every quadrature point `q`
and stores the result as a tuple at `geo[iel]`. Here `∂N∂x_q` is the matrix of
physical-space shape-function gradients (`N × nDim`) and `dΩ_q` is the
quadrature weight scaled by `|det J|`.

Because this kernel depends only on mesh geometry, it only needs to be called
once per mesh and the result can be reused across nonlinear or pseudo-transient
iterations.
"""
@kernel function precompute_geometry_kernel!(geo, @Const(coords), @Const(el2n), ∂N∂ξq, ω, ::Val{N}) where N
    iel = @index(Global)
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    c = element_coordinate_matrix(coords, local_nodes)
    geo[iel] = ntuple(Val(length(ω))) do q
        J = c' * ∂N∂ξq[q]
        (∂N∂ξq[q] * inv(J), abs(det(J)) * ω[q])
    end
end
