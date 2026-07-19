"""
    AbstractMesh

Abstract supertype for mesh containers.
"""
abstract type AbstractMesh end

"""
    Mesh(Ω, element, nels)
    Mesh(backend, Ω, element, nels)

Construct a structured mesh over domain `Ω` using `element` and `nels`.

The constructor derives the boundary `Γ`, coordinates, degrees of freedom,
element-to-node connectivity, boundary nodes, and stores the node and element
counts. For tensor-product domains, `nels` is a tuple such as `(nx, ny)` or
`(nx, ny, nz)`.

    Mesh(coords, el2n; order=1)
    Mesh(backend, coords, el2n; order=1)
    Mesh(coords, el2n, element; workgroup=256)
    Mesh(backend, coords, el2n, element; workgroup=256)

Construct an unstructured mesh from pre-built arrays.

`coords` is an `AbstractVector` of `SVector{nDim, T}` node coordinates and
`el2n` is an `N × nels` `AbstractMatrix{<:Integer}` of element-to-node
connectivity (one column per element). The keyword `order` declares the
polynomial order of the connectivity, stored in the mesh type parameter
(e.g. `order = 2` for T6 triangles). Boundary nodes are detected automatically
as nodes on mesh edges shared by exactly one element. `Ω` and `Γ` are set to
`nothing`. Passing `element` also stores the reference element and precomputes
the geometry used by single-field solvers. Constructors without `element`
retain `nothing` for both fields.
"""
struct Mesh{nDim, O, D, B, T1, T2, T3, T4, E, G} <: AbstractMesh
    Ω::D        # model domain
    Γ::B        # model boundary
    coords::T1  # vertex coordinates
    DoFs::T2    # degrees of freedom
    el2n::T3    # element-to-node connectivity
    Γnodes::T4  # boundary nodes
    nnodes::Int # number of nodes
    nels::Int   # number of elements
    element::E  # reference element
    geometry::G # precomputed element geometry

    function Mesh{nDim, O, D, B, T1, T2, T3, T4, E, G}(
        Ω,
        Γ,
        coords,
        DoFs,
        el2n,
        Γnodes,
        nnodes,
        nels,
        element,
        geometry,
    ) where {nDim, O, D, B, T1, T2, T3, T4, E, G}
        return new{nDim, O, D, B, T1, T2, T3, T4, E, G}(
            Ω, Γ, coords, DoFs, el2n, Γnodes, nnodes, nels, element, geometry,
        )
    end

    function Mesh(backend, Ω, element::ReferenceElement{T}, nels; workgroup = 256) where T<:AbstractElement{nDim} where nDim

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
        geometry = precompute_geometry(coords, el2n, element; backend, workgroup)

        return new{
            nDim,
            order(element),
            typeof(Ω),
            typeof(Γ),
            typeof(coords),
            typeof(DoFs),
            typeof(el2n),
            typeof(Γnodes),
            typeof(element),
            typeof(geometry),
        }(Ω, Γ, coords, DoFs, el2n, Γnodes, nnodes, size(el2n_cpu, 2), element, geometry)
    end

    Base.@constprop :aggressive function Mesh(
        backend,
        coords_cpu::AbstractVector{<:SVector{nDim}},
        el2n_cpu::AbstractMatrix{<:Integer};
        order::Int = 1,
    ) where {nDim}
        if order == 1
            return _unstructured_mesh(backend, coords_cpu, el2n_cpu, Val(1))
        elseif order == 2
            return _unstructured_mesh(backend, coords_cpu, el2n_cpu, Val(2))
        else
            return _unstructured_mesh(backend, coords_cpu, el2n_cpu, Val(order))
        end
    end
end

function Mesh{nDim, O, D, B, T1, T2, T3, T4}(
    Ω, Γ, coords, DoFs, el2n, Γnodes, nnodes, nels,
) where {nDim, O, D, B, T1, T2, T3, T4}
    return Mesh{nDim, O, D, B, T1, T2, T3, T4, Nothing, Nothing}(
        Ω, Γ, coords, DoFs, el2n, Γnodes, Int(nnodes), Int(nels), nothing, nothing,
    )
end

function _unstructured_mesh(
    backend,
    coords_cpu::AbstractVector{<:SVector{nDim}},
    el2n_cpu::AbstractMatrix{<:Integer},
    ::Val{O},
) where {nDim, O}
        TDev       = TA(backend)
        nnodes     = length(coords_cpu)
        DoFs_cpu   = Int32.(1:nnodes)
        Γnodes_cpu = _unstructured_boundary_nodes(el2n_cpu, Val(nDim))

        coords = TDev(coords_cpu)
        DoFs   = TDev(DoFs_cpu)
        el2n   = TDev(el2n_cpu)
        Γnodes = TDev(Γnodes_cpu)

        return Mesh{nDim, O, Nothing, Nothing, typeof(coords), typeof(DoFs), typeof(el2n), typeof(Γnodes), Nothing, Nothing}(
            nothing, nothing, coords, DoFs, el2n, Γnodes, nnodes, size(el2n_cpu, 2), nothing, nothing,
        )
end

function Mesh(
    backend,
    coords_cpu::AbstractVector{<:SVector{nDim}},
    el2n_cpu::AbstractMatrix{<:Integer},
    element::ReferenceElement{T};
    workgroup = 256,
) where {nDim, T <: AbstractElement{nDim}}
    mesh = Mesh(backend, coords_cpu, el2n_cpu; order = order(element))
    geometry = precompute_geometry(mesh.coords, mesh.el2n, element; backend, workgroup)
    return Mesh{nDim, order(element), Nothing, Nothing,
                typeof(mesh.coords), typeof(mesh.DoFs), typeof(mesh.el2n), typeof(mesh.Γnodes),
                typeof(element), typeof(geometry)}(
        nothing, nothing, mesh.coords, mesh.DoFs, mesh.el2n, mesh.Γnodes,
        mesh.nnodes, mesh.nels, element, geometry,
    )
end

function Base.show(io::IO, mesh::Mesh{nDim, O}) where {nDim, O}
    print(io, "Mesh{", nDim, ", ", O, "}(nnodes=", mesh.nnodes, ", nels=", mesh.nels, ")")
end

Mesh(Ω, element, nels; kwargs...) = Mesh(CPU(), Ω, element, nels; kwargs...)
Mesh(coords_cpu::AbstractVector{<:SVector}, el2n_cpu::AbstractMatrix{<:Integer}; kwargs...) =
    Mesh(CPU(), coords_cpu, el2n_cpu; kwargs...)
Mesh(coords_cpu::AbstractVector{<:SVector}, el2n_cpu::AbstractMatrix{<:Integer}, element::ReferenceElement; kwargs...) =
    Mesh(CPU(), coords_cpu, el2n_cpu, element; kwargs...)

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

    geometry = precompute_geometry(coords, el2n, element)

    return Mesh{
        nDim,
        order(element),
        D,
        B,
        typeof(coords),
        T2,
        T3,
        T4,
        typeof(element),
        typeof(geometry),
    }(Ω, Γ, coords, DoFs, el2n, Γnodes, length(coords), size(el2n, 2), element, geometry)

end

function _boundary_edge_paths_2d(nlocal::Int)
    if nlocal == 3
        return ((1, 2), (2, 3), (3, 1))
    elseif nlocal == 4
        return ((1, 2), (2, 3), (3, 4), (4, 1))
    elseif nlocal == 6 || nlocal == 7
        return ((1, 4, 2), (2, 5, 3), (3, 6, 1))
    elseif nlocal == 8 || nlocal == 9
        return ((1, 5, 2), (2, 6, 3), (3, 7, 4), (4, 8, 1))
    else
        throw(ArgumentError("cannot infer 2D boundary edge paths for elements with $nlocal local nodes"))
    end
end

_boundary_face_paths_3d(nlocal::Int) = _boundary_face_paths_3d(Val(nlocal))
_boundary_face_paths_3d(::Val{4}) =
    ((1, 2, 3), (1, 2, 4), (2, 3, 4), (1, 3, 4))
_boundary_face_paths_3d(::Val{8}) =
    ((1, 2, 4, 3), (5, 6, 8, 7), (1, 2, 6, 5),
     (3, 4, 8, 7), (1, 3, 7, 5), (2, 4, 8, 6))
_boundary_face_paths_3d(::Val{N}) where N =
    throw(ArgumentError("cannot infer 3D boundary face paths for elements with $N local nodes"))

"""
    _unstructured_boundary_nodes(el2n[, Val(nDim)])

Return sorted unique node indices that lie on the mesh boundary, as a vector
with the same integer eltype as `el2n`.

For 2-D meshes, a boundary edge is an edge whose corner endpoints appear in
exactly one element. For 3-D meshes, a boundary face is a face whose corner
nodes appear in exactly one element. Supported unstructured arities are T3,
Q4, T6/T7, Q8/Q9, Tet4, and Hex8.
"""
_unstructured_boundary_nodes(el2n::AbstractMatrix) = _unstructured_boundary_nodes(el2n, Val(2))

function _unstructured_boundary_nodes(el2n::AbstractMatrix{I}, ::Val{2}) where {I <: Integer}
    edge_paths = _boundary_edge_paths_2d(size(el2n, 1))
    edge_count = Dict{Tuple{I, I}, Int}()
    for iel in axes(el2n, 2), path in edge_paths
        a, b = minmax(el2n[first(path), iel], el2n[last(path), iel])
        edge_count[(a, b)] = get(edge_count, (a, b), 0) + 1
    end
    bnd = I[]
    for iel in axes(el2n, 2), path in edge_paths
        a, b = minmax(el2n[first(path), iel], el2n[last(path), iel])
        get(edge_count, (a, b), 0) == 1 || continue
        for i in path
            push!(bnd, el2n[i, iel])
        end
    end
    return sort!(unique!(bnd))
end

@inline function _face_key(el2n::AbstractMatrix{I}, path::NTuple{N, Int}, iel) where {I <: Integer, N}
    return Tuple(sort(SVector{N, I}(ntuple(i -> el2n[path[i], iel], Val(N)))))
end

function _unstructured_boundary_nodes(el2n::AbstractMatrix{I}, ::Val{3}) where {I <: Integer}
    return _unstructured_boundary_nodes_3d(el2n, Val(size(el2n, 1)))
end

function _unstructured_boundary_nodes_3d(el2n::AbstractMatrix{I}, ::Val{Nlocal}) where {I <: Integer, Nlocal}
    face_paths = _boundary_face_paths_3d(Val(Nlocal))
    Nface = length(first(face_paths))
    face_count = Dict{NTuple{Nface, I}, Int}()
    for iel in axes(el2n, 2), path in face_paths
        key = _face_key(el2n, path, iel)
        face_count[key] = get(face_count, key, 0) + 1
    end
    bnd = I[]
    for iel in axes(el2n, 2), path in face_paths
        key = _face_key(el2n, path, iel)
        get(face_count, key, 0) == 1 || continue
        for i in path
            push!(bnd, el2n[i, iel])
        end
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

function _integration_coordinates(ip::IntegrationPoints{nDim}, q) where {nDim}
    return SVector{nDim}(ntuple(d -> getfield(ip, d)[q], Val(nDim)))
end

_geometry_jacobian(element, point, ::Val{nDim}) where {nDim} =
    eval_shape_function_jacobian(element, point)
_geometry_jacobian(element::ReferenceElement{T}, point, ::Val{1}) where {N, T <: AbstractElement{1, N}} =
    SMatrix{N, 1}(eval_shape_function_jacobian(element, point))

"""
    precompute_geometry(coords, el2n, element; backend=KA.get_backend(coords), workgroup=256)

Precompute physical shape-function gradients and weighted element volumes for
all integration points of `element`.
"""
function precompute_geometry(
    coords,
    el2n,
    element::ReferenceElement{T};
    backend = KA.get_backend(coords),
    workgroup = 256,
) where {nDim, N, FP, T <: AbstractElement{nDim, N, FP}}
    ip = element.integration_points
    NQ = length(ip.ω)
    points = ntuple(q -> _integration_coordinates(ip, q), Val(NQ))
    jacobians = ntuple(q -> _geometry_jacobian(element, points[q], Val(nDim)), Val(NQ))
    Geometry = NTuple{NQ, Tuple{SMatrix{N, nDim, FP, N * nDim}, FP}}
    geometry = KA.allocate(backend, Geometry, size(el2n, 2))
    if backend isa CPU
        for iel in axes(el2n, 2)
            local_nodes = local_nodes_of(el2n, iel, Val(N))
            c = element_coordinate_matrix(coords, local_nodes)
            geometry[iel] = ntuple(Val(NQ)) do q
                J = c' * jacobians[q]
                (jacobians[q] * inv(J), abs(det(J)) * ip.ω[q])
            end
        end
        return geometry
    end
    precompute_geometry_kernel!(backend, workgroup)(
        geometry, coords, el2n, jacobians, ip.ω, Val(N); ndrange = size(el2n, 2),
    )
    KA.synchronize(backend)
    return geometry
end

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
