"""
    generate_element2node(element::ReferenceElement{<:LinearElement{1, N}}, nel)
    generate_element2node(element::ReferenceElement{<:QuadraticElement{1, N}}, nel)

Build the element-to-node connectivity matrix for a one-dimensional mesh.

Rows are local node ids and columns are element ids. For example, a quadratic
line mesh with three elements gives columns `[1, 2, 3]`, `[3, 4, 5]`, and
`[5, 6, 7]`.
"""
function generate_element2node(::ReferenceElement{LinearElement{1, N, T}}, nel) where {N, T}
    el2n = zeros(Int32, N, nel)
    for iel = axes(el2n, 2)
        for offset in 0:N-1
            el2n[offset + 1, iel] = (N - 1) * (iel - 1) + offset + 1
        end
    end
    return el2n
end

function generate_element2node(::ReferenceElement{QuadraticElement{1, N, T}}, nel) where {N, T}
    el2n = zeros(Int32, N, nel)
    for iel = axes(el2n, 2)
        for offset in 0:N-1
            el2n[offset + 1, iel] = (N - 1) * (iel - 1) + offset + 1
        end
    end
    return el2n
end

"""
    generate_element2node(element::ReferenceElement{<:LinearElement{2, 3}}, nels)

Build element-to-node connectivity for a structured linear triangular mesh.

Each quadrilateral cell `(ex, ey)` is split into two counter-clockwise triangles
using an **alternating-diagonal** (checker) pattern that restores the reflection
symmetry of the mesh:

- `(ex + ey)` even — BL→TR diagonal:
  - Triangle A: n1 (BL), n2 (BR), n3 (TR)
  - Triangle B: n1 (BL), n3 (TR), n4 (TL)
- `(ex + ey)` odd — BR→TL diagonal:
  - Triangle C: n1 (BL), n2 (BR), n4 (TL)
  - Triangle D: n2 (BR), n3 (TR), n4 (TL)

Nodes are on the `(nx+1) × (ny+1)` grid with the x-index varying fastest,
consistent with `generate_coordinates(::ReferenceElement{LinearElement{2, 3}})`.
"""
function generate_element2node(::ReferenceElement{LinearElement{2, 3, T}}, nels::NTuple{2, <:Integer}) where T
    nx, ny = nels
    stride = nx + 1
    el2n = zeros(Int32, 3, 2 * nx * ny)

    iel = 1
    for ey in 1:ny, ex in 1:nx
        n1 = (ey - 1) * stride + ex   # bottom-left
        n2 = n1 + 1                    # bottom-right
        n4 = n1 + stride               # top-left
        n3 = n4 + 1                    # top-right
        if iseven(ex + ey)
            el2n[:, iel]     .= Int32[n1, n2, n3]   # BL→TR diagonal: lower-right
            el2n[:, iel + 1] .= Int32[n1, n3, n4]   #                  upper-left
        else
            el2n[:, iel]     .= Int32[n1, n2, n4]   # BR→TL diagonal: lower-left
            el2n[:, iel + 1] .= Int32[n2, n3, n4]   #                  upper-right
        end
        iel += 2
    end

    return el2n
end

"""
    generate_element2node(element::ReferenceElement{<:LinearElement{2, 4}}, nels)

Build element-to-node connectivity for a structured linear quadrilateral mesh.

Nodes are numbered with the x-index varying fastest. Local nodes follow the
reference element order: bottom-left, bottom-right, top-right, top-left.
"""
function generate_element2node(::ReferenceElement{LinearElement{2, 4, T}}, nels::NTuple{2, <:Integer}) where T
    nx, ny = nels
    stride = nx + 1
    el2n = zeros(Int32, 4, nx * ny)

    iel = 1
    for ey in 1:ny, ex in 1:nx
        # `n1` is the lower-left corner of the current cell. The other corners
        # are offsets on the row-major node grid.
        n1 = (ey - 1) * stride + ex
        n2 = n1 + 1
        n4 = n1 + stride
        n3 = n4 + 1
        el2n[:, iel] .= Int32[n1, n2, n3, n4]
        iel += 1
    end

    return el2n
end

"""
    generate_element2node(element::ReferenceElement{<:QuadraticElement{2, 7}}, nels)

Build element-to-node connectivity for a structured T7 mesh.

Identical split pattern to `QuadraticElement{2, 6}` (alternating diagonal).
Nodes 1–6 are the T6 nodes on the `(2nx+1)×(2ny+1)` tensor grid; node 7 is
the element centroid, stored after all T6 nodes with index `n_T6 + iel`.
"""
function generate_element2node(::ReferenceElement{QuadraticElement{2, 7, T}}, nels::NTuple{2, <:Integer}) where T
    nx, ny  = nels
    stride  = 2nx + 1
    n_T6    = stride * (2ny + 1)
    el2n    = zeros(Int32, 7, 2 * nx * ny)

    node(ix, iy) = Int32(iy * stride + ix + 1)

    iel = 1
    for ey in 0:(ny - 1), ex in 0:(nx - 1)
        ix, iy = 2ex, 2ey
        BL  = node(ix,   iy);   BR  = node(ix+2, iy)
        TR  = node(ix+2, iy+2); TL  = node(ix,   iy+2)
        bot = node(ix+1, iy);   rgt = node(ix+2, iy+1)
        top = node(ix+1, iy+2); lft = node(ix,   iy+1)
        ctr = node(ix+1, iy+1)
        cA  = Int32(n_T6 + iel)
        cB  = Int32(n_T6 + iel + 1)
        if iseven(ex + ey)
            el2n[:, iel]     .= (BL, BR, TR, bot, rgt, ctr, cA)
            el2n[:, iel + 1] .= (BL, TR, TL, ctr, top, lft, cB)
        else
            el2n[:, iel]     .= (BL, BR, TL, bot, ctr, lft, cA)
            el2n[:, iel + 1] .= (BR, TR, TL, rgt, top, ctr, cB)
        end
        iel += 2
    end

    return el2n
end

"""
    generate_element2node(element::ReferenceElement{<:QuadraticElement{2, 6}}, nels)

Build element-to-node connectivity for a structured quadratic triangular (T6) mesh.

Each quadrilateral cell `(ex, ey)` is split into two T6 triangles using the same
alternating-diagonal pattern as `LinearElement{2, 3}`. Nodes are numbered on the
refined `(2nx + 1) × (2ny + 1)` tensor-product grid (x-index fastest), identical
to the Q9 grid. Local nodes follow the `QuadraticElement{2, 6}` ordering:
corners 1–3 then edge-midpoints 4 = mid(1,2), 5 = mid(2,3), 6 = mid(1,3).

- `(ex + ey)` even — BL→TR diagonal:
  - Triangle A (lower-right): BL, BR, TR + midpoints
  - Triangle B (upper-left):  BL, TR, TL + midpoints
- `(ex + ey)` odd — BR→TL diagonal:
  - Triangle C (lower-left):  BL, BR, TL + midpoints
  - Triangle D (upper-right): BR, TR, TL + midpoints
"""
function generate_element2node(::ReferenceElement{QuadraticElement{2, 6, T}}, nels::NTuple{2, <:Integer}) where T
    nx, ny  = nels
    stride  = 2nx + 1
    el2n    = zeros(Int32, 6, 2 * nx * ny)

    node(ix, iy) = Int32(iy * stride + ix + 1)

    iel = 1
    for ey in 0:(ny - 1), ex in 0:(nx - 1)
        ix, iy = 2ex, 2ey
        BL  = node(ix,   iy);   BR  = node(ix+2, iy)
        TR  = node(ix+2, iy+2); TL  = node(ix,   iy+2)
        bot = node(ix+1, iy);   rgt = node(ix+2, iy+1)
        top = node(ix+1, iy+2); lft = node(ix,   iy+1)
        ctr = node(ix+1, iy+1)
        if iseven(ex + ey)
            # Triangle A: corners BL,BR,TR; midpoints bot,rgt,ctr=mid(BL,TR)
            el2n[:, iel]     .= (BL, BR, TR, bot, rgt, ctr)
            # Triangle B: corners BL,TR,TL; midpoints ctr=mid(BL,TR),top,lft
            el2n[:, iel + 1] .= (BL, TR, TL, ctr, top, lft)
        else
            # Triangle C: corners BL,BR,TL; midpoints bot,ctr=mid(BR,TL),lft
            el2n[:, iel]     .= (BL, BR, TL, bot, ctr, lft)
            # Triangle D: corners BR,TR,TL; midpoints rgt,top,ctr=mid(BR,TL)
            el2n[:, iel + 1] .= (BR, TR, TL, rgt, top, ctr)
        end
        iel += 2
    end

    return el2n
end

"""
    generate_element2node(element::ReferenceElement{<:QuadraticElement{2, 9}}, nels)

Build element-to-node connectivity for a structured quadratic quadrilateral
mesh.

Nodes are numbered on the refined `(2nx + 1) × (2ny + 1)` grid with the x-index
varying fastest. Local nodes follow the `QuadraticElement{2, 9}` order.
"""
function generate_element2node(::ReferenceElement{QuadraticElement{2, 9, T}}, nels::NTuple{2, <:Integer}) where T
    nx, ny = nels
    stride = 2nx + 1
    el2n = zeros(Int32, 9, nx * ny)

    # Index into the refined tensor grid used by quadratic elements. `ix` and
    # `iy` are zero-based grid coordinates; the returned node id is one-based.
    node(ix, iy) = iy * stride + ix + 1

    iel = 1
    for ey in 0:(ny - 1), ex in 0:(nx - 1)
        ix = 2ex
        iy = 2ey
        el2n[:, iel] .= Int32[
            node(ix, iy),
            node(ix + 2, iy),
            node(ix + 2, iy + 2),
            node(ix, iy + 2),
            node(ix + 1, iy),
            node(ix + 2, iy + 1),
            node(ix + 1, iy + 2),
            node(ix, iy + 1),
            node(ix + 1, iy + 1),
        ]
        iel += 1
    end

    return el2n
end

"""
    generate_element2node(element::ReferenceElement{<:LinearElement{3, 8}}, nels)

Build element-to-node connectivity for a structured linear hexahedral mesh.

Nodes are numbered on the tensor grid with x varying fastest, then y, then z.
Local nodes follow the `LinearElement{3, 8}` hexahedron ordering.
"""
function generate_element2node(::ReferenceElement{LinearElement{3, 8, T}}, nels::NTuple{3, <:Integer}) where T
    nx, ny, nz = nels
    stride_y = nx + 1
    stride_z = (nx + 1) * (ny + 1)
    el2n = zeros(Int32, 8, nx * ny * nz)

    node(ix, iy, iz) = iz * stride_z + iy * stride_y + ix + 1

    iel = 1
    for ez in 0:(nz - 1), ey in 0:(ny - 1), ex in 0:(nx - 1)
        el2n[:, iel] .= Int32[
            node(ex, ey, ez),
            node(ex + 1, ey, ez),
            node(ex + 1, ey + 1, ez),
            node(ex, ey + 1, ez),
            node(ex, ey, ez + 1),
            node(ex + 1, ey, ez + 1),
            node(ex + 1, ey + 1, ez + 1),
            node(ex, ey + 1, ez + 1),
        ]
        iel += 1
    end

    return el2n
end

"""
    generate_element2node(element::ReferenceElement{<:QuadraticElement{3, 27}}, nels)

Build element-to-node connectivity for a structured quadratic hexahedral mesh.

Nodes are numbered on the refined tensor grid with x varying fastest, then y,
then z. Local nodes follow the `QuadraticElement{3, 27}` ordering.
"""
function generate_element2node(::ReferenceElement{QuadraticElement{3, 27, T}}, nels::NTuple{3, <:Integer}) where T
    nx, ny, nz = nels
    stride_y = 2nx + 1
    stride_z = (2nx + 1) * (2ny + 1)
    el2n = zeros(Int32, 27, nx * ny * nz)

    # Zero-based tensor-grid coordinates are converted to one-based Julia node
    # ids. Quadratic elements use every second refined-grid coordinate for
    # corners, with odd coordinates representing mid-edge, face, and cell nodes.
    node(ix, iy, iz) = iz * stride_z + iy * stride_y + ix + 1

    iel = 1
    for ez in 0:(nz - 1), ey in 0:(ny - 1), ex in 0:(nx - 1)
        ix = 2ex
        iy = 2ey
        iz = 2ez
        el2n[:, iel] .= Int32[
            node(ix, iy, iz),
            node(ix + 2, iy, iz),
            node(ix + 2, iy + 2, iz),
            node(ix, iy + 2, iz),
            node(ix, iy, iz + 2),
            node(ix + 2, iy, iz + 2),
            node(ix + 2, iy + 2, iz + 2),
            node(ix, iy + 2, iz + 2),
            node(ix + 1, iy, iz),
            node(ix + 2, iy + 1, iz),
            node(ix + 1, iy + 2, iz),
            node(ix, iy + 1, iz),
            node(ix + 1, iy, iz + 2),
            node(ix + 2, iy + 1, iz + 2),
            node(ix + 1, iy + 2, iz + 2),
            node(ix, iy + 1, iz + 2),
            node(ix, iy, iz + 1),
            node(ix + 2, iy, iz + 1),
            node(ix + 2, iy + 2, iz + 1),
            node(ix, iy + 2, iz + 1),
            node(ix + 1, iy + 1, iz),
            node(ix + 1, iy, iz + 1),
            node(ix + 2, iy + 1, iz + 1),
            node(ix + 1, iy + 2, iz + 1),
            node(ix, iy + 1, iz + 1),
            node(ix + 1, iy + 1, iz + 2),
            node(ix + 1, iy + 1, iz + 1),
        ]
        iel += 1
    end

    return el2n
end

"""
    generate_node2element(el2n, n_nodes=maximum(el2n))

Build node-to-element adjacency from an element-to-node connectivity matrix.

The result is a vector where entry `i` contains the element ids incident to
node `i`.

# Examples
```jldoctest
julia> generate_node2element([1 2; 2 3])
3-element Vector{Vector{Int32}}:
 [1]
 [1, 2]
 [2]
```
"""
function generate_node2element(el2n, n_nodes=maximum(el2n))
    n2el = [Int32[] for _ in 1:n_nodes]
    for iel in axes(el2n, 2)
        for node in @view el2n[:, iel]
            push!(n2el[node], Int32(iel))
        end
    end
    return n2el
end

"""
    generate_boundary_elements(Γnodes, n2el)

Return the unique element ids attached to the boundary nodes `Γnodes`.
"""
function generate_boundary_elements(Γnodes, n2el)
    Γels = Int32[]
    for node in Γnodes
        append!(Γels, n2el[node])
    end
    return unique!(sort!(Γels))
end

"""
    element_coordinate_matrix(coords, local_nodes) -> SMatrix{N, D, T}

Gather physical coordinates of `local_nodes` from `coords` into a dense
`N × D` matrix, where `N` is the number of local nodes and `D` is the spatial
dimension.
"""
@inline function element_coordinate_matrix(coords::AbstractArray{SVector{D, T}}, local_nodes::SVector{N, Int}) where {D, T, N}
    data = ntuple(Val(D * N)) do k
        col = cld(k, N)
        row = k - (col - 1) * N
        coords[local_nodes[row]][col]
    end
    return SMatrix{N, D, T, D * N}(data)
end

@inline function element_coordinate_matrix(coords::AbstractArray{T}, local_nodes::SVector{N, Int}) where {T <: Number, N}
    return SMatrix{N, 1, T, N}(ntuple(i -> coords[local_nodes[i]], Val(N)))
end

"""
    local_nodes_of(el2n, iel, Val(N)) -> SVector{N, Int}

Extract the `N` global node indices for element `iel` from `el2n` into a
statically-sized `SVector`. Used inside assembly kernels to avoid heap
allocation when gathering element-local field values.
"""
@inline local_nodes_of(el2n, iel, ::Val{N}) where N =
    SVector{N, Int}(ntuple(i -> Int(el2n[i, iel]), Val(N)))
