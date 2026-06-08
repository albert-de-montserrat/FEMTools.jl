"""
    generate_element2node(element::ReferenceElement{1, N}, nels)

Build the element-to-node connectivity matrix for a one-dimensional mesh.

Rows are local node ids and columns are element ids. For example, a quadratic
line mesh with three elements gives columns `[1, 2, 3]`, `[3, 4, 5]`, and
`[5, 6, 7]`.
"""
function generate_element2node(::ReferenceElement{1, N}, nel) where N
    el2n = zeros(Int32, N, nel)
    for iel = axes(el2n, 2)
        for offset in 0:N-1
            el2n[offset + 1, iel] = (N - 1) * (iel - 1) + offset + 1
        end
    end
    return el2n
end

"""
    generate_element2node(element::ReferenceElement{2, 4}, nels)

Build element-to-node connectivity for a structured linear quadrilateral mesh.

Nodes are numbered with the x-index varying fastest. Local nodes follow the
reference element order: bottom-left, bottom-right, top-right, top-left.
"""
function generate_element2node(::ReferenceElement{2, 4}, nels::NTuple{2, <:Integer})
    nx, ny = nels
    stride = nx + 1
    el2n = zeros(Int32, 4, nx * ny)

    iel = 1
    for ey in 1:ny, ex in 1:nx
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
    generate_element2node(element::ReferenceElement{2, 9}, nels)

Build element-to-node connectivity for a structured quadratic quadrilateral
mesh.

Nodes are numbered on the refined `(2nx + 1) × (2ny + 1)` grid with the x-index
varying fastest. Local nodes follow the `QuadraticElement{2, 9}` order.
"""
function generate_element2node(::ReferenceElement{2, 9}, nels::NTuple{2, <:Integer})
    nx, ny = nels
    stride = 2nx + 1
    el2n = zeros(Int32, 9, nx * ny)

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
    generate_element2node(element::ReferenceElement{3, 8}, nels)

Build element-to-node connectivity for a structured linear hexahedral mesh.
"""
function generate_element2node(::ReferenceElement{3, 8}, nels::NTuple{3, <:Integer})
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
    generate_element2node(element::ReferenceElement{3, 27}, nels)

Build element-to-node connectivity for a structured quadratic hexahedral mesh.
"""
function generate_element2node(::ReferenceElement{3, 27}, nels::NTuple{3, <:Integer})
    nx, ny, nz = nels
    stride_y = 2nx + 1
    stride_z = (2nx + 1) * (2ny + 1)
    el2n = zeros(Int32, 27, nx * ny * nz)

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
