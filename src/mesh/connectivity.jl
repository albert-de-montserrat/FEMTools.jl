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
