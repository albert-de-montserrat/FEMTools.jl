"""
    BoundaryFacets

Set of boundary edges of a 2-D mesh, stored as element ids `elements` and local
edge indices `local_edges` (positions in the element's edge-path list, see
`_boundary_edge_paths_2d`), both on the mesh backend.

Facets hold topology only, so they stay valid when the mesh coordinates change;
rebuild them after a remesh. Construct with [`boundary_facets`](@ref).
"""
struct BoundaryFacets{V <: AbstractVector{Int32}}
    elements::V
    local_edges::V
end

Base.length(facets::BoundaryFacets) = length(facets.elements)

"""
    boundary_facets(mesh, nodes) -> BoundaryFacets
    boundary_facets(mesh, predicate) -> BoundaryFacets

Select the boundary edges of a 2-D `Mesh` or `MixedMesh` (velocity connectivity)
whose nodes all lie in `nodes`, or whose node coordinates all satisfy
`predicate(x::SVector)`.

Only true boundary edges, whose corner pair belongs to a single element, are
candidates; an edge that touches the set at one corner only is not selected.
Throws an `ArgumentError` when no edge is selected.
"""
boundary_facets(mesh::Union{Mesh{2}, MixedMesh{2}}, nodes::AbstractVector{<:Integer}) =
    _boundary_facets(mesh, in(Set(nodes)))

function boundary_facets(mesh::Union{Mesh{2}, MixedMesh{2}}, predicate)
    coords = Array(mesh.coords)
    return _boundary_facets(mesh, node -> predicate(coords[node]))
end

function _boundary_facets(mesh, selected_node)
    el2n = Array(mesh.el2n)
    edge_paths = _boundary_edge_paths_2d(size(el2n, 1))
    corner_key(path, iel) = minmax(el2n[first(path), iel], el2n[last(path), iel])

    edge_count = Dict{NTuple{2, eltype(el2n)}, Int}()
    for iel in axes(el2n, 2), path in edge_paths
        key = corner_key(path, iel)
        edge_count[key] = get(edge_count, key, 0) + 1
    end

    elements, local_edges = Int32[], Int32[]
    for iel in axes(el2n, 2), (iedge, path) in enumerate(edge_paths)
        edge_count[corner_key(path, iel)] == 1 || continue
        all(i -> selected_node(el2n[i, iel]), path) || continue
        push!(elements, iel)
        push!(local_edges, iedge)
    end
    isempty(elements) && throw(
        ArgumentError("no boundary edges have all their nodes in the given set")
    )

    backend = KA.get_backend(mesh.coords)
    to_backend(v) = copyto!(KA.allocate(backend, Int32, length(v)), v)
    return BoundaryFacets(to_backend(elements), to_backend(local_edges))
end
