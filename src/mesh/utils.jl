"""
    renumber_connectivity(node_tags, element_node_tags, nodes_per_element)

Map arbitrary external node tags to Julia's one-based `Int32` numbering and
return one element per connectivity column.
"""
function renumber_connectivity(node_tags, element_node_tags, nodes_per_element::Integer)
    nodes_per_element > 0 || throw(ArgumentError("nodes_per_element must be positive"))
    length(element_node_tags) % nodes_per_element == 0 ||
        throw(DimensionMismatch("element node count is not divisible by nodes_per_element"))
    tag_to_node = Dict(tag => Int32(i) for (i, tag) in enumerate(node_tags))
    length(tag_to_node) == length(node_tags) || throw(ArgumentError("node tags must be unique"))
    all(haskey(tag_to_node, tag) for tag in element_node_tags) ||
        throw(ArgumentError("element connectivity contains an unknown node tag"))
    return reshape(Int32[tag_to_node[tag] for tag in element_node_tags], nodes_per_element, :)
end

"""
    orient_triangle_elements!(coords, el2n)

Orient T3, T6, or T7 connectivity counter-clockwise in place. Degenerate
corner triangles are rejected.
"""
function orient_triangle_elements!(coords::AbstractVector{<:StaticVector{2}}, el2n::AbstractMatrix{<:Integer})
    nlocal = size(el2n, 1)
    permutation = nlocal == 3 ? (1, 3, 2) :
                  nlocal == 6 ? (1, 3, 2, 6, 5, 4) :
                  nlocal == 7 ? (1, 3, 2, 6, 5, 4, 7) :
                  throw(ArgumentError("triangle connectivity must have 3, 6, or 7 local nodes"))
    for iel in axes(el2n, 2)
        p1, p2, p3 = coords[el2n[1, iel]], coords[el2n[2, iel]], coords[el2n[3, iel]]
        signed_area = det(hcat(p2 - p1, p3 - p1))
        iszero(signed_area) && throw(ArgumentError("degenerate triangle at element $iel"))
        signed_area < 0 && (el2n[:, iel] .= el2n[collect(permutation), iel])
    end
    return el2n
end

"""
    add_t7_bubbles!(coords, el2n_t6) -> (coords, el2n_t7)

Append one centroid node per T6 element to host `coords` and return T7
connectivity. The first six connectivity rows are preserved.
"""
function add_t7_bubbles!(coords, el2n_t6::AbstractMatrix{I}) where {I <: Integer}
    size(el2n_t6, 1) == 6 || throw(DimensionMismatch("T6 connectivity must have 6 rows"))
    n_t6, nels = length(coords), size(el2n_t6, 2)
    el2n_t7 = Matrix{I}(undef, 7, nels)
    el2n_t7[1:6, :] .= el2n_t6
    sizehint!(coords, n_t6 + nels)
    for iel in 1:nels
        push!(coords, sum(coords[el2n_t6[a, iel]] for a in 1:3) / 3)
        el2n_t7[7, iel] = I(n_t6 + iel)
    end
    return coords, el2n_t7
end

"""Straighten T7 edge and bubble nodes from the element corner coordinates."""
function straighten_t7_geometry!(coords, el2n::AbstractMatrix{<:Integer})
    size(el2n, 1) == 7 || throw(DimensionMismatch("T7 connectivity must have 7 rows"))
    for iel in axes(el2n, 2)
        n1, n2, n3 = el2n[1, iel], el2n[2, iel], el2n[3, iel]
        coords[el2n[4, iel]] = (coords[n1] + coords[n2]) / 2
        coords[el2n[5, iel]] = (coords[n2] + coords[n3]) / 2
        coords[el2n[6, iel]] = (coords[n3] + coords[n1]) / 2
        coords[el2n[7, iel]] = (coords[n1] + coords[n2] + coords[n3]) / 3
    end
    return coords
end

function _boundary_atol(atol, values...)
    tol = isnothing(atol) ? sqrt(eps(Float64)) * max(1.0, abs.(values)...) : atol
    tol >= 0 || throw(ArgumentError("atol must be non-negative"))
    return tol
end

"""Return node indices on the boundary of an axis-aligned rectangle."""
function rectangle_boundary_nodes(coords, x0, x1, y0, y1; atol = nothing)
    x0 <= x1 && y0 <= y1 || throw(ArgumentError("rectangle bounds must be ordered"))
    tol = _boundary_atol(atol, x0, x1, y0, y1)
    return Int32[i for i in eachindex(coords) if
        ((abs(coords[i][1] - x0) <= tol || abs(coords[i][1] - x1) <= tol) &&
         y0 - tol <= coords[i][2] <= y1 + tol) ||
        ((abs(coords[i][2] - y0) <= tol || abs(coords[i][2] - y1) <= tol) &&
         x0 - tol <= coords[i][1] <= x1 + tol)]
end

"""Return node indices on a circle with center `(cx, cy)` and radius `r`."""
function circle_boundary_nodes(coords, cx, cy, r; atol = nothing)
    r > 0 || throw(ArgumentError("circle radius must be positive"))
    tol = _boundary_atol(atol, cx, cy, r)
    return Int32[i for i in eachindex(coords) if
        abs(hypot(coords[i][1] - cx, coords[i][2] - cy) - r) <= tol]
end
