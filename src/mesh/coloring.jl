"""
    color_mesh_greedy(mesh)

Greedily color mesh elements so that elements sharing at least one node do not
share the same color.

Returns a vector `colors` where `colors[iel]` is the color assigned to element
`iel`.
"""
function color_mesh_greedy(mesh)
    colors = zeros(Int, mesh.nels)
    el2n = Array(mesh.el2n)
    n2el = generate_node2element(el2n, mesh.nnodes)

    for iel in 1:mesh.nels
        # Collect colors already used by node-adjacent elements.
        used_colors = Set{Int}()
        for node in @view el2n[:, iel]
            for jel in n2el[node]
                color = colors[jel]
                if color != 0
                    push!(used_colors, color)
                end
            end
        end

        # Assign the smallest available positive color for deterministic output.
        color = 1
        while color in used_colors
            color += 1
        end
        colors[iel] = color
    end

    return colors
end

"""
    color_mesh(mesh)

Color mesh elements for conflict-free element-wise operations.

For one-dimensional meshes this uses the optimal two-color alternating pattern.
Other mesh types fall back to [`color_mesh_greedy`](@ref).
"""
function color_mesh(mesh::Mesh{1})
    return [mod1(iel, 2) for iel in 1:mesh.nels]
end

color_mesh(mesh) = color_mesh_greedy(mesh)

"""
    color_structured_triangles(nels)

Return an optimal six-color element schedule for a structured `nels = (nx, ny)`
grid whose quadrilateral cells are each split into two triangles along the
bottom-left to top-right diagonal. The two triangles of cell `(ex, ey)` receive
colors separated by three; adjacent cells shift the base color periodically.

Six colors are minimal because six triangles meet at each interior grid node.
The returned ordering matches mesh builders that store the two triangles of
each cell consecutively with the x-index varying fastest.
"""
function color_structured_triangles(nels::NTuple{2, <:Integer})
    nx, ny = nels
    colors = Vector{Int}(undef, 2 * nx * ny)
    iel = 1
    for ey in 1:ny, ex in 1:nx
        base = mod1(ex + ey - 1, 6)
        colors[iel] = base
        colors[iel + 1] = mod1(base + 3, 6)
        iel += 2
    end
    return colors
end

"""
    reorder_connectivity_by_color(el2n, colors)

Reorder the columns of host element-to-node connectivity `el2n` so elements of
the same color are contiguous. Returns `(reordered_el2n, color_ranges,
permutation)`. `color_ranges[c]` is a contiguous element-index range suitable
for colored assembly, and `permutation` maps each reordered element back to its
original element index.

The sort is stable, preserving the original traversal order within every color.
Recompute element-indexed data such as geometry from `reordered_el2n`, or apply
the returned permutation to data that was computed before reordering.
"""
function reorder_connectivity_by_color(el2n::AbstractMatrix, colors::AbstractVector{<:Integer})
    nels = size(el2n, 2)
    length(colors) == nels || throw(DimensionMismatch(
        "expected one color per element ($nels), got $(length(colors))",
    ))
    isempty(colors) && return copy(el2n), UnitRange{Int}[], Int[]
    minimum(colors) >= 1 || throw(ArgumentError("element colors must be positive integers"))

    ncolors = maximum(colors)
    counts = [count(==(color), colors) for color in 1:ncolors]
    all(>(0), counts) || throw(ArgumentError("element colors must be contiguous from 1 to $ncolors"))

    permutation = sortperm(colors; alg = Base.Sort.MergeSort)
    reordered_el2n = el2n[:, permutation]
    stops = cumsum(counts)
    starts = [1; stops[1:(end - 1)] .+ 1]
    color_ranges = [starts[color]:stops[color] for color in 1:ncolors]
    return reordered_el2n, color_ranges, permutation
end

"""
    generate_element_groups(backend, colors)

Convert a flat `colors` vector (as returned by `color_mesh`) into a vector of
device integer arrays, one per color, each holding the indices of elements
assigned that color.

The returned groups are suitable for passing directly to
`assemble_diffusion_matrices_colored!`: elements within a group share no nodes,
so the assembly kernel can scatter without atomics.
"""
function generate_element_groups(backend, colors)
    TDev    = TA(backend)
    ncolors = maximum(colors)
    return [TDev(findall(==(c), colors)) for c in 1:ncolors]
end

Base.@deprecate build_element_groups generate_element_groups false
