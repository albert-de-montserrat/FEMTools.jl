"""
    color_mesh_greedy(mesh)

Greedily color mesh elements so that elements sharing at least one node do not
share the same color.

Returns a vector `colors` where `colors[iel]` is the color assigned to element
`iel`.
"""
function color_mesh_greedy(mesh)
    colors = zeros(Int, mesh.nels)

    for iel in 1:mesh.nels
        # Collect colors already used by node-adjacent elements.
        used_colors = Set{Int}()
        for node in @view mesh.el2n[:, iel]
            for jel in mesh.n2el[node]
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
