# Color grid so that all elements containing the i-th node have a different color.
# During threaded assembly each thread acts upon a single color. In this way, we 
# can multithread along each color avoiding synchronisations between threads. It 
# could be extended amongst different cpus, where each cpu takes different colors
function color_mesh(grid::Grid)
    ## ---- Make list of elements within i-th node 
    (; element2node) = grid
    nel = size(grid)
    els_in_node = elements_in_node(element2node)
    
    ## ---- Incidence matrix
    I, J, V = Int[], Int[], Bool[]
    for cells in els_in_node
        for cell1 in cells # All these cells have a neighboring node
            for cell2 in cells
                if cell1 != cell2
                    push!(I, cell1)
                    push!(J, cell2)
                    push!(V, true)
                end
            end
        end
    end
    incidence_matrix = sparse(I, J, V)

    ##  Fill elemental color matrix
    els_colors = fill(0,nel)
    max_color = 0
    @inbounds for iel in 1:nel
        occupied_color = Int64[]

        for r in nzrange(incidence_matrix, iel)
            el_neighbour = incidence_matrix.rowval[r]
            color = els_colors[el_neighbour]
            iszero(color) && continue
            push!(occupied_color, color)
        end

        free_color = 0
        for color in 1:max_color
            if color ∉ occupied_color
                free_color = color
                break
            end
        end

        if iszero(free_color) # no free color found, need to bump max colors
            max_color += 1
            free_color = max_color
        end

        els_colors[iel] = free_color
    end

    ## List of elements per color (this is what is needed for threading the assembler)
    # max_color -= 1
    color_list = [Int32[] for _ in 1:max_color]
    for (iel, color) in enumerate(els_colors)
        @inbounds push!(color_list[color], iel)
    end   

    return els_colors, sort!(color_list)
end

## Make list of elements sharing i-th node 
function elements_in_node(els)
    nodes = 1:maximum(els)
    els_in_node = [Int64[] for _ in axes(nodes,1)]
    sizehint!.(els_in_node, 10)
    for id in axes(els, 2)        
        Base.@nexprs 3 i -> begin 
            @inbounds vᵢ = els[i, id]
            @inbounds push!(els_in_node[vᵢ], id)
        end
    end
    els_in_node
end

function check_colored_mesh(color_list)
    ncolors = length(color_list)
    
    for (nc, node_list) in enumerate(color_list)
        for color in 1:ncolors
            if color!=nc
                for node in node_list
                    if node ∈ color_list[color]
                        println("node ", node, " found in color ", color)
                    end
                end
            end
        end
    end

end