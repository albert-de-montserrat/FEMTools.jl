"""
    generate_sparsity_pattern(mesh)

Build the global finite-element matrix sparsity pattern implied by
`mesh.el2n`.

The pattern connects every pair of local nodes within each element and returns
a sparse boolean matrix of size `(mesh.nnodes, mesh.nnodes)`. The stiffness and
mass matrices share this structural pattern for standard element-local
assembly.
"""
function generate_sparsity_pattern(mesh)
    rows = Int[]
    cols = Int[]

    for iel in axes(mesh.el2n, 2)
        local_nodes = @view mesh.el2n[:, iel]
        # A standard finite-element element matrix is dense over the element's
        # local nodes, so every pair contributes one structural nonzero.
        for j in local_nodes
            for i in local_nodes
                push!(rows, i)
                push!(cols, j)
            end
        end
    end

    pattern = sparse(rows, cols, trues(length(rows)), mesh.nnodes, mesh.nnodes)
    return pattern
end
