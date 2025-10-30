
"""
Preallocate sparse matrix structure for finite element assembly.
Returns the sparsity pattern (I, J indices) for efficient matrix assembly.
"""
function preallocate_sparse_matrix(grid, dofs_per_node)
    nnodes = length(grid)
    nel    = size(grid)
    
    # Estimate number of non-zero entries
    # this is suboptimal....
    (; element2node) = grid
    nnz_estimate = length(unique(element2node)) * dofs_per_node * size(element2node,1)
    
    # Preallocate index arrays
    I = Vector{Int}(undef, nnz_estimate)
    J = Vector{Int}(undef, nnz_estimate)
    
    idx = 1
    for iel in 1:nel
        global_dofs = getelement(grid, iel)
        n_dofs      = length(global_dofs)
        
        # Add all combinations of (i,j) for this element
        for i in 1:n_dofs, j in 1:n_dofs
            if idx ≤ nnz_estimate
                I[idx] = global_dofs[i]
                J[idx] = global_dofs[j]
                idx += 1
            end
        end
    end
    
    # Trim arrays to actual size
    resize!(I, idx - 1)
    resize!(J, idx - 1)
    
    # Create sparse matrix with zero values
    return sparse(I, J, zeros(length(I)), nnodes, nnodes)
end