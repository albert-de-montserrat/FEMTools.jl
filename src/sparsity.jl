"""
Preallocate sparse matrix structure for finite element assembly.
Returns the sparsity pattern (I, J indices) for efficient matrix assembly.
"""
function preallocate_sparse_matrix(element2node, nnodel)

    dummy  = 1:nnodel
    indx_j = dummy' .* ones(Int64,nnodel,nnodel)
    indx_i = dummy  .* ones(Int64,nnodel,nnodel)
    indx_i = tril(indx_i); indxx_i = vec(indx_i)
    indx_j = tril(indx_j); indxx_j = vec(indx_j)
    filter!(x->x>0, indxx_i)
    filter!(x->x>0, indxx_j)

    CM_i  = copy(@views element2node[indxx_i,:])
    CM_j  = copy(@views element2node[indxx_j,:])
    CMM_i = vec(CM_i)
    CMM_j = vec(CM_j)
    indx  = CMM_i .< CMM_j
    tmp   = @views CMM_j[indx]
    # tmp   = copy(@views CMM_j[indx])
    CMM_j[indx] = CM_i[indx]
    CMM_i[indx] = tmp
    K      = sparse(CMM_i,CMM_j, 1.0)
    K    .+= tril(K,-1)'

    return K

end
