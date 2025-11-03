function assemble_system!(KG, MG, FG, grid, element, κ, H)
    nel = size(grid)
    
    for iel in 1:nel
        KL = local_SMatrix(element)
        ML = local_SMatrix(element)
        FL = local_SVector(element)

        el_coords = getelementcoords(grid, iel)

        # loop over integration points (this can be unrolled if needed)
        for ip in eachindex(element.ip) 
            # extract integration point and shape functions data
            ξ, ω, N, ∇N_local = element[ip]
            
            # compute the Jacobian for 1D linear element
            J = ∇N_local' * el_coords'
            # in 1D the determinant of the Jacobian is the Jacobian
            detJ = det(J)
            
            # assemble local stiffness matrix (transform gradients to physical coordinates)
            ∇N_global = J \ ∇N_local'
            KL += (∇N_global' * κ * ∇N_global) * detJ * ω

            # assemble local mass matrix
            ML += (N * N') * detJ * ω

            # assemble local force vector
            FL += N * H * detJ * ω
        end

        # assemgle global matrices
        global_dofs = getelement(grid, iel)
        assemble_sparse_matrix!(KG, KL, global_dofs)
        assemble_sparse_matrix!(MG, ML, global_dofs)
        assemble_sparse_vector!(FG, FL, global_dofs)
    end
    
    return nothing
end

function assemble_sparse_matrix!(global_matrix, local_matrix, global_dofs)
    for j in eachindex(global_dofs), i in eachindex(global_dofs)
        @inbounds global_matrix[global_dofs[i], global_dofs[j]] += local_matrix[i, j]
    end
end

function assemble_sparse_vector!(global_vector, local_vector, global_dofs)
    @inbounds @simd for i in eachindex(global_dofs)
        global_vector[global_dofs[i]] += local_vector[i]
    end
end

function assemble_system_atomics!(KG, MG, FG, grid, element, κ, H)
    nel = size(grid)
    
    Threads.@threads for iel in 1:nel
        KL = local_SMatrix(element)
        ML = local_SMatrix(element)
        FL = local_SVector(element)

        el_coords = getelementcoords(grid, iel)

        # loop over integration points (this can be unrolled if needed)
        for ip in eachindex(element.ip) 
            # extract integration point and shape functions data
            ξ, ω, N, ∇N_local = element[ip]
            
            # compute the Jacobian for 1D linear element
            J = ∇N_local' * el_coords'
            # in 1D the determinant of the Jacobian is the Jacobian
            detJ = det(J)
            
            # assemble local stiffness matrix (transform gradients to physical coordinates)
            ∇N_global = J \ ∇N_local'
            KL += (∇N_global' * κ * ∇N_global) * detJ * ω

            # assemble local mass matrix
            ML += (N * N') * detJ * ω

            # assemble local force vector
            FL += N * H * detJ * ω
        end

        # assemgle global matrices
        global_dofs = getelement(grid, iel)
        assemble_sparse_matrix!(KG, KL, global_dofs)
        assemble_sparse_matrix!(MG, ML, global_dofs)
        assemble_sparse_vector!(FG, FL, global_dofs)
    end
    
    return nothing
end

function assemble_sparse_matrix_atomics!(global_matrix, local_matrix, global_dofs)
    @inbounds for j in eachindex(global_dofs), i in eachindex(global_dofs)
        Atomix.@atomic global_matrix[global_dofs[i], global_dofs[j]] += local_matrix[i, j]
    end
end

function assemble_sparse_vector_atomics!(global_vector, local_vector, global_dofs)
    @inbounds for i in eachindex(global_dofs)
        Atomix.@atomic global_vector[global_dofs[i]] += local_vector[i]
    end
end