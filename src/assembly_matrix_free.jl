function update_T!(T, T0, grid, element, ρ, Cp, κ,  Δt, θ, Δτ)
    nel = size(grid)
    
    # number of elements within block
    block_length = 800
    num_blocks   = cld(nel, block_length)

    for iblock in 1:num_blocks
        block_start = (iblock - 1) * block_length + 1
        block_end   = iblock * block_length 
        for iel in block_start:block_end
            iel > nel && break
            update_local_T!(T, T0, grid, element, ρ, Cp, κ,  Δt, θ, Δτ, iel)
        end
    end
    
    return nothing
end

function update_T_atomics!(T, T0, grid, element, ρ, Cp, κ,  Δt, θ, Δτ)
    nel = size(grid)
    
    for iel in 1:nel
        TL = local_SVector(element)
        # RL = local_SVector(element)

        el_coords  = getelementcoords(grid, iel)
        T_element  = getelementfield(T, grid, iel)
        T0_element = getelementfield(T0, grid, iel)

        # loop over integration points (this can be unrolled if needed)
        for ip in eachindex(element.ip) 
            # extract integration point and shape functions data
            ξ, ω, N, ∇N_local = element[ip]
            
            # compute the Jacobian for 1D linear element
            J         = ∇N_local' * el_coords'
            detJ      = det(J)
            ∇N_global = J \ ∇N_local'
            # precompute some stuff
            detJ_ω    = detJ * ω
            NT_N      = N * N'
            ### RESIDUAL
            # local stiffness matrix (transform gradients to physical coordinates)
            K  = (∇N_global' * κ * ∇N_global) * detJ_ω
            # local mass matrix
            M   = NT_N * ρ * Cp * detJ_ω
            # # assemble residual
            # RL += M * (T_element - T0_element) - K * T_element * Δt

            ### T UPDATE
            Q   = NT_N * (ρ * Cp / Δt + θ / Δτ) * detJ_ω - K
            rhs = M * T0_element + NT_N * (θ / Δτ) * detJ_ω * T_element
            TL += Q \ rhs 
        end

        # assemgle global matrices
        global_dofs = getelement(grid, iel)
        # assemble_sparse_vector_atomics!(RG, RL, global_dofs)
        assemble_sparse_vector_atomics!(T, TL, global_dofs)
    end
    
    return nothing
end

function update_T_colored!(T, T0, grid, element, ρ, Cp, κ,  Δt, θ, Δτ, color_list)
    Threads.@threads for colors in color_list
        for iel in colors
            update_local_T!(T, T0, grid, element, ρ, Cp, κ,  Δt, θ, Δτ, iel)
        end
    end
    return nothing
end

function update_local_T!(T, T0, q, grid, element, ρ, Cp, κ,  Δt, θ, Δτ, iel)
    TL = local_SVector(element)

    el_coords  = getelementcoords(grid, iel)
    T_element  = getelementfield(T, grid, iel)
    T0_element = getelementfield(T0, grid, iel)

    # loop over integration points (this can be unrolled if needed)
    for ip in eachindex(element.ip) 
        # extract integration point and shape functions data
        ξ, ω, N, ∇N_local = element[ip]
        
        # compute the Jacobian for 1D linear element
        J         = ∇N_local' * el_coords'
        detJ      = det(J)
        ∇N_global = J \ ∇N_local'
        # precompute some stuff
        detJ_ω    = detJ * ω
        NT_N      = N * N'
        ### RESIDUAL
        # local stiffness matrix (transform gradients to physical coordinates)
        K  = (∇N_global' * κ * ∇N_global) * detJ_ω
        # local mass matrix
        M   = NT_N * ρ * Cp * detJ_ω / Δt
        ### T UPDATE
        Q   = NT_N * (ρ * Cp / Δt + θ / Δτ) * detJ_ω - K
        rhs = M * T0_element + NT_N * (θ / Δτ) * detJ_ω * T_element
        TL += Q \ rhs 
    end

    # assemgle global matrices
    global_dofs = getelement(grid, iel)
    assemble_sparse_vector!(T, TL, global_dofs)
end

# function update_local_T!(T, T0, grid, element, ρ, Cp, κ,  Δt, θ, Δτ, iel)
#     TL = local_SVector(element)

#     el_coords  = getelementcoords(grid, iel)
#     T_element  = getelementfield(T, grid, iel)
#     T0_element = getelementfield(T0, grid, iel)

#     # loop over integration points (this can be unrolled if needed)
#     for ip in eachindex(element.ip) 
#         # extract integration point and shape functions data
#         ξ, ω, N, ∇N_local = element[ip]
        
#         # compute the Jacobian for 1D linear element
#         J         = ∇N_local' * el_coords'
#         detJ      = det(J)
#         ∇N_global = J \ ∇N_local'
#         # precompute some stuff
#         detJ_ω    = detJ * ω
#         NT_N      = N * N'
#         ### RESIDUAL
#         # local stiffness matrix (transform gradients to physical coordinates)
#         K  = (∇N_global' * κ * ∇N_global) * detJ_ω
#         # local mass matrix
#         M   = NT_N * ρ * Cp * detJ_ω / Δt
#         ### T UPDATE
#         Q   = NT_N * (ρ * Cp / Δt + θ / Δτ) * detJ_ω - K
#         rhs = M * T0_element + NT_N * (θ / Δτ) * detJ_ω * T_element
#         TL += Q \ rhs 
#     end

#     # assemgle global matrices
#     global_dofs = getelement(grid, iel)
#     assemble_sparse_vector!(T, TL, global_dofs)
# end

function update_R!(RG, grid, element, ρ, Cp, κ,  Δt)
    RL = local_SVector(element)

    el_coords  = getelementcoords(grid, iel)
    T_element  = getelementfield(T, grid, iel)
    T0_element = getelementfield(T0, grid, iel)

    # loop over integration points (this can be unrolled if needed)
    for ip in eachindex(element.ip) 
        # extract integration point and shape functions data
        ξ, ω, N, ∇N_local = element[ip]
        
        # compute the Jacobian for 1D linear element
        J         = ∇N_local' * el_coords'
        detJ      = det(J)
        ∇N_global = J \ ∇N_local'
        # precompute some stuff
        detJ_ω    = detJ * ω
        NT_N      = N * N'
        ### RESIDUAL
        # local stiffness matrix (transform gradients to physical coordinates)
        K  = (∇N_global' * κ * ∇N_global) * detJ_ω
        # local mass matrix
        M   = NT_N * ρ * Cp * detJ_ω
        # assemble residual
        RL += M * (T_element - T0_element) - K * T_element * Δt
    end

    # assemgle global matrices
    global_dofs = getelement(grid, iel)
    assemble_sparse_vector!(RG, RL, global_dofs)
end