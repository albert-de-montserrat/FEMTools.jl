using StaticArrays, ForwardDiff, LinearAlgebra, SparseArrays
using GLMakie
using TimerOutputs

include("elements.jl")
include("assembly.jl")
include("boundary_conditions.jl")
include("grid.jl")

function preallocate_sparse_matrix(grid, element, nel)
    """
    Preallocate sparse matrix structure for finite element assembly.
    Returns the sparsity pattern (I, J indices) for efficient matrix assembly.
    """
    nnodes = length(grid)
    nodes_per_element = num_nodes(element)
    
    # Estimate number of non-zero entries
    # Each node can be connected to roughly nodes_per_element * elements_per_node
    # For 1D: each internal node belongs to 2 elements, boundary nodes to 1 element
    # Conservative estimate: nodes_per_element^2 * nel entries
    nnz_estimate = nodes_per_element^2 * nel
    
    # Preallocate index arrays
    I = Vector{Int}(undef, nnz_estimate)
    J = Vector{Int}(undef, nnz_estimate)
    
    idx = 1
    for iel in 1:nel
        global_dofs = element_nodes(grid, iel, element)
        n_dofs = length(global_dofs)
        
        # Add all combinations of (i,j) for this element
        for i in 1:n_dofs
            for j in 1:n_dofs
                if idx <= nnz_estimate
                    I[idx] = global_dofs[i]
                    J[idx] = global_dofs[j]
                    idx += 1
                end
            end
        end
    end
    
    # Trim arrays to actual size
    resize!(I, idx - 1)
    resize!(J, idx - 1)
    
    # Create sparse matrix with zero values
    return sparse(I, J, zeros(length(I)), nnodes, nnodes)
end

num_nodes(element::LinearElement1D) = 2

num_nodes(element::QuadraticElement1D) = 3

function assemble_sparse_matrix!(global_matrix, local_matrix, global_dofs)
    """
    Efficiently assemble local matrix into sparse global matrix.
    """
    n_dofs = length(global_dofs)
    for i in 1:n_dofs
        for j in 1:n_dofs
            global_matrix[global_dofs[i], global_dofs[j]] += local_matrix[i, j]
        end
    end
end

function assemble_sparse_vector!(global_vector, local_vector, global_dofs)
    """
    Assemble local vector into global vector.
    """
    for i in eachindex(global_dofs)
        global_vector[global_dofs[i]] += local_vector[i]
    end
end

T_analytical(x, Tmax, t, κ, σ) = Tmax/ √(1+4 * t * κ / σ^2) * exp(-x^2 / (σ^2 + 4 * t * κ))

let 
    # grid limits
    x1 = 0e0
    x2 = 10e0
    # number of elements
    nel = 999
    # chose element type
    # element = LinearElement1D()
    element = QuadraticElement1D()
    # create the grid
    grid = Grid1D(x1, x2, nel, element)
    nnodes = length(grid)  # number of nodes

    # physical properties
    κ    = 1
    H    = 0
    ttot = 5
    nt   = 100
    dt   = ttot / nt

    # initial conditions
    Tmax = 10
    σ    = 1
    T    = Tmax * exp.(-grid.x.^2 ./ σ)
    # boundary conditions
    bc_dof = [nnodes]
    bc_val = [0e0]
    bcs    = DirichletBoundaryCondition(bc_dof, bc_val)

    
    # Preallocate sparse matrix structure
    KG = preallocate_sparse_matrix(grid, element, nel)
    MG = preallocate_sparse_matrix(grid, element, nel)
    FG  = zeros(nnodes)

    to = TimerOutput()
    @timeit to "Assembly" for iel in 1:nel
        KL = local_SMatrix(element)
        ML = local_SMatrix(element)
        FL = local_SVector(element)

        el_coords = element_coordinates(grid, iel, element)

        @timeit to "Integration" for ip in eachindex(element.ip)
            # extract integration point and shape functions data
            ξ, ω, N, ∇N_local = element[ip]
            
            # compute the Jacobian for 1D linear element
            J = ∇N_local ⋅ el_coords
            # in 1D the determinant of the Jacobian is the Jacobian
            detJ = J
            
            # assemble local stiffness matrix (transform gradients to physical coordinates)
            ∇N_global = J \ ∇N_local
            KL += (∇N_global * κ * ∇N_global') * detJ * ω

            # assemble local mass matrix
            ML += (N * N') * detJ * ω

            # assemble local force vector
            FL += N * H * detJ * ω
        end

        # assemgle global matrices
        @timeit to "Assembly" begin
            global_dofs = element_nodes(grid, iel, element)
            assemble_sparse_matrix!(KG, KL, global_dofs)
            assemble_sparse_matrix!(MG, ML, global_dofs)
            assemble_sparse_vector!(FG, FL, global_dofs)
        end
    end

    # Create system matrix (sparse)
    KLG = KG + MG / dt
    
    # Apply boundary conditions to sparse matrix
    @views KLG[bcs.dofs,:]         .= 0.0
    @views KLG[bcs.dofs, bcs.dofs] .= I(length(bcs.dofs))

    t = 0

    f,ax, = lines(T, label="T0")

    b = similar(T)

    @timeit to "Time loop" for _ in 1:nt
        t += dt
        @timeit to "build RHS" b .= MG / dt * T .+ FG

        # Apply boundary conditions
        bcs.value          .= T_analytical.(grid.x[bcs.dofs], Tmax, t, κ, σ) 
        @views b[bcs.dofs] .= bcs.value

        # Solve the linear system
        @timeit to "solve" T .= KLG \ b
    end

    @show to
    sol = T_analytical.(grid.x, Tmax, t, κ, σ) 
    lines!(ax, T, label="T FEM")
    lines!(ax, sol, label="T analytical")
    axislegend(ax)
    display(f)
end
