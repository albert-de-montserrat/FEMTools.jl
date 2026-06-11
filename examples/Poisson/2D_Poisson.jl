import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using SparseArrays
using LinearAlgebra
using StaticArrays
using Printf
using Atomix
using TimerOutputs
using DomainSets
using DomainSets: ×
using GLMakie
using FEMTools

function element_coordinate_matrix(mesh, local_nodes::NTuple{N}) where {N}
    data = ntuple(Val(2N)) do k
        col = cld(k, N)
        row = k - (col - 1) * N
        mesh.coords[local_nodes[row]][col]
    end
    return SMatrix{N, 2}(data...)
end
 
function integrate_residual!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, ::Val{N}; doatomic::Bool=false) where N
    local_nodes = ntuple(i -> mesh.el2n[i, iel], Val(N))
    coords = element_coordinate_matrix(mesh, local_nodes)

    for q in eachindex(ip.ω)
        Nv = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ' * coords
        dΩ = abs(det(J)) * ip.ω[q]
        ∂N∂x = ∂N∂ξ * inv(J)

        # # Naive
        # Hloc  = SVector{N}(H[local_nodes[i]] for i in  1:N)
        # Kloc  = D * (∂N∂x * ∂N∂x') * dΩ
        # Floc  = SVector{N}(source[local_nodes[i]] * Nv[i] * dΩ for i in  1:N)
        # R_e   = -Floc - Kloc * Hloc
        # for (i, inod) in enumerate(local_nodes)
        #     R[inod] += R_e[i]
        # end

        # opt
        Hloc   = SVector{N}(H[local_nodes[i]] for i in  1:N)
        tmp    = ∂N∂x' * Hloc
        KHloc  = (D*dΩ) * (∂N∂x * tmp)
        for (i, inod) in enumerate(local_nodes)
            if doatomic
                Atomix.@atomic :monotonic R[inod] += -source[local_nodes[i]] * Nv[i] * dΩ - KHloc[i]
            else
                R[inod] += -source[local_nodes[i]] * Nv[i] * dΩ - KHloc[i]
            end
        end
    end
end

function assemble_diffusion_matrices!(R, H, mesh, element, D, source)

    fill!(R, 0)

    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))
    N = length(element)
    
    for iel in 1:mesh.nels
        integrate_residual!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, Val(N))
    end
end

function assemble_diffusion_matrices_atomix!(R, H, mesh, element::ReferenceElement{T}, D, source) where T<:AbstractElement{nDim, N} where {nDim, N}

    fill!(R, 0)

    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))
    
    Threads.@threads :static for iel in 1:mesh.nels
        integrate_residual!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, Val(N); doatomic=true)
    end
end

function assemble_diffusion_matrices_colored!(R, H, mesh, element::ReferenceElement{T}, D, source, colors) where T<:AbstractElement{nDim, N} where {nDim, N}

    fill!(R, 0)

    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))

    for color in colors
        Threads.@threads for iel in color
            integrate_residual!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, Val(N))
        end
    end
end

function color_element_batches(mesh)
    colors = color_mesh(mesh)
    batches = [Int[] for _ in 1:maximum(colors)]
    for iel in 1:mesh.nels
        push!(batches[colors[iel]], iel)
    end
    return batches
end

function main(nel)
    Lx      = 1
    Ly      = 1

    Ωx = -Lx..Lx
    Ωy = -Ly..Ly
    Ω  = Ωx × Ωy
    element = ReferenceElement(LinearElement{2, 4, Float64})
    # element = ReferenceElement(QuadraticElement{1, 3, Float64})
    grid = FEMTools.Mesh(Ω, element, nel)
    colors = color_element_batches(grid)

    Δx     = 2*Lx/nel[1]                         # Size of elements
    Δy     = 2*Ly/nel[2]                         # Size of elements
    σ      = 0.1                                # Initial solution
    source = [2*exp(-(p[1]^2+p[2]^2)^2/(2σ^2) )  for p in grid.coords]  # Source  
    HW     = 1.0                                # Dirichlet value west
    HE     = 0.0                                # Dirichlet value east
    epsi   = 1e-9                               # Relative tolerance
    Ωe     = Δx * Δy                            # element volume = spacing in 1D

    # FEM PT solve
    D               = 1.0
    H_FEM           = zeros(grid.nnodes)
    R               = zeros(grid.nnodes)
    dHdτ            = zeros(grid.nnodes) 
    Δτ              = Δx^2/(2*D) * 2.0/Ωe / 1.1
    ρ               = 6.5/nel[1]
    nr0             = 0.0
    H_FEM          .= [exp(-(p[1]^2+p[2]^2)^2/(2σ^2) )  for p in grid.coords]
   
    left_boundary(p, D) = begin
        x, y = p
        I, J = factors(D)
        x == leftendpoint(I) && y ∈ J
    end
    right_boundary(p, D) = begin
        x, y = p
        I, J = factors(D)
        x == rightendpoint(I) && y ∈ J
    end
    Γl = [left_boundary(p, Ω) for p in grid.coords]
    Γr = [right_boundary(p, Ω) for p in grid.coords]
    Γl_dofs = grid.DoFs[Γl]
    Γr_dofs = grid.DoFs[Γr]
    Γl_vals = fill(HE, length(Γl_dofs))
    Γr_vals = fill(HW, length(Γr_dofs))
    Γ_dofs = vcat(Γl_dofs, Γr_dofs)
    Γ_vals = vcat(Γl_vals, Γr_vals)

    ΓD_H = DirichletBoundaryCondition(grid.Γ, Γ_dofs, Γ_vals)
    ΓD_R = DirichletBoundaryCondition(grid.Γ, Γ_dofs, zero(Γ_vals))
    apply_bc!(H_FEM, ΓD_H)

    # Benchmark assembly variants
    to = TimerOutput()
    for it in 1:100
        @timeit to "series" assemble_diffusion_matrices!(R, H_FEM, grid, element, D, source)
        @timeit to "atomix" assemble_diffusion_matrices_atomix!(R, H_FEM, grid, element, D, source)
        @timeit to "colors" assemble_diffusion_matrices_colored!(R, H_FEM, grid, element, D, source, colors)
    end
    display(to)

    apply_bc!(H_FEM, ΓD_H)
    xs    = LinRange(-Lx, Lx, nel[1] + 1)
    H_grid = reshape(H_FEM, nel .+ 1...)
    ymid  = nel[2] ÷ 2 + 1                  # middle y-index
    fig   = scatterlines(xs, H_grid[:, ymid]; label="initial")

    # PT solve
    for it in 1:10000
        assemble_diffusion_matrices_colored!(R, H_FEM, grid, element, D, source, colors)
        apply_bc!(R, ΓD_R)
        dHdτ .= (1-ρ).*dHdτ .+ R
        H_FEM .+= Δτ.*dHdτ
        apply_bc!(H_FEM, ΓD_H)
        if it%1000==0 || it==1
            nr = norm(R)
            if it==1
                nr0 = nr
            end
            @printf("Iter. %05d: %2.2e\n", it, nr/nr0)
            if nr/nr0 < epsi break end
        end
    end

    H_grid = reshape(H_FEM, nel .+ 1...)
    scatterlines!(xs, H_grid[:, ymid]; label="converged")
    axislegend(current_axis())
    display(fig)
end

n = 16
nel = (n, n).^2
main(nel)
println("nx, ny = $nel elements")
println("         $(prod(nel)) total elements")