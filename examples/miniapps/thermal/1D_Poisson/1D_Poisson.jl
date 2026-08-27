using SparseArrays
using LinearAlgebra
using StaticArrays
using Printf
using Atomix
using TimerOutputs
using DomainSets
using GLMakie
using FEMTools

function integrate_residual!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, ::Val{N}) where N
    local_nodes = ntuple(i -> mesh.el2n[i, iel], Val(N))
    coords = ntuple(i -> mesh.coords[local_nodes[i]], Val(N))

    for q in eachindex(ip.ω)
        Nv = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ ⋅ coords
        ∂N∂x = ∂N∂ξ / J
        dΩ = J * ip.ω[q]

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
            R[inod] += -source[local_nodes[i]] * Nv[i] * dΩ - KHloc[i]
        end
    end
end

function integrate_residual_atomic!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, ::Val{N}) where N
    local_nodes = ntuple(i -> mesh.el2n[i, iel], Val(N))
    coords = ntuple(i -> mesh.coords[local_nodes[i]], Val(N))

    for q in eachindex(ip.ω)
        Nv = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ ⋅ coords
        ∂N∂x = ∂N∂ξ / J
        dΩ = J * ip.ω[q]

        Hloc  = SVector{N}(H[local_nodes[i]] for i in  1:N)
        sloc  = SVector{N}(source[local_nodes[i]] for i in  1:N)
        Kloc  = D * (∂N∂x * ∂N∂x') * dΩ
        Floc  = @. Nv * sloc * dΩ

        R_e = -Floc - Kloc * Hloc
        for (i, inod) in enumerate(local_nodes)
            Atomix.@atomic :monotonic R[inod] += R_e[i]
        end
    end
end

function assemble_diffusion_matrices!(R, H, mesh, element, D, source)

    fill!(R, 0)

    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))
    N = length(element)
    VN = Val(N)
    for iel in 1:mesh.nels
        integrate_residual!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, Val(N))
    end
end

function assemble_diffusion_matrices_atomix!(R, H, mesh, element::ReferenceElement{T}, D, source) where T<:AbstractElement{nDim, N} where {nDim, N}

    fill!(R, 0)

    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))

    Threads.@threads :static for iel in 1:mesh.nels
        integrate_residual_atomic!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, Val(N))
    end
end

function assemble_diffusion_matrices_colored!(R, H, mesh, element::ReferenceElement{T}, D, source, colors) where T<:AbstractElement{nDim, N} where {nDim, N}

    fill!(R, 0)

    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))

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

    Ω = -Lx..Lx
    element = ReferenceElement(LinearElement{1, 2, Float64})
    # element = ReferenceElement(QuadraticElement{1, 3, Float64})
    mesh = Mesh(Ω, element, nel)
    colors = color_element_batches(mesh)

    Δx     = Lx/nel                             # Size of elements
    σ      = 0.1                                # Initial solution
    source = 2.0*exp.(-mesh.coords.^2/(2σ^2) )  # Source  
    HW     = 1.0                                # Dirichlet value west
    HE     = 0.0                                # Dirichlet value east
    epsi   = 1e-9                               # Relative tolerance
    Ωe     = Δx                                 # element volume = spacing in 1D

    # FEM PT solve
    D               = 1.0
    H_FEM           = zeros(mesh.nnodes)
    R               = zeros(mesh.nnodes)
    dHdτ            = zeros(mesh.nnodes) 
    Δτ              = Δx^2/(2*D) * 2.0/Ωe / 1.1
    ρ               = 6.5/nel
    nr0             = 0.0
    H_FEM[2:end-1] .= exp.(-mesh.coords[2:end-1].^2/(2σ^2) )  
    H_FEM[1]        = HW
    H_FEM[end]      = HE

    to = TimerOutput()
    for it=1:100#00
        @timeit to "series" assemble_diffusion_matrices!(R, H_FEM, mesh, element, D, source)
        @timeit to "atomix" assemble_diffusion_matrices_atomix!(R, H_FEM, mesh, element, D, source)
        @timeit to "colors" assemble_diffusion_matrices_colored!(R, H_FEM, mesh, element, D, source, colors)
        
        # Dirichlet BCs
        R[1]             = 0e0
        R[end]           = 0e0
        dHdτ            .= (1-ρ).*dHdτ .+ R
        H_FEM[2:end-1] .+= Δτ.*dHdτ[2:end-1]
        if it%1000==0 || it==1
            nr = norm(R)
            if it==1 
                nr0 = nr 
            end
            @printf("Iter. %05d: %2.2e\n", it, nr/nr0)
            if nr/nr0<epsi break end    
        end
    end

    display(to)

    scatterlines(mesh.coords[25:50:end], H_FEM[25:50:end])

end

nel = 150_000
main(nel)
