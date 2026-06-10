using ForwardDiff
using SparseArrays
using LinearAlgebra
using StaticArrays
using Printf
using Atomix
using TimerOutputs
using DomainSets
using GLMakie
using FEMTools

function integrate_residual(Hloc, local_nodes, source, D, mesh, Nq, ∂N∂ξq, ip, ::Val{N}) where N
    coords = ntuple(i -> mesh.coords[local_nodes[i]], Val(N))
    Re = zero(Hloc)
    for q in eachindex(ip.ω)
        Nv = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ ⋅ coords
        ∂N∂x = ∂N∂ξ / J
        dΩ = J * ip.ω[q]
        # opt
        tmp   = ∂N∂x' * Hloc
        KHloc = (D*dΩ) * (∂N∂x * tmp)
        Re    += SVector{N}(-source[local_nodes[i]] * Nv[i] * dΩ - KHloc[i] for i in  1:N)
    end
    return Re 
end

# per-element residual (and optionally Gershgorin row sum + |diagonal|) with plain scatter;
# safe serially or within a color batch
function integrate_residual_element!(R, ∂R∂H, PC, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, do_∂R∂H, ::Val{N}) where N
    local_nodes = ntuple(i -> mesh.el2n[i, iel], Val(N))
    Hloc = SVector{N}(H[local_nodes[i]] for i in  1:N)
    Re   = integrate_residual(Hloc, local_nodes, source, D, mesh, Nq, ∂N∂ξq, ip, Val(N))
    for (i, inod) in enumerate(local_nodes)
        R[inod] += Re[i]
    end
    if do_∂R∂H
        ∂Re∂He = ForwardDiff.jacobian(
            Hloc -> integrate_residual(Hloc, local_nodes, source, D, mesh, Nq, ∂N∂ξq, ip, Val(N)),
            Hloc
        )
        for (i, inod) in enumerate(local_nodes)
            ∂R∂H[inod] += sum(abs(∂Re∂He[i, j]) for j in 1:N) # sum over ||row||
            PC[inod]   += abs(∂Re∂He[i, i])                   # |diagonal|
        end
    end
end

# same as integrate_residual_element! but with atomic scatter, safe for unordered threading
function integrate_residual_atomic!(R, ∂R∂H, PC, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, do_∂R∂H, ::Val{N}) where N
    local_nodes = ntuple(i -> mesh.el2n[i, iel], Val(N))
    Hloc = SVector{N}(H[local_nodes[i]] for i in  1:N)
    Re   = integrate_residual(Hloc, local_nodes, source, D, mesh, Nq, ∂N∂ξq, ip, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic R[inod] += Re[i]
    end
    if do_∂R∂H
        ∂Re∂He = ForwardDiff.jacobian(
            Hloc -> integrate_residual(Hloc, local_nodes, source, D, mesh, Nq, ∂N∂ξq, ip, Val(N)),
            Hloc
        )
        for (i, inod) in enumerate(local_nodes)
            Atomix.@atomic :monotonic ∂R∂H[inod] += sum(abs(∂Re∂He[i, j]) for j in 1:N) # sum over ||row||
            Atomix.@atomic :monotonic PC[inod]   += abs(∂Re∂He[i, i])                   # |diagonal|
        end
    end
end

function assemble_diffusion_matrices!(R, ∂R∂H, PC, H, mesh, element, D, source, do_∂R∂H)

    fill!(R, 0)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
    end
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))
    N = length(element)

    for iel in 1:mesh.nels
        integrate_residual_element!(R, ∂R∂H, PC, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, do_∂R∂H, Val(N))
    end
end

function assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H, mesh, element::ReferenceElement{T}, D, source, do_∂R∂H) where T<:AbstractElement{nDim, N} where {nDim, N}

    fill!(R, 0)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
    end
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))

    Threads.@threads :static for iel in 1:mesh.nels
        integrate_residual_atomic!(R, ∂R∂H, PC, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, do_∂R∂H, Val(N))
    end
end

function assemble_diffusion_matrices_colored!(R, ∂R∂H, PC, H, mesh, element::ReferenceElement{T}, D, source, do_∂R∂H, colors) where T<:AbstractElement{nDim, N} where {nDim, N}

    fill!(R, 0)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
    end
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))

    for color in colors
        Threads.@threads for iel in color
            integrate_residual_element!(R, ∂R∂H, PC, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, do_∂R∂H, Val(N))
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

function update_rate!(∂u∂τ, R, PC, β)
    @. ∂u∂τ = R / PC + β * ∂u∂τ
end

function update_variable!(H, ∂u∂τ, α)
    @. H .+= α * ∂u∂τ
end

function main(nel)
    Lx      = 1

    Ω = -Lx..Lx
    # element = ReferenceElement(LinearElement{1, 2, Float64})
    element = ReferenceElement(QuadraticElement{1, 3, Float64})
    mesh = FEMTools.Mesh(Ω, element, nel)
    colors = color_element_batches(mesh)

    σ      = 0.1                                # Initial solution
    source = 2.0*exp.(-mesh.coords.^2/(2σ^2) )  # Source  
    HW     = 1.0                                # Dirichlet value west
    HE     = 0.0                                # Dirichlet value east
    epsi   = 1e-9                               # Relative tolerance

    # FEM PT solve
    D               = 1.0
    H_FEM           = zeros(mesh.nnodes)
    R               = zeros(mesh.nnodes)
    R0              = zeros(mesh.nnodes)
    ∂H∂τ            = zeros(mesh.nnodes)
    ∂R∂H            = zeros(mesh.nnodes)
    PC              = zeros(mesh.nnodes)
    # Δτ              = Δx^2/(2*D) * 2.0/Ωe / 1.1
    # ρ               = 6.5/nel
    nr0             = 0.0
    H_FEM[2:end-1] .= exp.(-mesh.coords[2:end-1].^2/(2σ^2) )  
    H_FEM[1]        = HW
    H_FEM[end]      = HE

    # Estimas min max λ
    do_∂R∂H = true 
    assemble_diffusion_matrices!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, do_∂R∂H)
    # PC = ∂R∂H ./ 2

    CFL    = 0.99
    c_fact = 0.9

    λmax = maximum(∂R∂H ./ PC)
    # λmax = maximum(PC ./ ∂R∂H)
    Δτ   = 2 / √(λmax) * CFL
    λmin = 0.0
    c    = 2 * √(λmin) * c_fact
    α    = 2 * Δτ^2 / (2 + c * Δτ)
    β    = (2 - c * Δτ) / (2 + c * Δτ)

    to = TimerOutput()
    ncheck = 100
    for it=1:1_000
        do_∂R∂H = if mod(it, ncheck) == 0
            copyto!(R0, R)
            true
        else
            false
        end
        @timeit to "series" assemble_diffusion_matrices!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, do_∂R∂H)
        @timeit to "atomix" assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, do_∂R∂H)
        @timeit to "colors" assemble_diffusion_matrices_colored!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, do_∂R∂H, colors)

        # Dirichlet BCs: constrain residual and rate *before* the update,
        # otherwise the (nonzero) reaction-force residual at the boundary
        # nodes accumulates into ∂H∂τ and poisons the λmin estimate
        R[1]            = 0e0
        R[end]          = 0e0
        ∂H∂τ[1]         = 0e0
        ∂H∂τ[end]       = 0e0

        update_rate!(∂H∂τ, R, PC, β)
        update_variable!(H_FEM, ∂H∂τ, α)

        H_FEM[1]        = HW
        H_FEM[end]      = HE

        if it % ncheck == 0 || it==1
            nr = norm(R)
            if it==1 
                nr0 = nr 
            end
            isnan(nr/nr0) && error("NaNs")

            if it < 10e3
                λmax = maximum(∂R∂H ./ PC)
                Δτ   = 2 / √(λmax) * CFL
                denom = sum( (Δτ.*∂H∂τ).^2 )
                λmin  = if it == 1 || denom == 0
                    0.0 # R0 not valid yet at it==1; denom==0 at convergence
                else
                    abs.((sum(Δτ.*∂H∂τ.*( (R .- R0) ./ PC )))) / denom
                end
                c    = 2 * √(λmin) * c_fact
                α    = 2 * Δτ^2 / (2 + c * Δτ)
                β    = (2 - c * Δτ) / (2 + c * Δτ)
            end
            @printf("Iter. %05d: %2.2e\n", it, nr/nr0)
            if nr/nr0<epsi break end    
        end
    end

    display(to)

    scatterlines(mesh.coords[25:50:end], H_FEM[25:50:end])

end

nel = 50_000
main(nel)
