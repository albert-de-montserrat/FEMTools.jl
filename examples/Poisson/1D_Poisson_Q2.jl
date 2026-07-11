import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using SparseArrays
using LinearAlgebra
using StaticArrays
using Printf
using Atomix
using TimerOutputs
using DomainSets
using GLMakie
using FEMTools

function integrate_residual!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, ::Val{N}; doatomic::Bool=false) where N
    local_nodes = ntuple(i -> mesh.el2n[i, iel], Val(N))
    coords = ntuple(i -> mesh.coords[local_nodes[i]], Val(N))

    for q in eachindex(ip.ω)
        Nv   = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J    = ∂N∂ξ ⋅ coords
        dΩ   = J * ip.ω[q]
        ∂N∂x = ∂N∂ξ / J

        Hloc  = SVector{N}(H[local_nodes[i]] for i in 1:N)
        tmp   = ∂N∂x' * Hloc
        KHloc = (D * dΩ) * (∂N∂x * tmp)
        for (i, inod) in enumerate(local_nodes)
            if doatomic
                Atomix.@atomic :monotonic R[inod] += source[local_nodes[i]] * Nv[i] * dΩ - KHloc[i]
            else
                R[inod] += source[local_nodes[i]] * Nv[i] * dΩ - KHloc[i]
            end
        end
    end
end

function assemble_diffusion_matrices!(R, H, mesh, element, D, source)
    fill!(R, 0)
    ip    = element.integration_points
    ξq    = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq    = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))
    N     = length(element)
    for iel in 1:mesh.nels
        integrate_residual!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, Val(N))
    end
end

function assemble_diffusion_matrices_atomix!(R, H, mesh, element::ReferenceElement{T}, D, source) where T<:AbstractElement{nDim, N} where {nDim, N}
    fill!(R, 0)
    ip    = element.integration_points
    ξq    = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq    = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))
    Threads.@threads for iel in 1:mesh.nels
        integrate_residual!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, Val(N); doatomic=true)
    end
end

function assemble_diffusion_matrices_colored!(R, H, mesh, element::ReferenceElement{T}, D, source, colors) where T<:AbstractElement{nDim, N} where {nDim, N}
    fill!(R, 0)
    ip    = element.integration_points
    ξq    = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq    = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))
    for color in colors
        Threads.@threads for iel in color
            integrate_residual!(R, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, Val(N))
        end
    end
end

function color_element_batches(mesh)
    colors  = color_mesh(mesh)
    batches = [Int[] for _ in 1:maximum(colors)]
    for iel in 1:mesh.nels
        push!(batches[colors[iel]], iel)
    end
    return batches
end

function main(nel)
    Lx = 1

    Ω       = -Lx..Lx
    element = ReferenceElement(QuadraticElement{1, 3, Float64})
    grid    = Mesh(Ω, element, nel)
    colors  = color_element_batches(grid)

    Δx   = 2*Lx / nel
    σ    = 0.1
    D    = 1.0
    HW   = 1.0
    HE   = 0.0
    epsi = 1e-9

    # Δτ          = Δx^2/(2*D) * 2.0/Ωe / 1.1

    source = 2.0 * exp.(-grid.coords.^2 / (2σ^2))

    H_FEM = zeros(grid.nnodes)
    R     = zeros(grid.nnodes)
    dHdτ  = zeros(grid.nnodes)
    h     = Δx / (length(element) - 1)      # nodal spacing = Δx/2 for Q2
    Δτ    = h^2 / (2*D) * 2.0/h * 0.65     # = h/1.1; keeps Δτ*λ_max ≈ 3.6 < 4
    ρ     = 6.5 / prod(nel) #/(nel-1)
    nr0   = 0.0
    H_FEM .= exp.(-grid.coords.^2 / (2σ^2))

    # left node (x=-Lx) → HW, right node (x=+Lx) → HE
    Γ_dofs = Int32[grid.Γnodes[1], grid.Γnodes[end]]
    Γ_vals = [HW, HE]
    ΓD_H   = DirichletBoundaryCondition(grid.Γ, Γ_dofs, Γ_vals)
    ΓD_R   = DirichletBoundaryCondition(grid.Γ, Γ_dofs, zero(Γ_vals))
    apply_bc!(H_FEM, ΓD_H)

    # Benchmark assembly variants
    to = TimerOutput()
    for _ in 1:100
        @timeit to "series" assemble_diffusion_matrices!(R, H_FEM, grid, element, D, source)
        @timeit to "atomix" assemble_diffusion_matrices_atomix!(R, H_FEM, grid, element, D, source)
        @timeit to "colors" assemble_diffusion_matrices_colored!(R, H_FEM, grid, element, D, source, colors)
    end
    display(to)

    apply_bc!(H_FEM, ΓD_H)
    xs  = grid.coords
    fig = scatterlines(xs[1:50:end], H_FEM[1:50:end]; label="initial")

    # PT solve
    for it in 1:100000
        assemble_diffusion_matrices_colored!(R, H_FEM, grid, element, D, source, colors)
        apply_bc!(R, ΓD_R)
        dHdτ .= (1 - ρ) .* dHdτ .+ R
        H_FEM .+= Δτ .* dHdτ
        apply_bc!(H_FEM, ΓD_H)
        if it % 1000 == 0 || it == 1
            nr = norm(R)
            if it == 1; nr0 = nr; end
            isnan(nr/nr0)  && error("NaNs")
            @printf("Iter. %05d: %2.2e\n", it, nr/nr0)
            if nr/nr0 < epsi; break; end
        end
    end

    scatterlines!(xs[1:50:end], H_FEM[1:50:end]; label="converged")
    axislegend(current_axis())
    display(fig)
end

nel = 10_000
main(nel)
println("nel = $nel elements  ($(2nel+1) nodes, QuadraticElement{1,3})")
