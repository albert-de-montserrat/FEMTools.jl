using LinearAlgebra
using SparseArrays
using StaticArrays

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DomainSets
using DomainSets: ×
using GLMakie
using FEMTools

function gaussian_temperature(x, y, t; κ=1.0, center=(0.5, 0.5), σ=0.08, amplitude=10.0)
    width² = σ^2 + 4κ * t
    dx = x - center[1]
    dy = y - center[2]
    return amplitude * σ^2 / width² * exp(-(dx^2 + dy^2) / width²)
end

function preallocate_sparse_matrix(pattern)
    A = Float64.(pattern)
    fill!(nonzeros(A), 0.0)
    return A
end

function element_coordinate_matrix(mesh, local_nodes::SVector{N, Int}) where {N}
    data = ntuple(Val(2N)) do k
        col = cld(k, N)
        row = k - (col - 1) * N
        mesh.coords[local_nodes[row]][col]
    end
    return SMatrix{N, 2, Float64, 2N}(data)
end

function assemble_diffusion_Kloc(coords, ip, ∂N∂ξq, ::ReferenceElement{2, N}, κ) where {N}
    Kloc = @SMatrix zeros(N, N)

    for q in eachindex(ip.ω)
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ' * coords
        dΩ = abs(det(J)) * ip.ω[q]
        ∂N∂x = ∂N∂ξ * inv(J)

        Kloc += κ * (∂N∂x * ∂N∂x') * dΩ
    end

    return Kloc
end

function assemble_diffusion_Mloc(coords, ip, Nq, ∂N∂ξq, ::ReferenceElement{2, N}) where {N}
    Mloc = @SMatrix zeros(N, N)

    for q in eachindex(ip.ω)
        Nv = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ' * coords
        dΩ = abs(det(J)) * ip.ω[q]

        Mloc += (Nv * Nv') * dΩ
    end

    return Mloc
end

function assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, ::ReferenceElement{2, N}, source) where {N}
    Floc = @SVector zeros(N)

    for q in eachindex(ip.ω)
        Nv = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ' * coords
        dΩ = abs(det(J)) * ip.ω[q]
        xq = sum(Nv[i] * coords[i, 1] for i in 1:N)
        yq = sum(Nv[i] * coords[i, 2] for i in 1:N)
        # source_value = source isa Function ? source(xq, yq) : source
        Floc += Nv * source * dΩ
    end

    return Floc
end

function assemble_diffusion_matrices!(K, M, F, mesh, element::ReferenceElement{2, N}, κ, source) where {N}
    fill!(K, 0.0)
    fill!(M, 0.0)
    fill!(F, 0.0)
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))

    for iel in 1:mesh.nels
        local_nodes = SVector{N, Int}(ntuple(i -> mesh.el2n[i, iel], Val(N)))
        coords = element_coordinate_matrix(mesh, local_nodes)

        Kloc = assemble_diffusion_Kloc(coords, ip, ∂N∂ξq, element, κ)
        Mloc = assemble_diffusion_Mloc(coords, ip, Nq, ∂N∂ξq, element)
        Floc = assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, element, source)

        @views K[local_nodes, local_nodes] .+= Kloc
        @views M[local_nodes, local_nodes] .+= Mloc
        @views F[local_nodes] .+= Floc
    end

    return nothing
end

function solve_2d_diffusion(;
    Lx=1.0,
    Ly=1.0,
    nels=(40, 40),
    κ=1.0e-2,
    t_total=0.1,
    n_steps=40,
    on_step=nothing,
)
    element = ReferenceElement(LinearElement{2, 4})
    Ω = (0.0..Lx) × (0.0..Ly)
    mesh = FEMTools.Mesh(Ω, element, nels)
    source = 0.0

    pattern = generate_sparsity_pattern(mesh)
    K = preallocate_sparse_matrix(pattern)
    M = preallocate_sparse_matrix(pattern)
    F = zeros(mesh.nnodes)

    assemble_diffusion_matrices!(K, M, F, mesh, element, κ, source)

    dt = t_total / n_steps
    lhs = M ./ dt .+ K
    lhs0 = copy(lhs)

    center = (Lx / 2, Ly / 2)
    T = [gaussian_temperature(coord[1], coord[2], 0.0; κ, center) for coord in mesh.coords]
    T_exact = similar(T)
    rhs = similar(T)
    T_exact .= [gaussian_temperature(coord[1], coord[2], 0.0; κ, center) for coord in mesh.coords]
    on_step !== nothing && on_step(0, 0.0, T, T_exact)

    for istep in 1:n_steps
        t = istep * dt
        T_exact .= [gaussian_temperature(coord[1], coord[2], t; κ, center) for coord in mesh.coords]
        mul!(rhs, M, T)
        @. rhs = (rhs / dt) + F
        ΓD = DirichletBoundaryCondition(mesh.Γ, mesh.Γnodes, T_exact[mesh.Γnodes])
        copyto!(lhs, lhs0)
        apply_bc!(lhs, rhs, ΓD)
        T .= lhs \ rhs
        on_step !== nothing && on_step(istep, t, T, T_exact)
    end

    nx, ny = nels
    xs = LinRange(0.0, Lx, nx + 1)
    ys = LinRange(0.0, Ly, ny + 1)
    return xs, ys, T, T_exact, mesh
end

function plot_solution(xs, ys, T, T_exact; output="2D_diffusion_FEMTools_solution.png")
    nx = length(xs)
    ny = length(ys)
    T_grid = reshape(T, nx, ny)
    error_grid = reshape(T - T_exact, nx, ny)

    fig = Figure(size=(1000, 500))
    ax_solution = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="2D diffusion solution", aspect=DataAspect())
    ax_error = Axis(fig[1, 2]; xlabel="x", ylabel="y", title="numerical - analytical", aspect=DataAspect())

    hm_solution = heatmap!(ax_solution, xs, ys, T_grid; colormap=:inferno)
    hm_error = heatmap!(ax_error, xs, ys, error_grid; colormap=:balance)
    Colorbar(fig[2, 1], hm_solution; vertical=false)
    Colorbar(fig[2, 2], hm_error; vertical=false)

    save(output, fig)
    display(fig)

    return fig
end

function main(; render=:plot, output="2D_diffusion_FEMTools_solution.png")
    @time xs, ys, T, T_exact, mesh = solve_2d_diffusion()
    error = norm(T - T_exact) / sqrt(length(T))

    println("Solved 2D diffusion with FEMTools linear rectangular elements.")
    println("nodes: ", mesh.nnodes)
    println("elements: ", mesh.nels)
    println("relative L2-like error: ", error)

    if render === :plot
        plot_solution(xs, ys, T, T_exact; output)
        println("saved solution plot: ", output)
    elseif render !== :none
        error("render must be one of :none or :plot")
    end

    return nothing
end

main(; render=:plot)
