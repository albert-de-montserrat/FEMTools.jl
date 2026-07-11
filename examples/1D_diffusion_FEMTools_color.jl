using LinearAlgebra
using SparseArrays
using StaticArrays
using Atomix

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DomainSets
using GLMakie
using FEMTools

using Base.Threads: @threads

function gaussian_temperature(x, t; κ=1.0, center=5.0, σ=0.6, amplitude=10.0)
    width² = σ^2 + 4κ * t
    return amplitude * σ / sqrt(width²) * exp(-((x - center)^2) / width²)
end

function preallocate_sparse_matrix(pattern)
    A = Float64.(pattern)
    fill!(nonzeros(A), 0.0)
    return A
end

function assemble_diffusion_Kloc(coords, ip, ∂N∂ξq, ::ReferenceElement{nDim, N}, κ) where {nDim, N}
    Kloc = @SMatrix zeros(N, N)

    for q in eachindex(ip.ω)
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ ⋅ coords
        ∂N∂x = ∂N∂ξ / J
        dΩ = J * ip.ω[q]

        Kloc += κ * (∂N∂x * ∂N∂x') * dΩ
    end

    return Kloc
end

function assemble_diffusion_Mloc(coords, ip, Nq, ∂N∂ξq, ::ReferenceElement{nDim, N}) where {nDim, N}
    Mloc = @SMatrix zeros(N, N)

    for q in eachindex(ip.ω)
        Nv = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ ⋅ coords
        dΩ = J * ip.ω[q]

        Mloc += (Nv * Nv') * dΩ
    end

    return Mloc
end

function assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, ::ReferenceElement{nDim, N}, source) where {nDim, N}
    Floc = @SVector zeros(N)

    for q in eachindex(ip.ω)
        Nv = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ ⋅ coords
        dΩ = J * ip.ω[q]

        Floc += Nv * source * dΩ
    end

    return Floc
end

function color_element_batches(mesh)
    colors = color_mesh(mesh)
    batches = [Int[] for _ in 1:maximum(colors)]
    for iel in 1:mesh.nels
        push!(batches[colors[iel]], iel)
    end
    return batches
end

function assemble_diffusion_matrices!(
    K,
    M,
    F,
    mesh,
    element::ReferenceElement{1, N},
    iel,
    ip,
    Nq,
    ∂N∂ξq,
    κ,
    source,
) where {N}

    local_nodes = SVector{N, Int}(ntuple(i -> mesh.el2n[i, iel], Val(N)))
    coords = SVector{N, Float64}(ntuple(i -> mesh.coords[local_nodes[i]], Val(N)))

    Kloc = assemble_diffusion_Kloc(coords, ip, ∂N∂ξq, element, κ)
    Mloc = assemble_diffusion_Mloc(coords, ip, Nq, ∂N∂ξq, element)
    Floc = assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, element, source)

    @views K[local_nodes, local_nodes] .+= Kloc
    @views M[local_nodes, local_nodes] .+= Mloc
    @views F[local_nodes] .+= Floc

    return nothing
end

function assemble_diffusion_matrices_color_coded!(K, M, F, mesh, element::ReferenceElement{1, N}, κ, source) where {N}
    fill!(K, 0.0)
    fill!(M, 0.0)
    fill!(F, 0.0)
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))
    colors = color_element_batches(mesh)

    for color in colors
        @threads for iel in color
            assemble_diffusion_matrices!(K, M, F, mesh, element, iel, ip, Nq, ∂N∂ξq, κ, source)
        end
    end
    return nothing
end

function solve_1d_diffusion(;
    Lx=10.0,
    nel=100,
    κ=1.0,
    t_total=1.0,
    n_steps=100,
    on_step=nothing,
)
    Ω = 0.0..Lx
    element = ReferenceElement(QuadraticElement{1, 3})
    mesh = Mesh(Ω, element, nel)
    # element = ReferenceElement(LinearElement{1, 2})

    # ΓD = DirichletBoundaryCondition(mesh.Γ, mesh.Γnodes, zeros(length(mesh.Γnodes)))

    source = 0.0

    pattern = generate_sparsity_pattern(mesh)
    K = preallocate_sparse_matrix(pattern)
    M = preallocate_sparse_matrix(pattern)
    F = zeros(mesh.nnodes)

    assemble_diffusion_matrices_color_coded!(K, M, F, mesh, element, κ, source)

    dt = t_total / n_steps
    lhs = M ./ dt .+ K
    lhs0 = copy(lhs)
    # apply_bc!(lhs, F, ΓD)
    
    center = Lx / 2
    T = gaussian_temperature.(mesh.coords, 0.0; κ, center)
    T_exact = similar(T)
    rhs  = similar(T)
    T_exact .= gaussian_temperature.(mesh.coords, 0.0; κ, center)
    on_step !== nothing && on_step(0, 0.0, T, T_exact)

    for i in 1:n_steps
        t = i * dt
        T_exact .= gaussian_temperature.(mesh.coords, t; κ, center)
        mul!(rhs, M, T)
        @. rhs = (rhs / dt) + F
        ΓD = DirichletBoundaryCondition(mesh.Γ, mesh.Γnodes, T_exact[mesh.Γnodes])
        copyto!(lhs, lhs0) 
        apply_bc!(lhs, rhs, ΓD)
        T .= lhs \ rhs
        on_step !== nothing && on_step(i, t, T, T_exact)
    end

    return mesh.coords, T, T_exact
end

function plot_solution(grid, T, T_exact; output="1D_diffusion_FEMTools_color_solution.png")
    error = T - T_exact

    fig = Figure(size=(900, 600))
    ax_solution = Axis(
        fig[1, 1];
        xlabel="x",
        ylabel="temperature",
        title="1D diffusion solution",
    )
    ax_error = Axis(
        fig[2, 1];
        xlabel="x",
        ylabel="numerical - analytical",
        title="pointwise error",
    )

    lines!(ax_solution, grid, T; color=:crimson, linewidth=3, label="FEMTools")
    lines!(ax_solution, grid, T_exact; color=:black, linewidth=2, linestyle=:dash, label="analytical")
    axislegend(ax_solution; position=:rt)

    lines!(ax_error, grid, error; color=:steelblue, linewidth=2)
    hlines!(ax_error, [0.0]; color=:gray40, linestyle=:dot)

    save(output, fig)
    display(fig)

    return fig
end

function record_time_evolution(;
    output="1D_diffusion_FEMTools_color_evolution.gif",
    framerate=24,
    kwargs...,
)
    frames = Tuple{Float64, Vector{Float64}, Vector{Float64}}[]
    on_step = (_, t, T, T_exact) -> begin
        push!(frames, (t, copy(T), copy(T_exact)))
    end
    grid, T, T_exact = solve_1d_diffusion(; on_step, kwargs...)

    T_obs = Observable(frames[1][2])
    T_exact_obs = Observable(frames[1][3])
    title_obs = Observable("1D diffusion solution, t = 0.000")
    err_obs = Observable(frames[1][2] - frames[1][3])

    y_min = minimum(minimum(frame[2]) for frame in frames)
    y_max = maximum(maximum(frame[2]) for frame in frames)
    y_pad = 0.05 * max(y_max - y_min, eps())

    fig = Figure(size=(900, 600))
    ax_solution = Axis(
        fig[1, 1];
        xlabel="x",
        ylabel="temperature",
        title=title_obs,
        limits=(nothing, (y_min - y_pad, y_max + y_pad)),
    )
    ax_error = Axis(
        fig[2, 1];
        xlabel="x",
        ylabel="numerical - analytical",
        title="pointwise error",
    )

    lines!(ax_solution, grid, T_obs; color=:crimson, linewidth=3, label="FEMTools")
    lines!(ax_solution, grid, T_exact_obs; color=:black, linewidth=2, linestyle=:dash, label="analytical")
    axislegend(ax_solution; position=:rt)
    lines!(ax_error, grid, err_obs; color=:steelblue, linewidth=2)
    hlines!(ax_error, [0.0]; color=:gray40, linestyle=:dot)

    record(fig, output, eachindex(frames); framerate) do iframe
        t, T_frame, T_exact_frame = frames[iframe]
        T_obs[] = T_frame
        T_exact_obs[] = T_exact_frame
        err_obs[] = T_frame - T_exact_frame
        title_obs[] = "1D diffusion solution, t = $(round(t; digits=3))"
    end

    return grid, T, T_exact
end

function main(; render=:none, output="1D_diffusion_FEMTools_color_solution.png", gif_output="1D_diffusion_FEMTools_color_evolution.gif")
    if render === :gif
        @time grid, T, T_exact = record_time_evolution(; output=gif_output)
    elseif render === :plot || render === :none
        @time grid, T, T_exact = solve_1d_diffusion()
    else
        error("render must be one of :none, :plot, or :gif")
    end

    error = norm(T - T_exact) / sqrt(length(T))

    println("Solved 1D diffusion with FEMTools color-coded quadrature assembly.")
    println("nodes: ", length(grid))
    println("relative L2-like error: ", error)

    if render === :plot
        plot_solution(grid, T, T_exact; output)
        println("saved solution plot: ", output)
    elseif render === :gif
        println("saved solution animation: ", gif_output)
    end

    nothing
end

main(; render=:gif)
# main(; render=:none)
