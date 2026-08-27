using LinearAlgebra
using SparseArrays
using StaticArrays

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using DomainSets
using DomainSets: ×
using GLMakie
using FEMTools

function gaussian_temperature(x, y, z, t; κ=1.0, center=(0.5, 0.5, 0.5), σ=0.08, amplitude=10.0)
    width² = σ^2 + 4κ * t
    dx = x - center[1]
    dy = y - center[2]
    dz = z - center[3]
    return amplitude * σ^3 / width²^(3 / 2) * exp(-(dx^2 + dy^2 + dz^2) / width²)
end

function preallocate_sparse_matrix(pattern)
    A = Float64.(pattern)
    fill!(nonzeros(A), 0.0)
    return A
end

function element_coordinate_matrix(mesh, local_nodes::SVector{N, Int}) where {N}
    data = ntuple(Val(3N)) do k
        col = cld(k, N)
        row = k - (col - 1) * N
        mesh.coords[local_nodes[row]][col]
    end
    return SMatrix{N, 3, Float64, 3N}(data)
end

function assemble_diffusion_Kloc(coords, ip, ∂N∂ξq, ::ReferenceElement{3, N}, κ) where {N}
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

function assemble_diffusion_Mloc(coords, ip, Nq, ∂N∂ξq, ::ReferenceElement{3, N}) where {N}
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

function assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, ::ReferenceElement{3, N}, source) where {N}
    Floc = @SVector zeros(N)

    for q in eachindex(ip.ω)
        Nv = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ' * coords
        dΩ = abs(det(J)) * ip.ω[q]

        Floc += Nv * source * dΩ
    end

    return Floc
end

function assemble_diffusion_matrices!(K, M, F, mesh, element::ReferenceElement{3, N}, κ, source) where {N}
    fill!(K, 0.0)
    fill!(M, 0.0)
    fill!(F, 0.0)
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q], ip.ζ[q]), length(ip.ω))
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

function solve_3d_diffusion(;
    Lx=1.0,
    Ly=1.0,
    Lz=1.0,
    nels=(1,1,1) .* 24,
    κ=1.0e-2,
    t_total=0.05,
    n_steps=20,
    on_step=nothing,
)
    element = ReferenceElement(LinearElement{3, 8})
    Ω = (0.0..Lx) × (0.0..Ly) × (0.0..Lz)
    mesh = Mesh(Ω, element, nels)
    source = 0.0

    pattern = generate_sparsity_pattern(mesh)
    K = preallocate_sparse_matrix(pattern)
    M = preallocate_sparse_matrix(pattern)
    F = zeros(mesh.nnodes)

    assemble_diffusion_matrices!(K, M, F, mesh, element, κ, source)

    dt = t_total / n_steps
    lhs = M ./ dt .+ K
    lhs0 = copy(lhs)

    center = (Lx / 2, Ly / 2, Lz / 2)
    T = [gaussian_temperature(coord[1], coord[2], coord[3], 0.0; κ, center) for coord in mesh.coords]
    T_exact = similar(T)
    rhs = similar(T)
    T_exact .= [gaussian_temperature(coord[1], coord[2], coord[3], 0.0; κ, center) for coord in mesh.coords]
    on_step !== nothing && on_step(0, 0.0, T, T_exact)

    for istep in 1:n_steps
        t = istep * dt
        T_exact .= [gaussian_temperature(coord[1], coord[2], coord[3], t; κ, center) for coord in mesh.coords]
        mul!(rhs, M, T)
        @. rhs = (rhs / dt) + F
        ΓD = DirichletBoundaryCondition(mesh.Γ, mesh.Γnodes, T_exact[mesh.Γnodes])
        copyto!(lhs, lhs0)
        apply_bc!(lhs, rhs, ΓD)
        T .= lhs \ rhs
        on_step !== nothing && on_step(istep, t, T, T_exact)
    end

    nx, ny, nz = nels
    xs = LinRange(0.0, Lx, nx + 1)
    ys = LinRange(0.0, Ly, ny + 1)
    zs = LinRange(0.0, Lz, nz + 1)
    return xs, ys, zs, T, T_exact, mesh
end

function plot_solution(xs, ys, zs, T, T_exact; output="3D_diffusion_FEMTools_slices.png")
    nx = length(xs)
    ny = length(ys)
    nz = length(zs)
    T_grid = reshape(T, nx, ny, nz)
    error_grid = reshape(T - T_exact, nx, ny, nz)
    error_max = maximum(abs, error_grid)
    error_range = error_max == 0 ? (-1.0, 1.0) : (-error_max, error_max)

    fig = Figure(size=(1220, 900))
    Label(fig[1, 1], "temperature"; fontsize=18)
    Label(fig[1, 2], "numerical - analytical"; fontsize=18)
    ax_solution = LScene(fig[2, 1], show_axis=true)
    ax_error = LScene(fig[2, 2], show_axis=true)
    colsize!(fig.layout, 1, Fixed(560))
    colsize!(fig.layout, 2, Fixed(560))
    rowsize!(fig.layout, 2, Fixed(560))
    rowgap!(fig.layout, 16)
    colgap!(fig.layout, 24)

    plt_solution = volumeslices!(ax_solution, xs, ys, zs, T_grid; colormap=:inferno)
    plt_error = volumeslices!(ax_error, xs, ys, zs, error_grid; colormap=:balance, colorrange=error_range)
    scale!(ax_solution.scene, 1, 1, 1)
    scale!(ax_error.scene, 1, 1, 1)
    cam3d!(ax_solution.scene, projectiontype=Makie.Orthographic)
    cam3d!(ax_error.scene, projectiontype=Makie.Orthographic)
    center!(ax_solution.scene)
    center!(ax_error.scene)
    zoom!(ax_solution.scene, 0.82)
    zoom!(ax_error.scene, 0.82)

    sgrid = SliderGrid(
        fig[3, 1:2],
        (label="yz plane - x axis", range=1:nx),
        (label="xz plane - y axis", range=1:ny),
        (label="xy plane - z axis", range=1:nz),
    )

    sl_yz, sl_xz, sl_xy = sgrid.sliders
    on(sl_yz.value) do value
        plt_solution[:update_yz][](value)
        plt_error[:update_yz][](value)
    end
    on(sl_xz.value) do value
        plt_solution[:update_xz][](value)
        plt_error[:update_xz][](value)
    end
    on(sl_xy.value) do value
        plt_solution[:update_xy][](value)
        plt_error[:update_xy][](value)
    end

    set_close_to!(sl_yz, cld(nx, 2))
    set_close_to!(sl_xz, cld(ny, 2))
    set_close_to!(sl_xy, cld(nz, 2))

    Colorbar(fig[4, 1], plt_solution; vertical=false, label="temperature")
    Colorbar(fig[4, 2], plt_error; vertical=false, label="numerical - analytical")

    save(output, fig)
    display(fig)

    return fig
end

function main(; render=:plot, output="3D_diffusion_FEMTools_slices.png")
    @time xs, ys, zs, T, T_exact, mesh = solve_3d_diffusion()
    error = norm(T - T_exact) / sqrt(length(T))

    println("Solved 3D diffusion with FEMTools linear hexahedral elements.")
    println("nodes: ", mesh.nnodes)
    println("elements: ", mesh.nels)
    println("relative L2-like error: ", error)

    if render === :plot
        plot_solution(xs, ys, zs, T, T_exact; output)
        println("saved solution slices: ", output)
    elseif render !== :none
        error("render must be one of :none or :plot")
    end

    return nothing
end

main(; render=:plot)
