using LinearAlgebra
using SparseArrays
using StaticArrays
using Atomix
using DomainSets
using DomainSets: ×
using GLMakie
using FEMTools

using Base.Threads: @threads

function preallocate_sparse_matrix(pattern)
    A = Float64.(pattern)
    fill!(nonzeros(A), 0.0)
    return A
end

function element_coordinate_matrix(mesh, local_nodes::SVector{N, Int}, ::Val{D}) where {N, D}
    data = ntuple(Val(D * N)) do k
        col = cld(k, N)
        row = k - (col - 1) * N
        mesh.coords[local_nodes[row]][col]
    end
    return SMatrix{N, D, Float64, D * N}(data)
end

function assemble_diffusion_Kloc(coords, ip, ∂N∂ξq, ::ReferenceElement{D, N}, κ) where {D, N}
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

function assemble_diffusion_Mloc(coords, ip, Nq, ∂N∂ξq, ::ReferenceElement{D, N}) where {D, N}
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

function assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, ::ReferenceElement{D, N}, source) where {D, N}
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

function color_element_batches(nels::NTuple{2, <:Integer})
    nx, ny = nels
    batches = [Int[] for _ in 1:4]

    iel = 1
    for ey in 0:(ny - 1), ex in 0:(nx - 1)
        color = 1 + (ex & 1) + 2 * (ey & 1)
        push!(batches[color], iel)
        iel += 1
    end

    return batches
end

function sparse_value_index(A::SparseMatrixCSC, row, col)
    rows = rowvals(A)
    for p in nzrange(A, col)
        rows[p] == row && return p
    end
    error("missing sparse entry at ($row, $col)")
end

function assemble_diffusion_matrices!(
    K,
    M,
    F,
    mesh,
    element::ReferenceElement{2, N},
    iel,
    ip,
    Nq,
    ∂N∂ξq,
    κ,
    source,
) where {N}
    local_nodes = SVector{N, Int}(ntuple(i -> mesh.el2n[i, iel], Val(N)))
    coords = element_coordinate_matrix(mesh, local_nodes, Val(2))

    Kloc = assemble_diffusion_Kloc(coords, ip, ∂N∂ξq, element, κ)
    Mloc = assemble_diffusion_Mloc(coords, ip, Nq, ∂N∂ξq, element)
    Floc = assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, element, source)

    for j in axes(local_nodes, 1)
        node_j = local_nodes[j]
        @inbounds F[node_j] += Floc[j]
        for i in axes(local_nodes, 1)
            node_i = local_nodes[i]
            @inbounds K[node_i, node_j] += Kloc[i, j]
            @inbounds M[node_i, node_j] += Mloc[i, j]
        end
    end

    return nothing
end

function assemble_diffusion_matrices_atomix!(
    K,
    M,
    F,
    mesh,
    element::ReferenceElement{2, N},
    iel,
    ip,
    Nq,
    ∂N∂ξq,
    κ,
    source,
) where {N}
    local_nodes = SVector{N, Int}(ntuple(i -> mesh.el2n[i, iel], Val(N)))
    coords = element_coordinate_matrix(mesh, local_nodes, Val(2))

    Kloc = assemble_diffusion_Kloc(coords, ip, ∂N∂ξq, element, κ)
    Mloc = assemble_diffusion_Mloc(coords, ip, Nq, ∂N∂ξq, element)
    Floc = assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, element, source)

    for j in axes(local_nodes, 1)
        node_j = local_nodes[j]
        Atomix.@atomic :monotonic F[node_j] += Floc[j]
        for i in axes(local_nodes, 1)
            node_i = local_nodes[i]
            Kidx = sparse_value_index(K, node_i, node_j)
            Midx = sparse_value_index(M, node_i, node_j)
            Atomix.@atomic :monotonic nonzeros(K)[Kidx] += Kloc[i, j]
            Atomix.@atomic :monotonic nonzeros(M)[Midx] += Mloc[i, j]
        end
    end

    return nothing
end

function quadrature_cache(element::ReferenceElement{2})
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))
    return ip, Nq, ∂N∂ξq
end

function assemble_diffusion_matrices_color_coded!(K, M, F, mesh, element::ReferenceElement{2, N}, κ, source, colors) where {N}
    ip, Nq, ∂N∂ξq = quadrature_cache(element)

    for color in colors
        @threads for iel in color
            assemble_diffusion_matrices!(K, M, F, mesh, element, iel, ip, Nq, ∂N∂ξq, κ, source)
        end
    end

    return nothing
end

function assemble_diffusion_matrices_nocolor_coded!(K, M, F, mesh, element::ReferenceElement{2, N}, κ, source) where {N}
    ip, Nq, ∂N∂ξq = quadrature_cache(element)

    for iel in 1:mesh.nels
        assemble_diffusion_matrices!(K, M, F, mesh, element, iel, ip, Nq, ∂N∂ξq, κ, source)
    end

    return nothing
end

function assemble_diffusion_matrices_atomics!(K, M, F, mesh, element::ReferenceElement{2, N}, κ, source) where {N}
    ip, Nq, ∂N∂ξq = quadrature_cache(element)

    @threads for iel in 1:mesh.nels
        assemble_diffusion_matrices_atomix!(K, M, F, mesh, element, iel, ip, Nq, ∂N∂ξq, κ, source)
    end

    return nothing
end

function main(ns)
    t_thread = Float64[]
    t_series = Float64[]
    t_atomics = Float64[]
    for n in ns
        κ = 1.0
        source = 0.0

        Ω = (0.0..1.0) × (0.0..1.0)
        element = ReferenceElement(LinearElement{2, 4})
        nels = (n, n)
        mesh = Mesh(Ω, element, nels)

        pattern = generate_sparsity_pattern(mesh)
        K = preallocate_sparse_matrix(pattern)
        M = preallocate_sparse_matrix(pattern)
        F = zeros(mesh.nnodes)
        colors = color_element_batches(nels)

        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        for _ in 1:10
            fill!(nonzeros(K), 0.0); fill!(nonzeros(M), 0.0); fill!(F, 0.0)
            t1 += @elapsed assemble_diffusion_matrices_color_coded!(K, M, F, mesh, element, κ, source, colors)
            fill!(nonzeros(K), 0.0); fill!(nonzeros(M), 0.0); fill!(F, 0.0)
            t2 += @elapsed assemble_diffusion_matrices_nocolor_coded!(K, M, F, mesh, element, κ, source)
            fill!(nonzeros(K), 0.0); fill!(nonzeros(M), 0.0); fill!(F, 0.0)
            t3 += @elapsed assemble_diffusion_matrices_atomics!(K, M, F, mesh, element, κ, source)
        end
        push!(t_thread, t1 / 10)
        push!(t_series, t2 / 10)
        push!(t_atomics, t3 / 10)
    end
    return t_thread, t_series, t_atomics
end

function foo()
    ns = 10:10:300
    t_thread, t_series, t_atomics = main(ns)
    nel = ns .^ 2

    nt = Threads.nthreads()

    f = Figure(size=(1000, 1000), fontsize=24)
    ax = Axis(f[1, 1], xscale=log10, yscale=log10)
    lines!(ax, nel, t_series, label="series")
    lines!(ax, nel, t_thread, label="$nt threads (colored)")
    lines!(ax, nel, t_atomics, label="$nt threads (atomics)")
    axislegend(ax)
    ax.xlabel = "# linear 2D rectangular elements"
    ax.ylabel = "walltime [s]"

    speedup_color = t_series ./ t_thread
    speedup_atomics = t_series ./ t_atomics
    ax_speedup = Axis(f[2, 1], xscale=log10, yscale=log10)
    lines!(ax_speedup, nel, speedup_color, label="color")
    lines!(ax_speedup, nel, speedup_atomics, label="atomics")
    hlines!(ax_speedup, [1.0], color=:gray50, linestyle=:dash, label="1x")
    axislegend(ax_speedup)
    ax_speedup.xlabel = "# linear 2D rectangular elements"
    ax_speedup.ylabel = "speedup"
    rowsize!(f.layout, 1, Relative(2 / 3))
    rowgap!(f.layout, 20)
    save("perf_assembly_2D_$(nt)threads.png", f)
    f
end

foo()
