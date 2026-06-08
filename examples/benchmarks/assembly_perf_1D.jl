using LinearAlgebra
using SparseArrays
using StaticArrays
using Atomix
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

function assemble_diffusion_matrices_color_coded!(K, M, F, mesh, element::ReferenceElement{1, N}, κ, source, colors) where {N}
    
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))

    for color in colors
        @threads for iel in color
            assemble_diffusion_matrices!(K, M, F, mesh, element, iel, ip, Nq, ∂N∂ξq, κ, source)
        end
    end

    return nothing
end

function assemble_diffusion_matrices_nocolor_coded!(K, M, F, mesh, element::ReferenceElement{1, N}, κ, source) where {N}
    
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))

    for iel in 1:mesh.nels
        assemble_diffusion_matrices!(K, M, F, mesh, element, iel, ip, Nq, ∂N∂ξq, κ, source)
    end

    return nothing
end

function assemble_diffusion_matrices_atomics!(K, M, F, mesh, element::ReferenceElement{1, N}, κ, source) where {N}
    
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q],), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_gradient(element, ξq[q]), length(ip.ω))

    @threads for iel in 1:mesh.nels
        assemble_diffusion_matrices_atomix!(K, M, F, mesh, element, iel, ip, Nq, ∂N∂ξq, κ, source)
    end

    return nothing
end

function main(nel)
    t_thread = Float64[]
    t_series = Float64[]
    t_atomics = Float64[]
    for nel in nel
        Lx = 10.0
        κ = 1.0
        source = 0.0
        t_total = 1.0
        n_steps = 100
        on_step = nothing

        Ω = 0.0..Lx
        element = ReferenceElement(QuadraticElement{1, 3})
        mesh = FEMTools.Mesh(Ω, element, nel)

        pattern = generate_sparsity_pattern(mesh)
        K = preallocate_sparse_matrix(pattern)
        M = preallocate_sparse_matrix(pattern)
        F = zeros(mesh.nnodes)
        colors = color_element_batches(mesh)

        t1 = 0e0 
        t2 = 0e0
        t3 = 0e0
        for _ in 1:10
            t1 += @elapsed assemble_diffusion_matrices_color_coded!(K, M, F, mesh, element, κ, source, colors)
            t2 += @elapsed assemble_diffusion_matrices_nocolor_coded!(K, M, F, mesh, element, κ, source)
            t3 += @elapsed assemble_diffusion_matrices_atomics!(K, M, F, mesh, element, κ, source)
            
        end
        push!(t_thread, t1 / 10)
        push!(t_series, t2 / 10)
        push!(t_atomics, t3 / 10)
    end
    return t_thread, t_series, t_atomics
end

function foo()
    nel = 1000:1000:100_000
    t_thread, t_series, t_atomics =  main(nel)

    nt = Threads.nthreads()

    f = Figure(size=(1000, 1000), fontsize=24)
    ax = Axis(f[1, 1], xscale=log10)
    lines!(ax, nel, t_series, label = "series")
    lines!(ax, nel, t_thread, label = "$nt threads (colored)")
    lines!(ax, nel, t_atomics, label = "$nt threads (atomics)")
    axislegend(ax)
    ax.xlabel = "# quadratic 1D elements"
    ax.ylabel = "walltime [s]"

    speedup_color = t_series ./ t_thread
    speedup_atomics = t_series ./ t_atomics
    ax_speedup = Axis(f[2, 1], xscale=log10)
    lines!(ax_speedup, nel, speedup_color, label = "color")
    lines!(ax_speedup, nel, speedup_atomics, label = "atomics")
    hlines!(ax_speedup, [1.0], color = :gray50, linestyle = :dash, label = "1x")
    axislegend(ax_speedup)
    ax_speedup.xlabel = "# quadratic 1D elements"
    ax_speedup.ylabel = "speedup"
    rowsize!(f.layout, 1, Relative(2 / 3))
    rowgap!(f.layout, 20)
    save("perf_assembly_$(nt)threads.png", f)
    f

end

foo()
