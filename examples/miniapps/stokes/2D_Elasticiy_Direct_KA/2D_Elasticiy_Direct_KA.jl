using ForwardDiff
using LinearAlgebra
using SparseArrays
using StaticArrays
using Printf
using TimerOutputs
using DomainSets
using DomainSets: ×
using KernelAbstractions
import KernelAbstractions as KA
using GLMakie
using FEMTools
# For CUDA, add `using CUDA` and switch the backend below to `CUDABackend()`.
# ---------------------------------------------------------------------------
# 2D linear elasticity (plane strain) of a cantilever clamped on the left
# face, loaded by gravity, solved by assembling the linear stiffness matrix and
# using Julia's sparse direct solver on the free displacement DoFs. KA is still
# used for backend data movement and geometry precomputation.
# ---------------------------------------------------------------------------

const backend   = CPU()   # swap to CUDABackend() after `using CUDA`
const workgroup = 1       # CPU: 1; CUDA: try 128 or 256
const FP        = Float64 # Float32 is a good default for GPU backends

@inline function element_coordinate_matrix(coords, local_nodes::SVector{N, Int}) where {N}
    data = ntuple(Val(2N)) do k
        col = cld(k, N)
        row = k - (col - 1) * N
        coords[local_nodes[row]][col]
    end
    return SMatrix{N, 2, FP, 2N}(data)
end

@inline local_nodes_of(el2n, iel, ::Val{N}) where N =
    SVector{N, Int}(ntuple(i -> Int(el2n[i, iel]), Val(N)))

# ---------------------------------------------------------------------------
# KA kernels
# ---------------------------------------------------------------------------

# (∂N∂x_q, dΩ_q) for every element and quadrature point. Depends only on the
# mesh, so it is computed once and reused for direct assembly.
@kernel function precompute_geometry_kernel!(geo, @Const(coords), @Const(el2n), ∂N∂ξq, ω, ::Val{N}) where N
    iel = @index(Global)
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    c = element_coordinate_matrix(coords, local_nodes)
    geo[iel] = ntuple(Val(length(ω))) do q
        J = ∂N∂ξq[q]' * c
        (∂N∂ξq[q] * inv(J), abs(det(J)) * ω[q])
    end
end

# stacked element residual: uloc = (ux_1..ux_N, uy_1..uy_N) -> (Rx_1..Rx_N, Ry_1..Ry_N)
# R = F - K·u with body force (bx, by); plane-strain Hooke law
@inline function elastic_residual(uloc::SVector{M}, geo_el, λ, μ, bx, by, Nq, ::Val{N}) where {M, N}
    Re = zero(uloc)
    uxloc = SVector{N}(ntuple(i -> uloc[i],     Val(N)))
    uyloc = SVector{N}(ntuple(i -> uloc[N + i], Val(N)))
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv = Nq[q]

        ∇ux = ∂N∂x' * uxloc
        ∇uy = ∂N∂x' * uyloc
        εxx = ∇ux[1]
        εyy = ∇uy[2]
        γxy = ∇ux[2] + ∇uy[1]          # engineering shear strain
        tr  = εxx + εyy
        σxx = λ * tr + 2μ * εxx
        σyy = λ * tr + 2μ * εyy
        σxy = μ * γxy

        Rx = SVector{N}(ntuple(i -> (bx * Nv[i] - (∂N∂x[i, 1] * σxx + ∂N∂x[i, 2] * σxy)) * dΩ, Val(N)))
        Ry = SVector{N}(ntuple(i -> (by * Nv[i] - (∂N∂x[i, 1] * σxy + ∂N∂x[i, 2] * σyy)) * dΩ, Val(N)))
        Re += vcat(Rx, Ry)
    end
    return Re
end

# ---------------------------------------------------------------------------
# launch wrappers
# ---------------------------------------------------------------------------

@inline function shape_function_values(element)
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    return ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
end

# in-place geometry build/rebuild for the current nodal coordinates.
function precompute_geometry!(geo, coords, el2n, nels, element::ReferenceElement{T}) where T<:AbstractElement{2, N} where N
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))
    precompute_geometry_kernel!(backend, workgroup)(geo, coords, el2n, ∂N∂ξq, ip.ω, Val(N); ndrange = nels)
    KA.synchronize(backend)
    return geo
end

function precompute_geometry(coords, el2n, nels, element::ReferenceElement{T}) where T<:AbstractElement{2, N} where N
    NQ = length(element.integration_points.ω)
    geo = KA.allocate(backend, NTuple{NQ, Tuple{SMatrix{N, 2, FP, 2N}, FP}}, nels)
    return precompute_geometry!(geo, coords, el2n, nels, element)
end

# triangulate the elements for rendering with GLMakie.mesh (CCW faces)
function plot_triangles(mesh, ::ReferenceElement{LinearElement{2, 4, T}}) where T
    faces = Matrix{Int32}(undef, 2 * mesh.nels, 3)
    for iel in 1:mesh.nels
        n1, n2, n3, n4 = @view mesh.el2n[:, iel]   # CCW corners
        faces[2iel - 1, :] .= (n1, n2, n3)
        faces[2iel,     :] .= (n1, n3, n4)
    end
    return faces
end

function plot_triangles(mesh, ::ReferenceElement{QuadraticElement{2, 9, T}}) where T
    # corners 1-4 (CCW), midsides 5=S 6=E 7=N 8=W, 9 = centre
    sub = ((1, 5, 9, 8), (5, 2, 6, 9), (8, 9, 7, 4), (9, 6, 3, 7))
    faces = Matrix{Int32}(undef, 8 * mesh.nels, 3)
    k = 0
    for iel in 1:mesh.nels
        ln = @view mesh.el2n[:, iel]
        for (a, b, c, d) in sub
            faces[k += 1, :] .= (ln[a], ln[b], ln[c])
            faces[k += 1, :] .= (ln[a], ln[c], ln[d])
        end
    end
    return faces
end

# move a host array to the compute backend
function to_backend(x::AbstractArray{T}) where T
    y = KA.allocate(backend, T, size(x))
    copyto!(y, x)
    return y
end

function to_backend_sparse(A::SparseMatrixCSC)
    backend isa CPU && return A
    if isdefined(@__MODULE__, :CUDA)
        return CUDA.CUSPARSE.CuSparseMatrixCSC(A)
    end
    error("No sparse matrix transfer is defined for backend $(typeof(backend)). For CUDA, load CUDA.jl and set `backend = CUDABackend()`.")
end

# ---------------------------------------------------------------------------

@inline function stacked_element_dofs(local_nodes::SVector{N, Int}, nnodes) where N
    return vcat(local_nodes, local_nodes .+ nnodes)
end

function color_element_batches(mesh)
    colors = color_mesh(mesh)
    batches = [Int[] for _ in 1:maximum(colors)]
    for iel in 1:mesh.nels
        push!(batches[colors[iel]], iel)
    end
    return [to_backend(batch) for batch in batches]
end

function sparse_value_index(A::SparseMatrixCSC, row, col)
    rows = rowvals(A)
    for p in nzrange(A, col)
        rows[p] == row && return p
    end
    error("missing sparse entry at ($row, $col)")
end

function elasticity_sparsity_pattern(mesh, element::ReferenceElement{T}) where T<:AbstractElement{2, N} where N
    nnodes = mesh.nnodes
    ndofs = 2 * nnodes
    el2n_host = Array(mesh.el2n)
    rows = Int[]
    cols = Int[]
    sizehint!(rows, mesh.nels * (2N)^2)
    sizehint!(cols, mesh.nels * (2N)^2)

    for iel in 1:mesh.nels
        local_nodes = SVector{N, Int}(ntuple(i -> Int(el2n_host[i, iel]), Val(N)))
        local_dofs = stacked_element_dofs(local_nodes, nnodes)
        for col in local_dofs, row in local_dofs
            push!(rows, row)
            push!(cols, col)
        end
    end

    return sparse(rows, cols, trues(length(rows)), ndofs, ndofs)
end

function element_sparse_value_indices(pattern::SparseMatrixCSC, mesh, element::ReferenceElement{T}) where T<:AbstractElement{2, N} where N
    nnodes = mesh.nnodes
    el2n_host = Array(mesh.el2n)
    kidx = Array{Int}(undef, 2N, 2N, mesh.nels)

    for iel in 1:mesh.nels
        local_nodes = SVector{N, Int}(ntuple(i -> Int(el2n_host[i, iel]), Val(N)))
        local_dofs = stacked_element_dofs(local_nodes, nnodes)
        for b in 1:2N, a in 1:2N
            kidx[a, b, iel] = sparse_value_index(pattern, local_dofs[a], local_dofs[b])
        end
    end

    return kidx
end

@kernel function assemble_elasticity_colored_system_kernel!(Kvals, F, @Const(el2n), @Const(geo), @Const(kidx), λ, μ, bx, by, Nq, @Const(elems), nnodes, ::Val{N}) where N
    idx = @index(Global)
    iel = elems[idx]
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    local_dofs = stacked_element_dofs(local_nodes, nnodes)
    geo_el = geo[iel]
    uloc0 = SVector{2N, FP}(ntuple(_ -> zero(FP), Val(2N)))
    Fe = elastic_residual(uloc0, geo_el, λ, μ, bx, by, Nq, Val(N))
    ∂Re∂ue = ForwardDiff.jacobian(
        uloc -> elastic_residual(uloc, geo_el, λ, μ, bx, by, Nq, Val(N)),
        uloc0
    )
    Ke = -∂Re∂ue

    for a in 1:2N
        F[local_dofs[a]] += Fe[a]
        for b in 1:2N
            Kvals[kidx[a, b, iel]] += Ke[a, b]
        end
    end
end

@kernel function scatter_free_solution_kernel!(U, @Const(Ufree), @Const(free))
    i = @index(Global)
    U[free[i]] = Ufree[i]
end

function assemble_elasticity_system(mesh, element::ReferenceElement{T}, geo, λ, μ, bx, by, to) where T<:AbstractElement{2, N} where N
    nnodes = mesh.nnodes
    ndofs = 2 * nnodes
    Nq = shape_function_values(element)

    pattern = @timeit to "sparse pattern" elasticity_sparsity_pattern(mesh, element)
    kidx_host = @timeit to "sparse value map" element_sparse_value_indices(pattern, mesh, element)
    K_cpu = @timeit to "empty sparse matrix" SparseMatrixCSC(ndofs, ndofs, copy(pattern.colptr), copy(rowvals(pattern)), zeros(FP, nnz(pattern)))
    K = @timeit to "copy sparse matrix to backend" to_backend_sparse(K_cpu)
    kidx = @timeit to "copy sparse map to backend" to_backend(kidx_host)
    el2n = @timeit to "copy connectivity to backend" to_backend(Array(mesh.el2n))
    Kvals = nonzeros(K)
    F = to_backend(zeros(FP, ndofs))
    colors = @timeit to "color batches" color_element_batches(mesh)

    @timeit to "colored KA assembly" begin
        for elems in colors
            assemble_elasticity_colored_system_kernel!(backend, workgroup)(Kvals, F, el2n, geo, kidx, λ, μ, bx, by, Nq, elems, nnodes, Val(N); ndrange = length(elems))
        end
        KA.synchronize(backend)
    end

    return K, F
end

function solve_elasticity_direct(mesh, element, geo, λ, μ, bx, by, Γ_nodes, to)
    K, F = @timeit to "assemble system" assemble_elasticity_system(mesh, element, geo, λ, μ, bx, by, to)
    nnodes = mesh.nnodes
    ndofs = 2 * nnodes
    free_host = @timeit to "free dofs" begin
        fixed = sort!(vcat(Γ_nodes, Γ_nodes .+ nnodes))
        isfree = trues(ndofs)
        isfree[fixed] .= false
        findall(isfree)
    end
    nfree = length(free_host)
    free = @timeit to "copy free dofs to backend" to_backend(free_host)

    Kfree = @timeit to "free sparse matrix" K[free_host, free_host]
    Ffree = @timeit to "free rhs" F[free_host]
    Ufree = @timeit to "backend sparse direct solve" try
        Kfree \ Ffree
    catch err
        error("""
        K and F were assembled on $(typeof(backend)), but the active backend does
        not provide a working sparse direct solver for this free-DoF system.

        On CUDA, make sure CUDA.jl is loaded, `backend = CUDABackend()`, and the
        sparse matrix transfer in `to_backend_sparse` creates a CUDA sparse
        matrix type supported by CUDA's sparse solver.

        Original solver error:
        $(sprint(showerror, err))
        """)
    end
    relres = @timeit to "residual check" norm(Kfree * Ufree .- Ffree) / norm(Ffree)
    U = KA.zeros(backend, FP, ndofs)
    @timeit to "scatter solution" begin
        scatter_free_solution_kernel!(backend, workgroup)(U, Ufree, free; ndrange = nfree)
        KA.synchronize(backend)
    end
    U_host = @timeit to "copy solution to host" Array(U)
    return FP.(U_host[1:nnodes]), FP.(U_host[nnodes + 1:end]), relres
end

function main(nels)
    to = TimerOutput()
    Lx, Ly = 4.0, 1.0                           # cantilever: length x thickness

    Ω = (0.0..Lx) × (0.0..Ly)
    # element = ReferenceElement(LinearElement{2, 4, FP})
    element = ReferenceElement(QuadraticElement{2, 9, FP})
    mesh = Mesh(Ω, element, nels)

    # material (plane strain) and gravity load, all in FP
    E      = FP(1.0)                            # Young's modulus
    ν      = FP(0.3)                            # Poisson ratio
    μ      = E / (2 * (1 + ν))                  # shear modulus
    λ      = E * ν / ((1 + ν) * (1 - 2ν))       # Lamé parameter
    bx       = FP(0.0)                          # body force x
    by_total = FP(-1e-3)                        # total gravity load

    # clamped on the left face: ux = uy = 0; all other faces traction-free
    clamped(p, D) = begin
        x, y = p
        I, J = factors(D)
        x == leftendpoint(I) && y ∈ J
    end
    Γc = [clamped(p, Ω) for p in mesh.coords]
    Γ_nodes = collect(mesh.DoFs[Γc])

    # mesh data and fields on the compute backend
    coords = @timeit to "copy mesh to backend" to_backend(mesh.coords)
    el2n   = @timeit to "copy mesh to backend" to_backend(mesh.el2n)

    geo = @timeit to "geometry precompute" precompute_geometry(coords, el2n, mesh.nels, element)
    itip = argmin([abs(p[1] - Lx) + abs(p[2] - Ly/2) for p in mesh.coords])
    Ux_host, Uy_host, relres = @timeit to "direct solve total" solve_elasticity_direct(mesh, element, geo, λ, μ, bx, by_total, Γ_nodes, to)
    Ux = @timeit to "copy solution to backend" to_backend(Ux_host)
    Uy = @timeit to "copy solution to backend" to_backend(Uy_host)

    # small-deflection reference for the total load (Euler-Bernoulli, plane
    # strain, no shear)
    E′ = E / (1 - ν^2)
    I = Ly^3 / 12
    w_eb = by_total * Ly * Lx^4 / (8 * E′ * I)
      # deformed grid rendered as a mesh, coloured by uy
    coords_def = [typeof(p)(p[1] + Ux_host[i], p[2] + Uy_host[i]) for (i, p) in pairs(mesh.coords)]
    coords = to_backend(coords_def)
    vertices = [GLMakie.Point2f(p[1], p[2]) for p in coords_def]
    faces = plot_triangles(mesh, element)

    fig = Figure(size = (1000, 350))
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="cantilever, direct sparse solve", aspect=DataAspect())
    m = GLMakie.mesh!(ax, vertices, faces; color = Uy_host, colormap = :bilbao, shading = NoShading)
    Colorbar(fig[1, 2], m)
    display(fig)

    display(to)

    @printf("direct solve relres = %.3e\n", relres)
    @printf("tip uy = %.4e  (linear Euler-Bernoulli estimate %.4e)\n",
            Uy_host[itip], w_eb)
    @printf("mesh: %d elements, %d nodes, %d DoFs\n",
            mesh.nels, mesh.nnodes, 2 * mesh.nnodes)
 
    # state for post-processing (coords are the displayed deformed grid;
    # mesh.coords and geo still represent the reference configuration)
    return (; mesh, element, coords, el2n, geo, Ux, Uy, λ, μ, E, ν, Lx, Ly, by_total)
    # return nothing
end

nels = (160, 40) .* 5
# nels = (1000, 250) 
state = main(nels);
println("solver done")
