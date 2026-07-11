using ForwardDiff
using LinearAlgebra
using StaticArrays
using Printf
using Atomix
using TimerOutputs
using DomainSets
using DomainSets: ×
using KernelAbstractions
import KernelAbstractions as KA
using GLMakie
using FEMTools

using CUDA

# const backend   = CPU()   # swap for e.g. MetalBackend() / CUDABackend(); CPU() threads with julia -t
const backend   = CUDABackend()   # swap for e.g. MetalBackend() / CUDABackend(); CPU() threads with julia -t
const workgroup = 32

# AD chunk size for the elemental Jacobian: the residual is re-evaluated
# ⌈N/chunk⌉ times with `chunk`-partial Duals instead of once with N partials,
# trading FLOPs for an ~N/chunk smaller per-thread (local-memory) footprint
const jacobian_chunk = 9

@inline function element_coordinate_matrix(coords, local_nodes::SVector{N, Int}) where {N}
    data = ntuple(Val(3N)) do k
        @inline 
        col = cld(k, N)
        row = k - (col - 1) * N
        coords[local_nodes[row]][col]
    end
    return SMatrix{N, 3, Float64, 3N}(data)
end

@inline local_nodes_of(el2n, iel, ::Val{N}) where N =
    SVector{N, Int}(ntuple(i -> Int(el2n[i, iel]), Val(N)))

# ---------------------------------------------------------------------------
# KA kernels
# ---------------------------------------------------------------------------

# (∂N∂x_q, dΩ_q) for every element and quadrature point, stored as an
# NQ × nels matrix so threads stream one quadrature point at a time from
# global memory instead of holding the whole element's geometry (O(N*NQ))
# in thread-private local memory. Depends only on the mesh, so it is
# computed once and reused across all PT iterations.
@kernel function precompute_geometry_kernel!(geo, @Const(coords), @Const(el2n), ∂N∂ξq, ω, ::Val{N}) where N
    iel = @index(Global)
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    c = element_coordinate_matrix(coords, local_nodes)
    for q in eachindex(ω)
        J = ∂N∂ξq[q]' * c
        geo[q, iel] = (∂N∂ξq[q] * inv(J), abs(det(J)) * ω[q])
    end
end


@inline function integrate_residual(Hloc, geo, iel, sloc, D, Nq, ::Val{N}) where N
    Re = zero(Hloc)
    for q in eachindex(Nq)
        ∂N∂x, dΩ = geo[q, iel]
        Nv = Nq[q]
        tmp   = ∂N∂x' * Hloc
        KHloc = (D * dΩ) * (∂N∂x * tmp)
        Re   += SVector{N}(ntuple(i -> -sloc[i] * Nv[i] * dΩ - KHloc[i], Val(N)))
    end
    return Re
end

# shared per-element work: gather, integrate, return (nodes, Re)
@inline function element_residual(H, source, el2n, geo, D, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    Hloc = SVector{N}(ntuple(i -> H[local_nodes[i]], Val(N)))
    sloc = SVector{N}(ntuple(i -> source[local_nodes[i]], Val(N)))
    Re   = integrate_residual(Hloc, geo, iel, sloc, D, Nq, Val(N))
    return local_nodes, Re
end

@inline function integrate_residual!(R, Hloc, geo, iel, source, D, Nq, local_nodes, ::Val{N}) where N

    for q in eachindex(Nq)
        ∂N∂x, dΩ = geo[q, iel]
        Nv = Nq[q]
        tmp   = ∂N∂x' * Hloc
        KHloc = (D * dΩ) * (∂N∂x * tmp)
        for (i, inod) in enumerate(local_nodes)
            R[inod] += -source[local_nodes[i]] * Nv[i] * dΩ - KHloc[i]
        end
    end
end

# shared per-element work: gather, integrate, return (nodes, Re)
@inline function element_residual!(R, H, source, el2n, geo, D, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    Hloc = SVector{N}(ntuple(i -> H[local_nodes[i]], Val(N)))
    integrate_residual!(R, Hloc, geo, iel, source, D, Nq, local_nodes, Val(N))
    return nothing
end


# chunked forward-mode Jacobian: seeds `C` columns at a time and re-evaluates
# `f` per chunk, accumulating Gershgorin row sums and |diagonal| on the fly so
# the full N×N Jacobian is never materialized. Per-pass live state is O(N*C)
# Duals instead of O(N*N), which keeps GPU local memory in check for large N.
struct JacobianChunkTag end

@inline function jacobian_rowsums_diags(f::F, x::SVector{N, T}, ::Val{C}) where {F, N, T, C}
    rowsums = zero(SVector{N, T})
    diags   = zero(SVector{N, T})
    for k in 0:cld(N, C) - 1
        offset = k * C
        xd = SVector{N}(ntuple(Val(N)) do i
            seed = ntuple(c -> T(i == offset + c), Val(C))
            ForwardDiff.Dual{JacobianChunkTag}(x[i], ForwardDiff.Partials(seed))
        end)
        yd = f(xd)
        # columns offset+1 … offset+C of ∂y∂x (those past N carry zero seeds,
        # so their partials are identically zero and safe to accumulate)
        rowsums += SVector{N}(ntuple(Val(N)) do i
            sum(abs, ForwardDiff.partials(yd[i]))
        end)
        diags += SVector{N}(ntuple(Val(N)) do i
            c = i - offset
            1 <= c <= C ? abs(ForwardDiff.partials(yd[i], c)) : zero(T)
        end)
    end
    return rowsums, diags
end

# shared per-element Jacobian: Gershgorin row sums + |diagonal| of ∂Re/∂He via
# chunked ForwardDiff
@inline function element_jacobian(H, source, el2n, geo, D, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    Hloc = SVector{N}(ntuple(i -> H[local_nodes[i]], Val(N)))
    sloc = SVector{N}(ntuple(i -> source[local_nodes[i]], Val(N)))
    rowsums, diags = jacobian_rowsums_diags(
        Hloc -> integrate_residual(Hloc, geo, iel, sloc, D, Nq, Val(N)),
        Hloc,
        Val(min(N, jacobian_chunk)),
    )
    return local_nodes, rowsums, diags
end

# --- atomic variants: all elements run concurrently, scatter with atomics ---

@kernel function residual_atomic_kernel!(R, @Const(H), @Const(source), @Const(el2n), @Const(geo), D, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, Re = element_residual(H, source, el2n, geo, D, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic R[inod] += Re[i]
    end
end

@kernel function jacobian_atomic_kernel!(∂R∂H, PC, @Const(H), @Const(source), @Const(el2n), @Const(geo), D, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, rowsums, diags = element_jacobian(H, source, el2n, geo, D, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic ∂R∂H[inod] += rowsums[i]
        Atomix.@atomic :monotonic PC[inod]   += diags[i]
    end
end

# --- colored variants: one launch per color batch; elements within a color
# share no nodes, so plain scatter is race-free. Launches on the same backend
# queue execute in order, so no synchronization is needed between colors. ---

@kernel function residual_colored_kernel!(R, @Const(H), @Const(source), @Const(el2n), @Const(geo), D, Nq, @Const(elems), ::Val{N}) where N
    idx = @index(Global)
    iel = elems[idx]
    element_residual!(R, H, source, el2n, geo, D, Nq, iel, Val(N))
end

@kernel function jacobian_colored_kernel!(∂R∂H, PC, @Const(H), @Const(source), @Const(el2n), @Const(geo), D, Nq, @Const(elems), ::Val{N}) where N
    idx = @index(Global)
    iel = elems[idx]
    local_nodes, rowsums, diags = element_jacobian(H, source, el2n, geo, D, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        ∂R∂H[inod] += rowsums[i]
        PC[inod]   += diags[i]
    end
end

# fused PT update: rate, variable, and Dirichlet handling in one launch over
# all nodes. On constrained nodes the residual and rate are zeroed *before*
# they enter the update — otherwise the (nonzero) reaction-force residual at
# the boundary accumulates into ∂u∂τ and poisons the λmin estimate — and H is
# pinned to its BC value. R is zeroed in-place there too, so the norm(R) /
# λmin reductions see the constrained residual without a separate scatter.
@kernel function update_pt_kernel!(H, ∂u∂τ, R, @Const(PC), α, β, @Const(isΓ), @Const(HΓ))
    i = @index(Global)
    if isΓ[i]
        R[i]    = 0.0
        ∂u∂τ[i] = 0.0
        H[i]    = HΓ[i]
    else
        rate    = R[i] / PC[i] + β * ∂u∂τ[i]
        ∂u∂τ[i] = rate
        H[i]   += α * rate
    end
end

# Dirichlet constraint: v[dofs[i]] = vals[i]
@kernel function dirichlet_kernel!(v, @Const(dofs), @Const(vals))
    i = @index(Global)
    v[dofs[i]] = vals[i]
end

# ---------------------------------------------------------------------------
# launch wrappers
# ---------------------------------------------------------------------------

@inline function shape_function_values(element)
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q], ip.ζ[q]), length(ip.ω))
    return ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
end

function assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H, el2n, geo, nels, element::ReferenceElement{T}, D, source, do_∂R∂H) where T<:AbstractElement{3, N} where N
    Nq = shape_function_values(element)

    fill!(R, 0)
    residual_atomic_kernel!(backend, workgroup)(R, H, source, el2n, geo, D, Nq, Val(N); ndrange = nels)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
        jacobian_atomic_kernel!(backend, workgroup)(∂R∂H, PC, H, source, el2n, geo, D, Nq, Val(N); ndrange = nels)
    end
    KA.synchronize(backend)
end

function assemble_diffusion_matrices_colored!(R, ∂R∂H, PC, H, el2n, geo, element::ReferenceElement{T}, D, source, do_∂R∂H, colors) where T<:AbstractElement{3, N} where N
    Nq = shape_function_values(element)

    fill!(R, 0)
    for elems in colors
        residual_colored_kernel!(backend, workgroup)(R, H, source, el2n, geo, D, Nq, elems, Val(N); ndrange = length(elems))
    end
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
        for elems in colors
            jacobian_colored_kernel!(backend, workgroup)(∂R∂H, PC, H, source, el2n, geo, D, Nq, elems, Val(N); ndrange = length(elems))
        end
    end
    KA.synchronize(backend)
end

# color batches as device arrays of element ids
function color_element_batches(mesh)
    colors = color_mesh(mesh)
    batches = [Int[] for _ in 1:maximum(colors)]
    for iel in 1:mesh.nels
        push!(batches[colors[iel]], iel)
    end
    return [to_backend(batch) for batch in batches]
end

function precompute_geometry(coords, el2n, nels, element::ReferenceElement{T}) where T<:AbstractElement{3, N} where N
    ip = element.integration_points
    NQ = length(ip.ω)
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q], ip.ζ[q]), NQ)
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), NQ)
    geo = KA.allocate(backend, Tuple{SMatrix{N, 3, Float64, 3N}, Float64}, (NQ, nels))
    precompute_geometry_kernel!(backend, workgroup)(geo, coords, el2n, ∂N∂ξq, ip.ω, Val(N); ndrange = nels)
    KA.synchronize(backend)
    return geo
end

function apply_dirichlet!(v, dofs, vals)
    dirichlet_kernel!(backend, workgroup)(v, dofs, vals; ndrange = length(dofs))
    KA.synchronize(backend)
    return nothing
end

# move a host array to the compute backend
function to_backend(x::AbstractArray{T}) where T
    y = KA.allocate(backend, T, size(x))
    copyto!(y, x)
    return y
end

# ---------------------------------------------------------------------------

function main(nels)
    TDev = FEMTools.TA(backend)

    Lx = Ly = Lz = 1

    Ω = (-Lx..Lx) × (-Ly..Ly) × (-Lz..Lz)
    element  = ReferenceElement(LinearElement{3, 8, Float64})
    # element = ReferenceElement(QuadraticElement{3, 27, Float64})
    mesh     = Mesh(backend, Ω, element, nels)
    mesh_cpu = Mesh(CPU(), Ω, element, nels)

    σ      = 0.1                                # Source width
    HW     = 1.0                                # Dirichlet value west
    HE     = 0.0                                # Dirichlet value east
    epsi   = 1e-9                               # Relative tolerance
    D      = 1.0

    # Dirichlet BCs on the west/east faces only (others natural), as in 3D_Poisson_AD.jl
    left_boundary(p, D) = begin
        x, y, z = p
        I, J, K = factors(D)
        x == leftendpoint(I) && y ∈ J && z ∈ K
    end
    right_boundary(p, D) = begin
        x, y, z = p
        I, J, K = factors(D)
        x == rightendpoint(I) && y ∈ J && z ∈ K
    end
    Γl = [left_boundary(p, Ω)  for p in Array(mesh.coords)]
    Γr = [right_boundary(p, Ω) for p in Array(mesh.coords)]
    Γ_dofs = vcat(mesh.DoFs[Γl], mesh.DoFs[Γr])
    Γ_vals = vcat(
        HE .* KA.ones(backend, Float64, count(Γl)),
        HW .* KA.ones(backend, Float64, count(Γr)),
    )

    # full-length nodal mask + BC values for the fused update/Dirichlet kernel
    HΓ_cpu = zeros(length(Γl))
    HΓ_cpu[Γl] .= HE
    HΓ_cpu[Γr] .= HW
    isΓ = TDev(Vector{Bool}(Γl .| Γr))
    HΓ  = TDev(HΓ_cpu)

    # mesh data and fields on the compute backend
    source  = TDev([2*exp(-(p[1]^2+p[2]^2+p[3]^2)^2/(2σ^2)) for p in mesh_cpu.coords])
    H_FEM   = TDev([exp(-(p[1]^2+p[2]^2+p[3]^2)^2/(2σ^2)) for p in mesh_cpu.coords])
    R       = KA.zeros(backend, Float64, mesh.nnodes)
    R0      = KA.zeros(backend, Float64, mesh.nnodes)
    ∂H∂τ    = KA.zeros(backend, Float64, mesh.nnodes)
    ∂R∂H    = KA.zeros(backend, Float64, mesh.nnodes)
    PC      = KA.zeros(backend, Float64, mesh.nnodes)
    nr0     = 0.0

    geo = precompute_geometry(mesh.coords, mesh.el2n, mesh.nels, element)
    colors = color_element_batches(mesh_cpu)

    apply_dirichlet!(H_FEM, Γ_dofs, Γ_vals)

    # Estimate min/max λ
    do_∂R∂H = true
    assemble_diffusion_matrices_colored!(R, ∂R∂H, PC, H_FEM, mesh.el2n, geo, element, D, source, do_∂R∂H, colors)

    CFL    = 0.99
    c_fact = 0.9

    λmax = maximum(∂R∂H ./ PC)
    Δτ   = 2 / √(λmax) * CFL
    λmin = 0.0
    c    = 2 * √(λmin) * c_fact
    α    = 2 * Δτ^2 / (2 + c * Δτ)
    β    = (2 - c * Δτ) / (2 + c * Δτ)

    to = TimerOutput()
    ncheck = 1000
    for it = 1:100#00
    # for it = 1:10_000
        do_∂R∂H = if mod(it, ncheck) == 0
            copyto!(R0, R)
            true
        else
            false
        end
        # @timeit to "atomix" assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H_FEM, mesh.el2n, geo, mesh.nels, element, D, source, do_∂R∂H)
        @timeit to "colors" assemble_diffusion_matrices_colored!(R, ∂R∂H, PC, H_FEM, mesh.el2n, geo, element, D, source, do_∂R∂H, colors)

        # rate + variable update + Dirichlet constraints in a single launch
        update_pt_kernel!(backend, workgroup)(H_FEM, ∂H∂τ, R, PC, α, β, isΓ, HΓ; ndrange = mesh.nnodes)

        if it % ncheck == 0 || it == 1
            # array reductions: backend-agnostic (run on the device for GPU arrays)
            nr = norm(R)
            if it == 1
                nr0 = nr
            end
            isnan(nr/nr0) && error("NaNs")

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
            @printf("Iter. %05d: %2.2e\n", it, nr/nr0)
            if nr/nr0 < epsi break end
        end
    end

    display(to)

    # nodes live on a (p*nx + 1) × (p*ny + 1) × (p*nz + 1) tensor grid, p = element order
    H_host = Array(H_FEM)
    nx, ny, nz = nels .* order(element)
    xs = LinRange(-Lx, Lx, nx + 1)
    ys = LinRange(-Ly, Ly, ny + 1)
    H_grid = reshape(H_host, nx + 1, ny + 1, nz + 1)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="3D Poisson PT solution (KA, z = 0 slice)", aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, H_grid[:, :, nz ÷ 2 + 1]; colormap=:inferno)
    Colorbar(fig[1, 2], hm)
    display(fig)

    return nothing
end

n = 32 ÷ 1
nels = (n, n, n) 
print("\n $(prod(nels)) elements in a ($n × $n × $n) grid\n")
main(nels)


# Q1 elements (32^3)
# ────────────────────────────────────────────────────────────────────
#                            Time                    Allocations      
#                   ───────────────────────   ────────────────────────
# Tot / % measured:      2.92s /  89.8%            216MiB /  89.2%    
# Section   ncalls     time    %tot     avg     alloc    %tot      avg
# ────────────────────────────────────────────────────────────────────
# atomix     3.00k    1.65s   62.9%   550μs   77.8MiB   40.4%  26.6KiB
# colors     3.00k    975ms   37.1%   325μs    115MiB   59.6%  39.1KiB
# ────────────────────────────────────────────────────────────────────

# Q2 elements (32^3)
# ────────────────────────────────────────────────────────────────────
#                            Time                    Allocations      
#                   ───────────────────────   ────────────────────────
# Tot / % measured:      30.3s /  98.0%           1.44GiB /  97.9%    
# Section   ncalls     time    %tot     avg     alloc    %tot      avg
# ────────────────────────────────────────────────────────────────────
# atomix     4.00k    17.1s   57.6%  4.28ms    384MiB   26.7%  98.3KiB
# colors     4.00k    12.6s   42.4%  3.15ms   1.03GiB   73.3%   270KiB
# ────────────────────────────────────────────────────────────────────
