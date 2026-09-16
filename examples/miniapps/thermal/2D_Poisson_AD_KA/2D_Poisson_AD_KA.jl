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
const workgroup = 64

@inline function element_coordinate_matrix(coords, local_nodes::SVector{N, Int}) where {N}
    data = ntuple(Val(2N)) do k
        col = cld(k, N)
        row = k - (col - 1) * N
        coords[local_nodes[row]][col]
    end
    return SMatrix{N, 2, Float64, 2N}(data)
end

@inline local_nodes_of(el2n, iel, ::Val{N}) where N =
    SVector{N, Int}(ntuple(i -> Int(el2n[i, iel]), Val(N)))

# ---------------------------------------------------------------------------
# KA kernels
# ---------------------------------------------------------------------------

# (∂N∂x_q, dΩ_q) for every element and quadrature point. Depends only on the
# mesh, so it is computed once and reused across all PT iterations.
@kernel function precompute_geometry_kernel!(geo, @Const(coords), @Const(el2n), ∂N∂ξq, ω, ::Val{N}) where N
    iel = @index(Global)
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    c = element_coordinate_matrix(coords, local_nodes)
    geo[iel] = ntuple(Val(length(ω))) do q
        J = ∂N∂ξq[q]' * c
        (∂N∂ξq[q] * inv(J), abs(det(J)) * ω[q])
    end
end

@inline function integrate_residual(Hloc, geo_el, sloc, D, Nq, ::Val{N}) where N
    Re = zero(Hloc)
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
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
    geo_el = geo[iel]
    Hloc = SVector{N}(ntuple(i -> H[local_nodes[i]], Val(N)))
    sloc = SVector{N}(ntuple(i -> source[local_nodes[i]], Val(N)))
    Re   = integrate_residual(Hloc, geo_el, sloc, D, Nq, Val(N))
    return local_nodes, Re
end

# shared per-element Jacobian: Gershgorin row sums + |diagonal| of ∂Re/∂He via ForwardDiff
@inline function element_jacobian(H, source, el2n, geo, D, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    Hloc = SVector{N}(ntuple(i -> H[local_nodes[i]], Val(N)))
    sloc = SVector{N}(ntuple(i -> source[local_nodes[i]], Val(N)))
    ∂Re∂He = ForwardDiff.jacobian(
        Hloc -> integrate_residual(Hloc, geo_el, sloc, D, Nq, Val(N)),
        Hloc
    )
    rowsums = SVector{N}(ntuple(i -> sum(abs(∂Re∂He[i, j]) for j in 1:N), Val(N)))
    diags   = SVector{N}(ntuple(i -> abs(∂Re∂He[i, i]), Val(N)))
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
    local_nodes, Re = element_residual(H, source, el2n, geo, D, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        R[inod] += Re[i]
    end
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

@kernel function update_rate_kernel!(∂u∂τ, @Const(R), @Const(PC), β)
    i = @index(Global)
    ∂u∂τ[i] = R[i] / PC[i] + β * ∂u∂τ[i]
end

@kernel function update_variable_kernel!(H, @Const(∂u∂τ), α)
    i = @index(Global)
    H[i] += α * ∂u∂τ[i]
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
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    return ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
end

function assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H, el2n, geo, nels, element::ReferenceElement{T}, D, source, do_∂R∂H) where T<:AbstractElement{2, N} where N
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

function assemble_diffusion_matrices_colored!(R, ∂R∂H, PC, H, el2n, geo, element::ReferenceElement{T}, D, source, do_∂R∂H, colors) where T<:AbstractElement{2, N} where N
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

function precompute_geometry(coords, el2n, nels, element::ReferenceElement{T}) where T<:AbstractElement{2, N} where N
    ip = element.integration_points
    NQ = length(ip.ω)
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), NQ)
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), NQ)
    geo = KA.allocate(backend, NTuple{NQ, Tuple{SMatrix{N, 2, Float64, 2N}, Float64}}, nels)
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
    
    Lx = Ly = 1

    Ω = (-Lx..Lx) × (-Ly..Ly)
    # element = ReferenceElement(LinearElement{2, 4, Float64})
    element  = ReferenceElement(QuadraticElement{2, 9, Float64})
    mesh     = Mesh(backend, Ω, element, nels)
    mesh_cpu = Mesh(CPU(), Ω, element, nels)

    σ      = 0.1                                # Source width
    HW     = 1.0                                # Dirichlet value west
    HE     = 0.0                                # Dirichlet value east
    epsi   = 1e-9                               # Relative tolerance
    D      = 1.0

    # Dirichlet BCs on the left/right faces only (top/bottom natural), as in 2D_Poisson.jl
    left_boundary(p, D) = begin
        x, y = p
        I, J = factors(D)
        x == leftendpoint(I) && y ∈ J
    end
    right_boundary(p, D) = begin
        x, y = p
        I, J = factors(D)
        x == rightendpoint(I) && y ∈ J
    end
    Γl = [left_boundary(p, Ω)  for p in Array(mesh.coords)]
    Γr = [right_boundary(p, Ω) for p in Array(mesh.coords)]
    Γ_dofs = vcat(mesh.DoFs[Γl], mesh.DoFs[Γr])
    Γ_vals = vcat(
        HE .* KA.ones(backend, Float64, count(Γl)),
        HW .* KA.ones(backend, Float64, count(Γr)),
    )

    # mesh data and fields on the compute backend
    source  = TDev([2*exp(-(p[1]^2+p[2]^2)^2/(2σ^2)) for p in mesh_cpu.coords])
    H_FEM   = TDev([exp(-(p[1]^2+p[2]^2)^2/(2σ^2)) for p in mesh_cpu.coords])
    Γ_zero  = zero(Γ_vals)
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
    assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H_FEM, mesh.el2n, geo, mesh.nels, element, D, source, do_∂R∂H)

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
    for it = 1:10_000
        do_∂R∂H = if mod(it, ncheck) == 0
            copyto!(R0, R)
            true
        else
            false
        end
        @timeit to "atomix" assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H_FEM, mesh.el2n, geo, mesh.nels, element, D, source, do_∂R∂H)
        @timeit to "colors" assemble_diffusion_matrices_colored!(R, ∂R∂H, PC, H_FEM, mesh.el2n, geo, element, D, source, do_∂R∂H, colors)

        # Dirichlet BCs: constrain residual and rate *before* the update,
        # otherwise the (nonzero) reaction-force residual at the boundary
        # nodes accumulates into ∂H∂τ and poisons the λmin estimate
        apply_dirichlet!(R, Γ_dofs, Γ_zero)
        apply_dirichlet!(∂H∂τ, Γ_dofs, Γ_zero)

        update_rate_kernel!(backend, workgroup)(∂H∂τ, R, PC, β; ndrange = mesh.nnodes)
        update_variable_kernel!(backend, workgroup)(H_FEM, ∂H∂τ, α; ndrange = mesh.nnodes)

        apply_dirichlet!(H_FEM, Γ_dofs, Γ_vals)

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

    # nodes live on a (p*nx + 1) × (p*ny + 1) tensor grid, p = element order
    H_host = Array(H_FEM)
    nx, ny = nels .* order(element)
    xs = LinRange(-Lx, Lx, nx + 1)
    ys = LinRange(-Ly, Ly, ny + 1)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="2D Poisson PT solution (KA)", aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, reshape(H_host, nx + 1, ny + 1); colormap=:inferno)
    Colorbar(fig[1, 2], hm)
    display(fig)

    return nothing
end

n = 110
nels = (n, n) .* 2
print("\n $(prod(nels)) elements in a ($n × $n) grid\n")
main(nels)


# ────────────────────────────────────────────────────────────────────
#                            Time                    Allocations      
#                   ───────────────────────   ────────────────────────
# Tot / % measured:      10.6s /  92.1%            341MiB /  86.0%    

# Section   ncalls     time    %tot     avg     alloc    %tot      avg
# ────────────────────────────────────────────────────────────────────
# colors     7.00k    5.90s   60.6%   843μs    225MiB   76.8%  32.9KiB
# atomix     7.00k    3.83s   39.4%   547μs   68.1MiB   23.2%  10.0KiB
# ────────────────────────────────────────────────────────────────────
