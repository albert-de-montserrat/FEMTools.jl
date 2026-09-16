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
# ---------------------------------------------------------------------------
# 2D linear elasticity (plane strain) of a cantilever clamped on the left
# face, loaded by gravity, solved with dynamic relaxation (second-order
# pseudo-transient iteration). Two coupled residuals Rx, Ry — one per
# displacement component — each with its own Gershgorin row sum (∂R∂Ux, ∂R∂Uy)
# and diagonal preconditioner (PCx, PCy). Assembly is atomics-based only.
# ---------------------------------------------------------------------------

const backend   = CPU()   # swap for e.g. MetalBackend() / CUDABackend(); CPU() threads with julia -t
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
# mesh, so it is computed once and reused across all DR iterations.
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

# element residuals with atomic scatter into the two component fields
@kernel function residual_atomic_kernel!(Rx, Ry, @Const(Ux), @Const(Uy), @Const(el2n), @Const(geo), λ, μ, bx, by, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    uloc = vcat(SVector{N}(ntuple(i -> Ux[local_nodes[i]], Val(N))),
                SVector{N}(ntuple(i -> Uy[local_nodes[i]], Val(N))))
    Re = elastic_residual(uloc, geo_el, λ, μ, bx, by, Nq, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic Rx[inod] += Re[i]
        Atomix.@atomic :monotonic Ry[inod] += Re[N + i]
    end
end

# Gershgorin row sums and |diagonal| of the full coupled 2N x 2N element
# Jacobian via ForwardDiff: rows 1..N feed (∂R∂Ux, PCx), rows N+1..2N feed
# (∂R∂Uy, PCy). Row sums include the x-y cross-coupling blocks.
@kernel function jacobian_atomic_kernel!(∂R∂Ux, ∂R∂Uy, PCx, PCy, @Const(Ux), @Const(Uy), @Const(el2n), @Const(geo), λ, μ, bx, by, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    uloc = vcat(SVector{N}(ntuple(i -> Ux[local_nodes[i]], Val(N))),
                SVector{N}(ntuple(i -> Uy[local_nodes[i]], Val(N))))
    ∂Re∂ue = ForwardDiff.jacobian(
        uloc -> elastic_residual(uloc, geo_el, λ, μ, bx, by, Nq, Val(N)),
        uloc
    )
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic ∂R∂Ux[inod] += sum(abs(∂Re∂ue[i, j])     for j in 1:2N) # ||row|| incl. coupling
        Atomix.@atomic :monotonic ∂R∂Uy[inod] += sum(abs(∂Re∂ue[N + i, j]) for j in 1:2N)
        Atomix.@atomic :monotonic PCx[inod]   += abs(∂Re∂ue[i, i])                        # |diagonal|
        Atomix.@atomic :monotonic PCy[inod]   += abs(∂Re∂ue[N + i, N + i])
    end
end

# colored variants: one launch per color batch; elements within a color share
# no nodes, so the scatter into nodal fields is race-free without atomics.
@kernel function residual_colored_kernel!(Rx, Ry, @Const(Ux), @Const(Uy), @Const(el2n), @Const(geo), λ, μ, bx, by, Nq, @Const(elems), ::Val{N}) where N
    idx = @index(Global)
    iel = elems[idx]
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    uloc = vcat(SVector{N}(ntuple(i -> Ux[local_nodes[i]], Val(N))),
                SVector{N}(ntuple(i -> Uy[local_nodes[i]], Val(N))))
    Re = elastic_residual(uloc, geo_el, λ, μ, bx, by, Nq, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Rx[inod] += Re[i]
        Ry[inod] += Re[N + i]
    end
end

@kernel function jacobian_colored_kernel!(∂R∂Ux, ∂R∂Uy, PCx, PCy, @Const(Ux), @Const(Uy), @Const(el2n), @Const(geo), λ, μ, bx, by, Nq, @Const(elems), ::Val{N}) where N
    idx = @index(Global)
    iel = elems[idx]
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    uloc = vcat(SVector{N}(ntuple(i -> Ux[local_nodes[i]], Val(N))),
                SVector{N}(ntuple(i -> Uy[local_nodes[i]], Val(N))))
    ∂Re∂ue = ForwardDiff.jacobian(
        uloc -> elastic_residual(uloc, geo_el, λ, μ, bx, by, Nq, Val(N)),
        uloc
    )
    for (i, inod) in enumerate(local_nodes)
        ∂R∂Ux[inod] += sum(abs(∂Re∂ue[i, j])     for j in 1:2N)
        ∂R∂Uy[inod] += sum(abs(∂Re∂ue[N + i, j]) for j in 1:2N)
        PCx[inod]   += abs(∂Re∂ue[i, i])
        PCy[inod]   += abs(∂Re∂ue[N + i, N + i])
    end
end

@kernel function update_rate_kernel!(∂u∂τ, @Const(R), @Const(PC), β)
    i = @index(Global)
    ∂u∂τ[i] = R[i] / PC[i] + β * ∂u∂τ[i]
end

@kernel function update_variable_kernel!(U, @Const(∂u∂τ), α)
    i = @index(Global)
    U[i] += α * ∂u∂τ[i]
end

# Dirichlet constraint: v[dofs[i]] = vals[i]
@kernel function dirichlet_kernel!(v, @Const(dofs), @Const(vals))
    i = @index(Global)
    v[dofs[i]] = vals[i]
end

# Lagrangian grid advection: x ← x + u (clamped nodes have u = 0 and stay put)
@kernel function advect_grid_kernel!(coords, @Const(Ux), @Const(Uy))
    i = @index(Global)
    p = coords[i]
    coords[i] = typeof(p)(p[1] + Ux[i], p[2] + Uy[i])
end

# ---------------------------------------------------------------------------
# launch wrappers
# ---------------------------------------------------------------------------

@inline function shape_function_values(element)
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    return ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
end

function assemble_elasticity_atomix!(Rx, Ry, ∂R∂Ux, ∂R∂Uy, PCx, PCy, Ux, Uy, el2n, geo, nels, element::ReferenceElement{T}, λ, μ, bx, by, do_∂R∂u) where T<:AbstractElement{2, N} where N
    Nq = shape_function_values(element)

    fill!(Rx, 0)
    fill!(Ry, 0)
    residual_atomic_kernel!(backend, workgroup)(Rx, Ry, Ux, Uy, el2n, geo, λ, μ, bx, by, Nq, Val(N); ndrange = nels)
    if do_∂R∂u
        fill!(∂R∂Ux, 0)
        fill!(∂R∂Uy, 0)
        fill!(PCx, 0)
        fill!(PCy, 0)
        jacobian_atomic_kernel!(backend, workgroup)(∂R∂Ux, ∂R∂Uy, PCx, PCy, Ux, Uy, el2n, geo, λ, μ, bx, by, Nq, Val(N); ndrange = nels)
    end
    KA.synchronize(backend)
end

function assemble_elasticity_colored!(Rx, Ry, ∂R∂Ux, ∂R∂Uy, PCx, PCy, Ux, Uy, el2n, geo, element::ReferenceElement{T}, λ, μ, bx, by, do_∂R∂u, colors) where T<:AbstractElement{2, N} where N
    Nq = shape_function_values(element)

    fill!(Rx, 0)
    fill!(Ry, 0)
    for elems in colors
        residual_colored_kernel!(backend, workgroup)(Rx, Ry, Ux, Uy, el2n, geo, λ, μ, bx, by, Nq, elems, Val(N); ndrange = length(elems))
    end
    if do_∂R∂u
        fill!(∂R∂Ux, 0)
        fill!(∂R∂Uy, 0)
        fill!(PCx, 0)
        fill!(PCy, 0)
        for elems in colors
            jacobian_colored_kernel!(backend, workgroup)(∂R∂Ux, ∂R∂Uy, PCx, PCy, Ux, Uy, el2n, geo, λ, μ, bx, by, Nq, elems, Val(N); ndrange = length(elems))
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

# in-place rebuild: required after every grid advection, since J/∂N∂x/dΩ are
# only constant while the nodes stand still
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
    geo = KA.allocate(backend, NTuple{NQ, Tuple{SMatrix{N, 2, Float64, 2N}, Float64}}, nels)
    return precompute_geometry!(geo, coords, el2n, nels, element)
end

apply_dirichlet!(v, dofs, vals) =
    dirichlet_kernel!(backend, workgroup)(v, dofs, vals; ndrange = length(dofs))

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

# ---------------------------------------------------------------------------

# one quasi-static DR solve for the incremental displacements (Ux, Uy) due to
# the load increment (bx, by) on the current grid configuration
function solve_increment!(Ux, Uy, Rx, Ry, Rx0, Ry0, ∂Ux∂τ, ∂Uy∂τ, ∂R∂Ux, ∂R∂Uy, PCx, PCy,
                          el2n, geo, nels, nnodes, element, λ, μ, bx, by, Γ_dofs, Γ_zero, epsi, to, colors)
    fill!(Ux, 0)
    fill!(Uy, 0)
    fill!(∂Ux∂τ, 0)
    fill!(∂Uy∂τ, 0)

    # estimate λmax on the current configuration
    assemble_elasticity_colored!(Rx, Ry, ∂R∂Ux, ∂R∂Uy, PCx, PCy, Ux, Uy, el2n, geo, element, λ, μ, bx, by, true, colors)

    CFL    = 0.99
    c_fact = 0.9

    λmax = max(maximum(∂R∂Ux ./ PCx), maximum(∂R∂Uy ./ PCy))
    Δτ   = 2 / √(λmax) * CFL
    λmin = 0.0
    c    = 2 * √(λmin) * c_fact
    α    = 2 * Δτ^2 / (2 + c * Δτ)
    β    = (2 - c * Δτ) / (2 + c * Δτ)

    nr0    = 0.0
    relres = NaN
    iters  = 0
    ncheck = 1000
    for it = 1:200_000
        iters = it
        do_∂R∂u = if mod(it, ncheck) == 0
            copyto!(Rx0, Rx)
            copyto!(Ry0, Ry)
            true
        else
            false
        end
        @timeit to "colors" assemble_elasticity_colored!(Rx, Ry, ∂R∂Ux, ∂R∂Uy, PCx, PCy, Ux, Uy, el2n, geo, element, λ, μ, bx, by, do_∂R∂u, colors)

        # clamped BCs: constrain residuals and rates *before* the update,
        # otherwise the (nonzero) reaction-force residual at the clamped
        # nodes accumulates into the rates and poisons the λmin estimate
        apply_dirichlet!(Rx, Γ_dofs, Γ_zero)
        apply_dirichlet!(Ry, Γ_dofs, Γ_zero)
        apply_dirichlet!(∂Ux∂τ, Γ_dofs, Γ_zero)
        apply_dirichlet!(∂Uy∂τ, Γ_dofs, Γ_zero)

        update_rate_kernel!(backend, workgroup)(∂Ux∂τ, Rx, PCx, β; ndrange = nnodes)
        update_rate_kernel!(backend, workgroup)(∂Uy∂τ, Ry, PCy, β; ndrange = nnodes)
        update_variable_kernel!(backend, workgroup)(Ux, ∂Ux∂τ, α; ndrange = nnodes)
        update_variable_kernel!(backend, workgroup)(Uy, ∂Uy∂τ, α; ndrange = nnodes)

        apply_dirichlet!(Ux, Γ_dofs, Γ_zero)
        apply_dirichlet!(Uy, Γ_dofs, Γ_zero)
        KA.synchronize(backend)

        if it % ncheck == 0 || it == 1
            # array reductions: backend-agnostic (run on the device for GPU arrays)
            nr = √(norm(Rx)^2 + norm(Ry)^2)
            if it == 1
                nr0 = nr
            end
            relres = nr / nr0
            isnan(relres) && error("NaNs")

            λmax = max(maximum(∂R∂Ux ./ PCx), maximum(∂R∂Uy ./ PCy))
            Δτ   = 2 / √(λmax) * CFL
            denom = sum( (Δτ.*∂Ux∂τ).^2 ) + sum( (Δτ.*∂Uy∂τ).^2 )
            λmin  = if it == 1 || denom == 0
                0.0 # R0 not valid yet at it==1; denom==0 at convergence
            else
                abs( sum(Δτ.*∂Ux∂τ.*( (Rx .- Rx0) ./ PCx )) +
                     sum(Δτ.*∂Uy∂τ.*( (Ry .- Ry0) ./ PCy )) ) / denom
            end
            c    = 2 * √(λmin) * c_fact
            α    = 2 * Δτ^2 / (2 + c * Δτ)
            β    = (2 - c * Δτ) / (2 + c * Δτ)
            if relres < epsi break end
        end
    end
    return iters, relres
end

function main(nels)
    Lx, Ly = 4.0, 1.0                           # cantilever: length x thickness

    Ω = (0.0..Lx) × (0.0..Ly)
    element = ReferenceElement(LinearElement{2, 4, Float64})
    # element = ReferenceElement(QuadraticElement{2, 9, Float64})
    mesh = Mesh(Ω, element, nels)

    # material (plane strain) and gravity load
    E      = 1.0                                # Young's modulus
    ν      = 0.3                                # Poisson ratio
    μ      = E / (2 * (1 + ν))                  # shear modulus
    λ      = E * ν / ((1 + ν) * (1 - 2ν))       # Lamé parameter
    bx       = 0.0                              # body force x
    by_total = -1e-3                            # total gravity, applied incrementally
    nsteps   = 1                                # load/time steps (Lagrangian advection)
    Δby      = by_total / nsteps                # gravity increment per step
    epsi     = 1e-9

    # clamped on the left face: ux = uy = 0; all other faces traction-free
    clamped(p, D) = begin
        x, y = p
        I, J = factors(D)
        x == leftendpoint(I) && y ∈ J
    end
    Γc = [clamped(p, Ω) for p in mesh.coords]
    Γ_dofs_host = mesh.DoFs[Γc]

    # mesh data and fields on the compute backend
    coords  = to_backend(mesh.coords)
    el2n    = to_backend(mesh.el2n)
    Γ_dofs  = to_backend(Γ_dofs_host)
    Γ_zero  = to_backend(zeros(Float64, length(Γ_dofs_host)))
    Ux      = KA.zeros(backend, Float64, mesh.nnodes)
    Uy      = KA.zeros(backend, Float64, mesh.nnodes)
    Uxtot   = KA.zeros(backend, Float64, mesh.nnodes)
    Uytot   = KA.zeros(backend, Float64, mesh.nnodes)
    Rx      = KA.zeros(backend, Float64, mesh.nnodes)
    Ry      = KA.zeros(backend, Float64, mesh.nnodes)
    Rx0     = KA.zeros(backend, Float64, mesh.nnodes)
    Ry0     = KA.zeros(backend, Float64, mesh.nnodes)
    ∂Ux∂τ   = KA.zeros(backend, Float64, mesh.nnodes)
    ∂Uy∂τ   = KA.zeros(backend, Float64, mesh.nnodes)
    ∂R∂Ux   = KA.zeros(backend, Float64, mesh.nnodes)
    ∂R∂Uy   = KA.zeros(backend, Float64, mesh.nnodes)
    PCx     = KA.zeros(backend, Float64, mesh.nnodes)
    PCy     = KA.zeros(backend, Float64, mesh.nnodes)

    geo = precompute_geometry(coords, el2n, mesh.nels, element)
    colors = color_element_batches(mesh)

    # time stepping with updated-Lagrangian grid advection: each step solves
    # linear elasticity for the *incremental* displacement due to the gravity
    # increment on the current configuration, then advects the grid with that
    # increment and rebuilds the geometry cache. No stress history is carried
    # between steps — a first-order approximation of finite deformation,
    # adequate for moderate deflections.
    itip = argmin([abs(p[1] - Lx) + abs(p[2] - Ly/2) for p in mesh.coords])
    to = TimerOutput()
    for step in 1:nsteps
        iters, relres = solve_increment!(Ux, Uy, Rx, Ry, Rx0, Ry0, ∂Ux∂τ, ∂Uy∂τ,
                                         ∂R∂Ux, ∂R∂Uy, PCx, PCy, el2n, geo,
                                         mesh.nels, mesh.nnodes, element, λ, μ,
                                         bx, Δby, Γ_dofs, Γ_zero, epsi, to, colors)

        # advect the grid with the increment and rebuild the geometry cache
        # (J/∂N∂x/dΩ are only constant while the nodes stand still)
        advect_grid_kernel!(backend, workgroup)(coords, Ux, Uy; ndrange = mesh.nnodes)
        KA.synchronize(backend)
        precompute_geometry!(geo, coords, el2n, mesh.nels, element)

        # accumulate total displacement
        update_variable_kernel!(backend, workgroup)(Uxtot, Ux, 1.0; ndrange = mesh.nnodes)
        update_variable_kernel!(backend, workgroup)(Uytot, Uy, 1.0; ndrange = mesh.nnodes)
        KA.synchronize(backend)

        @printf("step %02d: %6d DR iters (relres %.1e), tip uy = %+.4e\n",
                step, iters, relres, Array(Uytot)[itip])
    end

    display(to)

    # small-deflection reference for the *total* load (Euler-Bernoulli, plane
    # strain, no shear): the advected solution deviates from it as the
    # deflection grows and the configuration changes
    E′ = E / (1 - ν^2)
    I = Ly^3 / 12
    w_eb = by_total * Ly * Lx^4 / (8 * E′ * I)
    @printf("final tip uy = %.4e  (linear Euler-Bernoulli estimate %.4e)\n",
            Array(Uytot)[itip], w_eb)
    @printf("mesh: %d elements, %d nodes, %d DoFs\n",
            mesh.nels, mesh.nnodes, 2 * mesh.nnodes)

    # deformed (advected) grid rendered as a mesh, coloured by total uy
    coords_def = Array(coords)
    vertices = [GLMakie.Point2f(p[1], p[2]) for p in coords_def]
    faces = plot_triangles(mesh, element)
    fig = Figure(size = (1000, 350))
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="cantilever, advected grid (DR, KA colored)", aspect=DataAspect())
    m = GLMakie.mesh!(ax, vertices, faces; color = Array(Uytot), colormap = :bilbao, shading = NoShading)
    Colorbar(fig[1, 2], m)
    display(fig)

    # state for post-processing (coords/geo are the advected configuration;
    # mesh.coords still holds the reference configuration)
    return (; mesh, element, coords, el2n, geo, Ux = Uxtot, Uy = Uytot, λ, μ, E, ν, Lx, Ly, by_total)
    # return nothing
end

nels = (160, 40) 
# nels = (1000, 250) 
state = main(nels);
println("solver done")
