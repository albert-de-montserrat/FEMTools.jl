using ForwardDiff
using LinearAlgebra
using StaticArrays
using Printf
using Atomix
using TimerOutputs
using DomainSets
using DomainSets: ×
using GLMakie
using FEMTools

function element_coordinate_matrix(mesh, local_nodes::SVector{N, Int}) where {N}
    data = ntuple(Val(3N)) do k
        col = cld(k, N)
        row = k - (col - 1) * N
        mesh.coords[local_nodes[row]][col]
    end
    return SMatrix{N, 3, Float64, 3N}(data)
end

# (∂N∂x_q, dΩ_q) for every element and quadrature point. Depends only on the
# mesh, so it is computed once and reused across all PT iterations.
function precompute_geometry(mesh, element::ReferenceElement{T}) where T<:AbstractElement{3, N} where N
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q], ip.ζ[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))
    return map(1:mesh.nels) do iel
        local_nodes = SVector{N, Int}(ntuple(i -> mesh.el2n[i, iel], Val(N)))
        coords = element_coordinate_matrix(mesh, local_nodes)
        ntuple(length(ip.ω)) do q
            J = ∂N∂ξq[q]' * coords
            (∂N∂ξq[q] * inv(J), abs(det(J)) * ip.ω[q])
        end
    end
end

function integrate_residual(Hloc, geo_el, sloc, D, Nq, ::Val{N}) where N
    Re = zero(Hloc)
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv = Nq[q]
        tmp   = ∂N∂x' * Hloc
        KHloc = (D * dΩ) * (∂N∂x * tmp)
        Re   += SVector{N}(-sloc[i] * Nv[i] * dΩ - KHloc[i] for i in 1:N)
    end
    return Re
end

# per-element residual (and optionally Gershgorin row sum + |diagonal|) with plain scatter;
# safe serially or within a color batch
function integrate_residual_element!(R, ∂R∂H, PC, H, source, D, mesh, Nq, geo, iel, do_∂R∂H, ::Val{N}) where N
    local_nodes = SVector{N, Int}(ntuple(i -> mesh.el2n[i, iel], Val(N)))
    geo_el = geo[iel]
    Hloc   = SVector{N}(H[local_nodes[i]] for i in 1:N)
    sloc   = SVector{N}(source[local_nodes[i]] for i in 1:N)
    Re     = integrate_residual(Hloc, geo_el, sloc, D, Nq, Val(N))
    for (i, inod) in enumerate(local_nodes)
        R[inod] += Re[i]
    end
    if do_∂R∂H
        ∂Re∂He = ForwardDiff.jacobian(
            Hloc -> integrate_residual(Hloc, geo_el, sloc, D, Nq, Val(N)),
            Hloc
        )
        for (i, inod) in enumerate(local_nodes)
            ∂R∂H[inod] += sum(abs(∂Re∂He[i, j]) for j in 1:N) # sum over ||row||
            PC[inod]   += abs(∂Re∂He[i, i])                   # |diagonal|
        end
    end
end

# same as integrate_residual_element! but with atomic scatter, safe for unordered threading
function integrate_residual_atomic!(R, ∂R∂H, PC, H, source, D, mesh, Nq, geo, iel, do_∂R∂H, ::Val{N}) where N
    local_nodes = SVector{N, Int}(ntuple(i -> mesh.el2n[i, iel], Val(N)))
    geo_el = geo[iel]
    Hloc   = SVector{N}(H[local_nodes[i]] for i in 1:N)
    sloc   = SVector{N}(source[local_nodes[i]] for i in 1:N)
    Re     = integrate_residual(Hloc, geo_el, sloc, D, Nq, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic R[inod] += Re[i]
    end
    if do_∂R∂H
        ∂Re∂He = ForwardDiff.jacobian(
            Hloc -> integrate_residual(Hloc, geo_el, sloc, D, Nq, Val(N)),
            Hloc
        )
        for (i, inod) in enumerate(local_nodes)
            Atomix.@atomic :monotonic ∂R∂H[inod] += sum(abs(∂Re∂He[i, j]) for j in 1:N) # sum over ||row||
            Atomix.@atomic :monotonic PC[inod]   += abs(∂Re∂He[i, i])                   # |diagonal|
        end
    end
end

function assemble_diffusion_matrices!(R, ∂R∂H, PC, H, mesh, element::ReferenceElement{T}, D, source, geo, do_∂R∂H) where T<:AbstractElement{3, N} where N

    fill!(R, 0)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
    end
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q], ip.ζ[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))

    for iel in 1:mesh.nels
        integrate_residual_element!(R, ∂R∂H, PC, H, source, D, mesh, Nq, geo, iel, do_∂R∂H, Val(N))
    end
end

function assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H, mesh, element::ReferenceElement{T}, D, source, geo, do_∂R∂H) where T<:AbstractElement{3, N} where N

    fill!(R, 0)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
    end
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q], ip.ζ[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))

    Threads.@threads :static for iel in 1:mesh.nels
        integrate_residual_atomic!(R, ∂R∂H, PC, H, source, D, mesh, Nq, geo, iel, do_∂R∂H, Val(N))
    end
end

function assemble_diffusion_matrices_colored!(R, ∂R∂H, PC, H, mesh, element::ReferenceElement{T}, D, source, geo, do_∂R∂H, colors) where T<:AbstractElement{3, N} where N

    fill!(R, 0)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
    end
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q], ip.ζ[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))

    for color in colors
        Threads.@threads for iel in color
            integrate_residual_element!(R, ∂R∂H, PC, H, source, D, mesh, Nq, geo, iel, do_∂R∂H, Val(N))
        end
    end
end

function color_element_batches(mesh)
    colors = color_mesh(mesh)
    batches = [Int[] for _ in 1:maximum(colors)]
    for iel in 1:mesh.nels
        push!(batches[colors[iel]], iel)
    end
    return batches
end

function update_rate!(∂u∂τ, R, PC, β)
    @. ∂u∂τ = R / PC + β * ∂u∂τ
end

function update_variable!(H, ∂u∂τ, α)
    @. H += α * ∂u∂τ
end

function main(nels)
    Lx = Ly = Lz = 1

    Ω = (-Lx..Lx) × (-Ly..Ly) × (-Lz..Lz)
    element = ReferenceElement(LinearElement{3, 8, Float64})
    # element = ReferenceElement(QuadraticElement{3, 27, Float64})
    mesh = FEMTools.Mesh(Ω, element, nels)
    
    colors = color_element_batches(mesh)
    geo = precompute_geometry(mesh, element)

    σ      = 0.1                                # Source width
    source = [2*exp(-(p[1]^2+p[2]^2+p[3]^2)^2/(2σ^2)) for p in mesh.coords]  # Source
    HW     = 1.0                                # Dirichlet value west
    HE     = 0.0                                # Dirichlet value east
    epsi   = 1e-9                               # Relative tolerance

    # FEM PT solve
    D            = 1.0
    H_FEM        = [exp(-(p[1]^2+p[2]^2+p[3]^2)^2/(2σ^2)) for p in mesh.coords]
    R            = zeros(mesh.nnodes)
    R0           = zeros(mesh.nnodes)
    ∂H∂τ         = zeros(mesh.nnodes)
    ∂R∂H         = zeros(mesh.nnodes)
    PC           = zeros(mesh.nnodes)
    nr0          = 0.0

    # Dirichlet BCs on the west/east faces only (others natural), as in 2D_Poisson_AD.jl
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
    Γl = [left_boundary(p, Ω) for p in mesh.coords]
    Γr = [right_boundary(p, Ω) for p in mesh.coords]
    Γl_dofs = mesh.DoFs[Γl]
    Γr_dofs = mesh.DoFs[Γr]
    Γl_vals = fill(HE, length(Γl_dofs))
    Γr_vals = fill(HW, length(Γr_dofs))
    Γ_dofs = vcat(Γl_dofs, Γr_dofs)
    Γ_vals = vcat(Γl_vals, Γr_vals)

    ΓD_H = DirichletBoundaryCondition(mesh.Γ, Γ_dofs, Γ_vals)
    ΓD_R = DirichletBoundaryCondition(mesh.Γ, Γ_dofs, zero(Γ_vals))
    apply_bc!(H_FEM, ΓD_H)

    # Estimate min/max λ
    do_∂R∂H = true
    assemble_diffusion_matrices!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, geo, do_∂R∂H)

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
        @timeit to "series" assemble_diffusion_matrices!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, geo, do_∂R∂H)
        @timeit to "atomix" assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, geo, do_∂R∂H)
        # @timeit to "colors" assemble_diffusion_matrices_colored!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, geo, do_∂R∂H, colors)

        # Dirichlet BCs: constrain residual and rate *before* the update,
        # otherwise the (nonzero) reaction-force residual at the boundary
        # nodes accumulates into ∂H∂τ and poisons the λmin estimate
        apply_bc!(R, ΓD_R)
        apply_bc!(∂H∂τ, ΓD_R)

        update_rate!(∂H∂τ, R, PC, β)
        update_variable!(H_FEM, ∂H∂τ, α)

        apply_bc!(H_FEM, ΓD_H)

        if it % ncheck == 0 || it == 1
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
    nx, ny, nz = nels .* order(element)
    xs = LinRange(-Lx, Lx, nx + 1)
    ys = LinRange(-Ly, Ly, ny + 1)
    H_grid = reshape(H_FEM, nx + 1, ny + 1, nz + 1)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="3D Poisson PT solution (z = 0 slice)", aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, H_grid[:, :, nz ÷ 2 + 1]; colormap=:vikO)
    Colorbar(fig[1, 2], hm)
    
    # fig = scatterlines(xs, (reshape(H_FEM, nx + 1, ny + 1, nz +1))[:, ny>>>1, nz>>>1]; label="converged")
    display(fig)


    return nothing
end

n = 32
nels = (n, n, n)
main(nels)
