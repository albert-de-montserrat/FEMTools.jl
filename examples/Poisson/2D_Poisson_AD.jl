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
    data = ntuple(Val(2N)) do k
        col = cld(k, N)
        row = k - (col - 1) * N
        mesh.coords[local_nodes[row]][col]
    end
    return SMatrix{N, 2, Float64, 2N}(data)
end

# @inline function integrate_residual(Hloc, coords, sloc, D, Nq, ∂N∂ξq, ω, ::Val{N}) where N
#     Re = zero(Hloc)
#     for q in eachindex(ω)
#         Nv = Nq[q]
#         ∂N∂ξ = ∂N∂ξq[q]
#         J = ∂N∂ξ' * coords
#         ∂N∂x = ∂N∂ξ * inv(J)
#         dΩ = abs(det(J)) * ω[q]

#         tmp   = ∂N∂x' * Hloc
#         KHloc = (D * dΩ) * (∂N∂x * tmp)
#         Re   += SVector{N}(-sloc[i] * Nv[i] * dΩ - KHloc[i] for i in 1:N)
#     end
#     return Re
# end

@generated function integrate_residual(Hloc, coords, sloc, D, Nq, ∂N∂ξq, ω::SVector{M, T}, ::Val{N}) where {M, N, T}
    quote
        @inline
        Base.@nexprs $M q -> Re_q = zero(T)
        Base.@nexprs $M q -> begin
            Nv = Nq[q]
            ∂N∂ξ = ∂N∂ξq[q]
            J = ∂N∂ξ' * coords
            ∂N∂x = ∂N∂ξ * inv(J)
            dΩ = abs(det(J)) * ω[q]

            tmp   = ∂N∂x' * Hloc
            KHloc = (D * dΩ) * (∂N∂x * tmp)
            Base.@nexprs $N i -> Re_i += -sloc[i] * Nv[i] * dΩ - KHloc[i]
        end
        Re = Base.@ncall $N SVector Re
        return Re
    end
end

# per-element residual (and optionally Gershgorin row sum + |diagonal|) with plain scatter;
# safe serially or within a color batch
function integrate_residual_element!(R, ∂R∂H, PC, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, do_∂R∂H, ::Val{N}) where N
    local_nodes = SVector{N, Int}(ntuple(i -> mesh.el2n[i, iel], Val(N)))
    coords = element_coordinate_matrix(mesh, local_nodes)
    Hloc   = SVector{N}(H[local_nodes[i]] for i in 1:N)
    sloc   = SVector{N}(source[local_nodes[i]] for i in 1:N)
    Re     = integrate_residual(Hloc, coords, sloc, D, Nq, ∂N∂ξq, ip.ω, Val(N))
    for (i, inod) in enumerate(local_nodes)
        R[inod] += Re[i]
    end
    if do_∂R∂H
        ∂Re∂He = ForwardDiff.jacobian(
            Hloc -> integrate_residual(Hloc, coords, sloc, D, Nq, ∂N∂ξq, ip.ω, Val(N)),
            Hloc
        )
        for (i, inod) in enumerate(local_nodes)
            ∂R∂H[inod] += sum(abs(∂Re∂He[i, j]) for j in 1:N) # sum over ||row||
            PC[inod]   += abs(∂Re∂He[i, i])                   # |diagonal|
        end
    end
end

# same as integrate_residual_element! but with atomic scatter, safe for unordered threading
function integrate_residual_atomic!(R, ∂R∂H, PC, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, do_∂R∂H, ::Val{N}) where N
    local_nodes = SVector{N, Int}(ntuple(i -> mesh.el2n[i, iel], Val(N)))
    coords = element_coordinate_matrix(mesh, local_nodes)
    Hloc   = SVector{N}(H[local_nodes[i]] for i in 1:N)
    sloc   = SVector{N}(source[local_nodes[i]] for i in 1:N)
    Re     = integrate_residual(Hloc, coords, sloc, D, Nq, ∂N∂ξq, ip.ω, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic R[inod] += Re[i]
    end
    if do_∂R∂H
        ∂Re∂He = ForwardDiff.jacobian(
            Hloc -> integrate_residual(Hloc, coords, sloc, D, Nq, ∂N∂ξq, ip.ω, Val(N)),
            Hloc
        )
        for (i, inod) in enumerate(local_nodes)
            Atomix.@atomic :monotonic ∂R∂H[inod] += sum(abs(∂Re∂He[i, j]) for j in 1:N) # sum over ||row||
            Atomix.@atomic :monotonic PC[inod]   += abs(∂Re∂He[i, i])                   # |diagonal|
        end
    end
end

function assemble_diffusion_matrices!(R, ∂R∂H, PC, H, mesh, element::ReferenceElement{T}, D, source, do_∂R∂H) where T<:AbstractElement{2, N} where N

    fill!(R, 0)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
    end
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))

    for iel in 1:mesh.nels
        integrate_residual_element!(R, ∂R∂H, PC, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, do_∂R∂H, Val(N))
    end
end

function assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H, mesh, element::ReferenceElement{T}, D, source, do_∂R∂H) where T<:AbstractElement{2, N} where N

    fill!(R, 0)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
    end
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))

    Threads.@threads :static for iel in 1:mesh.nels
        integrate_residual_atomic!(R, ∂R∂H, PC, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, do_∂R∂H, Val(N))
    end
end

function assemble_diffusion_matrices_colored!(R, ∂R∂H, PC, H, mesh, element::ReferenceElement{T}, D, source, do_∂R∂H, colors) where T<:AbstractElement{2, N} where N

    fill!(R, 0)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
    end
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))

    for color in colors
        Threads.@threads for iel in color
            integrate_residual_element!(R, ∂R∂H, PC, H, source, D, mesh, Nq, ∂N∂ξq, ip, iel, do_∂R∂H, Val(N))
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
    Lx = Ly = 1

    Ω = (-Lx..Lx) × (-Ly..Ly)
    element = ReferenceElement(LinearElement{2, 4, Float64})
    # element = ReferenceElement(QuadraticElement{2, 9, Float64})
    mesh = FEMTools.Mesh(Ω, element, nels)
    colors = color_element_batches(mesh)

    σ      = 0.1                                # Source width
    r²     = [coord[1]^2 + coord[2]^2 for coord in mesh.coords]
    source = 2.0 * exp.(-r² / (2σ^2))           # Source
    HW     = 1.0                                # Dirichlet value west
    HE     = 0.0                                # Dirichlet value east
    epsi   = 1e-9                               # Relative tolerance

    # FEM PT solve
    D            = 1.0
    H_FEM        = zeros(mesh.nnodes)
    R            = zeros(mesh.nnodes)
    R0           = zeros(mesh.nnodes)
    ∂H∂τ         = zeros(mesh.nnodes)
    ∂R∂H         = zeros(mesh.nnodes)
    PC           = zeros(mesh.nnodes)
    nr0          = 0.0
    H_FEM       .= exp.(-r² / (2σ^2))

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
    assemble_diffusion_matrices!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, do_∂R∂H)

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
        @timeit to "series" assemble_diffusion_matrices!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, do_∂R∂H)
        # @timeit to "atomix" assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, do_∂R∂H)
        # @timeit to "colors" assemble_diffusion_matrices_colored!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, do_∂R∂H, colors)

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

    # nodes live on a (p*nx + 1) × (p*ny + 1) tensor grid, p = element order
    nx, ny = nels .* order(element)
    xs = LinRange(-Lx, Lx, nx + 1)
    ys = LinRange(-Ly, Ly, ny + 1)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="2D Poisson PT solution", aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, reshape(H_FEM, nx + 1, ny + 1); colormap=:inferno)
    Colorbar(fig[1, 2], hm)

    fig = scatterlines(xs, (reshape(H_FEM, nx + 1, ny + 1))[:, ny>>>1]; label="converged")

    display(fig)

    return nothing
end

nels = (100, 100) .*2
main(nels)
