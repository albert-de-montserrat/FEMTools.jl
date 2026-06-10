using ForwardDiff
using LinearAlgebra
using StaticArrays
using Printf
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

function integrate_residual(Hloc, coords, sloc, D, Nq, ∂N∂ξq, ip, ::Val{N}) where N
    Re = zero(Hloc)
    for q in eachindex(ip.ω)
        Nv = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ' * coords
        ∂N∂x = ∂N∂ξ * inv(J)
        dΩ = abs(det(J)) * ip.ω[q]

        tmp   = ∂N∂x' * Hloc
        KHloc = (D * dΩ) * (∂N∂x * tmp)
        Re   += SVector{N}(-sloc[i] * Nv[i] * dΩ - KHloc[i] for i in 1:N)
    end
    return Re
end

function assemble_diffusion_matrices!(R, ∂R∂H, PC, H, mesh, element::ReferenceElement{T}, D, source, do_∂R∂H) where T<:AbstractElement{3, N} where N

    fill!(R, 0)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
    end
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q], ip.ζ[q]), length(ip.ω))
    Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))

    for iel in 1:mesh.nels
        local_nodes = SVector{N, Int}(ntuple(i -> mesh.el2n[i, iel], Val(N)))
        coords = element_coordinate_matrix(mesh, local_nodes)
        Hloc   = SVector{N}(H[local_nodes[i]] for i in 1:N)
        sloc   = SVector{N}(source[local_nodes[i]] for i in 1:N)
        Re     = integrate_residual(Hloc, coords, sloc, D, Nq, ∂N∂ξq, ip, Val(N))
        for (i, inod) in enumerate(local_nodes)
            R[inod] += Re[i]
        end
        if do_∂R∂H
            ∂Re∂He = ForwardDiff.jacobian(
                Hloc -> integrate_residual(Hloc, coords, sloc, D, Nq, ∂N∂ξq, ip, Val(N)),
                Hloc
            )
            for (i, inod) in enumerate(local_nodes)
                ∂R∂H[inod] += sum(abs(∂Re∂He[i, j]) for j in 1:N) # sum over ||row||
                PC[inod]   += abs(∂Re∂He[i, i])                   # |diagonal|
            end
        end
    end
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
    mesh = FEMTools.Mesh(Ω, element, nels)
    Γnodes = mesh.Γnodes

    σ      = 0.1                                # Source width
    r²     = [coord[1]^2 + coord[2]^2 + coord[3]^2 for coord in mesh.coords]
    source = 2.0 * exp.(-r² / (2σ^2))           # Source
    Hbc    = 0.0                                # Dirichlet value on Γ
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
    H_FEM[Γnodes] .= Hbc

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
    ncheck = 100
    for it = 1:100_000
        do_∂R∂H = if mod(it, ncheck) == 0
            copyto!(R0, R)
            true
        else
            false
        end
        assemble_diffusion_matrices!(R, ∂R∂H, PC, H_FEM, mesh, element, D, source, do_∂R∂H)

        # Dirichlet BCs: constrain residual and rate *before* the update,
        # otherwise the (nonzero) reaction-force residual at the boundary
        # nodes accumulates into ∂H∂τ and poisons the λmin estimate
        R[Γnodes]    .= 0e0
        ∂H∂τ[Γnodes] .= 0e0

        update_rate!(∂H∂τ, R, PC, β)
        update_variable!(H_FEM, ∂H∂τ, α)

        H_FEM[Γnodes] .= Hbc

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

    # mid-plane slice at z = 0
    nx, ny, nz = nels
    xs = LinRange(-Lx, Lx, nx + 1)
    ys = LinRange(-Ly, Ly, ny + 1)
    H_grid = reshape(H_FEM, nx + 1, ny + 1, nz + 1)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="3D Poisson PT solution (z = 0 slice)", aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, H_grid[:, :, nz ÷ 2 + 1]; colormap=:inferno)
    Colorbar(fig[1, 2], hm)
    display(fig)

    return mesh, H_FEM
end

nels = (32, 32, 32)
main(nels)
