using LinearAlgebra
using SparseArrays
using StaticArrays


using DomainSets
using DomainSets: ×
using GLMakie
using FEMTools

function gaussian_temperature(x, y, t; κ=1.0, center=(0.5, 0.5), σ=0.08, amplitude=10.0)
    width² = σ^2 + 4κ * t
    dx = x - center[1]
    dy = y - center[2]
    return amplitude * σ^2 / width² * exp(-(dx^2 + dy^2) / width²)
end

function preallocate_sparse_matrix(pattern)
    A = Float64.(pattern)
    fill!(nonzeros(A), 0.0)
    return A
end

function element_coordinate_matrix(mesh, local_nodes::SVector{N, Int}) where {N}
    data = ntuple(Val(2N)) do k
        col = cld(k, N)
        row = k - (col - 1) * N
        mesh.coords[local_nodes[row]][col]
    end
    return SMatrix{N, 2, Float64, 2N}(data)
end

function assemble_diffusion_Kloc(coords, ip, ∂N∂ξq, ::ReferenceElement{T}, κ) where T<:AbstractElement{2, N} where N
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

function assemble_diffusion_Mloc(coords, ip, Nq, ∂N∂ξq, ::ReferenceElement{T}) where T<:AbstractElement{2, N} where N
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

function assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, ::ReferenceElement{T}, source) where T<:AbstractElement{2, N} where N
    Floc = @SVector zeros(N)

    for q in eachindex(ip.ω)
        Nv = Nq[q]
        ∂N∂ξ = ∂N∂ξq[q]
        J = ∂N∂ξ' * coords
        dΩ = abs(det(J)) * ip.ω[q]
        # xq = sum(Nv[i] * coords[i, 1] for i in 1:N)
        # yq = sum(Nv[i] * coords[i, 2] for i in 1:N)
        # source_value = source isa Function ? source(xq, yq) : source
        Floc += Nv * source * dΩ
    end

    return Floc
end

function assemble_diffusion_matrices!(K, M, F, mesh, element::ReferenceElement{T}, κ, source) where T<:AbstractElement{2, N} where N
    ip = element.integration_points
    ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
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

Lx=1.0
Ly=1.0
nels=(40, 40)
κ=1.0e-2

element = ReferenceElement(LinearElement{2, 4, Float64})
Ω = (0.0..Lx) × (0.0..Ly)
mesh = FEMTools.Mesh(Ω, element, nels)
source = 0.0

pattern = generate_sparsity_pattern(mesh)
K = preallocate_sparse_matrix(pattern)
M = preallocate_sparse_matrix(pattern)
F = zeros(mesh.nnodes)

# assemble_diffusion_matrices!(K, M, F, mesh, element, κ, source)
N = 4
ip = element.integration_points
ξq = ntuple(q -> (ip.ξ[q], ip.η[q]), length(ip.ω))
Nq = ntuple(q -> eval_shape_function(element, ξq[q]), length(ip.ω))
∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), length(ip.ω))

iel =1
local_nodes = SVector{N, Int}(ntuple(i -> mesh.el2n[i, iel], Val(N)))
coords = element_coordinate_matrix(mesh, local_nodes)

K_e  = assemble_diffusion_Kloc(coords, ip, ∂N∂ξq, element, κ)
M_e  = assemble_diffusion_Mloc(coords, ip, Nq, ∂N∂ξq, element)
F_e  = assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, element, source)
T_e  = assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, element, 1)
T0_e = assemble_diffusion_Floc(coords, ip, Nq, ∂N∂ξq, element, 2)

R_r = (-K_e + M_e) * T_e - (F_e  + T0_e)

R_r = (-K_e) * T_e # - (F_e  + T0_e)

q = 1

∂N∂ξ = ∂N∂ξq[q]
Nv = Nq[q]
J = ∂N∂ξ' * coords
dΩ = abs(det(J)) * ip.ω[q]
invJ =inv(J)
∂N∂x = ∂N∂ξ * invJ

Kloc = κ * (∂N∂x * ∂N∂x') * dΩ
Ke_ij = SVector{4}((κ * (∂N∂x[i, :] ⋅ ∂N∂x[j, :]) * dΩ)   for j in 1:4)

Mloc = (Nv * Nv') * dΩ
Me_ij = SVector{4}(((Nv[i] * Nv[j])) * dΩ for j in 1:4)

(Ke_ij .+ Me_ij) 

bar(K_e, M_e, T_e, Nv, F_e, T0_e) = (-K_e .+ M_e) * T_e - Nv .* (F_e  .+ T0_e)

@inline function compute_Re(Nv::SVector{M}, ∂N∂x::SMatrix{M, N}, κ, T_e, T0_e, F_e, dΩ, i::Int) where {M,N}
    # Ke_i  = SVector{M}((κ * (∂N∂x[i, :] ⋅ ∂N∂x[j, :]) * dΩ) * T_e[j]  for j in 1:M)
    # Me_i  = SVector{M}(((Nv[i] ⋅ Nv[j]) * dΩ) * T_e[j] for j in 1:M)
    # T0e_i = T0_e[i]
    # Fe_i  = F_e[i]

    # R_e = (-sum(Ke_i) + sum(Me_i)) - Nv[i] * (Fe_i  + T0e_i) * dΩ

    Ke_i  = SVector{M}(((∂N∂x[i, :] ⋅ ∂N∂x[j, :]) ) * T_e[j]  for j in 1:M)
    Me_i  = SVector{M}(((Nv[i] ⋅ Nv[j]) ) * T_e[j] for j in 1:M)
    T0e_i = T0_e[i]
    Fe_i  = F_e[i]
    R_e   = ((-κ * sum(Ke_i) + sum(Me_i))  - Nv[i] * (Fe_i  + T0e_i)) * dΩ

    return R_e
end

@inline compute_Re_ip(Nv::SVector{M}, ∂N∂x::SMatrix{M, N}, κ, T_e, T0_e, F_e, dΩ) where {M,N} = SVector{M}(compute_Re(Nv, ∂N∂x, κ, T_e, T0_e, F_e, dΩ, i) for i in 1:M)

function integrate_Re(Nq, ∂N∂ξq, coords, ip, κ, T_e, T0_e, F_e, dΩ)

    Re = @SVector zeros(4)
    for q in eachindex(Nq)
        ∂N∂ξ = ∂N∂ξq[q]
        Nv = Nq[q]
        J = ∂N∂ξ' * coords
        dΩ = abs(det(J)) * ip.ω[q]
        invJ =inv(J)
        ∂N∂x = ∂N∂ξ * invJ

        Re += compute_Re_ip(Nv, ∂N∂x, κ, T_e, T0_e, F_e, dΩ)
    end
    Re
end

R = zeros(4) 
function integrate_Re!(R, Nq, ∂N∂ξq, coords, ip, κ, T_e, T0_e, F_e, dΩ)

    fill!(R, 0)
    for q in eachindex(Nq)
        ∂N∂ξ = ∂N∂ξq[q]
        Nv = Nq[q]
        J = ∂N∂ξ' * coords
        dΩ = abs(det(J)) * ip.ω[q]
        invJ =inv(J)
        ∂N∂x = ∂N∂ξ * invJ

        # Re += compute_Re_ip(Nv, ∂N∂x, κ, T_e, T0_e, F_e, dΩ)
        # R .+= compute_Re_ip(Nv, ∂N∂x, κ, T_e, T0_e, F_e, dΩ)

        for qq in eachindex(Nv)
            R[qq] += compute_Re(Nv, ∂N∂x, κ, T_e, T0_e, F_e, dΩ, qq)
        end
    end
    # Re
end

@b integrate_Re($(Nq, ∂N∂ξq, coords, ip, κ, T_e, T0_e, F_e, dΩ))
@code_warntype integrate_Re(Nq, ∂N∂ξq, coords, ip, κ, T_e, T0_e, F_e, dΩ)
# compute_Re_ip($(Nv, ∂N∂x, κ, T_e, T0_e, F_e, dΩ)...)
