using LinearAlgebra

# ------------------------------------------------------------
# 1D Q2 basis on [-1, 1], nodes at ξ = -1, 0, 1
# ------------------------------------------------------------

@inline function q2_basis_1d(ξ)
    L = zeros(3)
    dL = zeros(3)

    L[1] = 0.5 * ξ * (ξ - 1.0)
    L[2] = 1.0 - ξ^2
    L[3] = 0.5 * ξ * (ξ + 1.0)

    dL[1] = ξ - 0.5
    dL[2] = -2.0 * ξ
    dL[3] = ξ + 0.5

    return L, dL
end

# ------------------------------------------------------------
# 3-point Gauss quadrature
# ------------------------------------------------------------

const ξq = [-sqrt(3 / 5), 0.0, sqrt(3 / 5)]
const wq = [5 / 9, 8 / 9, 5 / 9]

const Lq  = [q2_basis_1d(ξq[q])[1][a] for q in 1:3, a in 1:3]
const dLq = [q2_basis_1d(ξq[q])[2][a] for q in 1:3, a in 1:3]

# ------------------------------------------------------------
# Flattened Q2 node numbering
#
# i, j, k ∈ 1:3
#
# a = 1 + (i-1) + 3(j-1) + 9(k-1)
#
# So:
#
# k = 1 plane: a = 1:9
# k = 2 plane: a = 10:18
# k = 3 plane: a = 19:27
# ------------------------------------------------------------

@inline node_id(i, j, k) = i + 3 * (j - 1) + 9 * (k - 1)

# ------------------------------------------------------------
# Matrix-free flattened Q2 diffusion operator
# ------------------------------------------------------------

function apply_q2_diffusion_flat!(re, ue; κ = 1.0, hx = 1.0, hy = 1.0, hz = 1.0)

    fill!(re, 0.0)

    dξdx = 2.0 / hx
    dηdy = 2.0 / hy
    dζdz = 2.0 / hz

    detJ = hx * hy * hz / 8.0

    # Loop over quadrature points
    for qz in 1:3, qy in 1:3, qx in 1:3

        weight = wq[qx] * wq[qy] * wq[qz] * detJ

        # ----------------------------------------------------
        # Compute ∇u at quadrature point
        # ----------------------------------------------------

        dudx = 0.0
        dudy = 0.0
        dudz = 0.0

        for k in 1:3, j in 1:3, i in 1:3
            a = node_id(i, j, k)

            u = ue[a]

            Nx  = Lq[qx, i]
            Ny  = Lq[qy, j]
            Nz  = Lq[qz, k]

            dNx = dLq[qx, i] * dξdx
            dNy = dLq[qy, j] * dηdy
            dNz = dLq[qz, k] * dζdz

            dNdx = dNx * Ny  * Nz
            dNdy = Nx  * dNy * Nz
            dNdz = Nx  * Ny  * dNz

            dudx += dNdx * u
            dudy += dNdy * u
            dudz += dNdz * u
        end

        # Isotropic scalar diffusion flux
        qx_flux = κ * dudx
        qy_flux = κ * dudy
        qz_flux = κ * dudz

        # ----------------------------------------------------
        # Project flux back to flattened nodal residual
        # ----------------------------------------------------

        for k in 1:3, j in 1:3, i in 1:3
            a = node_id(i, j, k)

            Nx  = Lq[qx, i]
            Ny  = Lq[qy, j]
            Nz  = Lq[qz, k]

            dNx = dLq[qx, i] * dξdx
            dNy = dLq[qy, j] * dηdy
            dNz = dLq[qz, k] * dζdz

            dNdx = dNx * Ny  * Nz
            dNdy = Nx  * dNy * Nz
            dNdz = Nx  * Ny  * dNz

            re[a] += weight * (
                dNdx * qx_flux +
                dNdy * qy_flux +
                dNdz * qz_flux
            )
        end
    end

    return re
end

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

ue = zeros(27)

for k in 1:3, j in 1:3, i in 1:3
    a = node_id(i, j, k)

    x = (i - 1) / 2
    y = (j - 1) / 2
    z = (k - 1) / 2

    ue[a] = sin(pi * x) * cos(pi * y) + z^2
end

re = similar(ue)

apply_q2_diffusion_flat!(re, ue; κ = 2.0, hx = 1.0, hy = 1.0, hz = 1.0)

println("Flattened element vector ue:")
display(ue)

println("Flattened matrix-free residual re = Ke * ue:")
display(re)


#########
using StaticArrays
geo = [(@SMatrix(rand(27,3)), rand()) for q in 1:27, iel in 1:1]

Nq   = [@SVector(rand(27)) for q in 1:27]
Hloc = @SVector rand(27)
sloc = @SVector rand(27)
D    = 1
iel  = 1

@inline function integrate_residual(Hloc, geo, iel, sloc, D, Nq, ::Val{N}) where N
    Re = zero(Hloc)
    for q in eachindex(Nq)
        ∂N∂x, dΩ = geo[q, iel]
        Nv = Nq[q]
        tmp   = ∂N∂x' * Hloc
        KHloc = (D * dΩ) * (∂N∂x * tmp)
        # Re   += SVector{N}(ntuple(i -> KHloc[i], Val(N)))
        Re   += SVector{N}(ntuple(i -> -sloc[i] * Nv[i] * dΩ - KHloc[i], Val(N)))
    end
    return Re
end

@b integrate_residual($(Hloc, geo, iel, sloc, D, Nq, Val(27))...)

@inline function integrate_residual_opt(Hloc, geo, iel, sloc, D, Nq, ::Val{N}) where N
    Re = @MVector zeros(N)
    for q in eachindex(Nq)
        ∂N∂x, dΩ = geo[q, iel]
        Nv = Nq[q]
                
        dudx = 0.0
        dudy = 0.0
        dudz = 0.0
        for i in 1:N
            # interpolate ∇H to integration point
            H = Hloc[i]
            dudx += ∂N∂x[i, 1] * H
            dudy += ∂N∂x[i, 2] * H
            dudz += ∂N∂x[i, 3] * H
            # interpolate source to integration point
        end

        # Isotropic scalar diffusion flux
        qx_flux = D * dudx
        qy_flux = D * dudy
        qz_flux = D * dudz

        for i in 1:N
            Re[i] += dΩ * (
                -Nv[i] * sloc[i] - (∂N∂x[i, 1] * qx_flux + ∂N∂x[i, 2] * qy_flux +  ∂N∂x[i, 3] * qz_flux)
            )
        end
        # Re   += SVector{N}(ntuple(i -> -sloc[i] * Nv[i] * dΩ - KHloc[i], Val(N)))
    end
    return SVector(Re)
end
integrate_residual_opt(Hloc, geo, iel, sloc, D, Nq, Val(27))
integrate_residual_opt0(Hloc, geo, iel, sloc, D, Nq, Val(27))
integrate_residual(Hloc, geo, iel, sloc, D, Nq, Val(27))

@b integrate_residual_opt($(Hloc, geo, iel, sloc, D, Nq, Val(27))...)


@inline function integrate_residual_opt0(Hloc, geo, iel, sloc, D, Nq, ::Val{N}) where N
    Re = @MVector zeros(N)
    
    for q in eachindex(Nq)
        ∂N∂x, dΩ = geo[q, iel]
        Nv = Nq[q]
        # ∂N∂x_T = ∂N∂x'
                
        Base.@nexprs 3 i -> dud_i = 0e0
        for i in 1:N
            # interpolate ∇H to integration point
            H = Hloc[i]
            Base.@nexprs 3 j -> dud_j +=  ∂N∂x[i, j] * H
            # interpolate source to integration point
        end

        # Isotropic scalar diffusion flux
        Base.@nexprs 3 i -> q_flux_i =  D * dud_i

        for i in 1:N
            sum_flux = 0e0
            Base.@nexprs 3 j -> sum_flux += ∂N∂x[i, j] * q_flux_j
            Re[i] += dΩ * (
                -Nv[i] * sloc[i] -  sum_flux
            )
        end
    end
    return SVector(Re)
end

integrate_residual_opt(Hloc, geo, iel, sloc, D, Nq, Val(27))
integrate_residual_opt0(Hloc, geo, iel, sloc, D, Nq, Val(27))


@b integrate_residual_opt0($(Hloc, geo, iel, sloc, D, Nq, Val(27))...)


@generated function integrate_residual_opt2(Hloc, geo, iel, sloc, D, Nq, ::Val{nDim}, ::Val{N}) where {nDim, N}
    quote
        @inline 
        Re = @MVector zeros($N)
        @inbounds for q in eachindex(Nq)
            ∂N∂x, dΩ = geo[q, iel]
            Nv = Nq[q]
                    
            Base.@nexprs $nDim i -> ∂N∂x_i = 0e0
            for i in 1:N
                # interpolate ∇H to integration point
                H = Hloc[i]
                Base.@nexprs $nDim j -> ∂N∂x_j = muladd(∂N∂x[i, j], H, ∂N∂x_j)
            end

            # Isotropic scalar diffusion flux
            Base.@nexprs $nDim i -> q_flux_i =  D * ∂N∂x_i

            for i in 1:N
                sum_flux = 0e0
                Base.@nexprs 3 j -> sum_flux += ∂N∂x[i, j] * q_flux_j
                Re[i] = muladd(
                    dΩ,
                    -(Nv[i], sloc[i]) - sum_flux,
                    Re[i]
                )
            end
        end
        return SVector(Re)
    end
end

@b integrate_residual_opt2($(Hloc, geo, iel, sloc, D, Nq,  Val(3), Val(27))...)
