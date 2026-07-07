import Base.getindex

abstract type AbstractTensor end

struct SymmetricTensor2D{T} <: AbstractTensor
    xx::T
    yy::T
    xy::T
    II::T
end

struct SymmetricTensor3D{T} <: AbstractTensor
    xx::T
    yy::T
    zz::T
    yz::T
    xz::T
    xy::T
    II::T
end

@inline SymmetricTensor(xx, yy, xy)             = SymmetricTensor2D(xx, yy, xy, zero(xx))
@inline SymmetricTensor(xx, yy, zz, yz, xz, xy) = SymmetricTensor3D(xx, yy, zz, yz, xz, xy, zero(xx))

# Unpack the independent in-plane components `(xx, yy, xy)` (dropping the derived
# `II`) so the struct plugs straight into assemblers that dispatch on `NTuple{3}`.
@inline Base.Tuple(τ::SymmetricTensor2D) = (τ.xx, τ.yy, τ.xy)

# In-place component-wise copy, e.g. `copyto!(τ_old, τ)` to advance stress history.
function Base.copyto!(dst::SymmetricTensor2D, src::SymmetricTensor2D)
    copyto!(dst.xx, src.xx)
    copyto!(dst.yy, src.yy)
    copyto!(dst.xy, src.xy)
    return dst
end

# Integration-point components are stored as `nq × nels` matrices, so the Voigt
# accessor is indexed `[q, iel]` (quadrature point, element).
Base.getindex(A::SymmetricTensor2D, q::Int, iel::Int) = SA[
    A.xx[q, iel] A.yy[q, iel] A.xy[q, iel]
]
Base.getindex(A::SymmetricTensor3D, q::Int, iel::Int) = SA[
    A.xx[q, iel] A.yy[q, iel] A.zz[q, iel] A.yz[q, iel] A.xz[q, iel] A.xy[q, iel]
]


"""
    rotate_stress!(dr, mesh_stokes, cache, element_v, Δt)
    rotate_stress!(dr, mesh_stokes, geo_v, element_v, Δt)

Advance the deviatoric-stress history by rotating the current stress `dr.τ` with
the local vorticity over the time step `Δt`, writing the result into `dr.τ_old`.
Unpacks the solver state, connectivity (`mesh_stokes.el2n`), and element
geometry (`cache.geo_v`) for the low-level `_rotate_stress!` worker.
`element_v` supplies the velocity-node count `NV`.
"""
function rotate_stress!(dr, mesh_stokes, cache::MixedMeshCache, element_v, Δt)
    return rotate_stress!(dr, mesh_stokes, cache.geo_v, element_v, Δt)
end

function rotate_stress!(dr, mesh_stokes, geo_v, element_v, Δt)
    return _rotate_stress!(
        (dr.τxx_old, dr.τyy_old, dr.τxy_old),
        (dr.τxx, dr.τyy, dr.τxy),
        dr.vx, dr.vy, mesh_stokes.el2n, geo_v, Δt, element_v,
    )
end

function _rotate_stress!(
    τ_old,
    τ,
    vx, vy,
    el2n_v,
    geo_v,
    dt,
    ::ReferenceElement{TV},
) where {NV, TV <: AbstractElement{2, NV}}
    nels = size(el2n_v, 2)

    for iel in 1:nels
        local_nodes = SVector{NV}(ntuple(i -> el2n_v[i, iel], Val(NV)))
        vxloc = SVector{NV}(ntuple(i -> vx[local_nodes[i]], Val(NV)))
        vyloc = SVector{NV}(ntuple(i -> vy[local_nodes[i]], Val(NV)))
        geo_el = geo_v[iel]

        for q in eachindex(geo_el)
            ∂N∂x, = geo_el[q]
            # velocity gradients
            ∇vx = ∂N∂x' * vxloc
            ∇vy = ∂N∂x' * vyloc
            # vorticity ω = ½(∂vx/∂y − ∂vy/∂x)
            ωxy_q = (∇vx[2] - ∇vy[1]) / 2
            # rotate the current stress and store it as the old-stress history
            τxx_q, τyy_q, τxy_q = rotate_stress_tensor(τ, ωxy_q, dt, q, iel)
            τ_old[1][q, iel] = τxx_q
            τ_old[2][q, iel] = τyy_q
            τ_old[3][q, iel] = τxy_q
        end
    end

    return nothing
end

# Rigid-body rotation of the deviatoric stress at integration point (q, iel).
@inline function rotate_stress_tensor(τ, ω, dt, q::Int, iel::Int)
    τxx = τ[1][q, iel]
    τyy = τ[2][q, iel]
    τxy = τ[3][q, iel]
    sinθ, cosθ = sincos(ω * dt)
    return (
        cosθ^2 * τxx - 2 * sinθ * cosθ * τxy + sinθ^2 * τyy,
        sinθ^2 * τxx + 2 * sinθ * cosθ * τxy + cosθ^2 * τyy,
        sinθ * cosθ * (τxx - τyy) + (cosθ^2 - sinθ^2) * τxy,
    )
end
