import Base.getindex

"""
    AbstractTensor

Abstract supertype for symmetric-tensor containers.
"""
abstract type AbstractTensor end

"""
    SymmetricTensor2D{T}

Two-dimensional symmetric second-order tensor storing the independent
components `xx`, `yy`, `xy` and an invariant slot `II`.

Each component may be a scalar or an `nq × nels` matrix of integration-point
values. In the matrix case, `A[q, iel]` returns the Voigt component row vector
at quadrature point `q` of element `iel`.
"""
struct SymmetricTensor2D{T} <: AbstractTensor
    xx::T
    yy::T
    xy::T
    II::T
end

"""
    SymmetricTensor3D{T}

Three-dimensional symmetric second-order tensor storing the independent
components `xx`, `yy`, `zz`, `yz`, `xz`, `xy` and an invariant slot `II`.

Component storage and `A[q, iel]` indexing follow [`SymmetricTensor2D`](@ref).
"""
struct SymmetricTensor3D{T} <: AbstractTensor
    xx::T
    yy::T
    zz::T
    yz::T
    xz::T
    xy::T
    II::T
end

"""
    SymmetricTensor(xx, yy, xy) -> SymmetricTensor2D
    SymmetricTensor(xx, yy, zz, yz, xz, xy) -> SymmetricTensor3D

Construct a symmetric tensor from its independent components, initializing the
invariant slot `II` to zero.
"""
@inline SymmetricTensor(xx, yy, xy)             = SymmetricTensor2D(xx, yy, xy, zero(xx))
@inline SymmetricTensor(xx, yy, zz, yz, xz, xy) = SymmetricTensor3D(xx, yy, zz, yz, xz, xy, zero(xx))

_show_component(x) = x isa Number ? repr(x) : summary(x)

function Base.show(io::IO, τ::SymmetricTensor2D)
    print(io, "SymmetricTensor2D(xx=", _show_component(τ.xx),
          ", yy=", _show_component(τ.yy),
          ", xy=", _show_component(τ.xy),
          ", II=", _show_component(τ.II), ")")
end

function Base.show(io::IO, τ::SymmetricTensor3D)
    print(io, "SymmetricTensor3D(xx=", _show_component(τ.xx),
          ", yy=", _show_component(τ.yy),
          ", zz=", _show_component(τ.zz),
          ", yz=", _show_component(τ.yz),
          ", xz=", _show_component(τ.xz),
          ", xy=", _show_component(τ.xy),
          ", II=", _show_component(τ.II), ")")
end

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

Advance the deviatoric-stress history by rotating the current stress
(`dr.τxx`, `dr.τyy`, `dr.τxy`) with the local vorticity over the time step
`Δt`, writing the result into the corresponding `dr.τ*_old` fields.
Unpacks the solver state, connectivity (`mesh_stokes.el2n`), and element
geometry (`cache.geo_v`) for a KernelAbstractions launch on the state array's
backend. `element_v` supplies the velocity-node count `NV`.
"""
function rotate_stress!(dr, mesh_stokes, cache::MixedMeshCache, element_v, Δt)
    return rotate_stress!(dr, mesh_stokes, cache.geo_v, element_v, Δt)
end

function rotate_stress!(dr, mesh_stokes, geo_v, element_v, Δt)
    backend = KA.get_backend(dr.vx)
    rotate_stress_kernel!(backend)(
        dr.τxx_old, dr.τyy_old, dr.τxy_old,
        dr.τxx, dr.τyy, dr.τxy,
        dr.vx, dr.vy, mesh_stokes.el2n, geo_v, Δt, Val(length(element_v));
        ndrange = mesh_stokes.nels,
    )
    KA.synchronize(backend)
    return nothing
end

@kernel function rotate_stress_kernel!(
    τxx_old, τyy_old, τxy_old,
    @Const(τxx), @Const(τyy), @Const(τxy),
    @Const(vx), @Const(vy), @Const(el2n_v), @Const(geo_v), dt,
    ::Val{NV},
) where {NV}
    iel = @index(Global)
    local_nodes = SVector{NV}(ntuple(i -> el2n_v[i, iel], Val(NV)))
    vxloc = SVector{NV}(ntuple(i -> vx[local_nodes[i]], Val(NV)))
    vyloc = SVector{NV}(ntuple(i -> vy[local_nodes[i]], Val(NV)))
    geo_el = geo_v[iel]

    for q in eachindex(geo_el)
        ∂N∂x, = geo_el[q]
        ∇vx = ∂N∂x' * vxloc
        ∇vy = ∂N∂x' * vyloc
        ωxy_q = (∇vx[2] - ∇vy[1]) / 2
        sinθ, cosθ = sincos(ωxy_q * dt)
        τxx_q = τxx[q, iel]
        τyy_q = τyy[q, iel]
        τxy_q = τxy[q, iel]
        τxx_old[q, iel] = cosθ^2 * τxx_q - 2 * sinθ * cosθ * τxy_q + sinθ^2 * τyy_q
        τyy_old[q, iel] = sinθ^2 * τxx_q + 2 * sinθ * cosθ * τxy_q + cosθ^2 * τyy_q
        τxy_old[q, iel] = sinθ * cosθ * (τxx_q - τyy_q) + (cosθ^2 - sinθ^2) * τxy_q
    end
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
        local_nodes = local_nodes_of(el2n_v, iel, Val(NV))
        vxloc = _gather_local(vx, local_nodes, Val(NV))
        vyloc = _gather_local(vy, local_nodes, Val(NV))
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
