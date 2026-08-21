"""
    rotate_stress!(dr, mesh_stokes, cache, element_v, Δt)
    rotate_stress!(dr, mesh_stokes, geo_v, element_v, Δt)

Advance the deviatoric-stress history by rotating the current stress
`dr.τ` with the local vorticity over the time step `Δt`, writing the result
into the components of `dr.τ_old`.
Unpacks the solver state, connectivity (`mesh_stokes.el2n`), and element
geometry (`cache.geo_v`) for the low-level `_rotate_stress!` worker.
`element_v` supplies the velocity-node count `NV`.
"""
function rotate_stress!(dr, mesh_stokes, cache::MixedMeshCache, element_v, Δt)
    return rotate_stress!(dr, mesh_stokes, cache.geo_v, element_v, Δt)
end

function rotate_stress!(dr, mesh_stokes, geo_v, element_v, Δt)
    return _rotate_stress!(
        (dr.τ_old.xx, dr.τ_old.yy, dr.τ_old.xy),
        (dr.τ.xx, dr.τ.yy, dr.τ.xy),
        dr.v.x, dr.v.y, mesh_stokes.el2n, geo_v, Δt, element_v,
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
