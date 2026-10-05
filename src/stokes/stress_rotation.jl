"""
    rotate_stress!(dr, mesh_stokes::MixedMesh, Δt)
    rotate_stress!(dr, mesh_stokes, cache, element_v, Δt)
    rotate_stress!(dr, mesh_stokes, geo_v, element_v, Δt; workgroup=256)

Advance the deviatoric-stress history by rotating the current stress
`dr.τ` with the local vorticity over the time step `Δt`, writing the result
into the components of `dr.τ_old`. Works in 2-D and 3-D; the per-point
rotation is `GeoParams.rotate_elastic_stress`, with its vorticity conventions:
`ω = ½(∂vx/∂y − ∂vy/∂x)` in 2-D and the full curl of `v` in 3-D. For a plane
flow the two rotate in opposite senses; only the 3-D form is the Jaumann
co-rotation.
`element_v` supplies the velocity-node count. The `MixedMesh` form takes the
geometry and element from `mesh_stokes.geometry`.
"""
function rotate_stress!(dr, mesh_stokes::MixedMesh, Δt)
    cache = _mesh_geometry(mesh_stokes)
    return rotate_stress!(dr, mesh_stokes, cache, cache.element_v, Δt)
end

function rotate_stress!(dr, mesh_stokes, cache::MixedMeshCache, element_v, Δt)
    return rotate_stress!(dr, mesh_stokes, cache.geo_v, element_v, Δt)
end

function rotate_stress!(dr, mesh_stokes, geo_v, element_v, Δt; workgroup = 256)
    ∂N∂ξ_v = shape_function_gradients(element_v)
    return launch!(
        _rotate_stress_kernel!, KA.get_backend(dr.v.x), workgroup, size(mesh_stokes.el2n, 2),
        _stress_components(dr.τ_old), _stress_components(dr.τ), Tuple(dr.v),
        mesh_stokes.el2n, geo_v, ∂N∂ξ_v, Δt, _node_count(element_v),
    )
end

_node_count(::ReferenceElement{<:AbstractElement{<:Any, NV}}) where {NV} = Val(NV)

# Components in GeoParams' Voigt order: (xx, yy, xy) and (xx, yy, zz, yz, xz, xy).
_stress_components(τ::SymmetricTensor2D) = (τ.xx, τ.yy, τ.xy)
_stress_components(τ::SymmetricTensor3D) = (τ.xx, τ.yy, τ.zz, τ.yz, τ.xz, τ.xy)

# Vorticity in the form GeoParams expects: a scalar in 2-D, the curl in 3-D.
@inline _vorticity((∇vx, ∇vy)::NTuple{2}) = (∇vx[2] - ∇vy[1]) / 2
@inline _vorticity((∇vx, ∇vy, ∇vz)::NTuple{3}) =
    (∇vz[2] - ∇vy[3], ∇vx[3] - ∇vz[1], ∇vy[1] - ∇vx[2])

@kernel function _rotate_stress_kernel!(
        τ_old, @Const(τ), @Const(v), @Const(el2n_v), @Const(geo_v), ∂N∂ξ_v, dt, ::Val{NV},
    ) where {NV}
    iel = @index(Global)
    local_nodes = local_nodes_of(el2n_v, iel, Val(NV))
    vloc = map(vc -> _gather_local(vc, local_nodes, Val(NV)), v)
    geo_el = element_geometry(geo_v, iel, ∂N∂ξ_v)
    for q in eachindex(geo_el)
        ∂N∂x, = geo_el[q]
        ω = _vorticity(map(vc -> ∂N∂x' * vc, vloc))
        τq = map(c -> c[q, iel], τ)
        rotated = GeoParams.rotate_elastic_stress(ω, τq, dt)
        for c in eachindex(τ_old)
            τ_old[c][q, iel] = rotated[c]
        end
    end
end
