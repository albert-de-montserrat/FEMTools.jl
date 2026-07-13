"""
    precompute_stokes_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels, backend, workgroup)

Launch the Stokes geometry precompute kernel and synchronize the backend.
"""
# `backend::KA.Backend` keeps this disjoint from the deprecated leading-backend
# shim below: without it, a 9-argument call matches both methods ambiguously.
function precompute_stokes_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels, backend::KA.Backend, workgroup) where N
    precompute_geometry_kernel!(backend, workgroup)(
        geo, coords, el2n, ∂N∂ξq, ω, Val(N);
        ndrange = nels,
    )
    KA.synchronize(backend)
    return nothing
end

function precompute_stokes_geometry!(backend::KA.Backend, workgroup, geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels) where N
    Base.depwarn(
        "`precompute_stokes_geometry!(backend, workgroup, geo, …, nels)` is deprecated; pass `backend, workgroup` last: `precompute_stokes_geometry!(geo, …, nels, backend, workgroup)`.",
        :precompute_stokes_geometry!,
    )
    return precompute_stokes_geometry!(geo, coords, el2n, ∂N∂ξq, ω, Val(N), nels, backend, workgroup)
end

"""
    stokes_update_rate!(∂u∂τ, R, PC, β, ndofs, backend, workgroup)

Advance a pseudo-transient velocity-rate field with diagonal preconditioning.
"""
function stokes_update_rate!(∂u∂τ, R, PC, β, ndofs, backend, workgroup)
    update_rate_kernel!(backend, workgroup)(
        ∂u∂τ, R, PC, β;
        ndrange = ndofs,
    )
    KA.synchronize(backend)
    return nothing
end

function stokes_update_rate!(backend::KA.Backend, workgroup, ∂u∂τ, R, PC, β, ndofs)
    Base.depwarn(
        "`stokes_update_rate!(backend, workgroup, ∂u∂τ, R, PC, β, ndofs)` is deprecated; pass `backend, workgroup` last: `stokes_update_rate!(∂u∂τ, R, PC, β, ndofs, backend, workgroup)`.",
        :stokes_update_rate!,
    )
    return stokes_update_rate!(∂u∂τ, R, PC, β, ndofs, backend, workgroup)
end

"""
    stokes_update_variable!(u, ∂u∂τ, α_dr, ndofs, backend, workgroup)

Apply a damped pseudo-transient increment to a Stokes solution field.
"""
function stokes_update_variable!(u, ∂u∂τ, α_dr, ndofs, backend, workgroup)
    update_variable_kernel!(backend, workgroup)(
        u, ∂u∂τ, α_dr;
        ndrange = ndofs,
    )
    KA.synchronize(backend)
    return nothing
end

function stokes_update_variable!(backend::KA.Backend, workgroup, u, ∂u∂τ, α_dr, ndofs)
    Base.depwarn(
        "`stokes_update_variable!(backend, workgroup, u, ∂u∂τ, α_dr, ndofs)` is deprecated; pass `backend, workgroup` last: `stokes_update_variable!(u, ∂u∂τ, α_dr, ndofs, backend, workgroup)`.",
        :stokes_update_variable!,
    )
    return stokes_update_variable!(u, ∂u∂τ, α_dr, ndofs, backend, workgroup)
end

"""
    remove_pressure_mean!(P, M_P) -> p_mean

Remove the mass-weighted mean pressure and return the removed gauge offset.
"""
function remove_pressure_mean!(P, M_P)
    p_mean = sum(P .* M_P) / sum(M_P)
    @. P -= p_mean
    return p_mean
end

"""
    pressure_mass(dr::StokesDR)

Return the lumped pressure mass field (`M_P`) of a Stokes solver state.
"""
pressure_mass(dr::StokesDR) = dr.M_P

"""
    velocity_mass(dr::StokesDR)

Return the positive velocity mass diagonal (`M_V`) of a Stokes solver state.
"""
velocity_mass(dr::StokesDR) = dr.M_V
