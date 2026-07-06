"""
    precompute_stokes_geometry!(backend, workgroup, geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels)

Launch the Stokes geometry precompute kernel and synchronize the backend.
"""
function precompute_stokes_geometry!(backend, workgroup, geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels) where N
    precompute_geometry_kernel!(backend, workgroup)(
        geo, coords, el2n, ∂N∂ξq, ω, Val(N);
        ndrange = nels,
    )
    KA.synchronize(backend)
    return nothing
end

"""
    stokes_update_rate!(backend, workgroup, ∂u∂τ, R, PC, β, ndofs)

Advance a pseudo-transient velocity-rate field with diagonal preconditioning.
"""
function stokes_update_rate!(backend, workgroup, ∂u∂τ, R, PC, β, ndofs)
    update_rate_kernel!(backend, workgroup)(
        ∂u∂τ, R, PC, β;
        ndrange = ndofs,
    )
    KA.synchronize(backend)
    return nothing
end

"""
    stokes_update_variable!(backend, workgroup, u, ∂u∂τ, α_dr, ndofs)

Apply a damped pseudo-transient increment to a Stokes solution field.
"""
function stokes_update_variable!(backend, workgroup, u, ∂u∂τ, α_dr, ndofs)
    update_variable_kernel!(backend, workgroup)(
        u, ∂u∂τ, α_dr;
        ndrange = ndofs,
    )
    KA.synchronize(backend)
    return nothing
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
    pressure_mass(dr)

Return the pressure mass/scaling field stored on a Stokes solver state.
"""
pressure_mass(dr) = hasproperty(dr, :M_P) ? getproperty(dr, :M_P) : getproperty(dr, :PC_P)
