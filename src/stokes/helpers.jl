# `backend::KA.Backend` keeps this disjoint from the deprecated leading-backend
# shim below: without it, a 9-argument call matches both methods ambiguously.
"""
    precompute_stokes_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels, backend, workgroup)

Launch the Stokes geometry precompute kernel and synchronize the backend.
"""
precompute_stokes_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels, backend, workgroup) where {N} =
    launch!(precompute_geometry_kernel!, backend, workgroup, nels,
            geo, coords, el2n, ∂N∂ξq, ω, Val(N))

"""
    stokes_update_rate!(∂u∂τ, R, PC, β, ndofs, backend, workgroup)

Advance a pseudo-transient velocity-rate field with diagonal preconditioning.
"""
stokes_update_rate!(∂u∂τ, R, PC, β, ndofs, backend, workgroup) =
    launch!(update_rate_kernel!, backend, workgroup, ndofs, ∂u∂τ, R, PC, β)

"""
    stokes_update_variable!(u, ∂u∂τ, α_dr, ndofs, backend, workgroup)

Apply a damped pseudo-transient increment to a Stokes solution field.
"""
stokes_update_variable!(u, ∂u∂τ, α_dr, ndofs, backend, workgroup) =
    launch!(update_variable_kernel!, backend, workgroup, ndofs, u, ∂u∂τ, α_dr)

@kernel function update_stokes_velocity_kernel!(
        rate_x, rate_y, vx, vy,
        @Const(Rx), @Const(Ry), @Const(PCx), @Const(PCy),
        βx, βy, αx, αy,
    )
    i = @index(Global)
    new_rate_x = βx * rate_x[i] + Rx[i] / PCx[i]
    new_rate_y = βy * rate_y[i] + Ry[i] / PCy[i]
    rate_x[i] = new_rate_x
    rate_y[i] = new_rate_y
    vx[i] += αx * new_rate_x
    vy[i] += αy * new_rate_y
end

update_stokes_velocity!(
        rate_x, rate_y, vx, vy, Rx, Ry, PCx, PCy,
        βx, βy, αx, αy, ndofs, backend, workgroup,
    ) =
    launch!(update_stokes_velocity_kernel!, backend, workgroup, ndofs,
            rate_x, rate_y, vx, vy, Rx, Ry, PCx, PCy, βx, βy, αx, αy)

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
