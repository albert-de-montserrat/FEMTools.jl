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
        rate, v, @Const(R), @Const(PC), β, α_dr, ::Val{D},
    ) where {D}
    i = @index(Global)
    ntuple(Val(D)) do c
        new_rate = β[c] * rate[c][i] + R[c][i] / PC[c][i]
        rate[c][i] = new_rate
        v[c][i] += α_dr[c] * new_rate
        nothing
    end
end

"""
    update_stokes_velocity!(rate, v, R, PC, β, α_dr, ndofs, backend, workgroup)
    update_stokes_velocity!(rate_x, rate_y, vx, vy, Rx, Ry, PCx, PCy,
                            βx, βy, αx, αy, ndofs, backend, workgroup)

Advance the Chebyshev velocity recurrence for every spatial direction. The
first form takes one tuple entry per direction; the second takes the two
plane-strain components separately.
"""
update_stokes_velocity!(
        rate::NTuple{D}, v::NTuple{D}, R, PC, β, α_dr, ndofs, backend, workgroup,
    ) where {D} =
    launch!(update_stokes_velocity_kernel!, backend, workgroup, ndofs,
            rate, v, R, PC, β, α_dr, Val(D))

update_stokes_velocity!(
        rate_x, rate_y, vx, vy, Rx, Ry, PCx, PCy,
        βx, βy, αx, αy, ndofs, backend, workgroup,
    ) =
    update_stokes_velocity!((rate_x, rate_y), (vx, vy), (Rx, Ry), (PCx, PCy),
                            (βx, βy), (αx, αy), ndofs, backend, workgroup)

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
