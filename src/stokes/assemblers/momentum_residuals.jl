dot_or_zero(a, ::Nothing) = zero(eltype(a))
dot_or_zero(a, b) = dot(a, b)
@inline pressure_scale(γ_eff::Number, RP, MP) = γ_eff * RP ./ MP
@inline pressure_scale(γ_eff::SVector, RP, MP) = γ_eff .* RP ./ MP
@inline effective_viscosity(η, G, Δt) = inv(inv(η) + inv(G * Δt))
@inline function viscoelastic_coefficients_phase(Nv, η, G, phase_loc, Δt)
    ηq  = interp2ip_phase(Nv, η, phase_loc)
    # Interpolate compliance so G=Inf stays finite at quadratic IPs.
    invGq = interp2ip_phase(Nv, map(inv, G), phase_loc)
    ηve = inv(inv(ηq) + invGq / Δt)
    inv_2Gdt = invGq / (2 * Δt)
    return ηve, inv_2Gdt
end
@inline effective_viscosity_phase(Nv, η, G, phase_loc, Δt) =
    first(viscoelastic_coefficients_phase(Nv, η, G, phase_loc, Δt))
@inline zero_old_stress(::Type{T}) where T = (zero(T), zero(T), zero(T))
@inline old_stress_component_at_ip(_, τ::Number) = τ
@inline old_stress_component_at_ip(Nv, τ) = dot(Nv, τ)
@inline old_stress_at_ip(_, ::Nothing, ::Type{T}) where T = zero_old_stress(T)
@inline function old_stress_at_ip(Nv, τ_old::NTuple{3}, ::Type)
    return (
        old_stress_component_at_ip(Nv, τ_old[1]),
        old_stress_component_at_ip(Nv, τ_old[2]),
        old_stress_component_at_ip(Nv, τ_old[3]),
    )
end

@inline second_invariant(axx, ayy, axy) = second_invariant(tuple(axx, ayy, axy))

@inline function second_invariant(A::T) where {T <: Union{SVector{3}, NTuple{3}}}
    # Include the third diagonal component Tzz = -Txx - Tyy for 2D plane strain
    Azz = -A[1] - A[2]
    return √((A[1]^2 + A[2]^2 + Azz^2) / 2 + (A[3]^2))
end

# Pure viscoelastic stress — no yield criterion.
@inline function deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old)
    vxloc, vyloc = v
    ∇vx = ∂N∂x' * vxloc
    ∇vy = ∂N∂x' * vyloc

    εxx = ∇vx[1]
    εyy = ∇vy[2]
    εxy = (∇vx[2] + ∇vy[1]) / 2
    tr  = (εxx + εyy) / 3

    ηve, inv_2Gdt = viscoelastic_coefficients_phase(Nv, η, G, phase_loc, Δt)
    τxx_o, τyy_o, τxy_o = τ_old

    τxx = 2 * ηve * ((εxx - tr) + τxx_o * inv_2Gdt)
    τyy = 2 * ηve * ((εyy - tr) + τyy_o * inv_2Gdt)
    τxy = 2 * ηve * (εxy + τxy_o * inv_2Gdt)
    return τxx, τyy, τxy
end

# Dispatch: no plasticity when plastic===nothing.
@inline deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, _, ::Nothing) =
    deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old)

# Viscoelastic–Drucker-Prager plastic stress.
@inline function deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic::DruckerPrager)
    vxloc, vyloc = v
    ∇vx = ∂N∂x' * vxloc
    ∇vy = ∂N∂x' * vyloc

    εxx = ∇vx[1]
    εyy = ∇vy[2]
    εxy = (∇vx[2] + ∇vy[1]) / 2
    tr  = (εxx + εyy) / 3

    ηve, inv_2Gdt = viscoelastic_coefficients_phase(Nv, η, G, phase_loc, Δt)
    τxx_o, τyy_o, τxy_o = τ_old

    τxx = 2 * ηve * ((εxx - tr) + τxx_o * inv_2Gdt)
    τyy = 2 * ηve * ((εyy - tr) + τyy_o * inv_2Gdt)
    τxy = 2 * ηve * (εxy + τxy_o * inv_2Gdt)
    τij = τxx, τyy, τxy

    # Interpolate per-phase plastic parameters to the quadrature point.
    cosϕ  = interp2ip_phase(Nv, plastic.cosϕ,  phase_loc)
    sinϕ  = interp2ip_phase(Nv, plastic.sinϕ,  phase_loc)
    sinΨ  = interp2ip_phase(Nv, plastic.sinΨ,  phase_loc)
    C     = interp2ip_phase(Nv, plastic.C,      phase_loc)
    η_reg = interp2ip_phase(Nv, plastic.η_reg,  phase_loc)
    Kb    = interp2ip_phase(Nv, plastic.Kb,     phase_loc)

    # Drucker-Prager yield function.
    # second_invariant returns τxx²+τyy²+τzz²+2τxy² = 2J₂, so τII = sqrt(J₂) = sqrt(SI/2).
    τII      = second_invariant(τij)
    τII_safe = max(τII, eps(typeof(τII)))
    F        = τII - cosϕ * C - sinϕ * Pq
    ∂F∂P     = sinϕ

    # Derivatives of plastic potential Q (associated: Ψ = ϕ; non-associated: Ψ ≠ ϕ).
    ∂Q∂τxx = τxx / τII_safe * 0.5
    ∂Q∂τyy = τyy / τII_safe * 0.5
    ∂Q∂τxz = τxy / τII_safe
    ∂Q∂τ   = ∂Q∂τxx, ∂Q∂τyy, ∂Q∂τxz
    ∂Q∂P   = sinΨ

    λ = if F > 0
        F / (ηve + η_reg + Kb * Δt * ∂Q∂P * ∂F∂P)
    else
        zero(F)
    end

    τij = if λ > 0
        map((τ, ∂q) -> τ - 2 * ηve * λ * ∂q, τij, ∂Q∂τ)
    else
        τij
    end

    return τij
end

"""
    integrate_momentum_residual(v, P_loc, geo_v_el, phase_loc, η, G, Δt, Nq, NqP) -> (Rv_x, Rv_y)

Integrate the element momentum residual for a 2-D incompressible Stokes flow.

`v` is an `NTuple{2}` of element velocity vectors `(vxloc, vyloc)`, each of
length `N` (number of velocity nodes). `P_loc` holds the `NP` element pressure
values. `Nq` contains velocity shape-function values at each velocity
quadrature point; `NqP` contains pressure shape-function values at those same
points, used to interpolate pressure into the velocity quadrature points.

Weak form per node `i`:

    Rᵢˣ = ∫ (∂Nᵢ/∂x·(τxx − P) + ∂Nᵢ/∂y·τxy) dΩ
    Rᵢʸ = ∫ (∂Nᵢ/∂y·(τyy − P) + ∂Nᵢ/∂x·τxy) dΩ
"""
@inline function integrate_momentum_residual(
    v::NTuple{2, SVector{N, T}},
    P_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, Δt,
    Nq,
    NqP,
) where {N, T, NP}
    return integrate_momentum_residual(
        v, P_loc, geo_v_el, phase_loc, η, G, Δt, nothing, nothing, Nq, NqP,
    )
end

@inline function integrate_momentum_residual(
    v::NTuple{2, SVector{N, T}},
    P_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, Δt,
    τ_old,
    Nq,
    NqP,
) where {N, T, NP}
    return integrate_momentum_residual(
        v, P_loc, geo_v_el, phase_loc, η, G, Δt, τ_old, nothing, Nq, NqP,
    )
end

@inline function integrate_momentum_residual(
    v::NTuple{2, SVector{N, T}},
    P_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, Δt,
    τ_old,
    plastic,
    Nq,
    NqP,
) where {N, T, NP}
    Rv_x = zero(SVector{N, T})
    Rv_y = zero(SVector{N, T})
    for q in eachindex(geo_v_el)
        ∂N∂x, dΩ = geo_v_el[q]
        Nv = Nq[q]
        τ_old_q = old_stress_at_ip(Nv, τ_old, T)
        Pq  = dot(NqP[q], P_loc)
        τxx, τyy, τxy = deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old_q, Pq, plastic)
        # x-momentum: ∫ (∂Nᵢ/∂x·(τxx−P) + ∂Nᵢ/∂y·τxy) dΩ
        Rv_x += (∂N∂x[:, 1] * (τxx - Pq) + ∂N∂x[:, 2] * τxy) * dΩ
        # y-momentum: ∫ (∂Nᵢ/∂y·(τyy−P) + ∂Nᵢ/∂x·τxy) dΩ
        Rv_y += (∂N∂x[:, 2] * (τyy - Pq) + ∂N∂x[:, 1] * τxy) * dΩ
    end
    return Rv_x, Rv_y
end

"""
    integrate_momentum_residual(v, P_loc, Pnum_loc, T_loc, geo_v_el, phase_loc,
                                η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP) -> (Rv_x, Rv_y)

Integrate the element momentum residual including a numerical pressure correction
and a density-dependent gravitational body force.

`Pnum_loc` is an optional numerical pressure correction (`nothing` → zero).
`T_loc` holds temperature at the `NP` pressure nodes; it is interpolated to each
quadrature point via `NqP` to evaluate the linearised EOS
`ρ = ρ0·(1 − α·(T − Tref) + P/K)`. The body-force contribution is
`−∫ Nᵢ·ρg dΩ`, with `g` the 2-component gravity vector.
"""
@inline function integrate_momentum_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    Pnum_loc::Union{SVector{NP}, Nothing},
    T_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, α, ρ0, K,
    g,
    Tref::Real,
    Δt,
    Nq,
    NqP,
) where {N, NP}
    return integrate_momentum_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, nothing, nothing, Nq, NqP,
    )
end

@inline function integrate_momentum_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    Pnum_loc::Union{SVector{NP}, Nothing},
    T_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, α, ρ0, K,
    g,
    Tref::Real,
    Δt,
    τ_old,
    Nq,
    NqP,
) where {N, NP}
    return integrate_momentum_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old, nothing, Nq, NqP,
    )
end

@inline function integrate_momentum_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    Pnum_loc::Union{SVector{NP}, Nothing},
    T_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, α, ρ0, K,
    g,
    Tref::Real,
    Δt,
    τ_old,
    plastic,
    Nq,
    NqP,
) where {N, NP}
    vxloc, vyloc = v
    T    = promote_type(eltype(vxloc), eltype(vyloc))
    Rv_x = zero(SVector{N, T})
    Rv_y = zero(SVector{N, T})
    # Compressibility β = 1/K: safe for K=Inf (β=0) and avoids NaN from
    # interp2ip_phase when quadratic shape functions are negative.
    β = map(inv, K)
    for q in eachindex(geo_v_el)
        ∂N∂x, dΩ = geo_v_el[q]
        Nv = Nq[q]
        τ_old_q = old_stress_at_ip(Nv, τ_old, T)
        Pq    = dot(NqP[q], P_loc)
        τxx, τyy, τxy = deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old_q, Pq, plastic)
        Pnumq = dot_or_zero(NqP[q], Pnum_loc)
        Tq    = dot(NqP[q], T_loc)
        αq    = interp2ip_phase(Nv, α,  phase_loc)
        βq    = interp2ip_phase(Nv, β,  phase_loc)
        ρ0q   = interp2ip_phase(Nv, ρ0, phase_loc)
        ρq    = ρ0q * (1 - αq * (Tq - Tref) + βq * Pq)
        # x-momentum: ∫ (∂Nᵢ/∂x·(τxx−P) + ∂Nᵢ/∂y·τxy − Nᵢ·ρgₓ) dΩ
        Rv_x += (∂N∂x[:, 1] * (τxx - Pq - Pnumq) + ∂N∂x[:, 2] * τxy) * dΩ - Nv * (ρq * g[1] * dΩ)
        # y-momentum: ∫ (∂Nᵢ/∂y·(τyy−P) + ∂Nᵢ/∂x·τxy − Nᵢ·ρgᵧ) dΩ
        Rv_y += (∂N∂x[:, 2] * (τyy - Pq - Pnumq) + ∂N∂x[:, 1] * τxy) * dΩ - Nv * (ρq * g[2] * dΩ)
    end
    return Rv_x, Rv_y
end

"""
    integrate_momentum_x_residual(v, P_loc, Pnum_loc, T_loc, geo_v_el, phase_loc,
                                  η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP) -> Rv_x

Integrate only the x-component of the element momentum residual, including the
gravitational body force. Same physics as `integrate_momentum_residual` but
returns only `Rv_x`, useful when the x and y systems are solved independently.
"""
@inline function integrate_momentum_x_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    Pnum_loc::Union{SVector{NP}, Nothing},
    T_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, α, ρ0, K,
    g,
    Tref::Real,
    Δt,
    Nq,
    NqP,
) where {N, NP}
    Rv_x, _ = integrate_momentum_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, nothing, nothing, Nq, NqP,
    )
    return Rv_x
end

@inline function integrate_momentum_x_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    Pnum_loc::Union{SVector{NP}, Nothing},
    T_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, α, ρ0, K,
    g,
    Tref::Real,
    Δt,
    τ_old,
    Nq,
    NqP,
) where {N, NP}
    Rv_x, _ = integrate_momentum_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old, nothing, Nq, NqP,
    )
    return Rv_x
end

@inline function integrate_momentum_x_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    Pnum_loc::Union{SVector{NP}, Nothing},
    T_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, α, ρ0, K,
    g,
    Tref::Real,
    Δt,
    τ_old,
    plastic,
    Nq,
    NqP,
) where {N, NP}
    Rv_x, _ = integrate_momentum_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old, plastic, Nq, NqP,
    )
    return Rv_x
end

"""
    integrate_momentum_x_residual(v, P_loc, P0loc, T_loc, T0loc,
                                  geo_v_el, geo_P_el, phase_v, phase_P,
                                  η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
                                  MP_loc, Nq, NqP) -> Rv_x

Integrate the x-momentum residual with the DYREL numerical pressure correction
computed directly from the local pressure residual:

    Pnum = γ_eff * RP(v) / M_P

`RP(v)` is the weak pressure residual and `M_P` is the lumped pressure mass, so
`RP/M_P` matches the pointwise finite-difference residual used by JustRelax.
"""
@inline function integrate_momentum_x_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    P0loc::SVector{NP},
    T_loc::SVector{NP},
    T0loc::SVector{NP},
    geo_v_el,
    geo_P_el,
    phase_v,
    phase_P,
    η, G, α, ρ0, K,
    g,
    Tref::Real,
    ηb, Δt, γ_eff,
    MP_loc::SVector{NP},
    Nq,
    NqP,
) where {N, NP}
    Pnum_loc = pressure_scale(γ_eff, integrate_PH_pressure_residual(
        v, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
    ), MP_loc)
    return integrate_momentum_x_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, nothing, nothing, Nq, NqP,
    )
end

@inline function integrate_momentum_x_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    P0loc::SVector{NP},
    T_loc::SVector{NP},
    T0loc::SVector{NP},
    geo_v_el,
    geo_P_el,
    phase_v,
    phase_P,
    η, G, α, ρ0, K,
    g,
    Tref::Real,
    ηb, Δt, γ_eff,
    MP_loc::SVector{NP},
    τ_old,
    Nq,
    NqP,
) where {N, NP}
    Pnum_loc = pressure_scale(γ_eff, integrate_PH_pressure_residual(
        v, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
    ), MP_loc)
    return integrate_momentum_x_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, τ_old, nothing, Nq, NqP,
    )
end

@inline function integrate_momentum_x_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    P0loc::SVector{NP},
    T_loc::SVector{NP},
    T0loc::SVector{NP},
    geo_v_el,
    geo_P_el,
    phase_v,
    phase_P,
    η, G, α, ρ0, K,
    g,
    Tref::Real,
    ηb, Δt, γ_eff,
    MP_loc::SVector{NP},
    τ_old,
    plastic,
    Nq,
    NqP,
) where {N, NP}
    Pnum_loc = pressure_scale(γ_eff, integrate_PH_pressure_residual(
        v, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
    ), MP_loc)
    return integrate_momentum_x_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, τ_old, plastic, Nq, NqP,
    )
end

"""
    integrate_momentum_y_residual(v, P_loc, Pnum_loc, T_loc, geo_v_el, phase_loc,
                                  η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP) -> Rv_y

Integrate only the y-component of the element momentum residual, including the
gravitational body force. Same physics as `integrate_momentum_residual` but
returns only `Rv_y`, useful when the x and y systems are solved independently.
"""
@inline function integrate_momentum_y_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    Pnum_loc::Union{SVector{NP}, Nothing},
    T_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, α, ρ0, K,
    g,
    Tref::Real,
    Δt,
    Nq,
    NqP,
) where {N, NP}
    _, Rv_y = integrate_momentum_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, nothing, nothing, Nq, NqP,
    )
    return Rv_y
end

@inline function integrate_momentum_y_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    Pnum_loc::Union{SVector{NP}, Nothing},
    T_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, α, ρ0, K,
    g,
    Tref::Real,
    Δt,
    τ_old,
    Nq,
    NqP,
) where {N, NP}
    _, Rv_y = integrate_momentum_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old, nothing, Nq, NqP,
    )
    return Rv_y
end

@inline function integrate_momentum_y_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    Pnum_loc::Union{SVector{NP}, Nothing},
    T_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, α, ρ0, K,
    g,
    Tref::Real,
    Δt,
    τ_old,
    plastic,
    Nq,
    NqP,
) where {N, NP}
    _, Rv_y = integrate_momentum_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old, plastic, Nq, NqP,
    )
    return Rv_y
end

"""
    integrate_momentum_y_residual(v, P_loc, P0loc, T_loc, T0loc,
                                  geo_v_el, geo_P_el, phase_v, phase_P,
                                  η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
                                  MP_loc, Nq, NqP) -> Rv_y

Integrate the y-momentum residual with the local DYREL numerical pressure
correction `Pnum = γ_eff * RP(v) / M_P` computed internally.
"""
@inline function integrate_momentum_y_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    P0loc::SVector{NP},
    T_loc::SVector{NP},
    T0loc::SVector{NP},
    geo_v_el,
    geo_P_el,
    phase_v,
    phase_P,
    η, G, α, ρ0, K,
    g,
    Tref::Real,
    ηb, Δt, γ_eff,
    MP_loc::SVector{NP},
    Nq,
    NqP,
) where {N, NP}
    Pnum_loc = pressure_scale(γ_eff, integrate_PH_pressure_residual(
        v, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
    ), MP_loc)
    return integrate_momentum_y_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, nothing, nothing, Nq, NqP,
    )
end

@inline function integrate_momentum_y_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    P0loc::SVector{NP},
    T_loc::SVector{NP},
    T0loc::SVector{NP},
    geo_v_el,
    geo_P_el,
    phase_v,
    phase_P,
    η, G, α, ρ0, K,
    g,
    Tref::Real,
    ηb, Δt, γ_eff,
    MP_loc::SVector{NP},
    τ_old,
    Nq,
    NqP,
) where {N, NP}
    Pnum_loc = pressure_scale(γ_eff, integrate_PH_pressure_residual(
        v, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
    ), MP_loc)
    return integrate_momentum_y_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, τ_old, nothing, Nq, NqP,
    )
end

@inline function integrate_momentum_y_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    P0loc::SVector{NP},
    T_loc::SVector{NP},
    T0loc::SVector{NP},
    geo_v_el,
    geo_P_el,
    phase_v,
    phase_P,
    η, G, α, ρ0, K,
    g,
    Tref::Real,
    ηb, Δt, γ_eff,
    MP_loc::SVector{NP},
    τ_old,
    plastic,
    Nq,
    NqP,
) where {N, NP}
    Pnum_loc = pressure_scale(γ_eff, integrate_PH_pressure_residual(
        v, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
    ), MP_loc)
    return integrate_momentum_y_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, τ_old, plastic, Nq, NqP,
    )
end

"""
    assemble_momentum_residual_matrices_atomix!(Rv_x, Rv_y, vx, vy, P, T, Pnum,
                                                el2n_v, el2nP, geo_v, nels,
                                                element_v, element_P,
                                                phases, η, G, α, ρ0, K, g, Tref, Δt,
                                                backend, workgroup)

Assemble the Stokes momentum residuals `Rv_x` and `Rv_y` using Atomix-backed atomic scatter.

`Pnum` is an optional nodal array of numerical pressure corrections; pass `nothing`
to omit the correction. `T` is the temperature field on pressure nodes; it is used
together with `α`, `ρ0`, `K`, `g`, and `Tref` to compute the density body force
`ρ = ρ0·(1 − α·(T−Tref) + P/K)` and the resulting gravity term `−∫ Nᵢ·ρg dΩ`.
`phases` is a velocity-node integer array selecting the material phase.
"""
function assemble_momentum_residual_matrices_atomix!(
    Rv_x, Rv_y,
    vx, vy,
    P, T,
    Pnum,
    el2n_v, el2nP,
    geo_v,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    η, G, α, ρ0, K,
    g, Tref, Δt,
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    return assemble_momentum_residual_matrices_atomix!(
        Rv_x, Rv_y, vx, vy, P, T, Pnum,
        el2n_v, el2nP, geo_v, nels, element_v, element_P,
        phases, nothing, nothing, η, G, α, ρ0, K, g, Tref, Δt,
        backend, workgroup,
    )
end

function assemble_momentum_residual_matrices_atomix!(
    Rv_x, Rv_y,
    vx, vy,
    P, T,
    Pnum,
    el2n_v, el2nP,
    geo_v,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    τ_old,
    η, G, α, ρ0, K,
    g, Tref, Δt,
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    return assemble_momentum_residual_matrices_atomix!(
        Rv_x, Rv_y, vx, vy, P, T, Pnum,
        el2n_v, el2nP, geo_v, nels, element_v, element_P,
        phases, τ_old, nothing, η, G, α, ρ0, K, g, Tref, Δt,
        backend, workgroup,
    )
end

function assemble_momentum_residual_matrices_atomix!(
    Rv_x, Rv_y,
    vx, vy,
    P, T,
    Pnum,
    el2n_v, el2nP,
    geo_v,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    τ_old,
    plastic,
    η, G, α, ρ0, K,
    g, Tref, Δt,
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    Nq  = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)

    fill!(Rv_x, 0)
    fill!(Rv_y, 0)
    momentum_residual_atomic_kernel!(backend, workgroup)(
        Rv_x, Rv_y, vx, vy, P, T, Pnum, el2n_v, el2nP, geo_v, phases, τ_old, plastic,
        η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, Val(NV), Val(NP);
        ndrange = nels,
    )
    KA.synchronize(backend)
end

@kernel function momentum_residual_atomic_kernel!(
    Rv_x, Rv_y,
    @Const(vx), @Const(vy),
    @Const(P), @Const(T),
    Pnum,
    @Const(el2n_v), @Const(el2nP),
    @Const(geo_v),
    @Const(phases),
    τ_old,
    plastic,
    η, G, α, ρ0, K, g, Tref, Δt,
    Nq, NqP, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, Re_x, Re_y = momentum_element_residual(
        vx, vy, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
        τ_old, plastic, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP),
    )
    for (i, inod) in enumerate(local_nodes_v)
        Atomix.@atomic :monotonic Rv_x[inod] += Re_x[i]
        Atomix.@atomic :monotonic Rv_y[inod] += Re_y[i]
    end
end

"""
    momentum_element_residual(vx, vy, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
                              η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP))

Gather element-local nodal values and integrate the Stokes momentum residual for element `iel`.

Returns `(local_nodes_v, Re_x, Re_y)` ready for global scatter into `Rv_x` and `Rv_y`.
"""
@inline function momentum_element_residual(
    vx, vy, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
    η, G, α, ρ0, K, g, Tref, Δt,
    Nq, NqP, iel, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    return momentum_element_residual(
        vx, vy, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
        nothing, nothing, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP),
    )
end

@inline function momentum_element_residual(
    vx, vy, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
    τ_old, η, G, α, ρ0, K, g, Tref, Δt,
    Nq, NqP, iel, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    return momentum_element_residual(
        vx, vy, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
        τ_old, nothing, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP),
    )
end

@inline function momentum_element_residual(
    vx, vy, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
    τ_old, plastic, η, G, α, ρ0, K, g, Tref, Δt,
    Nq, NqP, iel, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP,  iel, Val(NP))
    geo_v_el  = geo_v[iel]
    vxloc     = SVector{NV}(ntuple(i -> vx[local_nodes_v[i]], Val(NV)))
    vyloc     = SVector{NV}(ntuple(i -> vy[local_nodes_v[i]], Val(NV)))
    P_loc     = SVector{NP}(ntuple(i ->  P[local_nodes_P[i]], Val(NP)))
    T_loc     = SVector{NP}(ntuple(i ->  T[local_nodes_P[i]], Val(NP)))
    Pnum_loc  = _gather_or_nothing(Pnum, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, Val(NV))
    phase_loc = _stokes_phase_loc(phases, local_nodes_v, iel, Val(NV))
    Re_x, Re_y = integrate_momentum_residual(
        (vxloc, vyloc), P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old_loc, plastic, Nq, NqP,
    )
    return local_nodes_v, Re_x, Re_y
end

_gather_or_nothing(::Nothing, _, ::Val) = nothing
@inline function _gather_or_nothing(arr, nodes, ::Val{N}) where N
    SVector{N}(ntuple(i -> arr[nodes[i]], Val(N)))
end

_gather_old_stress(::Nothing, _, ::Val) = nothing
@inline function _gather_old_stress(τ_old::NTuple{3}, nodes, ::Val{N}) where N
    return (
        SVector{N}(ntuple(i -> τ_old[1][nodes[i]], Val(N))),
        SVector{N}(ntuple(i -> τ_old[2][nodes[i]], Val(N))),
        SVector{N}(ntuple(i -> τ_old[3][nodes[i]], Val(N))),
    )
end

_gather_or_scalar(x::Number, _, ::Val) = x
@inline function _gather_or_scalar(arr, nodes, ::Val{N}) where N
    SVector{N}(ntuple(i -> arr[nodes[i]], Val(N)))
end

"""
    element_momentum_jacobians(vx, vy, P, T, el2n_v, el2nP, geo, phases,
                               η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP))

Compute per-element Jacobian diagnostics for the Stokes momentum residuals via
ForwardDiff, differentiating `Rv_x` w.r.t. `vx` and `Rv_y` w.r.t. `vy`.

Returns `(local_nodes_v, rowsums_x, diags_x, rowsums_y, diags_y)` where
`rowsums_*[i]` is the coupled L1 row sum, including the `vx↔vy` shear blocks,
and `diags_*[i]` is the absolute diagonal of the same-component block. The
row sums provide a conservative smoother/preconditioner and spectral estimate.
"""
@inline function element_momentum_jacobians(
    vx, vy, P, T, el2n_v, el2nP, geo, phases, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    return element_momentum_jacobians(
        vx, vy, P, T, el2n_v, el2nP, geo, phases, nothing, nothing,
        η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP),
    )
end

@inline function element_momentum_jacobians(
    vx, vy, P, T, el2n_v, el2nP, geo, phases, τ_old,
    η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    return element_momentum_jacobians(
        vx, vy, P, T, el2n_v, el2nP, geo, phases, τ_old, nothing,
        η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP),
    )
end

@inline function element_momentum_jacobians(
    vx, vy, P, T, el2n_v, el2nP, geo, phases, τ_old, plastic,
    η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP,  iel, Val(NP))
    geo_el    = geo[iel]
    vxloc     = SVector{NV}(ntuple(i -> vx[local_nodes_v[i]], Val(NV)))
    vyloc     = SVector{NV}(ntuple(i -> vy[local_nodes_v[i]], Val(NV)))
    P_loc     = SVector{NP}(ntuple(i ->  P[local_nodes_P[i]], Val(NP)))
    T_loc     = SVector{NP}(ntuple(i ->  T[local_nodes_P[i]], Val(NP)))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, Val(NV))
    phase_loc = _stokes_phase_loc(phases, local_nodes_v, iel, Val(NV))

    ∂RVx∂vx = ForwardDiff.jacobian(
        vx_loc -> integrate_momentum_x_residual(
            (vx_loc, vyloc), P_loc, nothing, T_loc,
            geo_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old_loc, plastic, Nq, NqP,
        ),
        vxloc,
    )
    ∂RVx∂vy = ForwardDiff.jacobian(
        vy_loc -> integrate_momentum_x_residual(
            (vxloc, vy_loc), P_loc, nothing, T_loc,
            geo_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old_loc, plastic, Nq, NqP,
        ),
        vyloc,
    )
    rowsums_x = SVector{NV}(ntuple(
        i -> sum(abs(∂RVx∂vx[i, j]) + abs(∂RVx∂vy[i, j]) for j in 1:NV),
        Val(NV),
    ))
    diags_x   = SVector{NV}(ntuple(i -> abs(∂RVx∂vx[i, i]), Val(NV)))

    ∂RVy∂vy = ForwardDiff.jacobian(
        vy_loc -> integrate_momentum_y_residual(
            (vxloc, vy_loc), P_loc, nothing, T_loc,
            geo_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old_loc, plastic, Nq, NqP,
        ),
        vyloc,
    )
    ∂RVy∂vx = ForwardDiff.jacobian(
        vx_loc -> integrate_momentum_y_residual(
            (vx_loc, vyloc), P_loc, nothing, T_loc,
            geo_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old_loc, plastic, Nq, NqP,
        ),
        vxloc,
    )
    rowsums_y = SVector{NV}(ntuple(
        i -> sum(abs(∂RVy∂vy[i, j]) + abs(∂RVy∂vx[i, j]) for j in 1:NV),
        Val(NV),
    ))
    diags_y   = SVector{NV}(ntuple(i -> abs(∂RVy∂vy[i, i]), Val(NV)))

    return local_nodes_v, rowsums_x, diags_x, rowsums_y, diags_y
end

"""
    assemble_momentum_jacobian_matrices_atomix!(∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
                                                vx, vy, P, T, el2n_v, el2nP, geo_v, nels,
                                                element_v, element_P,
                                                phases, η, G, α, ρ0, K, g, Tref, Δt,
                                                backend, workgroup)

Assemble the Stokes momentum Jacobian diagnostics using Atomix-backed atomic scatter.

Fills the row-sum Jacobian estimates `∂Rv_x∂vx`, `∂Rv_y∂vy` and row-sum
preconditioners `PC_vx`, `PC_vy` by differentiating the momentum residuals with
ForwardDiff via `element_momentum_jacobians`. The row sums include the
same-component and shear-coupling blocks, which is safer for the mixed T7/P1
Stokes smoother than using only the absolute diagonal.
"""
function assemble_momentum_jacobian_matrices_atomix!(
    ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
    vx, vy,
    P, T,
    el2n_v, el2nP,
    geo_v,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    η, G, α, ρ0, K,
    g, Tref, Δt,
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    return assemble_momentum_jacobian_matrices_atomix!(
        ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        vx, vy, P, T, el2n_v, el2nP, geo_v, nels, element_v, element_P,
        phases, nothing, nothing, η, G, α, ρ0, K, g, Tref, Δt,
        backend, workgroup,
    )
end

function assemble_momentum_jacobian_matrices_atomix!(
    ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
    vx, vy,
    P, T,
    el2n_v, el2nP,
    geo_v,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    τ_old,
    η, G, α, ρ0, K,
    g, Tref, Δt,
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    return assemble_momentum_jacobian_matrices_atomix!(
        ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        vx, vy, P, T, el2n_v, el2nP, geo_v, nels, element_v, element_P,
        phases, τ_old, nothing, η, G, α, ρ0, K, g, Tref, Δt,
        backend, workgroup,
    )
end

function assemble_momentum_jacobian_matrices_atomix!(
    ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
    vx, vy,
    P, T,
    el2n_v, el2nP,
    geo_v,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    τ_old,
    plastic,
    η, G, α, ρ0, K,
    g, Tref, Δt,
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    Nq  = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)

    fill!(∂Rv_x∂vx, 0)
    fill!(PC_vx,    0)
    fill!(∂Rv_y∂vy, 0)
    fill!(PC_vy,    0)
    momentum_jacobian_atomic_kernel!(backend, workgroup)(
        ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        vx, vy, P, T, el2n_v, el2nP, geo_v, phases,
        τ_old, plastic, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, Val(NV), Val(NP);
        ndrange = nels,
    )
    KA.synchronize(backend)
end

@inline function element_augmented_momentum_jacobians(
    vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
    phases_v, phases_P, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
    MP, Nq, NqP, iel, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    return element_augmented_momentum_jacobians(
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, nothing, nothing, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, iel, Val(NV), Val(NP),
    )
end

@inline function element_augmented_momentum_jacobians(
    vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
    phases_v, phases_P, τ_old, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
    MP, Nq, NqP, iel, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    return element_augmented_momentum_jacobians(
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, τ_old, nothing, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, iel, Val(NV), Val(NP),
    )
end

@inline function element_augmented_momentum_jacobians(
    vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
    phases_v, phases_P, τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
    MP, Nq, NqP, iel, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP,  iel, Val(NP))
    geo_v_el  = geo_v[iel]
    geo_P_el  = geo_P[iel]
    vxloc     = SVector{NV}(ntuple(i -> vx[local_nodes_v[i]], Val(NV)))
    vyloc     = SVector{NV}(ntuple(i -> vy[local_nodes_v[i]], Val(NV)))
    P_loc     = SVector{NP}(ntuple(i ->  P[local_nodes_P[i]], Val(NP)))
    P0loc     = SVector{NP}(ntuple(i -> P0[local_nodes_P[i]], Val(NP)))
    T_loc     = SVector{NP}(ntuple(i ->  T[local_nodes_P[i]], Val(NP)))
    T0loc     = SVector{NP}(ntuple(i -> T0[local_nodes_P[i]], Val(NP)))
    MP_loc    = SVector{NP}(ntuple(i -> MP[local_nodes_P[i]], Val(NP)))
    γ_eff_loc = _gather_or_scalar(γ_eff, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, Val(NV))
    phase_v   = _stokes_phase_loc(phases_v, local_nodes_v, iel, Val(NV))
    phase_P   = _stokes_phase_loc(phases_P, local_nodes_P, iel, Val(NP))

    ∂RVx∂vx = ForwardDiff.jacobian(
        vx_arg -> integrate_momentum_x_residual(
            (vx_arg, vyloc), P_loc, P0loc, T_loc, T0loc,
            geo_v_el, geo_P_el, phase_v, phase_P,
            η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff_loc, MP_loc, τ_old_loc, plastic, Nq, NqP,
        ),
        vxloc,
    )
    ∂RVx∂vy = ForwardDiff.jacobian(
        vy_arg -> integrate_momentum_x_residual(
            (vxloc, vy_arg), P_loc, P0loc, T_loc, T0loc,
            geo_v_el, geo_P_el, phase_v, phase_P,
            η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff_loc, MP_loc, τ_old_loc, plastic, Nq, NqP,
        ),
        vyloc,
    )
    rowsums_x = SVector{NV}(ntuple(
        i -> sum(abs(∂RVx∂vx[i, j]) + abs(∂RVx∂vy[i, j]) for j in 1:NV),
        Val(NV),
    ))
    diags_x = SVector{NV}(ntuple(i -> abs(∂RVx∂vx[i, i]), Val(NV)))

    ∂RVy∂vy = ForwardDiff.jacobian(
        vy_arg -> integrate_momentum_y_residual(
            (vxloc, vy_arg), P_loc, P0loc, T_loc, T0loc,
            geo_v_el, geo_P_el, phase_v, phase_P,
            η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff_loc, MP_loc, τ_old_loc, plastic, Nq, NqP,
        ),
        vyloc,
    )
    ∂RVy∂vx = ForwardDiff.jacobian(
        vx_arg -> integrate_momentum_y_residual(
            (vx_arg, vyloc), P_loc, P0loc, T_loc, T0loc,
            geo_v_el, geo_P_el, phase_v, phase_P,
            η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff_loc, MP_loc, τ_old_loc, plastic, Nq, NqP,
        ),
        vxloc,
    )
    rowsums_y = SVector{NV}(ntuple(
        i -> sum(abs(∂RVy∂vy[i, j]) + abs(∂RVy∂vx[i, j]) for j in 1:NV),
        Val(NV),
    ))
    diags_y = SVector{NV}(ntuple(i -> abs(∂RVy∂vy[i, i]), Val(NV)))

    return local_nodes_v, rowsums_x, diags_x, rowsums_y, diags_y
end

function assemble_augmented_momentum_jacobian_matrices_atomix!(
    ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
    vx, vy,
    P, P0,
    T, T0,
    el2n_v, el2nP,
    geo_v, geo_P,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases_v, phases_P,
    η, G, α, ρ0, K,
    g, Tref,
    ηb, Δt, γ_eff, MP,
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    return assemble_augmented_momentum_jacobian_matrices_atomix!(
        ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P, nels,
        element_v, element_P, phases_v, phases_P, nothing, nothing,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP,
        backend, workgroup,
    )
end

function assemble_augmented_momentum_jacobian_matrices_atomix!(
    ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
    vx, vy,
    P, P0,
    T, T0,
    el2n_v, el2nP,
    geo_v, geo_P,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases_v, phases_P,
    τ_old,
    η, G, α, ρ0, K,
    g, Tref,
    ηb, Δt, γ_eff, MP,
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    return assemble_augmented_momentum_jacobian_matrices_atomix!(
        ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P, nels,
        element_v, element_P, phases_v, phases_P, τ_old, nothing,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP,
        backend, workgroup,
    )
end

function assemble_augmented_momentum_jacobian_matrices_atomix!(
    ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
    vx, vy,
    P, P0,
    T, T0,
    el2n_v, el2nP,
    geo_v, geo_P,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases_v, phases_P,
    τ_old,
    plastic,
    η, G, α, ρ0, K,
    g, Tref,
    ηb, Δt, γ_eff, MP,
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    Nq  = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)

    fill!(∂Rv_x∂vx, 0)
    fill!(PC_vx,    0)
    fill!(∂Rv_y∂vy, 0)
    fill!(PC_vy,    0)
    augmented_momentum_jacobian_atomic_kernel!(backend, workgroup)(
        ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P, phases_v, phases_P,
        τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP, Nq, NqP, Val(NV), Val(NP);
        ndrange = nels,
    )
    KA.synchronize(backend)
end

@kernel function augmented_momentum_jacobian_atomic_kernel!(
    ∂Rv_x∂vx, PC_vx,
    ∂Rv_y∂vy, PC_vy,
    @Const(vx), @Const(vy),
    @Const(P), @Const(P0),
    @Const(T), @Const(T0),
    @Const(el2n_v), @Const(el2nP),
    @Const(geo_v), @Const(geo_P),
    @Const(phases_v), @Const(phases_P),
    τ_old,
    plastic,
    η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
    @Const(MP),
    Nq, NqP, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, rowsums_x, diags_x, rowsums_y, diags_y = element_augmented_momentum_jacobians(
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, iel, Val(NV), Val(NP),
    )
    for (i, inod) in enumerate(local_nodes_v)
        Atomix.@atomic :monotonic ∂Rv_x∂vx[inod] += rowsums_x[i]
        Atomix.@atomic :monotonic PC_vx[inod]    += diags_x[i]
        Atomix.@atomic :monotonic ∂Rv_y∂vy[inod] += rowsums_y[i]
        Atomix.@atomic :monotonic PC_vy[inod]    += diags_y[i]
    end
end

@kernel function momentum_jacobian_atomic_kernel!(
    ∂Rv_x∂vx, PC_vx,
    ∂Rv_y∂vy, PC_vy,
    @Const(vx), @Const(vy),
    @Const(P), @Const(T),
    @Const(el2n_v), @Const(el2nP),
    @Const(geo_v),
    @Const(phases),
    τ_old,
    plastic,
    η, G, α, ρ0, K, g, Tref, Δt,
    Nq, NqP, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, rowsums_x, diags_x, rowsums_y, diags_y = element_momentum_jacobians(
        vx, vy, P, T, el2n_v, el2nP, geo_v, phases, τ_old, plastic, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP),
    )
    for (i, inod) in enumerate(local_nodes_v)
        Atomix.@atomic :monotonic ∂Rv_x∂vx[inod] += rowsums_x[i]
        Atomix.@atomic :monotonic PC_vx[inod]    += diags_x[i]
        Atomix.@atomic :monotonic ∂Rv_y∂vy[inod] += rowsums_y[i]
        Atomix.@atomic :monotonic PC_vy[inod]    += diags_y[i]
    end
end
