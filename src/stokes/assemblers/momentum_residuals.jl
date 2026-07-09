dot_or_zero(a, ::Nothing) = zero(eltype(a))
dot_or_zero(a, b) = dot(a, b)
@inline pressure_scale(γ_eff::Number, RP, MP) = γ_eff * RP ./ MP
@inline pressure_scale(γ_eff::SVector, RP, MP) = γ_eff .* RP ./ MP
@inline function _local_pressure_correction(
    v, P_loc, P0loc, T_loc, T0loc, geo_v_el, geo_P_el, phase_P, α, ηb, Δt, γ_eff, MP_loc, NqP,
)
    RP_loc = integrate_PH_pressure_residual(
        v, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
    )
    return pressure_scale(γ_eff, RP_loc, MP_loc)
end
@inline effective_viscosity(η, G, Δt) = inv(inv(η) + inv(G * Δt))
"""
    viscoelastic_coefficients_phase(Nv, η, G, phase_loc, Δt) -> (ηve, inv_2Gdt)

Interpolate shear viscosity and elastic shear modulus to a quadrature point
and return the Maxwell viscoelastic effective viscosity `ηve` and the elastic
correction coefficient `1/(2G Δt)`.

Compliance `1/G` is interpolated (rather than `G` itself) so that the purely
viscous limit `G = Inf` stays numerically stable, including at quadratic
integration points where shape functions can be negative.
"""
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
"""
    IntegrationPointStress{TX, TY, TXY}

Old deviatoric-stress components stored at integration points for viscoelastic
memory. Each field is an `NQ × nels` matrix (integration-point index × element
index). Used by `_gather_old_stress` to recover `(τxx_q, τyy_q, τxy_q)` at
quadrature point `q` without going through nodal interpolation.
"""
struct IntegrationPointStress{TX, TY, TXY}
    τxx::TX
    τyy::TY
    τxy::TXY
end

"""
    IntegrationPointStressOutput{TX, TY, TXY}

Scratch buffer for writing the *current* deviatoric stress to integration
points during momentum-residual assembly. `iel` pins the buffer to a specific
element so that `store_stress_at_ip!` can index `τxx[q, iel]` directly.
"""
struct IntegrationPointStressOutput{TX, TY, TXY}
    τxx::TX
    τyy::TY
    τxy::TXY
    iel::Int
end
@inline old_stress_at_ip(_, ::Nothing, ::Type{T}, _) where T = zero_old_stress(T)
@inline function old_stress_at_ip(Nv, τ_old::NTuple{3}, ::Type, _)
    return (
        old_stress_component_at_ip(Nv, τ_old[1]),
        old_stress_component_at_ip(Nv, τ_old[2]),
        old_stress_component_at_ip(Nv, τ_old[3]),
    )
end
@inline old_stress_at_ip(_, τ_old::IntegrationPointStress, ::Type, q) =
    (τ_old.τxx[q], τ_old.τyy[q], τ_old.τxy[q])
@inline store_stress_at_ip!(::Nothing, _, _, _, _) = nothing
@inline function store_stress_at_ip!(τ_store::IntegrationPointStressOutput, q, τxx, τyy, τxy)
    τ_store.τxx[q, τ_store.iel] = τxx
    τ_store.τyy[q, τ_store.iel] = τyy
    τ_store.τxy[q, τ_store.iel] = τxy
    return nothing
end

"""
    second_invariant(τxx, τyy, τxy) -> τII
    second_invariant(A) -> τII

Compute the second invariant `τII = √J₂` of a 2-D symmetric deviatoric
stress tensor.

Includes the out-of-plane component `τzz = −τxx − τyy` required for
plane-strain consistency:

    τII = √((τxx² + τyy² + τzz²) / 2 + τxy²)

A small `eps²` floor is added under the square root so that the function is
safe for automatic differentiation (ForwardDiff, Enzyme) when the stress
components are zero: `d(√x)/dx = 1/(2√x)` diverges at `x = 0`, but
`d(√(x + ε²))/dx = 1/(2√(x + ε²))` remains finite. The floor is below
machine epsilon for any physically meaningful stress, so results are
unchanged in practice.
"""
@inline second_invariant(axx, ayy, axy) = second_invariant(tuple(axx, ayy, axy))

@inline function second_invariant(A::T) where {T <: Union{SVector{3}, NTuple{3}}}
    Azz = -A[1] - A[2]
    # typeof(real(A[1])) recovers the underlying float type when A[1] is a
    # ForwardDiff Dual (real(::Dual) = value(::Dual) is defined by ForwardDiff).
    FT  = typeof(real(A[1]))
    return √((A[1]^2 + A[2]^2 + Azz^2) / 2 + A[3]^2 + eps(FT)^2)
end

"""
    deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old) -> (τxx, τyy, τxy)

Compute the viscoelastic deviatoric stress at a quadrature point.

`v = (vxloc, vyloc)` are element velocity `SVector`s. Strain rates are
computed from the velocity gradients `∇v = ∂N∂x' * vloc`. The Maxwell
effective viscosity `ηve` and elastic correction term `τ_old / (2GΔt)` are
evaluated via `viscoelastic_coefficients_phase`. Pass `(0, 0, 0)` for
`τ_old` on the first time step.
"""
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

"""
    deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic::DruckerPrager) -> (τxx, τyy, τxy)

Compute the elasto-viscoplastic deviatoric stress at a quadrature point with
Drucker-Prager return mapping.

Computes the trial viscoelastic stress, evaluates the yield function
`F = τII − C·cos(ϕ) − P·sin(ϕ)`, and applies the plastic return
`τᵢⱼ ← τᵢⱼ − 2 ηve λ ∂Q/∂τᵢⱼ` when `F > 0`. The plastic multiplier uses the
regularized formula `λ = F / (ηve + η_reg - Kb Δt ∂Q/∂P ∂F/∂P)`.
"""
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

    εxx_eff = (εxx - tr) + τxx_o * inv_2Gdt
    εyy_eff = (εyy - tr) + τyy_o * inv_2Gdt
    εxy_eff = εxy + τxy_o * inv_2Gdt
    τxx = 2 * ηve * εxx_eff
    τyy = 2 * ηve * εyy_eff
    τxy = 2 * ηve * εxy_eff
    τij = τxx, τyy, τxy

    # Interpolate per-phase plastic parameters to the quadrature point.
    cosϕ  = interp2ip_phase(Nv, plastic.cosϕ,  phase_loc)
    sinϕ  = interp2ip_phase(Nv, plastic.sinϕ,  phase_loc)
    sinΨ  = interp2ip_phase(Nv, plastic.sinΨ,  phase_loc)
    C     = interp2ip_phase(Nv, plastic.C,     phase_loc)
    η_reg = interp2ip_phase(Nv, plastic.η_reg, phase_loc)
    Kb    = interp2ip_phase(Nv, plastic.Kb,    phase_loc)

    # Drucker-Prager yield function.
    # second_invariant returns τxx²+τyy²+τzz²+2τxy² = 2J₂, so τII = sqrt(J₂) = sqrt(SI/2).
    τII      = second_invariant(τij)
    τII_safe = τII + eps(typeof(τII))^2
    F        = τII - cosϕ * C - sinϕ * Pq
    ∂F∂P     = -sinϕ

    # Derivatives of the plane-strain invariant with τzz = -τxx - τyy.
    ∂Q∂τxx = (2 * τxx + τyy) / (2 * τII_safe)
    ∂Q∂τyy = (τxx + 2 * τyy) / (2 * τII_safe)
    ∂Q∂τxy = τxy / τII_safe
    ∂Q∂τ   = ∂Q∂τxx, ∂Q∂τyy, ∂Q∂τxy
    ∂Q∂P   = -sinΨ

    λ = if F > 0
        F / (ηve + η_reg - Kb * Δt * ∂Q∂P * ∂F∂P)
    else
        zero(F)
    end

    τij = λ > 0 ?
        map((τ, ∂q) -> τ - 2 * ηve * λ * ∂q, τij, ∂Q∂τ) :
        τij

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
    return integrate_momentum_residual(
        v, P_loc, geo_v_el, phase_loc, η, G, Δt, τ_old, plastic, nothing, Nq, NqP,
    )
end

@inline function integrate_momentum_residual(
    v::NTuple{2, SVector{N, T}},
    P_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, Δt,
    τ_old,
    plastic,
    τ_store,
    Nq,
    NqP,
) where {N, T, NP}
    Rv_x = zero(SVector{N, T})
    Rv_y = zero(SVector{N, T})
    for q in eachindex(geo_v_el)
        ∂N∂x, dΩ = geo_v_el[q]
        Nv = Nq[q]
        τ_old_q = old_stress_at_ip(Nv, τ_old, T, q)
        Pq  = dot(NqP[q], P_loc)
        τxx, τyy, τxy = deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old_q, Pq, plastic)
        store_stress_at_ip!(τ_store, q, τxx, τyy, τxy)
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
    return integrate_momentum_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old, plastic, nothing, Nq, NqP,
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
    τ_store,
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
        τ_old_q = old_stress_at_ip(Nv, τ_old, T, q)
        Pq    = dot(NqP[q], P_loc)
        τxx, τyy, τxy = deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old_q, Pq, plastic)
        store_stress_at_ip!(τ_store, q, τxx, τyy, τxy)
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

@inline function _integrate_momentum_x_with_pressure_correction(
    v, P_loc, P0loc, T_loc, T0loc, geo_v_el, geo_P_el, phase_v, phase_P,
    η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP_loc, τ_old, plastic, Nq, NqP,
)
    Pnum_loc = _local_pressure_correction(
        v, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, γ_eff, MP_loc, NqP,
    )
    return integrate_momentum_x_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, τ_old, plastic, Nq, NqP,
    )
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
    return _integrate_momentum_x_with_pressure_correction(
        v, P_loc, P0loc, T_loc, T0loc, geo_v_el, geo_P_el, phase_v, phase_P,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP_loc, nothing, nothing, Nq, NqP,
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
    return _integrate_momentum_x_with_pressure_correction(
        v, P_loc, P0loc, T_loc, T0loc, geo_v_el, geo_P_el, phase_v, phase_P,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP_loc, τ_old, nothing, Nq, NqP,
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
    return _integrate_momentum_x_with_pressure_correction(
        v, P_loc, P0loc, T_loc, T0loc, geo_v_el, geo_P_el, phase_v, phase_P,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP_loc, τ_old, plastic, Nq, NqP,
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

@inline function _integrate_momentum_y_with_pressure_correction(
    v, P_loc, P0loc, T_loc, T0loc, geo_v_el, geo_P_el, phase_v, phase_P,
    η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP_loc, τ_old, plastic, Nq, NqP,
)
    Pnum_loc = _local_pressure_correction(
        v, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, γ_eff, MP_loc, NqP,
    )
    return integrate_momentum_y_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, τ_old, plastic, Nq, NqP,
    )
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
    return _integrate_momentum_y_with_pressure_correction(
        v, P_loc, P0loc, T_loc, T0loc, geo_v_el, geo_P_el, phase_v, phase_P,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP_loc, nothing, nothing, Nq, NqP,
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
    return _integrate_momentum_y_with_pressure_correction(
        v, P_loc, P0loc, T_loc, T0loc, geo_v_el, geo_P_el, phase_v, phase_P,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP_loc, τ_old, nothing, Nq, NqP,
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
    return _integrate_momentum_y_with_pressure_correction(
        v, P_loc, P0loc, T_loc, T0loc, geo_v_el, geo_P_el, phase_v, phase_P,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP_loc, τ_old, plastic, Nq, NqP,
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
    return assemble_momentum_residual_matrices_atomix!(
        Rv_x, Rv_y, vx, vy, P, T, Pnum,
        el2n_v, el2nP, geo_v, nels, element_v, element_P,
        phases, τ_old, plastic, nothing, η, G, α, ρ0, K, g, Tref, Δt,
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
    τ_store,
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
        τ_store, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, Val(NV), Val(NP);
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
    τ_store,
    η, G, α, ρ0, K, g, Tref, Δt,
    Nq, NqP, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, Re_x, Re_y = momentum_element_residual(
        vx, vy, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
        τ_old, plastic, τ_store, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP),
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
    return momentum_element_residual(
        vx, vy, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
        τ_old, plastic, nothing, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP),
    )
end

@inline function momentum_element_residual(
    vx, vy, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
    τ_old, plastic, τ_store, η, G, α, ρ0, K, g, Tref, Δt,
    Nq, NqP, iel, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP,  iel, Val(NP))
    geo_v_el  = geo_v[iel]
    vxloc     = _gather_local(vx, local_nodes_v, Val(NV))
    vyloc     = _gather_local(vy, local_nodes_v, Val(NV))
    P_loc     = _gather_local(P, local_nodes_P, Val(NP))
    T_loc     = _gather_local(T, local_nodes_P, Val(NP))
    Pnum_loc  = _gather_or_nothing(Pnum, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, iel, Val(NV), Val(length(Nq)))
    τ_store_el = _stress_output(τ_store, iel)
    phase_loc = _gather_phase(phases, local_nodes_v, iel, Val(NV))
    Re_x, Re_y = integrate_momentum_residual(
        (vxloc, vyloc), P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt, τ_old_loc, plastic, τ_store_el, Nq, NqP,
    )
    return local_nodes_v, Re_x, Re_y
end

_gather_or_nothing(::Nothing, _, ::Val) = nothing
@inline function _gather_or_nothing(arr, nodes, ::Val{N}) where N
    _gather_local(arr, nodes, Val(N))
end

_gather_old_stress(::Nothing, _, _, ::Val, ::Val) = nothing
@inline function _gather_old_stress(τ_old::NTuple{3, <:AbstractMatrix}, _, iel, ::Val, ::Val{NQ}) where NQ
    return IntegrationPointStress(
        SVector{NQ}(ntuple(q -> τ_old[1][q, iel], Val(NQ))),
        SVector{NQ}(ntuple(q -> τ_old[2][q, iel], Val(NQ))),
        SVector{NQ}(ntuple(q -> τ_old[3][q, iel], Val(NQ))),
    )
end
@inline function _gather_old_stress(τ_old::NTuple{3}, nodes, _, ::Val{N}, ::Val) where N
    return (
        _gather_local(τ_old[1], nodes, Val(N)),
        _gather_local(τ_old[2], nodes, Val(N)),
        _gather_local(τ_old[3], nodes, Val(N)),
    )
end
_stress_output(::Nothing, _) = nothing
@inline _stress_output(τ_store::NTuple{3, <:AbstractMatrix}, iel) =
    IntegrationPointStressOutput(τ_store[1], τ_store[2], τ_store[3], Int(iel))

_gather_or_scalar(x::Number, _, ::Val) = x
@inline function _gather_or_scalar(arr, nodes, ::Val{N}) where N
    _gather_local(arr, nodes, Val(N))
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
    vxloc     = _gather_local(vx, local_nodes_v, Val(NV))
    vyloc     = _gather_local(vy, local_nodes_v, Val(NV))
    P_loc     = _gather_local(P, local_nodes_P, Val(NP))
    T_loc     = _gather_local(T, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, iel, Val(NV), Val(length(Nq)))
    phase_loc = _gather_phase(phases, local_nodes_v, iel, Val(NV))

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

"""
    element_augmented_momentum_jacobians(vx, vy, P, P0, T, T0,
                                          el2n_v, el2nP, geo_v, geo_P,
                                          phases_v, phases_P,
                                          η, G, α, ρ0, K, g, Tref,
                                          ηb, Δt, γ_eff, MP, Nq, NqP,
                                          iel, Val(NV), Val(NP))

Compute per-element Jacobian diagnostics for the augmented Stokes momentum
residuals via ForwardDiff, accounting for the DYREL/Arrow-Hurwicz numerical
pressure correction computed inline.

The pressure correction `Pnum = γ_eff * RP(v) / M_P` is re-evaluated inside
each ForwardDiff call from the current velocity, capturing the
velocity-to-pressure-to-velocity coupling introduced by the Arrow-Hurwicz
scheme. This makes the preconditioner more effective than
`element_momentum_jacobians` for problems where that coupling is significant.

Returns `(local_nodes_v, rowsums_x, diags_x, rowsums_y, diags_y)`.
"""
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
    vxloc     = _gather_local(vx, local_nodes_v, Val(NV))
    vyloc     = _gather_local(vy, local_nodes_v, Val(NV))
    P_loc     = _gather_local(P, local_nodes_P, Val(NP))
    P0loc     = _gather_local(P0, local_nodes_P, Val(NP))
    T_loc     = _gather_local(T, local_nodes_P, Val(NP))
    T0loc     = _gather_local(T0, local_nodes_P, Val(NP))
    MP_loc    = _gather_local(MP, local_nodes_P, Val(NP))
    γ_eff_loc = _gather_or_scalar(γ_eff, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, iel, Val(NV), Val(length(Nq)))
    phase_v   = _gather_phase(phases_v, local_nodes_v, iel, Val(NV))
    phase_P   = _gather_phase(phases_P, local_nodes_P, iel, Val(NP))

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

"""
    assemble_augmented_momentum_jacobian_matrices_atomix!(
        ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P, nels,
        element_v, element_P, phases_v, phases_P,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP,
        backend, workgroup)

Assemble the augmented Stokes momentum Jacobian diagnostics using
Atomix-backed atomic scatter.

Compared to `assemble_momentum_jacobian_matrices_atomix!`, this assembler
includes the DYREL/Arrow-Hurwicz numerical pressure correction in the
Jacobian by computing `Pnum = γ_eff * RP(v) / M_P` inline inside each
ForwardDiff call via `element_augmented_momentum_jacobians`. This captures
the velocity–pressure coupling and produces a more effective preconditioner
for incompressible Stokes flows. `P0`, `T0`, `ηb`, `γ_eff`, and `MP` are
the additional arguments relative to the non-augmented assembler.
"""
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
