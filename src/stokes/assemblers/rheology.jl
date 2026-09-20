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
`τ_old` on the first time step. This method has no yield criterion; the stress
is purely viscoelastic.
"""
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
regularized formula `λ = F / (ηve + η_reg + Kb Δt ∂Q/∂P ∂F/∂P)`, whose
denominator stays positive for any dilation angle.
"""
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

    # Plastic dilation feeds back on the yield surface through the pressure: the
    # volumetric plastic strain enters the mass balance as `P ← P - Kb Δt λ ∂Q/∂P`,
    # shifting `F` by `-Kb Δt λ ∂F/∂P ∂Q/∂P`, which the consistency condition moves
    # into the denominator with a positive sign. The product `∂Q/∂P ∂F/∂P = sinΨ sinϕ`
    # is non-negative, so the denominator cannot vanish for any dilation angle.
    λ = if F > 0
        F / (ηve + η_reg + Kb * Δt * ∂Q∂P * ∂F∂P)
    else
        zero(F)
    end

    τij = λ > 0 ?
        map((τ, ∂q) -> τ - 2 * ηve * λ * ∂q, τij, ∂Q∂τ) :
        τij

    return τij
end


# ---------------------------------------------------------------------------
# Drucker-Prager tensile cap
#
# Popov, Berlie & Kaus (2025), "A dilatant visco-elasto-viscoplasticity model
# with globally continuous tensile cap", Geosci. Model Dev. 18, 7035-7058,
# doi:10.5194/gmd-18-7035-2025. Equation numbers below refer to that paper.
#
# Pressure is compression-positive here and there (their Sect. 2.1), so the
# tensile strength `pT` is the pressure at which the rock fails in pure tension
# and is therefore negative. The shear branch reproduces the Drucker-Prager
# surface `deviatoric_stress` already uses, with `k = sinϕ` and `c = C cosϕ`.
# ---------------------------------------------------------------------------

"""
    cap_geometry(k, kq, c, pT) -> (; a, b, p_y, R_y, p_d, τ_d, p_q)

Derived geometry of the circular tensile cap that closes the Drucker-Prager
shear envelope on the tensile side (Popov et al. 2025, Eqs. 13-17).

Takes the *derived* Drucker-Prager coefficients rather than the raw angles:
friction coefficient `k = sin(ϕ)`, dilation coefficient `kq = sin(Ψ)`,
Drucker-Prager cohesion `c = C·cos(ϕ)`, and tensile strength `pT ≤ 0`.

Returns the scaling coefficients `a`, `b` (Eq. 14), the cap centre `p_y` and
radius `R_y` (Eq. 15), the delimiter point `(p_d, τ_d)` where the cap meets the
shear line (Eq. 16), and the flow-potential centre `p_q` (Eq. 17).

The cap is the circle of radius `R_y` centred at `(p_y, 0)` in the meridional
`(P, τII)` plane. `p_y` is fixed by two conditions at once: the circle meets the
pressure axis at `pT`, so `R_y = p_y − pT`, and it is tangent to the shear line
`τII = k·P + c`, so `R_y = (k·p_y + c)/a`. Eliminating `R_y` gives the `p_y`
below. Tangency is what makes the composite surface continuously
differentiable, which Perzyna viscoplasticity requires because `λ̇ = ⟨F⟩/η_reg`
reads `F` *away* from the surface, not only on it.

Nothing here is cached per phase: strain softening moves `k` and `c`, and the
whole geometry moves with them, so a precomputed cap would silently freeze
while the shear branch softened.

`R_y > 0` requires `c + k·pT > 0`, i.e. cohesion must exceed `k·|pT|`; the caller
is responsible for parameters that satisfy it.
"""
@inline function cap_geometry(k, kq, c, pT)
    a = √(one(k) + k * k)
    b = √(one(kq) + kq * kq)
    # Eq. 15 written as (pT + c/a)/(1 - k/a); the equivalent form below avoids
    # the nested division.
    p_y = (a * pT + c) / (a - k)
    R_y = p_y - pT
    p_d = p_y - R_y * k / a
    # Eq. 16 gives τ_d = k·p_d + c; tangency makes that equal R_y/a exactly.
    τ_d = R_y / a
    p_q = p_d + kq * τ_d
    return (; a, b, p_y, R_y, p_d, τ_d, p_q)
end

"""
    cap_yield_function(τII, P, k, c, geom) -> F

Composite yield function of the smooth tensile cap (Popov et al. 2025, Eq. 18),
where `geom` comes from [`cap_geometry`](@ref).

    F = τII − k·P − c              in the shear domain
    F = a·(R̂_y − R_y)              on the tensile cap,  R̂_y = √(τII² + (P − p_y)²)

The cap branch carries the factor `a` so that `‖∇F‖ = a` on *both* branches:
without it the two segments would agree on the surface but disagree everywhere
outside it, and the overstress `⟨F⟩` that drives Perzyna viscoplasticity would
jump across the delimiter.

Eq. 18 selects the shear branch with `τII·(p_y − p_d) ≥ τ_d·(p_y − P)`.
Substituting `p_y − p_d = R_y·k/a` and `τ_d = R_y/a` and cancelling the positive
factor `R_y/a` reduces that to the half-plane test used below.
"""
@inline function cap_yield_function(τII, P, k, c, geom)
    return if P + k * τII ≥ geom.p_y
        τII - k * P - c
    else
        R̂y = √(τII * τII + (P - geom.p_y)^2 + eps(typeof(τII))^2)
        geom.a * (R̂y - geom.R_y)
    end
end

"""
    cap_flow_direction(τII, P, kq, geom) -> (Bτ, Bp)

Prefactors of the flow-potential gradient `∂Q/∂σᵢⱼ = Bτ·τᵢⱼ + Bp·δᵢⱼ`
(Popov et al. 2025, Eqs. 21-22), where `geom` comes from [`cap_geometry`](@ref).

    Bτ, Bp = 1/(2τII),  kq/3                      in the shear domain
    Bτ, Bp = b/(2R̂_q), −b(P − p_q)/(3R̂_q)         on the tensile cap

with `R̂_q = √(τII² + (P − p_q)²)`. The domain test is the same reduction as in
[`cap_yield_function`](@ref), applied to Eq. 19 with the potential's own centre
`p_q`: `τII·(p_q − p_d) ≥ τ_d·(p_q − P)` becomes `P + kq·τII ≥ p_q`.

The potential is non-associated whenever `Ψ ≠ ϕ`, and `p_q` is placed so that its
own shear/tensile transition passes through the *same* delimiter as the yield
surface. That is what makes the return direction single-valued at every point
above the surface, so no active-surface search is needed.

Sign convention: the volumetric viscoplastic strain rate is
`ε̇_vol = λ·tr(∂Q/∂σ) = 3λ·Bp`, since `tr(τ) = 0`. In the shear domain this is
`λ·kq = λ·sin(Ψ)`, matching the existing `∂Q∂P = −sinΨ` of the Drucker-Prager
return map, where the pressure correction enters as `P ← P − Kb·Δt·λ·∂Q∂P`.
"""
@inline function cap_flow_direction(τII, P, kq, geom)
    ϵ² = eps(typeof(τII))^2
    return if P + kq * τII ≥ geom.p_q
        inv(2 * √(τII * τII + ϵ²)), kq / 3
    else
        R̂q = √(τII * τII + (P - geom.p_q)^2 + ϵ²)
        geom.b / (2 * R̂q), -geom.b * (P - geom.p_q) / (3 * R̂q)
    end
end

# ---------------------------------------------------------------------------
# Local return map for the tensile cap.
#
# The paper solves a 3-unknown system (τII, p, λ̇) because it carries diffusion
# and dislocation creep, so even its deviatoric relation is nonlinear. Here `ηve`
# is given and that relation is linear, so the system collapses to one scalar
# equation — but only in the right variable. Parametrising by
#
#     μ = 2 λ Bτ
#
# rather than by `λ` makes *both* returns explicit on *both* branches:
#
#     τII(μ) = τII_trial / (1 + ηve μ)                         radial, either branch
#     P(μ)   = P_trial + Kb Δt kq μ τII(μ)                     shear potential
#     P(μ)   = (P_trial + Kb Δt μ p_q) / (1 + Kb Δt μ)         cap potential
#
# so no inner iteration is needed to evaluate the residual. The multiplier comes
# back out as `λ = μ τII` on the shear branch and `λ = μ R̂_q / b` on the cap.
# ---------------------------------------------------------------------------

"""
    cap_residual_and_slope(μ, τII_trial, P_trial, ηve, KΔt, k, c, kq, η_reg, geom)
        -> (r, dr, τII, P, λ)

Perzyna consistency residual `r(μ) = F(τII(μ), P(μ)) − η_reg λ(μ)` and its exact
derivative, together with the state it implies. `geom` comes from
[`cap_geometry`](@ref) and `KΔt` is `Kb * Δt`.

The yield branch and the flow-potential branch are selected independently: `F`
switches on `P + k τII ≥ p_y`, the flow direction on `P + kq τII ≥ p_q`. They are
different rays and a point can sit on the shear side of one and the tensile side
of the other, which is exactly the mode-I/mode-II transition region.
"""
@inline function cap_residual_and_slope(μ, τII_trial, P_trial, ηve, KΔt, k, c, kq, η_reg, geom)
    ϵ²  = eps(typeof(μ))^2
    s   = inv(one(μ) + ηve * μ)
    τII = τII_trial * s
    dτII = -ηve * τII * s

    # Flow-potential branch fixes how the pressure returns.
    P_s = P_trial + KΔt * kq * μ * τII
    shear_Q = P_s + kq * τII ≥ geom.p_q
    P, dP = if shear_Q
        P_s, KΔt * kq * τII * s
    else
        den = one(μ) + KΔt * μ
        (P_trial + KΔt * μ * geom.p_q) / den, KΔt * (geom.p_q - P_trial) / (den * den)
    end

    # Yield branch fixes the residual.
    F, dF = if P + k * τII ≥ geom.p_y
        τII - k * P - c, dτII - k * dP
    else
        R̂y = √(τII * τII + (P - geom.p_y)^2 + ϵ²)
        geom.a * (R̂y - geom.R_y), geom.a * (τII * dτII + (P - geom.p_y) * dP) / R̂y
    end

    λ, dλ = if shear_Q
        μ * τII, τII * s
    else
        R̂q  = √(τII * τII + (P - geom.p_q)^2 + ϵ²)
        dR̂q = (τII * dτII + (P - geom.p_q) * dP) / R̂q
        μ * R̂q / geom.b, (R̂q + μ * dR̂q) / geom.b
    end

    return F - η_reg * λ, dF - η_reg * dλ, τII, P, λ
end

"""
    cap_return_map(τII_trial, P_trial, ηve, KΔt, k, kq, c, pT, η_reg, Val(maxiter))
        -> (; τII, P, λ, iters, converged)

Return-map the trial state onto the tensile-cap yield surface, solving the
Perzyna consistency condition for the return parameter `μ = 2λBτ`.

`KΔt` is `Kb * Δt`. `k = sin(ϕ)`, `kq = sin(Ψ)`, `c = C cos(ϕ)`, and `pT ≤ 0` is
the tensile strength. Returns the returned invariant and pressure, the
multiplier `λ`, and whether the local solve converged. The deviatoric return is
radial, so a caller rescales the trial tensor by `τII / τII_trial`.

An elastic trial state returns unchanged with `λ = 0`.

Uses a bracketed Newton iteration with bisection fallback rather than the
paper's Armijo line search. Both guard the same failure — the paper reports that
unguarded local iterations cycle in stress space and never converge even though
the surface is smooth — but a sign-change bracket is strictly more robust for a
scalar unknown and needs no tuning constants. The bracket is `[0, μ_hi]`:
`r(0) = F_trial > 0` whenever the trial state yields, and `μ_hi` is found by
doubling from the scale `1/(ηve + KΔt)`.

Both loops are fixed-trip with a guard rather than `while` on a residual test, so
the iteration count does not depend on the data. That keeps it launchable inside
a KernelAbstractions kernel without warp divergence, at the cost of always paying
the full trip count; a kernel-side version should revisit that trade.
"""
@inline function cap_return_map(
        τII_trial, P_trial, ηve, KΔt, k, kq, c, pT, η_reg, ::Val{maxiter} = Val(50)
    ) where {maxiter}
    FP   = typeof(τII_trial)
    geom = cap_geometry(k, kq, c, pT)
    F0   = cap_yield_function(τII_trial, P_trial, k, c, geom)

    if F0 ≤ 0
        return (; τII = τII_trial, P = P_trial, λ = zero(FP), iters = 0, converged = true)
    end

    scale = c + τII_trial + abs(P_trial)
    tol   = √(eps(FP)) * scale

    # Bracket: r(0) = F0 > 0, expand μ_hi by doubling until the residual changes
    # sign.  32 doublings span 10 orders of magnitude from the initial scale.
    μ_lo, r_lo = zero(FP), F0
    μ_hi = inv(ηve + KΔt)
    r_hi = first(cap_residual_and_slope(μ_hi, τII_trial, P_trial, ηve, KΔt, k, c, kq, η_reg, geom))
    bracketed = r_hi ≤ 0
    for _ in 1:32
        if !bracketed
            μ_lo, r_lo = μ_hi, r_hi
            μ_hi *= 2
            r_hi = first(
                cap_residual_and_slope(μ_hi, τII_trial, P_trial, ηve, KΔt, k, c, kq, η_reg, geom)
            )
            bracketed = r_hi ≤ 0
        end
    end

    μ    = (μ_lo + μ_hi) / 2
    τII  = τII_trial
    P    = P_trial
    λ    = zero(FP)
    done = false
    iters = 0
    for it in 1:maxiter
        if !done
            r, dr, τII, P, λ = cap_residual_and_slope(
                μ, τII_trial, P_trial, ηve, KΔt, k, c, kq, η_reg, geom
            )
            r > 0 ? (μ_lo = μ) : (μ_hi = μ)
            μ_newton = μ - r / dr
            # Take Newton only where it stays inside the bracket; bisect otherwise.
            μ = (isfinite(μ_newton) && μ_lo < μ_newton < μ_hi) ?
                μ_newton : (μ_lo + μ_hi) / 2
            done = abs(r) ≤ tol
            iters = it
        end
    end

    return (; τII, P, λ, iters, converged = done && bracketed)
end
