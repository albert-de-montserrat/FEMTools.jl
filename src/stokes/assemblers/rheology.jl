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
regularized formula `λ = F / (ηve + η_reg - Kb Δt ∂Q/∂P ∂F/∂P)`.
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

