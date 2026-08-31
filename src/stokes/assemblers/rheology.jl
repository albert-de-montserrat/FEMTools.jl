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
@inline zero_old_stress(::Type{T}, ::Val{Nτ}) where {T, Nτ} = ntuple(_ -> zero(T), Val(Nτ))
@inline old_stress_component_at_ip(_, τ::Number) = τ
@inline old_stress_component_at_ip(Nv, τ) = dot(Nv, τ)
"""
    IntegrationPointStress(τ::NTuple)
    IntegrationPointStress(τxx, τyy, τxy, ...)

Old deviatoric-stress components stored at integration points for viscoelastic
memory. Each component is an `NQ × nels` matrix (integration-point index ×
element index), ordered `(τxx, τyy, τxy)` in plane strain and
`(τxx, τyy, τzz, τxy, τxz, τyz)` in three dimensions. Used by
`_gather_old_stress` to recover the stress at quadrature point `q` without
going through nodal interpolation.
"""
struct IntegrationPointStress{Nτ, T}
    τ::NTuple{Nτ, T}
end
IntegrationPointStress(τxx, τyy, τxy, rest...) =
    IntegrationPointStress((τxx, τyy, τxy, rest...))

"""
    IntegrationPointStressOutput(τ, iel)

Scratch buffer for writing the *current* deviatoric stress to integration
points during momentum-residual assembly. `iel` pins the buffer to a specific
element so that `store_stress_at_ip!` can index `τ[c][q, iel]` directly.
"""
struct IntegrationPointStressOutput{Nτ, T}
    τ::NTuple{Nτ, T}
    iel::Int
end
@inline old_stress_at_ip(_, ::Nothing, ::Type{T}, _, ::Val{Nτ}) where {T, Nτ} =
    zero_old_stress(T, Val(Nτ))
@inline old_stress_at_ip(Nv, τ_old::NTuple{Nτ}, ::Type, _, ::Val{Nτ}) where {Nτ} =
    map(τ -> old_stress_component_at_ip(Nv, τ), τ_old)
@inline old_stress_at_ip(_, τ_old::IntegrationPointStress{Nτ}, ::Type, q, ::Val{Nτ}) where {Nτ} =
    ntuple(c -> τ_old.τ[c][q], Val(Nτ))
@inline store_stress_at_ip!(::Nothing, _, _...) = nothing
@inline function store_stress_at_ip!(
        τ_store::IntegrationPointStressOutput{Nτ}, q, τij::Vararg{Any, Nτ},
    ) where {Nτ}
    ntuple(Val(Nτ)) do c
        τ_store.τ[c][q, τ_store.iel] = τij[c]
        nothing
    end
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

@inline function second_invariant(A::Union{SVector{3}, Tuple{Any, Any, Any}})
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
@inline function deviatoric_stress(
        v::Tuple{<:SVector, <:SVector}, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old::NTuple{3},
    )
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
@inline function deviatoric_stress(
        v::Tuple{<:SVector, <:SVector}, ∂N∂x, Nv, η, G, phase_loc, Δt,
        τ_old::NTuple{3}, Pq, plastic::DruckerPrager,
    )
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


"""
    second_invariant(A::NTuple{6}) -> τII

Compute the second invariant `τII = √J₂` of a full three-dimensional symmetric
deviatoric tensor stored as `(τxx, τyy, τzz, τxy, τxz, τyz)`:

    τII = √((τxx² + τyy² + τzz²) / 2 + τxy² + τxz² + τyz²)

Unlike the plane-strain method, `τzz` is carried explicitly rather than being
reconstructed from the in-plane components. The same `eps²` floor keeps the
square root differentiable at zero stress.
"""
@inline function second_invariant(
        A::Union{SVector{6}, Tuple{Any, Any, Any, Any, Any, Any}},
    )
    FT = typeof(real(A[1]))
    return √((A[1]^2 + A[2]^2 + A[3]^2) / 2 + A[4]^2 + A[5]^2 + A[6]^2 + eps(FT)^2)
end

"""
    deviatoric_stress(v::NTuple{3}, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old)
        -> (τxx, τyy, τzz, τxy, τxz, τyz)

Compute the three-dimensional viscoelastic deviatoric stress at a quadrature
point.

`v` holds one element velocity `SVector` per spatial direction and `τ_old` the
six stress-history components in the same order as the result. Pass a tuple of
zeros for `τ_old` on the first time step. This method has no yield criterion.
"""
@inline function deviatoric_stress(
        v::Tuple{<:SVector, <:SVector, <:SVector}, ∂N∂x, Nv, η, G, phase_loc, Δt,
        τ_old::NTuple{6},
    )
    # Indexed literally rather than through `ntuple`: `v` is heterogeneous
    # while one direction is differentiated and the others are not, and a
    # closure index reaches `v[i]` non-constant, which widens every gradient
    # to a `Union` and makes the enclosing kernel uncompilable on GPU
    # back-ends.
    ∇v = (∂N∂x' * v[1], ∂N∂x' * v[2], ∂N∂x' * v[3])
    tr = (∇v[1][1] + ∇v[2][2] + ∇v[3][3]) / 3
    # `tr` reads all three directions, so it carries the widest element type in
    # `v`; holding the shear components to it keeps the strain rate on a single
    # type. `εyz` reads neither the trace nor, when `v` is mixed, the direction
    # being differentiated. Left narrower, it splits the Drucker-Prager return
    # of the caller below: the corrected branch promotes that component and the
    # unyielded branch returns it as is, so the two disagree and the stress type
    # widens to a `Union` that no GPU back-end can compile.
    ε = (
        ∇v[1][1] - tr, ∇v[2][2] - tr, ∇v[3][3] - tr,
        oftype(tr, (∇v[1][2] + ∇v[2][1]) / 2),
        oftype(tr, (∇v[1][3] + ∇v[3][1]) / 2),
        oftype(tr, (∇v[2][3] + ∇v[3][2]) / 2),
    )
    ηve, inv_2Gdt = viscoelastic_coefficients_phase(Nv, η, G, phase_loc, Δt)
    return map((εij, τij_o) -> 2 * ηve * (εij + τij_o * inv_2Gdt), ε, τ_old)
end

"""
    deviatoric_stress(v::NTuple{3}, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq,
                      plastic::DruckerPrager) -> (τxx, τyy, τzz, τxy, τxz, τyz)

Compute the three-dimensional elasto-viscoplastic deviatoric stress at a
quadrature point with Drucker-Prager return mapping.

The yield function, plastic multiplier, and regularization match the
plane-strain method. The flow direction is the radial return
`∂Q/∂τᵢⱼ = τᵢⱼ / (2 τII)`, which for the stored off-diagonal components — each
of which stands for two tensor entries — becomes `τᵢⱼ / τII`. The correction is
traceless because the normal components of `∂Q/∂τ` sum to `(τxx+τyy+τzz)/(2τII) = 0`.

The plane-strain method instead differentiates `τII` with respect to the two
free in-plane components, with `τzz = −τxx − τyy` slaved to them, so its flow
direction is not radial. The two return maps therefore differ even when the
three-dimensional kinematics reduce to plane strain.
"""
@inline function deviatoric_stress(
        v::Tuple{<:SVector, <:SVector, <:SVector}, ∂N∂x, Nv, η, G, phase_loc, Δt,
        τ_old::NTuple{6}, Pq,
        plastic::DruckerPrager,
    )
    τij = deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old)
    ηve = effective_viscosity_phase(Nv, η, G, phase_loc, Δt)

    # Interpolate per-phase plastic parameters to the quadrature point.
    cosϕ  = interp2ip_phase(Nv, plastic.cosϕ,  phase_loc)
    sinϕ  = interp2ip_phase(Nv, plastic.sinϕ,  phase_loc)
    sinΨ  = interp2ip_phase(Nv, plastic.sinΨ,  phase_loc)
    C     = interp2ip_phase(Nv, plastic.C,     phase_loc)
    η_reg = interp2ip_phase(Nv, plastic.η_reg, phase_loc)
    Kb    = interp2ip_phase(Nv, plastic.Kb,    phase_loc)

    τII      = second_invariant(τij)
    τII_safe = τII + eps(typeof(τII))^2
    F        = τII - cosϕ * C - sinϕ * Pq
    ∂F∂P     = -sinϕ
    ∂Q∂P     = -sinΨ
    # Normal components carry a factor 1/2 that the stored shear components,
    # which each represent two tensor entries, do not.
    ∂Q∂τ = (
        τij[1] / (2 * τII_safe), τij[2] / (2 * τII_safe), τij[3] / (2 * τII_safe),
        τij[4] / τII_safe, τij[5] / τII_safe, τij[6] / τII_safe,
    )

    λ = F > 0 ? F / (ηve + η_reg + Kb * Δt * ∂Q∂P * ∂F∂P) : zero(F)

    return λ > 0 ? map((τ, ∂q) -> τ - 2 * ηve * λ * ∂q, τij, ∂Q∂τ) : τij
end
