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
    ηq = interp2ip_phase(Nv, η, phase_loc)
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
    IntegrationPointStressOutput(τ, P, iel)

Scratch buffer for writing the *current* deviatoric stress to integration
points during momentum-residual assembly. `iel` pins the buffer to a specific
element so that `store_stress_at_ip!` can index `τ[c][q, iel]` directly.

`P` optionally receives the plastically corrected pressure returned by the
tensile-cap return map, the value the momentum balance actually uses. It is
`nothing` when the caller asks only for stress, and for yield models whose
return map leaves pressure unchanged it simply records the trial pressure.
"""
struct IntegrationPointStressOutput{Nτ, T, TP}
    τ::NTuple{Nτ, T}
    P::TP
    iel::Int
end
IntegrationPointStressOutput(τ::NTuple, iel) = IntegrationPointStressOutput(τ, nothing, iel)

"""
    IntegrationPointPlasticHistory(λ, ε̇pl, εpl)

Per-integration-point Drucker--Prager plastic state owned by a Stokes driver or
example. The arrays are indexed `(q, element)`. `λ` is the plastic multiplier
and `ε̇pl` the plastic strain-rate invariant; momentum assembly overwrites both
on every call. `εpl` is the accumulated plastic strain, advanced only by
[`update_plastic_history!`](@ref).
"""
struct IntegrationPointPlasticHistory{Tλ, Tr, Tε}
    λ::Tλ
    ε̇pl::Tr
    εpl::Tε
end

"""Element-pinned, isbits view used by a per-element assembly kernel."""
struct IntegrationPointPlasticHistoryOutput{Tλ, Tr}
    λ::Tλ
    ε̇pl::Tr
    iel::Int
end

@inline IntegrationPointPlasticHistoryOutput(history::IntegrationPointPlasticHistory, iel) =
    IntegrationPointPlasticHistoryOutput(history.λ, history.ε̇pl, Int(iel))

struct IntegrationPointPlasticMultiplierOutput{T}
    λ::T
    iel::Int
end

@inline store_plastic_multiplier_at_ip!(::Nothing, _, _, _) = nothing

@inline function store_plastic_multiplier_at_ip!(
        history::IntegrationPointPlasticHistoryOutput, q, λ, ε̇pl,
    )
    history.λ[q, history.iel] = λ
    history.ε̇pl[q, history.iel] = ε̇pl
    return nothing
end

@inline function store_plastic_multiplier_at_ip!(
        output::IntegrationPointPlasticMultiplierOutput, q, λ, _,
    )
    output.λ[q, output.iel] = λ
    return nothing
end

@kernel function _update_plastic_history_kernel!(εpl, ε̇pl, Δt)
    I = @index(Global, NTuple)
    εpl[I...] += Δt * ε̇pl[I...]
end

"""
    update_plastic_history!(history, Δt; workgroup=256)

Accept one step into the accumulated plastic strain, `εpl += Δt·ε̇pl`, using the
strain rate stored by the last momentum assembly. Call it exactly once per
converged step. The update is backend-neutral and preserves the array element
type.
"""
function update_plastic_history!(
        history::IntegrationPointPlasticHistory, Δt; workgroup = 256,
    )
    size(history.λ) == size(history.ε̇pl) == size(history.εpl) ||
        throw(DimensionMismatch("λ, ε̇pl, and εpl must have equal sizes"))
    Δt >= 0 || throw(ArgumentError("Δt must be nonnegative"))
    backend = KA.get_backend(history.εpl)
    _update_plastic_history_kernel!(backend, workgroup)(
        history.εpl, history.ε̇pl, Δt; ndrange = size(history.εpl),
    )
    KA.synchronize(backend)
    return history
end

@inline old_stress_at_ip(_, ::Nothing, ::Type{T}, _, ::Val{Nτ}) where {T, Nτ} =
    zero_old_stress(T, Val(Nτ))
@inline old_stress_at_ip(Nv, τ_old::NTuple{Nτ}, ::Type, _, ::Val{Nτ}) where {Nτ} =
    map(τ -> old_stress_component_at_ip(Nv, τ), τ_old)
@inline old_stress_at_ip(_, τ_old::IntegrationPointStress{Nτ}, ::Type, q, ::Val{Nτ}) where {Nτ} =
    ntuple(c -> τ_old.τ[c][q], Val(Nτ))
@inline store_stress_at_ip!(::Nothing, _, _...) = nothing
# `values` holds the `Nτ` stress components, optionally followed by the
# corrected pressure.
@inline function store_stress_at_ip!(
        τ_store::IntegrationPointStressOutput{Nτ}, q, values::Vararg{Any, M},
    ) where {Nτ, M}
    M == Nτ || M == Nτ + 1 || throw(ArgumentError("expected the stress components and an optional pressure"))
    ntuple(Val(Nτ)) do c
        τ_store.τ[c][q, τ_store.iel] = values[c]
        nothing
    end
    M > Nτ && _store_pressure_at_ip!(τ_store.P, q, τ_store.iel, values[M])
    return nothing
end
@inline _store_pressure_at_ip!(::Nothing, _, _, _) = nothing
@inline function _store_pressure_at_ip!(P_store, q, iel, P)
    P_store[q, iel] = P
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
    FT = typeof(real(A[1]))
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
    tr = (εxx + εyy) / 3

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

# Fluid pressure enters only the yield functions: they see the effective
# pressure P − Pf, and a pressure the return map corrects is shifted back by Pf,
# so the momentum balance keeps the total pressure. `nothing` means Pf = 0.
@inline _fluid_pressure_at_ip(_, ::Nothing) = nothing
@inline _fluid_pressure_at_ip(NPq, Pf_loc) = dot(NPq, Pf_loc)
@inline _effective_pressure(Pq, ::Nothing) = Pq
@inline _effective_pressure(Pq, Pfq) = Pq - Pfq
@inline _total_pressure(Pe, ::Nothing) = Pe
@inline _total_pressure(Pe, Pfq) = Pe + Pfq

# Internal momentum path. Existing stress-only API stays unchanged.
@inline deviatoric_stress_and_pressure(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, ::Nothing) =
    (deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old), Pq)

@inline deviatoric_stress_and_pressure(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic) =
    deviatoric_stress_and_pressure(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic, nothing)

# `γ` is the accepted cap history; only `DruckerPragerCap` reads it.
@inline deviatoric_stress_and_pressure(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic, γ) =
    (deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic), Pq)

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
@inline function _deviatoric_stress_with_multiplier(
        v::Tuple{<:SVector, <:SVector}, ∂N∂x, Nv, η, G, phase_loc, Δt,
        τ_old::NTuple{3}, Pq, plastic::DruckerPrager,
    )
    vxloc, vyloc = v
    ∇vx = ∂N∂x' * vxloc
    ∇vy = ∂N∂x' * vyloc

    εxx = ∇vx[1]
    εyy = ∇vy[2]
    εxy = (∇vx[2] + ∇vy[1]) / 2
    tr = (εxx + εyy) / 3

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
    cosϕ = interp2ip_phase(Nv, plastic.cosϕ, phase_loc)
    sinϕ = interp2ip_phase(Nv, plastic.sinϕ, phase_loc)
    sinΨ = interp2ip_phase(Nv, plastic.sinΨ, phase_loc)
    C = interp2ip_phase(Nv, plastic.C, phase_loc)
    η_reg = interp2ip_phase(Nv, plastic.η_reg, phase_loc)
    Kb = interp2ip_phase(Nv, plastic.Kb, phase_loc)

    # Drucker-Prager yield function.
    # second_invariant returns τxx²+τyy²+τzz²+2τxy² = 2J₂, so τII = sqrt(J₂) = sqrt(SI/2).
    τII = second_invariant(τij)
    τII_safe = τII + eps(typeof(τII))^2
    F = τII - cosϕ * C - sinϕ * Pq
    ∂F∂P = -sinϕ

    # Derivatives of the plane-strain invariant with τzz = -τxx - τyy.
    ∂Q∂τxx = (2 * τxx + τyy) / (2 * τII_safe)
    ∂Q∂τyy = (τxx + 2 * τyy) / (2 * τII_safe)
    ∂Q∂τxy = τxy / τII_safe
    ∂Q∂τ = ∂Q∂τxx, ∂Q∂τyy, ∂Q∂τxy
    ∂Q∂P = -sinΨ

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

    return τij, λ, ∂Q∂τ
end

"""
    plastic_strain_rate_invariant(λ, ∂Q∂τ) -> ε̇pl

Convert the Drucker--Prager multiplier and plane-strain deviatoric flow
direction into the scalar `J₂`-equivalent plastic strain-rate invariant used
by the history update.  The out-of-plane direction is reconstructed as
`∂Q∂τzz = -(∂Q∂τxx + ∂Q∂τyy)` and the shear contribution is counted twice.
"""
@inline function plastic_strain_rate_invariant(λ, ∂Q∂τ::NTuple{3})
    ∂Q∂τxx, ∂Q∂τyy, ∂Q∂τxy = ∂Q∂τ
    ∂Q∂τzz = -∂Q∂τxx - ∂Q∂τyy
    flow_norm = sqrt(
        (2 / 3) * (∂Q∂τxx^2 + ∂Q∂τyy^2 + ∂Q∂τzz^2 + 2 * ∂Q∂τxy^2),
    )
    return abs(λ) * flow_norm
end

"""Return the Drucker--Prager plastic multiplier at one quadrature point."""
@inline plastic_multiplier(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic::DruckerPrager) =
    _deviatoric_stress_with_multiplier(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic)[2]

@inline function plastic_strain_rate_invariant(
        v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic::DruckerPrager,
    )
    _, λ, ∂Q∂τ = _deviatoric_stress_with_multiplier(
        v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic,
    )
    return plastic_strain_rate_invariant(λ, ∂Q∂τ)
end

@inline deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic::DruckerPrager) =
    first(_deviatoric_stress_with_multiplier(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic))


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
@inline function _deviatoric_stress_with_multiplier(
        v::Tuple{<:SVector, <:SVector, <:SVector}, ∂N∂x, Nv, η, G, phase_loc, Δt,
        τ_old::NTuple{6}, Pq, plastic::DruckerPrager,
    )
    τij = deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old)
    ηve = effective_viscosity_phase(Nv, η, G, phase_loc, Δt)

    # Interpolate per-phase plastic parameters to the quadrature point.
    cosϕ = interp2ip_phase(Nv, plastic.cosϕ, phase_loc)
    sinϕ = interp2ip_phase(Nv, plastic.sinϕ, phase_loc)
    sinΨ = interp2ip_phase(Nv, plastic.sinΨ, phase_loc)
    C = interp2ip_phase(Nv, plastic.C, phase_loc)
    η_reg = interp2ip_phase(Nv, plastic.η_reg, phase_loc)
    Kb = interp2ip_phase(Nv, plastic.Kb, phase_loc)

    τII = second_invariant(τij)
    τII_safe = τII + eps(typeof(τII))^2
    F = τII - cosϕ * C - sinϕ * Pq
    ∂F∂P = -sinϕ
    ∂Q∂P = -sinΨ
    # Normal components carry a factor 1/2 that the stored shear components,
    # which each represent two tensor entries, do not.
    ∂Q∂τ = (
        τij[1] / (2 * τII_safe), τij[2] / (2 * τII_safe), τij[3] / (2 * τII_safe),
        τij[4] / τII_safe, τij[5] / τII_safe, τij[6] / τII_safe,
    )

    λ = F > 0 ? F / (ηve + η_reg + Kb * Δt * ∂Q∂P * ∂F∂P) : zero(F)

    τij = λ > 0 ? map((τ, ∂q) -> τ - 2 * ηve * λ * ∂q, τij, ∂Q∂τ) : τij
    return τij, λ, ∂Q∂τ
end

"""
    plastic_strain_rate_invariant(λ, ∂Q∂τ::NTuple{6}) -> ε̇pl

Three-dimensional counterpart of the plane-strain method, with the flow
direction stored as `(xx, yy, zz, xy, xz, yz)` and each shear contribution
counted twice.
"""
@inline function plastic_strain_rate_invariant(λ, ∂Q∂τ::NTuple{6})
    xx, yy, zz, xy, xz, yz = ∂Q∂τ
    return abs(λ) * sqrt((2 / 3) * (xx^2 + yy^2 + zz^2 + 2 * (xy^2 + xz^2 + yz^2)))
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

"""Return cohesion after linear strain softening at one integration point."""
@inline softened_cohesion(C, C_min, H_C, γ) = clamp(C + H_C * γ, C_min, C)
@inline _cohesion_at_history(C, C_min, H_C, ::Nothing) = C
@inline _cohesion_at_history(C, C_min, H_C, γ::Real) = softened_cohesion(C, C_min, H_C, γ)

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
    cap_invariants(s, p, k, kq, c, pT) -> (F, Aτ, Ap)
    cap_invariants(plastic::DruckerPragerCap, phase, s, p, γ) -> (F, Aτ, Ap)

Scalar coefficients for the coupled tensile-cap return: `Aτ = (∂Q/∂s)/2`
and `Ap = -∂Q/∂p`, matching JustRelax's invariant convention. Yield and
potential branches are selected independently. The material overload applies
bounded cohesion softening from accepted history `γ` (`nothing` means no history).
All inputs are plain numeric values; angles in `DruckerPragerCap` are radians.
"""
@inline function cap_invariants(s, p, k, kq, c, pT)
    geom = cap_geometry(k, kq, c, pT)
    F = cap_yield_function(s, p, k, c, geom)
    if p + kq * s ≥ geom.p_q
        return F, one(s) / 2, kq
    end
    Rq = hypot(s, p - geom.p_q)
    iszero(Rq) && return F, zero(s), zero(p)
    return F, geom.b * s / (2 * Rq), -geom.b * (p - geom.p_q) / Rq
end

@inline function cap_invariants(plastic::DruckerPragerCap, phase, s, p, γ)
    C = _cohesion_at_history(plastic.C[phase], plastic.C_min[phase], plastic.H_C[phase], γ)
    return cap_invariants(
        s, p, plastic.sinϕ[phase], plastic.sinΨ[phase],
        C * plastic.cosϕ[phase], plastic.pT[phase]
    )
end

@inline function _cap_residual(x, s_trial, p_trial, ηve, KΔt, k, kq, c, pT, η_reg)
    s, p, λ = x
    F, Aτ, Ap = cap_invariants(s, p, k, kq, c, pT)
    return SVector(s - s_trial + 2 * ηve * λ * Aτ, p - p_trial - KΔt * λ * Ap, F - η_reg * λ)
end

"""
    cap_return_map(s_trial, p_trial, ηve, KΔt, k, kq, c, pT, η_reg; maxiter = 40)
        -> (; τII, P, λ, Aτ, Ap, converged)

Coupled local return of the tensile cap (Popov et al. 2025, Eq. 42), solving
for `x = (s, p, λ)`:

    s - s_trial + 2 ηve λ Aτ(s, p) = 0
    p - p_trial - KΔt λ Ap(s, p)   = 0
    F(s, p) - η_reg λ              = 0

with `(F, Aτ, Ap)` from [`cap_invariants`](@ref). `KΔt` is `Kb * Δt`,
`k = sin(ϕ)`, `kq = sin(Ψ)`, `c = C cos(ϕ)`, and `pT ≤ 0` is the tensile
strength (compression-positive pressure). An elastic trial state (`F ≤ 0`)
returns unchanged with `λ = 0`.

Newton on residuals scaled by `max(|s_trial|, |p_trial|, |F_trial|, eps)` with a
ForwardDiff 3×3 Jacobian, converging at a scaled infinity norm `≤ 100 eps`.
Steps backtrack by halving (at most 24 times) until `s ≥ 0`, `λ ≥ 0`, the
residual is finite, and its squared norm satisfies an Armijo decrease. Any
failure — non-positive or non-finite `ηve`/`KΔt`, a non-finite step, a failed
line search, or an exhausted budget — returns `converged = false`.

A converged solve takes one final full Newton step, which leaves the value
unchanged to round-off but makes derivatives of the result through dual
numbers equal the implicit-function derivative.
"""
@inline function cap_return_map(
        s_trial, p_trial, ηve, KΔt, k, kq, c, pT, η_reg; maxiter = 40,
    )
    T = promote_type(typeof(s_trial), typeof(p_trial))
    F, Aτ, Ap = cap_invariants(s_trial, p_trial, k, kq, c, pT)
    x = SVector{3, T}(s_trial, p_trial, zero(T))
    F ≤ 0 && return (; τII = x[1], P = x[2], λ = x[3], Aτ, Ap, converged = true)
    failed = (; τII = x[1], P = x[2], λ = x[3], Aτ, Ap, converged = false)
    (isfinite(KΔt) && KΔt > 0 && isfinite(ηve) && ηve > 0) || return failed

    scale = max(abs(s_trial), abs(p_trial), abs(F), eps(real(T)))
    tol = 100 * eps(real(T))
    residual = y -> _cap_residual(y, s_trial, p_trial, ηve, KΔt, k, kq, c, pT, η_reg) / scale
    r = residual(x)
    converged = maximum(abs, r) ≤ tol
    for _ in 1:maxiter
        converged && break
        step = ForwardDiff.jacobian(residual, x) \ r
        all(isfinite, step) || return failed
        α = one(real(T))
        accepted = false
        for _ in 1:24
            candidate = x - α * step
            if candidate[1] ≥ 0 && candidate[3] ≥ 0
                r_new = residual(candidate)
                if all(isfinite, r_new) && sum(abs2, r_new) ≤ (1 - α / 10_000) * sum(abs2, r)
                    x, r = candidate, r_new
                    accepted = true
                    break
                end
            end
            α /= 2
        end
        accepted || return failed
        converged = maximum(abs, r) ≤ tol
    end
    converged || return failed
    x -= ForwardDiff.jacobian(residual, x) \ r
    _, Aτ, Ap = cap_invariants(x[1], x[2], k, kq, c, pT)
    return (; τII = x[1], P = x[2], λ = x[3], Aτ, Ap, converged)
end

"""
    cap_local_update(τ_trial, P_trial, ηve, KΔt, k, kq, c, pT, η_reg)
        -> (; τ, P, γdot, θdot)

Physical tensile-cap update at one integration point: the corrected deviatoric
stress tuple `τ`, corrected pressure `P`, and plastic history rates. With
`(s, p, λ)` from [`cap_return_map`](@ref) and `s_trial = second_invariant(τ_trial)`,

    εvp_ij = λ Aτ τ_trial_ij / s_trial      τ = τ_trial - 2 ηve εvp
    θdot   = λ Ap                           P = P_trial + KΔt θdot
    γdot   = λ Aτ = second_invariant(εvp)

The deviatoric return is radial, so `second_invariant(τ) == s`. A failed local
solve returns NaN everywhere so that the global solver's non-finite residual
check stops the run; a failed iterate is never returned as physical stress.
"""
@inline function cap_local_update(τ_trial, P_trial, ηve, KΔt, k, kq, c, pT, η_reg)
    s_trial = second_invariant(τ_trial)
    ret = cap_return_map(s_trial, P_trial, ηve, KΔt, k, kq, c, pT, η_reg)
    λ = ret.converged ? ret.λ : oftype(ret.λ, NaN)
    γdot = λ * ret.Aτ
    θdot = λ * ret.Ap
    # second_invariant floors s_trial at eps, so the division is always defined.
    τ = map(t -> t - 2 * ηve * γdot * t / s_trial, τ_trial)
    return (; τ, P = P_trial + KΔt * θdot, γdot, θdot)
end

# Interpolate cap parameters and the (softened) cohesion to one quadrature point
# and run the shared local update. Momentum, pressure, and history rates all use
# this one result.
@inline function _cap_ip_update(
        v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic::DruckerPragerCap, γ,
    )
    τ_trial = deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old)
    ηve, = viscoelastic_coefficients_phase(Nv, η, G, phase_loc, Δt)
    C = _cohesion_at_history(
        interp2ip_phase(Nv, plastic.C, phase_loc),
        interp2ip_phase(Nv, plastic.C_min, phase_loc),
        interp2ip_phase(Nv, plastic.H_C, phase_loc), γ,
    )
    return cap_local_update(
        τ_trial, Pq, ηve,
        interp2ip_phase(Nv, plastic.Kb, phase_loc) * Δt,
        interp2ip_phase(Nv, plastic.sinϕ, phase_loc),
        interp2ip_phase(Nv, plastic.sinΨ, phase_loc),
        C * interp2ip_phase(Nv, plastic.cosϕ, phase_loc),
        interp2ip_phase(Nv, plastic.pT, phase_loc),
        interp2ip_phase(Nv, plastic.η_reg, phase_loc),
    )
end

"""
    deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq,
                      plastic::DruckerPragerCap) -> (τxx, τyy, τxy)

Plane-strain elasto-viscoplastic stress with the globally continuous tensile
cap, from `cap_local_update` without cohesion softening.
"""
@inline deviatoric_stress(
    v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic::DruckerPragerCap,
) = _cap_ip_update(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic, nothing).τ

@inline function deviatoric_stress_and_pressure(
        v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq,
        plastic::DruckerPragerCap, γ::Union{Nothing, Real},
    )
    result = _cap_ip_update(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic, γ)
    return result.τ, result.P
end


"""
    plastic_history_rates(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic, γ)
        -> (γdot, θdot)

Deviatoric and volumetric plastic strain rates at one quadrature point, with
cohesion softened by accepted history `γ`. Zero for non-cap yield models.
"""
@inline plastic_history_rates(_, _, _, _, _, _, _, _, _, ::Nothing, _) = (0, 0)
@inline plastic_history_rates(_, _, _, _, _, _, _, _, _, ::DruckerPrager, _) = (0, 0)
@inline function plastic_history_rates(
        v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic::DruckerPragerCap, γ,
    )
    result = _cap_ip_update(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old, Pq, plastic, γ)
    return result.γdot, result.θdot
end
