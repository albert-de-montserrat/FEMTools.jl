"""
    assemble_shear_heating!(Φ, v, P, el2n_v, el2nP, geo_v, phases, τ_old, plastic,
                            η, G, Δt, Nq, NqP, ∂N∂ξ_v, γ_history,
                            Val(NV), Val(NP), backend, workgroup, Pf=nothing) -> Φ

Fill the `nq × nels` array `Φ` with the shear dissipation

    Φ = τ : (ε̇ − ε̇ᵉˡ),    ε̇ᵉˡ = (τ − τ_old) / (2GΔt),

at every velocity quadrature point, in plane strain (`v` has two components)
or in 3-D (three). `τ` is the deviatoric stress the momentum
residual evaluates for the current velocity and pressure, including any
plastic correction, so `Φ` is the viscous plus plastic shear dissipation; the
elastically stored power is excluded. Volumetric plastic work is not included.
`Pf` is the optional fluid pressure the yield model subtracts, as in the
momentum residual.
"""
function assemble_shear_heating!(
        Φ, v::NTuple, P, el2n_v, el2nP, geo_v, phases, τ_old, plastic,
        η, G, Δt, Nq, NqP, ∂N∂ξ_v, γ_history, ::Val{NV}, ::Val{NP}, backend, workgroup,
        Pf = nothing,
    ) where {NV, NP}
    shear_heating_kernel!(backend, workgroup)(
        Φ, v, P, el2n_v, el2nP, geo_v, phases, τ_old, plastic, γ_history,
        η, G, Δt, Nq, NqP, ∂N∂ξ_v, Val(NV), Val(NP), Pf;
        ndrange = size(Φ, 2),
    )
    KA.synchronize(backend)
    return Φ
end

@kernel function shear_heating_kernel!(
        Φ, @Const(v), @Const(P), @Const(el2n_v), @Const(el2nP), @Const(geo_v),
        @Const(phases), @Const(τ_old), @Const(plastic), @Const(γ_history),
        η, G, Δt, Nq, NqP, ∂N∂ξ_v, ::Val{NV}, ::Val{NP}, @Const(Pf),
    ) where {NV, NP}
    iel = @index(Global)
    Φ_el = shear_heating_element(
        v, P, el2n_v, el2nP, geo_v, phases, τ_old, plastic, γ_history,
        η, G, Δt, Nq, NqP, ∂N∂ξ_v, iel, Val(NV), Val(NP), Pf,
    )
    for q in eachindex(Φ_el)
        Φ[q, iel] = Φ_el[q]
    end
end

@inline function shear_heating_element(
        v, P, el2n_v, el2nP, geo_v, phases, τ_old, plastic, γ_history,
        η, G, Δt, Nq, NqP, ∂N∂ξ_v, iel, ::Val{NV}, ::Val{NP}, Pf = nothing,
    ) where {NV, NP}
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP, iel, Val(NP))
    geo_el = element_geometry(geo_v, iel, ∂N∂ξ_v)
    NQ = quadrature_points_val(geo_el)
    vloc = map(vc -> _gather_local(vc, local_nodes_v, Val(NV)), v)
    P_loc = _gather_local(P, local_nodes_P, Val(NP))
    Pf_loc = _gather_or_nothing(Pf, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, iel, Val(NV), NQ)
    γ_loc = _gather_history(γ_history, iel, NQ)
    phase_loc = _gather_phase(phases, local_nodes_v, iel, Val(NV))
    T = promote_type(map(eltype, vloc)...)
    return ntuple(NQ) do q
        ∂N∂x, _ = geo_el[q]
        Nv = Nq[q]
        τ_o = old_stress_at_ip(Nv, τ_old_loc, T, q, _stress_component_count(v))
        Peq = _effective_pressure(dot(NqP[q], P_loc), _fluid_pressure_at_ip(NqP[q], Pf_loc))
        τ, _ = deviatoric_stress_and_pressure(
            vloc, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_o, Peq, plastic,
            _history_at_ip(γ_loc, q),
        )
        _, inv_2Gdt = viscoelastic_coefficients_phase(Nv, η, G, phase_loc, Δt)
        shear_dissipation(τ, τ_o, map(vc -> ∂N∂x' * vc, vloc), inv_2Gdt)
    end
end

# Plane strain: the stress is deviatoric with τzz = −(τxx + τyy) and ε̇zz = 0,
# so the stress power needs only the in-plane terms, while the elastic power
# includes the out-of-plane stress.
@inline function shear_dissipation((τxx, τyy, τxy), (τxx_o, τyy_o, τxy_o), ∇v::NTuple{2, Any}, inv_2Gdt)
    ∇vx, ∇vy = ∇v
    τzz, τzz_o = -(τxx + τyy), -(τxx_o + τyy_o)
    power = τxx * ∇vx[1] + τyy * ∇vy[2] + τxy * (∇vx[2] + ∇vy[1])
    elastic = (
        τxx * (τxx - τxx_o) + τyy * (τyy - τyy_o) + τzz * (τzz - τzz_o) +
            2 * τxy * (τxy - τxy_o)
    ) * inv_2Gdt
    return power - elastic
end

# 3-D, in assembler order (xx, yy, zz, xy, xz, yz); shear terms count twice.
@inline function shear_dissipation(
        (τxx, τyy, τzz, τxy, τxz, τyz), (τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o),
        ∇v::NTuple{3, Any}, inv_2Gdt,
    )
    ∇vx, ∇vy, ∇vz = ∇v
    power = τxx * ∇vx[1] + τyy * ∇vy[2] + τzz * ∇vz[3] +
        τxy * (∇vx[2] + ∇vy[1]) + τxz * (∇vx[3] + ∇vz[1]) + τyz * (∇vy[3] + ∇vz[2])
    elastic = (
        τxx * (τxx - τxx_o) + τyy * (τyy - τyy_o) + τzz * (τzz - τzz_o) +
            2 * (τxy * (τxy - τxy_o) + τxz * (τxz - τxz_o) + τyz * (τyz - τyz_o))
    ) * inv_2Gdt
    return power - elastic
end
