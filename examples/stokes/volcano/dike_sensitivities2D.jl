using FEMTools
using StaticArrays
using LinearAlgebra
using ForwardDiff

# ---------------------------------------------------------------------------
# Parameter sensitivities of the adjoint
#
# `solve_stokes_adjoint_dyrel!` transposes the Powell-Hestenes augmented system
# [Rv + W·RP; RP], where W = ∂Rv/∂Pnum · γP/M_P is element-local. Its pressure
# multiplier λP is therefore related to the multiplier of the plain residual
# R = [Rv; RP] by λP_plain = λP + Wᵀλv, and for a parameter m
#
#     dJ/dm = λvᵀ ∂Rv/∂m + λP_plainᵀ ∂RP/∂m,
#
# contracted element by element, with the stress, pressure, plastic-strain and
# temperature history of the step held fixed.
# ---------------------------------------------------------------------------

"""
    element_state(data, iel, Val(NV), Val(NP)) -> NamedTuple

Element-local converged state, frozen history, and geometry of element `iel`,
gathered the way the forward solve's element residuals gather them. The
accepted pressure `P_old` is held at the integration points.
"""
function element_state(data, iel, ::Val{NV}, ::Val{NP}) where {NV, NP}
    nodes_v = FEMTools.local_nodes_of(data.el2n, iel, Val(NV))
    nodes_P = FEMTools.local_nodes_of(data.DoFsP, iel, Val(NP))
    geo_v = element_geometry(data.geo_v, iel, data.∂N∂ξ)
    NQ = FEMTools.quadrature_points_val(geo_v)
    gather_P(field) = FEMTools._gather_local(field, nodes_P, Val(NP))
    return (;
        nodes_v, nodes_P, geo_v, geo_P = element_geometry(data.geo_P, iel),
        v = map(vc -> FEMTools._gather_local(vc, nodes_v, Val(NV)), data.v),
        P = gather_P(data.P), T = gather_P(data.T), T0 = gather_P(data.T0),
        P_old = FEMTools._gather_old_pressure(data.P_old, nodes_P, iel, Val(NP), NQ),
        Pf = gather_P(data.Pf), P_litho = gather_P(data.P_litho), Q = gather_P(data.Q),
        τ_old = FEMTools._gather_old_stress(data.τ_old, nodes_v, iel, Val(NV), NQ),
        phase_v = FEMTools._gather_phase(data.phases_v, nodes_v, iel, Val(NV)),
        phase_P = FEMTools._gather_phase(data.phases_P, nodes_P, iel, Val(NP)),
        Γ = gather_P(data.γP) ./ gather_P(data.M_P),
    )
end

# Momentum residual of one element, stacked by component; `Pnum` is the
# Powell-Hestenes pressure, zero at the converged forward state. The cap's
# cohesion softening is off, so the plastic-strain history does not enter.
element_momentum(e, data, K, plastic, Pnum; η = data.η, G = data.G, Pf = e.Pf) = vcat(
    FEMTools.integrate_momentum_residual(
        e.v, e.P, Pnum, e.T, e.geo_v, e.phase_v,
        η, G, data.α, data.ρ0, K, data.g, data.Tref, data.Δt,
        data.Nq, data.NqP, e.τ_old, plastic, nothing, nothing, nothing, nothing, Pf,
    )...,
)

# With the tensile cap the continuity storage modulus is the bulk modulus `K`.
element_continuity(e, data, K, Q_scale) = FEMTools.integrate_PH_pressure_residual(
    e.v, e.P, e.P_old, e.T, e.T0, Q_scale * e.Q, e.geo_v, e.geo_P, e.phase_P,
    data.α, K, data.Δt, data.NqP; K,
)

# Perturb one material parameter, keeping the other cap parameters independent.
# Inner construction preserves the stored trigonometric coefficients under AD.
function perturb_cap(p::DruckerPragerCap{n}, parameter, s) where {n}
    factor = exp(s)
    ϕ = map((sn, cs) -> atan(sn, cs) * factor, p.sinϕ, p.cosϕ)
    return DruckerPragerCap{n, typeof(s)}(
        parameter == :ϕ ? map(cos, ϕ) : p.cosϕ,
        parameter == :ϕ ? map(sin, ϕ) : p.sinϕ, p.sinΨ,
        parameter == :C ? p.C .* factor : p.C,
        parameter == :pT ? p.pT .* factor : p.pT, p.η_reg,
        parameter == :K ? p.Kb .* factor : p.Kb,
        parameter == :C ? p.C_min .* factor : p.C_min, p.H_C,
    )
end

const DIKE_SENSITIVITY_PARAMETERS = (:K, :Q, :G, :η, :pT, :C, :ϕ, :λ_fluid)
const DIKE_SENSITIVITY_FIELDS = (
    :dlnJ_dlnK, :dlnJ_dlnQ, :dlnJ_dlnG, :dlnJ_dlneta,
    :dlnJ_dlnpT, :dlnJ_dlnC, :dlnJ_dlnphi, :dlnJ_dlambda_fluid,
)

function parameter_contraction(e, data, λv, λP, parameter, s)
    factor = exp(s)
    K = parameter == :K ? data.K .* factor : data.K
    plastic = parameter in (:K, :pT, :C, :ϕ) ? perturb_cap(data.plastic, parameter, s) : data.plastic
    Rv = element_momentum(e, data, K, plastic, nothing;
        G = parameter == :G ? data.G .* factor : data.G,
        η = parameter == :η ? data.η .* factor : data.η,
        Pf = parameter == :λ_fluid ? e.Pf + s * e.P_litho : e.Pf,
    )
    RP = element_continuity(e, data, K, parameter == :Q ? factor : one(s))
    return dot(λv, Rv) + dot(λP, RP)
end

"""
    box_mean_load(coords, el2n, geo_v, ∂N∂ξ, Nq, box, Val(NV)) -> (load, area)

Consistent load `∂J/∂vᵢ = ∫_box Nᵢ dΩ / |box|` of the box-mean objective
`J = ∫_box v dΩ / |box|`, and the box area. Membership is decided per
quadrature point, so a cut element contributes its quadrature-weighted share.
"""
function box_mean_load(coords, el2n, geo_v, ∂N∂ξ, Nq, box, ::Val{NV}) where {NV}
    xmin, xmax, ymin, ymax = box
    load = zeros(length(coords))
    area = 0.0
    for iel in axes(el2n, 2)
        nodes = FEMTools.local_nodes_of(el2n, iel, Val(NV))
        geo_el = element_geometry(geo_v, iel, ∂N∂ξ)
        for q in eachindex(geo_el)
            _, dΩ = geo_el[q]
            xq = sum(Nq[q][a] * coords[nodes[a]] for a in 1:NV)
            (xmin < xq[1] < xmax && ymin < xq[2] < ymax) || continue
            for a in 1:NV
                load[nodes[a]] += Nq[q][a] * dΩ
            end
            area += dΩ
        end
    end
    area > 0 || throw(ArgumentError("the observation box contains no quadrature point"))
    return load ./ area, area
end

"""
    log_sensitivities(data, λv, λP, Val(NV), Val(NP)) -> (derivatives, λP_plain)

Per-element `dJ/dln(parameter)` for K, Q, G, η, |pT|, C and ϕ (angle in
radians), plus `dJ/dλ_fluid`. Pore pressure changes by `δλ_fluid * P_litho`,
with lithostatic pressure held fixed, including when λ_fluid is zero.
K scales the EOS, continuity storage and cap bulk moduli together. Cohesion
and tensile strength are independent; the dilation angle and η_reg stay fixed.
History, current temperature, geometry and boundary values are frozen.
"""
function log_sensitivities(data, λv, λP, ::Val{NV}, ::Val{NP}) where {NV, NP}
    data.ηb == data.K ||
        throw(ArgumentError("the bulk-modulus sensitivity assumes the pressure storage modulus ηb equals K"))
    data.plastic.Kb == data.K ||
        throw(ArgumentError("the bulk-modulus sensitivity assumes the cap's bulk modulus equals K"))
    all(iszero, data.plastic.H_C) ||
        throw(ArgumentError("the adjoint assumes no cohesion softening (H_C = 0)"))
    derivatives = NamedTuple{DIKE_SENSITIVITY_FIELDS}(ntuple(_ -> zeros(size(data.el2n, 2)), 8))
    λP_plain = similar(λP)
    for iel in axes(data.el2n, 2)
        e = element_state(data, iel, Val(NV), Val(NP))
        λv_e = vcat(map(λ -> FEMTools._gather_local(λ, e.nodes_v, Val(NV)), λv)...)
        Bnum = ForwardDiff.jacobian(
            Pnum -> element_momentum(e, data, data.K, data.plastic, Pnum), zero(e.P),
        )
        λP_e = FEMTools._gather_local(λP, e.nodes_P, Val(NP)) + e.Γ .* (transpose(Bnum) * λv_e)
        λP_plain[e.nodes_P] = λP_e
        for (parameter, field) in zip(DIKE_SENSITIVITY_PARAMETERS, values(derivatives))
            field[iel] = ForwardDiff.derivative(
                s -> parameter_contraction(e, data, λv_e, λP_e, parameter, s), 0.0,
            )
        end
    end
    return derivatives, λP_plain
end

"""
    relative_sensitivities(derivatives, J)

Normalize cell derivatives by signed, finite, nonzero mean vertical velocity.
For either uplift or subsidence, `dJ/J` is `dln|J|`; it is undefined at zero.
"""
function relative_sensitivities(derivatives, J)
    isfinite(J) && !iszero(J) || error("relative vertical-velocity sensitivities require finite nonzero J, got $J")
    return map(f -> f ./ J, derivatives)
end
