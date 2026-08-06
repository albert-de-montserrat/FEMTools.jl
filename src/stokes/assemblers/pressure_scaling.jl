"""
    _finite_viscosity_mean(η) -> mean of the non-`Inf` entries of `η`

Mean of the finite-viscosity phases in a per-phase viscosity tuple, used as the
numerical-penalty scale for `Inf`-viscosity (rigid) phases: substituting `Inf`
directly into `γ_num = γfact * η` would propagate to `γ_eff = γ_phy*γ_num/(γ_phy+γ_num)`
as `NaN` (`Inf/Inf`) rather than the intended fully-numerical penalty.

Errors if every phase has infinite viscosity, since there is then no finite
scale to fall back to.
"""
function _finite_viscosity_mean(η)
    finite = filter(!isinf, η)
    isempty(finite) && throw(ArgumentError(
        "η_mean requires at least one finite-viscosity phase in η=$η"))
    return sum(finite) / length(finite)
end

"""
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, geo_P, element_v, element_P,
        γfact, Δt, backend, workgroup;
        phases_v=dr.phases_v, η=dr.η, K=dr.K, η_mean=_finite_viscosity_mean(η),
    )

Assemble the pressure mass and viscosity-weighted pressure scale using a
`StokesDR` state and `MixedMesh`.

`phases_v`, `η`, and `K` default to the solver state and may be overridden for
element-wise phase layouts or alternate pressure-scaling material properties.
`η_mean` is the fallback numerical-penalty scale substituted for any `Inf`
entries of `η` (rigid phases); it defaults to the mean of `η`'s finite entries
and rarely needs overriding.
"""
function assemble_viscosity_weighted_pressure_scaling!(
    γP,
    dr::StokesDR,
    mesh::MixedMesh{2},
    geo_P,
    element_v::ReferenceElement,
    element_P::ReferenceElement,
    γfact,
    Δt,
    backend, workgroup;
    phases_v = dr.phases_v,
    η = dr.η,
    K = dr.K,
    η_mean = _finite_viscosity_mean(η),
)
    return assemble_viscosity_weighted_pressure_scaling!(
        dr.M_P, γP,
        mesh.el2n, mesh.DoFsP, geo_P, mesh.nels,
        element_v, element_P,
        phases_v, η, γfact, K, Δt,
        backend, workgroup;
        η_mean,
    )
end

"""
    assemble_viscosity_weighted_pressure_scaling(
        dr, mesh, geo_P, element_v, element_P,
        γfact, Δt, backend, workgroup;
        phases_v=dr.phases_v, η=dr.η, K=dr.K, η_mean=_finite_viscosity_mean(η),
    ) -> γP

Allocate the viscosity-weighted pressure scale `γP` on pressure DoFs and return
it. Convenience wrapper around the in-place
[`assemble_viscosity_weighted_pressure_scaling!`](@ref).
"""
function assemble_viscosity_weighted_pressure_scaling(
    dr::StokesDR,
    mesh::MixedMesh{2},
    geo_P,
    element_v::ReferenceElement,
    element_P::ReferenceElement,
    γfact,
    Δt,
    backend, workgroup;
    phases_v = dr.phases_v,
    η = dr.η,
    K = dr.K,
    η_mean = _finite_viscosity_mean(η),
)
    γP = KA.zeros(backend, eltype(dr.M_P), mesh.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, geo_P, element_v, element_P,
        γfact, Δt, backend, workgroup;
        phases_v, η, K, η_mean,
    )
    return γP
end

"""
    assemble_viscosity_weighted_pressure_scaling!(
        MP, γP,
        el2n_v, dofs_P, geo_P, nels,
        element_v, element_P,
        phases_v, η, γfact, K, Δt,
        backend, workgroup,
    )

Assemble a lumped pressure mass `MP` and a viscosity-weighted pressure scale
`γP` on pressure DoFs.

The pressure residual assembled by FEMTools is weak/integrated,

    RP_i = ∫ N_i (-∇⋅v) dΩ,

so `MP_i = ∫ N_i dΩ` converts it to a pointwise residual `RP_i / MP_i`.
For the Arrow-Hurwicz/DYREL pressure update, this helper also computes a local
pressure scale

    γP_i = (∫ N_i γ_eff(q) dΩ) / MP_i,

where `γ_num = γfact * η_q`, `γ_phy = K_q * Δt`, and
`γ_eff = γ_num * γ_phy / (γ_num + γ_phy)`. If `K` is omitted, this falls back
to the incompressible JustRelax penalty branch where
`γ_eff = γ_num * γ_phy / (γ_num + γ_phy)` and `γ_num = γ_phy = γfact * η`.

Any `Inf` entries of `η` (rigid phases) are replaced by `η_mean` — the mean of
`η`'s finite entries by default — before interpolation, keeping `γ_eff` a
finite, purely-numerical penalty for those phases instead of `NaN`.
"""
function assemble_viscosity_weighted_pressure_scaling!(
    MP, γP,
    el2n_v, dofs_P,
    geo_P,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases_v,
    η,
    γfact,
    backend, workgroup;
    η_mean = _finite_viscosity_mean(η),
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    return assemble_viscosity_weighted_pressure_scaling!(
        MP, γP, el2n_v, dofs_P, geo_P, nels, element_v, element_P,
        phases_v, η, γfact, nothing, nothing, backend, workgroup;
        η_mean,
    )
end

function assemble_viscosity_weighted_pressure_scaling!(
    MP, γP,
    el2n_v, dofs_P,
    geo_P,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases_v,
    η,
    γfact,
    K,
    Δt,
    backend, workgroup;
    η_mean = _finite_viscosity_mean(η),
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    NqV = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)

    # `interp2ip_phase` interpolates via a shape-function-weighted sum
    # `Σ N[i]*η[phase[i]]`. Higher-order elements have quadrature points where
    # some `N[i] < 0`; with a uniform phase this sum is still exactly `η[phase]`
    # for finite values (the N[i] sum to 1), but for `η[phase] == Inf` the
    # positive- and negative-weighted terms become `Inf + (-Inf) = NaN` instead
    # of the intended `Inf`. Substituting `η_mean` for `Inf` entries here, before
    # interpolation, avoids that cancellation entirely.
    η_reg = map(ηi -> isinf(ηi) ? η_mean : ηi, η)

    fill!(MP, 0)
    fill!(γP, 0)
    viscosity_weighted_pressure_scaling_kernel!(backend, workgroup)(
        MP, γP, el2n_v, dofs_P, geo_P, phases_v, η_reg, γfact, K, Δt, NqV, NqP, Val(NV), Val(NP);
        ndrange = nels,
    )
    KA.synchronize(backend)
    pressure_scaling_finalize_kernel!(backend, workgroup)(
        γP, MP;
        ndrange = length(MP),
    )
    KA.synchronize(backend)
    return nothing
end

"""
    viscosity_weighted_pressure_scaling_kernel!(MP, γP, el2n_v, dofs_P, geo_P,
                                                phases_v, η, γfact, K, Δt,
                                                NqV, NqP, Val(NV), Val(NP))

KernelAbstractions kernel that accumulates the lumped pressure mass `MP` and
viscosity-weighted pressure scale `γP` by numerical quadrature.

At each quadrature point `q` in element `iel`, accumulates
`MP_a += N_a(q) dΩ` and `γP_a += N_a(q) γ_eff(q) dΩ` for every
pressure DoF `a`. Atomix atomics are used unconditionally for correctness
when pressure DoFs are shared across elements (continuous pressure spaces).

`η` must already have any `Inf` entries replaced by a finite fallback (see
[`assemble_viscosity_weighted_pressure_scaling!`](@ref)): interpolating an
`Inf` viscosity through `interp2ip_phase`'s shape-function-weighted sum can
produce `Inf + (-Inf) = NaN` at quadrature points where a higher-order
element's shape functions take negative values.
"""
@kernel function viscosity_weighted_pressure_scaling_kernel!(
    MP, γP,
    @Const(el2n_v), @Const(dofs_P),
    @Const(geo_P),
    @Const(phases_v),
    η, γfact, K, Δt,
    NqV, NqP, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_dofs_P  = local_nodes_of(dofs_P,  iel, Val(NP))
    phase_loc = _gather_phase(phases_v, local_nodes_v, iel, Val(NV))
    geo_P_el = geo_P[iel]

    for q in eachindex(geo_P_el)
        _, dΩ = geo_P_el[q]
        Nv = NqV[q]
        NPq = NqP[q]
        ηq = interp2ip_phase(Nv, η, phase_loc)
        γq = pressure_scale_at_ip(Nv, ηq, phase_loc, γfact, K, Δt)

        for a in 1:NP
            inod = local_dofs_P[a]
            weight = NPq[a] * dΩ
            Atomix.@atomic :monotonic MP[inod] += weight
            Atomix.@atomic :monotonic γP[inod] += weight * γq
        end
    end
end

@inline pressure_scale_at_ip(_, ηq, _, γfact, ::Nothing, _) = γfact * ηq / 2
@inline function pressure_scale_at_ip(Nv, ηq, phase_loc, γfact, K, Δt)
    γ_num = γfact * ηq
    βq = interp2ip_phase(Nv, map(inv, K), phase_loc)
    γ_phy = iszero(βq) ? γ_num : Δt / βq
    return γ_phy * γ_num / (γ_phy + γ_num)
end

@kernel function pressure_scaling_finalize_kernel!(γP, @Const(MP))
    i = @index(Global)
    γP[i] = MP[i] == 0 ? zero(γP[i]) : γP[i] / MP[i]
end
