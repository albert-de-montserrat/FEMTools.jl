"""
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, cache, γfact, Δt; workgroup=256, kwargs...)

Assemble pressure mass and scaling using the geometry and elements stored in
`cache`. The backend is inferred from `mesh.coords`.
"""
function assemble_viscosity_weighted_pressure_scaling!(
    γP,
    dr::StokesDR,
    mesh::MixedMesh{2},
    cache::MixedMeshCache,
    γfact,
    Δt;
    workgroup = 256,
    kwargs...,
)
    return assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, cache.geo_P, cache.element_v, cache.element_P,
        γfact, Δt, KA.get_backend(mesh.coords), workgroup; kwargs...,
    )
end

"""
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, geo_P, element_v, element_P,
        γfact, Δt, backend, workgroup;
        phases_v=dr.phases_v, η=dr.η, K=dr.K,
    )

Assemble the pressure mass and viscosity-weighted pressure scale using a
`StokesDR` state and `MixedMesh`.

`phases_v`, `η`, and `K` default to the solver state and may be overridden for
element-wise phase layouts or alternate pressure-scaling material properties.
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
)
    return assemble_viscosity_weighted_pressure_scaling!(
        dr.M_P, γP,
        mesh.el2n, mesh.DoFsP, geo_P, mesh.nels,
        element_v, element_P,
        phases_v, η, γfact, K, Δt,
        backend, workgroup,
    )
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
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    return assemble_viscosity_weighted_pressure_scaling!(
        MP, γP, el2n_v, dofs_P, geo_P, nels, element_v, element_P,
        phases_v, η, γfact, nothing, nothing, backend, workgroup,
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
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    NqV = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)

    fill!(MP, 0)
    fill!(γP, 0)
    viscosity_weighted_pressure_scaling_kernel!(backend, workgroup)(
        MP, γP, el2n_v, dofs_P, geo_P, phases_v, η, γfact, K, Δt, NqV, NqP, Val(NV), Val(NP);
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
                                                phases_v, η, γfact, K, Δt, NqV, NqP,
                                                Val(NV), Val(NP))

KernelAbstractions kernel that accumulates the lumped pressure mass `MP` and
viscosity-weighted pressure scale `γP` by numerical quadrature.

At each quadrature point `q` in element `iel`, accumulates
`MP_a += N_a(q) dΩ` and `γP_a += N_a(q) γ_eff(q) dΩ` for every
pressure DoF `a`. Atomix atomics are used unconditionally for correctness
when pressure DoFs are shared across elements (continuous pressure spaces).
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
