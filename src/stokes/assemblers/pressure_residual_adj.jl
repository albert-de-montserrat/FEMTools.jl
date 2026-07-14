"""
    assemble_pressure_residual_matrices_atomix_adj!(dr, dRP, dvx, dvy, dP,
                                                    mesh_stokes, geo_v, geo_P,
                                                    element_v, element_P,
                                                    phases, Δt, workgroup)

Apply the transpose pressure-residual Jacobian to the seed `dRP`.
"""
function assemble_pressure_residual_matrices_atomix_adj!(
    dr::StokesDR,
    dRP, dvx, dvy, dP,
    mesh_stokes::MixedMesh{2},
    geo_v, geo_P,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    Δt, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    NqP = shape_function_values(element_P, element_v.integration_points)
    backend = KA.get_backend(dRP)
    pressure_residual_adjoint_kernel!(backend, workgroup)(
        dRP, dvx, dvy, dP,
        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, phases,
        dr.α, dr.ηb, Δt, NqP, Val(NV), Val(NP);
        ndrange = mesh_stokes.nels,
    )
    KA.synchronize(backend)
    return nothing
end

@kernel function pressure_residual_adjoint_kernel!(
    @Const(dRP), dvx, dvy, dP,
    @Const(vx), @Const(vy), @Const(P), @Const(P0),
    @Const(T), @Const(T0),
    @Const(el2n_v), @Const(el2nP), @Const(geo_v), @Const(geo_P),
    @Const(phases), @Const(α), @Const(ηb), Δt, @Const(NqP),
    ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    nodes_P = local_nodes_of(el2nP, iel, Val(NP))
    vxloc = _gather_local(vx, nodes_v, Val(NV))
    vyloc = _gather_local(vy, nodes_v, Val(NV))
    Ploc = _gather_local(P, nodes_P, Val(NP))
    P0loc = _gather_local(P0, nodes_P, Val(NP))
    Tloc = _gather_local(T, nodes_P, Val(NP))
    T0loc = _gather_local(T0, nodes_P, Val(NP))
    seed = _gather_local(dRP, nodes_P, Val(NP))
    phase_loc = _gather_phase(phases, nodes_P, iel, Val(NP))
    geo_v_el = geo_v[iel]
    geo_P_el = geo_P[iel]

    Jvx = ForwardDiff.jacobian(v -> integrate_PH_pressure_residual(
        (v, vyloc), Ploc, P0loc, Tloc, T0loc,
        geo_v_el, geo_P_el, phase_loc, α, ηb, Δt, NqP), vxloc)
    Jvy = ForwardDiff.jacobian(v -> integrate_PH_pressure_residual(
        (vxloc, v), Ploc, P0loc, Tloc, T0loc,
        geo_v_el, geo_P_el, phase_loc, α, ηb, Δt, NqP), vyloc)
    JP = ForwardDiff.jacobian(p -> integrate_PH_pressure_residual(
        (vxloc, vyloc), p, P0loc, Tloc, T0loc,
        geo_v_el, geo_P_el, phase_loc, α, ηb, Δt, NqP), Ploc)
    dvx_e = SVector{NV}(ntuple(j -> sum(Jvx[i, j] * seed[i] for i in 1:NP), Val(NV)))
    dvy_e = SVector{NV}(ntuple(j -> sum(Jvy[i, j] * seed[i] for i in 1:NP), Val(NV)))
    dP_e = SVector{NP}(ntuple(j -> sum(JP[i, j] * seed[i] for i in 1:NP), Val(NP)))

    for i in 1:NV
        Atomix.@atomic :monotonic dvx[nodes_v[i]] += dvx_e[i]
        Atomix.@atomic :monotonic dvy[nodes_v[i]] += dvy_e[i]
    end
    for i in 1:NP
        dP[nodes_P[i]] += dP_e[i]
    end
end
