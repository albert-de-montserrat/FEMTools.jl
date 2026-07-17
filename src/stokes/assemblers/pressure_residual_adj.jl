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

"""
    assemble_pressure_adjoint_blocks(dr, mesh_stokes, geo_v, geo_P, element_v,
                                     element_P, phases, Δt, workgroup) -> blocks

Assemble the transposed element pressure-residual Jacobian blocks at the
state held in `dr`.

Per element the returned device vector holds `(Jvxᵀ, Jvyᵀ, JPᵀ)`, the
ForwardDiff Jacobians of the element PH pressure residual with respect to the
element velocities and pressure, stored transposed for
[`apply_pressure_adjoint_blocks!`](@ref). Assemble once per state; each apply
is then a differentiation-free sweep.
"""
function assemble_pressure_adjoint_blocks(
    dr::StokesDR,
    mesh_stokes::MixedMesh{2},
    geo_v, geo_P,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    Δt, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    NqP = shape_function_values(element_P, element_v.integration_points)
    backend = KA.get_backend(dr.P)
    Tel = eltype(dr.P)
    Tv = SMatrix{NV, NP, Tel, NV * NP}
    Tp = SMatrix{NP, NP, Tel, NP * NP}
    blocks = KA.allocate(backend, Tuple{Tv, Tv, Tp}, mesh_stokes.nels)
    pressure_adjoint_blocks_kernel!(backend, workgroup)(
        blocks,
        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, phases,
        dr.α, dr.ηb, Δt, NqP, Val(NV), Val(NP);
        ndrange = mesh_stokes.nels,
    )
    KA.synchronize(backend)
    return blocks
end

@kernel function pressure_adjoint_blocks_kernel!(
    blocks,
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
    blocks[iel] = (transpose(Jvx), transpose(Jvy), transpose(JP))
end

"""
    apply_pressure_adjoint_blocks!(dvx, dvy, dP, seed, blocks, mesh_stokes,
                                   element_v, element_P, workgroup)

Accumulate the transpose pressure-Jacobian products of `seed` into the output
arrays: `dv += (∂RP/∂v)ᵀ·seed` and `dP += (∂RP/∂P)ᵀ·seed`, using element
blocks precomputed by [`assemble_pressure_adjoint_blocks`](@ref). Outputs are
accumulated, not zeroed. Matches the contraction of
[`assemble_pressure_residual_matrices_atomix_adj!`](@ref) exactly.
"""
function apply_pressure_adjoint_blocks!(
    dvx, dvy, dP, seed, blocks,
    mesh_stokes::MixedMesh{2},
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    backend = KA.get_backend(dvx)
    pressure_adjoint_blocks_apply_kernel!(backend, workgroup)(
        dvx, dvy, dP, seed, blocks,
        mesh_stokes.el2n, mesh_stokes.DoFsP, Val(NV), Val(NP);
        ndrange = mesh_stokes.nels,
    )
    KA.synchronize(backend)
    return nothing
end

@kernel function pressure_adjoint_blocks_apply_kernel!(
    dvx, dvy, dP,
    @Const(seed), @Const(blocks),
    @Const(el2n_v), @Const(el2nP),
    ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    nodes_P = local_nodes_of(el2nP, iel, Val(NP))
    s = _gather_local(seed, nodes_P, Val(NP))
    Tvx, Tvy, TP = blocks[iel]
    dvx_e = Tvx * s
    dvy_e = Tvy * s
    dP_e = TP * s
    for i in 1:NV
        Atomix.@atomic :monotonic dvx[nodes_v[i]] += dvx_e[i]
        Atomix.@atomic :monotonic dvy[nodes_v[i]] += dvy_e[i]
    end
    # P1-disc pressure DoFs belong to exactly one element: no scatter races.
    for i in 1:NP
        dP[nodes_P[i]] += dP_e[i]
    end
end
