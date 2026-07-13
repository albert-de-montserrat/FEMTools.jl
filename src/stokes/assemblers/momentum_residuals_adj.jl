"""
    assemble_momentum_residual_matrices_atomix_adj!(...)

Apply the transpose momentum Jacobian to the output seeds `dRv_x` and
`dRv_y`. Each work item differentiates the scalar element contraction
`dRv_xᵉ⋅Rv_xᵉ + dRv_yᵉ⋅Rv_yᵉ` with ForwardDiff and scatters the
resulting pullback. Keeping the differentiation inside the
KernelAbstractions kernel makes the implementation portable to CPU and GPU
backends and avoids host-side AD/JIT trampolines.
"""
function assemble_momentum_residual_matrices_atomix_adj!(
    Rv_x, dRv_x, Rv_y, dRv_y,
    vx, dvx, vy, dvy, P, dP, T, Pnum, dPnum,
    mesh_stokes::MixedMesh{2}, geo_v,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases, τ_old, plastic,
    η, G, α, ρ0, K, g, Tref, Δt,
    workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    fill!(Rv_x, 0)
    fill!(Rv_y, 0)
    Nq  = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)
    backend = KA.get_backend(Rv_x)
    momentum_residual_adjoint_kernel!(backend, workgroup)(
        Rv_x, dRv_x, Rv_y, dRv_y,
        vx, dvx, vy, dvy, P, dP, T, Pnum, dPnum,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, phases,
        τ_old, plastic, η, G, α, ρ0, K, g, Tref, Δt,
        Nq, NqP, Val(NV), Val(NP);
        ndrange = mesh_stokes.nels,
    )
    KA.synchronize(backend)
    return nothing
end

@kernel function momentum_residual_adjoint_kernel!(
    Rv_x, @Const(dRv_x), Rv_y, @Const(dRv_y),
    @Const(vx), dvx, @Const(vy), dvy,
    @Const(P), dP, @Const(T), @Const(Pnum), dPnum,
    @Const(el2n_v), @Const(el2nP), @Const(geo_v), @Const(phases),
    @Const(τ_old), @Const(plastic),
    @Const(η), @Const(G), @Const(α), @Const(ρ0), @Const(K),
    @Const(g), Tref, Δt, @Const(Nq), @Const(NqP),
    ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    nodes_P = local_nodes_of(el2nP, iel, Val(NP))
    vxloc = _gather_local(vx, nodes_v, Val(NV))
    vyloc = _gather_local(vy, nodes_v, Val(NV))
    Ploc = _gather_local(P, nodes_P, Val(NP))
    Tloc = _gather_local(T, nodes_P, Val(NP))
    Pnumloc = _gather_local(Pnum, nodes_P, Val(NP))
    seed_x = _gather_local(dRv_x, nodes_v, Val(NV))
    seed_y = _gather_local(dRv_y, nodes_v, Val(NV))
    phase_loc = _gather_phase(phases, nodes_v, iel, Val(NV))
    τ_old_loc = _gather_old_stress(τ_old, nodes_v, iel, Val(NV), Val(length(Nq)))
    geo_el = geo_v[iel]

    Jxx = ForwardDiff.jacobian(v -> first(integrate_momentum_residual(
        (v, vyloc), Ploc, Pnumloc, Tloc, geo_el, phase_loc, η, G, α, ρ0, K,
        g, Tref, Δt, τ_old_loc, plastic, Nq, NqP)), vxloc)
    Jxy = ForwardDiff.jacobian(v -> first(integrate_momentum_residual(
        (vxloc, v), Ploc, Pnumloc, Tloc, geo_el, phase_loc, η, G, α, ρ0, K,
        g, Tref, Δt, τ_old_loc, plastic, Nq, NqP)), vyloc)
    Jyx = ForwardDiff.jacobian(v -> last(integrate_momentum_residual(
        (v, vyloc), Ploc, Pnumloc, Tloc, geo_el, phase_loc, η, G, α, ρ0, K,
        g, Tref, Δt, τ_old_loc, plastic, Nq, NqP)), vxloc)
    Jyy = ForwardDiff.jacobian(v -> last(integrate_momentum_residual(
        (vxloc, v), Ploc, Pnumloc, Tloc, geo_el, phase_loc, η, G, α, ρ0, K,
        g, Tref, Δt, τ_old_loc, plastic, Nq, NqP)), vyloc)
    JxP = ForwardDiff.jacobian(p -> first(integrate_momentum_residual(
        (vxloc, vyloc), p, Pnumloc, Tloc, geo_el, phase_loc, η, G, α, ρ0, K,
        g, Tref, Δt, τ_old_loc, plastic, Nq, NqP)), Ploc)
    JyP = ForwardDiff.jacobian(p -> last(integrate_momentum_residual(
        (vxloc, vyloc), p, Pnumloc, Tloc, geo_el, phase_loc, η, G, α, ρ0, K,
        g, Tref, Δt, τ_old_loc, plastic, Nq, NqP)), Ploc)
    JxPn = ForwardDiff.jacobian(p -> first(integrate_momentum_residual(
        (vxloc, vyloc), Ploc, p, Tloc, geo_el, phase_loc, η, G, α, ρ0, K,
        g, Tref, Δt, τ_old_loc, plastic, Nq, NqP)), Pnumloc)
    JyPn = ForwardDiff.jacobian(p -> last(integrate_momentum_residual(
        (vxloc, vyloc), Ploc, p, Tloc, geo_el, phase_loc, η, G, α, ρ0, K,
        g, Tref, Δt, τ_old_loc, plastic, Nq, NqP)), Pnumloc)
    dvx_e = SVector{NV}(ntuple(j -> sum(Jxx[i, j] * seed_x[i] + Jyx[i, j] * seed_y[i] for i in 1:NV), Val(NV)))
    dvy_e = SVector{NV}(ntuple(j -> sum(Jxy[i, j] * seed_x[i] + Jyy[i, j] * seed_y[i] for i in 1:NV), Val(NV)))
    dP_e = SVector{NP}(ntuple(j -> sum(JxP[i, j] * seed_x[i] + JyP[i, j] * seed_y[i] for i in 1:NV), Val(NP)))
    dPnum_e = SVector{NP}(ntuple(j -> sum(JxPn[i, j] * seed_x[i] + JyPn[i, j] * seed_y[i] for i in 1:NV), Val(NP)))

    Re_x, Re_y = integrate_momentum_residual(
        (vxloc, vyloc), Ploc, Pnumloc, Tloc,
        geo_el, phase_loc, η, G, α, ρ0, K, g, Tref, Δt,
        τ_old_loc, plastic, Nq, NqP,
    )
    for i in 1:NV
        Atomix.@atomic :monotonic Rv_x[nodes_v[i]] += Re_x[i]
        Atomix.@atomic :monotonic Rv_y[nodes_v[i]] += Re_y[i]
        Atomix.@atomic :monotonic dvx[nodes_v[i]] += dvx_e[i]
        Atomix.@atomic :monotonic dvy[nodes_v[i]] += dvy_e[i]
    end
    for i in 1:NP
        dP[nodes_P[i]] += dP_e[i]
        dPnum[nodes_P[i]] += dPnum_e[i]
    end
end
