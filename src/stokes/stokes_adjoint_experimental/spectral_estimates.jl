"""
    experimental_element_adjoint_spectral_diagnostics(...)

Compute element-local absolute row sums of the *adjoint* Jacobian for the
coupled `(λvx, λvy, λP)` system. Because the adjoint operator is the transpose
of the augmented forward Jacobian, these are assembled as absolute column sums
of the forward `3×3` block operator.

The returned row sums contain every coupling:

```
                 forward residual rows
              Rvx        Rvy        RP
λvx row:    Jxx'       Jyx'       JPx'
λvy row:    Jxy'       Jyy'       JPy'
λP  row:    JxP'       JyP'       JPP'
```

`free_residual_vx` and `free_residual_vy` mask forward momentum residual rows
that are overwritten by Dirichlet conditions. This makes the estimate match
the boundary-conditioned transpose used by the adjoint residual assembly.
"""
@inline function experimental_element_adjoint_spectral_diagnostics(
    vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
    phases_v, phases_P, free_residual_vx, free_residual_vy,
    τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
    MP, Nq, NqP, iel, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    nodes_P = local_nodes_of(el2nP, iel, Val(NP))
    geo_v_el = geo_v[iel]
    geo_P_el = geo_P[iel]
    vxloc = _gather_local(vx, nodes_v, Val(NV))
    vyloc = _gather_local(vy, nodes_v, Val(NV))
    Ploc = _gather_local(P, nodes_P, Val(NP))
    P0loc = _gather_local(P0, nodes_P, Val(NP))
    Tloc = _gather_local(T, nodes_P, Val(NP))
    T0loc = _gather_local(T0, nodes_P, Val(NP))
    MPloc = _gather_local(MP, nodes_P, Val(NP))
    γloc = _gather_or_scalar(γ_eff, nodes_P, Val(NP))
    phase_v = _gather_phase(phases_v, nodes_v, iel, Val(NV))
    phase_P = _gather_phase(phases_P, nodes_P, iel, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, nodes_v, iel, Val(NV), Val(length(Nq)))
    η_pc = _element_max_phase_property(η, phase_v)
    mask_x = _gather_local(free_residual_vx, nodes_v, Val(NV))
    mask_y = _gather_local(free_residual_vy, nodes_v, Val(NV))

    NALL = 2 * NV + NP
    u0 = SVector{NALL}(ntuple(
        i -> i <= NV ? vxloc[i] : i <= 2 * NV ? vyloc[i - NV] : Ploc[i - 2 * NV],
        Val(NALL),
    ))

    function coupled_forward_residual(u)
        vx_arg = SVector{NV}(ntuple(i -> u[i], Val(NV)))
        vy_arg = SVector{NV}(ntuple(i -> u[NV + i], Val(NV)))
        P_arg = SVector{NP}(ntuple(i -> u[2 * NV + i], Val(NP)))
        v_arg = (vx_arg, vy_arg)

        Rx = integrate_momentum_x_residual(
            v_arg, P_arg, P0loc, Tloc, T0loc,
            geo_v_el, geo_P_el, phase_v, phase_P,
            η_pc, G, α, ρ0, K, g, Tref, ηb, Δt, γloc, MPloc,
            τ_old_loc, plastic, Nq, NqP,
        )
        Ry = integrate_momentum_y_residual(
            v_arg, P_arg, P0loc, Tloc, T0loc,
            geo_v_el, geo_P_el, phase_v, phase_P,
            η_pc, G, α, ρ0, K, g, Tref, ηb, Δt, γloc, MPloc,
            τ_old_loc, plastic, Nq, NqP,
        )
        Rp = integrate_PH_pressure_residual(
            v_arg, P_arg, P0loc, Tloc, T0loc,
            geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
        )

        return SVector{NALL}(ntuple(
            i -> i <= NV ? Rx[i] : i <= 2 * NV ? Ry[i - NV] : Rp[i - 2 * NV],
            Val(NALL),
        ))
    end

    J = ForwardDiff.jacobian(coupled_forward_residual, u0)

    # Absolute column sums of J are absolute row sums of J'. Momentum output
    # masks reproduce the pullback of the forward Dirichlet overwrite.
    rowsum_vx = SVector{NV}(ntuple(j ->
        sum(abs(J[i, j]) * mask_x[i] + abs(J[NV + i, j]) * mask_y[i] for i in 1:NV) +
        sum(abs(J[2 * NV + i, j]) for i in 1:NP), Val(NV)))
    rowsum_vy = SVector{NV}(ntuple(j ->
        sum(abs(J[i, NV + j]) * mask_x[i] + abs(J[NV + i, NV + j]) * mask_y[i] for i in 1:NV) +
        sum(abs(J[2 * NV + i, NV + j]) for i in 1:NP), Val(NV)))
    rowsum_P = SVector{NP}(ntuple(j ->
        sum(abs(J[i, 2 * NV + j]) * mask_x[i] + abs(J[NV + i, 2 * NV + j]) * mask_y[i] for i in 1:NV) +
        sum(abs(J[2 * NV + i, 2 * NV + j]) for i in 1:NP), Val(NP)))

    diag_vx = SVector{NV}(ntuple(j -> abs(J[j, j]), Val(NV)))
    diag_vy = SVector{NV}(ntuple(j -> abs(J[NV + j, NV + j]), Val(NV)))
    diag_P = SVector{NP}(ntuple(j -> abs(J[2 * NV + j, 2 * NV + j]), Val(NP)))

    return nodes_v, nodes_P, rowsum_vx, diag_vx, rowsum_vy, diag_vy, rowsum_P, diag_P
end

"""
    assemble_experimental_adjoint_spectral_diagnostics!(
        rowsum_vx, diag_vx, rowsum_vy, diag_vy, rowsum_P, diag_P,
        dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
        phases_v, phases_P, free_residual_vx, free_residual_vy,
        τ_old, plastic, G, Δt, γP, backend, workgroup)

Assemble conservative Gershgorin bounds and absolute block diagonals for all
three rows of the coupled adjoint operator. Element absolute values are
scattered before summation, so shared-node cancellation cannot make the bound
non-conservative.
"""
function assemble_experimental_adjoint_spectral_diagnostics!(
    rowsum_vx, diag_vx, rowsum_vy, diag_vy, rowsum_P, diag_P,
    dr, mesh_stokes,
    geo_v, geo_P,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases_v, phases_P,
    free_residual_vx, free_residual_vy,
    τ_old, plastic, G, Δt, γP,
    backend, workgroup,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    Nq = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)
    fill!(rowsum_vx, 0)
    fill!(diag_vx, 0)
    fill!(rowsum_vy, 0)
    fill!(diag_vy, 0)
    fill!(rowsum_P, 0)
    fill!(diag_P, 0)

    experimental_adjoint_spectral_kernel!(backend, workgroup)(
        rowsum_vx, diag_vx, rowsum_vy, diag_vy, rowsum_P, diag_P,
        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P,
        phases_v, phases_P, free_residual_vx, free_residual_vy,
        τ_old, plastic, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
        dr.ηb, Δt, γP, dr.M_P, Nq, NqP, Val(NV), Val(NP);
        ndrange = mesh_stokes.nels,
    )
    KA.synchronize(backend)
    return nothing
end

@kernel function experimental_adjoint_spectral_kernel!(
    rowsum_vx, diag_vx, rowsum_vy, diag_vy, rowsum_P, diag_P,
    @Const(vx), @Const(vy), @Const(P), @Const(P0), @Const(T), @Const(T0),
    @Const(el2n_v), @Const(el2nP), @Const(geo_v), @Const(geo_P),
    @Const(phases_v), @Const(phases_P),
    @Const(free_residual_vx), @Const(free_residual_vy),
    @Const(τ_old), @Const(plastic),
    @Const(η), @Const(G), @Const(α), @Const(ρ0), @Const(K),
    @Const(g), Tref, @Const(ηb), Δt, @Const(γP), @Const(MP),
    @Const(Nq), @Const(NqP), ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    nodes_v, nodes_P, rsx, dx, rsy, dy, rsp, dp =
        experimental_element_adjoint_spectral_diagnostics(
            vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
            phases_v, phases_P, free_residual_vx, free_residual_vy,
            τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γP,
            MP, Nq, NqP, iel, Val(NV), Val(NP),
        )

    for i in 1:NV
        node = nodes_v[i]
        Atomix.@atomic :monotonic rowsum_vx[node] += rsx[i]
        Atomix.@atomic :monotonic diag_vx[node] += dx[i]
        Atomix.@atomic :monotonic rowsum_vy[node] += rsy[i]
        Atomix.@atomic :monotonic diag_vy[node] += dy[i]
    end
    for i in 1:NP
        node = nodes_P[i]
        Atomix.@atomic :monotonic rowsum_P[node] += rsp[i]
        Atomix.@atomic :monotonic diag_P[node] += dp[i]
    end
end
