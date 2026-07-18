"""
    assemble_momentum_residual_matrices_atomix_adj!(
        Rv_x, dRv_x, Rv_y, dRv_y,
        vx, dvx, vy, dvy, P, dP, T, Pnum, dPnum,
        mesh_stokes, geo_v, element_v, element_P,
        phases, τ_old, plastic,
        η, G, α, ρ0, K, g, Tref, Δt,
        workgroup)

Reverse-mode (Enzyme) transpose of `assemble_momentum_residual_kernel!`.

Given adjoint seeds `dRv_x`, `dRv_y` on the momentum residuals, accumulate the
transpose-Jacobian products into the input adjoints `dvx`, `dvy`, `dP`, and
`dPnum`. The primal residuals `Rv_x`, `Rv_y` are recomputed (overwritten) in
the process. All arrays must reside on the same backend; the kernel executes
on the backend of the output buffers.
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
    Nq  = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)
    el2n_v = mesh_stokes.el2n
    el2nP  = mesh_stokes.DoFsP
    nels   = mesh_stokes.nels

    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(assemble_momentum_residual_kernel!),
        Enzyme.Const,
        Enzyme.Duplicated(Rv_x, dRv_x),
        Enzyme.Duplicated(Rv_y, dRv_y),
        Enzyme.Duplicated(vx, dvx),
        Enzyme.Duplicated(vy, dvy),
        Enzyme.Duplicated(P, dP),
        Enzyme.Const(T),
        Enzyme.Duplicated(Pnum, dPnum),
        Enzyme.Const(el2n_v),
        Enzyme.Const(el2nP),
        Enzyme.Const(geo_v),
        Enzyme.Const(nels),
        Enzyme.Const(phases),
        Enzyme.Const(τ_old),
        Enzyme.Const(plastic),
        Enzyme.Const(nothing),
        Enzyme.Const(η),
        Enzyme.Const(G),
        Enzyme.Const(α),
        Enzyme.Const(ρ0),
        Enzyme.Const(K),
        Enzyme.Const(g),
        Enzyme.Const(Tref),
        Enzyme.Const(Δt),
        Enzyme.Const(Nq),
        Enzyme.Const(NqP),
        Enzyme.Const(Val(NV)),
        Enzyme.Const(Val(NP)),
        Enzyme.Const(workgroup),
    )
    return nothing
end
