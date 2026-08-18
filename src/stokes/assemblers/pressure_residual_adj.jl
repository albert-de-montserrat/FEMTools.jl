"""
    assemble_pressure_residual_matrices_atomix_adj!(
        dr, dRP, dvx, dvy, dP,
        mesh_stokes, geo_v, geo_P, element_v, element_P,
        phases, Δt, workgroup)

Reverse-mode (Enzyme) transpose of `assemble_pressure_residual_kernel!`.

Reads the primal fields (`RP`, `vx`, `vy`, `P`, `P0`, `T`, `T0`) and the
per-phase parameters `α`, `ηb` from the solver state `dr`. Given the adjoint
seed `dRP` on the pressure residual, accumulate the transpose-Jacobian
products into the input adjoints `dvx`, `dvy`, `dP`. The primal residual
`dr.RP` is recomputed (overwritten) in the process.
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
    RP, vx, vy, P = dr.RP, dr.vx, dr.vy, dr.P
    P0, T, T0 = dr.P0, dr.T, dr.T0
    el2n_v, dofs_P = mesh_stokes.el2n, mesh_stokes.DoFsP
    nels = mesh_stokes.nels

    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(assemble_pressure_residual_kernel!),
        Enzyme.Const,
        Enzyme.Duplicated(RP, dRP),
        Enzyme.Duplicated(vx, dvx),
        Enzyme.Duplicated(vy, dvy),
        Enzyme.Duplicated(P, dP),
        Enzyme.Const(P0),
        Enzyme.Const(T),
        Enzyme.Const(T0),
        Enzyme.Const(el2n_v),
        Enzyme.Const(dofs_P),
        Enzyme.Const(geo_v),
        Enzyme.Const(geo_P),
        Enzyme.Const(nels),
        Enzyme.Const(phases),
        Enzyme.Const(dr.α),
        Enzyme.Const(dr.ηb),
        Enzyme.Const(Δt),
        Enzyme.Const(NqP),
        Enzyme.Const(Val(NV)),
        Enzyme.Const(Val(NP)),
        Enzyme.Const(workgroup),
    )
    return nothing
end
