"""
    FrozenAdjointOperator(A, B, C)

Element blocks of the transposed Stokes adjoint operator, held for the lifetime of
one adjoint solve.

At a converged forward state the adjoint residual is affine in `λ` with a constant
operator, so the blocks are assembled once and applied many times:

```
ResλV = objective_v + Aᵀλv + CᵀλP
ResλP = Bᵀλv
```

- `A` is the augmented velocity block `∂Rv/∂v`, `2NV`×`2NV` per element, ordered
  with the `vx` degrees of freedom first. It already carries the Powell-Hestenes
  term `Bnum·(γP/M_P)·C`, because the assembler forms `Pnum` inline from
  element-local pressures; that coupling stays inside one element only while the
  pressure space is discontinuous.
- `B` is `∂Rv/∂P`, `2NV`×`NP` per element, differentiated with `Pnum` held as an
  independent variable so it excludes the augmentation already inside `A`.
- `C` is `∂RP/∂v`, `NP`×`2NV` per element.

Storage is `≈(4NV² + 4NV·NP)` floating-point numbers per element — 280 for T7/P1-disc
in two dimensions, about 2.2 kB.
"""
struct FrozenAdjointOperator{TA, TB, TC}
    A::TA
    B::TB
    C::TC
end

"""
    element_adjoint_operator_blocks(...) -> (local_nodes_v, local_nodes_P, A, B, C)

Build the three transposed-operator blocks for element `iel` at the current
forward state. See [`FrozenAdjointOperator`](@ref) for what each block contains.
"""
@inline function element_adjoint_operator_blocks(
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, iel, ::Val{NV}, ::Val{NP},
    ) where {NV, NP}
    local_nodes_v, ∂RVx∂vx, ∂RVx∂vy, ∂RVy∂vx, ∂RVy∂vy =
        element_augmented_momentum_jacobians(
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, iel, Val(NV), Val(NP),
    )
    A = vcat(hcat(∂RVx∂vx, ∂RVx∂vy), hcat(∂RVy∂vx, ∂RVy∂vy))

    local_nodes_P = local_nodes_of(el2nP, iel, Val(NP))
    geo_v_el = geo_v[iel]
    geo_P_el = geo_P[iel]
    vxloc = _gather_local(vx, local_nodes_v, Val(NV))
    vyloc = _gather_local(vy, local_nodes_v, Val(NV))
    P_loc = _gather_local(P, local_nodes_P, Val(NP))
    P0loc = _gather_local(P0, local_nodes_P, Val(NP))
    T_loc = _gather_local(T, local_nodes_P, Val(NP))
    T0loc = _gather_local(T0, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, iel, Val(NV), Val(length(Nq)))
    phase_v = _gather_phase(phases_v, local_nodes_v, iel, Val(NV))
    phase_P = _gather_phase(phases_P, local_nodes_P, iel, Val(NP))

    # ∂Rv/∂P with Pnum independent, matching the momentum residual the forward
    # solve evaluates: the augmentation belongs to A, not here.
    Pnum_loc = zero(P_loc)
    B = ForwardDiff.jacobian(
        P_arg -> begin
            Rx, Ry = integrate_momentum_residual(
                (vxloc, vyloc), P_arg, Pnum_loc, T_loc,
                geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt,
                τ_old_loc, plastic, nothing, Nq, NqP,
            )
            vcat(Rx, Ry)
        end,
        P_loc,
    )

    C = ForwardDiff.jacobian(
        v_arg -> integrate_PH_pressure_residual(
            (v_arg[SOneTo(NV)], v_arg[SVector{NV}(ntuple(i -> NV + i, Val(NV)))]),
            P_loc, P0loc, T_loc, T0loc,
            geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
        ),
        vcat(vxloc, vyloc),
    )

    return local_nodes_v, local_nodes_P, A, B, C
end

@kernel function adjoint_operator_assembly_kernel!(
        Ablocks, Bblocks, Cblocks,
        @Const(vx), @Const(vy),
        @Const(P), @Const(P0),
        @Const(T), @Const(T0),
        @Const(el2n_v), @Const(el2nP),
        @Const(geo_v), @Const(geo_P),
        @Const(phases_v), @Const(phases_P),
        τ_old, plastic,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        @Const(MP),
        Nq, NqP, ::Val{NV}, ::Val{NP},
    ) where {NV, NP}
    iel = @index(Global)
    _, _, A, B, C = element_adjoint_operator_blocks(
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, iel, Val(NV), Val(NP),
    )
    # One element owns one entry of each block array, so no atomics are needed.
    Ablocks[iel] = A
    Bblocks[iel] = B
    Cblocks[iel] = C
end

"""
    assemble_adjoint_operator(dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
                              phases_v, phases_P, τ_old, plastic, G, Δt, γP,
                              backend, workgroup) -> FrozenAdjointOperator

Assemble the transposed adjoint operator at the forward state held in `dr`.

The forward state must be converged: the blocks are frozen here and reused for
every subsequent apply, so they describe the linearisation at whatever state `dr`
holds at this moment.
"""
function assemble_adjoint_operator(
        dr, mesh_stokes, geo_v, geo_P,
        element_v::ReferenceElement{TV},
        element_P::ReferenceElement{TP},
        phases_v, phases_P, τ_old, plastic, G, Δt, γP,
        backend, workgroup,
    ) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    Nq = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)
    nels = mesh_stokes.nels
    Tv = eltype(dr.vx)

    Ablocks = similar(dr.vx, SMatrix{2NV, 2NV, Tv, 4NV * NV}, nels)
    Bblocks = similar(dr.vx, SMatrix{2NV, NP, Tv, 2NV * NP}, nels)
    Cblocks = similar(dr.vx, SMatrix{NP, 2NV, Tv, 2NV * NP}, nels)

    adjoint_operator_assembly_kernel!(backend, workgroup)(
        Ablocks, Bblocks, Cblocks,
        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P,
        phases_v, phases_P, τ_old, plastic,
        dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, dr.ηb, Δt, γP, dr.M_P,
        Nq, NqP, Val(NV), Val(NP);
        ndrange = nels,
    )
    KA.synchronize(backend)
    return FrozenAdjointOperator(Ablocks, Bblocks, Cblocks)
end

@kernel function adjoint_operator_apply_kernel!(
        dvx, dvy, dP,
        @Const(Ablocks), @Const(Bblocks), @Const(Cblocks),
        @Const(λvx), @Const(λvy), @Const(λP),
        @Const(el2n_v), @Const(el2nP),
        ::Val{NV}, ::Val{NP},
    ) where {NV, NP}
    iel = @index(Global)
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP, iel, Val(NP))

    λv = vcat(
        _gather_local(λvx, local_nodes_v, Val(NV)),
        _gather_local(λvy, local_nodes_v, Val(NV)),
    )
    λP_loc = _gather_local(λP, local_nodes_P, Val(NP))

    resv = transpose(Ablocks[iel]) * λv + transpose(Cblocks[iel]) * λP_loc
    resp = transpose(Bblocks[iel]) * λv

    for (i, inod) in enumerate(local_nodes_v)
        Atomix.@atomic :monotonic dvx[inod] += resv[i]
        Atomix.@atomic :monotonic dvy[inod] += resv[NV + i]
    end
    # Pressure degrees of freedom are discontinuous, hence unshared.
    _add_local!(dP, local_nodes_P, resp, Val(false))
end

"""
    apply_adjoint_operator!(dvx, dvy, dP, op, λvx, λvy, λP, mesh_stokes,
                            element_v, element_P, backend, workgroup)

Apply the transposed adjoint operator, overwriting `dvx`, `dvy`, and `dP` with
`Aᵀλv + CᵀλP` and `Bᵀλv`.

No rheology is evaluated and no primal residual is recomputed: this is a gather,
a pair of dense element products, and a scatter.
"""
function apply_adjoint_operator!(
        dvx, dvy, dP, op::FrozenAdjointOperator, λvx, λvy, λP, mesh_stokes,
        element_v::ReferenceElement{TV},
        element_P::ReferenceElement{TP},
        backend, workgroup,
    ) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    fill!(dvx, 0)
    fill!(dvy, 0)
    fill!(dP, 0)
    adjoint_operator_apply_kernel!(backend, workgroup)(
        dvx, dvy, dP, op.A, op.B, op.C, λvx, λvy, λP,
        mesh_stokes.el2n, mesh_stokes.DoFsP, Val(NV), Val(NP);
        ndrange = mesh_stokes.nels,
    )
    KA.synchronize(backend)
    return nothing
end
