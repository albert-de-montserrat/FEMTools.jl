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
  with the `vx` degrees of freedom first, or its packed upper triangle. It already carries the Powell-Hestenes
  term `Bnum·(γP/M_P)·C`, because the assembler forms `Pnum` inline from
  element-local pressures; that coupling stays inside one element only while the
  pressure space is discontinuous.
- `B` is `∂Rv/∂P`, `2NV`×`NP` per element, differentiated with `Pnum` held as an
  independent variable so it excludes the augmentation already inside `A`.
- `C` is `∂RP/∂v`, `NP`×`2NV` per element, or `nothing`.

A purely viscous forward model has a symmetric element tangent, and the layout
exploits that twice: `A` keeps only its upper triangle, and `C` is `nothing`
because `C == Bᵀ` exactly, so the apply reads `B` in its place. Storage falls from
`4NV² + 4NV·NP` floating-point numbers per element to `NV(2NV+1) + 2NV·NP` — 2 240
to 1 176 bytes for T7/P1-disc in two dimensions. `_symmetric_matvec` applies the
packed block without rebuilding it, at the same arithmetic cost.

Both identities are asserted per element at assembly rather than assumed, because
they are properties of the tangent, not of the code. A non-associated plastic flow
rule (`ψ ≠ ϕ`) makes the tangent non-normal and breaks them: on a shear-banding
case with 97 % of quadrature points at yield, `‖A − Aᵀ‖/‖A‖` is 2 × 10⁻⁴ and
`‖C − Bᵀ‖/‖C‖` is 0.125, against 7 × 10⁻¹⁵ and 0 at the same forward state with
the plastic tangent switched off.
"""
struct FrozenAdjointOperator{TA, TB, TC}
    A::TA
    B::TB
    C::TC
end

struct FrozenVelocityOperator{TA}
    A::TA
end

"""
    _packed_symmetric_length(N) -> Int

Number of entries in the upper triangle of an `N`×`N` symmetric matrix.
"""
@inline _packed_symmetric_length(N) = (N * (N + 1)) ÷ 2

# Column-major upper triangle: entry (i, j) with i ≤ j sits at j(j-1)/2 + i.
_triangle_index(i, j) = i ≤ j ? (j * (j - 1)) ÷ 2 + i : (i * (i - 1)) ÷ 2 + j

"""
    _pack_symmetric(A) -> SVector

Pack the upper triangle of a symmetric `SMatrix` column by column.

The lower triangle is not read: a caller storing the result has already
established that `A == Aᵀ`.
"""
@generated function _pack_symmetric(A::SMatrix{N, N, T}) where {N, T}
    entries = [:(A[$i, $j]) for j in 1:N for i in 1:j]
    return :(SVector{$(_packed_symmetric_length(N)), T}($(entries...)))
end

"""
    _unpack_symmetric(packed, Val(N)) -> SMatrix{N, N}

Rebuild the full symmetric matrix from a packed upper triangle. Used for
diagnostics and tests; the solver applies the packed form directly.
"""
@generated function _unpack_symmetric(packed::SVector{L, T}, ::Val{N}) where {L, N, T}
    entries = [:(packed[$(_triangle_index(i, j))]) for j in 1:N for i in 1:N]
    return :(SMatrix{$N, $N, T, $(N * N)}($(entries...)))
end

"""
    _symmetric_matvec(packed, x) -> SVector

Multiply a packed symmetric matrix by `x` without rebuilding it.

The same `N²` multiply-adds a dense product would perform, reading each stored
entry twice instead of storing it twice.
"""
@generated function _symmetric_matvec(packed::SVector{L}, x::SVector{N}) where {L, N}
    rows = [Expr(:call, :+, [:(packed[$(_triangle_index(i, j))] * x[$j]) for j in 1:N]...)
            for i in 1:N]
    return :(SVector{$N}($(rows...)))
end

# A symmetric velocity block is stored as its upper triangle; a non-symmetric one
# has to be kept whole and is transposed on apply.
@inline _store_velocity_block!(Ablocks::AbstractVector{<:SVector}, iel, A) =
    (Ablocks[iel] = _pack_symmetric(A); nothing)
@inline _store_velocity_block!(Ablocks, iel, A) = (Ablocks[iel] = A; nothing)
@inline _velocity_apply(A::SVector, λv) = _symmetric_matvec(A, λv)
@inline _velocity_apply(A, λv) = transpose(A) * λv

"""
    _assert_frozen_symmetry(worst, quantity, what, nels, T)

Raise if a structural identity the frozen operator's storage layout relies on
does not hold to `sqrt(eps(T))`.

The check cannot be tripped by a caller: it is `plastic === nothing` that makes
the element tangent symmetric, so the packing and the identity are enabled by the
same condition. It guards against the tangent itself changing — a new rheology, or
an edit to the momentum residual — in which case the adjoint must fail rather
than return a gradient computed from a layout the operator no longer satisfies.
"""
function _assert_frozen_symmetry(worst, quantity, what, nels, ::Type{T}) where {T}
    worst ≤ sqrt(eps(T)) && return nothing
    return error(
        "the frozen adjoint operator $what, which assumes the element tangent is " *
        "symmetric, but the assembled blocks disagree: max $quantity = $worst over " *
        "$nels elements. Without a plastic model the tangent is symmetric and this " *
        "cannot happen, so the element operator no longer has the symmetry the " *
        "storage layout depends on.")
end

# A symmetric operator keeps no pressure-coupling block: Cᵀ is B.
@inline _store_pressure_coupling!(::Nothing, _, _) = nothing
@inline _store_pressure_coupling!(Cblocks, iel, C) = (Cblocks[iel] = C; nothing)
@inline _pressure_coupling(::Nothing, Bblocks, iel) = Bblocks[iel]
@inline _pressure_coupling(Cblocks, _, iel) = transpose(Cblocks[iel])

@kernel function velocity_operator_assembly_kernel!(
        Ablocks, ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        @Const(vx), @Const(vy), @Const(P), @Const(P0), @Const(T), @Const(T0),
        @Const(el2n_v), @Const(el2nP), @Const(geo_v), @Const(geo_P),
        @Const(phases_v), @Const(phases_P), τ_old, plastic,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, @Const(MP),
        Nq, NqP, ∂N∂ξ_v, ::Val{NV}, ::Val{NP},
    ) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, Axx, Axy, Ayx, Ayy = element_augmented_momentum_jacobians(
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, η, G, α, ρ0, K, g, Tref, ηb, Δt,
        γ_eff, MP, Nq, NqP, ∂N∂ξ_v, iel, Val(NV), Val(NP), τ_old, plastic,
    )
    A = vcat(hcat(Axx, Axy), hcat(Ayx, Ayy))
    Ablocks[iel] = A
    for (i, inod) in enumerate(local_nodes_v)
        Atomix.@atomic :monotonic ∂Rv_x∂vx[inod] += sum(abs(A[i, j]) for j in 1:2NV)
        Atomix.@atomic :monotonic PC_vx[inod] += abs(A[i, i])
        Atomix.@atomic :monotonic ∂Rv_y∂vy[inod] += sum(abs(A[NV + i, j]) for j in 1:2NV)
        Atomix.@atomic :monotonic PC_vy[inod] += abs(A[NV + i, NV + i])
    end
end

function assemble_velocity_operator(
        dr, mesh_stokes, geo_v, geo_P,
        element_v::ReferenceElement{TV}, element_P::ReferenceElement{TP},
        phases_v, phases_P, τ_old, plastic, G, Δt, γP, backend, workgroup,
    ) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    Nq = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)
    ∂N∂ξ_v = shape_function_gradients(element_v)
    Ablocks = similar(dr.vx, SMatrix{2NV, 2NV, eltype(dr.vx), 4NV * NV}, mesh_stokes.nels)
    fill!(dr.∂Rv_x∂vx, 0)
    fill!(dr.PC_vx, 0)
    fill!(dr.∂Rv_y∂vy, 0)
    fill!(dr.PC_vy, 0)
    velocity_operator_assembly_kernel!(backend, workgroup)(
        Ablocks, dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, phases_v, phases_P,
        τ_old, plastic, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, dr.ηb,
        Δt, γP, dr.M_P, Nq, NqP, ∂N∂ξ_v, Val(NV), Val(NP);
        ndrange = mesh_stokes.nels,
    )
    KA.synchronize(backend)
    return FrozenVelocityOperator(Ablocks)
end

@kernel function velocity_operator_apply_kernel!(
        yx, yy, @Const(Ablocks), @Const(x), @Const(y), @Const(el2n_v), ::Val{NV},
    ) where {NV}
    iel = @index(Global)
    nodes = local_nodes_of(el2n_v, iel, Val(NV))
    out = Ablocks[iel] * vcat(
        _gather_local(x, nodes, Val(NV)), _gather_local(y, nodes, Val(NV)))
    for (i, inod) in enumerate(nodes)
        Atomix.@atomic :monotonic yx[inod] += out[i]
        Atomix.@atomic :monotonic yy[inod] += out[NV + i]
    end
end

function apply_velocity_operator!(
        yx, yy, op::FrozenVelocityOperator, x, y, mesh_stokes,
        element_v::ReferenceElement{TV}, backend, workgroup,
    ) where {TV <: AbstractElement{2, NV}} where {NV}
    fill!(yx, 0)
    fill!(yy, 0)
    velocity_operator_apply_kernel!(backend, workgroup)(
        yx, yy, op.A, x, y, mesh_stokes.el2n, Val(NV);
        ndrange = mesh_stokes.nels,
    )
    KA.synchronize(backend)
    return nothing
end

function estimate_velocity_λmax(
        op::FrozenVelocityOperator, mesh_stokes, element_v,
        PC_vx, PC_vy, vx_nodes, vy_nodes, backend, workgroup;
        max_iterations = 12, rtol = 1.0e-2, x = nothing, y = nothing,
    )
    if x === nothing
        x, y = similar(PC_vx), similar(PC_vy)
        x .= sin.(eachindex(x) .* 0.7)
        y .= cos.(eachindex(y) .* 1.3)
    end
    zx, zy, ax, ay = similar(PC_vx), similar(PC_vy), similar(PC_vx), similar(PC_vy)
    zero_x = fill!(similar(PC_vx, length(vx_nodes)), 0)
    zero_y = fill!(similar(PC_vy, length(vy_nodes)), 0)
    apply_dirichlet!(x, vx_nodes, zero_x, backend, workgroup)
    apply_dirichlet!(y, vy_nodes, zero_y, backend, workgroup)
    λ = zero(eltype(PC_vx))
    for it in 1:max_iterations
        n = sqrt(dot(x, x) + dot(y, y))
        n > 0 || throw(ArgumentError("power iteration collapsed to the zero vector"))
        x ./= n
        y ./= n
        @. zx = x / sqrt(PC_vx)
        @. zy = y / sqrt(PC_vy)
        apply_velocity_operator!(ax, ay, op, zx, zy, mesh_stokes, element_v, backend, workgroup)
        @. ax /= sqrt(PC_vx)
        @. ay /= sqrt(PC_vy)
        apply_dirichlet!(ax, vx_nodes, zero_x, backend, workgroup)
        apply_dirichlet!(ay, vy_nodes, zero_y, backend, workgroup)
        λnew = sqrt(dot(ax, ax) + dot(ay, ay))
        copyto!(x, ax)
        copyto!(y, ay)
        if it > 1 && abs(λnew - λ) ≤ rtol * λnew
            return λnew, it, x, y
        end
        λ = λnew
    end
    isfinite(λ) && λ > 0 ||
        throw(ArgumentError("power iteration produced a non-positive λmax: $λ"))
    return λ, max_iterations, x, y
end

"""
    element_adjoint_operator_blocks(...) -> (local_nodes_v, local_nodes_P, A, B, C)

Build the three transposed-operator blocks for element `iel` at the current
forward state. See [`FrozenAdjointOperator`](@ref) for what each block contains.
"""
@inline function element_adjoint_operator_blocks(
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, ∂N∂ξ_v, iel, ::Val{NV}, ::Val{NP},
    ) where {NV, NP}
    local_nodes_v, ∂RVx∂vx, ∂RVx∂vy, ∂RVy∂vx, ∂RVy∂vy =
        element_augmented_momentum_jacobians(
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, ∂N∂ξ_v, iel, Val(NV), Val(NP), τ_old, plastic,
    )
    A = vcat(hcat(∂RVx∂vx, ∂RVx∂vy), hcat(∂RVy∂vx, ∂RVy∂vy))

    local_nodes_P = local_nodes_of(el2nP, iel, Val(NP))
    geo_v_el = element_geometry(geo_v, iel, ∂N∂ξ_v)
    geo_P_el = geo_P[iel]
    vxloc = _gather_local(vx, local_nodes_v, Val(NV))
    vyloc = _gather_local(vy, local_nodes_v, Val(NV))
    P_loc = _gather_local(P, local_nodes_P, Val(NP))
    P0loc = _gather_local(P0, local_nodes_P, Val(NP))
    T_loc = _gather_local(T, local_nodes_P, Val(NP))
    T0loc = _gather_local(T0, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, iel, Val(NV), quadrature_points_val(geo_v_el))
    phase_v = _gather_phase(phases_v, local_nodes_v, iel, Val(NV))
    phase_P = _gather_phase(phases_P, local_nodes_P, iel, Val(NP))

    # ∂Rv/∂P with Pnum independent, matching the momentum residual the forward
    # solve evaluates: the augmentation belongs to A, not here.
    Pnum_loc = zero(P_loc)
    B = ForwardDiff.jacobian(
        P_arg -> begin
            Rx, Ry = integrate_momentum_residual(
                (vxloc, vyloc), P_arg, Pnum_loc, T_loc,
                geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP,
                τ_old_loc, plastic,
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
        Ablocks, Bblocks, Cblocks, defect_A, defect_C,
        ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        @Const(vx), @Const(vy),
        @Const(P), @Const(P0),
        @Const(T), @Const(T0),
        @Const(el2n_v), @Const(el2nP),
        @Const(geo_v), @Const(geo_P),
        @Const(phases_v), @Const(phases_P),
        τ_old, plastic,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        @Const(MP),
        Nq, NqP, ∂N∂ξ_v, ::Val{NV}, ::Val{NP},
    ) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, _, A, B, C = element_adjoint_operator_blocks(
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, ∂N∂ξ_v, iel, Val(NV), Val(NP),
    )
    # One element owns one entry of each block array, so no atomics are needed.
    _store_velocity_block!(Ablocks, iel, A)
    Bblocks[iel] = B
    _store_pressure_coupling!(Cblocks, iel, C)
    normA = norm(A)
    normC = norm(C)
    defect_A[iel] = iszero(normA) ? zero(normA) : norm(A - transpose(A)) / normA
    defect_C[iel] = iszero(normC) ? zero(normC) : norm(C - transpose(B)) / normC
    rowsums_x = SVector{NV}(ntuple(i -> sum(abs(A[i, j]) for j in 1:2NV), Val(NV)))
    rowsums_y = SVector{NV}(ntuple(i -> sum(abs(A[NV + i, j]) for j in 1:2NV), Val(NV)))
    for (i, inod) in enumerate(local_nodes_v)
        Atomix.@atomic :monotonic ∂Rv_x∂vx[inod] += rowsums_x[i]
        Atomix.@atomic :monotonic PC_vx[inod] += abs(A[i, i])
        Atomix.@atomic :monotonic ∂Rv_y∂vy[inod] += rowsums_y[i]
        Atomix.@atomic :monotonic PC_vy[inod] += abs(A[NV + i, NV + i])
    end
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
    ∂N∂ξ_v = shape_function_gradients(element_v)
    nels = mesh_stokes.nels
    Tv = eltype(dr.vx)

    # A viscous tangent is symmetric: A keeps only its upper triangle and C is Bᵀ
    # and is not stored at all. The kernel still forms both blocks whole and
    # reports how far each identity is from holding; the checks below turn a
    # violated assumption into an error rather than a wrong gradient.
    symmetric = plastic === nothing
    Ablocks = symmetric ?
        similar(dr.vx, SVector{_packed_symmetric_length(2NV), Tv}, nels) :
        similar(dr.vx, SMatrix{2NV, 2NV, Tv, 4NV * NV}, nels)
    Bblocks = similar(dr.vx, SMatrix{2NV, NP, Tv, 2NV * NP}, nels)
    Cblocks = symmetric ? nothing :
        similar(dr.vx, SMatrix{NP, 2NV, Tv, 2NV * NP}, nels)
    defect_A = similar(dr.vx, nels)
    defect_C = similar(dr.vx, nels)

    fill!(dr.∂Rv_x∂vx, 0)
    fill!(dr.PC_vx, 0)
    fill!(dr.∂Rv_y∂vy, 0)
    fill!(dr.PC_vy, 0)
    adjoint_operator_assembly_kernel!(backend, workgroup)(
        Ablocks, Bblocks, Cblocks, defect_A, defect_C,
        dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P,
        phases_v, phases_P, τ_old, plastic,
        dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, dr.ηb, Δt, γP, dr.M_P,
        Nq, NqP, ∂N∂ξ_v, Val(NV), Val(NP);
        ndrange = nels,
    )
    KA.synchronize(backend)
    if symmetric
        _assert_frozen_symmetry(maximum(defect_A), "‖A - Aᵀ‖/‖A‖",
            "packed only the upper triangle of its velocity block", nels, Tv)
        _assert_frozen_symmetry(maximum(defect_C), "‖C - Bᵀ‖/‖C‖",
            "dropped its pressure-coupling block", nels, Tv)
    end
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

    resv = _velocity_apply(Ablocks[iel], λv) + _pressure_coupling(Cblocks, Bblocks, iel) * λP_loc
    resp = transpose(Bblocks[iel]) * λv

    for (i, inod) in enumerate(local_nodes_v)
        Atomix.@atomic :monotonic dvx[inod] += resv[i]
        Atomix.@atomic :monotonic dvy[inod] += resv[NV + i]
    end
    # Pressure degrees of freedom are discontinuous, hence unshared.
    _add_local!(dP, local_nodes_P, resp, Val(false))
end

"""
    estimate_adjoint_λmax(op, mesh_stokes, element_v, element_P, PC_vx, PC_vy,
                          vx_nodes, vy_nodes, backend, workgroup;
                          max_iterations = 100, rtol = 1.0e-3) -> (λmax, iterations)

Estimate the largest eigenvalue of the Jacobi-preconditioned velocity block by
power iteration, returning it with the number of iterations taken.

The alternative is a Gershgorin bound, which is correct as a bound but typically
several times above the true value. Since the dynamic-relaxation step is
`Δτ = 2/sqrt(λmax)·CFL_v` and the iteration count scales as `sqrt(λmax/λmin)`,
overestimating λmax by a factor `f` costs about `sqrt(f)` in iterations. Each
power iteration costs one operator apply, which is negligible beside a solve.

The pressure adjoint is held at zero throughout, so this measures the velocity
block alone, and the Dirichlet rows are projected out at every step to match the
subspace the solver actually iterates on.
"""
function estimate_adjoint_λmax(
        op::FrozenAdjointOperator, mesh_stokes,
        element_v::ReferenceElement{TV},
        element_P::ReferenceElement{TP},
        PC_vx, PC_vy, vx_nodes, vy_nodes, backend, workgroup;
        max_iterations = 100, rtol = 1.0e-3,
    ) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    x_vx = similar(PC_vx)
    x_vy = similar(PC_vy)
    z_vx = similar(PC_vx)
    z_vy = similar(PC_vy)
    y_vx = similar(PC_vx)
    y_vy = similar(PC_vy)
    zeroP = fill!(similar(PC_vx, mesh_stokes.nnodesP), 0)
    scratchP = similar(zeroP)
    zero_vx_bc = fill!(similar(PC_vx, length(vx_nodes)), 0)
    zero_vy_bc = fill!(similar(PC_vy, length(vy_nodes)), 0)

    # A deterministic oscillatory start: a constant vector is a poor seed for an
    # elliptic operator, whose dominant mode is the most oscillatory one.
    x_vx .= sin.(eachindex(x_vx) .* 0.7)
    x_vy .= cos.(eachindex(x_vy) .* 1.3)
    apply_dirichlet!(x_vx, vx_nodes, zero_vx_bc, backend, workgroup)
    apply_dirichlet!(x_vy, vy_nodes, zero_vy_bc, backend, workgroup)

    λ = zero(eltype(PC_vx))
    iterations = 0
    for it in 1:max_iterations
        iterations = it
        nx = sqrt(dot(x_vx, x_vx) + dot(x_vy, x_vy))
        nx > 0 || throw(ArgumentError("power iteration collapsed to the zero vector"))
        x_vx ./= nx
        x_vy ./= nx

        # P⁻¹A and P⁻¹/²AP⁻¹/² are similar, hence have the same eigenvalues.
        # Iterating on the latter preserves symmetry and avoids the non-normal
        # transients introduced by left Jacobi scaling.
        @. z_vx = x_vx / sqrt(PC_vx)
        @. z_vy = x_vy / sqrt(PC_vy)
        apply_adjoint_operator!(
            y_vx, y_vy, scratchP, op, z_vx, z_vy, zeroP,
            mesh_stokes, element_v, element_P, backend, workgroup,
        )
        @. y_vx /= sqrt(PC_vx)
        @. y_vy /= sqrt(PC_vy)
        apply_dirichlet!(y_vx, vx_nodes, zero_vx_bc, backend, workgroup)
        apply_dirichlet!(y_vy, vy_nodes, zero_vy_bc, backend, workgroup)

        λ_new = sqrt(dot(y_vx, y_vx) + dot(y_vy, y_vy))
        copyto!(x_vx, y_vx)
        copyto!(x_vy, y_vy)
        converged = it > 1 && abs(λ_new - λ) ≤ rtol * λ_new
        λ = λ_new
        converged && break
    end
    isfinite(λ) && λ > 0 ||
        throw(ArgumentError("power iteration produced a non-positive λmax: $λ"))
    return λ, iterations
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
