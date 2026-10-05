"""
    FrozenAdjointOperator(A, B, C, D, W)

Element blocks of the transposed Stokes adjoint operator, held for the lifetime of
one adjoint solve.

At a converged forward state the adjoint residual is affine in `λ` with a constant
operator, so the blocks are assembled once and applied many times:

```
ResλV = objective_v + Aᵀλv + CᵀλP
ResλP = Bᵀλv + Dᵀ(Wᵀλv + λP)
```

- `A` is the augmented velocity block `∂Rv/∂v`, `dim·NV`×`dim·NV` per element in
  `dim` dimensions, with the degrees of freedom stacked by component (all of
  `vx`, then `vy`, …), or its packed upper triangle. It already carries the Powell-Hestenes
  term `Bnum·(γP/M_P)·C`, because the assembler forms `Pnum` inline from
  element-local pressures; that coupling stays inside one element only while the
  pressure space is discontinuous.
- `B` is `∂Rv/∂P`, `dim·NV`×`NP` per element, differentiated with `Pnum` held as an
  independent variable so it excludes the augmentation already inside `A`.
- `C` is `∂RP/∂v`, `NP`×`dim·NV` per element, or `nothing`.
- `D` is `∂RP/∂P`, `NP`×`NP` per element, or `nothing`. It is the storage term
  `−∫ Nᵢ Nⱼ/(ηb Δt) dΩ`, so it vanishes identically when every phase is
  incompressible in the bulk-viscosity sense, and only then.
- `W` is `∂Rv/∂Pnum · Γ`, `dim·NV`×`NP` per element, or `nothing`, where `Γ` is
  `γP/M_P` on the element's pressure nodes. It is the route from the velocity
  adjoint into the pressure row through the augmentation, carried as one block
  because only the product is ever needed.

The pressure row carries two terms rather than one, and both are consequences of
`A` carrying the augmentation. The system the operator transposes is
`[Aaug Baug; C D]` with `Aaug = A + Bnum·Γ·C` and `Baug = B + Bnum·Γ·D = B + W·D`,
because `Pnum` depends on `P` through `D` exactly as it depends on `v` through
`C`. So `Baugᵀλv = Bᵀλv + Dᵀ·Wᵀλv` and the whole row is `Bᵀλv + Dᵀ(Wᵀλv + λP)`.
Dropping it was GAP-25: with `ηb` finite the pressure row of the operator is not
the pressure row of the transposed Jacobian, and the gradient is wrong by that
amount while every incompressible test stays green, because `D` is exactly zero
there.

`Bnum` is kept rather than reusing `B` because the two differ: `Pnum` reaches the
momentum residual only through `Ptotal = P + Pnum`, while `P` also sets the
density through `ρ = ρ0(1 − α(T − Tref) + P/K)`. They coincide exactly in the
incompressible gauge and nowhere else, and an oracle that switches the
augmentation on at `K = 10` reads the difference as a 2.6 × 10⁻² relative error in
the pressure row.

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
struct FrozenAdjointOperator{TA, TB, TC, TD, TW}
    A::TA
    B::TB
    C::TC
    D::TD
    W::TW
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
    rows = [
        Expr(:call, :+, [:(packed[$(_triangle_index(i, j))] * x[$j]) for j in 1:N]...)
            for i in 1:N
    ]
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

Two conditions make the element tangent symmetric, and [`_symmetric_tangent`](@ref)
requires both before the packing is enabled: no plastic model, and no
pressure-dependent density. A non-associated flow rule breaks `A == Aᵀ`; a finite
bulk modulus breaks `C == Bᵀ`, because `ρ = ρ0(1 − α(T − Tref) + P/K)` puts a
pressure derivative in the momentum residual that the pressure residual has no
counterpart for. Measured on a 2×2 element mesh at `K = 10`, `‖C − Bᵀ‖/‖C‖` is
7.8 × 10⁻³ with `plastic === nothing`, which is why the condition is stated here
rather than assumed from the plastic model alone.

The check therefore guards two things: a caller reaching a configuration the
layout does not cover, and the tangent itself changing under a new rheology or an
edit to the momentum residual. Either way the adjoint must fail rather than return
a gradient computed from a layout the operator no longer satisfies.
"""
function _assert_frozen_symmetry(worst, quantity, what, nels, ::Type{T}) where {T}
    worst ≤ sqrt(eps(T)) && return nothing
    return error(
        "the frozen adjoint operator $what, which assumes the element tangent is " *
            "symmetric, but the assembled blocks disagree: max $quantity = $worst over " *
            "$nels elements. The layout is enabled only for a viscous model with an " *
            "incompressible momentum residual, so either that condition is no longer " *
            "what makes the tangent symmetric, or the element operator has changed."
    )
end

"""
    _symmetric_tangent(plastic, K) -> Bool

Whether the element tangent is symmetric, and hence whether the packed storage
layout of [`FrozenAdjointOperator`](@ref) may be used.

`K` is the per-phase bulk modulus; `Inf` is the incompressible gauge in which the
density carries no pressure dependence.
"""
@inline _symmetric_tangent(plastic, K) = plastic === nothing && all(isinf, K)

# A symmetric operator keeps no pressure-coupling block: Cᵀ is B.
@inline _store_pressure_coupling!(::Nothing, _, _) = nothing
@inline _store_pressure_coupling!(Cblocks, iel, C) = (Cblocks[iel] = C; nothing)
@inline _pressure_coupling(::Nothing, Bblocks, iel) = Bblocks[iel]
@inline _pressure_coupling(Cblocks, _, iel) = transpose(Cblocks[iel])

# An incompressible bulk keeps no storage block: ∂RP/∂P is identically zero, so
# the pressure row is Bᵀλv alone and nothing is allocated for it.
@inline _store_pressure_storage!(::Nothing, ::Nothing, _, _, _) = nothing
@inline _store_pressure_storage!(Dblocks, Wblocks, iel, D, W) =
    (Dblocks[iel] = D; Wblocks[iel] = W; nothing)
@inline _pressure_storage(::Nothing, ::Nothing, _, _, λP_loc) = zero(λP_loc)
@inline _pressure_storage(Dblocks, Wblocks, iel, λv, λP_loc) =
    transpose(Dblocks[iel]) * (transpose(Wblocks[iel]) * λv .+ λP_loc)

@kernel function velocity_operator_assembly_kernel!(
        Ablocks, ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        @Const(vx), @Const(vy), @Const(P), @Const(P0), @Const(T), @Const(T0),
        @Const(el2n_v), @Const(el2nP), @Const(geo_v), @Const(geo_P),
        @Const(phases_v), @Const(phases_P), τ_old, plastic, γ_history,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, @Const(MP),
        Nq, NqP, ∂N∂ξ_v, ::Val{NV}, ::Val{NP},
    ) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, Axx, Axy, Ayx, Ayy = element_augmented_momentum_jacobians(
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, η, G, α, ρ0, K, g, Tref, ηb, Δt,
        γ_eff, MP, Nq, NqP, ∂N∂ξ_v, iel, Val(NV), Val(NP), τ_old, plastic, γ_history,
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
        phases_v, phases_P, τ_old, plastic, G, Δt, γP, backend, workgroup;
        γ_history = nothing, pressure_bulk = dr.ηb,
    ) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    Nq = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)
    ∂N∂ξ_v = shape_function_gradients(element_v)
    Ablocks = similar(dr.v.x, SMatrix{2NV, 2NV, eltype(dr.v.x), 4NV * NV}, mesh_stokes.nels)
    fill!(dr.∂Rv∂v.x, 0)
    fill!(dr.PC_v.x, 0)
    fill!(dr.∂Rv∂v.y, 0)
    fill!(dr.PC_v.y, 0)
    velocity_operator_assembly_kernel!(backend, workgroup)(
        Ablocks, dr.∂Rv∂v.x, dr.PC_v.x, dr.∂Rv∂v.y, dr.PC_v.y,
        dr.v.x, dr.v.y, dr.P, dr.P0, dr.T, dr.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, phases_v, phases_P,
        τ_old, plastic, γ_history, dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, pressure_bulk,
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
        _gather_local(x, nodes, Val(NV)), _gather_local(y, nodes, Val(NV))
    )
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
    element_adjoint_operator_blocks(...) -> (local_nodes_v, local_nodes_P, A, B, C, D, W)

Build the transposed-operator blocks for element `iel` at the current forward
state. See [`FrozenAdjointOperator`](@ref) for what each block contains.

`D` and `W` are formed unconditionally — two small Jacobians cost nothing beside
`A` — and the assembler decides whether to keep them.
"""
@inline function element_adjoint_operator_blocks(
        v::NTuple{Dim}, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, ∂N∂ξ_v, iel, ::Val{NV}, ::Val{NP}, Pf = nothing,
    ) where {Dim, NV, NP}
    local_nodes_v, J = element_augmented_momentum_jacobians(
        v, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, ∂N∂ξ_v, iel, Val(NV), Val(NP), τ_old, plastic, nothing, Pf,
    )
    # Velocity degrees of freedom are stacked by component: all of vx, then vy, …
    A = vcat(map(row -> hcat(row...), J)...)

    local_nodes_P = local_nodes_of(el2nP, iel, Val(NP))
    geo_v_el = element_geometry(geo_v, iel, ∂N∂ξ_v)
    geo_P_el = geo_P[iel]
    vloc = ntuple(d -> _gather_local(v[d], local_nodes_v, Val(NV)), Val(Dim))
    P_loc = _gather_local(P, local_nodes_P, Val(NP))
    P0loc = _gather_local(P0, local_nodes_P, Val(NP))
    T_loc = _gather_local(T, local_nodes_P, Val(NP))
    T0loc = _gather_local(T0, local_nodes_P, Val(NP))
    Pf_loc = _gather_or_nothing(Pf, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, iel, Val(NV), quadrature_points_val(geo_v_el))
    phase_v = _gather_phase(phases_v, local_nodes_v, iel, Val(NV))
    phase_P = _gather_phase(phases_P, local_nodes_P, iel, Val(NP))
    momentum(v_arg, P_arg, Pnum_arg) = vcat(
        integrate_momentum_residual(
            v_arg, P_arg, Pnum_arg, T_loc,
            geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP,
            τ_old_loc, plastic, nothing, nothing, nothing, nothing, Pf_loc,
        )...,
    )

    # ∂Rv/∂P with Pnum independent, matching the momentum residual the forward
    # solve evaluates: the augmentation belongs to A, not here.
    Pnum_loc = zero(P_loc)
    B = ForwardDiff.jacobian(P_arg -> momentum(vloc, P_arg, Pnum_loc), P_loc)

    C = ForwardDiff.jacobian(
        v_arg -> integrate_PH_pressure_residual(
            _unstack_velocity(v_arg, Val(Dim), Val(NV)),
            P_loc, P0loc, T_loc, T0loc,
            geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP;
            K = _pressure_bulk_modulus(plastic, K),
        ),
        vcat(vloc...),
    )

    # ∂RP/∂P: the storage term alone, since the divergence, the thermal rate and
    # the source do not depend on the pressure.
    D = ForwardDiff.jacobian(
        P_arg -> integrate_PH_pressure_residual(
            vloc, P_arg, P0loc, T_loc, T0loc,
            geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
        ),
        P_loc,
    )

    # ∂Rv/∂Pnum, column-scaled by Γ. Pnum reaches the residual only through the
    # total pressure, while P also sets the density, so this is not B unless the
    # bulk modulus is infinite.
    Bnum = ForwardDiff.jacobian(Pnum_arg -> momentum(vloc, P_loc, Pnum_arg), Pnum_loc)
    Γ = _gather_or_scalar(γ_eff, local_nodes_P, Val(NP)) ./ _gather_local(MP, local_nodes_P, Val(NP))
    W = Bnum .* transpose(Γ)

    return local_nodes_v, local_nodes_P, A, B, C, D, W
end

@inline element_adjoint_operator_blocks(vx::AbstractVector, vy::AbstractVector, args...) =
    element_adjoint_operator_blocks((vx, vy), args...)

# Split a component-stacked element velocity back into one vector per component.
@inline _unstack_velocity(u, ::Val{Dim}, ::Val{NV}) where {Dim, NV} =
    ntuple(d -> u[SVector{NV}(ntuple(i -> (d - 1) * NV + i, Val(NV)))], Val(Dim))

# Element row sums and diagonal of the stacked velocity block, scattered per component.
@inline function _scatter_velocity_diagnostics!(
        ∂Rv∂v::NTuple{Dim}, PC::NTuple{Dim}, nodes, A, ::Val{NV},
    ) where {Dim, NV}
    for (i, inod) in enumerate(nodes)
        ntuple(Val(Dim)) do c
            r = (c - 1) * NV + i
            Atomix.@atomic :monotonic ∂Rv∂v[c][inod] += sum(abs(A[r, j]) for j in 1:(Dim * NV))
            Atomix.@atomic :monotonic PC[c][inod] += abs(A[r, r])
            nothing
        end
    end
    return nothing
end

# Scatter a component-stacked element vector into one global array per component.
@inline function _scatter_stacked!(dv::NTuple{Dim}, nodes, res, ::Val{NV}) where {Dim, NV}
    for (i, inod) in enumerate(nodes)
        ntuple(Val(Dim)) do c
            Atomix.@atomic :monotonic dv[c][inod] += res[(c - 1) * NV + i]
            nothing
        end
    end
    return nothing
end

@kernel function adjoint_operator_assembly_kernel!(
        Ablocks, Bblocks, Cblocks, Dblocks, Wblocks, defect_A, defect_C,
        ∂Rv∂v, PC,
        @Const(v),
        @Const(P), @Const(P0),
        @Const(T), @Const(T0),
        @Const(el2n_v), @Const(el2nP),
        @Const(geo_v), @Const(geo_P),
        @Const(phases_v), @Const(phases_P),
        τ_old, plastic,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        @Const(MP),
        Nq, NqP, ∂N∂ξ_v, ::Val{NV}, ::Val{NP}, @Const(Pf),
    ) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, _, A, B, C, D, W = element_adjoint_operator_blocks(
        v, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, ∂N∂ξ_v, iel, Val(NV), Val(NP), Pf,
    )
    # One element owns one entry of each block array, so no atomics are needed.
    _store_velocity_block!(Ablocks, iel, A)
    Bblocks[iel] = B
    _store_pressure_coupling!(Cblocks, iel, C)
    _store_pressure_storage!(Dblocks, Wblocks, iel, D, W)
    normA = norm(A)
    normC = norm(C)
    defect_A[iel] = iszero(normA) ? zero(normA) : norm(A - transpose(A)) / normA
    defect_C[iel] = iszero(normC) ? zero(normC) : norm(C - transpose(B)) / normC
    _scatter_velocity_diagnostics!(∂Rv∂v, PC, local_nodes_v, A, Val(NV))
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
    ) where {TV <: AbstractElement{Dim, NV}, TP <: AbstractElement{Dim, NP}} where {Dim, NV, NP}
    Nq = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)
    ∂N∂ξ_v = shape_function_gradients(element_v)
    nels = mesh_stokes.nels
    v = velocity(dr)
    ∂Rv∂v = Tuple(getfield(dr, :∂Rv∂v))
    PC = Tuple(getfield(dr, :PC_v))
    Tv = eltype(first(v))
    NU = Dim * NV

    # A viscous, incompressible tangent is symmetric: A keeps only its upper
    # triangle and C is Bᵀ and is not stored at all. The kernel still forms both
    # blocks whole and reports how far each identity is from holding; the checks
    # below turn a violated assumption into an error rather than a wrong gradient.
    symmetric = _symmetric_tangent(plastic, dr.K)
    Ablocks = symmetric ?
        similar(first(v), SVector{_packed_symmetric_length(NU), Tv}, nels) :
        similar(first(v), SMatrix{NU, NU, Tv, NU * NU}, nels)
    Bblocks = similar(first(v), SMatrix{NU, NP, Tv, NU * NP}, nels)
    Cblocks = symmetric ? nothing :
        similar(first(v), SMatrix{NP, NU, Tv, NU * NP}, nels)
    # ∂RP/∂P is the storage term, so it is identically zero under an infinite bulk
    # viscosity and is not allocated there.
    stores = any(isfinite, dr.ηb)
    Dblocks = stores ? similar(first(v), SMatrix{NP, NP, Tv, NP * NP}, nels) : nothing
    Wblocks = stores ? similar(first(v), SMatrix{NU, NP, Tv, NU * NP}, nels) : nothing
    defect_A = similar(first(v), nels)
    defect_C = similar(first(v), nels)

    foreach(a -> fill!(a, 0), ∂Rv∂v)
    foreach(a -> fill!(a, 0), PC)
    adjoint_operator_assembly_kernel!(backend, workgroup)(
        Ablocks, Bblocks, Cblocks, Dblocks, Wblocks, defect_A, defect_C,
        ∂Rv∂v, PC, v, dr.P, dr.P0, dr.T, dr.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P,
        phases_v, phases_P, τ_old, plastic,
        dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, dr.ηb, Δt, γP, dr.M_P,
        Nq, NqP, ∂N∂ξ_v, Val(NV), Val(NP), dr.Pf;
        ndrange = nels,
    )
    KA.synchronize(backend)
    if symmetric
        _assert_frozen_symmetry(
            maximum(defect_A), "‖A - Aᵀ‖/‖A‖",
            "packed only the upper triangle of its velocity block", nels, Tv
        )
        _assert_frozen_symmetry(
            maximum(defect_C), "‖C - Bᵀ‖/‖C‖",
            "dropped its pressure-coupling block", nels, Tv
        )
    end
    return FrozenAdjointOperator(Ablocks, Bblocks, Cblocks, Dblocks, Wblocks)
end

@kernel function adjoint_operator_apply_kernel!(
        dv, dP,
        @Const(Ablocks), @Const(Bblocks), @Const(Cblocks), @Const(Dblocks), @Const(Wblocks),
        @Const(λv), @Const(λP),
        @Const(el2n_v), @Const(el2nP),
        ::Val{NV}, ::Val{NP},
    ) where {NV, NP}
    iel = @index(Global)
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP, iel, Val(NP))

    λv_loc = vcat(map(λ -> _gather_local(λ, local_nodes_v, Val(NV)), λv)...)
    λP_loc = _gather_local(λP, local_nodes_P, Val(NP))

    resv = _velocity_apply(Ablocks[iel], λv_loc) + _pressure_coupling(Cblocks, Bblocks, iel) * λP_loc
    resp = transpose(Bblocks[iel]) * λv_loc + _pressure_storage(Dblocks, Wblocks, iel, λv_loc, λP_loc)

    _scatter_stacked!(dv, local_nodes_v, resv, Val(NV))
    # Pressure degrees of freedom are discontinuous, hence unshared.
    _add_local!(dP, local_nodes_P, resp, Val(false))
end

"""
    estimate_adjoint_λmax(op, mesh_stokes, element_v, element_P, PC, v_nodes,
                          backend, workgroup;
                          max_iterations = 100, rtol = 1.0e-3) -> (λmax, iterations)
    estimate_adjoint_λmax(op, mesh_stokes, element_v, element_P, PC_vx, PC_vy,
                          vx_nodes, vy_nodes, backend, workgroup; kwargs...)

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
        op, mesh_stokes,
        element_v::ReferenceElement{TV},
        element_P::ReferenceElement{TP},
        PC::NTuple{Dim}, v_nodes::NTuple{Dim}, backend, workgroup;
        max_iterations = 100, rtol = 1.0e-3,
    ) where {TV <: AbstractElement{Dim, NV}, TP <: AbstractElement{Dim, NP}} where {Dim, NV, NP}
    x = map(similar, PC)
    z = map(similar, PC)
    y = map(similar, PC)
    zeroP = fill!(similar(first(PC), mesh_stokes.nnodesP), 0)
    scratchP = similar(zeroP)
    zero_bc = map(nodes -> fill!(similar(first(PC), length(nodes)), 0), v_nodes)
    project!(u) = foreach((a, nodes, vals) -> apply_dirichlet!(a, nodes, vals, backend, workgroup), u, v_nodes, zero_bc)
    stacked_norm(u) = sqrt(sum(a -> dot(a, a), u))

    # A deterministic oscillatory start: a constant vector is a poor seed for an
    # elliptic operator, whose dominant mode is the most oscillatory one.
    foreach(enumerate(x)) do (d, a)
        a .= _power_seed.(eachindex(a), d)
    end
    project!(x)

    λ = zero(eltype(first(PC)))
    iterations = 0
    for it in 1:max_iterations
        iterations = it
        nx = stacked_norm(x)
        nx > 0 || throw(ArgumentError("power iteration collapsed to the zero vector"))
        foreach(a -> a ./= nx, x)

        # P⁻¹A and P⁻¹/²AP⁻¹/² are similar, hence have the same eigenvalues.
        # Iterating on the latter preserves symmetry and avoids the non-normal
        # transients introduced by left Jacobi scaling.
        foreach((zc, xc, pc) -> (@. zc = xc / sqrt(pc)), z, x, PC)
        apply_adjoint_operator!(
            y, scratchP, op, z, zeroP, mesh_stokes, element_v, element_P, backend, workgroup,
        )
        foreach((yc, pc) -> (@. yc /= sqrt(pc)), y, PC)
        project!(y)

        λ_new = stacked_norm(y)
        foreach(copyto!, x, y)
        converged = it > 1 && abs(λ_new - λ) ≤ rtol * λ_new
        λ = λ_new
        converged && break
    end
    isfinite(λ) && λ > 0 ||
        throw(ArgumentError("power iteration produced a non-positive λmax: $λ"))
    return λ, iterations
end

estimate_adjoint_λmax(
    op, mesh_stokes, element_v, element_P, PC_vx, PC_vy, vx_nodes, vy_nodes,
    backend, workgroup; kwargs...,
) = estimate_adjoint_λmax(
    op, mesh_stokes, element_v, element_P, (PC_vx, PC_vy), (vx_nodes, vy_nodes),
    backend, workgroup; kwargs...,
)

# Per-component seed of the power iteration.
@inline _power_seed(i, d) = d == 1 ? sin(0.7i) : d == 2 ? cos(1.3i) : sin(1.9i)

"""
    apply_adjoint_operator!(dv, dP, op, λv, λP, mesh_stokes,
                            element_v, element_P, backend, workgroup)
    apply_adjoint_operator!(dvx, dvy, dP, op, λvx, λvy, λP, mesh_stokes,
                            element_v, element_P, backend, workgroup)

Apply the transposed adjoint operator, overwriting the velocity components `dv`
(one array per direction) and `dP` with `Aᵀλv + CᵀλP` and `Baugᵀλv + DᵀλP`.

No rheology is evaluated and no primal residual is recomputed: this is a gather,
a handful of dense element products, and a scatter.
"""
function apply_adjoint_operator!(
        dv::NTuple{Dim}, dP, op::FrozenAdjointOperator, λv::NTuple{Dim}, λP, mesh_stokes,
        element_v::ReferenceElement{TV},
        element_P::ReferenceElement{TP},
        backend, workgroup,
    ) where {TV <: AbstractElement{Dim, NV}, TP <: AbstractElement{Dim, NP}} where {Dim, NV, NP}
    foreach(a -> fill!(a, 0), dv)
    fill!(dP, 0)
    adjoint_operator_apply_kernel!(backend, workgroup)(
        dv, dP, op.A, op.B, op.C, op.D, op.W, λv, λP,
        mesh_stokes.el2n, mesh_stokes.DoFsP, Val(NV), Val(NP);
        ndrange = mesh_stokes.nels,
    )
    KA.synchronize(backend)
    return nothing
end

apply_adjoint_operator!(
    dvx::AbstractVector, dvy::AbstractVector, dP, op::FrozenAdjointOperator,
    λvx::AbstractVector, λvy::AbstractVector, λP, args...,
) = apply_adjoint_operator!((dvx, dvy), dP, op, (λvx, λvy), λP, args...)

"""
    MatrixFreeAdjointOperator(state)

The transposed Stokes adjoint operator carried as the forward state it
linearises about, applied by directional differentiation rather than from stored
element blocks.

`state` is a `NamedTuple` holding the fields, geometry, material properties and
reference tables that the element residuals read. Nothing is stored per element,
so the operator's own footprint does not grow with the mesh. Every apply re-reads
those arrays, so mutating them changes the operator.

The apply evaluates the forward-mode products `A·λv`, `B·λP`, `C·λv` and `D·w`
where the adjoint residual calls for `Aᵀλv`, `CᵀλP`, `Bᵀλv` and `Dᵀw`. The two
agree exactly when `A == Aᵀ`, `C == Bᵀ` and `D == Dᵀ`, which a purely viscous
incompressible element tangent satisfies and a plastic or pressure-dependent one
does not; see [`FrozenAdjointOperator`](@ref) for what each block contains and how
far a non-associated flow rule moves each identity. `D` is a mass matrix weighted
by `1/(ηb Δt)`, so its symmetry is structural.

Cost per element per apply is two pressure residuals and two momentum residuals
carrying a single dual partial, against four dense element products for the
frozen blocks and three reverse sweeps for the Enzyme path.
"""
struct MatrixFreeAdjointOperator{TS}
    state::TS
end

# One fixed tag suffices: the seeded residual evaluations never nest, because
# the plastic return map is the only inner differentiation the momentum residual
# performs and a plastic model cannot reach this path.
struct AdjointDirectionalTag end

"""
    _seed_partials(x, ẋ) -> SVector{N, <:ForwardDiff.Dual}

Attach `ẋ` to `x` as a single dual partial, so that a residual evaluated at the
result carries its directional derivative along `ẋ`.
"""
@inline _seed_partials(x::SVector{N}, ẋ::SVector{N}) where {N} =
    SVector{N}(ntuple(i -> ForwardDiff.Dual{AdjointDirectionalTag}(x[i], ẋ[i]), Val(N)))

"""
    _partials(x) -> SVector{N}

Strip the directional derivative out of a seeded residual, discarding its value.
"""
@inline _partials(x::SVector{N}) where {N} =
    SVector{N}(ntuple(i -> ForwardDiff.partials(x[i], 1), Val(N)))

"""
    element_adjoint_matrix_free_apply(...)
        -> (local_nodes_v, local_nodes_P, resvx, resvy, resp)

Apply element `iel` of the adjoint operator to `λ` without forming any block,
returning `A·λv + B·λP` split by velocity component and `C·λv + D·(Γ·C·λv + λP)`.

Three seeded evaluations produce all four products. The velocity seed runs
through the pressure residual first: its directional derivative is `C·λv`, and
scaling it into `Pnum` and passing that on to the momentum residual reproduces
the velocity-to-pressure-to-velocity coupling that the augmented block `A`
carries. The pressure seed runs through the momentum residual with `Pnum` held
at a constant, matching the definition of `B`. A third seed runs the same
augmented combination back through the pressure residual with the velocity held
fixed, which is the storage block `D` applied to it; see
[`FrozenAdjointOperator`](@ref) for why the pressure row carries that term.
"""
@inline function element_adjoint_matrix_free_apply(
        λvx, λvy, λP, vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, τ_old, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, ∂N∂ξ_v, iel, ::Val{NV}, ::Val{NP},
    ) where {NV, NP}
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP, iel, Val(NP))
    geo_v_el = element_geometry(geo_v, iel, ∂N∂ξ_v)
    geo_P_el = geo_P[iel]
    vxloc = _gather_local(vx, local_nodes_v, Val(NV))
    vyloc = _gather_local(vy, local_nodes_v, Val(NV))
    P_loc = _gather_local(P, local_nodes_P, Val(NP))
    P0loc = _gather_local(P0, local_nodes_P, Val(NP))
    T_loc = _gather_local(T, local_nodes_P, Val(NP))
    T0loc = _gather_local(T0, local_nodes_P, Val(NP))
    MP_loc = _gather_local(MP, local_nodes_P, Val(NP))
    γ_eff_loc = _gather_or_scalar(γ_eff, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, iel, Val(NV), quadrature_points_val(geo_v_el))
    phase_v = _gather_phase(phases_v, local_nodes_v, iel, Val(NV))
    phase_P = _gather_phase(phases_P, local_nodes_P, iel, Val(NP))
    η_pc = _element_max_phase_property(η, phase_v)

    λvxloc = _gather_local(λvx, local_nodes_v, Val(NV))
    λvyloc = _gather_local(λvy, local_nodes_v, Val(NV))
    λP_loc = _gather_local(λP, local_nodes_P, Val(NP))

    vx_dual = _seed_partials(vxloc, λvxloc)
    vy_dual = _seed_partials(vyloc, λvyloc)
    RP_dual = integrate_PH_pressure_residual(
        (vx_dual, vy_dual), P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP;
        K = nothing,
    )
    Pnum_dual = pressure_scale(γ_eff_loc, RP_dual, MP_loc)
    Avx_dual, Avy_dual = integrate_momentum_residual(
        (vx_dual, vy_dual), P_loc, Pnum_dual, T_loc,
        geo_v_el, phase_v, η_pc, G, α, ρ0, K, g, Tref, Δt, Nq, NqP,
        τ_old_loc, nothing,
    )

    # Pnum enters as a constant, so the pressure seed sees ∂Rv/∂P alone and the
    # augmentation stays where it belongs, in the velocity product above.
    P_dual = _seed_partials(P_loc, λP_loc)
    Bvx_dual, Bvy_dual = integrate_momentum_residual(
        (vxloc, vyloc), P_dual, zero(P_loc), T_loc,
        geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP,
        τ_old_loc, nothing,
    )

    resvx = _partials(Avx_dual) + _partials(Bvx_dual)
    resvy = _partials(Avy_dual) + _partials(Bvy_dual)

    # The storage row. `u` is C·λv, and what the augmentation feeds back through
    # the pressure is the same combination the momentum residual saw as `Pnum`.
    u = _partials(RP_dual)
    RP_storage = integrate_PH_pressure_residual(
        (vxloc, vyloc),
        _seed_partials(P_loc, pressure_scale(γ_eff_loc, u, MP_loc) + λP_loc),
        P0loc, T_loc, T0loc, geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
    )
    return local_nodes_v, local_nodes_P, resvx, resvy, u + _partials(RP_storage)
end

@kernel function matrix_free_adjoint_apply_kernel!(
        dvx, dvy, dP,
        @Const(λvx), @Const(λvy), @Const(λP),
        @Const(vx), @Const(vy),
        @Const(P), @Const(P0),
        @Const(T), @Const(T0),
        @Const(el2n_v), @Const(el2nP),
        @Const(geo_v), @Const(geo_P),
        @Const(phases_v), @Const(phases_P),
        τ_old,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        @Const(MP),
        Nq, NqP, ∂N∂ξ_v, ::Val{NV}, ::Val{NP},
    ) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, local_nodes_P, resvx, resvy, resp = element_adjoint_matrix_free_apply(
        λvx, λvy, λP, vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, τ_old, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, ∂N∂ξ_v, iel, Val(NV), Val(NP),
    )
    for (i, inod) in enumerate(local_nodes_v)
        Atomix.@atomic :monotonic dvx[inod] += resvx[i]
        Atomix.@atomic :monotonic dvy[inod] += resvy[i]
    end
    # Pressure degrees of freedom are discontinuous, hence unshared.
    _add_local!(dP, local_nodes_P, resp, Val(false))
end

"""
    apply_adjoint_operator!(dvx, dvy, dP, op::MatrixFreeAdjointOperator,
                            λvx, λvy, λP, mesh_stokes, element_v, element_P,
                            backend, workgroup)

Apply the adjoint operator by directional differentiation, overwriting `dvx`,
`dvy`, and `dP` with `Aᵀλv + CᵀλP` and `Baugᵀλv + DᵀλP`.

The rheology and the primal residuals are re-evaluated on every call, in
exchange for storing no element blocks.
"""
function apply_adjoint_operator!(
        dvx, dvy, dP, op::MatrixFreeAdjointOperator, λvx, λvy, λP, mesh_stokes,
        element_v::ReferenceElement{TV},
        element_P::ReferenceElement{TP},
        backend, workgroup,
    ) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    s = op.state
    fill!(dvx, 0)
    fill!(dvy, 0)
    fill!(dP, 0)
    matrix_free_adjoint_apply_kernel!(backend, workgroup)(
        dvx, dvy, dP, λvx, λvy, λP,
        s.vx, s.vy, s.P, s.P0, s.T, s.T0,
        mesh_stokes.el2n, mesh_stokes.DoFsP, s.geo_v, s.geo_P,
        s.phases_v, s.phases_P, s.τ_old,
        s.η, s.G, s.α, s.ρ0, s.K, s.g, s.Tref, s.ηb, s.Δt, s.γ_eff, s.MP,
        s.Nq, s.NqP, s.∂N∂ξ_v, Val(NV), Val(NP);
        ndrange = mesh_stokes.nels,
    )
    KA.synchronize(backend)
    return nothing
end

apply_adjoint_operator!(
    dv::NTuple{2}, dP, op::MatrixFreeAdjointOperator, λv::NTuple{2}, λP, args...,
) = apply_adjoint_operator!(dv[1], dv[2], dP, op, λv[1], λv[2], λP, args...)

@inline _probe_norm(a, b, c) = sqrt(dot(a, a) + dot(b, b) + dot(c, c))

"""
    _assert_matrix_free_symmetry(op, mesh_stokes, element_v, element_P,
                                 backend, workgroup)

Raise unless the element operator `M = [A B; C D]` that the matrix-free apply
differentiates is symmetric to `sqrt(eps)`.

The apply substitutes forward-mode products for transposed ones, which is exact
only under that symmetry. Two applies test it through the bilinear form
`⟨u, Mw⟩ = ⟨Mu, w⟩`. As with [`_assert_frozen_symmetry`](@ref), no caller can
trip this: `plastic === nothing` is what makes the tangent symmetric and is
required to construct the operator at all. It catches the tangent itself
changing, in which case the adjoint must fail rather than return a gradient
built from products it was never entitled to substitute.
"""
function _assert_matrix_free_symmetry(
        op, mesh_stokes, element_v, element_P, backend, workgroup,
    )
    s = op.state
    Tv = eltype(s.vx)
    u_vx, u_vy, u_P = similar(s.vx), similar(s.vy), similar(s.P)
    w_vx, w_vy, w_P = similar(s.vx), similar(s.vy), similar(s.P)
    y_vx, y_vy, y_P = similar(s.vx), similar(s.vy), similar(s.P)

    # Two deterministic probes with different frequencies on every field, so that
    # an asymmetry in any block reaches the pairing instead of cancelling.
    u_vx .= sin.(eachindex(u_vx) .* 0.7)
    u_vy .= cos.(eachindex(u_vy) .* 1.3)
    u_P .= sin.(eachindex(u_P) .* 0.31)
    w_vx .= cos.(eachindex(w_vx) .* 0.53)
    w_vy .= sin.(eachindex(w_vy) .* 1.7)
    w_P .= cos.(eachindex(w_P) .* 0.11)
    nu = _probe_norm(u_vx, u_vy, u_P)
    u_vx ./= nu
    u_vy ./= nu
    u_P ./= nu
    nw = _probe_norm(w_vx, w_vy, w_P)
    w_vx ./= nw
    w_vy ./= nw
    w_P ./= nw

    apply_adjoint_operator!(
        y_vx, y_vy, y_P, op, w_vx, w_vy, w_P,
        mesh_stokes, element_v, element_P, backend, workgroup
    )
    uMw = dot(u_vx, y_vx) + dot(u_vy, y_vy) + dot(u_P, y_P)
    scale = _probe_norm(y_vx, y_vy, y_P)
    apply_adjoint_operator!(
        y_vx, y_vy, y_P, op, u_vx, u_vy, u_P,
        mesh_stokes, element_v, element_P, backend, workgroup
    )
    Muw = dot(y_vx, w_vx) + dot(y_vy, w_vy) + dot(y_P, w_P)
    scale = max(scale, _probe_norm(y_vx, y_vy, y_P))

    iszero(scale) && error(
        "the matrix-free adjoint operator sends both symmetry probes to zero, so " *
            "the symmetry its apply depends on cannot be established"
    )
    defect = abs(uMw - Muw) / scale
    defect ≤ sqrt(eps(Tv)) && return nothing
    return error(
        "the matrix-free adjoint operator applies forward-mode products in place " *
            "of transposed ones, which assumes the element operator [A B; C D] is " *
            "symmetric, but ⟨u, Mw⟩ and ⟨Mu, w⟩ differ by a relative $defect. The " *
            "operator is constructed only for a viscous model with an incompressible " *
            "momentum residual, so either that condition is no longer what makes the " *
            "tangent symmetric, or the element operator has changed."
    )
end

"""
    matrix_free_adjoint_operator(dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
                                 phases_v, phases_P, τ_old, plastic, G, Δt, γP,
                                 backend, workgroup) -> MatrixFreeAdjointOperator

Capture the forward state held in `dr` as an adjoint operator that stores no
element blocks.

The forward state must be converged, and must stay put: the operator keeps
references to `dr`'s fields and re-reads them on every apply, so it tracks
whatever those arrays hold at the time of the apply rather than at the time of
this call.

`plastic` must be `nothing`; a plastic tangent is not symmetric and the apply has
no transpose to fall back on. The reference tables go on `backend` as arrays
because the apply kernel launches on every iteration of the adjoint solve, and a
tuple would be rebuilt into the argument pack each time.
"""
function matrix_free_adjoint_operator(
        dr, mesh_stokes, geo_v, geo_P,
        element_v::ReferenceElement{TV},
        element_P::ReferenceElement{TP},
        phases_v, phases_P, τ_old, plastic, G, Δt, γP,
        backend, workgroup,
    ) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    _symmetric_tangent(plastic, dr.K) || throw(
        ArgumentError(
            "the matrix-free adjoint operator applies the element tangent through " *
                "forward-mode products, which reproduce the transpose only for a " *
                "symmetric tangent; a plastic model gives a non-normal one, and a finite " *
                "bulk modulus puts a pressure derivative in the density that the " *
                "pressure residual has no counterpart for. Assemble the element blocks " *
                "instead."
        )
    )
    state = (;
        vx = dr.v.x, vy = dr.v.y, dr.P, dr.P0, dr.T, dr.T0,
        geo_v, geo_P, phases_v, phases_P, τ_old,
        dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, dr.ηb, Δt,
        γ_eff = γP, MP = dr.M_P,
        Nq = quadrature_table(backend, shape_function_values(element_v)),
        NqP = quadrature_table(
            backend, shape_function_values(element_P, element_v.integration_points)
        ),
        ∂N∂ξ_v = quadrature_table(backend, shape_function_gradients(element_v)),
    )
    op = MatrixFreeAdjointOperator(state)
    _assert_matrix_free_symmetry(op, mesh_stokes, element_v, element_P, backend, workgroup)
    return op
end
