dot_or_zero(a, ::Nothing) = zero(eltype(a))
dot_or_zero(a, b) = dot(a, b)

"""
    eos_density(Nv, phase_loc, α, ρ0, β, Tq, Pq, Tref)

Density at a quadrature point from the linearised equation of state
`ρ = ρ0·(1 − α·(T − Tref) + β·P)`, with the per-phase properties interpolated
from the element's nodal phase assignment. `β` is the compressibility `1/K`.
"""
@inline eos_density(Nv, phase_loc, α, ρ0, β, Tq, Pq, Tref) =
    interp2ip_phase(Nv, ρ0, phase_loc) *
    (1 - interp2ip_phase(Nv, α, phase_loc) * (Tq - Tref) +
     interp2ip_phase(Nv, β, phase_loc) * Pq)

"""
    accumulate_momentum(R, ∂N∂x, Nv, τ, Ptotal, ρq, g, dΩ)

Add one quadrature point's contribution to the element momentum residual,

    Rᵢ += ∫ (∂Nᵢ/∂x_j·(τ_ji − P δ_ji) − Nᵢ·ρg_i) dΩ

for every spatial direction `i`. `τ` is the full deviatoric stress tensor at
the point and `Ptotal` the pressure including any numerical correction.
"""
@inline function accumulate_momentum(R::NTuple{D}, ∂N∂x, Nv, τ, Ptotal, ρq, g, dΩ) where {D}
    return ntuple(Val(D)) do i
        pressure = SVector{D}(ntuple(j -> j == i ? Ptotal : zero(Ptotal), Val(D)))
        R[i] + (∂N∂x * (τ[:, i] - pressure) - Nv * (ρq * g[i])) * dΩ
    end
end
@inline pressure_scale(γ_eff::Number, RP, MP) = γ_eff * RP ./ MP
@inline pressure_scale(γ_eff::SVector, RP, MP) = γ_eff .* RP ./ MP
@inline function _max_phase_value(var, phase_loc::SVector{N}) where N
    out = var[phase_loc[1]]
    for i in 2:N
        out = max(out, var[phase_loc[i]])
    end
    return out
end
@inline _element_max_phase_property(var, phase_loc) = map(_ -> _max_phase_value(var, phase_loc), var)
@inline function _local_pressure_correction(
    v, P_loc, P0loc, T_loc, T0loc, geo_v_el, geo_P_el, phase_P, α, ηb, Δt, γ_eff, MP_loc, NqP,
)
    RP_loc = integrate_PH_pressure_residual(
        v, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, NqP,
    )
    return pressure_scale(γ_eff, RP_loc, MP_loc)
end

"""
    integrate_momentum_residual(v, P_loc, geo_v_el, phase_loc, η, G, Δt, Nq, NqP,
                                τ_old=nothing, plastic=nothing, τ_store=nothing) -> (Rv_x, Rv_y)

Integrate the element momentum residual for a 2-D incompressible Stokes flow.

`v` is an `NTuple{2}` of element velocity vectors `(vxloc, vyloc)`, each of
length `N` (number of velocity nodes). `P_loc` holds the `NP` element pressure
values. `Nq` contains velocity shape-function values at each velocity
quadrature point; `NqP` contains pressure shape-function values at those same
points, used to interpolate pressure into the velocity quadrature points.

`τ_old` supplies the viscoelastic stress history and `plastic` the yield
model; `nothing` selects the purely viscous response for either. `τ_store`,
when given, receives the deviatoric stress at each quadrature point.

Weak form per node `i`:

    Rᵢˣ = ∫ (∂Nᵢ/∂x·(τxx − P) + ∂Nᵢ/∂y·τxy) dΩ
    Rᵢʸ = ∫ (∂Nᵢ/∂y·(τyy − P) + ∂Nᵢ/∂x·τxy) dΩ
"""
@inline function integrate_momentum_residual(
    v::NTuple{2, SVector{N, T}},
    P_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, Δt,
    Nq,
    NqP,
    τ_old = nothing,
    plastic = nothing,
    τ_store = nothing,
) where {N, T, NP}
    Rv_x = zero(SVector{N, T})
    Rv_y = zero(SVector{N, T})
    for q in eachindex(geo_v_el)
        ∂N∂x, dΩ = geo_v_el[q]
        Nv = Nq[q]
        τ_old_q = old_stress_at_ip(Nv, τ_old, T, q, Val(3))
        Pq  = dot(NqP[q], P_loc)
        τxx, τyy, τxy = deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old_q, Pq, plastic)
        store_stress_at_ip!(τ_store, q, τxx, τyy, τxy)
        # x-momentum: ∫ (∂Nᵢ/∂x·(τxx−P) + ∂Nᵢ/∂y·τxy) dΩ
        Rv_x += (∂N∂x[:, 1] * (τxx - Pq) + ∂N∂x[:, 2] * τxy) * dΩ
        # y-momentum: ∫ (∂Nᵢ/∂y·(τyy−P) + ∂Nᵢ/∂x·τxy) dΩ
        Rv_y += (∂N∂x[:, 2] * (τyy - Pq) + ∂N∂x[:, 1] * τxy) * dΩ
    end
    return Rv_x, Rv_y
end

"""
    integrate_momentum_residual(v, P_loc, Pnum_loc, T_loc, geo_v_el, phase_loc,
                                η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP,
                                τ_old=nothing, plastic=nothing, τ_store=nothing) -> (Rv_x, Rv_y)

Integrate the element momentum residual including a numerical pressure correction
and a density-dependent gravitational body force.

`Pnum_loc` is an optional numerical pressure correction (`nothing` → zero).
`T_loc` holds temperature at the `NP` pressure nodes; it is interpolated to each
quadrature point via `NqP` to evaluate the linearised EOS
`ρ = ρ0·(1 − α·(T − Tref) + P/K)`. The body-force contribution is
`−∫ Nᵢ·ρg dΩ`, with `g` the 2-component gravity vector.

`τ_old`, `plastic`, and `τ_store` carry the stress history, yield model, and
quadrature-point stress output, as for the body-force-free method.
"""
@inline function integrate_momentum_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    Pnum_loc::Union{SVector{NP}, Nothing},
    T_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, α, ρ0, K,
    g,
    Tref::Real,
    Δt,
    Nq,
    NqP,
    τ_old = nothing,
    plastic = nothing,
    τ_store = nothing,
) where {N, NP}
    T = promote_type(map(eltype, v)...)
    R = ntuple(_ -> zero(SVector{N, T}), Val(2))
    # Compressibility β = 1/K: safe for K=Inf (β=0) and avoids NaN from
    # interp2ip_phase when quadratic shape functions are negative.
    β = map(inv, K)
    for q in eachindex(geo_v_el)
        ∂N∂x, dΩ = geo_v_el[q]
        Nv = Nq[q]
        τ_old_q = old_stress_at_ip(Nv, τ_old, T, q, Val(3))
        Pq = dot(NqP[q], P_loc)
        # Plane strain: τzz = −(τxx + τyy) is carried by `deviatoric_stress`
        # and never enters the in-plane residual.
        τxx, τyy, τxy = deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old_q, Pq, plastic)
        store_stress_at_ip!(τ_store, q, τxx, τyy, τxy)
        τ = SMatrix{2, 2}(τxx, τxy, τxy, τyy)
        Ptotal = Pq + dot_or_zero(NqP[q], Pnum_loc)
        Tq = dot(NqP[q], T_loc)
        ρq = eos_density(Nv, phase_loc, α, ρ0, β, Tq, Pq, Tref)
        R = accumulate_momentum(R, ∂N∂x, Nv, τ, Ptotal, ρq, g, dΩ)
    end
    return R
end

"""
    integrate_momentum_residual(v::NTuple{3}, P_loc, Pnum_loc, T_loc,
                                geo_v_el, phase_loc, η, G, α, ρ0, K,
                                g, Tref, Δt, Nq, NqP,
                                τ_old=nothing, plastic=nothing) -> NTuple{3}

Integrate the 3-D element momentum residual, including pressure and gravity.
This is the element operator used by the Hex27 velocity path.

`τ_old` supplies the six viscoelastic stress-history components
`(τxx, τyy, τzz, τxy, τxz, τyz)` and `plastic` the yield model; `nothing`
selects the purely viscous response for either. `τ_store`, when given, receives
the deviatoric stress at each quadrature point in the same component order.
"""
@inline function integrate_momentum_residual(
    v::Tuple{<:SVector{N}, <:SVector{N}, <:SVector{N}},
    P_loc::SVector{NP},
    Pnum_loc::Union{SVector{NP}, Nothing},
    T_loc::SVector{NP},
    geo_v_el,
    phase_loc, η, G, α, ρ0, K,
    g::NTuple{3},
    Tref::Real,
    Δt,
    Nq,
    NqP,
    τ_old = nothing,
    plastic = nothing,
    τ_store = nothing,
) where {N, NP}
    T = promote_type(map(eltype, v)...)
    R = ntuple(_ -> zero(SVector{N, T}), 3)
    β = map(inv, K)
    for q in eachindex(geo_v_el)
        ∂N∂x, dΩ = geo_v_el[q]
        Nv = Nq[q]
        τ_old_q = old_stress_at_ip(Nv, τ_old, T, q, Val(6))
        Pq = dot(NqP[q], P_loc)
        τxx, τyy, τzz, τxy, τxz, τyz =
            deviatoric_stress(v, ∂N∂x, Nv, η, G, phase_loc, Δt, τ_old_q, Pq, plastic)
        store_stress_at_ip!(τ_store, q, τxx, τyy, τzz, τxy, τxz, τyz)
        τ = SMatrix{3, 3, T}(τxx, τxy, τxz, τxy, τyy, τyz, τxz, τyz, τzz)
        Ptotal = Pq + dot_or_zero(NqP[q], Pnum_loc)
        Tq = dot(NqP[q], T_loc)
        ρq = eos_density(Nv, phase_loc, α, ρ0, β, Tq, Pq, Tref)
        R = accumulate_momentum(R, ∂N∂x, Nv, τ, Ptotal, ρq, g, dΩ)
    end
    return R
end

"""
    integrate_momentum_residual(v, P_loc, P0loc, T_loc, T0loc,
                                geo_v_el, geo_P_el, phase_v, phase_P,
                                η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
                                MP_loc, Nq, NqP,
                                τ_old=nothing, plastic=nothing) -> NTuple{D}

Integrate the momentum residual with the DYREL numerical pressure correction
computed directly from the local pressure residual:

    Pnum = γ_eff * RP(v) / M_P

`RP(v)` is the weak pressure residual and `M_P` is the lumped pressure mass, so
`RP/M_P` matches the pointwise finite-difference residual used by JustRelax.
One element velocity vector is supplied per spatial direction, and the result
has one residual per direction.
"""
@inline function integrate_momentum_residual(
    v::Tuple{<:SVector{N}, Vararg{<:SVector{N}}},
    P_loc::SVector{NP},
    P0loc::SVector{NP},
    T_loc::SVector{NP},
    T0loc::SVector{NP},
    geo_v_el,
    geo_P_el,
    phase_v,
    phase_P,
    η, G, α, ρ0, K,
    g,
    Tref::Real,
    ηb, Δt, γ_eff,
    MP_loc::SVector{NP},
    Nq,
    NqP,
    τ_old = nothing,
    plastic = nothing,
) where {N, NP}
    Pnum_loc = _local_pressure_correction(
        v, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_P, α, ηb, Δt, γ_eff, MP_loc, NqP,
    )
    return integrate_momentum_residual(
        v, P_loc, Pnum_loc, T_loc,
        geo_v_el, phase_v, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, τ_old, plastic,
    )
end

"""
    integrate_momentum_x_residual(args...) -> Rv_x
    integrate_momentum_y_residual(args...) -> Rv_y

Integrate a single component of the element momentum residual. The arguments
are those of any [`integrate_momentum_residual`](@ref) method; each returns
the corresponding component of that method's result.

Useful when the x and y systems are advanced independently, and when
differentiating one component with respect to one velocity field.
"""
@inline integrate_momentum_x_residual(v::Tuple, args...) =
    integrate_momentum_residual(v, args...)[1]
@inline integrate_momentum_y_residual(v::Tuple, args...) =
    integrate_momentum_residual(v, args...)[2]


"""
    assemble_momentum_residual_matrices_atomix!(Rv, v, P, T, Pnum,
                                                el2n_v, el2nP, geo_v, nels,
                                                element_v, element_P,
                                                phases, τ_old, plastic, τ_store,
                                                η, G, α, ρ0, K, g, Tref, Δt,
                                                backend, workgroup)
    assemble_momentum_residual_matrices_atomix!(Rv_x, Rv_y, vx, vy, P, T, Pnum, ...)

Assemble the Stokes momentum residuals using Atomix-backed atomic scatter.

`Rv` and `v` hold one nodal array per spatial direction; the second form takes
the two plane-strain components separately. `Pnum` is an optional nodal array of
numerical pressure corrections; pass `nothing` to omit the correction. `T` is
the temperature field on pressure nodes; it is used together with `α`, `ρ0`,
`K`, `g`, and `Tref` to compute the density body force
`ρ = ρ0·(1 − α·(T−Tref) + P/K)` and the resulting gravity term `−∫ Nᵢ·ρg dΩ`.
`phases` is a velocity-node integer array selecting the material phase.
"""
function assemble_momentum_residual_matrices_atomix!(
    Rv::NTuple{D, <:AbstractVector},
    v::NTuple{D, <:AbstractVector},
    P, T,
    Pnum,
    el2n_v, el2nP,
    geo_v,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    τ_old,
    plastic,
    τ_store,
    η, G, α, ρ0, K,
    g, Tref, Δt,
    backend, workgroup,
) where {D, TV <: AbstractElement{D, NV}, TP <: AbstractElement{D, NP}} where {NV, NP}
    Nq  = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)

    return assemble_momentum_residual_kernel!(
        Rv, v, P, T, Pnum,
        el2n_v, el2nP, geo_v, nels, phases,
        τ_old, plastic, τ_store, η, G, α, ρ0, K,
        g, Tref, Δt, Nq, NqP, Val(NV), Val(NP), workgroup,
    )
end

assemble_momentum_residual_matrices_atomix!(
        Rv_x::AbstractVector, Rv_y::AbstractVector,
        vx::AbstractVector, vy::AbstractVector, args...) =
    assemble_momentum_residual_matrices_atomix!((Rv_x, Rv_y), (vx, vy), args...)

"""
    assemble_momentum_residual_kernel!(Rv, v, P, T, Pnum,
                                       el2n_v, el2nP, geo_v, nels, phases,
                                       τ_old, plastic, τ_store,
                                       η, G, α, ρ0, K, g, Tref, Δt,
                                       Nq, NqP, Val(NV), Val(NP), workgroup)
    assemble_momentum_residual_kernel!(Rv_x, Rv_y, vx, vy, P, T, Pnum, ...)

Zero the momentum residuals, launch the atomic momentum-residual kernel over
`nels` elements, and synchronize.

Low-level entry point beneath `assemble_momentum_residual_matrices_atomix!`:
the shape-function tables `Nq`, `NqP` and the local node counts `Val(NV)`,
`Val(NP)` are passed explicitly instead of `ReferenceElement`s, which makes
the call differentiable with Enzyme (see
`assemble_momentum_residual_matrices_atomix_adj!`). The backend is inferred
from the first residual array.
"""
function assemble_momentum_residual_kernel!(
    Rv::NTuple{D, <:AbstractVector},
    v::NTuple{D, <:AbstractVector}, P, T, Pnum,
    el2n_v, el2nP, geo_v, nels, phases,
    τ_old, plastic, τ_store,
    η, G, α, ρ0, K, g, Tref, Δt,
    Nq, NqP, ::Val{NV}, ::Val{NP}, workgroup,
) where {D, NV, NP}
    foreach(R -> fill!(R, 0), Rv)
    backend = KA.get_backend(first(Rv))
    momentum_residual_atomic_kernel!(backend, workgroup)(
        Rv, v, P, T, Pnum, el2n_v, el2nP, geo_v, phases, τ_old, plastic,
        τ_store, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, Val(NV), Val(NP);
        ndrange = nels,
    )
    KA.synchronize(backend)
    return nothing
end

assemble_momentum_residual_kernel!(
        Rv_x::AbstractVector, Rv_y::AbstractVector,
        vx::AbstractVector, vy::AbstractVector, args...) =
    assemble_momentum_residual_kernel!((Rv_x, Rv_y), (vx, vy), args...)

@kernel function momentum_residual_atomic_kernel!(
    Rv,
    @Const(v),
    @Const(P), @Const(T),
    @Const(Pnum),
    @Const(el2n_v), @Const(el2nP),
    @Const(geo_v),
    @Const(phases),
    @Const(τ_old),
    @Const(plastic),
    τ_store,
    @Const(η), @Const(G), @Const(α), @Const(ρ0), @Const(K),
    @Const(g), @Const(Tref), @Const(Δt),
    @Const(Nq), @Const(NqP), ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, Re = momentum_element_residual(
        v, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
        η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP),
        τ_old, plastic, τ_store,
    )
    _scatter_momentum!(Rv, local_nodes_v, Re)
end

# Walk the direction tuples with `ntuple` so the component index stays a
# compile-time constant and tuple indexing does not fall back to a dynamic path.
@inline function _scatter_momentum!(Rv::NTuple{D}, nodes, Re::NTuple{D}) where {D}
    for (i, inod) in enumerate(nodes)
        ntuple(Val(D)) do c
            Atomix.@atomic :monotonic Rv[c][inod] += Re[c][i]
            nothing
        end
    end
    return nothing
end

"""
    momentum_element_residual(v, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
                              η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel,
                              Val(NV), Val(NP),
                              τ_old=nothing, plastic=nothing, τ_store=nothing)

Gather element-local nodal values and integrate the Stokes momentum residual
for element `iel`.

`v` holds one nodal velocity array per spatial direction. Returns
`(local_nodes_v, Re)` with one element residual per direction, ready for global
scatter.
"""
@inline function momentum_element_residual(
    v::NTuple{D}, P, T, Pnum, el2n_v, el2nP, geo_v, phases,
    η, G, α, ρ0, K, g, Tref, Δt,
    Nq, NqP, iel, ::Val{NV}, ::Val{NP},
    τ_old = nothing,
    plastic = nothing,
    τ_store = nothing,
) where {D, NV, NP}
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP,  iel, Val(NP))
    geo_v_el  = geo_v[iel]
    vloc      = ntuple(i -> _gather_local(v[i], local_nodes_v, Val(NV)), Val(D))
    P_loc     = _gather_local(P, local_nodes_P, Val(NP))
    T_loc     = _gather_local(T, local_nodes_P, Val(NP))
    Pnum_loc  = _gather_or_nothing(Pnum, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, iel, Val(NV), Val(length(Nq)))
    phase_loc = _gather_phase(phases, local_nodes_v, iel, Val(NV))
    Re = integrate_momentum_residual(
        vloc, P_loc, Pnum_loc, T_loc, geo_v_el, phase_loc,
        η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, τ_old_loc, plastic,
        _stress_output(τ_store, iel),
    )
    return local_nodes_v, Re
end

_gather_or_nothing(::Nothing, _, ::Val) = nothing
@inline function _gather_or_nothing(arr, nodes, ::Val{N}) where N
    _gather_local(arr, nodes, Val(N))
end

_gather_old_stress(::Nothing, _, _, ::Val, ::Val) = nothing
@inline _gather_old_stress(
        τ_old::Tuple{AbstractMatrix, Vararg{AbstractMatrix}}, _, iel, ::Val, ::Val{NQ},
    ) where {NQ} =
    IntegrationPointStress(map(τ -> SVector{NQ}(ntuple(q -> τ[q, iel], Val(NQ))), τ_old))
@inline _gather_old_stress(
        τ_old::Tuple{AbstractVector, Vararg{AbstractVector}}, nodes, _, ::Val{N}, ::Val,
    ) where {N} =
    map(τ -> _gather_local(τ, nodes, Val(N)), τ_old)
_stress_output(::Nothing, _) = nothing
@inline _stress_output(τ_store::NTuple{Nτ, <:AbstractMatrix}, iel) where {Nτ} =
    IntegrationPointStressOutput(τ_store, Int(iel))

"""
    assemble_stokes_momentum_residual_3d!(R, v, P, mesh, cell_phase, η, ρ, g;
                                          workgroup=256)

Assemble the purely viscous 3-D momentum residual for continuous Hex27
velocity and four cell-local pressure modes `(1, ξ, η, ζ)`.
"""
function assemble_stokes_momentum_residual_3d!(
    R::NTuple{3}, v::NTuple{3}, P::AbstractMatrix, mesh::Mesh,
    cell_phase, η, ρ, g::NTuple{3}; workgroup = 256,
)
    size(P) == (4, mesh.nels) || throw(DimensionMismatch("P must be 4 × nels"))
    all(length(r) == mesh.nnodes for r in R) || throw(DimensionMismatch("residual size must match mesh nodes"))
    all(length(u) == mesh.nnodes for u in v) || throw(DimensionMismatch("velocity size must match mesh nodes"))
    length(cell_phase) == mesh.nels || throw(DimensionMismatch("cell_phase size must match mesh elements"))
    Nq = shape_function_values(mesh.element)
    ip = mesh.element.integration_points
    NqP = ntuple(q -> SVector(1.0, ip.ξ[q], ip.η[q], ip.ζ[q]), length(ip.ω))
    foreach(r -> fill!(r, 0), R)
    backend = KA.get_backend(first(R))
    stokes_momentum_residual_3d_kernel!(backend, workgroup)(
        R, v, P, mesh.el2n, mesh.geometry, cell_phase, η, ρ, g, Nq, NqP;
        ndrange = mesh.nels,
    )
    KA.synchronize(backend)
    return nothing
end

@kernel function stokes_momentum_residual_3d_kernel!(
    R, @Const(v), @Const(P), @Const(el2n), @Const(geometry), @Const(cell_phase),
    @Const(η), @Const(ρ), @Const(g), @Const(Nq), @Const(NqP),
)
    cell = @index(Global)
    nodes = local_nodes_of(el2n, cell, Val(27))
    velocity = ntuple(i -> _gather_local(v[i], nodes, Val(27)), 3)
    pressure = SVector{4}(ntuple(i -> P[i, cell], Val(4)))
    phase = Int(cell_phase[cell])
    phase_loc = SVector{27}(ntuple(_ -> phase, Val(27)))
    residual = integrate_momentum_residual(
        velocity, pressure, nothing, zero(pressure), geometry[cell], phase_loc,
        η, map(x -> oftype(x, Inf), η), map(zero, η), ρ,
        map(x -> oftype(x, Inf), η), g,
        zero(eltype(pressure)), one(eltype(pressure)), Nq, NqP,
    )
    for (i, node) in enumerate(nodes), component in 1:3
        Atomix.@atomic :monotonic R[component][node] += residual[component][i]
    end
end

function stokes_preconditioner_3d(mesh::Mesh, cell_phase, η; workgroup = 256)
    diagonal = ntuple(_ -> similar(mesh.coords, eltype(eltype(mesh.coords)), mesh.nnodes), 3)
    pressure_mass = similar(first(diagonal), 4, mesh.nels)
    foreach(x -> fill!(x, 0), diagonal)
    fill!(pressure_mass, 0)
    ip = mesh.element.integration_points
    modes = ntuple(q -> SVector(1.0, ip.ξ[q], ip.η[q], ip.ζ[q]), length(ip.ω))
    backend = KA.get_backend(first(diagonal))
    stokes_preconditioner_3d_kernel!(backend, workgroup)(
        diagonal, pressure_mass, mesh.el2n, mesh.geometry, cell_phase, η, modes;
        ndrange = mesh.nels,
    )
    KA.synchronize(backend)
    return diagonal, pressure_mass
end

@kernel function stokes_preconditioner_3d_kernel!(
    diagonal, pressure_mass, @Const(el2n), @Const(geometry), @Const(cell_phase),
    @Const(η), @Const(modes),
)
    cell = @index(Global)
    phase = Int(cell_phase[cell])
    for q in eachindex(geometry[cell])
        gradient, dΩ = geometry[cell][q]
        for a in 1:27, component in 1:3
            value = η[phase] * (dot(gradient[a, :], gradient[a, :]) +
                    gradient[a, component]^2 / 3) * dΩ
            Atomix.@atomic :monotonic diagonal[component][el2n[a, cell]] += value
        end
        for mode in 1:4
            pressure_mass[mode, cell] += modes[q][mode]^2 * dΩ
        end
    end
end

_gather_or_scalar(x::Number, _, ::Val) = x
@inline function _gather_or_scalar(arr, nodes, ::Val{N}) where N
    _gather_local(arr, nodes, Val(N))
end

"""
    _velocity_jacobian_blocks(momentum, vloc::NTuple{D}) -> NTuple{D, NTuple{D}}

Differentiate an element momentum residual with respect to every velocity
component. `momentum(v)` returns one residual per direction for the trial
velocity tuple `v`, so block `J[i][j]` is `∂Rᵢ/∂vⱼ` and the off-diagonal
blocks carry the shear coupling between directions.
"""
@inline function _velocity_jacobian_blocks(momentum, vloc::NTuple{D}) where {D}
    dirs = _direction_tags(Val(D))
    return map(vi -> map(vj -> _velocity_jacobian_block(momentum, vloc, vi, vj), dirs), dirs)
end

# The block indices travel as `Val` singletons rather than as integers: they end
# up captured by the closure `ForwardDiff.jacobian` differentiates, and only a
# type carries a value across that capture. An integer index would have to be
# constant-propagated into `momentum`, which is far too large for inference to
# do so; the index then reaches `_substitute_component` non-constant, the trial
# tuple comes back `Union`-typed, and the enclosing kernel loses inferability
# and cannot be compiled for GPU back-ends.
@inline _direction_tags(::Val{D}) where {D} = ntuple(i -> Val(i), Val(D))

@inline function _velocity_jacobian_block(
        momentum, vloc::NTuple{D}, vi::Val, vj::Val{J},
    ) where {D, J}
    return ForwardDiff.jacobian(
        u -> _component(momentum(_substitute_component(vloc, u, vj)), vi), vloc[J],
    )
end

@inline _component(R::Tuple, ::Val{I}) where {I} = R[I]

@inline _substitute_component(vloc::NTuple{D}, u, ::Val{J}) where {D, J} =
    ntuple(k -> k === J ? u : vloc[k], Val(D))

"""
    element_momentum_jacobians(vx, vy, P, T, el2n_v, el2nP, geo, phases,
                               η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel,
                               Val(NV), Val(NP), τ_old=nothing, plastic=nothing)

Compute per-element Jacobian diagnostics for the Stokes momentum residuals via
ForwardDiff, differentiating `Rv_x` w.r.t. `vx` and `Rv_y` w.r.t. `vy`.

Returns `(local_nodes_v, rowsums_x, diags_x, rowsums_y, diags_y)` where
`rowsums_*[i]` is the coupled L1 row sum, including the `vx↔vy` shear blocks,
and `diags_*[i]` is the absolute diagonal of the same-component block. The
row sums provide a conservative smoother/preconditioner and spectral estimate.
"""
@inline function element_momentum_jacobians(
    vx, vy, P, T, el2n_v, el2nP, geo, phases,
    η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, ::Val{NV}, ::Val{NP},
    τ_old = nothing,
    plastic = nothing,
) where {NV, NP}
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP,  iel, Val(NP))
    geo_el    = geo[iel]
    vxloc     = _gather_local(vx, local_nodes_v, Val(NV))
    vyloc     = _gather_local(vy, local_nodes_v, Val(NV))
    P_loc     = _gather_local(P, local_nodes_P, Val(NP))
    T_loc     = _gather_local(T, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, iel, Val(NV), Val(length(Nq)))
    phase_loc = _gather_phase(phases, local_nodes_v, iel, Val(NV))
    η_pc      = _element_max_phase_property(η, phase_loc)

    J = _velocity_jacobian_blocks(
        v_arg -> integrate_momentum_residual(
            v_arg, P_loc, nothing, T_loc,
            geo_el, phase_loc, η_pc, G, α, ρ0, K, g, Tref, Δt, Nq, NqP,
            τ_old_loc, plastic,
        ),
        (vxloc, vyloc),
    )
    rowsums_x, diags_x = jacobian_rowsums_and_diagonal(J[1][1], J[1][2])
    rowsums_y, diags_y = jacobian_rowsums_and_diagonal(J[2][2], J[2][1])

    return local_nodes_v, rowsums_x, diags_x, rowsums_y, diags_y
end

"""
    assemble_momentum_jacobian_matrices_atomix!(∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
                                                vx, vy, P, T, el2n_v, el2nP, geo_v, nels,
                                                element_v, element_P,
                                                phases, η, G, α, ρ0, K, g, Tref, Δt,
                                                backend, workgroup)

Assemble the Stokes momentum Jacobian diagnostics using Atomix-backed atomic scatter.

Fills the row-sum Jacobian estimates `∂Rv_x∂vx`, `∂Rv_y∂vy` and row-sum
preconditioners `PC_vx`, `PC_vy` by differentiating the momentum residuals with
ForwardDiff via `element_momentum_jacobians`. The row sums include the
same-component and shear-coupling blocks, which is safer for the mixed T7/P1
Stokes smoother than using only the absolute diagonal.
"""
function assemble_momentum_jacobian_matrices_atomix!(
    ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
    vx, vy,
    P, T,
    el2n_v, el2nP,
    geo_v,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases,
    η, G, α, ρ0, K,
    g, Tref, Δt,
    backend, workgroup;
    τ_old = nothing,
    plastic = nothing,
) where {TV <: AbstractElement{2, NV}, TP <: AbstractElement{2, NP}} where {NV, NP}
    Nq  = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)

    fill!(∂Rv_x∂vx, 0)
    fill!(PC_vx,    0)
    fill!(∂Rv_y∂vy, 0)
    fill!(PC_vy,    0)
    momentum_jacobian_atomic_kernel!(backend, workgroup)(
        ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        vx, vy, P, T, el2n_v, el2nP, geo_v, phases,
        τ_old, plastic, η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, Val(NV), Val(NP);
        ndrange = nels,
    )
    KA.synchronize(backend)
end

"""
    element_augmented_momentum_jacobians(vx, vy, P, P0, T, T0,
                                          el2n_v, el2nP, geo_v, geo_P,
                                          phases_v, phases_P,
                                          η, G, α, ρ0, K, g, Tref,
                                          ηb, Δt, γ_eff, MP, Nq, NqP,
                                          iel, Val(NV), Val(NP),
                                          τ_old=nothing, plastic=nothing)

Compute per-element Jacobian diagnostics for the augmented Stokes momentum
residuals via ForwardDiff, accounting for the DYREL/Arrow-Hurwicz numerical
pressure correction computed inline.

The pressure correction `Pnum = γ_eff * RP(v) / M_P` is re-evaluated inside
each ForwardDiff call from the current velocity, capturing the
velocity-to-pressure-to-velocity coupling introduced by the Arrow-Hurwicz
scheme. This makes the preconditioner more effective than
`element_momentum_jacobians` for problems where that coupling is significant.

Returns `(local_nodes_v, J)`, where `J[i][j]` is the `NV`×`NV` block
`∂Rᵢ/∂vⱼ` of the augmented element Jacobian. Because `Pnum` is formed inline
from element-local pressures, the blocks already carry the Powell-Hestenes
augmentation `Bnum·(γ_eff/MP)·C`; for a discontinuous pressure space that
coupling is element-local, so the blocks are exact rather than an
approximation.

The plane-strain form returns the four blocks positionally as
`(local_nodes_v, ∂RVx∂vx, ∂RVx∂vy, ∂RVy∂vx, ∂RVy∂vy)`.

`jacobian_rowsums_and_diagonal` reduces them to the preconditioner diagnostics.
"""
@inline function element_augmented_momentum_jacobians(
        v::NTuple{D}, args...,
    ) where {D}
    local_nodes_v, momentum, vloc = _element_augmented_momentum_problem(v, args...)
    return local_nodes_v, _velocity_jacobian_blocks(momentum, vloc)
end

# Gather the element-local state of the augmented momentum residual and return
# it as the residual closure the Jacobian differentiates, alongside the trial
# velocity it is differentiated at.
@inline function _element_augmented_momentum_problem(
    v::NTuple{D}, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
    phases_v, phases_P, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
    MP, Nq, NqP, iel, ::Val{NV}, ::Val{NP},
    τ_old = nothing,
    plastic = nothing,
) where {D, NV, NP}
    local_nodes_v = local_nodes_of(el2n_v, iel, Val(NV))
    local_nodes_P = local_nodes_of(el2nP,  iel, Val(NP))
    geo_v_el  = geo_v[iel]
    geo_P_el  = geo_P[iel]
    vloc      = ntuple(i -> _gather_local(v[i], local_nodes_v, Val(NV)), Val(D))
    P_loc     = _gather_local(P, local_nodes_P, Val(NP))
    P0loc     = _gather_local(P0, local_nodes_P, Val(NP))
    T_loc     = _gather_local(T, local_nodes_P, Val(NP))
    T0loc     = _gather_local(T0, local_nodes_P, Val(NP))
    MP_loc    = _gather_local(MP, local_nodes_P, Val(NP))
    γ_eff_loc = _gather_or_scalar(γ_eff, local_nodes_P, Val(NP))
    τ_old_loc = _gather_old_stress(τ_old, local_nodes_v, iel, Val(NV), Val(length(Nq)))
    phase_v   = _gather_phase(phases_v, local_nodes_v, iel, Val(NV))
    phase_P   = _gather_phase(phases_P, local_nodes_P, iel, Val(NP))
    η_pc      = _element_max_phase_property(η, phase_v)

    momentum = v_arg -> integrate_momentum_residual(
        v_arg, P_loc, P0loc, T_loc, T0loc,
        geo_v_el, geo_P_el, phase_v, phase_P,
        η_pc, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff_loc, MP_loc, Nq, NqP,
        τ_old_loc, plastic,
    )
    return local_nodes_v, momentum, vloc
end

@inline function element_augmented_momentum_jacobians(
        vx::AbstractVector, vy::AbstractVector, args...)
    local_nodes_v, J = element_augmented_momentum_jacobians((vx, vy), args...)
    return local_nodes_v, J[1][1], J[1][2], J[2][1], J[2][2]
end

"""
    assemble_augmented_momentum_jacobian_matrices_atomix!(
        ∂Rv_x∂vx, PC_vx, ∂Rv_y∂vy, PC_vy,
        vx, vy, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P, nels,
        element_v, element_P, phases_v, phases_P,
        η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP,
        backend, workgroup)

Assemble the augmented Stokes momentum Jacobian diagnostics using
Atomix-backed atomic scatter.

Compared to `assemble_momentum_jacobian_matrices_atomix!`, this assembler
includes the DYREL/Arrow-Hurwicz numerical pressure correction in the
Jacobian by computing `Pnum = γ_eff * RP(v) / M_P` inline inside each
ForwardDiff call via `element_augmented_momentum_jacobians`. This captures
the velocity–pressure coupling and produces a more effective preconditioner
for incompressible Stokes flows. `P0`, `T0`, `ηb`, `γ_eff`, and `MP` are
the additional arguments relative to the non-augmented assembler.
"""
function assemble_augmented_momentum_jacobian_matrices_atomix!(
    ∂Rv∂v::NTuple{D, <:AbstractVector},
    PC::NTuple{D, <:AbstractVector},
    v::NTuple{D, <:AbstractVector},
    P, P0,
    T, T0,
    el2n_v, el2nP,
    geo_v, geo_P,
    nels,
    element_v::ReferenceElement{TV},
    element_P::ReferenceElement{TP},
    phases_v, phases_P,
    η, G, α, ρ0, K,
    g, Tref,
    ηb, Δt, γ_eff, MP,
    backend, workgroup;
    τ_old = nothing,
    plastic = nothing,
) where {D, TV <: AbstractElement{D, NV}, TP <: AbstractElement{D, NP}} where {NV, NP}
    Nq  = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)

    foreach(A -> fill!(A, 0), ∂Rv∂v)
    foreach(A -> fill!(A, 0), PC)
    augmented_momentum_jacobian_atomic_kernel!(backend, workgroup)(
        ∂Rv∂v, PC,
        v, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P, phases_v, phases_P,
        τ_old, plastic, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff, MP, Nq, NqP, Val(NV), Val(NP);
        ndrange = nels,
    )
    KA.synchronize(backend)
end

assemble_augmented_momentum_jacobian_matrices_atomix!(
        ∂Rv_x∂vx::AbstractVector, PC_vx::AbstractVector,
        ∂Rv_y∂vy::AbstractVector, PC_vy::AbstractVector,
        vx::AbstractVector, vy::AbstractVector, args...; kwargs...) =
    assemble_augmented_momentum_jacobian_matrices_atomix!(
        (∂Rv_x∂vx, ∂Rv_y∂vy), (PC_vx, PC_vy), (vx, vy), args...; kwargs...)

@kernel function augmented_momentum_jacobian_atomic_kernel!(
    ∂Rv∂v, PC,
    @Const(v),
    @Const(P), @Const(P0),
    @Const(T), @Const(T0),
    @Const(el2n_v), @Const(el2nP),
    @Const(geo_v), @Const(geo_P),
    @Const(phases_v), @Const(phases_P),
    τ_old,
    plastic,
    η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
    @Const(MP),
    Nq, NqP, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, momentum, vloc = _element_augmented_momentum_problem(
        v, P, P0, T, T0, el2n_v, el2nP, geo_v, geo_P,
        phases_v, phases_P, η, G, α, ρ0, K, g, Tref, ηb, Δt, γ_eff,
        MP, Nq, NqP, iel, Val(NV), Val(NP), τ_old, plastic,
    )
    _scatter_folded_jacobian_diagnostics!(∂Rv∂v, PC, local_nodes_v, momentum, vloc)
end

# Reduce the element Jacobian to the preconditioner diagnostics one block at a
# time. The diagnostics need only an L1 row sum and a diagonal, so no block has
# to outlive its own contribution, and keeping just one live matters: a live
# block holds several KiB of ForwardDiff frame, and all `D²` at once reserve
# more thread-local memory than a GPU launch can back.
@inline function _scatter_folded_jacobian_diagnostics!(
        ∂Rv∂v::NTuple{D}, PC::NTuple{D}, nodes, momentum, vloc::NTuple{D},
    ) where {D}
    dirs = _direction_tags(Val(D))
    map(dirs) do vi
        rowsum, diagonal = _jacobian_row_diagnostics(momentum, vloc, vi, dirs)
        _scatter_row_diagnostics!(∂Rv∂v, PC, nodes, rowsum, diagonal, vi)
    end
    return nothing
end

@inline function _jacobian_row_diagnostics(momentum, vloc, vi, dirs)
    block = _velocity_jacobian_block(momentum, vloc, vi, first(dirs))
    diagonal = _abs_diagonal(block)
    seed = (_abs_row_sums(block), _same_direction(vi, first(dirs)) ? diagonal : zero(diagonal))
    return _fold_jacobian_row(momentum, vloc, vi, Base.tail(dirs), seed)
end

@inline _fold_jacobian_row(momentum, vloc, vi, ::Tuple{}, acc) = acc
@inline function _fold_jacobian_row(momentum, vloc, vi, dirs::Tuple, acc)
    vj = first(dirs)
    block = _velocity_jacobian_block(momentum, vloc, vi, vj)
    rowsum, diagonal = acc
    folded = (
        rowsum + _abs_row_sums(block),
        _same_direction(vi, vj) ? _abs_diagonal(block) : diagonal,
    )
    return _fold_jacobian_row(momentum, vloc, vi, Base.tail(dirs), folded)
end

@inline _same_direction(::Val{I}, ::Val{J}) where {I, J} = I === J
@inline _abs_row_sums(block::SMatrix{M, N}) where {M, N} =
    SVector{M}(ntuple(i -> sum(ntuple(j -> abs(block[i, j]), Val(N))), Val(M)))
@inline _abs_diagonal(block::SMatrix{M}) where {M} =
    SVector{M}(ntuple(i -> abs(block[i, i]), Val(M)))

@inline function _scatter_row_diagnostics!(
        ∂Rv∂v, PC, nodes, rowsum, diagonal, ::Val{C},
    ) where {C}
    for (i, inod) in enumerate(nodes)
        Atomix.@atomic :monotonic ∂Rv∂v[C][inod] += rowsum[i]
        Atomix.@atomic :monotonic PC[C][inod]    += diagonal[i]
    end
    return nothing
end

# Scatter the preconditioner diagnostics of an element Jacobian. For direction
# `c` the row sum spans every coupling block of that row, giving the
# conservative Gershgorin bound, while the diagonal comes from the
# same-direction block alone.
@inline function _scatter_jacobian_diagnostics!(∂Rv∂v::NTuple{D}, PC::NTuple{D}, nodes, J) where {D}
    ntuple(Val(D)) do c
        row  = J[c]
        same = row[c]
        for (i, inod) in enumerate(nodes)
            rowsum = sum(B -> sum(j -> abs(B[i, j]), axes(B, 2)), row)
            Atomix.@atomic :monotonic ∂Rv∂v[c][inod] += rowsum
            Atomix.@atomic :monotonic PC[c][inod]    += abs(same[i, i])
        end
        nothing
    end
    return nothing
end

@kernel function momentum_jacobian_atomic_kernel!(
    ∂Rv_x∂vx, PC_vx,
    ∂Rv_y∂vy, PC_vy,
    @Const(vx), @Const(vy),
    @Const(P), @Const(T),
    @Const(el2n_v), @Const(el2nP),
    @Const(geo_v),
    @Const(phases),
    τ_old,
    plastic,
    η, G, α, ρ0, K, g, Tref, Δt,
    Nq, NqP, ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    local_nodes_v, rowsums_x, diags_x, rowsums_y, diags_y = element_momentum_jacobians(
        vx, vy, P, T, el2n_v, el2nP, geo_v, phases,
        η, G, α, ρ0, K, g, Tref, Δt, Nq, NqP, iel, Val(NV), Val(NP), τ_old, plastic,
    )
    for (i, inod) in enumerate(local_nodes_v)
        Atomix.@atomic :monotonic ∂Rv_x∂vx[inod] += rowsums_x[i]
        Atomix.@atomic :monotonic PC_vx[inod]    += diags_x[i]
        Atomix.@atomic :monotonic ∂Rv_y∂vy[inod] += rowsums_y[i]
        Atomix.@atomic :monotonic PC_vy[inod]    += diags_y[i]
    end
end
