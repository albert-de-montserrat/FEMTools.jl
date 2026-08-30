import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using Enzyme
using Atomix
using KernelAbstractions
using Gmsh
using FEMTools
using GLMakie: Figure, Axis, Colorbar, poly!, lines!, Point2f, DataAspect, axislegend

const backend   = CPU()
const workgroup = 128

include(joinpath(@__DIR__, "..", "..", "gmsh_meshing.jl"))

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

"""
    objective_vy_kernel!(dJdvy, coords, el2n, geo, Nq,
                         xmin, xmax, ymin, ymax, Val(N))

Assemble the finite-element derivative of

    J(vᵧ) = -∫_Ωₒᵇₛ vᵧ dΩ,

where `Ωₒᵇₛ = (xmin, xmax) × (ymin, ymax)`.

For `vᵧʰ = Σᵢ Nᵢ Vᵧᵢ`, the derivative with respect to a global velocity degree of
freedom is the consistently assembled load vector `∂J/∂Vᵧᵢ = -∫_Ωₒᵇₛ Nᵢ dΩ`.

Each kernel invocation handles one element. Integration points outside the
observation box contribute nothing. `geo[iel][q][2]` is the physical quadrature
measure `|det(J)|ωq`. Neighbouring elements share velocity nodes, so element
load vectors are scattered with atomic additions.
"""
@kernel function objective_vy_kernel!(
    dJdvy,
    @Const(coords),
    @Const(el2n),
    @Const(geo),
    @Const(Nq),
    xmin,
    xmax,
    ymin,
    ymax,
    ::Val{N},
) where {N}
    iel = @index(Global)

    local_nodes = SVector{N, Int}(ntuple(i -> el2n[i, iel], Val(N)))

    element_load = zero(Nq[1])
    for q in eachindex(Nq)
        N_at_q = Nq[q]

        # Isoparametric map to the physical quadrature point: xq = Σᵢ Nᵢ(ξq) xᵢ.
        xq = zero(coords[local_nodes[1]])
        for i in 1:N
            xq += N_at_q[i] * coords[local_nodes[i]]
        end

        # For a discontinuous indicator, quadrature naturally approximates the
        # fraction of a cut element lying inside the observation box.
        if xmin < xq[1] < xmax && ymin < xq[2] < ymax
            dΩ = geo[iel][q][2]
            element_load -= N_at_q * dΩ
        end
    end

    for i in 1:N
        Atomix.@atomic :monotonic dJdvy[local_nodes[i]] += element_load[i]
    end
end

"""Assemble `∂J/∂Vy` for the observation-box integral objective."""
function assemble_objective_vy!(
    dJdvy, mesh, geo, element::ReferenceElement{E},
    xmin, xmax, ymin, ymax, workgroup,
) where {E <: AbstractElement{2, N}} where {N}
    backend = KernelAbstractions.get_backend(dJdvy)
    fill!(dJdvy, 0)

    # Use the same T7 quadrature rule as the momentum residual, so the objective
    # and residual discretizations are consistent.
    Nq = shape_function_values(element)
    objective_vy_kernel!(backend, workgroup)(
        dJdvy, mesh.coords, mesh.el2n, geo, Nq,
        xmin, xmax, ymin, ymax, Val(N);
        ndrange = mesh.nels,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Moduli contractions
#
# The elastic moduli reach the residual through two different blocks:
#
#   * The shear modulus `G` enters `deviatoric_stress` through the Maxwell
#     viscosity `ηve = (1/η + 1/(GΔt))⁻¹` and the elastic memory term
#     `τ_old/(2GΔt)`, so it appears only in the momentum residual.
#   * The bulk modulus `K` enters the momentum residual through the linearised
#     EOS `ρ = ρ0(1 - α(T-Tref) + P/K)`, and the pressure residual through the
#     storage term `∂P/∂t/(ηb Δt)` with `ηb = K` in this model.
#
# Both blocks therefore need their own contraction, one against the velocity
# multipliers and one against the pressure multiplier.
# ---------------------------------------------------------------------------

@kernel function momentum_moduli_contraction_kernel!(
    contracted_residual,
    @Const(vx), @Const(vy), @Const(P), @Const(T),
    @Const(λvx), @Const(λvy),
    @Const(el2n), @Const(dofsP), @Const(geo), @Const(phases),
    @Const(τ_old), G_element, K_element,
    @Const(η), @Const(α), @Const(ρ0), @Const(g), Tref, Δt,
    @Const(Nq), @Const(NqP), ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    velocity_nodes = SVector{NV, Int}(ntuple(i -> el2n[i, iel], Val(NV)))
    pressure_dofs  = SVector{NP, Int}(ntuple(i -> dofsP[i, iel], Val(NP)))
    phase = Int(phases[1, iel])
    vx_e = SVector{NV}(ntuple(i -> vx[velocity_nodes[i]], Val(NV)))
    vy_e = SVector{NV}(ntuple(i -> vy[velocity_nodes[i]], Val(NV)))
    P_e  = SVector{NP}(ntuple(i -> P[pressure_dofs[i]], Val(NP)))
    T_e  = SVector{NP}(ntuple(i -> T[pressure_dofs[i]], Val(NP)))
    λx_e = SVector{NV}(ntuple(i -> λvx[velocity_nodes[i]], Val(NV)))
    λy_e = SVector{NV}(ntuple(i -> λvy[velocity_nodes[i]], Val(NV)))
    # The sensitivity is taken with respect to this element's own G and K, so the
    # residual is evaluated with single-entry moduli tuples and an all-ones phase
    # vector that indexes that lone entry. Differentiating w.r.t. `G_element[iel]`
    # / `K_element[iel]` then gives the per-element sensitivity, while the
    # phase-indexed `η`, `α`, `ρ0` tuples are still read at the true `phase`.
    phase_e = SVector{NV, Int}(ntuple(_ -> 1, Val(NV)))
    τ_old_e = FEMTools.IntegrationPointStress(
        SVector(ntuple(q -> τ_old[1][q, iel], length(Nq))),
        SVector(ntuple(q -> τ_old[2][q, iel], length(Nq))),
        SVector(ntuple(q -> τ_old[3][q, iel], length(Nq))),
    )
    Re_x, Re_y = FEMTools.integrate_momentum_residual(
        (vx_e, vy_e), P_e, nothing, T_e, geo[iel], phase_e,
        (η[phase],), (G_element[iel],), (α[phase],),
        (ρ0[phase],), (K_element[iel],), g, Tref, Δt, Nq, NqP,
        τ_old_e,
    )
    contracted_residual[iel] = dot(λx_e, Re_x) + dot(λy_e, Re_y)
end

function launch_momentum_moduli_contraction!(out, vx, vy, P, T, λvx, λvy,
    el2n, dofsP, geo, phases, τ_old, G_element, K_element,
    η, α, ρ0, g, Tref, Δt, Nq, NqP, ::Val{NV}, ::Val{NP}, workgroup,
) where {NV, NP}
    fill!(out, 0)
    backend = KernelAbstractions.get_backend(out)
    momentum_moduli_contraction_kernel!(backend, workgroup)(out, vx, vy, P, T, λvx, λvy,
        el2n, dofsP, geo, phases, τ_old, G_element, K_element,
        η, α, ρ0, g, Tref, Δt, Nq, NqP, Val(NV), Val(NP); ndrange = size(el2n, 2))
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function pressure_moduli_contraction_kernel!(
    contracted_residual,
    @Const(vx), @Const(vy), @Const(P), @Const(P0), @Const(T), @Const(T0),
    @Const(λP),
    @Const(el2n), @Const(dofsP), @Const(geo_v), @Const(geo_P), @Const(phases),
    ηb_element, @Const(α), Δt, @Const(NqP), ::Val{NV}, ::Val{NP},
) where {NV, NP}
    iel = @index(Global)
    velocity_nodes = SVector{NV, Int}(ntuple(i -> el2n[i, iel], Val(NV)))
    pressure_dofs  = SVector{NP, Int}(ntuple(i -> dofsP[i, iel], Val(NP)))
    phase = Int(phases[1, iel])
    vx_e  = SVector{NV}(ntuple(i -> vx[velocity_nodes[i]], Val(NV)))
    vy_e  = SVector{NV}(ntuple(i -> vy[velocity_nodes[i]], Val(NV)))
    P_e   = SVector{NP}(ntuple(i -> P[pressure_dofs[i]], Val(NP)))
    P0_e  = SVector{NP}(ntuple(i -> P0[pressure_dofs[i]], Val(NP)))
    T_e   = SVector{NP}(ntuple(i -> T[pressure_dofs[i]], Val(NP)))
    T0_e  = SVector{NP}(ntuple(i -> T0[pressure_dofs[i]], Val(NP)))
    λP_e  = SVector{NP}(ntuple(i -> λP[pressure_dofs[i]], Val(NP)))
    # The pressure residual interpolates its material tuples with the NP pressure
    # shape functions, so the all-ones phase vector is sized to NP here.
    phase_e = SVector{NP, Int}(ntuple(_ -> 1, Val(NP)))
    RP_e = FEMTools.integrate_PH_pressure_residual(
        (vx_e, vy_e), P_e, P0_e, T_e, T0_e,
        geo_v[iel], geo_P[iel], phase_e,
        (α[phase],), (ηb_element[iel],), Δt, NqP,
    )
    contracted_residual[iel] = dot(λP_e, RP_e)
end

function launch_pressure_moduli_contraction!(out, vx, vy, P, P0, T, T0, λP,
    el2n, dofsP, geo_v, geo_P, phases, ηb_element, α, Δt, NqP,
    ::Val{NV}, ::Val{NP}, workgroup,
) where {NV, NP}
    fill!(out, 0)
    backend = KernelAbstractions.get_backend(out)
    pressure_moduli_contraction_kernel!(backend, workgroup)(out, vx, vy, P, P0, T, T0, λP,
        el2n, dofsP, geo_v, geo_P, phases, ηb_element, α, Δt, NqP,
        Val(NV), Val(NP); ndrange = size(el2n, 2))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    moduli_sensitivities(dr, mesh_stokes, λvx, λvy, λP, geo_v, geo_P, phases,
                         τ_old, element_v, element_P, η, α, ρ0, G, K,
                         cell_phase, g, Tref, Δt, Val(NV), Val(NP),
                         backend, workgroup)
        -> (shear_modulus_sensitivity, bulk_modulus_sensitivity, bulk_momentum_part, bulk_pressure_part)

Assemble the per-element elastic-moduli sensitivities `sᵉ(m) = λᵉᵀ ∂Rᵉ/∂m` of the
converged adjoint state by reverse-differentiating the momentum and pressure
contractions with respect to element moduli fields.

`cell_phase[iel]` selects the phase whose `G`/`K` value seeds element `iel`. The
bulk-modulus sensitivity is returned as its total and as its two separate
contributions, since the momentum path vanishes identically whenever gravity is
zero and only the pressure storage term then carries the signal.

Returns CPU arrays; summing the entries of one phase gives the derivative with
respect to that phase's global modulus.
"""
function moduli_sensitivities(
    dr, mesh_stokes, λvx, λvy, λP, geo_v, geo_P, phases, τ_old,
    element_v, element_P, η, α, ρ0, G, K, cell_phase,
    g, Tref, Δt, ::Val{NV}, ::Val{NP}, backend, workgroup,
) where {NV, NP}
    nels = mesh_stokes.nels
    Nq_v = shape_function_values(element_v)
    Nq_P = shape_function_values(element_P, element_v.integration_points)

    # --- momentum block: ∂R_v/∂G and ∂R_v/∂K -------------------------------
    contracted_momentum = KernelAbstractions.zeros(backend, Float64, nels)
    momentum_seed = KernelAbstractions.ones(backend, Float64, nels)
    G_element = KernelAbstractions.zeros(backend, Float64, nels)
    K_element = similar(G_element)
    copyto!(G_element, G[cell_phase])
    copyto!(K_element, K[cell_phase])
    shear_sensitivity = zero(G_element)
    bulk_momentum_sensitivity = zero(K_element)

    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(launch_momentum_moduli_contraction!), Enzyme.Const,
        Enzyme.Duplicated(contracted_momentum, momentum_seed),
        Enzyme.Const(dr.vx), Enzyme.Const(dr.vy), Enzyme.Const(dr.P), Enzyme.Const(dr.T),
        Enzyme.Const(λvx), Enzyme.Const(λvy),
        Enzyme.Const(mesh_stokes.el2n), Enzyme.Const(mesh_stokes.DoFsP),
        Enzyme.Const(geo_v), Enzyme.Const(phases), Enzyme.Const(τ_old),
        Enzyme.Duplicated(G_element, shear_sensitivity),
        Enzyme.Duplicated(K_element, bulk_momentum_sensitivity),
        Enzyme.Const(η), Enzyme.Const(α), Enzyme.Const(ρ0), Enzyme.Const(g),
        Enzyme.Const(Tref), Enzyme.Const(Δt), Enzyme.Const(Nq_v), Enzyme.Const(Nq_P),
        Enzyme.Const(Val(NV)), Enzyme.Const(Val(NP)), Enzyme.Const(workgroup),
    )

    # --- pressure block: ∂R_P/∂ηb, with ηb = K -----------------------------
    contracted_pressure = KernelAbstractions.zeros(backend, Float64, nels)
    pressure_seed = KernelAbstractions.ones(backend, Float64, nels)
    ηb_element = KernelAbstractions.zeros(backend, Float64, nels)
    copyto!(ηb_element, K[cell_phase])
    bulk_pressure_sensitivity = zero(ηb_element)

    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(launch_pressure_moduli_contraction!), Enzyme.Const,
        Enzyme.Duplicated(contracted_pressure, pressure_seed),
        Enzyme.Const(dr.vx), Enzyme.Const(dr.vy),
        Enzyme.Const(dr.P), Enzyme.Const(dr.P0),
        Enzyme.Const(dr.T), Enzyme.Const(dr.T0),
        Enzyme.Const(λP),
        Enzyme.Const(mesh_stokes.el2n), Enzyme.Const(mesh_stokes.DoFsP),
        Enzyme.Const(geo_v), Enzyme.Const(geo_P), Enzyme.Const(phases),
        Enzyme.Duplicated(ηb_element, bulk_pressure_sensitivity),
        Enzyme.Const(α), Enzyme.Const(Δt), Enzyme.Const(Nq_P),
        Enzyme.Const(Val(NV)), Enzyme.Const(Val(NP)), Enzyme.Const(workgroup),
    )

    shear = Array(shear_sensitivity)
    bulk_momentum = Array(bulk_momentum_sensitivity)
    bulk_pressure = Array(bulk_pressure_sensitivity)
    return shear, bulk_momentum .+ bulk_pressure, bulk_momentum, bulk_pressure
end

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    main(; n_circle=96, max_area=1 / (1 * 64^2), Δt=1 / 6, show_plot=true, kwargs...) -> NamedTuple

Solve one unstructured T7/P1-disc pure-shear Stokes step and its discrete
adjoint, then assemble the elastic-moduli sensitivities of the observation-box
velocity objective `J(vᵧ) = -∫_Ωₒᵇₛ vᵧ dΩ`.

The model is the viscoelastic pure-shear problem of
`stokes_2D_pure_shear_triangle.jl`: a square domain with a circular inclusion
that is softer in shear than the matrix, driven by pure-shear boundary
conditions. After the forward solve it solves the discrete adjoint
`(∂R/∂u)ᵀλ = -∂J/∂u` on the same discretisation and contracts `λᵀ ∂R/∂m` with
Enzyme for `m ∈ {G, K}`.

This is a *single-step* adjoint. The forward example advances a stress history
over many steps, but the multipliers here are those of one time step taken from
the zero-stress state; a multi-step gradient would have to propagate the adjoint
backwards through the stress history as well.

`plastic` must be `nothing`. The Drucker-Prager return map reads the bulk
modulus a third time through its regularised plastic multiplier, and that path is
not differentiated here, so a yielding configuration would report an incomplete
`K` gradient.

Returns a `NamedTuple` with the solver state (`dr`, `mesh_stokes`), forward and
adjoint statistics, the adjoint fields (`λvx`, `λvy`, `λP`), the objective load
`objective_vy`, the per-element `shear_modulus_sensitivity` and
`bulk_modulus_sensitivity` (with the momentum and pressure parts of the latter
kept separately), and the phase-summed gradients.
"""
function main(;
    n_circle = 96,
    max_area = 1 / (1 * 64^2),
    Δt = 1 / 6,
    G = (1e0, 0.5),
    K = (4e0, 8e0),
    η = (1.0, 1.0),
    show_plot = true,
    verbose = true,
    plastic = nothing,
    γfact = 20.0,
    # The adjoint operator does not tolerate the aggressive DYREL settings the
    # forward-only example uses (CFL 0.99, c_fact 0.9): its iteration diverges
    # there even though the forward solve converges.
    CFL_v = 0.9,
    CFL_P = 0.9,
    c_fact = 0.7,
    ncheck = 100,
    ϵ_tol = 1.0e-6,
    iterMax = 50_000,
    total_iterMax = 50_000,
    adjoint_tol = 1.0e-6,
    adjoint_rel_drop = 0.1,
    adjoint_iterMax = 50_000,
    adjoint_total_iterMax = 50_000,
    adjoint_max_ph_iterations = 100,
)
    plastic === nothing || throw(ArgumentError(
        "the moduli contractions do not differentiate the Drucker-Prager return " *
            "map, which reads the bulk modulus through its plastic multiplier; " *
            "the reported K sensitivity would be incomplete"
    ))

    # Domain
    Lx, Ly = 1.0, 1.0

    # Background pure-shear strain rate (non-dimensional)
    ε̇_bg = 1.0

    # Material (2 phases: matrix + inclusion)
    α     = (0.0,     0.0)   # thermal expansivity  (zero → isothermal)
    ρ0    = (1.0,     1.0)   # reference density
    # `G` and `K` are the parameters differentiated below, so they arrive as
    # keyword arguments; the inclusion defaults to being softer in shear and
    # stiffer in compression than the matrix.
    ηb    = K                # pressure storage modulus; residual uses ηb * Δt
    G_stokes = G
    g     = (0.0,     0.0)   # gravity vector
    Tref  = 0.0

    # Gmsh fragments the domain at this material interface.
    r_incl = 0.1
    cx     = Lx / 2
    cy     = Ly / 2

    # Observation band, offset above the inclusion. A band centred on the
    # inclusion would integrate the antisymmetric pure-shear vy to nearly zero.
    objective_bounds = (0.4, 0.6, 0.6, 0.7)

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    coords_v_cpu, el2n_v_cpu, outer_nodes, circle_nodes = build_gmsh_t7_circle_inclusion_mesh(;
        Lx, Ly,
        cx, cy, r = r_incl,
        n_circle,
        max_area,
    )
    DoFs_v_cpu = Int32.(1:length(coords_v_cpu))
    mesh_v = Mesh(
        element_v, nothing, nothing,
        coords_v_cpu, DoFs_v_cpu, el2n_v_cpu, outer_nodes,
    )
    mesh_stokes = MixedMesh(mesh_v, element_P)

    @info "Gmsh mixed mesh (T7/P1-disc)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels n_circle max_area n_interface_nodes=length(circle_nodes)

    # ---------------------------------------------------------------------------
    # Geometry precompute  (both fields evaluated at velocity integration points)
    # ---------------------------------------------------------------------------

    ip_v  = element_v.integration_points
    NQ_v  = length(ip_v.ω)
    NV    = length(element_v)
    NP    = length(element_P)

    cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
    geo_v, geo_P = cache.geo_v, cache.geo_P

    # ---------------------------------------------------------------------------
    # StokesDR struct
    # ---------------------------------------------------------------------------

    material = StokesMaterial(; η, ηb, G = G_stokes, α, ρ0, K, g = Tuple(g), Tref)
    dr = StokesDR(
        backend, mesh_stokes.nnodes, mesh_stokes.nnodesP, material;
        CFL_v, CFL_P, c_fact,
        stress_size = (NQ_v, mesh_stokes.nels),
    )
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

    # ---------------------------------------------------------------------------
    # Phase assignment — circular inclusion
    # ---------------------------------------------------------------------------

    in_incl(c) = (c[1] - cx)^2 + (c[2] - cy)^2 ≤ r_incl^2

    coords_v     = Array(mesh_stokes.coords)
    el2nP_cpu    = Array(mesh_stokes.el2nP)
    DoFsP_cpu    = Array(mesh_stokes.DoFsP)
    cell_phase = Int[
        in_incl(sum(a -> coords_v[mesh_stokes.el2n[a, iel]], 1:NV) / NV) ? 2 : 1
        for iel in 1:mesh_stokes.nels
    ]
    phases_v_cpu = repeat(reshape(cell_phase, 1, :), NV, 1)
    phases_P_cpu = repeat(reshape(cell_phase, 1, :), NP, 1)

    @info "Phases" n_incl=count(==(2), cell_phase)

    # ---------------------------------------------------------------------------
    # Boundary conditions — pure shear
    #   vx = +ε̇_bg * (x - Lx/2),   vy = -ε̇_bg * (y - Ly/2)
    # ---------------------------------------------------------------------------

    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    tol = max(Lx, Ly) * eps(Float64) * 32
    vx_nodes = Int32[n for n in Γnodes if abs(coords[n][1]) ≤ tol || abs(coords[n][1] - Lx) ≤ tol]
    vy_nodes = Int32[n for n in Γnodes if abs(coords[n][2]) ≤ tol || abs(coords[n][2] - Ly) ≤ tol]

    bc_vx_vals = [ ε̇_bg * (coords[n][1] - Lx / 2) for n in vx_nodes]
    bc_vy_vals = [-ε̇_bg * (coords[n][2] - Ly / 2) for n in vy_nodes]
    bc_vx = DirichletBoundaryCondition(nothing, vx_nodes, bc_vx_vals)
    bc_vy = DirichletBoundaryCondition(nothing, vy_nodes, bc_vy_vals)

    # Seed the full interior with the analytical pure-shear field so the solver
    # starts from a good initial guess.
    copyto!(dr.vx, [ ε̇_bg * (c[1] - Lx / 2) for c in coords_v])
    copyto!(dr.vy, [-ε̇_bg * (c[2] - Ly / 2) for c in coords_v])

    apply_bc!(dr.vx, bc_vx)
    apply_bc!(dr.vy, bc_vy)

    @info "BCs" n_vx = length(vx_nodes) n_vy = length(vy_nodes) max_vx = maximum(abs, bc_vx_vals) max_vy = maximum(abs, bc_vy_vals)

    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh_stokes, cache, γfact, Δt; workgroup, phases_v = phases_v_cpu,
    )

    # ---------------------------------------------------------------------------
    # Forward solve
    # ---------------------------------------------------------------------------

    copyto!(dr.P0, dr.P)
    copyto!(dr.T0, dr.T)

    @info "Starting PH/DYREL-style Stokes solver" Δt iterMax total_iterMax ncheck ϵ_tol

    t_forward = @elapsed solve_stats = solve_stokes_dyrel!(
        dr, mesh_stokes, cache, bc_vx, bc_vy, Δt, γP;
        phases_v = phases_v_cpu, phases_P = phases_P_cpu, τ_old, plastic, workgroup,
        ncheck, ϵ_tol, iterMax, total_iterMax, rel_drop0 = 0.75,
        verbose, verbose_inner = false,
        collect_history = true,
    )
    # The adjoint freezes its transpose Jacobian and preconditioner at the forward
    # state, so differentiating an unconverged one yields a gradient of nothing in
    # particular. Refusing here keeps that failure visible.
    solve_stats.converged || error(
        "Forward solve did not reach tolerance (err = $(solve_stats.err), " *
            "tol = $(ϵ_tol), iter = $(solve_stats.iter)/$(total_iterMax)); " *
            "the adjoint requires a converged forward state."
    )

    # ---------------------------------------------------------------------------
    # Adjoint solve: (dR/du)' λ + dJ/du = 0
    # ---------------------------------------------------------------------------

    objective_vx = zero(dr.Rv_x)
    objective_vy = zero(dr.Rv_y)
    assemble_objective_vy!(
        objective_vy, mesh_stokes, geo_v, element_v,
        objective_bounds..., workgroup,
    )

    λvx = zero(dr.vx)
    λvy = zero(dr.vy)
    λP  = zero(dr.P)

    t_adjoint = @elapsed adjoint_stats = solve_stokes_adjoint_dyrel!(
        dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
        phases_v_cpu, phases_P_cpu, τ_old, plastic, G_stokes, Δt, γP,
        objective_vx, objective_vy, λvx, λvy, λP,
        backend, workgroup;
        vx_nodes,
        vy_nodes,
        ncheck,
        adjoint_tol,
        rel_drop = adjoint_rel_drop,
        iterMax = adjoint_iterMax,
        total_iterMax = adjoint_total_iterMax,
        max_ph_iterations = adjoint_max_ph_iterations,
        verbose,
        verbose_inner = false,
        collect_history = true,
    )
    verbose && @info "Adjoint solve complete" adjoint_stats
    # A diverged adjoint still produces multipliers, and contracting them yields
    # finite-looking numbers that are pure noise. Refuse them here.
    adjoint_stats.converged || error(
        "Adjoint solve did not reach tolerance (err = $(adjoint_stats.err), " *
            "tol = $(adjoint_tol), iter = $(adjoint_stats.iter)/$(adjoint_total_iterMax)); " *
            "the contracted sensitivities would be meaningless. Lower `CFL_v` or " *
            "`c_fact`, or raise the iteration budget."
    )

    # J is linear in the velocity, so the assembled objective load is also the
    # objective's coefficient vector.
    objective = dot(objective_vy, dr.vy)

    # ---------------------------------------------------------------------------
    # Moduli sensitivities
    # ---------------------------------------------------------------------------
    #
    # For a modulus m and element residual Rᵉ the plotted contribution is
    #
    #     sᵉ(m) = λᵉᵀ (∂Rᵉ/∂m).
    #
    # Each entry is one element's contribution. Summing entries of one phase gives
    # the derivative with respect to that phase's single global modulus; keeping
    # them element-wise gives the spatial sensitivity maps.

    shear_modulus_sensitivity, bulk_modulus_sensitivity,
        bulk_momentum_part, bulk_pressure_part = moduli_sensitivities(
        dr, mesh_stokes, λvx, λvy, λP, geo_v, geo_P, phases_v_cpu, τ_old,
        element_v, element_P, η, α, ρ0, G_stokes, K, cell_phase,
        g, Tref, Δt, Val(NV), Val(NP), backend, workgroup,
    )

    shear_modulus_gradient_by_phase = ntuple(
        p -> sum(shear_modulus_sensitivity[cell_phase .== p]), length(G),
    )
    bulk_modulus_gradient_by_phase = ntuple(
        p -> sum(bulk_modulus_sensitivity[cell_phase .== p]), length(K),
    )
    @info "Moduli sensitivities assembled" shear_modulus_gradient_by_phase bulk_modulus_gradient_by_phase
    # With g = 0 the density never enters the body force, so the EOS path through
    # K contributes nothing and the whole bulk gradient comes from the pressure
    # storage term. Report both so that stays visible rather than assumed.
    @info "Bulk modulus gradient split" momentum_path=sum(bulk_momentum_part) pressure_path=sum(bulk_pressure_part)

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------

    pts   = [Point2f(c) for c in coords_v]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
             for i in 1:mesh_stokes.nels]

    # The raw sensitivities are un-normalized element integrals
    # sᵉ = λᵉᵀ ∂Rᵉ/∂m ≈ areaᵉ·(sensitivity density); dividing by areaᵉ recovers
    # the mesh-independent density used for plotting, while the phase-wise
    # gradients keep summing the raw integrals.
    element_area = [
        let a = coords_v[el2nP_cpu[1, i]], b = coords_v[el2nP_cpu[2, i]], c = coords_v[el2nP_cpu[3, i]]
            abs((b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])) / 2
        end
        for i in 1:mesh_stokes.nels
    ]
    obs_xlo, obs_xhi, obs_ylo, obs_yhi = objective_bounds

    function draw_geometry!(ax)
        θ    = LinRange(0, 2π, 300)
        lines!(ax, cx .+ r_incl .* cos.(θ), cy .+ r_incl .* sin.(θ);
            color = :white, linewidth = 1.5, linestyle = :dash)
        lines!(ax,
            [obs_xlo, obs_xhi, obs_xhi, obs_xlo, obs_xlo],
            [obs_ylo, obs_ylo, obs_yhi, obs_yhi, obs_ylo];
            color = :yellow, linewidth = 2)
        return nothing
    end

    function plot_summary()
        el(dofs, conn) = [mean(dofs[conn[:, i]]) for i in 1:mesh_stokes.nels]
        vx_cpu, vy_cpu, P_cpu = Array(dr.vx), Array(dr.vy), Array(dr.P)
        fig = Figure(size = (1100, 1300))
        for (row, col, title, values, colormap) in (
                (1, 1, "Vx", el(vx_cpu, el2nP_cpu), :vik),
                (1, 3, "Vy", el(vy_cpu, el2nP_cpu), :vik),
                (2, 1, "P",  el(P_cpu,  DoFsP_cpu), :glasgow),
                (2, 3, "Shear modulus sensitivity (per area)",
                    shear_modulus_sensitivity ./ element_area, :vik),
                (3, 1, "Bulk modulus sensitivity (per area)",
                    bulk_modulus_sensitivity ./ element_area, :vik),
            )
            limits = extrema(values)
            ax = Axis(fig[row, col]; aspect = DataAspect(), title, xlabel = "x", ylabel = "y")
            poly!(ax, polys; color = values, colormap, colorrange = limits, strokewidth = 0)
            Colorbar(fig[row, col + 1]; colormap, limits, width = 15, tellheight = false)
            draw_geometry!(ax)
        end

        ax = Axis(fig[3, 3:4]; xlabel = "iteration", ylabel = "log10 residual",
            title = "Residual evolution")
        logres(v) = log10.(max.(v, eps()))
        fwd_hist, adj_hist = solve_stats.history, adjoint_stats.history
        if !isempty(fwd_hist)
            it = Float64[h.iter for h in fwd_hist]
            lines!(ax, it, logres([h.err_v for h in fwd_hist]); label = "forward V")
            lines!(ax, it, logres([h.err_P for h in fwd_hist]); label = "forward P")
        end
        if !isempty(adj_hist)
            it = Float64[h.iter for h in adj_hist]
            lines!(ax, it, logres([h.err_v for h in adj_hist]); label = "adjoint V", linestyle = :dash)
            lines!(ax, it, logres([h.err_P for h in adj_hist]); label = "adjoint P", linestyle = :dash)
        end
        (isempty(fwd_hist) && isempty(adj_hist)) || axislegend(ax; position = :rt)

        display(fig)
        return fig
    end

    show_plot && plot_summary()

    return (;
        dr, mesh_stokes, solve_stats, adjoint_stats, t_forward, t_adjoint,
        λvx, λvy, λP, objective_vy, objective,
        shear_modulus_sensitivity, bulk_modulus_sensitivity,
        bulk_momentum_part, bulk_pressure_part,
        shear_modulus_gradient_by_phase, bulk_modulus_gradient_by_phase,
        internals = (; cache, element_v, element_P, cell_phase, phases_v_cpu,
            η, α, ρ0, G = G_stokes, K, g, Tref, Δt, NV, NP),
    )
end

main()
