import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Enzyme
using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using KernelAbstractions
using Atomix
using Triangulate
using FEMTools
using GLMakie: Figure, Axis, Colorbar, poly!, lines!, Point2f, DataAspect, axislegend

const backend   = CPU()
const workgroup = 128

# ---------------------------------------------------------------------------
# Mesh: unstructured T7 velocity mesh around a circular inclusion
# ---------------------------------------------------------------------------

"""
    build_triangle_t7_inclusion_mesh(; Lx, Ly, cx, cy, r, n_circle=96, max_area=nothing) -> Tuple

Build an unstructured T7 velocity mesh with a circular material interface.

Triangulate.jl generates a second-order T6 PSLG mesh with the circle as a
constrained internal boundary, so no element crosses the interface. The local
midpoint ordering is remapped to FEMTools' T6/T7 convention and one centroid
bubble node is appended per element. Returns the coordinates, the T7
connectivity, the outer-boundary nodes, and the circle-interface nodes.
"""
function build_triangle_t7_inclusion_mesh(; Lx, Ly, cx, cy, r, n_circle = 96, max_area = nothing)
    rect_pts  = Cdouble[0.0 Lx  Lx 0.0;
                        0.0 0.0 Ly Ly]
    rect_segs = Cint[1 2; 2 3; 3 4; 4 1]'

    θ = range(0, 2π; length = n_circle + 1)[1:end-1]
    circ_pts = Matrix{Cdouble}(hcat(cx .+ r .* cos.(θ), cy .+ r .* sin.(θ))')
    circ_segs = Matrix{Cint}(hcat([
        [4 + i; 4 + mod1(i + 1, n_circle)] for i in 1:n_circle
    ]...))

    tio = TriangulateIO()
    tio.pointlist = hcat(rect_pts, circ_pts)
    tio.segmentlist = hcat(rect_segs, circ_segs)

    flags = isnothing(max_area) ? "pqo2Q" : "pq30o2a$(max_area)Q"
    result, _ = triangulate(flags, tio)

    pts = result.pointlist
    tris_t6 = Matrix{Int32}(result.trianglelist)
    coords = [SVector{2, Float64}(pts[1, i], pts[2, i]) for i in axes(pts, 2)]

    # Project the circular-interface nodes back to the analytical radius so the
    # inclusion boundary is fitted by the high-order geometry.
    sagitta = r * (1 - cos(π / n_circle))
    circle_tol = max(2.5 * sagitta, 100eps(Float64) * max(Lx, Ly))
    for i in eachindex(coords)
        dx = coords[i][1] - cx
        dy = coords[i][2] - cy
        radius = hypot(dx, dy)
        if abs(radius - r) ≤ circle_tol && radius > 0
            coords[i] = SVector{2, Float64}(cx + r * dx / radius, cy + r * dy / radius)
        end
    end

    n_t6 = length(coords)
    nels = size(tris_t6, 2)
    el2n = Matrix{Int32}(undef, 7, nels)
    el2n[1:3, :] .= tris_t6[1:3, :]
    el2n[4, :] .= tris_t6[6, :] # FEMTools node 4 = mid(1, 2)
    el2n[5, :] .= tris_t6[4, :] # FEMTools node 5 = mid(2, 3)
    el2n[6, :] .= tris_t6[5, :] # FEMTools node 6 = mid(3, 1)
    sizehint!(coords, n_t6 + nels)
    for iel in 1:nels
        c1 = coords[tris_t6[1, iel]]
        c2 = coords[tris_t6[2, iel]]
        c3 = coords[tris_t6[3, iel]]
        push!(coords, (c1 + c2 + c3) / 3)
        el2n[7, iel] = Int32(n_t6 + iel)
    end

    tol = 100eps(Float64) * max(Lx, Ly)
    outer_nodes = Int32[
        i for i in 1:n_t6
        if abs(coords[i][1]) ≤ tol || abs(coords[i][1] - Lx) ≤ tol ||
           abs(coords[i][2]) ≤ tol || abs(coords[i][2] - Ly) ≤ tol
    ]
    circle_nodes = Int32[
        i for i in 1:n_t6
        if abs(hypot(coords[i][1] - cx, coords[i][2] - cy) - r) ≤ circle_tol
    ]

    return coords, el2n, sort!(unique!(outer_nodes)), sort!(unique!(circle_nodes))
end

# ---------------------------------------------------------------------------
# Objective:  J(vᵧ) = -∫_Ωₒᵦₛ vᵧ dΩ  over the observation box
# ---------------------------------------------------------------------------

"""
    objective_vy_kernel!(dJdvy, coords, el2n, geo, Nq, xmin, xmax, ymin, ymax, Val(N))

Assemble the consistent finite-element load `∂J/∂Vᵧᵢ = -∫_Ωₒᵦₛ Nᵢ dΩ` for the
objective `J(vᵧ) = -∫_Ωₒᵦₛ vᵧ dΩ`, where `Ωₒᵦₛ = (xmin, xmax) × (ymin, ymax)`.

Each element is handled by one work item. Integration points outside the box
contribute nothing; a cut element is weighted by the quadrature fraction of its
points that fall inside. Element loads are scattered with atomic additions
because neighbouring elements share velocity nodes.
"""
@kernel function objective_vy_kernel!(
    dJdvy, @Const(coords), @Const(el2n), @Const(geo), @Const(Nq),
    xmin, xmax, ymin, ymax, ::Val{N},
) where {N}
    iel = @index(Global)
    local_nodes = SVector{N, Int}(ntuple(i -> el2n[i, iel], Val(N)))
    element_load = zero(Nq[1])
    for q in eachindex(Nq)
        N_at_q = Nq[q]
        xq = zero(coords[local_nodes[1]])
        for i in 1:N
            xq += N_at_q[i] * coords[local_nodes[i]]
        end
        if xmin < xq[1] < xmax && ymin < xq[2] < ymax
            dΩ = geo[iel][q][2]
            element_load -= N_at_q * dΩ
        end
    end
    for i in 1:N
        Atomix.@atomic :monotonic dJdvy[local_nodes[i]] += element_load[i]
    end
end

"""Assemble `∂J/∂Vᵧ` for the observation-box velocity integral."""
function assemble_objective_vy!(
    dJdvy, mesh, geo, element::ReferenceElement{E},
    xmin, xmax, ymin, ymax, workgroup,
) where {E <: AbstractElement{2, N}} where {N}
    backend = KernelAbstractions.get_backend(dJdvy)
    fill!(dJdvy, 0)
    Nq = shape_function_values(element)
    objective_vy_kernel!(backend, workgroup)(
        dJdvy, mesh.coords, mesh.el2n, geo, Nq,
        xmin, xmax, ymin, ymax, Val(N); ndrange = mesh.nels,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Material contraction: sᵉ(G) = -λᵉᵀ ∂Rᵉ/∂G, differentiated by Enzyme
# ---------------------------------------------------------------------------

"""
    G_contraction_kernel!(...)

Contract the converged adjoint state with each element momentum residual,
writing `contracted_residual[iel] = λₓᵉ·Rₓᵉ + λᵧᵉ·Rᵧᵉ`.

The residual is evaluated with the element's own shear modulus threaded through
a single-entry `G` tuple `(G_element[iel],)` and a phase vector of all ones,
while `η`, `α`, `ρ0`, and `K` are still read at the true material `phase`.
Reverse-differentiating this kernel with respect to `G_element` then yields the
per-element sensitivity `∂(λᵀR)/∂G = λᵀ ∂R/∂G` at fixed forward and adjoint
states. With `λ` solving `(∂R/∂u)ᵀλ = -∂J/∂u`, this equals `dJ/dG`, the discrete
gradient of `J` with respect to element shear modulus.
"""
@kernel function G_contraction_kernel!(
    contracted_residual,
    @Const(vx), @Const(vy), @Const(P), @Const(T),
    @Const(λvx), @Const(λvy),
    @Const(el2n), @Const(dofsP), @Const(geo), @Const(phases),
    @Const(τ_old), G_element,
    @Const(η), @Const(α), @Const(ρ0), @Const(K), @Const(g), Tref, Δt,
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
    phase_e = SVector{NV, Int}(ntuple(_ -> 1, Val(NV)))
    τ_old_e = FEMTools.IntegrationPointStress(
        SVector(ntuple(q -> τ_old[1][q, iel], length(Nq))),
        SVector(ntuple(q -> τ_old[2][q, iel], length(Nq))),
        SVector(ntuple(q -> τ_old[3][q, iel], length(Nq))),
    )
    Re_x, Re_y = FEMTools.integrate_momentum_residual(
        (vx_e, vy_e), P_e, nothing, T_e, geo[iel], phase_e,
        (η[phase],), (G_element[iel],), (α[phase],),
        (ρ0[phase],), (K[phase],), g, Tref, Δt,
        τ_old_e, nothing, Nq, NqP,
    )
    contracted_residual[iel] = dot(λx_e, Re_x) + dot(λy_e, Re_y)
end

function launch_G_contraction!(out, vx, vy, P, T, λvx, λvy,
    el2n, dofsP, geo, phases, τ_old, G_element,
    η, α, ρ0, K, g, Tref, Δt, Nq, NqP, ::Val{NV}, ::Val{NP}, workgroup,
) where {NV, NP}
    fill!(out, 0)
    backend = KernelAbstractions.get_backend(out)
    G_contraction_kernel!(backend, workgroup)(out, vx, vy, P, T, λvx, λvy,
        el2n, dofsP, geo, phases, τ_old, G_element,
        η, α, ρ0, K, g, Tref, Δt, Nq, NqP, Val(NV), Val(NP); ndrange = size(el2n, 2))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    shear_modulus_sensitivities(...) -> Vector

Assemble the per-element shear-modulus sensitivity `sᵉ(G) = λᵉᵀ ∂Rᵉ/∂G` of the
converged adjoint state by reverse-differentiating `launch_G_contraction!` with
respect to the element shear-modulus field. `cell_phase[iel]` selects the phase
whose `G` value seeds element `iel`. Summing the entries of one phase gives
`dJ/dG` for that phase's global shear modulus.
"""
function shear_modulus_sensitivities(
    dr, mesh_stokes, λvx, λvy, geo_v, phases, τ_old,
    element_v, element_P, η, α, ρ0, K, G, cell_phase,
    g, Tref, Δt, ::Val{NV}, ::Val{NP}, backend, workgroup,
) where {NV, NP}
    contracted_residual = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nels)
    contracted_seed = KernelAbstractions.ones(backend, Float64, mesh_stokes.nels)
    G_element = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nels)
    copyto!(G_element, G[cell_phase])
    G_sensitivity_backend = zero(G_element)
    Nq_v = shape_function_values(element_v)
    Nq_P = shape_function_values(element_P, element_v.integration_points)

    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(launch_G_contraction!), Enzyme.Const,
        Enzyme.Duplicated(contracted_residual, contracted_seed),
        Enzyme.Const(dr.vx), Enzyme.Const(dr.vy), Enzyme.Const(dr.P), Enzyme.Const(dr.T),
        Enzyme.Const(λvx), Enzyme.Const(λvy),
        Enzyme.Const(mesh_stokes.el2n), Enzyme.Const(mesh_stokes.DoFsP),
        Enzyme.Const(geo_v), Enzyme.Const(phases), Enzyme.Const(τ_old),
        Enzyme.Duplicated(G_element, G_sensitivity_backend),
        Enzyme.Const(η), Enzyme.Const(α), Enzyme.Const(ρ0), Enzyme.Const(K), Enzyme.Const(g),
        Enzyme.Const(Tref), Enzyme.Const(Δt), Enzyme.Const(Nq_v), Enzyme.Const(Nq_P),
        Enzyme.Const(Val(NV)), Enzyme.Const(Val(NP)), Enzyme.Const(workgroup),
    )
    return Array(G_sensitivity_backend)
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    main(; max_area=1/64^2, Δt=1.0, n_circle=96, show_plot=true, check_fd=true, kwargs...) -> NamedTuple

Solve one viscoelastic pure-shear Stokes step on a T7/P1-disc mesh with a
circular inclusion, then its discrete adjoint, and assemble the sensitivity of

    J(vᵧ) = -∫_Ωₒᵦₛ vᵧ dΩ

with respect to the shear modulus `G`. The observation box `Ωₒᵦₛ` is a small
rectangle sitting on top of the circular inclusion.

The two phases (matrix / inclusion) differ only in `G`, so the sensitivity map
isolates how the shear modulus of each region controls the vertical velocity
sampled above the inclusion. When `check_fd` is true, the phase-summed adjoint
gradient is verified against a central finite difference of `J(G)`.

The adjoint gradient matches the central finite difference to a few parts in
1e5 across bulk moduli and elastic time steps, because `solve_stokes_adjoint_dyrel!`
carries the full transpose Jacobian including the elastic pressure self-coupling
`(∂RP/∂P)ᵀλP` (nonzero for finite `K`). The finite-difference check reports the
per-phase agreement.

Returns the solver state, the forward and adjoint statistics, the adjoint
fields, the per-element `G_sensitivity`, and its phase sums
`G_gradient_by_phase`.
"""
function main(;
    max_area = 1 / 64^2,
    Δt = 1.0,
    n_circle = 96,
    show_plot = true,
    check_fd = true,
    Kmat = (4.0, 4.0),
    ncheck = 100,
    ϵ_tol = 1.0e-8,
    iterMax = 50_000,
    total_iterMax = 50_000,
    adjoint_tol = 1.0e-8,
    adjoint_rel_drop = 0.1,
    adjoint_iterMax = 50_000,
    adjoint_total_iterMax = 50_000,
    adjoint_max_ph_iterations = 100,
    γfact = 20.0,
    CFL_v = 0.99,
    c_fact = 0.9,
)
    # Domain and background pure-shear strain rate
    Lx, Ly = 1.0, 1.0
    ε̇_bg = 1.0

    # Material (2 phases: matrix + inclusion). Only G differs between phases, so
    # the objective's sensitivity is driven entirely by the shear modulus.
    η    = (1.0, 1.0)   # shear viscosity
    α    = (0.0, 0.0)   # thermal expansivity (isothermal)
    ρ0   = (1.0, 1.0)   # reference density
    K    = Kmat         # bulk modulus (finite → weakly compressible)
    ηb   = K            # pressure storage modulus; residual uses ηb·Δt
    G    = (1.0, 0.5)   # shear modulus: softer inclusion
    g    = (0.0, 0.0)   # no gravity
    Tref = 0.0
    plastic = nothing   # purely viscoelastic, so the adjoint stays linear

    # Circular inclusion, centred in the box.
    r_incl = 0.1
    cx     = Lx / 2
    cy     = Ly / 2

    # Observation box: a small rectangle resting on top of the inclusion.
    obs_xlo, obs_xhi = cx - 0.05, cx + 0.05
    obs_ylo, obs_yhi = cy + r_incl + 0.01, cy + r_incl + 0.07
    objective_bounds = (obs_xlo, obs_xhi, obs_ylo, obs_yhi)

    # -----------------------------------------------------------------------
    # Mesh and geometry
    # -----------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    coords_v_cpu, el2n_v_cpu, outer_nodes, circle_nodes = build_triangle_t7_inclusion_mesh(;
        Lx, Ly, cx, cy, r = r_incl, n_circle, max_area,
    )
    DoFs_v_cpu = Int32.(1:length(coords_v_cpu))
    mesh_v = Mesh(element_v, nothing, nothing, coords_v_cpu, DoFs_v_cpu, el2n_v_cpu, outer_nodes)
    mesh_stokes = MixedMesh(mesh_v, element_P)

    TDev = FEMTools.TA(backend)

    @info "Triangle mixed mesh (T7/P1-disc)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels n_interface_nodes=length(circle_nodes)

    NV = length(element_v)
    NP = length(element_P)
    NQ_v = length(element_v.integration_points.ω)

    cache = MixedMeshCache(backend, workgroup, mesh_stokes, element_v, element_P)
    geo_v, geo_P = cache.geo_v, cache.geo_P

    # -----------------------------------------------------------------------
    # Solver state
    # -----------------------------------------------------------------------

    dr = StokesDR(
        backend, mesh_stokes.nnodes, mesh_stokes.nnodesP,
        η, ηb, α; ρ0, K, g, Tref,
        CFL_v, CFL_P = 0.9, c_fact,
        stress_size = (NQ_v, mesh_stokes.nels),
    )
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

    # -----------------------------------------------------------------------
    # Phase assignment — circular inclusion
    # -----------------------------------------------------------------------

    in_incl(c) = (c[1] - cx)^2 + (c[2] - cy)^2 ≤ r_incl^2
    coords_v  = Array(mesh_stokes.coords)
    el2nP_cpu = Array(mesh_stokes.el2nP)
    DoFsP_cpu = Array(mesh_stokes.DoFsP)
    cell_phase = Int[
        in_incl(sum(a -> coords_v[mesh_stokes.el2n[a, iel]], 1:NV) / NV) ? 2 : 1
        for iel in 1:mesh_stokes.nels
    ]
    phases_solve = TDev(reshape(cell_phase, 1, :))
    @info "Phases" n_incl=count(==(2), cell_phase)

    # -----------------------------------------------------------------------
    # Boundary conditions — pure shear
    #   vx = +ε̇_bg·(x − Lx/2),   vy = −ε̇_bg·(y − Ly/2)
    # -----------------------------------------------------------------------

    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)
    tol = max(Lx, Ly) * eps(Float64) * 32
    vx_nodes = TDev(Int32[n for n in Γnodes if abs(coords[n][1]) ≤ tol || abs(coords[n][1] - Lx) ≤ tol])
    vy_nodes = TDev(Int32[n for n in Γnodes if abs(coords[n][2]) ≤ tol || abs(coords[n][2] - Ly) ≤ tol])
    bc_vx_vals = TDev([ ε̇_bg * (coords[n][1] - Lx / 2) for n in Array(vx_nodes)])
    bc_vy_vals = TDev([-ε̇_bg * (coords[n][2] - Ly / 2) for n in Array(vy_nodes)])

    # Analytical pure-shear field used both as the initial guess and to re-seed
    # the forward solve for the finite-difference gradient check.
    vx_init = [ ε̇_bg * (c[1] - Lx / 2) for c in coords_v]
    vy_init = [-ε̇_bg * (c[2] - Ly / 2) for c in coords_v]

    @info "BCs" n_vx=length(vx_nodes) n_vy=length(vy_nodes)

    # Viscosity-weighted pressure update scale for the Powell–Hestenes/DYREL step.
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh_stokes, geo_P, element_v, element_P,
        γfact, Δt, backend, workgroup; phases_v = phases_solve,
    )

    # Objective load ∂J/∂vᵧ over the observation box (∂J/∂vₓ = 0).
    objective_vx = zero(dr.Rv_x)
    objective_vy = zero(dr.Rv_y)
    assemble_objective_vy!(objective_vy, mesh_stokes, geo_v, element_v, objective_bounds..., workgroup)

    # J = -∫_box vᵧ dΩ = ⟨∂J/∂vᵧ, vᵧ⟩ for the consistently assembled load.
    objective_value() = dot(objective_vy, dr.vy)

    # Re-runnable forward solve at a given shear-modulus tuple, returning J. The
    # inclusion phase reads Gval[2], the matrix Gval[1]; the mesh, boundary data,
    # and pressure scaling are held fixed.
    function run_forward!(Gval; verbose = false, collect_history = false)
        copyto!(dr.vx, vx_init); copyto!(dr.vy, vy_init)
        fill!(dr.P, 0); fill!(dr.P0, 0)
        apply_bc!(dr.vx, DirichletBoundaryCondition(nothing, vx_nodes, bc_vx_vals))
        apply_bc!(dr.vy, DirichletBoundaryCondition(nothing, vy_nodes, bc_vy_vals))
        stats = solve_stokes_dyrel!(
            dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
            phases_solve, phases_solve, τ_old, plastic, Gval, Δt, γP,
            Γnodes, bc_vx_vals, bc_vy_vals, backend, workgroup;
            ncheck, ϵ_tol, iterMax, total_iterMax, rel_drop0 = 0.1,
            verbose, verbose_inner = false, vx_nodes, vy_nodes, collect_history,
        )
        return stats
    end

    # -----------------------------------------------------------------------
    # Forward solve at the reference G
    # -----------------------------------------------------------------------

    @info "Forward viscoelastic pure-shear solve" Δt G
    solve_stats = run_forward!(G; verbose = true, collect_history = true)
    solve_stats.converged || @warn "Forward solve did not reach tolerance" solve_stats
    J0 = objective_value()
    @info "Objective at reference G" J0

    # Snapshot the forward fields for plotting before the FD check overwrites dr.
    vx_fwd = Array(dr.vx)
    vy_fwd = Array(dr.vy)
    P_fwd  = Array(dr.P)

    # -----------------------------------------------------------------------
    # Adjoint solve:  (∂R/∂u)ᵀ λ = −∂J/∂u
    # -----------------------------------------------------------------------

    λvx = zero(dr.vx)
    λvy = zero(dr.vy)
    λP  = zero(dr.P)
    adjoint_stats = solve_stokes_adjoint_dyrel!(
        dr, mesh_stokes, geo_v, geo_P, element_v, element_P,
        phases_solve, phases_solve, τ_old, plastic, G, Δt, γP,
        objective_vx, objective_vy, λvx, λvy, λP,
        backend, workgroup;
        vx_nodes, vy_nodes, ncheck, adjoint_tol,
        rel_drop = adjoint_rel_drop, iterMax = adjoint_iterMax,
        total_iterMax = adjoint_total_iterMax,
        max_ph_iterations = adjoint_max_ph_iterations,
        verbose = true, verbose_inner = false, collect_history = true,
    )
    @info "Adjoint solve complete" adjoint_stats.converged

    # -----------------------------------------------------------------------
    # Shear-modulus sensitivity:  sᵉ(G) = λᵉᵀ ∂Rᵉ/∂G = dJ/dGᵉ
    # -----------------------------------------------------------------------

    G_sensitivity = shear_modulus_sensitivities(
        dr, mesh_stokes, λvx, λvy, geo_v, phases_solve, τ_old,
        element_v, element_P, η, α, ρ0, K, G, cell_phase,
        g, Tref, Δt, Val(NV), Val(NP), backend, workgroup,
    )
    G_gradient_by_phase = ntuple(p -> sum(G_sensitivity[cell_phase .== p]), length(G))
    @info "Shear-modulus sensitivities assembled" G_gradient_by_phase

    # -----------------------------------------------------------------------
    # Finite-difference verification of the phase-summed gradient
    # -----------------------------------------------------------------------

    fd_gradient_by_phase = nothing
    if check_fd
        fd = zeros(length(G))
        for p in eachindex(G)
            h = 1e-3 * G[p]
            Gp = ntuple(q -> q == p ? G[q] + h : G[q], length(G))
            Gm = ntuple(q -> q == p ? G[q] - h : G[q], length(G))
            run_forward!(Gp); Jp = objective_value()
            run_forward!(Gm); Jm = objective_value()
            fd[p] = (Jp - Jm) / (2h)
        end
        fd_gradient_by_phase = Tuple(fd)
        rel_err = ntuple(p -> abs(G_gradient_by_phase[p] - fd[p]) /
                              max(abs(fd[p]), eps()), length(G))
        @info "Finite-difference check (dJ/dG per phase)" adjoint=G_gradient_by_phase finite_difference=fd_gradient_by_phase rel_err
    end

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------

    if show_plot
        pts   = [Point2f(c) for c in coords_v]
        polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
                 for i in 1:mesh_stokes.nels]

        # Element areas normalise the raw sensitivity integrals sᵉ = λᵉᵀ ∂Rᵉ/∂G
        # to a mesh-independent sensitivity density for plotting.
        element_area = [
            let a = coords_v[el2nP_cpu[1, i]], b = coords_v[el2nP_cpu[2, i]], c = coords_v[el2nP_cpu[3, i]]
                abs((b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2])) / 2
            end for i in 1:mesh_stokes.nels
        ]

        θ    = LinRange(0, 2π, 300)
        xs_c = cx .+ r_incl .* cos.(θ)
        ys_c = cy .+ r_incl .* sin.(θ)
        function draw_overlays!(ax)
            lines!(ax, xs_c, ys_c; color = :white, linewidth = 1.5, linestyle = :dash)
            lines!(ax,
                [obs_xlo, obs_xhi, obs_xhi, obs_xlo, obs_xlo],
                [obs_ylo, obs_ylo, obs_yhi, obs_yhi, obs_ylo];
                color = :yellow, linewidth = 2)
            return nothing
        end

        el(dofs, conn) = [mean(dofs[conn[:, i]]) for i in 1:mesh_stokes.nels]
        fig = Figure(size = (1100, 900))
        for (row, col, title, values, colormap) in (
                (1, 1, "Forward Vx", el(vx_fwd, el2nP_cpu), :vik),
                (1, 3, "Forward Vy", el(vy_fwd, el2nP_cpu), :vik),
                (2, 1, "Forward P",  el(P_fwd,  DoFsP_cpu), :glasgow),
                (2, 3, "dJ/dG sensitivity (per area)", G_sensitivity ./ element_area, :vik),
            )
            limits = extrema(values)
            ax = Axis(fig[row, col]; aspect = DataAspect(), title, xlabel = "x", ylabel = "y")
            poly!(ax, polys; color = values, colormap, colorrange = limits, strokewidth = 0)
            Colorbar(fig[row, col + 1]; colormap, limits, width = 15, tellheight = false)
            draw_overlays!(ax)
        end

        ax = Axis(fig[3, 1:4]; xlabel = "iteration", ylabel = "log10 residual",
            title = "Residual evolution")
        logres(v) = log10.(max.(v, eps()))
        if !isempty(solve_stats.history)
            it = Float64[h.iter for h in solve_stats.history]
            lines!(ax, it, logres([h.err_v for h in solve_stats.history]); label = "forward V")
            lines!(ax, it, logres([h.err_P for h in solve_stats.history]); label = "forward P")
        end
        if !isempty(adjoint_stats.history)
            it = Float64[h.iter for h in adjoint_stats.history]
            lines!(ax, it, logres([h.err_v for h in adjoint_stats.history]); label = "adjoint V", linestyle = :dash)
            lines!(ax, it, logres([h.err_P for h in adjoint_stats.history]); label = "adjoint P", linestyle = :dash)
        end
        (isempty(solve_stats.history) && isempty(adjoint_stats.history)) || axislegend(ax; position = :rt)
        display(fig)
    end

    return (;
        dr, mesh_stokes, solve_stats, adjoint_stats, λvx, λvy, λP,
        objective_vy, J0, G_sensitivity, G_gradient_by_phase, fd_gradient_by_phase,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
