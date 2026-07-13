using Test
using FEMTools
using StaticArrays
using LinearAlgebra: dot
using KernelAbstractions: CPU
using DomainSets
using DomainSets: ×

# Small T7/P1-disc buoyancy problem on the unit square: a denser rectangular
# block (phase 2) sinks through a lighter matrix (phase 1) under free-slip walls.
# The mesh, phases, boundary nodes, and objective are fixed; only the phase-2
# density varies, so the same discrete objective can be differentiated by the
# adjoint and by finite differences.
function _adjoint_gradient_case()
    backend = CPU()
    wg = 1
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh_v = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (3, 3))
    mesh = MixedMesh(mesh_v, element_P)
    cache = MixedMeshCache(backend, wg, mesh, element_v, element_P)

    coords = mesh.coords
    el2nP = mesh.el2nP
    centroid(i) = sum(coords[el2nP[a, i]] for a in 1:3) / 3
    cell_phase = [centroid(i)[2] < 0.5 && abs(centroid(i)[1] - 0.5) < 0.34 ? 2 : 1
                  for i in 1:mesh.nels]
    phases = reshape(cell_phase, 1, :)

    Γ = mesh_v.Γnodes
    tol = 1.0e-9
    on_x_wall(c) = abs(c[1]) ≤ tol || abs(c[1] - 1.0) ≤ tol
    on_y_wall(c) = abs(c[2]) ≤ tol || abs(c[2] - 1.0) ≤ tol
    vx_nodes = Int32[n for n in Γ if on_x_wall(coords[n])]
    vy_nodes = Int32[n for n in Γ if on_y_wall(coords[n])]
    bcx = zeros(length(vx_nodes))
    bcy = zeros(length(vy_nodes))

    # Objective J(vᵧ) = -Σ_obs vᵧ over the upper part of the domain, so its only
    # nonzero derivative is ∂J/∂vᵧ = -1 on the observation nodes.
    obs = Int32[n for n in 1:mesh.nnodes if coords[n][2] > 0.6]

    return (; backend, wg, element_v, element_P, mesh, cache, phases,
        Γ, vx_nodes, vy_nodes, bcx, bcy, obs)
end

# Solve the forward Stokes problem and its adjoint for the given material, then
# compare the discrete adjoint gradient dJ/dρ₂ = λᵀ ∂R/∂ρ₂ against a central
# finite difference of the same discrete objective. `∂R/∂ρ₂` is exact from a
# residual difference (the momentum residual is linear in density). Returns the
# two gradients and the forward/adjoint convergence flags.
function _density_gradient_check(case, η, ηb, K, G; Δt = 1.0, g = SVector(0.0, -1.0))
    (; backend, wg, element_v, element_P, mesh, cache, phases,
        Γ, vx_nodes, vy_nodes, bcx, bcy, obs) = case
    α = (0.0, 0.0)
    Tref = 0.0
    nq = length(element_v.integration_points.ω)

    objective(vy) = -sum(vy[obs])

    function solve_forward(ρ2)
        dr = StokesDR(backend, mesh.nnodes, mesh.nnodesP, η, ηb, α;
            ρ0 = (1.0, ρ2), K, g, Tref,
            CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.7, stress_size = (nq, mesh.nels))
        γP = zeros(Float64, mesh.nnodesP)
        FEMTools.assemble_viscosity_weighted_pressure_scaling!(
            γP, dr, mesh, cache.geo_P, element_v, element_P, 20.0, Δt, backend, wg;
            phases_v = phases, η)
        τ_old = ntuple(_ -> zeros(Float64, nq, mesh.nels), 3)
        stats = solve_stokes_dyrel!(
            dr, mesh, cache.geo_v, cache.geo_P, element_v, element_P,
            phases, phases, τ_old, nothing, G, Δt, γP,
            Γ, bcx, bcy, backend, wg;
            ncheck = 100, ϵ_tol = 1.0e-10, iterMax = 200_000, total_iterMax = 200_000,
            rel_drop0 = 0.1, verbose = false, verbose_inner = false, vx_nodes, vy_nodes)
        return dr, γP, τ_old, stats
    end

    ρ2 = 2.0
    dr, γP, τ_old, fwd = solve_forward(ρ2)

    objective_vx = zeros(mesh.nnodes)
    objective_vy = zeros(mesh.nnodes)
    objective_vy[obs] .= -1.0
    λvx = zeros(mesh.nnodes)
    λvy = zeros(mesh.nnodes)
    λP = zeros(mesh.nnodesP)
    adj = solve_stokes_adjoint_dyrel!(
        dr, mesh, cache.geo_v, cache.geo_P, element_v, element_P,
        phases, phases, τ_old, nothing, G, Δt, γP,
        objective_vx, objective_vy, λvx, λvy, λP, backend, wg;
        vx_nodes, vy_nodes, ncheck = 100, adjoint_tol = 1.0e-10, rel_drop = 0.1,
        iterMax = 200_000, total_iterMax = 200_000, max_ph_iterations = 200,
        verbose = false, verbose_inner = false)

    function residual(ρ0_vec)
        Rx = zeros(mesh.nnodes)
        Ry = zeros(mesh.nnodes)
        FEMTools.assemble_momentum_residual_matrices_atomix!(
            Rx, Ry, dr.vx, dr.vy, dr.P, dr.T, nothing,
            mesh.el2n, mesh.DoFsP, cache.geo_v, mesh.nels, element_v, element_P,
            phases, τ_old, nothing, nothing, η, G, α, ρ0_vec, K, g, Tref, Δt, backend, wg)
        return Rx, Ry
    end
    Rx1, Ry1 = residual((0.0, 1.0))
    Rx0, Ry0 = residual((0.0, 0.0))
    ∂R∂ρ2_x = Rx1 .- Rx0
    ∂R∂ρ2_y = Ry1 .- Ry0

    # dJ/dm = +λᵀ ∂R/∂m (with λ solving (∂R/∂u)ᵀλ = -∂J/∂u).
    grad_adjoint = dot(λvx, ∂R∂ρ2_x) + dot(λvy, ∂R∂ρ2_y)

    δ = 1.0e-4
    dr_plus, = solve_forward(ρ2 + δ)
    dr_minus, = solve_forward(ρ2 - δ)
    grad_fd = (objective(Array(dr_plus.vy)) - objective(Array(dr_minus.vy))) / (2δ)

    return (; grad_adjoint, grad_fd,
        fwd_converged = fwd.converged, adj_converged = adj.converged, λvy)
end

@testset "solve_stokes_adjoint_dyrel! — gradient matches finite differences" begin
    # Incompressible viscous limit (K = ηb = Inf): the elastic pressure
    # self-coupling of the adjoint is exactly zero here.
    r = _density_gradient_check(_adjoint_gradient_case(), (1.0, 1.0), (Inf, Inf), (Inf, Inf), (Inf, Inf))
    @test r.fwd_converged
    @test r.adj_converged
    @test any(!iszero, r.λvy)          # nontrivial adjoint field
    @test r.grad_adjoint ≈ r.grad_fd rtol = 1.0e-4
end

@testset "solve_stokes_adjoint_dyrel! — compressible viscoelastic gradient" begin
    # Finite bulk modulus (ηb) makes the pressure residual store pressure
    # elastically through -(P-P0)/(ηb·Δt), so the adjoint must carry the
    # self-coupling (∂RP/∂P)ᵀλP (both directly and through the augmented Pnum
    # chain). Omitting it biases every gradient by O(1/(ηb·Δt)); this case fails
    # by ~10% without those terms and matches finite differences with them.
    r = _density_gradient_check(_adjoint_gradient_case(), (1.0, 1.0), (4.0, 4.0), (4.0, 4.0), (1.0, 0.5))
    @test r.fwd_converged
    @test r.adj_converged
    @test r.grad_adjoint ≈ r.grad_fd rtol = 1.0e-4
end
