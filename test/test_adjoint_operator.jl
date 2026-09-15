using Test
using FEMTools
using StaticArrays
using LinearAlgebra
using KernelAbstractions: CPU
using DomainSets
using DomainSets: ×

# The frozen adjoint operator must reproduce, to round-off, the residual the
# Enzyme transpose assemblers produce at the same forward state. The two paths
# share no code: one differentiates the nonlinear residual on every call, the
# other applies element blocks built once by ForwardDiff.
@testset "frozen adjoint operator matches the Enzyme transpose" begin
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

    η = (1.0, 1.0)
    ηb = (Inf, Inf)
    α = (0.0, 0.0)
    K = (Inf, Inf)
    G = (Inf, Inf)
    g = SVector(0.0, -1.0)
    Tref = 0.0
    Δt = 1.0
    nq = length(element_v.integration_points.ω)

    dr = StokesDR(backend, mesh.nnodes, mesh.nnodesP, η, ηb, α;
        ρ0 = (1.0, 2.0), K, g, Tref,
        CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.7, stress_size = (nq, mesh.nels))
    γP = zeros(Float64, mesh.nnodesP)
    FEMTools.assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, cache.geo_P, element_v, element_P, 20.0, Δt, backend, wg;
        phases_v = phases, η)
    τ_old = ntuple(_ -> zeros(Float64, nq, mesh.nels), 3)
    fwd = solve_stokes_dyrel!(
        dr, mesh, cache.geo_v, cache.geo_P, element_v, element_P,
        phases, phases, τ_old, nothing, G, Δt, γP,
        Γ, bcx, bcy, backend, wg;
        ncheck = 100, ϵ_tol = 1.0e-10, iterMax = 200_000, total_iterMax = 200_000,
        rel_drop0 = 0.1, verbose = false, verbose_inner = false, vx_nodes, vy_nodes,
        measure_λmax = true)
    @test fwd.converged
    @test fwd.jacobian_assemblies == 1
    @test fwd.λmax_iterations > 0
    @test fwd.λmax < fwd.λmax_gershgorin

    # A seed unrelated to any solve, so agreement cannot come from both paths
    # returning something near zero.
    rng_state = 12345
    nextrand() = (rng_state = (1103515245 * rng_state + 12345) % 2147483648;
        rng_state / 2147483648 - 0.5)
    λvx = [nextrand() for _ in 1:(mesh.nnodes)]
    λvy = [nextrand() for _ in 1:(mesh.nnodes)]
    λP = [nextrand() for _ in 1:(mesh.nnodesP)]

    M_P = dr.M_P

    # Path 1: the Enzyme transpose assemblers, as the solver drives them.
    dvx = zeros(mesh.nnodes)
    dvy = zeros(mesh.nnodes)
    dP = zeros(mesh.nnodesP)
    dPnum = zeros(mesh.nnodesP)
    Pnum = zeros(mesh.nnodesP)
    Rv_x_buf = zeros(mesh.nnodes)
    Rv_y_buf = zeros(mesh.nnodes)
    FEMTools.assemble_momentum_residual_matrices_atomix_adj!(
        Rv_x_buf, copy(λvx), Rv_y_buf, copy(λvy),
        dr.v.x, dvx, dr.v.y, dvy, dr.P, dP, dr.T, Pnum, dPnum,
        mesh, cache.geo_v, element_v, element_P,
        phases, τ_old, nothing,
        dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt, wg)
    ResλP_enzyme = copy(dP)
    seed_RP = @. λP + γP * dPnum / M_P
    dP_scratch = zeros(mesh.nnodesP)
    FEMTools.assemble_pressure_residual_matrices_atomix_adj!(
        dr, seed_RP, dvx, dvy, dP_scratch,
        mesh, cache.geo_v, cache.geo_P, element_v, element_P,
        phases, Δt, wg)
    ResλVx_enzyme = copy(dvx)
    ResλVy_enzyme = copy(dvy)

    # Path 2: the frozen element blocks.
    op = FEMTools.assemble_adjoint_operator(
        dr, mesh, cache.geo_v, cache.geo_P, element_v, element_P,
        phases, phases, τ_old, nothing, G, Δt, γP, backend, wg)
    velocity_op = FEMTools.assemble_velocity_operator(
        dr, mesh, cache.geo_v, cache.geo_P, element_v, element_P,
        phases, phases, τ_old, nothing, G, Δt, γP, backend, wg)
    @test velocity_op.A ≈ op.A
    λcold, ncold, power_x, power_y = FEMTools.estimate_velocity_λmax(
        velocity_op, mesh, element_v, dr.PC_v.x, dr.PC_v.y,
        vx_nodes, vy_nodes, backend, wg)
    λwarm, nwarm, _, _ = FEMTools.estimate_velocity_λmax(
        velocity_op, mesh, element_v, dr.PC_v.x, dr.PC_v.y,
        vx_nodes, vy_nodes, backend, wg; x = power_x, y = power_y)
    @test λwarm ≈ λcold rtol = 1.0e-2
    @test nwarm < ncold
    rowsum_vx_op = copy(dr.∂Rv∂v.x)
    rowsum_vy_op = copy(dr.∂Rv∂v.y)
    PC_vx_op = copy(dr.PC_v.x)
    PC_vy_op = copy(dr.PC_v.y)
    ResλVx_op = zeros(mesh.nnodes)
    ResλVy_op = zeros(mesh.nnodes)
    ResλP_op = zeros(mesh.nnodesP)
    FEMTools.apply_adjoint_operator!(
        ResλVx_op, ResλVy_op, ResλP_op, op, λvx, λvy, λP,
        mesh, element_v, element_P, backend, wg)

    scale_v = max(maximum(abs, ResλVx_enzyme), maximum(abs, ResλVy_enzyme))
    scale_P = maximum(abs, ResλP_enzyme)
    @test scale_v > 0                                  # the seed produced a real residual
    @test maximum(abs, ResλVx_op .- ResλVx_enzyme) / scale_v < 1.0e-12
    @test maximum(abs, ResλVy_op .- ResλVy_enzyme) / scale_v < 1.0e-12
    @test maximum(abs, ResλP_op .- ResλP_enzyme) / scale_P < 1.0e-12

    # The measured λmax must reproduce the spectral radius of the symmetrically
    # Jacobi-scaled free-velocity block, not merely be smaller than Gershgorin.
    FEMTools.assemble_augmented_momentum_jacobian_matrices_atomix!(
        dr.∂Rv∂v.x, dr.PC_v.x, dr.∂Rv∂v.y, dr.PC_v.y,
        dr.v.x, dr.v.y, dr.P, dr.P0, dr.T, dr.T0,
        mesh.el2n, mesh.DoFsP, cache.geo_v, cache.geo_P, mesh.nels,
        element_v, element_P, phases, phases,
        dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
        dr.ηb, Δt, γP, dr.M_P, backend, wg; τ_old)
    @test rowsum_vx_op ≈ dr.∂Rv∂v.x
    @test rowsum_vy_op ≈ dr.∂Rv∂v.y
    @test PC_vx_op ≈ dr.PC_v.x
    @test PC_vy_op ≈ dr.PC_v.y
    λmax, λmax_iterations = FEMTools.estimate_adjoint_λmax(
        op, mesh, element_v, element_P, dr.PC_v.x, dr.PC_v.y,
        vx_nodes, vy_nodes, backend, wg)
    free_vx = setdiff(eachindex(dr.v.x), vx_nodes)
    free_vy = setdiff(eachindex(dr.v.y), vy_nodes)
    free = vcat(free_vx, length(dr.v.x) .+ free_vy)
    PC = vcat(dr.PC_v.x, dr.PC_v.y)
    A = zeros(length(free), length(free))
    basis_vx = zeros(mesh.nnodes)
    basis_vy = zeros(mesh.nnodes)
    out_vx = similar(basis_vx)
    out_vy = similar(basis_vy)
    for (j, dof) in enumerate(free)
        fill!(basis_vx, 0)
        fill!(basis_vy, 0)
        dof ≤ mesh.nnodes ? (basis_vx[dof] = inv(sqrt(PC[dof]))) :
            (basis_vy[dof - mesh.nnodes] = inv(sqrt(PC[dof])))
        FEMTools.apply_adjoint_operator!(
            out_vx, out_vy, ResλP_op, op, basis_vx, basis_vy, zero(λP),
            mesh, element_v, element_P, backend, wg)
        A[:, j] .= vcat(out_vx, out_vy)[free] ./ sqrt.(PC[free])
    end
    λmax_exact = maximum(abs, eigvals(Symmetric(A)))
    @test λmax ≈ λmax_exact rtol = 5.0e-3
    @test λmax_iterations < 100
end
