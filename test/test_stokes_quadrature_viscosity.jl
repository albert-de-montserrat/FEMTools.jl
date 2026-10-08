using Test, FEMTools, KernelAbstractions, DomainSets, StaticArrays, LinearAlgebra
using DomainSets: ×

@testset "quadrature viscosity and assembled body force" begin
    ev = ReferenceElement(QuadraticElement{2, 7, Float64})
    ep = ReferenceElement(LinearElement{2, 3, Float64})
    mesh = MixedMesh(Mesh(CPU(), (0.0 .. 1.0) × (0.0 .. 1.0), ev, (2, 2)), ep)
    nq = length(ev.integration_points.ω)
    material = StokesMaterial(; η = (2.0,), ηb = (Inf,), G = (Inf,), α = (0.0,), ρ0 = (0.0,))
    dr = StokesDR(CPU(), mesh.nnodes, mesh.nnodesP, material; stress_size = (nq, mesh.nels))
    Nq, Np, gradients = shape_function_values(ev), shape_function_values(ep, ev.integration_points), shape_function_gradients(ev)
    coords = Array(mesh.coords)
    copyto!(dr.v.x, [x[1]^2 for x in coords])
    copyto!(dr.v.y, [x[1] * x[2] for x in coords])
    function residual(η)
        r = (zeros(mesh.nnodes), zeros(mesh.nnodes))
        FEMTools.assemble_momentum_residual_kernel!(
            r, FEMTools.velocity(dr), dr.P, dr.T, nothing, mesh.el2n, mesh.DoFsP,
            mesh.geometry.geo_v, mesh.nels, dr.phases_v, nothing, nothing, nothing,
            η, dr.G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, 1.0,
            Nq, Np, gradients, Val(7), Val(3), 32)
        return r
    end
    constant = fill(2.0, nq, mesh.nels)
    @test all(a ≈ b for (a, b) in zip(residual(constant), residual(dr.η)))
    # A spatially varying field cannot be replaced by one element maximum.
    varying = copy(constant)
    varying[1, :] .= 4.0
    @test !isapprox(residual(varying)[1], residual(constant)[1])
    γ = zeros(mesh.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(γ, dr, mesh, 10.0, 1.0; η = constant)
    @test γ ≈ fill(10.0, mesh.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(γ, dr, mesh, 10.0, 1.0; η = varying)
    @test maximum(γ) > 10.0

    boundary = FEMTools.rectangle_boundary_nodes(coords, 0.0, 1.0, 0.0, 1.0)
    bc = ntuple(_ -> DirichletBoundaryCondition(boundary, zeros(length(boundary))), 2)
    # Equivalence with the existing gravitational assembly checks load sign and
    # its use in both the outer and inner residual loops.
    gravity_material = StokesMaterial(; η = (2.0,), ηb = (Inf,), G = (Inf,), α = (0.0,), g = (0.0, 1.0))
    gravity_dr = StokesDR(CPU(), mesh.nnodes, mesh.nnodesP, gravity_material; stress_size = (nq, mesh.nels))
    gravity_residual = (zeros(mesh.nnodes), zeros(mesh.nnodes))
    FEMTools.assemble_momentum_residual_kernel!(
        gravity_residual, FEMTools.velocity(gravity_dr), gravity_dr.P, gravity_dr.T,
        nothing, mesh.el2n, mesh.DoFsP, mesh.geometry.geo_v, mesh.nels,
        gravity_dr.phases_v, nothing, nothing, nothing, gravity_dr.η, gravity_dr.G,
        gravity_dr.α, gravity_dr.ρ0, gravity_dr.K, gravity_dr.g, gravity_dr.Tref,
        1.0, Nq, Np, gradients, Val(7), Val(3), 32)
    load = map(r -> -r, gravity_residual)
    fill!(dr.v.x, 0); fill!(dr.v.y, 0)
    assemble_viscosity_weighted_pressure_scaling!(γ, dr, mesh, 50.0, 1.0; η = constant)
    stats = solve_stokes_dyrel!(dr, mesh, bc, 1.0, γ; viscosity = constant,
        body_force = load, ncheck = 10, ϵ_tol = 1e-11, verbose = false)
    @test stats.converged
    # Hydrostatic pressure has the independent analytical solution P=y+C.
    pressure_coords = coords[vec(Array(mesh.el2nP))]
    pressure = dr.P .- sum(dr.P .* dr.M_P) / sum(dr.M_P)
    expected = [x[2] - 0.5 for x in pressure_coords]
    @test pressure ≈ expected atol = 1e-7
    @test maximum(abs, dr.v.x) < 1e-7
    @test maximum(abs, dr.v.y) < 1e-7
    @test_throws DimensionMismatch solve_stokes_dyrel!(dr, mesh, bc, 1.0, γ; viscosity = zeros(1, 1))
    @test_throws ArgumentError solve_stokes_dyrel!(dr, mesh, bc, 1.0, γ; viscosity = -constant)
    @test_throws ArgumentError solve_stokes_dyrel!(dr, mesh, bc, 1.0, γ; body_force = (fill(NaN, mesh.nnodes), zeros(mesh.nnodes)))
    @test_throws DimensionMismatch solve_stokes_dyrel!(dr, mesh, bc, 1.0, γ; body_force = (zeros(1), zeros(1)))

    # The state-owned scale reproduces the caller-prepared one exactly, and a
    # separate scaling viscosity changes only the scale, not the momentum solve.
    manual = (copy(dr.v.x), copy(dr.v.y), copy(dr.P))
    fill!(dr.v.x, 0); fill!(dr.v.y, 0); fill!(dr.P, 0)
    owned = solve!(dr, mesh, bc; dt = 1.0, pressure_factor = 50.0,
        viscosity = constant, body_force = load, check_interval = 10, tolerance = 1e-11, verbose = false)
    @test owned.converged && owned.iterations == owned.iter && owned.residual == owned.err_abs
    @test dr.γP == γ
    @test (dr.v.x, dr.v.y, dr.P) == manual
    solve!(dr, mesh, bc; dt = 1.0, pressure_factor = 50.0, viscosity = constant,
        scaling_viscosity = varying, body_force = load, check_interval = 10, tolerance = 1e-11, verbose = false)
    @test maximum(dr.γP) > maximum(γ)

    # A budget too small to converge throws by default and reports failure on request.
    fill!(dr.v.x, 0); fill!(dr.v.y, 0); fill!(dr.P, 0)
    @test_throws ErrorException solve!(dr, mesh, bc; dt = 1.0, viscosity = constant,
        body_force = load, check_interval = 1, max_iterations = 2, verbose = false)
    failed = solve!(dr, mesh, bc; dt = 1.0, viscosity = constant, body_force = load,
        check_interval = 1, max_iterations = 2, verbose = false, throw_on_failure = false)
    @test !failed.converged
end

@testset "Q2/P1 pressure topology and hydrostatics" begin
    ev = ReferenceElement(QuadraticElement{2, 9, Float64})
    ep = ReferenceElement(LinearElement{2, 3, Float64})
    mesh = MixedMesh(Mesh(CPU(), (0.0 .. 1.0) × (0.0 .. 1.0), ev, (2, 2)), ep)
    @test mesh.el2nP == mesh.el2n[[9, 6, 7], :]
    @test mesh.nnodesP == 3 * mesh.nels
    material = StokesMaterial(; η = (2.0,), ηb = (Inf,), G = (Inf,), α = (0.0,), g = (0.0, 1.0))
    dr = StokesDR(CPU(), mesh.nnodes, mesh.nnodesP, material)
    γ = zeros(mesh.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(γ, dr, mesh, 50.0, 1.0)
    @test all(>(0), dr.M_P)
    @test γ ≈ fill(50.0, mesh.nnodesP)
    @test sum(mesh.geometry.geo_P[1]) ≈ 0.25
    coords = Array(mesh.coords)
    nodes = FEMTools.rectangle_boundary_nodes(coords, 0.0, 1.0, 0.0, 1.0)
    bc = ntuple(_ -> DirichletBoundaryCondition(nodes, zeros(length(nodes))), 2)
    stats = solve_stokes_dyrel!(dr, mesh, bc, 1.0, γ; ncheck = 10, ϵ_tol = 1e-11, collect_history = true, verbose = false)
    @test stats.converged
    @test first(stats.history).iter == 0
    @test last(stats.history).iter == stats.iter
    @test last(stats.history).err_P ≈ stats.err_P
    @test all(h -> length(h.err_v_components) == 2 && maximum(h.err_v_components) ≈ h.err_v, stats.history)
    pressure_coords = coords[vec(Array(mesh.el2nP))]
    expected = [x[2] for x in pressure_coords]
    difference = dr.P .- expected
    @test maximum(difference) - minimum(difference) < 1e-7
    @test norm(dr.v.x) + norm(dr.v.y) < 1e-7
end
