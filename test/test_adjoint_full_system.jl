using Test
using FEMTools
using StaticArrays
using LinearAlgebra
using KernelAbstractions: CPU
using DomainSets
using DomainSets: ×

# GAP-25: the frozen adjoint operator must be the transpose of the Jacobian of the residual system
# it claims to linearise — not of a simpler one that happens to agree in the incompressible gauge.
#
# `test_adjoint_operator.jl` checks the operator against the Enzyme transpose, and
# `test_stokes_adjoint_api.jl` checks the gradient against finite differences, but both run with an
# infinite bulk viscosity and an infinite bulk modulus. Two blocks are then identically zero or
# identically equal to blocks that are already stored, so a missing term cannot show up. This file
# builds the Jacobian itself by central differences on a 2×2 element mesh and compares `Jᵀa` term by
# term, with storage, elasticity, pressure-dependent density and the Powell-Hestenes augmentation
# switched on in every combination.
#
# The two identities that must hold:
#   Jᵀa  matches the operator's apply, row block by row block
#   aᵀ(Jb) == (Jᵀa)ᵀb  for arbitrary a and b, which is the definition of a transpose
@testset "the adjoint operator transposes the compressible Jacobian" begin
    backend = CPU()
    wg = 1
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh_v = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (2, 2))
    mesh = MixedMesh(mesh_v, element_P; workgroup = wg)
    (; geo_v, geo_P) = mesh.geometry
    nels, nn, nnP = mesh.nels, mesh.nnodes, mesh.nnodesP
    NU = 2nn + nnP
    phases = ones(Int32, 1, nels)
    nq = length(element_v.integration_points.ω)
    g, Tref, Δt = SVector(0.0, -1.0), 0.0, 1.0
    η, α = (1.0, 1.0), (0.0, 0.0)

    # A fixed pseudo-random state, so a failure is reproducible and not a lucky probe.
    seed = 12345
    nextrand() = (seed = (1103515245 * seed + 12345) % 2147483648; seed / 2147483648 - 0.5)
    vx0 = [0.1nextrand() for _ in 1:nn]
    vy0 = [0.1nextrand() for _ in 1:nn]
    Pstate = [nextrand() for _ in 1:nnP]
    Pold = [0.2nextrand() for _ in 1:nnP]
    Qsrc = [0.3 * sin(3.0i) for i in 1:nnP]
    avec = [nextrand() for _ in 1:NU]
    bvec = [nextrand() for _ in 1:NU]
    u0 = vcat(vx0, vy0, Pstate)

    # The central difference is taken on the residual the operator describes, which for a nonzero γP
    # is the augmented one: the pressure residual is assembled first and folded into the momentum
    # residual as Pnum, exactly as the forward solve does.
    function full_residual(u, dr, G, γP, MP, τ_old)
        vx, vy, P = u[1:nn], u[(nn + 1):(2nn)], u[(2nn + 1):NU]
        Rvx, Rvy, RP = zeros(nn), zeros(nn), zeros(nnP)
        FEMTools.assemble_pressure_residual_matrices_atomix!(
            RP, (vx, vy), P, dr.P0, dr.T, dr.T0, Qsrc, mesh.el2n, mesh.DoFsP, geo_v, geo_P, nels,
            element_v, element_P, phases, dr.α, dr.ηb, Δt, backend, wg,
        )
        Pnum = @. γP * RP / MP
        FEMTools.assemble_momentum_residual_matrices_atomix!(
            (Rvx, Rvy), (vx, vy), P, dr.T, Pnum, mesh.el2n, mesh.DoFsP, geo_v, nels,
            element_v, element_P, phases, τ_old, nothing, nothing,
            dr.η, G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt, backend, wg,
        )
        return vcat(Rvx, Rvy, RP)
    end

    function setup(K, G, ηb, augmented)
        dr = StokesDR(
            backend, nn, nnP, η, ηb, α; ρ0 = (1.0, 2.0), K, g, Tref,
            CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.7, stress_size = (nq, nels),
        )
        τ_old = ntuple(_ -> zeros(Float64, nq, nels), 3)
        γP = zeros(Float64, nnP)
        FEMTools.assemble_viscosity_weighted_pressure_scaling!(
            γP, dr, mesh, geo_P, element_v, element_P, 20.0, Δt, backend, wg; phases_v = phases, η,
        )
        augmented || (γP .= 0.0)
        dr.v.x .= vx0
        dr.v.y .= vy0
        dr.P .= Pstate
        dr.P0 .= Pold
        dr.T .= 0.0
        dr.T0 .= 0.0
        return dr, γP, copy(dr.M_P), τ_old
    end

    cases = (
        (; name = "incompressible", K = (Inf, Inf), G = (Inf, Inf), ηb = (Inf, Inf)),
        (; name = "finite storage", K = (Inf, Inf), G = (3.0, 3.0), ηb = (5.0, 5.0)),
        (; name = "compressible", K = (10.0, 10.0), G = (3.0, 3.0), ηb = (5.0, 5.0)),
    )

    for case in cases, augmented in (false, true)
        dr, γP, MP, τ_old = setup(case.K, case.G, case.ηb, augmented)
        residual(u) = full_residual(u, dr, case.G, γP, MP, τ_old)

        J = zeros(NU, NU)
        h = 1.0e-6
        for j in 1:NU
            up, um = copy(u0), copy(u0)
            up[j] += h
            um[j] -= h
            J[:, j] = (residual(up) - residual(um)) ./ (2h)
        end

        op = FEMTools.assemble_adjoint_operator(
            dr, mesh, geo_v, geo_P, element_v, element_P, phases, phases, τ_old, nothing,
            case.G, Δt, γP, backend, wg,
        )

        # The storage block exists exactly when the bulk viscosity is finite, and the
        # finite-difference Jacobian agrees about which case that is.
        stores = any(isfinite, case.ηb)
        @test (op.D === nothing) == !stores
        @test (op.W === nothing) == !stores
        @test (norm(J[(2nn + 1):NU, (2nn + 1):NU]) > 0) == stores

        λvx, λvy, λP = avec[1:nn], avec[(nn + 1):(2nn)], avec[(2nn + 1):NU]
        dvx, dvy, dP = zeros(nn), zeros(nn), zeros(nnP)
        FEMTools.apply_adjoint_operator!(
            dvx, dvy, dP, op, λvx, λvy, λP, mesh, element_v, element_P, backend, wg,
        )
        Jt_op = vcat(dvx, dvy, dP)
        Jt_fd = transpose(J) * avec

        # The central difference itself carries about 1e-10 of truncation and round-off on this
        # residual, so the tolerance is set by the oracle rather than by the operator.
        tol = 1.0e-7
        for (what, rows) in (("velocity", 1:(2nn)), ("pressure", (2nn + 1):NU))
            o, f = Jt_op[rows], Jt_fd[rows]
            relative = norm(o - f) / norm(f)
            @test relative < tol
            relative < tol || @info "$(case.name), augmented = $augmented: $what rows" relative
        end
        lhs, rhs = dot(avec, J * bvec), dot(Jt_op, bvec)
        @test abs(lhs - rhs) / abs(lhs) < tol
    end
end

# The two storage layouts are enabled by a condition on the model, not by a wish, and each must
# refuse the configurations it does not cover rather than return a gradient from a layout that no
# longer holds.
@testset "the adjoint operators refuse an asymmetric tangent" begin
    backend = CPU()
    wg = 1
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh_v = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (2, 2))
    mesh = MixedMesh(mesh_v, element_P; workgroup = wg)
    (; geo_v, geo_P) = mesh.geometry
    nels, nn, nnP = mesh.nels, mesh.nnodes, mesh.nnodesP
    phases = ones(Int32, 1, nels)
    nq = length(element_v.integration_points.ω)
    g, Tref, Δt = SVector(0.0, -1.0), 0.0, 1.0
    η, α = (1.0, 1.0), (0.0, 0.0)

    @test FEMTools._symmetric_tangent(nothing, (Inf, Inf))
    @test !FEMTools._symmetric_tangent(nothing, (10.0, Inf))

    function operator(K, G)
        dr = StokesDR(
            backend, nn, nnP, η, (Inf, Inf), α; ρ0 = (1.0, 2.0), K, g, Tref,
            CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.7, stress_size = (nq, nels),
        )
        τ_old = ntuple(_ -> zeros(Float64, nq, nels), 3)
        γP = zeros(Float64, nnP)
        # M_P is the mass scaling Pnum is divided by; leaving it unfilled makes the augmented
        # residual NaN even at γP = 0, so it is assembled before the blocks are.
        FEMTools.assemble_viscosity_weighted_pressure_scaling!(
            γP, dr, mesh, geo_P, element_v, element_P, 20.0, Δt, backend, wg; phases_v = phases, η,
        )
        γP .= 0.0
        return FEMTools.assemble_adjoint_operator(
            dr, mesh, geo_v, geo_P, element_v, element_P, phases, phases, τ_old, nothing,
            G, Δt, γP, backend, wg,
        ), dr, τ_old, γP
    end

    # Incompressible: the packing is used, and C is Bᵀ so it is not stored at all.
    op, dr, τ_old, γP = operator((Inf, Inf), (Inf, Inf))
    @test op.C === nothing
    @test op.A[1] isa SVector

    # Compressible: ρ(P) breaks C == Bᵀ, so both blocks must be kept whole rather than asserted away.
    op, dr, τ_old, γP = operator((10.0, 10.0), (3.0, 3.0))
    @test op.C !== nothing
    @test op.A[1] isa SMatrix

    # The matrix-free operator has no stored blocks to fall back on, so it refuses instead.
    @test_throws ArgumentError FEMTools.matrix_free_adjoint_operator(
        dr, mesh, geo_v, geo_P, element_v, element_P, phases, phases, τ_old, nothing,
        (3.0, 3.0), Δt, γP, backend, wg,
    )
end


# The frozen adjoint operator on the 3-D T11/P1-discontinuous mixed mesh must be the transpose of
# the Jacobian of the augmented residual the forward solve drives to zero. Checked through the
# transpose identity aᵀ(J b) == (Jᵀa)ᵀ b, with J b from central differences of the residual along
# b, once per block row and column: compressible viscoelastic, and Drucker-Prager plastic with a
# nonzero fluid pressure, which the yield model subtracts from the pressure.
@testset "3-D adjoint operator and solver on tetrahedra" begin
    include(joinpath(@__DIR__, "tet11_box_mesh.jl"))
    backend = CPU()
    wg = 1
    element_v = ReferenceElement(QuadraticElement{3, 11, Float64})
    element_P = ReferenceElement(LinearElement{3, 4, Float64})
    coords, el2n, groups = build_tet11_inclusion_mesh()
    mesh = MixedMesh(Mesh(backend, coords, el2n, element_v; workgroup = wg), element_P)
    (; geo_v, geo_P) = mesh.geometry
    nels, nn, nnP = mesh.nels, mesh.nnodes, mesh.nnodesP
    NV, NP = length(element_v), length(element_P)
    nq = length(element_v.integration_points.ω)
    phases_v = repeat(reshape(groups.phase, 1, :), NV, 1)
    phases_P = repeat(reshape(groups.phase, 1, :), NP, 1)
    Δt = 1.0
    Nq = FEMTools.quadrature_table(backend, FEMTools.shape_function_values(element_v))
    NqP = FEMTools.quadrature_table(
        backend, FEMTools.shape_function_values(element_P, element_v.integration_points),
    )
    ∂N∂ξ = FEMTools.quadrature_table(backend, FEMTools.shape_function_gradients(element_v))

    seed = 2024
    nextrand() = (seed = (1103515245 * seed + 12345) % 2147483648; seed / 2147483648 - 0.5)
    randvec(n, scale = 1.0) = [scale * nextrand() for _ in 1:n]

    K = (10.0, 5.0)
    material = StokesMaterial(;
        η = (1.0, 0.1), ηb = K, G = (3.0, 1.0), α = (0.0, 0.0), ρ0 = (1.0, 0.9), K,
        g = (0.0, 0.0, -1.0), Tref = 0.0,
    )

    function setup(Pf_scale)
        dr = StokesDR(backend, nn, nnP, material; stress_size = (nq, nels))
        foreach((vc, x) -> copyto!(vc, x), FEMTools.velocity(dr), ntuple(_ -> randvec(nn, 0.1), 3))
        copyto!(dr.P, randvec(nnP))
        copyto!(dr.P0, randvec(nnP, 0.2))
        copyto!(dr.Pf, randvec(nnP, Pf_scale))
        τ_old = ntuple(_ -> reshape(randvec(nq * nels, 0.1), nq, nels), 6)
        γP = zeros(nnP)
        assemble_viscosity_weighted_pressure_scaling!(γP, dr, mesh, 20.0, Δt; workgroup = wg, phases_v)
        return dr, τ_old, γP
    end

    # The augmented residual: the pressure residual is assembled first and folded into the
    # momentum residual as Pnum = γP·RP/M_P, exactly as the forward solve does.
    function residual(u, dr, τ_old, plastic, γP)
        v = (u[1:nn], u[(nn + 1):(2nn)], u[(2nn + 1):(3nn)])
        P = u[(3nn + 1):end]
        RP = zeros(nnP)
        FEMTools.assemble_pressure_residual_kernel!(
            RP, v, P, dr.P0, dr.T, dr.T0, nothing, mesh.el2n, mesh.DoFsP, geo_v, geo_P, nels,
            phases_P, dr.α, dr.ηb, Δt, NqP, ∂N∂ξ, Val(NV), Val(NP), wg, dr.ηb,
        )
        Pnum = γP .* RP ./ dr.M_P
        Rv = ntuple(_ -> zeros(nn), 3)
        FEMTools.assemble_momentum_residual_kernel!(
            Rv, v, P, dr.T, Pnum, mesh.el2n, mesh.DoFsP, geo_v, nels, phases_v,
            τ_old, plastic, nothing, dr.η, dr.G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
            Nq, NqP, ∂N∂ξ, Val(NV), Val(NP), wg, nothing, nothing, nothing, dr.Pf,
        )
        return vcat(Rv..., RP)
    end

    plastic = DruckerPrager(
        (deg2rad(30), deg2rad(30)), (deg2rad(10), deg2rad(10)), (0.05, 0.05), (0.5, 0.5), K,
    )
    for (name, plastic, Pf_scale) in (("viscoelastic", nothing, 0.0), ("plastic with Pf", plastic, 0.5))
        dr, τ_old, γP = setup(Pf_scale)
        op = FEMTools.assemble_adjoint_operator(
            dr, mesh, geo_v, geo_P, element_v, element_P, phases_v, phases_P, τ_old, plastic,
            dr.G, Δt, γP, backend, wg,
        )
        u0 = vcat(map(copy, FEMTools.velocity(dr))..., copy(dr.P))
        a = randvec(length(u0))
        dv = ntuple(_ -> zeros(nn), 3)
        dP = zeros(nnP)
        h = 1.0e-6
        velocity_rows, pressure_rows = 1:(3nn), (3nn + 1):length(u0)
        for (what, cols) in (("velocity", velocity_rows), ("pressure", pressure_rows))
            b = zeros(length(u0))
            b[cols] = randvec(length(cols))
            Jb = (residual(u0 + h * b, dr, τ_old, plastic, γP) - residual(u0 - h * b, dr, τ_old, plastic, γP)) / 2h
            for (block, rows) in (("velocity", velocity_rows), ("pressure", pressure_rows))
                lhs = dot(a[rows], Jb[rows])
                a_block = zeros(length(u0))
                a_block[rows] = a[rows]
                FEMTools.apply_adjoint_operator!(
                    dv, dP, op,
                    (a_block[1:nn], a_block[(nn + 1):(2nn)], a_block[(2nn + 1):(3nn)]),
                    a_block[(3nn + 1):end], mesh, element_v, element_P, backend, wg,
                )
                rhs = dot(vcat(dv..., dP), b)
                relative = abs(lhs - rhs) / max(abs(lhs), eps())
                @test relative < 1.0e-6
                relative < 1.0e-6 || @info "$name: $block rows × $what columns" lhs rhs relative
            end
        end

        # The solver iterates to the adjoint of the same operator: Jᵀλ = −∂J/∂u on the free rows.
        fixed = (groups.left ∪ groups.right, groups.front ∪ groups.back, groups.bottom)
        v_nodes = map(n -> Int32.(sort!(collect(n))), fixed)
        objective = (zeros(nn), zeros(nn), [c[3] > -0.5 ? 1.0 : 0.0 for c in coords] ./ nn)
        λv = ntuple(_ -> zeros(nn), 3)
        λP = zeros(nnP)
        stats = solve_stokes_adjoint_dyrel!(
            dr, mesh, geo_v, geo_P, element_v, element_P, phases_v, phases_P, τ_old, plastic,
            dr.G, Δt, γP, objective, λv, λP, backend, wg;
            v_nodes, adjoint_tol = 1.0e-10, verbose = false,
            iterMax = 200_000, total_iterMax = 200_000, max_ph_iterations = 1000,
        )
        @test stats.converged
        FEMTools.apply_adjoint_operator!(dv, dP, op, λv, λP, mesh, element_v, element_P, backend, wg)
        residual_v = map(.+, dv, objective)
        foreach((r, nodes) -> (r[nodes] .= 0), residual_v, v_nodes)
        @test maximum(norm, residual_v) ≤ 1.0e-6 * maximum(norm, objective)
        @test norm(dP) ≤ 1.0e-6 * maximum(norm, objective)
    end

    dr, τ_old, γP = setup(0.0)
    @test_throws "two dimensions only" solve_stokes_adjoint_dyrel!(
        dr, mesh, geo_v, geo_P, element_v, element_P, phases_v, phases_P, τ_old, nothing,
        dr.G, Δt, γP, ntuple(_ -> zeros(nn), 3), ntuple(_ -> zeros(nn), 3), zeros(nnP), backend, wg;
        v_nodes = (Int32[], Int32[], Int32[]), operator = :matrix_free, verbose = false,
    )
end
