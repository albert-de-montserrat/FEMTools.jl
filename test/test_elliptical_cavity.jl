using LinearAlgebra: norm, normalize

@testset "elliptical cavity analytic solution" begin
    include(joinpath(pkgdir(FEMTools), "examples", "reykjanes", "cavity_analytic.jl"))

    G, ν, p = 3.0e10, 0.25, 1.0e7
    λ = 2G * ν / (1 - 2ν)

    # A circle is Lamé's pressurised hole, `u_r = p a² / (2G r)`.
    a = 2.0e3
    ux, uy = elliptical_cavity_displacement(3a, 0.0; a, b = a, p, G, ν)
    @test ux ≈ p * a^2 / (2G * 3a)
    @test abs(uy) < 1.0e-12 * abs(ux)

    # The compliance reduces to the circle and to the Griffith crack, `π (1 − ν) a² / G`.
    @test elliptical_cavity_area_compliance(a, a, G, ν) ≈ π * a^2 / G
    @test elliptical_cavity_area_compliance(a, 0.0, G, ν) ≈ π * (1 - ν) * a^2 / G

    for (a, b) in ((2.5e3, 0.5e3), (0.5e3, 2.5e3))
        # The closed-form compliance is the area change of the displacement field around the hole.
        n = 4000
        ΔA = sum(0:(n - 1)) do k
            θ = 2π * (k + 0.5) / n
            ux, uy = elliptical_cavity_displacement(a * cos(θ), b * sin(θ); a, b, p, G, ν)
            (ux * b * cos(θ) + uy * a * sin(θ)) * 2π / n
        end
        @test ΔA ≈ elliptical_cavity_area_compliance(a, b, G, ν) * p rtol = 1.0e-8

        # The displacement carries the load: the traction on the hole wall is the pressure, `σ·n = −p n`.
        # Strains come from second-order one-sided differences that step away from the hole.
        worst = 0.0
        for θ in range(0.05, 2π - 0.05; length = 25)
            x, y = a * cos(θ), b * sin(θ)
            n̂ = normalize([x / a^2, y / b^2])
            h = 1.0e-4 * min(a, b)
            u(x, y) = collect(elliptical_cavity_displacement(x, y; a, b, p, G, ν))
            step = sign.(n̂)
            derivative(f) = (-3f(0) + 4f(1) - f(2)) / (2h)
            ∂x = step[1] * derivative(k -> u(x + step[1] * k * h, y))
            ∂y = step[2] * derivative(k -> u(x, y + step[2] * k * h))
            εxx, εyy, εxy = ∂x[1], ∂y[2], (∂x[2] + ∂y[1]) / 2
            σ = [λ * (εxx + εyy) + 2G * εxx  2G * εxy; 2G * εxy  λ * (εxx + εyy) + 2G * εyy]
            worst = max(worst, norm(σ * n̂ + p * n̂) / p)
        end
        @test worst < 1.0e-4
    end
end

@testset "pressurised elliptical cavity benchmark" begin
    include(joinpath(pkgdir(FEMTools), "examples", "reykjanes", "elliptical_cavity_setup.jl"))

    # Coarse mesh (about 440 elements): the discretisation error is a few percent, and the
    # inclusion's shear modulus, 1e-3 of the host's, adds 0.3 %.
    result = solve_elliptical_cavity(; max_area = 4.0e6, refinement = 4)
    @test result.converged
    @test result.P_cavity ≈ result.P_reference rtol = 0.01
    @test result.ΔA ≈ result.ΔA_reference rtol = 0.03
    @test result.opening_ratio ≈ 1 rtol = 0.02
    @test result.error_L2 < 0.03
    # The pressure follows from the injected and the displaced volume through the continuity
    # equation, which fails if the source `Q` enters with the wrong sign or scale.
    @test result.P_cavity ≈ result.P_balance rtol = 1.0e-4
end

@testset "reservoir source normalisation" begin
    include(joinpath(pkgdir(FEMTools), "examples", "reykjanes", "elliptical_cavity_setup.jl"))
    include(joinpath(pkgdir(FEMTools), "examples", "reykjanes", "injection_source.jl"))
    include(joinpath(pkgdir(FEMTools), "examples", "reykjanes", "dike_functions", "dike_eigenstrain.jl"))

    a, b = 2.5e3, 0.5e3
    coords, el2n, groups = build_triangulate_t7_cavity_mesh(;
        radius = 6a, cavity_radii = (a, b), max_area = 4.0e6, refinement = 4,
    )
    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh = MixedMesh(Mesh(CPU(), coords, el2n, element_v; workgroup = 64), element_P; workgroup = 64)
    DoFsP = Array(mesh.DoFsP)
    geo_P = Array(mesh.geometry.geo_P)
    NqP = shape_function_values(element_P, element_v.integration_points)
    cavity = findall(==(2), groups.phase)
    host = findall(==(1), groups.phase)

    corner_area(iel) = begin
        p, q, r = coords[el2n[1, iel]], coords[el2n[2, iel]], coords[el2n[3, iel]]
        ((q[1] - p[1]) * (r[2] - p[2]) - (r[1] - p[1]) * (q[2] - p[2])) / 2
    end
    @test sum(sum(geo_P[iel]) for iel in cavity) ≈ sum(corner_area, cavity)

    rate = 3.7e3
    Q = uniform_pressure_source(rate, DoFsP, geo_P, cavity)
    @test pressure_source_integral(Q, DoFsP, geo_P, NqP, cavity) ≈ rate rtol = 1.0e-13
    @test pressure_source_integral(Q, DoFsP, geo_P, NqP) ≈ rate rtol = 1.0e-13
    @test pressure_source_integral(Q, DoFsP, geo_P, NqP, host) == 0
    @test uniform_pressure_source(rate, DoFsP, geo_P, [cavity; cavity]) == Q

    # With no motion and no storage or thermal change the assembled continuity residual is
    # `∫ Nᵢ Q dΩ`, which sums to `∫ Q dΩ`: the assembler and the oracle integrate the same source.
    zero_v, zero_P = zeros(mesh.nnodes), zeros(mesh.nnodesP)
    phases_P = repeat(reshape(groups.phase, 1, :), length(element_P), 1)
    RP = zeros(mesh.nnodesP)
    FEMTools.assemble_pressure_residual_matrices_atomix!(
        RP, (zero_v, zero_v), zero_P, zero_P, zero_P, zero_P, Q,
        mesh.el2n, mesh.DoFsP, mesh.geometry.geo_v, mesh.geometry.geo_P, mesh.nels,
        element_v, element_P, phases_P, (0.0, 0.0), (1.0, 1.0), 1.0, CPU(), 64,
    )
    @test sum(RP) ≈ rate rtol = 1.0e-12

    # A transfer: the sink normalised with the same quadrature cancels a constant band source.
    band = host[1:5]
    Q_band = add_dike_source!(zeros(mesh.nnodesP), DoFsP, band, 0.02)
    band_integral = pressure_source_integral(Q_band, DoFsP, geo_P, NqP)
    @test band_integral ≈ 0.02 * sum(corner_area, band)
    Q_sink = uniform_pressure_source(-band_integral, DoFsP, geo_P, cavity)
    @test abs(pressure_source_integral(Q_band + Q_sink, DoFsP, geo_P, NqP)) < 1.0e-13 * band_integral

    # `balance_dike_source!` does that in place, from the source already in `Q`.
    Q_transfer = add_dike_source!(zeros(mesh.nnodesP), DoFsP, band, 0.02)
    @test balance_dike_source!(Q_transfer, DoFsP, geo_P, NqP, band, cavity) ≈ band_integral
    @test Q_transfer ≈ Q_band + Q_sink
    @test abs(pressure_source_integral(Q_transfer, DoFsP, geo_P, NqP)) < 1.0e-13 * band_integral
    # It leaves the band's own source alone: only the sink is added.
    @test pressure_source_integral(Q_transfer, DoFsP, geo_P, NqP, band) ≈ band_integral
    @test_throws ArgumentError balance_dike_source!(Q_transfer, DoFsP, geo_P, NqP, band, [band[1]])
    @test_throws ArgumentError balance_dike_source!(Q_transfer, DoFsP, geo_P, NqP, band, Int[])

    @test_throws ArgumentError uniform_pressure_source(NaN, DoFsP, geo_P, cavity)
    @test_throws ArgumentError uniform_pressure_source(1.0, DoFsP, geo_P, Int[])
    @test_throws BoundsError uniform_pressure_source(1.0, DoFsP, geo_P, [mesh.nels + 1])
end
