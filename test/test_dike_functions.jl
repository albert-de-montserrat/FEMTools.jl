@testset "dike eigenstrain helpers" begin
    include(joinpath(pkgdir(FEMTools), "examples", "reykjanes", "dike_functions", "dike_eigenstrain.jl"))

    strain = dike_eigenstrain_increment(0.2, 2.0, (3.0, 4.0))
    @test strain.volumetric ≈ 0.1
    @test isapprox(strain.xx + strain.yy + strain.zz, 0; atol = 1.0e-15)
    @test strain.xx ≈ 0.1 * (0.36 - 1 / 3)
    @test strain.yy ≈ 0.1 * (0.64 - 1 / 3)
    @test strain.xy ≈ 0.048

    τ_old = ([10.0 20.0; 30.0 40.0], [1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0])
    apply_dike_eigenstrain!(τ_old, [2.0, 3.0], strain, (2,))
    @test τ_old[1][:, 1] == [10.0, 30.0]
    @test τ_old[2][:, 1] == [1.0, 3.0]
    @test τ_old[3][:, 1] == [5.0, 7.0]
    @test τ_old[1][:, 2] ≈ [20.0 - 2 * 3 * strain.xx, 40.0 - 2 * 3 * strain.xx]
    @test τ_old[2][:, 2] ≈ [2.0 - 2 * 3 * strain.yy, 4.0 - 2 * 3 * strain.yy]
    @test τ_old[3][:, 2] ≈ [6.0 - 2 * 3 * strain.xy, 8.0 - 2 * 3 * strain.xy]

    Q = zeros(6)
    DoFsP = [1 4; 2 5; 3 6]
    returned = apply_dike_opening!(τ_old, Q, DoFsP, [2.0, 3.0], (1,), 0.2, 2.0, (1.0, 0.0), 0.5)
    @test returned.volumetric ≈ 0.1
    @test Q == [0.2, 0.2, 0.2, 0.0, 0.0, 0.0]

    @test_throws ArgumentError dike_eigenstrain_increment(-1.0, 2.0, (1.0, 0.0))
    @test_throws ArgumentError dike_eigenstrain_increment(1.0, 0.0, (1.0, 0.0))
    @test_throws ArgumentError dike_eigenstrain_increment(1.0, 2.0, (0.0, 0.0))
end

@testset "plane-strain dike failure diagnostics" begin
    include(joinpath(pkgdir(FEMTools), "examples", "reykjanes", "dike_functions", "dike_eigenstrain.jl"))

    τ = (reshape([2.0, 0.0], 1, 2), reshape([-1.0, 0.0], 1, 2), reshape([0.0, 2.0], 1, 2))
    P = reshape([10.0, 10.0], 1, 2)
    diag = dike_failure_diagnostics(τ, P, 1.0, π / 6, 12.0, 0.5)
    @test diag.s3 ≈ reshape([8.0, 8.0], 1, 2)
    @test diag.F[:, 1] ≈ [sqrt(3.0) - (cos(π / 6) + 10sin(π / 6))]
    @test diag.hydraulic_margin[:, 1] ≈ [3.5]
    @test diag.in_plane[:, 1] == [true]
    @test diag.tensile_failed[:, 1] == [true]
    @test diag.shear_failed[:, 1] == [false]

    τpure = (reshape([0.0], 1, 1), reshape([0.0], 1, 1), reshape([2.0], 1, 1))
    pure = dike_failure_diagnostics(τpure, reshape([10.0], 1, 1), 100.0, 0.0, 0.0, 0.0)
    @test pure.s3[1] ≈ 8.0
    @test pure.orientation[1] ≈ π / 4
    @test !pure.degenerate[1]
    @test !pure.failed[1]
end

@testset "dike connectivity and opening path" begin
    adjacency = [Int32[2], Int32[1, 3], Int32[2, 4], Int32[3, 5], Int32[4]]
    active = Bool[true, true, false, true, true]
    @test dike_connected_component(adjacency, active, (1,)) == Int32[1, 2]
    @test dike_shortest_path(adjacency, active, (1,), (5,)) == Int32[]
    active[3] = true
    @test dike_connected_component(adjacency, active, (1,)) == Int32[1, 2, 3, 4, 5]
    @test dike_shortest_path(adjacency, active, (1,), (5,)) == Int32[1, 2, 3, 4, 5]
    @test dike_shortest_path(adjacency, active, (1,), (1,)) == Int32[1]
end

@testset "the eigenstrain band opens like a pressurised crack" begin
    include(joinpath(pkgdir(FEMTools), "examples", "reykjanes", "dike_crack_setup.jl"))

    # Coarse, because the point is the mechanism, not the discretisation: the band is ordinary host
    # material and its opening comes from `τ_old` and `Q` alone.
    result = solve_dike_crack(; max_area = (2.5e3)^2 / 4, refinement = 4)
    @test result.converged

    # The plan's gate for this row: the opening profile within 2% of the closed form.
    @test result.profile_error < 0.02
    @test result.n_profile > 20
    @test isapprox(result.opening_ratio, 1; atol = 0.02)
    @test result.error_L2 < 0.02

    # Hydraulic consistency: the traction that would close the band is the crack pressure.
    @test isapprox(result.closure_stress / result.P_reference, 1; atol = 0.02)
    # The band's mean pressure is a different quantity and must not be read as the dike pressure.
    @test result.P_band / result.P_reference < 0.9

    # The band's own continuity identity, independent of the closed form: what was injected, less
    # what its bulk modulus stored, is what it delivered.
    @test isapprox(result.balance, 1; atol = 1.0e-5)
    @test isapprox(result.ΔA / result.ΔA_reference, 1; atol = 1.0e-3)

    # A thinner numerical band with the same opening volume is the same crack.
    thin = solve_dike_crack(; b = 62.5, max_area = (2.5e3)^2 / 4, refinement = 4)
    @test thin.converged
    @test thin.profile_error < 0.02
    @test isapprox(thin.closure_stress / result.closure_stress, 1; atol = 0.01)

    @test_throws ArgumentError solve_dike_crack(; a = 100.0, b = 200.0)
end

# A uniform grid of square cells over `x_range × y_range`, as centroids and areas. It stands in for a
# mesh: the corridor detector only ever sees centroids, areas and failure flags, so two of these at
# different spacings are two meshes of the same section.
function _uniform_cells(x_range, y_range, h)
    centroids = Tuple{Float64, Float64}[]
    for x in (first(x_range) + h / 2):h:(last(x_range) - h / 4),
            y in (first(y_range) + h / 2):h:(last(y_range) - h / 4)
        push!(centroids, (x, y))
    end
    return centroids, fill(h^2, length(centroids))
end

@testset "the detection corridor is fixed geometry, not the element graph" begin
    include(joinpath(pkgdir(FEMTools), "examples", "reykjanes", "dike_functions", "dike_eigenstrain.jl"))

    centroids, areas = _uniform_cells((-2.0e3, 2.0e3), (-4.0e3, -2.0e3), 250.0)
    corridor = build_dike_corridor(
        centroids, areas;
        x_center = 0.0, half_width = 1.0e3, y_bottom = -4.0e3, y_top = -2.0e3, nbins = 4,
    )
    # Only the strip enters the corridor, every bin is sampled, and the mesh tiles it exactly here.
    @test length(corridor.cells) == 8 * 8
    @test all(>(0), corridor.bin_area)
    @test corridor.area ≈ corridor.geometric_area
    @test sum(corridor.bin_area) ≈ corridor.area
    @test all(==(16), corridor.counts)

    # A corridor finer than the mesh is under-resolved, and an under-resolved corridor never trips,
    # however much has failed: a bin the mesh cannot sample is missing evidence, not absence of it.
    thin = build_dike_corridor(
        centroids, areas;
        x_center = 0.0, half_width = 1.0e3, y_bottom = -4.0e3, y_top = -2.0e3, nbins = 32,
    )
    @test any(iszero, thin.bin_area)
    @test corridor_verdict(thin, trues(length(areas)), areas, :corridor_column, 0.5).under_resolved
    @test !corridor_verdict(thin, trues(length(areas)), areas, :corridor_column, 0.5).tripped

    # `eligible` keeps material out of the corridor altogether — the reservoir's own elements.
    eligible = [c[2] > -3.0e3 for c in centroids]
    host = build_dike_corridor(
        centroids, areas;
        x_center = 0.0, half_width = 1.0e3, y_bottom = -4.0e3, y_top = -2.0e3, nbins = 4,
        eligible,
    )
    @test iszero(host.bin_area[1]) && iszero(host.bin_area[2])
    @test host.counts[3] == host.counts[4] == 16

    @test_throws DimensionMismatch build_dike_corridor(
        centroids[1:3], areas; x_center = 0.0, half_width = 1.0e3,
        y_bottom = -4.0e3, y_top = -2.0e3, nbins = 4,
    )
    @test_throws ArgumentError build_dike_corridor(
        centroids, areas; x_center = 0.0, half_width = 0.0,
        y_bottom = -4.0e3, y_top = -2.0e3, nbins = 4,
    )
    @test_throws ArgumentError build_dike_corridor(
        centroids, areas; x_center = 0.0, half_width = 1.0e3,
        y_bottom = -2.0e3, y_top = -4.0e3, nbins = 4,
    )
    @test_throws ArgumentError corridor_verdict(corridor, trues(length(areas)), areas, :whatever, 0.5)
    @test_throws ArgumentError corridor_verdict(corridor, trues(length(areas)), areas, :corridor_column, 0.0)
end

@testset "the corridor verdict is mesh independent where the graph path is not" begin
    include(joinpath(pkgdir(FEMTools), "examples", "reykjanes", "dike_functions", "dike_eigenstrain.jl"))

    # The same physical failure — every point below −3 km has failed, nothing above — sampled by two
    # meshes whose element sizes differ by a factor of four.
    failed_region(c) = c[2] <= -3.0e3
    verdicts = map((500.0, 125.0)) do h
        centroids, areas = _uniform_cells((-2.0e3, 2.0e3), (-4.0e3, -2.0e3), h)
        corridor = build_dike_corridor(
            centroids, areas;
            x_center = 0.0, half_width = 1.0e3, y_bottom = -4.0e3, y_top = -2.0e3, nbins = 4,
        )
        failed = map(failed_region, centroids)
        (;
            corridor, failed, areas,
            column = corridor_verdict(corridor, failed, areas, :corridor_column, 0.5),
            fraction = corridor_verdict(corridor, failed, areas, :corridor_fraction, 0.5),
            full = corridor_verdict(corridor, trues(length(areas)), areas, :corridor_column, 0.5),
        )
    end
    coarse, fine = verdicts

    # Identical to the last digit, at a 4× change of element size: the corridor reads the field, and
    # the field is the same. This is the property Gate G1 asks for and the adjacency rule lacks.
    @test coarse.column.fractions == fine.column.fractions == [1.0, 1.0, 0.0, 0.0]
    @test coarse.column.fraction == fine.column.fraction == 0.5
    # Half the corridor has failed: the column rule says no — the failure has not reached the target
    # depth — and the area-fraction rule says yes at a 0.5 threshold. They are different questions.
    @test !coarse.column.tripped && !fine.column.tripped
    @test coarse.fraction.tripped && fine.fraction.tripped
    # Fill the corridor and both rules trip on both meshes, and the band is the corridor's own rock.
    @test coarse.full.tripped && fine.full.tripped
    @test length(coarse.full.cells) == length(coarse.corridor.cells)
    # The band's area is a physical area, so it too is mesh independent.
    @test sum(coarse.areas[coarse.full.cells]) ≈ sum(fine.areas[fine.full.cells])
end

@testset "the corridor weighs an element by the quadrature of it that failed" begin
    include(joinpath(pkgdir(FEMTools), "examples", "reykjanes", "dike_functions", "dike_eigenstrain.jl"))

    centroids, areas = _uniform_cells((-1.0e3, 1.0e3), (-4.0e3, -3.0e3), 500.0)   # 4 × 2 cells
    corridor = build_dike_corridor(
        centroids, areas;
        x_center = 0.0, half_width = 1.0e3, y_bottom = -4.0e3, y_top = -3.0e3, nbins = 2,
    )
    @test corridor.counts == [4, 4]

    # Four integration points per element, one of them failed in every element of the lower bin.
    points = falses(4, length(areas))
    for iel in eachindex(areas)
        centroids[iel][2] < -3.5e3 && (points[1, iel] = true)
    end
    @test corridor_bin_fractions(corridor, points, areas) ≈ [0.25, 0.0]
    @test corridor_failed_fraction(corridor, points, areas) ≈ 0.125
    # The element reduction of the same field is Boolean, and reads the bin as wholly failed: that
    # coarsening is what the point weighting avoids.
    elements = vec(any(points; dims = 1))
    @test corridor_bin_fractions(corridor, elements, areas) ≈ [1.0, 0.0]
    # The band is whole elements either way: an element with any failed quadrature is in it.
    @test corridor_cells(corridor, points) == corridor_cells(corridor, elements)
    @test corridor_failed_weight(points, 1) ≈ 0.25 || corridor_failed_weight(points, 1) == 0.0

    # Half the quadrature of every element: the corridor is half failed, the column rule trips at a
    # 0.5 threshold and not at 0.75, and none of it depends on how the elements were drawn.
    half = falses(4, length(areas))
    half[1:2, :] .= true
    @test corridor_failed_fraction(corridor, half, areas) ≈ 0.5
    @test corridor_verdict(corridor, half, areas, :corridor_column, 0.5).tripped
    @test !corridor_verdict(corridor, half, areas, :corridor_column, 0.75).tripped
    @test_throws DimensionMismatch corridor_bin_fractions(corridor, falses(4, 3), areas)
end
