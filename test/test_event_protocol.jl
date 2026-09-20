const _PROTOCOL_EXAMPLES = joinpath(pkgdir(FEMTools), "examples", "reykjanes")

@testset "event protocol validation" begin
    include(joinpath(_PROTOCOL_EXAMPLES, "event_protocol.jl"))

    protocol = EventProtocol()
    @test protocol.criterion === :tensile
    @test protocol.arrest_target === :reservoir_pressure
    # The two defaults that make the detector usable at all: a nonzero tensile strength and a
    # magmastatic head. With neither, a connected path from the sill to the surface exists at rest.
    @test protocol.tensile_strength > 0
    @test protocol.magmastatic_head
    # An intrusion step short against the Maxwell time but not so short that the tectonic scaling of
    # the model breaks the relaxation.
    @test 0 < protocol.intrusion_Δt < 10 * 365.25 * 24 * 3600

    @test EventProtocol(; criterion = :both).criterion === :both
    @test_throws ArgumentError EventProtocol(; criterion = :whenever)
    @test_throws ArgumentError EventProtocol(; arrest_target = :whatever)
    @test_throws ArgumentError EventProtocol(; tensile_strength = -1.0)
    @test_throws ArgumentError EventProtocol(; magma_density = 0.0)
    @test_throws ArgumentError EventProtocol(; crossing_tolerance = 0.0)
    @test_throws ArgumentError EventProtocol(; crossing_samples = 0)
    @test_throws ArgumentError EventProtocol(; band_width = 0.0)
    @test_throws ArgumentError EventProtocol(; intrusion_Δt = 0.0)
    @test_throws ArgumentError EventProtocol(; max_opening = 0.0)
    @test_throws ArgumentError EventProtocol(; amplitude_rtol = 1.0)
    @test_throws ArgumentError EventProtocol(; dike_normal = (0.0, 0.0))
end

@testset "the detector rule is a protocol choice, and its default is mesh independent" begin
    include(joinpath(_PROTOCOL_EXAMPLES, "event_protocol.jl"))

    # Gate G1: the default detector must not be the one that lives on the element graph. The
    # adjacency rule is kept, because the sweep quotes the three rules against each other, but it is
    # not what a threshold is measured with.
    protocol = EventProtocol()
    @test protocol.detector === :corridor_column
    @test protocol.detector in DETECTOR_RULES
    @test :graph_path in DETECTOR_RULES
    # The corridor is wide against an element and narrow against the section, and its bins are fixed
    # in geometry, so refining the mesh cannot move them.
    @test 0 < protocol.corridor_width < 1.0e4
    @test protocol.corridor_bins >= 1
    @test 0 < protocol.corridor_fill <= 1

    @test EventProtocol(; detector = :graph_path).detector === :graph_path
    @test EventProtocol(; corridor_bins = 8).corridor_bins == 8
    @test_throws ArgumentError EventProtocol(; detector = :whatever)
    @test_throws ArgumentError EventProtocol(; corridor_width = 0.0)
    @test_throws ArgumentError EventProtocol(; corridor_bins = 0)
    @test_throws ArgumentError EventProtocol(; corridor_fill = 0.0)
    @test_throws ArgumentError EventProtocol(; corridor_fill = 1.5)
end

@testset "the detector's Boolean rule is the protocol's choice" begin
    include(joinpath(_PROTOCOL_EXAMPLES, "event_protocol.jl"))

    # Three elements, two integration points each: one fails only in shear, one only in tension, one
    # in both but at different points.
    shear = [true false false; false false true]
    tensile = [false true false; false false false]
    diagnostics = (; shear_failed = shear, tensile_failed = tensile, failed = shear .| tensile)

    @test failed_elements(diagnostics, :shear) == [true, false, true]
    @test failed_elements(diagnostics, :tensile) == [false, true, false]
    @test failed_elements(diagnostics, :either) == [true, true, true]
    # `:both` is the same point failing twice, not the same element failing in two ways.
    @test failed_elements(diagnostics, :both) == [false, false, false]
    diagnostics.tensile_failed[2, 3] = true
    @test failed_elements(diagnostics, :both) == [false, false, true]
end

@testset "path averages weight by element area" begin
    include(joinpath(_PROTOCOL_EXAMPLES, "event_protocol.jl"))

    s3 = [10.0 20.0 30.0; 30.0 40.0 50.0]          # element means 20, 30, 40
    areas = [1.0, 3.0, 5.0]
    @test closure_stress(s3, areas, [1, 2]) ≈ (20 + 3 * 30) / 4
    @test closure_stress(s3, areas, 1:3) ≈ (20 + 3 * 30 + 5 * 40) / 9
    @test closure_stress(s3, areas, [2]) ≈ 30
    @test_throws ArgumentError closure_stress(s3, areas, Int[])
    @test_throws ArgumentError closure_stress(s3, zeros(3), [1, 2])

    # The closure traction of the band is `P − τ_nn`, not the pressure alone.
    P_ip = fill(100.0, 2, 3)
    τ_nn = [1.0 2.0 3.0; 3.0 4.0 5.0]
    @test band_closure_traction(τ_nn, P_ip, areas, [1, 2]) ≈ 100 - (2 + 3 * 3) / 4
    @test_throws ArgumentError band_closure_traction(τ_nn, P_ip, areas, Int[])

    # The magma pressure is the reservoir pressure plus the head, weighted the same way, so that it
    # can be compared with the closure stress element by element.
    head = [-1.0e6 -2.0e6 -3.0e6; -3.0e6 -4.0e6 -5.0e6]
    @test magma_pressure(1.0e8, head, areas, [1, 2]) ≈ 1.0e8 - (2.0e6 + 3 * 3.0e6) / 4
    @test magma_pressure(1.0e8, zeros(2, 3), areas, 1:3) == 1.0e8
    @test_throws ArgumentError magma_pressure(1.0e8, head, areas, Int[])
end

@testset "the deviatoric normal stress follows the dike normal" begin
    include(joinpath(_PROTOCOL_EXAMPLES, "event_protocol.jl"))

    τ = ([2.0 0.0], [-2.0 0.0], [0.5 0.0])          # τxx, τyy, τxy at one point of two elements
    @test normal_deviatoric_stress(τ, (1.0, 0.0))[1] ≈ 2.0
    @test normal_deviatoric_stress(τ, (0.0, 1.0))[1] ≈ -2.0
    @test normal_deviatoric_stress(τ, (1.0, 1.0))[1] ≈ 0.5      # (2 − 2)/2 + 2(0.5)/2
    # The normal is normalised, so its length does not change the answer.
    @test normal_deviatoric_stress(τ, (3.0, 0.0)) == normal_deviatoric_stress(τ, (1.0, 0.0))
end
