for FP in (FP32, FP64)
    @testset "mixed mesh constructor" begin
        velocity_element = ReferenceElement(QuadraticElement{2, 6, FP})
        pressure_element = ReferenceElement(LinearElement{2, 3, FP})

        coords = SVector{2, FP}[
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (0.5, 0),
            (0.5, 0.5),
            (0, 0.5),
            (1, 0.5),
            (0.5, 1),
        ]
        el2n = Int32[
            1 2
            2 4
            3 3
            5 8
            6 9
            7 6
        ]
        el2nP = Int32[
            1 4
            2 5
            3 6
        ]
        DoFs = Int32.(1:length(coords))
        DoFsP = Int32.(1:length(el2nP))

        mesh = MixedMesh(velocity_element, pressure_element, coords, DoFs, el2n, DoFsP, el2nP)

        @test mesh isa AbstractMesh
        @test mesh isa MixedMesh{2, 2, 1}
        @test mesh.coords === coords
        @test mesh.DoFs == DoFs
        @test mesh.el2n == el2n
        @test mesh.nnodes == length(coords)
        @test mesh.nels == 2
        @test mesh.DoFsP == DoFsP
        @test mesh.el2nP == el2nP
        @test mesh.nnodesP == 6
    end

    @testset "mixed mesh rejects inconsistent pressure connectivity" begin
        velocity_element = ReferenceElement(QuadraticElement{2, 6, FP})
        pressure_element = ReferenceElement(LinearElement{2, 3, FP})

        coords = SVector{2, FP}[
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (0.5, 0),
            (0.5, 0.5),
        ]
        el2n = reshape(Int32[1, 2, 3, 4, 5, 6], 6, 1)
        bad_el2nP = Int32[
            1 4
            2 5
            3 6
        ]

        @test_throws ArgumentError MixedMesh(
            velocity_element,
            pressure_element,
            coords,
            Int32.(1:length(coords)),
            el2n,
            Int32.(1:length(bad_el2nP)),
            bad_el2nP,
        )
    end
end
