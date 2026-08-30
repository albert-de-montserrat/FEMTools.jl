using KernelAbstractions: CPU
using LinearAlgebra: dot, norm

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
        @test length(mesh.normals) == mesh.nnodes
        @test norm(mesh.normals[1]) ≈ one(FP)
        @test dot(mesh.normals[1], SVector{2, FP}(-1, -1) / sqrt(FP(2))) ≈ one(FP)
        @test mesh.normals[5] ≈ SVector{2, FP}(0, -1)
        @test iszero(norm(mesh.normals[6]))
        @test sprint(show, mesh) == "MixedMesh{2, 2, 1}(nnodes=9, nnodesP=6, nels=2)"
    end

    @testset "mixed mesh supports continuous pressure connectivity" begin
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
            1 2
            2 4
            3 3
        ]
        mesh = MixedMesh(
            velocity_element,
            pressure_element,
            coords,
            Int32.(1:length(coords)),
            el2n,
            Int32.(1:4),
            el2nP,
        )

        @test mesh.nnodesP == 4
        @test sprint(show, mesh) == "MixedMesh{2, 2, 1}(nnodes=9, nnodesP=4, nels=2)"
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

    @testset "mixed mesh cache" begin
        velocity_element = ReferenceElement(QuadraticElement{2, 7, FP})
        pressure_element = ReferenceElement(LinearElement{2, 3, FP})
        mesh_v = Mesh(CPU(), (FP(0)..FP(1)) × (FP(0)..FP(1)), velocity_element, (1, 1))
        mesh = MixedMesh(mesh_v, pressure_element)

        cache = MixedMeshCache(CPU(), 1, mesh, velocity_element, pressure_element)

        @test cache isa MixedMeshCache
        @test length(cache.geo_v) == mesh.nels
        @test length(cache.geo_P) == mesh.nels
        @test length(cache.geo_v[1]) == length(velocity_element.integration_points.ω)
        @test length(cache.geo_P[1]) == length(velocity_element.integration_points.ω)
        @test cache.geo_v[1][1][2] isa FP
        @test cache.geo_P[1][1][2] isa FP
        @test cache.element_v === velocity_element
        @test cache.element_P === pressure_element
        @test length(mesh.normals) == mesh.nnodes
        @test eltype(mesh.normals) == SVector{2, FP}
    end
end

@testset "3D mixed pressure scaling has a positive modal mass" begin
    velocity_element = ReferenceElement(QuadraticElement{3, 27, Float64})
    pressure_element = ReferenceElement(LinearElement{3, 4, Float64})
    mesh_v = Mesh(
        CPU(), (0.0..1.0) × (0.0..1.0) × (0.0..1.0),
        velocity_element, (1, 1, 1),
    )
    mesh = MixedMesh(mesh_v, pressure_element)
    cache = MixedMeshCache(CPU(), 1, mesh, velocity_element, pressure_element)
    dr = StokesDR(
        CPU(), mesh.nnodes, mesh.nnodesP,
        StokesMaterial(; η = (1.0,), ηb = (Inf,), G = (Inf,), α = (0.0,),
                       ρ0 = (1.0,), K = (Inf,), g = (0.0, 0.0, 0.0), Tref = 0.0),
    )
    γP = zeros(mesh.nnodesP)

    FEMTools.assemble_viscosity_weighted_pressure_scaling!(
        γP, dr, mesh, cache, 2.0, 1.0; workgroup = 1,
    )

    @test all(>(0), dr.M_P)
    @test all(isfinite, γP)
    @test all(>(0), γP)
end
