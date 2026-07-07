for FP in (FP32, FP64)
    @testset "mesh coordinates" begin
        Ω = zero(FP) .. FP(10.0)

        linear = ReferenceElement(LinearElement{1, 2, FP})
        @test generate_coordinates(linear, Ω, 4) == FP.([0.0, 2.5, 5.0, 7.5, 10.0])

        quadratic = ReferenceElement(QuadraticElement{1, 3, FP})
        @test generate_coordinates(quadratic, Ω, 2) == FP.([0.0, 2.5, 5.0, 7.5, 10.0])

        quad4 = ReferenceElement(LinearElement{2, 4, FP})
        @test generate_coordinates(quad4, (zero(FP) .. FP(2)) × (zero(FP) .. FP(1)), (2, 1)) == [
            SVector{2, FP}(0.0, 0.0),
            SVector{2, FP}(1.0, 0.0),
            SVector{2, FP}(2.0, 0.0),
            SVector{2, FP}(0.0, 1.0),
            SVector{2, FP}(1.0, 1.0),
            SVector{2, FP}(2.0, 1.0),
        ]

        quad9 = ReferenceElement(QuadraticElement{2, 9, FP})
        @test generate_coordinates(quad9, (zero(FP) .. FP(2)) × (zero(FP) .. FP(1)), (1, 1)) == [
            SVector{2, FP}(0.0, 0.0),
            SVector{2, FP}(1.0, 0.0),
            SVector{2, FP}(2.0, 0.0),
            SVector{2, FP}(0.0, 0.5),
            SVector{2, FP}(1.0, 0.5),
            SVector{2, FP}(2.0, 0.5),
            SVector{2, FP}(0.0, 1.0),
            SVector{2, FP}(1.0, 1.0),
            SVector{2, FP}(2.0, 1.0),
        ]

        hex8 = ReferenceElement(LinearElement{3, 8, FP})
        @test generate_coordinates(hex8, (zero(FP) .. FP(1)) × (zero(FP) .. FP(2)) × (zero(FP) .. FP(3)), (1, 1, 1)) == [
            SVector{3, FP}(0.0, 0, 0),
            SVector{3, FP}(1.0, 0, 0),
            SVector{3, FP}(0.0, 2, 0),
            SVector{3, FP}(1.0, 2, 0),
            SVector{3, FP}(0.0, 0, 3),
            SVector{3, FP}(1.0, 0, 3),
            SVector{3, FP}(0.0, 2, 3),
            SVector{3, FP}(1.0, 2, 3),
        ]

        hex27 = ReferenceElement(QuadraticElement{3, 27, FP})
        coords = generate_coordinates(hex27, (0.0 .. 1.0) × (0.0 .. 2.0) × (0.0 .. 3.0), (1, 1, 1))
        @test length(coords) == 27
        @test first(coords) == SVector{3, FP}(0.0, 0.0, 0.0)
        @test coords[14] == SVector{3, FP}(0.5, 1.0, 1.5)
        @test last(coords) == SVector{3, FP}(1.0, 2.0, 3.0)
    end

    @testset "mesh connectivity" begin
        linear = ReferenceElement(LinearElement{1, 2, FP})
        el2n = generate_element2node(linear, 3)
        @test el2n == Int32[1 2 3; 2 3 4]

        n2el = generate_node2element(el2n)
        @test n2el == Vector{Int32}[[1], [1, 2], [2, 3], [3]]

        quadratic = ReferenceElement(QuadraticElement{1, 3, FP})
        el2n = generate_element2node(quadratic, 3)
        @test el2n == Int32[1 3 5; 2 4 6; 3 5 7]

        n2el = generate_node2element(el2n)
        @test n2el == Vector{Int32}[[1], [1], [1, 2], [2], [2, 3], [3], [3]]
        @test generate_boundary_elements(Int32[1, 7], n2el) == Int32[1, 3]

        quad4 = ReferenceElement(LinearElement{2, 4, FP})
        el2n = generate_element2node(quad4, (2, 1))
        @test el2n == Int32[1 2; 2 3; 5 6; 4 5]

        quad9 = ReferenceElement(QuadraticElement{2, 9, FP})
        el2n = generate_element2node(quad9, (1, 1))
        @test el2n == reshape(Int32[1, 3, 9, 7, 2, 6, 8, 4, 5], 9, 1)

        hex8 = ReferenceElement(LinearElement{3, 8, FP})
        el2n = generate_element2node(hex8, (1, 1, 1))
        @test el2n == reshape(Int32[1, 2, 4, 3, 5, 6, 8, 7], 8, 1)

        hex27 = ReferenceElement(QuadraticElement{3, 27, FP})
        el2n = generate_element2node(hex27, (1, 1, 1))
        @test el2n == reshape(
            Int32[
                1,
                3,
                9,
                7,
                19,
                21,
                27,
                25,
                2,
                6,
                8,
                4,
                20,
                24,
                26,
                22,
                10,
                12,
                18,
                16,
                5,
                11,
                15,
                17,
                13,
                23,
                14,
            ],
            27,
            1,
        )
    end

    @testset "mesh constructor" begin
        Ω = FP(0.0) .. FP(10.0)
        element = ReferenceElement(QuadraticElement{1, 3, FP})
        mesh = Mesh(Ω, element, 3)

        @test mesh isa FEMTools.AbstractMesh
        @test mesh isa Mesh{1, 2}
        @test mesh.Ω == Ω
        @test mesh.Γ == boundary(Ω)
        @test mesh.coords ≈ [0.0, 10 / 6, 10 / 3, 5.0, 20 / 3, 50 / 6, 10.0]
        @test mesh.DoFs == Int32[1, 2, 3, 4, 5, 6, 7]
        @test mesh.el2n == Int32[1 3 5; 2 4 6; 3 5 7]
        @test mesh.Γnodes == Int32[1, 7]
        @test mesh.nnodes == 7
        @test mesh.nels == 3

        quad4 = ReferenceElement(LinearElement{2, 4, FP})
        mesh2D = Mesh((FP(0.0) .. FP(2.0)) × (FP(0.0) .. FP(1.0)), quad4, (2, 1))
        @test mesh2D isa Mesh{2, 1}
        @test mesh2D.coords == [
            SVector{2, FP}(0.0, 0.0),
            SVector{2, FP}(1.0, 0.0),
            SVector{2, FP}(2.0, 0.0),
            SVector{2, FP}(0.0, 1.0),
            SVector{2, FP}(1.0, 1.0),
            SVector{2, FP}(2.0, 1.0),
        ]
        @test mesh2D.el2n == Int32[1 2; 2 3; 5 6; 4 5]
        @test mesh2D.nnodes == 6
        @test mesh2D.nels == 2
    end

    @testset "mesh sparsity pattern" begin
        element = ReferenceElement(QuadraticElement{1, 3, FP})
        mesh = Mesh(FP(0) .. FP(10), element, 3)
        pattern = generate_sparsity_pattern(mesh)

        @test pattern isa SparseMatrixCSC{Bool, Int}
        @test size(pattern) == (mesh.nnodes, mesh.nnodes)
        @test pattern == pattern'
        @test nnz(pattern) == 25

        for iel in 1:mesh.nels
            local_nodes = @view mesh.el2n[:, iel]
            for j in local_nodes, i in local_nodes
                @test pattern[i, j]
            end
        end
    end

    @testset "mesh coloring" begin
        element = ReferenceElement(QuadraticElement{1, 3, FP})
        mesh = Mesh(FP(0) .. FP(10), element, 6)
        greedy_colors = FEMTools.color_mesh_greedy(mesh)
        colors = color_mesh(mesh)

        function is_valid_coloring(mesh, colors)
            n2el = generate_node2element(mesh.el2n, mesh.nnodes)
            for iel in 1:mesh.nels
                for node in @view mesh.el2n[:, iel]
                    for jel in n2el[node]
                        if jel != iel && colors[jel] == colors[iel]
                            return false
                        end
                    end
                end
            end
            return true
        end

        @test colors == [1, 2, 1, 2, 1, 2]
        @test maximum(colors) == 2
        @test is_valid_coloring(mesh, colors)
        @test is_valid_coloring(mesh, greedy_colors)
    end
end
