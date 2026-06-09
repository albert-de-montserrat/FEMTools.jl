for FP in (FP32, FP64)
    @testset "shape function evaluation helpers" begin
        line = ReferenceElement(LinearElement{1, 2, FP})
        ξ = FP(0.25)

        @test eval_shape_function(line, (ξ,)) == FP.([0.375, 0.625])
        @test eval_shape_function_gradient(line, (ξ,)) == FP.([-0.5, 0.5])
        @test eval_shape_function_jacobian(line, (ξ,)) == FP.([-0.5, 0.5])

        quadratic_line = ReferenceElement(QuadraticElement{1, 3, FP})
        ξ = FP(-0.2)

        @test eval_shape_function(quadratic_line, (ξ,)) ==
            FP.([ξ * (ξ - 1) * 0.5, 1 - ξ^2, ξ * (ξ + 1) * 0.5])
        @test eval_shape_function_gradient(quadratic_line, (ξ,)) ==
            FP.([ξ - 0.5, -2ξ, ξ + 0.5])
        @test eval_shape_function_jacobian(quadratic_line, (ξ,)) ==
            FP.([ξ - 0.5, -2ξ, ξ + 0.5])

        triangle = ReferenceElement(LinearElement{2, 3, FP})
        ξ, η = FP.((0.2, 0.3))

        @test eval_shape_function(triangle, (ξ, η)) == [1 - ξ - η, ξ, η]
        @test eval_shape_function_gradient(triangle, (ξ, η)) ==
            [FP.((-one(FP), -one(FP))), FP.((1.0, +zero(FP))), FP.((+zero(FP), 1.0))]
        @test eval_shape_function_jacobian(triangle, (ξ, η)) ==
            FP.([-one(FP) -one(FP); 1.0 +zero(FP); +zero(FP) 1.0])

        quadratic_triangle = ReferenceElement(QuadraticElement{2, 6, FP})
        ξ, η = FP.((0.2, 0.3))
        L1 = 1 - ξ - η

        @test eval_shape_function(quadratic_triangle, (ξ, η)) ≈ [
            L1 * (2L1 - 1),
            ξ * (2ξ - 1),
            η * (2η - 1),
            4L1 * ξ,
            4ξ * η,
            4η * L1,
        ]
        @test eval_shape_function_gradient(quadratic_triangle, (ξ, η)) == [
            (1 - 4L1, 1 - 4L1),
            (4ξ - 1, +zero(FP)),
            (+zero(FP), 4η - 1),
            (4 * (1 - 2ξ - η), -4ξ),
            (4η, 4ξ),
            (-4η, 4 * (1 - ξ - 2η)),
        ]
        @test eval_shape_function_jacobian(quadratic_triangle, (ξ, η)) ≈ [
            1 - 4L1 1 - 4L1
            4ξ - 1 +zero(FP)
            +zero(FP) 4η - 1
            4 * (1 - 2ξ - η) -4ξ
            4η 4ξ
            -4η 4 * (1 - ξ - 2η)
        ]

        quadrilateral = ReferenceElement(LinearElement{2, 4, FP})
        ξ, η = FP.((0.2, -0.4))

        @test eval_shape_function(quadrilateral, (ξ, η)) == [
            (1 - ξ) * (1 - η) * FP(0.25),
            (1 + ξ) * (1 - η) * FP(0.25),
            (1 + ξ) * (1 + η) * FP(0.25),
            (1 - ξ) * (1 + η) * FP(0.25),
        ]
        @test eval_shape_function_gradient(quadrilateral, (ξ, η)) == [
            (-(1 - η) * FP(0.25), -(1 - ξ) * FP(0.25)),
            (+(1 - η) * FP(0.25), -(1 + ξ) * FP(0.25)),
            (+(1 + η) * FP(0.25), +(1 + ξ) * FP(0.25)),
            (-(1 + η) * FP(0.25), +(1 - ξ) * FP(0.25)),
        ]
        @test eval_shape_function_jacobian(quadrilateral, (ξ, η)) ≈ [
            -(1 - η) * FP(0.25) -(1 - ξ) * FP(0.25)
            +(1 - η) * FP(0.25) -(1 + ξ) * FP(0.25)
            +(1 + η) * FP(0.25) +(1 + ξ) * FP(0.25)
            -(1 + η) * FP(0.25) +(1 - ξ) * FP(0.25)
        ]

        quadratic_quadrilateral = ReferenceElement(QuadraticElement{2, 9, FP})
        ξ, η = FP.((0.2, -0.4))
        l1(x)  = FP(x * (x - 1) / 2)
        l2(x)  = FP(x * (x + 1) / 2)
        l3(x)  = FP(1 - x^2)
        dl1(x) = FP(x - 1/2)
        dl2(x) = FP(x + 1/2)
        dl3(x) = FP(-2x)

        @test eval_shape_function(quadratic_quadrilateral, (ξ, η)) ≈ [
            l1(ξ) * l1(η),
            l2(ξ) * l1(η),
            l2(ξ) * l2(η),
            l1(ξ) * l2(η),
            l3(ξ) * l1(η),
            l2(ξ) * l3(η),
            l3(ξ) * l2(η),
            l1(ξ) * l3(η),
            l3(ξ) * l3(η),
        ]
        @test eval_shape_function_gradient(quadratic_quadrilateral, (ξ, η)) == [
            (dl1(ξ) * l1(η), l1(ξ) * dl1(η)),
            (dl2(ξ) * l1(η), l2(ξ) * dl1(η)),
            (dl2(ξ) * l2(η), l2(ξ) * dl2(η)),
            (dl1(ξ) * l2(η), l1(ξ) * dl2(η)),
            (dl3(ξ) * l1(η), l3(ξ) * dl1(η)),
            (dl2(ξ) * l3(η), l2(ξ) * dl3(η)),
            (dl3(ξ) * l2(η), l3(ξ) * dl2(η)),
            (dl1(ξ) * l3(η), l1(ξ) * dl3(η)),
            (dl3(ξ) * l3(η), l3(ξ) * dl3(η)),
        ]
        @test eval_shape_function_jacobian(quadratic_quadrilateral, (ξ, η)) ≈ [
            dl1(ξ) * l1(η) l1(ξ) * dl1(η)
            dl2(ξ) * l1(η) l2(ξ) * dl1(η)
            dl2(ξ) * l2(η) l2(ξ) * dl2(η)
            dl1(ξ) * l2(η) l1(ξ) * dl2(η)
            dl3(ξ) * l1(η) l3(ξ) * dl1(η)
            dl2(ξ) * l3(η) l2(ξ) * dl3(η)
            dl3(ξ) * l2(η) l3(ξ) * dl2(η)
            dl1(ξ) * l3(η) l1(ξ) * dl3(η)
            dl3(ξ) * l3(η) l3(ξ) * dl3(η)
        ]

        hexahedron = ReferenceElement(LinearElement{3, 8, FP})
        hex_nodes = (
            (-one(FP), -one(FP), -one(FP)),
            (+one(FP), -one(FP), -one(FP)),
            (+one(FP), +one(FP), -one(FP)),
            (-one(FP), +one(FP), -one(FP)),
            (-one(FP), -one(FP), +one(FP)),
            (+one(FP), -one(FP), +one(FP)),
            (+one(FP), +one(FP), +one(FP)),
            (-one(FP), +one(FP), +one(FP)),
        )
        for (inode, coords) in pairs(hex_nodes)
            values = eval_shape_function(hexahedron, coords)
            @test values[inode] == 1.0
            @test sum(values) == 1.0
            @test count(!iszero, values) == 1
        end

        quadratic_hexahedron = ReferenceElement(QuadraticElement{3, 27, FP})
        qhex_nodes = (
            (-one(FP), -one(FP), -one(FP)),
            (+one(FP), -one(FP), -one(FP)),
            (+one(FP), +one(FP), -one(FP)),
            (-one(FP), +one(FP), -one(FP)),
            (-one(FP), -one(FP), +one(FP)),
            (+one(FP), -one(FP), +one(FP)),
            (+one(FP), +one(FP), +one(FP)),
            (-one(FP), +one(FP), +one(FP)),
            (+zero(FP), -one(FP), -one(FP)),
            (+one(FP), +zero(FP), -one(FP)),
            (+zero(FP), +one(FP), -one(FP)),
            (-one(FP), +zero(FP), -one(FP)),
            (+zero(FP), -one(FP), +one(FP)),
            (+one(FP), +zero(FP), +one(FP)),
            (+zero(FP), +one(FP), +one(FP)),
            (-one(FP), +zero(FP), +one(FP)),
            (-one(FP), -one(FP), +zero(FP)),
            (+one(FP), -one(FP), +zero(FP)),
            (+one(FP), +one(FP), +zero(FP)),
            (-one(FP), +one(FP), +zero(FP)),
            (+zero(FP), +zero(FP), -one(FP)),
            (+zero(FP), -one(FP), +zero(FP)),
            (+one(FP), +zero(FP), +zero(FP)),
            (+zero(FP), +one(FP), +zero(FP)),
            (-one(FP), +zero(FP), +zero(FP)),
            (+zero(FP), +zero(FP), +one(FP)),
            (+zero(FP), +zero(FP), +zero(FP)),
        )
        for (inode, coords) in pairs(qhex_nodes)
            values = eval_shape_function(quadratic_hexahedron, coords)
            @test values[inode] == 1.0
            @test sum(values) == 1.0
            @test count(!iszero, values) == 1
        end
    end
end