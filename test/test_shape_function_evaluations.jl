using Test

using FEMTools

@testset "shape function evaluation helpers" begin
    line = ReferenceElement(LinearElement{1, 2})
    ξ = 0.25

    @test eval_shape_function(line, (ξ,)) == [0.375, 0.625]
    @test eval_shape_function_gradient(line, (ξ,)) == [-0.5, 0.5]
    @test eval_shape_function_jacobian(line, (ξ,)) == [-0.5, 0.5]

    quadratic_line = ReferenceElement(QuadraticElement{1, 3})
    ξ = -0.2

    @test eval_shape_function(quadratic_line, (ξ,)) ==
          [ξ * (ξ - 1) * 0.5, 1 - ξ^2, ξ * (ξ + 1) * 0.5]
    @test eval_shape_function_gradient(quadratic_line, (ξ,)) ==
          [ξ - 0.5, -2ξ, ξ + 0.5]
    @test eval_shape_function_jacobian(quadratic_line, (ξ,)) ==
          [ξ - 0.5, -2ξ, ξ + 0.5]

    triangle = ReferenceElement(LinearElement{2, 3})
    ξ, η = 0.2, 0.3

    @test eval_shape_function(triangle, (ξ, η)) == [1 - ξ - η, ξ, η]
    @test eval_shape_function_gradient(triangle, (ξ, η)) ==
          [(-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)]
    @test eval_shape_function_jacobian(triangle, (ξ, η)) ==
          [-1.0 -1.0; 1.0 0.0; 0.0 1.0]

    quadratic_triangle = ReferenceElement(QuadraticElement{2, 6})
    ξ, η = 0.2, 0.3
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
        (4ξ - 1, 0.0),
        (0.0, 4η - 1),
        (4 * (1 - 2ξ - η), -4ξ),
        (4η, 4ξ),
        (-4η, 4 * (1 - ξ - 2η)),
    ]
    @test eval_shape_function_jacobian(quadratic_triangle, (ξ, η)) ≈ [
        1 - 4L1 1 - 4L1
        4ξ - 1 0.0
        0.0 4η - 1
        4 * (1 - 2ξ - η) -4ξ
        4η 4ξ
        -4η 4 * (1 - ξ - 2η)
    ]

    quadrilateral = ReferenceElement(LinearElement{2, 4})
    ξ, η = 0.2, -0.4

    @test eval_shape_function(quadrilateral, (ξ, η)) == [
        (1 - ξ) * (1 - η) * 0.25,
        (1 + ξ) * (1 - η) * 0.25,
        (1 + ξ) * (1 + η) * 0.25,
        (1 - ξ) * (1 + η) * 0.25,
    ]
    @test eval_shape_function_gradient(quadrilateral, (ξ, η)) == [
        (-(1 - η) * 0.25, -(1 - ξ) * 0.25),
        (+(1 - η) * 0.25, -(1 + ξ) * 0.25),
        (+(1 + η) * 0.25, +(1 + ξ) * 0.25),
        (-(1 + η) * 0.25, +(1 - ξ) * 0.25),
    ]
    @test eval_shape_function_jacobian(quadrilateral, (ξ, η)) ≈ [
        -(1 - η) * 0.25 -(1 - ξ) * 0.25
        +(1 - η) * 0.25 -(1 + ξ) * 0.25
        +(1 + η) * 0.25 +(1 + ξ) * 0.25
        -(1 + η) * 0.25 +(1 - ξ) * 0.25
    ]

    quadratic_quadrilateral = ReferenceElement(QuadraticElement{2, 9})
    ξ, η = 0.2, -0.4
    l1(x) = x * (x - 1) * 0.5
    l2(x) = x * (x + 1) * 0.5
    l3(x) = 1 - x^2
    dl1(x) = x - 0.5
    dl2(x) = x + 0.5
    dl3(x) = -2x

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
        dl1(ξ)*l1(η) l1(ξ)*dl1(η)
        dl2(ξ)*l1(η) l2(ξ)*dl1(η)
        dl2(ξ)*l2(η) l2(ξ)*dl2(η)
        dl1(ξ)*l2(η) l1(ξ)*dl2(η)
        dl3(ξ)*l1(η) l3(ξ)*dl1(η)
        dl2(ξ)*l3(η) l2(ξ)*dl3(η)
        dl3(ξ)*l2(η) l3(ξ)*dl2(η)
        dl1(ξ)*l3(η) l1(ξ)*dl3(η)
        dl3(ξ)*l3(η) l3(ξ)*dl3(η)
    ]
end
