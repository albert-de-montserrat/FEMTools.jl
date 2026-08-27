using Test

using FEMTools
using StaticArrays

const SUPPORTED_ELEMENTS = (
    LinearElement{1, 2, FP64},
    QuadraticElement{1, 3, FP64},
    LinearElement{2, 3, FP64},
    QuadraticElement{2, 6, FP64},
    QuadraticElement{2, 7, FP64},
    LinearElement{2, 4, FP64},
    QuadraticElement{2, 9, FP64},
    LinearElement{3, 8, FP64},
    QuadraticElement{3, 27, FP64},
    LinearElement{3, 4, FP64},
    QuadraticElement{3, 10, FP64},
    QuadraticElement{3, 11, FP64},
    LinearElement{1, 2, FP32},
    QuadraticElement{1, 3, FP32},
    LinearElement{2, 3, FP32},
    QuadraticElement{2, 6, FP32},
    QuadraticElement{2, 7, FP32},
    LinearElement{2, 4, FP32},
    QuadraticElement{2, 9, FP32},
    LinearElement{3, 8, FP32},
    QuadraticElement{3, 27, FP32},
    LinearElement{3, 4, FP32},
    QuadraticElement{3, 10, FP32},
    QuadraticElement{3, 11, FP32},

)


@testset "latest element hierarchy" begin
    for FP in (FP64, FP32)
        @test LinearElement{1, 2, FP} <: FEMTools.AbstractLinearElement{1, 2, FP}
        @test LinearElement{2, 3, FP} <: FEMTools.AbstractLinearElement{2, 3, FP}
        @test LinearElement{2, 4, FP} <: FEMTools.AbstractLinearElement{2, 4, FP}
        @test LinearElement{3, 8, FP} <: FEMTools.AbstractLinearElement{3, 8, FP}
        @test LinearElement{3, 4, FP} <: FEMTools.AbstractLinearElement{3, 4, FP}

        @test QuadraticElement{1, 3, FP} <: FEMTools.AbstractQuadraticElement{1, 3, FP}
        @test QuadraticElement{2, 6, FP} <: FEMTools.AbstractQuadraticElement{2, 6, FP}
        @test QuadraticElement{2, 7, FP} <: FEMTools.AbstractQuadraticElement{2, 7, FP}
        @test QuadraticElement{2, 9, FP} <: FEMTools.AbstractQuadraticElement{2, 9, FP}
        @test QuadraticElement{3, 27, FP} <: FEMTools.AbstractQuadraticElement{3, 27, FP}
        @test QuadraticElement{3, 10, FP} <: FEMTools.AbstractQuadraticElement{3, 10, FP}
        @test QuadraticElement{3, 11, FP} <: FEMTools.AbstractQuadraticElement{3, 11, FP}

        @test LinearElement{1, 2, FP} <: FEMTools.AbstractElement{1, 2, FP}
        @test QuadraticElement{1, 3, FP} <: FEMTools.AbstractElement{1, 3, FP}

        @test !(LinearElement{1, 2, FP} <: FEMTools.AbstractQuadraticElement{1, 2, FP})
        @test !(QuadraticElement{1, 3, FP} <: FEMTools.AbstractLinearElement{1, 3, FP})
        @test !(LinearElement{2, 4, FP} <: FEMTools.AbstractElement{2, 3, FP})
    end
end

@testset "latest reference element constructors" begin
    for Element in SUPPORTED_ELEMENTS
        element_from_type = ReferenceElement(Element)
        element_from_instance = ReferenceElement(Element())

        @test typeof(element_from_type) === typeof(element_from_instance)
        @test length(element_from_type) == Element.parameters[2]
        @test element_from_type isa ReferenceElement{Element}
        @test element_from_type.shape_functions isa FEMTools.AbstractShapeFunction
        @test element_from_type.integration_points isa FEMTools.AbstractIntegrationPoints
        @test sprint(show, element_from_type) ==
            "ReferenceElement{$Element}(order=$(order(element_from_type)), nodes=$(length(element_from_type)), nips=$(length(element_from_type.integration_points.ω)))"
    end
end

@testset "T-less element constructors default to Float64" begin
    cases = (
        (LinearElement{1, 2}, LinearElement{1, 2, Float64}),
        (QuadraticElement{1, 3}, QuadraticElement{1, 3, Float64}),
        (LinearElement{2, 3}, LinearElement{2, 3, Float64}),
        (QuadraticElement{2, 6}, QuadraticElement{2, 6, Float64}),
        (QuadraticElement{2, 7}, QuadraticElement{2, 7, Float64}),
        (LinearElement{2, 4}, LinearElement{2, 4, Float64}),
        (QuadraticElement{2, 9}, QuadraticElement{2, 9, Float64}),
        (LinearElement{3, 8}, LinearElement{3, 8, Float64}),
        (QuadraticElement{3, 27}, QuadraticElement{3, 27, Float64}),
        (LinearElement{3, 4}, LinearElement{3, 4, Float64}),
        (QuadraticElement{3, 10}, QuadraticElement{3, 10, Float64}),
        (QuadraticElement{3, 11}, QuadraticElement{3, 11, Float64}),
    )

    for (Element, ConcreteElement) in cases
        @test ReferenceElement(Element) isa ReferenceElement{ConcreteElement}
        @test typeof(ShapeFunctions(Element)) === typeof(ShapeFunctions(ConcreteElement))
        @test IntegrationPoints(Element) == IntegrationPoints(ConcreteElement)
    end
end

@testset "element order" begin
    for FP in (FP64, FP32)
        linear_elements = (
            LinearElement{1, 2, FP},
            LinearElement{2, 3, FP},
            LinearElement{2, 4, FP},
            LinearElement{3, 8, FP},
            LinearElement{3, 4, FP},
        )
        quadratic_elements = (
            QuadraticElement{1, 3, FP},
            QuadraticElement{2, 6, FP},
            QuadraticElement{2, 7, FP},
            QuadraticElement{2, 9, FP},
            QuadraticElement{3, 27, FP},
            QuadraticElement{3, 10, FP},
            QuadraticElement{3, 11, FP},
        )

        for Element in linear_elements
            element = ReferenceElement(Element)

            @test order(Element) == 1
            @test order(Element()) == 1
            @test order(element) == 1
        end

        for Element in quadratic_elements
            element = ReferenceElement(Element)

            @test order(Element) == 2
            @test order(Element()) == 2
            @test order(element) == 2
        end

        @test_throws MethodError order(CubicElement{1, 4, FP})
        @test_throws MethodError order(CubicElement{1, 4, FP}())
    end
end

@testset "latest integration point metadata" begin
    for FP in (FP64, FP32)
        cases = (
            (LinearElement{1, 2, FP}, 1, 2),
            (QuadraticElement{1, 3, FP}, 1, 3),
            (LinearElement{2, 3, FP}, 2, 3),
            (QuadraticElement{2, 6, FP}, 2, 6),
            (QuadraticElement{2, 7, FP}, 2, 7),
            (LinearElement{2, 4, FP}, 2, 4),
            (QuadraticElement{2, 9, FP}, 2, 9),
            (LinearElement{3, 8, FP}, 3, 8),
            (QuadraticElement{3, 27, FP}, 3, 27),
            (LinearElement{3, 4, FP}, 3, 1),
            (QuadraticElement{3, 10, FP}, 3, 4),
            (QuadraticElement{3, 11, FP}, 3, 15),
        )

        for (Element, nDim, nIp) in cases
            ip = IntegrationPoints(Element)

            @test ip isa FEMTools.IntegrationPoints{nDim, nIp, FP}
            @test ip.ξ isa SVector{nIp, FP}
            @test ip.ω isa SVector{nIp, FP}
            @test length(ip.ξ) == nIp
            @test length(ip.ω) == nIp
            @test nDim == 1 ? ip.η === nothing : ip.η isa SVector{nIp, FP}
            @test nDim < 3 ? ip.ζ === nothing : ip.ζ isa SVector{nIp, FP}
        end
    end
end

@testset "generated triangle quadrature input validation" begin
    @test_throws "quadrature order must be positive, got 0" gauss_legendre_triangle(0)
    @test_throws "quadrature order must be positive, got -1" gauss_legendre_triangle(Float32, -1)
    @test_throws "quadrature order must be positive, got 0" FEMTools._gauss_legendre_01(Float64, 0)
end

@testset "Gauss-Legendre rule on the unit interval" begin
    for FP in (FP64, FP32), n in 1:5
        pts, wts = FEMTools._gauss_legendre_01(FP, n)
        @test length(pts) == n
        @test length(wts) == n
        @test eltype(pts) === FP
        @test eltype(wts) === FP
        @test all(p -> 0 < p < 1, pts)
        @test sum(wts) ≈ 1 rtol = 8eps(FP)
        # An n-point rule is exact for polynomials of degree ≤ 2n-1.
        for k in 0:(2n - 1)
            @test sum(wts .* pts .^ k) ≈ 1 / (k + 1) rtol = 1.0e3 * eps(FP)
        end
    end
end

@testset "generated triangle quadrature accuracy" begin
    for n in 1:5
        ip = gauss_legendre_triangle(n)
        @test length(ip.ω) == n^2
        @test ip.ζ === nothing
        @test all(>(0), ip.ξ)
        @test all(>(0), ip.η)
        @test all(ip.ξ .+ ip.η .< 1)
        @test sum(ip.ω) ≈ 1 / 2          # area of the reference triangle
        # The Duffy Jacobian costs one degree in ξ, leaving exactness up to
        # total degree 2n-2; ∫ ξᵃ ηᵇ dΩ = a! b! / (a+b+2)! on the reference triangle.
        for a in 0:(2n - 2), b in 0:(2n - 2 - a)
            exact = factorial(a) * factorial(b) / factorial(a + b + 2)
            @test sum(ip.ω .* ip.ξ .^ a .* ip.η .^ b) ≈ exact
        end
    end
    ip32 = gauss_legendre_triangle(Float32, 3)
    @test eltype(ip32.ω) === Float32
    @test sum(ip32.ω) ≈ 0.5f0
end

@testset "tetrahedron integration rules" begin
    for FP in (FP64, FP32)
        for (Element, degree) in ((QuadraticElement{3, 10, FP}, 2),
                                  (QuadraticElement{3, 11, FP}, 5))
            ip = IntegrationPoints(Element)
            @test sum(ip.ω) ≈ FP(1/6) atol = 20eps(FP)
            @test all(ip.ξ .>= 0) && all(ip.η .>= 0) && all(ip.ζ .>= 0)
            @test all(ip.ξ .+ ip.η .+ ip.ζ .<= 1)

            for i in 0:degree, j in 0:(degree - i), k in 0:(degree - i - j)
                numerical = sum(ip.ω .* ip.ξ.^i .* ip.η.^j .* ip.ζ.^k)
                exact = FP(factorial(i) * factorial(j) * factorial(k) /
                           factorial(i + j + k + 3))
                @test numerical ≈ exact atol = 100eps(FP)
            end
        end
    end
end

@testset "triangle integration rules" begin
    for FP in (FP64, FP32)
        tri3 = IntegrationPoints(LinearElement{2, 3, FP})
        @test tri3 isa FEMTools.IntegrationPoints{2, 3, FP}
        @test tri3.ξ == SVector{3, FP}(1 / 6, 2 / 3, 1 / 6)
        @test tri3.η == SVector{3, FP}(1 / 6, 1 / 6, 2 / 3)
        @test tri3.ω == SVector{3, FP}(1 / 6, 1 / 6, 1 / 6)
        @test sum(tri3.ω) ≈ FP(1 / 2)

        tri6 = IntegrationPoints(QuadraticElement{2, 6, FP})
        @test tri6 isa FEMTools.IntegrationPoints{2, 6, FP}
        @test length(tri6.ω) == 6
        @test sum(tri6.ω) ≈ FP(1 / 2)
    end
end

@testset "latest quadratic line ordering" begin
    for FP in (FP64, FP32)
        element = ReferenceElement(QuadraticElement{1, 3, FP})
        nodes = FP.((-1.0, 0.0, 1.0))

        for (i, ξ) in pairs(nodes)
            values = eval_shape_function(element, (ξ,))
            gradients = eval_shape_function_gradient(element, (ξ,))

            @test values[i] == 1.0
            @test sum(values) == 1.0
            @test sum(gradients) == 0.0
        end

        ξ = FP(0.37)
        @test eval_shape_function(element, (ξ,)) == [
            ξ * (ξ - 1) * FP(1/2),
            1 - FP(ξ^2),
            ξ * (ξ + 1) * FP(1/2),
        ]
        @test eval_shape_function_gradient(element, (ξ,)) == [
            ξ - FP(1/2),
            -2ξ,
            ξ + FP(1/2),
        ]
    end
end

@testset "latest unsupported element methods" begin
    for FP in (FP64, FP32)
        @test CubicElement{1, 4, FP} <: FEMTools.AbstractElement{1, 4, FP}

        @test_throws MethodError ShapeFunctions(CubicElement{1, 4, FP})
        @test_throws MethodError IntegrationPoints(CubicElement{1, 4, FP})
        @test_throws MethodError ReferenceElement(CubicElement{1, 4, FP})
    end
end
