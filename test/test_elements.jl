using Test

using FEMTools
using StaticArrays

const SUPPORTED_ELEMENTS = (
    LinearElement{1, 2, FP64},
    QuadraticElement{1, 3, FP64},
    LinearElement{2, 3, FP64},
    QuadraticElement{2, 6, FP64},
    LinearElement{2, 4, FP64},
    QuadraticElement{2, 9, FP64},
    LinearElement{3, 8, FP64},
    QuadraticElement{3, 27, FP64},
    LinearElement{1, 2, FP32},
    QuadraticElement{1, 3, FP32},
    LinearElement{2, 3, FP32},
    QuadraticElement{2, 6, FP32},
    LinearElement{2, 4, FP32},
    QuadraticElement{2, 9, FP32},
    LinearElement{3, 8, FP32},
    QuadraticElement{3, 27, FP32},

)


@testset "latest element hierarchy" begin
    for FP in (FP64, FP32)
        @test LinearElement{1, 2, FP} <: FEMTools.AbstractLinearElement{1, 2, FP}
        @test LinearElement{2, 3, FP} <: FEMTools.AbstractLinearElement{2, 3, FP}
        @test LinearElement{2, 4, FP} <: FEMTools.AbstractLinearElement{2, 4, FP}
        @test LinearElement{3, 8, FP} <: FEMTools.AbstractLinearElement{3, 8, FP}

        @test QuadraticElement{1, 3, FP} <: FEMTools.AbstractQuadraticElement{1, 3, FP}
        @test QuadraticElement{2, 6, FP} <: FEMTools.AbstractQuadraticElement{2, 6, FP}
        @test QuadraticElement{2, 9, FP} <: FEMTools.AbstractQuadraticElement{2, 9, FP}
        @test QuadraticElement{3, 27, FP} <: FEMTools.AbstractQuadraticElement{3, 27, FP}

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
        (LinearElement{2, 4}, LinearElement{2, 4, Float64}),
        (QuadraticElement{2, 9}, QuadraticElement{2, 9, Float64}),
        (LinearElement{3, 8}, LinearElement{3, 8, Float64}),
        (QuadraticElement{3, 27}, QuadraticElement{3, 27, Float64}),
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
        )
        quadratic_elements = (
            QuadraticElement{1, 3, FP},
            QuadraticElement{2, 6, FP},
            QuadraticElement{2, 9, FP},
            QuadraticElement{3, 27, FP},
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
            (LinearElement{2, 4, FP}, 2, 4),
            (QuadraticElement{2, 9, FP}, 2, 9),
            (LinearElement{3, 8, FP}, 3, 8),
            (QuadraticElement{3, 27, FP}, 3, 27),
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
