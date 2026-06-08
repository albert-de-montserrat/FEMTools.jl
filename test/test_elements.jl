using Test

using FEMTools
using StaticArrays

const SUPPORTED_ELEMENTS = (
    LinearElement{1, 2},
    QuadraticElement{1, 3},
    LinearElement{2, 3},
    QuadraticElement{2, 6},
    LinearElement{2, 4},
    QuadraticElement{2, 9},
)

@testset "latest element hierarchy" begin
    @test LinearElement{1, 2} <: FEMTools.AbstractLinearElement{1, 2}
    @test LinearElement{2, 3} <: FEMTools.AbstractLinearElement{2, 3}
    @test LinearElement{2, 4} <: FEMTools.AbstractLinearElement{2, 4}

    @test QuadraticElement{1, 3} <: FEMTools.AbstractQuadraticElement{1, 3}
    @test QuadraticElement{2, 6} <: FEMTools.AbstractQuadraticElement{2, 6}
    @test QuadraticElement{2, 9} <: FEMTools.AbstractQuadraticElement{2, 9}

    @test LinearElement{1, 2} <: FEMTools.AbstractElement{1, 2}
    @test QuadraticElement{1, 3} <: FEMTools.AbstractElement{1, 3}

    @test !(LinearElement{1, 2} <: FEMTools.AbstractQuadraticElement{1, 2})
    @test !(QuadraticElement{1, 3} <: FEMTools.AbstractLinearElement{1, 3})
    @test !(LinearElement{2, 4} <: FEMTools.AbstractElement{2, 3})
end

@testset "latest reference element constructors" begin
    for Element in SUPPORTED_ELEMENTS
        element_from_type = ReferenceElement(Element)
        element_from_instance = ReferenceElement(Element())

        @test typeof(element_from_type) === typeof(element_from_instance)
        @test length(element_from_type) == Element.parameters[2]
        @test element_from_type isa ReferenceElement{Element.parameters[1], Element.parameters[2], Element}
        @test element_from_type.shape_functions isa FEMTools.AbstractShapeFunction
        @test element_from_type.integration_points isa FEMTools.AbstractIntegrationPoints
    end
end

@testset "element order" begin
    linear_elements = (
        LinearElement{1, 2},
        LinearElement{2, 3},
        LinearElement{2, 4},
    )
    quadratic_elements = (
        QuadraticElement{1, 3},
        QuadraticElement{2, 6},
        QuadraticElement{2, 9},
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

    @test_throws MethodError order(CubicElement{1, 4})
    @test_throws MethodError order(CubicElement{1, 4}())
end

@testset "latest integration point metadata" begin
    cases = (
        (LinearElement{1, 2}, 1, 2),
        (QuadraticElement{1, 3}, 1, 3),
        (LinearElement{2, 3}, 2, 1),
        (QuadraticElement{2, 6}, 2, 3),
        (LinearElement{2, 4}, 2, 4),
        (QuadraticElement{2, 9}, 2, 9),
    )

    for (Element, nDim, nIp) in cases
        ip = IntegrationPoints(Element)

        @test ip isa FEMTools.IntegrationPoints{nDim, nIp, Float64}
        @test ip.ξ isa SVector{nIp, Float64}
        @test ip.ω isa SVector{nIp, Float64}
        @test length(ip.ξ) == nIp
        @test length(ip.ω) == nIp
        @test nDim == 1 ? ip.η === nothing : ip.η isa SVector{nIp, Float64}
        @test ip.ζ === nothing
    end
end

@testset "latest quadratic line ordering" begin
    element = ReferenceElement(QuadraticElement{1, 3})
    nodes = (-1.0, 0.0, 1.0)

    for (i, ξ) in pairs(nodes)
        values = eval_shape_function(element, (ξ,))
        gradients = eval_shape_function_gradient(element, (ξ,))

        @test values[i] == 1.0
        @test sum(values) == 1.0
        @test sum(gradients) == 0.0
    end

    ξ = 0.37
    @test eval_shape_function(element, (ξ,)) == [
        ξ * (ξ - 1) * 0.5,
        1 - ξ^2,
        ξ * (ξ + 1) * 0.5,
    ]
    @test eval_shape_function_gradient(element, (ξ,)) == [
        ξ - 0.5,
        -2ξ,
        ξ + 0.5,
    ]
end

@testset "latest unsupported element methods" begin
    @test CubicElement{1, 4} <: FEMTools.AbstractElement{1, 4}

    @test_throws MethodError ShapeFunctions(CubicElement{1, 4})
    @test_throws MethodError IntegrationPoints(CubicElement{1, 4})
    @test_throws MethodError ReferenceElement(CubicElement{1, 4})
end
