using Test

using FEMTools

function element_constructor_allocations(::Type{T}) where {T <: FEMTools.AbstractElement}
    ShapeFunctions(T)
    IntegrationPoints(T)
    ReferenceElement(T)

    return (
        shape_functions = @allocated(ShapeFunctions(T)),
        integration_points = @allocated(IntegrationPoints(T)),
        reference_element = @allocated(ReferenceElement(T)),
    )
end

function shape_function_evaluation_allocations(element, coords)
    eval_shape_function(element, coords)
    eval_shape_function_gradient(element, coords)
    eval_shape_function_jacobian(element, coords)

    return (
        shape_function = @allocated(eval_shape_function(element, coords)),
        gradient = @allocated(eval_shape_function_gradient(element, coords)),
        jacobian = @allocated(eval_shape_function_jacobian(element, coords)),
    )
end

if Base.JLOptions().code_coverage == 0
    @testset "allocations" begin
        for FP in (FP32, FP64)
            element_cases = (
                (LinearElement{1, 2, FP}, (0.25,)),
                (QuadraticElement{1, 3, FP}, (-0.2,)),
                (LinearElement{2, 3, FP}, (0.2, 0.3)),
                (QuadraticElement{2, 6, FP}, (0.2, 0.3)),
                (LinearElement{2, 4, FP}, (0.2, -0.4)),
                (QuadraticElement{2, 9, FP}, (0.2, -0.4)),
            )

            for (Element, coords) in element_cases
                constructor_allocs = element_constructor_allocations(Element)
                @test constructor_allocs.shape_functions == 0
                @test constructor_allocs.integration_points == 0
                @test constructor_allocs.reference_element == 0

                element = ReferenceElement(Element)
                evaluation_allocs = shape_function_evaluation_allocations(element, coords)
                @test evaluation_allocs.shape_function == 0
                @test evaluation_allocs.gradient == 0
                @test evaluation_allocs.jacobian == 0
            end
        end
    end
end
