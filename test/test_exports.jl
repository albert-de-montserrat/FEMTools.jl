using Test

import FEMTools

@testset "exports" begin
    exported_names = (
        :LinearElement,
        :QuadraticElement,
        :CubicElement,
        :ReferenceElement,
        :ShapeFunctions,
        :IntegrationPoints,
        :Mesh,
        :DirichletBoundaryCondition,
        :apply_bc!,
        :generate_element2node,
        :generate_node2element,
        :generate_boundary_elements,
        :generate_coordinates,
        :generate_dofs,
        :generate_sparsity_pattern,
        :color_mesh,
        :color_mesh_greedy,
        :eval_shape_function,
        :eval_shape_function_gradient,
        :eval_shape_function_jacobian,
        :order,
        :ThermalDiffusionDR,
        :solver!,
        :assemble_diffusion_matrices_atomix!,
        :apply_dirichlet!,
        :shape_function_values,
        :AbstractElement,
        :AbstractLinearElement,
        :AbstractQuadraticElement,
        :AbstractShapeFunction,
        :AbstractIntegrationPoints,
        :AbstractMesh,
        :AbstractBoundaryCondition,
    )

    public_only_names = ()

    for name in exported_names
        @test Base.isexported(FEMTools, name)
        @test Base.ispublic(FEMTools, name)
    end

    for name in public_only_names
        @test !Base.isexported(FEMTools, name)
        @test Base.ispublic(FEMTools, name)
    end
end
