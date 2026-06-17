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
        :build_element_groups,
        :eval_shape_function,
        :eval_shape_function_gradient,
        :eval_shape_function_jacobian,
        :shape_function_values,
        :order,
        :ThermalDiffusionDR,
        :solver!,
        :assemble_diffusion_matrices_atomix!,
        :assemble_diffusion_matrices_colored!,
        :apply_dirichlet!,
        :LithostaticPressureDR,
        :assemble_lithostatic_pressure_matrices_atomix!,
        :assemble_lithostatic_pressure_matrices_colored!,
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
