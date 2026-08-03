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
        :MixedMesh,
        :MixedMeshCache,
        :DirichletBoundaryCondition,
        :apply_bc!,
        :generate_element2node,
        :generate_node2element,
        :generate_boundary_elements,
        :generate_coordinates,
        :generate_dofs,
        :generate_sparsity_pattern,
        :color_mesh,
        :color_structured_triangles,
        :reorder_connectivity_by_color,
        :generate_element_groups,
        :generate_discontinuous_linear_mesh,
        :eval_shape_function,
        :eval_shape_function_gradient,
        :eval_shape_function_jacobian,
        :shape_function_values,
        :order,
        :ThermalMaterial,
        :ThermalDiffusionDR,
        :StokesMaterial,
        :solver!,
        :solve_stokes_dyrel!,
        :solve_stokes_3d!,
        :solve_stokes_adjoint_3d!,
        :solve_stokes_adjoint_dyrel!,
        :LithostaticPressureDR,
        :AbstractElement,
        :AbstractLinearElement,
        :AbstractQuadraticElement,
        :AbstractShapeFunction,
        :AbstractIntegrationPoints,
        :AbstractMesh,
        :AbstractBoundaryCondition,
        :assemble_viscosity_weighted_pressure_scaling!,
        :pressure_mass,
    )

    public_only_names = (
        :TA,
        :interp2ip,
        :interp2ip_phase,
        :apply_dirichlet!,
        :assemble_diffusion_matrices_atomix!,
        :assemble_diffusion_matrices_colored!,
        :assemble_lithostatic_pressure_matrices_atomix!,
        :assemble_lithostatic_pressure_matrices_colored!,
        :assemble_momentum_residual_matrices_atomix!,
        :assemble_momentum_jacobian_matrices_atomix!,
        :assemble_augmented_momentum_jacobian_matrices_atomix!,
        :assemble_pressure_residual_matrices_atomix!,
        :update_rate_kernel!,
        :update_variable_kernel!,
        :precompute_geometry_kernel!,
        :stokes_update_rate!,
        :stokes_update_variable!,
        :precompute_stokes_geometry!,
        :color_mesh_greedy,
        :remove_pressure_mean!,
        :velocity,
        :stress,
        :pressure,
        :temperature,
    )

    for name in exported_names
        @test Base.isexported(FEMTools, name)
        @test Base.ispublic(FEMTools, name)
    end

    for name in public_only_names
        @test !Base.isexported(FEMTools, name)
        @test Base.ispublic(FEMTools, name)
    end
end
