using Test

using DomainSets
using DomainSets: ×
using FEMTools
using KernelAbstractions: CPU
using SparseArrays
using StaticArrays

const TRI_DNDX = @SMatrix [-1.0 -1.0; 1.0 0.0; 0.0 1.0]
const TRI_GEO = ((TRI_DNDX, 0.5),)
const TRI_NQ = (SVector(1 / 3, 1 / 3, 1 / 3),)

function _element_eval_case(::Type{FP}) where FP
    element = ReferenceElement(LinearElement{2, 3, FP})
    ξ = SVector(FP(0.2), FP(0.3))
    return eval_shape_function(element, ξ)
end

function _triangle_mesh_case()
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
    ]
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    return Mesh(CPU(), coords, el2n)
end

function _tet_mesh_case()
    coords = SVector{3, Float64}[
        SVector(0.0, 0.0, 0.0),
        SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0),
        SVector(0.0, 0.0, 1.0),
    ]
    el2n = reshape(Int32[1, 2, 3, 4], 4, 1)
    return Mesh(CPU(), coords, el2n)
end

function _structured_mesh_case(::Type{FP}) where FP
    element = ReferenceElement(LinearElement{2, 4, FP})
    return Mesh(CPU(), (FP(0) .. FP(1)) × (FP(0) .. FP(1)), element, (1, 1))
end

function _mesh_utilities_case()
    mesh = _structured_mesh_case(Float64)
    colors = color_mesh(mesh)
    return (
        FEMTools.generate_sparsity_pattern(mesh),
        colors,
        generate_element_groups(CPU(), colors),
    )
end

function _mixed_mesh_case()
    element_v = ReferenceElement(QuadraticElement{2, 6, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
        SVector(0.5, 0.0),
        SVector(0.5, 0.5),
        SVector(0.0, 0.5),
    ]
    el2n = reshape(Int32[1, 2, 3, 4, 5, 6], 6, 1)
    dofs = Int32.(1:length(coords))
    mesh_v = Mesh(element_v, nothing, nothing, coords, dofs, el2n, Int32[])
    return MixedMesh(mesh_v, element_P)
end

function _heat_integrate_case()
    Tloc = SVector(1.0, 2.0, 3.0)
    T0loc = SVector(1.0, 1.0, 1.0)
    sloc = SVector(0.0, 0.0, 0.0)
    phase_loc = SVector(1, 1, 1)
    Ploc = SVector(0.0, 0.0, 0.0)
    return FEMTools.integrate_residual(
        Tloc, T0loc, TRI_GEO, sloc, phase_loc,
        (2.0,), (1.0,), (1.0,), (0.0,), (Inf,),
        Ploc, 1.0, 0.0, TRI_NQ, Val(3),
    )
end

function _lithostatic_integrate_case()
    Ploc = SVector(0.0, 0.0, 0.0)
    Tloc = SVector(0.0, 0.0, 0.0)
    phase_loc = SVector(1, 1, 1)
    return FEMTools.lp_integrate_residual(
        Ploc, Tloc, TRI_GEO, phase_loc,
        (1.0,), (0.0,), (Inf,), 0.0, (0.0, -1.0), TRI_NQ, Val(3),
    )
end

function _stokes_integrate_case()
    vxloc = SVector(0.0, 0.1, 0.0)
    vyloc = SVector(0.0, 0.0, -0.1)
    P_loc = SVector(0.0, 0.0, 0.0)
    phase_loc = SVector(1, 1, 1)
    return FEMTools.integrate_momentum_residual(
        (vxloc, vyloc), P_loc, TRI_GEO, phase_loc,
        (1.0,), (Inf,), 1.0, TRI_NQ, TRI_NQ,
    )
end

function _heat_element_jacobian_case()
    T = [1.0, 2.0, 3.0]
    T0 = [1.0, 1.0, 1.0]
    source = [0.0, 0.0, 0.0]
    P = [0.0, 0.0, 0.0]
    phases = [1, 1, 1]
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    geo = [TRI_GEO]
    return FEMTools.element_jacobian(
        T, T0, source, el2n, geo, phases,
        (2.0,), (1.0,), (1.0,), (0.0,), (Inf,),
        P, 1.0, 0.0, TRI_NQ, 1, Val(3),
    )
end

function _lithostatic_element_jacobian_case()
    T = [0.0, 0.0, 0.0]
    P = [0.0, 0.0, 0.0]
    phases = [1, 1, 1]
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    geo = [TRI_GEO]
    return FEMTools.lp_element_jacobian(
        T, P, el2n, geo, phases,
        (1.0,), (0.0,), (Inf,), 0.0, (0.0, -1.0), TRI_NQ, 1, Val(3),
    )
end

function _stokes_augmented_component_case()
    vxloc = SVector(0.0, 0.1, 0.0)
    vyloc = SVector(0.0, 0.0, -0.1)
    P_loc = SVector(0.0, 0.0, 0.0)
    P0loc = SVector(0.0, 0.0, 0.0)
    T_loc = SVector(0.0, 0.0, 0.0)
    T0loc = SVector(0.0, 0.0, 0.0)
    phase_loc = SVector(1, 1, 1)
    MP_loc = SVector(1.0, 1.0, 1.0)
    args = (
        (vxloc, vyloc), P_loc, P0loc, T_loc, T0loc,
        TRI_GEO, TRI_GEO, phase_loc, phase_loc,
        (1.0,), (Inf,), (0.0,), (1.0,), (Inf,),
        (0.0, -1.0), 0.0, (Inf,), 1.0, 0.5, MP_loc, TRI_NQ, TRI_NQ,
    )
    return (
        FEMTools.integrate_momentum_x_residual(args...),
        FEMTools.integrate_momentum_y_residual(args...),
    )
end

function _pressure_element_residual_case()
    vx = zeros(3)
    vy = zeros(3)
    P = zeros(3)
    P0 = zeros(3)
    T = zeros(3)
    T0 = zeros(3)
    phases = ones(Int, 3)
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    geo = [TRI_GEO]
    return FEMTools.pressure_element_residual(
        (vx, vy), P, P0, T, T0, el2n, el2n, geo, geo, phases,
        (0.0,), (Inf,), 1.0, TRI_NQ, 1, Val(3), Val(3),
    )
end

function _heat_dr_case(::Type{FP}) where FP
    return ThermalDiffusionDR(
        CPU(), 3,
        (FP(2),), (FP(1),), (FP(1),), (FP(0),), (FP(Inf),),
    )
end

function _lithostatic_dr_case(::Type{FP}) where FP
    return LithostaticPressureDR(
        CPU(), 3,
        (FP(1),), (FP(0),), (FP(Inf),),
    )
end

function _stokes_dr_case(::Type{FP}) where FP
    material = StokesMaterial(;
        η = (FP(1),), ηb = (FP(Inf),), G = (FP(Inf),), α = (FP(0),),
        ρ0 = (FP(1),), K = (FP(Inf),), g = (FP(0), -FP(1)), Tref = FP(0),
    )
    return StokesDR(
        CPU(), 6, 3, material;
        stress_size = (1, 1),
    )
end

function _mixed_mesh_cache_case()
    mesh = _mixed_mesh_case()
    element_v = ReferenceElement(QuadraticElement{2, 6, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    return MixedMeshCache(CPU(), 1, mesh, element_v, element_P)
end

function _heat_assembly_case()
    element = ReferenceElement(LinearElement{2, 3, Float64})
    R = zeros(3)
    ∂R∂T = zeros(3)
    PC = zeros(3)
    T = [1.0, 2.0, 3.0]
    T0 = ones(3)
    source = zeros(3)
    P = zeros(3)
    phases = ones(Int, 3)
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    geo = [TRI_GEO]
    return FEMTools.assemble_diffusion_matrices_atomix!(
        R, ∂R∂T, PC, T, T0, el2n, geo, 1, element, phases,
        (2.0,), (1.0,), (1.0,), (0.0,), (Inf,),
        P, 1.0, source, 0.0, CPU(), 1; compute_jacobian = true,
    )
end

function _lithostatic_assembly_case()
    element = ReferenceElement(LinearElement{2, 3, Float64})
    R = zeros(3)
    ∂R∂P = zeros(3)
    PC = zeros(3)
    T = zeros(3)
    P = zeros(3)
    phases = ones(Int, 3)
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    geo = [TRI_GEO]
    return FEMTools.assemble_lithostatic_pressure_matrices_atomix!(
        R, ∂R∂P, PC, T, P, el2n, geo, 1, element, phases,
        (1.0,), (0.0,), (Inf,), 0.0, (0.0, -1.0), CPU(), 1;
        compute_jacobian = true,
    )
end

function _stokes_assembly_case()
    element = ReferenceElement(LinearElement{2, 3, Float64})
    Rvx = zeros(3)
    Rvy = zeros(3)
    vx = zeros(3)
    vy = zeros(3)
    P = zeros(3)
    T = zeros(3)
    Pnum = zeros(3)
    phases = ones(Int, 3)
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    geo = [TRI_GEO]
    return FEMTools.assemble_momentum_residual_matrices_atomix!(
        Rvx, Rvy, vx, vy, P, T, Pnum, el2n, el2n, geo, 1,
        element, element, phases, nothing, nothing, nothing,
        (1.0,), (Inf,), (0.0,), (1.0,), (Inf,),
        (0.0, -1.0), 0.0, 1.0, CPU(), 1,
    )
end

function _pressure_assembly_case()
    element = ReferenceElement(LinearElement{2, 3, Float64})
    RP = zeros(3)
    vx = zeros(3)
    vy = zeros(3)
    P = zeros(3)
    P0 = zeros(3)
    T = zeros(3)
    T0 = zeros(3)
    phases = ones(Int, 3)
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    geo = [TRI_GEO]
    return FEMTools.assemble_pressure_residual_matrices_atomix!(
        RP, vx, vy, P, P0, T, T0, el2n, el2n, geo, geo, 1,
        element, element, phases, (0.0,), (Inf,), 1.0, CPU(), 1,
    )
end

function _pressure_scaling_assembly_case()
    element = ReferenceElement(LinearElement{2, 3, Float64})
    MP = zeros(3)
    γP = zeros(3)
    phases = ones(Int, 3)
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    geo = [TRI_GEO]
    return FEMTools.assemble_viscosity_weighted_pressure_scaling!(
        MP, γP, el2n, el2n, geo, 1, element, element, phases,
        (1.0,), 0.5, (Inf,), 1.0, CPU(), 1,
    )
end

@testset "core type inference" begin
    @test (@inferred _element_eval_case(Float32)) isa SVector{3, Float32}
    @test (@inferred _element_eval_case(Float64)) isa SVector{3, Float64}
    @test (@inferred _triangle_mesh_case()) isa Mesh{2}
    @test (@inferred _tet_mesh_case()) isa Mesh{3}
    @test (@inferred _structured_mesh_case(Float32)) isa Mesh{2, 1, <:Rectangle, <:UnionDomain, <:Vector{SVector{2, Float32}}}
    @test (@inferred _structured_mesh_case(Float64)) isa Mesh{2, 1, <:Rectangle, <:UnionDomain, <:Vector{SVector{2, Float64}}}
    @test (@inferred _mesh_utilities_case()) isa Tuple{SparseMatrixCSC{Bool, Int}, Vector{Int}, Vector{Vector{Int}}}
    @test (@inferred _mixed_mesh_case()) isa MixedMesh{2}
    @test (@inferred _heat_integrate_case()) isa SVector{3, Float64}
    @test (@inferred _lithostatic_integrate_case()) isa SVector{3, Float64}
    @test (@inferred _stokes_integrate_case()) isa Tuple{SVector{3, Float64}, SVector{3, Float64}}
    @test (@inferred _heat_element_jacobian_case()) isa Tuple{<:SVector{3, <:Integer}, SVector{3, Float64}, SVector{3, Float64}}
    @test (@inferred _lithostatic_element_jacobian_case()) isa Tuple{<:SVector{3, <:Integer}, SVector{3, Float64}, SVector{3, Float64}}
    @test (@inferred _stokes_augmented_component_case()) isa Tuple{SVector{3, Float64}, SVector{3, Float64}}
    @test (@inferred _pressure_element_residual_case()) isa Tuple{<:SVector{3, <:Integer}, SVector{3, Float64}}
    @test (@inferred _heat_dr_case(Float32)) isa ThermalDiffusionDR{1, <:Vector{Float32}, <:Vector{Int}, Float32}
    @test (@inferred _heat_dr_case(Float64)) isa ThermalDiffusionDR{1, <:Vector{Float64}, <:Vector{Int}, Float64}
    @test (@inferred _lithostatic_dr_case(Float64)) isa LithostaticPressureDR{1, <:Vector{Float64}, <:Vector{Int}, Float64}
    @test (@inferred _stokes_dr_case(Float64)) isa StokesDR{2, 1, <:Vector{Float64}, <:Vector{Int}, <:Matrix{Float64}, Float64, 3}
    @test (@inferred _mixed_mesh_cache_case()) isa MixedMeshCache
    @test (@inferred _heat_assembly_case()) === nothing
    @test (@inferred _lithostatic_assembly_case()) === nothing
    @test (@inferred _stokes_assembly_case()) === nothing
    @test (@inferred _pressure_assembly_case()) === nothing
    @test (@inferred _pressure_scaling_assembly_case()) === nothing
end

@testset "JET optimization checks" begin
    if Base.find_package("JET") === nothing
        @test_skip "JET is not available in this environment"
    else
        @eval using JET
        @eval JET.@test_opt _element_eval_case(Float64)
        @eval JET.@test_opt _structured_mesh_case(Float64)
        @eval JET.@test_opt _heat_integrate_case()
        @eval JET.@test_opt _lithostatic_integrate_case()
        @eval JET.@test_opt _stokes_integrate_case()
        @eval JET.@test_opt _heat_element_jacobian_case()
        @eval JET.@test_opt _lithostatic_element_jacobian_case()
        @eval JET.@test_opt _stokes_augmented_component_case()
        @eval JET.@test_opt _pressure_element_residual_case()
    end
end

@testset "maintained example scripts parse" begin
    scripts = [
        "examples/1D_diffusion_FEMTools.jl",
        "examples/1D_diffusion_FEMTools_color.jl",
        "examples/2D_diffusion_FEMTools.jl",
        "examples/3D_diffusion_FEMTools.jl",
        "examples/gmsh_meshing.jl",
        "examples/Poisson/1D_Poisson.jl",
        "examples/Poisson/1D_Poisson_AD.jl",
        "examples/Poisson/1D_Poisson_Q2.jl",
        "examples/Poisson/2D_Poisson.jl",
        "examples/Poisson/2D_Poisson_AD.jl",
        "examples/Poisson/2D_Poisson_AD_KA.jl",
        "examples/Poisson/3D_Poisson.jl",
        "examples/Poisson/3D_Poisson_AD.jl",
        "examples/Poisson/3D_Poisson_AD_KA.jl",
        "examples/elasticity/2D_Elasticity_stress_postprocess.jl",
        "examples/elasticity/2D_Elasticiy_DR_KA.jl",
        "examples/elasticity/2D_Elasticiy_Direct_KA.jl",
        "examples/heat_diffusion/2D_heat_diffusion.jl",
        "examples/heat_diffusion/2D_heat_diffusion_triangles.jl",
        "examples/heat_diffusion/2D_heat_diffusion_unstructured.jl",
        "examples/heat_diffusion/2D_heat_diffusion_unstructured_T6.jl",
        "examples/heat_diffusion/3D_heat_diffusion_unstructured_hex.jl",
        "examples/stokes/buildup/stokes_2D_elastic_buildup.jl",
        "examples/stokes/buildup/stokes_2D_elastic_buildup_hole.jl",
        "examples/stokes/sinking_block/sinking_block.jl",
        "examples/stokes/volcano/volcano_mesh_3D.jl",
        "examples/stokes/volcano/volcano_thermal_stokes.jl",
        "examples/stokes/volcano/volcano_thermal_stokes_3D.jl",
        "examples/stokes/stokes_2D_pure_shear_triangle_hole.jl",
        "examples/stokes/vevp/stokes_2D_pure_shear.jl",
        "examples/stokes/vevp/stokes_2D_pure_shear_triangle.jl",
        "examples/stokes/vevp/stokes_2D_pure_shear_triangle_adj.jl",
        "examples/stokes/vevp/stokes_2D_pure_shear_triangle_adv.jl",
        "examples/stokes/vevp/stokes_2D_dike_triangle_adv.jl",
        "examples/stokes/vevp/stokes_2D_shear_bands_triangle.jl",
        "examples/stokes/vevp/stokes_2D_viscous_inclusion_triangle.jl",
    ]
    for script in scripts
        @test Meta.parseall(read(joinpath(pkgdir(FEMTools), script), String)) isa Expr
    end
end
