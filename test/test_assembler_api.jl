using Test

using FEMTools
using KernelAbstractions: CPU
using StaticArrays

function _line_assembler_fixture(dNdx)
    element = ReferenceElement(LinearElement{1, 2, Float64})
    geo_el = ntuple(_ -> (dNdx, 1.0), length(shape_function_values(element)))
    return element, reshape(Int32[1, 2], 2, 1), [geo_el], [Int32[1]], CPU(), 1
end

@testset "heat assemblers accept compute_jacobian keyword" begin
    element, el2n, geo, groups, backend, workgroup = _line_assembler_fixture(@SMatrix [0.0; 0.0])
    T = [300.0, 300.0]
    T0 = [300.0, 300.0]
    phases = [1, 1]
    k = (1.0,)
    Cp = (1.0,)
    ρ0 = (2.0,)
    α = (0.01,)
    K = (100.0,)
    P = [0.0, 0.0]
    source = [1.0, 1.0]
    Δt = 1.0
    Tref = 300.0

    R_kw = zeros(2); J_kw = zeros(2); PC_kw = zeros(2)
    FEMTools.assemble_diffusion_matrices_atomix!(
        R_kw, J_kw, PC_kw, T, T0, el2n, geo, 1, element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref,
        backend, workgroup; compute_jacobian = true,
    )

    R_old = zeros(2); J_old = zeros(2); PC_old = zeros(2)
    @test_deprecated FEMTools.assemble_diffusion_matrices_atomix!(
        R_old, J_old, PC_old, T, T0, el2n, geo, 1, element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref, true,
        backend, workgroup,
    )
    @test R_old ≈ R_kw
    @test J_old ≈ J_kw
    @test PC_old ≈ PC_kw

    R_col = zeros(2); J_col = zeros(2); PC_col = zeros(2)
    FEMTools.assemble_diffusion_matrices_colored!(
        R_col, J_col, PC_col, T, T0, el2n, geo, groups, element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref,
        backend, workgroup; compute_jacobian = true,
    )
    @test R_col ≈ R_kw
    @test J_col ≈ J_kw
    @test PC_col ≈ PC_kw

    R_col_old = zeros(2); J_col_old = zeros(2); PC_col_old = zeros(2)
    @test_deprecated FEMTools.assemble_diffusion_matrices_colored!(
        R_col_old, J_col_old, PC_col_old, T, T0, el2n, geo, groups, element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref, true,
        backend, workgroup,
    )
    @test R_col_old ≈ R_col
    @test J_col_old ≈ J_col
    @test PC_col_old ≈ PC_col
end

@testset "lithostatic assemblers accept compute_jacobian keyword" begin
    element, el2n, geo, groups, backend, workgroup = _line_assembler_fixture(@SMatrix [0.0; 1.0])
    T = [300.0, 300.0]
    P = [1.0, 0.0]
    phases = [1, 1]
    ρ0 = (2.0,)
    α = (0.01,)
    K = (100.0,)
    Tref = 300.0
    g = SVector(-1.0)

    R_kw = zeros(2); J_kw = zeros(2); PC_kw = zeros(2)
    FEMTools.assemble_lithostatic_pressure_matrices_atomix!(
        R_kw, J_kw, PC_kw, T, P, el2n, geo, 1, element, phases, ρ0, α, K, Tref, g,
        backend, workgroup; compute_jacobian = true,
    )

    R_old = zeros(2); J_old = zeros(2); PC_old = zeros(2)
    @test_deprecated FEMTools.assemble_lithostatic_pressure_matrices_atomix!(
        R_old, J_old, PC_old, T, P, el2n, geo, 1, element, phases, ρ0, α, K, Tref, g, true,
        backend, workgroup,
    )
    @test R_old ≈ R_kw
    @test J_old ≈ J_kw
    @test PC_old ≈ PC_kw

    R_col = zeros(2); J_col = zeros(2); PC_col = zeros(2)
    FEMTools.assemble_lithostatic_pressure_matrices_colored!(
        R_col, J_col, PC_col, T, P, el2n, geo, groups, element, phases, ρ0, α, K, Tref, g,
        backend, workgroup; compute_jacobian = true,
    )
    @test R_col ≈ R_kw
    @test J_col ≈ J_kw
    @test PC_col ≈ PC_kw

    R_col_old = zeros(2); J_col_old = zeros(2); PC_col_old = zeros(2)
    @test_deprecated FEMTools.assemble_lithostatic_pressure_matrices_colored!(
        R_col_old, J_col_old, PC_col_old, T, P, el2n, geo, groups, element, phases, ρ0, α, K, Tref, g, true,
        backend, workgroup,
    )
    @test R_col_old ≈ R_col
    @test J_col_old ≈ J_col
    @test PC_col_old ≈ PC_col
end
