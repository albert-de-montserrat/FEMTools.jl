using Test

using FEMTools
using KernelAbstractions: CPU
using StaticArrays

# One two-node line element of unit physical length: the reference element spans
# ξ ∈ [-1, 1], so J = 1/2 and J⁻¹ = 2. These tests compare the atomic and colored
# assembly paths against each other, so only their sharing one geometry matters.
function _line_assembler_fixture()
    element = ReferenceElement(LinearElement{1, 2, Float64})
    geo_el = ntuple(_ -> QuadraturePointGeometry(SMatrix{1, 1}(2.0), 1.0),
                    length(shape_function_values(element)))
    return element, reshape(Int32[1, 2], 2, 1), [geo_el], [Int32[1]], CPU(), 1
end

@testset "heat assemblers accept compute_jacobian keyword" begin
    element, el2n, geo, groups, backend, workgroup = _line_assembler_fixture()
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

    R_col = zeros(2); J_col = zeros(2); PC_col = zeros(2)
    FEMTools.assemble_diffusion_matrices_colored!(
        R_col, J_col, PC_col, T, T0, el2n, geo, groups, element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref,
        backend, workgroup; compute_jacobian = true,
    )
    @test R_col ≈ R_kw
    @test J_col ≈ J_kw
    @test PC_col ≈ PC_kw

    R_only = zeros(2); J_untouched = fill(NaN, 2); PC_untouched = fill(NaN, 2)
    FEMTools.assemble_diffusion_matrices_atomix!(
        R_only, J_untouched, PC_untouched, T, T0, el2n, geo, 1, element, phases, k, Cp, ρ0, α, K, P, Δt, source, Tref,
        backend, workgroup,
    )
    @test R_only ≈ R_kw
    @test all(isnan, J_untouched)
    @test all(isnan, PC_untouched)
end

@testset "lithostatic assemblers accept compute_jacobian keyword" begin
    element, el2n, geo, groups, backend, workgroup = _line_assembler_fixture()
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

    R_col = zeros(2); J_col = zeros(2); PC_col = zeros(2)
    FEMTools.assemble_lithostatic_pressure_matrices_colored!(
        R_col, J_col, PC_col, T, P, el2n, geo, groups, element, phases, ρ0, α, K, Tref, g,
        backend, workgroup; compute_jacobian = true,
    )
    @test R_col ≈ R_kw
    @test J_col ≈ J_kw
    @test PC_col ≈ PC_kw

    R_only = zeros(2); J_untouched = fill(NaN, 2); PC_untouched = fill(NaN, 2)
    FEMTools.assemble_lithostatic_pressure_matrices_atomix!(
        R_only, J_untouched, PC_untouched, T, P, el2n, geo, 1, element, phases, ρ0, α, K, Tref, g,
        backend, workgroup,
    )
    @test R_only ≈ R_kw
    @test all(isnan, J_untouched)
    @test all(isnan, PC_untouched)
end
