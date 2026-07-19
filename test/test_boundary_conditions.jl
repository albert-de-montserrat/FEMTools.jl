using Test

using DomainSets
using FEMTools
using SparseArrays

@testset "dirichlet boundary condition" begin
    Γ = boundary(0.0 .. 1.0)
    DoFs = Int32[1, 4]
    vals = [10.0, -2.0]

    ΓD = DirichletBoundaryCondition(Γ, DoFs, vals)

    @test ΓD isa FEMTools.AbstractBoundaryCondition
    @test ΓD.Γ == Γ
    @test ΓD.DoFs == DoFs
    @test ΓD.vals == vals
    @test ΓD.zero_vals == zero(vals)

    rhs = zeros(4)
    returned_rhs = apply_bc!(rhs, ΓD)

    @test returned_rhs === nothing
    @test rhs == [10.0, 0.0, 0.0, -2.0]

    A = fill(2.0, 4, 4)
    rhs = zeros(4)
    returned_system = apply_bc!(A, rhs, ΓD)

    @test returned_system === nothing
    @test rhs == [10.0, -16.0, -16.0, -2.0]
    @test A[1, :] == [1.0, 0.0, 0.0, 0.0]
    @test A[4, :] == [0.0, 0.0, 0.0, 1.0]
    @test A[2, :] == [0.0, 2.0, 2.0, 0.0]
    @test A[3, :] == [0.0, 2.0, 2.0, 0.0]

    A = sparse(fill(2.0, 4, 4))
    rhs = zeros(4)
    apply_bc!(A, rhs, ΓD)

    @test rhs == [10.0, -16.0, -16.0, -2.0]
    @test Matrix(A) == [
        1.0 0.0 0.0 0.0
        0.0 2.0 2.0 0.0
        0.0 2.0 2.0 0.0
        0.0 0.0 0.0 1.0
    ]

    @test_throws DimensionMismatch apply_bc!(ones(2, 3), zeros(2), ΓD)
    @test_throws DimensionMismatch apply_bc!(ones(4, 4), zeros(3), ΓD)

    bad_dof = DirichletBoundaryCondition(Γ, Int32[5], [1.0])
    @test_throws BoundsError apply_bc!(zeros(4), bad_dof)
    @test_throws BoundsError apply_bc!(zeros(4, 4), bad_dof)
    @test_throws BoundsError apply_bc!(zeros(4, 4), zeros(4), bad_dof)
end
