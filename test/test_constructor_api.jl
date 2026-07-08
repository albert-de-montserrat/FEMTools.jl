using Test

using FEMTools
using StaticArrays

@testset "constructor API tidying" begin
    bc = DirichletBoundaryCondition(:left, Int32[1, 3], [2.0, 4.0])
    @test bc isa FEMTools.DirichletBoundaryCondition{Symbol, Vector{Int32}, Vector{Float64}}
    @test bc.Γ === :left
    @test bc.DoFs == Int32[1, 3]
    @test bc.vals == [2.0, 4.0]

    N = (x -> x,)
    ∇N = (x -> one(x),)
    sf = ShapeFunctions(N, ∇N)
    @test sf isa FEMTools.ShapeFunctions{typeof(N), typeof(∇N)}
    @test sf.N === N
    @test sf.∇N === ∇N

    coords = SVector{2, Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
    ]
    DoFs = Int32[1, 2, 3]
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    Γnodes = Int32[1, 2, 3]
    mesh = Mesh{2, 1, Nothing, Nothing, typeof(coords), typeof(DoFs), typeof(el2n), typeof(Γnodes)}(
        nothing, nothing, coords, DoFs, el2n, Γnodes, Int32(length(coords)), Int32(size(el2n, 2)),
    )

    @test mesh.coords === coords
    @test mesh.DoFs === DoFs
    @test mesh.el2n === el2n
    @test mesh.Γnodes === Γnodes
    @test mesh.nnodes === length(coords)
    @test mesh.nels === size(el2n, 2)
end

@testset "unstructured Mesh accepts abstract array types" begin
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
    ]
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    reference = Mesh(coords, el2n)

    # A view over the coordinates and an Int64 connectivity are a strict
    # superset of the previously required Vector{SVector}/Matrix{Int32} forms.
    coords_view = view(coords, :)
    el2n_int64 = Int64.(el2n)
    loosened = Mesh(coords_view, el2n_int64)

    @test loosened.coords == reference.coords
    @test loosened.el2n == reference.el2n
    @test loosened.Γnodes == reference.Γnodes
    @test eltype(loosened.Γnodes) === Int64
    @test loosened.nnodes === reference.nnodes
    @test loosened.nels === reference.nels
end
