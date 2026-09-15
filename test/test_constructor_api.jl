using Test

using FEMTools
using KernelAbstractions: CPU
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


@testset "element-aware Mesh precomputes geometry" begin
    element = ReferenceElement(LinearElement{2, 3, Float32})
    coords = SVector{2, Float32}[
        SVector(0, 0), SVector(1, 0), SVector(0, 1),
    ]
    el2n = reshape(Int32[1, 2, 3], 3, 1)
    mesh = Mesh(CPU(), coords, el2n, element; workgroup = 1)

    @test mesh.element === element
    @test size(mesh.geometry) == (3, mesh.nels)
    @test eltype(mesh.geometry) == Tuple{SMatrix{3, 2, Float32, 6}, Float32}
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

@testset "unstructured Mesh boundary detection handles high-order paths" begin
    coords = SVector{2, Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
        SVector(1.0, 1.0),
        SVector(0.5, 0.0),
        SVector(0.5, 0.5),
        SVector(0.0, 0.5),
        SVector(1.0, 0.5),
        SVector(0.5, 1.0),
    ]
    el2n = Int32[
        1 2
        2 4
        3 3
        5 8
        6 9
        7 6
    ]
    mesh = Mesh(coords, el2n; order = 2)

    @test mesh.Γnodes == Int32[1, 2, 3, 4, 5, 7, 8, 9]

    coords3 = SVector{3, Float64}[
        SVector(0.0, 0.0, 0.0),
        SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0),
        SVector(0.0, 0.0, 1.0),
        SVector(1.0, 1.0, 1.0),
    ]
    bad_el2n = reshape(Int32[1, 2, 3, 4, 5], 5, 1)
    @test_throws ArgumentError Mesh(coords3, bad_el2n)
end

@testset "unstructured Mesh boundary detection on Q4 and Hex8 grids" begin
    # 2x2 grid of Q4 cells over the unit square: node 5 is the only one not on
    # a boundary edge.
    #   7---8---9
    #   | 3 | 4 |
    #   4---5---6
    #   | 1 | 2 |
    #   1---2---3
    coords = SVector{2, Float64}[SVector(x, y) for y in (0.0, 0.5, 1.0) for x in (0.0, 0.5, 1.0)]
    el2n = Int32[
        1 2 4 5
        2 3 5 6
        5 6 8 9
        4 5 7 8
    ]
    mesh = Mesh(coords, el2n)

    @test mesh.nels == 4
    @test mesh.nnodes == 9
    @test mesh.Γnodes == Int32[1, 2, 3, 4, 6, 7, 8, 9]

    # 2x2x2 grid of Hex8 cells over the unit cube, node ordering x fastest then
    # y then z: the centre node 14 is the only interior one.
    coords3 = SVector{3, Float64}[
        SVector(x, y, z) for z in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0) for x in (0.0, 0.5, 1.0)
    ]
    node(i, j, k) = Int32(i + 3 * (j - 1) + 9 * (k - 1))
    el2n3 = reduce(hcat, [
        Int32[node(a, b, c), node(a + 1, b, c), node(a, b + 1, c), node(a + 1, b + 1, c),
              node(a, b, c + 1), node(a + 1, b, c + 1), node(a, b + 1, c + 1), node(a + 1, b + 1, c + 1)]
            for c in 1:2 for b in 1:2 for a in 1:2
    ])
    mesh3 = Mesh(coords3, el2n3)

    @test mesh3.nels == 8
    @test mesh3.nnodes == 27
    @test mesh3.Γnodes == Int32[n for n in 1:27 if n != 14]
end

@testset "unstructured Mesh rejects unsupported element arities" begin
    coords = SVector{2, Float64}[SVector(Float64(i), 0.0) for i in 1:5]
    bad_el2n = reshape(Int32[1, 2, 3, 4, 5], 5, 1)
    @test_throws "cannot infer 2D boundary edge paths for elements with 5 local nodes" Mesh(coords, bad_el2n)

    coords3 = SVector{3, Float64}[SVector(Float64(i), 0.0, 0.0) for i in 1:5]
    @test_throws "cannot infer 3D boundary face paths for elements with 5 local nodes" Mesh(coords3, bad_el2n)
end
