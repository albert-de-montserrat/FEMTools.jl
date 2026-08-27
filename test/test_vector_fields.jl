using KernelAbstractions: CPU

using FEMTools: AbstractVectorField, VectorField, VectorField2D, VectorField3D

@testset "VectorField construction" begin
    v2 = VectorField2D(CPU(), FP64, 4)
    v3 = VectorField3D(CPU(), 4)

    @test v2 isa VectorField2D{Vector{FP64}}
    @test v3 isa VectorField3D{Vector{FP64}}
    @test v2 isa AbstractVectorField{Vector{FP64}}
    @test all(iszero, v2.x) && all(iszero, v2.y)
    @test VectorField2D(CPU(), FP32, 4) isa VectorField2D{Vector{FP32}}
    @test length(VectorField3D(CPU(), FP32, 7).z) == 7

    # components supplied directly, and the parametric form
    x, y, z = ntuple(_ -> rand(4), 3)
    @test VectorField2D(x, y).y === y
    @test VectorField3D(x, y, z).z === z
    @test VectorField2D{Vector{FP64}}(x, y) isa VectorField2D{Vector{FP64}}

    # the dimension-agnostic factory picks the concrete type from the arity
    @test VectorField(x, y) === VectorField2D(x, y)
    @test VectorField(x, y, z) === VectorField3D(x, y, z)
    @test_throws "2D (2 components) and 3D (3 components)" VectorField(x)
    @test_throws ArgumentError VectorField(x, y, z, x)

    # components must share a storage type
    @test_throws MethodError VectorField(rand(4), rand(2, 2))
end

@testset "VectorField indexing" begin
    v2 = VectorField2D(CPU(), FP64, 4)
    v3 = VectorField3D(CPU(), FP64, 4)

    @test eltype(v2) === FP64
    @test eltype(VectorField3D(CPU(), FP32, 4)) === FP32
    @test size(v2) == (4,)
    @test length(v3) == 4
    @test axes(v2) == (Base.OneTo(4),)
    @test collect(eachindex(v3)) == 1:4
    @test firstindex(v2) == 1 && lastindex(v2) == 4

    # writes accept any indexable of the right length
    v2[2] = SA[1.0, 2.0]
    @test v2[2] == SA[1.0, 2.0]
    v2[3] = (3.0, 4.0)
    @test v2[3] == SA[3.0, 4.0]
    @test v2[end] == SA[0.0, 0.0]

    v3[1] = SA[1.0, 2.0, 3.0]
    @test v3[1] == SA[1.0, 2.0, 3.0]
    @test v3.z[1] == 3.0

    # a GPU backend supplies Int32 indices
    @test v3[Int32(1)] == v3[1]

    # integration-point storage is indexed by (quadrature point, element)
    B = VectorField(zeros(2, 3), zeros(2, 3))
    B[2, 3] = SA[7.0, 8.0]
    @test B[2, 3] == SA[7.0, 8.0]
    @test B.x[2, 3] == 7.0
end

@testset "VectorField copy, unpack, and display" begin
    src = VectorField2D(fill(1.0, 2), fill(2.0, 2))
    dst = VectorField2D(CPU(), FP64, 2)
    copyto!(dst, src)
    @test Tuple(dst) == Tuple(src)

    src3 = VectorField3D(fill(1.0, 2), fill(2.0, 2), fill(3.0, 2))
    dst3 = VectorField3D(CPU(), FP64, 2)
    copyto!(dst3, src3)
    @test Tuple(dst3) == Tuple(src3)

    @test Tuple(VectorField(1.0, 2.0)) === (1.0, 2.0)
    @test Tuple(VectorField(1.0, 2.0, 3.0)) === (1.0, 2.0, 3.0)

    @test sprint(show, VectorField(1.0, 2.0)) == "VectorField2D(x=1.0, y=2.0)"
    @test sprint(show, VectorField(1.0, 2.0, 3.0)) == "VectorField3D(x=1.0, y=2.0, z=3.0)"
    @test sprint(show, VectorField(zeros(2, 3), zeros(2, 3))) ==
        "VectorField2D(x=2×3 Matrix{Float64}, y=2×3 Matrix{Float64})"
end

@testset "VectorField stays on the stack" begin
    # measured through a function barrier: at global scope `@allocated` also
    # counts compilation
    component(A, i) = A[i]
    store!(A, v, i) = A[i] = v

    v2 = VectorField2D(CPU(), FP64, 4)
    v3 = VectorField3D(CPU(), FP64, 4)
    component(v2, 1); component(v3, 1)
    store!(v2, SA[1.0, 2.0], 1); store!(v3, SA[1.0, 2.0, 3.0], 1)

    @test @allocated(component(v2, 2)) == 0
    @test @allocated(component(v3, 2)) == 0
    @test @allocated(store!(v2, SA[1.0, 2.0], 2)) == 0
    @test @allocated(store!(v3, SA[1.0, 2.0, 3.0], 2)) == 0
end
