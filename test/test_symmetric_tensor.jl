using KernelAbstractions: CPU

using FEMTools: AbstractSymmetricTensor, AbstractTensor, AbstractVoigtTensor,
    SymmetricTensor, SymmetricTensor2D, SymmetricTensor3D,
    VoigtTensor, VoigtTensor2D, VoigtTensor3D

@testset "SymmetricTensor construction" begin
    A2 = SymmetricTensor2D(CPU(), FP64, 4)
    A3 = SymmetricTensor3D(CPU(), 4)

    @test A2 isa SymmetricTensor2D{Vector{FP64}}
    @test A3 isa SymmetricTensor3D{Vector{FP64}}
    @test A2 isa AbstractSymmetricTensor{Vector{FP64}} <: AbstractTensor{Vector{FP64}}
    @test all(iszero, A2.xx) && all(iszero, A2.II)
    @test SymmetricTensor2D(CPU(), FP32, 4) isa SymmetricTensor2D{Vector{FP32}}

    # components supplied directly, and the parametric form
    xx, yy, xy, II = ntuple(_ -> rand(4), 4)
    @test SymmetricTensor2D(xx, yy, xy, II).II === II
    @test SymmetricTensor2D{Vector{FP64}}(xx, yy, xy, II) isa SymmetricTensor2D{Vector{FP64}}
    @test SymmetricTensor3D(ntuple(_ -> rand(4), 7)...) isa SymmetricTensor3D{Vector{FP64}}

    # `SymmetricTensor` zeroes the invariant slot
    @test SymmetricTensor(1.0, 2.0, 3.0) === SymmetricTensor2D(1.0, 2.0, 3.0, 0.0)
    @test SymmetricTensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0) ===
        SymmetricTensor3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0)
    @test SymmetricTensor(xx, yy, xy).II == zero(xx)
end

@testset "SymmetricTensor indexing" begin
    A2 = SymmetricTensor2D(CPU(), FP64, 4)
    A3 = SymmetricTensor3D(CPU(), FP64, 4)

    @test eltype(A2) === FP64
    @test eltype(SymmetricTensor2D(CPU(), FP32, 4)) === FP32
    @test size(A2) == (4,)
    @test length(A3) == 4
    @test axes(A2) == (Base.OneTo(4),)
    @test collect(eachindex(A3)) == 1:4
    @test firstindex(A2) == 1 && lastindex(A2) == 4

    # component writes go through Voigt-ordered vectors, tensors, or tuples
    A2[2] = SA[1.0, 2.0, 3.0]
    @test A2[2] == SA[1.0, 2.0, 3.0]
    @test A2[end] == SA[0.0, 0.0, 0.0]
    A2[3] = VoigtTensor(4.0, 5.0, 6.0)
    @test A2[3] == SA[4.0, 5.0, 6.0]

    A3[1] = SA[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    @test A3[1] == SA[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    @test VoigtTensor(A3, 1) === VoigtTensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    # a GPU backend supplies Int32 indices
    @test A3[Int32(1)] == A3[1]
    @test VoigtTensor(A2, Int32(2)) === VoigtTensor(1.0, 2.0, 3.0)

    # writes leave the derived invariant slot alone
    @test all(iszero, A2.II)

    # integration-point storage is indexed by (quadrature point, element)
    B = SymmetricTensor(zeros(2, 3), zeros(2, 3), zeros(2, 3))
    B[2, 3] = SA[7.0, 8.0, 9.0]
    @test B[2, 3] == SA[7.0, 8.0, 9.0]
    @test B.xx[2, 3] == 7.0
end

@testset "SymmetricTensor copy, unpack, and display" begin
    src = SymmetricTensor2D(fill(1.0, 2), fill(2.0, 2), fill(3.0, 2), fill(4.0, 2))
    dst = SymmetricTensor2D(CPU(), FP64, 2)
    copyto!(dst, src)
    @test Tuple(dst) == Tuple(src)
    @test dst.II == src.II

    src3 = SymmetricTensor3D(ntuple(i -> fill(Float64(i), 2), 7)...)
    dst3 = SymmetricTensor3D(CPU(), FP64, 2)
    copyto!(dst3, src3)
    @test Tuple(dst3) == Tuple(src3)

    @test Tuple(SymmetricTensor(1.0, 2.0, 3.0)) === (1.0, 2.0, 3.0)
    @test Tuple(SymmetricTensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)) === (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    @test sprint(show, SymmetricTensor(1.0, 2.0, 3.0)) ==
        "SymmetricTensor2D(xx=1.0, yy=2.0, xy=3.0, II=0.0)"
    @test sprint(show, SymmetricTensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)) ==
        "SymmetricTensor3D(xx=1.0, yy=2.0, zz=3.0, yz=4.0, xz=5.0, xy=6.0, II=0.0)"
    @test sprint(show, SymmetricTensor(zeros(2, 3), zeros(2, 3), zeros(2, 3))) ==
        "SymmetricTensor2D(xx=2×3 Matrix{Float64}, yy=2×3 Matrix{Float64}, xy=2×3 Matrix{Float64}, II=2×3 Matrix{Float64})"
end

@testset "VoigtTensor construction" begin
    @test VoigtTensor(1.0, 2.0, 3.0) isa VoigtTensor2D{FP64}
    @test VoigtTensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0) isa VoigtTensor3D{FP64}
    @test VoigtTensor(SA[1.0, 2.0, 3.0]) === VoigtTensor(1.0, 2.0, 3.0)

    # mixed argument types promote
    @test VoigtTensor2D(1, 0.0, 0.0) === VoigtTensor2D{FP64}(1.0, 0.0, 0.0)
    @test VoigtTensor(1, 2, 3) isa VoigtTensor2D{Int}
    @test VoigtTensor3D(1, 2, 3, 4, 5, 6.0f0) isa VoigtTensor3D{FP32}

    # the parametric form converts rather than demanding a matching type
    @test VoigtTensor2D{FP64}(1, 0, 0) === VoigtTensor2D(1.0, 0.0, 0.0)
    @test VoigtTensor3D{FP32}(1, 2, 3, 4, 5, 6) isa VoigtTensor3D{FP32}

    @test_throws "2D (3 components) and 3D (6 components)" VoigtTensor(1.0, 2.0, 3.0, 4.0)
    @test_throws ArgumentError VoigtTensor(SA[1.0, 2.0, 3.0, 4.0])
end

@testset "VoigtTensor conversions" begin
    a = VoigtTensor(1.0, 2.0, 3.0)
    b = VoigtTensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    @test SVector(a) === SA[1.0, 2.0, 3.0]
    @test SVector(b) === SA[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    @test MVector(a) == SVector(a)
    @test MVector(b) == SVector(b)

    @test SMatrix(a) === SA[1.0 3.0; 3.0 2.0]
    @test SMatrix(b) === SA[1.0 6.0 5.0; 6.0 2.0 4.0; 5.0 4.0 3.0]
    @test SMatrix(a) == SMatrix(a)'
    @test SMatrix(b) == SMatrix(b)'
    @test MMatrix(a) == SMatrix(a)
    @test MMatrix(b) == SMatrix(b)
end

@testset "VoigtTensor arithmetic" begin
    a = VoigtTensor(1.0, 2.0, 3.0)
    b = VoigtTensor(4.0, 5.0, 6.0)

    @test a + b === VoigtTensor(5.0, 7.0, 9.0)
    @test b - a === VoigtTensor(3.0, 3.0, 3.0)
    @test -a === VoigtTensor(-1.0, -2.0, -3.0)
    @test 2 * a === VoigtTensor(2.0, 4.0, 6.0)
    @test a * 2 === 2 * a
    @test a / 2 === VoigtTensor(0.5, 1.0, 1.5)

    # operands of different element types promote
    @test a + VoigtTensor(1.0f0, 1.0f0, 1.0f0) === VoigtTensor(2.0, 3.0, 4.0)
    @test 2.0f0 * VoigtTensor(1, 2, 3) isa VoigtTensor2D{FP32}

    c = VoigtTensor(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    @test c + c === 2 * c
    @test SVector(c - c) == zeros(SVector{6})

    # matrix products and solves agree with the dense form
    x = SA[1.0, 2.0]
    @test a * x ≈ SMatrix(a) * x
    @test a * b ≈ SMatrix(a) * SMatrix(b)
    @test a \ x ≈ SMatrix(a) \ x
    @test a * (a \ x) ≈ x

    y = SA[1.0, 2.0, 3.0]
    @test c * y ≈ SMatrix(c) * y
    @test c \ y ≈ SMatrix(c) \ y

    # the product of two symmetric tensors is generally not symmetric
    @test a * b != (a * b)'
end

@testset "VoigtTensor stays on the stack" begin
    a = VoigtTensor(1.0, 2.0, 3.0)
    x = @SVector rand(2)
    m = @MVector rand(2)

    # a mutable result would be heap-allocated, so every product returns an SVector
    @test (a * m) isa SVector
    @test (a \ m) isa SVector

    # measured through a function barrier: at global scope `@allocated` also
    # counts compilation
    mul(A, v) = A * v
    div(A, v) = A \ v
    add(A, B) = A + B
    scale(alpha, A) = alpha * A
    component(A, i) = A[i]
    voigt(A, i) = VoigtTensor(A, i)

    A3 = SymmetricTensor3D(CPU(), FP64, 4)
    for f in (mul, div), v in (x, m)
        f(a, v)
    end
    add(a, a); scale(2, a); component(A3, 2); voigt(A3, 2)

    @test @allocated(mul(a, x)) == 0
    @test @allocated(mul(a, m)) == 0
    @test @allocated(div(a, m)) == 0
    @test @allocated(add(a, a)) == 0
    @test @allocated(scale(2, a)) == 0
    @test @allocated(component(A3, 2)) == 0
    @test @allocated(voigt(A3, 2)) == 0
end
