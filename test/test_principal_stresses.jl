using Test
using FEMTools
using LinearAlgebra
using StaticArrays

_principal_test_components(A::AbstractMatrix) = size(A, 1) == 2 ?
    (A[1, 1], A[2, 2], A[1, 2]) :
    (A[1, 1], A[2, 2], A[3, 3], A[1, 2], A[1, 3], A[2, 3])

function _principal_test_allocations(τ)
    FEMTools._principal_eigenpairs(τ)
    return @allocated FEMTools._principal_eigenpairs(τ)
end

function _principal_test_tensors(::Type{T}, D) where {T}
    Q = D == 2 ? T[3 -4; 4 3] / T(5) : T[1 2 -2; 2 1 2; 2 -2 -1] / T(3)
    diagonal = D == 2 ? T[3, -2] : T[3, -2, 1]
    repeated = D == 2 ? T[2, 2] : T[2, 2, -1]
    near = copy(repeated)
    near[2] += T(16) * eps(T)
    rotated = Q * Diagonal(diagonal) * Q'
    shear = zeros(T, D, D)
    shear[1, 2] = shear[2, 1] = -T(2)
    small = sqrt(floatmin(T))
    large = sqrt(floatmax(T))
    tensors = [
        zeros(T, D, D), Matrix{T}(I, D, D) * T(2),
        Matrix(Diagonal(diagonal)), rotated, shear,
        Q * Diagonal(repeated) * Q', Q * Diagonal(near) * Q',
        rotated * small, rotated * large,
        Matrix(Diagonal(fill(T(-2), D))),
    ]
    # Deviatoric plane strain: the out-of-plane value is the largest.
    D == 3 && push!(tensors, T[-2 0 0; 0 -1 0; 0 0 3])
    return tensors
end

# Optional hardware gate: load the backend package, disable scalar indexing,
# then call this with CUDA.CuArray.
function _principal_stress_backend_tests(to_backend; types = (Float32, Float64))
    @testset "principal stress accelerator" begin
        for T in types, D in (2, 3)
            tensors = _principal_test_tensors(T, D)
            samples = map(_principal_test_components, tensors)
            τ_cpu = ntuple(c -> T[s[c] for s in samples], D == 2 ? 3 : 6)
            τ = map(to_backend, τ_cpu)
            out = compute_principal_stresses(τ; workgroup = 32)
            host(out) = (;
                values = map(Array, out.values),
                directions = map(v -> map(Array, v), out.directions),
            )
            _check_principal_test_eigenpairs(host(out), tensors, T, D)
            P_cpu = fill(T(7), length(tensors))
            P = to_backend(P_cpu)
            @test compute_principal_stresses!(out, τ; pressure = P) === out
            expected = compute_principal_stresses(τ_cpu; pressure = P_cpu)
            actual = host(out)
            @test all(k -> actual.values[k] ≈ expected.values[k], 1:D)
            matrix_τ = map(a -> reshape(a, 1, :), τ)
            matrix_out = compute_principal_stresses(matrix_τ)
            @test all(a -> size(a) == (1, length(tensors)), matrix_out.values)
            @test_throws ArgumentError compute_principal_stresses(τ; pressure = P_cpu)
            @test out isa PrincipalStresses
            @test_throws ArgumentError compute_principal_stresses!(PrincipalStresses((τ[1], out.values[2:end]...), out.directions), τ)
            @test_throws ArgumentError compute_principal_stresses!(PrincipalStresses((out.values[1], out.values[1], out.values[3:end]...), out.directions), τ)
            bad = map(copy, τ_cpu)
            bad[end][1] = T(NaN)
            @test_throws DomainError compute_principal_stresses!(out, map(to_backend, bad))
            @test isnan(first(Array(out.values[1])))
        end
    end
    return nothing
end

function _check_principal_test_eigenpairs(out, tensors, ::Type{T}, D) where {T}
    tol = T(128) * eps(T)
    for (i, A) in enumerate(tensors)
        λ = T[a[i] for a in out.values]
        V = T[out.directions[k][c][i] for c in 1:D, k in 1:D]
        scale = max(maximum(abs, A), floatmin(T))
        B = A / scale
        μ = λ / scale
        oracle = eigen(Symmetric(B))
        @test issorted(λ; rev = true)
        @test μ ≈ reverse(oracle.values) atol = tol rtol = tol
        @test B * V ≈ V * Diagonal(μ) atol = tol rtol = tol
        @test V' * V ≈ Matrix{T}(I, D, D) atol = tol rtol = tol
        @test V * Diagonal(μ) * V' ≈ B atol = tol rtol = tol
        for k in 1:D
            pivot = argmax(abs.(V[:, k]))
            @test V[pivot, k] >= zero(T)
            # A repeated eigenvalue defines a subspace, not a unique vector.
            cluster = findall(x -> abs(x - μ[k]) <= tol, oracle.values)
            @test norm(V[:, k] - oracle.vectors[:, cluster] * (oracle.vectors[:, cluster]' * V[:, k])) <= T(8) * tol
        end
    end
end

@testset "principal stress eigenpairs" begin
    for T in (Float32, Float64), D in (2, 3)
        tensors = _principal_test_tensors(T, D)
        samples = map(_principal_test_components, tensors)
        τ = ntuple(c -> T[s[c] for s in samples], D == 2 ? 3 : 6)
        out = compute_principal_stresses(τ; workgroup = 4)
        @test out isa PrincipalStresses
        @test PrincipalStresses(out.values, out.directions).values === out.values
        @test all(a -> eltype(a) === T && size(a) == size(τ[1]), out.values)
        _check_principal_test_eigenpairs(out, tensors, T, D)
        @test compute_principal_stresses!(out, τ) === out

        tensor = D == 2 ? FEMTools.SymmetricTensor2D(τ..., similar(τ[1], 0)) :
            FEMTools.SymmetricTensor3D(τ[1], τ[2], τ[3], τ[6], τ[5], τ[4], similar(τ[1], 0))
        tensor_out = compute_principal_stresses(tensor)
        @test tensor_out.values == out.values
        @test tensor_out.directions == out.directions
        local_result = @inferred FEMTools._principal_eigenpairs(samples[4])
        @test local_result[3]
        @test _principal_test_allocations(samples[4]) == 0

        P = fill(T(7), length(tensors))
        shifted = compute_principal_stresses(τ; pressure = P)
        @test shifted.values == map(a -> a .- P, out.values)
        @test shifted.directions == out.directions
        value_buffers, direction_buffers = out.values, out.directions
        @test compute_principal_stresses!(out, τ; pressure = P) === out
        @test out.values === value_buffers
        @test out.directions === direction_buffers
        @test out.values == shifted.values
        @test out.directions == shifted.directions
        # Large pressure must not destroy direction information.
        huge = compute_principal_stresses(τ; pressure = fill(sqrt(floatmax(T)), length(P)))
        @test huge.directions == out.directions

        matrix_τ = map(a -> reshape(copy(a), 1, :), τ)
        matrix_out = compute_principal_stresses(matrix_τ; pressure = reshape(P, 1, :))
        @test matrix_out.values == map(a -> reshape(a, 1, :), shifted.values)
        @test matrix_out.directions == map(v -> map(a -> reshape(a, 1, :), v), shifted.directions)

        empty_τ = map(a -> similar(a, 0, 2), τ)
        empty_out = compute_principal_stresses(empty_τ)
        @test all(a -> size(a) == (0, 2) && eltype(a) === T, empty_out.values)

        for bad in (T(NaN), T(Inf), T(-Inf))
            bad_τ = map(copy, τ)
            bad_τ[end][1] = bad
            @test_throws DomainError compute_principal_stresses!(out, bad_τ)
            @test all(a -> isnan(a[1]), out.values)
            @test all(v -> all(a -> isnan(a[1]), v), out.directions)
            @test_throws DomainError compute_principal_stresses(τ; pressure = fill(bad, length(P)))
        end
        overflow = ntuple(_ -> fill(floatmax(T), 1), length(τ))
        @test_throws DomainError compute_principal_stresses(overflow)
    end
end

@testset "principal stress validation before mutation" begin
    τ = ([2.0, 3.0], [-1.0, -2.0], [0.5, 0.0])
    out = compute_principal_stresses(τ)
    @test_throws ArgumentError compute_principal_stresses(τ[1:2])
    @test_throws ArgumentError compute_principal_stresses((1.0, 2.0, 3.0))
    @test_throws ArgumentError compute_principal_stresses(map(a -> Int.(round.(a)), τ))
    @test_throws ArgumentError compute_principal_stresses((Float32.(τ[1]), τ[2], τ[3]))
    @test_throws DimensionMismatch compute_principal_stresses((τ[1], τ[2], [0.0]))
    @test_throws DimensionMismatch compute_principal_stresses(τ; pressure = zeros(2, 1))
    @test_throws ArgumentError compute_principal_stresses(τ; pressure = zeros(Float32, 2))
    @test_throws ArgumentError compute_principal_stresses(τ; pressure = 1.0)
    @test_throws ArgumentError compute_principal_stresses(τ; workgroup = 0)
    @test_throws ArgumentError compute_principal_stresses(τ; workgroup = 1.5)
    @test_throws DimensionMismatch compute_principal_stresses!(PrincipalStresses(out.values[1:1], out.directions), τ)
    @test_throws DimensionMismatch compute_principal_stresses!(PrincipalStresses(out.values, ((out.directions[1][1],), out.directions[2])), τ)
    @test_throws ArgumentError compute_principal_stresses!(PrincipalStresses((τ[1], out.values[2]), out.directions), τ)
    @test_throws ArgumentError compute_principal_stresses!(PrincipalStresses((out.values[1], out.values[1]), out.directions), τ)
    @test_throws ArgumentError compute_principal_stresses!(out, τ; pressure = out.directions[1][1])
    aliased = @view τ[1][:]
    @test_throws ArgumentError compute_principal_stresses!(PrincipalStresses((aliased, out.values[2]), out.directions), τ)
    expected = compute_principal_stresses(τ)
    @test out.values == expected.values
    @test out.directions == expected.directions
end
