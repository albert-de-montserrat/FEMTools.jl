function assemble_diffusion_matrices_atomix!(R, ∂R∂T, PC, T, el2n, geo, nels, element::ReferenceElement{T}, D, source, do_∂R∂T) where T<:AbstractElement{2, N} where N
    Nq = shape_function_values(element)

    fill!(R, 0)
    residual_atomic_kernel!(backend, workgroup)(R, T, source, el2n, geo, D, Nq, Val(N); ndrange = nels)
    if do_∂R∂T
        fill!(∂R∂T, 0)
        fill!(PC, 0)
        jacobian_atomic_kernel!(backend, workgroup)(∂R∂T, PC, T, source, el2n, geo, D, Nq, Val(N); ndrange = nels)
    end
    KA.synchronize(backend)
end

# --- atomic variants: all elements run concurrently, scatter with atomics ---

@kernel function residual_atomic_kernel!(R, @Const(T), @Const(source), @Const(el2n), @Const(geo), D, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, Re = element_residual(T, source, el2n, geo, D, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic R[inod] += Re[i]
    end
end

@kernel function jacobian_atomic_kernel!(∂R∂T, PC, @Const(T), @Const(source), @Const(el2n), @Const(geo), D, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, rowsums, diags = element_jacobian(T, source, el2n, geo, D, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic ∂R∂T[inod] += rowsums[i]
        Atomix.@atomic :monotonic PC[inod]   += diags[i]
    end
end

# (∂N∂x_q, dΩ_q) for every element and quadrature point. Depends only on the
# mesh, so it is computed once and reused across all PT iterations.
@kernel function precompute_geometry_kernel!(geo, @Const(coords), @Const(el2n), ∂N∂ξq, ω, ::Val{N}) where N
    iel = @index(Global)
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    c = element_coordinate_matrix(coords, local_nodes)
    geo[iel] = ntuple(Val(length(ω))) do q
        J = ∂N∂ξq[q]' * c
        (∂N∂ξq[q] * inv(J), abs(det(J)) * ω[q])
    end
end

# numerical integration of the residual
@inline function integrate_residual(Hloc, geo_el, sloc, D, Nq, ::Val{N}) where N
    Re = zero(Hloc)
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv = Nq[q]
        tmp   = ∂N∂x' * Hloc
        KHloc = (D * dΩ) * (∂N∂x * tmp)
        Re   += SVector{N}(
            -sloc[i] * Nv[i] * dΩ - KHloc[i] for i in 1:N
        )
    end
    return Re
end

# shared per-element work: gather, integrate, return (nodes, Re)
@inline function element_residual(T, source, el2n, geo, D, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    Hloc = SVector{N}(ntuple(i -> T[local_nodes[i]], Val(N)))
    sloc = SVector{N}(ntuple(i -> source[local_nodes[i]], Val(N)))
    Re   = integrate_residual(Hloc, geo_el, sloc, D, Nq, Val(N))
    return local_nodes, Re
end

# shared per-element Jacobian: Gershgorin row sums + |diagonal| of ∂Re/∂He via ForwardDiff
@inline function element_jacobian(T, source, el2n, geo, D, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    Hloc = SVector{N}(ntuple(i -> T[local_nodes[i]], Val(N)))
    sloc = SVector{N}(ntuple(i -> source[local_nodes[i]], Val(N)))
    ∂Re∂He = ForwardDiff.jacobian(
        Hloc -> integrate_residual(Hloc, geo_el, sloc, D, Nq, Val(N)),
        Hloc
    )
    rowsums = SVector{N}(ntuple(i -> sum(abs(∂Re∂He[i, j]) for j in 1:N), Val(N)))
    diags   = SVector{N}(ntuple(i -> abs(∂Re∂He[i, i]), Val(N)))
    return local_nodes, rowsums, diags
end