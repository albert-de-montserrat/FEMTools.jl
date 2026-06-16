"""
    assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H, el2n, geo, nels,
                                        element, D, source, do_∂R∂H)

Assemble the diffusion residual `R` with Atomix-backed atomic scatter.

When `do_∂R∂H` is true, also assemble row-sum estimates of the residual
Jacobian into `∂R∂H` and the absolute diagonal contribution into `PC`. The
geometry tuple `geo` is expected to contain precomputed `(∂N∂x, dΩ)` data per
element and quadrature point.
"""
function assemble_diffusion_matrices_atomix!(R, ∂R∂H, PC, H, el2n, geo, nels, element::ReferenceElement{T}, D, source, do_∂R∂H) where T<:AbstractElement{2, N} where N
    Nq = shape_function_values(element)

    fill!(R, 0)
    residual_atomic_kernel!(backend, workgroup)(R, H, source, el2n, geo, D, Nq, Val(N); ndrange = nels)
    if do_∂R∂H
        fill!(∂R∂H, 0)
        fill!(PC, 0)
        jacobian_atomic_kernel!(backend, workgroup)(∂R∂H, PC, H, source, el2n, geo, D, Nq, Val(N); ndrange = nels)
    end
    KA.synchronize(backend)
end

# Atomic variants: all elements run concurrently and scatter their local
# contributions into global arrays. Element coloring is unnecessary here because
# write conflicts are resolved with atomic additions.

@kernel function residual_atomic_kernel!(R, @Const(H), @Const(source), @Const(el2n), @Const(geo), D, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, Re = element_residual(H, source, el2n, geo, D, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic R[inod] += Re[i]
    end
end

@kernel function jacobian_atomic_kernel!(∂R∂H, PC, @Const(H), @Const(source), @Const(el2n), @Const(geo), D, Nq, ::Val{N}) where N
    iel = @index(Global)
    local_nodes, rowsums, diags = element_jacobian(H, source, el2n, geo, D, Nq, iel, Val(N))
    for (i, inod) in enumerate(local_nodes)
        Atomix.@atomic :monotonic ∂R∂H[inod] += rowsums[i]
        Atomix.@atomic :monotonic PC[inod]   += diags[i]
    end
end

"""
    precompute_geometry_kernel!(geo, coords, el2n, ∂N∂ξq, ω, Val(N))

KernelAbstractions kernel that fills `geo` with per-element geometry data.

For each element `iel`, computes `(∂N∂x_q, dΩ_q)` at every quadrature point `q`
and stores the result as a tuple at `geo[iel]`. Here `∂N∂x_q` is the matrix of
physical-space shape-function gradients (`N × nDim`) and `dΩ_q` is the
quadrature weight scaled by `|det J|`.

Because this kernel depends only on mesh geometry, it only needs to be called
once per mesh and the result can be reused across nonlinear or pseudo-transient
iterations.
"""
@kernel function precompute_geometry_kernel!(geo, @Const(coords), @Const(el2n), ∂N∂ξq, ω, ::Val{N}) where N
    iel = @index(Global)
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    c = element_coordinate_matrix(coords, local_nodes)
    geo[iel] = ntuple(Val(length(ω))) do q
        J = ∂N∂ξq[q]' * c
        (∂N∂ξq[q] * inv(J), abs(det(J)) * ω[q])
    end
end

"""
    integrate_residual(Hloc, geo_el, sloc, D, Nq, Val(N))

Integrate the element residual for a scalar diffusion equation.

`Hloc` and `sloc` are local nodal values, `geo_el[q]` supplies `(∂N∂x, dΩ)` at
quadrature point `q`, and `Nq[q]` supplies the shape-function values at that
same point.
"""
@inline function integrate_residual(Hloc, geo_el, sloc, D, Nq, ::Val{N}) where N
    Re = zero(Hloc)
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv = Nq[q]
        # Diffusive flux contribution, evaluated through the element-local
        # gradient of the nodal field.
        tmp   = ∂N∂x' * Hloc
        KHloc = (D * dΩ) * (∂N∂x * tmp)
        Re   += SVector{N}(ntuple(i -> -sloc[i] * Nv[i] * dΩ - KHloc[i], Val(N)))
    end
    return Re
end

"""
    element_residual(H, source, el2n, geo, D, Nq, iel, Val(N))

Gather element-local nodal values, integrate the residual, and return
`(local_nodes, Re)` for global scatter.
"""
@inline function element_residual(H, source, el2n, geo, D, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    Hloc = SVector{N}(ntuple(i -> H[local_nodes[i]], Val(N)))
    sloc = SVector{N}(ntuple(i -> source[local_nodes[i]], Val(N)))
    Re   = integrate_residual(Hloc, geo_el, sloc, D, Nq, Val(N))
    return local_nodes, Re
end

"""
    element_jacobian(H, source, el2n, geo, D, Nq, iel, Val(N))

Compute per-element Jacobian diagnostics for the diffusion residual.

The full element Jacobian `∂Re/∂He` is obtained by ForwardDiff. The returned
row sums are Gershgorin-style absolute row sums, while `diags` stores the
absolute diagonal entries for preconditioning.
"""
@inline function element_jacobian(H, source, el2n, geo, D, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    Hloc = SVector{N}(ntuple(i -> H[local_nodes[i]], Val(N)))
    sloc = SVector{N}(ntuple(i -> source[local_nodes[i]], Val(N)))
    ∂Re∂He = ForwardDiff.jacobian(
        Hloc -> integrate_residual(Hloc, geo_el, sloc, D, Nq, Val(N)),
        Hloc
    )
    rowsums = SVector{N}(ntuple(i -> sum(abs(∂Re∂He[i, j]) for j in 1:N), Val(N)))
    diags   = SVector{N}(ntuple(i -> abs(∂Re∂He[i, i]), Val(N)))
    return local_nodes, rowsums, diags
end
