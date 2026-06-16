"""
    assemble_diffusion_matrices_atomix!(R, ∂R∂T, PC, T, el2n, geo, nels,
                                        element, D, source, do_∂R∂T)

Assemble the heat-diffusion residual `R` using Atomix-backed atomic scatter.

When `do_∂R∂T` is true, also assemble row-sum estimates of the residual
Jacobian into `∂R∂T` and the absolute diagonal contribution into `PC`. The
geometry tuple `geo` is expected to hold precomputed `(∂N∂x, dΩ)` data per
element and quadrature point (see `precompute_geometry_kernel!`).
"""
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

# Atomic variants: all elements run concurrently and scatter their local
# contributions into global arrays. Element coloring is unnecessary here because
# write conflicts are resolved with atomic additions.

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

"""
    integrate_residual(Tloc, geo_el, sloc, D, Nq, Val(N))

Integrate the element residual for a scalar heat-diffusion equation.

`Tloc` and `sloc` are local nodal values of temperature and source,
`geo_el[q]` supplies `(∂N∂x, dΩ)` at quadrature point `q`, and `Nq[q]`
supplies the shape-function values at that same point. The diffusivity `D`
is assumed to be isotropic and element-uniform.

The weak-form contribution at each quadrature point is

    Re_q = -N * s * dΩ - D * (∂N∂x) * (∂N∂x)ᵀ * T * dΩ
"""
@inline function integrate_residual(Tloc, geo_el, sloc, D, Nq, ::Val{N}) where N
    Re = zero(Tloc)
    for q in eachindex(geo_el)
        ∂N∂x, dΩ = geo_el[q]
        Nv = Nq[q]
        tmp   = ∂N∂x' * Tloc
        KTloc = (D * dΩ) * (∂N∂x * tmp)
        Re   += SVector{N}(
            -sloc[i] * Nv[i] * dΩ - KTloc[i] for i in 1:N
        )
    end
    return Re
end

"""
    element_residual(T, source, el2n, geo, D, Nq, iel, Val(N))

Gather element-local nodal values and integrate the heat-diffusion residual.

Returns `(local_nodes, Re)` where `local_nodes` is the global-index `SVector`
for element `iel` and `Re` is the element residual vector ready for global
scatter.
"""
@inline function element_residual(T, source, el2n, geo, D, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    Tloc = SVector{N}(ntuple(i -> T[local_nodes[i]], Val(N)))
    sloc = SVector{N}(ntuple(i -> source[local_nodes[i]], Val(N)))
    Re   = integrate_residual(Tloc, geo_el, sloc, D, Nq, Val(N))
    return local_nodes, Re
end

"""
    element_jacobian(T, source, el2n, geo, D, Nq, iel, Val(N))

Compute per-element Jacobian diagnostics for the heat-diffusion residual.

The full element Jacobian `∂Re/∂Te` is obtained via ForwardDiff. Returns
`(local_nodes, rowsums, diags)` where `rowsums` are Gershgorin-style absolute
row sums and `diags` stores the absolute diagonal entries, both useful for
diagonal preconditioning in pseudo-transient solvers.
"""
@inline function element_jacobian(T, source, el2n, geo, D, Nq, iel, ::Val{N}) where N
    local_nodes = local_nodes_of(el2n, iel, Val(N))
    geo_el = geo[iel]
    Tloc = SVector{N}(ntuple(i -> T[local_nodes[i]], Val(N)))
    sloc = SVector{N}(ntuple(i -> source[local_nodes[i]], Val(N)))
    ∂Re∂He = ForwardDiff.jacobian(
        Tloc -> integrate_residual(Tloc, geo_el, sloc, D, Nq, Val(N)),
        Tloc
    )
    rowsums = SVector{N}(ntuple(i -> sum(abs(∂Re∂He[i, j]) for j in 1:N), Val(N)))
    diags   = SVector{N}(ntuple(i -> abs(∂Re∂He[i, i]), Val(N)))
    return local_nodes, rowsums, diags
end
