"""
    interp2ip(N, v)

Interpolate nodal values `v` to one integration point using the shape-function
weights `N`.

Both `N` and `v` are `SVector`s with the same length. The result has the same
scalar type as the entries of `N`.

# Examples
```jldoctest
julia> using StaticArrays

julia> FEMTools.interp2ip(SVector(0.25, 0.25, 0.5), SVector(1.0, 2.0, 4.0))
2.75
```
"""
@generated function interp2ip(N::SVector{M, T}, v::SVector{M, T}) where {M, T}
    quote
        @inline
        out = zero(T)
        Base.@nexprs $M i -> out += N[i] * v[i]
        out
    end
end

"""
    interp2ip_phase(N, var, phase)

Interpolate a per-phase material property to one integration point.

`N` is the `SVector{M}` of shape-function values at the quadrature point.
`var` is an `NTuple` of per-phase scalars. `phase` is an `SVector{M, Int}` of
per-node phase indices (1-based). At node `i`, `var[phase[i]]` is the property
value, and it is weighted by `N[i]` and accumulated.
"""
@generated function interp2ip_phase(N::SVector{M, T}, var, phase) where {M, T}
    quote
        @inline
        out = zero(T)
        Base.@nexprs $M i -> out += N[i] * var[phase[i]]
        out
    end
end

"""
    interp2ip(N, f, args)

Evaluate `f` at each node using nodal argument vectors in `args`, then
interpolate those nodal values to one integration point with weights `N`.

`args` must be a tuple of `SVector`s whose entries are aligned by node. At node
`i`, `f(args[1][i], args[2][i], ...)` is multiplied by `N[i]` and accumulated.
"""
@generated function interp2ip(N::SVector{M, T}, f::F, args::NTuple{A, SVector}) where {M, F, A, T}
    quote
        @inline
        out = zero(T)
        Base.@nexprs $M i -> out += begin
            argsᵢ = Base.@ntuple $A j -> getindex(args[j], i)
            N[i] * f(argsᵢ...)
        end
        out
    end
end

@inline _gather_local(arr, nodes, ::Val{N}) where N =
    SVector{N}(ntuple(i -> arr[nodes[i]], Val(N)))

@inline _phase_at(phases::AbstractMatrix, _, i, iel) =
    Int(phases[size(phases, 1) == 1 ? 1 : i, iel])
@inline _phase_at(phases, nodes, i, _) = Int(phases[nodes[i]])
@inline _gather_phase(phases, nodes, iel, ::Val{N}) where N =
    SVector{N}(ntuple(i -> _phase_at(phases, nodes, i, iel), Val(N)))

@inline function _add_local!(dest, nodes, values, ::Val{false})
    for (i, node) in enumerate(nodes)
        dest[node] += values[i]
    end
    return nothing
end

@inline function _add_local!(dest, nodes, values, ::Val{true})
    for (i, node) in enumerate(nodes)
        Atomix.@atomic :monotonic dest[node] += values[i]
    end
    return nothing
end

@inline function _add_local_pair!(dest_a, dest_b, nodes, values_a, values_b, ::Val{false})
    for (i, node) in enumerate(nodes)
        dest_a[node] += values_a[i]
        dest_b[node] += values_b[i]
    end
    return nothing
end

@inline function _add_local_pair!(dest_a, dest_b, nodes, values_a, values_b, ::Val{true})
    for (i, node) in enumerate(nodes)
        Atomix.@atomic :monotonic dest_a[node] += values_a[i]
        Atomix.@atomic :monotonic dest_b[node] += values_b[i]
    end
    return nothing
end

function _checked_λmax(jacobian, PC, label)
    λmax = maximum(jacobian ./ PC)
    if !(isfinite(λmax) && λmax > zero(λmax))
        error("$label preconditioner produced invalid λmax = $λmax; check for zero or non-finite preconditioner entries")
    end
    return λmax
end
