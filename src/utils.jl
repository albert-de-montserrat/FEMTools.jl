"""
    interp2ip(N, v)

Interpolate nodal values `v` to one integration point using the shape-function
weights `N`.

Both `N` and `v` are `SVector`s with the same length. The result has the same
scalar type as the entries of `N`.
"""
@generated function interp2ip(N::SVector{M, T}, v::SVector{M, T}) where {M, T}
    quote
        @inline
        out = zero(T)
        Base.@nexprs $M i -> out += N[i] * v[i]
        out
    end
end

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
