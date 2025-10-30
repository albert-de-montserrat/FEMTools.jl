@inline get_dofs(args,         condition::F, flag::Symbol) where F<:Function = flag => findall(condition, args)
@inline get_dofs(args::NTuple, condition::F, flag::Symbol) where F<:Function = flag => mapreduce(arg->findall(condition, arg), intersect , args)

@inline get_dofs(c::Pair{T, Pair{F, Symbol}}) where {T, F<:Function} = get_dofs(c.first, c.second.first, c.second.second)
@inline get_dofs(c::Vararg{Pair, N} ) where {N} = Dict(ntuple(i -> get_dofs(c[i]), Val(N)))
