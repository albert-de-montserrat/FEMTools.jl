struct Point{N, T}
    data::SVector{N, T}

    function Point(points::Vararg{T, N}) where {N, T}
        new{N, T}(SVector{N, T}(points...))
    end
end

abstract type AbstractFiniteElementGrid end

struct Grid{dim, N, T, M} <: AbstractFiniteElementGrid
    x::T
    y::T
    z::T
    element2node::M

    function Grid(x::T, y::T, z::T, elemet2node::M, element) where {T, M}
        N = number_of_nodes(element)
        new{3, N, T, M}(x, y, z, elemet2node)
    end

    function Grid(x::T, y::T, elemet2node::M, element) where {T, M}
        N = number_of_nodes(element)
        new{2, N, T, M}(x, y, zeros(eltype(x), 1), elemet2node)
    end
end

@inline Base.getindex(grid::Grid{2}, i::Int) = Point(grid.x[i], grid.y[i])
@inline Base.getindex(grid::Grid{3}, i::Int) = Point(grid.x[i], grid.y[i], grid.z[i])

@inline Base.length(grid::Grid) = length(grid.x)

@inline Base.size(grid::Grid) = size(grid.element2node, 2)

@inline getelement(grid::Grid{M, N}, el::Int) where {M, N} = ntuple(node -> @inbounds(grid.element2node[node, el]), Val(N))

@inline function getelementcoords(grid::Grid{2, N}, el::Int) where {N} 
    ind = getelement(grid, el)
    T = eltype(grid.x)
    x = SVector{N, T}(@inbounds grid.x[ind[i]] for i in 1:N)
    y = SVector{N, T}(@inbounds grid.y[ind[i]] for i in 1:N)
    return vcat(x',y')
end

@inline function getelementfield(A::AbstractVector, grid::Grid{2, N}, el::Int) where {N} 
    ind = getelement(grid, el)
    T = eltype(grid.x)
    A_element = SVector{N, T}(@inbounds A[ind[i]] for i in 1:N)
    return A_element
end



@inline node_x_el(grid::Grid) = size(grid.element2node, 1)
