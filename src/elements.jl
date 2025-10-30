abstract type AbstractFiniteElement{NN, T} end

struct ElementDispatcher{ND, NN} end

struct LinearElement1D{NN, T} <: AbstractFiniteElement{NN, T}
    N::NTuple{2, SVector{2, T}}   # shape functions
    ∇N::NTuple{2, SVector{2, T}}  # gradient of the shape functions
    ip::SVector{2, T}             # integrations points
    ω::SVector{2, T}              # weights for integration

    function LinearElement1D(::Type{T}) where {T}
        element = ElementDispatcher{1, 2}()
        
        ξ  = integration_points(element)
        ω  = integration_weights(element)
        N  = ntuple(i -> shape_functions(ξ[i], element), Val(2))
        ∇N = ntuple(i -> gradient_shape_functions(ξ[i], element), Val(2))
        
        new{2, T}(N, ∇N, ξ, ω)
    end
end

LinearElement1D() = LinearElement1D(Float64)

struct QuadraticElement1D{NN, T} <: AbstractFiniteElement{NN, T}
    N::NTuple{3, SVector{3, T}}   # shape functions
    ∇N::NTuple{3, SVector{3, T}}  # gradient of the shape functions
    ip::SVector{3, T}             # integrations points
    ω::SVector{3, T}              # weights for integration

    function QuadraticElement1D(::Type{T}) where {T}
        element = ElementDispatcher{1, 3}()

        ξ  = integration_points(element)
        ω  = integration_weights(element)
        N  = ntuple(i -> shape_functions(ξ[i], element), Val(3))
        ∇N = ntuple(i -> gradient_shape_functions(ξ[i], element), Val(3))
        new{3, T}(N, ∇N, ξ, ω)
    end
end

QuadraticElement1D() = QuadraticElement1D(Float64)

struct Q1Element2D{NN, T} <: AbstractFiniteElement{NN, T}
    N::NTuple{4, SVector{4, T}}     # shape functions (4 integration points)
    ∇N::NTuple{4, SVector{4, T}}    # gradient of the shape functions
    ip::NTuple{2, SVector{4, T}}    # integrations points (ξ, η)
    ω::SVector{4, T}                # weights for integration

    function Q1Element2D(::Type{T}) where {T}
        element = ElementDispatcher{2, 4}()
        
        ξη = integration_points(element)  # returns (ξ_coords, η_coords)
        ω  = integration_weights(element)
        N  = ntuple(i -> shape_functions(ξη[1][i], ξη[2][i], element), Val(4))
        ∇N = ntuple(i -> gradient_shape_functions(SA[ξη[1][i], ξη[2][i]], element), Val(4))
        
        new{4, T}(N, ∇N, ξη, ω)
    end
end

Q1Element2D() = Q1Element2D(Float64)

struct Q2Element2D{NN, T} <: AbstractFiniteElement{NN, T}
    N::NTuple{9, SVector{9, T}}     # shape functions (9 integration points, 9 nodes)
    ∇N::NTuple{9, SVector{9, T}}    # gradient of the shape functions  
    ip::NTuple{2, SVector{9, T}}    # integrations points (ξ, η)
    ω::SVector{9, T}                # weights for integration

    function Q2Element2D(::Type{T}) where {T}
        element = ElementDispatcher{2, 9}()

        ξη = integration_points(element)  # returns (ξ_coords, η_coords)
        ω  = integration_weights(element)
        N  = ntuple(i -> shape_functions(ξη[1][i], ξη[2][i], element), Val(9))
        ∇N = ntuple(i -> gradient_shape_functions(SA[ξη[1][i], ξη[2][i]], element), Val(9))
        
        new{9, T}(N, ∇N, ξη, ω)
    end
end

Q2Element2D() = Q2Element2D(Float64)

struct T1Element{NN, T} <: AbstractFiniteElement{NN, T}
    N::NTuple{3,  SVector{3, T}}       # shape functions (3 integration points)
    ∇N::NTuple{3, SMatrix{3, 2, T, 6}} # gradient of the shape functions
    ip::NTuple{2, SVector{3, T}}       # integrations points (ξ, η)
    ω::SVector{3, T}                   # weights for integration

    function T1Element(::Type{T}) where {T}
        element = ElementDispatcher{2, 3}()
        
        ξη = integration_points(element)  # returns (ξ_coords, η_coords)
        ω  = integration_weights(element)
        N  = ntuple(i -> shape_functions(ξη[1][i], ξη[2][i], element), Val(3))
        ∇N = ntuple(i -> gradient_shape_functions(SA[ξη[1][i], ξη[2][i]], element), Val(3))
        
        new{4, T}(N, ∇N, ξη, ω)
    end
end

T1Element() = T1Element(Float64)

struct T2Element{NN, T} <: AbstractFiniteElement{NN, T}
    N::NTuple{3,  SVector{6, T}}        # shape functions (6 integration points)
    ∇N::NTuple{3, SMatrix{6, 2, T, 12}} # gradient of the shape functions
    ip::NTuple{2, SVector{6, T}}        # integrations points (ξ, η)
    ω::SVector{3, T}                    # weights for integration

    function T2Element(::Type{T}) where {T}
        element = ElementDispatcher{2, 6}()
        
        ξη = integration_points(element) 
        ω  = integration_weights(element)
        N  = ntuple(i -> shape_functions(ξη[1][i], ξη[2][i], element), Val(6))
        ∇N = ntuple(i -> gradient_shape_functions(SA[ξη[1][i], ξη[2][i]], element), Val(6))
        
        new{4, T}(N, ∇N, ξη, ω)
    end
end

T2Element() = T2Element(Float64)

Base.getindex(element::AbstractFiniteElement, ip::Int) = element.ip[ip], element.ω[ip], element.N[ip], element.∇N[ip]

@inline number_of_nodes(::T1Element) = 3
@inline number_of_nodes(::T2Element) = 6

local_SMatrix(::T1Element) = @SMatrix zeros(3,3)
local_SVector(::T1Element) = @SVector zeros(3)
