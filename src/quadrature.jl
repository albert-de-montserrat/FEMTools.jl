# abstract type AbstractQuadrature end

# struct Quadrature2D{N, T} <: AbstractQuadrature
#     ξ::SVector{N, T}
#     η::SVector{N, T}
#     ω::SVector{N, T}
# end

# struct Quadrature3D{N, T} <: AbstractQuadrature
#     ξ::SVector{N, T}
#     η::SVector{N, T}
#     ζ::SVector{N, T}
#     ω::SVector{N, T}
# end

# @inline Base.getindex(q::Quadrature2D, i::Int) = SA[q.ξ[i], q.η[i]]
# @inline Base.getindex(q::Quadrature3D, i::Int) = SA[q.ξ[i], q.η[i], q.ζ[i]]

# @inline getweight(q::AbstractQuadrature, i::Int) = q.ω[i]

# 1D integration points and weights
@inline integration_points(::ElementDispatcher{1, 2})  = SA[-1/√(3), 1/√(3)]
@inline integration_points(::ElementDispatcher{1, 3})  = SA[-1/√(3/5), 0, 1/√(3/5)]

@inline integration_weights(::ElementDispatcher{1, 2}) = SA[1, 1]
@inline integration_weights(::ElementDispatcher{1, 3}) = SA[5/9, 8/9, 5/9]

# 2D integration points and weights for P1 (linear triangular) elements
@inline function integration_points(::ElementDispatcher{2, 3})
    # 1-point Gaussian quadrature for P1 triangular elements (centroid rule)
    # Reference triangle: (0,0), (1,0), (0,1)
    ξ = SA[
        1/6
        2/3
        1/6
    ]

    η = SA[
        1/6
        1/6
        2/3
    ]

    return ξ, η
end


function ip_triangle3()
    # -- Integration point coordinates
    ipx = @SMatrix [
        1/6 1/6
        2/3 1/6
        1/6 2/3
    ]

    # -- Weights
    ipw = @SVector [
        1/6,
        1/6,
        1/6,
    ]

    return ipx, ipw

end

@inline integration_weights(::ElementDispatcher{2, 3}) = SA[1/6, 1/6, 1/6]

# 2D integration points and weights for P2 (quadratic triangular) elements
@inline function integration_points(::ElementDispatcher{2, 6})

    # -- Integration point coordinates
    g1  = (8 - √10 + √(38 - 44 * √2/5)) / 18
    g2  = (8 - √10 - √(38 - 44 * √2/5)) / 18
    ξ   = SA[
        1 - 2*g1
        g1
        g1
        1 - 2*g2
        g2
        g2
    ]

    η = @SMatrix [
        g1
        1.0-2.0*g1
        g1
        g2
        1.0-2.0*g2
        g2
    ]

    
    return ξ, η
end

@inline function integration_weights(::ElementDispatcher{2, 6})

    # -- Weights
    w1 = (620 + √(213125 - 53320 * √10)) / 3720
    w2 = (620 - √(213125 - 53320 * √10)) / 3720
    ω  =  SA[
        w1, # 0.223381589678011
        w1, # 0.223381589678011
        w1, # 0.223381589678011
        w2, # 0.109951743655322
        w2, # 0.109951743655322
        w2, # 0.109951743655322
    ] ./ 2

    return ω
end

# 2D integration points and weights for Q1 (bilinear) elements
@inline function integration_points(::ElementDispatcher{2, 4}) 
    # 2x2 Gaussian quadrature for Q1 elements
    ξ₁ = -1/√3
    ξ₂ = 1/√3
    ξ = SA[ξ₁, ξ₂, ξ₁, ξ₂]
    η = SA[ξ₁, ξ₁, ξ₂, ξ₂]
    
    return ξ, η
end

@inline integration_weights(::ElementDispatcher{2, 4}) = SA[1, 1, 1, 1]

# 2D integration points and weights for Q2 (biquadratic) elements  
@inline function integration_points(::ElementDispatcher{2, 9})
    # 3x3 Gaussian quadrature for Q2 elements
    ξ₁ = -√(3/5)
    ξ₂ = 0
    ξ₃ = √(3/5)
    
    ξ = SA[ξ₁, ξ₂, ξ₃, ξ₁, ξ₂, ξ₃, ξ₁, ξ₂, ξ₃]
    η = SA[ξ₁, ξ₁, ξ₁, ξ₂, ξ₂, ξ₂, ξ₃, ξ₃, ξ₃]
    
    return ξ, η
end

@inline function integration_weights(::ElementDispatcher{2, 9})
    # 3x3 Gaussian quadrature weights
    wᵢ =SA[
        5/9
        8/9
        5/9
    ]
    # Tensor product of 1D weights
    ω = kron(wᵢ, wᵢ)
    
    return ω
end
