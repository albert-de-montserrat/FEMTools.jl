
struct IntegrationPoints{T}
    η::T    # local η coordinates
    ξ::T    # local ξ coordinates
    vals::T # weights
end

function IntegrationPoints(::Union{LinearTriangle, Type{LinearTriangle}})
    # Integration points coordinates
    η = SA[
        1/6, 2/3, 1/6
    ]
    ξ = SA[
        1/6, 1/6, 2/3
    ]
    # Weights
    vals = SA[
        1/6, 1/6, 1/6
    ]
    # Type
    T = typeof(η)
    # instantiate
    IntegrationPoints{T}(η, ξ, vals)
end

function IntegrationPoints(::Union{QuadraticTriangle, Type{QuadraticTriangle}})
    # Integration points coordinates
    g1  = (8 - √10 + √(38 - 44 * √(2/5))) / 18
    g2  = (8 - √10 - √(38 - 44 * √(2/5))) / 18
    η   = SA[
        1-2*g1, g1, g1, 1-2*g2, g2, g2
    ]
    ξ   = SA[
        g1, 1-2*g1, g1, g2, 1-2*g2, g2
    ]
    # Weights
    w1 = (620 + √(213125 - 53320 * √10)) / 3720
    w2 = (620 - √(213125 - 53320 * √10)) / 3720
    vals = SA[
        w1, w1, w1, w2, w2, w2
    ]
    # Type
    T = typeof(η)
    # instantiate
    IntegrationPoints{T}(η, ξ, vals)
end

function IntegrationPoints(::Union{QuadraticTriangleBubble, Type{QuadraticTriangleBubble}})
    # Integration points coordinates
    g1 = (6 - √15) / 21
    g2 = (6 + √15) / 21
    η   = SA[
        1/3, 1-2*g1, g1, g1, 1-2*g2, g2, g2
    ]
    ξ   = SA[
        1/3, g1, 1-2*g1, g1, g2, 1-2*g2, g2
    ]
    # Weights
    w1 = (155 - √15) / 1200
    w2 = (155 + √15) / 1200
    vals = SA[
        0.225, w1, w1, w1, w2, w2, w2
    ]
    # Type
    T = typeof(η)
    # instantiate
    IntegrationPoints{T}(η, ξ, vals)
end