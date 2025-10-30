function shape_functions(ξ, ::ElementDispatcher{1, 2})
    # Linear shape functions for a 1D element
    # x: position where the shape function is evaluated
    # x1: left node position
    # x2: right node position
    N1 = (1 - ξ) / 2
    N2 = (1 + ξ) / 2
    return SA[N1, N2]
end

function shape_functions(ξ, ::ElementDispatcher{1, 3})
    # Quadratic shape functions for a 1D element
    # x: position where the shape function is evaluated
    # x1: left node position
    # x2: middle node position
    # x3: right node position
    N1 = 1/2 * ξ * (ξ - 1)
    N2 = 1 - ξ^2
    N3 = 1/2 * ξ * (ξ + 1)
    return SA[N1, N2, N3]
end

# 2D shape functions for P1 (linear triangular) elements
function shape_functions(ξ, η, ::ElementDispatcher{2, 3})
    # Linear shape functions for a 2D triangular element
    # Node numbering (counterclockwise):
    # 3       
    # |\      
    # | \     
    # |  \    
    # |   \   
    # |    \  
    # 1-----2 
    #
    # Reference element: (0,0), (1,0), (0,1)
    
    N1 = 1 - ξ - η  # vertex 1 (bottom-left)
    N2 = ξ          # vertex 2 (bottom-right)
    N3 = η          # vertex 3 (top)
    
    return SA[N1, N2, N3]
end

# 2D shape functions for P2 (quadratic triangular) elements
function shape_functions(ξ, η, ::ElementDispatcher{2, 6})
    # Quadratic shape functions for a 2D triangular element (6-node)
    # Node numbering (counterclockwise):
    # 3       
    # |\      
    # | \     
    # 6  5    
    # |   \   
    # |    \  
    # 1--4--2 
    #
    # Reference element vertices: (0,0), (1,0), (0,1)
    # Mid-side nodes: (0.5,0), (0.5,0.5), (0,0.5)
    
    λ1 = 1 - ξ - η  # area coordinate for vertex 1
    λ2 = ξ          # area coordinate for vertex 2
    λ3 = η          # area coordinate for vertex 3
    
    # Vertex nodes (quadratic bubble)
    N1 = λ1 * (2*λ1 - 1)  # vertex 1 (bottom-left)
    N2 = λ2 * (2*λ2 - 1)  # vertex 2 (bottom-right)
    N3 = λ3 * (2*λ3 - 1)  # vertex 3 (top)
    
    # Mid-side nodes
    N4 = 4 * λ1 * λ2      # mid-point between 1 and 2 (bottom edge)
    N5 = 4 * λ2 * λ3      # mid-point between 2 and 3 (right edge)
    N6 = 4 * λ3 * λ1      # mid-point between 3 and 1 (left edge)
    
    return SA[N1, N2, N3, N4, N5, N6]
end

# 2D shape functions for Q1 (bilinear) elements
function shape_functions(ξ, η, ::ElementDispatcher{2, 4})
    # Bilinear shape functions for a 2D rectangular element
    # Node numbering (counterclockwise):
    #   4 ---- 3
    #   |      |
    #   |      |
    #   1 ---- 2
    
    N1 = (1 - ξ) * (1 - η) / 4  # bottom-left
    N2 = (1 + ξ) * (1 - η) / 4  # bottom-right
    N3 = (1 + ξ) * (1 + η) / 4  # top-right
    N4 = (1 - ξ) * (1 + η) / 4  # top-left
    
    return SA[N1, N2, N3, N4]
end

# 2D shape functions for Q2 (biquadratic) elements
function shape_functions(ξ, η, ::ElementDispatcher{2, 9})
    # Biquadratic shape functions for a 2D rectangular element (9-node)
    # Node numbering:
    #   4 ---- 7 ---- 3
    #   |             |
    #   8      9      6
    #   |             |
    #   1 ---- 5 ---- 2
    
    # Corner nodes
    N1 = (ξ - 1) * (η - 1) * ξ * η / 4         # bottom-left corner
    N2 = (ξ + 1) * (η - 1) * ξ * η / 4         # bottom-right corner  
    N3 = (ξ + 1) * (η + 1) * ξ * η / 4         # top-right corner
    N4 = (ξ - 1) * (η + 1) * ξ * η / 4         # top-left corner
    
    # Mid-side nodes
    N5 = (1 - ξ^2) * (η - 1) * η / 2           # bottom edge
    N6 = (ξ + 1) * (1 - η^2) * ξ / 2           # right edge
    N7 = (1 - ξ^2) * (η + 1) * η / 2           # top edge
    N8 = (ξ - 1) * (1 - η^2) * ξ / 2           # left edge
    
    # Center node
    N9 = (1 - ξ^2) * (1 - η^2)                 # center node
    
    return SA[N1, N2, N3, N4, N5, N6, N7, N8, N9]
end

@inline shape_functions(ξη::SVector, element::ElementDispatcher{2}) = shape_functions(ξη..., element)

function gradient_shape_functions(ξ, element::ElementDispatcher{1}) 
    ∇N = ForwardDiff.derivative(ξ -> shape_functions(ξ, element), ξ)
    return ∇N
end

# 2D gradient functions
function gradient_shape_functions(ξη, element::ElementDispatcher{N}) where N
    @assert N > 1
    # Use ForwardDiff to compute gradients automatically
    ∇N = ForwardDiff.jacobian(ξη -> shape_functions(ξη, element), ξη)
    return ∇N
end
