abstract type AbstractShapeFunction end 

struct ShapeFunction{T1,T2} <:AbstractShapeFunction
    N::T1
    ∇N::T2
end

# Find shape functions and their derivatives at given points on the
# master element for 3 node triangle
# 3-node triangle (node numbering is important)
#
#        3
#        | \
# s-axis |   \
#        |     \
#        1 - - - 2
#          r axis -->
function ShapeFunction(r, s, ::Union{LinearTriangle, Type{LinearTriangle}})
    t = 1 - r - s
    N = SA[
        t, r, s
    ]
    ∇N = SA[
        -1.0   1.0   0.0  # w.r.t. r
        -1.0   0.0   1.0  # w.r.t. s
    ];
    T1, T2 = typeof(N), typeof(∇N)
    return ShapeFunction{T1, T2}(N, ∇N)
end 

# Find shape functions and their derivatives at given points on the
# master element for a 7 node triangle
# 7-node triangle (node numbering is important)
#
#        3
#        | \
# s-axis 6   5
#        |    \
#        1 - 4 - 2
#          r-axis
function ShapeFunction(r, s, ::Union{QuadraticTriangle, Type{QuadraticTriangle}})
    t  = 1-r-s
    N  = SA[
        t*(2*t-1)+3*r*s*t, r*(2*r-1)+3*r*s*t, s*(2*s-1)+3*r*s*t, 4*r*t-12*r*s*t, 4*r*s-12*r*s*t, 4*s*t-12*r*s*t, 27*r*s*t
    ]
    ∇N = SA[
        1-4*t+3*s*t-3*r*s  -1+4*r+3*s*t-3*r*s  3*s*t-3*r*s        4*t-4*r+12*r*s-12*s*t 4*s+12*r*s-12*s*t -4*s+12*r*s-12*s*t    -27*r*s+27*s*t
        1-4*t+3*r*t-3*r*s   3*r*t-3*r*s       -1+4*s+3*r*t-3*r*s -4*r-12*r*t+12*r*s     4*r-12*r*t+12*r*s  4*t-4*s-12*r*t+12*r*s 27*r*t-27*r*s
    ]
    T1, T2 = typeof(N), typeof(∇N)
    return ShapeFunction{T1, T2}(N, ∇N)
end 

# Find shape functions and their derivatives at given points on the
# master element for a 7 node triangle
# 7-node triangle (node numbering is important)
#
#        3
#        | \
# s-axis 6   5
#        | 7  \
#        1 - 4 - 2
#          r-axis
function ShapeFunction(r, s, ::Union{QuadraticTriangleBubble, Type{QuadraticTriangleBubble}})
    t  = 1-r-s
    N  = SA[
        t*(2*t-1)  r*(2*r-1) s*(2*s-1) 4*r*t 4*r*s 4*s*t
    ]
    ∇N = SA[
        -(4*t-1)  4*r-1  0  4*(t-r)  4*s  -4*s    
        -(4*t-1)  0      4*s-1  -4*r  4*r  4*(t-s)
    ]
    T1, T2 = typeof(N), typeof(∇N)
    return ShapeFunction{T1, T2}(N, ∇N)
end 
