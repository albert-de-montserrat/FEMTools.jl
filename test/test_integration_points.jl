using Test
using StaticArrays

include("../triangular_elements.jl")
include("../shape_functions.jl")
include("../integration_points.jl")


linTri = IntegrationPoints(LinearTriangle)

@test linTri.η == SA[
    0.16666666666666666
    0.6666666666666666
    0.16666666666666666
]

@test linTri.ξ == SA[
    0.16666666666666666
    0.16666666666666666
    0.6666666666666666
]

@test linTri.vals == SA[
    0.16666666666666666
    0.16666666666666666
    0.16666666666666666
]


quadTri = IntegrationPoints(QuadraticTriangle)

@test linTri.η == SA[
    0.16666666666666666
    0.6666666666666666
    0.16666666666666666
]

@test linTri.ξ == SA[
    0.16666666666666666
    0.16666666666666666
    0.6666666666666666
]

@test linTri.vals == SA[
    0.16666666666666666
    0.16666666666666666
    0.16666666666666666
]