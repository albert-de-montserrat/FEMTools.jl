module FEMTools

using StaticArrays

include("triangular_elements.jl")
export 
    Element, 
    LinearTriangle,
    QuadraticTriangle, 
    QuadraticTriangleBubble, 
    CruziexRaviart, 
    element_type
 
include("shape_functions.jl")
include("integration_points.jl")

end # module FEMTools
