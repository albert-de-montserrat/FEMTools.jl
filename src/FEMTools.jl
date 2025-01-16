module FEMTools

using StaticArrays

include("elements/triangular_elements.jl")
export 
    Element, 
    LinearTriangle,
    QuadraticTriangle, 
    QuadraticTriangleBubble, 
    CruziexRaviart, 
    element_type
 
include("elements/shape_functions.jl")
include("elements/integration_points.jl")

end # module FEMTools
