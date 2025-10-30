using Test

using FEMTools

el = Element(LinearTriangle)
@test length(el) == 3

el = Element(QuadraticTriangle)
@test length(el) == 6

el = Element(QuadraticTriangleBubble)
@test length(el) == 7

CR = CruziexRaviart()

@test element_type(CR.VelocityElement) === QuadraticTriangleBubble
@test element_type(CR.PressureElement) === LinearTriangle

vel = CR.VelocityElement
pel = CR.PressureElement

ρ = rand(6)

@b ipValue($ρ, $vel, 1) 

function ipValue(x, el::Element, ip) 
    N = el[ip].N 
    @assert length(N) == length(x)
    x ⋅ N
end
