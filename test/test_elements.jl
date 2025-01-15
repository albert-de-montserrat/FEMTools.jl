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