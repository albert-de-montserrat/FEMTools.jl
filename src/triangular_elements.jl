abstract type AbstractElementType end 

struct LinearTriangle <:AbstractElementType end
struct QuadraticTriangle <:AbstractElementType end
struct QuadraticTriangleBubble <:AbstractElementType end

struct Element{T1, T2}
    SF::T1 # shape functions
    ip::T2 # integration points
end

function Element(::Union{T, Type{T}}) where T <: AbstractElementType
    # integration points
    ip = IntegrationPoints(T)
    # shape functions
    SF = [ShapeFunction(rᵢ, sᵢ, T) for (rᵢ, sᵢ) in zip(ip.η, ip.ξ)]
    return Element{typeof(SF), typeof(ip)}(SF, ip)
end

Base.getindex(el::Element, i::Int) = el.SF[i]
Base.length(el::Element) = length(el.ip.vals)

function element_type(el::Element)
    @inline element_type(::Val{3}) = LinearTriangle
    @inline element_type(::Val{6}) = QuadraticTriangle
    @inline element_type(::Val{7}) = QuadraticTriangleBubble
        
    return element_type(Val(length(el)))
end

abstract type AbstractElement end 

struct CruziexRaviart{VE, PE} <:AbstractElement 
    VelocityElement::VE
    PressureElement::PE
    
    function CruziexRaviart()
        VelocityElement = Element(QuadraticTriangleBubble)
        PressureElement = Element(LinearTriangle)
        new{typeof(VelocityElement), typeof(PressureElement)}(VelocityElement, PressureElement)
    end
end

