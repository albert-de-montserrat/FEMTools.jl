using StaticArrays

abstract type AbstractElementType end 

struct LinearTriangle <:AbstractElementType end
struct Quadratic <:AbstractElementType end
struct QuadraticBubble <:AbstractElementType end

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

abstract type AbstractElement end 

struct CruziexRaviart{VE, PE} <:AbstractElement 
    VelocityElement::VE
    PressureElement::PE
end


el = Element(LinearTriangle)
