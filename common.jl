### helper functions
struct LinearElement{T,F1,F2}
    ip::T
    ω::T
    S1::F1
    S2::F2
   
    function LinearElement()
        ip = [-√(1/3), √(1/3)]
        ω  = [1, 1]
        S1 = ξ -> (1 - ξ)  / 2
        S2 = ξ -> (1 + ξ)  / 2

        new{typeof(ip),typeof(S1), typeof(S2)}(ip, ω, S1, S2)
    end
end

# struct LinearElement2D{T,F1,F2}
#     ip::T
#     ω::T
#     S1::F1
#     S2::F2
#     S3::F3
#     S4::F4
    
#     function LinearElement()
#         ip = √(1/3) .* [
#             -1  1  1 -1
#             -1 -1  1  1
#         ]       
#         ω  = [1, 1, 1, 1]
#         S1 = (ξ, η) -> (1 - ξ) * (1 - η) / 4
#         S2 = (ξ, η) -> (1 + ξ) * (1 - η) / 4
#         S3 = (ξ, η) -> (1 + ξ) * (1 + η) / 4
#         S4 = (ξ, η) -> (1 - ξ) * (1 + η) / 4

#         new{typeof(ip),typeof(S1), typeof(S2), typeof(S3), typeof(S4)}(ip, ω, S1, S2, S3, S4)
#     end
# end

struct Jacobian{T}
    J::T
    
    function Jacobian(SF)
        nip = length(SF.ip)
        J   = [∇N(SF, i) for i in 1:nip]
        new{typeof(J)}(J)
    end
end

# evaluate shape function
function N(SF::LinearElement, x)
    (; S1, S2) = SF
    @SVector [S1(x), S2(x)]
end

function Base.getindex(SF::LinearElement, I::Integer)
    (; S1, S2, ip) = SF
    S = S1, S2
    SVector{2, Float64}(f(ip[I]) for f in S)
    # @SVector [S1(ip[I]), S2(ip[I])]
end

# function Base.getindex(SF::LinearElement2D, I::Integer)
#     (; S1, S2, S3, S4, ip) = SF
#     S = S1, S2, S3, S4
#     SVector{4, Float64}(f(ip[I]) for f in S)
#     # @SVector [S1(ip[I]), S2(ip[I])]
# end

# compute jacobian of the shape function
∇N(SF::LinearElement, ip::Integer) = ForwardDiff.derivative(x -> N(SF, x), SF.ip[ip])
∇N(SF::LinearElement, x::Float64)  = ForwardDiff.derivative(x -> N(SF, x), x)
