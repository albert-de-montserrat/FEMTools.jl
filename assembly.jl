using FEMTools
using StaticArrays
using LinearAlgebra

function compute_B(∇N, ::Val{N}) where N
    B_ip = ntuple(Val(N)) do i
        @SMatrix [
            ∇N[1, i] 0
            0        ∇N[2, i]
            ∇N[2, i] ∇N[1, i]
        ]
    end
    reduce(hcat, B_ip)
end

el = CruziexRaviart()

ε  = @MVector zeros(3)
τ  = @MVector zeros(3)
P  = @MMatrix ones(3,3)
Pb = @MVector ones(3)
m  = SA[1, 1, 0]
η  = 1e0


D  = SA[
    4/3 -2/3  0e0
   -2/3  4/3  0e0
    0e0  0e0  1e0
]

ip = 1

J = el.PressureElement[1].∇N
detJ = det(J)

B_ip = @SMatrix [
    1 0
    0 1
    1 1
]

∇N_vel = el.VelocityElement[1].∇N

B = compute_B(∇N_vel, Val(6))


A = B' * ((η .* D) * B)
G = B' * m * el.PressureElement[1].N'