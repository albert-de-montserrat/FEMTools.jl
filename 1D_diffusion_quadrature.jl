using LinearAlgebra
using StaticArrays
using ForwardDiff
using GLMakie

include("common.jl")
## helper functions
struct LinearElement{T,F1,F2}
    ip::T
    ω::T
    S1::F1
    S2::F2
   
    function LinearElement()
        ip = √(1/3) .* [1, -1]
        ω  = [1, 1]
        S1 = ξ -> (1 - ξ)  / 2
        S2 = ξ -> (1 + ξ)  / 2

        new{typeof(ip),typeof(S1), typeof(S2)}(ip, ω, S1, S2)
    end
end

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
    @SVector [S1(ip[I]), S2(ip[I])]
end

# compute jacobian of the shape function
∇N(SF::LinearElement, ip::Integer) = ForwardDiff.derivative(x -> N(SF, x), SF.ip[ip])
∇N(SF::LinearElement, x::Float64)  = ForwardDiff.derivative(x -> N(SF, x), x)

function analytical_solution(κ, σ, x, t, Tmax)
    sol = Tmax / √(1 + 4 * t * κ / σ^2) * exp(-x^2 / σ^2 + 4 * t * κ)
    return sol
end

###
function main()
    # GEOMETRICAL PARAMETERS
    Lx	    = 10
    # PHYSICAL PARAMETERS
    κ       = 1
    Q       = 0     # source term
    ttot	= 5     # total time
    # NUMERICAL PARAMETERS
    nel     = 100      # number of elements
    nnods   = nel + 1  # number of nodes
    n_x_el  = 2
    nt      = 100      # number of iterations
    # calculated from above
    dt      = ttot / nt
    time    = dt:dt:ttot
    # NUMERICAL GRID
    dx      = Lx / nel
    GCOORD  = 0:dx:Lx

    # Local-to-global mapping
    EL2N   = [
        (1:nnods-1)'
        (2:nnods)'
    ]

    # Shape functions
    SF     = LinearElement()
    JacSF  = Jacobian(SF)

    # Initial conditions
    Tmax    = 10
    σ       = 1
    T       = @. Tmax * exp(-GCOORD^2 / σ^2)

    # Boundary conditions
    bc_dof = [1, nnods] # degrees of freedom corresponding to boundary conditions
    bc_val = [0, 0] # dirichlet valyes at the boundary conditions

    # initialize global matrices
    KG = zeros(nnods, nnods)
    MG = zeros(nnods, nnods)
    FG = zeros(nnods)

    # assembly
    for iel in 1:nel
        local_nodes = @view EL2N[:, iel]
        coord       = @view GCOORD[local_nodes]
        Kloc        = @SMatrix zeros(n_x_el, n_x_el)
        Mloc        = @SMatrix zeros(n_x_el, n_x_el)
        Floc        = @SVector zeros(n_x_el)

        for i in 1:n_x_el
            # we need to compute them only once
            Nᵢ    = N(SF, SF.ip[i])
            JacSFᵢ= JacSF.J[i]'
            # JacSFᵢ= ∇N(SF, JacSF.J[i])'
            ωᵢ    = SF.ω[i]
            # actual computation
            J     = JacSFᵢ * coord
            ∂N∂x  = J \ JacSFᵢ
            Kloc += (∂N∂x' .* κ) * ∂N∂x * (J * ωᵢ)
            Mloc += Nᵢ * Nᵢ' * J * ωᵢ
            Floc += Nᵢ * (Q * J * ωᵢ)
        end
        
        @views KG[local_nodes, local_nodes] .+= Kloc
        @views MG[local_nodes, local_nodes] .+= Mloc
        @views FG[local_nodes]              .+= Floc
    end

    KG_sparse = sparse(KG)
    MG_sparse = sparse(MG)
    KLG       = @. KG_sparse + MG_sparse / dt
    b         = similar(T)
    sol       = similar(T)

    # time loop
    fig = Figure()
    ax  = Axis(fig[1,1])
    t   = 0
    for _ in 1:100
        sol  .= analytical_solution.(κ, σ, GCOORD, t, Tmax)
        b    .= MG ./ dt * T .+ FG

        # apply BCs
        @views KLG[bc_dof, :]      .= 0
        @views KLG[bc_dof, bc_dof] .= I + zeros(length(bc_dof), length(bc_dof))
        # @views b[bc_dof]           .= analytical_solution.(κ, σ, GCOORD[bc_dof], t, Tmax)
        @views b[bc_dof]           .= sol[bc_dof]
        T                          .= KLG \ b
        t                          += dt
    end

    # xsol = LinRange(0e0, Lx, 1_000) 
    # sol  = analytical_solution.(κ, σ, GCOORD, t-dt, Tmax)
    lines!(ax, GCOORD, T , color=:red)
    lines!(ax, GCOORD, sol, color=:black)

    fig
end

main()
