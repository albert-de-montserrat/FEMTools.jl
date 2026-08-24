
# using Metal
using KernelAbstractions
using ForwardDiff, StaticArrays
import KernelAbstractions as KA
import LinearAlgebra: norm
using GLMakie
using CUDA

const DAT = Float64
const derivative = true
const primitive  = false 
using TimerOutputs

# 2 step:
# 1) fluxes as function of u: q = f(u)
# 2) residual as function of fluxes: r = f(q)
# ∂r∂u = ∂r∂q * ∂q∂u

############################### Helpers ###############################
@inline function assign_value(v, 𝐼)
    return @inbounds v[𝐼...]
end

@inline function create_mask(dim) 
    mask = if dim == 1
        (0, 1, 1)
    elseif dim == 2
        (1, 0, 1)
    else
        (1, 1, 0)
    # else
    #     error("Dimension $dim not supported")
    end
    return mask
end

# Function that launches a kernel
function all!(out, func, args, launch)
    backend, wg, nd = launch
    run!(backend, wg)(
            out, func, args;
            ndrange = nd
        )
end

############################### Kernels ###############################

@kernel function run!(out::AbstractArray{T, 3}, func::F, args) where {F,T}
    i, j, k = @index(Global, NTuple)
    I, J, K = i+1, j+1, k+1
    if I>1 && I <= size(out,1)-1 && J>1 && J <= size(out,2)-1 && K>1 && K <= size(out,3)-1
        @inbounds out[I, J, K] = func( args..., (I, J, K) )
    end
end

@kernel function run_flux!(q, func::F, args) where {F}
    I, J, K = @index(Global, NTuple)

    if all((I, J, K) .≤ size(q.x))
        @inbounds q.x[I, J, K] = func( args..., (I, J, K), 1 )
    end 

    if all((I, J, K) .≤ size(q.y))
        @inbounds q.y[I, J, K] = func( args..., (I, J, K), 2 )
    end

    if all((I, J, K) .≤ size(q.z))
        @inbounds  q.z[I, J, K] = func( args..., (I, J, K), 3 )
    end
end

######################## Physics ########################

# Local Poisson function
@inline function Poisson_local(q, b, Δ)
    # Flux components on each face
    qxW = q[1]; qxE = q[2] # dim 1
    qyS = q[3]; qyN = q[4] # dim 2
    qzB = q[5]; qzF = q[6] # dim 3
    # Flux divergence
    r   = -(( (qxE - qxW) / Δ.x + (qyN - qyS) / Δ.y + (qzF - qzB) / Δ.z ) + b)
    return r
end

# Poisson wrapper
function Poisson(q, Gq, b, Δ, deriv, 𝐼)
    i, j, k = 𝐼
    # Flux
    @inbounds begin
        qxW = q.x[i-1,j-1,k-1]; qxE = q.x[i+0,j-1,k-1] # dim 1
        qyS = q.y[i-1,j-1,k-1]; qyN = q.y[i-1,j+0,k-1] # dim 2
        qzB = q.z[i-1,j-1,k-1]; qzF = q.z[i-1,j-1,k+0] # dim 3
    end
    # RHS
    𝑏 = b[𝐼...]
    # Call Poisson evaluation
    if deriv
        @inbounds begin
            ∂qxW = Gq.x[i-1,j-1,k-1]; ∂qxE = Gq.x[i+0,j-1,k-1] # dim 1
            ∂qyS = Gq.y[i-1,j-1,k-1]; ∂qyN = Gq.y[i-1,j+0,k-1] # dim 2
            ∂qzB = Gq.z[i-1,j-1,k-1]; ∂qzF = Gq.z[i-1,j-1,k+0] # dim 3
        end
        ∂𝑞∂𝑢 = @SVector [∂qxW, ∂qxE, ∂qyS, ∂qyN, ∂qzB, ∂qzF]
        𝑞    = @SVector [qxW, qxE, qyS, qyN, qzB, qzF]
        ∂r∂𝑞 = ForwardDiff.gradient(x->Poisson_local(x, 𝑏, Δ), 𝑞) 
        return sum(abs.(∂r∂𝑞.*∂𝑞∂𝑢))
    else
        𝑞    = @SVector [qxW, qxE, qyS, qyN, qzB, qzF]
        return Poisson_local(𝑞, 𝑏, Δ)
    end
end

# Flux component
@inline function qi_local(u, Δ, si_u, 𝐼, dim)
    # Boundaries
    uW = 𝐼[dim] ==1           ?  -u[2] : u[1]
    uE = 𝐼[dim] ==si_u[dim]-1 ?  -u[1] : u[2]
    # Flux component
    Δxi = values(Δ)[dim]
    return -(uE - uW) / Δxi
end

# Flux component wrapper
function qi(u, Δ, deriv, 𝐼, dim)
    # Primitive
    mask = create_mask(dim)
    ones = (1, 1, 1)
    # Neighbours
    uW   = u[𝐼 .+ mask...]
    uE   = u[𝐼 .+ ones... ]
    # Call flux component evaluation
    if deriv 
        𝑢    = @SVector [uW, uE]
        ∂q∂𝑢 = ForwardDiff.gradient(x->qi_local(x, Δ, size(u), 𝐼, dim), 𝑢) 
        return sum(abs.(∂q∂𝑢))
    else
        𝑢    = @SVector [uW, uE]
        return qi_local(𝑢, Δ, size(u), 𝐼, dim)
    end
end

@inline function update_rate(∂u∂τ, r, D, β, 𝐼)
    return @inbounds r[𝐼...] / D[𝐼...] + β * ∂u∂τ[𝐼...]
end

@inline function update_variable(u, ∂u∂τ, α, 𝐼)
    return @inbounds u[𝐼...] + α * ∂u∂τ[𝐼...]
end

@inline function updates(u, ∂u∂τ, r, D, α, β, 𝐼)
    ∂u∂τ_new = @inbounds r[𝐼...] / D[𝐼...] + β * ∂u∂τ[𝐼...]
    u_new    = @inbounds u[𝐼...] + α * ∂u∂τ_new
    return u_new, ∂u∂τ_new
end

######################## Main ########################
function main(ncx)

    Gershgorin = :analytics
    Gershgorin = :enzyme

    # backend = CPU()
    backend = CUDABackend()
    # backend = MetalBackend()

    # Resolution
    nc  = ncx, ncx, ncx # here we need size in Int64 for Metal (at least)
    nce = nc .+ 2
    Δ   = (x=DAT(1/nc[1]), y=DAT(1/nc[2]), z=DAT(1/nc[3]))

    # Memory allocations
    r    = KA.zeros(backend, DAT, nce...)
    ∂u∂τ = KA.zeros(backend, DAT, nce...)
    r0   = KA.zeros(backend, DAT, nce...) 
    u    =  KA.ones(backend, DAT, nce...)
    b    = KA.zeros(backend, DAT, nce...)
    D    =  KA.ones(backend, DAT, nce...)
    G    =  KA.ones(backend, DAT, nce...)
    𝐪    = (
        x = KA.zeros(backend, DAT, nc[1]+1, nc[2],   nc[3]  ),
        y = KA.zeros(backend, DAT, nc[1],   nc[2]+1, nc[3]  ),
        z = KA.zeros(backend, DAT, nc[1],   nc[2],   nc[3]+1),
    )
    𝐆𝐪  = (
        x = KA.zeros(backend, DAT, nc[1]+1, nc[2],   nc[3]  ),
        y = KA.zeros(backend, DAT, nc[1],   nc[2]+1, nc[3]  ),
        z = KA.zeros(backend, DAT, nc[1],   nc[2],   nc[3]+1),
    )

    # Set initial u and b
    u  .= one(DAT)
    b  .= one(DAT)

    # Kernel launch params
    wg     = (8, 8, 8)
    nd     = nc
    nd_q   = nc .+ 1
    launch = (backend, wg, nd)

    # Gershgorin 
    if Gershgorin == :analytics 
        D[2:end-1,2:end-1,2:end-1] .= 6  / Δ.x^2
        G[2:end-1,2:end-1,2:end-1] .= 12 / Δ.x^2 
    elseif Gershgorin == :enzyme 
        # Step 1: compute flux Gershgorin first Gq = ∂q∂u 
        run_flux!(backend, wg)(
                𝐆𝐪, qi, (u, Δ, derivative);
                ndrange = nd_q
            )
        # Step 2: compute Poisson Gershgorin first Gq = ∂r∂q*∂q∂u 
        all!(G, Poisson, (𝐪, 𝐆𝐪, b, Δ, derivative), launch)

        # Approximate diagonal PC (does not see BC effect)
        @. D = G / 2
    end

    # Iteration parameters
    tol    = DAT(1e-7)
    niter  = DAT(2000)
    CFL    = DAT(0.99)
    c_fact = DAT(0.9)
    nr0    = one(DAT)
    n      = DAT(sqrt(prod(nce)))
    
    λmax   = maximum(G ./ D)
    Δτ     = 2 / sqrt(maximum(λmax)) * CFL
    λmin   = DAT(0.0)
    c      = 2*sqrt(λmin)*c_fact
    α      = 2 * Δτ^2 / (2 + c.*Δτ)
    β      = (2 - c * Δτ) / (2 + c.*Δτ)

    @info "Iteration parameters"
    @show  Δτ, c

    # Iterations
    to = TimerOutput()
    @timeit to "solver" for iter=1:1000

        all!(r0, assign_value, (r,), launch)
        #-------------------------
        # 1 - flux
        @timeit to "residual" begin
            run_flux!(backend, wg)(
                    𝐪, qi, (u, Δ, primitive);
                    ndrange = nd_q
                )
            # 2 - balance
            all!(r, Poisson, (𝐪, 𝐆𝐪, b, Δ, primitive), launch)
        end
        #-------------------------
        all!(∂u∂τ, update_rate, (∂u∂τ, r, D, β), launch)
        all!(u, update_variable, (u, ∂u∂τ, α), launch)

        # if iter==1 || mod(iter, 100)==0
        #     nr = norm(r) / n
        #     nr0 = iter==1 ? nr : nr0
        #     @info "iter. $(iter) --- abs. |r| =  $(norm(r)) ---  nr = $(nr/nr0)"
        #     if nr/nr0 < tol break end

        #     # backend, wg, nd = launch
        #     # GershgorinDiagonal!(backend, wg)(
        #     #     D, G, Poisson_Diff, (u, b, Δ);
        #     #     ndrange = nd
        #     # )
        #     # λmax   = maximum(G ./ D)
        #     # Δτ     = 2 / sqrt(maximum(λmax)) * CFL

        #     λmin  = abs.((sum(Δτ.*∂u∂τ.*( (r .- r0) ./ D )))) / sum( (Δτ.*∂u∂τ).^2 )
        #     c      = 2 * sqrt(λmin) * c_fact
        #     α      = 2 * Δτ^2 / (2 + c*Δτ)
        #     β      = (2 - c * Δτ) / (2 + c*Δτ)
        # end
    end

    display(to)
    # interior cells (ghost shell stripped) at the z mid-slice; cell centers
    # live at Δ/2 + i*Δ on the unit cube
    # u_host = Array(u)
    # xc = LinRange(Δ.x / 2, 1 - Δ.x / 2, nc[1])
    # yc = LinRange(Δ.y / 2, 1 - Δ.y / 2, nc[2])
    # fig = Figure()
    # ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="3D Poisson PT solution (2-step, z = 0.5 slice)", aspect=DataAspect())
    # hm = heatmap!(ax, xc, yc, u_host[2:end-1, 2:end-1, nc[3] ÷ 2 + 1]; colormap=:inferno)
    # Colorbar(fig[1, 2], hm)
    # display(fig)

    return nothing
end

main(64)
# main(256)

### CPU TIMES
# 64^3 |-> 12.8 ms
# 32^3 |-> 12.8 ms