using Metal
using KernelAbstractions
using ForwardDiff, StaticArrays
import KernelAbstractions as KA
import LinearAlgebra: norm

const DAT = Float32
const derivative = true
const primitive  = false 

############################### Kernels ###############################

@kernel function update2fields!(u::AbstractArray{T, 3}, ∂u∂τ::AbstractArray{T, 3}, func::F, args) where {F,T}
    i, j, k = @index(Global, NTuple)
    I, J, K = i+1, j+1, k+1 # shift for ghost nodes
    if I>1 && I <= size(u,1)-1 && J>1 && J <= size(u,2)-1 && K>1 && K <= size(u,3)-1
        result = func( args..., (I, J, K) )
        u[I, J, K]    = result[1]
        ∂u∂τ[I, J, K] = result[2]
    end
end

# Local Poisson function
function Poisson_local_allinone(u, b, Δ, sz_u, 𝐼)
    i, j, k = 𝐼
    uC = u[1]
    # Boundaries
    uW = i==2         ?  -uC : u[2]
    uE = i==sz_u[1]-1 ?  -uC : u[3]
    uS = j==2         ?  -uC : u[4]
    uN = j==sz_u[2]-1 ?  -uC : u[5]
    uB = k==2         ?  -uC : u[6]
    uF = k==sz_u[3]-1 ?  -uC : u[7]
    # Flux components on each face
    qxW = -(uC - uW) / Δ.x;  qxE = -(uE - uC) / Δ.x
    qyS = -(uC - uS) / Δ.y;  qyN = -(uN - uC) / Δ.y
    qzB = -(uC - uB) / Δ.z;  qzF = -(uF - uC) / Δ.z
    # Flux divergence
    return -(( (qxE - qxW) / Δ.x + (qyN - qyS) / Δ.y + (qzF - qzB) / Δ.z ) + b)
end

# Poisson, standard call
function Poisson_allinone(u, b, r0, Δ, deriv, 𝐼)
    i, j, k = 𝐼
    # Primitive
    uC = u[i,j,k]  
    uW = u[i-1,j,k]; uE = u[i+1,j,k]
    uS = u[i,j-1,k]; uN = u[i,j+1,k]
    uB = u[i,j,k-1]; uF = u[i,j,k+1]
    # RHS
    𝑏 = b[𝐼...]
    𝑢 = @SVector [uC, uW, uE, uS, uN, uB, uF]
    # Poisson
    if deriv==true
        ∂r∂𝑢 = ForwardDiff.gradient(x->Poisson_local_allinone(x, 𝑏, Δ, size(u), 𝐼), 𝑢) 
        D = abs(∂r∂𝑢[1])
        G = sum(abs.(∂r∂𝑢))
        out = D, G
    else
        r = Poisson_local_allinone(𝑢, 𝑏, Δ, size(u), 𝐼)
        out = r, r0[𝐼...]
    end
    return out
end

############################### Helpers ###############################
function assign_value(v, 𝐼)
    return v[𝐼...]
end

# Function that launches a kernel
function all!(out, func, args, launch)
    backend, wg, nd = launch
    run!(backend, wg)(
            out, func, args;
            ndrange = nd
        )
end

function updates(u, ∂u∂τ, r, D, α, β, 𝐼)
    ∂u∂τ_new =  r[𝐼...] / D[𝐼...] + β * ∂u∂τ[𝐼...]
    u_new    = u[𝐼...] + α * ∂u∂τ_new
    return u_new, ∂u∂τ_new
end

######################## Main ########################
function main(ncx)

    Gershgorin = :analytics
    Gershgorin = :enzyme

    backend = CPU()
    backend = MetalBackend()

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
 
    # Set initial u and b
    u  .= one(DAT)
    b  .= one(DAT)

    # Kernel launch params
    wg     = (8, 8, 8)
    nd     = nc

    # Gershgorin 
    if Gershgorin == :analytics 
        D[2:end-1,2:end-1,2:end-1] .= 6  / Δ.x^2
        G[2:end-1,2:end-1,2:end-1] .= 12 / Δ.x^2 
    elseif Gershgorin == :enzyme 
        update2fields!(backend, wg)(
            D, G, Poisson_allinone, (u, b, r, Δ, derivative);
                ndrange = nd
            )
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
    @time for iter=1:niter

        update2fields!(backend, wg)(
                r, r0, Poisson_allinone, (u, b, r, Δ, primitive);
                ndrange = nd
            )

        update2fields!(backend, wg)(
                u, ∂u∂τ, updates, (u, ∂u∂τ, r, D, α, β);
                ndrange = nd
            )

        if iter==1 || mod(iter, 100)==0
            nr = norm(r) / n
            nr0 = iter==1 ? nr : nr0
            @info "iter. $(iter) --- abs. |r| =  $(norm(r)) ---  nr = $(nr/nr0)"
            if nr/nr0 < tol break end

            update2fields!(backend, wg)(
            D, G, Poisson_allinone, (u, b, r, Δ, derivative);
                ndrange = nd
            )
            λmax   = maximum(G ./ D)
            Δτ     = 2 / sqrt(maximum(λmax)) * CFL

            λmin  = abs.((sum(Δτ.*∂u∂τ.*( (r .- r0) ./ D )))) / sum( (Δτ.*∂u∂τ).^2 )
            c      = 2 * sqrt(λmin) * c_fact
            α      = 2 * Δτ^2 / (2 + c*Δτ)
            β      = (2 - c * Δτ) / (2 + c*Δτ)
        end
    end

end

main(16)