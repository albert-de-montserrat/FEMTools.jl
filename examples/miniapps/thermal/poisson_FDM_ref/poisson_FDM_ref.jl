using GLMakie, LinearAlgebra, Printf
using TimerOutputs

let 
    
    # let
    xmin  = -1.0
    xmax  = 1.0 
    D     = 1.0
    ncx   = 150_000
    xv    = LinRange(xmin, xmax, ncx+1)        # Vertices / faces
    xc    = 1//2 .* (xv[1:end-1] .+ xv[2:end]) # Centroids of elements
    Δx    = (xmax-xmin)/ncx                    # Size of elements
    σ     = 0.1                                # Initial solution
    s     = 2.0*exp.(-xc.^2/(2σ^2) )           # Source  
    HW    = 1.0                                # Dirichlet value west
    HE    = 0.0                                # Dirichlet value east
    epsi  = 1e-9                               # Relative tolerance
    Ωe      = Δx                   # element volume = spacing in 1D
    
    
    # FD PT solve
    H_FD   = zeros(ncx+2) 
    f      = zeros(ncx)
    q      = zeros(ncx+1)         # Storage for flux
    dHdτ   = zeros(ncx)
    ρ      = 6.5/ncx              # Optimal damping (dispersion analysis)
    Δτ     = Δx^2/(2*D) * 1.9     # Optimal step determined by von Neumann analysis
    nr0    = 0.0
    H_FD[2:end-1] .= exp.(-xc.^2/(2σ^2) ) # Initial solution
    to = TimerOutput()
    for it=1:100#00
        @timeit to "threads FD" begin
            H_FD[1]        = 2.0*HW - H_FD[2]
            H_FD[end]      = 2.0*HE - H_FD[end-1]
            # q             .= @views  D.*diff(H_FD, dims=1)/Δx
            # f             .= @views diff(q,dims=1)/Δx .- s
            Threads.@threads for i in eachindex(q)
                q[i] = D * (H_FD[i+1] - H_FD[i])/Δx
            end
            Threads.@threads for i in eachindex(f)
                f[i] = (q[i+1]-q[i])/Δx - s[i]
            end
        end
        @timeit to "series FD" begin
            H_FD[1]        = 2.0*HW - H_FD[2]
            H_FD[end]      = 2.0*HE - H_FD[end-1]
            # q             .= @views  D.*diff(H_FD, dims=1)/Δx
            # f             .= @views diff(q,dims=1)/Δx .- s
            for i in eachindex(q)
                q[i] = D * (H_FD[i+1] - H_FD[i])/Δx
            end
            for i in eachindex(f)
                f[i] = (q[i+1]-q[i])/Δx - s[i]
            end
        end
        dHdτ          .= (1.0 .- ρ).*dHdτ .+ f
        H_FD[2:end-1] .+= Δτ.*dHdτ          
        if it%1000==0 || it==1
            nr = norm(f)
            if it==1 nr0 = nr end
            @printf("Iter. %05d: %2.2e\n", it, nr/nr0)
            if nr/nr0<epsi break end    
        end
    end

    display(to)
    scatterlines(xv[25:50:end], H_FD[25:50:end-1])
    # end
end