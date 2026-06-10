using GLMakie, LinearAlgebra, Printf
using TimerOutputs
let 
    
    # let
    xmin  = -1.0
    xmax  = 1.0 
    D     = 1.0
    ncx   = 2000
    xv    = LinRange(xmin, xmax, ncx+1)        # Vertices / faces
    xc    = 1//2 .* (xv[1:end-1] .+ xv[2:end]) # Centroids of elements
    Δx    = (xmax-xmin)/ncx                    # Size of elements
    σ     = 0.1                                # Initial solution
    s     = 2.0*exp.(-xc.^2/(2σ^2) )           # Source  
    HW    = 1.0                                # Dirichlet value west
    HE    = 0.0                                # Dirichlet value east
    epsi  = 1e-9                               # Relative tolerance
    Ωe      = Δx                   # element volume = spacing in 1D
    
    # Specific to FEM 1D linear elements + analytic integration
    KM   = D.*[1/Δx -1/Δx; -1/Δx 1/Δx]
    F    = [Δx/2; Δx/2]
    nel  = ncx
    ndof = ncx+1
    
    """
    FEM Residual of Poisson problem in 1D (Dirichlet BC's)
    """
    @views function Residual_FEM_1D_vec!(f, h, s, ndof, KM, F, ncx)
        f           .= 0.0
        f[2:end-1] .+= (-s[1:end-1].*F[2]) .- (KM[2,1].*h[1:end-2] .+ KM[2,2].*h[2:end-1])
        f[2:end-1] .+= (-s[2:end-0].*F[2]) .- (KM[1,1].*h[2:end-1] .+ KM[1,2].*h[3:end-0])
        return nothing
    end
    
    """
    FEM Residual of Poisson problem in 1D (Dirichlet BC's)
    """
    function Residual_FEM_1D_loop!(f, h, s, ndof, KM, F, ncx)
        f .= 0.0
        # K11, K12, K21, K22 = K[1,1], K[1,2], K[2,1], K[2,2]
        @inbounds for idof=2:ndof-1
            # West 
            iel     = idof-1
            be      = -s[iel].*F
            dofs    = [idof-1; idof]
            H_loc   = h[dofs]
            fe      = be .- KM*H_loc 
            f[idof]+= fe[2] # (K21*H_loc[1] + K22*H_loc[2])
            # East 
            iel     = idof
            be      = -s[iel].*F
            dofs    = [idof; idof+1]
            H_loc   = h[dofs]
            fe      = be .- KM*H_loc
            f[idof]+= fe[1] # (K11*H_loc[1] + K12*H_loc[2])
        end
        return nothing
    end
    
    # FEM PT solve
    H_FEM       = zeros(ncx+1)
    f           = zeros(ncx+1) 
    dHdτ        = zeros(ncx+1) 
    Δτ          = Δx^2/(2*D) * 2.0/Ωe / 1.1
    ρ           = 6.5/ncx
    nr0         = 0.0
    H_FEM[2:end-1] .= exp.(-xv[2:end-1].^2/(2σ^2) )  
    H_FEM[1] = HW
    H_FEM[end] = HE
    to = TimerOutput()
    for it=1:10000
        @timeit to "crap" Residual_FEM_1D_vec!(f, H_FEM, s, ndof, KM, F, ncx)
        @timeit to "crap loop" Residual_FEM_1D_loop!(f, H_FEM, s, ndof, KM, F, ncx)
        dHdτ            .= (1-ρ).*dHdτ .+ f
        H_FEM[2:end-1] .+= Δτ.*dHdτ[2:end-1]
        if it%1000==0 || it==1
            nr = norm(f)
            if it==1 nr0 = nr end
            @printf("Iter. %05d: %2.2e\n", it, nr/nr0)
            if nr/nr0<epsi break end    
        end
    end
    display(to)
    scatterlines!(xv[25:50:end], H_FEM[25:50:end])
    # end
end