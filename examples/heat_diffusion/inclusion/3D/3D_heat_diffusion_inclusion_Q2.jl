using DomainSets
using DomainSets: ×
using KernelAbstractions
using StaticArrays
using Printf
using FEMTools

const backend   = CPU()
const workgroup = 1

# ---------------------------------------------------------------------------
# Geometry precomputation  (local helper — wraps FEMTools kernel)
# ---------------------------------------------------------------------------

function precompute_geometry(coords, el2n, nels, element::ReferenceElement{T}) where T<:AbstractElement{3, N} where N
    ip    = element.integration_points
    NQ    = length(ip.ω)
    ξq    = ntuple(q -> SVector(ip.ξ[q], ip.η[q], ip.ζ[q]), NQ)
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), NQ)
    geo   = KernelAbstractions.allocate(backend, NTuple{NQ, Tuple{SMatrix{N, 3, Float64, 3N}, Float64}}, nels)
    FEMTools.precompute_geometry_kernel!(backend, workgroup)(geo, coords, el2n, ∂N∂ξq, ip.ω, Val(N); ndrange = nels)
    KernelAbstractions.synchronize(backend)
    return geo
end

# ---------------------------------------------------------------------------

function main(nels; do_plot = true, nsteps = 50, verbose = true)
    TDev = FEMTools.TA(backend)

    L        = 50e3
    Ω        = (-L..L) × (-L..L) × (-L..L)
    element  = ReferenceElement(QuadraticElement{3, 27, Float64})
    mesh     = FEMTools.Mesh(backend, Ω, element, nels)
    mesh_cpu = FEMTools.Mesh(CPU(), Ω, element, nels)

    # --- material properties (single phase) ---
    k    = (3.0,)     # thermal conductivity   [W m⁻¹ K⁻¹]
    Cp   = (1200.0,)  # specific heat          [J kg⁻¹ K⁻¹]
    ρ0   = (3300.0,)  # reference density      [kg m⁻³]
    α    = (3e-5,)    # thermal expansivity    [K⁻¹]
    K    = (1e11,)    # bulk modulus           [Pa]
    Tref = 273.0      # reference temperature  [K]
    Δt   = 100e3 * 365 * 24 * 3600 # time step [s]

    # --- Gaussian thermal anomaly at the domain center ---
    T_bg = 273.0    # background temperature [K]
    ΔT   = 500.0    # anomaly amplitude      [K]
    σ    = L / 8    # anomaly half-width     [m]

    # --- Dirichlet BCs: T = T_bg on all six walls ---
    I_Ω, J_Ω, K_Ω = factors(Ω)
    coords_cpu = Array(mesh_cpu.coords)
    Γ = findall(coords_cpu) do p
        x, y, z = p
        x == leftendpoint(I_Ω) || x == rightendpoint(I_Ω) ||
        y == leftendpoint(J_Ω) || y == rightendpoint(J_Ω) ||
        z == leftendpoint(K_Ω) || z == rightendpoint(K_Ω)
    end

    # DOF index arrays and value arrays are kept separate from dr
    Γ_dofs = TDev(mesh_cpu.DoFs[Γ])
    Γ_vals = TDev(fill(T_bg, length(Γ)))
    Γ_zero = zero(Γ_vals)

    # --- precompute geometry (∂N∂x, dΩ per element per quadrature point) ---
    geo = precompute_geometry(mesh.coords, mesh.el2n, mesh.nels, element)

    # --- ThermalDiffusionDR bundles all solver state and material properties ---
    dr = ThermalDiffusionDR(backend, mesh.nnodes, k, Cp, ρ0, α, K; CFL=0.9, ϵ=1e-8)

    # Gaussian hot spherical inclusion centered at the origin
    copyto!(dr.T, TDev(Float64[T_bg + ΔT * exp(-(p[1]^2 + p[2]^2 + p[3]^2) / (2σ^2)) for p in coords_cpu]))
    apply_dirichlet!(dr.T, Γ_dofs, Γ_vals, backend, workgroup)
    copyto!(dr.T0, dr.T)

    # --- time loop ---
    history    = Tuple{Vector{Int}, Vector{Float64}}[]  # (iters, resids) per step
    solve_time = 0.0

    for step in 1:nsteps
        verbose && @printf("─── time step %d / %d  (t = %.3g s) ───\n", step, nsteps, step * Δt)
        copyto!(dr.T0, dr.T)
        fill!(dr.∂T∂τ, 0)

        # one call — everything lives in FEMTools
        Δwall = @elapsed hist = solver!(dr, Δt, mesh, geo, element, Γ_dofs, Γ_zero, Γ_vals, backend, workgroup; ncheck = 5, Tref = Tref, verbose = verbose)
        solve_time += Δwall
        push!(history, hist)
    end

    if do_plot
        @eval using GLMakie

        p_ord = order(element)
        nx, ny, nz = nels .* p_ord
        xs = LinRange(-L, L, nx + 1) ./ 1e3
        ys = LinRange(-L, L, ny + 1) ./ 1e3
        iz = div(nz, 2) + 1

        T_host = reshape(Array(dr.T), nx + 1, ny + 1, nz + 1)
        fig = Figure(size = (700, 600))
        ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "mid-z slice")
        heatmap!(ax, xs, ys, T_host[:, :, iz]; colormap = :thermal, colorrange = (T_bg, T_bg + ΔT))
        Colorbar(fig[1, 2]; colormap = :thermal, limits = (T_bg, T_bg + ΔT), label = "T [K]")
        display(fig)
    end

    return (; nels = mesh.nels, ndofs = mesh.nnodes, history, solve_time, dr)
end

# skipped when included by a driver (which defines RUN_SWEEP)
if !(@isdefined RUN_SWEEP)
    nels = (16, 16, 16)
    main(nels)
end
