using DomainSets
using DomainSets: ×
using KernelAbstractions
using StaticArrays
using Printf
using FEMTools

const backend   = @isdefined(SWEEP_BACKEND) ? SWEEP_BACKEND : CPU()
const workgroup = @isdefined(SWEEP_WORKGROUP) ? SWEEP_WORKGROUP : 1

# ---------------------------------------------------------------------------
# Geometry precomputation  (local helper — wraps FEMTools kernel)
# ---------------------------------------------------------------------------

function precompute_geometry(coords, el2n, nels, element::ReferenceElement{T}) where T<:AbstractElement{2, N} where N
    ip    = element.integration_points
    NQ    = length(ip.ω)
    ξq    = ntuple(q -> SVector(ip.ξ[q], ip.η[q]), NQ)
    ∂N∂ξq = ntuple(q -> eval_shape_function_jacobian(element, ξq[q]), NQ)
    geo   = KernelAbstractions.allocate(backend, NTuple{NQ, Tuple{SMatrix{N, 2, Float64, 2N}, Float64}}, nels)
    FEMTools.precompute_geometry_kernel!(backend, workgroup)(geo, coords, el2n, ∂N∂ξq, ip.ω, Val(N); ndrange = nels)
    KernelAbstractions.synchronize(backend)
    return geo
end

# ---------------------------------------------------------------------------

function main(nels; do_plot = true, nsteps = 50, verbose = true, reorder_colors = true)
    TDev = FEMTools.TA(backend)

    L        = 50e3
    Ω        = (-L..L) × (-L..L)
    element  = ReferenceElement(LinearElement{2, 4, Float64})
    mesh     = FEMTools.Mesh(backend, Ω, element, nels)
    mesh_cpu = FEMTools.Mesh(CPU(), Ω, element, nels)

    # elements sharing no nodes are grouped by color so the assembly kernels
    # can scatter into R, ∂R∂T, and PC without atomics
    colors = color_mesh(mesh_cpu)
    if reorder_colors
        el2n_reordered, el_groups, _ = FEMTools.reorder_connectivity_by_color(mesh_cpu.el2n, colors)
        copyto!(mesh.el2n, el2n_reordered)
    else
        el_groups = generate_element_groups(backend, colors)
    end

    # --- material properties (single phase) ---
    k   = (3.0,)     # thermal conductivity   [W m⁻¹ K⁻¹]
    Cp  = (1200.0,)  # specific heat          [J kg⁻¹ K⁻¹]
    ρ0  = (3300.0,)  # reference density      [kg m⁻³]
    α   = (3e-5,)    # thermal expansivity    [K⁻¹]
    K   = (1e11,)    # bulk modulus           [Pa]
    Tref = 273.0     # reference temperature [K]
    Δt  = 100e3 * 365 * 24 * 3600 # time step             [s]

    # --- Gaussian thermal anomaly at the domain center ---
    T_bg = 273.0    # background temperature       [K]
    ΔT   = 500.0    # anomaly amplitude            [K]
    σ    = L / 8    # anomaly half-width           [m]

    # --- Dirichlet BCs: T = T_bg on all four walls ---
    I_Ω, J_Ω   = factors(Ω)
    coords_cpu = Array(mesh_cpu.coords)
    Γ = findall(coords_cpu) do p
        x, y = p
        x == leftendpoint(I_Ω) || x == rightendpoint(I_Ω) ||
        y == leftendpoint(J_Ω) || y == rightendpoint(J_Ω)
    end

    # DOF index arrays and value arrays are kept separate from dr
    Γ_dofs = TDev(mesh_cpu.DoFs[Γ])
    Γ_vals = TDev(fill(T_bg, length(Γ)))
    Γ_zero = zero(Γ_vals)

    # --- precompute geometry (∂N∂x, dΩ per element per quadrature point) ---
    geo = precompute_geometry(mesh.coords, mesh.el2n, mesh.nels, element)

    # --- ThermalDiffusionDR bundles all solver state and material properties ---
    dr = ThermalDiffusionDR(backend, mesh.nnodes, k, Cp, ρ0, α, K; CFL=0.9, ϵ=1e-8)

    # Gaussian hot inclusion centered at the origin
    copyto!(dr.T, TDev(Float64[T_bg + ΔT * exp(-(p[1]^2 + p[2]^2) / (2σ^2)) for p in coords_cpu]))
    FEMTools.apply_dirichlet!(dr.T, Γ_dofs, Γ_vals, backend, workgroup)
    copyto!(dr.T0, dr.T)

    # --- time loop ---
    history    = Tuple{Vector{Int}, Vector{Float64}}[]
    solve_time = 0.0

    for step in 1:nsteps
        verbose && @printf("─── time step %d / %d  (t = %.3g s) ───\n", step, nsteps, step * Δt)
        copyto!(dr.T0, dr.T)
        fill!(dr.∂T∂τ, 0)

        # one call — everything lives in FEMTools
        Δwall = @elapsed hist = FEMTools.solver_colored!(dr, Δt, mesh, geo, element, Γ_dofs, Γ_zero, Γ_vals, el_groups, backend, workgroup; ncheck = 5, Tref = Tref, verbose = verbose)
        solve_time += Δwall
        push!(history, hist)
    end

    if do_plot
        @eval using GLMakie

        p_ord  = order(element)
        nx, ny = nels .* p_ord
        xs = LinRange(-L, L, nx + 1) ./ 1e3
        ys = LinRange(-L, L, ny + 1) ./ 1e3

        T_host = reshape(Array(dr.T), nx + 1, ny + 1)
        fig = Figure(size = (600, 600))
        ax = Axis(fig[1, 1]; aspect = DataAspect())
        heatmap!(ax, xs, ys, T_host; colormap = :thermal, colorrange = (T_bg, T_bg + ΔT))
        Colorbar(fig[1, 2]; colormap = :thermal, limits = (T_bg, T_bg + ΔT), label = "T [K]")
        display(fig)
    end

    return (; nels = mesh.nels, ndofs = mesh.nnodes, history, solve_time, dr)
end

# skipped when included by the sweep driver (which defines RUN_SWEEP)
if !(@isdefined RUN_SWEEP)
    nels = (30, 30)
    main(nels)
end
