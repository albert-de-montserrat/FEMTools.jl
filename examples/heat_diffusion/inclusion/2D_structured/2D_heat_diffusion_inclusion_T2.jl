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
# Mesh generation — regular grid, each quad split into 2 quadratic triangles
# ---------------------------------------------------------------------------

"""
    build_mesh_T2(Ω, nels)

Build a structured T6 mesh over the rectangular domain `Ω` by splitting each
quadrilateral cell of the `nels = (nx, ny)` grid into **two** counter-clockwise
quadratic triangles along the bottom-left to top-right diagonal.

Node numbering: the refined `(2nx+1) × (2ny+1)` tensor grid first (x-index
fastest) — cell corners, cell-edge midpoints, and cell centers. Local nodes
follow the `QuadraticElement{2, 6}` ordering: corners 1–3, then
4 = mid(1,2), 5 = mid(2,3), 6 = mid(1,3). Returns `(coords, el2n)` with
`el2n` of size `6 × 2·nx·ny`.
"""
function build_mesh_T2(Ω, nels)
    I_Ω, J_Ω = factors(Ω)
    nx, ny   = nels
    xs = LinRange(leftendpoint(I_Ω), rightendpoint(I_Ω), 2nx + 1)
    ys = LinRange(leftendpoint(J_Ω), rightendpoint(J_Ω), 2ny + 1)

    stride = 2nx + 1
    n_grid = stride * (2ny + 1)
    coords = Vector{SVector{2, Float64}}(undef, n_grid)
    for iy in 0:2ny, ix in 0:2nx
        coords[iy * stride + ix + 1] = SVector(xs[ix + 1], ys[iy + 1])
    end

    node(ix, iy) = Int32(iy * stride + ix + 1)

    el2n = zeros(Int32, 6, 2 * nx * ny)
    iel  = 1
    for ey in 0:(ny - 1), ex in 0:(nx - 1)
        ix, iy = 2ex, 2ey
        BL  = node(ix,   iy);   BR  = node(ix+2, iy)
        TR  = node(ix+2, iy+2); TL  = node(ix,   iy+2)
        bot = node(ix+1, iy);   rgt = node(ix+2, iy+1)
        top = node(ix+1, iy+2); lft = node(ix,   iy+1)
        C   = node(ix+1, iy+1)

        el2n[:, iel]     .= Int32[BL, BR, TR, bot, rgt, C]
        el2n[:, iel + 1] .= Int32[BL, TR, TL, C, top, lft]
        iel += 2
    end

    return coords, el2n
end

# ---------------------------------------------------------------------------

function main(nels; do_plot = true, nsteps = 50, verbose = true)
    TDev = FEMTools.TA(backend)

    L        = 50e3
    Ω        = (-L..L) × (-L..L)
    element  = ReferenceElement(QuadraticElement{2, 6, Float64})

    coords_cpu, el2n_cpu = build_mesh_T2(Ω, nels)
    mesh = FEMTools.Mesh(backend, coords_cpu, el2n_cpu; order = 2)
    verbose && @printf("mesh: %d nodes, %d T6 elements\n", mesh.nnodes, mesh.nels)

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
    I_Ω, J_Ω = factors(Ω)
    Γ = findall(coords_cpu) do p
        x, y = p
        x == leftendpoint(I_Ω) || x == rightendpoint(I_Ω) ||
        y == leftendpoint(J_Ω) || y == rightendpoint(J_Ω)
    end

    # DOF index arrays and value arrays are kept separate from dr
    Γ_dofs = TDev(Int32.(Γ))
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
        @eval using GeometryBasics

        # --- plot on the linearized (corner-node) triangulation ---
        pts2d = [Point2f(p[1] / 1e3, p[2] / 1e3) for p in coords_cpu]
        faces = [GeometryBasics.TriangleFace(Int(el2n_cpu[1, i]), Int(el2n_cpu[2, i]), Int(el2n_cpu[3, i]))
                 for i in axes(el2n_cpu, 2)]

        fig = Figure(size = (600, 600))
        ax  = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x [km]", ylabel = "y [km]")
        m   = mesh!(ax, pts2d, faces;
                    color      = Array(dr.T),
                    colormap   = :thermal,
                    colorrange = (T_bg, T_bg + ΔT),
                    shading    = NoShading)
        Colorbar(fig[1, 2], m; label = "T [K]")
        display(fig)
    end

    return (; nels = mesh.nels, ndofs = mesh.nnodes, history, solve_time, dr)
end

# skipped when included by the sweep driver (which defines RUN_SWEEP)
if !(@isdefined RUN_SWEEP)
    nels = (30, 30)
    main(nels)
end
