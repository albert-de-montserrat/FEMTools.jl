using DomainSets
using DomainSets: ×
using KernelAbstractions
using StaticArrays
using Printf
using GLMakie
using FEMTools

# const backend   = CUDABackend()
const backend   = CPU()
const workgroup = 64

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

function main(nels)
    TDev = FEMTools.TA(backend)

    Lx, Ly   = 50e3, 100e3
    Ω        = (-Lx..Lx) × (-Ly..Ly)
    element  = ReferenceElement(QuadraticElement{2, 9, Float64})
    mesh     = Mesh(backend, Ω, element, nels)
    mesh_cpu = Mesh(CPU(), Ω, element, nels)

    # --- two-phase material properties ---
    k   = (3.0,    2.5)     # thermal conductivity   [W m⁻¹ K⁻¹]
    Cp  = (1200.0, 1100.0)  # specific heat          [J kg⁻¹ K⁻¹]
    ρ0  = (3300.0, 2700.0)  # reference density      [kg m⁻³]
    α   = (3e-5,   2e-5)    # thermal expansivity    [K⁻¹]
    K   = (1e11,   8e10)    # bulk modulus           [Pa]
    Tref = 273.0            # reference temperature [K]
    Δt  = 100e3 * 365 * 24 * 3600 # time step             [s]

    # --- Dirichlet BCs: T = 1573 K at bottom wall, T = 273 K at top wall ---
    T_bot = 1300.0 + 273.0   # 1573 K
    T_top = 273.0            # 273 K

    _, J_Ω     = factors(Ω)
    coords_cpu = Array(mesh_cpu.coords)
    Γb = findall(p -> last(p) == leftendpoint(J_Ω),  coords_cpu)
    Γt = findall(p -> last(p) == rightendpoint(J_Ω), coords_cpu)

    # DOF index arrays and value arrays are kept separate from dr
    Γ_dofs = TDev(vcat(mesh_cpu.DoFs[Γb], mesh_cpu.DoFs[Γt]))
    Γ_vals = TDev(vcat(fill(T_bot, length(Γb)), fill(T_top, length(Γt))))
    Γ_zero = zero(Γ_vals)

    # --- precompute geometry (∂N∂x, dΩ per element per quadrature point) ---
    geo = precompute_geometry(mesh.coords, mesh.el2n, mesh.nels, element)

    # --- ThermalDiffusionDR bundles all solver state and material properties ---
    dr = ThermalDiffusionDR(backend, mesh.nnodes, k, Cp, ρ0, α, K; CFL=0.9, ϵ=1e-8)

    # Phase assignment: dr.phases defaults to all-ones (single phase).
    # Overwrite to set a two-phase layout, e.g. upper half = phase 2:
    # copyto!(dr.phases, TDev(Int[last(p) > 0 ? 2 : 1 for p in coords_cpu]))

    # linear initial profile interpolating between bottom (hot) and top (cold)
    # copyto!(dr.T,  TDev(Float64[T_bot + (T_top - T_bot) * (p[2] - leftendpoint(J_Ω)) / (2Ly) for p in coords_cpu]))
    dr.T .= (1300 + 273 * 2) / 2
    apply_dirichlet!(dr.T, Γ_dofs, Γ_vals, backend, workgroup)
    copyto!(dr.T0, dr.T)

    # --- time loop ---
    nsteps = 50
    p_ord  = order(element)
    nx, ny = nels .* p_ord
    xs = LinRange(-Lx, Lx, nx + 1) ./ 1e3
    ys = LinRange(-Ly, Ly, ny + 1) ./ 1e3

    fig = Figure(size = (600, 600))

    for step in 1:nsteps
        @printf("─── time step %d / %d  (t = %.3g s) ───\n", step, nsteps, step * Δt)
        copyto!(dr.T0, dr.T)
        fill!(dr.∂T∂τ, 0)

        # one call — everything lives in FEMTools
        solver!(dr, Δt, mesh, geo, element, Γ_dofs, Γ_zero, Γ_vals, backend, workgroup; ncheck = 50, Tref = Tref)
    end

    T_host = reshape(Array(dr.T), nx + 1, ny + 1)
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    heatmap!(ax, xs, ys, T_host; colormap = :thermal, colorrange = (T_top, T_bot))
    Colorbar(fig[1, 2]; colormap = :thermal, limits = (T_top, T_bot), label = "T [K]")
    display(fig)

    return dr
end

nels = (30, 30)
main(nels)
