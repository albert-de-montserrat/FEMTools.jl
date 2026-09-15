using DomainSets
using DomainSets: ×
using KernelAbstractions
using StaticArrays
using Printf
using GLMakie
using FEMTools

# const backend   = CUDABackend()
const backend   = CPU()
const workgroup = 128

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

    Lx, Ly   = 10e3, 20e3
    Ω        = (-Lx..Lx) × (-Ly..0e0)
    element  = ReferenceElement(LinearElement{2, 3, Float64})
    mesh     = Mesh(backend, Ω, element, nels)
    mesh_cpu = Mesh(CPU(), Ω, element, nels)

    # --- two-phase material properties ---
    material = ThermalMaterial(;
        k = (3.0, 2.5), Cp = (1200.0, 1100.0), ρ0 = (3300.0, 2700.0),
        α = (3e-5, 2e-5), K = (1e11, 8e10),
    )
    Tref = 273.0            # reference temperature [K]
    Δt  = 1e3 * 365 * 24 * 3600 # time step           [s]

    # --- Dirichlet BCs: T = 1573 K at bottom wall, T = 273 K at top wall ---
    T_bot = 600.0 + 273.0   # 873 K
    T_top = 273.0           # 273 K

    _, J_Ω     = factors(Ω)
    coords_cpu = Array(mesh_cpu.coords)
    Γb = findall(p -> last(p) == leftendpoint(J_Ω),  coords_cpu)
    Γt = findall(p -> last(p) == rightendpoint(J_Ω), coords_cpu)

    Γ_dofs = TDev(vcat(mesh_cpu.DoFs[Γb], mesh_cpu.DoFs[Γt]))
    Γ_vals = TDev(vcat(fill(T_bot, length(Γb)), fill(T_top, length(Γt))))
    Γ_zero = zero(Γ_vals)

    # --- precompute geometry (∂N∂x, dΩ per element per quadrature point) ---
    geo = precompute_geometry(mesh.coords, mesh.el2n, mesh.nels, element)

    # --- ThermalDiffusionDR bundles all solver state and material properties ---
    dr = ThermalDiffusionDR(backend, mesh.nnodes, material; CFL=0.9, ϵ=1e-8)

    # Phase assignment: dr.phases defaults to all-ones (single phase).
    # Overwrite to set a two-phase layout, e.g. upper half = phase 2:
    # copyto!(dr.phases, TDev(Int[last(p) > 0 ? 2 : 1 for p in coords_cpu]))

    dr.T .= (T_bot + T_top) / 2
    apply_dirichlet!(dr.T, Γ_dofs, Γ_vals, backend, workgroup)
    copyto!(dr.T0, dr.T)

    # --- time loop ---
    # Nodes are on a regular (nx+1)×(ny+1) grid regardless of triangle splitting.
    nx, ny = nels
    xs = LinRange(-Lx, Lx, nx + 1) ./ 1e3
    ys = LinRange(-Ly, 0e0, ny + 1) ./ 1e3

    nsteps = 50

    for step in 1:nsteps
        @printf("─── time step %d / %d  (t = %.3g yr) ───\n", step, nsteps, step * Δt / (365*24*3600))
        copyto!(dr.T0, dr.T)
        fill!(dr.∂T∂τ, 0)

        solver!(dr, Δt, mesh, geo, element, Γ_dofs, Γ_zero, Γ_vals, backend, workgroup; Tref = Tref)
    end
    
    fig = Figure(size = (600, 600))
    T_host = reshape(Array(dr.T), nx + 1, ny + 1)
    ax = Axis(fig[1, 1]; title = "t = $(nsteps * Δt / (365*24*3600*1e3)) kyr", aspect = DataAspect())
    heatmap!(ax, xs, ys, T_host; colormap = :thermal, colorrange = (T_top, T_bot))
    Colorbar(fig[1, 2]; colormap = :thermal, limits = (T_top, T_bot), label = "T [K]")
    display(fig)

    return nothing #dr
end

nels = (120, 120)
@time main(nels)
