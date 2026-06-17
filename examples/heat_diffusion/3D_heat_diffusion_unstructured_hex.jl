# Heat diffusion on a 3D box with cylindrical inclusions, meshed with Gmsh.jl
# (linear tetrahedral elements, P1).
#
# 3D analogue of 2D_heat_diffusion_unstructured.jl:
#   Domain   x ∈ [−Lx, Lx],  y ∈ [−Ly, 0],  z ∈ [−Lz, Lz]
#   Holes    two z-aligned cylinders  (same (cx, cy, r) as the 2D example)
#   BCs      T = T_top  at y = 0   (cold surface)
#            T = T_bot  at y = −Ly (hot base)
#            T = T_cyl  at each cylindrical wall
#            Neumann (zero flux) on all remaining faces
#   Solver   ThermalDiffusionDR  (pseudo-transient dynamic relaxation)
#   Output   VTK time-series + progress printed to stdout
#
# Requires Gmsh.jl  (install once: import Pkg; Pkg.add("Gmsh"))

import Pkg
# Pkg.activate(joinpath(@__DIR__, "../.."))

using Gmsh
using StaticArrays
using KernelAbstractions
using Printf
using WriteVTK
using TimerOutputs
using FEMTools

const backend   = CPU()
const workgroup = 128

# ---------------------------------------------------------------------------
# Geometry precomputation  (∂N∂x, dΩ per element per quadrature point)
# ---------------------------------------------------------------------------

function precompute_geometry(coords, el2n, nels,
                              element::ReferenceElement{T}) where T <: AbstractElement{3, N} where N
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
# Mesh generation with Gmsh
# ---------------------------------------------------------------------------

"""
    build_mesh(; Lx, Ly, Lz, cylinders, mesh_size)

Generate a tetrahedral mesh of [−Lx,Lx]×[−Ly,0]×[−Lz,Lz] with z-aligned
cylindrical holes using Gmsh.jl.

Each entry of `cylinders` is `(cx, cy, r)` for a cylinder whose axis runs
parallel to z through the point `(cx, cy)` in the x-y plane.

Returns `(coords, el2n, top_nodes, bottom_nodes, cyl_nodes_per_cyl)`.
"""
function build_mesh(; Lx, Ly, Lz, cylinders, mesh_size = 2e3)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("heat_3d")

    # Box: x ∈ [−Lx, Lx], y ∈ [−Ly, 0], z ∈ [−Lz, Lz]
    box_tag = gmsh.model.occ.addBox(-Lx, -Ly, -Lz, 2Lx, Ly, 2Lz)

    # Cylinders extend slightly beyond z = ±Lz for a clean boolean cut
    δ = 0.01Lz
    cyl_tags = Int[]
    for (cx, cy, r) in cylinders
        tag = gmsh.model.occ.addCylinder(cx, cy, -Lz - δ, 0, 0, 2(Lz + δ), r)
        push!(cyl_tags, tag)
    end

    gmsh.model.occ.cut([(3, box_tag)], [(3, t) for t in cyl_tags])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.model.mesh.generate(3)

    # --- Node extraction ---
    nodetags, coords_flat, _ = gmsh.model.mesh.getNodes()
    nnodes  = length(nodetags)
    max_tag = Int(maximum(nodetags))
    tag2idx = zeros(Int32, max_tag)
    for (i, tag) in enumerate(nodetags)
        tag2idx[Int(tag)] = Int32(i)
    end

    coords_cpu = [SVector{3, Float64}(
        coords_flat[3(i-1)+1],
        coords_flat[3(i-1)+2],
        coords_flat[3(i-1)+3],
    ) for i in 1:nnodes]

    # --- Tet4 element extraction (Gmsh element type 4) ---
    elemtypes, _, elemnodetags = gmsh.model.mesh.getElements(3)
    tet_idx = findfirst(==(4), elemtypes)
    isnothing(tet_idx) && error("No tet4 elements found in 3D mesh.")

    tet_flat = elemnodetags[tet_idx]
    nels     = length(tet_flat) ÷ 4
    el2n     = Matrix{Int32}(undef, 4, nels)
    for iel in 1:nels, i in 1:4
        el2n[i, iel] = tag2idx[Int(tet_flat[4(iel-1)+i])]
    end

    gmsh.finalize()

    # --- Boundary node detection ---
    tol = max(Lx, Ly) * 1e-6
    top_nodes    = Int32[i for i in 1:nnodes if abs(coords_cpu[i][2])      < tol]
    bottom_nodes = Int32[i for i in 1:nnodes if abs(coords_cpu[i][2] + Ly) < tol]

    cyl_nodes_per_cyl = map(cylinders) do (cx, cy, r)
        tol_r = r * 1e-4
        Int32[i for i in 1:nnodes if
            abs(sqrt((coords_cpu[i][1] - cx)^2 + (coords_cpu[i][2] - cy)^2) - r) < tol_r]
    end

    return coords_cpu, el2n, top_nodes, bottom_nodes, cyl_nodes_per_cyl
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(; mesh_size = 3e3)
    FP   = Float64
    TDev = FEMTools.TA(backend)

    # Domain geometry (same x-y footprint as 2D unstructured example)
    Lx, Ly, Lz = 15e3, 30e3, 10e3
    cylinders   = [(-7e3,  -10e3, 2.5e3),    # (cx, cy, r)
                   (  0.0, -11.5e3, 2e3)]
    T_holes = FP[873, 1173]

    # Generate mesh
    @printf("Generating Gmsh mesh …\n")
    to = TimerOutput()
    @timeit to "mesh" coords_cpu, el2n_cpu, top_nodes, bottom_nodes, cyl_nodes_per_cyl =
        build_mesh(; Lx, Ly, Lz, cylinders, mesh_size)
    mesh = FEMTools.Mesh(backend, coords_cpu, el2n_cpu)
    @printf("mesh: %d nodes, %d elements\n", mesh.nnodes, mesh.nels)

    # Dirichlet BCs --------------------------------------------------------
    T_top    = FP(273)    # cold top wall  [K]
    T_bottom = FP(873)    # hot bottom wall [K]

    hole_dof_vecs = [fill(T_holes[h], length(cyl_nodes_per_cyl[h]))
                     for h in eachindex(cylinders)]

    Γ_dofs = TDev(vcat(top_nodes, bottom_nodes, cyl_nodes_per_cyl...))
    Γ_vals = TDev(vcat(
        fill(T_top,    length(top_nodes)),
        fill(T_bottom, length(bottom_nodes)),
        hole_dof_vecs...,
    ))
    Γ_zero = zero(Γ_vals)

    @printf("Dirichlet: %d top, %d bottom", length(top_nodes), length(bottom_nodes))
    for (h, hn) in enumerate(cyl_nodes_per_cyl)
        @printf(", %d cyl-%d (T=%.0f K)", length(hn), h, T_holes[h])
    end
    @printf("\n")

    # Element and geometry ------------------------------------------------
    element = ReferenceElement(LinearElement{3, 4, FP})
    @timeit to "geo" geo = precompute_geometry(mesh.coords, mesh.el2n, mesh.nels, element)

    # Material properties (single homogeneous phase) ----------------------
    k    = (FP(3.0),)
    Cp   = (FP(1200.0),)
    ρ0   = (FP(3300.0),)
    α    = (FP(3e-5),)
    K    = (FP(1e11),)
    Tref = FP(273.0)
    g    = SA[FP(0.0), -FP(9.81), FP(0.0)]
    Δt   = FP(20e3 * 365.25 * 24 * 3600)   # 20 kyr time step [s]

    # Solver state --------------------------------------------------------
    dr = ThermalDiffusionDR(backend, mesh.nnodes, k, Cp, ρ0, α, K;
                            CFL = FP(0.9), ϵ = FP(1e-8))

    T_init = FP[T_top + (T_bottom - T_top) * (-coords_cpu[i][2] / Ly) for i in eachindex(coords_cpu)]
    copyto!(dr.T, T_init)
    apply_dirichlet!(dr.T, Γ_dofs, Γ_vals, backend, workgroup)
    copyto!(dr.T0, dr.T)

    # Lithostatic pressure: solve ∫ ∇P·∇v dΩ = ∫ ρ(T) g·∇v dΩ on the initial T.
    lp_dr = LithostaticPressureDR(backend, mesh.nnodes, ρ0, α, K; CFL = FP(0.9), ϵ = FP(1e-2))
    copyto!(lp_dr.T, dr.T)
    P0_litho = FP[ρ0[1] * (-g[2]) * (-coords_cpu[i][2]) for i in eachindex(coords_cpu)]
    copyto!(lp_dr.P, P0_litho)
    Γ_P_dofs      = TDev(top_nodes)
    Γ_P_zero_vals = zero(dr.P[top_nodes])
    @printf("solving initial lithostatic pressure …\n")
    @timeit to "litho P init" solver!(lp_dr, mesh, geo, element, Γ_P_dofs, Γ_P_zero_vals, Γ_P_zero_vals,
                                       backend, workgroup; ncheck = 50, Tref = Tref, g = g)
    copyto!(dr.P, lp_dr.P)

    # VTK time-series setup -----------------------------------------------
    out_dir = "output"
    mkpath(out_dir)
    vtk_pts = zeros(3, mesh.nnodes)
    for (i, p) in enumerate(coords_cpu)
        vtk_pts[1, i] = p[1]; vtk_pts[2, i] = p[2]; vtk_pts[3, i] = p[3]
    end
    cells = [MeshCell(VTKCellTypes.VTK_TETRA, el2n_cpu[:, i]) for i in axes(el2n_cpu, 2)]
    pvd    = paraview_collection(joinpath(out_dir, "heat_diffusion_3d_tet"))
    t_phys = FP(0)

    # Time loop -----------------------------------------------------------
    nsteps = 75
    for step in 1:nsteps
        @printf("─── time step %2d / %d ───\n", step, nsteps)
        copyto!(dr.T0, dr.T)
        fill!(dr.∂T∂τ, 0)
        @timeit to "solver" solver!(dr, Δt, mesh, geo, element, Γ_dofs, Γ_zero, Γ_vals,
                                     backend, workgroup; ncheck = 100, Tref = Tref)
        t_phys += Δt

        @timeit to "vtk" vtk_grid(joinpath(out_dir, "heat_diffusion_3d_tet_$step"), vtk_pts, cells) do vtk
            vtk["T"] = Array(dr.T)
            vtk["P"] = Array(dr.P)
            pvd[t_phys] = vtk
        end
    end
    vtk_save(pvd)
    @printf("saved %s (%d steps)\n", joinpath(out_dir, "heat_diffusion_3d_tet.pvd"), nsteps)
    print_timer(to; title = "3D heat diffusion — tet", sortby = :firstexec)

    @printf("mesh: %d nodes, %d elements\n", mesh.nnodes, mesh.nels)
    return nothing
end

main(; mesh_size = 1e3)
