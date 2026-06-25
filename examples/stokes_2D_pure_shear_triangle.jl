import Pkg
Pkg.activate(@__DIR__)

using Printf
using Statistics
using StaticArrays
using LinearAlgebra
using DomainSets
using DomainSets: ×
using KernelAbstractions
using Triangulate
using FEMTools
using GLMakie: Figure, Axis, Colorbar, poly!, scatterlines!, lines!, Point2f, DataAspect

const backend   = CPU()
const workgroup = 128

function precompute_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels) where N
    precompute_geometry_kernel!(backend, workgroup)(
        geo, coords, el2n, ∂N∂ξq, ω, Val(N);
        ndrange = nels,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function update_rate!(∂u∂τ, R, PC, β, ndofs)
    update_rate_kernel!(backend, workgroup)(
        ∂u∂τ, R, PC, β;
        ndrange = ndofs,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function update_variable!(u, ∂u∂τ, α_dr, ndofs)
    update_variable_kernel!(backend, workgroup)(
        u, ∂u∂τ, α_dr;
        ndrange = ndofs,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function remove_pressure_mean!(P, MP)
    p_mean = sum(P .* MP) / sum(MP)
    @. P -= p_mean
    return p_mean
end

@inline _phase_at_postprocess(phases::AbstractMatrix, _, i, iel) = Int(phases[i, iel])
@inline _phase_at_postprocess(phases, local_nodes, i, _) = Int(phases[local_nodes[i]])
@inline _phase_loc_postprocess(phases, local_nodes, iel, ::Val{N}) where N =
    SVector{N}(ntuple(i -> _phase_at_postprocess(phases, local_nodes, i, iel), Val(N)))

function compute_strain_rate_stress_postprocess(
    vx, vy,
    el2n_v,
    geo_v,
    phases_v,
    τ_old,
    η, G, Δt,
    element_v::ReferenceElement{TV},
) where {TV <: AbstractElement{2, NV}} where NV
    nels = size(el2n_v, 2)
    Nq = shape_function_values(element_v)

    εxx = zeros(FP, nels)
    εyy = zeros(FP, nels)
    εzz = zeros(FP, nels)
    εxy = zeros(FP, nels)
    εII = zeros(FP, nels)
    τxx = zeros(FP, nels)
    τyy = zeros(FP, nels)
    τzz = zeros(FP, nels)
    τxy = zeros(FP, nels)
    τII = zeros(FP, nels)

    for iel in 1:nels
        local_nodes = SVector{NV}(ntuple(i -> el2n_v[i, iel], Val(NV)))
        vxloc = SVector{NV}(ntuple(i -> vx[local_nodes[i]], Val(NV)))
        vyloc = SVector{NV}(ntuple(i -> vy[local_nodes[i]], Val(NV)))
        τxx_old_loc = SVector{NV}(ntuple(i -> τ_old[1][local_nodes[i]], Val(NV)))
        τyy_old_loc = SVector{NV}(ntuple(i -> τ_old[2][local_nodes[i]], Val(NV)))
        τxy_old_loc = SVector{NV}(ntuple(i -> τ_old[3][local_nodes[i]], Val(NV)))
        phase_loc = _phase_loc_postprocess(phases_v, local_nodes, iel, Val(NV))
        geo_el = geo_v[iel]
        volume = zero(FP)

        for q in eachindex(geo_el)
            ∂N∂x, dΩ = geo_el[q]
            Nv = Nq[q]

            ∇vx = ∂N∂x' * vxloc
            ∇vy = ∂N∂x' * vyloc

            εxx_q = ∇vx[1]
            εyy_q = ∇vy[2]
            εzz_q = zero(εxx_q)
            εxy_q = (∇vx[2] + ∇vy[1]) / 2

            tr = (εxx_q + εyy_q + εzz_q) / 3
            εxx_dev = εxx_q - tr
            εyy_dev = εyy_q - tr
            εzz_dev = εzz_q - tr

            ηq = FEMTools.interp2ip_phase(Nv, η, phase_loc)
            invGq = FEMTools.interp2ip_phase(Nv, map(inv, G), phase_loc)
            ηeff_q = inv(inv(ηq) + invGq / Δt)
            inv_2Gdt = invGq / (2 * Δt)

            τxx_old_q = dot(Nv, τxx_old_loc)
            τyy_old_q = dot(Nv, τyy_old_loc)
            τxy_old_q = dot(Nv, τxy_old_loc)
            τzz_old_q = -(τxx_old_q + τyy_old_q)

            τxx_q = 2 * ηeff_q * (εxx_dev + τxx_old_q * inv_2Gdt)
            τyy_q = 2 * ηeff_q * (εyy_dev + τyy_old_q * inv_2Gdt)
            τzz_q = 2 * ηeff_q * (εzz_dev + τzz_old_q * inv_2Gdt)
            τxy_q = 2 * ηeff_q * (εxy_q   + τxy_old_q * inv_2Gdt)

            εII_q = sqrt((εxx_dev^2 + εyy_dev^2 + εzz_dev^2) / 2 + εxy_q^2)
            τII_q = sqrt((τxx_q^2 + τyy_q^2 + τzz_q^2) / 2 + τxy_q^2)

            εxx[iel] += εxx_q * dΩ
            εyy[iel] += εyy_q * dΩ
            εzz[iel] += εzz_q * dΩ
            εxy[iel] += εxy_q * dΩ
            εII[iel] += εII_q * dΩ
            τxx[iel] += τxx_q * dΩ
            τyy[iel] += τyy_q * dΩ
            τzz[iel] += τzz_q * dΩ
            τxy[iel] += τxy_q * dΩ
            τII[iel] += τII_q * dΩ
            volume += dΩ
        end

        εxx[iel] /= volume
        εyy[iel] /= volume
        εzz[iel] /= volume
        εxy[iel] /= volume
        εII[iel] /= volume
        τxx[iel] /= volume
        τyy[iel] /= volume
        τzz[iel] /= volume
        τxy[iel] /= volume
        τII[iel] /= volume
    end

    return (;
        εxx, εyy, εzz, εxy, εII,
        τxx, τyy, τzz, τxy, τII,
        tauII = τII,
    )
end

function update_old_stress_from_cells!(τ_old, post, el2n_v, nnodes_v)
    τxx_nodes = zeros(FP, nnodes_v)
    τyy_nodes = zeros(FP, nnodes_v)
    τxy_nodes = zeros(FP, nnodes_v)
    counts = zeros(Int, nnodes_v)

    for iel in axes(el2n_v, 2)
        for a in axes(el2n_v, 1)
            inode = el2n_v[a, iel]
            τxx_nodes[inode] += post.τxx[iel]
            τyy_nodes[inode] += post.τyy[iel]
            τxy_nodes[inode] += post.τxy[iel]
            counts[inode] += 1
        end
    end

    for inode in eachindex(counts)
        if counts[inode] > 0
            τxx_nodes[inode] /= counts[inode]
            τyy_nodes[inode] /= counts[inode]
            τxy_nodes[inode] /= counts[inode]
        end
    end

    copyto!(τ_old[1], τxx_nodes)
    copyto!(τ_old[2], τyy_nodes)
    copyto!(τ_old[3], τxy_nodes)
    return nothing
end

function write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post)
    NP = size(el2nP_cpu, 1)
    vtk_nodes = sort!(unique(vec(el2nP_cpu)))
    vtk_node_map = zeros(Int32, length(coords_v))
    for (new_i, old_i) in enumerate(vtk_nodes)
        vtk_node_map[old_i] = Int32(new_i)
    end

    vtk_P = zeros(FP, length(vtk_nodes))
    vtk_P_count = zeros(Int, length(vtk_nodes))
    for iel in 1:mesh_stokes.nels
        for a in 1:NP
            inode = vtk_node_map[el2nP_cpu[a, iel]]
            vtk_P[inode] += P_cpu[DoFsP_cpu[a, iel]]
            vtk_P_count[inode] += 1
        end
    end
    @. vtk_P /= vtk_P_count

    vtk_Vx = [vx_cpu[old_i] for old_i in vtk_nodes]
    vtk_Vy = [vy_cpu[old_i] for old_i in vtk_nodes]
    vtk_V  = hypot.(vtk_Vx, vtk_Vy)

    open(vtk_path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, "FEMTools Stokes 2D pure shear")
        println(io, "ASCII")
        println(io, "DATASET UNSTRUCTURED_GRID")

        println(io, "POINTS $(length(vtk_nodes)) float")
        for old_i in vtk_nodes
            c = coords_v[old_i]
            println(io, "$(c[1]) $(c[2]) 0.0")
        end

        println(io, "CELLS $(mesh_stokes.nels) $(4 * mesh_stokes.nels)")
        for iel in 1:mesh_stokes.nels
            i1 = vtk_node_map[el2nP_cpu[1, iel]] - 1
            i2 = vtk_node_map[el2nP_cpu[2, iel]] - 1
            i3 = vtk_node_map[el2nP_cpu[3, iel]] - 1
            println(io, "3 $i1 $i2 $i3")
        end

        println(io, "CELL_TYPES $(mesh_stokes.nels)")
        for _ in 1:mesh_stokes.nels
            println(io, "5") # VTK_TRIANGLE
        end

        println(io, "POINT_DATA $(length(vtk_nodes))")
        for (name, field) in (("P", vtk_P), ("Vx", vtk_Vx), ("Vy", vtk_Vy), ("V", vtk_V))
            println(io, "SCALARS $name float 1")
            println(io, "LOOKUP_TABLE default")
            for value in field
                println(io, value)
            end
        end

        println(io, "CELL_DATA $(mesh_stokes.nels)")
        for (name, field) in (
            ("strain_xx", post.εxx),
            ("strain_yy", post.εyy),
            ("strain_zz", post.εzz),
            ("strain_xy", post.εxy),
            ("strain_II", post.εII),
            ("tau_xx", post.τxx),
            ("tau_yy", post.τyy),
            ("tau_zz", post.τzz),
            ("tau_xy", post.τxy),
            ("tau_II", post.τII),
        )
            println(io, "SCALARS $name float 1")
            println(io, "LOOKUP_TABLE default")
            for value in field
                println(io, value)
            end
        end
    end
    return nothing
end

function build_triangle_t7_inclusion_mesh(; Lx, Ly, cx, cy, r, n_circle = 96, max_area = nothing)
    rect_pts  = Cdouble[0.0 Lx  Lx 0.0;
                        0.0 0.0 Ly Ly]
    rect_segs = Cint[1 2; 2 3; 3 4; 4 1]'

    θ = range(0, 2π; length = n_circle + 1)[1:end-1]
    circ_pts = Matrix{Cdouble}(hcat(cx .+ r .* cos.(θ), cy .+ r .* sin.(θ))')
    circ_segs = Matrix{Cint}(hcat([
        [4 + i; 4 + mod1(i + 1, n_circle)] for i in 1:n_circle
    ]...))

    tio = TriangulateIO()
    tio.pointlist = hcat(rect_pts, circ_pts)
    tio.segmentlist = hcat(rect_segs, circ_segs)

    flags = isnothing(max_area) ? "pqo2Q" : "pq30o2a$(max_area)Q"
    result, _ = triangulate(flags, tio)

    pts = result.pointlist
    tris_t6 = Matrix{Int32}(result.trianglelist)
    coords = [SVector{2, Float64}(pts[1, i], pts[2, i]) for i in axes(pts, 2)]

    # Triangle's second-order boundary nodes lie on straight constrained
    # segments. Project the circular-interface nodes back to the analytical
    # radius so the inclusion boundary is fitted by the high-order geometry.
    sagitta = r * (1 - cos(π / n_circle))
    circle_tol = max(2.5 * sagitta, 100eps(Float64) * max(Lx, Ly))
    for i in eachindex(coords)
        dx = coords[i][1] - cx
        dy = coords[i][2] - cy
        radius = hypot(dx, dy)
        if abs(radius - r) ≤ circle_tol && radius > 0
            coords[i] = SVector{2, Float64}(cx + r * dx / radius, cy + r * dy / radius)
        end
    end

    n_t6 = length(coords)
    nels = size(tris_t6, 2)
    el2n = Matrix{Int32}(undef, 7, nels)
    el2n[1:3, :] .= tris_t6[1:3, :]
    el2n[4, :] .= tris_t6[6, :] # FEMTools node 4 = mid(1, 2)
    el2n[5, :] .= tris_t6[4, :] # FEMTools node 5 = mid(2, 3)
    el2n[6, :] .= tris_t6[5, :] # FEMTools node 6 = mid(3, 1)
    sizehint!(coords, n_t6 + nels)
    for iel in 1:nels
        c1 = coords[tris_t6[1, iel]]
        c2 = coords[tris_t6[2, iel]]
        c3 = coords[tris_t6[3, iel]]
        push!(coords, (c1 + c2 + c3) / 3)
        el2n[7, iel] = Int32(n_t6 + iel)
    end

    tol = 100eps(Float64) * max(Lx, Ly)
    outer_nodes = Int32[
        i for i in 1:n_t6
        if abs(coords[i][1]) ≤ tol ||
           abs(coords[i][1] - Lx) ≤ tol ||
           abs(coords[i][2]) ≤ tol ||
           abs(coords[i][2] - Ly) ≤ tol
    ]
    circle_nodes = Int32[
        i for i in 1:n_t6
        if abs(hypot(coords[i][1] - cx, coords[i][2] - cy) - r) ≤ circle_tol
    ]

    return coords, el2n, sort!(unique!(outer_nodes)), sort!(unique!(circle_nodes))
end

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const FP = Float64
function main(; nsteps = 20, n_circle = 96, max_area = 1 / (2 * 32^2), Δt = 0.167, show_plot = true)
    # Domain
    Lx, Ly = 1.0, 1.0

    # Background pure-shear strain rate (non-dimensional)
    ε̇_bg = 1.0

    # Material (2 phases: matrix + inclusion)
    η     = (1.0,     1.0)   # shear viscosity
    α     = (0.0,     0.0)   # thermal expansivity  (zero → isothermal)
    ρ0    = (0.0,     0.0)   # reference density
    K     = (4,         4)   # bulk modulus  (Inf → incompressible)
    G     = (1.0,     0.5)   # Shear modulus  (Inf → purely viscous)
    G_stokes = NTuple{2, FP}(G)
    τy   = 1.6 /cosd(30) #* Inf
    plastic = DruckerPrager(
        NTuple{2, FP}((π/6, π/6)),        # friction angle ϕ  [rad]
        NTuple{2, FP}((0,   0)),          # dilation angle Ψ  [rad]
        NTuple{2, FP}((τy, τy)),          # cohesion C        [Pa]
        NTuple{2, FP}((8.0e-3,  8.0e-3)), # η_reg             [Pa s]
        NTuple{2, FP}(K),                 # Kb ≈ bulk modulus [Pa]
    )
    γfact = 20 
    ηbi   = if K[1]<Inf 
        γfact * sum(η) / length(η)
    else
        K[1] * Δt
    end
    ηb    = (ηbi,     ηbi)   # bulk  viscosity
   
    # plastic = nothing
    g     = (0.0,     0.0)   # gravity vector
    Tref  = 0.0

    # DR solver
    ncheck = 50          # convergence check interval
    ϵ_tol  = 1e-6        # relative residual tolerance

    # Inclusion geometry. The Triangle PSLG uses this circle as an internal
    # constrained boundary, so no element crosses the material interface.
    r_incl = FP(0.1)
    cx     = FP(Lx / 2)
    cy     = FP(Ly / 2)

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 7, FP})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, FP})      # P1-disc

    coords_v_cpu, el2n_v_cpu, outer_nodes, circle_nodes = build_triangle_t7_inclusion_mesh(;
        Lx = FP(Lx), Ly = FP(Ly),
        cx, cy, r = r_incl,
        n_circle,
        max_area,
    )
    DoFs_v_cpu = Int32.(1:length(coords_v_cpu))
    mesh_v = FEMTools.Mesh(
        element_v, nothing, nothing,
        coords_v_cpu, DoFs_v_cpu, el2n_v_cpu, outer_nodes,
    )
    mesh_stokes = MixedMesh(mesh_v, element_P)

    @info "Triangle mixed mesh (T7/P1-disc)" nnodes_v=mesh_stokes.nnodes nnodes_P=mesh_stokes.nnodesP nels=mesh_stokes.nels n_circle max_area n_interface_nodes=length(circle_nodes)

    # ---------------------------------------------------------------------------
    # Geometry precompute  (both fields evaluated at velocity integration points)
    # ---------------------------------------------------------------------------

    ip_v  = element_v.integration_points
    NQ_v  = length(ip_v.ω)
    NV    = length(element_v)
    NP    = length(element_P)

    ξq_v    = ntuple(q -> SVector(ip_v.ξ[q], ip_v.η[q]), NQ_v)
    ∂N∂ξq_v = ntuple(q -> eval_shape_function_jacobian(element_v, ξq_v[q]), NQ_v)
    ∂N∂ξq_P = ntuple(q -> eval_shape_function_jacobian(element_P, ξq_v[q]), NQ_v)

    geo_v = Vector{NTuple{NQ_v, Tuple{SMatrix{NV, 2, FP, 2NV}, FP}}}(undef, mesh_stokes.nels)
    geo_P = Vector{NTuple{NQ_v, Tuple{SMatrix{NP, 2, FP, 2NP}, FP}}}(undef, mesh_stokes.nels)

    precompute_geometry!(geo_v, mesh_stokes.coords, mesh_stokes.el2n, ∂N∂ξq_v, ip_v.ω, Val(NV), mesh_stokes.nels)
    precompute_geometry!(geo_P, mesh_stokes.coords, mesh_stokes.el2nP, ∂N∂ξq_P, ip_v.ω, Val(NP), mesh_stokes.nels)

    # ---------------------------------------------------------------------------
    # StokesDR struct
    # ---------------------------------------------------------------------------

    dr = StokesDR(
        backend,
        mesh_stokes.nnodes,
        mesh_stokes.nnodesP,
        NTuple{2, FP}(η), NTuple{2, FP}(ηb), NTuple{2, FP}(α);
        ρ0   = NTuple{2, FP}(ρ0),
        K    = NTuple{2, FP}(K),
        g    = NTuple{2, FP}(g),
        Tref = FP(Tref),
        CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.9,
    )
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

    # ---------------------------------------------------------------------------
    # Phase assignment — circular inclusion
    # ---------------------------------------------------------------------------

    in_incl(c) = (c[1] - cx)^2 + (c[2] - cy)^2 ≤ r_incl^2

    coords_v     = Array(mesh_stokes.coords)
    el2n_v_cpu   = Array(mesh_stokes.el2n)
    phases_v_cpu = Int[in_incl(c) ? 2 : 1 for c in coords_v]
    copyto!(dr.phases_v, phases_v_cpu)

    el2nP_cpu    = Array(mesh_stokes.el2nP)
    DoFsP_cpu    = Array(mesh_stokes.DoFsP)
    phases_P_cpu = Int[
        in_incl(coords_v[el2nP_cpu[mod1(d, 3), cld(d, 3)]]) ? 2 : 1
        for d in 1:mesh_stokes.nnodesP
    ]
    copyto!(dr.phases_P, phases_P_cpu)

    element_phases = Vector{Int}(undef, mesh_stokes.nels)
    for iel in 1:mesh_stokes.nels
        c1 = coords_v[el2n_v_cpu[1, iel]]
        c2 = coords_v[el2n_v_cpu[2, iel]]
        c3 = coords_v[el2n_v_cpu[3, iel]]
        element_phases[iel] = in_incl((c1 + c2 + c3) / 3) ? 2 : 1
    end
    phases_v_el = Matrix{Int}(undef, NV, mesh_stokes.nels)
    phases_P_el = Matrix{Int}(undef, NP, mesh_stokes.nels)
    for iel in 1:mesh_stokes.nels
        phases_v_el[:, iel] .= element_phases[iel]
        phases_P_el[:, iel] .= element_phases[iel]
    end

    @info "Phases" n_incl_nodes=count(==(2), phases_v_cpu) n_incl_elements=count(==(2), element_phases)

    # ---------------------------------------------------------------------------
    # Boundary conditions — pure shear + free slip
    #   left/right walls  (x = 0 or Lx): fix Vx = +ε̇·(x − Lx/2),  Vy free
    #   top/bottom walls  (y = 0 or Ly): fix Vy = −ε̇·(y − Ly/2),  Vx free
    # ---------------------------------------------------------------------------

    coords  = Array(mesh_v.coords)
    Γnodes  = Array(mesh_v.Γnodes)
    tol_bc  = 100 * eps(Float64) * max(Lx, Ly)

    lr_nodes = Int32[n for n in Γnodes if abs(coords[n][1])      ≤ tol_bc ||
                                          abs(coords[n][1] - Lx) ≤ tol_bc]
    tb_nodes = Int32[n for n in Γnodes if abs(coords[n][2])      ≤ tol_bc ||
                                          abs(coords[n][2] - Ly) ≤ tol_bc]

    bc_vx_lr = FP[ ε̇_bg * (coords[n][1] - Lx / 2) for n in lr_nodes]
    bc_vy_tb = FP[-ε̇_bg * (coords[n][2] - Ly / 2) for n in tb_nodes]

    apply_bc!(dr.vx, DirichletBoundaryCondition(nothing, lr_nodes, bc_vx_lr))
    apply_bc!(dr.vy, DirichletBoundaryCondition(nothing, tb_nodes, bc_vy_tb))

    zero_lr = zero(bc_vx_lr)
    zero_tb = zero(bc_vy_tb)

    @info "BCs" n_lr = length(lr_nodes) n_tb = length(tb_nodes) max_vx = maximum(abs, bc_vx_lr)

    # ---------------------------------------------------------------------------
    # Seed preconditioners — used on the very first kernel call before the
    # ForwardDiff Jacobian has been assembled.  Overwritten by Jacobian on iter 1.
    # ---------------------------------------------------------------------------

    h    = sqrt(Lx * Ly / mesh_stokes.nels)
    ηmax = FP(maximum(η))

    Δτ_V_seed = dr.CFL_v * h^2 / (4 * ηmax)

    fill!(dr.PC_vx, 1 / Δτ_V_seed)
    fill!(dr.PC_vy, 1 / Δτ_V_seed)

    # FEM pressure residuals are assembled in weak form:
    #
    #     RP_i = ∫ N_i (-∇⋅v) dΩ
    #
    # The Arrow-Hurwicz pressure step and numerical pressure
    # correction are calibrated for that pointwise residual.  If we feed the weak
    # residual directly into Pnum or P += γP*RP/M_P, the update is scaled by element
    # volume and pressure convergence stalls/refines incorrectly.
    #
    # Use the helper to assemble both:
    #   dr.PC_P = M_P = ∫ N_i dΩ
    #   γP      = local viscosity-weighted pressure update scale
    # Then γP * RP/M_P matches the pointwise FD-style pressure correction, but
    # adapts the pressure step to viscosity contrasts.
    γP = KernelAbstractions.zeros(backend, FP, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        dr.PC_P, γP,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_P, mesh_stokes.nels,
        element_v, element_P,
        phases_v_el, dr.η, FP(γfact),
        backend, workgroup,
    )

    # Δt = Δt === nothing ? FP(0.5 / max(abs(ε̇_bg), eps(FP))) : FP(Δt)
    Δt = 0.167
    time_history = zeros(FP, nsteps)
    mean_tauII_history = zeros(FP, nsteps)

    # ---------------------------------------------------------------------------
    # Powell-Hestenes / DYREL-style Stokes solver
    #
    # Structure (mirrors DYREL / JustRelax solver_DR_FD.jl):
    #
    #   Outer Powell-Hestenes loop (itPH)
    #   ├─ assemble physical Rv and RP
    #   ├─ compute err and set inner tolerance ϵ_vel = err * rel_drop
    #   ├─ Inner DR loop for velocity while err > ϵ_vel
    #   │   ├─ copy previous residuals
    #   │   ├─ assemble RP from current v
    #   │   ├─ Pnum = γP·RP/M_P
    #   │   ├─ assemble augmented Rv with Pnum
    #   │   ├─ update v with damped pseudo-transient DR
    #   │   └─ every ncheck: update λmin, λmax, αV, βV
    #   └─ P += γP·RP/M_P  (Arrow-Hurwicz pressure update)
    # ---------------------------------------------------------------------------

    niter_PH      = 20       # max outer PH iterations
    niter_inner   = 50_000    # max inner DR iterations per PH step
    total_iterMax = 50_000   # max total inner DR iterations
    rel_drop0     = FP(0.75) # inner convergence: velocity residual drops by this factor
    # rel_drop0     = FP(1e-2) # inner convergence: velocity residual drops by this factor
    verbose_DR    = false

    @info "Starting PH/DYREL-style Stokes solver" nsteps Δt niter_PH niter_inner ncheck ϵ_tol

    # λ_min helper (Rayleigh quotient from consecutive residual snapshots).
    # JustRelax computes this with the actual velocity increment, dV = βV*dτV*dVdτ.
    _λmin(step, rate, ΔR, PC) = begin
        dV = step .* rate
        denom = sum(dV .^ 2)
        denom == 0 ? FP(0) : abs(sum(dV .* (ΔR ./ PC))) / denom
    end

    # Damped DYREL/Chebyshev step from spectral step Δτ and damping λmin.
    # Clamp c so that β = (2-c·Δτ)/(2+c·Δτ) ≥ 0; without the clamp the row-sum
    # preconditioner (λmax≈1, λmin≈1) drives c·Δτ > 2, flipping momentum sign and
    # causing a divergence spike when free-slip boundary DOFs have large rates.
    _cheb(Δτ, λmin, c_fact) = begin
        c = 2 * √(λmin) * c_fact
        (2 * Δτ^2 / (2 + c * Δτ), (2 - c * Δτ) / (2 + c * Δτ))
    end

    el2n_v_cpu = Array(mesh_stokes.el2n)
    out_dir = joinpath(@__DIR__, "output_stokes")
    mkpath(out_dir)
    post = nothing

    for istep in 1:nsteps
        t = istep * Δt
        time_history[istep] = t
        copyto!(dr.P0, dr.P)
        copyto!(dr.T0, dr.T)
        fill!(dr.∂vx∂τ, 0)
        fill!(dr.∂vy∂τ, 0)
        fill!(dr.Rv_x0, 0)
        fill!(dr.Rv_y0, 0)
        @info "Physical time step" istep nsteps t

    α_vx = FP(0)
    β_vx = FP(0)
    α_vy = FP(0)
    β_vy = FP(0)
    err_min = FP(Inf)
    err = FP(2ϵ_tol)
    err_v0 = FP(0)
    err_P0 = FP(0)
    err_v00 = FP(0)
    iter = 0
    rel_drop = rel_drop0

    for itPH in 1:niter_PH

        # ── Outer residuals (fresh momentum + pressure) for convergence check ────
        assemble_momentum_residual_matrices_atomix!(
            dr.Rv_x, dr.Rv_y,
            dr.vx, dr.vy, dr.P, dr.T, nothing,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
            element_v, element_P,
            phases_v_el, τ_old, plastic, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
            backend, workgroup,
        )
        apply_dirichlet!(dr.Rv_x, lr_nodes, zero_lr, backend, workgroup)
        apply_dirichlet!(dr.Rv_y, tb_nodes, zero_tb, backend, workgroup)

        assemble_pressure_residual_matrices_atomix!(
            dr.RP,
            dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
            element_v, element_P,
            phases_P_el, dr.α, dr.ηb, Δt,
            backend, workgroup,
        )

        # ── Outer convergence check ─────────────────────────────────────────────
        # Compare the FD-like, pointwise pressure residual RP/M_P, not the weak
        # integrated residual RP.
        err_P = norm(dr.RP ./ dr.PC_P) / √mesh_stokes.nnodesP
        err_v = max(norm(dr.Rv_x), norm(dr.Rv_y)) / (2 * √mesh_stokes.nnodes)
        if itPH == 1
            err_P0 = err_P + eps(err_P)
            err_v0 = err_v + eps(err_v)
        end
        if itPH == 2
            err_P0 = err_P + eps(err_P)
        end
        err_v_rel = min(err_v / err_v0, err_v)
        err_P_rel = min(err_P / err_P0, err_P)
        err = max(err_v_rel, err_P_rel)

        isnan(err) && error("NaN detected in outer loop at PH=$itPH")
        err > FP(1e10) && error("Kaboom! Error > 1e10 in outer loop at PH=$itPH")

        @printf("itPH = %02d iter = %06d err = %.3e - norm[Rv=%.3e %.3e, Rp=%.3e %.3e]\n",
                itPH, iter, err, err_v, err_v / err_v0, err_P, err_P / err_P0)
        err_P / err_P0 < ϵ_tol && err_v / err_v0 < ϵ_tol &&
            (@info "PH converged at outer iteration $itPH"; break)

        if err > err_min * FP(1.05)
            rel_drop = max(rel_drop * FP(0.1), FP(1e-3))
        end
        err_min = min(err_min, err)

        ϵ_vel = err * rel_drop
        itPT  = 0

        # ── Inner DR loop for velocity (P held fixed) ───────────────────────────
        while err > ϵ_vel && itPT ≤ niter_inner
            itPT += 1
            iter += 1
            do_jac = (mod(itPT, ncheck) == 0) || (itPT == 1)

            # Reassemble pressure residual (v is changing → ∇·v changes → RP changes)
            assemble_pressure_residual_matrices_atomix!(
                dr.RP,
                dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                element_v, element_P,
                phases_P_el, dr.α, dr.ηb, Δt,
                backend, workgroup,
            )

            # Numerical pressure correction
            # P_num = γP * RP.  Here RP is weak, so use γP*RP/M_P.
            @. dr.Pnum = γP * dr.RP / dr.PC_P

            # Momentum residuals with pressure correction
            assemble_momentum_residual_matrices_atomix!(
                dr.Rv_x, dr.Rv_y,
                dr.vx, dr.vy, dr.P, dr.T, dr.Pnum,
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
                element_v, element_P,
                phases_v_el, τ_old, plastic, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
                backend, workgroup,
            )

            # Momentum Jacobian (ForwardDiff — expensive; every ncheck)
            if do_jac
                assemble_augmented_momentum_jacobian_matrices_atomix!(
                    dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
                    dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                    mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                    element_v, element_P,
                    phases_v_el, phases_P_el, τ_old, plastic, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
                    dr.ηb, Δt, γP, dr.PC_P,
                    backend, workgroup,
                )
            end

            # Enforce Dirichlet BCs on residuals and rates
            apply_dirichlet!(dr.Rv_x,  lr_nodes, zero_lr, backend, workgroup)
            apply_dirichlet!(dr.∂vx∂τ, lr_nodes, zero_lr, backend, workgroup)
            apply_dirichlet!(dr.Rv_y,  tb_nodes, zero_tb, backend, workgroup)
            apply_dirichlet!(dr.∂vy∂τ, tb_nodes, zero_tb, backend, workgroup)
            # DYREL-style velocity update
            update_rate!(dr.∂vx∂τ, dr.Rv_x, dr.PC_vx, β_vx, mesh_stokes.nnodes)
            update_variable!(dr.vx, dr.∂vx∂τ, -α_vx, mesh_stokes.nnodes)
            update_rate!(dr.∂vy∂τ, dr.Rv_y, dr.PC_vy, β_vy, mesh_stokes.nnodes)
            update_variable!(dr.vy, dr.∂vy∂τ, -α_vy, mesh_stokes.nnodes)

            # Re-pin Dirichlet values
            apply_dirichlet!(dr.vx, lr_nodes, bc_vx_lr, backend, workgroup)
            apply_dirichlet!(dr.vy, tb_nodes, bc_vy_tb, backend, workgroup)

            # Inner convergence check + damped step-size update (every ncheck)
            if do_jac
                err_v_inner = max(norm(dr.Rv_x), norm(dr.Rv_y)) / (2 * √mesh_stokes.nnodes)
                if iter == 1
                    err_v00 = err_v_inner + eps(err_v_inner)
                end
                err = err_v_inner / err_v00
                isnan(err) && error("NaN detected in inner loop PH=$itPH PT=$itPT")
                err > FP(1e10) && error("Kaboom! Error > 1e10 in inner loop PH=$itPH PT=$itPT")

                verbose_DR && @printf("  it = %d, iter = %d, err = %.3e\n", itPT, iter, err)

                # λmax → Δτ → λmin → damped step for velocity.
                # Cap Δτ at the elastic VE limit (h²/4η_max) to prevent the plastic
                # softening from inflating the step size when λmax drops.
                λmax_vx  = maximum(dr.∂Rv_x∂vx ./ dr.PC_vx)
                λmax_vy  = maximum(dr.∂Rv_y∂vy ./ dr.PC_vy)
                # Δτ_v_max = h^2 / (4 * ηmax)
                # Δτ_vx    = min(2 / √(λmax_vx) * dr.CFL_v, Δτ_v_max)
                # Δτ_vy    = min(2 / √(λmax_vy) * dr.CFL_v, Δτ_v_max)

                Δτ_vx    = 2 / √(λmax_vx) * dr.CFL_v
                Δτ_vy    = 2 / √(λmax_vy) * dr.CFL_v

                λmin_vx = itPT == 1 ? FP(0) : _λmin(α_vx, dr.∂vx∂τ, dr.Rv_x .- dr.Rv_x0, dr.PC_vx)
                λmin_vy = itPT == 1 ? FP(0) : _λmin(α_vy, dr.∂vy∂τ, dr.Rv_y .- dr.Rv_y0, dr.PC_vy)

                α_vx, β_vx = _cheb(Δτ_vx, λmin_vx, dr.c_fact)
                α_vy, β_vy = _cheb(Δτ_vy, λmin_vy, dr.c_fact)

                # Snapshot residuals so ΔR in the next do_jac spans the full ncheck block.
                copyto!(dr.Rv_x0, dr.Rv_x)
                copyto!(dr.Rv_y0, dr.Rv_y)
            end

            itPT == niter_inner && @printf("  inner: max iters (%d) reached at PH=%d\n", niter_inner, itPH)
            iter > total_iterMax && break
        end  # inner PT loop

        # ── Arrow-Hurwicz pressure update (after inner velocity convergence) ─────
        # Same mass-lumped residual as Pnum: pressure is updated from the
        # pointwise divergence residual, not from the weak residual integral.
        @. dr.P += γP * dr.RP / dr.PC_P
        remove_pressure_mean!(dr.P, dr.PC_P)

        iter > total_iterMax && break
    end  # outer PH loop

        P_cpu  = Array(dr.P)
        vx_cpu = Array(dr.vx)
        vy_cpu = Array(dr.vy)
        post = compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu,
            el2n_v_cpu,
            Array(geo_v),
            phases_v_el,
            τ_old,
            dr.η, G_stokes, Δt,
            element_v,
        )
        mean_tauII_history[istep] = mean(post.tauII)
        update_old_stress_from_cells!(τ_old, post, el2n_v_cpu, mesh_stokes.nnodes)

        vtk_path = joinpath(out_dir, @sprintf("stokes_2D_pure_shear_triangle_%04d.vtk", istep))
        write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post)
        @info "Wrote VTK file" vtk_path mean_tauII=mean_tauII_history[istep]
    end  # physical time step loop

    P_cpu  = Array(dr.P)
    vx_cpu = Array(dr.vx)
    vy_cpu = Array(dr.vy)

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------

    # Per-element average pressure (mean of 3 pressure DoFs per element)
    el_P = [mean(P_cpu[DoFsP_cpu[:, i]]) for i in 1:mesh_stokes.nels]

    pts   = [Point2f(c) for c in coords_v]
    polys = [[pts[el2nP_cpu[1, i]], pts[el2nP_cpu[2, i]], pts[el2nP_cpu[3, i]]]
            for i in 1:mesh_stokes.nels]

    fig = Figure(size = (1200, 520))

    # Pressure
    clims_P = extrema(el_P)
    ax1 = Axis(fig[1, 1]; aspect = DataAspect(),
            title = "Pressure  (T7/P1-disc, pure shear)", xlabel = "x", ylabel = "y")
    poly!(ax1, polys; color = el_P, colormap = :vik, colorrange = clims_P, strokewidth = 0)
    Colorbar(fig[1, 2]; colormap = :vik, limits = clims_P,
            label = "P", width = 15, tellheight = false)

    ax2 = Axis(fig[1, 3];
            title = "Mean τII history", xlabel = "time", ylabel = "mean(post.tauII)")
    scatterlines!(ax2, time_history, mean_tauII_history; color = :black, linewidth = 2)

    θ    = LinRange(0, 2π, 300)
    xs_c = cx .+ r_incl .* cos.(θ)
    ys_c = cy .+ r_incl .* sin.(θ)
    lines!(ax1, xs_c, ys_c; color = :white, linewidth = 1.5, linestyle = :dash)

    # show_plot && display(fig)
    display(fig)
    return (; time = time_history, mean_tauII = mean_tauII_history, post)
end

main(; nsteps = 5)
