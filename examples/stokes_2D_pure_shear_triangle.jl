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

"""
    precompute_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels) -> Nothing

Fill per-element geometry data on the configured backend.

This wrapper launches `precompute_geometry_kernel!` with the example-wide
`backend` and `workgroup` constants, then synchronizes before returning.
"""
function precompute_geometry!(geo, coords, el2n, ∂N∂ξq, ω, ::Val{N}, nels) where N
    precompute_geometry_kernel!(backend, workgroup)(
        geo, coords, el2n, ∂N∂ξq, ω, Val(N);
        ndrange = nels,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    update_rate!(∂u∂τ, R, PC, β, ndofs) -> Nothing

Advance a pseudo-transient rate field with diagonal preconditioning.
"""
function update_rate!(∂u∂τ, R, PC, β, ndofs)
    update_rate_kernel!(backend, workgroup)(
        ∂u∂τ, R, PC, β;
        ndrange = ndofs,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    update_variable!(u, ∂u∂τ, α_dr, ndofs) -> Nothing

Apply a damped pseudo-transient increment to a solution field.
"""
function update_variable!(u, ∂u∂τ, α_dr, ndofs)
    update_variable_kernel!(backend, workgroup)(
        u, ∂u∂τ, α_dr;
        ndrange = ndofs,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    remove_pressure_mean!(P, MP) -> Number

Remove the mass-weighted mean pressure and return the removed gauge offset.

The Stokes pressure is defined up to an additive constant. Projecting out this
constant mode prevents the Powell-Hestenes pressure update from drifting by a
large uniform offset that has no physical content.
"""
function remove_pressure_mean!(P, MP)
    p_mean = sum(P .* MP) / sum(MP)
    @. P -= p_mean
    return p_mean
end

pressure_mass(dr) = hasproperty(dr, :M_P) ? getproperty(dr, :M_P) : getproperty(dr, :PC_P)

@inline _phase_at_postprocess(phases::AbstractMatrix, _, i, iel) = Int(phases[i, iel])
@inline _phase_at_postprocess(phases, local_nodes, i, _) = Int(phases[local_nodes[i]])
@inline _phase_loc_postprocess(phases, local_nodes, iel, ::Val{N}) where N =
    SVector{N}(ntuple(i -> _phase_at_postprocess(phases, local_nodes, i, iel), Val(N)))

"""
    compute_strain_rate_stress_postprocess(vx, vy, el2n_v, geo_v, τ_ip, element_v) -> NamedTuple

Compute element-averaged strain-rate and deviatoric-stress diagnostics.

The returned fields are cell averages over the velocity quadrature rule and
include `εxx`, `εyy`, `εzz`, `εxy`, `εII`, `τxx`, `τyy`, `τzz`, `τxy`, `τII`,
and the ASCII alias `tauII`. Stresses are averaged from quadrature-point
history written by the momentum residual assembly.
"""
function compute_strain_rate_stress_postprocess(
    vx, vy,
    el2n_v,
    geo_v,
    τ_ip,
    element_v::ReferenceElement{TV},
) where {TV <: AbstractElement{2, NV}} where NV
    nels = size(el2n_v, 2)
    Nq = shape_function_values(element_v)

    εxx = zeros(Float64, nels)
    εyy = zeros(Float64, nels)
    εzz = zeros(Float64, nels)
    εxy = zeros(Float64, nels)
    εII = zeros(Float64, nels)
    τxx = zeros(Float64, nels)
    τyy = zeros(Float64, nels)
    τzz = zeros(Float64, nels)
    τxy = zeros(Float64, nels)
    τII = zeros(Float64, nels)

    for iel in 1:nels
        local_nodes = SVector{NV}(ntuple(i -> el2n_v[i, iel], Val(NV)))
        vxloc = SVector{NV}(ntuple(i -> vx[local_nodes[i]], Val(NV)))
        vyloc = SVector{NV}(ntuple(i -> vy[local_nodes[i]], Val(NV)))
        geo_el = geo_v[iel]
        volume = 0.0

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

            τxx_q = τ_ip[1][q, iel]
            τyy_q = τ_ip[2][q, iel]
            τxy_q = τ_ip[3][q, iel]
            τzz_q = -(τxx_q + τyy_q)

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

"""
    write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post) -> Nothing

Write pressure, velocity, strain-rate, and stress fields to an ASCII VTK file.

The VTK mesh uses the pressure triangle corners, while nodal velocity fields are
sampled from the corresponding velocity nodes.
"""
function write_stokes_vtk(vtk_path, mesh_stokes, coords_v, el2nP_cpu, DoFsP_cpu, P_cpu, vx_cpu, vy_cpu, post)
    NP = size(el2nP_cpu, 1)
    vtk_nodes = sort!(unique(vec(el2nP_cpu)))
    vtk_node_map = zeros(Int32, length(coords_v))
    for (new_i, old_i) in enumerate(vtk_nodes)
        vtk_node_map[old_i] = Int32(new_i)
    end

    vtk_P = zeros(Float64, length(vtk_nodes))
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

"""
    build_triangle_t7_inclusion_mesh(; Lx, Ly, cx, cy, r, n_circle=96, max_area=nothing) -> Tuple

Build an unstructured T7 velocity mesh around a circular inclusion.

Triangulate.jl generates a second-order T6 PSLG mesh with the circle as a
constrained internal boundary. The local midpoint ordering is remapped to
FEMTools' T6/T7 convention, then one centroid bubble node is appended per
element.
"""
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

"""
    main(; nsteps=15, n_circle=96, max_area=1 / (1 * 64^2), Δt=1 / 6, show_plot=true) -> NamedTuple

Run the unstructured T7/P1-disc pure-shear Stokes example.

The model builds a square domain with a circular inclusion, applies pure-shear
boundary conditions, advances the viscoelastic-plastic Stokes solve, writes one
VTK file per physical step, and returns the stress-history diagnostics.
"""
function main(; nsteps = 15, n_circle = 96, max_area = 1 / (1 * 64^2), Δt = 1 / 6, show_plot = true)
    # Domain
    Lx, Ly = 1.0, 1.0

    # Background pure-shear strain rate (non-dimensional)
    ε̇_bg = 1.0

    # Material (2 phases: matrix + inclusion)
    η     = (1.0,     1.0)   # shear viscosity
    γfact = 20.0
    α     = (0.0,     0.0)   # thermal expansivity  (zero → isothermal)
    ρ0    = (1.0,     1.0)   # reference density
    K     = (4e0,     4e0)   # bulk modulus  (Inf → incompressible)
    ηb    = K                # pressure storage modulus; residual uses ηb * Δt
    G     = (1e0,     0.5)   # Shear modulus
    G_stokes = G
    # Cohesion chosen so the yield stress C·cosϕ = 1.6 at zero pressure.
    # Background deviatoric stress in pure shear is 2η·ε̇_bg = 2, so the
    # inclusion (lower G) will enter the plastic regime after a few steps.
    τy      = 1.6 / cosd(30)                   # cohesion C; yield stress = C·cosϕ = 1.6
    plastic = DruckerPrager(
        (π/6, π/6),                            # friction angle ϕ = 30° [rad]
        (0.0, 0.0),                            # dilation angle Ψ = 0°  [rad] (non-associated)
        (τy, τy),                              # cohesion C [same for both phases]
        (8.0e-3,  8.0e-3),                     # plastic regularisation viscosity η_reg
        K,                                     # Kb (passed separately from elastic K)
    )
    g     = (0.0,     0.0)   # gravity vector
    Tref  = 0.0

    # DR solver
    ncheck = 50          # convergence check interval
    ϵ_tol  = 1e-6        # relative residual tolerance

    # Inclusion geometry. The Triangle PSLG uses this circle as an internal
    # constrained boundary, so no element crosses the material interface.
    r_incl = 0.1
    cx     = Lx / 2
    cy     = Ly / 2

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})   # T7 (bubble)
    element_P = ReferenceElement(LinearElement{2, 3, Float64})      # P1-disc

    coords_v_cpu, el2n_v_cpu, outer_nodes, circle_nodes = build_triangle_t7_inclusion_mesh(;
        Lx, Ly,
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

    geo_v = Vector{NTuple{NQ_v, Tuple{SMatrix{NV, 2, Float64, 2NV}, Float64}}}(undef, mesh_stokes.nels)
    geo_P = Vector{NTuple{NQ_v, Tuple{SMatrix{NP, 2, Float64, 2NP}, Float64}}}(undef, mesh_stokes.nels)

    precompute_geometry!(geo_v, mesh_stokes.coords, mesh_stokes.el2n, ∂N∂ξq_v, ip_v.ω, Val(NV), mesh_stokes.nels)
    precompute_geometry!(geo_P, mesh_stokes.coords, mesh_stokes.el2nP, ∂N∂ξq_P, ip_v.ω, Val(NP), mesh_stokes.nels)

    # ---------------------------------------------------------------------------
    # StokesDR struct
    # ---------------------------------------------------------------------------

    dr = StokesDR(
        backend,
        mesh_stokes.nnodes,
        mesh_stokes.nnodesP,
        η, ηb, α;
        ρ0,
        K,
        g,
        Tref,
        CFL_v = 0.9, CFL_P = 0.9, c_fact = 0.9,
        stress_size = (NQ_v, mesh_stokes.nels),
        # CFL_v = 0.03, CFL_P = 0.9, c_fact = 0.5,
    )
    M_P = pressure_mass(dr)
    τ = (dr.τxx, dr.τyy, dr.τxy)
    τ_old = (dr.τxx_old, dr.τyy_old, dr.τxy_old)

    # ---------------------------------------------------------------------------
    # Phase assignment — circular inclusion
    # ---------------------------------------------------------------------------

    in_incl(c) = (c[1] - cx)^2 + (c[2] - cy)^2 ≤ r_incl^2

    coords_v     = Array(mesh_stokes.coords)
    phases_v_cpu = Int[in_incl(c) ? 2 : 1 for c in coords_v]
    copyto!(dr.phases_v, phases_v_cpu)

    el2nP_cpu    = Array(mesh_stokes.el2nP)
    DoFsP_cpu    = Array(mesh_stokes.DoFsP)
    phases_P_cpu = Int[
        in_incl(coords_v[el2nP_cpu[mod1(d, 3), cld(d, 3)]]) ? 2 : 1
        for d in 1:mesh_stokes.nnodesP
    ]
    copyto!(dr.phases_P, phases_P_cpu)

    @info "Phases" n_incl_v=count(==(2), phases_v_cpu) n_incl_P=count(==(2), phases_P_cpu)

    # ---------------------------------------------------------------------------
    # Boundary conditions — pure shear
    #   vx = +ε̇_bg * (x - Lx/2),   vy = -ε̇_bg * (y - Ly/2)
    # ---------------------------------------------------------------------------

    Γnodes = Array(mesh_v.Γnodes)
    coords = Array(mesh_v.coords)

    x_bc = Float64[coords[n][1] for n in Γnodes]
    y_bc = Float64[coords[n][2] for n in Γnodes]

    bc_vx_vals = Float64[@.( ε̇_bg * (x_bc - Lx / 2))...]
    bc_vy_vals = Float64[@.(-ε̇_bg * (y_bc - Ly / 2))...]

    # Seed the full interior with the analytical pure-shear field so the
    # solver starts with a good initial guess (boundary nodes are overwritten
    # by apply_bc! below; the result is identical on those nodes).
    copyto!(dr.vx, Float64[ ε̇_bg * (c[1] - Lx / 2) for c in coords_v])
    copyto!(dr.vy, Float64[-ε̇_bg * (c[2] - Ly / 2) for c in coords_v])

    apply_bc!(dr.vx, DirichletBoundaryCondition(nothing, Γnodes, bc_vx_vals))
    apply_bc!(dr.vy, DirichletBoundaryCondition(nothing, Γnodes, bc_vy_vals))

    zero_vbc = zero(bc_vx_vals)

    @info "BCs" n_bc = length(Γnodes) max_vx = maximum(abs, bc_vx_vals)

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
    #   dr.M_P = ∫ N_i dΩ
    #   γP      = local viscosity-weighted pressure update scale
    # Then γP * RP/M_P matches the pointwise FD-style pressure correction, but
    # adapts the pressure step to viscosity contrasts.
    γP = KernelAbstractions.zeros(backend, Float64, mesh_stokes.nnodesP)
    assemble_viscosity_weighted_pressure_scaling!(
        M_P, γP,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_P, mesh_stokes.nels,
        element_v, element_P,
        dr.phases_v, dr.η, γfact, dr.K, Δt,
        backend, workgroup,
    )

    Δt = Δt === nothing ? 0.5 / max(abs(ε̇_bg), eps(Float64)) : Float64(Δt)
    time_history = zeros(Float64, nsteps)
    mean_tauII_history = zeros(Float64, nsteps)

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

    iterMax       = 50_000   # max inner DR iterations per PH step
    total_iterMax = 50_000   # max total inner DR iterations
    nout          = ncheck   # residual / spectral update cadence
    rel_drop0     = 1e-2     # inner convergence: velocity residual drops by this factor
    verbose_PH    = true
    verbose_DR    = false

    @info "Starting PH/DYREL-style Stokes solver" nsteps Δt iterMax total_iterMax nout ϵ_tol

    # λ_min helper (Rayleigh quotient from consecutive residual snapshots).
    # JustRelax computes this with the actual velocity increment, dV = βV*dτV*dVdτ.
    _λmin(step, rate, ΔR, PC) = begin
        dV = step .* rate
        denom = sum(dV .^ 2)
        denom == 0 ? 0.0 : abs(sum(dV .* (ΔR ./ PC))) / denom
    end

    # Damped DYREL/Chebyshev step from spectral step Δτ and damping λmin.
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

        assemble_augmented_momentum_jacobian_matrices_atomix!(
            dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
            dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
            element_v, element_P,
            dr.phases_v, dr.phases_P, τ_old, plastic, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
            dr.ηb, Δt, γP, M_P,
            backend, workgroup,
        )
        λmax_vx = maximum(dr.∂Rv_x∂vx ./ dr.PC_vx)
        λmax_vy = maximum(dr.∂Rv_y∂vy ./ dr.PC_vy)
        Δτ_vx   = 2 / √(λmax_vx) * dr.CFL_v
        Δτ_vy   = 2 / √(λmax_vy) * dr.CFL_v
        α_vx, β_vx = _cheb(Δτ_vx, 0.0, dr.c_fact)
        α_vy, β_vy = _cheb(Δτ_vy, 0.0, dr.c_fact)
        @info "Initial momentum preconditioner" λmax_vx λmax_vy Δτ_vx Δτ_vy

        err_min = Inf
        ϵ = Float64(ϵ_tol)
        err = 2 * ϵ
        err_v0 = 1.0
        err_P0 = 1.0
        err_v00 = 1.0
        iter = 0
        rel_drop = rel_drop0

        for itPH in 1:1000

            # ── Outer residuals (fresh momentum + pressure) for convergence check ────
            assemble_pressure_residual_matrices_atomix!(
                dr.RP,
                dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                element_v, element_P,
                dr.phases_P, dr.α, dr.ηb, Δt,
                backend, workgroup,
            )

            assemble_momentum_residual_matrices_atomix!(
                dr.Rv_x, dr.Rv_y,
                dr.vx, dr.vy, dr.P, dr.T, nothing,
                mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
                element_v, element_P,
                dr.phases_v, τ_old, plastic, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
                backend, workgroup,
            )
            apply_dirichlet!(dr.Rv_x, Γnodes, zero_vbc, backend, workgroup)
            apply_dirichlet!(dr.Rv_y, Γnodes, zero_vbc, backend, workgroup)

            # ── Outer convergence check ─────────────────────────────────────────────
            # Compare the FD-like, pointwise pressure residual RP/M_P, not the weak
            # integrated residual RP.
            err_P = norm(dr.RP ./ M_P) / √mesh_stokes.nnodesP
            err_v = max(norm(dr.Rv_x), norm(dr.Rv_y)) / (2 * √mesh_stokes.nnodes)
            if itPH == 1
                err_P0 = err_P + eps(err_P)
                err_v0 = err_v + eps(err_v)
            end
            if itPH == 2
                err_P0 = err_P + eps(err_P)
            end
            err_v_rel = err_v / err_v0
            err_P_rel = err_P / err_P0
            err_abs = max(err_v, err_P)
            err_rel = max(err_v_rel, err_P_rel)
            err = min(err_abs, err_rel)

            isnan(err) && error("NaN detected in outer loop at PH=$itPH")
            err > 1e10 && error("Kaboom! Error > 1e10 in outer loop at PH=$itPH")

            if verbose_PH
                @printf("itPH = %02d iter = %06d err = %.3e abs = %.3e rel = %.3e - norm[Rv=%.3e %.3e, Rp=%.3e %.3e]\n",
                        itPH, iter, err, err_abs, err_rel, err_v, err_v_rel, err_P, err_P_rel)
            end
            err < ϵ && break

            if err > err_min * 1.05
                rel_drop = max(rel_drop * 0.1, 1e-3)
            end
            err_min = min(err_min, err)

            ϵ_vel = err * rel_drop
            itPT  = 0

            # ── Inner DR loop for velocity (P held fixed) ───────────────────────────
            while err > ϵ_vel && itPT ≤ iterMax
                itPT += 1
                iter += 1

                copyto!(dr.Rv_x0, dr.Rv_x)
                copyto!(dr.Rv_y0, dr.Rv_y)

                # Reassemble pressure residual (v is changing → ∇·v changes → RP changes)
                assemble_pressure_residual_matrices_atomix!(
                    dr.RP,
                    dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                    mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                    element_v, element_P,
                    dr.phases_P, dr.α, dr.ηb, Δt,
                    backend, workgroup,
                )

                # Numerical pressure correction, matching JustRelax's pointwise DYREL
                # P_num = γP * RP.  Here RP is weak, so use γP*RP/M_P.
                @. dr.Pnum = γP * dr.RP / M_P

                # Momentum residuals with pressure correction
                assemble_momentum_residual_matrices_atomix!(
                    dr.Rv_x, dr.Rv_y,
                    dr.vx, dr.vy, dr.P, dr.T, dr.Pnum,
                    mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
                    element_v, element_P,
                    dr.phases_v, τ_old, plastic, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
                    backend, workgroup,
                )

                # Enforce Dirichlet BCs on residuals and rates
                for field in (dr.Rv_x, dr.Rv_y, dr.∂vx∂τ, dr.∂vy∂τ)
                    apply_dirichlet!(field, Γnodes, zero_vbc, backend, workgroup)
                end

                # DYREL-style velocity update
                update_rate!(dr.∂vx∂τ, dr.Rv_x, dr.PC_vx, β_vx, mesh_stokes.nnodes)
                update_variable!(dr.vx, dr.∂vx∂τ, -α_vx, mesh_stokes.nnodes)
                update_rate!(dr.∂vy∂τ, dr.Rv_y, dr.PC_vy, β_vy, mesh_stokes.nnodes)
                update_variable!(dr.vy, dr.∂vy∂τ, -α_vy, mesh_stokes.nnodes)

                # Re-pin Dirichlet values
                apply_dirichlet!(dr.vx, Γnodes, bc_vx_vals, backend, workgroup)
                apply_dirichlet!(dr.vy, Γnodes, bc_vy_vals, backend, workgroup)

                # Inner convergence check + damped step-size update
                if iszero(iter % nout)
                    err_v_inner = max(norm(dr.Rv_x), norm(dr.Rv_y)) / (2 * √mesh_stokes.nnodes)
                    if iter == nout
                        err_v00 = err_v_inner + eps(err_v_inner)
                    end
                    err = max(err_v_inner / err_v00, err_v_inner)
                    isnan(err) && error("NaN detected in inner loop PH=$itPH PT=$itPT")
                    err > 1e10 && error("Kaboom! Error > 1e10 in inner loop PH=$itPH PT=$itPT")

                    verbose_DR && @printf("  it = %d, iter = %d, err = %.3e\n", itPT, iter, err)

                    λmin_vx = _λmin(α_vx, dr.∂vx∂τ, dr.Rv_x .- dr.Rv_x0, dr.PC_vx)
                    λmin_vy = _λmin(α_vy, dr.∂vy∂τ, dr.Rv_y .- dr.Rv_y0, dr.PC_vy)

                    assemble_augmented_momentum_jacobian_matrices_atomix!(
                        dr.∂Rv_x∂vx, dr.PC_vx, dr.∂Rv_y∂vy, dr.PC_vy,
                        dr.vx, dr.vy, dr.P, dr.P0, dr.T, dr.T0,
                        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, geo_P, mesh_stokes.nels,
                        element_v, element_P,
                        dr.phases_v, dr.phases_P, τ_old, plastic, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref,
                        dr.ηb, Δt, γP, M_P,
                        backend, workgroup,
                    )

                    # λmax → Δτ → damped step for velocity.
                    λmax_vx  = maximum(dr.∂Rv_x∂vx ./ dr.PC_vx)
                    λmax_vy  = maximum(dr.∂Rv_y∂vy ./ dr.PC_vy)
                    Δτ_vx    = 2 / √(λmax_vx) * dr.CFL_v
                    Δτ_vy    = 2 / √(λmax_vy) * dr.CFL_v

                    α_vx, β_vx = _cheb(Δτ_vx, λmin_vx, dr.c_fact)
                    α_vy, β_vy = _cheb(Δτ_vy, λmin_vy, dr.c_fact)
                end

                itPT == iterMax && @printf("  inner: max iters (%d) reached at PH=%d\n", iterMax, itPH)
                iter > total_iterMax && break
            end  # inner PT loop

            # ── Arrow-Hurwicz pressure update (after inner velocity convergence) ─────
            # Same mass-lumped residual as Pnum: pressure is updated from the
            # pointwise divergence residual, not from the weak residual integral.
            @. dr.P += γP * dr.RP / M_P
            remove_pressure_mean!(dr.P, M_P)

            iter > total_iterMax && break
        end  # outer PH loop

        assemble_momentum_residual_matrices_atomix!(
            dr.Rv_x, dr.Rv_y,
            dr.vx, dr.vy, dr.P, dr.T, nothing,
            mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
            element_v, element_P,
            dr.phases_v, τ_old, plastic, τ, dr.η, G_stokes, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, Δt,
            backend, workgroup,
        )

        P_cpu  = Array(dr.P)
        vx_cpu = Array(dr.vx)
        vy_cpu = Array(dr.vy)
        post = compute_strain_rate_stress_postprocess(
            vx_cpu, vy_cpu,
            el2n_v_cpu,
            Array(geo_v),
            τ,
            element_v,
        )
        mean_tauII_history[istep] = mean(post.tauII)
        copyto!(dr.τxx_old, dr.τxx)
        copyto!(dr.τyy_old, dr.τyy)
        copyto!(dr.τxy_old, dr.τxy)

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

    show_plot && display(fig)
    return (; time = time_history, mean_tauII = mean_tauII_history, post)
end

main()
